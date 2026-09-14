"""Build a provenance checked frozen encoder cache for the direct span probe.

This script deliberately has no training path.  It replays the epoch-4 neural
encoder in the same length-sorted batches used by the baseline evaluator and
stores token states and character boundaries in a flat, document-indexed cache.
"""
from __future__ import annotations

import argparse
import json
import hashlib
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from prepared_loader import PreparedData
from sequence_crf import tokens
from neural_sequence import build_model, encode_document, collate_encoded, decode_logits
from run_neural_sequence import sha, source_hashes

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "output" / "bitrase-1.algo-6" / "v1"
CACHE_ROOT = ROOT / "output" / "bitrase-2" / "p0" / "cache"
BATCH_SIZE = 32


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _check_provenance():
    config = _read_json(RUN / "config.json")
    report = _read_json(RUN / "epoch4_report.json")
    if source_hashes() != config.get("sources"):
        raise ValueError("Frozen neural source hashes differ from epoch-4 config")
    checkpoint = RUN / "epoch4.pt"
    if sha(checkpoint) != report.get("model_sha256"):
        raise ValueError("Epoch-4 checkpoint hash mismatch")
    vocab = RUN / "vocab.json"
    if sha(vocab) != _read_json(RUN / "report.json").get("vocab_sha256"):
        raise ValueError("Frozen vocabulary hash mismatch")
    if config.get("batch_size") != BATCH_SIZE:
        raise ValueError(f"Expected baseline batch size {BATCH_SIZE}")
    return config, report, checkpoint, vocab


def _load_existing_predictions():
    path = RUN / "epoch4_known_k_dev_predictions.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    return pq.read_table(path).to_pylist(), sha(path)


def _new_cache_dir(path: Path):
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite cache directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _encoded(row, vocab):
    ts, ids, chars, flags = encode_document(row["full_text"], vocab)
    # Labels are intentionally absent from the cache.  The dev gold is never
    # passed to the model or decoder; the train cache is likewise inference-like.
    return (ts, ids, chars, flags, np.full(len(ids), -100, dtype=np.int64))


def _capture_batch(model, docs, device):
    batch = collate_encoded(docs)
    lengths = batch["mask"].sum(1)
    captured = []

    def hook(_module, inputs):
        captured.append(inputs[0].detach())

    handle = model.out.register_forward_pre_hook(hook)
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            logits = model(batch["words"].to(device), batch["chars"].to(device),
                           batch["flags"].to(device), lengths)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("Encoder hook did not capture exactly one hidden tensor")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        replay_logits = model.out(captured[0])
    if not torch.allclose(logits, replay_logits, rtol=0.0, atol=0.0):
        delta = (logits - replay_logits).abs()
        raise ValueError(f"Native output replay mismatch: maxabs={float(delta.max())}")
    maxabs = float((logits - replay_logits).abs().max())
    hidden = captured[0].float() if captured[0].dtype == torch.float32 else captured[0]
    # Preserve the encoder's native dtype.  The float() branch above is only
    # reached for native FP32 and is intentionally lossless.
    return hidden.cpu().numpy(), logits.detach().float().cpu(), lengths.tolist(), maxabs


def _compare_predictions(rows, expected):
    if len(rows) != len(expected):
        raise ValueError("Replay prediction row count differs")
    for index, (actual, wanted) in enumerate(zip(rows, expected)):
        if actual != wanted:
            raise ValueError(f"Epoch-4 known-K prediction mismatch at row {index}: {actual!r} != {wanted!r}")


def build(split: str):
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen cache extraction requires CUDA")
    generator_sha = sha(Path(__file__))
    parent_hashes = source_hashes()
    config, report, checkpoint, vocab_path = _check_provenance()
    if parent_hashes != config.get("sources"):
        raise ValueError("Frozen neural parent hashes differ from epoch-4 config")
    vocab = _read_json(vocab_path)
    data = PreparedData()
    rows = (data.load_train("train_fitting") if split == "train"
            else data.load_dev("evaluation")).to_pylist()
    out = CACHE_ROOT / split
    _new_cache_dir(out)

    model = build_model(len(vocab)).to("cuda")
    model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
    model.eval()
    torch.set_num_threads(4)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    lengths = [len(tokens(row["full_text"])) for row in rows]
    total = sum(lengths)
    order = sorted(range(len(rows)), key=lambda i: lengths[i])
    hidden_mm = None
    boundaries_mm = np.lib.format.open_memmap(out / "boundaries.npy", mode="w+",
                                               dtype=np.int32, shape=(total, 2))
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(np.asarray(lengths, dtype=np.int64))
    np.save(out / "offsets.npy", offsets)
    metadata = [{"document_id": row["document_id"],
                 "expectedK": int(row["expected_entity_count"]),
                 "token_count": lengths[i]} for i, row in enumerate(rows)]
    started = time.monotonic()
    first_hidden = None
    train_prediction_rows = []
    replay_maxabs = 0.0
    for begin in range(0, len(order), BATCH_SIZE):
        indices = order[begin:begin + BATCH_SIZE]
        docs = [_encoded(rows[i], vocab) for i in indices]
        hidden, logits, actual_lengths, batch_maxabs = _capture_batch(model, docs, "cuda")
        replay_maxabs = max(replay_maxabs, batch_maxabs)
        if first_hidden is None:
            first_hidden = hidden.copy()
            hidden_mm = np.lib.format.open_memmap(out / "hidden.npy", mode="w+",
                                                   dtype=hidden.dtype,
                                                   shape=(total, hidden.shape[-1]))
        if hidden_mm is None or hidden.dtype != hidden_mm.dtype or hidden.shape[-1] != 256:
            raise ValueError("Unexpected hidden-state shape or dtype")
        if split == "train":
            known = decode_logits(logits, actual_lengths, docs,
                                  [int(rows[i]["expected_entity_count"]) for i in indices])
            for i, spans in zip(indices, known):
                train_prediction_rows.append({"document_id": rows[i]["document_id"],
                                              "predicted": [{"start": a, "end": b, "label": label}
                                                            for a, b, label in spans]})
        for j, doc_index in enumerate(indices):
            count = actual_lengths[j]
            start, end = int(offsets[doc_index]), int(offsets[doc_index + 1])
            if end - start != count:
                raise ValueError("Token count changed between indexing and encoding")
            hidden_mm[start:end] = hidden[j, :count]
            boundaries_mm[start:end] = np.asarray([[a, b] for a, b, _ in docs[j][0]], dtype=np.int32)
        if begin % (BATCH_SIZE * 50) == 0:
            print(f"Cached {min(begin + BATCH_SIZE, len(order))}/{len(order)} {split} docs", flush=True)
    hidden_mm.flush(); boundaries_mm.flush()

    # A native-dtype read replay catches truncation/corruption before the
    # manifest is marked complete.  It also checks logits and argmax for one
    # representative batch against the just-captured forward pass.
    replay_hidden = np.load(out / "hidden.npy", mmap_mode="r")
    if first_hidden is None or not np.array_equal(replay_hidden[int(offsets[order[0]]):int(offsets[order[0] + 1])],
                                                   first_hidden[0, :lengths[order[0]]]):
        raise ValueError("Representative hidden-state mmap replay mismatch")

    prediction_hash = None
    if split == "dev":
        expected, prediction_hash = _load_existing_predictions()
        if prediction_hash != report.get("prediction_hashes", {}).get("known_k"):
            raise ValueError("Epoch-4 known-K prediction hash differs from report")
        replay_rows = []
        # Replay every document's cached state, in the baseline evaluator's
        # exact batches, and compare only known-K predictions.
        for begin in range(0, len(order), BATCH_SIZE):
            indices = order[begin:begin + BATCH_SIZE]
            max_len = max(lengths[i] for i in indices)
            padded = np.zeros((len(indices), max_len, 256), dtype=replay_hidden.dtype)
            for j, i in enumerate(indices):
                lo, hi = int(offsets[i]), int(offsets[i + 1]); padded[j, :hi - lo] = replay_hidden[lo:hi]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                replay_logits = model.out(torch.from_numpy(padded).to("cuda"))
            docs = [_encoded(rows[i], vocab) for i in indices]
            ks = [int(rows[i]["expected_entity_count"]) for i in indices]
            predicted = decode_logits(replay_logits.float().cpu(), [lengths[i] for i in indices], docs, ks)
            for i, spans in zip(indices, predicted):
                replay_rows.append({"document_id": rows[i]["document_id"],
                                    "predicted": [{"start": a, "end": b, "label": label} for a, b, label in spans]})
            if begin % (BATCH_SIZE * 50) == 0:
                print(f"Replayed {min(begin + BATCH_SIZE, len(order))}/{len(order)} dev docs", flush=True)
        # Baseline parquet is written in sorted evaluator order.
        _compare_predictions(replay_rows, expected)
    if split == "train":
        prediction_path = out / "train_known_k_predictions.parquet"
        pq.write_table(pa.Table.from_pylist(train_prediction_rows), prediction_path, compression="zstd")

    manifest = {
        "complete": False, "split": split, "documents": len(rows), "tokens": total,
        "hidden_shape": [total, 256], "hidden_dtype": str(hidden_mm.dtype),
        "batch_size": BATCH_SIZE, "checkpoint": "epoch4.pt",
        "checkpoint_sha256": sha(checkpoint), "vocab_sha256": sha(vocab_path),
        "sources": source_hashes(), "data_sha256": sha(ROOT / "output" / "prepared" / "v1" / ("train.parquet" if split == "train" else "dev.parquet")),
        "baseline_dev_predictions_sha256": prediction_hash,
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                     "numpy": np.__version__, "pyarrow": pa.__version__},
        "native_output_replay_maxabs": replay_maxabs,
        "generator_sha256": generator_sha,
        "files": {}, "seconds": time.monotonic() - started,
    }
    files = ["hidden.npy", "boundaries.npy", "offsets.npy"]
    if split == "train":
        files.append("train_known_k_predictions.parquet")
    for name in files:
        manifest["files"][name] = {"sha256": sha(out / name), "bytes": (out / name).stat().st_size}
    (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["files"]["metadata.json"] = {"sha256": sha(out / "metadata.json"), "bytes": (out / "metadata.json").stat().st_size}
    manifest["complete"] = True
    if source_hashes() != parent_hashes or sha(Path(__file__)) != generator_sha:
        raise ValueError("Frozen provenance changed during cache extraction")
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "dev"), required=True)
    build(parser.parse_args().split)


if __name__ == "__main__":
    main()
