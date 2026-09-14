"""Inference for a promoted P0 frozen direct-span probe.

The command performs a selected-checkpoint dev replay before writing the test
submission.  It never loads holdout data and has no fitting path.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from prepared_loader import PreparedData
from neural_sequence import build_model, encode_document, collate_encoded
from submit_m0 import format_prediction, validate_rows
from run_neural_sequence import sha
from run_span_probe import provenance
from span_probe import SpanHead, universe, padded_hidden, decode_spans, character_spans
from build_span_cache import _check_provenance, _capture_batch

ROOT = Path(__file__).resolve().parents[1]
CHUNK = 32768
BATCH_SIZE = 32


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _encoded(row, vocab):
    ts, ids, chars, flags = encode_document(row["full_text"], vocab)
    return (ts, ids, chars, flags, np.full(len(ids), -100, dtype=np.int64))


def _predict_documents(encoder, head, rows, vocab, progress=False):
    docs = [_encoded(row, vocab) for row in rows]
    lengths = [len(doc[1]) for doc in docs]
    if any(length <= 0 for length in lengths):
        raise ValueError("P0 inference requires every document to contain at least one token")
    order = sorted(range(len(rows)), key=lambda i: lengths[i])
    result = {}
    encoder.eval(); head.eval()
    with torch.no_grad():
        for begin in range(0, len(order), BATCH_SIZE):
            ids = order[begin:begin + BATCH_SIZE]
            batch_docs = [docs[i] for i in ids]
            hidden, _, actual_lengths, _ = _capture_batch(encoder, batch_docs, "cuda")
            h, prefix = padded_hidden([hidden[j, :actual_lengths[j]] for j in range(len(ids))])
            pairs = [universe(actual_lengths[j]) for j in range(len(ids))]
            counts = [len(pair) for pair in pairs]
            allpairs = np.concatenate(pairs)
            document_ids = np.repeat(np.arange(len(ids)), counts)
            values=[]; labels=[]
            for offset in range(0, len(allpairs), CHUNK):
                pp = allpairs[offset:offset + CHUNK]
                dd = document_ids[offset:offset + CHUNK]
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = head(h, prefix,
                                  torch.as_tensor(dd, device="cuda"),
                                  torch.as_tensor(pp[:, 0], device="cuda"),
                                  torch.as_tensor(pp[:, 1], device="cuda"))
                z = logits.float().cpu().numpy()
                if not np.isfinite(z).all():
                    raise ValueError("Nonfinite span logits")
                lab = z[:, 1:].argmax(1) + 1
                values.append(z[np.arange(len(z)), lab] - z[:, 0])
                labels.append(lab)
            values = np.concatenate(values)
            labels = np.concatenate(labels)
            offset = 0
            for local, i in enumerate(ids):
                count = counts[local]
                token_spans = decode_spans(actual_lengths[local],
                                           values[offset:offset + count],
                                           labels[offset:offset + count],
                                           int(rows[i]["expected_entity_count"]))
                token_bounds = np.asarray([[a, b] for a, b, _ in batch_docs[local][0]], dtype=np.int32)
                result[rows[i]["document_id"]] = character_spans(token_spans, token_bounds)
                offset += count
            if progress and begin % (BATCH_SIZE * 50) == 0:
                print(f"Predicted {min(begin + BATCH_SIZE, len(order))}/{len(order)} docs", flush=True)
    return result


def _prediction_rows(rows, predictions):
    output=[]
    for row in rows:
        spans=predictions[row["document_id"]]
        if len(spans) != int(row["expected_entity_count"]):
            raise ValueError("Exact-K inference contract failure")
        output.append({"document_id": row["document_id"],
                       "predicted": [{"start": a, "end": b, "label": label}
                                     for a, b, label in spans]})
    return output


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=ROOT / "output/bitrase-2/p0/seed2026")
    args=parser.parse_args(); run=args.run.resolve()
    if not torch.cuda.is_available():
        raise RuntimeError("P0 inference requires CUDA")
    if (run / "submission.csv").exists():
        raise FileExistsError(f"Refusing to overwrite {run / 'submission.csv'}")
    config=read_json(run / "config.json"); report=read_json(run / "report.json")
    if config.get("replication") or config.get("seed") != 2026:
        raise ValueError("Predictor is restricted to discovery seed2026")
    if not report.get("promotion_candidate"):
        raise ValueError("Discovery run is not promotion_candidate")
    if config.get("sources") != provenance():
        raise ValueError("P0 provenance mismatch")
    epoch=int(report["selected_epoch"]); checkpoint=run / f"epoch{epoch}.pt"
    if sha(checkpoint) != report.get("selected_model_sha256"):
        raise ValueError("Selected span-head checkpoint hash mismatch")
    _, _, encoder_checkpoint, vocab_path = _check_provenance()
    vocab=read_json(vocab_path)
    encoder=build_model(len(vocab)).cuda()
    encoder.load_state_dict(torch.load(encoder_checkpoint, map_location="cuda", weights_only=True))
    head=SpanHead().cuda();head.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
    torch.set_num_threads(4)
    data=PreparedData(); dev=data.load_dev("evaluation").to_pylist()
    predictor_hash=sha(Path(__file__))
    started=time.monotonic()
    dev_pred=_predict_documents(encoder, head, dev, vocab, progress=True)
    dev_file=run / f"epoch{epoch}_dev_predictions.parquet"
    if sha(dev_file)!=report['epochs'][str(epoch)]['prediction_sha256']:
        raise ValueError('Selected dev prediction hash mismatch')
    import pyarrow.parquet as pq
    expected=pq.read_table(dev_file).to_pylist()
    actual=_prediction_rows(dev, dev_pred)
    actual_map={r['document_id']:r['predicted'] for r in actual}
    expected_map={r['document_id']:r['predicted'] for r in expected}
    if len(actual)!=len(actual_map) or len(expected)!=len(expected_map) or actual_map != expected_map:
        raise ValueError("Dev replay differs from selected P0 predictions")
    print('All selected dev predictions replay identically',flush=True)
    test=data.load_test().to_pylist(); slots=data.load_test_slots().to_pylist()
    test_pred=_predict_documents(encoder, head, test, vocab, progress=True)
    text={row["document_id"]:row["full_text"] for row in test}
    counts={row["document_id"]:int(row["expected_entity_count"]) for row in test}
    rows=[]
    for slot in slots:
        spans=test_pred[slot["document_id"]]
        rows.append({"row_id":slot["row_id"],"document_id":slot["document_id"],
                     "Predicted":format_prediction(spans[slot["slot"] - 1])})
    validate_rows(rows, text, counts)
    if provenance()!=config['sources'] or sha(Path(__file__))!=predictor_hash:
        raise ValueError('Inference provenance changed')
    run.mkdir(parents=True, exist_ok=True)
    target=run / "submission.csv"
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer=csv.DictWriter(stream, fieldnames=["row_id", "Predicted"],extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    if provenance() != config["sources"]:
        raise ValueError("P0 provenance changed during inference")
    manifest={"rows":len(rows),"documents":len(test),"epoch":epoch,"pretrained":False,"uploaded":False,
              "dev_replay_verified":True,"submission_sha256":sha(target),"head_sha256":sha(checkpoint),
              "encoder_sha256":sha(encoder_checkpoint),"vocab_sha256":sha(vocab_path),
              "test_sha256":sha(ROOT/'output/prepared/v1/test.parquet'),
              "slots_sha256":sha(ROOT/'output/prepared/v1/test_slots.parquet'),
              "predictor_sha256":sha(Path(__file__)),"seconds":time.monotonic()-started}
    (run/'submission_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':main()
