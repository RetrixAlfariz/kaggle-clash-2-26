# /// script
# requires-python = "==3.13.2"
# dependencies = ["pyarrow==25.0.1", "scikit-learn==1.9.1"]
# ///
"""bitrase-1.algo-1: frozen classifier, MAP/slot-MBR decoder ablation."""
import argparse
import collections
import csv
import json
import pickle
from pathlib import Path
import time

import pyarrow as pa
import pyarrow.parquet as pq
from baseline_probe import decode as parent_decode
from ner_evaluation import Evaluation, triples
from prepared_loader import PreparedData
from slot_decoder import decode
from submit_m0 import load_dictionary, format_prediction, validate_rows
from train_m1 import propose, score, sha, model_sources

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/bitrase-1.algo-1/v1"
METHODS = ("slot_mbr", "map")


def load_parent():
    parent = ROOT / "output/m1/v1"
    config = json.loads((parent / "config.json").read_text())
    report = json.loads((parent / "report.json").read_text())
    if model_sources() != config["sources"] or sha(parent / "model.pkl") != report["model_sha256"]:
        raise ValueError("Frozen parent source/model mismatch")
    if sha(parent / "dev_predictions.parquet") != report["prediction_sha256"]:
        raise ValueError("Parent predictions changed")
    with (parent / "model.pkl").open("rb") as stream:
        fitted = pickle.load(stream)
    trie, confidence = load_dictionary(ROOT / "output/m0/v1/dictionary.parquet")
    return fitted, trie, confidence, report


def sources():
    paths = ["src/run_algo1.py", "src/slot_decoder.py", "note/bitrase-1.algo-1/protocol.md",
             "output/m1/v1/config.json", "output/m1/v1/model.pkl", "output/m1/v1/dev_predictions.parquet",
             "output/m1/v1/report.json", "src/verify_submission.py"]
    return {**model_sources(), **{p: sha(ROOT / p) for p in paths}}


def evaluate():
    if OUT.exists(): raise ValueError("Run exists; preserve original experiment")
    fitted, trie, confidence, parent_report = load_parent()
    config = {"experiment": "bitrase-1.algo-1", "parent": "bitrase-1", "methods": METHODS,
              "changed": "decoder only", "pretrained": False, "refit": False,
              "clip_probability": [1e-6, 1 - 1e-6], "sources": sources()}
    OUT.mkdir(parents=True)
    (OUT / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    saved = {r["document_id"]: triples(r["predicted"]) for r in pq.read_table(ROOT / "output/m1/v1/dev_predictions.parquet").to_pylist()}
    metrics = {m: Evaluation() for m in METHODS}
    predictions = {m: [] for m in METHODS}
    paired = {m: collections.Counter() for m in METHODS}
    groups = {m: collections.defaultdict(collections.Counter) for m in METHODS}
    documents = []
    start = time.monotonic()
    dev = PreparedData().load_dev("evaluation").to_pylist()
    if set(saved) != {r["document_id"] for r in dev}: raise ValueError("Dev coverage mismatch")
    for i, row in enumerate(dev, 1):
        text = row["full_text"]
        gold = triples(row["entities"])
        k = row["expected_entity_count"]
        pool = propose(text, trie, confidence)
        probabilities = score(text, pool, fitted["model"], fitted["hasher"])
        parent = parent_decode(probabilities, k)
        if parent != saved[row["document_id"]]: raise ValueError("Parent replay mismatch")
        actual = sum(a == b for a, b in zip(gold, parent))
        group = "all_gold_available" if all(s in pool for s in gold) else "missing_gold_candidates"
        doc = {"document_id": row["document_id"], "gold_slots": k, "parent_correct": actual, "group": group}
        for method in METHODS:
            pred = decode(probabilities, k, method)
            if len(pred) != k or any(s not in pool for s in pred): raise ValueError("Decoder contract broken")
            metrics[method].add(text, gold, pred, pool)
            predictions[method].append({"document_id": row["document_id"], "predicted": [{"start": a, "end": b, "label": label} for a, b, label in pred]})
            correct = sum(a == b for a, b in zip(gold, pred))
            delta = correct - actual
            paired[method].update({"better_documents": int(delta > 0), "worse_documents": int(delta < 0),
                                   "tied_documents": int(delta == 0), "changed_documents": int(pred != parent),
                                   "gained_slots": max(0, delta), "lost_slots": max(0, -delta), "net_slots": delta})
            groups[method][group].update({"documents": 1, "gold_slots": k, "parent_correct": actual, "correct": correct, "delta": delta})
            doc[method + "_correct"] = correct
        documents.append(doc)
        if i % 1500 == 0: print(f"Decoded {i}/{len(dev)} dev docs", flush=True)
    reports = {m: metrics[m].report() for m in METHODS}
    best = max(METHODS, key=lambda m: reports[m]["slot_accuracy"])
    winner = best if reports[best]["slot_accuracy"] > parent_report["dev"]["slot_accuracy"] else "parent"
    hashes = {}
    for method in METHODS:
        file = OUT / f"{method}_dev_predictions.parquet"
        pq.write_table(pa.Table.from_pylist(predictions[method]), file, compression="zstd")
        hashes[method] = sha(file)
    if sources() != config["sources"]: raise ValueError("Sources changed during evaluation")
    result = {"selected": winner, "parent_dev": parent_report["dev"], "dev": reports,
              "paired_documents": {m: dict(c) for m, c in paired.items()},
              "groups": {m: {g: dict(c) for g, c in gs.items()} for m, gs in groups.items()},
              "prediction_hashes": hashes, "parent_replayed_exactly": True, "seconds": time.monotonic() - start}
    (OUT / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUT / "paired_documents.json").write_text(json.dumps(documents, indent=2), encoding="utf-8")
    print(json.dumps({"selected": winner, "metrics": {m: {"slot_accuracy": r["slot_accuracy"], "entity_f1": r["entity_micro"]["f1"], "exact_docs": r["counts"]["exact_documents"]} for m, r in reports.items()}, "paired": result["paired_documents"], "groups": result["groups"]}, indent=2))


def submission():
    report = json.loads((OUT / "report.json").read_text())
    config = json.loads((OUT / "config.json").read_text())
    if sources() != config["sources"]: raise ValueError("Experiment source mismatch")
    selected = report["selected"]
    if selected not in METHODS: raise ValueError("Challenger not accepted; use parent CSV")
    target = OUT / "submission.csv"
    if target.exists(): raise ValueError("Submission already exists")
    fitted, trie, confidence, parent = load_parent()
    data = PreparedData()
    test = data.load_test().to_pylist()
    slots = data.load_test_slots().to_pylist()
    results = {}
    for i, row in enumerate(test, 1):
        pool = propose(row["full_text"], trie, confidence)
        probabilities = score(row["full_text"], pool, fitted["model"], fitted["hasher"])
        pred = decode(probabilities, row["expected_entity_count"], selected)
        results[row["document_id"]] = [format_prediction(s) for s in pred]
        if i % 5000 == 0: print(f"Decoded {i}/{len(test)} test docs", flush=True)
    final = [{"row_id": s["row_id"], "document_id": s["document_id"], "Predicted": results[s["document_id"]][s["slot"] - 1]} for s in slots]
    validate_rows(final, {r["document_id"]: r["full_text"] for r in test}, {r["document_id"]: r["expected_entity_count"] for r in test})
    if sources() != config["sources"]: raise ValueError("Sources changed during inference")
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["row_id", "Predicted"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(final)
    manifest = {"experiment": "bitrase-1.algo-1", "decoder": selected, "rows": len(final), "documents": len(test),
                "sha256": sha(target), "parent_model_sha256": parent["model_sha256"],
                "test_sha256": sha(ROOT / "output/prepared/v1/test.parquet"),
                "slots_sha256": sha(ROOT / "output/prepared/v1/test_slots.parquet"), "pretrained": False,
                "uploaded": False, "upload_owner": "Vian"}
    (OUT / "submission_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-only", action="store_true")
    args = parser.parse_args()
    submission() if args.submission_only else evaluate()
