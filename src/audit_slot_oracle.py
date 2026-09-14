# /// script
# requires-python = "==3.13.2"
# dependencies = ["pyarrow==25.0.1", "scikit-learn==1.9.1"]
# ///
"""Gold-assisted dev diagnostic only; never an inference/submission method."""
import bisect
import collections
import json
from pathlib import Path
import time

import pyarrow.parquet as pq
from train_m1 import propose, sha
from submit_m0 import load_dictionary
from prepared_loader import PreparedData
from ner_evaluation import triples

ROOT = Path(__file__).resolve().parents[1]


def slot_oracle(candidates, gold):
    """Maximum exact positional matches among exactly K compatible candidates.

    DP[i][k] = optimum using first i end-sorted candidates and exactly k picks.
    A selected interval is the kth chronological span, so reward compares gold[k-1].
    Non-overlap and positive lengths ensure end order equals start order for a path.
    Return None when exactly K cannot be achieved, rather than relaxing K.
    """
    items = sorted(set(candidates), key=lambda s: (s[1], s[0], s[2]))
    if any(a < 0 or b <= a for a, b, _ in items):
        raise ValueError("Invalid candidate interval")
    if list(gold) != sorted(gold) or any(a[1] > b[0] for a, b in zip(gold, gold[1:])):
        raise ValueError("Gold must be sorted and non-overlapping")
    count = len(gold)
    ends = [s[1] for s in items]
    impossible = -10**9
    table = [[0] + [impossible] * count]
    for i, span in enumerate(items):
        previous = bisect.bisect_right(ends, span[0], hi=i)
        row = table[-1].copy()
        for k in range(1, count + 1):
            if table[previous][k - 1] != impossible:
                row[k] = max(row[k], table[previous][k - 1] + int(span == gold[k - 1]))
        table.append(row)
    result = table[-1][count]
    return None if result == impossible else result


def main():
    out = ROOT / "output/bitrase-1/slot_oracle"
    if out.exists():
        raise ValueError("Diagnostic output already exists")
    config = json.loads((ROOT / "output/m1/v1/config.json").read_text())
    # Check frozen source identities without reading train or reserved holdout.
    sources = {p: h for p, h in config["sources"].items() if p != "output/prepared/v1/train.parquet"}
    for p, expected in sources.items():
        if sha(ROOT / p) != expected:
            raise ValueError(f"Frozen source mismatch: {p}")
    sources["src/audit_slot_oracle.py"] = sha(Path(__file__))
    prediction_file = ROOT / "output/m1/v1/dev_predictions.parquet"
    saved_report = json.loads((ROOT / "output/m1/v1/report.json").read_text())["dev"]
    sources["output/m1/v1/dev_predictions.parquet"] = sha(prediction_file)
    predictions = {r["document_id"]: triples(r["predicted"]) for r in pq.read_table(prediction_file).to_pylist()}
    trie, confidence = load_dictionary(ROOT / "output/m0/v1/dictionary.parquet")
    data = PreparedData().load_dev("evaluation").to_pylist()
    if set(predictions) != {r["document_id"] for r in data}:
        raise ValueError("Prediction coverage differs from dev")
    total = collections.Counter()
    missing_labels = collections.Counter()
    groups = collections.defaultdict(collections.Counter)
    documents = []
    start = time.monotonic()
    for i, row in enumerate(data, 1):
        gold = triples(row["entities"])
        pred = predictions[row["document_id"]]
        pool = propose(row["full_text"], trie, confidence)
        k = len(gold)
        if len(pred) != k or any(s not in pool for s in pred):
            raise ValueError("Baseline predictions must be feasible in reconstructed pool")
        if any(a[1] > b[0] for a, b in zip(pred, pred[1:])):
            raise ValueError("Baseline overlap")
        actual = sum(a == b for a, b in zip(gold, pred))
        oracle = slot_oracle(pool, gold)
        found = sum(s in pool for s in gold)
        if oracle is None or not actual <= oracle <= found <= k:
            raise ValueError("Oracle bound invariant failed")
        missing = [s for s in gold if s not in pool]
        missing_labels.update(s[2] for s in missing)
        group = "all_gold_available" if not missing else "missing_gold_candidates"
        counts = {"documents": 1, "gold_slots": k, "actual_correct": actual,
                  "oracle_correct": oracle, "candidate_gold_found": found,
                  "selection_gap": oracle - actual, "pool_ceiling_gap": k - oracle,
                  "direct_missing_gold": k - found, "additional_slot_constraint_gap": found - oracle,
                  "actual_exact_documents": int(actual == k), "oracle_exact_documents": int(oracle == k),
                  "documents_with_selection_gap": int(oracle > actual)}
        total.update(counts)
        groups[group].update(counts)
        documents.append({"document_id": row["document_id"], "candidates": len(pool),
                          **{key: value for key, value in counts.items() if key != "documents"},
                          "missing_by_label": dict(collections.Counter(s[2] for s in missing))})
        if i % 1500 == 0:
            print(f"Oracle evaluated {i}/{len(data)} dev docs", flush=True)
    if total["actual_correct"] != saved_report["counts"]["slot_correct"] or total["candidate_gold_found"] != saved_report["counts"]["candidate_gold_found"]:
        raise ValueError("Reconstructed baseline metrics disagree with frozen report")
    for p, expected in sources.items():
        if sha(ROOT / p) != expected:
            raise ValueError(f"Source changed during diagnostic: {p}")
    metrics = {"actual_slot_accuracy": total["actual_correct"] / total["gold_slots"],
               "oracle_slot_accuracy": total["oracle_correct"] / total["gold_slots"],
               "selection_gap_pp": 100 * total["selection_gap"] / total["gold_slots"],
               "pool_ceiling_gap_pp": 100 * total["pool_ceiling_gap"] / total["gold_slots"]}
    report = {"baseline": "bitrase-1", "purpose": "gold-assisted diagnostic, not deployable prediction",
              "objective": "maximum exact-slot matches at exactly K non-overlapping candidate spans",
              "split": "prepared v1 dev only", "counts": dict(total), "metrics": metrics,
              "groups": {key: dict(value) for key, value in groups.items()},
              "missing_gold_by_label": dict(missing_labels), "source_sha256": sources,
              "seconds": time.monotonic() - start}
    out.mkdir(parents=True)
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "documents.json").write_text(json.dumps(documents, indent=2), encoding="utf-8")
    print(json.dumps({"counts": dict(total), "metrics": metrics, "groups": report["groups"], "missing": dict(missing_labels)}, indent=2))


if __name__ == "__main__":
    main()
