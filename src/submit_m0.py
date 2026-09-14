# /// script
# requires-python = ">=3.13"
# dependencies = ["pyarrow>=18"]
# ///
"""Generate the frozen M0 test submission without refitting or holdout access."""
from __future__ import annotations

import csv, hashlib, json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from baseline_probe import END, decode
from prepared_loader import PreparedData
from run_m0 import candidates_for

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "submissions" / "m0"


def sha(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def load_dictionary(path):
    trie, confidence = {}, {}
    for row in pq.read_table(path).to_pylist():
        phrase = row["phrase"]
        confidence[phrase] = (row["label"], float(row["score"]))
        node = trie
        for char in phrase:
            node = node.setdefault(char, {})
        node[END] = phrase
    return trie, confidence


def format_prediction(span):
    start, end, label = span
    return f"{label}:{start}:{end}"


def validate_rows(rows, texts, counts):
    if [r["row_id"] for r in rows] != list(dict.fromkeys(r["row_id"] for r in rows)):
        raise ValueError("Duplicate submission row IDs")
    by_doc = defaultdict(list)
    for row in rows:
        doc = row["document_id"]
        if doc not in texts or doc not in counts:
            raise ValueError("Unknown submission document")
        by_doc[doc].append(row)
        parts = row["Predicted"].split(":")
        if len(parts) != 3 or parts[0] not in {"NAME", "DATE", "EMAIL", "PHONE", "ADDRESS", "USERNAME", "JOB_TITLE"}:
            raise ValueError("Invalid prediction format")
        try:
            a, b = int(parts[1]), int(parts[2])
        except ValueError as exc:
            raise ValueError("Invalid prediction offsets") from exc
        if not 0 <= a < b <= len(texts[doc]):
            raise ValueError("Prediction offset out of bounds")
    if set(by_doc) != set(texts) or any(len(v) != counts[d] for d, v in by_doc.items()):
        raise ValueError("Submission document or slot coverage mismatch")
    for doc, doc_rows in by_doc.items():
        spans = [(int(r["Predicted"].split(":")[1]), int(r["Predicted"].split(":")[2])) for r in doc_rows]
        if spans != sorted(spans) or any(b > c for (_, b), (c, _) in zip(spans, spans[1:])):
            raise ValueError(f"Unsorted or overlapping predictions for {doc}")


def main():
    out = OUT.resolve()
    if out.exists():
        raise ValueError(f"Output already exists; choose a new version: {out}")
    data = PreparedData()
    test = data.load_test("inference").to_pylist()
    slot_rows = data.load_test_slots("inference").to_pylist()
    texts = {r["document_id"]: r["full_text"] for r in test}
    counts = defaultdict(int)
    for row in slot_rows:
        counts[row["document_id"]] += 1
    trie, confidence = load_dictionary(ROOT / "output/m0/v1/dictionary.parquet")
    predictions, blocked = [], []
    for row in test:
        doc, text, expected = row["document_id"], row["full_text"], row["expected_entity_count"]
        spans = decode(candidates_for(text, trie, confidence), expected)
        if len(spans) != expected:
            blocked.append({"document_id": doc, "expected_count": expected, "decoded_count": len(spans)})
            continue
        predictions_by_slot = [format_prediction(span) for span in spans]
        for slot, predicted in enumerate(predictions_by_slot, 1):
            predictions.append({"document_id": doc, "slot": slot, "Predicted": predicted})
    report = {"run": "m0-v1-test-submission", "test_documents": len(test), "expected_rows": len(slot_rows),
              "decoded_rows": len(predictions), "blocked_document_count": len(blocked), "blocked_documents": blocked,
              "dictionary_sha256": sha(ROOT / "output/m0/v1/dictionary.parquet"), "test_manifest_sha256": sha(ROOT / "output/prepared/v1/manifest.json"),
              "model": "frozen output/m0/v1/dictionary.parquet + run_m0 candidates_for + baseline_probe.decode; no refit", "holdout_read": False}
    out.mkdir(parents=True)
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if blocked:
        raise ValueError(f"Cannot produce complete submission: {len(blocked)} documents have infeasible K")
    by_key = {(r["document_id"], r["slot"]): r["Predicted"] for r in predictions}
    final = [{"row_id": r["row_id"], "Predicted": by_key[(r["document_id"], r["slot"])]} for r in slot_rows]
    validate_rows([{**r, "document_id": slot_rows[i]["document_id"]} for i, r in enumerate(final)], texts, counts)
    with (out / "submission.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["row_id", "Predicted"])
        writer.writeheader(); writer.writerows(final)
    report.update({"csv_sha256": sha(out / "submission.csv"), "csv_rows": len(final)})
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("csv_rows", "blocked_document_count", "csv_sha256")}, indent=2))


if __name__ == "__main__":
    main()
