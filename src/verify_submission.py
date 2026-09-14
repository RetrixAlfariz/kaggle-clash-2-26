"""Read-only validation of a generated competition submission."""
from __future__ import annotations
import argparse, csv, hashlib, json
from collections import defaultdict
from pathlib import Path
from prepared_loader import PreparedData

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, default=ROOT / "output/submissions/m0/submission.csv")
    parser.add_argument("--report", type=Path, default=ROOT / "output/submissions/readiness.json")
    args = parser.parse_args()
    source = args.submission.resolve()
    sample = ROOT / "Data/sample_submission.csv"
    data = PreparedData()
    texts = {r["document_id"]: r["full_text"] for r in data.load_test("inference").to_pylist()}
    slots = data.load_test_slots("inference").to_pylist()
    with sample.open(encoding="utf-8-sig", newline="") as f: expected = list(csv.DictReader(f))
    with source.open(encoding="utf-8-sig", newline="") as f: actual = list(csv.DictReader(f))
    errors = []
    if [r["row_id"] for r in actual] != [r["row_id"] for r in expected]: errors.append("row_id order/content differs from sample_submission")
    if not actual or list(actual[0]) != ["row_id", "Predicted"]: errors.append("empty file or wrong header")
    counts = defaultdict(int)
    for s in slots: counts[s["document_id"]] += 1
    seen = set(); by_doc = defaultdict(list)
    for row in actual:
        rid, pred = row.get("row_id", ""), row.get("Predicted", "")
        if rid in seen: errors.append(f"duplicate row_id: {rid}")
        seen.add(rid)
        parts = pred.split(":")
        if len(parts) != 3 or parts[0] not in {"NAME", "DATE", "EMAIL", "PHONE", "ADDRESS", "USERNAME", "JOB_TITLE"}:
            errors.append(f"invalid prediction format: {rid}"); continue
        try: a, b = int(parts[1]), int(parts[2])
        except ValueError: errors.append(f"non-integer offsets: {rid}"); continue
        doc = rid.rsplit("_", 1)[0]; by_doc[doc].append((a, b))
        if doc not in texts or not 0 <= a < b <= len(texts[doc]): errors.append(f"out-of-bounds: {rid}")
    for doc, spans in by_doc.items():
        if len(spans) != counts.get(doc): errors.append(f"count mismatch: {doc}")
        if spans != sorted(spans): errors.append(f"unsorted spans: {doc}")
        if any(x[1] > y[0] for x, y in zip(spans, spans[1:])): errors.append(f"overlap: {doc}")
    if set(by_doc) != set(texts): errors.append("document coverage mismatch")
    report = {"submission": str(source), "csv_sha256": sha(source), "dictionary_sha256": sha(ROOT / "output/m0/v1/dictionary.parquet"),
              "sample_sha256": sha(sample), "rows": len(actual), "expected_rows": len(expected),
              "documents": len(by_doc), "expected_documents": len(texts), "errors": errors, "ready": not errors}
    out = args.report.resolve(); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors: raise SystemExit(1)

if __name__ == "__main__": main()
