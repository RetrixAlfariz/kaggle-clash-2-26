# /// script
# requires-python = ">=3.13"
# dependencies = ["pyarrow>=18"]
# ///
"""Audit raw reconstruction and annotation/slot integrity without modifying inputs."""
import collections
import csv
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import pyarrow
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/data_understanding"
LABELS = {"NAME", "DATE", "EMAIL", "PHONE", "JOB_TITLE", "ADDRESS", "USERNAME"}


def read_csv(path, columns):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != columns:
            raise ValueError(f"Unexpected columns in {path.name}: {reader.fieldnames}")
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"Malformed CSV row in {path.name}, record {reader.line_num}")
            yield row


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    assert "\n".join(["A", "", "é😀"]) == "A\n\né😀"
    assert len("A\n\né😀") == 5
    report = {"executed_at": datetime.now(timezone.utc).isoformat(),
              "python": platform.python_version(), "pyarrow": pyarrow.__version__,
              "method": "Full local-file scan; CSV empty strings preserved, numeric indices sorted, one LF between rows, Python Unicode code-point offsets.",
              "checks": [], "splits": {}, "sources": []}

    def check(name, failures, denominator, examples=None):
        report["checks"].append({"name": name, "status": "pass" if failures == 0 else "review",
                                 "failures": failures, "denominator": denominator,
                                 "examples": (examples or [])[:8]})

    all_texts = {}
    hashes = {}
    for split in ("train", "test"):
        groups = collections.defaultdict(list)
        for row in read_csv(ROOT / f"Data/{split}/{split}_segments.csv", ["document_id", "segment_index", "text"]):
            groups[row["document_id"]].append((int(row["segment_index"]), row["text"]))
        meta = pq.read_table(ROOT / f"Data/{split}/{split}_metadata.parquet").to_pylist()
        meta_ids = [row["document_id"] for row in meta]
        check(f"{split}: unique metadata document IDs", len(meta_ids) - len(set(meta_ids)), len(meta_ids))
        missing = set(groups) ^ set(meta_ids)
        check(f"{split}: metadata/segments document join", len(missing), len(set(groups) | set(meta_ids)), sorted(missing))
        by_id = {row["document_id"]: row for row in meta}
        texts = {}
        stats = collections.Counter()
        mismatch = collections.defaultdict(list)
        exact = collections.defaultdict(list)
        normalized = collections.defaultdict(list)
        for doc, rows in groups.items():
            rows.sort()
            indices = [i for i, _ in rows]
            if len(indices) != len(set(indices)):
                mismatch["unique segment indices"].append(doc)
            if indices != list(range(len(rows))):
                mismatch["contiguous zero-based segment indices"].append(doc)
            text = "\n".join(t for _, t in rows)
            texts[doc] = text
            stats["segments"] += len(rows)
            stats["empty_segments"] += sum(t == "" for _, t in rows)
            stats["whitespace_only_nonempty_segments"] += sum(bool(t) and not t.strip() for _, t in rows)
            stats["documents_with_non_ascii"] += any(ord(c) > 127 for c in text)
            stats["documents_with_cr"] += "\r" in text
            stats["documents_with_embedded_segment_newline"] += any("\n" in t for _, t in rows)
            measured = {"n_chars": len(text), "n_segments": len(rows), "n_words": len(text.split()),
                        "n_nonempty_segments": sum(bool(t.strip()) for _, t in rows)}
            if doc in by_id:
                for field, actual in measured.items():
                    if actual != by_id[doc][field]:
                        mismatch[field].append({"document_id": doc, "metadata": by_id[doc][field], "reconstructed": actual})
            exact[digest(text)].append(doc)
            normalized[digest(" ".join(text.split()))].append(doc)
        for field in ("unique segment indices", "contiguous zero-based segment indices", "n_chars", "n_segments", "n_words", "n_nonempty_segments"):
            check(f"{split}: {field}", len(mismatch[field]), len(texts), mismatch[field])
        duplicate_groups = [ids for ids in exact.values() if len(ids) > 1]
        report["splits"][split] = {"documents": len(texts), **stats,
            "exact_duplicate_groups": duplicate_groups,
            "exact_duplicate_excess_documents": sum(len(g) - 1 for g in duplicate_groups),
            "whitespace_normalized_duplicate_excess_documents": sum(len(g) - 1 for g in normalized.values())}
        all_texts[split] = texts
        hashes[split] = (exact, normalized)
        print(f"Audited reconstruction: {split}, {len(texts):,} documents", flush=True)

    overlap_ids = set(all_texts["train"]) & set(all_texts["test"])
    check("train/test document IDs disjoint", len(overlap_ids), sum(map(len, all_texts.values())), sorted(overlap_ids))
    for index, name in enumerate(("exact", "whitespace_normalized")):
        common = hashes["train"][index].keys() & hashes["test"][index].keys()
        report[f"train_test_{name}_overlap"] = {"shared_text_groups": len(common),
            "test_documents": sum(len(hashes["test"][index][h]) for h in common)}

    annotations = collections.defaultdict(list)
    bad = collections.defaultdict(list)
    label_rows = 0
    for row in read_csv(ROOT / "Data/train/train_labels.csv", ["document_id", "label", "start_offset", "end_offset"]):
        label_rows += 1
        doc, label = row["document_id"], row["label"]
        a, b = int(row["start_offset"]), int(row["end_offset"])
        if doc not in all_texts["train"]:
            bad["label document joins"].append(row)
            continue
        text = all_texts["train"][doc]
        if not 0 <= a < b <= len(text):
            bad["label offset bounds"].append(row)
        if label not in LABELS:
            bad["label enum"].append(row)
        if not text[a:b].strip():
            bad["nonempty entity text"].append(row)
        annotations[doc].append((a, b, label))
    duplicate_spans = overlap_spans = tied_bounds = 0
    for doc, spans in annotations.items():
        spans.sort()
        duplicate_spans += len(spans) - len(set(spans))
        tied_bounds += len(spans) - len({(a, b) for a, b, _ in spans})
        max_end = -1
        for a, b, _ in spans:
            overlap_spans += a < max_end
            max_end = max(max_end, b)
    for name in ("label document joins", "label offset bounds", "label enum", "nonempty entity text"):
        check(name, len(bad[name]), label_rows, bad[name])
    check("duplicate gold span rows", duplicate_spans, label_rows)
    check("gold spans with shared bounds", tied_bounds, label_rows)
    check("gold span overlaps (running maximum end)", overlap_spans, label_rows)
    no_gold = all_texts["train"].keys() - annotations.keys()
    report["training_labels"] = {"rows": label_rows, "documents_without_labels": sorted(no_gold)}
    duplicate_conflicts = []
    for ids in report["splits"]["train"]["exact_duplicate_groups"]:
        if len({tuple(annotations[d]) for d in ids}) > 1:
            duplicate_conflicts.append(ids)
    report["duplicate_text_annotation_conflicts"] = duplicate_conflicts

    slots = collections.defaultdict(list)
    row_ids = set()
    duplicate_row_ids = malformed = 0
    slot_rows = 0
    sample_values = collections.Counter()
    for row in read_csv(ROOT / "Data/sample_submission.csv", ["row_id", "Predicted"]):
        slot_rows += 1
        rid = row["row_id"]
        duplicate_row_ids += rid in row_ids
        row_ids.add(rid)
        doc, suffix = rid.rsplit("_", 1)
        try:
            slot = int(suffix)
            malformed += suffix != f"{slot:02d}" or slot < 1
            slots[doc].append(slot)
        except ValueError:
            malformed += 1
        sample_values[row["Predicted"]] += 1
    check("unique submission row IDs", duplicate_row_ids, slot_rows)
    check("submission slot suffix format", malformed, slot_rows)
    check("submission contiguous slots starting at 01", sum(sorted(s) != list(range(1, len(s) + 1)) for s in slots.values()), len(slots))
    slot_missing = set(slots) ^ set(all_texts["test"])
    check("test/slot document coverage", len(slot_missing), len(all_texts["test"]), sorted(slot_missing))
    report["submission"] = {"rows": slot_rows, "documents": len(slots), "min_slots": min(map(len, slots.values())),
        "max_slots": max(map(len, slots.values())), "placeholder_values": sample_values.most_common(5),
        "interpretation": "Slot count is exposed test entity cardinality, not hidden labels. Preserve source row order when writing submissions."}

    weak_ids = set()
    weak_counts = collections.Counter()
    weak_invalid = []
    weak_count = weak_duplicates = 0
    with (ROOT / "Data/extra/weak_labeled_documents.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            weak_count += 1
            doc, text = row["document_id"], row["full_text"]
            weak_duplicates += doc in weak_ids
            weak_ids.add(doc)
            for entity in row["weak_entities"]:
                a, b = entity["start"], entity["end"]
                weak_counts[entity["label"]] += 1
                if not (0 <= a < b <= len(text) and text[a:b] == entity["text"] and entity["label"] in LABELS):
                    weak_invalid.append({"document_id": doc, "entity": entity})
    check("weak document ID uniqueness", weak_duplicates, weak_count)
    check("weak entity bounds/text/label structural validity", len(weak_invalid), sum(weak_counts.values()), weak_invalid)
    report["weak_structural_audit"] = {"documents": weak_count, "label_counts": weak_counts,
        "gold_or_test_id_overlap": len(weak_ids & (set(all_texts["train"]) | set(all_texts["test"]))),
        "limitation": "Structural validity does not establish semantic annotation correctness or completeness. Weak data is not used as gold."}

    for path in sorted((ROOT / "Data").rglob("*")):
        if path.is_file():
            h = hashlib.sha256()
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            report["sources"].append({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": h.hexdigest()})
    report["summary"] = {"checks": len(report["checks"]), "passed": sum(c["status"] == "pass" for c in report["checks"]),
                         "review": sum(c["status"] != "pass" for c in report["checks"])}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "reconstruction.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "splits": report["splits"], "submission": report["submission"],
                      "weak": report["weak_structural_audit"], "conflicting_duplicate_text": duplicate_conflicts}, indent=2))


if __name__ == "__main__":
    main()
