# /// script
# requires-python = ">=3.13"
# dependencies = ["pyarrow==25.0.1"]
# ///
"""Independent full readback audit; never imports preparation implementation."""
import collections
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "output/prepared/v1"
OUT = ROOT / "output/preparation_audit"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)


def normalized(text):
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    checks = []

    def check(name, failures, units):
        checks.append({"name": name, "failed_units": failures, "scoped_units": units})

    manifest = json.loads((BUILD / "manifest.json").read_text())
    check("source_hashes", sum(sha(ROOT / p) != h for p, h in manifest["sources"].items()), len(manifest["sources"]))
    check("artifact_hashes", sum(sha(BUILD / p) != v["sha256"] for p, v in manifest["files"].items()), len(manifest["files"]))
    assignment_rows = pq.read_table(BUILD / "split_assignments.parquet").to_pylist()
    assignments = {r["document_id"]: r for r in assignment_rows}
    check("unique_assignment_ids", len(assignment_rows) - len(assignments), len(assignment_rows))
    texts, rawmeta = {}, {}
    for split in ("train", "test"):
        segments = collections.defaultdict(list)
        for r in csv_rows(ROOT / f"Data/{split}/{split}_segments.csv"):
            segments[r["document_id"]].append((int(r["segment_index"]), r["text"]))
        metadata_rows = pq.read_table(ROOT / f"Data/{split}/{split}_metadata.parquet").to_pylist()
        meta = {r["document_id"]: r for r in metadata_rows}
        check(f"{split}_raw_metadata_uniqueness", len(metadata_rows) - len(meta), len(metadata_rows))
        check(f"{split}_raw_segment_coverage", len(set(meta) ^ set(segments)), len(meta))
        bad_indices = bad_metadata = 0
        for d, parts in segments.items():
            parts.sort()
            bad_indices += [i for i, _ in parts] != list(range(len(parts)))
            text = "\n".join(s for _, s in parts)
            texts[d] = text
            m = meta[d]
            bad_metadata += any((m["n_chars"] != len(text), m["n_words"] != len(text.split()), m["n_segments"] != len(parts), m["n_nonempty_segments"] != sum(bool(s.strip()) for _, s in parts)))
        check(f"{split}_segment_indices", bad_indices, len(meta))
        check(f"{split}_metadata_counts", bad_metadata, len(meta))
        rawmeta[split] = meta
    check("raw_train_test_id_overlap", len(set(rawmeta["train"]) & set(rawmeta["test"])), len(texts))
    gold = collections.defaultdict(list)
    for r in csv_rows(ROOT / "Data/train/train_labels.csv"):
        d = r["document_id"]
        start, end = int(r["start_offset"]), int(r["end_offset"])
        gold[d].append({"label": r["label"], "start": start, "end": end, "text": texts[d][start:end]})
    for entities in gold.values():
        entities.sort(key=lambda e: (e["start"], e["end"]))
    check("assignment_id_coverage", len(set(assignments) ^ set(rawmeta["train"])), len(rawmeta["train"]))
    partitions, distributions, group_parts = {}, {}, collections.defaultdict(set)
    seen = []
    for part in ("train", "dev", "holdout"):
        rows = pq.read_table(BUILD / f"{part}.parquet").to_pylist()
        ids = [r["document_id"] for r in rows]
        expected = {d for d, a in assignments.items() if a["partition"] == part}
        check(f"{part}_id_coverage", len(set(ids) ^ expected) + len(ids) - len(set(ids)), len(expected))
        bad_text = bad_entities = bad_counts = bad_spans = 0
        labels = collections.Counter()
        categories = collections.Counter()
        for r in rows:
            d = r["document_id"]
            partitions[d] = part
            bad_text += r["full_text"] != texts[d]
            bad_entities += r["entities"] != gold[d]
            bad_counts += r["expected_entity_count"] != len(gold[d])
            prev = 0
            for e in r["entities"]:
                bad_spans += not (prev <= e["start"] < e["end"] <= len(r["full_text"]) and r["full_text"][e["start"]:e["end"]] == e["text"] and e["label"] in {"NAME", "DATE", "EMAIL", "PHONE", "ADDRESS", "USERNAME", "JOB_TITLE"})
                prev = e["end"]
                labels[e["label"]] += 1
            m = rawmeta["train"][d]
            categories[m["channel"] + " / " + m["domain"]] += 1
            group_parts[normalized(r["full_text"])].add(part)
        for name, bad in (("text", bad_text), ("entities", bad_entities), ("entity_counts", bad_counts)):
            check(f"{part}_{name}", bad, len(rows))
        check(f"{part}_span_contract", bad_spans, sum(labels.values()))
        distributions[part] = {"documents": len(rows), "entities": sum(labels.values()), "labels": dict(labels), "channel_domain": dict(categories)}
        seen.extend(ids)
    check("gold_global_id_coverage", len(set(seen) ^ set(rawmeta["train"])) + len(seen) - len(set(seen)), len(rawmeta["train"]))
    check("normalized_group_isolation", sum(len(v) != 1 for v in group_parts.values()), len(group_parts))
    check("normalized_train_test_overlap", len(set(group_parts) & {normalized(texts[d]) for d in rawmeta["test"]}), len(rawmeta["test"]))
    bad_assign = 0
    for d, a in assignments.items():
        m = rawmeta["train"][d]
        bad_assign += a["partition"] != partitions[d] or a["duplicate_group"] != normalized(texts[d]) or a["channel"] != m["channel"] or a["domain"] != m["domain"] or a["entity_count"] != len(gold[d])
    check("assignment_content", bad_assign, len(assignments))
    check("historical_train_membership", sum(a["partition"] != "train" for a in assignments.values() if a["prior_probe_role"] == "train"), sum(a["prior_probe_role"] == "train" for a in assignments.values()))
    check("historical_dev_membership", sum(a["partition"] != "dev" for a in assignments.values() if a["prior_probe_role"] == "dev"), sum(a["prior_probe_role"] == "dev" for a in assignments.values()))
    check("reviewed_holdout_exclusion", sum(a["partition"] == "holdout" for a in assignments.values() if a["reviewed_example"]), sum(a["reviewed_example"] for a in assignments.values()))
    def reviewed_ids(value):
        if isinstance(value, dict):
            found = {value["document_id"]} if "document_id" in value else set()
            for child in value.values():
                found.update(reviewed_ids(child))
            return found
        if isinstance(value, list):
            return set().union(*(reviewed_ids(child) for child in value))
        return set()
    reviewed = {"DOC_000002", "DOC_000003"}
    for name in ("annotations_examples.json", "annotations.json"):
        reviewed.update(reviewed_ids(json.loads((ROOT / "output/data_understanding" / name).read_text())))
    reviewed &= assignments.keys()
    check("reviewed_flags_against_saved_examples", len(reviewed ^ {d for d, a in assignments.items() if a["reviewed_example"]}), len(reviewed))
    baseline = json.loads((ROOT / "output/baseline_probe/report.json").read_text())["split"]
    for role, key in (("train", "train"), ("dev", "validation")):
        expected_ids = set(baseline[key])
        check(f"historical_{role}_flags_against_probe", len(expected_ids ^ {d for d, a in assignments.items() if a["prior_probe_role"] == role}), len(expected_ids))
    source_slots = list(csv_rows(ROOT / "Data/sample_submission.csv"))
    slots = pq.read_table(BUILD / "test_slots.parquet").to_pylist()
    expected_slots = []
    slot_counts = collections.Counter()
    for i, r in enumerate(source_slots):
        d, s = r["row_id"].rsplit("_", 1)
        expected_slots.append({"row_order": i, "row_id": r["row_id"], "document_id": d, "slot": int(s)})
        slot_counts[d] += 1
    check("slot_readback_exact_order", sum(a != b for a, b in zip(slots, expected_slots)) + abs(len(slots) - len(expected_slots)), len(expected_slots))
    test = pq.read_table(BUILD / "test.parquet").to_pylist()
    test_ids = [r["document_id"] for r in test]
    check("test_id_coverage", len(set(test_ids) ^ set(rawmeta["test"])) + len(test_ids) - len(set(test_ids)), len(rawmeta["test"]))
    check("test_text_counts", sum(r["full_text"] != texts[r["document_id"]] or r["expected_entity_count"] != slot_counts[r["document_id"]] for r in test), len(test))
    metadata = pq.read_table(BUILD / "metadata_audit_only.parquet").to_pylist()
    expected_meta = {d: {**m, "partition": partitions[d]} for d, m in rawmeta["train"].items()}
    expected_meta.update({d: {**m, "partition": "test"} for d, m in rawmeta["test"].items()})
    meta_ids = [r["document_id"] for r in metadata]
    check("metadata_id_coverage", len(set(meta_ids) ^ set(expected_meta)) + len(meta_ids) - len(set(meta_ids)), len(expected_meta))
    check("metadata_content", sum(r != expected_meta.get(r["document_id"]) for r in metadata), len(metadata))
    for d in rawmeta["test"]:
        group_parts[normalized(texts[d])].add("test")
    weak = pq.read_table(BUILD / "weak_pool_not_for_supervised_training.parquet").to_pylist()
    with (ROOT / "Data/extra/weak_labeled_documents.jsonl").open(encoding="utf-8") as stream:
        weak_raw = {r["document_id"]: r for r in map(json.loads, stream)}
    weak_ids = [r["document_id"] for r in weak]
    check("weak_id_coverage", len(set(weak_ids) ^ set(weak_raw)) + len(weak_ids) - len(set(weak_ids)), len(weak_raw))
    bad_weak = 0
    overlap = collections.Counter()
    for r in weak:
        raw = weak_raw[r["document_id"]]
        expected = {k: raw[k] for k in ("document_id", "full_text", "weak_entities", "annotation_source")}
        expected["overlap_partitions"] = sorted(group_parts.get(normalized(raw["full_text"]), set()))
        bad_weak += r != expected
        overlap.update(r["overlap_partitions"])
    check("weak_content_and_overlap_flags", bad_weak, len(weak))
    check("model_facing_metadata_exclusion", sum(bool({"source_system", "ingested_at"} & set(pq.read_schema(BUILD / f"{p}.parquet").names)) for p in ("train", "dev", "holdout", "test")), 4)
    rebuilt = OUT / "rebuild"
    if (rebuilt / "manifest.json").exists():
        check("fresh_rebuild_parquet_bytes", sum(sha(rebuilt / p) != sha(BUILD / p) for p in manifest["files"]), len(manifest["files"]))
    result = {"created_at": datetime.now(timezone.utc).isoformat(), "audit_code_sha256": sha(Path(__file__)), "build_manifest_sha256": sha(BUILD / "manifest.json"), "checks": checks, "failed_checks": sum(c["failed_units"] > 0 for c in checks), "distributions": distributions, "weak_overlap": dict(overlap), "limits": ["Saved reviewed IDs cannot reconstruct unrecorded prior human exposure.", "No semantic near-duplicate detection, tokenizer validation, or predictive modelling."]}
    (OUT / "data_checks.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"checks": len(checks), "failed_checks": result["failed_checks"], "distributions": {k: {"documents": v["documents"], "entities": v["entities"]} for k, v in distributions.items()}, "weak_overlap": dict(overlap)}, indent=2))
    if result["failed_checks"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
