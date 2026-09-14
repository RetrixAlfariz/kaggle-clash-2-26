# /// script
# requires-python = "==3.13.2"
# dependencies = ["pyarrow==25.0.1"]
# ///
"""Prepare immutable, verified document data; no normalization or tokenization.

Run: uv run src/prepare_data.py
Use --output for a new version. Existing different builds are never overwritten.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from audit_reconstruction import LABELS, read_csv

ROOT = Path(__file__).resolve().parents[1]
SEED = 2026
ENTITY = pa.struct([("label", pa.string()), ("start", pa.int32()),
                    ("end", pa.int32()), ("text", pa.string())])
GOLD_SCHEMA = pa.schema([("document_id", pa.string()), ("full_text", pa.string()),
                        ("entities", pa.list_(ENTITY)), ("expected_entity_count", pa.int32())])
TEST_SCHEMA = pa.schema([("document_id", pa.string()), ("full_text", pa.string()),
                        ("expected_entity_count", pa.int32())])
WEAK_SCHEMA = pa.schema([("document_id", pa.string()), ("full_text", pa.string()),
                        ("weak_entities", pa.list_(ENTITY)), ("annotation_source", pa.string()),
                        ("overlap_partitions", pa.list_(pa.string()))])


def file_hash(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def group_hash(text):
    # Group key only. The stored text is NEVER normalized.
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def assemble(parts, metadata):
    parts = sorted(parts)
    if [i for i, _ in parts] != list(range(len(parts))):
        raise ValueError("Segment indices must be unique, contiguous and zero-based")
    text = "\n".join(t for _, t in parts)
    actual = {"n_chars": len(text), "n_segments": len(parts), "n_words": len(text.split()),
              "n_nonempty_segments": sum(bool(t.strip()) for _, t in parts)}
    for field, value in actual.items():
        if metadata[field] != value:
            raise ValueError(f"Metadata mismatch: {field}, expected {metadata[field]}, got {value}")
    return text


def validate_entities(text, entities):
    last_end = -1
    for entity in entities:
        a, b = entity["start"], entity["end"]
        if entity["label"] not in LABELS or not 0 <= a < b <= len(text):
            raise ValueError("Invalid entity label or offset")
        if text[a:b] != entity["text"]:
            raise ValueError("Entity text disagrees with its character slice")
        if a < last_end:
            raise ValueError("Gold entities overlap or are not sorted")
        last_end = b


def load_documents(split):
    metadata = pq.read_table(ROOT / f"Data/{split}/{split}_metadata.parquet").to_pylist()
    by_id = {r["document_id"]: r for r in metadata}
    if len(by_id) != len(metadata):
        raise ValueError("Duplicate metadata IDs")
    segments = collections.defaultdict(list)
    for row in read_csv(ROOT / f"Data/{split}/{split}_segments.csv", ["document_id", "segment_index", "text"]):
        segments[row["document_id"]].append((int(row["segment_index"]), row["text"]))
    if set(segments) != set(by_id):
        raise ValueError("Metadata/segment document coverage differs")
    documents = {doc: assemble(parts, by_id[doc]) for doc, parts in segments.items()}
    return documents, by_id


def count_band(count):
    return "1-10" if count <= 10 else "11-15" if count <= 15 else "16+"


def assign_splits(records, prior_train, prior_dev, reviewed, seed=SEED):
    """Allocate whole normalized-text groups, stratified on metadata and gold count.

    Prior probe train/dev membership is hard; other reviewed examples cannot enter
    holdout. Hash ordering supplies deterministic pseudo-random ordering.
    """
    groups = collections.defaultdict(list)
    for record in records:
        groups[record["duplicate_group"]].append(record)
    strata = collections.defaultdict(list)
    split = {}
    for key, members in groups.items():
        member_ids = {r["document_id"] for r in members}
        if member_ids & prior_train and member_ids & prior_dev:
            raise ValueError("A duplicate group crosses the historical probe split")
        protected = "train" if member_ids & prior_train else "dev" if member_ids & prior_dev else None
        if protected:
            split.update({d: protected for d in member_ids})
        elif member_ids & reviewed:
            protected = "dev"
            split.update({d: protected for d in member_ids})
        # Identical normalized texts should share semantics, but grouping wins if metadata differs.
        representative = min(members, key=lambda r: r["document_id"])
        stratum = representative["stratum"]
        strata[stratum].append((key, members, protected))
    allocation = []
    for stratum, members in sorted(strata.items()):
        total = sum(len(rows) for _, rows, _ in members)
        target = max(1, round(total * .10)) if total >= 10 else 0
        used_dev = sum(len(rows) for _, rows, fixed in members if fixed == "dev")
        free = sorted([(key, rows) for key, rows, fixed in members if fixed is None],
                      key=lambda item: hashlib.sha256(f"{seed}:{item[0]}".encode()).hexdigest())
        used_holdout = 0
        for key, rows in free:
            size = len(rows)
            if used_holdout < target:
                destination = "holdout"
                used_holdout += size
            elif used_dev < target:
                destination = "dev"
                used_dev += size
            else:
                destination = "train"
            split.update({r["document_id"]: destination for r in rows})
        allocation.append({"stratum": stratum, "documents": total, "target_dev_or_holdout": target,
                           "actual_dev": used_dev, "actual_holdout": used_holdout})
    return split, allocation


def collect_ids(value):
    if isinstance(value, dict):
        if isinstance(value.get("document_id"), str):
            yield value["document_id"]
        for child in value.values():
            yield from collect_ids(child)
    elif isinstance(value, list):
        for child in value:
            yield from collect_ids(child)


def write_parquet(folder, name, rows, schema=None):
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, folder / name, compression="zstd", row_group_size=1024)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_parquet(path, expected_rows, expected_schema):
    """Validate complete serialized content, including missing/duplicate rows."""
    loaded = pq.read_table(path)
    # Parquet renames list children item -> element; Arrow semantic equality
    # preserves field names/types/nullability while accepting that round trip.
    require(loaded.schema.equals(expected_schema) and loaded.schema.metadata == expected_schema.metadata,
            f"{path.name}: schema mismatch")
    require(loaded.num_rows == len(expected_rows), f"{path.name}: row count mismatch")
    key = "row_id" if path.name == "test_slots.parquet" else "document_id"
    rows = loaded.to_pylist()
    expected_ids = [r[key] for r in expected_rows]
    actual_ids = [r[key] for r in rows]
    require(len(set(expected_ids)) == len(expected_ids), f"{path.name}: duplicate source IDs")
    require(len(set(actual_ids)) == len(actual_ids), f"{path.name}: duplicate serialized IDs")
    require(set(actual_ids) == set(expected_ids), f"{path.name}: ID coverage mismatch")
    for index, (actual, expected) in enumerate(zip(rows, expected_rows)):
        require(actual == expected, f"{path.name}: content/order mismatch at row {index}, ID {expected[key]}")


ARTIFACT_NAMES = {"train.parquet", "dev.parquet", "holdout.parquet", "test.parquet",
                  "test_slots.parquet", "split_assignments.parquet", "metadata_audit_only.parquet",
                  "weak_pool_not_for_supervised_training.parquet"}


def verify_artifacts(folder, expected):
    require(set(expected) == ARTIFACT_NAMES, "Expected artifact inventory mismatch")
    require({p.name for p in folder.glob("*.parquet")} == ARTIFACT_NAMES, "Serialized artifact inventory mismatch")
    for name, (rows, schema) in expected.items():
        verify_parquet(folder / name, rows, schema)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New build path; historical v1 is preserved")
    args = parser.parse_args()
    destination = args.output.resolve()
    require(platform.python_version() == "3.13.2" and pa.__version__ == "25.0.1", "Preparation requires Python 3.13.2 and PyArrow 25.0.1")
    audit = ROOT / "output/data_understanding"
    baseline = ROOT / "output/baseline_probe/report.json"
    provenance = [ROOT / "Data" / path for path in (
        "train/train_segments.csv", "train/train_metadata.parquet", "train/train_labels.csv",
        "test/test_segments.csv", "test/test_metadata.parquet", "sample_submission.csv",
        "extra/weak_labeled_documents.jsonl", "README.txt")]
    provenance += [baseline, audit / "annotations_examples.json", audit / "annotations.json",
                   Path(__file__).resolve(), ROOT / "src/audit_reconstruction.py",
                   ROOT / "src/prepare_data.py.lock", ROOT / ".python-version"]
    sources = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in provenance}
    config = {"seed": SEED, "split_fraction_target": {"train": .8, "dev": .1, "holdout": .1},
              "stratification": ["channel", "domain", "entity_count_band"], "format_version": 1,
              "validation_revision": 2}
    if destination.exists():
        previous = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        if previous["sources"] != sources or previous["config"] != config:
            raise ValueError("Existing output belongs to a different build. Choose a new --output path.")
        require(set(previous["files"]) == ARTIFACT_NAMES, "Manifest artifact inventory mismatch")
        require({p.name for p in destination.glob("*.parquet")} == ARTIFACT_NAMES, "Build artifact inventory mismatch")
        for name, info in previous["files"].items():
            if file_hash(destination / name) != info["sha256"]:
                raise ValueError(f"Prepared artifact changed: {name}")
        print(f"Verified existing build; no files changed: {destination}")
        return

    train_text, train_meta = load_documents("train")
    test_text, test_meta = load_documents("test")
    if train_text.keys() & test_text.keys():
        raise ValueError("Train and test IDs overlap")
    entities = collections.defaultdict(list)
    for row in read_csv(ROOT / "Data/train/train_labels.csv", ["document_id", "label", "start_offset", "end_offset"]):
        d, a, b = row["document_id"], int(row["start_offset"]), int(row["end_offset"])
        entities[d].append({"label": row["label"], "start": a, "end": b, "text": train_text[d][a:b]})
    if entities.keys() != train_text.keys():
        raise ValueError("Gold annotation document coverage differs")
    for d, spans in entities.items():
        spans.sort(key=lambda e: (e["start"], e["end"]))
        validate_entities(train_text[d], spans)

    historical = json.loads(baseline.read_text(encoding="utf-8"))["split"]
    prior_train, prior_dev = set(historical["train"]), set(historical["validation"])
    if (prior_train | prior_dev) - train_text.keys() or prior_train & prior_dev:
        raise ValueError("Invalid historical probe membership")
    reviewed = set()
    for name in ("annotations_examples.json", "annotations.json"):
        reviewed.update(collect_ids(json.loads((audit / name).read_text(encoding="utf-8"))))
    # These two documents were printed during the first exploratory inspection.
    reviewed.update({"DOC_000002", "DOC_000003"})
    records = []
    for doc in sorted(train_text):
        m = train_meta[doc]
        records.append({"document_id": doc, "duplicate_group": group_hash(train_text[doc]),
                        "channel": m["channel"], "domain": m["domain"], "entity_count": len(entities[doc]),
                        "stratum": "|".join((m["channel"], m["domain"], count_band(len(entities[doc]))))})
    partition, strata = assign_splits(records, prior_train, prior_dev, reviewed)
    if set(partition) != set(train_text):
        raise ValueError("Incomplete split assignment")
    for doc in prior_train:
        require(partition[doc] == "train", f"Historical train role changed: {doc}")
    for doc in prior_dev:
        require(partition[doc] == "dev", f"Historical dev role changed: {doc}")
    for doc in reviewed & train_text.keys():
        require(partition[doc] != "holdout", f"Reviewed document entered holdout: {doc}")
    group_partitions = collections.defaultdict(set)
    for row in records:
        row["partition"] = partition[row["document_id"]]
        row["prior_probe_role"] = "train" if row["document_id"] in prior_train else "dev" if row["document_id"] in prior_dev else "none"
        row["reviewed_example"] = row["document_id"] in reviewed
        group_partitions[row["duplicate_group"]].add(row["partition"])
    require(all(len(parts) == 1 for parts in group_partitions.values()), "Duplicate group crosses partitions")
    test_groups = {group_hash(text) for text in test_text.values()}
    if test_groups & group_partitions.keys():
        raise ValueError("Normalized text overlap with competition test; review before preparing")
    group_partitions.update({g: {"test"} for g in test_groups})

    slots = []
    slots_by_doc = collections.defaultdict(list)
    seen_rows = set()
    for index, row in enumerate(read_csv(ROOT / "Data/sample_submission.csv", ["row_id", "Predicted"])):
        rid = row["row_id"]
        d, suffix = rid.rsplit("_", 1)
        number = int(suffix)
        if rid in seen_rows or d not in test_text or number < 1 or suffix != f"{number:02d}":
            raise ValueError("Invalid submission row")
        seen_rows.add(rid)
        slots_by_doc[d].append(number)
        slots.append({"row_order": index, "row_id": rid, "document_id": d, "slot": number})
    if set(slots_by_doc) != set(test_text):
        raise ValueError("Test slot coverage mismatch")
    require(all(sorted(numbers) == list(range(1, len(numbers) + 1)) for numbers in slots_by_doc.values()), "Noncontiguous submission slots")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".preparing-", dir=destination.parent))
    totals = {}
    expected_artifacts = {}

    def emit(name, rows, schema=None):
        expected_schema = schema if schema is not None else pa.Table.from_pylist(rows).schema
        expected_artifacts[name] = (rows, expected_schema)
        write_parquet(staging, name, rows, expected_schema)

    for part in ("train", "dev", "holdout"):
        rows = [{"document_id": d, "full_text": train_text[d], "entities": entities[d],
                 "expected_entity_count": len(entities[d])} for d in sorted(train_text) if partition[d] == part]
        emit(f"{part}.parquet", rows, GOLD_SCHEMA)
        totals[part] = {"documents": len(rows), "entities": sum(len(r["entities"]) for r in rows),
                        "labels": dict(collections.Counter(e["label"] for r in rows for e in r["entities"]))}
        print(f"Prepared {part}: {len(rows):,} documents", flush=True)
    test_rows = [{"document_id": d, "full_text": test_text[d], "expected_entity_count": len(slots_by_doc[d])} for d in sorted(test_text)]
    emit("test.parquet", test_rows, TEST_SCHEMA)
    emit("test_slots.parquet", slots)
    emit("split_assignments.parquet", records)
    metadata = [{**train_meta[d], "partition": partition[d]} for d in sorted(train_meta)]
    metadata += [{**test_meta[d], "partition": "test"} for d in sorted(test_meta)]
    emit("metadata_audit_only.parquet", metadata)

    weak_rows, weak_ids = [], set()
    weak_overlap = collections.Counter()
    with (ROOT / "Data/extra/weak_labeled_documents.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            d, text = row["document_id"], row["full_text"]
            if d in weak_ids or d in train_text or d in test_text:
                raise ValueError("Weak ID collision")
            weak_ids.add(d)
            weak_spans = [{"label": e["label"], "start": e["start"], "end": e["end"], "text": e["text"]} for e in row["weak_entities"]]
            # Weak annotations may overlap; do not impose gold sequence semantics.
            for e in weak_spans:
                if e["label"] not in LABELS or not 0 <= e["start"] < e["end"] <= len(text) or text[e["start"]:e["end"]] != e["text"]:
                    raise ValueError("Invalid weak annotation")
            overlap = sorted(group_partitions.get(group_hash(text), set()))
            weak_overlap.update(overlap)
            weak_rows.append({"document_id": d, "full_text": text, "weak_entities": weak_spans,
                              "annotation_source": row["annotation_source"], "overlap_partitions": overlap})
    emit("weak_pool_not_for_supervised_training.parquet", sorted(weak_rows, key=lambda r: r["document_id"]), WEAK_SCHEMA)

    verify_artifacts(staging, expected_artifacts)
    require(sum(v["documents"] for v in totals.values()) == len(train_text), "Gold document total mismatch")
    require(sum(v["entities"] for v in totals.values()) == sum(map(len, entities.values())), "Gold entity total mismatch")
    for path, expected in sources.items():
        if file_hash(ROOT / path) != expected:
            raise ValueError(f"Input changed during preparation: {path}")
    manifest = {"format_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                "runtime": {"python": platform.python_version(), "pyarrow": pa.__version__},
                "config": config, "sources": sources, "gold_partitions": totals,
                "test": {"documents": len(test_rows), "slots": len(slots)},
                "weak": {"documents": len(weak_rows), "normalized_text_overlap_by_partition": dict(weak_overlap), "used_for_training": False},
                "historical_exposure": {"prior_probe_train_preserved": len(prior_train), "prior_probe_dev_preserved": len(prior_dev),
                                        "reviewed_document_ids_excluded_from_holdout": len(reviewed & train_text.keys())},
                "stratum_allocation": strata, "files": {},
                "limitations": ["The entire corpus was profiled earlier: reserved holdout is not fully blind.",
                                "Only exact/whitespace-normalized duplicates are grouped; semantic near duplicates remain unaudited.",
                                "Split proportions are approximate because historical membership and whole groups take priority.",
                                "Gold labels are preserved, including suspected inconsistencies. No tokenization, truncation, BIO conversion or relabeling.",
                                "Expected entity counts mirror exposed competition slots; this is not a general-purpose unknown-count NER protocol.",
                                "Metadata is audit-only; weak labels are a separate unused pool, and absent weak labels are not negatives."]}
    for path in sorted(staging.glob("*.parquet")):
        manifest["files"][path.name] = {"sha256": file_hash(path), "rows": pq.read_metadata(path).num_rows,
                                       "schema": str(pq.read_schema(path))}
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    staging.rename(destination)
    print(json.dumps({"output": str(destination), "gold_partitions": totals, "test": manifest["test"], "weak": manifest["weak"]}, indent=2))


if __name__ == "__main__":
    main()
