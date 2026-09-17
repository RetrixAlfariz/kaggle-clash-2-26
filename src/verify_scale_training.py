"""Independent, train-only validation of the algo-3 training artifacts.

Run after ``training_complete.json`` is written. This verifier never loads dev,
holdout, or test data and never evaluates EVAL labels.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/bitrase-2/algo-3"
REPORT = OUT / "independent_training_validation.json"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def metadata_identities(path: Path):
    """Read only identity fields from metadata; skip gold values lexically."""
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    size = len(text)
    pos = 0

    def ws(i):
        while i < size and text[i].isspace():
            i += 1
        return i

    def parse_value(i):
        i = ws(i)
        value, end = decoder.raw_decode(text, i)
        return value, end

    def skip_value(i):
        i = ws(i)
        if i >= size:
            raise ValueError("Unexpected end of metadata JSON")
        if text[i] == '"':
            _, end = decoder.raw_decode(text, i)
            return end
        if text[i] not in "[{":
            end = i
            while end < size and text[end] not in ",]}" and not text[end].isspace():
                end += 1
            if end == i:
                raise ValueError("Invalid scalar in metadata JSON")
            return end
        stack = ["}" if text[i] == "{" else "]"]
        j = i + 1
        in_string = escaped = False
        while j < size and stack:
            ch = text[j]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "{":
                stack.append("}")
            elif ch == "[":
                stack.append("]")
            elif ch in "]}":
                if not stack or ch != stack[-1]:
                    raise ValueError("Mismatched delimiters in metadata JSON")
                stack.pop()
            j += 1
        if stack or in_string:
            raise ValueError("Truncated metadata JSON value")
        return j

    pos = ws(pos)
    require(pos < size and text[pos] == "[", "Metadata JSON must be an array")
    pos = ws(pos + 1)
    while pos < size and text[pos] != "]":
        require(text[pos] == "{", "Metadata array item must be an object")
        pos = ws(pos + 1)
        identity = {}
        seen = set()
        while pos < size and text[pos] != "}":
            key, pos = parse_value(pos)
            require(isinstance(key, str) and key not in seen, "Invalid or duplicate metadata field")
            seen.add(key)
            pos = ws(pos)
            require(pos < size and text[pos] == ":", "Missing metadata field colon")
            pos = ws(pos + 1)
            if key in ("document_id", "group_id"):
                identity[key], pos = parse_value(pos)
            else:
                pos = skip_value(pos)
            pos = ws(pos)
            if pos < size and text[pos] == ",":
                pos = ws(pos + 1)
            else:
                break
        require(pos < size and text[pos] == "}", "Malformed metadata object")
        pos = ws(pos + 1)
        require("document_id" in identity and "group_id" in identity, "Missing metadata identity")
        yield identity
        if pos < size and text[pos] == ",":
            pos = ws(pos + 1)
        else:
            break
    require(pos < size and text[pos] == "]", "Malformed metadata array")
    require(ws(pos + 1) == size, "Trailing data after metadata JSON")


def main() -> None:
    complete_path = OUT / "training_complete.json"
    require(complete_path.is_file(), "Training is not complete; run this verifier after training_complete.json exists")
    completion = load_json(complete_path)
    require(completion.get("complete") is True, "Training completion marker is not complete")

    input_manifest = load_json(OUT / "training_inputs.json")
    required_inputs = {
        "src/train_scale_ranking.py",
        "src/neural_sequence.py",
        "src/span_probe.py",
        "src/run_span_probe.py",
        "src/sequence_crf.py",
        "src/structure_families.py",
        "src/audit_template_similarity.py",
        "src/prepared_loader.py",
        "note/bitrase-2/algo-3/protocol.md",
        "output/prepared/v1/train.parquet",
    }
    require(set(input_manifest) == required_inputs, "Training input provenance list differs from the frozen expected list")
    for relative, digest in input_manifest.items():
        require(sha(ROOT / relative) == digest, f"Training input hash changed: {relative}")
    require(completion.get("input_hashes_unchanged") is True, "Training did not attest unchanged inputs")

    assignments_path = OUT / "assignments.parquet"
    assignments_hash = sha(assignments_path)
    config = load_json(OUT / "config.json")
    require(config.get("split_sha256") == assignments_hash, "Config assignment hash mismatch")
    rows = pq.read_table(assignments_path).to_pylist()
    train_ids = pq.read_table(ROOT / "output/prepared/v1/train.parquet", columns=["document_id"]).column("document_id").to_pylist()
    require(len(train_ids) == len(set(train_ids)), "Prepared TRAIN document IDs are not unique")
    require(len(rows) == len(train_ids), "Assignment row count differs from prepared TRAIN")
    assignment_by_id = {}
    for row in rows:
        doc_id = row["document_id"]
        require(doc_id not in assignment_by_id, f"Duplicate assignment document: {doc_id}")
        require(row["role"] in ("FIT", "CAL", "EVAL"), f"Invalid role for {doc_id}")
        assignment_by_id[doc_id] = row
    require(set(assignment_by_id) == set(train_ids), "Assignments do not exactly cover prepared TRAIN IDs")

    grouping = load_json(OUT / "grouping.json")
    family_map = grouping.get("sampled_families")
    require(isinstance(family_map, dict) and family_map, "Grouping report has no sampled_families map")
    expected_role_by_group = {}
    ordered_groups = sorted(
        family_map,
        key=lambda group: (hashlib.sha256(f"20260917:{group}".encode()).digest(), group),
    )
    role_counts = {role: 0 for role in ("FIT", "CAL", "EVAL")}
    for group in ordered_groups:
        role = "CAL" if role_counts["CAL"] < 1000 else "EVAL" if role_counts["EVAL"] < 3000 else "FIT"
        expected_role_by_group[group] = role
        role_counts[role] += len(family_map[group])
    family_docs = set()
    for group, docs in family_map.items():
        require(docs, f"Empty family in grouping report: {group}")
        require(len(docs) == len(set(docs)), f"Duplicate documents inside family: {group}")
        for doc_id in docs:
            require(doc_id not in family_docs, f"Document appears in multiple families: {doc_id}")
            family_docs.add(doc_id)
            assigned = assignment_by_id.get(doc_id)
            require(assigned is not None, f"Grouped document missing from assignments: {doc_id}")
            require(assigned["group_id"] == group, f"Assignment family mismatch for {doc_id}")
            require(assigned["role"] == expected_role_by_group[group], f"Non-deterministic role assignment for {doc_id}")
    require(family_docs == set(train_ids), "Grouping families do not exactly cover prepared TRAIN")
    for group in family_map:
        roles = {assignment_by_id[doc_id]["role"] for doc_id in family_map[group]}
        require(len(roles) == 1, f"Family crosses role boundary: {group}")

    for filename, key in (("encoder_epoch4.pt", "encoder_sha256"), ("head_epoch2.pt", "head_sha256"), ("vocab.json", "vocab_sha256")):
        actual = sha(OUT / filename)
        require(completion.get(key) == actual, f"Completion hash mismatch: {filename}")

    role_summaries = {}
    for role in ("CAL", "EVAL"):
        role_path = OUT / role
        manifest = load_json(role_path / "manifest.json")
        require(manifest.get("complete") is True and manifest.get("split") == role, f"Invalid {role} score manifest")
        for filename, digest in manifest.get("files", {}).items():
            artifact = role_path / filename
            require(Path(filename).name == filename, f"Unsafe {role} artifact path: {filename}")
            require(artifact.is_file() and sha(artifact) == digest, f"{role} artifact hash mismatch: {filename}")
        role_rows = list(metadata_identities(role_path / "metadata.json"))
        role_ids = [r["document_id"] for r in role_rows]
        require(len(role_ids) == len(set(role_ids)), f"Duplicate {role} metadata IDs")
        expected_ids = {doc_id for doc_id, a in assignment_by_id.items() if a["role"] == role}
        require(set(role_ids) == expected_ids, f"{role} metadata IDs differ from frozen assignments")
        for row in role_rows:
            assignment = assignment_by_id[row["document_id"]]
            require(assignment["role"] == role and assignment["group_id"] == row["group_id"], f"{role} metadata family/role mismatch")
        require(manifest.get("documents") == len(role_ids), f"{role} manifest document count mismatch")
        role_summaries[role] = {"documents": len(role_ids), "manifest_sha256": sha(role_path / "manifest.json"),
                                "metadata_sha256": sha(role_path / "metadata.json")}

    counts = {role: sum(a["role"] == role for a in rows) for role in ("FIT", "CAL", "EVAL")}
    require(config.get("counts") == counts, "Config split counts differ from assignments")
    report = {
        "complete": True,
        "validation": "independent train-only artifact integrity, split reconstruction, and role coverage",
        "prepared_train_documents": len(train_ids),
        "family_count": len(family_map),
        "role_document_counts": counts,
        "assignments_sha256": assignments_hash,
        "training_inputs_verified": len(input_manifest),
        "training_complete_sha256": sha(complete_path),
        "checkpoints_and_vocab": {name: completion[key] for name, key in (
            ("encoder_epoch4.pt", "encoder_sha256"), ("head_epoch2.pt", "head_sha256"), ("vocab.json", "vocab_sha256"))},
        "frozen_role_artifacts": role_summaries,
        "data_access": {"prepared_train": "loaded for ID coverage only", "dev": False, "holdout": False, "test": False,
                        "CAL_EVAL_labels_used": False},
    }
    REPORT.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
