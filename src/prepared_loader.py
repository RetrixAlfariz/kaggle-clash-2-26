"""Explicit, manifest-verified access to prepared model data.

This is a workflow guard against accidental leakage, not a security boundary.
Callers can deliberately bypass it by reading parquet directly.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "output" / "prepared" / "v1"
MODEL_FILES = {"train.parquet", "dev.parquet", "holdout.parquet", "test.parquet", "test_slots.parquet"}
PURPOSES = {
    "train": {"train_fitting"}, "dev": {"tuning", "evaluation"},
    "holdout": {"final_evaluation"}, "test": {"inference"}, "test_slots": {"inference"},
}
ENTITY = pa.struct([("label", pa.string()), ("start", pa.int32()),
                    ("end", pa.int32()), ("text", pa.string())])
EXPECTED_SCHEMAS = {
    "train": pa.schema([("document_id", pa.string()), ("full_text", pa.string()),
                         ("entities", pa.list_(ENTITY)), ("expected_entity_count", pa.int32())]),
    "dev": pa.schema([("document_id", pa.string()), ("full_text", pa.string()),
                      ("entities", pa.list_(ENTITY)), ("expected_entity_count", pa.int32())]),
    "holdout": pa.schema([("document_id", pa.string()), ("full_text", pa.string()),
                           ("entities", pa.list_(ENTITY)), ("expected_entity_count", pa.int32())]),
    "test": pa.schema([("document_id", pa.string()), ("full_text", pa.string()),
                        ("expected_entity_count", pa.int32())]),
    "test_slots": pa.schema([("row_order", pa.int64()), ("row_id", pa.string()),
                              ("document_id", pa.string()), ("slot", pa.int64())]),
}


class PreparedDataError(ValueError):
    pass


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PreparedData:
    """Load only explicitly named, manifest-verified model-facing artifacts."""

    def __init__(self, root: str | Path = DEFAULT_ROOT):
        self.root = Path(root).resolve()
        self.manifest_path = self.root / "manifest.json"
        if not self.manifest_path.is_file():
            raise PreparedDataError(f"Missing prepared manifest: {self.manifest_path}")
        try:
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PreparedDataError("Invalid prepared manifest") from exc
        files = self.manifest.get("files")
        if not isinstance(files, dict) or not MODEL_FILES.issubset(files):
            raise PreparedDataError("Manifest does not contain the complete model file allowlist")
        unexpected = set(files) - MODEL_FILES - {"split_assignments.parquet", "metadata_audit_only.parquet", "weak_pool_not_for_supervised_training.parquet"}
        if unexpected:
            raise PreparedDataError(f"Unexpected manifest files: {sorted(unexpected)}")

    def _load(self, name: str, purpose: str):
        if name not in MODEL_FILES:
            raise PreparedDataError(f"Artifact is not model-facing: {name}")
        stem = name.removesuffix(".parquet")
        allowed = PURPOSES[stem]
        if purpose not in allowed:
            raise PreparedDataError(f"Purpose {purpose!r} is not allowed for {stem}; allowed={sorted(allowed)}")
        path = (self.root / name).resolve()
        if path.parent != self.root or not path.is_file():
            raise PreparedDataError(f"Missing or unsafe prepared artifact: {name}")
        info = self.manifest["files"].get(name)
        if not isinstance(info, dict) or _hash(path) != info.get("sha256"):
            raise PreparedDataError(f"Hash mismatch: {name}")
        table = pq.read_table(path)
        expected = EXPECTED_SCHEMAS[stem]
        if not table.schema.equals(expected):
            raise PreparedDataError(f"Unexpected model schema: {name}")
        if table.num_rows != info.get("rows"):
            raise PreparedDataError(f"Row-count mismatch: {name}")
        if str(table.schema) != info.get("schema"):
            raise PreparedDataError(f"Schema mismatch: {name}")
        return table

    def load_train(self, purpose: str = "train_fitting"):
        return self._load("train.parquet", purpose)

    def load_dev(self, purpose: str = "tuning"):
        return self._load("dev.parquet", purpose)

    def load_holdout(self, purpose: str = "development", *, allow_final_evaluation: bool = False):
        if purpose == "final_evaluation" and not allow_final_evaluation:
            raise PreparedDataError("Holdout requires allow_final_evaluation=True")
        return self._load("holdout.parquet", purpose)

    def load_test(self, purpose: str = "inference"):
        return self._load("test.parquet", purpose)

    def load_test_slots(self, purpose: str = "inference"):
        return self._load("test_slots.parquet", purpose)
