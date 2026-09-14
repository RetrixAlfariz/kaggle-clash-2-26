import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prepared_loader import PreparedData, PreparedDataError


def make_fixture(root):
    schemas = {
        "train.parquet": pa.schema([("document_id", pa.string()), ("full_text", pa.string()), ("entities", pa.list_(pa.struct([("label", pa.string()), ("start", pa.int32()), ("end", pa.int32()), ("text", pa.string())]))), ("expected_entity_count", pa.int32())]),
        "dev.parquet": pa.schema([("document_id", pa.string()), ("full_text", pa.string()), ("entities", pa.list_(pa.struct([("label", pa.string()), ("start", pa.int32()), ("end", pa.int32()), ("text", pa.string())]))), ("expected_entity_count", pa.int32())]),
        "holdout.parquet": pa.schema([("document_id", pa.string()), ("full_text", pa.string()), ("entities", pa.list_(pa.struct([("label", pa.string()), ("start", pa.int32()), ("end", pa.int32()), ("text", pa.string())]))), ("expected_entity_count", pa.int32())]),
        "test.parquet": pa.schema([("document_id", pa.string()), ("full_text", pa.string()), ("expected_entity_count", pa.int32())]),
        "test_slots.parquet": pa.schema([("row_order", pa.int64()), ("row_id", pa.string()), ("document_id", pa.string()), ("slot", pa.int64())]),
    }
    rows = {"document_id": ["d"], "full_text": ["Alice"], "entities": [[{"label": "NAME", "start": 0, "end": 5, "text": "Alice"}]], "expected_entity_count": [1]}
    files = {}
    for name, schema in schemas.items():
        if name in {"test.parquet", "test_slots.parquet"}:
            table = pa.Table.from_pylist([{"document_id": "x", "full_text": "text", "expected_entity_count": 1}] if name == "test.parquet" else [{"row_order": 0, "row_id": "x_01", "document_id": "x", "slot": 1}], schema=schema)
        else:
            table = pa.Table.from_pydict(rows, schema=schema)
        pq.write_table(table, root / name)
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        files[name] = {"sha256": digest, "rows": table.num_rows, "schema": str(pq.read_schema(root / name))}
    (root / "manifest.json").write_text(json.dumps({"files": files}), encoding="utf-8")


class PreparedLoaderTests(unittest.TestCase):
    def test_explicit_purposes_and_holdout_gate(self):
        with tempfile.TemporaryDirectory() as d:
            make_fixture(Path(d)); loader = PreparedData(d)
            self.assertEqual(loader.load_train().num_rows, 1)
            self.assertEqual(loader.load_dev("evaluation").num_rows, 1)
            with self.assertRaises(PreparedDataError): loader.load_holdout()
            with self.assertRaises(PreparedDataError): loader.load_holdout("final_evaluation")
            self.assertEqual(loader.load_holdout("final_evaluation", allow_final_evaluation=True).num_rows, 1)
            self.assertEqual(loader.load_test().num_rows, 1)

    def test_hash_corruption_and_unsafe_purpose_fail(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); make_fixture(root)
            (root / "dev.parquet").write_bytes((root / "dev.parquet").read_bytes() + b"corruption")
            with self.assertRaises(PreparedDataError): PreparedData(root).load_dev()
            make_fixture(root)
            with self.assertRaises(PreparedDataError): PreparedData(root).load_train("evaluation")

    def test_model_loader_does_not_expose_audit_or_weak_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); make_fixture(root); loader = PreparedData(root)
            with self.assertRaises(PreparedDataError): loader._load("metadata_audit_only.parquet", "inference")
            self.assertNotIn("metadata_audit_only.parquet", {"train.parquet", "dev.parquet", "holdout.parquet", "test.parquet", "test_slots.parquet"})

    def test_independent_schema_rejects_manifest_updated_extra_or_wrong_type(self):
        for fault in ("source_system", "weak_entities", "wrong_type"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as d:
                root = Path(d); make_fixture(root)
                original = pq.read_table(root / "train.parquet")
                table = (original.set_column(3, "expected_entity_count", pa.array([1], type=pa.int64()))
                         if fault == "wrong_type" else original.append_column(fault, pa.array(["leak"])))
                path = root / "train.parquet"
                pq.write_table(table, path)
                manifest = json.loads((root / "manifest.json").read_text())
                manifest["files"][path.name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": table.num_rows, "schema": str(pq.read_schema(path))}
                (root / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaisesRegex(PreparedDataError, "Unexpected model schema"):
                    PreparedData(root).load_train()

    def test_manifest_count_and_schema_are_checked(self):
        for field, value, message in (("rows", 2, "Row-count"), ("schema", "wrong", "Schema mismatch")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as d:
                root = Path(d); make_fixture(root)
                manifest = json.loads((root / "manifest.json").read_text())
                manifest["files"]["train.parquet"][field] = value
                (root / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaisesRegex(PreparedDataError, message):
                    PreparedData(root).load_train()

    def test_fit_purposes_are_denied_for_dev_and_test(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); make_fixture(root); loader = PreparedData(root)
            with self.assertRaises(PreparedDataError): loader.load_dev("train_fitting")
            with self.assertRaises(PreparedDataError): loader.load_test("train_fitting")


if __name__ == "__main__": unittest.main()
