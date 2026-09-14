"""Contract tests: uv run --with pyarrow python -m unittest discover -s tests -v."""
import sys
import unittest
import tempfile
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prepare_data import assemble, assign_splits, group_hash, validate_entities, verify_parquet, verify_artifacts, ARTIFACT_NAMES


class PreparationContracts(unittest.TestCase):
    def test_serialized_faults_rejected(self):
        fixtures = [
            ("train.parquet", [{"document_id": "a"}, {"document_id": "b"}], [], "row count"),
            ("train.parquet", [{"document_id": "a"}, {"document_id": "b"}], [{"document_id": "a"}, {"document_id": "a"}], "duplicate"),
            ("train.parquet", [{"document_id": "a"}], [{"document_id": "z"}], "coverage"),
            ("split_assignments.parquet", [{"document_id": "a", "partition": "train"}], [{"document_id": "a", "partition": "dev"}], "content"),
            ("metadata_audit_only.parquet", [{"document_id": "a", "n_chars": 4}], [{"document_id": "a", "n_chars": 5}], "content"),
            ("test_slots.parquet", [{"row_id": "a_01"}, {"row_id": "a_02"}], [{"row_id": "a_02"}, {"row_id": "a_01"}], "order"),
            ("train.parquet", [{"document_id": "a", "full_text": "é\n"}], [{"document_id": "a", "full_text": "e\n"}], "content"),
        ]
        with tempfile.TemporaryDirectory() as folder:
            for name, expected, actual, message in fixtures:
                with self.subTest(name=name, fault=message):
                    schema = pa.Table.from_pylist(expected).schema
                    path = Path(folder) / name
                    pq.write_table(pa.Table.from_pylist(actual, schema=schema), path)
                    with self.assertRaisesRegex(ValueError, message):
                        verify_parquet(path, expected, schema)

    def test_schema_change_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "train.parquet"
            rows = [{"document_id": "a", "extra": "unexpected"}]
            pq.write_table(pa.Table.from_pylist(rows), path)
            with self.assertRaisesRegex(ValueError, "schema"):
                verify_parquet(path, [{"document_id": "a"}], pa.schema([("document_id", pa.string())]))

    def test_nested_gold_schema_roundtrip(self):
        from prepare_data import GOLD_SCHEMA
        rows = [{"document_id": "a", "full_text": "Alice", "entities": [{"label": "NAME", "start": 0, "end": 5, "text": "Alice"}], "expected_entity_count": 1}]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "train.parquet"
            pq.write_table(pa.Table.from_pylist(rows, schema=GOLD_SCHEMA), path)
            verify_parquet(path, rows, GOLD_SCHEMA)

    def test_all_eight_artifacts_required_and_verified(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            expected = {}
            for name in ARTIFACT_NAMES:
                key = "row_id" if name == "test_slots.parquet" else "document_id"
                rows = [{key: "a"}]
                table = pa.Table.from_pylist(rows)
                expected[name] = (rows, table.schema)
                pq.write_table(table, folder / name)
            verify_artifacts(folder, expected)
            pq.write_table(pa.table({"document_id": ["wrong"]}), folder / "metadata_audit_only.parquet")
            with self.assertRaisesRegex(ValueError, "metadata_audit_only.*coverage"):
                verify_artifacts(folder, expected)
            (folder / "metadata_audit_only.parquet").unlink()
            with self.assertRaisesRegex(ValueError, "inventory"):
                verify_artifacts(folder, expected)

    def test_shuffled_blank_lines_and_unicode_keep_python_offsets(self):
        text = assemble([(2, "é😀"), (0, "A"), (1, "")],
                        {"n_chars": 5, "n_segments": 3, "n_words": 2, "n_nonempty_segments": 2})
        self.assertEqual(text, "A\n\né😀")
        validate_entities(text, [{"label": "NAME", "start": 3, "end": 5, "text": "é😀"}])

    def test_duplicate_gap_and_metadata_mismatch_fail(self):
        meta = {"n_chars": 3, "n_segments": 2, "n_words": 2, "n_nonempty_segments": 2}
        for parts in [[(0, "A"), (0, "B")], [(0, "A"), (2, "B")], [(0, "AA"), (1, "B")]]:
            with self.subTest(parts=parts), self.assertRaises(ValueError):
                assemble(parts, meta)

    def test_boundaries_preserve_whitespace_and_multiline_addresses(self):
        text = " Dr. A\nStreet\nCity"
        spans = [{"label": "NAME", "start": 0, "end": 6, "text": " Dr. A"},
                 {"label": "ADDRESS", "start": 7, "end": len(text), "text": "Street\nCity"}]
        validate_entities(text, spans)
        self.assertTrue(spans[0]["text"].startswith(" "))

    def test_invalid_bounds_label_slice_and_nested_spans_fail(self):
        valid = {"label": "NAME", "start": 0, "end": 3, "text": "ABC"}
        cases = [[{**valid, "end": 4}], [{**valid, "label": "PERSON"}],
                 [{**valid, "text": "abc"}], [valid, {"label": "NAME", "start": 1, "end": 2, "text": "B"}]]
        for spans in cases:
            with self.subTest(spans=spans), self.assertRaises(ValueError):
                validate_entities("ABC", spans)

    def test_deterministic_split_groups_and_historical_exposure(self):
        rows = [{"document_id": f"d{i}", "duplicate_group": f"g{i}",
                 "stratum": "EMAIL|IT|1-10" if i < 30 else "LETTER|LEGAL|11-15"} for i in range(60)]
        rows.append({**rows[5], "document_id": "duplicate"})
        first, _ = assign_splits(rows, {"d0"}, {"d1"}, {"d2"}, seed=2026)
        second, _ = assign_splits(list(reversed(rows)), {"d0"}, {"d1"}, {"d2"}, seed=2026)
        self.assertEqual(first, second)
        self.assertEqual(first["d0"], "train")
        self.assertEqual(first["d1"], "dev")
        self.assertNotEqual(first["d2"], "holdout")
        self.assertEqual(first["d5"], first["duplicate"])
        self.assertEqual(set(first.values()), {"train", "dev", "holdout"})
        self.assertEqual(len(first), len(rows))

    def test_duplicate_group_conflicting_probe_roles_fail(self):
        rows = [{"document_id": d, "duplicate_group": "same", "stratum": "s"} for d in ["a", "b"]]
        with self.assertRaises(ValueError):
            assign_splits(rows, {"a"}, {"b"}, set())

    def test_normalization_is_only_a_duplicate_key(self):
        self.assertEqual(group_hash("A\n\nB"), group_hash("A B"))
        self.assertNotEqual("A\n\nB", "A B")


if __name__ == "__main__":
    unittest.main()
