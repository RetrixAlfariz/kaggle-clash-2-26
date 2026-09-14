import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from submit_m0 import validate_rows


class SubmissionContracts(unittest.TestCase):
    def test_valid_sorted_nonoverlapping_rows(self):
        texts = {"d": "Alice 2026"}; counts = {"d": 2}
        rows = [{"row_id": "d_01", "document_id": "d", "Predicted": "NAME:0:5"},
                {"row_id": "d_02", "document_id": "d", "Predicted": "DATE:6:10"}]
        validate_rows(rows, texts, counts)

    def test_invalid_count_overlap_and_bounds(self):
        base = {"row_id": "d_01", "document_id": "d", "Predicted": "NAME:0:5"}
        with self.assertRaises(ValueError): validate_rows([base], {"d": "Alice"}, {"d": 2})
        rows = [base, {"row_id": "d_02", "document_id": "d", "Predicted": "DATE:4:6"}]
        with self.assertRaises(ValueError): validate_rows(rows, {"d": "Alice!"}, {"d": 2})
        rows[1]["Predicted"] = "DATE:5:99"
        with self.assertRaises(ValueError): validate_rows(rows, {"d": "Alice!"}, {"d": 2})


if __name__ == "__main__": unittest.main()
