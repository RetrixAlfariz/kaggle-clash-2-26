import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from structure_families import group_and_sample


def _base_text(number: int) -> str:
    family = chr(ord("a") + number)
    unique = " ".join(f"u{family}x{i}" for i in range(40))
    return f"document family{family} {unique}"


class StructureFamilyTests(unittest.TestCase):
    def test_exact_normalized_and_numeric_masked_templates_join(self):
        template = "sender 12345 reference alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon"
        rows = [
            {"document_id": "a", "full_text": template},
            {"document_id": "b", "full_text": "  SENDER   12345 reference alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon  "},
            {"document_id": "c", "full_text": template.replace("12345", "998877")},
            {"document_id": "d", "full_text": template.replace("upsilon", "omegaalt")},
        ]
        rows += [{"document_id": f"u{i:02}", "full_text": _base_text(i)} for i in range(12)]
        selected, groups, folds, report = group_and_sample(
            rows, sample_size=len(rows), max_sample_size=20, min_families=2
        )
        selected_by_id = {rows[idx]["document_id"]: (groups[pos], folds[pos]) for pos, idx in enumerate(selected)}
        self.assertEqual(selected_by_id["a"][0], selected_by_id["b"][0])
        self.assertEqual(selected_by_id["a"][0], selected_by_id["c"][0])
        self.assertEqual(selected_by_id["a"][0], selected_by_id["d"][0])
        self.assertEqual(report["exact_duplicate_links"], 1)
        self.assertGreaterEqual(report["verified_candidate_edges"], 1)
        self.assertEqual(selected_by_id["a"][1], selected_by_id["c"][1])

    def test_sampling_is_deterministic_and_keeps_whole_families(self):
        rows = [{"document_id": "dup1", "full_text": "same exact text"},
                {"document_id": "dup2", "full_text": " SAME   EXACT TEXT "}]
        rows += [{"document_id": f"doc{i:02}", "full_text": _base_text(i)} for i in range(12)]
        args = dict(sample_size=8, max_sample_size=12, min_families=2)
        first = group_and_sample(rows, **args)
        second = group_and_sample(rows, **args)
        self.assertEqual(first, second)
        selected, groups, folds, report = first
        members = {group: [selected[i] for i, value in enumerate(groups) if value == group] for group in set(groups)}
        self.assertIn([0, 1], list(members.values()))
        self.assertGreaterEqual(len(selected), 8)
        self.assertEqual(len(selected), len(groups))
        self.assertEqual(len(selected), len(folds))
        for group in set(groups):
            self.assertEqual(len({folds[i] for i, value in enumerate(groups) if value == group}), 1)
        self.assertTrue(all(count > 0 for count in report["fold_family_counts"]))

    def test_rejects_sample_overflow_and_insufficient_families(self):
        rows = [{"document_id": f"large{i}", "full_text": "same"} for i in range(4)]
        with self.assertRaisesRegex(ValueError, "exceeding max"):
            group_and_sample(rows, sample_size=1, max_sample_size=2, min_families=1)
        with self.assertRaisesRegex(ValueError, "need at least"):
            group_and_sample(rows[:2], sample_size=1, max_sample_size=10, min_families=2)


if __name__ == "__main__":
    unittest.main()
