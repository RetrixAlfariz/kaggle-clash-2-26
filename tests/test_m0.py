import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ner_evaluation import Evaluation
from baseline_probe import decode
from run_m0 import fit, candidates_for


class M0Contracts(unittest.TestCase):
    def test_missing_first_entity_shifts_slots_but_keeps_set_matches(self):
        gold = [(0, 1, "NAME"), (2, 3, "DATE"), (4, 5, "EMAIL")]
        predicted = gold[1:]
        e = Evaluation()
        e.add("A B C", gold, predicted, dict.fromkeys(predicted, .9))
        r = e.report()
        self.assertEqual(r["slot_accuracy"], 0)
        self.assertEqual(r["entity_micro"]["precision"], 1)
        self.assertAlmostEqual(r["entity_micro"]["recall"], 2/3)
        self.assertEqual(r["output_stage"]["exact_span_wrong_slot"], 2)

    def test_label_and_boundary_errors_do_not_earn_exact_credit(self):
        e = Evaluation()
        gold = [(0, 2, "NAME"), (3, 6, "ADDRESS")]
        pred = [(0, 2, "USERNAME"), (3, 5, "ADDRESS")]
        e.add("AB CDE", gold, pred, dict.fromkeys(pred, .8))
        r = e.report()
        self.assertEqual(r["entity_micro"]["f1"], 0)
        self.assertEqual(r["boundary_micro"]["f1"], .5)
        self.assertEqual(r["output_stage"], {"same_boundary_wrong_label": 1, "overlap_same_label": 1})
        self.assertEqual(sum(r["candidate_stage"].values()), 2)

    def test_decoder_fixed_count_and_insufficient_candidates(self):
        choices = {(0, 5, "NAME"): .99, (0, 2, "NAME"): .8, (3, 5, "NAME"): .8}
        self.assertEqual(decode(choices, 2), [(0, 2, "NAME"), (3, 5, "NAME")])
        self.assertEqual(len(decode(choices, 3)), 2)
        self.assertEqual(decode({}, 2), [])

    def test_invalid_and_overlapping_predictions_are_rejected(self):
        for predicted in [[(0, 4, "NAME")], [(0, 2, "NAME"), (1, 3, "DATE")]]:
            with self.assertRaises(ValueError):
                Evaluation().add("ABC", [], predicted, {})

    def test_train_fit_scores_and_predictions_need_no_dev_labels(self):
        train = [{"full_text": "Alice Alice", "entities": [{"start": 0, "end": 5, "label": "NAME", "text": "Alice"}]}]
        trie, dictionary = fit(train)
        self.assertEqual(dictionary[0]["occurrences"], 2)
        self.assertEqual(dictionary[0]["score"], .5)
        confidence = {r["phrase"]: (r["label"], r["score"]) for r in dictionary}
        found = candidates_for("Alice writes to new@example.com", trie, confidence)
        self.assertIn((0, 5, "NAME"), found)
        self.assertTrue(any(s[2] == "EMAIL" for s in found))


if __name__ == "__main__":
    unittest.main()
