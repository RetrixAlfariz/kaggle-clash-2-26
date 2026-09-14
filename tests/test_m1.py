import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from train_m1 import propose, features, score, FeatureHasher, SGDClassifier


class M1Contracts(unittest.TestCase):
    def test_username_context_preserves_offsets(self):
        text = "Unicode é. Your username is 'rjohnson_prime'."
        candidates = propose(text, {}, {})
        expected = text.index("rjohnson_prime")
        self.assertIn((expected, expected + len("rjohnson_prime"), "USERNAME"), candidates)
        self.assertFalse(any(span[2] == "USERNAME" for span in propose("your username should be private", {}, {})))

    def test_boundary_context_features_distinguish_title_and_partial_identifier(self):
        text = "Dear Mr. Chen, username: chen_prime"
        full = features(text, (5, 13, "NAME"), .5)
        short = features(text, (9, 13, "NAME"), .5)
        self.assertEqual(full["NAME|starts_title"], 1)
        self.assertEqual(short["NAME|left_title"], 1)
        a = text.index("chen_prime")
        partial = features(text, (a, a + 4, "USERNAME"), .5)
        self.assertEqual(partial["USERNAME|right_identifier"], 1)

    def test_fresh_classifier_can_score_valid_probabilities(self):
        h = FeatureHasher(n_features=1024, input_type="dict")
        model = SGDClassifier(loss="log_loss", random_state=2026)
        x = h.transform([features("Alice", (0, 5, "NAME"), .8), features("wrong", (0, 5, "NAME"), .1)])
        model.partial_fit(x, np.array([1, 0]), classes=np.array([0, 1]))
        result = score("Alice", {(0, 5, "NAME"): .8}, model, h)
        self.assertGreaterEqual(result[0, 5, "NAME"], 0)
        self.assertLessEqual(result[0, 5, "NAME"], 1)
        self.assertEqual(score("", {}, model, h), {})


if __name__ == "__main__":
    unittest.main()
