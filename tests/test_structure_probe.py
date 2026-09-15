import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from structure_probe import fit_predict, weighted_auc, weighted_logloss


class StructureProbeTests(unittest.TestCase):
  def test_weighted_auc_counts_ties_as_half_and_respects_weights(self):
    y = np.array([1, 0, 1, 0])
    p = np.array([0.5, 0.5, 0.9, 0.1])
    self.assertAlmostEqual(weighted_auc(y, p), 0.875)
    self.assertAlmostEqual(weighted_auc([1, 0], [0.4, 0.4]), 0.5)
    self.assertAlmostEqual(weighted_auc([1, 0], [0.1, 0.9], [1, 3]), 0.0)


  def test_weighted_logloss_matches_weighted_formula_and_clips_edges(self):
    got = weighted_logloss([1, 0], [0.8, 0.25], [3, 1])
    expected = (3 * -np.log(0.8) - np.log(0.75)) / 4
    self.assertAlmostEqual(got, expected)
    self.assertTrue(np.isfinite(weighted_logloss([1, 0], [1.0, 0.0])))


  def test_fit_predict_learns_signal_and_reports_convergence(self):
    rng = np.random.default_rng(20260916)
    x = rng.normal(size=(400, 3))
    latent = 2.0 * x[:, 0] - 1.2 * x[:, 1]
    y = (latent + rng.normal(scale=0.5, size=latent.size) > 0).astype(np.int64)
    pred, diag = fit_predict(x, y, x[:20])
    self.assertEqual(pred.shape, (20,))
    self.assertTrue(np.all((pred > 0) & (pred < 1)))
    self.assertGreater(weighted_auc(y[:20], pred), 0.8)
    self.assertTrue(diag["converged"])
    self.assertLessEqual(diag["gradient_inf_norm"], 1e-8)


  def test_standardization_uses_training_rows_and_handles_constant_features(self):
    x = np.array([[0.0, 7.0], [1.0, 7.0], [2.0, 7.0], [3.0, 7.0]])
    y = np.array([0, 0, 1, 1])
    pred_a, diag_a = fit_predict(x, y, [[1.5, 7.0]])
    pred_b, diag_b = fit_predict(x, y, [[1.5, 7.0], [1e9, -1e9]])
    self.assertAlmostEqual(pred_a[0], pred_b[0], delta=1e-12)
    self.assertEqual(diag_a["constant_feature_count"], 1)
    self.assertEqual(diag_b["standardization_fit_rows"], len(y))


  def test_fit_predict_rejects_one_class_and_bad_weights(self):
    with self.assertRaisesRegex(ValueError, "both classes"):
        fit_predict([[0.0], [1.0]], [0, 0], [[0.5]])
    with self.assertRaisesRegex(ValueError, "weight_train"):
        fit_predict([[0.0], [1.0]], [0, 1], [[0.5]], [-1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
