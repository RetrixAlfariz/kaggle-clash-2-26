import itertools
import math
import unittest

import numpy as np

from src.posterior_math import fixed_k_posterior, posterior_statistics
from src.slot_mbr import decode


def exhaustive(n, pairs, energies, k, tau):
    worlds = []
    m, labels = energies.shape
    for ids in itertools.combinations(range(m), k):
        ordered = sorted(ids, key=lambda i: (pairs[i, 1], pairs[i, 0]))
        if any(pairs[a, 1] > pairs[b, 0] for a, b in zip(ordered, ordered[1:])):
            continue
        for ls in itertools.product(range(labels), repeat=k):
            seq = list(zip(ordered, ls))
            score = sum(energies[i, lab] / tau for i, lab in seq)
            worlds.append((seq, score))
    assert worlds
    scores = np.array([score for _, score in worlds])
    top = scores.max()
    logz = float(top + np.log(np.exp(scores - top).sum()))
    probs = np.exp(scores - logz)
    mu = np.zeros((k, m, labels), dtype=np.float64)
    for (seq, _), prob in zip(worlds, probs):
        for slot, (i, lab) in enumerate(seq):
            mu[slot, i, lab] += prob
    entropy = float(-np.sum(probs * np.log(probs)))
    return logz, entropy, math.log(len(worlds)), mu


class PosteriorMathTests(unittest.TestCase):
    def test_tiny_exhaustive_and_existing_decoder_agree(self):
        pairs = np.array([[0, 1], [1, 2], [2, 3], [0, 2], [1, 3]], dtype=np.int64)
        energies = np.array([[.5, 1], [1.5, -1.5], [-.5, .5], [-1, 0], [-.5, -1]], dtype=np.float64)
        for tau in (.5, 1., 2.):
            expected = exhaustive(3, pairs, energies, 2, tau)
            logz, entropy, log_count, mu = fixed_k_posterior(3, pairs, energies, 2, tau)
            np.testing.assert_allclose((logz, entropy, log_count), expected[:3], rtol=0, atol=1e-12)
            np.testing.assert_allclose(mu, expected[3], rtol=0, atol=1e-12)
            _, decoder_diag = decode(3, pairs, energies / tau, 2, return_marginals=True)
            self.assertAlmostEqual(logz, decoder_diag["logZ"], places=12)
            np.testing.assert_allclose(mu, decoder_diag["slot_marginals"], rtol=0, atol=1e-12)
            self.assertAlmostEqual(float(mu.sum(axis=(1, 2)).min()), 1.0, places=12)

    def test_equal_scores_count_and_uniform_marginals(self):
        pairs = np.array([[0, 1], [1, 2], [2, 3], [0, 2], [1, 3]], dtype=np.int64)
        energies = np.zeros((len(pairs), 3))
        z, h, log_count, mu = fixed_k_posterior(3, pairs, energies, 2, 1)
        self.assertAlmostEqual(z, log_count, places=12)
        self.assertAlmostEqual(h, log_count, places=12)
        np.testing.assert_allclose(mu.sum(axis=(1, 2)), 1.0, atol=1e-12)
        result = posterior_statistics(3, pairs, energies, 2, 2)
        self.assertEqual(set(result), {"logZ", "entropy", "log_count", "mu"})

    def test_zero_k_is_single_empty_structure(self):
        result = posterior_statistics(0, np.empty((0, 2), dtype=np.int64), np.empty((0, 2)), 0, 1)
        self.assertEqual(result["logZ"], 0.0)
        self.assertEqual(result["entropy"], 0.0)
        self.assertEqual(result["log_count"], 0.0)
        self.assertEqual(result["mu"].shape, (0, 0, 2))

    def test_infeasible_and_invalid_temperature_rejected(self):
        with self.assertRaises(ValueError):
            fixed_k_posterior(2, np.array([[0, 1]]), np.zeros((1, 1)), 2, 1)
        for tau in (0, -1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                fixed_k_posterior(1, np.array([[0, 1]]), np.zeros((1, 1)), 1, tau)


if __name__ == "__main__":
    unittest.main()
