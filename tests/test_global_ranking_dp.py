import unittest

import numpy as np

from src.global_ranking_dp import best_two


def brute(n, pairs, energies, k):
    rows = [(int(a), int(b), i) for i, (a, b) in enumerate(pairs)]
    structures = []

    def visit(pos, chosen, score):
        if len(chosen) == k:
            structures.append((float(score), tuple(chosen)))
            return
        for a, b, i in rows:
            if a >= pos:
                for lab in range(energies.shape[1]):
                    visit(b, chosen + [(a, b, lab)], score + energies[i, lab])

    if k == 0:
        return [(0.0, ())]
    visit(0, [], 0.0)
    structures.sort(key=lambda item: -item[0])
    return structures[:2]


class GlobalRankingDPTests(unittest.TestCase):
    def test_matches_brute_force_random_tiny_cases(self):
        rng = np.random.default_rng(1221)
        for n in range(1, 7):
            pairs = np.array([(a, b) for a in range(n) for b in range(a + 1, n + 1)])
            for labels in (1, 2, 3):
                energies = rng.normal(size=(len(pairs), labels))
                for k in range(0, min(4, n + 1)):
                    with self.subTest(n=n, labels=labels, k=k):
                        got = best_two(n, pairs, energies, k)
                        expected = brute(n, pairs, energies, k)
                        self.assertEqual(len(got), len(expected))
                        self.assertEqual([x[1] for x in got], list(dict.fromkeys(x[1] for x in got)))
                        np.testing.assert_allclose([x[0] for x in got], [x[0] for x in expected], rtol=1e-12, atol=1e-12)

    def test_same_boundary_label_alternative_and_skip_distinctness(self):
        pairs = np.array([[0, 1], [1, 2], [0, 2]])
        energies = np.array([[0.0, 2.0], [0.0, 3.0], [1.0, 0.5]])
        got = best_two(2, pairs, energies, 1)
        self.assertEqual(got[0][1], ((1, 2, 1),))
        self.assertEqual(got[1][1], ((0, 1, 1),))
        self.assertEqual(best_two(2, pairs, energies, 2)[0][1], ((0, 1, 1), (1, 2, 1)))

    def test_tied_scores_are_deterministic(self):
        pairs = np.array([[0, 1], [1, 2], [0, 2]])
        energies = np.zeros((3, 2))
        first = best_two(2, pairs, energies, 1)
        self.assertEqual(first, best_two(2, pairs, energies, 1))
        self.assertEqual(len({x[1] for x in first}), len(first))

    def test_validation_and_infeasible_k(self):
        with self.assertRaises(ValueError):
            best_two(2, np.array([[0, 3]]), np.zeros((1, 1)), 1)
        self.assertEqual(best_two(2, np.array([[0, 1]]), np.zeros((1, 1)), 2), [])


if __name__ == "__main__":
    unittest.main()
