import itertools
import math
from pathlib import Path
import random
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from slot_decoder import decode, slot_marginals


class DecoderTests(unittest.TestCase):
    def test_exhaustive_distribution_and_decisions(self):
        rng = random.Random(2026)
        universe = [(a, b, label) for a in range(6) for b in range(a + 1, 7) for label in ("NAME", "DATE")]
        for _ in range(180):
            pool = {s: rng.uniform(.001, .999) for s in rng.sample(universe, rng.randrange(1, 10))}
            k = rng.randrange(4)
            choices = [x for x in itertools.combinations(sorted(pool), k) if all(a[1] <= b[0] for a, b in zip(x, x[1:]))]
            if not choices:
                for method in ("map", "slot_mbr"):
                    with self.assertRaises(ValueError): decode(pool, k, method)
                continue
            logw = [sum(math.log(pool[s] / (1 - pool[s])) for s in x) for x in choices]
            weights = np.exp(np.array(logw) - max(logw))
            weights /= weights.sum()
            spans, _, marginals = slot_marginals(pool, k)
            expected = np.zeros_like(marginals)
            index = {s: i for i, s in enumerate(spans)}
            for choice, weight in zip(choices, weights):
                for j, span in enumerate(choice): expected[index[span], j] += weight
            np.testing.assert_allclose(marginals, expected, atol=1e-10)
            mbr = decode(pool, k)
            utility = lambda choice: sum(expected[index[s], j] for j, s in enumerate(choice))
            self.assertAlmostEqual(utility(mbr), max(map(utility, choices)))
            chosen_map = decode(pool, k, "map")
            self.assertAlmostEqual(sum(math.log(pool[s] / (1 - pool[s])) for s in chosen_map), max(logw))

    def test_extreme_probabilities_and_touching_intervals(self):
        pool = {(0, 1, "NAME"): 0., (1, 2, "DATE"): 1.}
        self.assertEqual(decode(pool, 2), sorted(pool))
        self.assertEqual(decode({}, 0), [])

    def test_invalid_inputs(self):
        for p in (float('nan'), -1., 2.):
            with self.assertRaises(ValueError): decode({(0, 1, "NAME"): p}, 1)
        with self.assertRaises(ValueError): decode({(1, 1, "NAME"): .5}, 1)
        with self.assertRaises(ValueError): decode({}, -1)


if __name__ == "__main__": unittest.main()
