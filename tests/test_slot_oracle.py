import itertools
from pathlib import Path
import random
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from audit_slot_oracle import slot_oracle


def exhaustive(pool, gold):
    values = []
    for choice in itertools.combinations(sorted(set(pool)), len(gold)):
        if all(a[1] <= b[0] for a, b in zip(choice, choice[1:])):
            values.append(sum(a == b for a, b in zip(choice, gold)))
    return max(values) if values else None


class OracleTests(unittest.TestCase):
    def test_missing_first_entity_forces_shift(self):
        gold = [(0, 1, "NAME"), (2, 3, "NAME"), (4, 5, "NAME")]
        pool = gold[1:] + [(6, 7, "NAME")]
        self.assertEqual(slot_oracle(pool, gold), 0)

    def test_wrong_filler_can_restore_later_slots(self):
        gold = [(0, 1, "NAME"), (2, 3, "NAME"), (4, 5, "NAME")]
        self.assertEqual(slot_oracle([(0, 1, "DATE")] + gold[1:], gold), 2)

    def test_exact_count_infeasible_and_empty(self):
        self.assertIsNone(slot_oracle([(0, 4, "NAME"), (1, 3, "DATE")], [(0, 1, "NAME"), (4, 5, "DATE")]))
        self.assertEqual(slot_oracle([], []), 0)

    def test_dp_matches_exhaustive_enumeration(self):
        rng = random.Random(2026)
        universe = [(a, b, label) for a in range(7) for b in range(a + 1, 8) for label in ("NAME", "DATE")]
        for _ in range(300):
            gold = [(2 * i, 2 * i + 1, rng.choice(("NAME", "DATE"))) for i in range(rng.randrange(5))]
            pool = rng.sample(universe, rng.randrange(11))
            self.assertEqual(slot_oracle(pool, gold), exhaustive(pool, gold), (pool, gold))


if __name__ == "__main__":
    unittest.main()
