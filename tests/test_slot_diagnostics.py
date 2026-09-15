import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from slot_diagnostics import compare


class SlotDiagnosticsTests(unittest.TestCase):
    def test_missing_gold_does_not_terminate_shift(self):
        gold=[(2,3,'A'),(6,7,'B'),(10,11,'C'),(14,15,'D')]
        pred=[(0,1,'X'),gold[0],(8,9,'Y'),gold[2]]
        out=compare(gold,pred,pred)
        self.assertEqual(len(out['simple_episodes']),1)
        self.assertEqual(out['simple_episodes'][0]['gold_ranks'],[1,3])
        self.assertEqual(out['episode_entity_counts']['RETAINED_DISPLACED'],2)

    def test_randomized_accounting_partition(self):
        import random
        rng=random.Random(20260915)
        for _ in range(200):
            k=rng.randrange(1,9)
            universe=[(i*2,i*2+1,'A') for i in range(16)]
            gold=sorted(rng.sample(universe,k));mp=sorted(rng.sample(universe,k));bp=sorted(rng.sample(universe,k))
            out=compare(gold,mp,bp);t=out['transitions']
            self.assertEqual(sum(t.values()),k)
            expected=sum(g==p for g,p in zip(gold,bp))-sum(g==p for g,p in zip(gold,mp))
            observed=t['map_wrong_to_mbr_correct']+t['map_absent_to_mbr_correct']-t['map_correct_to_mbr_wrong']-t['map_correct_to_mbr_lost']
            self.assertEqual(expected,observed)
            self.assertEqual(out['gaps'][-1]['cumulative'],0)
            self.assertEqual(out['mbr_gaps'][-1]['cumulative'],0)

    def test_identical_predictions_have_no_events(self):
        x = [(0, 1, "A"), (2, 3, "B")]
        out = compare(x, x, x)
        self.assertFalse(out["simple_episodes"])
        self.assertEqual(out["episode_entity_counts"], {"REPAIRED": 0, "RETAINED_DISPLACED": 0, "LOST": 0})
        self.assertEqual(out["transitions"]["map_correct_to_mbr_correct"], 2)

    def test_shift_repair_and_loss_is_not_repair(self):
        gold = [(1, 2, "A"), (3, 4, "B"), (5, 6, "C")]
        # A filler before the first two entities creates a +1 episode.
        mp = [(0, 1, "X"), (1, 2, "A"), (3, 4, "B")]
        mbr = [(5, 6, "C"), (7, 8, "Y"), (9, 10, "Z")]
        out = compare(gold, mp, mbr)
        self.assertEqual(out["episode_entity_counts"]["LOST"], 2)
        self.assertEqual(out["episode_entity_counts"]["REPAIRED"], 0)
        self.assertEqual(out["transitions"]["map_correct_to_mbr_correct"], 0)

    def test_simple_shift_is_repaired(self):
        gold = [(1, 2, "A"), (3, 4, "B"), (5, 6, "C")]
        mp = [(0, 1, "X"), (1, 2, "A"), (3, 4, "B")]
        out = compare(gold, mp, gold)
        self.assertEqual(len(out["simple_episodes"]), 1)
        self.assertEqual(out["simple_episodes"][0]["length"], 2)
        self.assertTrue(out["simple_episodes"][0]["complete_repair"])
        self.assertEqual(out["episode_entity_counts"]["REPAIRED"], 2)

    def test_complex_displacement_and_sentinel(self):
        gold = [(2, 3, "A"), (4, 5, "B"), (6, 7, "C"), (8, 9, "D")]
        mp = [(0, 1, "X"), (1, 2, "Y"), (2, 3, "A"), (6, 7, "C")]
        out = compare(gold, mp, mp)
        self.assertTrue(out["complex_displacement_runs"])
        self.assertEqual(out["gaps"][0]["cumulative"], 2)
        self.assertEqual(out["gaps"][-1]["cumulative"], 0)

    def test_json_safe_and_validation(self):
        out = compare([], [], [])
        self.assertEqual(out["gold_count"], 0)
        self.assertFalse(out["documents_affected"])
        with self.assertRaises(ValueError):
            compare([(1, 2, "A"), (0, 1, "B")], [], [])


if __name__ == "__main__":
    unittest.main()
