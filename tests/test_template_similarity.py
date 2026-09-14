import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from audit_template_similarity import candidates, jaccard, shingles, signature

class SimilarityTests(unittest.TestCase):
  def test_identical_and_near_duplicate_shingles(self):
    base="one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"
    a=shingles(base)
    b=shingles(base)
    self.assertEqual(jaccard(a,b),1.0)
    c=shingles(base.replace("twenty","twentyone"))
    self.assertGreaterEqual(jaccard(a,c),.8); self.assertLess(jaccard(a,c),1)

  def test_lsh_retrieves_identical_signature(self):
    idx={}
    s=signature(shingles("a b c d e f"))
    from audit_template_similarity import add_index
    add_index(idx,s,"doc")
    self.assertIn("doc",candidates(s,idx))

  def test_near_duplicate_is_retrieved_and_empty_is_excluded(self):
    from audit_template_similarity import add_index
    text = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"
    idx = {}
    add_index(idx, signature(shingles(text)), "original")
    self.assertIn("original", candidates(signature(shingles(text.replace("twenty", "changed"))), idx))
    self.assertIsNone(signature(shingles("too short")))
    self.assertEqual(candidates(None, idx), set())
    self.assertEqual(jaccard(set(), set()), 0)

  def test_threshold_components_are_transitive(self):
    from audit_template_similarity import families
    pairs = [("d1", "t1", .85), ("d1", "t2", .95), ("d2", "t2", .81)]
    self.assertEqual(families(pairs, .8)["largest_component_documents"], 4)
    self.assertEqual(families(pairs, .9)["largest_component_documents"], 2)
