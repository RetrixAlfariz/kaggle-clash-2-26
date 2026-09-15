import unittest
from src.run_slot_mbr import canonical_hash,pred_map,record,paired_bootstrap


class ReplayTests(unittest.TestCase):
    def test_only_document_serialization_order_normalized(self):
        a=record('a',[(0,1,'NAME'),(2,3,'DATE')]);b=record('b',[(0,1,'EMAIL')])
        self.assertEqual(canonical_hash([a,b]),canonical_hash([b,a]))
        changed=record('a',[(0,1,'NAME'),(2,3,'NAME')])
        self.assertNotEqual(canonical_hash([a,b]),canonical_hash([changed,b]))
        self.assertNotEqual(canonical_hash([a,b]),canonical_hash([a]))
        for rows in ([a,a],[record('x',[(2,3,'NAME'),(0,1,'DATE')])],
                     [record('x',[(0,3,'NAME'),(2,4,'DATE')])]):
            with self.assertRaises(ValueError):pred_map(rows)

    def test_bootstrap_uses_pooled_slot_counts_and_pairing(self):
        base=[{'document_id':'a','K':1,'correct':0},{'document_id':'b','K':9,'correct':0}]
        cur=[{'document_id':'a','K':1,'correct':1},{'document_id':'b','K':9,'correct':0}]
        result=paired_bootstrap(cur,base)
        self.assertAlmostEqual(result['observed_delta'],.1)
        self.assertEqual(result['replicates'],10000)
        with self.assertRaises(ValueError):paired_bootstrap(cur,base[::-1])


if __name__=='__main__':unittest.main()
