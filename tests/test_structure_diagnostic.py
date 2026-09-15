import unittest
import numpy as np
from src.run_structure_diagnostic import FastAUC,summarize
from src.structure_probe import weighted_auc


class DiagnosticTests(unittest.TestCase):
    def test_exact_cluster_weighted_auc_with_ties(self):
        rng=np.random.default_rng(17)
        for _ in range(100):
            y=rng.integers(0,2,50);p=rng.integers(0,6,50)/5;g=rng.integers(0,5,50);w=rng.integers(1,5,5)
            self.assertAlmostEqual(FastAUC(y,p,g).value(w),weighted_auc(y,p,w[g]),places=12)

    def test_pair_weighting_and_improvement_direction(self):
        x={'target':np.array([1,0,1,0]),'pair_pos':np.array([0,2]),'pair_neg':np.array([1,3])}
        pred=np.array([[.8,.9],[.9,.1],[.9,.9],[.1,.1]])
        r=summarize(np.ones(4,dtype=bool),np.ones(2,dtype=bool),x,pred,np.array([0,0,1,1]))
        self.assertEqual(r['control']['pair_accuracy'],.5)
        self.assertEqual(r['structural']['pair_accuracy'],1.)
        self.assertGreater(r['improvement']['logloss'],0)


if __name__=='__main__':unittest.main()
