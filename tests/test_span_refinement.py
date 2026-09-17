import sys
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from span_probe import SpanHead, padded_hidden, sampling_plan, sample_plan, universe
from train_span_refinement import multipliers, weighted_ce


class RefinementTests(unittest.TestCase):
    def test_overlap_uses_max_and_random_remains_one(self):
        np.testing.assert_array_equal(multipliers(np.array([1,2,4,6,8])),[4,2,4,4,1])

    def test_weighted_loss_equals_full_small_population(self):
        torch.manual_seed(1); head=SpanHead().eval()
        bounds=np.array([[0,1],[2,3],[4,5]])
        plan=sampling_plan(bounds,[dict(start=0,end=1,label='NAME')],[(2,3,'NAME')]); plan['n']=3
        h,pr=padded_hidden([np.ones((3,256),dtype=np.float32)],device='cpu')
        actual=weighted_ce(head,h,pr,[0],[plan],3)
        indices,targets,tags,weights=sample_plan(plan,2026,3,0); p=universe(3)[indices]
        z=head(h,pr,torch.zeros(len(p),dtype=torch.long),torch.tensor(p[:,0]),torch.tensor(p[:,1]))
        w=torch.tensor(weights*multipliers(tags))
        expected=(torch.nn.functional.cross_entropy(z,torch.tensor(targets),reduction='none')*w).sum()/w.sum()
        torch.testing.assert_close(actual,expected)


if __name__=='__main__': unittest.main()
