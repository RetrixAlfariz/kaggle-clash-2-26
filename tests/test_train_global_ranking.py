import copy
import hashlib
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from train_global_ranking import ce_loss, gold_tokens, rank_loss
from span_probe import SpanHead, padded_hidden, sampling_plan


class RankingTrainingTests(unittest.TestCase):
    def test_rank_gradient_matches_direct_energy_difference(self):
        torch.manual_seed(9)
        head=SpanHead().eval()
        h,pr=padded_hidden([np.random.default_rng(2).normal(size=(3,256)).astype(np.float32)],device='cpu')
        gold=[((0,1,0),(2,3,1))]; rival={0:((0,1,0),(1,3,2))}
        actual=rank_loss(head,h,pr,[0],gold,rival)
        z=head(h,pr,torch.tensor([0,0]),torch.tensor([1,2]),torch.tensor([3,3]))
        expected=torch.nn.functional.softplus(((z[0,3]-z[0,0])-(z[1,2]-z[1,0]))/2)
        self.assertAlmostEqual(float(actual.detach()),float(expected.detach()),places=6)
        ga=torch.autograd.grad(actual,tuple(head.parameters()))
        ge=torch.autograd.grad(expected,tuple(head.parameters()))
        for a,e in zip(ga,ge): torch.testing.assert_close(a,e,atol=2e-6,rtol=2e-5)

    def test_ce_replay_survives_intervening_ranking_forward(self):
        torch.manual_seed(5); head=SpanHead(); other=copy.deepcopy(head)
        h,pr=padded_hidden([np.ones((3,256),dtype=np.float32)],device='cpu')
        entities=[dict(start=0,end=1,label='NAME')]
        plan=sampling_plan(np.array([[0,1],[2,3],[4,5]]),entities,[]); plan['n']=3
        digest1=hashlib.sha256(); digest2=hashlib.sha256()
        torch.manual_seed(41); head.train(); a=ce_loss(head,h,pr,[0],[plan],1,digest1)
        other.eval(); rank_loss(other,h,pr,[0],[((0,1,0),)],{0:((1,2,0),)})
        torch.manual_seed(41); other.train(); b=ce_loss(other,h,pr,[0],[plan],1,digest2)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(digest1.hexdigest(),digest2.hexdigest())

    def test_unsupported_gold_is_excluded(self):
        row=dict(entities=[dict(start=1,end=2,label='NAME')],expected_entity_count=1)
        self.assertIsNone(gold_tokens(row,np.array([[0,2]])))
        h,pr=padded_hidden([np.zeros((1,256),dtype=np.float32)],device='cpu')
        loss=rank_loss(SpanHead(),h,pr,[0],[None],{0:None})
        self.assertEqual(float(loss),0.)
        self.assertFalse(loss.requires_grad)


if __name__=='__main__': unittest.main()
