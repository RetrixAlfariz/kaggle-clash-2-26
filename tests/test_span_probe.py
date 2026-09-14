import itertools
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from span_probe import universe,span_index,sampling_plan,sample_plan,decode_spans,SpanHead,padded_hidden

class SpanTests(unittest.TestCase):
    def test_index_and_complete_universe(self):
        for n in (0,1,3,16,17,31):
            pairs=universe(n)
            self.assertEqual(len(pairs),sum(min(b,16) for b in range(1,n+1)))
            for i,(a,b) in enumerate(pairs):self.assertEqual(span_index(int(a),int(b)),i)

    def test_sampling_gold_priority_protection_and_ht(self):
        bounds=np.array([[i*2,i*2+1] for i in range(40)])
        gold=[{'start':4,'end':9,'label':'NAME'},{'start':20,'end':24,'label':'USERNAME'}]
        # Second gold ends inside a gap: it is unrepresentable, never rounded.
        plan=sampling_plan(bounds,gold,[(4,9,'DATE'),(0,1,'NAME')])
        self.assertEqual(plan['unaligned'],1)
        positive=span_index(2,5);self.assertIn(positive,plan['positive'])
        self.assertEqual(plan['targets'][list(plan['positive']).index(positive)],1)
        a=sample_plan(plan,2026,1,0);b=sample_plan(plan,2026,1,0)
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        indices,targets,tags,weights=a
        self.assertEqual(len(indices),len(set(indices)))
        self.assertAlmostEqual(float(weights.sum(dtype=np.float64)),plan['population'],places=3)
        for index,target in zip(indices,targets):
            x,y=universe(40)[index]
            if not target:self.assertFalse(bounds[x,0]<24 and bounds[y-1,1]>20)
        # HT total is unbiased for arbitrary fixed candidate losses.
        losses=np.arange(len(universe(40)),dtype=float)/100
        eligible=np.concatenate([plan['positive'],plan['forced'],plan['remaining']])
        estimates=[]
        for seed in range(300):
            ii,_,_,ww=sample_plan(plan,seed,1,0);estimates.append(float((losses[ii]*ww).sum()))
        self.assertLess(abs(np.mean(estimates)-losses[eligible].sum()),.015*losses[eligible].sum())

    def test_interval_dp_exhaustive(self):
        rng=np.random.default_rng(2026);n=4;pairs=universe(n)
        for _ in range(25):
            scores=rng.normal(size=len(pairs));labels=rng.integers(1,8,len(pairs))
            for k in range(n+1):
                best=-np.inf
                for indices in itertools.combinations(range(len(pairs)),k):
                    selected=sorted(pairs[i].tolist() for i in indices)
                    if any(x[1]>y[0] for x,y in zip(selected,selected[1:])):continue
                    best=max(best,sum(scores[i] for i in indices))
                chosen=decode_spans(n,scores,labels,k)
                self.assertEqual(len(chosen),k)
                self.assertAlmostEqual(sum(scores[span_index(a,b)] for a,b,l in chosen),best)
        with self.assertRaises(ValueError):decode_spans(n,np.full(len(pairs),np.nan),labels,2)

    def test_prefix_mean_and_head_gradient(self):
        arrays=[np.random.default_rng(1).normal(size=(5,256)).astype('float32'),np.ones((2,256),dtype='float32')]
        h,p=padded_hidden(arrays,'cpu')
        torch.testing.assert_close((p[0,4]-p[0,1])/3,h[0,1:4].mean(0))
        head=SpanHead();logits=head(h,p,torch.tensor([0,1]),torch.tensor([1,0]),torch.tensor([4,2]))
        self.assertEqual(tuple(logits.shape),(2,8))
        torch.nn.functional.cross_entropy(logits,torch.tensor([1,0])).backward()
        self.assertTrue(all(torch.isfinite(x.grad).all() for x in head.parameters()))
        self.assertFalse(h.requires_grad)

    def test_chunked_weighted_gradient_equals_full_batch(self):
        torch.manual_seed(2)
        head=SpanHead().eval()  # Disable dropout to compare identical objectives.
        h,p=padded_hidden([np.ones((4,256),dtype='float32')],'cpu')
        d=torch.zeros(4,dtype=torch.long);a=torch.tensor([0,1,0,2]);b=torch.tensor([1,3,4,4])
        y=torch.tensor([1,0,3,0]);w=torch.tensor([1.,4.,1.,4.]);denom=10
        full=(torch.nn.functional.cross_entropy(head(h,p,d,a,b),y,reduction='none')*w).sum()/denom
        full.backward();expected=[x.grad.clone() for x in head.parameters()];head.zero_grad()
        for sl in (slice(0,2),slice(2,4)):
            loss=(torch.nn.functional.cross_entropy(head(h,p,d[sl],a[sl],b[sl]),y[sl],reduction='none')*w[sl]).sum()/denom
            loss.backward()
        for x,gradient in zip(head.parameters(),expected):torch.testing.assert_close(x.grad,gradient,rtol=1e-4,atol=1e-5)

if __name__=='__main__':unittest.main()
