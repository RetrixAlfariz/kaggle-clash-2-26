import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'src'))
import numpy as np
import torch
from run_neural_transition import sequence_nll, legal_mask, LABELS
from run_neural_transition import batch_nll
import itertools

def test_partition_matches_exhaustive_and_padded_batch():
    torch.manual_seed(2026)
    e=torch.randn(3,len(LABELS));t=torch.randn(len(LABELS),len(LABELS),requires_grad=True);mask=legal_mask()
    gold=torch.tensor([1,2,0]);scores=[]
    for path in itertools.product(range(len(LABELS)),repeat=3):
        if LABELS[path[0]].startswith('I-') or any(not mask[a,b] for a,b in zip(path,path[1:])):continue
        scores.append(sum(e[i,j] for i,j in enumerate(path))+t[path[0],path[1]]+t[path[1],path[2]])
    score=sum(e[i,j] for i,j in enumerate(gold))+t[1,2]+t[2,0]
    expected=torch.logsumexp(torch.stack(scores),0)-score
    assert torch.allclose(sequence_nll(e,gold,t,mask),expected,atol=1e-5)
    emissions=torch.stack([e,e]);golds=torch.stack([gold,torch.tensor([1,0,0])])
    losses=batch_nll(emissions,golds,torch.tensor([3,1]),t,mask)
    assert torch.allclose(losses[0],expected,atol=1e-5)
    assert torch.allclose(losses[1],sequence_nll(e[:1],gold[:1],t,mask),atol=1e-5)
    losses.mean().backward();assert torch.isfinite(t.grad).all()

def test_transition_mask_and_finite_gradient():
    e=torch.randn(5,len(LABELS),requires_grad=True); t=torch.zeros((len(LABELS),len(LABELS)),requires_grad=True)
    g=torch.tensor([0,1,2,0,3]); loss=sequence_nll(e,g,t,legal_mask()); loss.backward()
    assert torch.isfinite(loss); assert torch.isfinite(t.grad).all()

def test_initial_i_is_illegal():
    m=legal_mask(); assert bool(m[0,0]); assert not bool(m[0,2]) if LABELS[2].startswith('I-') else True
