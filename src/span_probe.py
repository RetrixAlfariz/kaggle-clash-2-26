"""P0 span head, train-only sampling, and exact-K interval selection.

Internal token intervals are half-open [a,b), matching character conventions.
The encoder remains in its original module and is never modified here.
"""
from functools import lru_cache
import numpy as np
import torch
from torch import nn

LABELS = ('NONE','NAME','DATE','EMAIL','PHONE','ADDRESS','USERNAME','JOB_TITLE')
WIDTH = 16

@lru_cache(maxsize=1024)
def universe(n):
    pairs=[(a,b) for b in range(1,n+1) for a in range(max(0,b-WIDTH),b)]
    return np.asarray(pairs,dtype=np.int64).reshape(-1,2)

def span_index(a,b):
    if not 0<=a<b or b-a>WIDTH:raise ValueError('Invalid token interval')
    t=b-1
    previous=t*(t+1)//2 if t<=WIDTH else WIDTH*(WIDTH+1)//2+(t-WIDTH)*WIDTH
    return previous+a-max(0,b-WIDTH)

def sampling_plan(bounds,gold,old_predictions):
    """No dev input. Protected-gold overlap is excluded only during train."""
    n=len(bounds);pairs=universe(n);m=len(pairs)
    starts={int(a):i for i,(a,b) in enumerate(bounds)}
    ends={int(b):i+1 for i,(a,b) in enumerate(bounds)}
    positives={};protected=[];unaligned=long=0
    for g in gold:
        a,b=g['start'],g['end']
        if a not in starts or b not in ends:
            protected.append((a,b));unaligned+=1;continue
        i,j=starts[a],ends[b]
        if j-i>WIDTH:
            protected.append((a,b));long+=1;continue
        index=span_index(i,j)
        if index in positives:raise ValueError('Duplicate gold boundary')
        positives[index]=LABELS.index(g['label'])
    eligible=np.ones(m,dtype=bool)
    for a,b in protected:
        eligible &= ~((bounds[pairs[:,0],0]<b)&(bounds[pairs[:,1]-1,1]>a))
    for i in positives:eligible[i]=False
    forced={}
    for index in positives:
        a,b=pairs[index]
        for delta in (-2,-1,1,2):
            for x,y in ((a+delta,b),(a,b+delta)):
                if 0<=x<y<=n and y-x<=WIDTH:
                    j=span_index(int(x),int(y))
                    if eligible[j]:forced[j]=forced.get(j,0)|2
    for a,b,label in old_predictions:
        if a in starts and b in ends and 0<ends[b]-starts[a]<=WIDTH:
            j=span_index(starts[a],ends[b])
            if eligible[j]:forced[j]=forced.get(j,0)|4
    fi=np.asarray(sorted(forced),dtype=np.int32)
    remaining_mask=eligible.copy();remaining_mask[fi]=False
    pi=np.asarray(sorted(positives),dtype=np.int32)
    return {'positive':pi,'targets':np.asarray([positives[i] for i in pi],dtype=np.int64),
            'forced':fi,'tags':np.asarray([forced[i] for i in fi],dtype=np.int64),
            'remaining':np.flatnonzero(remaining_mask).astype(np.int32),
            'population':int(eligible.sum())+len(pi),'unaligned':unaligned,'long':long,
            'raw_population':m}

def sample_plan(plan,seed,epoch,document_index):
    remaining=plan['remaining'];r=len(remaining);m=min(256,r)
    rng=np.random.default_rng(np.random.SeedSequence([seed,epoch,document_index]))
    random=remaining if m==r else rng.choice(remaining,m,replace=False)
    indices=np.concatenate([plan['positive'],plan['forced'],random])
    targets=np.concatenate([plan['targets'],np.zeros(len(plan['forced'])+m,dtype=np.int64)])
    tags=np.concatenate([np.ones(len(plan['positive']),dtype=np.int64),plan['tags'],np.full(m,8,dtype=np.int64)])
    weights=np.ones(len(indices),dtype=np.float32)
    if m:weights[-m:]=r/m
    if not np.isclose(weights.sum(dtype=np.float64),plan['population'],atol=.01):raise ValueError('HT denominator mismatch')
    return indices,targets,tags,weights

class SpanHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.width=nn.Embedding(WIDTH+1,32,padding_idx=0)
        self.net=nn.Sequential(nn.LayerNorm(1056),nn.Linear(1056,256),nn.GELU(),nn.Dropout(.2),nn.Linear(256,8))
    def forward(self,hidden,prefix,documents,starts,ends):
        left=hidden[documents,starts];right=hidden[documents,ends-1]
        mean=(prefix[documents,ends]-prefix[documents,starts])/(ends-starts)[:,None]
        features=torch.cat([left,right,left*right,mean,self.width(ends-starts)],dim=1)
        return self.net(features)

def padded_hidden(arrays,device='cuda'):
    result=np.zeros((len(arrays),max(len(a) for a in arrays),256),dtype=np.float32)
    for i,a in enumerate(arrays):result[i,:len(a)]=a
    hidden=torch.from_numpy(result).to(device)
    prefix=torch.cat([hidden.new_zeros((len(arrays),1,256)),hidden.cumsum(1)],dim=1)
    return hidden,prefix

def decode_spans(n,values,labels,count):
    """Exact weighted interval DP over all <=16-token spans, O(T*W*K).

    End-position grouping is equivalent to the candidate-prefix recurrence.
    Skip wins ties, then smaller start; label ties are resolved before entry.
    """
    pairs=universe(n)
    if not isinstance(count,(int,np.integer)) or not 0<=count<=n:raise ValueError('Invalid K')
    if len(values)!=len(pairs) or len(labels)!=len(pairs):raise ValueError('Candidate coverage mismatch')
    if not np.isfinite(values).all():raise ValueError('Nonfinite span scores')
    if np.any((np.asarray(labels)<1)|(np.asarray(labels)>=len(LABELS))):raise ValueError('Invalid labels')
    dp=np.full((n+1,count+1),-np.inf);dp[:,0]=0
    choice=np.full((n+1,count+1),-1,dtype=np.int32)
    for b in range(1,n+1):
        dp[b,1:]=dp[b-1,1:]
        lo=span_index(max(0,b-WIDTH),b);hi=lo+min(b,WIDTH)
        starts=pairs[lo:hi,0]
        options=dp[starts,:count]+np.asarray(values[lo:hi],dtype=np.float64)[:,None]
        if count:
            arg=options.argmax(0);best=options[arg,np.arange(count)];take=best>dp[b,1:]
            dp[b,1:][take]=best[take];choice[b,1:][take]=lo+arg[take]
    if not np.isfinite(dp[n,count]):raise ValueError('Infeasible exact K')
    selected=[];b=n;k=count
    while k:
        i=int(choice[b,k])
        if i<0:b-=1;continue
        a,end=pairs[i];selected.append((int(a),int(end),LABELS[int(labels[i])]))
        b=int(a);k-=1
    return sorted(selected)

def character_spans(token_spans,bounds):
    return [(int(bounds[a,0]),int(bounds[b-1,1]),label) for a,b,label in token_spans]
