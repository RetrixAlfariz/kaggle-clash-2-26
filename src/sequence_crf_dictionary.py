"""Sequence features enriched with frozen train dictionary candidate boundaries."""
import bisect
from pathlib import Path
from sequence_crf import features as base_features
from run_m0 import candidates_for
from submit_m0 import load_dictionary

ROOT=Path(__file__).resolve().parents[1]
_dictionary=None


def features(text,ts):
    global _dictionary
    if _dictionary is None:_dictionary=load_dictionary(ROOT/'output/m0/v1/dictionary.parquet')
    seq=base_features(text,ts)
    starts={a:i for i,(a,b,w) in enumerate(ts)};ends={b:i for i,(a,b,w) in enumerate(ts)}
    scores=[{} for _ in ts]
    for (a,b,label),prior in candidates_for(text,*_dictionary).items():
        if a not in starts or b not in ends:continue
        i,j=starts[a],ends[b]
        for t in range(i,j+1):
            roles=['B' if t==i else 'I']
            if t==j:roles.append('E')
            for role in roles:
                key='dict_'+role+'_'+label
                scores[t][key]=max(scores[t].get(key,0.),prior)
    for fs,values in zip(seq,scores):
        for key,prior in sorted(values.items()):
            fs.extend([key,key+'_confidence='+str(min(9,int(prior*10)))])
    return seq
