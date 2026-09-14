# /// script
# requires-python = "==3.13.2"
# dependencies = ["lightgbm==4.6.0", "pyarrow==25.0.1", "scikit-learn==1.9.1"]
# ///
"""Corrected train-only categorical boosted candidate scorer, separate v2 run."""
import bisect
import hashlib
import json
import math
from pathlib import Path
import pickle
import random
import re
import time
import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from train_m1 import propose,sha,model_sources
from submit_m0 import load_dictionary
from prepared_loader import PreparedData
from ner_evaluation import Evaluation,triples
from baseline_probe import decode as sum_decode
from slot_decoder import decode as slot_decode

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/bitrase-1.algo-2/v2'
WORD=re.compile(r'\w+|[^\w\s]');TITLE=re.compile(r'^(Mr|Mrs|Ms|Miss|Dr|Prof|Mx)\.?\s+')
NCAT=12;MAXNEG=40


def context(text):
    matches=list(WORD.finditer(text))
    return [m.start() for m in matches],[m.end() for m in matches],[m.group().lower() for m in matches]


def feature_values(text,span,prior,ctx):
    a,b,label=span;v=text[a:b];low=v.lower();vs=low.split();starts,ends,words=ctx
    left=bisect.bisect_right(ends,a);right=bisect.bisect_left(starts,b)
    lw=words[max(0,left-2):left];rw=words[right:right+2]
    title=TITLE.match(v)
    shp=''.join('X' if c.isupper() else 'x' if c.islower() else 'd' if c.isdigit() else c for c in v)
    shp=re.sub(r'(.)\1+',r'\1',shp)[:40]
    cats=[label,low,vs[0] if vs else '',vs[-1] if vs else '',lw[-1] if lw else '',lw[-2] if len(lw)>1 else '',
          rw[0] if rw else '',rw[1] if len(rw)>1 else '', ' '.join(lw),' '.join(rw),shp,title.group(1) if title else '']
    nums=[math.log(max(prior,1e-6)/max(1-prior,1e-6)),len(v),len(vs),a/max(1,len(text)),len(text),
          v.count('\n'),sum(c.isdigit() for c in v),int('_' in v),int('@' in v),int(v.islower()),int(v.isupper()),int(v.istitle()),
          int(bool(title)),int(a>0 and (text[a-1].isalnum() or text[a-1]=='_')),int(b<len(text) and (text[b].isalnum() or text[b]=='_')),
          int(a>0 and text[a-1]=='\n'),int(b<len(text) and text[b]=='\n'),int(v[0].isspace()),int(v[-1].isspace()),
          int(bool(re.search(r'\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s*$',text[max(0,a-20):a])))]
    return cats,nums


def encode(cats,nums,maps,fit):
    codes=[]
    for value,mapping in zip(cats,maps):
        if fit and value not in mapping:mapping[value]=len(mapping)
        codes.append(mapping.get(value,-1))
    return codes+nums


def main():
    if OUT.exists():raise ValueError('Output exists')
    sources={**model_sources(),**{s:sha(ROOT/s) for s in ['src/run_boosted_v2.py','src/slot_decoder.py','note/bitrase-1.algo-2/protocol_v2.md']}}
    params={'n_estimators':500,'learning_rate':.06,'num_leaves':63,'max_depth':-1,'min_child_samples':40,
            'colsample_bytree':.9,'reg_lambda':2.,'cat_smooth':10.,'max_cat_threshold':32,'random_state':2026,'n_jobs':8,'verbosity':-1}
    config={'experiment':'bitrase-1.algo-2','run':'v2','pretrained':False,'params':params,'negative_cap':MAXNEG,
            'sampling':'all positives; deterministic negatives with inverse inclusion probability sample weights','sources':sources}
    OUT.mkdir(parents=True);(OUT/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    rows=PreparedData().load_train().to_pylist();trie,confidence=load_dictionary(ROOT/'output/m0/v1/dictionary.parquet')
    capacity=sum(len(r['entities'])+MAXNEG for r in rows);x=np.empty((capacity,NCAT+20),dtype=np.float32)
    y=np.empty(capacity,dtype=np.int8);weights=np.empty(capacity,dtype=np.float32);maps=[{} for _ in range(NCAT)]
    cursor=0;start=time.monotonic()
    for i,row in enumerate(rows,1):
        text=row['full_text'];pool=propose(text,trie,confidence);gold=set(triples(row['entities']));ctx=context(text)
        positives=[s for s in sorted(pool) if s in gold];negatives=[s for s in sorted(pool) if s not in gold]
        rng=random.Random(2026+int(hashlib.sha256(row['document_id'].encode()).hexdigest()[:8],16));rng.shuffle(negatives)
        negweight=max(1,len(negatives)/MAXNEG)
        for label,spans,w in [(1,positives,1.),(0,negatives[:MAXNEG],negweight)]:
            for span in spans:
                cats,nums=feature_values(text,span,pool[span],ctx);x[cursor]=encode(cats,nums,maps,True);y[cursor]=label;weights[cursor]=w;cursor+=1
        if i%10000==0:print(f'Encoded {i}/{len(rows)} train docs',flush=True)
    x=x[:cursor];y=y[:cursor];weights=weights[:cursor];del rows
    print(f'Fit500trees on {cursor} candidate examples',flush=True)
    model=lgb.LGBMClassifier(**params);model.fit(x,y,sample_weight=weights,categorical_feature=list(range(NCAT)))
    with (OUT/'model.pkl').open('wb') as f:pickle.dump({'model':model,'maps':maps},f,protocol=5)
    del x,weights
    print('Training finished, building dev matrix',flush=True)
    dev=PreparedData().load_dev('evaluation').to_pylist();cache=[];vectors=[]
    for row in dev:
        text=row['full_text'];pool=propose(text,trie,confidence);spans=sorted(pool);ctx=context(text)
        for span in spans:
            cats,nums=feature_values(text,span,pool[span],ctx);vectors.append(encode(cats,nums,maps,False))
        cache.append((row,pool,spans))
    probability=model.predict_proba(np.asarray(vectors,dtype=np.float32))[:,1];del vectors
    metrics={m:Evaluation() for m in ['sum_probability','slot_mbr']};predictions={m:[] for m in metrics};cursor=0
    for i,(row,pool,spans) in enumerate(cache,1):
        scored=dict(zip(spans,map(float,probability[cursor:cursor+len(spans)])));cursor+=len(spans);gold=triples(row['entities']);k=row['expected_entity_count']
        for method in metrics:
            pred=sum_decode(scored,k) if method=='sum_probability' else slot_decode(scored,k,'slot_mbr')
            if len(pred)!=k:raise ValueError('K infeasible')
            metrics[method].add(row['full_text'],gold,pred,pool)
            predictions[method].append({'document_id':row['document_id'],'predicted':[{'start':a,'end':b,'label':lab} for a,b,lab in pred]})
        if i%1500==0:print(f'Decoded {i}/{len(dev)} dev docs',flush=True)
    reports={m:e.report() for m,e in metrics.items()};ph={}
    for m,preds in predictions.items():
        file=OUT/f'{m}_dev_predictions.parquet';pq.write_table(pa.Table.from_pylist(preds),file,compression='zstd');ph[m]=sha(file)
    for path,h in sources.items():
        if sha(ROOT/path)!=h:raise ValueError('Source changed: '+path)
    result={'dev':reports,'selected':max(reports,key=lambda m:reports[m]['slot_accuracy']),'train_examples':len(y),
            'train_positives':int(y.sum()),'seconds':time.monotonic()-start,'model_sha256':sha(OUT/'model.pkl'),'prediction_hashes':ph}
    (OUT/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({m:{'slot':r['slot_accuracy'],'entity_f1':r['entity_micro']['f1']} for m,r in reports.items()},indent=2))


if __name__=='__main__':main()
