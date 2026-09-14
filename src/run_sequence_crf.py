# /// script
# requires-python = "==3.13.2"
# dependencies = ["pyarrow==25.0.1", "numpy==2.5.3", "python-crfsuite==0.9.12"]
# ///
"""Sequence CRF from scratch, raw Viterbi and exposed-count constrained decoding."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import time
import pyarrow as pa
import pyarrow.parquet as pq
import pycrfsuite
from prepared_loader import PreparedData
from ner_evaluation import Evaluation, triples
from sequence_crf import tokens, features, gold_tags, spans_from_tags, model_arrays, emissions, decode_k

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train-docs',type=int,default=12000)
    p.add_argument('--iterations',type=int,default=60)
    p.add_argument('--output',type=Path,default=ROOT/'output/bitrase-1.algo-3/crf12k')
    args=p.parse_args(); out=args.output.resolve()
    if out.exists(): raise ValueError('Preserve previous run')
    out.mkdir(parents=True)
    paths=['src/sequence_crf.py','src/run_sequence_crf.py','src/prepared_loader.py','src/ner_evaluation.py',
           'output/prepared/v1/train.parquet','output/prepared/v1/dev.parquet','note/bitrase-1.algo-3/protocol.md']
    hashes={s:sha(ROOT/s) for s in paths}
    params={'c1':.1,'c2':.1,'max_iterations':args.iterations,'feature.possible_transitions':True,'feature.minfreq':2.}
    config={'experiment':'bitrase-1.algo-3','method':'linear-chain BIO CRF from scratch',
            'pretrained':False,'external_data':False,'train_docs_requested':args.train_docs,'seed':2026,'params':params,
            'versions':{s:importlib.metadata.version(s) for s in ('pyarrow','numpy','python-crfsuite')},'sources':hashes}
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    data=PreparedData(); train=data.load_train().to_pylist()
    random.Random(2026).shuffle(train);train=train[:args.train_docs]
    trainer=pycrfsuite.Trainer(verbose=False)
    trainer.set_params(params)
    unaligned=ntokens=0; ids=[]; start=time.monotonic()
    for i,row in enumerate(train,1):
        ts=tokens(row['full_text']); seq=features(row['full_text'],ts)
        tags,bad=gold_tags(ts,row['entities']);unaligned+=bad;ntokens+=len(ts)
        trainer.append(seq,tags);ids.append(row['document_id'])
        if i%2000==0:print(f'Prepared {i}/{len(train)} train docs',flush=True)
    (out/'train_ids.json').write_text(json.dumps(ids),encoding='utf-8')
    print(f'Training CRF: {ntokens} tokens, {unaligned} unaligned gold entities',flush=True)
    trainer.train(str(out/'model.crfsuite'))
    del trainer,train
    print('Training complete, evaluating dev',flush=True)
    train_seconds=time.monotonic()-start
    tagger=pycrfsuite.Tagger();tagger.open(str(out/'model.crfsuite'))
    labels,lookup,transitions=model_arrays(tagger)
    evaluators={m:Evaluation() for m in ('raw','known_k')}
    predictions={m:[] for m in evaluators};bad_dev=0
    for i,row in enumerate(data.load_dev('evaluation').to_pylist(),1):
        ts=tokens(row['full_text']); seq=features(row['full_text'],ts)
        gold=triples(row['entities']);tagged=tagger.tag(seq)
        raw=spans_from_tags(ts,tagged)
        # If raw path is valid BIO and already has K, it remains optimal under the constraints.
        valid=all(not tag.startswith('I-') or (j>0 and tagged[j-1] in ('B-'+tag[2:],tag)) for j,tag in enumerate(tagged))
        if len(raw)==row['expected_entity_count'] and valid:
            constrained=raw
        else:
            constrained=spans_from_tags(ts,decode_k(emissions(seq,lookup,len(labels)),transitions,labels,row['expected_entity_count']))
        _,bad=gold_tags(ts,row['entities']);bad_dev+=bad
        for method,pred in [('raw',raw),('known_k',constrained)]:
            evaluators[method].add(row['full_text'],gold,pred,pred)
            predictions[method].append({'document_id':row['document_id'],'predicted':[{'start':a,'end':b,'label':label} for a,b,label in pred]})
        if i%1500==0:print(f'Evaluated {i}/6943 dev docs',flush=True)
    reports={m:e.report() for m,e in evaluators.items()}
    # Sequence model does not enumerate a candidate pool; remove inapplicable candidate diagnostics.
    for r in reports.values():
        r.pop('candidate_recall',None);r.pop('candidate_stage',None)
        r['counts'].pop('candidate_gold_found',None);r['counts'].pop('exact_candidate_not_selected',None)
        for v in r['per_label'].values():v.pop('candidate_recall',None);v.pop('candidate_found',None)
    phashes={}
    for method,rows in predictions.items():
        file=out/f'{method}_dev_predictions.parquet';pq.write_table(pa.Table.from_pylist(rows),file,compression='zstd');phashes[method]=sha(file)
    if {s:sha(ROOT/s) for s in paths}!=hashes:raise ValueError('Sources changed during run')
    result={'train_docs':len(ids),'train_tokens':ntokens,'train_unaligned_entities':unaligned,'dev_unaligned_entities':bad_dev,
            'train_seconds':train_seconds,'seconds':time.monotonic()-start,'dev':reports,'model_sha256':sha(out/'model.crfsuite'),'prediction_hashes':phashes}
    (out/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({m:{'slot':r['slot_accuracy'],'entity_f1':r['entity_micro']['f1'],'exact_docs':r['counts']['exact_documents']} for m,r in reports.items()},indent=2))


if __name__=='__main__':main()
