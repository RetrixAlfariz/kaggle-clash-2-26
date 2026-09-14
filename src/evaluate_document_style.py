"""Text-only style prediction on every dev document; no gold eligibility gating."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from prepared_loader import PreparedData
from ner_evaluation import Evaluation,triples
from run_document_style import adjust

ROOT=Path(__file__).resolve().parents[1]


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,default=ROOT/'output/bitrase-1.algo-3/crf12k/known_k_dev_predictions.parquet')
    p.add_argument('--output',type=Path,default=ROOT/'output/bitrase-1.algo-5/text_only_v2');args=p.parse_args();out=args.output.resolve()
    if out.exists():raise ValueError('Preserve previous results')
    modelpath=ROOT/'output/bitrase-1.algo-5/model.pkl'
    sources={str(s):sha(s) for s in [Path(__file__),ROOT/'src/run_document_style.py',args.base,modelpath]}
    with modelpath.open('rb') as f:model=pickle.load(f)
    dev=PreparedData().load_dev('evaluation').to_pylist()
    base={r['document_id']:triples(r['predicted']) for r in pq.read_table(args.base).to_pylist()}
    if set(base)!={r['document_id'] for r in dev}:raise ValueError('Coverage mismatch')
    classes=list(model.named_steps['clf'].classes_)
    probs=model.predict_proba([r['full_text'] for r in dev])[:,classes.index('included')]
    # Prediction stage sees only text, pre-existing predicted spans, and model probabilities.
    adjusted={};invalid=0
    for doc,text,prob in zip([r['document_id'] for r in dev],[r['full_text'] for r in dev],probs):
        try:adjusted[doc]=adjust(text,base[doc],'included' if prob>=.5 else 'excluded')
        except ValueError:adjusted[doc]=base[doc];invalid+=1
    evaluators={m:Evaluation() for m in ('base','adjusted')}
    for row in dev:
        gold=triples(row['entities'])
        for m,pred in [('base',base[row['document_id']]),('adjusted',adjusted[row['document_id']])]:
            evaluators[m].add(row['full_text'],gold,pred,pred)
    reports={m:e.report() for m,e in evaluators.items()}
    for r in reports.values():
        r.pop('candidate_recall',None);r.pop('candidate_stage',None);r['counts'].pop('candidate_gold_found',None)
        for v in r['per_label'].values():v.pop('candidate_recall',None);v.pop('candidate_found',None)
    out.mkdir(parents=True)
    rows=[{'document_id':doc,'predicted':[{'start':a,'end':b,'label':l} for a,b,l in spans]} for doc,spans in adjusted.items()]
    pq.write_table(pa.Table.from_pylist(rows),out/'dev_predictions.parquet',compression='zstd')
    report={'dev':reports,'invalid_adjustments':invalid,'gold_gating':False,'model_refit':False,'sources':sources,'prediction_sha256':sha(out/'dev_predictions.parquet')}
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({m:{'slot':r['slot_accuracy'],'entity_f1':r['entity_micro']['f1']} for m,r in reports.items()},indent=2))


if __name__=='__main__':main()
