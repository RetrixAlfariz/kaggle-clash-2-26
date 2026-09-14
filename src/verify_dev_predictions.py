"""Independent counts from saved predictions; dev only, no model imports."""
import argparse
import hashlib
import json
from pathlib import Path
import pyarrow.parquet as pq
from prepared_loader import PreparedData


def main():
    p=argparse.ArgumentParser();p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--method',required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    saved=pq.read_table(args.predictions).to_pylist();pred={r['document_id']:r['predicted'] for r in saved}
    dev=PreparedData().load_dev('evaluation').to_pylist();errors=[]
    if len(pred)!=len(saved):errors.append('duplicate prediction document')
    if set(pred)!={r['document_id'] for r in dev}:errors.append('document coverage mismatch')
    total_gold=total_pred=slot=entity=boundary=exact=count_correct=0
    labels={'NAME','DATE','PHONE','EMAIL','ADDRESS','USERNAME','JOB_TITLE'}
    for row in dev:
        gold=sorted((e['start'],e['end'],e['label']) for e in row['entities'])
        spans=[(e['start'],e['end'],e['label']) for e in pred.get(row['document_id'],[])]
        if spans!=sorted(spans) or len(set(spans))!=len(spans):errors.append('sort/duplicate:'+row['document_id'])
        if any(not 0<=a<b<=len(row['full_text']) or label not in labels for a,b,label in spans):errors.append('bounds/label:'+row['document_id'])
        if any(a[1]>b[0] for a,b in zip(spans,spans[1:])):errors.append('overlap:'+row['document_id'])
        total_gold+=len(gold);total_pred+=len(spans);slot+=sum(a==b for a,b in zip(gold,spans))
        entity+=len(set(gold)&set(spans));boundary+=len({s[:2] for s in gold}&{s[:2] for s in spans})
        exact+=int(gold==spans);count_correct+=int(len(gold)==len(spans))
    metrics={'slot_accuracy':slot/total_gold,'entity_f1':2*entity/(total_gold+total_pred),
             'boundary_f1':2*boundary/(total_gold+total_pred),'slot_correct':slot,'entity_correct':entity,
             'gold':total_gold,'predicted':total_pred,'exact_documents':exact,'count_correct_documents':count_correct}
    reported=json.loads(args.report.read_text())['dev'][args.method]
    comparisons={'slot_accuracy':reported['slot_accuracy'],'entity_f1':reported['entity_micro']['f1'],
                 'boundary_f1':reported['boundary_micro']['f1'],**{k:reported['counts'][k] for k in ('slot_correct','entity_correct','gold','predicted','exact_documents','count_correct_documents')}}
    for k,v in comparisons.items():
        if abs(metrics[k]-v)>1e-12:errors.append('metric mismatch:'+k)
    with args.predictions.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    out={'predictions':str(args.predictions.resolve()),'prediction_sha256':digest,'documents':len(dev),'metrics':metrics,'errors':errors,'verified':not errors}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
