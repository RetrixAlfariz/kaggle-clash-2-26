# /// script
# requires-python = "==3.13.2"
# dependencies = ["pyarrow==25.0.1", "numpy==2.5.3", "python-crfsuite==0.9.12"]
# ///
"""Generate exactly-K test CSV from a verified local from-scratch CRF checkpoint."""
import argparse
import csv
import json
from pathlib import Path
import time
import pycrfsuite
from prepared_loader import PreparedData
from submit_m0 import validate_rows, format_prediction
from sequence_crf import tokens, features, spans_from_tags, model_arrays, emissions, decode_k
from run_sequence_crf import sha

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();out=args.run.resolve()
    config=json.loads((out/'config.json').read_text());report=json.loads((out/'report.json').read_text())
    for s,h in config['sources'].items():
        if sha(ROOT/s)!=h:raise ValueError('Source hash mismatch: '+s)
    if sha(out/'model.crfsuite')!=report['model_sha256']:raise ValueError('Model hash mismatch')
    target=out/'submission.csv'
    if target.exists():raise ValueError('CSV already exists')
    tagger=pycrfsuite.Tagger();tagger.open(str(out/'model.crfsuite'))
    labels,lookup,transitions=model_arrays(tagger)
    data=PreparedData();test=data.load_test().to_pylist();slots=data.load_test_slots().to_pylist()
    bydoc={};constrained_docs=0;start=time.monotonic()
    for i,row in enumerate(test,1):
        text=row['full_text'];ts=tokens(text);seq=features(text,ts);tags=tagger.tag(seq)
        pred=spans_from_tags(ts,tags);k=row['expected_entity_count']
        valid=all(not tag.startswith('I-') or (j>0 and tags[j-1] in ('B-'+tag[2:],tag)) for j,tag in enumerate(tags))
        if len(pred)!=k or not valid:
            pred=spans_from_tags(ts,decode_k(emissions(seq,lookup,len(labels)),transitions,labels,k));constrained_docs+=1
        if len(pred)!=k:raise ValueError('K contract failure')
        bydoc[row['document_id']]=[format_prediction(s) for s in pred]
        if i%5000==0:print(f'Predicted {i}/{len(test)} test docs',flush=True)
    rows=[{'row_id':s['row_id'],'document_id':s['document_id'],'Predicted':bydoc[s['document_id']][s['slot']-1]} for s in slots]
    validate_rows(rows,{r['document_id']:r['full_text'] for r in test},{r['document_id']:r['expected_entity_count'] for r in test})
    for s,h in config['sources'].items():
        if sha(ROOT/s)!=h:raise ValueError('Source changed during inference: '+s)
    with target.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['row_id','Predicted'],extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    manifest={'rows':len(rows),'documents':len(test),'count_constrained_docs':constrained_docs,'sha256':sha(target),
              'model_sha256':report['model_sha256'],'inference_source_sha256':sha(Path(__file__)),
              'test_sha256':sha(ROOT/'output/prepared/v1/test.parquet'),'slots_sha256':sha(ROOT/'output/prepared/v1/test_slots.parquet'),
              'pretrained':False,'uploaded':False,'seconds':time.monotonic()-start}
    (out/'submission_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
