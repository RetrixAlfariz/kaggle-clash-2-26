"""Test inference from selected, hash-verified local from-scratch neural checkpoint."""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch
from prepared_loader import PreparedData
from neural_sequence import encode_document,collate_encoded,build_model,decode_logits
from submit_m0 import validate_rows,format_prediction
from run_neural_sequence import sha,source_hashes

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=ROOT/'output/bitrase-1.algo-6/v1')
    p.add_argument('--ensemble',action='store_true');args=p.parse_args();out=args.run.resolve()
    config=json.loads((out/'config.json').read_text(encoding='utf-8'));report=json.loads((out/'report.json').read_text(encoding='utf-8'));epoch=report['selected_epoch']
    expected=report['epochs'][str(epoch)]['model_sha256'];checkpoint=out/f'epoch{epoch}.pt'
    if source_hashes()!=config['sources']:raise ValueError('Source hashes differ')
    if sha(checkpoint)!=expected or sha(out/'vocab.json')!=report['vocab_sha256']:raise ValueError('Checkpoint/vocabulary hash mismatch')
    destination=ROOT/'output/bitrase-1.algo-6/ensemble_4_8' if args.ensemble else out
    target=destination/'submission.csv'
    if target.exists():raise ValueError('Preserve prior CSV')
    vocab=json.loads((out/'vocab.json').read_text(encoding='utf-8'));model=build_model(len(vocab)).cuda()
    model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True))
    checkpoint_hashes={str(epoch):expected}
    if args.ensemble:
        ensemble_config=json.loads((destination/'config.json').read_text(encoding='utf-8'))
        if ensemble_config['parent_report_sha256']!=sha(out/'report.json') or ensemble_config['weights']!=[.5,.5]:
            raise ValueError('Ensemble provenance mismatch')
        models=[];checkpoint_hashes={}
        for e in (4,8):
            path=out/f'epoch{e}.pt';checkpoint_hashes[str(e)]=sha(path)
            if checkpoint_hashes[str(e)]!=report['epochs'][str(e)]['model_sha256']:raise ValueError('Ensemble checkpoint mismatch')
            member=build_model(len(vocab)).cuda();member.load_state_dict(torch.load(path,map_location='cuda',weights_only=True));models.append(member)
        class Ensemble(torch.nn.Module):
            def __init__(self):
                super().__init__();self.models=torch.nn.ModuleList(models)
            def forward(self,*inputs):
                return sum(m(*inputs).float() for m in self.models)/len(self.models)
        model=Ensemble()
    model.eval();torch.set_num_threads(4)
    data=PreparedData();test=data.load_test().to_pylist();slots=data.load_test_slots().to_pylist();encoded=[]
    for r in test:
        ts,ids,ch,fl=encode_document(r['full_text'],vocab)
        encoded.append((ts,ids,ch,fl,np.full(len(ids),-100,dtype=np.int64)))
    order=sorted(range(len(test)),key=lambda i:len(encoded[i][1]));results={};batch_size=config['batch_size'];start=time.monotonic()
    with torch.no_grad():
        for offset in range(0,len(order),batch_size):
            indices=order[offset:offset+batch_size];docs=[encoded[i] for i in indices];batch=collate_encoded(docs);lengths=batch['mask'].sum(1)
            with torch.autocast('cuda',dtype=torch.float16):
                logits=model(batch['words'].cuda(),batch['chars'].cuda(),batch['flags'].cuda(),lengths)
            counts=[test[i]['expected_entity_count'] for i in indices]
            spans=decode_logits(logits.float().cpu(),lengths.tolist(),docs,counts)
            for j,i in enumerate(indices):
                if len(spans[j])!=counts[j]:raise ValueError('K contract failure')
                results[test[i]['document_id']]=[format_prediction(s) for s in spans[j]]
            if offset%(batch_size*100)==0:print(f'Predicted {min(offset+batch_size,len(order))}/{len(order)} test docs',flush=True)
    rows=[{'row_id':s['row_id'],'document_id':s['document_id'],'Predicted':results[s['document_id']][s['slot']-1]} for s in slots]
    validate_rows(rows,{r['document_id']:r['full_text'] for r in test},{r['document_id']:r['expected_entity_count'] for r in test})
    if source_hashes()!=config['sources']:raise ValueError('Sources changed during inference')
    with target.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['row_id','Predicted'],extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    manifest={'rows':len(rows),'documents':len(test),'epoch':None if args.ensemble else epoch,'ensemble':args.ensemble,'sha256':sha(target),'checkpoint_hashes':checkpoint_hashes,
              'test_sha256':sha(ROOT/'output/prepared/v1/test.parquet'),'slots_sha256':sha(ROOT/'output/prepared/v1/test_slots.parquet'),
              'inference_script_sha256':sha(Path(__file__)),'pretrained':False,'uploaded':False,'seconds':time.monotonic()-start}
    (destination/'submission_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
