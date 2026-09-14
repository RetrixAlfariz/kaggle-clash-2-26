"""GPU training of a from-scratch byte-CNN BiLSTM; no pretrained artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from prepared_loader import PreparedData
from ner_evaluation import Evaluation,triples
from neural_sequence import build_vocab,encode_document,tags_to_ids,collate_encoded,build_model,decode_logits,LABELS
from sequence_crf import spans_from_tags

ROOT=Path(__file__).resolve().parents[1]


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def source_hashes():
    return {s:sha(ROOT/s) for s in ['src/run_neural_sequence.py','src/neural_sequence.py','src/sequence_crf.py',
            'src/prepared_loader.py','src/ner_evaluation.py','note/bitrase-1.algo-6/protocol.md',
            'output/prepared/v1/train.parquet','output/prepared/v1/dev.parquet']}


def clean_report(r):
    r.pop('candidate_recall',None);r.pop('candidate_stage',None)
    r['counts'].pop('candidate_gold_found',None);r['counts'].pop('exact_candidate_not_selected',None)
    for v in r['per_label'].values():v.pop('candidate_recall',None);v.pop('candidate_found',None)
    return r


def evaluate(model,dev,encoded,batch_size):
    model.eval();evaluators={m:Evaluation() for m in ('known_k','raw')};predictions={m:[] for m in evaluators}
    order=sorted(range(len(dev)),key=lambda i:len(encoded[i][1]))
    with torch.no_grad():
        for start in range(0,len(order),batch_size):
            indices=order[start:start+batch_size];docs=[encoded[i] for i in indices];batch=collate_encoded(docs);lengths=batch['mask'].sum(1)
            with torch.autocast('cuda',dtype=torch.float16):
                logits=model(batch['words'].cuda(),batch['chars'].cuda(),batch['flags'].cuda(),lengths)
            logits=logits.float().cpu();ks=[dev[i]['expected_entity_count'] for i in indices]
            constrained=decode_logits(logits,lengths.tolist(),docs,ks)
            for j,i in enumerate(indices):
                row=dev[i];ts=docs[j][0];raw=spans_from_tags(ts,[LABELS[int(v)] for v in logits[j,:len(ts)].argmax(1)])
                gold=triples(row['entities'])
                for m,pred in [('raw',raw),('known_k',constrained[j])]:
                    if m=='known_k' and len(pred)!=ks[j]:raise ValueError('K mismatch')
                    evaluators[m].add(row['full_text'],gold,pred,pred)
                    predictions[m].append({'document_id':row['document_id'],'predicted':[{'start':a,'end':b,'label':l} for a,b,l in pred]})
            if start%(batch_size*50)==0:print(f'Dev {min(start+batch_size,len(order))}/{len(order)}',flush=True)
    return {m:clean_report(e.report()) for m,e in evaluators.items()},predictions


def main():
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=8);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--output',type=Path,default=ROOT/'output/bitrase-1.algo-6/v1');a=p.parse_args();out=a.output.resolve()
    if out.exists():raise ValueError('Preserve existing run')
    if not torch.cuda.is_available():raise RuntimeError('This protocol requires local CUDA GPU')
    if a.epochs not in (4,8):raise ValueError('Protocol allows4 or8 epochs')
    random.seed(2026);np.random.seed(2026);torch.manual_seed(2026);torch.cuda.manual_seed_all(2026)
    torch.set_num_threads(4)
    hashes=source_hashes();out.mkdir(parents=True)
    config={'experiment':'bitrase-1.algo-6','pretrained':False,'seed':2026,'epochs':a.epochs,'batch_size':a.batch_size,
            'optimizer':'AdamW','lr':.001,'weight_decay':.01,'gradient_clip':1.,'evaluation_epochs':[4,8],
            'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,
            'numpy':np.__version__,'pyarrow':pa.__version__,'sources':hashes}
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    data=PreparedData();train=data.load_train().to_pylist();dev=data.load_dev('evaluation').to_pylist()
    print('Building train-only vocabulary',flush=True);vocab=build_vocab(r['full_text'] for r in train)
    (out/'vocab.json').write_text(json.dumps(vocab,ensure_ascii=False),encoding='utf-8')
    encoded=[];unaligned=ntokens=0;start_time=time.monotonic()
    for i,row in enumerate(train,1):
        ts,ids,ch,fl=encode_document(row['full_text'],vocab);ys,bad=tags_to_ids(ts,row['entities']);unaligned+=bad;ntokens+=len(ts)
        encoded.append((None,ids,ch,fl,ys))
        if i%10000==0:print(f'Encoded {i}/{len(train)} train docs',flush=True)
    del train
    dev_encoded=[];dev_bad=0
    for row in dev:
        ts,ids,ch,fl=encode_document(row['full_text'],vocab);ys,bad=tags_to_ids(ts,row['entities']);dev_bad+=bad
        dev_encoded.append((ts,ids,ch,fl,ys))
    model=build_model(len(vocab)).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda');loss_fn=torch.nn.CrossEntropyLoss(ignore_index=-100)
    epoch_logs=[];results={}
    for epoch in range(1,a.epochs+1):
        rng=random.Random(2026+epoch);order=list(range(len(encoded)));rng.shuffle(order)
        batches=[]
        for offset in range(0,len(order),a.batch_size*32):
            block=sorted(order[offset:offset+a.batch_size*32],key=lambda i:len(encoded[i][1]))
            batches.extend(block[j:j+a.batch_size] for j in range(0,len(block),a.batch_size))
        rng.shuffle(batches);model.train();total_loss=total_tokens=0;epoch_start=time.monotonic()
        for step,indices in enumerate(batches,1):
            batch=collate_encoded([encoded[i] for i in indices]);lengths=batch['mask'].sum(1);target=batch['labels'].cuda()
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                logits=model(batch['words'].cuda(),batch['chars'].cuda(),batch['flags'].cuda(),lengths)
                loss=loss_fn(logits.reshape(-1,len(LABELS)),target.reshape(-1))
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(opt);scaler.update();n=int(lengths.sum());total_loss+=float(loss.detach())*n;total_tokens+=n
            if step%250==0:print(f'Epoch{epoch}/{a.epochs} step{step}/{len(batches)} loss{total_loss/total_tokens:.5f}',flush=True)
        log={'epoch':epoch,'loss':total_loss/total_tokens,'seconds':time.monotonic()-epoch_start};epoch_logs.append(log)
        (out/'training_log.json').write_text(json.dumps(epoch_logs,indent=2),encoding='utf-8');print(json.dumps(log),flush=True)
        if epoch not in (4,8):continue
        checkpoint=out/f'epoch{epoch}.pt';torch.save(model.state_dict(),checkpoint)
        reports,predictions=evaluate(model,dev,dev_encoded,a.batch_size)
        ph={}
        for m,rows in predictions.items():
            file=out/f'epoch{epoch}_{m}_dev_predictions.parquet';pq.write_table(pa.Table.from_pylist(rows),file,compression='zstd');ph[m]=sha(file)
        r={'epoch':epoch,'dev':reports,'model_sha256':sha(checkpoint),'prediction_hashes':ph}
        (out/f'epoch{epoch}_report.json').write_text(json.dumps(r,indent=2),encoding='utf-8');results[str(epoch)]=r
        print(json.dumps({'epoch':epoch,'slot_accuracy':reports['known_k']['slot_accuracy'],'entity_f1':reports['known_k']['entity_micro']['f1']}),flush=True)
    if source_hashes()!=hashes:raise ValueError('Sources changed during training')
    selected=max(results,key=lambda e:results[e]['dev']['known_k']['slot_accuracy'])
    final={'selected_epoch':int(selected),'epochs':results,'train_docs':len(encoded),'train_tokens':ntokens,
           'train_unaligned_entities':unaligned,'dev_unaligned_entities':dev_bad,'seconds':time.monotonic()-start_time,'vocab_sha256':sha(out/'vocab.json')}
    (out/'report.json').write_text(json.dumps(final,indent=2),encoding='utf-8')


if __name__=='__main__':main()
