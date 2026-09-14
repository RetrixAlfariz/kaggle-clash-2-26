# /// script
# requires-python = "==3.13.2"
# dependencies = ["numpy==2.5.3", "pyarrow==25.0.1", "torch==2.8.0"]
# ///
"""Fit a small CRF transition matrix on frozen from-scratch neural emissions."""
import argparse, hashlib, json, random, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path.cwd()/"src"))
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from prepared_loader import PreparedData
from ner_evaluation import Evaluation, triples
from neural_sequence import LABELS, encode_document, collate_encoded, build_model
from sequence_crf import gold_tags, decode_k, spans_from_tags
from run_neural_sequence import source_hashes, clean_report

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'output/bitrase-1.algo-6/v1'
OUT=ROOT/'output/bitrase-1.algo-7/v1'
SEED=2026; TRAIN_DOCS=4000; EPOCHS=3
L=len(LABELS); ID={s:i for i,s in enumerate(LABELS)}

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def legal_mask():
    m=torch.zeros((L,L),dtype=torch.bool)
    for i,a in enumerate(LABELS):
        for j,b in enumerate(LABELS):
            # O/B may follow any non-I; I only follows same-label B or I.
            m[i,j]=not b.startswith('I-') or a in ('B-'+b[2:],b)
    return m

def sequence_nll(emission, gold, transition, mask):
    """CRF NLL; initial I tags are disallowed, matching decode_k."""
    legal=torch.where(mask, transition, torch.tensor(-torch.inf,device=transition.device))
    alpha=torch.full((L,),-torch.inf,device=emission.device)
    alpha[~torch.tensor([x.startswith('I-') for x in LABELS],device=emission.device)]=emission[0,~torch.tensor([x.startswith('I-') for x in LABELS],device=emission.device)]
    for t in range(1,emission.shape[0]):
        alpha=torch.logsumexp(alpha[:,None]+legal,dim=0)+emission[t]
    logz=torch.logsumexp(alpha,dim=0)
    score=emission[0,gold[0]]
    for t in range(1,len(gold)): score=score+emission[t,gold[t]]+legal[gold[t-1],gold[t]]
    return logz-score

def load_parent():
    report=json.loads((PARENT/'report.json').read_text(encoding='utf-8')); epoch=report['selected_epoch']
    config=json.loads((PARENT/'config.json').read_text(encoding='utf-8'))
    if source_hashes()!=config['sources']:raise ValueError('Parent source/data mismatch')
    if sha(PARENT/'vocab.json')!=report['vocab_sha256']:raise ValueError('Parent vocabulary mismatch')
    vocab=json.loads((PARENT/'vocab.json').read_text(encoding='utf-8'))
    model=build_model(len(vocab)).cuda(); ck=PARENT/f'epoch{epoch}.pt'
    if sha(ck)!=report['epochs'][str(epoch)]['model_sha256']:raise ValueError('Parent checkpoint mismatch')
    model.load_state_dict(torch.load(ck,map_location='cuda',weights_only=True)); model.eval()
    return model,vocab,epoch,ck

def emissions_for(model,vocab,row):
    ts,ids,ch,fl=encode_document(row['full_text'],vocab)
    doc=(ts,ids,ch,fl,np.full(len(ids),-100,dtype=np.int64)); batch=collate_encoded([doc]); lengths=batch['mask'].sum(1)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16): logits=model(batch['words'].cuda(),batch['chars'].cuda(),batch['flags'].cuda(),lengths)
    return ts,logits[0,:len(ts)].float().cpu()

def select_train(rows):
    return sorted(rows,key=lambda r:hashlib.sha1(r['document_id'].encode()).hexdigest())

def cache_rows(model,vocab,rows):
    encoded=[]
    for row in rows:
        ts,ids,ch,fl=encode_document(row['full_text'],vocab)
        encoded.append((ts,ids,ch,fl,np.full(len(ids),-100,dtype=np.int64)))
    order=sorted(range(len(rows)),key=lambda i:len(encoded[i][1])); result=[None]*len(rows)
    with torch.no_grad():
        for offset in range(0,len(order),32):
            indices=order[offset:offset+32];batch=collate_encoded([encoded[i] for i in indices]);lengths=batch['mask'].sum(1)
            with torch.autocast('cuda',dtype=torch.float16):
                logits=model(batch['words'].cuda(),batch['chars'].cuda(),batch['flags'].cuda(),lengths)
            logits=logits.float().cpu()
            for j,i in enumerate(indices):result[i]=(encoded[i][0],logits[j,:int(lengths[j])].clone())
    return result

def batch_nll(emission,gold,lengths,transition,mask):
    legal=transition.masked_fill(~mask,-torch.inf)
    initial_i=torch.tensor([s.startswith('I-') for s in LABELS],device=emission.device)
    alpha=emission[:,0].masked_fill(initial_i[None,:],-torch.inf)
    for t in range(1,emission.shape[1]):
        updated=torch.logsumexp(alpha[:,:,None]+legal[None,:,:],dim=1)+emission[:,t]
        alpha=torch.where((t<lengths)[:,None],updated,alpha)
    valid=torch.arange(emission.shape[1],device=emission.device)[None,:]<lengths[:,None]
    score=(emission.gather(2,gold[:,:,None]).squeeze(2)*valid).sum(1)
    # Gold padding uses O; torch.where avoids multiplying masked infinities.
    pair=legal[gold[:,:-1],gold[:,1:]]
    score+=torch.where(valid[:,1:],pair,torch.zeros_like(pair)).sum(1)
    return torch.logsumexp(alpha,dim=1)-score

def main():
    p=argparse.ArgumentParser();p.add_argument('--train-docs',type=int,default=TRAIN_DOCS);p.add_argument('--epochs',type=int,default=EPOCHS);p.add_argument('--output',type=Path,default=OUT);a=p.parse_args();out=a.output.resolve()
    if out.exists():raise ValueError('Preserve existing run')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for frozen parent inference')
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(4)
    paths=['src/run_neural_transition.py','src/run_neural_sequence.py','src/neural_sequence.py','src/sequence_crf.py','src/prepared_loader.py','src/ner_evaluation.py','note/bitrase-1.algo-7/protocol.md','output/prepared/v1/train.parquet','output/prepared/v1/dev.parquet','output/bitrase-1.algo-6/v1/report.json','output/bitrase-1.algo-6/v1/vocab.json']
    model,vocab,epoch,ck=load_parent(); paths.append(f'output/bitrase-1.algo-6/v1/epoch{epoch}.pt'); hashes={s:sha(ROOT/s) for s in paths}
    out.mkdir(parents=True); config={'experiment':'bitrase-1.algo-7','parent':'bitrase-1.algo-6','method':'CRF transition matrix on frozen neural emissions','pretrained':False,'train_docs':a.train_docs,'epochs':a.epochs,'seed':SEED,'sources':hashes}
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    train=select_train(PreparedData().load_train().to_pylist())[:a.train_docs]; mask=legal_mask(); transition=torch.nn.Parameter(torch.zeros((L,L))); opt=torch.optim.Adam([transition],lr=.03)
    cached=[]; start=time.monotonic()
    unaligned=0
    for row,(ts,e) in zip(train,cache_rows(model,vocab,train)):
        tags,bad=gold_tags(ts,row['entities']);unaligned+=bad
        cached.append((e,torch.tensor([ID[t] for t in tags],dtype=torch.long)))
    print(f'Cached {len(cached)} train docs; unaligned={unaligned}',flush=True)
    torch.set_num_threads(1)
    logs=[]
    for ep in range(a.epochs):
        order=sorted(range(len(cached)),key=lambda i:len(cached[i][0]));batches=[order[j:j+32] for j in range(0,len(order),32)]
        random.Random(SEED+ep).shuffle(batches);total=0.;ep_start=time.monotonic()
        for step,indices in enumerate(batches,1):
            es=[cached[i][0] for i in indices];gs=[cached[i][1] for i in indices]
            e=torch.nn.utils.rnn.pad_sequence(es,batch_first=True);g=torch.nn.utils.rnn.pad_sequence(gs,batch_first=True)
            lengths=torch.tensor([len(x) for x in es]);opt.zero_grad(set_to_none=True)
            loss=batch_nll(e,g,lengths,transition,mask).mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward()
            if not torch.isfinite(transition.grad).all():raise ValueError('Nonfinite gradients')
            torch.nn.utils.clip_grad_norm_([transition],5.);opt.step();total+=float(loss.detach())*len(indices)
            if step%25==0:print(f'Transition epoch{ep+1} batch{step}/{len(batches)}',flush=True)
        logs.append({'epoch':ep+1,'nll':total/len(cached),'seconds':time.monotonic()-ep_start})
        print(json.dumps(logs[-1]),flush=True)
        (out/'training_log.json').write_text(json.dumps(logs,indent=2),encoding='utf-8')
    trans=transition.detach().cpu().numpy(); np.save(out/'transitions.npy',trans)
    dev=PreparedData().load_dev('evaluation').to_pylist(); ev=Evaluation(); preds=[]
    torch.set_num_threads(4)
    for i,(row,(ts,e)) in enumerate(zip(dev,cache_rows(model,vocab,dev)),1):
        tags=decode_k(e.numpy(),trans,LABELS,row['expected_entity_count']); pred=spans_from_tags(ts,tags); ev.add(row['full_text'],triples(row['entities']),pred,pred); preds.append({'document_id':row['document_id'],'predicted':[{'start':x,'end':y,'label':z} for x,y,z in pred]})
        if i%1500==0:print(f'Evaluated {i}/{len(dev)} dev docs',flush=True)
    pq.write_table(pa.Table.from_pylist(preds),out/'dev_predictions.parquet',compression='zstd'); report={'epoch':epoch,'train_docs':len(train),'train_unaligned_entities':unaligned,'training':logs,'dev':{'known_k':clean_report(ev.report())},'seconds':time.monotonic()-start,'transition_sha256':sha(out/'transitions.npy'),'prediction_sha256':sha(out/'dev_predictions.parquet'),'pretrained':False}
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if {s:sha(ROOT/s) for s in paths}!=hashes:raise ValueError('Source changed during run')
    print(json.dumps({'slot_accuracy':report['dev']['known_k']['slot_accuracy'],'entity_f1':report['dev']['known_k']['entity_micro']['f1']},indent=2))

if __name__=='__main__':main()
