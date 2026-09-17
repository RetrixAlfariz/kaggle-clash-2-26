"""Fresh FIT-only encoder/head for a three-way TRAIN diagnostic; no dev access."""
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from prepared_loader import PreparedData
from neural_sequence import build_vocab, encode_document, tags_to_ids, collate_encoded, build_model, decode_logits, LABELS
from structure_families import group_and_sample
from span_probe import SpanHead, sampling_plan, sample_plan, universe, padded_hidden
from run_span_probe import optimize_batch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/bitrase-2/algo-3'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')

def split_rows(rows):
    idx,groups,_,report=group_and_sample(rows,sample_size=len(rows),max_sample_size=len(rows),min_families=1,seed=20260917)
    members={}
    for i,g in zip(idx,groups):members.setdefault(g,[]).append(i)
    ordered=sorted(members,key=lambda g:(hashlib.sha256(f'20260917:{g}'.encode()).digest(),g))
    split={'FIT':[],'CAL':[],'EVAL':[]}; assignment=[]
    for g in ordered:
        role='CAL' if len(split['CAL'])<1000 else 'EVAL' if len(split['EVAL'])<3000 else 'FIT'
        for i in members[g]:
            split[role].append(i);assignment.append(dict(document_id=rows[i]['document_id'],group_id=g,role=role,original_index=i))
    assert sum(map(len,split.values()))==len(rows)
    assert len({a['document_id'] for a in assignment})==len(rows)
    for role in split:split[role].sort()
    return split,assignment,report

def encode(rows,vocab,supervised):
    result=[];bad=0
    for row in rows:
        ts,ids,ch,fl=encode_document(row['full_text'],vocab)
        if supervised:y,b=tags_to_ids(ts,row['entities']);bad+=b
        else:y=np.full(len(ids),-100,dtype=np.int64)
        result.append((ts,ids,ch,fl,y))
    return result,bad

def train_encoder(encoded,vocab):
    model=build_model(len(vocab)).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024);loss_fn=torch.nn.CrossEntropyLoss(ignore_index=-100);logs=[]
    for epoch in range(1,5):
        rng=random.Random(2026+epoch);order=list(range(len(encoded)));rng.shuffle(order);batches=[]
        for offset in range(0,len(order),1024):
            block=sorted(order[offset:offset+1024],key=lambda i:len(encoded[i][1]))
            batches.extend(block[j:j+32] for j in range(0,len(block),32))
        rng.shuffle(batches);model.train();total=nt=0;start=time.monotonic()
        for step,ids in enumerate(batches,1):
            batch=collate_encoded([encoded[i] for i in ids]);lengths=batch['mask'].sum(1);target=batch['labels'].cuda()
            # Retry overflow on the identical dropout draw, never silently skip FIT examples.
            state=torch.cuda.get_rng_state()
            for attempt in range(16):
                if attempt:torch.cuda.set_rng_state(state)
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):
                    z=model(batch['words'].cuda(),batch['chars'].cuda(),batch['flags'].cuda(),lengths)
                    loss=loss_fn(z.reshape(-1,len(LABELS)),target.reshape(-1))
                if not torch.isfinite(loss):raise ValueError('Nonfinite encoder loss')
                scaler.scale(loss).backward();scaler.unscale_(opt)
                finite=all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
                if finite:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(opt);scaler.update();break
                scaler.step(opt);scaler.update()
            else:raise ValueError('Encoder overflow retry limit')
            nn=int(lengths.sum());total+=float(loss.detach())*nn;nt+=nn
            if step%250==0:print(f'Encoder epoch {epoch}/4 step {step}/{len(batches)} loss {total/nt:.5f}',flush=True)
        logs.append(dict(epoch=epoch,loss=total/nt,seconds=time.monotonic()-start));write(OUT/'encoder_training.json',logs)
        print(logs[-1],flush=True)
    torch.save(model.state_dict(),OUT/'encoder_epoch4.pt');model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model

def hidden_cache(model,rows,encoded,role,groups):
    path=OUT/role;path.mkdir();lengths=[len(x[1]) for x in encoded]
    offsets=np.r_[0,np.cumsum(lengths)].astype(np.int64);np.save(path/'token_offsets.npy',offsets)
    bounds=np.lib.format.open_memmap(path/'boundaries.npy',mode='w+',dtype=np.int32,shape=(int(offsets[-1]),2))
    hidden=None;old={};order=sorted(range(len(rows)),key=lambda i:lengths[i]);captured=[]
    hook=model.out.register_forward_pre_hook(lambda module,inputs:captured.append(inputs[0].detach()))
    try:
        with torch.no_grad():
            for begin in range(0,len(order),32):
                ids=order[begin:begin+32];docs=[encoded[i] for i in ids];batch=collate_encoded(docs);lens=batch['mask'].sum(1);captured.clear()
                with torch.autocast('cuda',dtype=torch.float16):
                    z=model(batch['words'].cuda(),batch['chars'].cuda(),batch['flags'].cuda(),lens)
                assert len(captured)==1
                h=captured[0].cpu().numpy()
                if hidden is None:hidden=np.lib.format.open_memmap(path/'hidden.npy',mode='w+',dtype=h.dtype,shape=(int(offsets[-1]),256))
                for j,i in enumerate(ids):
                    a,b=offsets[i:i+2];hidden[a:b]=h[j,:lengths[i]];bounds[a:b]=[(x,y) for x,y,w in docs[j][0]]
                if role=='FIT':
                    pred=decode_logits(z.float().cpu(),lens.tolist(),docs,[rows[i]['expected_entity_count'] for i in ids])
                    for i,pp in zip(ids,pred):old[i]=pp
                if (begin+32)%6400==0:print(f'{role} hidden {begin+32}/{len(rows)}',flush=True)
    finally:hook.remove()
    hidden.flush();bounds.flush()
    meta=[dict(document_id=r['document_id'],group_id=groups[r['document_id']],expectedK=r['expected_entity_count'],
               token_count=lengths[i],gold=[{k:g[k] for k in ('start','end','label')} for g in r['entities']]) for i,r in enumerate(rows)]
    write(path/'metadata.json',meta)
    return hidden,bounds,offsets,old

def train_head(rows,hidden,bounds,offsets,old):
    plans=[sampling_plan(bounds[offsets[i]:offsets[i+1]],r['entities'],old[i]) for i,r in enumerate(rows)]
    # Same P0 head seed, independent of how many FIT encoder batches consumed RNG.
    random.seed(2026);np.random.seed(2026);torch.manual_seed(2026);torch.cuda.manual_seed_all(2026)
    head=SpanHead().cuda();opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024);logs=[]
    for epoch in (1,2):
        head.train();order=list(range(len(rows)));random.Random(2026+epoch).shuffle(order);total=pop=0;start=time.monotonic()
        for begin in range(0,len(order),32):
            ids=order[begin:begin+32];hs=[hidden[offsets[i]:offsets[i+1]] for i in ids];h,pr=padded_hidden(hs)
            samples=[sample_plan(plans[i],2026,epoch,i) for i in ids]
            pairs=np.concatenate([universe(len(hh))[s[0]] for hh,s in zip(hs,samples)])
            dd=np.repeat(np.arange(len(ids)),[len(s[0]) for s in samples]);yy=np.concatenate([s[1] for s in samples]);tags=np.concatenate([s[2] for s in samples]);ww=np.concatenate([s[3] for s in samples])
            den=sum(plans[i]['population'] for i in ids)
            loss,_,_=optimize_batch(head,opt,scaler,h,pr,pairs,dd,yy,tags,ww,den);total+=loss;pop+=den
            if begin%6400==0:print(f'Head epoch {epoch}/2 docs {begin}/{len(rows)} loss {total/pop:.5f}',flush=True)
        logs.append(dict(epoch=epoch,weighted_ce=total/pop,seconds=time.monotonic()-start));write(OUT/'head_training.json',logs)
    torch.save(head.state_dict(),OUT/'head_epoch2.pt');head.eval();return head

def score(head,role,hidden,offsets):
    path=OUT/role;counts=[len(universe(int(offsets[i+1]-offsets[i]))) for i in range(len(offsets)-1)]
    so=np.r_[0,np.cumsum(counts)].astype(np.int64);np.save(path/'span_offsets.npy',so)
    pairs=np.lib.format.open_memmap(path/'pairs.npy',mode='w+',dtype=np.int32,shape=(int(so[-1]),2))
    logits=np.lib.format.open_memmap(path/'logits.npy',mode='w+',dtype=np.float32,shape=(int(so[-1]),8))
    order=sorted(range(len(counts)),key=lambda i:counts[i])
    with torch.no_grad():
        for begin in range(0,len(order),32):
            ids=order[begin:begin+32];hs=[hidden[offsets[i]:offsets[i+1]] for i in ids];h,pr=padded_hidden(hs)
            pp=np.concatenate([universe(len(x)) for x in hs]);dd=np.repeat(np.arange(len(ids)),[counts[i] for i in ids]);values=[]
            for start in range(0,len(pp),32768):
                p=pp[start:start+32768];d=dd[start:start+32768]
                with torch.autocast('cuda',dtype=torch.float16):
                    z=head(h,pr,torch.as_tensor(d,device='cuda'),torch.as_tensor(p[:,0],device='cuda'),torch.as_tensor(p[:,1],device='cuda'))
                values.append(z.float().cpu().numpy())
            zz=np.concatenate(values);assert np.isfinite(zz).all();j=0
            for i in ids:
                a,b=so[i:i+2];pairs[a:b]=pp[j:j+counts[i]];logits[a:b]=zz[j:j+counts[i]];j+=counts[i]
    logits.flush();pairs.flush()
    write(path/'manifest.json',dict(complete=True,split=role,documents=len(counts),
        files={p.name:sha(p) for p in path.iterdir() if p.is_file()}))

def main():
    if OUT.exists():raise FileExistsError(OUT)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(4);random.seed(2026);np.random.seed(2026);torch.manual_seed(2026);torch.cuda.manual_seed_all(2026)
    paths=['src/train_scale_ranking.py','src/neural_sequence.py','src/span_probe.py','src/run_span_probe.py',
           'src/sequence_crf.py','src/structure_families.py','src/audit_template_similarity.py','src/prepared_loader.py',
           'note/bitrase-2/algo-3/protocol.md','output/prepared/v1/train.parquet']
    hashes={p:sha(ROOT/p) for p in paths};OUT.mkdir(parents=True);write(OUT/'training_inputs.json',hashes)
    rows=PreparedData().load_train('train_fitting').to_pylist();print('Grouping prepared TRAIN only',flush=True)
    split,assignment,group_report=split_rows(rows);pq.write_table(pa.Table.from_pylist(assignment),OUT/'assignments.parquet');write(OUT/'grouping.json',group_report)
    groups={a['document_id']:a['group_id'] for a in assignment};fit=[rows[i] for i in split['FIT']]
    write(OUT/'config.json',dict(seed=2026,encoder_epochs=4,head_epochs=2,counts={k:len(v) for k,v in split.items()},
        split_sha256=sha(OUT/'assignments.parquet'),gpu=torch.cuda.get_device_name(),torch=torch.__version__,pretrained=False,dev_accessed=False))
    print('Split sizes', {k:len(v) for k,v in split.items()},flush=True)
    vocab=build_vocab(r['full_text'] for r in fit);write(OUT/'vocab.json',vocab)
    encoded,bad=encode(fit,vocab,True);write(OUT/'fit_alignment.json',{'unaligned':bad});print('FIT encoded; fresh training starts',flush=True)
    model=train_encoder(encoded,vocab);hidden,bounds,offsets,old=hidden_cache(model,fit,encoded,'FIT',groups)
    del encoded
    head=train_head(fit,hidden,bounds,offsets,old);del hidden,bounds,offsets,old,fit
    for role in ('CAL','EVAL'):
        subset=[rows[i] for i in split[role]];enc,_=encode(subset,vocab,False)
        hh,bb,oo,_=hidden_cache(model,subset,enc,role,groups);score(head,role,hh,oo)
        del hh,bb,oo,enc,subset
        print(role+' frozen scores saved',flush=True)
    assert hashes=={p:sha(ROOT/p) for p in paths}
    write(OUT/'training_complete.json',dict(complete=True,encoder_sha256=sha(OUT/'encoder_epoch4.pt'),
        head_sha256=sha(OUT/'head_epoch2.pt'),vocab_sha256=sha(OUT/'vocab.json'),input_hashes_unchanged=True))

if __name__=='__main__':main()
