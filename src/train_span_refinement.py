"""Algo-5 adaptive development: conservative continuation versus refreshed weighted CE."""
import hashlib
import json
import random
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import train_global_ranking as g
from span_probe import SpanHead, sampling_plan, sample_plan, universe, padded_hidden, decode_spans, character_spans

ROOT=g.ROOT
SOURCE=g.OUT
OUT=ROOT/'output/bitrase-2/algo-5/run'


def multipliers(tags):
    return np.where((tags&1)!=0,4.,np.where((tags&4)!=0,4.,np.where((tags&2)!=0,2.,1.))).astype(np.float32)


def weighted_ce(head,h,pr,ids,plans,epoch):
    samples=[sample_plan(plans[i],2026,epoch,i) for i in ids]
    pp=np.concatenate([universe(plans[i]['n'])[s[0]] for i,s in zip(ids,samples)])
    dd=np.repeat(np.arange(len(ids)),[len(s[0]) for s in samples]); yy=np.concatenate([s[1] for s in samples])
    weights=np.concatenate([s[3]*multipliers(s[2]) for s in samples])
    denominator=sum(4*len(plans[i]['positive'])+multipliers(plans[i]['tags']).sum()+len(plans[i]['remaining']) for i in ids)
    if not np.isclose(weights.sum(dtype=np.float64),denominator,rtol=1e-5): raise ValueError('Weighted HT denominator mismatch')
    loss=h.new_zeros(())
    for off in range(0,len(pp),g.CHUNK):
        p=pp[off:off+g.CHUNK]; d=dd[off:off+g.CHUNK]
        z=head(h,pr,torch.as_tensor(d,device=h.device),torch.as_tensor(p[:,0],device=h.device),torch.as_tensor(p[:,1],device=h.device))
        ce=F.cross_entropy(z,torch.as_tensor(yy[off:off+g.CHUNK],device=h.device),reduction='none')
        loss=loss+(ce*torch.as_tensor(weights[off:off+g.CHUNK],device=h.device)).sum()/denominator
    return loss


def map_worker(task):
    n,z,k=task; energies=z[:,1:].astype(np.float64)-z[:,:1].astype(np.float64)
    labels=energies.argmax(1)
    return decode_spans(n,energies[np.arange(len(labels)),labels],labels+1,k)


def refresh(head,hidden,bounds,offsets,rows,old):
    predictions={}; t=time.monotonic()
    with ProcessPoolExecutor(max_workers=4) as pool:
        for begin in range(0,len(rows),32):
            ids=list(range(begin,min(begin+32,len(rows)))); arrays=[hidden[offsets[i]:offsets[i+1]] for i in ids]
            values=g.score_batch(head,arrays)
            tasks=[(len(h),z,rows[i]['expected_entity_count']) for h,z,i in zip(arrays,values,ids)]
            for i,tokens in zip(ids,pool.map(map_worker,tasks)):
                predictions[i]=character_spans(tokens,bounds[offsets[i]:offsets[i+1]])
            if begin%6400==0: print(f'CE mining {begin}/{len(rows)} {time.monotonic()-t:.1f}s',flush=True)
    plans=[]
    for i,r in enumerate(rows):
        bb=bounds[offsets[i]:offsets[i+1]]
        p=sampling_plan(bb,r['entities'],old[i]+predictions[i]); p['n']=len(bb); plans.append(p)
    return plans,predictions


def score(head,arm):
    cache=SOURCE/'cache/EVAL'; hh=np.load(cache/'hidden.npy',mmap_mode='r'); oo=np.load(cache/'token_offsets.npy')
    src=SOURCE/'EVAL/control'; path=OUT/'EVAL'/arm; path.mkdir(parents=True)
    for name in ('pairs.npy','span_offsets.npy','token_offsets.npy','boundaries.npy','metadata.json'): shutil.copyfile(src/name,path/name)
    so=np.load(path/'span_offsets.npy')
    zz=np.lib.format.open_memmap(path/'logits.npy',mode='w+',dtype=np.float32,shape=(int(so[-1]),8))
    for start in range(0,len(oo)-1,32):
        ids=list(range(start,min(start+32,len(oo)-1))); arrays=[hh[oo[i]:oo[i+1]] for i in ids]
        for i,z in zip(ids,g.score_batch(head,arrays)): zz[so[i]:so[i+1]]=z
    zz.flush()
    g.write(path/'manifest.json',dict(complete=True,documents=len(oo)-1,files={p.name:g.sha(p) for p in path.iterdir() if p.is_file()}))


def main():
    if OUT.exists(): raise FileExistsError(OUT)
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; g.seed()
    inputs=['src/train_span_refinement.py','src/train_global_ranking.py','src/span_probe.py',
        'note/bitrase-2/algo-5/protocol.md','output/bitrase-2/algo-4/run/control_epoch2.pt',
        'output/bitrase-2/algo-4/run/assignments.parquet','output/bitrase-2/algo-4/run/cache/encoder_epoch4.pt']
    hashes={p:g.sha(ROOT/p) for p in inputs}; OUT.mkdir(parents=True); g.write(OUT/'inputs.json',hashes)
    g.write(OUT/'config.json',dict(epochs=[3,4],lr=.0003,weight_decay=.01,clip=1,batch=32,seed=2026,
        positives_weight=4,hard_fp_weight=4,near_only_weight=2,other_weight=1,pretrained=False,
        adaptive_development=True,prior_eval='algo-4 EVAL2, now reused for development',no_dev_holdout_test=True))
    rows, assignments=g.input_rows(); fit=[r for r,a in zip(rows,assignments) if a['role']=='FIT']
    cache=SOURCE/'cache/FIT'; hh=np.load(cache/'hidden.npy',mmap_mode='r'); bb=np.load(cache/'boundaries.npy',mmap_mode='r'); oo=np.load(cache/'token_offsets.npy')
    old={int(i):p for i,p in json.loads((SOURCE/'cache/old_predictions.json').read_text()).items()}
    original=[]
    for i,r in enumerate(fit):
        bounds=bb[oo[i]:oo[i+1]]; p=sampling_plan(bounds,r['entities'],old[i]); p['n']=len(bounds); original.append(p)
    logs={}
    for arm in ('control','challenger'):
        head=SpanHead().cuda(); head.load_state_dict(torch.load(SOURCE/'control_epoch2.pt',weights_only=True)); opt=torch.optim.AdamW(head.parameters(),lr=.0003,weight_decay=.01)
        logs[arm]=[]
        for epoch in (3,4):
            if arm=='challenger':
                plans,pred=refresh(head,hh,bb,oo,fit,old); g.write(OUT/f'mined_epoch{epoch}.json',pred)
            else: plans=original
            order=list(range(len(fit))); random.Random(2026+epoch).shuffle(order)
            g.write(OUT/f'{arm}_epoch{epoch}_order.json',order)
            losses=[]; start=time.monotonic()
            for begin in range(0,len(order),32):
                ids=order[begin:begin+32]; h,pr=padded_hidden([hh[oo[i]:oo[i+1]] for i in ids]); head.train()
                rng=2026+100000*epoch+begin//32; torch.manual_seed(rng); torch.cuda.manual_seed_all(rng); opt.zero_grad(set_to_none=True)
                loss=weighted_ce(head,h,pr,ids,plans,epoch) if arm=='challenger' else g.ce_loss(head,h,pr,ids,plans,epoch)
                if not torch.isfinite(loss): raise ValueError('Nonfinite CE')
                loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True); opt.step(); losses.append(float(loss.detach()))
                if begin%6400==0: print(f'{arm} epoch{epoch} {begin}/{len(order)} loss={np.mean(losses):.6f}',flush=True)
            logs[arm].append(dict(epoch=epoch,mean_batch_loss=float(np.mean(losses)),seconds=time.monotonic()-start)); g.write(OUT/'training.json',logs)
        torch.save(head.state_dict(),OUT/f'{arm}_epoch4.pt')
        # Fixed final-only scoring; no intermediate evaluation/checkpoint selection.
        score(head,arm)
    assert hashes=={p:g.sha(ROOT/p) for p in inputs}
    g.write(OUT/'complete.json',dict(complete=True,inputs_unchanged=True,checkpoints={arm:g.sha(OUT/f'{arm}_epoch4.pt') for arm in logs}))


if __name__=='__main__': main()
