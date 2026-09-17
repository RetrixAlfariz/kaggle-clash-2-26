"""Algo-4: fixed fresh-encoder, matched CE and global-ranking heads on TRAIN only."""
import argparse
import hashlib
import json
import random
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

import train_scale_ranking as base
from audit_algo4_exposure import whole_family_eligible
from prepared_loader import PreparedData
from span_probe import SpanHead, LABELS, padded_hidden, sample_plan, sampling_plan, universe

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/bitrase-2/algo-4/run'
SEED = 2026
SPLIT_SEED = 20260919
WORKERS = 4
CHUNK = 16384
sha, write = base.sha, base.write


def ordered_hash(ids, tag):
    return sorted(ids, key=lambda x: (hashlib.sha256(f'{SPLIT_SEED}:{tag}:{x}'.encode()).digest(), x))


def seed(value=SEED):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(value)


def prepare():
    if OUT.exists(): raise FileExistsError(OUT)
    registry_path = OUT.parent / 'exposure/registry.parquet'
    registry = pq.read_table(registry_path).to_pylist()
    families = {}
    for r in registry: families.setdefault(r['family_id'], []).append(r['document_id'])
    eligible = whole_family_eligible(families, {
        r['document_id']: r['conditional_eligibility_if_training_metric_reuse_were_admissible'] for r in registry})
    count = sum(map(len, eligible.values()))
    if count < 2000: raise ValueError('Insufficient eligible EVAL2 documents')
    selected = set()
    for g in ordered_hash(eligible, 'eval2'):
        if len(selected) >= min(3000, count): break
        selected.update(eligible[g])
    rows = PreparedData().load_train('train_fitting').to_pylist()
    groups = {r['document_id']: r['family_id'] for r in registry}
    assert len(rows) == len(groups) == 55582
    assert {r['document_id'] for r in rows} == set(groups)
    assignment = [dict(document_id=r['document_id'], group_id=groups[r['document_id']],
                       role='EVAL' if r['document_id'] in selected else 'FIT', original_index=i)
                  for i, r in enumerate(rows)]
    assert not {a['group_id'] for a in assignment if a['role']=='FIT'} & {a['group_id'] for a in assignment if a['role']=='EVAL'}
    OUT.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(assignment), OUT/'assignments.parquet')
    write(OUT/'config.json', dict(seed=SEED, split_seed=SPLIT_SEED, encoder_epochs=4, head_epochs=2,
        batch_size=32, lr=.001, weight_decay=.01, clip=1., head_precision='float32', tf32=False,
        eligible_documents=count, FIT=len(rows)-len(selected), EVAL=len(selected), workers=WORKERS,
        lambda_docs=512, lambda_epsilon=1e-12, zero_rank_norm=1e-12, lambda_bounds=[.01,100],
        exposure_policy='Prior in-sample training diagnostics permitted; dedicated evaluation/calibration/probes and unknowns excluded at family level.',
        authorization='User requested autonomous improvements toward 82%; operational exposure decision documented before selection.',
        exposure_registry_sha256=sha(registry_path), assignment_sha256=sha(OUT/'assignments.parquet'),
        no_dev_holdout_test=True, pretrained=False, target_internal_exact_slot=.82))
    print(json.dumps(json.loads((OUT/'config.json').read_text()), indent=2), flush=True)


def input_rows():
    config=json.loads((OUT/'config.json').read_text())
    assert sha(OUT/'assignments.parquet')==config['assignment_sha256']
    rows=PreparedData().load_train('train_fitting').to_pylist()
    assignments=pq.read_table(OUT/'assignments.parquet').to_pylist()
    assert [a['document_id'] for a in assignments]==[r['document_id'] for r in rows]
    return rows, assignments


def runtime_worker(task):
    from global_ranking_dp import best_two
    n, energies, k = task
    t=time.monotonic(); result=best_two(n, universe(n), energies, k)
    return result, time.monotonic()-t, peak_rss()


def peak_rss():
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(x,ctypes.c_size_t) for x in (
            'PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
            'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    value=Counters(); value.cb=ctypes.sizeof(value)
    get_process=ctypes.windll.kernel32.GetCurrentProcess; get_process.restype=wintypes.HANDLE
    query=ctypes.windll.psapi.GetProcessMemoryInfo
    query.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    if not query(get_process(),ctypes.byref(value),value.cb): raise ctypes.WinError()
    return int(value.PeakWorkingSetSize)


def score_batch(head, arrays):
    h, prefix=padded_hidden(arrays)
    pairs=np.concatenate([universe(len(x)) for x in arrays])
    dd=np.repeat(np.arange(len(arrays)), [len(universe(len(x))) for x in arrays])
    values=[]
    head.eval()
    with torch.no_grad():
        for off in range(0,len(pairs),CHUNK):
            pp=pairs[off:off+CHUNK]; ds=dd[off:off+CHUNK]
            z=head(h,prefix,torch.as_tensor(ds,device='cuda'),torch.as_tensor(pp[:,0],device='cuda'),torch.as_tensor(pp[:,1],device='cuda'))
            values.append(z.cpu().numpy())
    zz=np.concatenate(values)
    if not np.isfinite(zz).all(): raise ValueError('Nonfinite scoring')
    sizes=[len(universe(len(x))) for x in arrays]
    return np.split(zz, np.cumsum(sizes)[:-1])


def benchmark():
    if (OUT/'benchmark.json').exists(): raise FileExistsError('Benchmark already frozen')
    from neural_sequence import encode_document
    rows, assignments=input_rows()
    fit={r['document_id']:r for r,a in zip(rows,assignments) if a['role']=='FIT'}
    ids=ordered_hash(fit,'runtime')[:256]
    # Timing-only synthetic encoder states at actual FIT token lengths: no old encoder or evaluation labels.
    arrays=[np.random.default_rng(i).normal(size=(len(encode_document(fit[d]['full_text'],{})[1]),256)).astype(np.float16) for i,d in enumerate(ids)]
    head=SpanHead().cuda().eval()
    score_batch(head,arrays[:2]); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    peak_ram=peak_rss()
    scoring=0.; dp_cpu=0.; t0=time.monotonic()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for start in range(0,len(ids),32):
            t=time.monotonic(); values=score_batch(head,arrays[start:start+32]); scoring+=time.monotonic()-t
            tasks=[(len(h),z[:,1:].astype(np.float64)-z[:,:1].astype(np.float64),fit[d]['expected_entity_count']) for h,z,d in zip(arrays[start:start+32],values,ids[start:start+32])]
            results=list(pool.map(runtime_worker,tasks)); dp_cpu+=sum(x[1] for x in results)
            rss=peak_rss()+WORKERS*max(x[2] for x in results)
            peak_ram=max(peak_ram,rss)
    elapsed=time.monotonic()-t0
    report=dict(documents=len(ids),document_ids=ids,synthetic_hidden=True,labels_used=False,
        miner_sha256=sha(ROOT/'src/global_ranking_dp.py'),runner_sha256=sha(Path(__file__)),
        span_scoring_seconds=scoring,dp_cpu_seconds=dp_cpu,wall_seconds=elapsed,
        projected_full_fit_epoch_seconds=elapsed/len(ids)*len(fit),
        process_tree_peak_rss_upper_estimate_bytes=peak_ram,ram_method='parent peak RSS + 4 times largest observed worker peak RSS; conservative estimate, not simultaneous peak',peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        practical_limit_projected_seconds=7200,within_runtime_budget=elapsed/len(ids)*len(fit)<=7200)
    write(OUT/'benchmark.json',report); print(json.dumps({k:v for k,v in report.items() if k!='document_ids'},indent=2),flush=True)
    if not report['within_runtime_budget']: raise RuntimeError('Exact mining exceeds declared two-hour/epoch runtime gate')


def gold_tokens(row, bounds):
    starts={int(a):i for i,(a,b) in enumerate(bounds)}
    ends={int(b):i+1 for i,(a,b) in enumerate(bounds)}
    gold=[]
    for ent in row['entities']:
        a=starts.get(ent['start']); b=ends.get(ent['end'])
        if a is None or b is None or not 0<b-a<=16: return None
        gold.append((a,b,LABELS.index(ent['label'])-1))
    gold=tuple(sorted(gold))
    if not gold or len(gold)!=row['expected_entity_count'] or any(gold[j-1][1]>gold[j][0] for j in range(1,len(gold))): return None
    return gold


def mine(head, hidden, offsets, rows, gold, ids=None):
    ids=list(range(len(rows))) if ids is None else list(ids)
    rivals={}; start_time=time.monotonic()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for begin in range(0,len(ids),32):
            batch=ids[begin:begin+32]
            hs=[hidden[offsets[i]:offsets[i+1]] for i in batch]
            zs=score_batch(head,hs)
            tasks=[(len(h),z[:,1:].astype(np.float64)-z[:,:1].astype(np.float64),rows[i]['expected_entity_count']) for h,z,i in zip(hs,zs,batch)]
            for i,(top,_,_rss) in zip(batch,pool.map(runtime_worker,tasks)):
                if gold[i] is None: rivals[i]=None; continue
                wrong=next((p for s,p in top if p!=gold[i]),None)
                if wrong is None: raise ValueError('No incorrect feasible competitor')
                rivals[i]=wrong
            if begin%2048==0: print(f'Mining {begin}/{len(ids)} elapsed {time.monotonic()-start_time:.1f}s',flush=True)
    return rivals


def ce_loss(head, h, prefix, ids, plans, epoch, digest=None):
    samples=[sample_plan(plans[i],SEED,epoch,i) for i in ids]
    pairs=np.concatenate([universe(plans[i]['n'])[s[0]] for i,s in zip(ids,samples)])
    dd=np.repeat(np.arange(len(ids)),[len(s[0]) for s in samples])
    yy=np.concatenate([s[1] for s in samples]); ww=np.concatenate([s[3] for s in samples])
    den=sum(plans[i]['population'] for i in ids)
    if digest is not None:
        for x in (np.asarray(ids,dtype=np.int64),pairs,dd,yy,ww): digest.update(x.tobytes())
    total=h.new_zeros(())
    for off in range(0,len(pairs),CHUNK):
        pp=pairs[off:off+CHUNK]
        z=head(h,prefix,torch.as_tensor(dd[off:off+CHUNK],device=h.device),torch.as_tensor(pp[:,0],device=h.device),torch.as_tensor(pp[:,1],device=h.device))
        ce=F.cross_entropy(z,torch.as_tensor(yy[off:off+CHUNK],device=h.device),reduction='none')
        total=total+(ce*torch.as_tensor(ww[off:off+CHUNK],device=h.device)).sum()/den
    return total


def rank_loss(head, h, prefix, ids, gold, rivals):
    supported=[(j,i) for j,i in enumerate(ids) if gold[i] is not None]
    if not supported: return h.new_zeros(())
    spans=[]; ds=[]; targets=[]; signs=[]; groups=[]; counts=[]
    for q,(j,i) in enumerate(supported):
        counts.append(len(gold[i]))
        for structure,sign in ((gold[i],-1.),(rivals[i],1.)):
            assert len(structure)==len(gold[i])
            for a,b,label in structure:
                spans.append((a,b)); ds.append(j); targets.append(label+1); signs.append(sign); groups.append(q)
    pp=torch.as_tensor(spans,device=h.device)
    z=head(h,prefix,torch.as_tensor(ds,device=h.device),pp[:,0],pp[:,1])
    energies=z.gather(1,torch.as_tensor(targets,device=h.device)[:,None]).squeeze(1)-z[:,0]
    delta=z.new_zeros(len(supported)).index_add(0,torch.as_tensor(groups,device=h.device),energies*torch.as_tensor(signs,device=h.device))
    return F.softplus(delta/torch.as_tensor(counts,device=h.device)).mean()


def grad_norm(head):
    return float(torch.sqrt(sum((p.grad.detach().double().square().sum() for p in head.parameters() if p.grad is not None))))


def measurement(head, hidden, offsets, plans, gold, rivals, ids, epoch):
    result=[]; head.eval()
    for start in range(0,len(ids),32):
        batch=ids[start:start+32]; h,pr=padded_hidden([hidden[offsets[i]:offsets[i+1]] for i in batch])
        head.zero_grad(set_to_none=True); ce=ce_loss(head,h,pr,batch,plans,epoch); ce.backward(); cn=grad_norm(head)
        head.zero_grad(set_to_none=True); rank=rank_loss(head,h,pr,batch,gold,rivals); rank.backward(); rn=grad_norm(head)
        if not all(np.isfinite(x) for x in (float(ce),float(rank),cn,rn)): raise ValueError('Invalid gradient measurement')
        result.append(dict(ce=float(ce.detach()),rank=float(rank.detach()),ce_norm=cn,rank_norm=rn))
    head.zero_grad(set_to_none=True)
    return result


def train():
    if (OUT/'training_complete.json').exists(): raise FileExistsError('Training already complete')
    if not json.loads((OUT/'benchmark.json').read_text())['within_runtime_budget']: raise RuntimeError('Benchmark gate failed')
    sources=['src/train_global_ranking.py','src/global_ranking_dp.py','src/span_probe.py','src/neural_sequence.py',
             'src/train_scale_ranking.py','note/bitrase-2/algo-4/protocol.md','output/prepared/v1/train.parquet']
    inputs={p:sha(ROOT/p) for p in sources}
    write(OUT/'training_inputs.json',inputs)
    rows, assignments=input_rows(); groups={a['document_id']:a['group_id'] for a in assignments}
    fit=[r for r,a in zip(rows,assignments) if a['role']=='FIT']
    cache=OUT/'cache'; cache.mkdir(exist_ok=True); base.OUT=cache
    if (cache/'fit_ready.json').exists():
        ready=json.loads((cache/'fit_ready.json').read_text())
        assert ready['input_hashes']==inputs
        hidden=np.load(cache/'FIT/hidden.npy',mmap_mode='r'); bounds=np.load(cache/'FIT/boundaries.npy',mmap_mode='r'); offsets=np.load(cache/'FIT/token_offsets.npy')
        old={int(i):p for i,p in json.loads((cache/'old_predictions.json').read_text()).items()}
    else:
        vocab=base.build_vocab(r['full_text'] for r in fit); write(OUT/'vocab.json',vocab)
        encoded,bad=base.encode(fit,vocab,True); write(OUT/'fit_alignment.json',dict(unaligned=bad))
        model=base.train_encoder(encoded,vocab)
        hidden,bounds,offsets,old=base.hidden_cache(model,fit,encoded,'FIT',groups)
        write(cache/'old_predictions.json',old)
        write(cache/'fit_ready.json',dict(input_hashes=inputs,encoder_sha256=sha(cache/'encoder_epoch4.pt')))
        del encoded,model
    plans=[]; gold=[]
    for i,r in enumerate(fit):
        bb=bounds[offsets[i]:offsets[i+1]]; plan=sampling_plan(bb,r['entities'],old[i]); plan['n']=len(bb)
        plans.append(plan); gold.append(gold_tokens(r,bb))
    eligible={r['document_id']:i for i,r in enumerate(fit) if gold[i] is not None and len(gold[i])>0}
    measure_ids=[eligible[d] for d in ordered_hash(eligible,'lambda')[:512]]
    assert len(measure_ids)==512
    seed(); head=SpanHead().cuda(); torch.save(head.state_dict(),OUT/'initial_head.pt')
    initial_rivals=mine(head,hidden,offsets,fit,gold,measure_ids)
    initial=measurement(head,hidden,offsets,plans,gold,initial_rivals,measure_ids,1)
    cn=float(np.median([r['ce_norm'] for r in initial])); rn=float(np.median([r['rank_norm'] for r in initial]))
    lam=cn/(rn+1e-12)
    write(OUT/'lambda.json',dict(value=lam,ce_median=cn,rank_median=rn,measurements=initial,
        document_ids=[fit[i]['document_id'] for i in measure_ids],supported_fit_documents=len(eligible)))
    if not np.isfinite(lam) or rn<=1e-12 or not .01<=lam<=100: raise RuntimeError('Lambda predeclared gate failed')
    print(f'Frozen lambda={lam:.8f}; supported FIT={len(eligible)}',flush=True)
    del head
    logs={}; expected_digests=[]; epoch_orders={}
    for epoch in (1,2):
        order=list(range(len(fit))); random.Random(SEED+epoch).shuffle(order); epoch_orders[epoch]=order
    write(OUT/'ce_orders.json',epoch_orders)
    for arm in ('control','challenger'):
        head=SpanHead().cuda(); head.load_state_dict(torch.load(OUT/'initial_head.pt',weights_only=True)); opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
        logs[arm]=[]
        for epoch in (1,2):
            rivals=None
            if arm=='challenger':
                rivals=mine(head,hidden,offsets,fit,gold)
                write(OUT/f'competitors_epoch{epoch}.json',rivals)
                before=measurement(head,hidden,offsets,plans,gold,rivals,measure_ids,1)
            digest=hashlib.sha256(); total=0.; pop=0; ranks=0.; batches=0; t=time.monotonic()
            for begin in range(0,len(fit),32):
                ids=epoch_orders[epoch][begin:begin+32]; h,pr=padded_hidden([hidden[offsets[i]:offsets[i+1]] for i in ids])
                # Reset the complete CE RNG stream identically in both arms.
                draw_seed=SEED+epoch*100000+begin//32
                torch.manual_seed(draw_seed); torch.cuda.manual_seed_all(draw_seed)
                digest.update(np.asarray([draw_seed],dtype=np.int64).tobytes())
                head.train(); opt.zero_grad(set_to_none=True)
                ce=ce_loss(head,h,pr,ids,plans,epoch,digest); ce.backward()
                rr=0.
                if rivals is not None:
                    head.eval(); rank=rank_loss(head,h,pr,ids,gold,rivals)
                    if rank.requires_grad: (lam*rank).backward()
                    rr=float(rank.detach())
                if not np.isfinite(float(ce.detach())): raise ValueError('Nonfinite CE')
                norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True); opt.step()
                den=sum(plans[i]['population'] for i in ids); total+=float(ce.detach())*den; pop+=den; ranks+=rr; batches+=1
                if begin%6400==0: print(f'{arm} epoch{epoch} {begin}/{len(fit)} CE={total/pop:.6f} rank={ranks/batches:.4f}',flush=True)
            record=dict(epoch=epoch,ce=total/pop,rank_per_batch=ranks/batches,seconds=time.monotonic()-t,ce_exposure_sha256=digest.hexdigest())
            if arm=='control': expected_digests.append(digest.hexdigest())
            else:
                assert digest.hexdigest()==expected_digests[epoch-1], 'CE exposure mismatch'
                record['gradient_monitor_before']=before
                record['gradient_monitor_after']=measurement(head,hidden,offsets,plans,gold,rivals,measure_ids,1)
            logs[arm].append(record); write(OUT/'head_training.json',logs)
            torch.save(head.state_dict(),OUT/f'{arm}_epoch{epoch}.pt')
        del head
    assert inputs=={p:sha(ROOT/p) for p in sources}
    write(OUT/'training_complete.json',dict(complete=True,inputs_unchanged=True,ce_exposure_matched=True,
        encoder_sha256=sha(cache/'encoder_epoch4.pt'),control_sha256=sha(OUT/'control_epoch2.pt'),challenger_sha256=sha(OUT/'challenger_epoch2.pt')))


def score_eval():
    assert (OUT/'training_complete.json').exists()
    if (OUT/'scores_complete.json').exists(): raise FileExistsError('Scores already frozen')
    rows, assignments=input_rows(); groups={a['document_id']:a['group_id'] for a in assignments}
    subset=[r for r,a in zip(rows,assignments) if a['role']=='EVAL']
    vocab=json.loads((OUT/'vocab.json').read_text()); enc,_=base.encode(subset,vocab,False)
    model=base.build_model(len(vocab)).cuda(); model.load_state_dict(torch.load(OUT/'cache/encoder_epoch4.pt',weights_only=True)); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    base.OUT=OUT/'cache'
    hh,bb,oo,_=base.hidden_cache(model,subset,enc,'EVAL',groups)
    del model,enc
    counts=[len(universe(int(oo[i+1]-oo[i]))) for i in range(len(subset))]
    so=np.r_[0,np.cumsum(counts)].astype(np.int64)
    for arm in ('control','challenger'):
        path=OUT/'EVAL'/arm; path.mkdir(parents=True)
        for name in ('metadata.json','token_offsets.npy','boundaries.npy'): shutil.copyfile(OUT/'cache/EVAL'/name,path/name)
        np.save(path/'span_offsets.npy',so)
        pairs=np.lib.format.open_memmap(path/'pairs.npy',mode='w+',dtype=np.int32,shape=(int(so[-1]),2))
        logits=np.lib.format.open_memmap(path/'logits.npy',mode='w+',dtype=np.float32,shape=(int(so[-1]),8))
        head=SpanHead().cuda(); head.load_state_dict(torch.load(OUT/f'{arm}_epoch2.pt',weights_only=True)); head.eval()
        for start in range(0,len(subset),32):
            ids=list(range(start,min(start+32,len(subset)))); arrays=[hh[oo[i]:oo[i+1]] for i in ids]
            for i,z in zip(ids,score_batch(head,arrays)):
                a,b=so[i:i+2]; logits[a:b]=z; pairs[a:b]=universe(int(oo[i+1]-oo[i]))
        logits.flush(); pairs.flush()
        write(path/'manifest.json',dict(complete=True,documents=len(subset),files={p.name:sha(p) for p in path.iterdir() if p.is_file()}))
        print(f'{arm} EVAL scores frozen',flush=True)
    write(OUT/'scores_complete.json',dict(complete=True,documents=len(subset)))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['prepare','benchmark','train','score']); args=parser.parse_args()
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    seed()
    {'prepare':prepare,'benchmark':benchmark,'train':train,'score':score_eval}[args.stage]()


if __name__=='__main__': main()
