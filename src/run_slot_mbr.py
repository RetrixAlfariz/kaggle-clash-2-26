"""P1 frozen-score experiment. Dev only; no fitting or submission path."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/bitrase-2/algo-1'
FROZEN = OUT / 'frozen'
P0 = ROOT / 'output/bitrase-2/p0/seed2026'
LABELS = ('NAME','DATE','EMAIL','PHONE','ADDRESS','USERNAME','JOB_TITLE')
BOOTSTRAP_SEED = 20260915


def peak_working_set():
    """Windows process peak resident memory, including mmap pages and imports."""
    if os.name!='nt':return None
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD),
                  *[(name,ctypes.c_size_t) for name in ('PeakWorkingSetSize','WorkingSetSize',
                    'QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage',
                    'QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    psapi=ctypes.WinDLL('psapi',use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    info=Counters();info.cb=ctypes.sizeof(info)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(info),info.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(info.PeakWorkingSetSize)


def verify():
    OUT.mkdir(parents=True,exist_ok=True)
    files=['tests/test_slot_mbr.py','tests/test_slot_diagnostics.py','tests/test_slot_mbr_runner.py']
    hashes={**code_hashes(),**{name:sha(ROOT/name) for name in files}}
    result=subprocess.run([sys.executable,'-B','-m','unittest',
                           'tests.test_slot_mbr','tests.test_slot_diagnostics','tests.test_slot_mbr_runner','-v'],
                          cwd=ROOT,capture_output=True,text=True)
    (OUT/'verification.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    if result.returncode or hashes!={**code_hashes(),**{name:sha(ROOT/name) for name in files}}:
        raise ValueError('Verification failed or sources changed; see verification.log')
    write(OUT/'verification.json',{'passed':True,'files':hashes,'log_sha256':sha(OUT/'verification.log'),
                                 'command':result.args,'returncode':result.returncode})
    print(result.stdout+result.stderr,flush=True)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def code_hashes():
    names = ['src/run_slot_mbr.py','src/slot_mbr.py','src/slot_diagnostics.py',
             'src/span_probe.py','src/ner_evaluation.py',
             'note/bitrase-2/algo-1/protocol.md']
    return {name: sha(ROOT/name) for name in names}


def verify_evidence():
    v = read(OUT/'verification.json')
    if not v['passed']:
        raise ValueError('Exhaustive verification has not passed')
    for name, digest in v['files'].items():
        if sha(ROOT/name) != digest:
            raise ValueError(f'Verified source changed: {name}')
    return v


def freeze():
    import torch
    from run_span_probe import Cache
    from span_probe import SpanHead, universe, padded_hidden
    verify_evidence()
    if FROZEN.exists():
        raise FileExistsError('Frozen artifact already exists; never overwrite')
    started = time.monotonic()
    sources = code_hashes()
    cache = Cache('dev')  # verifies P0 cache and original source provenance
    report = read(P0/'report.json')
    if report['selected_epoch'] != 2 or sha(P0/'epoch2.pt') != report['selected_model_sha256']:
        raise ValueError('P0 selected checkpoint mismatch')
    saved = P0/'epoch2_dev_predictions.parquet'
    if sha(saved) != report['epochs']['2']['prediction_sha256']:
        raise ValueError('P0 predictions changed')
    torch.set_num_threads(4)
    head = SpanHead().cuda()
    head.load_state_dict(torch.load(P0/'epoch2.pt', map_location='cuda', weights_only=True))
    head.eval()
    FROZEN.mkdir(parents=True)
    offsets = np.zeros(len(cache.meta)+1, dtype=np.int64)
    for i,m in enumerate(cache.meta):
        offsets[i+1] = offsets[i] + len(universe(m['token_count']))
    logits = np.lib.format.open_memmap(FROZEN/'logits.npy', mode='w+', dtype=np.float32, shape=(int(offsets[-1]),8))
    pairsfile = np.lib.format.open_memmap(FROZEN/'pairs.npy', mode='w+', dtype=np.int32, shape=(int(offsets[-1]),2))
    order = sorted(range(len(cache.meta)), key=lambda i: cache.meta[i]['token_count'])
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for begin in range(0,len(order),32):
            ids = order[begin:begin+32]
            h,prefix = padded_hidden([cache.h(i) for i in ids])
            pairs = [universe(len(cache.h(i))) for i in ids]
            counts = [len(p) for p in pairs]
            pp = np.concatenate(pairs)
            dd = np.repeat(np.arange(len(ids)),counts)
            chunks = []
            for start in range(0,len(pp),32768):
                p = pp[start:start+32768]; d = dd[start:start+32768]
                with torch.autocast('cuda',dtype=torch.float16):
                    z = head(h,prefix,torch.as_tensor(d,device='cuda'),
                             torch.as_tensor(p[:,0],device='cuda'),torch.as_tensor(p[:,1],device='cuda'))
                chunks.append(z.float().cpu().numpy())
            z = np.concatenate(chunks)
            if not np.isfinite(z).all():
                raise ValueError('Nonfinite frozen logits')
            cursor = 0
            for i,p,count in zip(ids,pairs,counts):
                lo,hi = offsets[i:i+2]
                logits[lo:hi] = z[cursor:cursor+count]
                pairsfile[lo:hi] = p
                cursor += count
            if begin % 1600 == 0:
                print(f'Frozen logits {min(begin+32,len(order))}/{len(order)}',flush=True)
    logits.flush(); pairsfile.flush()
    np.save(FROZEN/'span_offsets.npy',offsets)
    for source,target in [('offsets.npy','token_offsets.npy'),('boundaries.npy','boundaries.npy'),('metadata.json','metadata.json')]:
        shutil.copyfile(cache.path/source,FROZEN/target)
    files = {p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in FROZEN.iterdir()}
    if code_hashes() != sources:
        raise ValueError('Sources changed during freezing')
    manifest = {'complete':True,'sources':sources,'files':files,'documents':len(cache.meta),
                'candidates':int(offsets[-1]),'labels':['NONE',*LABELS],
                'head_sha256':sha(P0/'epoch2.pt'),'p0_predictions_sha256':sha(saved),
                'cache_manifest_sha256':sha(cache.path/'manifest.json'),
                'dev_sha256':sha(ROOT/'output/prepared/v1/dev.parquet'),
                'bootstrap_seed':BOOTSTRAP_SEED,'primary_tau':1.,'diagnostic_taus':[.5,2.],
                'batch_documents':32,'chunk_spans':32768,'raw_dtype':'float32 (native FP16 logits losslessly promoted)',
                'energy_arithmetic':'subtract raw logits in float64; MAP replay uses original P0 float32 subtraction',
                'seconds':time.monotonic()-started,'torch':torch.__version__,'numpy':np.__version__,
                'gpu':torch.cuda.get_device_name(),'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
                'process_peak_working_set_bytes':peak_working_set(),
                'max_tokens':max(m['token_count'] for m in cache.meta),'max_K':max(m['expectedK'] for m in cache.meta)}
    write(FROZEN/'manifest.json',manifest)
    # OS read-only flags supplement hash validation; every consumer verifies hashes.
    for p in FROZEN.iterdir():
        p.chmod(0o444)
    print(json.dumps({k:manifest[k] for k in ('documents','candidates','seconds','max_tokens','max_K')}),flush=True)


def validate_frozen():
    verify_evidence()
    m = read(FROZEN/'manifest.json')
    if not m['complete'] or m['sources'] != code_hashes():
        raise ValueError('Frozen source provenance mismatch')
    for name, info in m['files'].items():
        if sha(FROZEN/name) != info['sha256']:
            raise ValueError(f'Frozen file mismatch: {name}')
    if sha(P0/'epoch2_dev_predictions.parquet') != m['p0_predictions_sha256']:
        raise ValueError('P0 prediction hash mismatch')
    return m


def load_arrays():
    return {k:np.load(FROZEN/f'{k}.npy',mmap_mode='r') for k in
            ('logits','pairs','span_offsets','token_offsets','boundaries')}


def pred_map(records):
    result = {}
    for row in records:
        doc = row['document_id']
        if doc in result:
            raise ValueError('Duplicate prediction document')
        p = [(e['start'],e['end'],e['label']) for e in row['predicted']]
        if p != sorted(p) or len(p) != len(set(p)):
            raise ValueError('Unsorted/duplicate prediction entities')
        if any(a<0 or b<=a or l not in LABELS for a,b,l in p) or any(x[1]>y[0] for x,y in zip(p,p[1:])):
            raise ValueError('Invalid or overlapping prediction spans')
        result[doc] = p
    return result


def canonical_hash(records):
    # Only document order is normalized. Entity order must already be valid.
    mapping = pred_map(records)
    payload = [(doc,mapping[doc]) for doc in sorted(mapping)]
    return hashlib.sha256(json.dumps(payload,ensure_ascii=False,separators=(',',':')).encode('utf-8')).hexdigest()


def record(doc,p):
    return {'document_id':doc,'predicted':[{'start':int(a),'end':int(b),'label':l} for a,b,l in p]}


def replay():
    from span_probe import universe,decode_spans,character_spans
    manifest = validate_frozen(); arrays = load_arrays(); meta = read(FROZEN/'metadata.json')
    target = OUT/'map_predictions.parquet'
    if target.exists():
        raise FileExistsError(target)
    expected_records = pq.read_table(P0/'epoch2_dev_predictions.parquet').to_pylist()
    expected = pred_map(expected_records)
    if len(meta)!=len({m['document_id'] for m in meta}) or set(expected)!={m['document_id'] for m in meta}:
        raise ValueError('MAP document coverage mismatch')
    started=time.monotonic(); output=[]
    for i,m in enumerate(meta):
        lo,hi=arrays['span_offsets'][i:i+2];a,b=arrays['token_offsets'][i:i+2]
        z=arrays['logits'][lo:hi]; pp=arrays['pairs'][lo:hi]
        if not np.array_equal(pp,universe(m['token_count'])):
            raise ValueError('Candidate universe changed')
        labels=z[:,1:].argmax(1)+1
        values=z[np.arange(len(z)),labels]-z[:,0]
        pred=character_spans(decode_spans(m['token_count'],values,labels,m['expectedK']),arrays['boundaries'][a:b])
        if len(pred)!=m['expectedK'] or pred!=expected[m['document_id']]:
            raise ValueError(f'MAP replay mismatch: {m["document_id"]}; STOP before MBR')
        output.append(record(m['document_id'],pred))
    actual_hash=canonical_hash(output);expected_hash=canonical_hash(expected_records)
    if actual_hash!=expected_hash:
        raise ValueError('Canonical MAP mismatch')
    pq.write_table(pa.Table.from_pylist(output),target,compression='zstd')
    result={'passed':True,'documents':len(output),'canonical_actual_sha256':actual_hash,
            'canonical_expected_sha256':expected_hash,'prediction_sha256':sha(target),
            'frozen_manifest_sha256':sha(FROZEN/'manifest.json'),'seconds':time.monotonic()-started,
            'comparison':'all documents; exact counts, ordered triples, coverage; normalized document serialization order only'}
    write(OUT/'map_replay.json',result)
    print(json.dumps(result),flush=True)


_ARRAYS=None
_META=None


def worker_init():
    global _ARRAYS,_META
    _ARRAYS=load_arrays();_META=read(FROZEN/'metadata.json')


def worker(task):
    from slot_mbr import decode
    i,tau=task;m=_META[i];v=_ARRAYS
    lo,hi=v['span_offsets'][i:i+2];a,b=v['token_offsets'][i:i+2]
    z=np.asarray(v['logits'][lo:hi],dtype=np.float64)
    energy=(z[:,1:]-z[:,:1])/tau
    started=time.monotonic()
    pred,diagnostics=decode(m['token_count'],v['pairs'][lo:hi],energy,m['expectedK'])
    diagnostics['process_peak_working_set_bytes']=peak_working_set()
    diagnostics['document_id']=m['document_id']
    bounds=v['boundaries'][a:b]
    out=[(int(bounds[x,0]),int(bounds[y-1,1]),LABELS[label]) for x,y,label in pred]
    return record(m['document_id'],out),diagnostics,time.monotonic()-started


def evaluate_records(records,rows):
    from ner_evaluation import Evaluation,triples
    from run_neural_sequence import clean_report
    mapping=pred_map(records)
    if set(mapping)!={r['document_id'] for r in rows}:
        raise ValueError('Evaluation coverage mismatch')
    ev=Evaluation();documents=[]
    for r in rows:
        gold=triples(r['entities']);p=mapping[r['document_id']]
        if len(p)!=r['expected_entity_count'] or len(gold)!=len(p):
            raise ValueError('Exact-K mismatch')
        ev.add(r['full_text'],gold,p,p)
        documents.append({'document_id':r['document_id'],'K':len(gold),'correct':sum(g==q for g,q in zip(gold,p)),
                          'entity_correct':len(set(gold)&set(p))})
    return clean_report(ev.report()),documents


def paired_bootstrap(current,baseline):
    if [r['document_id'] for r in current]!=[r['document_id'] for r in baseline]:
        raise ValueError('Bootstrap pairing mismatch')
    k=np.array([r['K'] for r in current]);delta=np.array([c['correct']-b['correct'] for c,b in zip(current,baseline)])
    rng=np.random.default_rng(BOOTSTRAP_SEED);draws=[]
    for begin in range(0,10000,128):
        idx=rng.integers(0,len(k),size=(min(128,10000-begin),len(k)))
        draws.extend((delta[idx].sum(1)/k[idx].sum(1)).tolist())
    return {'observed_delta':float(delta.sum()/k.sum()),'median':float(np.median(draws)),
            'ci95':np.quantile(draws,[.025,.975]).tolist(),'seed':BOOTSTRAP_SEED,'replicates':10000,
            'unit':'paired document; ratio of pooled correct counts to pooled slots','selection_bias_corrected':False}


def run(tau,workers):
    from prepared_loader import PreparedData
    manifest=validate_frozen();replayed=read(OUT/'map_replay.json')
    if not replayed['passed'] or sha(OUT/'map_predictions.parquet')!=replayed['prediction_sha256'] or sha(FROZEN/'manifest.json')!=replayed['frozen_manifest_sha256']:
        raise ValueError('MAP replay gate not satisfied')
    if tau!=1 and not (OUT/'tau1'/'mechanism.json').exists():
        raise ValueError('Primary performance and mechanism must precede sensitivity')
    target=OUT/f'tau{tau:g}'
    if target.exists():raise FileExistsError(target)
    target.mkdir()
    started=time.monotonic();records=[];diags=[];times=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=worker_init) as pool:
        for p,d,t in pool.map(worker,((i,tau) for i in range(manifest['documents'])),chunksize=16):
            records.append(p);diags.append(d);times.append(t)
            if len(records)%1000==0:print(f'tau={tau:g}: decoded {len(records)}/{manifest["documents"]}',flush=True)
    decode_seconds=time.monotonic()-started
    file=target/'predictions.parquet';pq.write_table(pa.Table.from_pylist(records),file,compression='zstd')
    write(target/'decoder_diagnostics.json',diags)
    rows=PreparedData().load_dev('evaluation').to_pylist()
    baseline_records=pq.read_table(OUT/'map_predictions.parquet').to_pylist()
    baseline,base_documents=evaluate_records(baseline_records,rows)
    metrics,documents=evaluate_records(records,rows)
    boot=paired_bootstrap(documents,base_documents)
    pq.write_table(pa.Table.from_pylist(documents),target/'documents.parquet',compression='zstd')
    result={'tau':tau,'primary':tau==1,'metrics':metrics,'baseline_metrics':baseline,'bootstrap':boot,
            'research_positive':tau==1 and boot['observed_delta']>=.002 and boot['ci95'][0]>0,
            'sensitivity_eligible_for_selection':False,'prediction_sha256':sha(file),
            'frozen_manifest_sha256':sha(FROZEN/'manifest.json'),'code_hashes':code_hashes(),
            'decode_wall_seconds':decode_seconds,'document_decode_seconds_quantiles':np.quantile(times,[0,.5,.95,1]).tolist(),
            'workers':workers,'seconds':time.monotonic()-started,'holdout_accessed':False,'uploaded':False}
    result['parent_peak_working_set_bytes']=peak_working_set()
    write(target/'performance.json',result)
    # Gold is used only after predictions and performance have been computed.
    mechanism(rows,baseline_records,records,target)
    if code_hashes()!=manifest['sources']:raise ValueError('Source changed during comparison')
    print(json.dumps({'tau':tau,'slot_accuracy':metrics['slot_accuracy'],'entity_f1':metrics['entity_micro']['f1'],
                      'bootstrap':boot,'research_positive':result['research_positive'],'seconds':result['seconds']}),flush=True)


def mechanism(rows,baseline_records,current_records,target):
    from collections import Counter
    from ner_evaluation import triples
    from slot_diagnostics import compare
    base=pred_map(baseline_records);current=pred_map(current_records)
    details=[]
    for r in rows:
        details.append({'document_id':r['document_id'],**compare(triples(r['entities']),base[r['document_id']],current[r['document_id']])})
    write(target/'mechanism_documents.json',details)
    # Preserve exact per-document diagnostics; aggregate scalar counters by path.
    totals={}
    def add(obj,prefix=''):
        for key,value in obj.items():
            path=f'{prefix}.{key}' if prefix else key
            if isinstance(value,dict):add(value,path)
            elif isinstance(value,(int,float)) and not isinstance(value,bool):totals[path]=totals.get(path,0)+value
            elif isinstance(value,bool):totals[path]=totals.get(path,0)+int(value)
    for d in details:add({k:v for k,v in d.items() if k!='document_id'})
    lengths={model:dict(sorted(Counter(length for d in details for length in d['model_stats'][model]['episode_lengths']).items())) for model in ('map','mbr')}
    write(target/'mechanism.json',{'documents':len(details),'summed_scalar_diagnostics':totals,
                                 'episode_length_histograms':lengths,
                                 'new_episode_definition':'Different reference entity set or displacement sign from all MAP simple episodes; includes modified episodes, not necessarily new harm',
                                 'details_sha256':sha(target/'mechanism_documents.json')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['verify','freeze','replay','run'])
    p.add_argument('--tau',type=float,choices=[.5,1.,2.],default=1.)
    p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    if args.stage=='verify':verify()
    elif args.stage=='freeze':freeze()
    elif args.stage=='replay':replay()
    else:run(args.tau,args.workers)
