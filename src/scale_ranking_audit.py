"""Train-only scalar-energy and candidate-ranking diagnostic.

The module consumes frozen span-logit caches. It never imports the prepared
dataset loader or training code, so FIT/CAL/EVAL boundaries are supplied by
the caller as separate cache directories.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

LABELS = ('NAME', 'DATE', 'EMAIL', 'PHONE', 'ADDRESS', 'USERNAME', 'JOB_TITLE')
BETA_BOUNDS = (0.25, 4.0)
FIT_ITERATIONS = 18
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260917


def _lse(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + math.log(float(np.exp(x - m).sum()))


def partition_stats(n: int, pairs: np.ndarray, energies: np.ndarray,
                    k: int, beta: float) -> tuple[float, float]:
    """Return exact-K log partition and posterior expected *raw* total energy.

    The second quantity is computed by an expectation semiring alongside the
    log-space DP; it is the derivative of logZ(beta*energies) w.r.t. beta.
    """
    p = np.asarray(pairs, dtype=np.int64)
    e = np.asarray(energies, dtype=np.float64) * float(beta)
    if p.ndim != 2 or p.shape[1] != 2 or e.ndim != 2 or e.shape[0] != len(p):
        raise ValueError('expected pairs Mx2 and energies MxL')
    if not np.isfinite(e).all() or beta <= 0 or k < 0 or k > n:
        raise ValueError('invalid finite scores, beta, or K')
    if len(p) and (np.any(p[:, 0] < 0) or np.any(p[:, 1] > n) or np.any(p[:, 0] >= p[:, 1])):
        raise ValueError('invalid interval bounds')
    # Collapse labels per interval while preserving their conditional expected
    # raw energy, then combine intervals using an expectation semiring.
    raw = np.asarray(energies, dtype=np.float64)
    if len(p):
        vmax=np.max(e,axis=1); interval_z=vmax+np.log(np.exp(e-vmax[:,None]).sum(axis=1))
        probs=np.exp(e-interval_z[:,None]);interval_mean=np.sum(probs*raw,axis=1)
    else:
        interval_z=np.empty(0);interval_mean=np.empty(0)
    by_end: list[list[int]] = [[] for _ in range(n + 1)]
    for i, (_a, b) in enumerate(p): by_end[int(b)].append(i)
    z = np.full((n + 1, k + 1), -np.inf); z[:, 0] = 0.0
    ex = np.zeros((n + 1, k + 1), dtype=np.float64)
    for b in range(1, n + 1):
        z[b] = z[b - 1]; ex[b] = ex[b - 1]
        ids=np.asarray(by_end[b],dtype=np.int64)
        if not len(ids) or not k: continue
        starts=p[ids,0]
        trans=z[starts,:-1]+interval_z[ids,None]
        trans_ex=ex[starts,:-1]+interval_mean[ids,None]
        old=z[b,1:].copy(); all_terms=np.concatenate((old[None,:],trans),axis=0)
        merged=np.logaddexp.reduce(all_terms,axis=0)
        valid=np.isfinite(merged)
        old_w=np.zeros_like(old);take_w=np.zeros_like(trans)
        old_w[valid]=np.exp(old[valid]-merged[valid])
        take_w[:,valid]=np.exp(trans[:,valid]-merged[None,valid])
        ex[b,1:]=old_w*ex[b,1:]+np.sum(take_w*trans_ex,axis=0)
        z[b,1:]=merged
    if not math.isfinite(float(z[n, k])): raise ValueError('no feasible exact-K structure')
    return float(z[n, k]), float(ex[n, k])


def fit_beta(calibration: list[dict[str, Any]], iterations: int = FIT_ITERATIONS,
             progress: Any = None) -> dict[str, Any]:
    """Fit one inverse temperature by CAL pooled per-slot structured NLL."""
    docs = [d for d in calibration if d['supported']]
    if not docs: raise ValueError('CAL has no fully representable gold structures')
    def derivative(beta: float) -> float:
        total = 0.0
        for d in docs:
            _z, mean = partition_stats(d['n'], d['pairs'], d['energies'], d['k'], beta)
            total += mean - d['gold_energy']
        return total
    lo, hi = BETA_BOUNDS
    dlo, dhi = derivative(lo), derivative(hi)
    if dlo >= 0: beta, bound = lo, 'lower'
    elif dhi <= 0: beta, bound = hi, 'upper'
    else:
        for it in range(iterations):
            mid = (lo + hi) / 2
            deriv=derivative(mid)
            if deriv < 0: lo = mid
            else: hi = mid
            if progress: progress(it+1,mid,deriv,lo,hi)
        beta, bound = (lo + hi) / 2, None
    denom = sum(int(d['k']) for d in docs)
    nll1 = sum(partition_stats(d['n'], d['pairs'], d['energies'], d['k'], 1.0)[0] - d['gold_energy'] for d in docs) / denom
    nllfit = sum(partition_stats(d['n'], d['pairs'], d['energies'], d['k'], beta)[0] - beta*d['gold_energy'] for d in docs) / denom
    return {'beta': float(beta), 'bounds': list(BETA_BOUNDS), 'iterations': int(iterations),
            'boundary_solution': bound, 'cal_docs_total': len(calibration), 'cal_docs_supported': len(docs),
            'cal_slots_supported': denom, 'cal_nll_per_slot_beta1': float(nll1),
            'cal_nll_per_slot_fitted': float(nllfit), 'cal_nll_improvement_fraction': float((nll1-nllfit)/nll1) if nll1 else 0.0}


def _best_two(n: int, pairs: np.ndarray, energies: np.ndarray, k: int) -> list[tuple[float, tuple[tuple[int,int,int], ...]]]:
    """Exact top two distinct labeled structures; deterministic tie ordering."""
    p=np.asarray(pairs,dtype=np.int64); e=np.asarray(energies,dtype=np.float64)
    top_labels=np.argsort(-e,axis=1,kind='stable')[:,:min(2,e.shape[1])]
    by_end=[[] for _ in range(n+1)]
    for i,(_a,b) in enumerate(p): by_end[int(b)].append(i)
    # Each DP cell keeps two unique structural paths.
    dp=[[[] for _ in range(k+1)] for _ in range(n+1)]
    for b in range(n+1): dp[b][0]=[(0.0,())]
    for b in range(1,n+1):
        for q in range(1,k+1):
            cand=list(dp[b-1][q])
            for i in by_end[b]:
                a=int(p[i,0])
                for lab in top_labels[i]:
                    for score,path in dp[a][q-1]:
                        span=(int(p[i,0]),int(p[i,1]),int(lab))
                        cand.append((score+float(e[i,lab]),path+(span,)))
            # Sort by energy then stable lexicographic path, deduplicate.
            cand.sort(key=lambda x:(-x[0],x[1]))
            seen=set(); dp[b][q]=[]
            for item in cand:
                if item[1] not in seen:
                    seen.add(item[1]);dp[b][q].append(item)
                    if len(dp[b][q])==2:break
    return dp[n][k]


def ranking_record(n: int, pairs: np.ndarray, energies: np.ndarray, k: int,
                   gold: tuple[tuple[int,int,int], ...]) -> dict[str, Any]:
    top=_best_two(n,pairs,energies,k)
    gold_score=None
    index={tuple(map(int,p)):i for i,p in enumerate(np.asarray(pairs))}
    if len(gold)==k and all((a,b) in index for a,b,_l in gold):
        gold_score=float(sum(energies[index[a,b],l] for a,b,l in gold))
    if gold_score is None:
        return {'gold_supported':False,'gold_energy':None,'map_energy':top[0][0],
                'map_is_gold':False,'strongest_wrong_energy':top[0][0], 'gold_minus_wrong_gap':None,
                'second_energy':top[1][0] if len(top)>1 else None}
    map_is_gold=(top[0][1]==gold)
    rival=top[1] if map_is_gold and len(top)>1 else top[0]
    return {'gold_supported':True,'gold_energy':gold_score,'map_energy':top[0][0],
            'map_is_gold':map_is_gold,'strongest_wrong_energy':rival[0],
            'gold_minus_wrong_gap':float(gold_score-rival[0]) if len(top)>1 or not map_is_gold else None,
            'second_energy':top[1][0] if len(top)>1 else None,'map_structure':top[0][1]}


def local_span_ranking(pairs: np.ndarray, energies: np.ndarray,
                       gold: tuple[tuple[int,int,int], ...]) -> list[dict[str, Any]]:
    """Gold-label span ordering against feasible one-slot substitutions.

    Other gold slots remain fixed; alternatives must lie between neighboring
    gold intervals. Candidate labels are held to the gold label.
    """
    p=np.asarray(pairs,dtype=np.int64); lookup={tuple(map(int,x)):i for i,x in enumerate(p)}
    out=[]
    for j,(a,b,l) in enumerate(gold):
        left=gold[j-1][1] if j else 0; right=gold[j+1][0] if j+1<len(gold) else math.inf
        gi=lookup.get((a,b))
        if gi is None: out.append({'slot':j,'label':LABELS[l],'supported':False,'competitors':0});continue
        alternatives=[(i,ll) for i,(x,y) in enumerate(p) if int(x)>=left and int(y)<=right
                      for ll in range(energies.shape[1]) if (int(x),int(y),ll)!=(a,b,l)]
        if not alternatives: out.append({'slot':j,'label':LABELS[l],'supported':True,'competitors':0,'gap':None});continue
        rival=max(float(energies[i,ll]) for i,ll in alternatives)
        out.append({'slot':j,'label':LABELS[l],'supported':True,'competitors':len(alternatives),
                    'gold_energy':float(energies[gi,l]),'strongest_alternative_energy':rival,
                    'gold_minus_alternative_gap':float(energies[gi,l]-rival)})
    return out


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def load_split(path: Path) -> tuple[list[dict[str,Any]],dict[str,str]]:
    path=Path(path); required=['logits.npy','pairs.npy','span_offsets.npy','token_offsets.npy','boundaries.npy','metadata.json']
    missing=[x for x in required if not (path/x).is_file()]
    if missing: raise FileNotFoundError(f'{path}: missing {missing}')
    hashes={name:sha256(path/name) for name in required}
    logits=np.load(path/'logits.npy',mmap_mode='r'); pairs=np.load(path/'pairs.npy',mmap_mode='r')
    so=np.load(path/'span_offsets.npy',mmap_mode='r');to=np.load(path/'token_offsets.npy',mmap_mode='r');bounds=np.load(path/'boundaries.npy',mmap_mode='r')
    meta=json.loads((path/'metadata.json').read_text(encoding='utf-8'))
    if logits.ndim!=2 or logits.shape[1]!=8 or pairs.shape!=(len(logits),2) or len(so)!=len(meta)+1 or len(to)!=len(meta)+1 or bounds.ndim!=2 or bounds.shape[1]!=2:
        raise ValueError(f'{path}: cache array shape mismatch')
    if int(so[0])!=0 or int(so[-1])!=len(logits) or int(to[0])!=0 or int(to[-1])!=len(bounds):
        raise ValueError(f'{path}: offsets do not cover the arrays exactly')
    if np.any(np.diff(so)<0) or np.any(np.diff(to)<0): raise ValueError(f'{path}: offsets are not monotone')
    for start in range(0,len(logits),1_000_000):
        if not np.isfinite(logits[start:start+1_000_000]).all(): raise ValueError(f'{path}: nonfinite logits')
    if np.any(bounds[:,0]>=bounds[:,1]): raise ValueError(f'{path}: invalid character token boundaries')
    ids=[m.get('document_id') for m in meta]
    if any(x is None for x in ids) or len(set(ids))!=len(ids): raise ValueError(f'{path}: document IDs are missing or duplicated')
    manifest_path=path/'manifest.json'
    if not manifest_path.is_file(): raise ValueError(f'{path}: immutable cache manifest is required')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    listed=manifest.get('files',manifest.get('file_sha256',{}))
    if isinstance(listed,list): listed={x['name']:x.get('sha256') for x in listed}
    for name,digest in hashes.items():
        expected=listed.get(name) if isinstance(listed,dict) else None
        if expected is None: raise ValueError(f'{path}: manifest has no hash for {name}')
        if expected!=digest: raise ValueError(f'{path}: manifest hash mismatch for {name}')
    hashes['manifest.json']=sha256(manifest_path)
    docs=[]
    for i,m in enumerate(meta):
        lo,hi=int(so[i]),int(so[i+1]);ta,tb=int(to[i]),int(to[i+1]); n=int(m['token_count']);k=int(m['expectedK'])
        if tb-ta!=n or hi<lo or hi>len(logits): raise ValueError(f'{path}: bad offsets for {m.get("document_id")}')
        z=np.asarray(logits[lo:hi],dtype=np.float64); en=z[:,1:]-z[:,:1]
        pp=np.asarray(pairs[lo:hi],dtype=np.int64)
        expected=np.asarray([(a,b) for b in range(1,n+1) for a in range(max(0,b-16),b)],dtype=np.int64).reshape(-1,2)
        if not np.array_equal(pp,expected): raise ValueError(f'{path}: candidate universe/order mismatch for {m.get("document_id")}')
        char_bounds=np.asarray(bounds[ta:tb],dtype=np.int64)
        starts={int(x):j for j,(x,_y) in enumerate(char_bounds)}
        ends={int(y):j+1 for j,(_x,y) in enumerate(char_bounds)}
        gold=[]; representable=True
        gold_char=[]
        for ent in m.get('gold',[]):
            lab=ent['label']; l=LABELS.index(lab) if isinstance(lab,str) else int(lab)
            if not 0<=l<len(LABELS): raise ValueError(f'{path}: invalid gold label')
            gold_char.append((int(ent['start']),int(ent['end']),l))
            a=starts.get(int(ent['start'])); b=ends.get(int(ent['end']))
            if a is None or b is None: representable=False;continue
            gold.append((a,b,l))
        gold=tuple(sorted(gold,key=lambda x:(x[0],x[1],x[2])))
        if any(gold[j-1][1]>gold[j][0] for j in range(1,len(gold))): representable=False
        ix={tuple(map(int,x)):j for j,x in enumerate(pairs[lo:hi])}
        supported=representable and len(gold)==k and all((a,b) in ix for a,b,_l in gold)
        gold_energy=float(sum(en[ix[a,b],l] for a,b,l in gold)) if supported else None
        docs.append({'document_id':m['document_id'],'n':n,'k':k,'pairs':np.asarray(pairs[lo:hi]),'energies':en,
                     'gold':gold,'gold_char':tuple(sorted(gold_char)),'supported':bool(supported),'gold_energy':gold_energy,
                     'boundaries':char_bounds,'group_id':m.get('group_id')})
    return docs,hashes


def _to_char(pred: list[tuple[int,int,int]], boundaries: np.ndarray) -> tuple[tuple[int,int,int],...]:
    return tuple(sorted((int(boundaries[a,0]),int(boundaries[b-1,1]),int(l)) for a,b,l in pred))


def _ordered_matches(gold: tuple[tuple[int,int,int],...] | list[tuple[int,int,int]],
                     pred: tuple[tuple[int,int,int],...] | list[tuple[int,int,int]]) -> list[int]:
    """Exact chronological slot matches, retaining order as part of the metric."""
    return [int(j<len(pred) and a==pred[j]) for j,a in enumerate(gold)]


def _bootstrap(deltas: list[int], slots: list[int]) -> dict[str,Any]:
    x=np.asarray(deltas,dtype=np.float64); w=np.asarray(slots,dtype=np.float64)
    if not len(x): return {'n':0,'mean':None,'ci95':None}
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    samples=np.empty(BOOTSTRAP_REPLICATES)
    for j in range(BOOTSTRAP_REPLICATES):
        ix=rng.integers(0,len(x),len(x)); samples[j]=x[ix].sum()/w[ix].sum()
    mean=float(x.sum()/w.sum())
    return {'n':len(x),'mean':mean,'ci95':[float(v) for v in np.quantile(samples,[.025,.975])],
            'replicates':BOOTSTRAP_REPLICATES,'seed':BOOTSTRAP_SEED}


def _paired_relative_bootstrap(base: list[float], fitted: list[float], slots: list[int]) -> dict[str,Any]:
    """Document-paired bootstrap of pooled NLL reduction, weighted by slots."""
    a=np.asarray(base,dtype=np.float64);b=np.asarray(fitted,dtype=np.float64)
    w=np.asarray(slots,dtype=np.float64)
    if not len(a): return {'n':0,'relative_improvement':None,'ci95':None}
    rng=np.random.default_rng(BOOTSTRAP_SEED);vals=np.empty(BOOTSTRAP_REPLICATES)
    for j in range(BOOTSTRAP_REPLICATES):
        ix=rng.integers(0,len(a),len(a)); denom=float(a[ix]@w[ix])
        vals[j]=(denom-float(b[ix]@w[ix]))/denom if denom else 0.0
    denom=float(a@w);point=(denom-float(b@w))/denom if denom else 0.0
    ci=[float(v) for v in np.quantile(vals,[.025,.975])]
    return {'n':len(a),'relative_improvement':point,'ci95':ci,'replicates':BOOTSTRAP_REPLICATES,'seed':BOOTSTRAP_SEED,
            'gate_pass':bool(point>=.01 and ci[0]>0)}


def _evaluate_one(d: dict[str,Any], beta: float) -> dict[str,Any]:
    try:
        from .slot_mbr import decode
    except ImportError:
        from slot_mbr import decode
    n,p,e,k=d['n'],d['pairs'],d['energies'],d['k']
    rank=ranking_record(n,p,e,k,d['gold'])
    base_mbr,base_diag=decode(n,p,e,k);map1=base_diag['map_prediction']
    fit_mbr,fit_diag=decode(n,p,e*beta,k);map_scaled=fit_diag['map_prediction']
    if [tuple(x) for x in map1] != [tuple(x) for x in map_scaled]:
        raise RuntimeError(f"MAP replay changed under positive scaling for {d['document_id']}")
    base_map_char=_to_char([tuple(x) for x in map1],d['boundaries'])
    base_mbr_char=_to_char(base_mbr,d['boundaries']);fit_mbr_char=_to_char(fit_mbr,d['boundaries'])
    goldchar=d['gold_char']
    gold_sorted=sorted(goldchar);map_sorted=sorted(base_map_char);base_sorted=sorted(base_mbr_char);fit_sorted=sorted(fit_mbr_char)
    map_slot=_ordered_matches(gold_sorted,map_sorted)
    base_slot=_ordered_matches(gold_sorted,base_sorted)
    fit_slot=_ordered_matches(gold_sorted,fit_sorted)
    map_correct=sum(map_slot);base_correct=sum(base_slot);fit_correct=sum(fit_slot)
    row={'document_id':d['document_id'],'K':k,**rank,'gold_supported':d['supported'],
         'map_gold_correct':map_correct,'mbr_tau1_correct':base_correct,'mbr_fitted_correct':fit_correct,
         'mbr_correct_delta':fit_correct-base_correct,
         'mbr_expected_gain_tau1_vs_map':float(base_diag['mbr_expected_utility']-base_diag['map_expected_utility']),
         'mbr_expected_gain_fitted_vs_map':float(fit_diag['mbr_expected_utility']-fit_diag['map_expected_utility']),
         'mbr_realized_gain_tau1_vs_map':base_correct-map_correct,
         'mbr_realized_gain_fitted_vs_map':fit_correct-map_correct,
         'map_replay_equal_under_scaling':True,'gold_slots':goldchar,'map_prediction':base_map_char,
         'mbr_tau1_prediction':base_mbr_char,'mbr_fitted_prediction':fit_mbr_char,
         'structured_nll_tau1_per_slot':None,'structured_nll_fitted_per_slot':None,
         'local_span_ranking':local_span_ranking(p,e,d['gold']) if d['supported'] else []}
    if d['supported']:
        z1,_=partition_stats(n,p,e,k,1.0);zb,_=partition_stats(n,p,e,k,beta)
        row['structured_nll_tau1_per_slot']=(z1-rank['gold_energy'])/k
        row['structured_nll_fitted_per_slot']=(zb-beta*rank['gold_energy'])/k
    return row


def run(cal_path: Path, eval_path: Path, out_path: Path, workers: int = 4) -> None:
    out_path=Path(out_path)
    if out_path.exists(): raise FileExistsError(f'refusing existing output directory: {out_path}')
    out_path.mkdir(parents=True)
    cal,ch_cal=load_split(cal_path)
    root=Path(__file__).resolve().parents[1]
    source_names=('src/scale_ranking_audit.py','src/slot_mbr.py','tests/test_scale_ranking_audit.py','note/bitrase-2/algo-3/protocol.md')
    source_hashes={name:sha256(root/name) for name in source_names}
    print(f'CAL loaded: {len(cal)} docs; fitting beta on supported structures',flush=True)
    fit=fit_beta(cal,progress=lambda it,beta,d,lo,hi: print(f'CAL bisection {it:02d}/{FIT_ITERATIONS}: beta={beta:.8f} derivative={d:.6g}',flush=True))
    fit['cal_input_hashes']=ch_cal
    fit['diagnostic_source_hashes']=source_hashes
    (out_path/'calibration.json').write_text(json.dumps(fit,indent=2,allow_nan=False),encoding='utf-8')
    # EVAL is opened only after the fitted scalar and its provenance are frozen.
    ev, ch_eval=load_split(eval_path)
    if any(not isinstance(d['group_id'],str) or not d['group_id'] for d in cal+ev):
        raise ValueError('all CAL/EVAL metadata rows must carry nonempty group_id')
    cal_ids={d['document_id'] for d in cal}; eval_ids={d['document_id'] for d in ev}
    if cal_ids & eval_ids: raise ValueError('CAL/EVAL document overlap')
    cal_groups={d['group_id'] for d in cal if d['group_id'] is not None}
    eval_groups={d['group_id'] for d in ev if d['group_id'] is not None}
    if cal_groups & eval_groups: raise ValueError('CAL/EVAL family overlap')
    if workers<1: raise ValueError('workers must be positive')
    rows=[None]*len(ev);per_label={l:[] for l in LABELS}
    if workers==1:
        for i,d in enumerate(ev):
            rows[i]=_evaluate_one(d,fit['beta'])
            if (i+1)%100==0: print(f'EVAL decoded {i+1}/{len(ev)}',flush=True)
    else:
        # Bound in-flight pickle payloads: score slices can be large.
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending={};next_i=0;limit=workers*2;done_count=0
            while next_i<len(ev) or pending:
                while next_i<len(ev) and len(pending)<limit:
                    fut=pool.submit(_evaluate_one,ev[next_i],fit['beta']);pending[fut]=next_i;next_i+=1
                finished,_=wait(pending,return_when=FIRST_COMPLETED)
                for fut in finished:
                    ix=pending.pop(fut);rows[ix]=fut.result();done_count+=1
                    if done_count%100==0: print(f'EVAL decoded {done_count}/{len(ev)}',flush=True)
    for r in rows:
        gold=sorted(r['gold_slots']);base=sorted(r['mbr_tau1_prediction']);fitted=sorted(r['mbr_fitted_prediction'])
        for j,g in enumerate(gold):
            l=g[2]
            per_label[LABELS[l]].append({'tau1_correct':int(j<len(base) and g==base[j]),
                                         'fitted_correct':int(j<len(fitted) and g==fitted[j])})
    supported=[r for r in rows if r['gold_supported']]
    eval_base_nll=[r['structured_nll_tau1_per_slot'] for r in rows if r['gold_supported']]
    eval_fit_nll=[r['structured_nll_fitted_per_slot'] for r in rows if r['gold_supported']]
    supported_slots=[d['k'] for d in ev if d['supported']]
    nll_boot=_paired_relative_bootstrap(eval_base_nll,eval_fit_nll,supported_slots)
    dec_boot=_bootstrap([r['mbr_correct_delta'] for r in rows],[r['K'] for r in rows])
    dec_boot['gain_percentage_points']=float(dec_boot['mean']*100) if dec_boot['mean'] is not None else None
    dec_boot['gate_pass']=bool(dec_boot['mean'] is not None and dec_boot['mean']>=.002 and dec_boot['ci95'][0]>0)
    report={'schema':'bitrase-2.algo-3.scale-ranking-audit.v1','fit':fit,
      'eval_input_hashes':ch_eval,'eval_docs':len(rows),'eval_supported_docs':len(supported),
      'eval_supported_slots':sum(r['K'] for r in supported),
      'structured_nll_per_slot':{key:float(sum(r[key]*r['K'] for r in supported)/sum(r['K'] for r in supported)) if supported else None for key in ('structured_nll_tau1_per_slot','structured_nll_fitted_per_slot')},
      'structured_nll_paired_doc_bootstrap':nll_boot,
      'ranking':{'supported_docs':sum(r['gold_supported'] for r in rows),'map_gold_docs':sum(r.get('map_is_gold',False) for r in rows),
                 'gold_vs_strongest_wrong':{'strict_win':sum(r['gold_minus_wrong_gap']>1e-12 for r in supported if r['gold_minus_wrong_gap'] is not None),
                   'tie':sum(abs(r['gold_minus_wrong_gap'])<=1e-12 for r in supported if r['gold_minus_wrong_gap'] is not None),
                   'loss':sum(r['gold_minus_wrong_gap']<-1e-12 for r in supported if r['gold_minus_wrong_gap'] is not None)},
                 'gold_minus_strongest_wrong_gap_mean':float(np.mean([r['gold_minus_wrong_gap'] for r in supported if r['gold_minus_wrong_gap'] is not None])) if any(r['gold_minus_wrong_gap'] is not None for r in supported) else None},
      'mbr':{'tau1_correct_slots':sum(r['mbr_tau1_correct'] for r in rows),'fitted_correct_slots':sum(r['mbr_fitted_correct'] for r in rows),
             'map_correct_slots':sum(r['map_gold_correct'] for r in rows),
             'all_eval_slots':sum(r['K'] for r in rows),
             'map_accuracy':sum(r['map_gold_correct'] for r in rows)/sum(r['K'] for r in rows) if rows else None,
             'tau1_accuracy':sum(r['mbr_tau1_correct'] for r in rows)/sum(r['K'] for r in rows) if rows else None,
             'fitted_accuracy':sum(r['mbr_fitted_correct'] for r in rows)/sum(r['K'] for r in rows) if rows else None,
             'paired_doc_bootstrap':dec_boot,
             'expected_gain_tau1_vs_map_per_slot':sum(r['mbr_expected_gain_tau1_vs_map'] for r in rows)/sum(r['K'] for r in rows) if rows else None,
             'expected_gain_fitted_vs_map_per_slot':sum(r['mbr_expected_gain_fitted_vs_map'] for r in rows)/sum(r['K'] for r in rows) if rows else None,
             'realized_gain_tau1_vs_map_per_slot':sum(r['mbr_realized_gain_tau1_vs_map'] for r in rows)/sum(r['K'] for r in rows) if rows else None,
             'realized_gain_fitted_vs_map_per_slot':sum(r['mbr_realized_gain_fitted_vs_map'] for r in rows)/sum(r['K'] for r in rows) if rows else None},
      'by_label':{l:{'slots':len(v),'tau1_accuracy':float(np.mean([x['tau1_correct'] for x in v])) if v else None,
                         'fitted_accuracy':float(np.mean([x['fitted_correct'] for x in v])) if v else None} for l,v in per_label.items()},
      'claims_limit':'Descriptive calibration/ranking evidence on the supplied split caches; no causal attribution or generalization claim.'}
    post_hashes={name:sha256(root/name) for name in source_hashes}
    if post_hashes!=source_hashes: raise RuntimeError('diagnostic source changed during run')
    report['diagnostic_source_hashes_before']=source_hashes;report['diagnostic_source_hashes_after']=post_hashes
    (out_path/'eval_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    (out_path/'eval_documents.json').write_text(json.dumps(rows,indent=2,allow_nan=False),encoding='utf-8')


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cal',type=Path,required=True);ap.add_argument('--eval',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=4)
    a=ap.parse_args();run(a.cal,a.eval,a.out,a.workers)


if __name__=='__main__': main()
