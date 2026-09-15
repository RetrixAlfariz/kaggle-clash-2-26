"""Read-only audit of frozen P1 predictions; never fits or decodes a new model."""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from run_slot_mbr import ROOT, OUT as P1, FROZEN, LABELS, read, write, sha, validate_frozen
from posterior_math import posterior_statistics

OUT = ROOT / 'output/bitrase-2/algo-2'
TAUS = (.5, 1, 2)

def triples(items):
    return [(int(x['start']), int(x['end']), x['label']) for x in items]

def margin_bin(x):
    return '<0' if x < 0 else '0–2' if x < 2 else '2–5' if x < 5 else '>=5'

def init():
    global ARR, META, GOLD, PRED, DIAG
    from prepared_loader import PreparedData
    ARR = {k: np.load(FROZEN / (k+'.npy'), mmap_mode='r') for k in
           ('logits','pairs','span_offsets','token_offsets','boundaries')}
    META = read(FROZEN/'metadata.json')
    GOLD = {r['document_id']: sorted(triples(r['entities'])) for r in
            PreparedData().load_dev('evaluation').to_pylist()}
    PRED = {name: {r['document_id']: triples(r['predicted']) for r in
                  pq.read_table(P1/path).to_pylist()} for name,path in
            [('MAP','map_predictions.parquet')]+[(str(t),f'tau{t:g}/predictions.parquet') for t in TAUS]}
    DIAG = {str(t): {r['document_id']:r for r in read(P1/f'tau{t:g}/decoder_diagnostics.json')} for t in TAUS}

def worker(i):
    m=META[i]; doc=m['document_id']; k=m['expectedK']; n=m['token_count']
    a,b=ARR['span_offsets'][i:i+2]; c,d=ARR['token_offsets'][i:i+2]
    p=ARR['pairs'][a:b]; z=ARR['logits'][a:b].astype(np.float64); e=z[:,1:]-z[:,:1]
    bd=ARR['boundaries'][c:d]
    lookup={(int(bd[s,0]),int(bd[t-1,1])):j for j,(s,t) in enumerate(p)}
    gold=GOLD[doc]; mp=PRED['MAP'][doc]
    assert len(gold)==len(mp)==k
    rows=[]; slots=[]; previous=-1.
    for tau in TAUS:
        stats=posterior_statistics(n,p,e,k,tau); mu=stats['mu']; diag=DIAG[str(tau)][doc]
        h=float(stats['entropy']); lc=float(stats['log_count'])
        assert -1e-8<=h<=lc+1e-8 and h>=previous-1e-8
        previous=h
        assert abs(stats['logZ']-diag['logZ'])<1e-8
        assert np.max(np.abs(mu.sum(axis=(1,2))-1))<1e-8
        predictions={'MAP':mp,'MBR':PRED[str(tau)][doc]}
        utilities={}; energies={}; correct={}
        for rule,pred in predictions.items():
            assert len(pred)==k
            confidence=[]; energy=0.; correctness=[]
            for j,tr in enumerate(pred):
                ix=lookup[tr[:2]]; label=LABELS.index(tr[2]); prob=float(mu[j,ix,label]); raw=float(e[ix,label])
                ok=tr==gold[j]; confidence.append(prob); energy+=raw; correctness.append(ok)
                mapix=lookup[mp[j][:2]]; mapmargin=float(e[mapix,LABELS.index(mp[j][2])])
                slots.append(dict(document_id=doc,tau=tau,rule=rule,slot=j+1,confidence=prob,
                    correct=int(ok),label=tr[2],gold_label=gold[j][2],margin_bin=margin_bin(raw),
                    map_margin_bin=margin_bin(mapmargin),changed=mp[j]!=predictions['MBR'][j],
                    k_bin='1–10' if k<=10 else '11–15' if k<=15 else '16+',
                    length_bin='<=128' if n<=128 else '129–256' if n<=256 else '>256'))
            utilities[rule]=sum(confidence); energies[rule]=energy; correct[rule]=sum(correctness)
        assert abs(utilities['MAP']-diag['map_expected_utility'])<1e-8
        assert abs(utilities['MBR']-diag['mbr_expected_utility'])<1e-8
        assert energies['MAP']>=energies['MBR']-1e-8
        assert utilities['MBR']>=utilities['MAP']-1e-8
        mb=predictions['MBR']
        rows.append(dict(document_id=doc,tau=tau,K=k,tokens=n,entropy=h,log_count=lc,
            normalized_entropy=h/lc,entropy_per_slot=h/k,
            map_structure_probability=float(np.exp(energies['MAP']/tau-stats['logZ'])),
            delta_U=utilities['MBR']-utilities['MAP'],delta_C=correct['MBR']-correct['MAP'],
            map_correct=correct['MAP'],mbr_correct=correct['MBR'],
            raw_energy_sacrifice=energies['MAP']-energies['MBR'],
            changed_slots=sum(x!=y for x,y in zip(mp,mb)),
            changed_boundaries=sum(x[:2]!=y[:2] for x,y in zip(mp,mb)),
            removed=len(set(mp)-set(mb)),added=len(set(mb)-set(mp)),
            k_bin=slots[-1]['k_bin'],length_bin=slots[-1]['length_bin']))
    return rows,slots

def reliability(rows):
    p=np.array([r['confidence'] for r in rows]); y=np.array([r['correct'] for r in rows])
    bins=[]; ece=0.
    for b in range(10):
        mask=np.minimum((p*10).astype(int),9)==b
        if mask.any():
            conf=float(p[mask].mean()); acc=float(y[mask].mean()); count=int(mask.sum())
            bins.append(dict(lower=b/10,upper=(b+1)/10,n=count,confidence=conf,accuracy=acc))
            ece+=count*abs(conf-acc)/len(p)
    return dict(n=len(p),confidence=float(p.mean()),accuracy=float(y.mean()),
                brier=float(np.mean((p-y)**2)),ece=ece,bins=bins)

def summarize(docs,slots):
    result={}
    for tau in TAUS:
        dd=[r for r in docs if r['tau']==tau]; ss=[r for r in slots if r['tau']==tau]
        k=np.array([r['K'] for r in dd]); dc=np.array([r['delta_C'] for r in dd]); du=np.array([r['delta_U'] for r in dd])
        rng=np.random.default_rng(20260915); boot=[]
        for _ in range(2000):
            ids=rng.integers(0,len(dd),len(dd)); den=k[ids].sum()
            boot.append([dc[ids].sum()/den,du[ids].sum()/den,(du[ids]-dc[ids]).sum()/den])
        strata={}
        for field in ('k_bin','length_bin'):
            strata[field]={}
            for value in sorted({r[field] for r in dd}):
                rr=[r for r in dd if r[field]==value]; den=sum(r['K'] for r in rr)
                strata[field][value]=dict(n=len(rr),delta_C_per_slot=sum(r['delta_C'] for r in rr)/den,
                    delta_U_per_slot=sum(r['delta_U'] for r in rr)/den,
                    entropy_per_slot=np.mean([r['entropy_per_slot'] for r in rr]).item())
        paired={}
        for field in ('gold_label','map_margin_bin','changed'):
            paired[field]={}
            for value in sorted({r[field] for r in ss},key=str):
                rr=[r for r in ss if r[field]==value]; mm=[r for r in rr if r['rule']=='MAP']; bb=[r for r in rr if r['rule']=='MBR']
                assert len(mm)==len(bb)
                paired[field][str(value)]={'n':len(mm),'delta_C_per_slot':
                    (sum(r['correct'] for r in bb)-sum(r['correct'] for r in mm))/len(mm)}
        rel={}
        for rule in ('MAP','MBR'):
            rr=[r for r in ss if r['rule']==rule]; rel[rule]={'overall':reliability(rr)}
            for field in ('label','margin_bin','k_bin','length_bin','changed'):
                rel[rule][field]={str(v):reliability([r for r in rr if r[field]==v]) for v in sorted({r[field] for r in rr},key=str)}
        result[str(tau)]=dict(documents=len(dd),slots=int(k.sum()),
            delta_C_per_slot=float(dc.sum()/k.sum()),delta_U_per_slot=float(du.sum()/k.sum()),
            bootstrap_columns=['delta_C_per_slot','delta_U_per_slot','expected_minus_realized'],
            bootstrap_ci=np.quantile(boot,[.025,.975],axis=0).T.tolist(),
            delta_c_u_correlation=float(np.corrcoef(dc,du)[0,1]),
            improved=int((dc>0).sum()),worsened=int((dc<0).sum()),unchanged=int((dc==0).sum()),
            means={key:float(np.mean([r[key] for r in dd])) for key in
                   ('entropy','log_count','normalized_entropy','entropy_per_slot','map_structure_probability','raw_energy_sacrifice')},
            churn={key:sum(r[key] for r in dd) for key in ('changed_slots','changed_boundaries','removed','added')},
            strata=strata,paired_slot_strata=paired,reliability=rel,
            frozen_mechanism=read(P1/f'tau{tau:g}/mechanism.json'))
    return result

def main():
    validate_frozen()
    if OUT.exists(): raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    paths=[FROZEN/'manifest.json',P1/'map_predictions.parquet',ROOT/'src/run_posterior_audit.py',
           ROOT/'src/posterior_math.py',ROOT/'tests/test_posterior_math.py',ROOT/'note/bitrase-2/algo-2/protocol.md',
           ROOT/'output/prepared/v1/dev.parquet']
    for tau in TAUS:
        paths += [P1/f'tau{tau:g}'/name for name in ('predictions.parquet','decoder_diagnostics.json','mechanism.json','mechanism_documents.json')]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}; write(OUT/'input_manifest.json',hashes)
    docs=[]; slots=[]; count=len(read(FROZEN/'metadata.json'))
    with ProcessPoolExecutor(max_workers=4,initializer=init) as pool:
        for i,(dd,ss) in enumerate(pool.map(worker,range(count),chunksize=8)):
            docs.extend(dd);slots.extend(ss)
            if (i+1)%500==0: print(f'{i+1}/{count} documents audited',flush=True)
    pq.write_table(pa.Table.from_pylist(docs),OUT/'documents.parquet')
    pq.write_table(pa.Table.from_pylist(slots),OUT/'slots.parquet')
    write(OUT/'summary.json',summarize(docs,slots))
    assert hashes=={str(p.relative_to(ROOT)):sha(p) for p in paths}
    validate_frozen()
    write(OUT/'validation.json',dict(passed=True,documents=count,temperatures=list(TAUS),
        invariants='all partitions, utilities, normalization, entropy bounds/monotonicity, input hashes passed',
        outputs={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}))
    print('Audit complete',flush=True)

if __name__=='__main__': main()
