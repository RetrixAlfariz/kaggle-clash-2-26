"""Independent checks of completed algo-3 metrics and numerical implementation."""
import json
from pathlib import Path
import numpy as np
from scale_ranking_audit import load_split, sha256, LABELS
from posterior_math import posterior_statistics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/bitrase-2/algo-3'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def main():
    audit=OUT/'audit';summary=read(audit/'eval_summary.json');rows=read(audit/'eval_documents.json')
    fit=read(audit/'calibration.json');beta=fit['beta'];docs,hashes=load_split(OUT/'EVAL')
    assert hashes==summary['eval_input_hashes']
    cal_hashes=fit['cal_input_hashes']
    assert all(sha256(OUT/'CAL'/name)==digest for name,digest in cal_hashes.items())
    assert len(rows)==len(docs)==len({r['document_id'] for r in rows})
    totals={'map_gold_correct':0,'mbr_tau1_correct':0,'mbr_fitted_correct':0}
    labels={l:[0,0,0] for l in LABELS}
    for row,doc in zip(rows,docs):
        assert row['document_id']==doc['document_id'] and row['K']==doc['k']
        gold=tuple(doc['gold_char']);assert tuple(map(tuple,row['gold_slots']))==gold
        for key,col in [('map_prediction','map_gold_correct'),('mbr_tau1_prediction','mbr_tau1_correct'),('mbr_fitted_prediction','mbr_fitted_correct')]:
            pred=tuple(map(tuple,row[key]));assert len(pred)==len(gold)==row['K']
            assert list(pred)==sorted(pred) and len(set(pred))==len(pred)
            assert all(a[1]<=b[0] for a,b in zip(pred,pred[1:]))
            correctness=[int(g==p) for g,p in zip(gold,pred)]
            assert sum(correctness)==row[col];totals[col]+=sum(correctness)
            if key!='map_prediction':
                for g,ok in zip(gold,correctness):labels[LABELS[g[2]]][1 if key=='mbr_tau1_prediction' else 2]+=ok
        for g in gold:labels[LABELS[g[2]]][0]+=1
        assert row['mbr_correct_delta']==row['mbr_fitted_correct']-row['mbr_tau1_correct']
        starts={int(a):i for i,(a,b) in enumerate(doc['boundaries'])}
        ends={int(b):i+1 for i,(a,b) in enumerate(doc['boundaries'])}
        ix={tuple(p):i for i,p in enumerate(doc['pairs'])}
        energy=sum(float(doc['energies'][ix[starts[a],ends[b]],l]) for a,b,l in row['map_prediction'])
        assert abs(energy-row['map_energy'])<1e-9
        if row['gold_supported']:
            assert abs(doc['gold_energy']-row['gold_energy'])<1e-9
            if row['gold_minus_wrong_gap']>1e-12:assert row['map_gold_correct']==row['K']
    k=sum(r['K'] for r in rows)
    assert totals['mbr_tau1_correct']==summary['mbr']['tau1_correct_slots']
    assert totals['mbr_fitted_correct']==summary['mbr']['fitted_correct_slots']
    assert k==summary['mbr']['all_eval_slots']
    for l,(nn,aa,bb) in labels.items():
        assert summary['by_label'][l]['slots']==nn
        assert abs(summary['by_label'][l]['tau1_accuracy']-aa/nn)<1e-12
        assert abs(summary['by_label'][l]['fitted_accuracy']-bb/nn)<1e-12
    supported=[r for r in rows if r['gold_supported']]
    for key in ('structured_nll_tau1_per_slot','structured_nll_fitted_per_slot'):
        pooled=sum(r[key]*r['K'] for r in supported)/sum(r['K'] for r in supported)
        assert abs(pooled-summary['structured_nll_per_slot'][key])<1e-12
    # Independent F/B marginal implementation checks expectation-semiring NLL
    # on the first 20 supported document IDs, fixed by identity rather than scores.
    byid={d['document_id']:d for d in docs};checked=[]
    for row in sorted(supported,key=lambda r:r['document_id'])[:20]:
        d=byid[row['document_id']]
        for b,key in [(1.,'structured_nll_tau1_per_slot'),(beta,'structured_nll_fitted_per_slot')]:
            stat=posterior_statistics(d['n'],d['pairs'],d['energies'],d['k'],1/b)
            nll=(stat['logZ']-b*d['gold_energy'])/d['k']
            assert abs(nll-row[key])<1e-9
        checked.append(d['document_id'])
    source=summary['diagnostic_source_hashes_before']
    assert source==summary['diagnostic_source_hashes_after']
    assert all(sha256(ROOT/name)==digest for name,digest in source.items())
    result=dict(passed=True,documents=len(rows),slots=k,independently_recomputed_counts=totals,
                independent_partition_checks=checked,source_sha256=sha256(Path(__file__)),
                hashes={p.name:sha256(p) for p in audit.iterdir() if p.is_file()})
    (OUT/'independent_audit_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Independent audit checks passed',totals,flush=True)

if __name__=='__main__':main()
