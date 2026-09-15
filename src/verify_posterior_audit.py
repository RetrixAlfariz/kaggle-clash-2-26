"""Independent artifact-level checks, separate from posterior calculation."""
import numpy as np
import pyarrow.parquet as pq
from run_slot_mbr import ROOT, OUT as P1, FROZEN, read, write, sha

def main():
    out=ROOT/'output/bitrase-2/algo-2'
    docs=pq.read_table(out/'documents.parquet').to_pylist()
    slots=pq.read_table(out/'slots.parquet').to_pylist()
    meta=read(FROZEN/'metadata.json'); ids={r['document_id'] for r in meta}
    assert sha(P1/'map_predictions.parquet')==read(P1/'map_replay.json')['prediction_sha256']
    summary=read(out/'summary.json')
    for path in [P1/'map_predictions.parquet']+[P1/f'tau{t:g}/predictions.parquet' for t in (.5,1,2)]:
        records=pq.read_table(path).to_pylist()
        assert len(records)==len(ids) and {r['document_id'] for r in records}==ids
        for record in records:
            pp=[(x['start'],x['end'],x['label']) for x in record['predicted']]
            assert pp==sorted(pp) and len(pp)==len(set(pp))
            assert all(a[1]<=b[0] for a,b in zip(pp,pp[1:]))
    for tau in (.5,1,2):
        perf=read(P1/f'tau{tau:g}/performance.json')
        assert sha(P1/f'tau{tau:g}/predictions.parquet')==perf['prediction_sha256']
        assert sha(FROZEN/'manifest.json')==perf['frozen_manifest_sha256']
        dd=[r for r in docs if r['tau']==tau]
        assert len(dd)==len(ids) and {r['document_id'] for r in dd}==ids
        sr=summary[str(tau)] if str(tau) in summary else summary[str(float(tau))]
        for rule in ('MAP','MBR'):
            ss=[r for r in slots if r['tau']==tau and r['rule']==rule]
            assert len(ss)==85336 and len({(r['document_id'],r['slot']) for r in ss})==85336
            assert abs(np.mean([r['correct'] for r in ss])-sr['reliability'][rule]['overall']['accuracy'])<1e-12
        mechanism=sr['frozen_mechanism']['summed_scalar_diagnostics']
        assert sum(r['mbr_correct'] for r in dd)==mechanism['model_stats.mbr.correct_entities']
        assert sum(r['map_correct'] for r in dd)==69067
        assert sum(r['delta_C'] for r in dd)==sum(r['mbr_correct']-r['map_correct'] for r in dd)
        assert sum(v for key,v in mechanism.items() if key.startswith('transitions.'))==85336
    counts={}
    for r in docs:
        key=r['document_id']
        if key in counts: assert counts[key]==r['log_count']
        counts[key]=r['log_count']
    write(out/'independent_validation.json',dict(passed=True,documents=len(ids),
        checks=['P1 historical prediction hashes','unique complete document and slot coverage',
                'same structure count at all temperatures','independent aggregate correctness',
                'frozen positional transition counts'],
        source_sha256=sha(__file__)))
    print('Independent artifact checks passed')

if __name__=='__main__':main()
