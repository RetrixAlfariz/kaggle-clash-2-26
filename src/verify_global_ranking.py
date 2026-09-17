"""Independent artifact reconciliation for completed algo-4 training/evaluation."""
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from train_global_ranking import OUT, ROOT, gold_tokens, sha, write


def main():
    config=json.loads((OUT/'config.json').read_text())
    assignments=pq.read_table(OUT/'assignments.parquet').to_pylist()
    registry={r['document_id']:r for r in pq.read_table(OUT.parent/'exposure/registry.parquet').to_pylist()}
    fit=[a for a in assignments if a['role']=='FIT']; evaluation=[a for a in assignments if a['role']=='EVAL']
    assert len(fit)==52582 and len(evaluation)==3000
    assert len({a['document_id'] for a in assignments})==len(assignments)==55582
    assert not {a['group_id'] for a in fit}&{a['group_id'] for a in evaluation}
    assert all(registry[a['document_id']]['conditional_eligibility_if_training_metric_reuse_were_admissible'] for a in evaluation)
    eval_families={a['group_id'] for a in evaluation}
    assert all(r['conditional_eligibility_if_training_metric_reuse_were_admissible'] for r in registry.values() if r['family_id'] in eval_families)
    assert sha(OUT/'assignments.parquet')==config['assignment_sha256']
    inputs=json.loads((OUT/'training_inputs.json').read_text())
    assert inputs=={p:sha(ROOT/p) for p in inputs}
    complete=json.loads((OUT/'training_complete.json').read_text())
    assert complete['complete'] and complete['ce_exposure_matched']
    assert sha(OUT/'cache/encoder_epoch4.pt')==complete['encoder_sha256']
    assert sha(OUT/'control_epoch2.pt')==complete['control_sha256']
    assert sha(OUT/'challenger_epoch2.pt')==complete['challenger_sha256']
    logs=json.loads((OUT/'head_training.json').read_text())
    assert [r['epoch'] for r in logs['control']]==[1,2]==[r['epoch'] for r in logs['challenger']]
    assert all(c['ce_exposure_sha256']==r['ce_exposure_sha256'] for c,r in zip(logs['control'],logs['challenger']))
    lam=json.loads((OUT/'lambda.json').read_text()); measures=lam['measurements']
    expected=np.median([m['ce_norm'] for m in measures])/(np.median([m['rank_norm'] for m in measures])+1e-12)
    assert np.isclose(expected,lam['value'],rtol=1e-12) and .01<=expected<=100
    fit_ids={a['document_id'] for a in fit}
    assert len(set(lam['document_ids']))==512 and set(lam['document_ids'])<=fit_ids
    metadata=json.loads((OUT/'cache/FIT/metadata.json').read_text())
    assert [m['document_id'] for m in metadata]==[a['document_id'] for a in fit]
    offsets=np.load(OUT/'cache/FIT/token_offsets.npy'); bounds=np.load(OUT/'cache/FIT/boundaries.npy',mmap_mode='r')
    gold=[gold_tokens(dict(entities=m['gold'],expected_entity_count=m['expectedK']),bounds[offsets[i]:offsets[i+1]]) for i,m in enumerate(metadata)]
    for epoch in (1,2):
        rivals=json.loads((OUT/f'competitors_epoch{epoch}.json').read_text())
        assert set(rivals)=={str(i) for i in range(len(fit))}
        for i,m in enumerate(metadata):
            rival=rivals[str(i)]
            if gold[i] is None:
                assert rival is None; continue
            assert len(rival)==m['expectedK']
            assert all(0<=a<b<=m['token_count'] and b-a<=16 and 0<=l<7 for a,b,l in rival)
            assert all(rival[j-1][1]<=rival[j][0] for j in range(1,len(rival)))
            assert tuple(map(tuple,rival))!=gold[i]
    for arm in ('control','challenger'):
        checkpoint=torch.load(OUT/f'{arm}_epoch2.pt',map_location='cpu',weights_only=True)
        assert all(bool(torch.isfinite(t).all()) for t in checkpoint.values())
    report=dict(training_verified=True,counts={'FIT':len(fit),'EVAL':len(evaluation)},supported_fit=sum(g is not None for g in gold),
        ce_exposure_matched=True,lambda_reconciled=True,all_competitors_feasible_incorrect=True,source_hashes_unchanged=True)
    result_path=OUT/'EVAL/evaluation/eval_summary.json'
    if result_path.exists():
        result=json.loads(result_path.read_text()); documents=json.loads((result_path.parent/'eval_documents.json').read_text())
        assert {r['document_id'] for r in documents}=={a['document_id'] for a in evaluation}
        assert len(documents)==len(evaluation)
        slots=sum(r['K'] for r in documents); assert slots==result['eval_slots']
        for decoder in ('map','mbr'):
            for arm in ('control','challenger'):
                correct=0; tp=0
                for r in documents:
                    goldchar=list(map(tuple,r['gold_slots'])); pred=list(map(tuple,r[f'{decoder}_{arm}']))
                    assert len(pred)==len(goldchar)==r['K'] and pred==sorted(pred)
                    matches=sum(a==b for a,b in zip(goldchar,pred))
                    assert matches==r[f'{decoder}_correct_{arm}']
                    correct+=matches; tp+=len(set(goldchar)&set(pred))
                assert correct==result['metrics'][decoder][arm]['correct_slots']
                assert tp==result['metrics'][decoder][arm]['entity_f1']['true_positive']
        report['evaluation_predictions_reconciled']=True; report['eval_slots']=slots
    write(OUT/'verification.json',report); print(json.dumps(report,indent=2))


if __name__=='__main__': main()
