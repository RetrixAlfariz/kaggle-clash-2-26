"""Evaluate fixed algo-5 heads; preserve explicit adaptive-development provenance."""
import json
from pathlib import Path
from evaluate_global_ranking import run, _family_bootstrap, sha256


def main():
    root=Path(__file__).resolve().parents[1]
    path=root/'output/bitrase-2/algo-5/run/EVAL'
    own_hash=sha256(Path(__file__))
    result=run(path/'control',path/'challenger',path/'evaluation')
    current=json.loads((path/'evaluation/eval_documents.json').read_text())
    previous_path=root/'output/bitrase-2/algo-4/run/EVAL/evaluation/eval_documents.json'
    previous={r['document_id']:r for r in json.loads(previous_path.read_text())}
    assert len(previous)==len(current) and set(previous)=={r['document_id'] for r in current}
    comparisons={}
    for arm in ('control','challenger'):
        comparisons[arm]={}
        for decoder in ('map','mbr'):
            paired=[]
            for r in current:
                old=previous[r['document_id']]
                assert (r['K'],r['group_id'],r['gold_slots'])==(old['K'],old['group_id'],old['gold_slots'])
                paired.append(dict(group_id=r['group_id'],K=r['K'],**{
                    f'{decoder}_correct_control':old[f'{decoder}_correct_control'],
                    f'{decoder}_correct_challenger':r[f'{decoder}_correct_{arm}']}))
            comparisons[arm][decoder]=_family_bootstrap(paired,decoder)
    result.update(schema='bitrase-2.algo-5.ce-refinement-eval.v1',experiment='bitrase-2.algo-5',
        adaptive_development=True,comparison_to_original_algo4_ce_control=comparisons,
        original_predictions_sha256=sha256(previous_path),wrapper_sha256=own_hash,
        claims_limit='EVAL2 has already informed this follow-up. Scores and bootstrap intervals are adaptive internal development evidence, not untouched confirmation or Kaggle results.')
    assert own_hash==sha256(Path(__file__))
    (path/'evaluation/eval_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('metrics','paired_family_bootstrap','comparison_to_original_algo4_ce_control')},indent=2))


if __name__=='__main__': main()
