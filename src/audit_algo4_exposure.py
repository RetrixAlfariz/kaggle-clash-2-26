"""Reconstruct TRAIN exposure before selecting any algo-4 evaluation cohort."""
import hashlib
import json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/bitrase-2/algo-4/exposure'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def whole_family_eligible(families, eligible):
    return {g:ids for g,ids in families.items() if all(eligible[i] for i in ids)}

def main():
    if OUT.exists():raise FileExistsError(OUT)
    names={
        'prepared_train':'output/prepared/v1/train.parquet',
        'prepared_assignments':'output/prepared/v1/split_assignments.parquet',
        'prepared_manifest':'output/prepared/v1/manifest.json',
        'family_memberships':'output/bitrase-2/algo-3/grouping.json',
        'algo3_roles':'output/bitrase-2/algo-3/assignments.parquet',
        'structure_memberships':'output/bitrase-2/structure-diagnostic/data/documents.json',
        'structure_manifest':'output/bitrase-2/structure-diagnostic/data/manifest.json',
        'p0_sampling':'output/bitrase-2/p0/seed2026/sampling_documents.parquet',
        'p0_training_metrics':'output/bitrase-2/p0/seed2026/training_log.json',
        'p0_sampling_report':'output/bitrase-2/p0/seed2026/sampling_report.json',
        'p0_diagnostic_interpretation':'note/bitrase-2/p0/research_questions.md',
        'p0_training_source':'src/run_span_probe.py',
        'p0_sampling_source':'src/span_probe.py',
        'p0_protocol':'note/bitrase-2/p0/protocol.md',
        'probe_source':'src/baseline_probe.py',
        'probe_report':'output/baseline_probe/report.json',
        'preparation_source':'src/prepare_data.py',
        'data_understanding_report':'note/REPORT.md',
        'honorific_data_audit':'output/experiments88/honorific_train_audit.json',
        'style_source':'src/run_document_style.py',
        'style_report':'note/bitrase-1.algo-5/report.md',
        'algo3_training_source':'src/train_scale_ranking.py',
        'algo3_protocol':'note/bitrase-2/algo-3/protocol.md',
        'family_source':'src/structure_families.py',
        'structure_protocol':'note/bitrase-2/structure-diagnostic/protocol.md',
    }
    hashes={name:sha(ROOT/path) for name,path in names.items()}
    manifest=read(ROOT/names['prepared_manifest'])
    assert hashes['prepared_train']==manifest['files']['train.parquet']['sha256']
    ids=pq.read_table(ROOT/names['prepared_train'],columns=['document_id']).column('document_id').to_pylist()
    assert len(ids)==len(set(ids))==55582
    universe=set(ids)
    assignments=pq.read_table(ROOT/names['prepared_assignments'],filters=[('partition','=','train')]).to_pylist()
    assert len(assignments)==len(ids) and {r['document_id'] for r in assignments}==universe
    meta={r['document_id']:r for r in assignments}
    roles=pq.read_table(ROOT/names['algo3_roles']).to_pylist()
    assert {r['document_id'] for r in roles}==universe and len(roles)==len(ids)
    role={r['document_id']:r['role'] for r in roles}
    structure={r['document_id'] for r in read(ROOT/names['structure_memberships'])}
    assert len(structure)==6000 and structure<=universe
    sample=pq.read_table(ROOT/names['p0_sampling']).to_pylist()
    assert len(sample)==len(ids) and {r['document_id'] for r in sample}==universe
    random_members={r['document_id'] for r in sample if r['random_draws']>0}
    epoch2=next(r for r in read(ROOT/names['p0_training_metrics']) if r['epoch']==2)
    assert sum(r['random_draws'] for r in sample)==epoch2['strata']['random']['count']==14228992
    assert random_members==universe
    families=read(ROOT/names['family_memberships'])['sampled_families']
    members=[d for ds in families.values() for d in ds]
    assert len(members)==len(ids) and set(members)==universe
    family={d:g for g,ds in families.items() for d in ds}
    rows=[]
    for doc in ids:
        dedicated=[];cal=[];selection=[];uncertain=[]
        if doc in structure:
            dedicated.append('structure_probe_6000');selection.append('structure_feature_gate')
        if role[doc]=='EVAL':
            dedicated.append('algo3_eval');selection.append('algo3_scale_gate')
        if role[doc]=='CAL':cal.append('algo3_cal')
        if meta[doc]['prior_probe_role']=='dev':dedicated.append('initial_baseline_probe_validation')
        if meta[doc]['reviewed_example']:uncertain.append('historical_reviewed_example_purpose_requires_classification')
        # Ordinary supervised fitting is allowed. This separate event records
        # the explicit later research use of a training-accuracy comparison.
        reused=['p0_epoch2_random_accuracy_reused_in_research'] if doc in random_members else []
        rows.append(dict(document_id=doc,family_id=family[doc],prior_supervised_training='yes',
            prior_evaluation=dedicated+reused,prior_dedicated_evaluation=dedicated,
            prior_calibration=cal,prior_structure_probe=['structure_probe_6000'] if doc in structure else [],
            prior_model_selection=selection,prior_training_metric_diagnostic_reuse=reused,
            prior_data_audit=['reconstruction_annotation_metadata_audits','honorific_annotation_profile'],
            historical_reviewed_example=bool(meta[doc]['reviewed_example']),
            uncertain_history=bool(uncertain),uncertainty_reasons=uncertain,
            history_scope='reconstructed named artifacts; no claim of exhaustive external history',
            membership_source=[names['p0_sampling'],names['algo3_roles'],names['structure_memberships'],names['prepared_assignments']],
            eligible_under_broad_no_diagnostic_exposure=not(dedicated or reused or cal or selection or uncertain),
            conditional_eligibility_if_training_metric_reuse_were_admissible=not(dedicated or cal or selection or uncertain)))
    strict=whole_family_eligible(families,{r['document_id']:r['eligible_under_broad_no_diagnostic_exposure'] for r in rows})
    narrower=whole_family_eligible(families,{r['document_id']:r['conditional_eligibility_if_training_metric_reuse_were_admissible'] for r in rows})
    count=sum(map(len,strict.values()));assert count==0
    OUT.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows),OUT/'registry.parquet')
    sources=[dict(key=k,path=path,sha256=hashes[k]) for k,path in names.items()]
    (OUT/'searched_sources.json').write_text(json.dumps(sources,indent=2),encoding='utf-8')
    summary=dict(status='PENDING_EXPOSURE_CLASSIFICATION',prepared_train_documents=len(ids),discovered_families=len(families),
        strict_eligible_documents=count,strict_eligible_families=len(strict),minimum_eval2_documents=2000,target_eval2_documents=3000,
        p0_training_metric_research_reuse_documents=len(random_members),
        p0_epoch2_random_predictions=epoch2['strata']['random']['count'],
        p0_epoch2_random_accuracy=epoch2['strata']['random']['accuracy'],p0_epoch2_old_fp_accuracy=epoch2['strata']['old_fp']['accuracy'],
        structure_probe_documents=len(structure),algo3_cal_documents=sum(x=='CAL' for x in role.values()),
        algo3_eval_documents=sum(x=='EVAL' for x in role.values()),
        dedicated_evaluation_or_calibration_union=sum(bool(r['prior_dedicated_evaluation'] or r['prior_calibration']) for r in rows),
        prior_probe_validation_in_prepared_train=sum(meta[d]['prior_probe_role']=='dev' for d in ids),
        uncertain_reviewed_examples=sum(r['uncertain_history'] for r in rows),
        conditional_narrower_policy_documents=sum(map(len,narrower.values())),
        conditional_narrower_policy_families=len(narrower),
        narrower_policy_authorized=False,eval2_selected=False,training_started=False,mining_benchmarked=False,
        classification_note='The zero count depends on counting explicitly repurposed training-accuracy diagnostics as prior diagnostic exposure. It does not follow from supervised training alone. No alternative exposure rule was silently adopted.',
        sources_unchanged=hashes=={name:sha(ROOT/path) for name,path in names.items()},
        registry_sha256=sha(OUT/'registry.parquet'),source_sha256=sha(Path(__file__)))
    assert summary['sources_unchanged']
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
