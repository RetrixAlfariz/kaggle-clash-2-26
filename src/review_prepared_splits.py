# /// script
# requires-python = ">=3.13"
# dependencies = ["pyarrow>=18"]
# ///
"""Read-only audit of prepared/v1 split roles and distributions."""
import csv, json, collections
from pathlib import Path
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"output/preparation_audit"
def shares(counter,total): return {k:{"count":v,"pct":round(100*v/total,4)} for k,v in sorted(counter.items())}
def main():
    a=pq.read_table(ROOT/"output/prepared/v1/split_assignments.parquet").to_pylist()
    byid={r["document_id"]:r for r in a}; parts=collections.Counter(r["partition"] for r in a)
    base=json.loads((ROOT/"output/baseline_probe/report.json").read_text(encoding="utf-8"))
    btrain=set(base["split"]["train"]); bvalid=set(base["split"]["validation"])
    overlap={"baseline_train_by_partition":collections.Counter(byid[d]["partition"] for d in btrain),
             "baseline_validation_by_partition":collections.Counter(byid[d]["partition"] for d in bvalid)}
    labs=collections.defaultdict(collections.Counter)
    for row in csv.DictReader((ROOT/"Data/train/train_labels.csv").open(encoding="utf-8",newline="")):
        labs[byid[row["document_id"]]["partition"]][row["label"]]+=1
    prof={}
    for part in sorted(parts):
        rows=[r for r in a if r["partition"]==part]; n=len(rows)
        prof[part]={"documents":n,"document_share_pct":round(100*n/len(a),4),
                    "channel":shares(collections.Counter(r["channel"] for r in rows),n),
                    "domain":shares(collections.Counter(r["domain"] for r in rows),n),
                    "entity_count":{"total":sum(r["entity_count"] for r in rows),"mean":round(sum(r["entity_count"] for r in rows)/n,4),"zero_docs":sum(r["entity_count"]==0 for r in rows)},
                    "labels":shares(labs[part],sum(labs[part].values()))}
    role=collections.Counter((r["partition"],r["prior_probe_role"]) for r in a)
    report={"scope":"read-only audit; prepared/v1 unchanged","source_files":["output/prepared/v1/split_assignments.parquet","output/baseline_probe/report.json","Data/train/train_labels.csv"],"partition_counts":dict(parts),"assignment_roles":{"prior_probe_role_by_partition":{p:dict(collections.Counter(r["prior_probe_role"] for r in a if r["partition"]==p)) for p in sorted(parts)},"reviewed_examples_total":sum(r["reviewed_example"] for r in a)},"baseline_overlap":{"baseline_train_count":len(btrain),"baseline_validation_count":len(bvalid),"baseline_train_by_prepared_partition":dict(overlap["baseline_train_by_partition"]),"baseline_validation_by_prepared_partition":dict(overlap["baseline_validation_by_partition"]),"baseline_train_all_present":btrain<=byid.keys(),"baseline_validation_all_present":bvalid<=byid.keys()},"distributions":prof,"differences":{"label_pct_point_vs_train":{p:{l:round(prof[p]["labels"].get(l,{}).get("pct",0)-prof["train"]["labels"].get(l,{}).get("pct",0),4) for l in set(prof[p]["labels"])|set(prof["train"]["labels"])} for p in ["dev","holdout"]},"channel_pct_point_vs_train":{p:{l:round(prof[p]["channel"].get(l,{}).get("pct",0)-prof["train"]["channel"].get(l,{}).get("pct",0),4) for l in set(prof[p]["channel"])|set(prof["train"]["channel"])} for p in ["dev","holdout"]},"domain_pct_point_vs_train":{p:{l:round(prof[p]["domain"].get(l,{}).get("pct",0)-prof["train"]["domain"].get(l,{}).get("pct",0),4) for l in set(prof[p]["domain"])|set(prof["train"]["domain"])} for p in ["dev","holdout"]}},"caveats":["Label percentages use annotation counts, while channel/domain percentages use document counts.","Baseline validation is prior 2,000-document probe validation; prepared dev and holdout are separate current partitions.","No near-duplicate or semantic leakage audit was performed here.","Counts are descriptive and do not establish predictive validity."]}
    OUT.mkdir(parents=True,exist_ok=True); (OUT/"split_review.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"partition_counts":dict(parts),"baseline_overlap":report["baseline_overlap"],"differences":report["differences"]},indent=2))
if __name__=="__main__": main()
