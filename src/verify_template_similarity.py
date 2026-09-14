"""Independent checks for report arithmetic, examples and holdout disclosure."""
import hashlib
import json
import re
from pathlib import Path
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/template_similarity"


def main():
    report = json.loads((OUT / "template_similarity_report.json").read_text())
    docs = {}
    for split in ("train", "dev"):
        docs.update({r["document_id"]: r["full_text"] for r in pq.read_table(ROOT / "output/prepared/v1" / f"{split}.parquet", columns=["document_id", "full_text"]).to_pylist()})
    def exact(text):
        words = text.lower().split()
        return {tuple(words[i:i + 5]) for i in range(len(words) - 4)}
    mismatch = 0
    for example in report["examples_train_dev"]:
        left, right = exact(docs[example["dev_id"]]), exact(docs[example["train_id"]])
        actual = len(left.intersection(right)) / len(left.union(right))
        mismatch += abs(actual - example["exact_string_jaccard"]) > 1e-12
    arithmetic = all(sum(c["candidate_score_bins"].values()) == c["unique_candidate_pairs_scored"] and c["threshold_pair_counts"]["0.9"] <= c["threshold_pair_counts"]["0.8"] for c in report["comparisons"].values())
    holdout_ids = set(pq.read_table(ROOT / "output/prepared/v1/holdout.parquet", columns=["document_id"])["document_id"].to_pylist())
    exposed = set()
    for name in ("template_similarity_report.json", "train_dev_pairs.json"):
        exposed.update(set(re.findall(r"DOC_\d+", (OUT / name).read_text())) & holdout_ids)
    source_hashes_valid = True
    for path, expected in report["source_hashes"].items():
        with (ROOT / path).open("rb") as stream:
            source_hashes_valid &= hashlib.file_digest(stream, "sha256").hexdigest() == expected
    code_hash_valid = hashlib.sha256((ROOT / "src/audit_template_similarity.py").read_bytes()).hexdigest() == report["audit_code_sha256"]
    result = {"examples_checked": len(report["examples_train_dev"]), "example_score_mismatches": mismatch, "arithmetic_valid": arithmetic, "persisted_holdout_id_count": len(exposed), "source_hashes_valid": source_hashes_valid, "code_hash_valid": code_hash_valid}
    (OUT / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if mismatch or exposed or not all((arithmetic, source_hashes_valid, code_hash_valid)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
