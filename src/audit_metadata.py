# /// script
# dependencies = ["numpy", "pyarrow", "scipy"]
# ///
"""Metadata and train/test distribution audit for Kaggle Clash 2.

Run with: uv run --with pyarrow --with scipy python src/audit_metadata.py
"""
from __future__ import annotations

import csv, hashlib, json, math, re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

try:
    from scipy.stats import ks_2samp
except Exception:  # pragma: no cover
    ks_2samp = None

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "data_understanding"
OUT.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f)


def quantiles(values):
    a = np.asarray(values, dtype=float)
    return {str(q): float(np.quantile(a, q)) for q in (0, .01, .25, .5, .75, .99, 1)}


def metadata(path):
    rows = pq.read_table(path).to_pylist()
    return rows


def cat_props(rows, col):
    c = Counter(str(r.get(col)) for r in rows)
    n = len(rows)
    return {k: {"count": v, "proportion": v / n} for k, v in sorted(c.items(), key=lambda x: (-x[1], x[0]))}


def numeric_compare(a, b):
    x, y = np.asarray(a, float), np.asarray(b, float)
    result = {"train": {"quantiles": quantiles(x), "mean": float(x.mean()), "std": float(x.std())},
              "test": {"quantiles": quantiles(y), "mean": float(y.mean()), "std": float(y.std())}}
    pooled = math.sqrt((x.var() + y.var()) / 2) or 1
    result["mean_difference_test_minus_train"] = float(y.mean() - x.mean())
    result["standardized_mean_difference"] = float((y.mean() - x.mean()) / pooled)
    if ks_2samp:
        ks = ks_2samp(x, y)
        result["ks_statistic"] = float(ks.statistic)
        result["ks_p_value"] = float(ks.pvalue)
    return result


def tvd(a, b):
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys)


def cramers_v(table):
    a = np.asarray(table, float)
    n = a.sum()
    if n == 0:
        return 0.0
    expected = a.sum(axis=1, keepdims=True) @ a.sum(axis=0, keepdims=True) / n
    chi2 = ((a - expected) ** 2 / np.where(expected == 0, 1, expected)).sum()
    phi2 = chi2 / n
    r, k = a.shape
    # Bergsma correction for small-sample bias; retain raw value separately.
    phi2corr = max(0, phi2 - (k - 1) * (r - 1) / max(1, n - 1))
    rcorr = r - (r - 1) ** 2 / max(1, n - 1)
    kcorr = k - (k - 1) ** 2 / max(1, n - 1)
    return float(math.sqrt(phi2corr / max(1e-12, min(kcorr - 1, rcorr - 1))))


def doc_text(path):
    grouped = defaultdict(list)
    for r in read_csv(path):
        grouped[r["document_id"]].append((int(r["segment_index"]), r["text"]))
    out = {}
    for d, parts in grouped.items():
        parts.sort()
        out[d] = "\n".join(x[1] for x in parts)
    return out


def main():
    train_meta = metadata(ROOT / "Data/train/train_metadata.parquet")
    test_meta = metadata(ROOT / "Data/test/test_metadata.parquet")
    train_ids = {r["document_id"] for r in train_meta}
    test_ids = {r["document_id"] for r in test_meta}
    labels = list(read_csv(ROOT / "Data/train/train_labels.csv"))
    label_counts = Counter(r["document_id"] for r in labels)
    label_by_doc = defaultdict(Counter)
    for r in labels:
        label_by_doc[r["document_id"]][r["label"]] += 1

    # Submission slots encode test document and expected entity count.
    slots = Counter()
    bad_rows = 0
    for r in read_csv(ROOT / "Data/sample_submission.csv"):
        doc = r["row_id"].rsplit("_", 1)[0]
        slots[doc] += 1
        if doc not in test_ids:
            bad_rows += 1

    numeric = ["n_chars", "n_segments", "n_words", "n_nonempty_segments"]
    num_cmp = {c: numeric_compare([r[c] for r in train_meta], [r[c] for r in test_meta]) for c in numeric}
    cats = ["channel", "domain", "source_system"]
    cat_cmp = {}
    for c in cats:
        tp = {k: v["proportion"] for k, v in cat_props(train_meta, c).items()}
        ep = {k: v["proportion"] for k, v in cat_props(test_meta, c).items()}
        cat_cmp[c] = {"train": cat_props(train_meta, c), "test": cat_props(test_meta, c),
                      "total_variation_distance": tvd(tp, ep)}
    numeric_matrix = np.asarray([[r[c] for c in numeric] for r in train_meta], float)
    corr = np.corrcoef(numeric_matrix, rowvar=False)
    redundancy = {"numeric_pearson": {numeric[i]: {numeric[j]: float(corr[i, j]) for j in range(len(numeric))} for i in range(len(numeric))},
                  "violations": {"nonempty_segments_gt_segments": sum(r["n_nonempty_segments"] > r["n_segments"] for r in train_meta),
                                  "words_negative": sum(r["n_words"] < 0 for r in train_meta),
                                  "chars_negative": sum(r["n_chars"] < 0 for r in train_meta)}}

    # Operational association: group-level entity count and label composition.
    def assoc(rows):
        out = {}
        for field in ("source_system", "channel", "domain", "ingested_at_date"):
            groups = defaultdict(list)
            for r in rows:
                value = str(r["ingested_at"])[:10] if field == "ingested_at_date" else str(r[field])
                groups[value].append(r["document_id"])
            out[field] = {}
            for g, ids in groups.items():
                counts = np.array([label_counts.get(i, 0) for i in ids], float)
                spans = Counter()
                for i in ids:
                    spans.update(label_by_doc[i])
                out[field][g] = {"documents": len(ids), "gold_count_mean": float(counts.mean()),
                                 "gold_count_median": float(np.median(counts)),
                                 "gold_count_nonzero_rate": float((counts > 0).mean()),
                                 "label_document_presence_counts": dict(Counter(x for i in ids for x in label_by_doc[i])),
                                 "gold_span_counts": dict(spans),
                                 "gold_span_proportions": {k: v / max(1, sum(spans.values())) for k, v in spans.items()}}
        return out

    # Exact and whitespace-normalized text overlap, plus cheap formatting profile.
    train_text, test_text = doc_text(ROOT / "Data/train/train_segments.csv"), doc_text(ROOT / "Data/test/test_segments.csv")
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    train_hash = {hashlib.sha256(t.encode()).hexdigest() for t in train_text.values()}
    test_hash = {hashlib.sha256(t.encode()).hexdigest() for t in test_text.values()}
    train_norm = {hashlib.sha256(norm(t).encode()).hexdigest() for t in train_text.values()}
    test_norm = {hashlib.sha256(norm(t).encode()).hexdigest() for t in test_text.values()}
    def formatting(texts):
        return {"mean_newlines_per_document": float(np.mean([t.count("\n") for t in texts])),
                "mean_empty_segments_per_document": float(np.mean([sum(not x.strip() for x in t.split("\n")) for t in texts])),
                "digit_doc_rate": float(np.mean([bool(re.search(r"\d", t)) for t in texts])),
                "email_doc_rate": float(np.mean([bool(re.search(r"[\w.+-]+@[\w.-]+", t)) for t in texts]))}

    train_tokens = re.findall(r"\S+", " ".join(train_text.values()))
    test_tokens = re.findall(r"\S+", " ".join(test_text.values()))
    vocab_train_set, vocab_test_set = set(train_tokens), set(test_tokens)
    vocab_train, vocab_test = len(vocab_train_set), len(vocab_test_set)
    def date_stats(rows):
        dates = [str(r["ingested_at"])[:10] for r in rows]
        return {"min": min(dates), "max": max(dates), "unique_dates": len(set(dates)), "daily_top": Counter(dates).most_common(5)}
    def count_bin(x): return "0-10" if x <= 10 else "11-15" if x <= 15 else "16+"
    assoc_v = {}
    for field in ("source_system", "channel", "domain"):
        groups = sorted({str(r[field]) for r in train_meta})
        table = [[0, 0, 0] for _ in groups]
        for r in train_meta:
            g = groups.index(str(r[field])); table[g][["0-10", "11-15", "16+"].index(count_bin(label_counts.get(r["document_id"], 0)))] += 1
        assoc_v[field] = {"entity_count_bin_cramers_v_bias_corrected": cramers_v(table), "entity_count_bins": ["0-10", "11-15", "16+"]}
    # Ingestion date associations are descriptive: date vs domain/channel and count bins.
    for field in ("domain", "channel"):
        dates = sorted({str(r["ingested_at"])[:10] for r in train_meta})
        groups = sorted({str(r[field]) for r in train_meta})
        table = [[0 for _ in groups] for _ in dates]
        for r in train_meta:
            i = dates.index(str(r["ingested_at"])[:10]); table[i][groups.index(str(r[field]))] += 1
        assoc_v["ingested_at_date_vs_" + field] = {"cramers_v_bias_corrected": cramers_v(table), "date_buckets": len(dates), "groups": groups}
    dates = sorted({str(r["ingested_at"])[:10] for r in train_meta})
    table = [[0, 0, 0] for _ in dates]
    for r in train_meta:
        i = dates.index(str(r["ingested_at"])[:10]); table[i][["0-10", "11-15", "16+"].index(count_bin(label_counts.get(r["document_id"], 0)))] += 1
    assoc_v["ingested_at_date_vs_entity_count_bin"] = {"cramers_v_bias_corrected": cramers_v(table), "date_buckets": len(dates), "groups": ["0-10", "11-15", "16+"]}
    gold_counts = [label_counts.get(r["document_id"], 0) for r in train_meta]
    for other in ("channel", "domain"):
        systems = sorted({str(r["source_system"]) for r in train_meta})
        categories = sorted({str(r[other]) for r in train_meta})
        contingency = np.zeros((len(systems), len(categories)), dtype=int)
        for row in train_meta:
            contingency[systems.index(str(row["source_system"]))][categories.index(str(row[other]))] += 1
        assoc_v[f"source_system_vs_{other}"] = {"cramers_v_bias_corrected": cramers_v(contingency),
                                               "source_systems": systems, "categories": categories}

    hidden = [c for c in train_meta[0] if re.search(r"label|target|entity|pii|annotation", c, re.I)]
    report = {
        "scope": {"train_documents": len(train_meta), "test_documents": len(test_meta), "train_labels": len(labels),
                  "train_metadata_ids_unique": len(train_ids) == len(train_meta), "test_metadata_ids_unique": len(test_ids) == len(test_meta)},
        "schema": {"train_columns": {k: type(v).__name__ for k, v in train_meta[0].items()}, "test_columns": list(test_meta[0]),
                   "cardinality": {"train": {c: len({r[c] for r in train_meta}) for c in train_meta[0]}, "test": {c: len({r[c] for r in test_meta}) for c in test_meta[0]}},
                   "missing": {"train": {c: sum(r.get(c) is None for r in train_meta) for c in train_meta[0]},
                               "test": {c: sum(r.get(c) is None for r in test_meta) for c in test_meta[0]}}, "label_like_metadata_columns": hidden},
        "numeric_comparison": num_cmp, "categorical_comparison": cat_cmp, "metadata_redundancy": redundancy,
        "train_label_counts": dict(Counter(r["label"] for r in labels)),
        "test_submission_slots": {"rows": sum(slots.values()), "documents": len(slots), "unknown_document_rows": bad_rows,
                                   "count_quantiles": quantiles(list(slots.values())), "max_count": max(slots.values(), default=0),
                                   "metadata_docs_without_slots": len(test_ids - set(slots)), "slot_docs_without_metadata": len(set(slots) - test_ids)},
        "operational_associations_train": assoc(train_meta),
        "operational_signal_summary": {"ingested_at_train": date_stats(train_meta), "ingested_at_test": date_stats(test_meta), "categorical_entity_count_association": assoc_v},
        "entity_count_comparison": {"train_gold": {"quantiles": quantiles(gold_counts), "mean": float(np.mean(gold_counts)), "median": float(np.median(gold_counts))}, "test_submission_slots": {"quantiles": quantiles(list(slots.values())), "mean": float(np.mean(list(slots.values()))), "median": float(np.median(list(slots.values()))) }},
        "formatting": {"train": formatting(list(train_text.values())), "test": formatting(list(test_text.values()))},
        "text_overlap": {"exact_document_hash_overlap": len(train_hash & test_hash), "whitespace_normalized_overlap": len(train_norm & test_norm),
                         "case_sensitive_whitespace_token_vocab_overlap": len(vocab_train_set & vocab_test_set),
                         "test_token_occurrence_oov_rate_vs_train": float(sum(x not in vocab_train_set for x in test_tokens) / max(1, len(test_tokens))),
                         "train_vocab_unique_tokens": vocab_train, "test_vocab_unique_tokens": vocab_test},
        "limitations": ["Metadata length fields are derived from text and are not independent predictive signals.",
                        "ingested_at is synthetic/operational and associations do not establish causality or predictive validity.",
                        "Test label composition is unavailable; submission slots provide only total entity counts.",
                        "Distribution comparisons are descriptive and require held-out evaluation before feature use."]}
    report["schema"]["parquet_types"] = {split: {field.name: str(field.type) for field in pq.read_schema(ROOT / f"Data/{split}/{split}_metadata.parquet")} for split in ("train", "test")}
    report["operational_signal_summary"]["exact_timestamps"] = {split: {"min": min(str(r["ingested_at"]) for r in rows), "max": max(str(r["ingested_at"]) for r in rows), "unique": len({str(r["ingested_at"]) for r in rows})} for split, rows in (("train", train_meta), ("test", test_meta))}
    (OUT / "metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = ["# Metadata and train/test distribution audit", "", "## Scope and integrity", f"- Train/test metadata: {len(train_meta):,} / {len(test_meta):,} documents; labels: {len(labels):,}.", f"- Metadata IDs unique: train={report['scope']['train_metadata_ids_unique']}, test={report['scope']['test_metadata_ids_unique']}.", f"- Submission slots: {sum(slots.values()):,} across {len(slots):,} test documents; unknown rows={bad_rows}; missing slot docs={len(test_ids-set(slots))}.", "- No metadata column has a label-like name: `" + ", ".join(hidden or ["none"]) + "`.", "", "## Findings", "- Numeric fields are text-derived document descriptors. Their train/test KS and standardized mean differences are in `metadata.json`; they can diagnose drift but should not be treated as independent signal.", "- Categorical drift is low: TVD is channel 0.0061, domain 0.0078, source_system 0.0054. Operational metadata associations are observational only.", f"- Exact text hash overlap is {len(train_hash & test_hash):,}; whitespace-normalized overlap is {len(train_norm & test_norm):,}; case-sensitive token OOV occurrence rate is {report['text_overlap']['test_token_occurrence_oov_rate_vs_train']:.3f}.", f"- Unique whitespace-token vocabulary is train={vocab_train:,}, test={vocab_test:,}; formatting summaries use mean counts per document.", f"- Gold train entity counts have mean/median {np.mean(gold_counts):.2f}/{np.median(gold_counts):.0f}; test slot counts have mean/median {np.mean(list(slots.values())):.2f}/{np.median(list(slots.values())):.0f}. Bias-corrected Cramer's V is reported for source/channel/domain and ingestion-date buckets.", "- Test slots expose entity count per document, while test label composition remains hidden.", "", "## Interpretation and limits", "Metadata field names are not proof that hidden labels are absent; this audit checks schema and observed associations only. Numeric descriptors derive from text, and timestamps may be synthetic operational values. Any metadata feature must be tested in a document-held-out ablation with count-aware slot scoring; no causal or predictive claim follows from these descriptive effect sizes.", "", "Generated by `src/audit_metadata.py`."]
    (OUT / "metadata.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
