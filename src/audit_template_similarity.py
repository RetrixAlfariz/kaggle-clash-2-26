# /// script
# requires-python = "==3.13.2"
# dependencies = ["pyarrow==25.0.1"]
# ///
"""Label-independent cross-split shingle audit; holdout outputs are aggregate only."""
from __future__ import annotations
import collections
import hashlib
import heapq
import json
import platform
from pathlib import Path
from functools import lru_cache
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/template_similarity"
K = 5
THRESHOLDS = (.8, .9)
BANDS = 4
MASK = (1 << 64) - 1
# Fixed deterministic affine permutations, set before any similarity scoring.
PERMUTATIONS = [(int.from_bytes(hashlib.sha256(f"a-{i}".encode()).digest()[:8], "little") | 1,
                 int.from_bytes(hashlib.sha256(f"b-{i}".encode()).digest()[:8], "little")) for i in range(8)]


def shingles(text):
    words = text.lower().split()
    return {" ".join(words[i:i + K]) for i in range(len(words) - K + 1)}


def hashed_shingles(text):
    return {int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little") for s in shingles(text)}


def signature(values):
    # Accept raw shingles in small fixtures, hashed shingles in the full run.
    values = {int.from_bytes(hashlib.blake2b(v.encode(), digest_size=8).digest(), "little") if isinstance(v, str) else v for v in values}
    if not values:
        return None
    return tuple(min((a * v + b) & MASK for v in values) for a, b in PERMUTATIONS)


def jaccard(a, b):
    return len(a & b) / len(a | b) if a or b else 0.0


def candidates(sig, index):
    if sig is None:
        return set()
    out = set()
    for band in range(BANDS):
        out.update(index.get((band, sig[band * 2:band * 2 + 2]), ()))
    return out


def add_index(index, sig, doc):
    if sig is not None:
        for band in range(BANDS):
            index.setdefault((band, sig[band * 2:band * 2 + 2]), []).append(doc)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(part):
    return {r["document_id"]: r["full_text"] for r in pq.read_table(ROOT / "output/prepared/v1" / f"{part}.parquet", columns=["document_id", "full_text"]).to_pylist()}


def families(pairs, threshold):
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for left, right, score in pairs:
        if score >= threshold:
            parent[find(left)] = find(right)
    sizes = collections.Counter(find(x) for x in list(parent))
    return {"connected_components": len(sizes), "documents": len(parent), "largest_component_documents": max(sizes.values(), default=0)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [ROOT / "output/prepared/v1" / f"{p}.parquet" for p in ("train", "dev", "holdout")]
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    train = load("train")
    train_sh = {}
    index = {}
    for n, (d, text) in enumerate(train.items(), 1):
        train_sh[d] = hashed_shingles(text)
        add_index(index, signature(train_sh[d]), d)
        if n % 10000 == 0:
            print(f"Indexed train documents: {n}", flush=True)
    @lru_cache(maxsize=512)
    def exact_train(d):
        return shingles(train[d])
    results = {}
    dev_pairs = []
    strongest_dev_pairs = []
    dev_text = None
    for part in ("dev", "holdout"):
        documents = load(part)
        if part == "dev":
            dev_text = documents
        pair_counts = collections.Counter()
        document_counts = collections.Counter()
        score_bins = collections.Counter()
        best_scores = []
        candidate_count = zero_candidates = short_docs = 0
        all_ref_docs = set()
        for d, text in documents.items():
            raw = shingles(text)
            hs = {int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little") for s in raw}
            short_docs += not bool(hs)
            refs = candidates(signature(hs), index)
            candidate_count += len(refs)
            zero_candidates += not bool(refs)
            best = 0.0
            best_ref = None
            hit_thresholds = set()
            for ref in refs:
                score = jaccard(hs, train_sh[ref])
                if score >= THRESHOLDS[0]:
                    score = jaccard(raw, exact_train(ref))
                if score > best or (score == best and (best_ref is None or ref < best_ref)):
                    best, best_ref = score, ref
                score_bins["0.9-1.0" if score >= .9 else "0.8-0.9" if score >= .8 else "0.5-0.8" if score >= .5 else "0-0.5"] += 1
                for threshold in THRESHOLDS:
                    if score >= threshold:
                        pair_counts[str(threshold)] += 1
                        hit_thresholds.add(str(threshold))
                if score >= THRESHOLDS[0]:
                    all_ref_docs.add(ref)
                    if part == "dev":
                        dev_pairs.append((d, ref, score))
            document_counts.update(hit_thresholds)
            best_scores.append(best)
            if part == "dev" and best_ref is not None:
                heapq.heappush(strongest_dev_pairs, (best, d, best_ref))
                if len(strongest_dev_pairs) > 8:
                    heapq.heappop(strongest_dev_pairs)
        ordered = sorted(best_scores)
        results[part + "_to_train"] = {
            "query_documents": len(documents), "short_documents_excluded": short_docs,
            "unique_candidate_pairs_scored": candidate_count,
            "queries_without_candidates": zero_candidates,
            "threshold_pair_counts": {str(t): pair_counts[str(t)] for t in THRESHOLDS},
            "threshold_query_document_counts": {str(t): document_counts[str(t)] for t in THRESHOLDS},
            "reference_documents_in_verified_pairs_at_0.8": len(all_ref_docs),
            "candidate_score_bins": dict(score_bins),
            "best_retrieved_score": {"mean": sum(ordered) / len(ordered), "p50": ordered[len(ordered) // 2], "p95": ordered[int(len(ordered) * .95)], "max": max(ordered)}}
        print(f"Completed {part}: {len(documents)} queries; {candidate_count} candidate pairs", flush=True)
    dev_pairs.sort(key=lambda p: (-p[2], p[0], p[1]))
    examples = []
    for score, d, ref in sorted(strongest_dev_pairs, reverse=True):
        # Only train/dev excerpts; no holdout identifiers or text are persisted.
        exact_score = jaccard(shingles(dev_text[d]), shingles(train[ref]))
        examples.append({"dev_id": d, "train_id": ref, "retrieved_jaccard": score,
                         "exact_string_jaccard": exact_score, "meets_0.8": exact_score >= .8,
                         "dev_excerpt": dev_text[d][:1000], "train_excerpt": train[ref][:1000]})
    if any(sha(ROOT / path) != expected for path, expected in source_hashes.items()):
        raise ValueError("Source changed during similarity audit")
    report = {
        "method": {"normalization": "lowercase and whitespace word splitting; original files unchanged", "shingle_words": K,
                   "thresholds_fixed_before_scoring": list(THRESHOLDS), "signature": "8 deterministic affine minhash permutations of 64-bit BLAKE2b shingle hashes", "bands": BANDS, "rows_per_band": 2,
                   "verification": "64-bit hashed-shingle Jaccard on retrieved candidates; every pair reaching 0.8 rechecked with exact string-shingle Jaccard",
                   "pair_scope": "all dev versus train and all holdout versus train; no query or candidate caps; within-split pairs not searched"},
        "runtime": {"python": platform.python_version(), "pyarrow": pa.__version__},
        "source_hashes": source_hashes, "audit_code_sha256": sha(Path(__file__)),
        "train_documents": len(train), "train_short_documents_excluded": sum(not x for x in train_sh.values()),
        "comparisons": results,
        "observed_train_dev_families": {str(t): families(dev_pairs, t) for t in THRESHOLDS},
        "examples_train_dev": examples,
        "limitations": ["Approximate retrieval: verified pair counts are lower bounds, not exhaustive overlap counts.",
                        "Best retrieved scores are not guaranteed true nearest-neighbor scores; zero can mean no candidate retrieved.",
                        "Family counts are connected components of discovered CROSS-SPLIT edges only; they exclude unseen/within-split edges and are not generator-family ground truth.",
                        "Below-threshold candidate scores use 64-bit hashes; hash collisions are possible. Reported positive pairs use exact string shingles.",
                        "No gold labels, entity masking, holdout IDs/examples, test documents or model scores used in output.",
                        "No structural-line or semantic similarity audit performed; substantially rewritten templates can evade word shingles."]}
    (OUT / "template_similarity_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "train_dev_pairs.json").write_text(json.dumps(dev_pairs), encoding="utf-8")
    print(json.dumps({"comparisons": results, "families": report["observed_train_dev_families"]}, indent=2))


if __name__ == "__main__":
    main()
