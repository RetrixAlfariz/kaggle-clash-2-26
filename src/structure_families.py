"""Deterministic train-only duplicate/template-family grouping and sampling."""

from __future__ import annotations

import hashlib
import re
import statistics
from collections import defaultdict
from typing import Any, Iterable

from audit_template_similarity import add_index, candidates, jaccard, shingles as shingle, signature


SEED = 20260916
MIN_SAMPLE_SIZE = 6000
MAX_SAMPLE_SIZE = 12000
MIN_FAMILIES_PER_FOLD = 30
FOLD_COUNT = 3
JACCARD_THRESHOLD = 0.8


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _masked_norm(text: str) -> str:
    # Preserve the number of digits only as a single placeholder per run.
    return re.sub(r"\d+", "0", _norm(text))


def _order_hash(seed: int, document_id: str) -> bytes:
    return hashlib.sha256(f"{seed}:{document_id}".encode("utf-8")).digest()


def group_and_sample(
    rows: Iterable[dict[str, Any]],
    sample_size: int = MIN_SAMPLE_SIZE,
    max_sample_size: int = MAX_SAMPLE_SIZE,
    min_families: int = MIN_FAMILIES_PER_FOLD,
    seed: int = SEED,
) -> tuple[list[int], list[str], list[int], dict[str, Any]]:
    """Group train documents, choose complete families, and assign 3 folds.

    ``rows`` must contain only prepared-train ``document_id`` and ``full_text``
    values. The function deliberately does not inspect annotation fields.
    Returned original row indices follow deterministic family-sampling order,
    with members of each selected family ordered by their original row index.
    """
    records = list(rows)
    if sample_size <= 0 or max_sample_size < sample_size or min_families < 1:
        raise ValueError("invalid sample/family bounds")
    ids: list[str] = []
    texts: list[str] = []
    for i, row in enumerate(records):
        if "document_id" not in row or "full_text" not in row:
            raise ValueError(f"row {i} must contain document_id and full_text")
        doc_id = str(row["document_id"])
        text = row["full_text"]
        if not doc_id or not isinstance(text, str):
            raise ValueError(f"row {i} has invalid document_id/full_text")
        ids.append(doc_id)
        texts.append(text)
    if len(set(ids)) != len(ids):
        raise ValueError("document_id values must be unique")

    uf = _UnionFind(len(records))
    exact_index: dict[str, int] = {}
    exact_duplicate_edges = 0
    for i, text in enumerate(texts):
        key = _norm(text)
        if key in exact_index:
            uf.union(i, exact_index[key])
            exact_duplicate_edges += 1
        else:
            exact_index[key] = i

    # Retrieve approximate candidate pairs using the existing fixed 8-way
    # MinHash signature and 4 bands of 2 hashes. Index one representative per
    # distinct digit-masked text: members of the same bucket have identical
    # shingle sets, so comparing every repeated document pair is redundant.
    band_index: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    masked_representatives: dict[str, int] = {}
    representative_shingles: dict[int, set[str]] = {}
    candidate_edges = 0
    verified_edges = 0
    approximate_unions = 0
    masked_identical_links = 0
    for i, text in enumerate(texts):
        masked_text = _masked_norm(text)
        if masked_text in masked_representatives:
            representative = masked_representatives[masked_text]
            uf.union(i, representative)
            masked_identical_links += 1
            continue
        values = shingle(masked_text)
        shingles_i = set(values)
        sig = signature(values)
        masked_representatives[masked_text] = i
        representative_shingles[i] = shingles_i
        retrieved = candidates(sig, band_index)
        candidate_edges += len(retrieved)
        for j in sorted(retrieved):
            if jaccard(shingles_i, representative_shingles[j]) >= JACCARD_THRESHOLD:
                verified_edges += 1
                if uf.union(i, j):
                    approximate_unions += 1
        add_index(band_index, sig, i)

    components_by_root: dict[int, list[int]] = defaultdict(list)
    for i in range(len(records)):
        components_by_root[uf.find(i)].append(i)
    components: list[tuple[str, list[int]]] = []
    for members in components_by_root.values():
        members.sort()
        group_id = min(ids[i] for i in members)
        components.append((group_id, members))

    # Stable family order is determined solely by seed and minimum document ID.
    components.sort(key=lambda item: (_order_hash(seed, item[0]), item[0]))
    selected_components: list[tuple[str, list[int]]] = []
    selected_count = 0
    for component in components:
        if selected_count >= sample_size:
            break
        selected_components.append(component)
        selected_count += len(component[1])
    if selected_count > max_sample_size:
        raise ValueError(
            f"complete-family sample has {selected_count} documents, exceeding max {max_sample_size}"
        )

    if len(selected_components) < FOLD_COUNT * min_families:
        raise ValueError(
            f"only {len(selected_components)} sampled families; need at least "
            f"{FOLD_COUNT * min_families} for {FOLD_COUNT} folds"
        )

    # Greedy family-level fold allocation, largest family first. Stable seeded
    # hash breaks size ties; equal fold counts go to the lowest fold index.
    allocation_order = sorted(
        selected_components,
        key=lambda item: (-len(item[1]), _order_hash(seed, item[0]), item[0]),
    )
    fold_doc_counts = [0] * FOLD_COUNT
    fold_family_counts = [0] * FOLD_COUNT
    component_fold: dict[str, int] = {}
    for group_id, members in allocation_order:
        fold = min(range(FOLD_COUNT), key=lambda f: (fold_doc_counts[f], f))
        component_fold[group_id] = fold
        fold_doc_counts[fold] += len(members)
        fold_family_counts[fold] += 1
    if min(fold_family_counts) < min_families:
        raise ValueError(
            f"balanced assignment leaves fold family counts {fold_family_counts}; "
            f"each fold needs at least {min_families}"
        )

    selected_indices: list[int] = []
    group_ids: list[str] = []
    fold_ids: list[int] = []
    for group_id, members in selected_components:
        fold = component_fold[group_id]
        for original_index in members:
            selected_indices.append(original_index)
            group_ids.append(group_id)
            fold_ids.append(fold)

    all_family_sizes = sorted((len(members) for _, members in components), reverse=True)
    selected_memberships = {
        group_id: [ids[i] for i in members]
        for group_id, members in selected_components
    }
    report = {
        "seed": int(seed),
        "policy": {
            "normalization": "lowercase, collapse whitespace, strip",
            "numeric_mask": "replace each consecutive digit run with 0",
            "shingles": "5-word shingles",
            "minhash_permutations": 8,
            "bands": 4,
            "rows_per_band": 2,
            "verified_jaccard_threshold": JACCARD_THRESHOLD,
            "sample_minimum_documents": int(sample_size),
            "sample_maximum_documents": int(max_sample_size),
            "fold_count": FOLD_COUNT,
            "minimum_families_per_fold": int(min_families),
        },
        "input_documents": len(records),
        "exact_duplicate_links": exact_duplicate_edges,
        "digit_mask_identical_links": masked_identical_links,
        "retrieved_candidate_edges": candidate_edges,
        "verified_candidate_edges": verified_edges,
        "approximate_union_edges": approximate_unions,
        "family_count": len(components),
        "family_size_summary": {
            "minimum": min(all_family_sizes, default=0),
            "median": float(statistics.median(all_family_sizes)) if all_family_sizes else 0.0,
            "maximum": max(all_family_sizes, default=0),
            "sizes_descending": all_family_sizes,
        },
        "sampled_documents": len(selected_indices),
        "sampled_family_count": len(selected_components),
        "sampled_families": selected_memberships,
        "fold_document_counts": fold_doc_counts,
        "fold_family_counts": fold_family_counts,
        "fold_family_memberships": {
            str(fold): [group_id for group_id, _ in selected_components if component_fold[group_id] == fold]
            for fold in range(FOLD_COUNT)
        },
    }
    return selected_indices, group_ids, fold_ids, report
