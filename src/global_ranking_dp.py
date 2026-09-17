"""Exact top-two labeled interval structures with compact backpointers.

The tie rule is deterministic: equal scores prefer a skip transition, then
intervals ordered by (start, end, input index), then labels ordered by
descending energy and ascending label, then predecessor rank. This differs
from lexical path ordering used by the older audit helper; non-tied scores are
unaffected.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def best_two(
    n: int,
    pairs: np.ndarray,
    energies: np.ndarray,
    k: int,
) -> list[tuple[float, tuple[tuple[int, int, int], ...]]]:
    """Return the two highest-scoring distinct exact-k labeled structures.

    Structures are sorted, non-overlapping sequences of half-open spans. Each
    span receives one label from the energy row associated with its interval.
    Returns one result when only one feasible structure exists, and an empty
    list when no exact-k structure exists.
    """
    if not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    if not isinstance(k, (int, np.integer)) or k < 0:
        raise ValueError("k must be a nonnegative integer")
    p = np.asarray(pairs, dtype=np.int64)
    e = np.asarray(energies, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or e.ndim != 2 or e.shape[0] != len(p):
        raise ValueError("pairs must be Mx2 and energies must be MxL")
    if len({(int(a), int(b)) for a, b in p}) != len(p):
        raise ValueError("pairs must not contain duplicate intervals")
    if e.shape[1] == 0 or not np.isfinite(e).all():
        raise ValueError("energies must have at least one label and be finite")
    if np.any(p[:, 0] < 0) or np.any(p[:, 0] >= p[:, 1]) or np.any(p[:, 1] > n):
        raise ValueError("pairs must be valid nonempty half-open spans within n")
    if k == 0:
        return [(0.0, ())]
    if k > n or len(p) == 0:
        return []

    by_end: list[list[int]] = [[] for _ in range(n + 1)]
    for i, (_, end) in enumerate(p):
        by_end[int(end)].append(i)
    for bucket in by_end:
        bucket.sort(key=lambda i: (int(p[i, 0]), int(p[i, 1]), i))

    # Scores hold the two best unique paths for each (prefix, entity count).
    scores = np.full((n + 1, k + 1, 2), -math.inf, dtype=np.float64)
    scores[:, 0, 0] = 0.0
    # A choice is either -1 (skip one token) or the selected interval index.
    choice = np.full((n + 1, k + 1, 2), -2, dtype=np.int32)
    choice_label = np.full((n + 1, k + 1, 2), -1, dtype=np.int16)
    prev_rank = np.full((n + 1, k + 1, 2), -1, dtype=np.int8)
    choice[:, 0, 0] = -1

    # Only the two strongest labels on an interval can occur in the global top
    # two: every weaker label has the same predecessor structure and a score
    # no greater than these two alternatives.
    best_labels = np.argsort(-e, axis=1, kind="stable")[:, : min(2, e.shape[1])]

    for end in range(1, n + 1):
        qmax = min(k, end)
        bucket = np.asarray(by_end[end], dtype=np.int64)
        skip_rows = scores[end - 1, 1 : qmax + 1, :].T
        if bucket.size:
            labels = best_labels[bucket]
            starts = p[bucket, 0]
            predecessor = scores[starts, :qmax, :]  # M x Q x rank
            label_energy = e[bucket[:, None], labels]  # M x strongest-label
            # Keep interval, label, predecessor-rank order, while computing
            # all their energy vectors with a single NumPy gather/broadcast.
            selected = (predecessor[:, :, None, :] + label_energy[:, None, :, None])
            selected = selected.transpose(0, 2, 3, 1).reshape(-1, qmax)
            candidate_scores = np.concatenate((skip_rows, selected), axis=0)
            label_count = labels.shape[1]
            option_i = np.concatenate((np.full(2, -1, dtype=np.int32), np.repeat(bucket, label_count * 2)))
            option_label = np.concatenate((np.full(2, -1, dtype=np.int16),
                                           np.broadcast_to(labels[:, :, None], (len(bucket), label_count, 2)).reshape(-1)))
            option_rank = np.concatenate((np.arange(2, dtype=np.int8),
                                          np.tile(np.arange(2, dtype=np.int8), len(bucket) * label_count)))
        else:
            candidate_scores = skip_rows
            option_i = np.full(2, -1, dtype=np.int32)
            option_label = np.full(2, -1, dtype=np.int16)
            option_rank = np.arange(2, dtype=np.int8)
        # argmax returns the first maximum, so masking each winner gives the
        # top two with stable row-order tie breaking at a fraction of the cost
        # of sorting every candidate column.
        winner0 = np.argmax(candidate_scores, axis=0)
        remaining = candidate_scores.copy()
        remaining[winner0, np.arange(qmax)] = -math.inf
        winner1 = np.argmax(remaining, axis=0)
        winners = (winner0, winner1)
        for out_rank in range(2):
            selected = winners[out_rank]
            q_indices = np.arange(1, qmax + 1)
            scores[end, q_indices, out_rank] = candidate_scores[selected, np.arange(qmax)] if out_rank == 0 else remaining[selected, np.arange(qmax)]
            choice[end, q_indices, out_rank] = option_i[selected]
            choice_label[end, q_indices, out_rank] = option_label[selected]
            prev_rank[end, q_indices, out_rank] = option_rank[selected]

    out: list[tuple[float, tuple[tuple[int, int, int], ...]]] = []
    for rank in range(2):
        score = float(scores[n, k, rank])
        if not math.isfinite(score):
            continue
        spans: list[tuple[int, int, int]] = []
        end, q, r = n, k, rank
        while q:
            i = int(choice[end, q, r])
            r0 = int(prev_rank[end, q, r])
            if i == -1:
                end -= 1
                r = r0
                continue
            if i < 0:
                raise RuntimeError("broken top-two DP backpointer")
            a, b = map(int, p[i])
            spans.append((a, b, int(choice_label[end, q, r])))
            end, q, r = a, q - 1, r0
        spans.reverse()
        out.append((score, tuple(spans)))
    return out
