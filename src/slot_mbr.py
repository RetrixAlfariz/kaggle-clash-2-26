"""Exact fixed-K interval MAP and Slot-MBR decoding for frozen span scores.

``energies`` contains one direct energy per positive entity label.  The
partition sums labels for each interval; the final MBR decision keeps the
highest marginal label for each selected interval.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _lse(values: list[float]) -> float:
    if not values:
        return -math.inf
    m = max(values)
    if not math.isfinite(m):
        return m
    return m + math.log(sum(math.exp(x - m) for x in values))


def _validate(pairs: np.ndarray, energies: np.ndarray, k: int, n: int) -> tuple[np.ndarray, np.ndarray, int]:
    p = np.asarray(pairs)
    e = np.asarray(energies, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or e.ndim != 2 or e.shape[0] != p.shape[0]:
        raise ValueError("pairs must be Mx2 and energies must be MxL")
    if e.shape[1] < 1 or not isinstance(k, (int, np.integer)) or k < 0:
        raise ValueError("invalid energies or K")
    if p.shape[0] and not np.issubdtype(p.dtype, np.integer):
        if not np.all(np.isfinite(p)) or not np.all(p == p.astype(np.int64)):
            raise ValueError("pairs must contain integer boundaries")
    p = p.astype(np.int64, copy=False)
    if p.size and (np.any(p[:, 0] < 0) or np.any(p[:, 1] <= p[:, 0]) or np.any(p[:, 1] - p[:, 0] > 16)):
        raise ValueError("invalid span interval")
    if len(p) != len({tuple(x) for x in p.tolist()}):
        raise ValueError("duplicate interval")
    if not np.all(np.isfinite(e)):
        raise ValueError("energies must be finite")
    t = int(p[:, 1].max()) if len(p) else 0
    if t > n:
        raise ValueError("interval end exceeds n")
    if k > t:
        raise ValueError("no feasible exactly-K interval selection")
    return p, e, t


def _backtrack(decision: np.ndarray, pairs: np.ndarray, k: int, t: int) -> list[int]:
    out: list[int] = []
    x, q = t, k
    while q:
        i = int(decision[x, q])
        if i < 0:
            x -= 1
        else:
            out.append(i)
            x = int(pairs[i, 0])
            q -= 1
    return out[::-1]


def decode(n: int, pairs: np.ndarray, energies: np.ndarray, K: int, return_marginals: bool = False) -> tuple[list[tuple[int, int, int]], dict[str, Any]]:
    """Decode a complete interval universe using exact Slot-MBR.

    ``n`` is the token count.  ``pairs`` are half-open ``[start,end)``
    intervals and ``energies`` has one column per entity label.  The returned
    triples use zero-based label indices.  Ties prefer skipping, then the
    smaller start boundary.
    """
    if not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    p, e, tmax = _validate(pairs, energies, K, int(n))
    t = int(n)
    if K == 0:
        return [], {"method": "slot_mbr", "K": 0, "logZ": 0.0, "partition_residual": 0.0,
                    "max_slot_norm_residual": 0.0, "workspace_bytes": 0,
                    "map_expected_utility": 0.0, "mbr_expected_utility": 0.0}
    m, labels = len(p), e.shape[1]
    by_end = [[] for _ in range(t + 1)]
    for i, (a, b) in enumerate(p):
        by_end[int(b)].append(i)
    for bucket in by_end:
        bucket.sort(key=lambda i: (int(p[i, 0]), int(p[i, 1]), i))

    # Grouped-label forward/backward log partition.
    interval_lse = np.array([_lse(e[i].tolist()) for i in range(m)])
    f = np.full((t + 1, K + 1), -math.inf)
    f[:, 0] = 0.0
    for end in range(1, t + 1):
        f[end, 1:] = f[end - 1, 1:]
        ids = by_end[end]
        if ids:
            starts = p[ids, 0]
            terms = f[starts, :-1] + interval_lse[ids, None]
            f[end, 1:] = np.logaddexp(f[end, 1:], np.logaddexp.reduce(terms, axis=0))
    logz = float(f[t, K])
    if not math.isfinite(logz):
        raise ValueError("no feasible exactly-K interval selection")
    bwd = np.full((t + 1, K + 1), -math.inf)
    bwd[:, 0] = 0.0
    by_start = [[] for _ in range(t + 1)]
    for i, (a, _b) in enumerate(p):
        by_start[int(a)].append(i)
    for bucket in by_start:
        bucket.sort(key=lambda i: (int(p[i, 1]), int(p[i, 0]), i))
    for start in range(t - 1, -1, -1):
        bwd[start, 1:] = bwd[start + 1, 1:]
        ids = by_start[start]
        if ids:
            ends = p[ids, 1]
            terms = interval_lse[ids, None] + bwd[ends, :-1]
            bwd[start, 1:] = np.logaddexp(bwd[start, 1:], np.logaddexp.reduce(terms, axis=0))
    partition_residual = abs(logz - float(bwd[0, K]))
    if not math.isclose(logz, float(bwd[0, K]), rel_tol=0.0, abs_tol=1e-10):
        raise RuntimeError("forward/backward partition mismatch")

    # Slot-specific labelled marginals and max-label rewards.
    base = f[p[:,0], :K] + bwd[p[:,1], K-1::-1] - logz
    mu = np.exp(base.T[:,:,None] + e[None,:,:])
    max_slot_norm_residual = float(np.max(np.abs(mu.sum(axis=(1, 2)) - 1.0)))
    if max_slot_norm_residual > 2e-10:
        raise RuntimeError("slot marginals do not sum to one")
    reward = mu.max(axis=2)
    # Labels are independent of slot; preserve energy ordering even if a
    # numerically negligible marginal underflows to zero for every label.
    reward_label = np.argmax(e,axis=1)

    # Reward DP indexed by token prefix.  A strict comparison preserves skip;
    # candidate buckets are start-sorted, so equal take values prefer smaller start.
    r = np.full((t + 1, K + 1), -math.inf)
    r[:, 0] = 0.0
    take = np.full((t + 1, K + 1), -1, dtype=np.int64)
    # Keep the grouped MBR decision vectorized over all slots.  A strict
    # comparison preserves the documented skip-on-exact-tie rule.
    for end in range(1, t + 1):
        r[end, 1:] = r[end - 1, 1:]
        ids = by_end[end]
        if ids:
            starts = p[ids, 0]
            vals = r[starts, :-1] + reward[:, ids].T
            best = np.argmax(vals, axis=0)
            bestvals = vals[best, np.arange(K)]
            mask = bestvals > r[end, 1:]
            r[end, 1:][mask] = bestvals[mask]
            take[end, 1:][mask] = np.asarray(ids, dtype=np.int64)[best[mask]]
    if not math.isfinite(float(r[t, K])):
        raise ValueError("no feasible exactly-K MBR selection")
    selected = _backtrack(take, p, K, t)
    result = [(int(p[i, 0]), int(p[i, 1]), int(reward_label[i])) for i in selected]
    # A parallel grouped MAP DP is retained for the required exact replay and
    # for the MBR-vs-MAP expected-risk invariant.
    map_score = np.full((t + 1, K + 1), -math.inf)
    map_score[:, 0] = 0.0
    map_take = np.full((t + 1, K + 1), -1, dtype=np.int64)
    map_label = np.argmax(e, axis=1)
    map_local = e[np.arange(m),map_label]
    for end in range(1, t + 1):
        map_score[end, 1:] = map_score[end - 1, 1:]
        ids = by_end[end]
        if ids:
            vals = map_score[p[ids, 0], :-1] + map_local[ids, None]
            best = np.argmax(vals, axis=0)
            bestvals = vals[best, np.arange(K)]
            mask = bestvals > map_score[end, 1:]
            map_score[end, 1:][mask] = bestvals[mask]
            map_take[end, 1:][mask] = np.asarray(ids, dtype=np.int64)[best[mask]]
    map_ids = _backtrack(map_take, p, K, t)
    map_result = [(int(p[i, 0]), int(p[i, 1]), int(map_label[i])) for i in map_ids]
    map_expected = float(sum(mu[q, i, map_label[i]] for q, i in enumerate(map_ids)))
    mbr_expected = float(r[t, K])
    if mbr_expected + 1e-10 < map_expected:
        raise RuntimeError("MBR expected utility is below MAP")
    diagnostics: dict[str, Any] = {
        "method": "slot_mbr", "K": int(K), "logZ": logz,
        "partition_residual": float(partition_residual),
        "max_slot_norm_residual": max_slot_norm_residual,
        "workspace_bytes": int(sum(x.nbytes for x in (f,bwd,mu,r,map_score,take,map_take,base,reward,interval_lse))),
        "map_expected_utility": map_expected, "mbr_expected_utility": mbr_expected,
        "expected_utility": mbr_expected,
        "map_prediction": map_result,
        "map_score": float(map_score[t,K]),
    }
    if return_marginals:
        diagnostics["slot_marginals"] = mu.tolist()
        diagnostics["interval_log_partition"] = interval_lse.tolist()
    return result, diagnostics
