"""Exact fixed-cardinality posterior statistics for labeled intervals.

This module is diagnostic-only. Each structure contains exactly ``K``
nonoverlapping intervals and one of ``L`` labels per interval. Energies are
divided by ``tau`` before normalization.
"""

from __future__ import annotations

import math

import numpy as np


def fixed_k_posterior(
    n: int,
    pairs: np.ndarray,
    raw_energies: np.ndarray,
    K: int,
    tau: float,
) -> tuple[float, float, float, np.ndarray]:
    """Return ``(logZ, entropy, log_count, mu)`` for an exact-K posterior.

    ``pairs`` is ``M x 2`` half-open token intervals; ``raw_energies`` is
    ``M x L`` and excludes the NONE label. ``mu[q,i,l]`` is the probability
    that interval ``i`` with label ``l`` occupies slot ``q`` (chronological,
    zero-based). ``log_count`` counts labeled structures and is independent
    of energies and temperature. Empty-K has one empty structure and an empty
    ``(0,M,L)`` marginal array.
    """
    if not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    if not isinstance(K, (int, np.integer)) or K < 0:
        raise ValueError("K must be a nonnegative integer")
    if not math.isfinite(float(tau)) or tau <= 0:
        raise ValueError("tau must be finite and positive")

    p = np.asarray(pairs)
    e = np.asarray(raw_energies, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or e.ndim != 2 or e.shape[0] != len(p) or e.shape[1] < 1:
        raise ValueError("pairs must be Mx2 and energies must be MxL with L >= 1")
    if p.size and not np.issubdtype(p.dtype, np.integer):
        if not np.all(np.isfinite(p)) or not np.all(p == p.astype(np.int64)):
            raise ValueError("pairs must contain integer boundaries")
    p = p.astype(np.int64, copy=False)
    if p.size and (np.any(p[:, 0] < 0) or np.any(p[:, 1] <= p[:, 0]) or np.any(p[:, 1] > n)):
        raise ValueError("invalid interval boundaries")
    if len(p) != len({(int(a), int(b)) for a, b in p}):
        raise ValueError("duplicate interval")
    if not np.all(np.isfinite(e)):
        raise ValueError("energies must be finite")
    m, labels = e.shape
    if K == 0:
        return 0.0, 0.0, 0.0, np.empty((0, m, labels), dtype=np.float64)

    scaled = e / float(tau)
    # Stable interval label log-sum-exp.
    rowmax = scaled.max(axis=1)
    interval_lse = rowmax + np.log(np.exp(scaled - rowmax[:, None]).sum(axis=1))
    by_end: list[list[int]] = [[] for _ in range(n + 1)]
    by_start: list[list[int]] = [[] for _ in range(n + 1)]
    for i, (a, b) in enumerate(p):
        by_end[int(b)].append(i)
        by_start[int(a)].append(i)
    for bucket in by_end:
        bucket.sort(key=lambda i: (int(p[i, 0]), int(p[i, 1]), i))
    for bucket in by_start:
        bucket.sort(key=lambda i: (int(p[i, 1]), int(p[i, 0]), i))

    f = np.full((n + 1, K + 1), -np.inf, dtype=np.float64)
    f[:, 0] = 0.0
    for end in range(1, n + 1):
        f[end, 1:] = f[end - 1, 1:]
        ids = by_end[end]
        if ids:
            starts = p[ids, 0]
            f[end, 1:] = np.logaddexp(
                f[end, 1:],
                np.logaddexp.reduce(f[starts, :-1] + interval_lse[ids, None], axis=0),
            )
    logz = float(f[n, K])
    if not math.isfinite(logz):
        raise ValueError("no feasible exactly-K interval selection")

    bwd = np.full((n + 1, K + 1), -np.inf, dtype=np.float64)
    bwd[:, 0] = 0.0
    for start in range(n - 1, -1, -1):
        bwd[start, 1:] = bwd[start + 1, 1:]
        ids = by_start[start]
        if ids:
            ends = p[ids, 1]
            bwd[start, 1:] = np.logaddexp(
                bwd[start, 1:],
                np.logaddexp.reduce(interval_lse[ids, None] + bwd[ends, :-1], axis=0),
            )
    if not math.isclose(logz, float(bwd[0, K]), rel_tol=0.0, abs_tol=1e-10):
        raise RuntimeError("forward/backward partition mismatch")

    # A selected interval's preceding and following structures determine its
    # slot. Convert joint labeled weight into the slot/candidate marginal.
    base = f[p[:, 0], :K] + bwd[p[:, 1], K - 1::-1] - logz
    mu = np.exp(base.T[:, :, None] + scaled[None, :, :])
    residual = np.max(np.abs(mu.sum(axis=(1, 2)) - 1.0))
    if residual > 2e-10:
        raise RuntimeError(f"slot marginals do not sum to one (residual={residual})")

    expected_scaled_energy = float(np.sum(mu * scaled[None, :, :]))
    entropy = logz - expected_scaled_energy
    # Count all labeled structures by assigning zero energy to every label.
    # A stable log-domain DP avoids constructing potentially huge integers.
    log_label_choices = math.log(labels)
    count_f = np.full((n + 1, K + 1), -np.inf, dtype=np.float64)
    count_f[:, 0] = 0.0
    for end in range(1, n + 1):
        count_f[end, 1:] = count_f[end - 1, 1:]
        ids = by_end[end]
        if ids:
            starts = p[ids, 0]
            count_f[end, 1:] = np.logaddexp(
                count_f[end, 1:],
                np.logaddexp.reduce(count_f[starts, :-1] + log_label_choices, axis=0),
            )
    log_count = float(count_f[n, K])
    if not math.isfinite(log_count):
        raise ValueError("no feasible exactly-K interval selection")
    # Remove tiny negative roundoff in the degenerate one-structure case.
    if entropy < 0.0 and entropy > -1e-12:
        entropy = 0.0
    return logz, float(entropy), log_count, mu


def posterior_statistics(n: int, pairs: np.ndarray, energies: np.ndarray, K: int, tau: float) -> dict[str, object]:
    """Dictionary-returning API used by the audit runner."""
    logz, entropy, log_count, mu = fixed_k_posterior(n, pairs, energies, K, tau)
    return {"logZ": logz, "entropy": entropy, "log_count": log_count, "mu": mu}
