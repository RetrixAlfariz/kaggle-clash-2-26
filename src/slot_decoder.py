"""Fixed-K interval MAP and slot-marginal minimum Bayes risk decoding.

No gold inputs. Distribution is conditional independent candidate Bernoullis,
restricted to exactly K pairwise non-overlapping spans.
"""
import bisect
import math
import numpy as np


def prepare(probabilities, count):
    if not isinstance(count, int) or count < 0:
        raise ValueError("K must be a nonnegative integer")
    spans = sorted(probabilities, key=lambda s: (s[1], s[0], s[2]))
    for a, b, label in spans:
        if a < 0 or a >= b or not label:
            raise ValueError("Invalid span")
    probs = np.array([probabilities[s] for s in spans], dtype=float)
    if np.any(~np.isfinite(probs)) or np.any((probs < 0) | (probs > 1)):
        raise ValueError("Invalid probability")
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    weights = np.log(probs) - np.log1p(-probs)
    ends = [s[1] for s in spans]
    previous = [bisect.bisect_right(ends, s[0], hi=i) for i, s in enumerate(spans)]
    return spans, weights, previous


def maximize(spans, previous, rewards, count):
    """rewards[i,k-1] rewards putting interval i at chronological slot k."""
    n = len(spans)
    values = np.full((n + 1, count + 1), -np.inf)
    values[:, 0] = 0
    took = np.zeros((n + 1, count + 1), dtype=bool)
    for i in range(n):
        values[i + 1, 1:] = values[i, 1:]
        take = values[previous[i], :-1] + rewards[i]
        mask = take > values[i + 1, 1:]
        values[i + 1, 1:][mask] = take[mask]
        took[i + 1, 1:] = mask
    if not np.isfinite(values[n, count]):
        raise ValueError("No feasible exactly-K interval selection")
    result = []
    i, k = n, count
    while k:
        if took[i, k]:
            result.append(spans[i - 1])
            i = previous[i - 1]
            k -= 1
        else:
            i -= 1
    return sorted(result)


def slot_marginals(probabilities, count):
    spans, weights, previous = prepare(probabilities, count)
    n = len(spans)
    # F[i,k]: total weight of compatible k-subsets in end-sorted prefix i.
    forward = np.full((n + 1, count + 1), -np.inf)
    forward[:, 0] = 0
    for i in range(n):
        forward[i + 1, 1:] = np.logaddexp(forward[i, 1:], forward[previous[i], :-1] + weights[i])
    z = forward[n, count]
    if not np.isfinite(z):
        raise ValueError("No feasible exactly-K interval selection")
    order = sorted(range(n), key=lambda i: (spans[i][0], spans[i][1], spans[i][2]))
    starts = [spans[i][0] for i in order]
    # G[j,k]: total weight of compatible k-subsets in start-sorted suffix j.
    backward = np.full((n + 1, count + 1), -np.inf)
    backward[:, 0] = 0
    for j in range(n - 1, -1, -1):
        i = order[j]
        following = bisect.bisect_left(starts, spans[i][1], lo=j + 1)
        backward[j, 1:] = np.logaddexp(backward[j + 1, 1:], backward[following, :-1] + weights[i])
    if not math.isclose(float(z), float(backward[0, count]), abs_tol=1e-8):
        raise ValueError("Forward/backward partition mismatch")
    marginals = np.zeros((n, count))
    for i, span in enumerate(spans):
        following = bisect.bisect_left(starts, span[1])
        marginals[i] = np.exp(forward[previous[i], :count] + weights[i] + backward[following, count - 1::-1] - z) if count else []
    if count and not np.allclose(marginals.sum(axis=0), 1., atol=1e-7):
        raise ValueError("Slot marginal probabilities do not sum to one")
    return spans, previous, marginals


def decode(probabilities, count, method="slot_mbr"):
    if method == "slot_mbr":
        spans, previous, rewards = slot_marginals(probabilities, count)
    elif method == "map":
        spans, weights, previous = prepare(probabilities, count)
        rewards = np.repeat(weights[:, None], count, axis=1)
    else:
        raise ValueError("Unknown decoder")
    return maximize(spans, previous, rewards, count)
