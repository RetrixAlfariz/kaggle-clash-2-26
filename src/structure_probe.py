"""Small deterministic utilities for train-only candidate diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np


def _matrix(name: str, value: Any) -> np.ndarray:
    x = np.asarray(value, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} must contain only finite values")
    return x


def fit_predict(
    X_train: Any,
    y_train: Any,
    X_eval: Any,
    weight_train: Any | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one L2 logistic probe and return probabilities and fit diagnostics.

    Features are standardized from the training rows only. The objective is
    weighted mean binary cross-entropy plus ``0.001 / 2 * ||w||²``; the
    intercept is not penalized. Newton steps use a deterministic backtracking
    line search and a small dense Hessian, appropriate for low-dimensional
    probes.
    """
    x = _matrix("X_train", X_train)
    xe = _matrix("X_eval", X_eval)
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    if x.shape[0] != y.size:
        raise ValueError("X_train and y_train row counts differ")
    if xe.shape[1] != x.shape[1]:
        raise ValueError("X_train and X_eval feature counts differ")
    if y.size == 0 or not np.isfinite(y).all() or not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("y_train must be a nonempty finite binary vector")
    if np.unique(y).size != 2:
        raise ValueError("both classes must be present in y_train")
    if weight_train is None:
        sample_weight = np.ones(y.size, dtype=np.float64)
    else:
        sample_weight = np.asarray(weight_train, dtype=np.float64).reshape(-1)
        if sample_weight.size != y.size or not np.isfinite(sample_weight).all():
            raise ValueError("weight_train must be finite and match y_train")
        if np.any(sample_weight < 0) or sample_weight.sum() <= 0:
            raise ValueError("weight_train must be nonnegative with positive sum")

    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    xs = (x - mean) / scale
    xes = (xe - mean) / scale
    design = np.column_stack((np.ones(y.size), xs))
    eval_design = np.column_stack((np.ones(xe.shape[0]), xes))
    penalty = np.full(design.shape[1], 0.001, dtype=np.float64)
    penalty[0] = 0.0
    weight_sum = float(sample_weight.sum())

    def objective(beta: np.ndarray) -> float:
        z = design @ beta
        # logaddexp is stable for both large positive and negative logits.
        ce = np.logaddexp(0.0, z) - y * z
        return float(np.dot(sample_weight, ce) / weight_sum + 0.5 * np.dot(penalty, beta * beta))

    beta = np.zeros(design.shape[1], dtype=np.float64)
    converged = False
    grad_norm = float("inf")
    iterations = 0
    for iteration in range(100):
        z = design @ beta
        # Stable sigmoid without overflowing exp for extreme logits.
        p = np.empty_like(z)
        positive = z >= 0
        p[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
        ez = np.exp(z[~positive])
        p[~positive] = ez / (1.0 + ez)
        residual = sample_weight * (p - y) / weight_sum
        grad = design.T @ residual + penalty * beta
        grad_norm = float(np.max(np.abs(grad)))
        iterations = iteration
        if grad_norm <= 1e-8:
            converged = True
            break
        curvature = sample_weight * p * (1.0 - p) / weight_sum
        hessian = (design.T * curvature) @ design
        hessian.flat[:: hessian.shape[0] + 1] += penalty
        try:
            step = np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, grad, rcond=1e-12)[0]
        if not np.isfinite(step).all():
            break
        old_obj = objective(beta)
        directional = float(grad @ step)
        rate = 1.0
        accepted = False
        for _ in range(40):
            candidate = beta - rate * step
            if objective(candidate) <= old_obj - 1e-4 * rate * directional:
                beta = candidate
                accepted = True
                break
            rate *= 0.5
        if not accepted:
            # Near the optimum, objective changes can round to zero. Accept a
            # sufficiently small Newton step only if it does not worsen loss.
            candidate = beta - rate * step
            if objective(candidate) <= old_obj + 1e-14:
                beta = candidate
            else:
                break
        iterations = iteration + 1

    z_eval = np.clip(eval_design @ beta, -35.0, 35.0)
    predictions = 1.0 / (1.0 + np.exp(-z_eval))
    diagnostics = {
        "converged": converged,
        "iterations": int(iterations),
        "gradient_inf_norm": grad_norm,
        "objective": objective(beta),
        "ridge_lambda": 0.001,
        "standardization_fit_rows": int(x.shape[0]),
        "constant_feature_count": int(np.count_nonzero(x.std(axis=0) < 1e-12)),
        "coefficients_standardized": beta[1:].copy(),
        "feature_mean": mean.copy(),
        "feature_scale": scale.copy(),
        "intercept": float(beta[0]),
    }
    return predictions.astype(np.float64, copy=False), diagnostics


def weighted_auc(y: Any, prediction: Any, weight: Any | None = None) -> float:
    """Return weighted ROC AUC, assigning half credit to tied scores."""
    labels = np.asarray(y).reshape(-1)
    scores = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if labels.size != scores.size or labels.size == 0:
        raise ValueError("y and prediction must be nonempty vectors of equal length")
    if not np.isin(labels, [0, 1, False, True]).all() or not np.isfinite(scores).all():
        raise ValueError("y must be binary and prediction must be finite")
    labels = labels.astype(bool)
    if weight is None:
        w = np.ones(scores.size, dtype=np.float64)
    else:
        w = np.asarray(weight, dtype=np.float64).reshape(-1)
        if w.size != scores.size or not np.isfinite(w).all() or np.any(w < 0):
            raise ValueError("weight must be finite, nonnegative, and match y")
    pos_total = float(w[labels].sum())
    neg_total = float(w[~labels].sum())
    if pos_total <= 0 or neg_total <= 0:
        raise ValueError("weighted AUC requires positive weight in both classes")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    sorted_w = w[order]
    concordant = 0.0
    negative_before = 0.0
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        group_pos = float(sorted_w[start:end][sorted_labels[start:end]].sum())
        group_neg = float(sorted_w[start:end][~sorted_labels[start:end]].sum())
        concordant += group_pos * (negative_before + 0.5 * group_neg)
        negative_before += group_neg
        start = end
    return concordant / (pos_total * neg_total)


def weighted_logloss(y: Any, prediction: Any, weight: Any | None = None) -> float:
    """Return weighted binary log-loss with probabilities clipped at 1e-12."""
    labels = np.asarray(y, dtype=np.float64).reshape(-1)
    probs = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if labels.size != probs.size or labels.size == 0:
        raise ValueError("y and prediction must be nonempty vectors of equal length")
    if not np.isfinite(labels).all() or not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError("y must be a finite binary vector")
    if not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)):
        raise ValueError("prediction must contain finite probabilities in [0, 1]")
    if weight is None:
        w = np.ones(labels.size, dtype=np.float64)
    else:
        w = np.asarray(weight, dtype=np.float64).reshape(-1)
        if w.size != labels.size or not np.isfinite(w).all() or np.any(w < 0):
            raise ValueError("weight must be finite, nonnegative, and match y")
    if w.sum() <= 0:
        raise ValueError("weight must have positive sum")
    p = np.clip(probs, 1e-12, 1.0 - 1e-12)
    losses = -(labels * np.log(p) + (1.0 - labels) * np.log1p(-p))
    return float(np.dot(w, losses) / w.sum())
