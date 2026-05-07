"""Multiclass softmax focal-loss custom objective for LightGBM v4.x.

Focal loss (Lin et al., 2017) for multiclass softmax:

    L_i = -alpha * (1 - p_y)^gamma * log(p_y)

where p_y = softmax(z)_{y_i}.

Analytical gradient w.r.t. raw scores z_k for sample i (true class y):

    dL/dz_k = alpha * (p_k - 1{k==y}) * f_i

with the "modulator"

    f_i = (1 - p_y)^gamma + gamma * p_y * (1 - p_y)^(gamma-1) * (-log(p_y))

For the Hessian we use the standard CE Hessian h_k = p_k * (1 - p_k) scaled by
the focal modulator f_i. This is the conventional approximation used in most
focal LightGBM/XGBoost implementations and is sufficient for stable training.

LightGBM v4 multiclass custom-objective API:
    objective(preds, train_data) -> (grad, hess)
    preds, grad, hess all shape (n_samples, n_classes).
"""
from __future__ import annotations

import numpy as np


class MulticlassFocalObjective:
    """Stateless callable. Stores alpha, gamma, num_class."""

    def __init__(self, num_class: int = 3, alpha: float = 0.25, gamma: float = 2.0,
                 class_weights: np.ndarray | None = None):
        self.num_class = int(num_class)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        if class_weights is not None:
            cw = np.asarray(class_weights, dtype=np.float64)
            assert cw.shape == (self.num_class,)
            self.class_weights = cw
        else:
            self.class_weights = None

    def __call__(self, preds: np.ndarray, train_data) -> tuple:
        K = self.num_class
        y = train_data.get_label().astype(np.int64)
        n = y.shape[0]
        # v4 hands preds as (n, K)
        z = np.asarray(preds, dtype=np.float64)
        if z.ndim == 1:
            z = z.reshape(n, K)
        # Stable softmax
        z_max = z.max(axis=1, keepdims=True)
        e = np.exp(z - z_max)
        p = e / e.sum(axis=1, keepdims=True)
        eps = 1e-7
        p = np.clip(p, eps, 1.0 - eps)

        rows = np.arange(n)
        p_y = p[rows, y]                              # (n,)
        one_m_py = 1.0 - p_y
        log_py = np.log(p_y)
        if abs(self.gamma - 0.0) < 1e-12:
            modulator = np.ones(n, dtype=np.float64)
        else:
            modulator = (one_m_py ** self.gamma) + \
                        self.gamma * p_y * (one_m_py ** (self.gamma - 1.0)) * (-log_py)

        per_sample = self.alpha * modulator
        if self.class_weights is not None:
            per_sample = per_sample * self.class_weights[y]

        grad = p.copy()
        grad[rows, y] -= 1.0
        grad *= per_sample[:, None]
        hess = per_sample[:, None] * p * (1.0 - p)
        return grad.astype(np.float64), hess.astype(np.float64)


def softmax_2d(raw: np.ndarray, num_class: int = 3) -> np.ndarray:
    """Softmax along axis=1. Reshape if 1D."""
    z = np.asarray(raw, dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(-1, num_class)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)
