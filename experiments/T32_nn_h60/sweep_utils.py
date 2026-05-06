"""Shared helpers for thresholded predictions / per-horizon metrics."""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.eval.pnl import _per_horizon_metrics  # noqa: E402


def thresholded_pred(probs: np.ndarray, T: float, delta: float) -> np.ndarray:
    p0 = probs[:, 0]; p1 = probs[:, 1]; p2 = probs[:, 2]
    side_max = np.maximum(p0, p2)
    take = (side_max >= T) & (side_max > p1 + delta)
    pred_side = np.where(p2 > p0, 2, 0)
    return np.where(take, pred_side, 1).astype(np.int8)


def thresholded_metrics(
    probs: np.ndarray, true_lab: np.ndarray,
    mp_t: np.ndarray, mp_th: np.ndarray,
    T: float, delta: float,
) -> dict:
    pred = thresholded_pred(probs, T, delta)
    m = _per_horizon_metrics(
        pred, true_lab.astype(np.int64),
        mp_t.astype(np.float32), mp_th.astype(np.float32),
        fee_rate=0.0001,
    )
    return {
        "cum_pnl": float(m["cum_pnl"]),
        "single_pnl": float(m["single_pnl"]),
        "accuracy": float(m["accuracy"]),
        "n_active": int(m["n_predictions_active"]),
        "f0_5_macro": float(m["f0_5_macro"]),
    }
