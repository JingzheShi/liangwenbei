"""
T34.3 — Synthetic perturbation stress test.

Goal: simulate distribution shift / OOD by perturbing the model's confidence
in each direction. We CANNOT perturb raw features (we only have OOF preds),
but we can perturb the probability vector (which is the proximate signal
that drives the threshold decision rule).

Two perturbation modes:

    (a) Probability noise (additive Gaussian on logits, then renormalize)
        Models "model is more uncertain than it thinks". Stress on
        decision-rule robustness to confidence calibration drift.

    (b) Probability scale (multiplicative on side probs vs flat prob)
        scale s: p0' = s*p0, p2' = s*p2, p1' = 1 - s*(p0+p2). For s=1 unchanged.
        Models a global "alpha shrinkage" — confidence in side directions
        is dampened by factor s. This is a clean proxy for OOD shrinkage.

We report PnL decay vs noise level → brittleness score:
    brittleness = 1 - PnL(noise=med) / PnL(noise=0)

A robust model has low brittleness (PnL holds up under noise).
A brittle model has high brittleness (PnL collapses).

We sweep r ∈ {0.0, 0.05, 0.1, 0.2, 0.3, 0.5} (interpreted as logit-noise std
or as scale=(1-r) for shrinkage).
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from oof_io import HORIZONS, FEE_RATE, compute_pnl_per_row, load_oof  # noqa: E402

RNG_SEED = 1234


def _safe_log(x: np.ndarray) -> np.ndarray:
    return np.log(np.clip(x, 1e-9, 1.0))


def perturb_probs_logit_noise(probs: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Additive Gaussian noise on log-probs; renormalize. sigma=0 → identity."""
    if sigma == 0.0:
        return probs.copy()
    noise = rng.normal(0.0, sigma, size=probs.shape).astype(probs.dtype)
    logp = _safe_log(probs) + noise
    # softmax
    logp = logp - logp.max(axis=1, keepdims=True)
    e = np.exp(logp)
    return e / e.sum(axis=1, keepdims=True)


def perturb_probs_shrinkage(probs: np.ndarray, scale: float) -> np.ndarray:
    """Multiplicative shrinkage of side probs; absorb residual into p1.

    scale=1 → unchanged. scale=0.5 → halve confidence. scale<1 = shrinkage,
    scale>1 = inflation.
    """
    if scale == 1.0:
        return probs.copy()
    out = probs.copy()
    out[:, 0] = probs[:, 0] * scale
    out[:, 2] = probs[:, 2] * scale
    out[:, 1] = 1.0 - out[:, 0] - out[:, 2]
    out[:, 1] = np.clip(out[:, 1], 0.0, 1.0)
    # Renormalize defensively
    s = out.sum(axis=1, keepdims=True)
    return out / np.maximum(s, 1e-9)


def threshold_predict(probs: np.ndarray, T: float, delta: float) -> np.ndarray:
    p0, p1, p2 = probs[:, 0], probs[:, 1], probs[:, 2]
    side_max = np.maximum(p0, p2)
    take = (side_max >= T) & (side_max > p1 + delta)
    side_pred = np.where(p2 > p0, 2, 0)
    return np.where(take, side_pred, 1).astype(np.int64)


def stress_one_oof(df: pd.DataFrame, T: float, delta: float, perturbations: List[float],
                   mode: str, rng: np.random.Generator) -> Dict:
    """Apply perturbation sweep and report cum_pnl + n_active."""
    base_probs = df[["prob_0", "prob_1", "prob_2"]].to_numpy().astype(np.float64)
    mp_t = df["midprice_t"].to_numpy(dtype=np.float64)
    mp_th = df["midprice_th"].to_numpy(dtype=np.float64)
    diff = mp_th - mp_t
    denom = mp_t + 1.0
    fee_term = FEE_RATE * np.abs((mp_th + 1.0) + (mp_t + 1.0))

    out = []
    for r in perturbations:
        if mode == "logit_noise":
            p_pert = perturb_probs_logit_noise(base_probs, r, rng)
        elif mode == "shrinkage":
            p_pert = perturb_probs_shrinkage(base_probs, scale=1.0 - r)
        else:
            raise ValueError(mode)
        pred = threshold_predict(p_pert, T=T, delta=delta)
        side = pred.astype(np.float64) - 1.0
        abs_side = np.abs(side)
        pnl = (side * diff - fee_term * abs_side) / denom
        cum = float(pnl.sum())
        n_active = int((pred != 1).sum())
        out.append({
            "r": float(r),
            "cum_pnl": cum,
            "n_active": n_active,
            "per_trade": cum / n_active if n_active else 0.0,
        })
    return out


def model_thresholds(model: str, h: int):
    """Return (T, delta) used by the deployed Predictor for this (model, h)."""
    if model == "iter_002":
        cfg = {5: (0.6, 0.1), 10: (0.55, 0.1), 20: (0.5, 0.05), 40: (0.5, 0.0), 60: (0.5, 0.2)}
        return cfg[h]
    if model in ("iter_005b", "t26_aug_a"):
        if h == 60:
            return (0.45, 0.10)
        if h == 40:
            return (0.50, 0.0)
    raise ValueError(f"no threshold for {model} h={h}")


def main():
    rng = np.random.default_rng(RNG_SEED)
    perturbations = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    results = {}

    targets = (
        [("iter_002", h) for h in HORIZONS]
        + [("iter_005b", 40), ("iter_005b", 60), ("t26_aug_a", 60)]
    )

    for mode in ("shrinkage", "logit_noise"):
        rng = np.random.default_rng(RNG_SEED)  # reset for reproducibility
        print(f"\n=== Perturbation mode: {mode} ===")
        print(f"{'model_h':<20} " + " ".join([f"r={r:>4.2f}" for r in perturbations])
              + "   brittleness@0.2")
        for model, h in targets:
            df = load_oof(model, h)
            T, delta = model_thresholds(model, h)
            sweep = stress_one_oof(df, T, delta, perturbations, mode, rng)
            base_pnl = sweep[0]["cum_pnl"]
            row_str = " ".join([f"{s['cum_pnl']:>+7.2f}" for s in sweep])
            # Brittleness: 1 - PnL(r=0.2)/PnL(r=0). Higher = more brittle.
            r02 = sweep[3]["cum_pnl"]
            brittleness = 1.0 - r02 / base_pnl if base_pnl > 0 else float("nan")
            print(f"{model+'_h'+str(h):<20} {row_str}   {brittleness:>+7.3f}")
            key = f"{model}_h{h}_{mode}"
            results[key] = {
                "T": T, "delta": delta,
                "base_cum_pnl": base_pnl,
                "sweep": sweep,
                "brittleness_at_0.2": brittleness,
            }

    out_path = os.path.join(HERE, "perturbation_stress_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[perturbation_stress] saved {out_path}")


if __name__ == "__main__":
    main()
