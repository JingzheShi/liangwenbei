"""T83 extra strategies — conformal, quantile-fixed, ensemble.

Testing strategies that fix or constrain the threshold to global structure
(distribution quantiles, fee multiples) rather than tuning to PnL on train.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from core import (
    load_pred_avg, sym_sum, gate_asym, eval_strategy,
    fit_de_asym, fit_de_asym_kfold, fit_bootstrap_de, fit_de_regularized,
    fit_symmetric_grid, FEE,
)


def fit_quantile_fixed(df_train: pd.DataFrame, q_up: float, q_dn: float) -> dict:
    """Fix thresholds at the q-th quantile of pred distribution (sym-agnostic).

    q_up = upper-tail trim point (e.g., 0.75 → top 25% of preds taken as long).
    q_dn = lower-tail trim point (e.g., 0.25 → bottom 25% as short).

    Robust because it adapts to the distribution shape, not to PnL fit.
    """
    pred = df_train["pred"].to_numpy()
    thr_up = float(np.quantile(pred, q_up))
    thr_dn = float(-np.quantile(pred, q_dn))
    if thr_up < 0 or thr_dn < 0:
        # ensure positive (clip if pred dist is shifted)
        thr_up = max(thr_up, 1e-6)
        thr_dn = max(thr_dn, 1e-6)
    a = gate_asym(pred, thr_up, thr_dn)
    train_pnl = sym_sum(df_train, a)
    return {"strategy": f"quantile_q_up{q_up}_q_dn{q_dn}",
            "thr_up": thr_up, "thr_dn": thr_dn, "train_pnl": train_pnl,
            "q_up": q_up, "q_dn": q_dn}


def fit_quantile_grid(df_train: pd.DataFrame,
                       q_up_grid=None, q_dn_grid=None) -> dict:
    """Grid-search (q_up, q_dn) on train half, return best."""
    if q_up_grid is None:
        q_up_grid = np.arange(0.55, 0.96, 0.025)
    if q_dn_grid is None:
        q_dn_grid = np.arange(0.05, 0.46, 0.025)
    best = None
    for qu in q_up_grid:
        for qd in q_dn_grid:
            rec = fit_quantile_fixed(df_train, qu, qd)
            if best is None or rec["train_pnl"] > best["train_pnl"]:
                best = rec
    best["strategy"] = "quantile_grid"
    return best


def fit_ensemble_avg(df_train: pd.DataFrame) -> dict:
    """Average the thresholds from {sym_grid, kfold_de_5_date, bootstrap_de}."""
    s = fit_symmetric_grid(df_train)
    k = fit_de_asym_kfold(df_train, n_folds=5, fold_axis="date", maxiter=40, popsize=16)
    b = fit_bootstrap_de(df_train, n_boot=20, frac=0.7, seed=42, maxiter=35, popsize=14)

    thr_up_avg = (s["thr_up"] + k["thr_up"] + b["thr_up"]) / 3.0
    thr_dn_avg = (s["thr_dn"] + k["thr_dn"] + b["thr_dn"]) / 3.0
    a = gate_asym(df_train["pred"].to_numpy(), thr_up_avg, thr_dn_avg)
    train_pnl = sym_sum(df_train, a)
    return {"strategy": "ensemble_avg_3",
            "thr_up": thr_up_avg, "thr_dn": thr_dn_avg, "train_pnl": train_pnl,
            "components": {"sym_grid_thr": (s["thr_up"], s["thr_dn"]),
                           "kfold5_date_thr": (k["thr_up"], k["thr_dn"]),
                           "bootstrap_thr": (b["thr_up"], b["thr_dn"])}}


def fit_pnl_sharpe_combined(df_train: pd.DataFrame, lam_sharpe: float = 0.0,
                             bounds=((1e-5, 8e-4), (1e-5, 8e-4)),
                             de_seeds=(0, 1, 2)) -> dict:
    """DE that maximizes train_pnl + lam_sharpe * Sharpe-like ratio.

    Sharpe = (total_pnl) / (n_active * std_per_row + ε). Penalizes thresholds
    that produce a lot of low-quality trades.
    """
    from scipy.optimize import differential_evolution

    pred = df_train["pred"].to_numpy()
    mp_t = df_train["mp_t"].to_numpy()
    mp_th = df_train["mp_th"].to_numpy()

    def obj(x):
        thr_up, thr_dn = x
        a = gate_asym(pred, thr_up, thr_dn)
        side = a.astype(np.float64) - 1.0
        abs_side = np.abs(side)
        fee = FEE * abs_side * np.abs((mp_th + 1.0) + (mp_t + 1.0))
        pnl_row = (side * (mp_th - mp_t) - fee) / (mp_t + 1.0)
        n_active = float((a != 1).sum())
        total = float(pnl_row.sum())
        if n_active < 100:
            return 1e9
        if lam_sharpe > 0:
            std = float(pnl_row[a != 1].std() + 1e-10)
            sharpe_term = lam_sharpe * total / (np.sqrt(n_active) * std)
            return -(total + sharpe_term)
        return -total

    best = None
    for sd in de_seeds:
        result = differential_evolution(
            obj, bounds=bounds, seed=sd, maxiter=50, popsize=18,
            polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            updating="deferred", workers=1, init="sobol",
        )
        rec = {"thr_up": float(result.x[0]), "thr_dn": float(result.x[1])}
        a = gate_asym(pred, rec["thr_up"], rec["thr_dn"])
        rec["train_pnl"] = sym_sum(df_train, a)
        if best is None or rec["train_pnl"] > best["train_pnl"]:
            best = rec
    return {"strategy": f"pnl_sharpe_lam{lam_sharpe}",
            "thr_up": best["thr_up"], "thr_dn": best["thr_dn"],
            "train_pnl": best["train_pnl"]}


if __name__ == "__main__":
    df = load_pred_avg()
    print("--- quantile_grid on full 442k ---")
    rec = fit_quantile_grid(df)
    print(f"  q_up={rec['q_up']:.3f} q_dn={rec['q_dn']:.3f} "
          f"thr_up={rec['thr_up']:.4e} thr_dn={rec['thr_dn']:.4e} "
          f"train_pnl={rec['train_pnl']:.4f}")
