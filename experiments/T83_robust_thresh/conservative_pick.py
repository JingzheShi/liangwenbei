"""T83 follow-up — conservative threshold picks via bootstrap quartiles.

Key insight from diagnostic.py: the +6 PnL gap on date splits is ≥90% due to
per-date PnL variation — NOT threshold overfit. Choosing the wrong thr only
costs ~0.3-0.4 PnL on a 15-PnL eval.

So the lever isn't "find a smarter thr" — it's **trade less when the model is
likely wrong**. Conservative thresholds (slightly higher than DE-optimal) cut
the variance of expected PnL on out-of-sample (where realised PnL per active
trade may be lower than train suggested).

Strategy: bootstrap-DE on full 442k → take an UPPER QUARTILE thr (e.g., 75th
percentile of bootstrap thr_up, 75th of -thr_dn). This gives a thr that DE
considers "conservative" but plausible.

We compare:
  - iter_013 full-fit DE (thr_up=3.72e-4, thr_dn=1.61e-4) → +36.15
  - bootstrap median DE
  - bootstrap q75 thr_up + q25 -thr_dn (= more conservative both sides; trades less)
  - bootstrap q90 thr_up + q90 -thr_dn (very conservative)
  - symmetric k=1.25 (T75 conservative fallback) → +33.72
  - symmetric k=1.5 → +33.28

For each, also report n_active fraction.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from core import load_pred_avg, sym_sum, gate_asym, gate_sym, FEE


def fit_de_quick(df, seed=0, maxiter=50, popsize=18):
    pred = df["pred"].to_numpy()
    mp_t = df["mp_t"].to_numpy()
    mp_th = df["mp_th"].to_numpy()
    sym_arr = df["sym"].to_numpy()

    def obj(x):
        thr_up, thr_dn = x
        a = gate_asym(pred, thr_up, thr_dn)
        side = a.astype(np.float64) - 1.0
        abs_side = np.abs(side)
        fee = FEE * abs_side * np.abs((mp_th + 1.0) + (mp_t + 1.0))
        pnl_row = (side * (mp_th - mp_t) - fee) / (mp_t + 1.0)
        s = 0.0
        for k in (0, 1, 2, 3, 4):
            mask = sym_arr == k
            if mask.any():
                s += pnl_row[mask].sum()
        return -s

    result = differential_evolution(
        obj, bounds=((1e-5, 8e-4), (1e-5, 8e-4)),
        seed=seed, maxiter=maxiter, popsize=popsize, polish=True, tol=1e-7,
        mutation=(0.5, 1.0), recombination=0.7, updating="deferred",
        workers=1, init="sobol")
    return float(result.x[0]), float(result.x[1]), float(-result.fun)


def main():
    df = load_pred_avg()
    print(f"Loaded {len(df)} rows")

    # Fit on FULL data with multiple seeds → see seed variance
    print("\n--- Full-data DE asym across seeds ---")
    seed_results = []
    for sd in range(8):
        thr_up, thr_dn, p = fit_de_quick(df, seed=sd, maxiter=50, popsize=18)
        seed_results.append((thr_up, thr_dn, p))
        print(f"  seed {sd}: thr_up={thr_up:.4e} thr_dn={thr_dn:.4e} pnl={p:.4f}")

    # Bootstrap on FULL data
    print("\n--- Bootstrap-DE on full 442k (B=20, frac=0.7) ---")
    rng = np.random.default_rng(42)
    boot_results = []
    n = len(df)
    for b in range(20):
        idx = rng.integers(0, n, int(0.7 * n))
        df_b = df.iloc[idx].reset_index(drop=True)
        thr_up, thr_dn, _ = fit_de_quick(df_b, seed=b, maxiter=40, popsize=14)
        boot_results.append((thr_up, thr_dn))
        print(f"  boot {b}: thr_up={thr_up:.4e} thr_dn={thr_dn:.4e}")

    arr = np.array(boot_results)
    print(f"\nBootstrap thr_up: median={np.median(arr[:, 0]):.4e}  q25={np.quantile(arr[:, 0], 0.25):.4e}  "
          f"q75={np.quantile(arr[:, 0], 0.75):.4e}  q90={np.quantile(arr[:, 0], 0.90):.4e}")
    print(f"Bootstrap thr_dn: median={np.median(arr[:, 1]):.4e}  q25={np.quantile(arr[:, 1], 0.25):.4e}  "
          f"q75={np.quantile(arr[:, 1], 0.75):.4e}  q90={np.quantile(arr[:, 1], 0.90):.4e}")

    # Evaluate candidates on full data (no train/eval split — just to see PnL)
    print("\n--- Candidate threshold settings on full 442k ---")
    candidates = {
        "iter_013 (DE 4D)":                (3.723e-4, 1.613e-4),
        "bootstrap median":                 (float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))),
        "bootstrap q75 thr_up + q75 thr_dn (conservative)":
            (float(np.quantile(arr[:, 0], 0.75)), float(np.quantile(arr[:, 1], 0.75))),
        "bootstrap q60 thr_up + q60 thr_dn (mild)":
            (float(np.quantile(arr[:, 0], 0.60)), float(np.quantile(arr[:, 1], 0.60))),
        "bootstrap q90 thr_up + q90 thr_dn (very conservative)":
            (float(np.quantile(arr[:, 0], 0.90)), float(np.quantile(arr[:, 1], 0.90))),
    }
    for name, (thr_up, thr_dn) in candidates.items():
        a = gate_asym(df["pred"].to_numpy(), thr_up, thr_dn)
        pnl = sym_sum(df, a)
        n_active = int((a != 1).sum())
        n_act_frac = n_active / float(len(df))
        print(f"  {name}: thr=({thr_up:.4e}, {thr_dn:.4e}) pnl={pnl:.4f} n_active={n_active} ({n_act_frac:.1%})")

    # Also symmetric k variants
    for k in (1.0, 1.25, 1.5, 1.75, 2.0):
        thr = k * 2 * FEE
        a = gate_sym(df["pred"].to_numpy(), thr)
        pnl = sym_sum(df, a)
        n_active = int((a != 1).sum())
        print(f"  symmetric k={k}: thr={thr:.4e} pnl={pnl:.4f} n_active={n_active} ({n_active/float(len(df)):.1%})")

    # Save bootstrap stats
    out = {
        "seed_results": [{"seed": i, "thr_up": s[0], "thr_dn": s[1], "pnl": s[2]} for i, s in enumerate(seed_results)],
        "bootstrap_thrs": [{"thr_up": float(t[0]), "thr_dn": float(t[1])} for t in boot_results],
        "boot_stats": {
            "thr_up_median": float(np.median(arr[:, 0])),
            "thr_up_q25": float(np.quantile(arr[:, 0], 0.25)),
            "thr_up_q75": float(np.quantile(arr[:, 0], 0.75)),
            "thr_up_q90": float(np.quantile(arr[:, 0], 0.90)),
            "thr_dn_median": float(np.median(arr[:, 1])),
            "thr_dn_q25": float(np.quantile(arr[:, 1], 0.25)),
            "thr_dn_q75": float(np.quantile(arr[:, 1], 0.75)),
            "thr_dn_q90": float(np.quantile(arr[:, 1], 0.90)),
        },
    }
    with open(os.path.join(HERE, "conservative_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nWrote conservative_results.json")


if __name__ == "__main__":
    main()
