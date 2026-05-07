"""T47: Baseline single-seed h_60-only DE 4D thresh (apples-to-apples vs stacking).

This computes the LOSO sum cum_pnl when using ONLY h_60 probs (no stacking)
with the same single-seed-42 + aug_a model trained in train_loso.py.

Used as a sanity baseline: iter_006 +13.61 is 5-seed averaged, our single-seed
should be lower. This measures the stacking gain over single-seed h_60.
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
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

N_FOLDS = 5
FEE = 0.0001


def gate_asymmetric(probs, T_up, T_dn, d_up, d_dn):
    p0 = probs[:, 0]; p1 = probs[:, 1]; p2 = probs[:, 2]
    take_up = (p2 >= T_up) & (p2 > p1 + d_up) & (p2 > p0)
    take_dn = (p0 >= T_dn) & (p0 > p1 + d_dn) & (p0 > p2)
    pred = np.full(probs.shape[0], 1, dtype=np.int8)
    pred[take_up] = 2
    pred[take_dn] = 0
    return pred


def vectorized_pnl(pred, label, mp_t, mp_th):
    side = pred.astype(np.float64) - 1.0
    abs_side = np.abs(side)
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee = FEE * abs_side * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee) / denom


def main():
    t0 = time.time()
    print("=== T47 baseline single-seed h_60 + DE 4D thresh ===")

    label = []
    mp_t = []
    mp_th = []
    probs = []
    for k in range(N_FOLDS):
        p = os.path.join(HERE, f"loso_pred_h60_held{k}.parquet")
        df = pd.read_parquet(p)
        label.append(df["true_label_60"].to_numpy(np.int64))
        mp_t.append(df["midprice_t"].to_numpy(np.float64))
        mp_th.append(df["midprice_t60"].to_numpy(np.float64))
        probs.append(df[["prob_0", "prob_1", "prob_2"]].to_numpy(np.float32))

    def objective(params):
        Tu, Td, du, dd = params
        total = 0.0
        for k in range(N_FOLDS):
            pred = gate_asymmetric(probs[k], Tu, Td, du, dd)
            total += vectorized_pnl(pred, label[k], mp_t[k], mp_th[k]).sum()
        return -total

    bounds = [(0.34, 0.55), (0.34, 0.55), (0.0, 0.30), (0.0, 0.30)]
    print(f"  bounds: {bounds}; running scipy DE ...")
    de_result = differential_evolution(
        objective, bounds,
        popsize=30, maxiter=100, tol=1e-4,
        mutation=(0.5, 1.5), recombination=0.7,
        seed=42, polish=True, workers=1, updating="deferred",
    )

    Tu, Td, du, dd = de_result.x
    sum_pnl = -de_result.fun
    per_fold = []
    for k in range(N_FOLDS):
        pred = gate_asymmetric(probs[k], Tu, Td, du, dd)
        per_fold.append(float(vectorized_pnl(pred, label[k], mp_t[k], mp_th[k]).sum()))

    print(f"  T_up={Tu:.4f} T_dn={Td:.4f} d_up={du:.4f} d_dn={dd:.4f}")
    print(f"  sum_cum_pnl = {sum_pnl:+.4f}")
    for k in range(N_FOLDS):
        print(f"  fold k={k}: cum_pnl={per_fold[k]:+.4f}")

    # Argmax baseline (no thresh)
    argmax_per_fold = []
    for k in range(N_FOLDS):
        pred = probs[k].argmax(axis=1).astype(np.int8)
        argmax_per_fold.append(float(vectorized_pnl(pred, label[k], mp_t[k], mp_th[k]).sum()))
    print(f"  ARGMAX baseline sum = {sum(argmax_per_fold):+.4f}")

    out = {
        "task": "T47 baseline single-seed h_60 + DE 4D thresh",
        "argmax_per_fold": argmax_per_fold,
        "argmax_sum": float(sum(argmax_per_fold)),
        "de_thresh": {"T_up": float(Tu), "T_dn": float(Td),
                      "d_up": float(du), "d_dn": float(dd)},
        "de_sum_pnl": float(sum_pnl),
        "de_per_fold": per_fold,
        "elapsed_sec": time.time() - t0,
    }
    with open(os.path.join(HERE, "baseline_h60_de_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nresults -> {os.path.join(HERE, 'baseline_h60_de_results.json')}")


if __name__ == "__main__":
    main()
