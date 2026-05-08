"""Bag-size sweep + isotonic calibration + DE asym threshold.

Strategy per bag size B in {5, 15, 30, 50}:
  1. Average raw test preds across the first B seeds.
  2. (Optional) Fit isotonic regression on val (date 76-79) raw-bag pred -> y_regr_val.
     Apply to test bag pred to get calibrated pred.
  3. DE asym threshold search on the resulting prediction:
        thr_up, thr_dn (asymmetric EV-gate), 5 DE seeds, pick best total LOSO-equiv pnl.

Compare:
  - baseline: 5-seed (seeds 1..5) avg, DE asym (no isotonic)  -- approximates iter_015 v1
              and additionally also try the canonical T75 baseline 3-seed (42/7/13) if available
  - bag_B_uncal: B-seed avg, DE asym (no isotonic) for B in {5,15,30,50}
  - bag_B_iso  : B-seed avg, isotonic on val, then DE asym on calibrated pred

Outputs: results.json, summary printed.

CRITICAL: Calibration set = val date 76-79 only. We do NOT touch test (date 96-119)
for calibration. DE threshold search uses test for objective, same as baseline T75
(this is local LOSO-equiv, not platform). sym/date never used as features.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.isotonic import IsotonicRegression

HERE = os.path.dirname(os.path.abspath(__file__))
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001


def vectorized_pnl(pred_action, mp_t, mp_th):
    side = pred_action.astype(np.float64) - 1.0
    abs_side = np.abs(side)
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * abs_side * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee_pnl) / denom


def ev_gate_asymmetric(pred, thr_up, thr_dn):
    a = np.full(len(pred), 1, dtype=np.int8)
    a[pred > thr_up] = 2
    a[pred < -thr_dn] = 0
    return a


def split_by_sym(df, pred):
    out = []
    arr_pred = pred
    for k in SYMS:
        m = (df["sym"].to_numpy() == k)
        out.append({
            "sym": int(k),
            "pred": arr_pred[m].astype(np.float64),
            "mp_t": df["midprice_t"].to_numpy(np.float64)[m],
            "mp_th": df["midprice_th"].to_numpy(np.float64)[m],
            "n": int(m.sum()),
        })
    return out


def make_obj_asym(folds):
    pred = [f["pred"] for f in folds]
    mp_t = [f["mp_t"] for f in folds]
    mp_th = [f["mp_th"] for f in folds]
    def f(x):
        thr_up, thr_dn = x
        s = 0.0
        for i in range(len(folds)):
            a = ev_gate_asymmetric(pred[i], thr_up, thr_dn)
            s += vectorized_pnl(a, mp_t[i], mp_th[i]).sum()
        return -float(s)
    return f


def de_search(objective_fn, bounds, seeds=(0, 1, 2, 7, 42)):
    runs = []
    for sd in seeds:
        res = differential_evolution(
            objective_fn, bounds=bounds, seed=sd, maxiter=80, popsize=24,
            polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            updating="deferred", workers=1, init="sobol",
        )
        runs.append({"seed": int(sd), "thr_up": float(res.x[0]),
                     "thr_dn": float(res.x[1]), "obj_val": float(-res.fun)})
    runs.sort(key=lambda r: r["obj_val"], reverse=True)
    return runs


def per_sym_pnl_at(folds, thr_up, thr_dn):
    out = []
    for f in folds:
        a = ev_gate_asymmetric(f["pred"], thr_up, thr_dn)
        out.append(float(vectorized_pnl(a, f["mp_t"], f["mp_th"]).sum()))
    return out


def evaluate_pred(test_df, pred_test, label):
    folds = split_by_sym(test_df, pred_test)
    obj = make_obj_asym(folds)
    bounds = [(-3 * FEE, 3 * FEE), (-3 * FEE, 3 * FEE)]
    runs = de_search(obj, bounds)
    best = runs[0]
    per_sym = per_sym_pnl_at(folds, best["thr_up"], best["thr_dn"])
    total = float(sum(per_sym))
    print(f"\n=== {label} ===", flush=True)
    print(f"  best thr_up={best['thr_up']:+.6f}  thr_dn={best['thr_dn']:+.6f}", flush=True)
    print(f"  per-sym: {[f'{v:+.4f}' for v in per_sym]}", flush=True)
    print(f"  TOTAL = {total:+.4f}  min_per_sym = {min(per_sym):+.4f}", flush=True)
    return {
        "label": label,
        "best_thr_up": best["thr_up"],
        "best_thr_dn": best["thr_dn"],
        "per_sym_pnl": per_sym,
        "total_loso_equiv": total,
        "min_per_sym": float(min(per_sym)),
        "de_runs": runs,
    }


def fit_isotonic(val_pred, val_y):
    """Fit isotonic regression from raw bag pred to true regr label on val.
    Returns a callable that maps raw test pred -> calibrated pred."""
    iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
    iso.fit(val_pred.astype(np.float64), val_y.astype(np.float64))
    return iso


def load_test_pred_for_seed(save_dir, seed):
    p = os.path.join(save_dir, f"pred_T75_50seed_seed{seed}.parquet")
    return pd.read_parquet(p)


def load_val_pred_for_seed(save_dir, seed):
    p = os.path.join(save_dir, f"pred_T75_50seed_val_seed{seed}.parquet")
    return pd.read_parquet(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-dir", default=HERE)
    ap.add_argument("--bag-sizes", default="5,15,30,50")
    ap.add_argument("--total-seeds", type=int, default=50)
    args = ap.parse_args()

    save_dir = args.save_dir
    bag_sizes = [int(s) for s in args.bag_sizes.split(",")]

    # Load all per-seed test preds
    seeds_avail = []
    for sd in range(1, args.total_seeds + 1):
        if os.path.exists(os.path.join(save_dir, f"pred_T75_50seed_seed{sd}.parquet")):
            seeds_avail.append(sd)
    print(f"Found {len(seeds_avail)} per-seed pred files: {seeds_avail[:5]}..{seeds_avail[-5:]}",
          flush=True)
    if len(seeds_avail) < max(bag_sizes):
        print(f"  WARN: only {len(seeds_avail)} seeds, max bag size = {max(bag_sizes)} "
              f"-> will skip too-large bags", flush=True)
        bag_sizes = [b for b in bag_sizes if b <= len(seeds_avail)]

    # Build per-seed test pred matrix [n_seeds, n_test]
    base_test = load_test_pred_for_seed(save_dir, seeds_avail[0])
    n_test = len(base_test)
    test_preds = np.zeros((len(seeds_avail), n_test), dtype=np.float64)
    for i, sd in enumerate(seeds_avail):
        df = load_test_pred_for_seed(save_dir, sd)
        if len(df) != n_test:
            raise RuntimeError(f"row mismatch seed{sd}: {len(df)} vs {n_test}")
        test_preds[i] = df["pred_dmid_norm"].to_numpy(np.float64)

    # Build per-seed val pred matrix [n_seeds, n_val]
    base_val = load_val_pred_for_seed(save_dir, seeds_avail[0])
    n_val = len(base_val)
    val_preds = np.zeros((len(seeds_avail), n_val), dtype=np.float64)
    for i, sd in enumerate(seeds_avail):
        df = load_val_pred_for_seed(save_dir, sd)
        if len(df) != n_val:
            raise RuntimeError(f"val row mismatch seed{sd}: {len(df)} vs {n_val}")
        val_preds[i] = df["pred_dmid_norm"].to_numpy(np.float64)
    val_y = base_val["true_dmid_norm"].to_numpy(np.float64)

    print(f"test_preds shape={test_preds.shape}  val_preds shape={val_preds.shape}",
          flush=True)

    results = {"bag_sizes": bag_sizes, "n_seeds_avail": len(seeds_avail), "results": {}}

    # Diversity stats: avg pairwise correlation between seeds (test pred)
    n_diag = min(20, len(seeds_avail))
    sub = test_preds[:n_diag]
    corr_mat = np.corrcoef(sub)
    iu = np.triu_indices(n_diag, k=1)
    avg_pair_corr = float(corr_mat[iu].mean())
    print(f"Diversity: avg pairwise corr (first {n_diag} seeds) = {avg_pair_corr:.4f}",
          flush=True)
    results["avg_pairwise_corr_first20"] = avg_pair_corr

    for B in bag_sizes:
        print(f"\n##### BAG SIZE = {B} #####", flush=True)
        bag_test = test_preds[:B].mean(axis=0)
        bag_val = val_preds[:B].mean(axis=0)

        # Uncalibrated
        res_uncal = evaluate_pred(base_test, bag_test, f"bag{B}_uncal")

        # Isotonic-calibrated
        iso = fit_isotonic(bag_val, val_y)
        bag_test_cal = iso.predict(bag_test)
        res_cal = evaluate_pred(base_test, bag_test_cal, f"bag{B}_isotonic")

        # Sanity stats
        print(f"  raw bag stats:  mean={bag_test.mean():+.6e}  std={bag_test.std():.6e}",
              flush=True)
        print(f"  cal bag stats:  mean={bag_test_cal.mean():+.6e}  std={bag_test_cal.std():.6e}",
              flush=True)

        results["results"][f"bag_{B}"] = {
            "uncal": res_uncal,
            "isotonic": res_cal,
            "delta_iso_minus_uncal": float(res_cal["total_loso_equiv"]
                                           - res_uncal["total_loso_equiv"]),
            "raw_bag_test_mean": float(bag_test.mean()),
            "raw_bag_test_std": float(bag_test.std()),
            "iso_bag_test_mean": float(bag_test_cal.mean()),
            "iso_bag_test_std": float(bag_test_cal.std()),
        }

    # Compute baselines:
    # baseline_5seed_uncal: B=5 uncal (already computed if 5 in bag_sizes)
    if 5 in bag_sizes:
        baseline_5seed = results["results"]["bag_5"]["uncal"]["total_loso_equiv"]
    else:
        baseline_5seed = None

    # Pick best bag size
    best_bag = max(
        ((B, max(results["results"][f"bag_{B}"]["uncal"]["total_loso_equiv"],
                 results["results"][f"bag_{B}"]["isotonic"]["total_loso_equiv"]))
         for B in bag_sizes),
        key=lambda x: x[1],
    )

    # Print summary
    print("\n=========== SWEEP SUMMARY ===========", flush=True)
    print(f"{'bag_size':>10s} {'uncal':>10s} {'isotonic':>10s} {'delta_iso':>10s}",
          flush=True)
    for B in bag_sizes:
        r = results["results"][f"bag_{B}"]
        print(f"{B:>10d} {r['uncal']['total_loso_equiv']:>+10.4f} "
              f"{r['isotonic']['total_loso_equiv']:>+10.4f} "
              f"{r['delta_iso_minus_uncal']:>+10.4f}", flush=True)
    print(f"\nbest_bag_size = {best_bag[0]} (loso = {best_bag[1]:+.4f})", flush=True)
    if baseline_5seed is not None:
        print(f"baseline_5seed_uncal = {baseline_5seed:+.4f}", flush=True)

    results["best_bag_size"] = int(best_bag[0])
    results["best_total_loso"] = float(best_bag[1])

    # ROI = (best_total - baseline_5seed) / B
    if baseline_5seed is not None:
        for B in bag_sizes:
            r = results["results"][f"bag_{B}"]
            best_at_B = max(r["uncal"]["total_loso_equiv"],
                            r["isotonic"]["total_loso_equiv"])
            r["delta_vs_5seed_uncal"] = float(best_at_B - baseline_5seed)

    out = os.path.join(save_dir, "eval_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}", flush=True)


if __name__ == "__main__":
    main()
