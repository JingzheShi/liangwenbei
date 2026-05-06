"""T6: Threshold sweep on ensemble OOF predictions.

Inputs:
    experiments/T6_ensemble_multiseed/ensemble_pred_scheme{A,B}_held{K}.parquet
        produced by aggregate_predictions.py

Sweep T x delta grid (same as T5a) to find best (T*, delta*) on the ensemble.

Reference comparisons:
    Scheme B single seed=42 + T=0.45 delta=0.05 -> sum_cum_pnl = +11.11 (iter_001d)
    Scheme A single seed=42 + T=0.50 delta=0.15 -> sum_cum_pnl = +6.45  (T4 sweep)

Outputs:
    sweep_scheme{X}_results.csv  -- full grid sorted by sum_cum_pnl
    best_per_fold_scheme{X}.csv  -- per-fold breakdown for best (T*, delta*)
    raw_argmax_per_fold_scheme{X}.csv -- baseline ensemble argmax (no threshold)
    sweep_results_scheme{X}.json -- summary

Usage:
    python threshold_sweep_ensemble.py --scheme A
    python threshold_sweep_ensemble.py --scheme B
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from src.eval.pnl import _per_horizon_metrics  # noqa: E402

T_GRID = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
D_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]


def thresholded_pred(probs: np.ndarray, T: float, delta: float) -> np.ndarray:
    p0 = probs[:, 0]
    p1 = probs[:, 1]
    p2 = probs[:, 2]
    side_max = np.maximum(p0, p2)
    take = (side_max >= T) & (side_max > p1 + delta)
    pred_side = np.where(p2 > p0, 2, 0)
    return np.where(take, pred_side, 1).astype(np.int8)


def per_fold_metrics(df_fold: pd.DataFrame, T: float, delta: float) -> dict:
    probs = df_fold[["prob_0", "prob_1", "prob_2"]].to_numpy(np.float32)
    pred = thresholded_pred(probs, T, delta)
    m = _per_horizon_metrics(
        pred,
        df_fold["true_label_60"].to_numpy(np.int64),
        df_fold["midprice_t"].to_numpy(np.float32),
        df_fold["midprice_t60"].to_numpy(np.float32),
        fee_rate=0.0001,
    )
    return {
        "cum_pnl": float(m["cum_pnl"]),
        "single_pnl": float(m["single_pnl"]),
        "accuracy": float(m["accuracy"]),
        "n_active": int(m["n_predictions_active"]),
        "f0_5_macro": float(m["f0_5_macro"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", choices=("A", "B"), required=True)
    args = ap.parse_args()

    t0 = time.time()
    scheme = args.scheme

    fold_dfs = {}
    for k in range(5):
        p = os.path.join(HERE, f"ensemble_pred_scheme{scheme}_held{k}.parquet")
        fold_dfs[k] = pd.read_parquet(p)
        print(f"  fold {k}: {len(fold_dfs[k]):,} rows  ({os.path.relpath(p, ROOT)})", flush=True)
    n_total = sum(len(v) for v in fold_dfs.values())

    # ---- Baseline: ensemble raw argmax ----
    print("\n=== Baseline (ensemble raw argmax) per-fold ===", flush=True)
    baseline_pf = []
    for k in range(5):
        df = fold_dfs[k]
        pred = df[["prob_0", "prob_1", "prob_2"]].to_numpy(np.float32).argmax(1).astype(np.int8)
        m = _per_horizon_metrics(
            pred,
            df["true_label_60"].to_numpy(np.int64),
            df["midprice_t"].to_numpy(np.float32),
            df["midprice_t60"].to_numpy(np.float32),
            fee_rate=0.0001,
        )
        baseline_pf.append({"sym": k, "cum_pnl": float(m["cum_pnl"]),
                            "n_active": int(m["n_predictions_active"]),
                            "accuracy": float(m["accuracy"])})
        print(f"  sym {k}: cum_pnl={m['cum_pnl']:+.4f} n_active={m['n_predictions_active']:,}", flush=True)
    baseline_sum = sum(r["cum_pnl"] for r in baseline_pf)
    print(f"  ensemble argmax 5-fold sum cum_pnl = {baseline_sum:+.4f}", flush=True)
    pd.DataFrame(baseline_pf).to_csv(
        os.path.join(HERE, f"raw_argmax_per_fold_scheme{scheme}.csv"), index=False
    )

    # ---- Sweep grid ----
    rows = []
    print(f"\n=== Sweeping {len(T_GRID)} x {len(D_GRID)} = {len(T_GRID)*len(D_GRID)} combos ===", flush=True)
    for T in T_GRID:
        for d in D_GRID:
            per_fold = []
            for k in range(5):
                m = per_fold_metrics(fold_dfs[k], T, d)
                per_fold.append(m)
            sum_cum = sum(r["cum_pnl"] for r in per_fold)
            mean_acc = float(np.mean([r["accuracy"] for r in per_fold]))
            sum_active = int(sum(r["n_active"] for r in per_fold))
            n_pos = int(sum(1 for r in per_fold if r["cum_pnl"] > 0))
            rows.append({
                "T": T, "delta": d,
                "sum_cum_pnl": sum_cum,
                "mean_cum_pnl": sum_cum / 5.0,
                "n_pos_folds": n_pos,
                "mean_accuracy": mean_acc,
                "sum_n_active": sum_active,
                "fold0_pnl": per_fold[0]["cum_pnl"],
                "fold1_pnl": per_fold[1]["cum_pnl"],
                "fold2_pnl": per_fold[2]["cum_pnl"],
                "fold3_pnl": per_fold[3]["cum_pnl"],
                "fold4_pnl": per_fold[4]["cum_pnl"],
            })
            print(f"  T={T:.2f} d={d:.2f} -> sum={sum_cum:+.4f} pos={n_pos}/5 active={sum_active:,}",
                  flush=True)

    sweep_df = pd.DataFrame(rows).sort_values("sum_cum_pnl", ascending=False)
    sweep_df.to_csv(os.path.join(HERE, f"sweep_scheme{scheme}_results.csv"), index=False)

    best_row = sweep_df.iloc[0].to_dict()
    best_T = float(best_row["T"])
    best_d = float(best_row["delta"])
    best_sum = float(best_row["sum_cum_pnl"])
    print(f"\nBest (T*, delta*) = ({best_T}, {best_d}): sum_cum_pnl={best_sum:+.4f}", flush=True)

    # Per-fold @ best
    print(f"\n=== Per-fold @ best (T={best_T}, delta={best_d}) ===", flush=True)
    best_pf_rows = []
    for k in range(5):
        m = per_fold_metrics(fold_dfs[k], best_T, best_d)
        best_pf_rows.append({"sym": k, **m})
        print(f"  sym {k}: cum_pnl={m['cum_pnl']:+.4f} n_active={m['n_active']:,} acc={m['accuracy']:.4f}",
              flush=True)
    pd.DataFrame(best_pf_rows).to_csv(
        os.path.join(HERE, f"best_per_fold_scheme{scheme}.csv"), index=False
    )

    # iter_001d benchmark for Scheme B
    iter001d_match = None
    if scheme == "B":
        try:
            iter001d_match = sweep_df[
                (sweep_df["T"] == 0.45) & (sweep_df["delta"] == 0.05)
            ].iloc[0].to_dict()
        except IndexError:
            iter001d_match = None

    out = {
        "task": f"T6 ensemble threshold sweep on Scheme {scheme}",
        "scheme": scheme,
        "n_oof_total": int(n_total),
        "T_grid": T_GRID,
        "delta_grid": D_GRID,
        "ensemble_baseline_argmax": {
            "sum_cum_pnl": float(baseline_sum),
            "per_fold": baseline_pf,
        },
        "ensemble_best": {
            "T": best_T,
            "delta": best_d,
            "sum_cum_pnl": best_sum,
            "n_pos_folds": int(best_row["n_pos_folds"]),
            "mean_accuracy": float(best_row["mean_accuracy"]),
            "sum_n_active": int(best_row["sum_n_active"]),
            "per_fold": best_pf_rows,
        },
        "iter_001d_threshold_on_ensemble": (
            {
                "T": 0.45, "delta": 0.05,
                "sum_cum_pnl": float(iter001d_match["sum_cum_pnl"]),
                "n_pos_folds": int(iter001d_match["n_pos_folds"]),
            }
            if iter001d_match is not None else None
        ),
        "elapsed_sec": time.time() - t0,
    }
    with open(os.path.join(HERE, f"sweep_results_scheme{scheme}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nresults -> sweep_results_scheme{scheme}.json", flush=True)


if __name__ == "__main__":
    main()
