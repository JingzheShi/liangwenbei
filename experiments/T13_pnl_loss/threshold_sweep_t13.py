"""Threshold sweep for T13 (S1/S2) LOSO predictions.

Reuses the same (T_grid, delta_grid) as T5b. Reads LOSO prediction parquets
matching <prefix>_pred_h{H}_held{K}*.parquet.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from src.eval.pnl import _per_horizon_metrics  # noqa: E402

T_GRID = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
D_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]


def thresholded_pred(probs: np.ndarray, T: float, delta: float) -> np.ndarray:
    p0 = probs[:, 0]; p1 = probs[:, 1]; p2 = probs[:, 2]
    side_max = np.maximum(p0, p2)
    take = (side_max >= T) & (side_max > p1 + delta)
    pred_side = np.where(p2 > p0, 2, 0)
    return np.where(take, pred_side, 1).astype(np.int8)


def per_fold_metrics(df_fold: pd.DataFrame, T: float, delta: float) -> dict:
    probs = df_fold[["prob_0", "prob_1", "prob_2"]].to_numpy(np.float32)
    pred = thresholded_pred(probs, T, delta)
    m = _per_horizon_metrics(
        pred,
        df_fold["true_label"].to_numpy(np.int64),
        df_fold["midprice_t"].to_numpy(np.float32),
        df_fold["midprice_th"].to_numpy(np.float32),
        fee_rate=0.0001,
    )
    return {
        "cum_pnl": float(m["cum_pnl"]),
        "single_pnl": float(m["single_pnl"]),
        "accuracy": float(m["accuracy"]),
        "n_active": int(m["n_predictions_active"]),
        "f0_5_macro": float(m["f0_5_macro"]),
    }


def sweep_one_horizon(prefix: str, H: int, suffix: str = "") -> dict:
    fold_dfs = {}
    for k in range(5):
        cand = os.path.join(HERE, f"{prefix}_pred_h{H}_held{k}{suffix}.parquet")
        if not os.path.exists(cand):
            # try wildcard
            matches = glob.glob(os.path.join(HERE, f"{prefix}_pred_h{H}_held{k}*.parquet"))
            if not matches:
                return {"error": f"missing pred for held={k}"}
            cand = sorted(matches)[0]
        fold_dfs[k] = pd.read_parquet(cand)

    n_total = sum(len(v) for v in fold_dfs.values())
    print(f"\n=== Horizon h={H}, total OOF rows: {n_total:,} ===", flush=True)

    baseline_pf = []
    for k in range(5):
        df = fold_dfs[k]
        pred = df[["prob_0", "prob_1", "prob_2"]].to_numpy(np.float32).argmax(1).astype(np.int8)
        m = _per_horizon_metrics(
            pred, df["true_label"].to_numpy(np.int64),
            df["midprice_t"].to_numpy(np.float32),
            df["midprice_th"].to_numpy(np.float32),
            fee_rate=0.0001,
        )
        baseline_pf.append({"sym": k, "cum_pnl": float(m["cum_pnl"]),
                            "n_active": int(m["n_predictions_active"]),
                            "accuracy": float(m["accuracy"])})
    baseline_sum = sum(r["cum_pnl"] for r in baseline_pf)
    print(f"  baseline argmax sum_cum_pnl = {baseline_sum:+.4f}  per-fold={[round(r['cum_pnl'],3) for r in baseline_pf]}", flush=True)

    rows = []
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
                "T": T, "delta": d, "sum_cum_pnl": sum_cum,
                "mean_cum_pnl": sum_cum / 5.0, "n_pos_folds": n_pos,
                "mean_accuracy": mean_acc, "sum_n_active": sum_active,
                **{f"fold{k}_pnl": per_fold[k]["cum_pnl"] for k in range(5)},
            })

    sweep_df = pd.DataFrame(rows).sort_values("sum_cum_pnl", ascending=False)
    out_csv = os.path.join(HERE, f"sweep_{prefix}_h{H}{suffix}.csv")
    sweep_df.to_csv(out_csv, index=False)
    print(f"  saved sweep -> {out_csv}", flush=True)

    best_row = sweep_df.iloc[0].to_dict()
    best_T = float(best_row["T"]); best_d = float(best_row["delta"])
    best_sum = float(best_row["sum_cum_pnl"])
    print(f"  best (T={best_T}, d={best_d}) sum={best_sum:+.4f} pos={best_row['n_pos_folds']}/5",
          flush=True)
    print(f"  per-fold: {[round(best_row[f'fold{k}_pnl'], 3) for k in range(5)]}",
          flush=True)

    return {
        "horizon": H,
        "n_oof_total": int(n_total),
        "baseline": {
            "raw_argmax_sum_cum_pnl": float(baseline_sum),
            "per_fold": baseline_pf,
        },
        "best": {
            "T": best_T, "delta": best_d, "sum_cum_pnl": best_sum,
            "n_pos_folds": int(best_row["n_pos_folds"]),
            "mean_accuracy": float(best_row["mean_accuracy"]),
            "sum_n_active": int(best_row["sum_n_active"]),
            "per_fold_pnl": [float(best_row[f"fold{k}_pnl"]) for k in range(5)],
        },
        "top10_grid": sweep_df.head(10).to_dict(orient="records"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="e.g. s1 or s2")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--suffix", default="", help="optional suffix like _normalized")
    ap.add_argument("--out-tag", default=None)
    args = ap.parse_args()

    H = args.horizon
    t0 = time.time()
    res = sweep_one_horizon(args.prefix, H, args.suffix)
    out = {
        "task": f"T13 {args.prefix} threshold sweep h={H}{args.suffix}",
        "horizon": H,
        "T_grid": T_GRID,
        "delta_grid": D_GRID,
        "result": res,
        "elapsed_sec": time.time() - t0,
    }
    out_path = os.path.join(HERE, f"sweep_results_{args.prefix}_h{H}{args.suffix}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsweep results -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
