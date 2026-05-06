"""T19: Threshold sweep on NN OOF predictions for h_10.

Reads loso_pred_nn_h10_held{K}.parquet (K in 0..4), runs the same
(T, delta) sweep grid as T11, picks the best by sum of cum_pnl across folds.

Outputs:
    sweep_results_nn_h10.csv
    threshold_results_nn.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

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


def per_fold_metrics(df: pd.DataFrame, T: float, delta: float) -> dict:
    probs = df[["prob_0", "prob_1", "prob_2"]].to_numpy(np.float32)
    pred = thresholded_pred(probs, T, delta)
    m = _per_horizon_metrics(
        pred,
        df["true_label"].to_numpy(np.int64),
        df["midprice_t"].to_numpy(np.float32),
        df["midprice_th"].to_numpy(np.float32),
        fee_rate=0.0001,
    )
    return {
        "cum_pnl": float(m["cum_pnl"]),
        "single_pnl": float(m["single_pnl"]),
        "accuracy": float(m["accuracy"]),
        "n_active": int(m["n_predictions_active"]),
    }


def baseline_argmax(fold_dfs):
    out = []
    for k in sorted(fold_dfs.keys()):
        df = fold_dfs[k]
        pred = df[["prob_0", "prob_1", "prob_2"]].to_numpy(np.float32).argmax(1).astype(np.int8)
        m = _per_horizon_metrics(
            pred, df["true_label"].to_numpy(np.int64),
            df["midprice_t"].to_numpy(np.float32),
            df["midprice_th"].to_numpy(np.float32),
            fee_rate=0.0001,
        )
        out.append({"sym": k, "cum_pnl": float(m["cum_pnl"]),
                    "n_active": int(m["n_predictions_active"]),
                    "accuracy": float(m["accuracy"])})
    return out


def sweep(fold_dfs, label_for_csv: str):
    folds = sorted(fold_dfs.keys())
    rows = []
    for T in T_GRID:
        for d in D_GRID:
            per_fold = []
            for k in folds:
                m = per_fold_metrics(fold_dfs[k], T, d)
                per_fold.append(m)
            sum_cum = sum(r["cum_pnl"] for r in per_fold)
            mean_acc = float(np.mean([r["accuracy"] for r in per_fold]))
            sum_active = int(sum(r["n_active"] for r in per_fold))
            n_pos = int(sum(1 for r in per_fold if r["cum_pnl"] > 0))
            rows.append({
                "T": T, "delta": d, "sum_cum_pnl": sum_cum,
                "mean_cum_pnl": sum_cum / len(folds), "n_pos_folds": n_pos,
                "mean_accuracy": mean_acc, "sum_n_active": sum_active,
                **{f"fold{k}_pnl": per_fold[i]["cum_pnl"] for i, k in enumerate(folds)},
            })
    df = pd.DataFrame(rows).sort_values("sum_cum_pnl", ascending=False)
    out_csv = os.path.join(HERE, f"sweep_results_{label_for_csv}.csv")
    df.to_csv(out_csv, index=False)
    return df, folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syms", default="0,1,2,3,4")
    ap.add_argument("--in_pattern", default="loso_pred_nn_h10_held{K}.parquet")
    ap.add_argument("--label", default="nn_h10")
    args = ap.parse_args()

    target_syms = [int(x) for x in args.syms.split(",")]
    fold_dfs = {}
    for k in target_syms:
        p = os.path.join(HERE, args.in_pattern.format(K=k))
        if not os.path.exists(p):
            sys.exit(f"missing {p}")
        fold_dfs[k] = pd.read_parquet(p)
        print(f"  fold{k}: {len(fold_dfs[k]):,} rows")

    n_total = sum(len(v) for v in fold_dfs.values())
    base = baseline_argmax(fold_dfs)
    base_sum = sum(r["cum_pnl"] for r in base)
    print(f"\nbaseline argmax sum = {base_sum:+.4f}")
    for r in base:
        print(f"  sym={r['sym']}: cum_pnl={r['cum_pnl']:+.4f} n_active={r['n_active']} acc={r['accuracy']:.3f}")

    sweep_df, folds = sweep(fold_dfs, args.label)
    best = sweep_df.iloc[0].to_dict()
    print(f"\nbest (T={best['T']}, d={best['delta']}) sum={best['sum_cum_pnl']:+.4f} pos={best['n_pos_folds']}/{len(folds)}")
    print(f"per-fold pnl: {[round(best[f'fold{k}_pnl'], 3) for k in folds]}")
    print(f"\ntop 10 grid:")
    print(sweep_df.head(10).to_string())

    out = {
        "label": args.label,
        "n_oof_total": n_total,
        "T_grid": T_GRID, "delta_grid": D_GRID,
        "baseline": {"raw_argmax_sum_cum_pnl": float(base_sum), "per_fold": base},
        "best": {
            "T": float(best["T"]), "delta": float(best["delta"]),
            "sum_cum_pnl": float(best["sum_cum_pnl"]),
            "n_pos_folds": int(best["n_pos_folds"]),
            "mean_accuracy": float(best["mean_accuracy"]),
            "sum_n_active": int(best["sum_n_active"]),
            "per_fold_pnl": [float(best[f"fold{k}_pnl"]) for k in folds],
        },
        "top10_grid": sweep_df.head(10).to_dict(orient="records"),
    }
    out_json = os.path.join(HERE, f"threshold_results_{args.label}.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {out_json}")


if __name__ == "__main__":
    main()
