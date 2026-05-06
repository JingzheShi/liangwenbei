"""T21b: Single-seed (seed=42) threshold sweep on Scheme H 291-d LOSO OOF.

Mirrors T11's sweep_one() but operates only on per-seed parquets — there is no
ensemble (per user request: drop ensemble, single seed only).

Outputs:
    sweep_results_seed42_h10.csv
    threshold_results_seed42.json
"""
from __future__ import annotations

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
H = 10
SEED = 42
SYMS = (0, 1, 2, 3, 4)


def thresholded_pred(probs, T, delta):
    p0 = probs[:, 0]; p1 = probs[:, 1]; p2 = probs[:, 2]
    side_max = np.maximum(p0, p2)
    take = (side_max >= T) & (side_max > p1 + delta)
    pred_side = np.where(p2 > p0, 2, 0)
    return np.where(take, pred_side, 1).astype(np.int8)


def per_fold_metrics(df, T, delta):
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
        "f0_5_macro": float(m["f0_5_macro"]),
    }


def baseline_argmax(fold_dfs):
    pf = []
    for k in sorted(fold_dfs.keys()):
        df = fold_dfs[k]
        pred = df[["prob_0", "prob_1", "prob_2"]].to_numpy(np.float32).argmax(1).astype(np.int8)
        m = _per_horizon_metrics(
            pred, df["true_label"].to_numpy(np.int64),
            df["midprice_t"].to_numpy(np.float32),
            df["midprice_th"].to_numpy(np.float32),
            fee_rate=0.0001,
        )
        pf.append({"sym": k, "cum_pnl": float(m["cum_pnl"]),
                   "n_active": int(m["n_predictions_active"]),
                   "accuracy": float(m["accuracy"])})
    return pf


def sweep_grid(fold_dfs):
    rows = []
    folds = sorted(fold_dfs.keys())
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
    return pd.DataFrame(rows).sort_values("sum_cum_pnl", ascending=False), folds


def main():
    t0 = time.time()
    fold_dfs = {}
    for k in SYMS:
        p = os.path.join(HERE, f"loso_pred_h{H}_seed{SEED}_held{k}.parquet")
        if not os.path.exists(p):
            sys.exit(f"missing {p}")
        fold_dfs[k] = pd.read_parquet(p)

    n_total = sum(len(v) for v in fold_dfs.values())
    print(f"=== single-seed sweep seed={SEED} h={H} ({n_total:,} OOF rows) ===", flush=True)

    base_pf = baseline_argmax(fold_dfs)
    base_sum = sum(r["cum_pnl"] for r in base_pf)
    print(f"\nbaseline argmax sum = {base_sum:+.4f}")
    for r in base_pf:
        print(f"  fold{r['sym']}: cum_pnl={r['cum_pnl']:+.4f} n_active={r['n_active']} acc={r['accuracy']:.4f}")

    sweep_df, folds = sweep_grid(fold_dfs)
    out_csv = os.path.join(HERE, f"sweep_results_seed{SEED}_h{H}.csv")
    sweep_df.to_csv(out_csv, index=False)
    print(f"\nsweep csv -> {out_csv}")

    print(f"\n=== TOP 10 (T, d) BY sum_cum_pnl ===")
    print(sweep_df.head(10).to_string(index=False))

    best = sweep_df.iloc[0].to_dict()
    print(f"\nBEST: T={best['T']:.2f} d={best['delta']:.2f} sum={best['sum_cum_pnl']:+.4f} "
          f"pos={best['n_pos_folds']}/5 mean_acc={best['mean_accuracy']:.4f} "
          f"n_active={best['sum_n_active']}")
    print(f"per-fold: {[round(best[f'fold{k}_pnl'], 4) for k in folds]}")

    # comparison
    iter_002_baseline = 21.86  # T11 seed_42 baseline best ~21.715 too; iter_002 is +21.86
    decision = "build_iter_004" if best["sum_cum_pnl"] > 22.5 else "no_iter_004"
    print(f"\n=== DECISION ===")
    print(f"iter_002 baseline       : +21.86")
    print(f"T21 SchemeH seed42 best : {best['sum_cum_pnl']:+.4f}")
    print(f"diff vs iter_002        : {best['sum_cum_pnl'] - iter_002_baseline:+.4f}")
    print(f"decision                : {decision} (need > +22.5)")

    out = {
        "task": "T21b single-seed (42) threshold sweep on Scheme H 291-d",
        "horizon": H,
        "seed": SEED,
        "T_grid": T_GRID,
        "delta_grid": D_GRID,
        "baseline_argmax": {
            "sum_cum_pnl": float(base_sum),
            "per_fold": base_pf,
        },
        "best": {
            "T": float(best["T"]), "delta": float(best["delta"]),
            "sum_cum_pnl": float(best["sum_cum_pnl"]),
            "n_pos_folds": int(best["n_pos_folds"]),
            "mean_accuracy": float(best["mean_accuracy"]),
            "sum_n_active": int(best["sum_n_active"]),
            "per_fold_pnl": [float(best[f"fold{k}_pnl"]) for k in folds],
        },
        "top10_grid": sweep_df.head(10).to_dict(orient="records"),
        "decision": decision,
        "iter_002_baseline": iter_002_baseline,
        "diff_vs_iter_002": float(best["sum_cum_pnl"]) - iter_002_baseline,
        "elapsed_sec": time.time() - t0,
    }
    out_json = os.path.join(HERE, "threshold_results_seed42.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nthreshold_results -> {out_json}")


if __name__ == "__main__":
    main()
