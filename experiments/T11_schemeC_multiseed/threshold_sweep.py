"""T11: Per-horizon threshold sweep on multi-seed ensemble OOF.

Rule: side_max = max(p0, p2);
      take if side_max >= T and side_max > p1 + delta;
      pred 2 if p2 > p0 else 0; otherwise 1.

Operates on loso_pred_ensemble_h{H}_held{K}.parquet (output of aggregate_ensemble.py).
Also evaluates raw argmax baselines (no threshold).

Compared against per-seed individual sweeps (so we can tell if ensemble adds value
on top of best individual seed) and against iter_002 (single seed=42).

Outputs:
    sweep_results_ensemble_h{H}.csv
    sweep_results_seed{S}_h{H}.csv (per-seed for reference)
    threshold_results.json
"""
from __future__ import annotations

import argparse
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


def baseline_argmax(fold_dfs):
    """Raw argmax PnL per fold (no threshold)."""
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


def sweep_grid(fold_dfs, label_for_csv):
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
    sweep_df = pd.DataFrame(rows).sort_values("sum_cum_pnl", ascending=False)
    out_csv = os.path.join(HERE, f"sweep_results_{label_for_csv}.csv")
    sweep_df.to_csv(out_csv, index=False)
    return sweep_df, folds


def sweep_one(fold_dfs, label_for_csv):
    print(f"\n=== sweeping {label_for_csv} ===", flush=True)
    n_total = sum(len(v) for v in fold_dfs.values())
    print(f"  total OOF rows: {n_total:,}", flush=True)

    base_pf = baseline_argmax(fold_dfs)
    base_sum = sum(r["cum_pnl"] for r in base_pf)
    print(f"  baseline argmax sum = {base_sum:+.4f} per_fold={[round(r['cum_pnl'],3) for r in base_pf]}",
          flush=True)

    sweep_df, folds = sweep_grid(fold_dfs, label_for_csv)
    best = sweep_df.iloc[0].to_dict()
    print(f"  best (T={best['T']}, d={best['delta']}) sum={best['sum_cum_pnl']:+.4f} "
          f"pos={best['n_pos_folds']}/{len(folds)}", flush=True)
    print(f"  per-fold: {[round(best[f'fold{k}_pnl'], 3) for k in folds]}", flush=True)

    return {
        "n_oof_total": int(n_total),
        "baseline": {
            "raw_argmax_sum_cum_pnl": float(base_sum),
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
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="10")
    ap.add_argument("--seeds", default="42,1,7,13,100")
    ap.add_argument("--syms", default="0,1,2,3,4")
    args = ap.parse_args()

    horizons = [int(x) for x in args.horizons.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    target_syms = [int(x) for x in args.syms.split(",")]

    t0 = time.time()
    out_results = {"per_horizon": {}, "horizons": horizons, "seeds": seeds,
                   "T_grid": T_GRID, "delta_grid": D_GRID,
                   "task": "T11 per-horizon threshold sweep on multi-seed ensemble OOF"}

    cross_summary = []

    for H in horizons:
        print(f"\n{'='*78}\n=== HORIZON h={H} ===\n{'='*78}", flush=True)
        # 1) Ensemble sweep
        ens_dfs = {}
        for k in target_syms:
            p = os.path.join(HERE, f"loso_pred_ensemble_h{H}_held{k}.parquet")
            if not os.path.exists(p):
                sys.exit(f"missing {p}")
            ens_dfs[k] = pd.read_parquet(p)
        ens_res = sweep_one(ens_dfs, f"ensemble_h{H}")

        # 2) Per-seed sweep (so we can compare)
        seed_results = {}
        for s in seeds:
            sd_dfs = {}
            for k in target_syms:
                p = os.path.join(HERE, f"loso_pred_h{H}_seed{s}_held{k}.parquet")
                if not os.path.exists(p):
                    sys.exit(f"missing {p}")
                sd_dfs[k] = pd.read_parquet(p)
            seed_results[s] = sweep_one(sd_dfs, f"seed{s}_h{H}")

        out_results["per_horizon"][f"h{H}"] = {
            "horizon": H,
            "ensemble": ens_res,
            "per_seed": {f"seed_{s}": seed_results[s] for s in seeds},
        }

        # Cross summary row
        best_seed_sum = max(seed_results[s]["best"]["sum_cum_pnl"] for s in seeds)
        best_seed = max(seeds, key=lambda s: seed_results[s]["best"]["sum_cum_pnl"])
        cross_summary.append({
            "horizon": H,
            "ensemble_argmax_sum": ens_res["baseline"]["raw_argmax_sum_cum_pnl"],
            "ensemble_best_T": ens_res["best"]["T"],
            "ensemble_best_delta": ens_res["best"]["delta"],
            "ensemble_best_sum": ens_res["best"]["sum_cum_pnl"],
            "ensemble_n_pos": ens_res["best"]["n_pos_folds"],
            "best_single_seed": best_seed,
            "best_single_seed_sum": best_seed_sum,
            "ensemble_uplift_vs_best_seed": ens_res["best"]["sum_cum_pnl"] - best_seed_sum,
        })

    out_results["summary_rows"] = cross_summary
    out_results["elapsed_sec"] = time.time() - t0
    out_path = os.path.join(HERE, "threshold_results.json")
    with open(out_path, "w") as f:
        json.dump(out_results, f, indent=2)
    print(f"\nthreshold_results.json -> {out_path}", flush=True)

    # Print final cross summary
    print(f"\n{'='*78}\n=== CROSS HORIZON SUMMARY ===\n{'='*78}", flush=True)
    print(f"{'h':>3} {'ens_argmax':>11} {'ens_best':>10} {'(T,d)':>14} "
          f"{'best_seed':>10} {'seed_best':>10} {'uplift':>9}", flush=True)
    for row in cross_summary:
        print(f"{row['horizon']:>3} {row['ensemble_argmax_sum']:>+11.4f} "
              f"{row['ensemble_best_sum']:>+10.4f} "
              f"({row['ensemble_best_T']:.2f},{row['ensemble_best_delta']:.2f})  "
              f"seed{row['best_single_seed']:<6} {row['best_single_seed_sum']:>+10.4f} "
              f"{row['ensemble_uplift_vs_best_seed']:>+9.4f}", flush=True)


if __name__ == "__main__":
    main()
