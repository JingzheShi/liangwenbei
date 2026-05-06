"""T18: Combine LightGBM + XGBoost OOF probabilities and threshold-sweep.

For each fold, average LightGBM seed probs (T11_schemeC_multiseed/loso_pred_h10_seed{S}_held{K}.parquet)
with XGBoost seed probs (T18_xgboost/loso_pred_xgb_h10_seed{S}_held{K}.parquet)
to form a 2-algorithm (or N×M) ensemble, then sweep thresholds.

Two modes:
- 1+1: LightGBM seed=42 + XGBoost seed=42 (default)
- N+M: arbitrary subset of LightGBM seeds + XGBoost seeds (--lgbm-seeds, --xgb-seeds)

Outputs:
- loso_pred_lgbmxgb_ens_h{H}_held{K}.parquet (averaged)
- sweep_results_lgbmxgb_<tag>_h{H}.csv
- combine_results.json
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
T11_DIR = os.path.join(ROOT, "experiments", "T11_schemeC_multiseed")
sys.path.insert(0, ROOT)

from src.eval.pnl import _per_horizon_metrics  # noqa: E402

T_GRID = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
D_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]


def thresholded_pred(probs, T, delta):
    p0 = probs[:, 0]; p1 = probs[:, 1]; p2 = probs[:, 2]
    side_max = np.maximum(p0, p2)
    take = (side_max >= T) & (side_max > p1 + delta)
    pred_side = np.where(p2 > p0, 2, 0)
    return np.where(take, pred_side, 1).astype(np.int8)


def per_fold_metrics(df_fold, T, delta):
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


def average_probs(parquet_paths):
    """Average prob_0/1/2 across a list of parquet paths, validating key alignment."""
    dfs = [pd.read_parquet(p) for p in parquet_paths]
    n0 = len(dfs[0])
    for p, df in zip(parquet_paths, dfs):
        assert len(df) == n0, f"row count mismatch {p}"
    ref_keys = dfs[0][["sym", "date", "session", "t"]].to_numpy()
    for p, df in list(zip(parquet_paths, dfs))[1:]:
        cur_keys = df[["sym", "date", "session", "t"]].to_numpy()
        if not np.array_equal(ref_keys, cur_keys):
            sys.exit(f"key mismatch with {p}")
    prob_stack = np.stack(
        [df[["prob_0", "prob_1", "prob_2"]].to_numpy(np.float32) for df in dfs], axis=0
    )
    prob_avg = prob_stack.mean(axis=0).astype(np.float32)
    out = pd.DataFrame({
        "sym": dfs[0]["sym"].astype(np.int8),
        "date": dfs[0]["date"].astype(np.int16),
        "session": dfs[0]["session"],
        "t": dfs[0]["t"].astype(np.int16),
        "true_label": dfs[0]["true_label"].astype(np.int8),
        "midprice_t": dfs[0]["midprice_t"].astype(np.float32),
        "midprice_th": dfs[0]["midprice_th"].astype(np.float32),
        "prob_0": prob_avg[:, 0],
        "prob_1": prob_avg[:, 1],
        "prob_2": prob_avg[:, 2],
    })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="10")
    ap.add_argument("--lgbm-seeds", default="42",
                    help="LightGBM seeds (from T11)")
    ap.add_argument("--xgb-seeds", default="42",
                    help="XGBoost seeds (from T18)")
    ap.add_argument("--syms", default="0,1,2,3,4")
    ap.add_argument("--tag", default="lgbm_xgb_42",
                    help="tag for output csv name")
    args = ap.parse_args()

    horizons = [int(x) for x in args.horizons.split(",")]
    lgbm_seeds = [int(x) for x in args.lgbm_seeds.split(",")]
    xgb_seeds = [int(x) for x in args.xgb_seeds.split(",")]
    target_syms = [int(x) for x in args.syms.split(",")]

    print(f"Combining LGBM seeds={lgbm_seeds} + XGB seeds={xgb_seeds}", flush=True)

    out_results = {"per_horizon": {}, "horizons": horizons,
                   "lgbm_seeds": lgbm_seeds, "xgb_seeds": xgb_seeds,
                   "tag": args.tag, "task": "T18 LightGBM+XGBoost combine"}

    for H in horizons:
        print(f"\n{'='*78}\n=== HORIZON h={H} ===\n{'='*78}", flush=True)
        ens_dfs = {}
        for held in target_syms:
            paths = []
            for s in lgbm_seeds:
                paths.append(os.path.join(T11_DIR, f"loso_pred_h{H}_seed{s}_held{held}.parquet"))
            for s in xgb_seeds:
                paths.append(os.path.join(HERE, f"loso_pred_xgb_h{H}_seed{s}_held{held}.parquet"))
            print(f"  fold{held}: averaging {len(paths)} parquets ({len(lgbm_seeds)} LGBM + {len(xgb_seeds)} XGB)",
                  flush=True)
            for p in paths:
                if not os.path.exists(p):
                    sys.exit(f"missing {p}")
            avg = average_probs(paths)
            out_path = os.path.join(HERE, f"loso_pred_lgbmxgb_{args.tag}_h{H}_held{held}.parquet")
            avg.to_parquet(out_path, index=False)
            ens_dfs[held] = avg

        sweep_res = sweep_one(ens_dfs, f"lgbmxgb_{args.tag}_h{H}")
        out_results["per_horizon"][f"h{H}"] = {
            "horizon": H,
            "n_lgbm_seeds": len(lgbm_seeds),
            "n_xgb_seeds": len(xgb_seeds),
            "ensemble_sweep": sweep_res,
        }

    out_path = os.path.join(HERE, f"combine_results_{args.tag}.json")
    with open(out_path, "w") as f:
        json.dump(out_results, f, indent=2)
    print(f"\ncombine_results_{args.tag}.json -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
