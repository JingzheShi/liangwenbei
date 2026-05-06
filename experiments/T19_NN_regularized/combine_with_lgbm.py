"""T19: Combine NN OOF with LightGBM seed=42 OOF, then threshold sweep.

Reads:
  experiments/T19_NN_regularized/loso_pred_nn_h10_held{K}.parquet (NN)
  experiments/T11_schemeC_multiseed/loso_pred_h10_seed42_held{K}.parquet (LGBM seed=42)

Per fold, aligns by (sym, date, session, t), averages prob_0/1/2.
Saves ensemble OOF parquets and runs sweep.

Outputs:
  loso_pred_nn_lgbm42_h10_held{K}.parquet (averaged)
  sweep_results_nn_lgbm42_h10.csv
  threshold_results_nn_lgbm42.json
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


def align_and_average(nn_df: pd.DataFrame, lgb_df: pd.DataFrame) -> pd.DataFrame:
    """Align two OOF parquets by (sym, date, session, t), average prob_0/1/2.
    Both must have same number of rows after alignment.
    """
    keys = ["sym", "date", "session", "t"]
    nn_df = nn_df.sort_values(keys).reset_index(drop=True)
    lgb_df = lgb_df.sort_values(keys).reset_index(drop=True)
    # Sanity: same row count + key alignment
    if len(nn_df) != len(lgb_df):
        raise RuntimeError(f"row count mismatch: NN {len(nn_df)} vs LGB {len(lgb_df)}")
    for c in keys:
        if not (nn_df[c].to_numpy() == lgb_df[c].to_numpy()).all():
            raise RuntimeError(f"key '{c}' mismatch between NN and LGB OOF")
    # Sanity: midprice should agree (allowing fp32 noise)
    if not np.allclose(nn_df["midprice_t"], lgb_df["midprice_t"], atol=1e-4):
        max_d = float((nn_df["midprice_t"] - lgb_df["midprice_t"]).abs().max())
        print(f"  WARN: midprice_t differs by up to {max_d:.6g} (atol=1e-4)")
    if not np.allclose(nn_df["midprice_th"], lgb_df["midprice_th"], atol=1e-4):
        max_d = float((nn_df["midprice_th"] - lgb_df["midprice_th"]).abs().max())
        print(f"  WARN: midprice_th differs by up to {max_d:.6g} (atol=1e-4)")
    # Average probs
    nn_p = nn_df[["prob_0", "prob_1", "prob_2"]].to_numpy(np.float32)
    lg_p = lgb_df[["prob_0", "prob_1", "prob_2"]].to_numpy(np.float32)
    avg = ((nn_p + lg_p) / 2.0).astype(np.float32)
    out = pd.DataFrame({
        "sym": nn_df["sym"].astype(np.int8),
        "date": nn_df["date"].astype(np.int16),
        "session": nn_df["session"].astype(object),
        "t": nn_df["t"].astype(np.int16),
        "true_label": nn_df["true_label"].astype(np.int8),
        "pred_label": avg.argmax(1).astype(np.int8),
        "prob_0": avg[:, 0],
        "prob_1": avg[:, 1],
        "prob_2": avg[:, 2],
        "midprice_t": nn_df["midprice_t"].astype(np.float32),
        "midprice_th": nn_df["midprice_th"].astype(np.float32),
    })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syms", default="0,1,2,3,4")
    ap.add_argument("--nn_pattern", default="loso_pred_nn_h10_held{K}.parquet")
    ap.add_argument("--lgb_pattern", default=os.path.join(
        ROOT, "experiments", "T11_schemeC_multiseed", "loso_pred_h10_seed42_held{K}.parquet"))
    ap.add_argument("--label", default="nn_lgbm42_h10")
    args = ap.parse_args()

    target_syms = [int(x) for x in args.syms.split(",")]
    nn_dfs = {}
    lgb_dfs = {}
    fold_dfs = {}
    for k in target_syms:
        nn_p = os.path.join(HERE, args.nn_pattern.format(K=k))
        lg_p = args.lgb_pattern.format(K=k)
        if not os.path.exists(nn_p):
            sys.exit(f"missing NN OOF: {nn_p}")
        if not os.path.exists(lg_p):
            sys.exit(f"missing LGB OOF: {lg_p}")
        nn_dfs[k] = pd.read_parquet(nn_p)
        lgb_dfs[k] = pd.read_parquet(lg_p)
        print(f"  fold{k}: NN={len(nn_dfs[k]):,} LGB={len(lgb_dfs[k]):,} rows")
        fold_dfs[k] = align_and_average(nn_dfs[k], lgb_dfs[k])
        out_p = os.path.join(HERE, f"loso_pred_{args.label}_held{k}.parquet")
        fold_dfs[k].to_parquet(out_p, index=False)
        print(f"    -> {out_p}")

    print("\n--- baselines ---")
    nn_base = baseline_argmax(nn_dfs)
    lg_base = baseline_argmax(lgb_dfs)
    en_base = baseline_argmax(fold_dfs)
    print(f"NN raw argmax sum         = {sum(r['cum_pnl'] for r in nn_base):+.4f}")
    print(f"LGB seed42 raw argmax sum = {sum(r['cum_pnl'] for r in lg_base):+.4f}")
    print(f"Ensemble raw argmax sum   = {sum(r['cum_pnl'] for r in en_base):+.4f}")

    print("\n--- sweep on ensemble OOF ---")
    sweep_df, folds = sweep(fold_dfs, args.label)
    best = sweep_df.iloc[0].to_dict()
    print(f"best (T={best['T']}, d={best['delta']}) sum={best['sum_cum_pnl']:+.4f} pos={best['n_pos_folds']}/{len(folds)}")
    print(f"per-fold pnl: {[round(best[f'fold{k}_pnl'], 3) for k in folds]}")
    print(f"\ntop 10 grid:")
    print(sweep_df.head(10).to_string())

    out = {
        "label": args.label,
        "T_grid": T_GRID, "delta_grid": D_GRID,
        "baselines": {
            "nn_raw_argmax_sum": float(sum(r["cum_pnl"] for r in nn_base)),
            "lgb42_raw_argmax_sum": float(sum(r["cum_pnl"] for r in lg_base)),
            "ensemble_raw_argmax_sum": float(sum(r["cum_pnl"] for r in en_base)),
            "nn_per_fold": nn_base,
            "lgb42_per_fold": lg_base,
            "ensemble_per_fold": en_base,
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
    out_json = os.path.join(HERE, f"threshold_results_{args.label}.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {out_json}")


if __name__ == "__main__":
    main()
