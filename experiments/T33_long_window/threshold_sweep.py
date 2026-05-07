"""T33 threshold sweep on Scheme J LOSO OOF predictions.

Sweeps (T, delta) grid per (variant, seed) on h=60 5-fold OOF and reports best.
Also supports multi-seed averaging (ensemble).
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


def per_fold_metrics(df_fold: pd.DataFrame, T: float, delta: float,
                     prob_cols=("prob_0", "prob_1", "prob_2")) -> dict:
    probs = df_fold[list(prob_cols)].to_numpy(np.float32)
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


def load_fold_dfs(variant: str, seeds, H: int = 60):
    """Return dict {held: df with averaged probs across seeds}."""
    fold_dfs = {}
    for k in range(5):
        per_seed = []
        for s in seeds:
            p = os.path.join(HERE, f"loso_pred_h{H}_{variant}_seed{s}_held{k}.parquet")
            if not os.path.exists(p):
                # Fallback: legacy naming without seed
                p = os.path.join(HERE, f"loso_pred_h{H}_{variant}_held{k}.parquet")
                if not os.path.exists(p):
                    return {"error": f"missing held{k} for {variant} seeds={seeds}"}
            per_seed.append(pd.read_parquet(p))
        if len(per_seed) == 1:
            fold_dfs[k] = per_seed[0]
        else:
            base = per_seed[0][["sym", "date", "session", "t", "true_label",
                                 "midprice_t", "midprice_th"]].copy()
            probs = np.zeros((len(base), 3), dtype=np.float32)
            for d in per_seed:
                probs += d[["prob_0", "prob_1", "prob_2"]].to_numpy(np.float32)
            probs /= float(len(per_seed))
            base["prob_0"] = probs[:, 0]
            base["prob_1"] = probs[:, 1]
            base["prob_2"] = probs[:, 2]
            base["pred_label"] = probs.argmax(axis=1).astype(np.int8)
            fold_dfs[k] = base
    return fold_dfs


def sweep_one(variant: str, seeds, H: int = 60) -> dict:
    fold_dfs = load_fold_dfs(variant, seeds, H)
    if isinstance(fold_dfs, dict) and "error" in fold_dfs:
        print(f"  skip: {fold_dfs['error']}", flush=True)
        return fold_dfs

    n_total = sum(len(v) for v in fold_dfs.values())
    label = f"{variant}_seeds={list(seeds)}"
    print(f"\n=== {label}, h={H}, OOF rows: {n_total:,} ===", flush=True)

    # baseline raw argmax
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
    print(f"  raw_argmax sum = {baseline_sum:+.4f}  per_fold={[round(r['cum_pnl'],3) for r in baseline_pf]}",
          flush=True)

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
            std_pnl = float(np.std([r["cum_pnl"] for r in per_fold], ddof=0))
            rows.append({
                "T": T, "delta": d, "sum_cum_pnl": sum_cum,
                "mean_cum_pnl": sum_cum / 5.0,
                "std_cum_pnl": std_pnl,
                "n_pos_folds": n_pos,
                "mean_accuracy": mean_acc, "sum_n_active": sum_active,
                "per_fold": [r["cum_pnl"] for r in per_fold],
            })

    df_sw = pd.DataFrame(rows)
    df_sw.to_csv(
        os.path.join(HERE, f"sweep_{variant}_seeds_{'_'.join(map(str,seeds))}_h{H}.csv"),
        index=False,
    )
    best = df_sw.loc[df_sw["sum_cum_pnl"].idxmax()].to_dict()
    top10 = df_sw.sort_values("sum_cum_pnl", ascending=False).head(10).to_dict("records")
    print(f"  BEST (T,δ) = ({best['T']}, {best['delta']}) sum={best['sum_cum_pnl']:+.4f} "
          f"mean={best['mean_cum_pnl']:+.4f} std={best['std_cum_pnl']:.4f} "
          f"pos_folds={best['n_pos_folds']}/5 active={best['sum_n_active']:,} acc={best['mean_accuracy']:.4f}",
          flush=True)
    print("  per-fold at BEST:", best["per_fold"], flush=True)
    return {
        "variant": variant,
        "seeds": list(seeds),
        "horizon": H,
        "raw_argmax_sum": baseline_sum,
        "best": best,
        "top10": top10,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="baseline,aug_a")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--ensemble-seeds", default="",
                    help="comma list to also run as a multi-seed ensemble (e.g. 42,1,7,13,100)")
    ap.add_argument("--ensemble-variant", default="aug_a")
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    results = {}
    for v in variants:
        for s in seeds:
            r = sweep_one(v, [s], args.horizon)
            if "error" not in r:
                results[f"{v}_seed{s}"] = r

    if args.ensemble_seeds:
        ens_seeds = [int(s) for s in args.ensemble_seeds.split(",")]
        r = sweep_one(args.ensemble_variant, ens_seeds, args.horizon)
        if "error" not in r:
            results[f"{args.ensemble_variant}_ensemble_{'_'.join(map(str,ens_seeds))}"] = r

    out_path = os.path.join(HERE, "threshold_sweep_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n--- Final summary saved to {out_path} ---")
    for key, r in results.items():
        b = r["best"]
        print(f"  {key:40s}: best=(T={b['T']}, δ={b['delta']}) sum={b['sum_cum_pnl']:+.4f}  "
              f"std={b['std_cum_pnl']:.4f}  pos={b['n_pos_folds']}/5", flush=True)


if __name__ == "__main__":
    main()
