"""T35: Threshold sweep on Scheme K LOSO predictions.

For a given (variant, seeds, horizon), loads loso_pred_h{H}_{variant}_seed{S}_held{K}.parquet
and:
  1) computes per-seed best (T, delta) and the raw-argmax baseline
  2) if multiple seeds, computes the seed-mean ensemble probs and sweeps that

Outputs sweep_{tag}.csv and threshold_results_{tag}.json.
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
    }


def baseline_argmax(fold_dfs):
    pf = []
    for k in sorted(fold_dfs):
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
    folds = sorted(fold_dfs)
    for T in T_GRID:
        for d in D_GRID:
            per_fold = [per_fold_metrics(fold_dfs[k], T, d) for k in folds]
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
    out_csv = os.path.join(HERE, f"sweep_{label_for_csv}.csv")
    sweep_df.to_csv(out_csv, index=False)
    return sweep_df, folds


def sweep_one(fold_dfs, label_for_csv):
    print(f"\n=== sweeping {label_for_csv} ===", flush=True)
    n_total = sum(len(v) for v in fold_dfs.values())
    print(f"  total OOF rows: {n_total:,}", flush=True)

    base_pf = baseline_argmax(fold_dfs)
    base_sum = sum(r["cum_pnl"] for r in base_pf)
    print(
        f"  baseline argmax sum = {base_sum:+.4f} "
        f"per_fold={[round(r['cum_pnl'], 3) for r in base_pf]}",
        flush=True,
    )

    sweep_df, folds = sweep_grid(fold_dfs, label_for_csv)
    best = sweep_df.iloc[0].to_dict()
    print(f"  best (T={best['T']}, d={best['delta']}) sum={best['sum_cum_pnl']:+.4f} "
          f"pos={best['n_pos_folds']}/{len(folds)}", flush=True)
    print(f"  per-fold @best: {[round(best[f'fold{k}_pnl'], 3) for k in folds]}",
          flush=True)

    return {
        "n_oof_total": int(n_total),
        "baseline_argmax_sum_cum_pnl": float(base_sum),
        "baseline_per_fold": base_pf,
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


def ensemble_seed_dfs(per_seed_dfs):
    """Mean-prob ensemble across seeds per held-out sym."""
    folds = sorted(next(iter(per_seed_dfs.values())))
    out = {}
    for k in folds:
        prob_acc = None
        ref_df = None
        n = 0
        for s, dfs in per_seed_dfs.items():
            df = dfs[k]
            p = df[["prob_0", "prob_1", "prob_2"]].to_numpy(np.float32)
            prob_acc = p if prob_acc is None else prob_acc + p
            ref_df = df
            n += 1
        prob_avg = prob_acc / float(n)
        ens = ref_df.copy()
        ens["prob_0"] = prob_avg[:, 0]
        ens["prob_1"] = prob_avg[:, 1]
        ens["prob_2"] = prob_avg[:, 2]
        out[k] = ens
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--variant", default="baseline")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--syms", default="0,1,2,3,4")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    H = args.horizon
    seeds = [int(x) for x in args.seeds.split(",")]
    target_syms = [int(x) for x in args.syms.split(",")]
    tag = args.tag or f"h{H}_{args.variant}_{'-'.join(map(str, seeds))}seed"

    per_seed_dfs = {}
    seed_results = {}
    for s in seeds:
        sd = {}
        for k in target_syms:
            p = os.path.join(HERE, f"loso_pred_h{H}_{args.variant}_seed{s}_held{k}.parquet")
            if not os.path.exists(p):
                sys.exit(f"missing {p}")
            sd[k] = pd.read_parquet(p)
        per_seed_dfs[s] = sd
        seed_results[s] = sweep_one(sd, f"seed{s}_h{H}_{args.variant}")

    out = {
        "horizon": H, "variant": args.variant, "seeds": seeds,
        "T_grid": T_GRID, "delta_grid": D_GRID,
        "per_seed": {f"seed_{s}": seed_results[s] for s in seeds},
    }
    if len(seeds) > 1:
        ens_dfs = ensemble_seed_dfs(per_seed_dfs)
        out["ensemble"] = sweep_one(ens_dfs, f"ensemble_h{H}_{args.variant}")

    out_path = os.path.join(HERE, f"threshold_results_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nthreshold_results -> {out_path}", flush=True)

    print(f"\n{'='*78}\n=== SUMMARY tag={tag} ===\n{'='*78}", flush=True)
    for s in seeds:
        r = seed_results[s]
        print(
            f"  seed{s}: argmax={r['baseline_argmax_sum_cum_pnl']:+.4f}  "
            f"best=(T={r['best']['T']:.2f},d={r['best']['delta']:.2f}) -> "
            f"{r['best']['sum_cum_pnl']:+.4f}  pos={r['best']['n_pos_folds']}/5",
            flush=True,
        )
    if len(seeds) > 1:
        e = out["ensemble"]
        print(
            f"  ENS:   argmax={e['baseline_argmax_sum_cum_pnl']:+.4f}  "
            f"best=(T={e['best']['T']:.2f},d={e['best']['delta']:.2f}) -> "
            f"{e['best']['sum_cum_pnl']:+.4f}  pos={e['best']['n_pos_folds']}/5",
            flush=True,
        )


if __name__ == "__main__":
    main()
