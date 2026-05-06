"""T32: Combine NN h_60 OOF with LightGBM 5-seed aug_a OOF and threshold sweep.

Loads:
  T32 OOFs: experiments/T32_nn_h60/loso_pred_nn_h60_held{K}.parquet  (K in 0..4)
  LightGBM OOFs:
    seed 42:  experiments/T26_domain_randomization/loso_pred_h60_aug_a_held{K}.parquet
    seeds 1,7,13,100: experiments/T27_iter005/loso_pred_h60_aug_a_seed{S}_held{K}.parquet

For each fold:
  - Inner-join NN <-> LightGBM by (sym, date, session, t)
  - Per-seed LightGBM ensemble = mean of 5 prob arrays
  - Final ensemble = (NN_probs + LightGBM_5seed_avg) / 2
    (alternative: weighted; also report 3:1 LGB:NN, etc.)

Then:
  - Threshold sweep on (T, delta) ∈ T_GRID × D_GRID over the 5-fold concat
  - Report best, top-K
  - Compare:
      LGB-only 5-seed best     (T27 reported = +11.46 at T=0.45,d=0.10)
      NN-only LOSO best
      6-model ensemble best (50/50)
      Weighted ensemble best (α=NN weight, varied)
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
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sweep_utils import thresholded_metrics  # noqa: E402

T26_DIR = os.path.join(ROOT, "experiments", "T26_domain_randomization")
T27_DIR = os.path.join(ROOT, "experiments", "T27_iter005")

T_GRID = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
D_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]


def load_lgb_seed_fold(seed: int, held: int) -> pd.DataFrame:
    if seed == 42:
        p = os.path.join(T26_DIR, f"loso_pred_h60_aug_a_held{held}.parquet")
    else:
        p = os.path.join(T27_DIR, f"loso_pred_h60_aug_a_seed{seed}_held{held}.parquet")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return pd.read_parquet(p)


def load_nn_fold(held: int) -> pd.DataFrame:
    p = os.path.join(HERE, f"loso_pred_nn_h60_held{held}.parquet")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return pd.read_parquet(p)


def per_fold_blob(held: int, lgb_seeds: list[int]):
    """Aligned per-fold ensemble: returns dict with NN probs, LGB-avg probs, joined meta."""
    nn = load_nn_fold(held).rename(columns={
        "prob_0": "nn_p0", "prob_1": "nn_p1", "prob_2": "nn_p2",
        "pred_label": "nn_pred",
    })
    keys = ["sym", "date", "session", "t"]
    # LGB seeds — average prob first, then merge
    lgb_per = []
    meta_lgb = None
    for s in lgb_seeds:
        d = load_lgb_seed_fold(s, held)
        lgb_per.append(d[["sym", "date", "session", "t", "prob_0", "prob_1", "prob_2"]].rename(
            columns={"prob_0": f"lgb{s}_p0", "prob_1": f"lgb{s}_p1", "prob_2": f"lgb{s}_p2"}
        ))
        if meta_lgb is None:
            meta_lgb = d[keys + ["true_label", "midprice_t", "midprice_th"]].copy()
    lgb_merged = meta_lgb
    for d in lgb_per:
        lgb_merged = lgb_merged.merge(d, on=keys, how="inner", validate="one_to_one")
    # Average lgb probs
    lgb_p0 = np.mean(np.stack([lgb_merged[f"lgb{s}_p0"].to_numpy(np.float32) for s in lgb_seeds], axis=0), axis=0)
    lgb_p1 = np.mean(np.stack([lgb_merged[f"lgb{s}_p1"].to_numpy(np.float32) for s in lgb_seeds], axis=0), axis=0)
    lgb_p2 = np.mean(np.stack([lgb_merged[f"lgb{s}_p2"].to_numpy(np.float32) for s in lgb_seeds], axis=0), axis=0)
    lgb_merged["lgb_p0"] = lgb_p0
    lgb_merged["lgb_p1"] = lgb_p1
    lgb_merged["lgb_p2"] = lgb_p2
    lgb_merged = lgb_merged[keys + ["true_label", "midprice_t", "midprice_th", "lgb_p0", "lgb_p1", "lgb_p2"]]

    # Inner-join NN and LGB
    df = lgb_merged.merge(nn[keys + ["nn_p0", "nn_p1", "nn_p2"]], on=keys, how="inner", validate="one_to_one")
    return df


def sweep_probs(probs: np.ndarray, true_lab: np.ndarray, mp_t: np.ndarray, mp_th: np.ndarray,
                fold_split: list[int], label: str) -> pd.DataFrame:
    """Sweep T_GRID x D_GRID, with per-fold breakdown. fold_split = list of n_rows per fold (cum-sum)."""
    rows = []
    fold_starts = [0] + np.cumsum(fold_split).tolist()
    for T in T_GRID:
        for d in D_GRID:
            per_fold = []
            for i in range(len(fold_split)):
                a, b = fold_starts[i], fold_starts[i + 1]
                if b <= a:
                    per_fold.append({"cum_pnl": 0.0, "n_active": 0})
                    continue
                r = thresholded_metrics(
                    probs[a:b], true_lab[a:b], mp_t[a:b], mp_th[a:b],
                    T=T, delta=d,
                )
                per_fold.append(r)
            sum_cum = sum(r["cum_pnl"] for r in per_fold)
            std_pnl = float(np.std([r["cum_pnl"] for r in per_fold], ddof=0))
            n_pos = int(sum(1 for r in per_fold if r["cum_pnl"] > 0))
            sum_active = int(sum(r["n_active"] for r in per_fold))
            row = {
                "label": label, "T": T, "delta": d, "sum_cum_pnl": sum_cum,
                "mean_cum_pnl": sum_cum / max(len(per_fold), 1),
                "std_cum_pnl": std_pnl, "n_pos_folds": n_pos,
                "sum_n_active": sum_active,
            }
            for i in range(len(fold_split)):
                row[f"fold{i}_pnl"] = per_fold[i]["cum_pnl"]
            rows.append(row)
    return pd.DataFrame(rows).sort_values("sum_cum_pnl", ascending=False).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lgb_seeds", default="42,1,7,13,100")
    ap.add_argument("--syms", default="0,1,2,3,4")
    ap.add_argument("--alphas", default="0.0,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0",
                    help="NN weight α; final = α*NN + (1-α)*LGB")
    args = ap.parse_args()

    lgb_seeds = [int(s) for s in args.lgb_seeds.split(",")]
    target_syms = [int(s) for s in args.syms.split(",")]
    alphas = [float(a) for a in args.alphas.split(",")]

    print(f"=== T32 combine NN h_60 + LGB {lgb_seeds}-seed aug_a ===", flush=True)
    t0 = time.time()

    # Build per-fold dfs with NN + LGB-avg probs
    dfs = []
    for k in target_syms:
        df = per_fold_blob(k, lgb_seeds)
        print(f"  fold {k}: {len(df):,} rows after NN<->LGB inner-join")
        dfs.append(df)

    # Concat
    big = pd.concat(dfs, axis=0, ignore_index=True)
    fold_split = [len(d) for d in dfs]
    true_lab = big["true_label"].to_numpy()
    mp_t = big["midprice_t"].to_numpy()
    mp_th = big["midprice_th"].to_numpy()

    nn_probs = big[["nn_p0", "nn_p1", "nn_p2"]].to_numpy(np.float32)
    lgb_probs = big[["lgb_p0", "lgb_p1", "lgb_p2"]].to_numpy(np.float32)

    summaries = {}

    # 1) NN-only
    df_nn = sweep_probs(nn_probs, true_lab, mp_t, mp_th, fold_split, label="NN_only")
    print("\n--- NN-only top 5 ---")
    print(df_nn.head(5).to_string(index=False))
    summaries["nn_only"] = df_nn.iloc[0].to_dict()

    # 2) LGB-only (5-seed avg)
    df_lgb = sweep_probs(lgb_probs, true_lab, mp_t, mp_th, fold_split, label="LGB_5seed")
    print("\n--- LGB 5-seed top 5 ---")
    print(df_lgb.head(5).to_string(index=False))
    summaries["lgb_5seed"] = df_lgb.iloc[0].to_dict()

    # 3) Ensemble per α
    all_alpha_dfs = []
    for a in alphas:
        ens = a * nn_probs + (1.0 - a) * lgb_probs
        df_e = sweep_probs(ens, true_lab, mp_t, mp_th, fold_split, label=f"ENS_alpha={a:.2f}")
        print(f"\n--- ENS alpha={a:.2f} top 3 ---")
        print(df_e.head(3).to_string(index=False))
        summaries[f"ens_alpha_{a:.2f}"] = df_e.iloc[0].to_dict()
        all_alpha_dfs.append(df_e)

    # Save full sweep
    full = pd.concat([df_nn, df_lgb] + all_alpha_dfs, axis=0, ignore_index=True)
    out_csv = os.path.join(HERE, "combine_sweep.csv")
    full.to_csv(out_csv, index=False)
    print(f"\nSaved sweep CSV -> {out_csv}", flush=True)

    # Best across all alphas
    best_overall = max(summaries.items(), key=lambda kv: kv[1]["sum_cum_pnl"])
    print(f"\nBEST OVERALL: {best_overall[0]} -> "
          f"T={best_overall[1]['T']} d={best_overall[1]['delta']} "
          f"sum={best_overall[1]['sum_cum_pnl']:+.4f}")

    out_json = os.path.join(HERE, "combine_results.json")
    with open(out_json, "w") as f:
        json.dump({
            "lgb_seeds": lgb_seeds,
            "alphas": alphas,
            "summaries": {k: {kk: float(vv) if isinstance(vv, (int, float, np.floating, np.integer)) else vv
                              for kk, vv in v.items()} for k, v in summaries.items()},
            "best": {
                "label": best_overall[0],
                **{kk: float(vv) if isinstance(vv, (int, float, np.floating, np.integer)) else vv
                   for kk, vv in best_overall[1].items()},
            },
            "elapsed_sec": time.time() - t0,
        }, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else int(o) if isinstance(o, np.integer) else str(o))
    print(f"Saved combine results -> {out_json}")


if __name__ == "__main__":
    main()
