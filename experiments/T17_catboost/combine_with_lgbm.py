"""T17: Combine CatBoost OOF with LightGBM (T11) OOF and run threshold sweep.

Modes:
  - cat-only:   sweep on CatBoost OOF only (per CatBoost seed and CatBoost ensemble)
  - 2model-42:  average CatBoost seed=42 + LightGBM seed=42, sweep
  - 2pop:       average CatBoost-ensemble + LightGBM-ensemble, sweep
                (uses all available CatBoost seeds + all LightGBM seeds)
  - 10model:    per-model average across all (CatBoost x seeds) + (LightGBM x seeds)
                (equivalent to 2pop when each pop is uniform-averaged)
  - all:        run all of the above

Each mode produces:
  - sweep_results_<mode>_h{H}.csv    (full grid)
  - top10 print + per-fold breakdown for best
And writes combined results into:
  - combine_results.json
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

T11_DIR = os.path.join(ROOT, "experiments", "T11_schemeC_multiseed")

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
        pf.append({"sym": int(k), "cum_pnl": float(m["cum_pnl"]),
                   "n_active": int(m["n_active" if "n_active" in m else "n_predictions_active"]),
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


def sweep_one(fold_dfs, label):
    print(f"\n=== {label} ===", flush=True)
    n_total = sum(len(v) for v in fold_dfs.values())
    print(f"  total OOF rows: {n_total:,}", flush=True)
    base_pf = baseline_argmax(fold_dfs)
    base_sum = sum(r["cum_pnl"] for r in base_pf)
    print(f"  baseline argmax: sum={base_sum:+.4f} per_fold={[round(r['cum_pnl'],3) for r in base_pf]}",
          flush=True)
    sweep_df, folds = sweep_grid(fold_dfs, label)
    best = sweep_df.iloc[0].to_dict()
    print(f"  best (T={best['T']}, d={best['delta']}) sum={best['sum_cum_pnl']:+.4f} "
          f"pos={best['n_pos_folds']}/{len(folds)} active={best['sum_n_active']:,}", flush=True)
    print(f"  per-fold: {[round(best[f'fold{k}_pnl'], 3) for k in folds]}", flush=True)
    return {
        "label": label,
        "n_oof_total": int(n_total),
        "baseline_sum": float(base_sum),
        "baseline_per_fold": base_pf,
        "best_T": float(best["T"]),
        "best_delta": float(best["delta"]),
        "best_sum": float(best["sum_cum_pnl"]),
        "best_n_pos": int(best["n_pos_folds"]),
        "best_n_active": int(best["sum_n_active"]),
        "best_per_fold_pnl": [float(best[f"fold{k}_pnl"]) for k in folds],
        "best_mean_accuracy": float(best["mean_accuracy"]),
        "top10": sweep_df.head(10).to_dict(orient="records"),
    }


def load_cat_seed(seed: int, syms, H=10):
    out = {}
    for k in syms:
        p = os.path.join(HERE, f"loso_pred_cat_h{H}_seed{seed}_held{k}.parquet")
        if not os.path.exists(p):
            return None
        out[k] = pd.read_parquet(p)
    return out


def load_lgbm_seed(seed: int, syms, H=10):
    out = {}
    for k in syms:
        p = os.path.join(T11_DIR, f"loso_pred_h{H}_seed{seed}_held{k}.parquet")
        if not os.path.exists(p):
            return None
        out[k] = pd.read_parquet(p)
    return out


def assert_aligned(dfs_list, k):
    """Make sure all dfs_list[i][k] share the same key ordering."""
    base = dfs_list[0][k][["sym", "date", "session", "t"]].to_numpy()
    base_y = dfs_list[0][k]["true_label"].to_numpy()
    for j, dfs in enumerate(dfs_list[1:], 1):
        cur = dfs[k][["sym", "date", "session", "t"]].to_numpy()
        cur_y = dfs[k]["true_label"].to_numpy()
        if not np.array_equal(base, cur):
            sys.exit(f"key misalignment at component {j} fold {k}")
        if not np.array_equal(base_y, cur_y):
            sys.exit(f"label misalignment at component {j} fold {k}")


def average_pred_dfs(dfs_list, syms):
    """Average prob_0/1/2 across each component dict in dfs_list, return aligned dict."""
    out = {}
    for k in syms:
        assert_aligned(dfs_list, k)
        prob_stack = np.stack([
            d[k][["prob_0", "prob_1", "prob_2"]].to_numpy(np.float32) for d in dfs_list
        ], axis=0)
        prob_avg = prob_stack.mean(axis=0).astype(np.float32)
        ref = dfs_list[0][k]
        out[k] = pd.DataFrame({
            "sym": ref["sym"].values,
            "date": ref["date"].values,
            "session": ref["session"].values,
            "t": ref["t"].values,
            "true_label": ref["true_label"].values,
            "pred_label": prob_avg.argmax(1).astype(np.int8),
            "prob_0": prob_avg[:, 0],
            "prob_1": prob_avg[:, 1],
            "prob_2": prob_avg[:, 2],
            "midprice_t": ref["midprice_t"].values,
            "midprice_th": ref["midprice_th"].values,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--syms", default="0,1,2,3,4")
    ap.add_argument("--cat-seeds", default="42",
                    help="CatBoost seeds for which OOF parquets exist")
    ap.add_argument("--lgbm-seeds", default="42,1,7,13,100",
                    help="LightGBM seeds for which T11 OOF parquets exist")
    ap.add_argument("--modes", default="cat-only,2model-42,2pop",
                    help="comma-separated mode names")
    args = ap.parse_args()

    H = args.horizon
    syms = [int(x) for x in args.syms.split(",")]
    cat_seeds = [int(x) for x in args.cat_seeds.split(",")]
    lgbm_seeds = [int(x) for x in args.lgbm_seeds.split(",")]
    modes = [m.strip() for m in args.modes.split(",")]
    print(f"H={H} syms={syms} cat_seeds={cat_seeds} lgbm_seeds={lgbm_seeds} modes={modes}", flush=True)

    # Load all available CatBoost seeds
    cat_loaded = {}
    for s in cat_seeds:
        d = load_cat_seed(s, syms, H)
        if d is None:
            print(f"  [warn] CatBoost seed={s} not fully available, skipping", flush=True)
            continue
        cat_loaded[s] = d
        print(f"  CatBoost seed={s} loaded ({len(d)} folds)", flush=True)
    lgbm_loaded = {}
    for s in lgbm_seeds:
        d = load_lgbm_seed(s, syms, H)
        if d is None:
            print(f"  [warn] LightGBM seed={s} not fully available, skipping", flush=True)
            continue
        lgbm_loaded[s] = d
        print(f"  LightGBM seed={s} loaded ({len(d)} folds)", flush=True)

    out = {
        "task": "T17 CatBoost + LightGBM combine sweeps",
        "horizon": H,
        "cat_seeds": list(cat_loaded.keys()),
        "lgbm_seeds": list(lgbm_loaded.keys()),
        "T_grid": T_GRID, "delta_grid": D_GRID,
        "modes": {},
    }

    # Mode: cat-only — per seed + ensemble of CatBoost seeds
    if "cat-only" in modes:
        for s, d in cat_loaded.items():
            r = sweep_one(d, f"cat_seed{s}_h{H}")
            out["modes"][f"cat_seed{s}"] = r
        if len(cat_loaded) > 1:
            ens = average_pred_dfs(list(cat_loaded.values()), syms)
            r = sweep_one(ens, f"cat_ensemble_h{H}")
            out["modes"]["cat_ensemble"] = r

    # Mode: 2model-42 — CatBoost seed=42 + LightGBM seed=42 averaged
    if "2model-42" in modes:
        if 42 in cat_loaded and 42 in lgbm_loaded:
            avg = average_pred_dfs([cat_loaded[42], lgbm_loaded[42]], syms)
            r = sweep_one(avg, f"cat42+lgbm42_h{H}")
            out["modes"]["2model_seed42"] = r
        else:
            print("  [skip] 2model-42 needs both seed=42", flush=True)

    # Mode: 2pop — average all CatBoost (uniform across seeds) + all LightGBM (uniform across seeds)
    # then average those two pops 50/50
    if "2pop" in modes:
        if cat_loaded and lgbm_loaded:
            cat_pop = average_pred_dfs(list(cat_loaded.values()), syms)
            lgbm_pop = average_pred_dfs(list(lgbm_loaded.values()), syms)
            mix = average_pred_dfs([cat_pop, lgbm_pop], syms)
            r = sweep_one(mix, f"2pop_cat{len(cat_loaded)}+lgbm{len(lgbm_loaded)}_h{H}")
            out["modes"]["2pop_average"] = r

    # Mode: 10model — per-model uniform average across all CatBoost+LightGBM seeds
    if "10model" in modes:
        if cat_loaded and lgbm_loaded:
            all_dfs = list(cat_loaded.values()) + list(lgbm_loaded.values())
            mix = average_pred_dfs(all_dfs, syms)
            n_models = len(all_dfs)
            r = sweep_one(mix, f"{n_models}model_uniform_h{H}")
            out["modes"][f"{n_models}model_uniform"] = r

    out_path = os.path.join(HERE, "combine_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n--> {out_path}", flush=True)

    # Print final summary
    print(f"\n{'='*78}\n=== FINAL SUMMARY ===\n{'='*78}", flush=True)
    print(f"{'mode':35s} {'argmax':>10s} {'best':>10s} {'(T,d)':>14s} {'pos':>5s} {'active':>10s}", flush=True)
    for name, r in out["modes"].items():
        T_d = f"({r['best_T']:.2f},{r['best_delta']:.2f})"
        print(f"{name:35s} {r['baseline_sum']:>+10.4f} {r['best_sum']:>+10.4f} {T_d:>14s} "
              f"{r['best_n_pos']}/5 {r['best_n_active']:>10,d}", flush=True)


if __name__ == "__main__":
    main()
