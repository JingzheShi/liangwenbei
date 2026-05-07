"""T50 DE 4D threshold search on 5-seed-averaged OOF h_60 LightGBM-DART predictions.

For a given tag (default 'dart'), loads
    {tag}_pred_seed{S}_held{K}.parquet for S in {42,1,7,13,100}, K in 0..4
averages probs across 5 seeds per fold, then runs:
  1) Coarse asymmetric grid sweep (4D)
  2) Differential evolution (4D continuous) over (T_up, T_dn, d_up, d_dn) with 5 DE seeds
  3) Local neighborhood sanity check around DE optimum

Writes {tag}_thresh_results.json into HERE.

Schema is 1:1 the same as T37/T46 (parquet column names match).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.eval.pnl import _per_horizon_metrics  # noqa: E402

SEEDS_DEFAULT = [42, 1, 7, 13, 100]
N_FOLDS = 5
FEE = 0.0001
PROB_COLS = ["prob_0", "prob_1", "prob_2"]


def load_fold_avg(tag: str, k: int, seeds: List[int]) -> pd.DataFrame:
    base = pd.read_parquet(os.path.join(HERE, f"{tag}_pred_seed{seeds[0]}_held{k}.parquet"))
    p_sum = base[PROB_COLS].to_numpy(np.float64).copy()
    n = len(base)
    for s in seeds[1:]:
        df_s = pd.read_parquet(os.path.join(HERE, f"{tag}_pred_seed{s}_held{k}.parquet"))
        if len(df_s) != n:
            raise RuntimeError(f"row count mismatch tag={tag} fold={k} seed={s}: {len(df_s)} vs {n}")
        p_sum += df_s[PROB_COLS].to_numpy(np.float64)
    p_avg = p_sum / float(len(seeds))
    out = base.drop(columns=PROB_COLS).copy()
    out["prob_0"] = p_avg[:, 0].astype(np.float32)
    out["prob_1"] = p_avg[:, 1].astype(np.float32)
    out["prob_2"] = p_avg[:, 2].astype(np.float32)
    out["fold"] = np.int8(k)
    return out


def gate_asymmetric(probs: np.ndarray, T_up: float, T_dn: float,
                    d_up: float, d_dn: float) -> np.ndarray:
    p0 = probs[:, 0]; p1 = probs[:, 1]; p2 = probs[:, 2]
    take_up = (p2 >= T_up) & (p2 > p1 + d_up) & (p2 > p0)
    take_dn = (p0 >= T_dn) & (p0 > p1 + d_dn) & (p0 > p2)
    pred = np.full(probs.shape[0], 1, dtype=np.int8)
    pred[take_up] = 2
    pred[take_dn] = 0
    return pred


def vectorized_pnl(pred: np.ndarray, label: np.ndarray,
                   mp_t: np.ndarray, mp_th: np.ndarray) -> np.ndarray:
    side = pred.astype(np.float64) - 1.0
    abs_side = np.abs(side)
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee = FEE * abs_side * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee) / denom


@dataclass
class FoldArrays:
    fold: int
    probs: np.ndarray
    label: np.ndarray
    mp_t: np.ndarray
    mp_th: np.ndarray


def fold_arrays_from_dfs(folds: Dict[int, pd.DataFrame]) -> List[FoldArrays]:
    out = []
    for k in range(N_FOLDS):
        df = folds[k]
        out.append(FoldArrays(
            fold=k,
            probs=df[PROB_COLS].to_numpy(np.float32),
            label=df["true_label"].to_numpy(np.int64),
            mp_t=df["midprice_t"].to_numpy(np.float64),
            mp_th=df["midprice_th"].to_numpy(np.float64),
        ))
    return out


def coarse_sweep(fas):
    T_up = np.round(np.arange(0.40, 0.6501, 0.025), 4)
    T_dn = np.round(np.arange(0.40, 0.6501, 0.025), 4)
    d_up = np.round(np.arange(0.00, 0.2001, 0.025), 4)
    d_dn = np.round(np.arange(0.00, 0.2001, 0.025), 4)
    print(f"  coarse 4D grid: {len(T_up)*len(T_dn)*len(d_up)*len(d_dn)} combos", flush=True)
    rows = []
    label = [fa.label for fa in fas]
    mp_t = [fa.mp_t for fa in fas]
    mp_th = [fa.mp_th for fa in fas]
    probs = [fa.probs for fa in fas]
    for Tu in T_up:
        for Td in T_dn:
            for du in d_up:
                for dd in d_dn:
                    per = []
                    for k in range(5):
                        pred = gate_asymmetric(probs[k], float(Tu), float(Td), float(du), float(dd))
                        per.append(float(vectorized_pnl(pred, label[k], mp_t[k], mp_th[k]).sum()))
                    sum_p = sum(per)
                    rows.append({
                        "T_up": float(Tu), "T_dn": float(Td),
                        "d_up": float(du), "d_dn": float(dd),
                        "sum_cum_pnl": sum_p,
                        "n_pos_folds": int(sum(1 for x in per if x > 0)),
                        "per_fold": per,
                    })
    df = pd.DataFrame(rows).sort_values("sum_cum_pnl", ascending=False).reset_index(drop=True)
    return df


def make_objective(fas):
    label = [fa.label for fa in fas]
    mp_t = [fa.mp_t for fa in fas]
    mp_th = [fa.mp_th for fa in fas]
    probs = [fa.probs for fa in fas]

    def f(x):
        Tu, Td, du, dd = x
        s = 0.0
        for k in range(5):
            pred = gate_asymmetric(probs[k], Tu, Td, du, dd)
            s += vectorized_pnl(pred, label[k], mp_t[k], mp_th[k]).sum()
        return -float(s)
    return f


def de_search(fas, bounds, seeds=(0, 1, 2, 7, 42)):
    f = make_objective(fas)
    runs = []
    for seed in seeds:
        t_run = time.time()
        result = differential_evolution(
            f, bounds=bounds, seed=seed, maxiter=80, popsize=24,
            polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            updating="deferred", workers=1, init="sobol",
        )
        sum_pnl = -result.fun
        Tu, Td, du, dd = result.x
        per = []
        for k in range(5):
            pred = gate_asymmetric(fas[k].probs, Tu, Td, du, dd)
            per.append(float(vectorized_pnl(pred, fas[k].label, fas[k].mp_t, fas[k].mp_th).sum()))
        n_active = []
        for k in range(5):
            pred = gate_asymmetric(fas[k].probs, Tu, Td, du, dd)
            n_active.append(int((pred != 1).sum()))
        runs.append({
            "seed": int(seed),
            "T_up": float(Tu), "T_dn": float(Td),
            "d_up": float(du), "d_dn": float(dd),
            "sum_cum_pnl": float(sum_pnl),
            "per_fold_pnl": per,
            "n_active_per_fold": n_active,
            "n_pos_folds": int(sum(1 for x in per if x > 0)),
            "std_cum_pnl": float(np.std(per, ddof=0)),
            "nfev": int(result.nfev),
            "elapsed_sec": float(time.time() - t_run),
        })
        print(f"  DE seed={seed}: T_up={Tu:.4f} T_dn={Td:.4f} d_up={du:.4f} d_dn={dd:.4f} "
              f"-> sum={sum_pnl:+.4f}  per_fold={[round(x,3) for x in per]} "
              f"({result.nfev} evals, {time.time()-t_run:.1f}s)", flush=True)
    runs.sort(key=lambda r: r["sum_cum_pnl"], reverse=True)
    return runs


def neighborhood_check(fas, Tu0, Td0, du0, dd0, half_T=0.04, half_d=0.04, step=0.01):
    Tu_grid = np.round(np.arange(max(0.34, Tu0 - half_T), Tu0 + half_T + 1e-9, step), 4)
    Td_grid = np.round(np.arange(max(0.34, Td0 - half_T), Td0 + half_T + 1e-9, step), 4)
    du_grid = np.round(np.arange(max(0.0, du0 - half_d), min(0.30, du0 + half_d) + 1e-9, step), 4)
    dd_grid = np.round(np.arange(max(0.0, dd0 - half_d), min(0.30, dd0 + half_d) + 1e-9, step), 4)
    print(f"  neighborhood: {len(Tu_grid)}x{len(Td_grid)}x{len(du_grid)}x{len(dd_grid)} = "
          f"{len(Tu_grid)*len(Td_grid)*len(du_grid)*len(dd_grid)} combos", flush=True)
    rows = []
    label = [fa.label for fa in fas]
    mp_t = [fa.mp_t for fa in fas]
    mp_th = [fa.mp_th for fa in fas]
    probs = [fa.probs for fa in fas]
    for Tu in Tu_grid:
        for Td in Td_grid:
            for du in du_grid:
                for dd in dd_grid:
                    per = []
                    for k in range(5):
                        pred = gate_asymmetric(probs[k], float(Tu), float(Td), float(du), float(dd))
                        per.append(float(vectorized_pnl(pred, label[k], mp_t[k], mp_th[k]).sum()))
                    rows.append({
                        "T_up": float(Tu), "T_dn": float(Td),
                        "d_up": float(du), "d_dn": float(dd),
                        "sum_cum_pnl": float(sum(per)),
                        "n_pos_folds": int(sum(1 for x in per if x > 0)),
                    })
    return pd.DataFrame(rows).sort_values("sum_cum_pnl", ascending=False).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dart")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS_DEFAULT))
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"=== T50 DE thresh search tag={args.tag} seeds={seeds} ===", flush=True)

    t0 = time.time()
    folds = {k: load_fold_avg(args.tag, k, seeds) for k in range(N_FOLDS)}
    fas = fold_arrays_from_dfs(folds)
    print(f"  loaded {sum(len(f) for f in folds.values()):,} OOF rows in {time.time()-t0:.1f}s", flush=True)

    # Raw argmax baseline
    raw_pf = []
    for fa in fas:
        pred = fa.probs.argmax(axis=1).astype(np.int8)
        raw_pf.append(float(vectorized_pnl(pred, fa.label, fa.mp_t, fa.mp_th).sum()))
    print(f"  raw argmax sum={sum(raw_pf):+.4f}  per_fold={[round(x,3) for x in raw_pf]}", flush=True)

    print(f"\n[1/3] Coarse asymmetric sweep ...", flush=True)
    t1 = time.time()
    coarse_df = coarse_sweep(fas)
    print(f"  coarse done in {time.time()-t1:.1f}s. Top 10:", flush=True)
    print(coarse_df.head(10)[["T_up", "T_dn", "d_up", "d_dn", "sum_cum_pnl", "n_pos_folds"]].to_string(index=False))

    print(f"\n[2/3] DE 4D differential evolution ...", flush=True)
    bounds = [(0.34, 0.75), (0.34, 0.75), (0.0, 0.30), (0.0, 0.30)]
    de_runs = de_search(fas, bounds)
    best = de_runs[0]
    print(f"\n  Best DE run: sum={best['sum_cum_pnl']:+.4f} at "
          f"(T_up={best['T_up']:.4f}, T_dn={best['T_dn']:.4f}, "
          f"d_up={best['d_up']:.4f}, d_dn={best['d_dn']:.4f})", flush=True)
    print(f"  per_fold: {[round(x,3) for x in best['per_fold_pnl']]} "
          f"std={best['std_cum_pnl']:.4f} pos={best['n_pos_folds']}/5", flush=True)

    print(f"\n[3/3] Neighborhood sanity check around DE optimum ...", flush=True)
    nb = neighborhood_check(fas, best["T_up"], best["T_dn"], best["d_up"], best["d_dn"])
    print(f"  neighborhood top 10:", flush=True)
    print(nb.head(10).to_string(index=False))
    n_robust = int((nb["sum_cum_pnl"] >= best["sum_cum_pnl"] - 0.5).sum())
    print(f"  configs within 0.5 of optimum: {n_robust} of {len(nb)}", flush=True)

    out = {
        "tag": args.tag,
        "seeds": seeds,
        "raw_argmax_sum": float(sum(raw_pf)),
        "raw_argmax_per_fold": raw_pf,
        "coarse_top10": coarse_df.head(10).to_dict(orient="records"),
        "coarse_best_sum": float(coarse_df.iloc[0]["sum_cum_pnl"]),
        "de_runs": de_runs,
        "de_best": best,
        "neighborhood_top10": nb.head(10).to_dict(orient="records"),
        "neighborhood_n_within_0.5": int(n_robust),
        "elapsed_sec": float(time.time() - t0),
    }
    out_path = os.path.join(HERE, f"{args.tag}_thresh_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDone in {time.time()-t0:.1f}s. -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
