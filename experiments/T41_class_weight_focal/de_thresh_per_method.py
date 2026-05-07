"""T41 per-method DE 4D thresh search on single-seed OOF (seed=42).

For each method in --methods, loads
    loso_pred_h60_<method>_seed42_held{K}.parquet  for K in 0..4
and runs:
  1) Coarse asymmetric grid sweep
  2) DE 4D over (T_up, T_dn, d_up, d_dn) with multiple seeds
  3) Local neighborhood sanity check around DE optimum

Writes per_method_thresh_results.json into HERE, with one entry per method.
Also prints a ranked summary at the end.
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

N_FOLDS = 5
H = 60
SEED = 42
FEE = 0.0001
PROB_COLS = ["prob_0", "prob_1", "prob_2"]


def load_method_folds(method: str) -> Dict[int, pd.DataFrame]:
    folds = {}
    for k in range(N_FOLDS):
        p = os.path.join(HERE, f"loso_pred_h{H}_{method}_seed{SEED}_held{k}.parquet")
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
        df = pd.read_parquet(p)
        folds[k] = df
    return folds


def gate_asymmetric(probs, T_up, T_dn, d_up, d_dn):
    p0 = probs[:, 0]; p1 = probs[:, 1]; p2 = probs[:, 2]
    take_up = (p2 >= T_up) & (p2 > p1 + d_up) & (p2 > p0)
    take_dn = (p0 >= T_dn) & (p0 > p1 + d_dn) & (p0 > p2)
    pred = np.full(probs.shape[0], 1, dtype=np.int8)
    pred[take_up] = 2
    pred[take_dn] = 0
    return pred


def vectorized_pnl(pred, label, mp_t, mp_th):
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


def coarse_sweep(fas, T_grid=None, d_grid=None):
    if T_grid is None:
        T_grid = np.round(np.arange(0.40, 0.6501, 0.025), 4)
    if d_grid is None:
        d_grid = np.round(np.arange(0.00, 0.2001, 0.025), 4)
    probs = [fa.probs for fa in fas]
    label = [fa.label for fa in fas]
    mp_t = [fa.mp_t for fa in fas]
    mp_th = [fa.mp_th for fa in fas]
    rows = []
    for Tu in T_grid:
        for Td in T_grid:
            for du in d_grid:
                for dd in d_grid:
                    per = []
                    for k in range(N_FOLDS):
                        pred = gate_asymmetric(probs[k], float(Tu), float(Td), float(du), float(dd))
                        per.append(float(vectorized_pnl(pred, label[k], mp_t[k], mp_th[k]).sum()))
                    rows.append({
                        "T_up": float(Tu), "T_dn": float(Td),
                        "d_up": float(du), "d_dn": float(dd),
                        "sum_cum_pnl": sum(per),
                        "n_pos_folds": int(sum(1 for x in per if x > 0)),
                        "per_fold": per,
                    })
    return pd.DataFrame(rows).sort_values("sum_cum_pnl", ascending=False).reset_index(drop=True)


def make_objective(fas):
    label = [fa.label for fa in fas]
    mp_t = [fa.mp_t for fa in fas]
    mp_th = [fa.mp_th for fa in fas]
    probs = [fa.probs for fa in fas]

    def f(x):
        Tu, Td, du, dd = x
        s = 0.0
        for k in range(N_FOLDS):
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
        n_active = []
        for k in range(N_FOLDS):
            pred = gate_asymmetric(fas[k].probs, Tu, Td, du, dd)
            per.append(float(vectorized_pnl(pred, fas[k].label, fas[k].mp_t, fas[k].mp_th).sum()))
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
    runs.sort(key=lambda r: r["sum_cum_pnl"], reverse=True)
    return runs


def neighborhood_check(fas, Tu0, Td0, du0, dd0, half_T=0.04, half_d=0.04, step=0.01):
    Tu_grid = np.round(np.arange(max(0.34, Tu0 - half_T), Tu0 + half_T + 1e-9, step), 4)
    Td_grid = np.round(np.arange(max(0.34, Td0 - half_T), Td0 + half_T + 1e-9, step), 4)
    du_grid = np.round(np.arange(max(0.0, du0 - half_d), min(0.30, du0 + half_d) + 1e-9, step), 4)
    dd_grid = np.round(np.arange(max(0.0, dd0 - half_d), min(0.30, dd0 + half_d) + 1e-9, step), 4)
    probs = [fa.probs for fa in fas]
    label = [fa.label for fa in fas]
    mp_t = [fa.mp_t for fa in fas]
    mp_th = [fa.mp_th for fa in fas]
    rows = []
    for Tu in Tu_grid:
        for Td in Td_grid:
            for du in du_grid:
                for dd in dd_grid:
                    per = []
                    for k in range(N_FOLDS):
                        pred = gate_asymmetric(probs[k], float(Tu), float(Td), float(du), float(dd))
                        per.append(float(vectorized_pnl(pred, label[k], mp_t[k], mp_th[k]).sum()))
                    rows.append({
                        "T_up": float(Tu), "T_dn": float(Td),
                        "d_up": float(du), "d_dn": float(dd),
                        "sum_cum_pnl": float(sum(per)),
                        "n_pos_folds": int(sum(1 for x in per if x > 0)),
                    })
    return pd.DataFrame(rows).sort_values("sum_cum_pnl", ascending=False).reset_index(drop=True)


def process_method(method: str) -> dict:
    print(f"\n{'='*78}\n=== METHOD {method} ===\n{'='*78}", flush=True)
    t0 = time.time()
    folds = load_method_folds(method)
    fas = fold_arrays_from_dfs(folds)
    n_rows = sum(len(f) for f in folds.values())
    print(f"  loaded {n_rows:,} OOF rows in {time.time()-t0:.1f}s", flush=True)

    raw_pf = []
    for fa in fas:
        pred = fa.probs.argmax(axis=1).astype(np.int8)
        raw_pf.append(float(vectorized_pnl(pred, fa.label, fa.mp_t, fa.mp_th).sum()))
    raw_sum = sum(raw_pf)
    print(f"  raw argmax sum={raw_sum:+.4f}  per_fold={[round(x,3) for x in raw_pf]}", flush=True)

    print(f"\n[1/3] Coarse asymmetric sweep ...", flush=True)
    t1 = time.time()
    coarse_df = coarse_sweep(fas)
    print(f"  coarse done in {time.time()-t1:.1f}s. Top 5:", flush=True)
    print(coarse_df.head(5)[["T_up", "T_dn", "d_up", "d_dn", "sum_cum_pnl", "n_pos_folds"]].to_string(index=False))

    print(f"\n[2/3] DE 4D ...", flush=True)
    bounds = [(0.34, 0.75), (0.34, 0.75), (0.0, 0.30), (0.0, 0.30)]
    de_runs = de_search(fas, bounds)
    best = de_runs[0]
    print(f"  Best DE: sum={best['sum_cum_pnl']:+.4f} at "
          f"T_up={best['T_up']:.4f} T_dn={best['T_dn']:.4f} "
          f"d_up={best['d_up']:.4f} d_dn={best['d_dn']:.4f}", flush=True)
    print(f"  per_fold: {[round(x,3) for x in best['per_fold_pnl']]} "
          f"std={best['std_cum_pnl']:.4f} pos={best['n_pos_folds']}/{N_FOLDS}", flush=True)
    print(f"  n_active: {best['n_active_per_fold']} (sum={sum(best['n_active_per_fold'])})", flush=True)

    print(f"\n[3/3] Neighborhood check ...", flush=True)
    nb = neighborhood_check(fas, best["T_up"], best["T_dn"], best["d_up"], best["d_dn"])
    n_robust = int((nb["sum_cum_pnl"] >= best["sum_cum_pnl"] - 0.5).sum())
    print(f"  configs within 0.5 of optimum: {n_robust}/{len(nb)}", flush=True)

    return {
        "method": method,
        "raw_argmax_sum": float(raw_sum),
        "raw_argmax_per_fold": raw_pf,
        "coarse_top10": coarse_df.head(10).to_dict(orient="records"),
        "coarse_best_sum": float(coarse_df.iloc[0]["sum_cum_pnl"]),
        "de_runs": de_runs,
        "de_best": best,
        "neighborhood_top10": nb.head(10).to_dict(orient="records"),
        "neighborhood_n_within_0.5": int(n_robust),
        "elapsed_sec": float(time.time() - t0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", required=True,
                    help="comma list of method tags matching loso_pred_h60_<m>_seed42_held*.parquet")
    ap.add_argument("--out", default=os.path.join(HERE, "per_method_thresh_results.json"))
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    print(f"=== T41 per-method DE thresh search methods={methods} ===", flush=True)

    t0 = time.time()
    results = []
    for m in methods:
        try:
            results.append(process_method(m))
        except FileNotFoundError as e:
            print(f"  SKIP {m}: missing {e}", flush=True)

    print(f"\n{'='*78}\n=== RANKED SUMMARY ===\n{'='*78}", flush=True)
    ranked = sorted(results, key=lambda r: r["de_best"]["sum_cum_pnl"], reverse=True)
    for r in ranked:
        b = r["de_best"]
        print(
            f"  {r['method']:24s} DE_sum={b['sum_cum_pnl']:+8.4f} "
            f"raw_argmax={r['raw_argmax_sum']:+8.4f} "
            f"pos={b['n_pos_folds']}/{N_FOLDS} std={b['std_cum_pnl']:.3f} "
            f"n_active={sum(b['n_active_per_fold']):>6} "
            f"T_up={b['T_up']:.3f} T_dn={b['T_dn']:.3f} d_up={b['d_up']:.3f} d_dn={b['d_dn']:.3f}",
            flush=True,
        )

    out = {"methods": methods, "results": results, "elapsed_sec": time.time() - t0}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDone in {time.time()-t0:.1f}s. -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
