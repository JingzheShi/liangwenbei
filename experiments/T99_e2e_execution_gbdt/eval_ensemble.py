"""T99: Ensemble evaluation — combine T99 model with T87/T89/T75 baselines.

Input: a list of (label, parquet_paths_glob, weight) tuples.
For each combo, compute DE asym LOSO PnL.
Also compute pairwise cross-correlation matrix on test predictions.

Usage:
  python eval_ensemble.py --combos huber_a0.001 cb_huber_a0.001 ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from glob import glob

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001


def load_avg_pred_from_paths(paths):
    base = pd.read_parquet(paths[0])
    p_sum = base["pred_dmid_norm"].to_numpy(np.float64).copy()
    n = len(base)
    for p in paths[1:]:
        df = pd.read_parquet(p)
        if len(df) != n:
            raise RuntimeError(f"len mismatch {p}")
        if not (df["sym"].equals(base["sym"]) and df["date"].equals(base["date"])
                and df["t"].equals(base["t"])):
            raise RuntimeError(f"order mismatch {p}")
        p_sum += df["pred_dmid_norm"].to_numpy(np.float64)
    return base, p_sum / float(len(paths))


def vectorized_pnl(pred_action, mp_t, mp_th):
    side = pred_action.astype(np.float64) - 1.0
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee_pnl) / denom


def ev_gate_asymmetric(pred, thr_up, thr_dn):
    a = np.full(len(pred), 1, dtype=np.int8)
    a[pred > thr_up] = 2
    a[pred < -thr_dn] = 0
    return a


def make_obj_loso_per_sym(pred, mp_t, mp_th, sym_arr):
    fold_pred = []
    fold_mpt = []
    fold_mpth = []
    for k in SYMS:
        m = sym_arr == k
        fold_pred.append(pred[m])
        fold_mpt.append(mp_t[m])
        fold_mpth.append(mp_th[m])

    def f(x):
        thr_up, thr_dn = x
        s = 0.0
        for i in range(len(SYMS)):
            a = ev_gate_asymmetric(fold_pred[i], thr_up, thr_dn)
            s += vectorized_pnl(a, fold_mpt[i], fold_mpth[i]).sum()
        return -float(s)
    return f, fold_pred, fold_mpt, fold_mpth


def de_search(obj_fn, bounds, seeds=(0, 1, 2, 7, 42), maxiter=80, popsize=24):
    runs = []
    for sd in seeds:
        result = differential_evolution(
            obj_fn, bounds=bounds, seed=sd, maxiter=maxiter, popsize=popsize,
            polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            updating="deferred", workers=1, init="sobol",
        )
        runs.append({"seed": int(sd), "thr_up": float(result.x[0]),
                     "thr_dn": float(result.x[1]), "obj_val": float(-result.fun)})
    runs.sort(key=lambda r: r["obj_val"], reverse=True)
    return runs


def eval_combo(label, pred, base, bounds=(0.0, 0.0040), de_seeds=(0,1,2,7,42)):
    sym_arr = base["sym"].to_numpy()
    mp_t = base["midprice_t"].to_numpy(np.float64)
    mp_th = base["midprice_th"].to_numpy(np.float64)

    obj_fn, fold_pred, fold_mpt, fold_mpth = make_obj_loso_per_sym(
        pred, mp_t, mp_th, sym_arr)
    runs = de_search(obj_fn, [bounds, bounds], seeds=de_seeds)
    best = runs[0]
    per_sym = []
    for i in range(len(SYMS)):
        a = ev_gate_asymmetric(fold_pred[i], best["thr_up"], best["thr_dn"])
        per_sym.append(float(vectorized_pnl(a, fold_mpt[i], fold_mpth[i]).sum()))
    return {
        "label": label,
        "thr_up": best["thr_up"], "thr_dn": best["thr_dn"],
        "de_sum": float(sum(per_sym)),
        "per_sym": per_sym,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ensemble_results.json")
    ap.add_argument("--bounds-hi", type=float, default=0.0040)
    args = ap.parse_args()

    # Define streams
    T75_DIR = os.path.join(ROOT, "experiments", "T75_regression_dmid")
    T87_DIR = os.path.join(ROOT, "experiments", "T87_spo_dfl")
    T89_DIR = os.path.join(ROOT, "experiments", "T89_catboost_regression")

    seeds = [1, 7, 13, 42, 100]

    streams = {}

    # T75 LGB L2 (5 seeds avg)
    paths = [os.path.join(T75_DIR, f"pred_T75_seed{s}.parquet") for s in seeds]
    if all(os.path.exists(p) for p in paths):
        base, p = load_avg_pred_from_paths(paths)
        streams["T75_LGB_L2"] = (base, p)
        print(f"  loaded T75 LGB L2 (5 seeds)", flush=True)

    # T87 NN SPO+ DFL (5 seeds avg)
    paths = [os.path.join(T87_DIR, f"pred_T87_seed{s}_main.parquet") for s in seeds]
    if all(os.path.exists(p) for p in paths):
        base, p = load_avg_pred_from_paths(paths)
        streams["T87_NN_SPO+"] = (base, p)
        print(f"  loaded T87 NN SPO+ (5 seeds)", flush=True)

    # T89 CB RMSE (5 seeds avg)
    paths = [os.path.join(T89_DIR, f"pred_T89_seed{s}.parquet") for s in seeds]
    if all(os.path.exists(p) for p in paths):
        base, p = load_avg_pred_from_paths(paths)
        streams["T89_CB_RMSE"] = (base, p)
        print(f"  loaded T89 CB RMSE (5 seeds)", flush=True)

    # T99 LGB Huber 1e-3 (whatever seeds available)
    paths_huber = sorted(glob(os.path.join(HERE, "pred_T99_huber_a0.001_seed*.parquet")))
    if len(paths_huber) >= 1:
        base, p = load_avg_pred_from_paths(paths_huber)
        streams[f"T99_LGB_Huber1e-3_{len(paths_huber)}seed"] = (base, p)
        print(f"  loaded T99 LGB Huber1e-3 ({len(paths_huber)} seeds)", flush=True)

    # T99 CB Huber 1e-3
    paths_cbh = sorted(glob(os.path.join(HERE, "pred_T99_cb_huber_a0.001_seed*.parquet")))
    if len(paths_cbh) >= 1:
        base, p = load_avg_pred_from_paths(paths_cbh)
        streams[f"T99_CB_Huber1e-3_{len(paths_cbh)}seed"] = (base, p)
        print(f"  loaded T99 CB Huber1e-3 ({len(paths_cbh)} seeds)", flush=True)

    # Reference base for combos (use T75)
    if not streams:
        print("ERROR: no streams loaded", flush=True)
        sys.exit(1)
    ref_base = next(iter(streams.values()))[0]

    # Cross-correlation matrix
    print(f"\n[Cross-correlation matrix]", flush=True)
    keys = list(streams.keys())
    n = len(keys)
    corr = np.zeros((n, n))
    for i, ki in enumerate(keys):
        pi = streams[ki][1]
        for j, kj in enumerate(keys):
            pj = streams[kj][1]
            corr[i, j] = float(np.corrcoef(pi, pj)[0, 1])
    print(f"  {'':25s} " + " ".join(f"{k[:14]:>14s}" for k in keys))
    for i, ki in enumerate(keys):
        print(f"  {ki[:25]:25s} " + " ".join(f"{corr[i,j]:>+14.4f}" for j in range(n)))

    # Standalone (each stream alone)
    print(f"\n[Standalone DE asym LOSO]", flush=True)
    standalone = {}
    for k, (base, p) in streams.items():
        r = eval_combo(k, p, base, bounds=(0.0, args.bounds_hi))
        standalone[k] = r
        print(f"  {k:30s} thr=({r['thr_up']:.5f},{r['thr_dn']:.5f}) de_sum={r['de_sum']:+8.4f}",
              flush=True)

    # Pairwise ensembles
    print(f"\n[Pairwise ensembles (avg w=1:1)]", flush=True)
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            ki, kj = keys[i], keys[j]
            p_avg = (streams[ki][1] + streams[kj][1]) / 2.0
            r = eval_combo(f"{ki}+{kj}_w1:1", p_avg, ref_base,
                           bounds=(0.0, args.bounds_hi))
            pairs.append(r)
            print(f"  {ki:25s}+{kj:25s} de_sum={r['de_sum']:+8.4f}", flush=True)

    # Triplets — top 3 streams from standalone
    print(f"\n[Triplet ensembles (top streams)]", flush=True)
    standalone_ranked = sorted(standalone.items(), key=lambda x: -x[1]["de_sum"])
    top_keys = [k for k, _ in standalone_ranked[:5]]
    triplets = []
    for i in range(len(top_keys)):
        for j in range(i+1, len(top_keys)):
            for k in range(j+1, len(top_keys)):
                ki, kj, kk = top_keys[i], top_keys[j], top_keys[k]
                p_avg = (streams[ki][1] + streams[kj][1] + streams[kk][1]) / 3.0
                r = eval_combo(f"{ki}+{kj}+{kk}_w1:1:1", p_avg, ref_base,
                               bounds=(0.0, args.bounds_hi))
                triplets.append(r)
                print(f"  {ki[:18]:18s}+{kj[:18]:18s}+{kk[:18]:18s}  de_sum={r['de_sum']:+8.4f}",
                      flush=True)

    # Sort all by de_sum
    all_combos = list(standalone.values()) + pairs + triplets
    all_combos.sort(key=lambda r: -r["de_sum"])
    print(f"\n=== TOP 10 ===", flush=True)
    for r in all_combos[:10]:
        print(f"  {r['label']:60s} de_sum={r['de_sum']:+8.4f}  thr=({r['thr_up']:.5f},{r['thr_dn']:.5f})",
              flush=True)

    out = {
        "task": "T99 ensemble eval",
        "fee_rate": FEE,
        "streams": list(streams.keys()),
        "cross_corr_matrix": {keys[i]: {keys[j]: float(corr[i,j]) for j in range(n)} for i in range(n)},
        "standalone": standalone,
        "pairs": pairs,
        "triplets": triplets,
        "top10": all_combos[:10],
    }
    out_path = os.path.join(HERE, args.out)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
