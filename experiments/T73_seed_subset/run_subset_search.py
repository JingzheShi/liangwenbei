"""T73: enumerate all non-empty subsets of T70's 5 seeds, find best by DE-loso sum.

Strategy:
  - Load 5 seed prob arrays once into memory
  - For each of 31 non-empty subsets:
      Stage A (coarse): DE 4D with maxiter=40, popsize=15, 1 seed (sd=42)
                         objective = sum-of-per-sym cum_pnl
  - Sort coarse results, pick top 5
  - Stage B (fine): rerun those 5 with maxiter=80, popsize=24, 4 DE seeds
  - Save results.json with full ranking

Comparison:
  iter_011 LOSO (T64 5-seed)            +25.94
  iter_012 LOSO (T70 5-seed all)        +26.4396
  T73 best subset (TBD)
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
T53_DIR = os.path.join(ROOT, "experiments", "T53_r34_stage3")
T70_DIR = os.path.join(ROOT, "experiments", "T70_v4_stage5")
sys.path.insert(0, T53_DIR)
from de_thresh import PROB_COLS, gate_asymmetric, vectorized_pnl  # noqa: E402

ALL_SEEDS = (1, 7, 13, 42, 100)
SYMS = (0, 1, 2, 3, 4)
ITER011 = 25.94
ITER012 = 26.4396  # T70 5-seed-all DE-loso (existing baseline)
BOUNDS = [(0.34, 0.75), (0.34, 0.75), (0.0, 0.30), (0.0, 0.30)]


def load_all_seed_arrays():
    """Returns dict: seed -> np.ndarray (N,3) prob array, plus shared meta."""
    base = pd.read_parquet(os.path.join(T70_DIR, f"pred_T70_seed{ALL_SEEDS[0]}.parquet"))
    n = len(base)
    sym = base["sym"].to_numpy(np.int64)
    label = base["true_label"].to_numpy(np.int64)
    mp_t = base["midprice_t"].to_numpy(np.float64)
    mp_th = base["midprice_th"].to_numpy(np.float64)
    probs = {ALL_SEEDS[0]: base[PROB_COLS].to_numpy(np.float64)}
    for s in ALL_SEEDS[1:]:
        df_s = pd.read_parquet(os.path.join(T70_DIR, f"pred_T70_seed{s}.parquet"))
        if len(df_s) != n:
            raise RuntimeError(f"row count mismatch seed={s}: {len(df_s)} vs {n}")
        if not (df_s["sym"].equals(base["sym"]) and df_s["date"].equals(base["date"]) and df_s["t"].equals(base["t"])):
            raise RuntimeError(f"row order mismatch seed={s}")
        probs[s] = df_s[PROB_COLS].to_numpy(np.float64)
    print(f"  loaded all 5 seeds, n={n:,}", flush=True)
    return probs, sym, label, mp_t, mp_th, n


def avg_subset(probs_per_seed, subset):
    p_sum = probs_per_seed[subset[0]].copy()
    for s in subset[1:]:
        p_sum += probs_per_seed[s]
    return (p_sum / float(len(subset))).astype(np.float32)


def split_folds(probs_avg, sym, label, mp_t, mp_th):
    folds = []
    for k in SYMS:
        m = sym == k
        folds.append({
            "sym": int(k),
            "probs": probs_avg[m],
            "label": label[m],
            "mp_t": mp_t[m],
            "mp_th": mp_th[m],
        })
    return folds


def make_obj_loso(folds):
    def f(x):
        Tu, Td, du, dd = x
        s = 0.0
        for fk in folds:
            pred = gate_asymmetric(fk["probs"], Tu, Td, du, dd)
            s += vectorized_pnl(pred, fk["label"], fk["mp_t"], fk["mp_th"]).sum()
        return -float(s)
    return f


def per_sym_pnl(folds, Tu, Td, du, dd):
    per = []
    n_active = []
    for f in folds:
        pred = gate_asymmetric(f["probs"], Tu, Td, du, dd)
        per.append(float(vectorized_pnl(pred, f["label"], f["mp_t"], f["mp_th"]).sum()))
        n_active.append(int((pred != 1).sum()))
    return per, n_active


def de_run(obj, seed, maxiter, popsize):
    t0 = time.time()
    res = differential_evolution(
        obj, bounds=BOUNDS, seed=seed, maxiter=maxiter, popsize=popsize,
        polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
        updating="deferred", workers=1, init="sobol",
    )
    return {
        "seed": int(seed),
        "T_up": float(res.x[0]),
        "T_dn": float(res.x[1]),
        "d_up": float(res.x[2]),
        "d_dn": float(res.x[3]),
        "obj_val": float(-res.fun),
        "nfev": int(res.nfev),
        "elapsed_sec": float(time.time() - t0),
    }


def all_nonempty_subsets(seeds):
    out = []
    for r in range(1, len(seeds) + 1):
        for c in itertools.combinations(seeds, r):
            out.append(tuple(c))
    return out


def stage_a_coarse(subsets, probs_per_seed, sym, label, mp_t, mp_th):
    rows = []
    for i, sub in enumerate(subsets):
        p_avg = avg_subset(probs_per_seed, sub)
        folds = split_folds(p_avg, sym, label, mp_t, mp_th)
        obj = make_obj_loso(folds)
        run = de_run(obj, seed=42, maxiter=40, popsize=15)
        per_sym, na = per_sym_pnl(folds, run["T_up"], run["T_dn"], run["d_up"], run["d_dn"])
        sum_per_sym = float(sum(per_sym))
        rows.append({
            "subset": list(sub),
            "subset_size": len(sub),
            "subset_str": "_".join(str(s) for s in sub),
            "coarse": run,
            "coarse_sum_per_sym": sum_per_sym,
            "coarse_per_sym": per_sym,
            "coarse_n_active": na,
        })
        print(f"  [{i+1:2d}/{len(subsets)}] subset={sub}  coarse_sum={sum_per_sym:+.4f}  obj={run['obj_val']:+.4f}  ({run['elapsed_sec']:.1f}s)", flush=True)
    rows.sort(key=lambda r: r["coarse_sum_per_sym"], reverse=True)
    return rows


def stage_b_fine(top_rows, probs_per_seed, sym, label, mp_t, mp_th):
    de_seeds = (0, 1, 2, 7, 42)
    refined = []
    for i, row in enumerate(top_rows):
        sub = tuple(row["subset"])
        p_avg = avg_subset(probs_per_seed, sub)
        folds = split_folds(p_avg, sym, label, mp_t, mp_th)
        obj = make_obj_loso(folds)
        runs = []
        for sd in de_seeds:
            r = de_run(obj, seed=sd, maxiter=80, popsize=24)
            runs.append(r)
        runs.sort(key=lambda r: r["obj_val"], reverse=True)
        best = runs[0]
        per_sym, na = per_sym_pnl(folds, best["T_up"], best["T_dn"], best["d_up"], best["d_dn"])
        sum_per_sym = float(sum(per_sym))
        refined.append({
            **row,
            "fine_runs": runs,
            "fine_best": best,
            "fine_sum_per_sym": sum_per_sym,
            "fine_per_sym": per_sym,
            "fine_n_active": na,
        })
        print(f"  [{i+1}/{len(top_rows)}] subset={sub}  fine_sum={sum_per_sym:+.4f}  best_obj={best['obj_val']:+.4f}", flush=True)
    refined.sort(key=lambda r: r["fine_sum_per_sym"], reverse=True)
    return refined


def main():
    print(f"=== T73 seed subset search ===", flush=True)
    t0 = time.time()
    probs_per_seed, sym, label, mp_t, mp_th, n = load_all_seed_arrays()

    subsets = all_nonempty_subsets(ALL_SEEDS)
    print(f"  {len(subsets)} non-empty subsets", flush=True)

    print(f"\n[Stage A] coarse DE on all subsets (maxiter=40 popsize=15 seed=42)", flush=True)
    coarse_rows = stage_a_coarse(subsets, probs_per_seed, sym, label, mp_t, mp_th)

    K_FINE = 5
    print(f"\n[Stage B] fine DE on top {K_FINE} subsets (maxiter=80 popsize=24 5 DE seeds)", flush=True)
    top = coarse_rows[:K_FINE]
    fine_rows = stage_b_fine(top, probs_per_seed, sym, label, mp_t, mp_th)

    fine_lookup = {tuple(r["subset"]): r for r in fine_rows}
    final = []
    for r in coarse_rows:
        key = tuple(r["subset"])
        if key in fine_lookup:
            final.append(fine_lookup[key])
        else:
            final.append(r)

    best = max(final, key=lambda r: r.get("fine_sum_per_sym", r["coarse_sum_per_sym"]))
    best_metric = best.get("fine_sum_per_sym", best["coarse_sum_per_sym"])

    out = {
        "task": "T73 seed subset enumeration",
        "n_test": int(n),
        "all_seeds": list(ALL_SEEDS),
        "n_subsets": len(subsets),
        "iter011_loso": ITER011,
        "iter012_loso": ITER012,
        "stage_a_params": {"maxiter": 40, "popsize": 15, "seed": 42},
        "stage_b_params": {"maxiter": 80, "popsize": 24, "de_seeds": [0, 1, 2, 7, 42]},
        "all_subsets_ranked": final,
        "best_subset": best["subset"],
        "best_metric_source": "fine" if "fine_sum_per_sym" in best else "coarse",
        "best_sum_per_sym": best_metric,
        "vs_iter012": best_metric - ITER012,
        "vs_iter011": best_metric - ITER011,
        "elapsed_sec_total": time.time() - t0,
    }
    out_path = os.path.join(HERE, "results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote -> {out_path}", flush=True)

    print(f"\n{'='*78}\n=== T73 SUMMARY ===\n{'='*78}", flush=True)
    print(f"iter_011 (T64 5-seed):   +{ITER011:.4f}", flush=True)
    print(f"iter_012 (T70 5-seed):   +{ITER012:.4f}  <- baseline", flush=True)
    print(f"T73 best subset:         {best['subset']}", flush=True)
    print(f"  sum_per_sym = +{best_metric:.4f}  ({out['best_metric_source']})", flush=True)
    print(f"  vs iter_012 = {out['vs_iter012']:+.4f}", flush=True)
    print(f"\nTop 10 subsets (fine if available else coarse):", flush=True)
    for i, r in enumerate(final[:10]):
        m = r.get("fine_sum_per_sym", r["coarse_sum_per_sym"])
        src = "fine" if "fine_sum_per_sym" in r else "coarse"
        print(f"  {i+1:2d}. {tuple(r['subset'])} ({len(r['subset'])}-seed)  {src:6s} sum={m:+.4f}", flush=True)


if __name__ == "__main__":
    main()
