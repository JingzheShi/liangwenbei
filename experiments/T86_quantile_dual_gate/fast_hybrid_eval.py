"""T86 fast eval: hybrid of (q30+q70)/2, T75 mean, T81 NN with grid weights.

KEY OPTIMIZATION: when the decision rule is global (no per-sym thresholds),
LOSO-equiv = sum_per_sym(cum_pnl) = total cum_pnl. So we can use a single
fully-vectorized eval over all 442k rows instead of looping over 5 folds.

Goal: answer the headline question — can T86 quantile predictions add to the
T75+T81 ensemble (+38.28 LOSO-equiv)?
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
T75_DIR = os.path.join(ROOT, "experiments", "T75_regression_dmid")
T81_DIR = os.path.join(ROOT, "experiments", "T81_nn_regression_pnl")

SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001
ITER013_LOSO = 36.23
ITER014_LOSO = 38.28
SEEDS = (1, 7, 13, 42, 100)


def load_avg_quantile_each(seeds, q_tag):
    base = pd.read_parquet(os.path.join(HERE, f"pred_T86_{q_tag}_seed{seeds[0]}.parquet"))
    p = np.zeros(len(base), dtype=np.float64)
    for s in seeds:
        df_s = pd.read_parquet(os.path.join(HERE, f"pred_T86_{q_tag}_seed{s}.parquet"))
        p += df_s["pred_dmid_norm"].to_numpy(np.float64)
    p /= float(len(seeds))
    return p, base


def load_avg_t75():
    base = pd.read_parquet(os.path.join(T75_DIR, f"pred_T75_seed{SEEDS[0]}.parquet"))
    p = np.zeros(len(base), dtype=np.float64)
    for s in SEEDS:
        df_s = pd.read_parquet(os.path.join(T75_DIR, f"pred_T75_seed{s}.parquet"))
        p += df_s["pred_dmid_norm"].to_numpy(np.float64)
    p /= float(len(SEEDS))
    return p, base


def load_avg_t81():
    base = pd.read_parquet(os.path.join(T81_DIR, f"pred_T81_seed{SEEDS[0]}.parquet"))
    p = np.zeros(len(base), dtype=np.float64)
    for s in SEEDS:
        df_s = pd.read_parquet(os.path.join(T81_DIR, f"pred_T81_seed{s}.parquet"))
        p += df_s["pred_dmid_norm"].to_numpy(np.float64)
    p /= float(len(SEEDS))
    return p, base


# Pre-multiply constants so eval is just two boolean masks + dot products
# pnl_per_row = (side*diff - fee*|side|*|denom_sum|) / denom
# When action ∈ {0, 1, 2}, side = action - 1 ∈ {-1, 0, 1}, |side| = abs(action-1)
# Long: side=+1 → pnl = (diff - fee*denom_sum) / denom
# Short: side=-1 → pnl = (-diff - fee*denom_sum) / denom
# Flat: side=0 → pnl = 0
# Define:
#   long_pnl  = (mp_th - mp_t - fee*((mp_th+1)+(mp_t+1))) / (mp_t + 1)
#   short_pnl = (mp_t - mp_th - fee*((mp_th+1)+(mp_t+1))) / (mp_t + 1)


def precompute_pnl_arrays(mp_t, mp_th):
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    denom = mp_t.astype(np.float64) + 1.0
    fee_term = FEE * np.abs((mp_th.astype(np.float64) + 1.0) +
                            (mp_t.astype(np.float64) + 1.0))
    long_pnl = (diff - fee_term) / denom
    short_pnl = (-diff - fee_term) / denom
    return long_pnl, short_pnl


def total_pnl_at_thresh(pred, thr_up, thr_dn, long_pnl, short_pnl):
    """thr_up, thr_dn are positive scalars; rule is global so total = LOSO-equiv."""
    long_m = pred > thr_up
    short_m = pred < -thr_dn
    return float(long_pnl[long_m].sum() + short_pnl[short_m].sum())


def per_sym_pnl(pred, thr_up, thr_dn, long_pnl, short_pnl, sym, syms=SYMS):
    out = []
    nact = []
    for k in syms:
        m = sym == k
        long_m = m & (pred > thr_up)
        short_m = m & (pred < -thr_dn)
        out.append(float(long_pnl[long_m].sum() + short_pnl[short_m].sum()))
        nact.append(int(long_m.sum() + short_m.sum()))
    return out, nact


def de_thresh_2d(pred, long_pnl, short_pnl, seeds=(0, 7), maxiter=30, popsize=12):
    def obj(x):
        return -total_pnl_at_thresh(pred, x[0], x[1], long_pnl, short_pnl)
    bounds = [(0.0, 0.004), (0.0, 0.004)]
    best = None
    for sd in seeds:
        result = differential_evolution(
            obj, bounds=bounds, seed=sd, maxiter=maxiter, popsize=popsize,
            polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            updating="deferred", workers=1, init="sobol",
        )
        if best is None or -result.fun > best["loso"]:
            best = {"loso": float(-result.fun), "thr_up": float(result.x[0]),
                    "thr_dn": float(result.x[1])}
    return best


def main():
    print("=== T86 fast hybrid eval ===", flush=True)
    t0 = time.time()
    q30, base30 = load_avg_quantile_each(SEEDS, "q030")
    q70, base70 = load_avg_quantile_each(SEEDS, "q070")
    p_t75, base_t75 = load_avg_t75()
    p_t81, base_t81 = load_avg_t81()
    base = base30
    print(f"  loaded all 4 prediction sets in {time.time()-t0:.1f}s", flush=True)

    for tag, b in [("q70", base70), ("t75", base_t75), ("t81", base_t81)]:
        if not (b["sym"].equals(base["sym"]) and b["t"].equals(base["t"])):
            raise RuntimeError(f"row order mismatch: {tag}")

    q_mid = 0.5 * (q30 + q70)
    sym = base["sym"].to_numpy(np.int8)
    mp_t = base["midprice_t"].to_numpy(np.float64)
    mp_th = base["midprice_th"].to_numpy(np.float64)
    long_pnl, short_pnl = precompute_pnl_arrays(mp_t, mp_th)

    print(f"  pred-pred correlations:", flush=True)
    print(f"    corr(q_mid, t75) = {np.corrcoef(q_mid, p_t75)[0,1]:+.4f}", flush=True)
    print(f"    corr(q_mid, t81) = {np.corrcoef(q_mid, p_t81)[0,1]:+.4f}", flush=True)
    print(f"    corr(q30, q70)   = {np.corrcoef(q30, q70)[0,1]:+.4f}", flush=True)
    print(f"    corr(t75, t81)   = {np.corrcoef(p_t75, p_t81)[0,1]:+.4f}", flush=True)

    # Sanity: iter_014 baseline
    print(f"\n  sanity: w_q=0, w_t75=1.5, w_t81=1.0  → iter_014 baseline", flush=True)
    pred_iter014 = (1.5 * p_t75 + 1.0 * p_t81) / 2.5
    iter14_check = total_pnl_at_thresh(pred_iter014, 4.21e-4, 1.86e-4, long_pnl, short_pnl)
    print(f"    fixed thresh (4.21e-4, 1.86e-4) → LOSO {iter14_check:+.4f} "
          f"(published: +{ITER014_LOSO})", flush=True)
    iter14_de = de_thresh_2d(pred_iter014, long_pnl, short_pnl)
    print(f"    re-DE (2 seed × 30 iter × 12 pop): {iter14_de}", flush=True)

    # Time one DE call to estimate grid duration
    print(f"\n  timing single DE call ...", flush=True)
    t_de = time.time()
    _ = de_thresh_2d(pred_iter014, long_pnl, short_pnl)
    de_dt = time.time() - t_de
    print(f"    one DE call = {de_dt:.2f}s", flush=True)

    # Grid over weights
    weight_grid = [0.0, 0.5, 1.0, 1.5, 2.0]
    runs = []
    n_total = sum(1 for wq in weight_grid for w75 in weight_grid for w81 in weight_grid
                  if (wq + w75 + w81) > 0)
    print(f"\n  grid: {n_total} cells (estimated {n_total * de_dt:.0f}s)", flush=True)
    print(f"  {'w_q':>4s} {'w_t75':>5s} {'w_t81':>5s}  {'thr_up':>8s} {'thr_dn':>8s}  "
          f"{'LOSO':>8s}  vs_iter014  vs_iter013", flush=True)
    print(f"  {'-'*4} {'-'*5} {'-'*5}  {'-'*8} {'-'*8}  {'-'*8}  {'-'*10}  {'-'*10}", flush=True)
    t_start = time.time()
    for wq in weight_grid:
        for w75 in weight_grid:
            for w81 in weight_grid:
                w_total = wq + w75 + w81
                if w_total <= 0:
                    continue
                pred = (wq * q_mid + w75 * p_t75 + w81 * p_t81) / w_total
                best = de_thresh_2d(pred, long_pnl, short_pnl)
                runs.append({"w_q": wq, "w_t75": w75, "w_t81": w81,
                             "loso": best["loso"], "thr_up": best["thr_up"],
                             "thr_dn": best["thr_dn"]})
                print(f"  {wq:>4.1f} {w75:>5.1f} {w81:>5.1f}  "
                      f"{best['thr_up']:>8.5f} {best['thr_dn']:>8.5f}  "
                      f"{best['loso']:>+8.4f}  {best['loso']-ITER014_LOSO:>+10.4f}  "
                      f"{best['loso']-ITER013_LOSO:>+10.4f}",
                      flush=True)
    elapsed = time.time() - t_start
    print(f"\n  grid done in {elapsed:.1f}s", flush=True)

    runs.sort(key=lambda r: r["loso"], reverse=True)
    print(f"\n  TOP 10 weight combos:", flush=True)
    for r in runs[:10]:
        print(f"    w_q={r['w_q']:.1f} w_t75={r['w_t75']:.1f} w_t81={r['w_t81']:.1f}  "
              f"→ LOSO {r['loso']:+.4f}  thr_up={r['thr_up']:.5f} thr_dn={r['thr_dn']:.5f}",
              flush=True)
    best = runs[0]

    # Per-sym at best, and stronger DE polish on the best weight combo
    print(f"\n  STRONGER DE polish for best combo ...", flush=True)
    pred_best = (best["w_q"] * q_mid + best["w_t75"] * p_t75 + best["w_t81"] * p_t81) \
                / (best["w_q"] + best["w_t75"] + best["w_t81"])
    polished = de_thresh_2d(pred_best, long_pnl, short_pnl,
                            seeds=(0, 1, 2, 7, 42), maxiter=80, popsize=24)
    per_sym, n_active = per_sym_pnl(
        pred_best, polished["thr_up"], polished["thr_dn"], long_pnl, short_pnl, sym)
    print(f"    polished: {polished}", flush=True)
    print(f"    per_sym: [{', '.join(f'{x:+.3f}' for x in per_sym)}]  "
          f"n_active={n_active}", flush=True)

    print(f"\n  BEST: w_q={best['w_q']} w_t75={best['w_t75']} w_t81={best['w_t81']}  "
          f"polished LOSO {polished['loso']:+.4f}  vs_iter014 {polished['loso']-ITER014_LOSO:+.4f}",
          flush=True)

    out = {
        "task": "T86 fast hybrid eval",
        "iter013_ref": ITER013_LOSO,
        "iter014_ref": ITER014_LOSO,
        "iter014_recheck": {"thr_up_fixed": 4.21e-4, "thr_dn_fixed": 1.86e-4,
                            "loso_fixed": iter14_check, "de": iter14_de},
        "correlations": {
            "qmid_t75": float(np.corrcoef(q_mid, p_t75)[0, 1]),
            "qmid_t81": float(np.corrcoef(q_mid, p_t81)[0, 1]),
            "q30_q70": float(np.corrcoef(q30, q70)[0, 1]),
            "t75_t81": float(np.corrcoef(p_t75, p_t81)[0, 1]),
        },
        "all_runs_sorted": runs,
        "best_grid": best,
        "best_polished": {**polished, "w_q": best["w_q"], "w_t75": best["w_t75"],
                          "w_t81": best["w_t81"], "per_sym": per_sym, "n_active": n_active},
        "best_polished_vs_iter014": float(polished["loso"] - ITER014_LOSO),
        "best_polished_vs_iter013": float(polished["loso"] - ITER013_LOSO),
    }
    out_path = os.path.join(HERE, "fast_hybrid_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
