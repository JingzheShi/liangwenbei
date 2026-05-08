"""R_multitick_window: 3-seed ensemble × 2-variant DE evaluation.

Per ensemble:
  1. Average the 3-seed predictions
  2. DE 2D thresh (T_up, T_dn shared) → LOSO-equiv (sum per-sym)
  3. DE 4D thresh (T_up_am, T_dn_am, T_up_pm, T_dn_pm) → LOSO-equiv
  4. Per-sym PnL with both gates

Outputs: de_results.json with baseline / trick / delta.
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
SEEDS = (1, 7, 42)
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001
VARIANTS = ("baseline", "trick")


def vectorized_pnl(action, mp_t, mp_th):
    side = action.astype(np.float64) - 1.0
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee_pnl) / denom


def gate_2d(p, tu, td):
    a = np.full(len(p), 1, dtype=np.int8)
    a[p > tu] = 2
    a[p < -td] = 0
    return a


def gate_4d_session(p, sess_arr, tu_am, td_am, tu_pm, td_pm):
    a = np.full(len(p), 1, dtype=np.int8)
    is_am = (sess_arr == 0)
    is_pm = ~is_am
    a[is_am & (p > tu_am)] = 2
    a[is_am & (p < -td_am)] = 0
    a[is_pm & (p > tu_pm)] = 2
    a[is_pm & (p < -td_pm)] = 0
    return a


def avg_preds(parquet_paths):
    base = pd.read_parquet(parquet_paths[0])
    p = np.zeros(len(base), dtype=np.float64)
    for pp in parquet_paths:
        df = pd.read_parquet(pp)
        if not (df["sym"].equals(base["sym"]) and df["t"].equals(base["t"])
                and df["date"].equals(base["date"]) and df["session"].equals(base["session"])):
            raise RuntimeError(f"row order mismatch in {pp}")
        p += df["pred_dmid_norm"].to_numpy(np.float64)
    return base, p / len(parquet_paths)


def fit_de_2d(p_arr, sym_arr, mp_t, mp_th, de_seeds=(0, 42, 1)):
    folds = [(p_arr[sym_arr == k], mp_t[sym_arr == k], mp_th[sym_arr == k])
             for k in SYMS]

    def obj(x):
        s = 0.0
        for fp, fmt, fmth in folds:
            a = gate_2d(fp, x[0], x[1])
            s += vectorized_pnl(a, fmt, fmth).sum()
        return -float(s)

    best_x, best_y = None, np.inf
    for ds in de_seeds:
        r = differential_evolution(obj, bounds=[(0.0, 0.005), (0.0, 0.005)],
                                   seed=ds, maxiter=80, popsize=24,
                                   tol=1e-7, polish=True, workers=1, init="sobol")
        if r.fun < best_y:
            best_y, best_x = float(r.fun), tuple(float(v) for v in r.x)
    return best_x, -best_y


def fit_de_4d(p_arr, sym_arr, sess_arr, mp_t, mp_th, de_seeds=(0, 42, 1)):
    masks = [(sym_arr == k) for k in SYMS]
    folds = [(p_arr[m], sess_arr[m], mp_t[m], mp_th[m]) for m in masks]

    def obj(x):
        tu_am, td_am, tu_pm, td_pm = x
        s = 0.0
        for fp, fs, fmt, fmth in folds:
            a = gate_4d_session(fp, fs, tu_am, td_am, tu_pm, td_pm)
            s += vectorized_pnl(a, fmt, fmth).sum()
        return -float(s)

    bounds = [(0.0, 0.005), (0.0, 0.005), (0.0, 0.005), (0.0, 0.005)]
    best_x, best_y = None, np.inf
    for ds in de_seeds:
        r = differential_evolution(obj, bounds=bounds,
                                   seed=ds, maxiter=120, popsize=28,
                                   tol=1e-7, polish=True, workers=1, init="sobol")
        if r.fun < best_y:
            best_y, best_x = float(r.fun), tuple(float(v) for v in r.x)
    return best_x, -best_y


def per_sym_pnl_2d(p_arr, sym_arr, mp_t, mp_th, tu, td):
    out = {}
    for k in SYMS:
        m = sym_arr == k
        a = gate_2d(p_arr[m], tu, td)
        pnl = vectorized_pnl(a, mp_t[m], mp_th[m]).sum()
        out[int(k)] = float(pnl)
    return out


def per_sym_pnl_4d(p_arr, sym_arr, sess_arr, mp_t, mp_th, tu_am, td_am, tu_pm, td_pm):
    out = {}
    for k in SYMS:
        m = sym_arr == k
        a = gate_4d_session(p_arr[m], sess_arr[m], tu_am, td_am, tu_pm, td_pm)
        pnl = vectorized_pnl(a, mp_t[m], mp_th[m]).sum()
        out[int(k)] = float(pnl)
    return out


def evaluate_arm(name, paths):
    print(f"\n=== {name} ({len(paths)} seeds) ===", flush=True)
    base, p_avg = avg_preds(paths)
    sym_arr = base["sym"].to_numpy(np.int8)
    sess_arr = (base["session"].to_numpy() == "pm").astype(np.int8)
    mp_t = base["midprice_t"].to_numpy(np.float64)
    mp_th = base["midprice_th"].to_numpy(np.float64)

    t0 = time.time()
    (tu, td), tot_2d = fit_de_2d(p_avg, sym_arr, mp_t, mp_th)
    per_2d = per_sym_pnl_2d(p_avg, sym_arr, mp_t, mp_th, tu, td)
    loso_2d = sum(per_2d.values())
    print(f"  DE 2D:  tu={tu:.4e} td={td:.4e} loso={loso_2d:+.4f}  ({time.time()-t0:.1f}s)",
          flush=True)
    print(f"  DE 2D per-sym: {per_2d}", flush=True)

    t0 = time.time()
    (tu_am, td_am, tu_pm, td_pm), tot_4d = fit_de_4d(p_avg, sym_arr, sess_arr, mp_t, mp_th)
    per_4d = per_sym_pnl_4d(p_avg, sym_arr, sess_arr, mp_t, mp_th, tu_am, td_am, tu_pm, td_pm)
    loso_4d = sum(per_4d.values())
    print(f"  DE 4D:  am=({tu_am:.3e},{td_am:.3e}) pm=({tu_pm:.3e},{td_pm:.3e}) loso={loso_4d:+.4f}  ({time.time()-t0:.1f}s)",
          flush=True)
    print(f"  DE 4D per-sym: {per_4d}", flush=True)

    return {
        "arm": name, "n_seeds": len(paths),
        "de_2d": {"T_up": tu, "T_dn": td, "loso_equiv": loso_2d, "per_sym": per_2d},
        "de_4d_session": {
            "T_up_am": tu_am, "T_dn_am": td_am,
            "T_up_pm": tu_pm, "T_dn_pm": td_pm,
            "loso_equiv": loso_4d, "per_sym": per_4d,
        },
    }


def main():
    print("=== R_multitick_window 3-seed × 2-variant DE eval ===", flush=True)
    arms = {
        v: [os.path.join(HERE, f"pred_{v}_seed{s}.parquet") for s in SEEDS]
        for v in VARIANTS
    }
    for k, paths in arms.items():
        for p in paths:
            if not os.path.exists(p):
                print(f"MISSING: {p}", flush=True)
                sys.exit(1)

    results = {}
    for name, paths in arms.items():
        results[name] = evaluate_arm(name, paths)

    b2 = results["baseline"]["de_2d"]["loso_equiv"]
    b4 = results["baseline"]["de_4d_session"]["loso_equiv"]
    t2 = results["trick"]["de_2d"]["loso_equiv"]
    t4 = results["trick"]["de_4d_session"]["loso_equiv"]

    print("\n=== Summary ===", flush=True)
    print(f"  Baseline DE 2D LOSO:  {b2:+.4f}", flush=True)
    print(f"  Baseline DE 4D LOSO:  {b4:+.4f}", flush=True)
    print(f"  +trick   DE 2D LOSO:  {t2:+.4f}  (Δ = {t2 - b2:+.4f})", flush=True)
    print(f"  +trick   DE 4D LOSO:  {t4:+.4f}  (Δ = {t4 - b4:+.4f})", flush=True)

    base_per = results["baseline"]["de_4d_session"]["per_sym"]
    trk_per = results["trick"]["de_4d_session"]["per_sym"]
    deltas = {k: trk_per[k] - base_per[k] for k in base_per}
    print(f"\n  Per-sym Δ (4D): {deltas}", flush=True)
    print(f"  Per-sym Δ min/max: {min(deltas.values()):+.4f} / {max(deltas.values()):+.4f}", flush=True)

    out = {
        "arms": results,
        "delta_2d": t2 - b2,
        "delta_4d": t4 - b4,
        "per_sym_delta_4d": deltas,
        "per_sym_delta_4d_min": min(deltas.values()),
        "per_sym_delta_4d_max": max(deltas.values()),
    }
    with open(os.path.join(HERE, "de_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved de_results.json", flush=True)


if __name__ == "__main__":
    main()
