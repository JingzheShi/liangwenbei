"""DE + LOSO evaluation for Hawkes L2 experiment.

Per variant (baseline/hawkes):
  1. Average 3-seed predictions
  2. DE 2D thresh (T_up, T_dn) optimizing LOSO PnL
  3. Per-sym PnL breakdown

Outputs: de_results.json with baseline / hawkes / delta.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (1, 7, 42)
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001


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


def avg_preds(variant):
    parquet_paths = [os.path.join(HERE, f"pred_{variant}_seed{s}.parquet") for s in SEEDS]
    for pp in parquet_paths:
        if not os.path.exists(pp):
            raise FileNotFoundError(f"Missing: {pp}")

    base = pd.read_parquet(parquet_paths[0])
    p = np.zeros(len(base), dtype=np.float64)
    for pp in parquet_paths:
        df = pd.read_parquet(pp)
        p += df["pred_dmid_norm"].to_numpy(np.float64)
    p /= len(parquet_paths)

    return (p, base["sym"].to_numpy(np.int8),
            base["midprice_t"].to_numpy(np.float64),
            base["midprice_th"].to_numpy(np.float64),
            base["session"].to_numpy())


def fit_de_loso(p_arr, sym_arr, mp_t, mp_th, de_seeds=(0, 42, 1)):
    """DE 2D: optimize sum of per-sym PnL (LOSO-equivalent)."""
    folds = [(p_arr[sym_arr == k], mp_t[sym_arr == k], mp_th[sym_arr == k])
             for k in SYMS]

    def obj(x):
        tu, td = x
        s = 0.0
        for fp, fmt, fmth in folds:
            a = gate_2d(fp, tu, td)
            s += vectorized_pnl(a, fmt, fmth).sum()
        return -float(s)

    best_x, best_y = None, np.inf
    for ds in de_seeds:
        r = differential_evolution(obj, bounds=[(0.0, 0.008), (0.0, 0.008)],
                                   seed=ds, maxiter=100, popsize=32,
                                   tol=1e-7, polish=True, workers=1, init="sobol")
        if r.fun < best_y:
            best_y, best_x = float(r.fun), tuple(float(v) for v in r.x)
    return best_x, -best_y


def per_sym_pnl(p_arr, sym_arr, mp_t, mp_th, tu, td):
    results = {}
    for s in SYMS:
        m = (sym_arr == s)
        a = gate_2d(p_arr[m], tu, td)
        pnl = vectorized_pnl(a, mp_t[m], mp_th[m])
        results[f"sym{s}"] = float(pnl.sum())
        results[f"sym{s}_n_active"] = int((a != 1).sum())
    return results


def main():
    print("=== DE + LOSO Evaluation ===")

    results = {}

    for variant in ["baseline", "hawkes"]:
        print(f"\n>>> {variant}")
        try:
            p, sym, mp_t, mp_th, sess = avg_preds(variant)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        print(f"  samples: {len(p):,}")

        thresh, loso_pnl = fit_de_loso(p, sym, mp_t, mp_th)
        print(f"  DE thresh: T_up={thresh[0]:.6f}, T_dn={thresh[1]:.6f}")
        print(f"  LOSO PnL: {loso_pnl:+.4f}")

        per_sym = per_sym_pnl(p, sym, mp_t, mp_th, thresh[0], thresh[1])
        print(f"  per-sym: {[round(per_sym[f'sym{s}'], 2) for s in SYMS]}")

        total_active = sum(per_sym[f"sym{s}_n_active"] for s in SYMS)

        results[variant] = {
            "thresh_up": thresh[0],
            "thresh_dn": thresh[1],
            "loso_pnl": loso_pnl,
            "per_sym": per_sym,
            "total_n_active": total_active,
        }

    if "baseline" in results and "hawkes" in results:
        delta = results["hawkes"]["loso_pnl"] - results["baseline"]["loso_pnl"]
        results["delta"] = delta
        print(f"\n>>> DELTA: {delta:+.4f}")
        if delta >= 0.5:
            print("  ** iter_019 CANDIDATE **")
            results["candidate"] = True
        else:
            results["candidate"] = False

    out_path = os.path.join(HERE, "de_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
