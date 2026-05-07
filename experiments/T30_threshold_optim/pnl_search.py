"""Differential evolution direct PnL maximization over (T_up, T_dn, d_up, d_dn).

Continuous 4D space; might find sweet spots the grid misses.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
from scipy.optimize import differential_evolution

from _common import fold_arrays, gate_asymmetric, load_all_folds, vectorized_pnl

HERE = os.path.dirname(os.path.abspath(__file__))


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


def main():
    t0 = time.time()
    folds = load_all_folds()
    fas = fold_arrays(folds)
    f = make_objective(fas)

    bounds = [(0.34, 0.75), (0.34, 0.75), (0.0, 0.30), (0.0, 0.30)]

    # Run with multiple seeds for reproducibility check
    runs = []
    for seed in (0, 1, 2, 7, 42):
        t_run = time.time()
        result = differential_evolution(
            f, bounds=bounds, seed=seed, maxiter=80, popsize=24,
            polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            updating="deferred", workers=1, init="sobol",
        )
        sum_pnl = -result.fun
        # per-fold breakdown
        per = []
        Tu, Td, du, dd = result.x
        for k in range(5):
            pred = gate_asymmetric(fas[k].probs, Tu, Td, du, dd)
            per.append(float(vectorized_pnl(pred, fas[k].label, fas[k].mp_t, fas[k].mp_th).sum()))
        n_active = []
        for k in range(5):
            pred = gate_asymmetric(fas[k].probs, Tu, Td, du, dd)
            n_active.append(int((pred != 1).sum()))

        runs.append({
            "seed": seed,
            "T_up": float(Tu), "T_dn": float(Td),
            "d_up": float(du), "d_dn": float(dd),
            "sum_cum_pnl": float(sum_pnl),
            "per_fold_pnl": per,
            "n_active_per_fold": n_active,
            "n_pos_folds": int(sum(1 for x in per if x > 0)),
            "std_cum_pnl": float(np.std(per, ddof=0)),
            "nfev": int(result.nfev),
            "elapsed_sec": time.time() - t_run,
        })
        print(f"seed={seed}: T_up={Tu:.4f} T_dn={Td:.4f} d_up={du:.4f} d_dn={dd:.4f} "
              f"-> sum={sum_pnl:+.4f}, per_fold={[round(x,3) for x in per]} "
              f"({result.nfev} evals, {time.time()-t_run:.1f}s)")

    runs.sort(key=lambda r: r["sum_cum_pnl"], reverse=True)
    best = runs[0]
    print(f"\nBest across {len(runs)} DE runs: sum={best['sum_cum_pnl']:+.4f} at "
          f"(T_up={best['T_up']:.4f}, T_dn={best['T_dn']:.4f}, "
          f"d_up={best['d_up']:.4f}, d_dn={best['d_dn']:.4f})")
    print(f"per_fold: {[round(x,3) for x in best['per_fold_pnl']]} "
          f"std={best['std_cum_pnl']:.4f} pos={best['n_pos_folds']}/5")

    out = {
        "task": "T30 differential evolution PnL search 4D",
        "bounds": bounds,
        "runs": runs,
        "best": best,
        "elapsed_sec": time.time() - t0,
    }
    with open(os.path.join(HERE, "pnl_search_results.json"), "w") as f_out:
        json.dump(out, f_out, indent=2)


if __name__ == "__main__":
    main()
