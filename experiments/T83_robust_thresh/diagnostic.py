"""T83 diagnostic — per-half eval-optimal vs train-optimal PnL.

Asks: how much of the gap is "we can't pick the right threshold" vs
"the eval-distribution itself is just less profitable"?

If eval-optimal PnL ≈ train-optimal PnL on eval half, then no threshold
strategy helps — the gap is intrinsic.
If eval-optimal PnL >> our eval PnL, there's room: a smarter strategy could
close part of the gap.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from core import load_pred_avg, gate_asym, sym_sum, FEE


def fit_de_quick(df, seeds=(0, 1, 2)):
    pred = df["pred"].to_numpy()
    mp_t = df["mp_t"].to_numpy()
    mp_th = df["mp_th"].to_numpy()
    sym_arr = df["sym"].to_numpy()

    def obj(x):
        thr_up, thr_dn = x
        a = gate_asym(pred, thr_up, thr_dn)
        side = a.astype(np.float64) - 1.0
        abs_side = np.abs(side)
        fee = FEE * abs_side * np.abs((mp_th + 1.0) + (mp_t + 1.0))
        pnl_row = (side * (mp_th - mp_t) - fee) / (mp_t + 1.0)
        s = 0.0
        for k in (0, 1, 2, 3, 4):
            mask = sym_arr == k
            if mask.any():
                s += pnl_row[mask].sum()
        return -s

    best = None
    for sd in seeds:
        result = differential_evolution(
            obj, bounds=((1e-5, 8e-4), (1e-5, 8e-4)),
            seed=sd, maxiter=60, popsize=20, polish=True, tol=1e-7,
            mutation=(0.5, 1.0), recombination=0.7, updating="deferred",
            workers=1, init="sobol")
        if best is None or -result.fun > best[2]:
            best = (float(result.x[0]), float(result.x[1]), float(-result.fun))
    return best


def main():
    df = load_pred_avg()
    halfA = df[df["date"].isin(range(96, 108))].reset_index(drop=True)   # 96-107
    halfB = df[df["date"].isin(range(108, 120))].reset_index(drop=True)  # 108-119
    print(f"halfA(96-107)={len(halfA)} halfB(108-119)={len(halfB)}")

    # Full-fit thr (from iter_013)
    THR = (3.723e-4, 1.613e-4)
    a_A = gate_asym(halfA["pred"].to_numpy(), *THR)
    a_B = gate_asym(halfB["pred"].to_numpy(), *THR)
    print(f"\niter_013 thr ({THR[0]:.3e}, {THR[1]:.3e}) applied:")
    print(f"  halfA (96-107):  {sym_sum(halfA, a_A):.4f}")
    print(f"  halfB (108-119): {sym_sum(halfB, a_B):.4f}")
    print(f"  total = {sym_sum(halfA, a_A) + sym_sum(halfB, a_B):.4f}  (matches full-fit)")

    # Best-possible thr on each half (what an oracle would pick)
    print("\nOracle thr (DE on each half independently):")
    thA_up, thA_dn, oA = fit_de_quick(halfA)
    print(f"  halfA-optimal: thr=({thA_up:.3e}, {thA_dn:.3e}) pnl={oA:.4f}")
    thB_up, thB_dn, oB = fit_de_quick(halfB)
    print(f"  halfB-optimal: thr=({thB_up:.3e}, {thB_dn:.3e}) pnl={oB:.4f}")

    # Cross-evaluate: thr_A on halfB and vice versa
    print("\nCross-application:")
    a_AB = gate_asym(halfB["pred"].to_numpy(), thA_up, thA_dn)
    print(f"  thr_optimal_A on halfB: pnl={sym_sum(halfB, a_AB):.4f}  (vs halfB-optimal {oB:.4f})")
    a_BA = gate_asym(halfA["pred"].to_numpy(), thB_up, thB_dn)
    print(f"  thr_optimal_B on halfA: pnl={sym_sum(halfA, a_BA):.4f}  (vs halfA-optimal {oA:.4f})")

    # The key diagnostic: how much eval-PnL does each strategy "cost" us
    # vs the oracle?
    print("\nGap analysis:")
    print(f"  S1 (train=A, eval=B): thr_A on B = {sym_sum(halfB, a_AB):.4f}, oracle B = {oB:.4f}, lost = {oB - sym_sum(halfB, a_AB):.4f}")
    print(f"  S2 (train=B, eval=A): thr_B on A = {sym_sum(halfA, a_BA):.4f}, oracle A = {oA:.4f}, lost = {oA - sym_sum(halfA, a_BA):.4f}")

    # PnL distribution per date
    print("\nPer-date PnL under iter_013 thr (lets us see distribution shifts):")
    pnl_per_date = {}
    for d in sorted(df["date"].unique()):
        sub = df[df["date"] == d]
        a = gate_asym(sub["pred"].to_numpy(), *THR)
        pnl_per_date[int(d)] = float(sym_sum(sub, a))
    for d, p in pnl_per_date.items():
        print(f"  date {d}: {p:.3f}")

    # Mean per half
    pa = np.mean([pnl_per_date[d] for d in range(96, 108)])
    pb = np.mean([pnl_per_date[d] for d in range(108, 120)])
    print(f"\nMean per-date PnL: halfA={pa:.3f}, halfB={pb:.3f}, ratio = {pa/pb:.3f}x")


if __name__ == "__main__":
    main()
