"""
T34.1 — Leave-2-Sym-Out (LO2SO) PnL on existing OOF predictions.

Methodology:
    True LO2SO would train models holding out 2 syms, but we have only LOSO
    (1-sym hold-out) trained models. We approximate LO2SO by combining the
    held-out predictions of two LOSO folds: e.g. for held-out (sym_A, sym_B),
    take sym_A's OOF row from fold "trained on {0..4}\\{A}" and sym_B's row
    from fold "trained on {0..4}\\{B}".

    This is a *strict* approximation: each sym's prediction comes from a model
    that has seen 4 syms (NOT 3). Real LO2SO would be strictly harder. But the
    *combined PnL distribution across 10 (a,b) pairs* gives a much richer
    picture of OOD behavior than the 5 single-sym folds alone.

Why useful:
    - 10 fold combinations vs 5 → tighter estimate of OOD mean/median
    - Worst-pair PnL is a brittleness proxy
    - Comparing single-LOSO sum vs LO2SO mean reveals whether the held-out
      sym is the dominant cause of variance

NOTE: We work entirely from existing OOF parquets — no training.
"""
from __future__ import annotations

import json
import os
import sys
from itertools import combinations
from typing import Dict, List

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from oof_io import HORIZONS, compute_pnl_per_row, cum_pnl_metrics, load_oof  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def lo2so_summary(df: pd.DataFrame) -> Dict:
    """For an OOF DataFrame containing predictions for all 5 syms, compute
    PnL per (sym_a, sym_b) pair (10 pairs) plus mean/median/worst.
    """
    pnl = compute_pnl_per_row(df)
    syms = sorted(df["sym"].unique().tolist())
    df = df.copy()
    df["_pnl"] = pnl
    df["_active"] = (df["pred_label"].to_numpy() != 1).astype(int)

    per_sym_cum = df.groupby("sym")["_pnl"].sum().to_dict()
    per_sym_n_active = df.groupby("sym")["_active"].sum().to_dict()

    pair_results = []
    for a, b in combinations(syms, 2):
        pair_pnl = per_sym_cum[a] + per_sym_cum[b]
        pair_n_active = per_sym_n_active[a] + per_sym_n_active[b]
        pair_per_trade = pair_pnl / pair_n_active if pair_n_active else 0.0
        pair_results.append({
            "pair": (int(a), int(b)),
            "cum_pnl": float(pair_pnl),
            "n_active": int(pair_n_active),
            "per_trade_pnl": float(pair_per_trade),
        })

    cums = np.array([p["cum_pnl"] for p in pair_results])
    return {
        "n_pairs": len(pair_results),
        "pair_results": pair_results,
        "lo2so_mean": float(cums.mean()),
        "lo2so_median": float(np.median(cums)),
        "lo2so_min_pair": float(cums.min()),
        "lo2so_max_pair": float(cums.max()),
        "lo2so_std": float(cums.std(ddof=1)) if len(cums) > 1 else 0.0,
        # equivalent total: cum_pnl summed across all pairs / 4 (each sym in 4 pairs)
        # === single-LOSO sum (since each sym appears in 4 of 10 pairs)
        "loso_sum_pnl": float(sum(per_sym_cum.values())),
        "loso_per_sym": {int(k): float(v) for k, v in per_sym_cum.items()},
    }


def main():
    out: Dict[str, Dict] = {}

    # iter_002 — all 5 horizons
    for h in HORIZONS:
        df = load_oof("iter_002", h)
        out[f"iter_002_h{h}"] = lo2so_summary(df)

    # iter_005b — h=60 (5-seed ensemble) + h=40 (reuses iter_002)
    for h in (40, 60):
        df = load_oof("iter_005b", h)
        out[f"iter_005b_h{h}"] = lo2so_summary(df)

    # t26 single-seed aug_a baseline (for comparison: iter_005b w/o ensemble)
    out["t26_aug_a_h60"] = lo2so_summary(load_oof("t26_aug_a", 60))
    out["t26_baseline_h60"] = lo2so_summary(load_oof("t26_baseline", 60))

    # Print summary table
    print(f"{'model_h':<28} {'LOSO sum':>10} {'LO2SO mean':>11} {'LO2SO med':>11} "
          f"{'min pair':>10} {'max pair':>10} {'std':>8}")
    for k, v in out.items():
        print(f"{k:<28} {v['loso_sum_pnl']:>+10.4f} {v['lo2so_mean']:>+11.4f} "
              f"{v['lo2so_median']:>+11.4f} {v['lo2so_min_pair']:>+10.4f} "
              f"{v['lo2so_max_pair']:>+10.4f} {v['lo2so_std']:>+8.4f}")

    # Save
    out_path = os.path.join(HERE, "lo2so_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[lo2so] saved {out_path}")


if __name__ == "__main__":
    main()
