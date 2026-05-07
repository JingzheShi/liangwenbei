"""Per-regime threshold sweep.

Discretizes a sym-agnostic regime indicator (computed from 100-tick window
only) into 3-5 bins, then sweeps (T_up, T_dn, d_up, d_dn) per bin on the
5-seed aug_a OOF h_60.

Regime selection: tries each indicator individually and picks the one that
gives the largest LOSO improvement vs the global asymmetric best.

Important: bin edges are computed from the OOF data quantiles. At inference
time, the same edges would be used (they are constants), preserving the
sym-agnostic / stateless contract — bin assignment uses only the current
100-tick window.
"""
from __future__ import annotations

import json
import os
import time
from itertools import product

import numpy as np
import pandas as pd

from scipy.optimize import differential_evolution

from _common import fold_arrays, gate_asymmetric, load_all_folds, vectorized_pnl

HERE = os.path.dirname(os.path.abspath(__file__))
REGIME_PARQUET = os.path.join(HERE, "regime_features.parquet")

DE_BOUNDS = [(0.34, 0.75), (0.34, 0.75), (0.0, 0.30), (0.0, 0.30)]


def attach_regime(folds, regime_df: pd.DataFrame, indicator: str, n_bins: int):
    """Return list of (FoldArrays-like dict + regime bin assignments)."""
    edges = np.quantile(regime_df[indicator].dropna().to_numpy(), np.linspace(0, 1, n_bins + 1))
    edges[0] = -np.inf; edges[-1] = np.inf
    out_folds = {}
    for k, df in folds.items():
        merged = df.merge(regime_df[["sym", "date", "session", "t", indicator]],
                          on=["sym", "date", "session", "t"], how="left")
        if merged[indicator].isna().any():
            print(f"  warn: fold{k} has {merged[indicator].isna().sum()} missing {indicator}")
            merged[indicator] = merged[indicator].fillna(merged[indicator].median())
        bins = np.searchsorted(edges[1:-1], merged[indicator].to_numpy()).astype(np.int8)
        out_folds[k] = {
            "df": merged,
            "probs": merged[["prob_0","prob_1","prob_2"]].to_numpy(np.float32),
            "label": merged["true_label"].to_numpy(np.int64),
            "mp_t": merged["midprice_t"].to_numpy(np.float64),
            "mp_th": merged["midprice_th"].to_numpy(np.float64),
            "bins": bins,
        }
    return out_folds, edges


def best_threshold_for_subset(probs, label, mp_t, mp_th):
    """DE search over (T_up, T_dn, d_up, d_dn) for one subset."""
    def neg(x):
        Tu, Td, du, dd = x
        pred = gate_asymmetric(probs, float(Tu), float(Td), float(du), float(dd))
        return -float(vectorized_pnl(pred, label, mp_t, mp_th).sum())
    result = differential_evolution(
        neg, bounds=DE_BOUNDS, seed=0, maxiter=40, popsize=18,
        polish=True, tol=1e-7, init="sobol")
    return -float(result.fun), tuple(float(v) for v in result.x)


def evaluate_regime(folds_with_regime, n_bins: int) -> dict:
    """For each bin: pool OOF rows across all 5 folds, find best threshold,
    apply per-fold and report sum_cum_pnl."""
    all_probs = np.concatenate([fwr["probs"] for fwr in folds_with_regime.values()])
    all_label = np.concatenate([fwr["label"] for fwr in folds_with_regime.values()])
    all_mp_t = np.concatenate([fwr["mp_t"]  for fwr in folds_with_regime.values()])
    all_mp_th = np.concatenate([fwr["mp_th"] for fwr in folds_with_regime.values()])
    all_bins = np.concatenate([fwr["bins"]  for fwr in folds_with_regime.values()])

    # Per-bin best threshold (selected on POOLED data, not per-fold)
    bin_thresholds = {}
    for b in range(n_bins):
        mask = all_bins == b
        if mask.sum() < 1000:
            bin_thresholds[b] = (0.45, 0.45, 0.10, 0.10)
            continue
        s, best = best_threshold_for_subset(
            all_probs[mask], all_label[mask], all_mp_t[mask], all_mp_th[mask])
        bin_thresholds[b] = best
        # print(f"    bin {b}: n={mask.sum()}, best (Tu,Td,du,dd)={best}, sum={s:.3f}")

    # Apply per-fold and sum
    per_fold = []
    for k, fwr in folds_with_regime.items():
        pred = np.full(len(fwr["bins"]), 1, dtype=np.int8)
        for b in range(n_bins):
            mask = fwr["bins"] == b
            if not mask.any():
                continue
            Tu, Td, du, dd = bin_thresholds[b]
            pb = gate_asymmetric(fwr["probs"][mask], Tu, Td, du, dd)
            pred[mask] = pb
        per_fold.append(float(vectorized_pnl(pred, fwr["label"],
                                              fwr["mp_t"], fwr["mp_th"]).sum()))
    sum_p = sum(per_fold)
    return {
        "sum_cum_pnl": sum_p,
        "per_fold_pnl": per_fold,
        "n_pos_folds": int(sum(1 for x in per_fold if x > 0)),
        "std_cum_pnl": float(np.std(per_fold, ddof=0)),
        "bin_thresholds": {int(b): list(v) for b, v in bin_thresholds.items()},
    }


def main():
    t0 = time.time()
    folds = load_all_folds()
    regime_df = pd.read_parquet(REGIME_PARQUET)
    print(f"Loaded {len(regime_df)} regime rows")

    all_results = {}
    for indicator in ["regime_vol", "regime_imb", "regime_intst",
                      "regime_spread", "regime_ret"]:
        for n_bins in [3, 4]:
            t1 = time.time()
            fwr, edges = attach_regime(folds, regime_df, indicator, n_bins)
            res = evaluate_regime(fwr, n_bins)
            res["indicator"] = indicator
            res["n_bins"] = n_bins
            res["bin_edges"] = [float(e) for e in edges]
            res["elapsed_sec"] = time.time() - t1
            key = f"{indicator}_b{n_bins}"
            all_results[key] = res
            print(f"  {key}: sum={res['sum_cum_pnl']:+.4f}, "
                  f"per_fold={[round(x,3) for x in res['per_fold_pnl']]}, "
                  f"pos={res['n_pos_folds']}/5, "
                  f"({res['elapsed_sec']:.1f}s)")

    best_key = max(all_results, key=lambda k: all_results[k]["sum_cum_pnl"])
    print(f"\nBest regime: {best_key} sum={all_results[best_key]['sum_cum_pnl']:+.4f}")

    out = {
        "task": "T30 per-regime threshold sweep",
        "n_indicators": 5,
        "n_bin_options": 3,
        "results": all_results,
        "best_key": best_key,
        "elapsed_sec": time.time() - t0,
    }
    with open(os.path.join(HERE, "per_regime_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
