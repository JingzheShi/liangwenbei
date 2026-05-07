"""Honest LOSO-CV variant of per-regime threshold sweep.

For each held fold k, pick per-bin asymmetric thresholds using ONLY the other
4 folds' pooled data, then apply to fold k. The reported sum_cum_pnl is a
genuine OOF estimate (no per-bin overfit to the held fold).

If this score is well below the in-sample per_regime score, the per-bin
thresholds are overfitting noise.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from _common import N_FOLDS, fold_arrays, gate_asymmetric, load_all_folds, vectorized_pnl

HERE = os.path.dirname(os.path.abspath(__file__))
REGIME_PARQUET = os.path.join(HERE, "regime_features.parquet")
DE_BOUNDS = [(0.34, 0.75), (0.34, 0.75), (0.0, 0.30), (0.0, 0.30)]


def attach_regime(folds, regime_df: pd.DataFrame, indicator: str, n_bins: int):
    edges = np.quantile(regime_df[indicator].dropna().to_numpy(), np.linspace(0, 1, n_bins + 1))
    edges[0] = -np.inf; edges[-1] = np.inf
    out = {}
    for k, df in folds.items():
        merged = df.merge(regime_df[["sym","date","session","t",indicator]],
                          on=["sym","date","session","t"], how="left")
        merged[indicator] = merged[indicator].fillna(merged[indicator].median())
        bins = np.searchsorted(edges[1:-1], merged[indicator].to_numpy()).astype(np.int8)
        out[k] = {
            "probs": merged[["prob_0","prob_1","prob_2"]].to_numpy(np.float32),
            "label": merged["true_label"].to_numpy(np.int64),
            "mp_t": merged["midprice_t"].to_numpy(np.float64),
            "mp_th": merged["midprice_th"].to_numpy(np.float64),
            "bins": bins,
        }
    return out, edges


def best_threshold_for_subset_de(probs, label, mp_t, mp_th):
    def neg(x):
        Tu, Td, du, dd = x
        pred = gate_asymmetric(probs, float(Tu), float(Td), float(du), float(dd))
        return -float(vectorized_pnl(pred, label, mp_t, mp_th).sum())
    if len(probs) < 500:
        return -np.inf, (0.45, 0.45, 0.10, 0.10)
    result = differential_evolution(
        neg, bounds=DE_BOUNDS, seed=0, maxiter=40, popsize=18,
        polish=True, tol=1e-7, init="sobol")
    return -float(result.fun), tuple(float(v) for v in result.x)


def evaluate_regime_cv(folds_with_regime, n_bins: int):
    """Honest LOSO CV: thresholds picked on 4 folds, applied to held-out fold."""
    fold_keys = sorted(folds_with_regime.keys())
    per_fold_pnl = []
    per_fold_thresh = {}
    for k in fold_keys:
        # Calibration set = other 4 folds pooled
        cal_idx = [j for j in fold_keys if j != k]
        cal_probs = np.concatenate([folds_with_regime[j]["probs"] for j in cal_idx])
        cal_label = np.concatenate([folds_with_regime[j]["label"] for j in cal_idx])
        cal_mp_t = np.concatenate([folds_with_regime[j]["mp_t"]  for j in cal_idx])
        cal_mp_th = np.concatenate([folds_with_regime[j]["mp_th"] for j in cal_idx])
        cal_bins = np.concatenate([folds_with_regime[j]["bins"]  for j in cal_idx])

        bin_thresh = {}
        for b in range(n_bins):
            mask = cal_bins == b
            _, best = best_threshold_for_subset_de(
                cal_probs[mask], cal_label[mask], cal_mp_t[mask], cal_mp_th[mask])
            bin_thresh[b] = best
        per_fold_thresh[k] = bin_thresh

        # Apply to held fold k
        held = folds_with_regime[k]
        pred = np.full(len(held["bins"]), 1, dtype=np.int8)
        for b in range(n_bins):
            mask = held["bins"] == b
            if not mask.any():
                continue
            Tu, Td, du, dd = bin_thresh[b]
            pb = gate_asymmetric(held["probs"][mask], Tu, Td, du, dd)
            pred[mask] = pb
        per_fold_pnl.append(float(vectorized_pnl(pred, held["label"], held["mp_t"], held["mp_th"]).sum()))

    sum_p = sum(per_fold_pnl)
    return {
        "sum_cum_pnl": sum_p,
        "per_fold_pnl": per_fold_pnl,
        "n_pos_folds": int(sum(1 for x in per_fold_pnl if x > 0)),
        "std_cum_pnl": float(np.std(per_fold_pnl, ddof=0)),
        "per_fold_thresholds": {int(k): {int(b): list(v) for b, v in t.items()}
                                for k, t in per_fold_thresh.items()},
    }


def main():
    t0 = time.time()
    folds = load_all_folds()
    regime_df = pd.read_parquet(REGIME_PARQUET)

    all_results = {}
    for indicator in ["regime_intst", "regime_vol", "regime_imb"]:
        for n_bins in [3, 4]:
            t1 = time.time()
            fwr, edges = attach_regime(folds, regime_df, indicator, n_bins)
            res = evaluate_regime_cv(fwr, n_bins)
            res["indicator"] = indicator
            res["n_bins"] = n_bins
            res["bin_edges"] = [float(e) for e in edges]
            res["elapsed_sec"] = time.time() - t1
            key = f"{indicator}_b{n_bins}"
            all_results[key] = res
            print(f"  {key}: CV-OOF sum={res['sum_cum_pnl']:+.4f}, "
                  f"per_fold={[round(x,3) for x in res['per_fold_pnl']]}, "
                  f"pos={res['n_pos_folds']}/5, "
                  f"({res['elapsed_sec']:.1f}s)")

    best_key = max(all_results, key=lambda k: all_results[k]["sum_cum_pnl"])
    print(f"\nBest CV-OOF regime: {best_key} sum={all_results[best_key]['sum_cum_pnl']:+.4f}")

    out = {
        "task": "T30 per-regime threshold sweep — HONEST LOSO CV (4-fold-pick, 1-fold-test)",
        "results": all_results,
        "best_key": best_key,
        "elapsed_sec": time.time() - t0,
    }
    with open(os.path.join(HERE, "per_regime_cv_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
