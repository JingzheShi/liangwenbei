"""Threshold post-processor experiment.

Theory: LightGBM 默认 argmax 阈值 = 1/3 太低，让 brittle 模型在低置信度上乱出手。
强制要求 max(prob_0, prob_2) > T 且 > prob_1 + delta 时才出手 → 减少假阳性 → 减亏。

We sweep (T, delta) on the OOF predictions:
  - "OOF" here = LOSO fold concat: each held-out sym's predictions, where the model never saw
    that sym during training.  Stitching all 5 folds gives an honest cross-sym OOF set.

For each (T, delta):
  if max(prob_0, prob_2) > T and max(prob_0, prob_2) > prob_1 + delta:
      pred = argmax(prob_0, prob_2)
  else:
      pred = 1 (flat)

Report best (T, delta) by cum_pnl on:
  - Per-fold OOF
  - Aggregated all-fold OOF

Also report on the original T2 Scheme B test (sym in 0..4, dates 96..119, IID) for comparison.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from src.eval.pnl import _per_horizon_metrics  # noqa: E402


def apply_threshold(probs: np.ndarray, T: float, delta: float) -> np.ndarray:
    """probs: (N, 3). Returns (N,) int8 in {0,1,2}."""
    p0 = probs[:, 0]
    p1 = probs[:, 1]
    p2 = probs[:, 2]
    side_max = np.maximum(p0, p2)
    take_side = (side_max > T) & (side_max > p1 + delta)
    pred_side = np.where(p2 > p0, 2, 0)
    pred = np.where(take_side, pred_side, 1).astype(np.int8)
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="A")
    ap.add_argument("--T-grid", default="0.34,0.38,0.42,0.46,0.50,0.55,0.60,0.65,0.70")
    ap.add_argument("--delta-grid", default="0.0,0.03,0.05,0.10,0.15,0.20")
    args = ap.parse_args()

    T_grid = [float(x) for x in args.T_grid.split(",")]
    d_grid = [float(x) for x in args.delta_grid.split(",")]
    print(f"=== threshold post-processor scheme {args.scheme} ===", flush=True)
    print(f"T_grid: {T_grid}\nd_grid: {d_grid}", flush=True)

    # ---- Load LOSO predictions across 5 folds ----
    dfs = []
    for k in range(5):
        p = os.path.join(HERE, f"loso_pred_scheme{args.scheme}_held{k}.parquet")
        if not os.path.isfile(p):
            print(f"  warn: missing {p}", flush=True)
            continue
        df = pd.read_parquet(p)
        df["held_out_sym"] = k
        dfs.append(df)
    all_oof = pd.concat(dfs, ignore_index=True)
    print(f"loaded LOSO OOF: {len(all_oof):,} rows from {len(dfs)} folds", flush=True)

    # ---- Baseline (default argmax) ----
    probs = all_oof[["prob_0", "prob_1", "prob_2"]].to_numpy(np.float32)
    y = all_oof["true_label_60"].to_numpy(np.int64)
    mp_t = all_oof["midprice_t"].to_numpy(np.float32)
    mp_t60 = all_oof["midprice_t60"].to_numpy(np.float32)

    # Default argmax (no thresholding)
    pred_default = probs.argmax(axis=1).astype(np.int8)
    m_default = _per_horizon_metrics(pred_default, y, mp_t, mp_t60, fee_rate=0.0001)
    print(f"\n=== Baseline (default argmax) on LOSO OOF (5 folds, all syms) ===", flush=True)
    print(f"  cum_pnl    = {m_default['cum_pnl']:+.4f}", flush=True)
    print(f"  single_pnl = {m_default['single_pnl']:+.6f}", flush=True)
    print(f"  accuracy   = {m_default['accuracy']:.4f}", flush=True)
    print(f"  n_active   = {m_default['n_predictions_active']:,}", flush=True)

    # ---- Sweep ----
    rows = []
    for T in T_grid:
        for delta in d_grid:
            pred = apply_threshold(probs, T, delta)
            m = _per_horizon_metrics(pred, y, mp_t, mp_t60, fee_rate=0.0001)
            rows.append({
                "T": T, "delta": delta,
                "cum_pnl": m["cum_pnl"], "single_pnl": m["single_pnl"],
                "accuracy": m["accuracy"], "n_active": m["n_predictions_active"],
                "f0_5_macro": m["f0_5_macro"],
            })
    sweep = pd.DataFrame(rows)
    sweep_path = os.path.join(HERE, f"threshold_sweep_scheme{args.scheme}.csv")
    sweep.to_csv(sweep_path, index=False)
    print(f"\nfull sweep saved -> {sweep_path}", flush=True)

    # Top 5 by cum_pnl
    print("\nTop 10 (T, delta) by LOSO OOF cum_pnl:", flush=True)
    top = sweep.sort_values("cum_pnl", ascending=False).head(10)
    print(top.to_string(index=False), flush=True)

    # Best
    best = sweep.iloc[sweep["cum_pnl"].idxmax()]
    print(f"\nBest: T={best['T']} delta={best['delta']}: "
          f"cum_pnl={best['cum_pnl']:+.4f} (vs baseline {m_default['cum_pnl']:+.4f})", flush=True)

    # Per-fold breakdown for best (T, delta)
    print(f"\nPer-fold breakdown @ best (T={best['T']}, delta={best['delta']}):", flush=True)
    for k in range(5):
        if all_oof[all_oof["held_out_sym"] == k].empty:
            continue
        sub = all_oof[all_oof["held_out_sym"] == k]
        ps = sub[["prob_0", "prob_1", "prob_2"]].to_numpy(np.float32)
        pred_k = apply_threshold(ps, best["T"], best["delta"])
        m = _per_horizon_metrics(
            pred_k, sub["true_label_60"].to_numpy(np.int64),
            sub["midprice_t"].to_numpy(np.float32),
            sub["midprice_t60"].to_numpy(np.float32), fee_rate=0.0001
        )
        print(f"  sym={k}: cum_pnl={m['cum_pnl']:+.4f} acc={m['accuracy']:.4f} "
              f"n_active={m['n_predictions_active']:,}", flush=True)

    # ---- Same sweep on T2 Scheme B test (IID; sanity check that thresholding doesn't hurt IID) ----
    t2_pred_path = os.path.join(ROOT, "experiments", "T2_gbdt_lgbm",
                                f"test_predictions_label60_scheme{args.scheme}.parquet")
    if os.path.isfile(t2_pred_path):
        t2 = pd.read_parquet(t2_pred_path)
        ps = t2[["prob_0", "prob_1", "prob_2"]].to_numpy(np.float32)
        y2 = t2["true_label_60"].to_numpy(np.int64)
        mt = t2["midprice_t"].to_numpy(np.float32)
        mt60 = t2["midprice_t60"].to_numpy(np.float32)
        baseline_t2 = _per_horizon_metrics(ps.argmax(1), y2, mt, mt60, fee_rate=0.0001)
        pred_t2 = apply_threshold(ps, best["T"], best["delta"])
        m_t2 = _per_horizon_metrics(pred_t2, y2, mt, mt60, fee_rate=0.0001)
        print(f"\n=== Sanity: on T2 Scheme {args.scheme} IID test (sym 0-4 train+test) ===", flush=True)
        print(f"  baseline (argmax)               : cum_pnl={baseline_t2['cum_pnl']:+.4f}", flush=True)
        print(f"  best (T={best['T']}, delta={best['delta']}): cum_pnl={m_t2['cum_pnl']:+.4f}", flush=True)

    out = {
        "scheme": args.scheme,
        "baseline_loso_oof_cum_pnl": float(m_default["cum_pnl"]),
        "best_T": float(best["T"]),
        "best_delta": float(best["delta"]),
        "best_loso_oof_cum_pnl": float(best["cum_pnl"]),
        "best_loso_oof_acc": float(best["accuracy"]),
        "best_loso_oof_n_active": int(best["n_active"]),
        "T_grid": T_grid, "delta_grid": d_grid,
    }
    out_path = os.path.join(HERE, f"threshold_results_scheme{args.scheme}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n--> {out_path}", flush=True)


if __name__ == "__main__":
    main()
