"""T24 iter_004b: aggressive threshold sweep on T5b LOSO OOF.

Goal: pick (T, delta) for short horizons (h_5, h_10, h_20) such that
  1. LOSO sum_cum_pnl is still >> 0 (model still works)
  2. per-trade_pnl >> +0.0001 (2x fee buffer; robust to OOD drift)
  3. n_active is small enough to drop fee-noise (selective)

Why: iter_002 short horizons hit per-trade ≈ -fee on platform. Going more
selective on T should let only "obvious" wins through, leaving per-trade
positive even after the LOSO -> platform calibration drift.

For h_40 / h_60 we keep iter_002 settings.

Outputs:
  T24_iter004/sweep_aggressive_h{H}.csv
  T24_iter004/sweep_aggressive_summary.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
T5B = os.path.join(ROOT, "experiments", "T5b_features_multihorizon")
sys.path.insert(0, ROOT)

from src.eval.pnl import _per_horizon_metrics  # noqa: E402

T_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
D_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]


def thresholded_pred(probs: np.ndarray, T: float, delta: float) -> np.ndarray:
    p0 = probs[:, 0]; p1 = probs[:, 1]; p2 = probs[:, 2]
    side_max = np.maximum(p0, p2)
    take = (side_max >= T) & (side_max > p1 + delta)
    pred_side = np.where(p2 > p0, 2, 0)
    return np.where(take, pred_side, 1).astype(np.int8)


def fold_metrics(df: pd.DataFrame, T: float, delta: float) -> dict:
    probs = df[["prob_0", "prob_1", "prob_2"]].to_numpy(np.float32)
    pred = thresholded_pred(probs, T, delta)
    m = _per_horizon_metrics(
        pred,
        df["true_label"].to_numpy(np.int64),
        df["midprice_t"].to_numpy(np.float32),
        df["midprice_th"].to_numpy(np.float32),
        fee_rate=0.0001,
    )
    return {
        "cum_pnl": float(m["cum_pnl"]),
        "single_pnl": float(m["single_pnl"]),
        "accuracy": float(m["accuracy"]),
        "n_active": int(m["n_predictions_active"]),
    }


def sweep_horizon(H: int) -> dict:
    fold_dfs = {}
    for k in range(5):
        p = os.path.join(T5B, f"loso_pred_h{H}_held{k}.parquet")
        if not os.path.exists(p):
            return {"error": f"missing {p}"}
        fold_dfs[k] = pd.read_parquet(p)
    n_total = sum(len(v) for v in fold_dfs.values())
    print(f"\n=== h={H} OOF rows: {n_total:,} ===", flush=True)

    rows = []
    for T in T_GRID:
        for d in D_GRID:
            per_fold = [fold_metrics(fold_dfs[k], T, d) for k in range(5)]
            sum_cum = sum(r["cum_pnl"] for r in per_fold)
            sum_active = sum(r["n_active"] for r in per_fold)
            n_pos = sum(1 for r in per_fold if r["cum_pnl"] > 0)
            mean_acc = float(np.mean([r["accuracy"] for r in per_fold]))
            per_trade = sum_cum / sum_active if sum_active > 0 else 0.0
            rows.append({
                "T": T, "delta": d,
                "sum_cum_pnl": sum_cum,
                "n_pos_folds": n_pos,
                "sum_n_active": sum_active,
                "per_trade_pnl": per_trade,
                "mean_accuracy": mean_acc,
                **{f"f{k}": per_fold[k]["cum_pnl"] for k in range(5)},
                **{f"a{k}": per_fold[k]["n_active"] for k in range(5)},
            })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(HERE, f"sweep_aggressive_h{H}.csv")
    df.sort_values("per_trade_pnl", ascending=False).to_csv(out_csv, index=False)
    print(f"  saved {out_csv}", flush=True)

    # Top by per_trade_pnl with min n_active and n_pos_folds filter
    candidates = df[(df["sum_n_active"] >= 5000) & (df["n_pos_folds"] >= 4)].copy()
    candidates = candidates.sort_values("per_trade_pnl", ascending=False)
    print(f"  top by per_trade_pnl (n_active>=5000, n_pos>=4):", flush=True)
    for _, r in candidates.head(8).iterrows():
        print(f"    T={r['T']:.2f} d={r['delta']:.2f}  "
              f"per_trade={r['per_trade_pnl']:+.6f}  sum={r['sum_cum_pnl']:+.3f}  "
              f"n_act={int(r['sum_n_active']):>6d}  pos={int(r['n_pos_folds'])}", flush=True)

    return {
        "horizon": H,
        "all_rows": rows,
        "best_per_trade_filtered": candidates.head(8).to_dict(orient="records"),
    }


def main():
    horizons = [5, 10, 20, 40, 60]
    t0 = time.time()
    out = {}
    for H in horizons:
        out[H] = sweep_horizon(H)
    out_path = os.path.join(HERE, "sweep_aggressive_summary.json")
    with open(out_path, "w") as f:
        json.dump({
            "task": "T24 aggressive threshold sweep",
            "T_grid": T_GRID, "delta_grid": D_GRID,
            "per_horizon": {f"h{H}": out[H] for H in horizons},
            "elapsed_sec": time.time() - t0,
        }, f, indent=2)
    print(f"\nDone in {time.time()-t0:.1f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
