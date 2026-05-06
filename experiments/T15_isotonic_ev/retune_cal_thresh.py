"""Sanity: sweep (T, δ) on CALIBRATED OOF for h_10 to confirm EV ≥ best (T, δ) post-cal."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from src.eval.pnl import _per_horizon_metrics  # noqa: E402

HORIZONS = (5, 10, 20, 40, 60)
T_GRID = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
D_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
FEE = 0.0001


def threshold_pred(probs, T, delta):
    p0 = probs[:, 0]; p1 = probs[:, 1]; p2 = probs[:, 2]
    side_max = np.maximum(p0, p2)
    take = (side_max >= T) & (side_max > p1 + delta)
    side = np.where(p2 > p0, 2, 0)
    return np.where(take, side, 1).astype(np.int64)


def fold_pnl(df, pred):
    m = _per_horizon_metrics(
        pred,
        df["true_label"].to_numpy(np.int64),
        df["midprice_t"].to_numpy(np.float64),
        df["midprice_th"].to_numpy(np.float64),
        fee_rate=FEE,
    )
    return float(m["cum_pnl"]), int(m["n_predictions_active"])


def main():
    summary = {}
    for H in HORIZONS:
        fold_dfs = [pd.read_parquet(os.path.join(HERE, f"loso_pred_h{H}_held{k}_cal.parquet")) for k in range(5)]
        rows = []
        for T in T_GRID:
            for d in D_GRID:
                pf = []
                for k in range(5):
                    df = fold_dfs[k]
                    probs = df[["prob_0_cal", "prob_1_cal", "prob_2_cal"]].to_numpy(np.float32)
                    pred = threshold_pred(probs, T, d)
                    cp, na = fold_pnl(df, pred)
                    pf.append(cp)
                s = sum(pf)
                rows.append({"T": T, "delta": d, "sum": s, "pos": sum(1 for x in pf if x > 0)})
        rows.sort(key=lambda x: x["sum"], reverse=True)
        best = rows[0]
        summary[f"h_{H}"] = {"best_T": best["T"], "best_delta": best["delta"], "best_sum": best["sum"], "best_pos": best["pos"]}
        print(f"h_{H}: best (T={best['T']:.2f}, δ={best['delta']:.2f}) cal-thresh sum={best['sum']:+.4f} pos={best['pos']}/5")
    with open(os.path.join(HERE, "retune_cal_thresh_results.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
