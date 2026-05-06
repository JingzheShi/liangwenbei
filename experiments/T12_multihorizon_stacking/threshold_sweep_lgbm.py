"""T12: Threshold sweep on LightGBM meta OOF (h_10)."""
from __future__ import annotations

import json
import os
import sys
import time
from itertools import product

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from src.eval.pnl import _per_horizon_metrics  # noqa: E402

T_GRID = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
D_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]


def main():
    df = pd.read_parquet(os.path.join(HERE, "meta_oof_h10_lgbm.parquet"))
    H = 10
    print(f"Loaded LGBM meta OOF: {df.shape}", flush=True)

    rows = []
    for T, d in product(T_GRID, D_GRID):
        per_fold = []
        for k in range(5):
            sub = df[df.sym == k].reset_index(drop=True)
            probs = sub[["meta_p0", "meta_p1", "meta_p2"]].to_numpy(np.float32)
            p0, p1, p2 = probs[:, 0], probs[:, 1], probs[:, 2]
            side_max = np.maximum(p0, p2)
            take = (side_max >= T) & (side_max > p1 + d)
            side_pred = np.where(p2 > p0, 2, 0)
            pred = np.where(take, side_pred, 1).astype(np.int64)
            m = _per_horizon_metrics(
                pred, sub[f"true_label_{H}"].to_numpy(np.int64),
                sub["midprice_t"].to_numpy(np.float32),
                sub[f"midprice_th_{H}"].to_numpy(np.float32),
                fee_rate=0.0001,
            )
            per_fold.append({"cum_pnl": float(m["cum_pnl"]),
                             "n_active": int(m["n_predictions_active"]),
                             "accuracy": float(m["accuracy"])})
        sum_cum = sum(r["cum_pnl"] for r in per_fold)
        n_pos = int(sum(1 for r in per_fold if r["cum_pnl"] > 0))
        rows.append({
            "T": T, "delta": d,
            "sum_cum_pnl": sum_cum, "n_pos_folds": n_pos,
            "sum_n_active": int(sum(r["n_active"] for r in per_fold)),
            "mean_accuracy": float(np.mean([r["accuracy"] for r in per_fold])),
            **{f"fold{k}_pnl": per_fold[k]["cum_pnl"] for k in range(5)},
        })
    df_sweep = pd.DataFrame(rows).sort_values("sum_cum_pnl", ascending=False)
    df_sweep.to_csv(os.path.join(HERE, "sweep_results_h10_meta_lgbm.csv"),
                    index=False)
    print("\n=== Top-10 (LGBM meta) ===", flush=True)
    print(df_sweep.head(10).to_string(index=False), flush=True)

    best = df_sweep.iloc[0].to_dict()
    print(f"\nLGBM meta BEST: T={best['T']} d={best['delta']} "
          f"sum={best['sum_cum_pnl']:+.4f} pos={best['n_pos_folds']}/5",
          flush=True)
    out = {
        "task": "T12 LGBM-meta OOF threshold sweep on h_10",
        "iter002_h10_baseline_sum": 21.859534071535453,
        "best": {k: float(v) if isinstance(v, (int, float, np.floating))
                 else v for k, v in best.items()},
        "top10_grid": df_sweep.head(10).to_dict(orient="records"),
    }
    with open(os.path.join(HERE, "threshold_results_lgbm.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)


if __name__ == "__main__":
    main()
