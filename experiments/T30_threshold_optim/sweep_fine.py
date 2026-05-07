"""Fine-grained (T, delta) symmetric grid sweep on 5-seed aug_a OOF h_60.

Grid: T in [0.30..0.75 step 0.01] (46), delta in [0.00..0.30 step 0.01] (31)
     = 1426 combos. Reports best by sum_cum_pnl across 5 LOSO folds.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from _common import (FoldArrays, fold_arrays, gate_symmetric, load_all_folds,
                     vectorized_pnl)

HERE = os.path.dirname(os.path.abspath(__file__))
T_GRID = np.round(np.arange(0.30, 0.7501, 0.01), 4)
D_GRID = np.round(np.arange(0.00, 0.3001, 0.01), 4)


def main():
    t0 = time.time()
    folds = load_all_folds()
    fas = fold_arrays(folds)

    rows = []
    for T in T_GRID:
        for d in D_GRID:
            per = []
            for fa in fas:
                pred = gate_symmetric(fa.probs, float(T), float(d))
                pnl = vectorized_pnl(pred, fa.label, fa.mp_t, fa.mp_th)
                per.append(pnl.sum())
            sum_p = float(sum(per))
            rows.append({
                "T": float(T), "delta": float(d),
                "sum_cum_pnl": sum_p,
                "mean_cum_pnl": sum_p / 5.0,
                "std_cum_pnl": float(np.std(per, ddof=0)),
                "n_pos_folds": int(sum(1 for x in per if x > 0)),
                **{f"fold{k}_pnl": float(per[k]) for k in range(5)},
            })

    df = pd.DataFrame(rows).sort_values("sum_cum_pnl", ascending=False).reset_index(drop=True)
    out_csv = os.path.join(HERE, "sweep_fine_results.csv")
    df.to_csv(out_csv, index=False)

    best = df.iloc[0].to_dict()
    elapsed = time.time() - t0

    print(f"Fine sweep: {len(df)} combos, {elapsed:.1f}s")
    print("Top 15:")
    print(df.head(15).to_string(index=False))
    print(f"\nBest: T={best['T']}, delta={best['delta']}, sum={best['sum_cum_pnl']:+.4f}, "
          f"std={best['std_cum_pnl']:.4f}, pos={int(best['n_pos_folds'])}/5")

    out = {
        "task": "T30 fine-grained symmetric (T, delta) sweep",
        "n_combos": int(len(df)),
        "T_grid": T_GRID.tolist(),
        "D_grid": D_GRID.tolist(),
        "best": {k: float(best[k]) if k != "n_pos_folds" else int(best[k])
                 for k in ["T", "delta", "sum_cum_pnl", "std_cum_pnl",
                           "mean_cum_pnl", "n_pos_folds",
                           "fold0_pnl", "fold1_pnl", "fold2_pnl",
                           "fold3_pnl", "fold4_pnl"]},
        "top20": df.head(20).to_dict(orient="records"),
        "elapsed_sec": elapsed,
    }
    with open(os.path.join(HERE, "sweep_fine_results.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
