"""Sanity check the DE optimum is a robust ridge, not a spike of OOF noise.

Sweeps a fine grid around the DE optimum (T_up=0.448, T_dn=0.391, d_up=0.260,
d_dn=0.024) to verify nearby configurations also score high.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from _common import fold_arrays, gate_asymmetric, load_all_folds, vectorized_pnl

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    t0 = time.time()
    folds = load_all_folds()
    fas = fold_arrays(folds)

    Tu0, Td0, du0, dd0 = 0.448, 0.391, 0.260, 0.024

    # Sweep ±0.06 in T, ±0.10 in d, step 0.01
    Tu_grid = np.round(np.arange(max(0.34, Tu0 - 0.06), Tu0 + 0.061, 0.01), 4)
    Td_grid = np.round(np.arange(max(0.34, Td0 - 0.06), Td0 + 0.061, 0.01), 4)
    du_grid = np.round(np.arange(max(0.0, du0 - 0.10), min(0.30, du0 + 0.10) + 1e-9, 0.01), 4)
    dd_grid = np.round(np.arange(max(0.0, dd0 - 0.04), min(0.30, dd0 + 0.06) + 1e-9, 0.01), 4)

    print(f"Neighborhood: |Tu|={len(Tu_grid)}, |Td|={len(Td_grid)}, "
          f"|du|={len(du_grid)}, |dd|={len(dd_grid)} = "
          f"{len(Tu_grid)*len(Td_grid)*len(du_grid)*len(dd_grid)} combos")

    rows = []
    label = [fa.label for fa in fas]
    mp_t = [fa.mp_t for fa in fas]
    mp_th = [fa.mp_th for fa in fas]
    probs = [fa.probs for fa in fas]
    for Tu in Tu_grid:
        for Td in Td_grid:
            for du in du_grid:
                for dd in dd_grid:
                    per = []
                    for k in range(5):
                        pred = gate_asymmetric(probs[k], float(Tu), float(Td),
                                               float(du), float(dd))
                        per.append(float(vectorized_pnl(pred, label[k], mp_t[k], mp_th[k]).sum()))
                    sum_p = sum(per)
                    rows.append({
                        "T_up": float(Tu), "T_dn": float(Td),
                        "d_up": float(du), "d_dn": float(dd),
                        "sum_cum_pnl": sum_p,
                        "n_pos_folds": int(sum(1 for x in per if x > 0)),
                        "std_cum_pnl": float(np.std(per, ddof=0)),
                        **{f"fold{k}_pnl": per[k] for k in range(5)},
                    })

    df = pd.DataFrame(rows).sort_values("sum_cum_pnl", ascending=False).reset_index(drop=True)
    df.to_csv(os.path.join(HERE, "de_neighborhood.csv"), index=False)
    print(f"\nDone in {time.time()-t0:.1f}s. Top 25:")
    print(df.head(25).to_string(index=False))

    # Distribution stats
    print(f"\n>= 13.0: {(df['sum_cum_pnl']>=13.0).sum()} of {len(df)} configs")
    print(f">= 12.5: {(df['sum_cum_pnl']>=12.5).sum()} of {len(df)} configs")
    print(f">= 12.0: {(df['sum_cum_pnl']>=12.0).sum()} of {len(df)} configs")
    print(f"max sum_cum_pnl: {df['sum_cum_pnl'].max():+.4f}")
    print(f"all 5 pos fold count: {(df['n_pos_folds']==5).sum()}")

    out = {
        "task": "T30 DE-optimum neighborhood sanity check",
        "center": {"T_up": Tu0, "T_dn": Td0, "d_up": du0, "d_dn": dd0},
        "n_combos": int(len(df)),
        "top25": df.head(25).to_dict(orient="records"),
        "stats": {
            "max_sum": float(df["sum_cum_pnl"].max()),
            "n_ge_13": int((df["sum_cum_pnl"]>=13.0).sum()),
            "n_ge_125": int((df["sum_cum_pnl"]>=12.5).sum()),
            "n_ge_12": int((df["sum_cum_pnl"]>=12.0).sum()),
            "n_5pos": int((df["n_pos_folds"]==5).sum()),
        },
        "elapsed_sec": time.time() - t0,
    }
    with open(os.path.join(HERE, "de_neighborhood_results.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
