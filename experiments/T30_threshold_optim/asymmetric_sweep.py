"""Asymmetric (T_up, T_dn, delta_up, delta_dn) sweep on 5-seed OOF h_60.

Allows different gating strength for long vs short — the ensemble may not be
calibrated symmetrically across up/down.

Coarse 4D grid:
  T_up, T_dn  in {0.40..0.65 step 0.025} = 11 each   -> 121
  d_up, d_dn  in {0.00..0.20 step 0.025} = 9  each   ->  81
Total 121*81 = 9801. Vectorized; each combo is dirt cheap.

Then medium-grid local refinement around the best.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from _common import fold_arrays, gate_asymmetric, load_all_folds, vectorized_pnl

HERE = os.path.dirname(os.path.abspath(__file__))


def evaluate_grid(fas, T_up_grid, T_dn_grid, d_up_grid, d_dn_grid):
    """Run all combos. Returns DataFrame sorted desc by sum."""
    rows = []
    label = [fa.label for fa in fas]
    mp_t = [fa.mp_t for fa in fas]
    mp_th = [fa.mp_th for fa in fas]
    probs = [fa.probs for fa in fas]
    for Tu in T_up_grid:
        for Td in T_dn_grid:
            for du in d_up_grid:
                for dd in d_dn_grid:
                    per = []
                    for k in range(5):
                        pred = gate_asymmetric(probs[k], float(Tu), float(Td),
                                               float(du), float(dd))
                        pnl = vectorized_pnl(pred, label[k], mp_t[k], mp_th[k])
                        per.append(float(pnl.sum()))
                    sum_p = sum(per)
                    rows.append({
                        "T_up": float(Tu), "T_dn": float(Td),
                        "d_up": float(du), "d_dn": float(dd),
                        "sum_cum_pnl": sum_p,
                        "std_cum_pnl": float(np.std(per, ddof=0)),
                        "n_pos_folds": int(sum(1 for x in per if x > 0)),
                        **{f"fold{k}_pnl": per[k] for k in range(5)},
                    })
    return pd.DataFrame(rows).sort_values("sum_cum_pnl", ascending=False).reset_index(drop=True)


def main():
    t0 = time.time()
    folds = load_all_folds()
    fas = fold_arrays(folds)

    # Coarse grid
    T_up = np.round(np.arange(0.40, 0.6501, 0.025), 4)   # 11
    T_dn = np.round(np.arange(0.40, 0.6501, 0.025), 4)   # 11
    d_up = np.round(np.arange(0.00, 0.2001, 0.025), 4)   # 9
    d_dn = np.round(np.arange(0.00, 0.2001, 0.025), 4)   # 9

    print(f"Coarse 4D grid: {len(T_up)*len(T_dn)*len(d_up)*len(d_dn)} combos")
    df = evaluate_grid(fas, T_up, T_dn, d_up, d_dn)
    print(f"Coarse done in {time.time()-t0:.1f}s. Top 10:")
    print(df.head(10).to_string(index=False))
    df.to_csv(os.path.join(HERE, "asym_sweep_coarse.csv"), index=False)

    # Refine around best
    best = df.iloc[0]
    print(f"\nRefining around T_up={best['T_up']}, T_dn={best['T_dn']}, "
          f"d_up={best['d_up']}, d_dn={best['d_dn']}")

    def around(c, lo=-0.05, hi=0.06, step=0.01):
        v = np.round(np.arange(c + lo, c + hi, step), 4)
        return np.unique(np.clip(v, 0.30, 0.80))

    Tu2 = around(best["T_up"])
    Td2 = around(best["T_dn"])
    du2 = around(best["d_up"], lo=-0.04, hi=0.05)
    dd2 = around(best["d_dn"], lo=-0.04, hi=0.05)
    du2 = np.unique(np.clip(du2, 0.0, 0.30))
    dd2 = np.unique(np.clip(dd2, 0.0, 0.30))
    print(f"Fine 4D: {len(Tu2)*len(Td2)*len(du2)*len(dd2)} combos")
    df2 = evaluate_grid(fas, Tu2, Td2, du2, dd2)

    df_all = pd.concat([df, df2], ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["T_up","T_dn","d_up","d_dn"]).reset_index(drop=True)
    df_all = df_all.sort_values("sum_cum_pnl", ascending=False).reset_index(drop=True)
    df_all.to_csv(os.path.join(HERE, "asym_sweep_all.csv"), index=False)

    print(f"\nFinal top 15 (combined):")
    print(df_all.head(15).to_string(index=False))

    best = df_all.iloc[0].to_dict()
    elapsed = time.time() - t0

    out = {
        "task": "T30 asymmetric (T_up, T_dn, d_up, d_dn) sweep",
        "n_combos_total": int(len(df_all)),
        "best": {k: (int(best[k]) if k == "n_pos_folds" else float(best[k]))
                 for k in ["T_up", "T_dn", "d_up", "d_dn", "sum_cum_pnl",
                           "std_cum_pnl", "n_pos_folds",
                           "fold0_pnl", "fold1_pnl", "fold2_pnl",
                           "fold3_pnl", "fold4_pnl"]},
        "top20": df_all.head(20).to_dict(orient="records"),
        "elapsed_sec": elapsed,
    }
    with open(os.path.join(HERE, "asym_sweep_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDone in {elapsed:.1f}s. Best sum={best['sum_cum_pnl']:+.4f}")


if __name__ == "__main__":
    main()
