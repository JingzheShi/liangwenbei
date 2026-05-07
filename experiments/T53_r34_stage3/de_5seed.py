"""T53b DE 4D threshold search on 5-seed-averaged R34 Stage 3 LOSO h_60 OOF.

Reuses the T53 single-seed DE pipeline (de_thresh.py) but with seeds=1,7,13,42,100.

Output: de_results_5seed.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Reuse helpers
sys.path.insert(0, HERE)
from de_thresh import (  # noqa: E402
    N_FOLDS, FEE, PROB_COLS,
    load_fold_avg, fold_arrays_from_dfs, gate_asymmetric, vectorized_pnl,
    coarse_sweep, de_search, neighborhood_check,
)


def main():
    prefix = "loso_pred_h60"
    seeds = [1, 7, 13, 42, 100]
    out_name = "de_results_5seed.json"

    print(f"=== T53b DE 5-seed thresh search prefix={prefix} seeds={seeds} ===", flush=True)

    t0 = time.time()
    folds = {k: load_fold_avg(prefix, k, seeds) for k in range(N_FOLDS)}
    fas = fold_arrays_from_dfs(folds)
    print(f"  loaded {sum(len(f) for f in folds.values()):,} OOF rows in {time.time()-t0:.1f}s", flush=True)

    raw_pf = []
    for fa in fas:
        pred = fa.probs.argmax(axis=1).astype(np.int8)
        raw_pf.append(float(vectorized_pnl(pred, fa.label, fa.mp_t, fa.mp_th).sum()))
    print(f"  raw argmax sum={sum(raw_pf):+.4f}  per_fold={[round(x,3) for x in raw_pf]}", flush=True)

    print(f"\n[1/3] Coarse asymmetric sweep ...", flush=True)
    t1 = time.time()
    coarse_df = coarse_sweep(fas)
    print(f"  coarse done in {time.time()-t1:.1f}s. Top 10:", flush=True)
    print(coarse_df.head(10)[["T_up", "T_dn", "d_up", "d_dn", "sum_cum_pnl", "n_pos_folds"]].to_string(index=False))

    print(f"\n[2/3] DE 4D differential evolution ...", flush=True)
    bounds = [(0.34, 0.75), (0.34, 0.75), (0.0, 0.30), (0.0, 0.30)]
    de_runs = de_search(fas, bounds, seeds=(0, 1, 2, 7, 42))
    best = de_runs[0]
    print(f"\n  Best DE run: sum={best['sum_cum_pnl']:+.4f} at "
          f"(T_up={best['T_up']:.4f}, T_dn={best['T_dn']:.4f}, "
          f"d_up={best['d_up']:.4f}, d_dn={best['d_dn']:.4f})", flush=True)
    print(f"  per_fold: {[round(x,3) for x in best['per_fold_pnl']]} "
          f"std={best['std_cum_pnl']:.4f} pos={best['n_pos_folds']}/5", flush=True)

    print(f"\n[3/3] Neighborhood sanity check around DE optimum ...", flush=True)
    nb = neighborhood_check(fas, best["T_up"], best["T_dn"], best["d_up"], best["d_dn"])
    print(f"  neighborhood top 10:", flush=True)
    print(nb.head(10).to_string(index=False))
    n_robust = int((nb["sum_cum_pnl"] >= best["sum_cum_pnl"] - 0.5).sum())
    print(f"  configs within 0.5 of optimum: {n_robust} of {len(nb)}", flush=True)

    out = {
        "prefix": prefix,
        "seeds": seeds,
        "raw_argmax_sum": float(sum(raw_pf)),
        "raw_argmax_per_fold": raw_pf,
        "coarse_top10": coarse_df.head(10).to_dict(orient="records"),
        "coarse_best_sum": float(coarse_df.iloc[0]["sum_cum_pnl"]),
        "de_runs": de_runs,
        "de_best": best,
        "neighborhood_top10": nb.head(10).to_dict(orient="records"),
        "neighborhood_robust_count": n_robust,
    }
    out_path = os.path.join(HERE, out_name)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
