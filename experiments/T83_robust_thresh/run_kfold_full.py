"""T83 follow-up — exhaustive K-fold CV evaluation on the full 442k OOF.

For each candidate strategy we hold out one fold and fit on the rest, evaluate
on the held-out fold. The resulting K eval PnLs are summed (an unbiased estimate
of the strategy's "first-time" PnL on a similar distribution).

This is more rigorous than the 50/50 split because every row contributes to one
eval fold exactly, and strategies that overfit folds are penalized symmetrically.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from core import (
    load_pred_avg, sym_sum, gate_asym, eval_strategy,
    fit_symmetric_grid, fit_de_asym, fit_de_asym_kfold, fit_bootstrap_de,
    fit_de_regularized, FEE,
)


def date_kfold_assignments(df: pd.DataFrame, n_folds: int):
    dates = sorted(df["date"].unique())
    chunks = np.array_split(dates, n_folds)
    masks = []
    for ch in chunks:
        masks.append(df["date"].isin(ch.tolist()).to_numpy())
    return masks, [list(map(int, ch)) for ch in chunks]


def main():
    df = load_pred_avg()
    n_folds = 6  # 24 dates / 6 = 4 dates per fold
    masks, fold_dates = date_kfold_assignments(df, n_folds)

    print(f"Date K-fold ({n_folds} folds; 4 dates each):")
    for i, dt in enumerate(fold_dates):
        print(f"  fold {i}: {dt}")

    strategies = [
        ("sym_grid", lambda dtr: fit_symmetric_grid(dtr)),
        ("de_asym", lambda dtr: fit_de_asym(dtr, de_seeds=(0, 1, 2), maxiter=50, popsize=18)),
        ("de_reg_lam1", lambda dtr: fit_de_regularized(dtr, lam=1.0, de_seeds=(0,), maxiter=50, popsize=16)),
        ("de_reg_lam10", lambda dtr: fit_de_regularized(dtr, lam=10.0, de_seeds=(0,), maxiter=50, popsize=16)),
        ("kfold_de_5_date", lambda dtr: fit_de_asym_kfold(dtr, n_folds=5, fold_axis="date", maxiter=40, popsize=16)),
        ("kfold_de_5_row", lambda dtr: fit_de_asym_kfold(dtr, n_folds=5, fold_axis="row", maxiter=40, popsize=16)),
        ("bootstrap30", lambda dtr: fit_bootstrap_de(dtr, n_boot=20, frac=0.7, seed=42, maxiter=35, popsize=14)),
    ]

    results = []
    fold_evals = {name: [] for name, _ in strategies}
    fold_thrs = {name: [] for name, _ in strategies}

    for fi in range(n_folds):
        eval_mask = masks[fi]
        df_train = df[~eval_mask].reset_index(drop=True)
        df_eval = df[eval_mask].reset_index(drop=True)
        print(f"\n--- Fold {fi}: train={len(df_train)} eval={len(df_eval)} (eval dates={fold_dates[fi]})")

        for name, fitter in strategies:
            t0 = time.time()
            rec = fitter(df_train)
            ev = eval_strategy(df_eval, rec["thr_up"], rec["thr_dn"])
            fold_evals[name].append(ev["eval_pnl"])
            fold_thrs[name].append((rec["thr_up"], rec["thr_dn"]))
            results.append({
                "fold": fi, "strategy": name,
                "thr_up": rec["thr_up"], "thr_dn": rec["thr_dn"],
                "train_pnl": rec.get("train_pnl"), "eval_pnl": ev["eval_pnl"],
                "n_active": ev["n_active"], "per_sym": ev["per_sym"],
                "elapsed_s": float(time.time() - t0),
            })
            print(f"  [{name}] thr=({rec['thr_up']:.2e},{rec['thr_dn']:.2e}) "
                  f"train={rec.get('train_pnl', float('nan')):.3f} eval={ev['eval_pnl']:.3f} ({time.time()-t0:.1f}s)")

    print("\n=== K-fold aggregate (sum of 6 eval folds = unbiased full PnL estimate) ===")
    agg_rows = []
    for name in fold_evals:
        evals = np.array(fold_evals[name])
        agg_rows.append({
            "strategy": name,
            "kfold_sum_eval": float(evals.sum()),
            "kfold_mean_per_fold": float(evals.mean()),
            "kfold_std_per_fold": float(evals.std()),
            "thr_up_med": float(np.median([t[0] for t in fold_thrs[name]])),
            "thr_dn_med": float(np.median([t[1] for t in fold_thrs[name]])),
        })
    df_agg = pd.DataFrame(agg_rows).sort_values("kfold_sum_eval", ascending=False)
    print(df_agg.to_string(index=False))

    out_json = os.path.join(HERE, "results_kfold.json")
    with open(out_json, "w") as f:
        json.dump({"n_folds": n_folds, "fold_dates": fold_dates,
                   "results": results, "aggregate": agg_rows}, f, indent=2)
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
