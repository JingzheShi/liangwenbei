"""T83 main experiment runner — V2 (faster).

Tests several robust-threshold strategies on iter_013 OOF predictions and
evaluates each via train/eval splits to estimate the unbiased eval PnL.

Splits:
  S1 — date 50/50 forward: train = dates 96..107, eval = dates 108..119
  S2 — date 50/50 backward: train = dates 108..119, eval = dates 96..107
  S3 — random row 50/50 (seed=0)

Strategies:
  A — sym_grid                   (1D, k ∈ [0.25, 3.5])
  B — de_asym                    (2D, baseline = iter_013 form)
  C — de_reg_lam0.1, lam1, lam10 (DE asym + symmetry penalty)
  D — kfold_de_5_date            (K-fold-on-train inner CV-DE, median thr)
  E — kfold_de_5_row             (same but row-folds)
  F — bootstrap_de_n15_f0.7      (bootstrap median DE; 15 boots)
  G — quantile_grid              (grid q_up, q_dn over pred quantiles)
  H — ensemble_avg_3             (avg of sym_grid + kfold_5_date + bootstrap)
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
    fit_de_regularized, split_by_date, split_random, FEE,
)
from extra_strategies import fit_quantile_grid, fit_ensemble_avg


def run_strategies(df_train: pd.DataFrame, df_eval: pd.DataFrame, label: str) -> list:
    out = []
    print(f"\n=== Split: {label} | train={len(df_train)} eval={len(df_eval)} ===", flush=True)

    def go(name, rec):
        ev = eval_strategy(df_eval, rec["thr_up"], rec["thr_dn"])
        rec.update({"split": label, "eval_pnl": ev["eval_pnl"],
                    "gap": ev["eval_pnl"] - rec["train_pnl"], "n_active_eval": ev["n_active"]})
        # drop verbose internals for json
        for k in ("de_runs", "fold_thrs", "components"):
            rec.pop(k, None)
        print(f"  [{name}] thr=({rec['thr_up']:.2e},{rec['thr_dn']:.2e}) "
              f"train={rec['train_pnl']:.3f} eval={rec['eval_pnl']:.3f} gap={rec['gap']:.3f}", flush=True)
        out.append(rec)

    # A
    t0 = time.time()
    rec = fit_symmetric_grid(df_train); rec["strategy"] = "sym_grid"
    go("sym_grid", rec)
    print(f"    ({time.time()-t0:.1f}s)", flush=True)

    # B
    t0 = time.time()
    rec = fit_de_asym(df_train, de_seeds=(0, 1, 2), maxiter=50, popsize=18)
    go("de_asym", rec)
    print(f"    ({time.time()-t0:.1f}s)", flush=True)

    # C
    for lam in (0.1, 1.0, 10.0):
        t0 = time.time()
        rec = fit_de_regularized(df_train, lam=lam, de_seeds=(0,), maxiter=50, popsize=18)
        go(f"de_reg_lam{lam}", rec)
        print(f"    ({time.time()-t0:.1f}s)", flush=True)

    # D
    t0 = time.time()
    rec = fit_de_asym_kfold(df_train, n_folds=5, fold_axis="date", de_seed=0,
                             maxiter=40, popsize=16)
    go("kfold5_date", rec)
    print(f"    ({time.time()-t0:.1f}s)", flush=True)

    # E
    t0 = time.time()
    rec = fit_de_asym_kfold(df_train, n_folds=5, fold_axis="row", de_seed=0,
                             maxiter=40, popsize=16)
    go("kfold5_row", rec)
    print(f"    ({time.time()-t0:.1f}s)", flush=True)

    # F
    t0 = time.time()
    rec = fit_bootstrap_de(df_train, n_boot=15, frac=0.7, seed=42, maxiter=35, popsize=14)
    go("bootstrap15", rec)
    print(f"    ({time.time()-t0:.1f}s)", flush=True)

    # G
    t0 = time.time()
    rec = fit_quantile_grid(df_train); rec["strategy"] = "quantile_grid"
    go("quantile_grid", rec)
    print(f"    ({time.time()-t0:.1f}s)", flush=True)

    # H
    t0 = time.time()
    rec = fit_ensemble_avg(df_train); rec["strategy"] = "ensemble_avg_3"
    go("ensemble_avg_3", rec)
    print(f"    ({time.time()-t0:.1f}s)", flush=True)

    return out


def main():
    df = load_pred_avg()
    print(f"Loaded {len(df)} rows. Dates {df['date'].min()}..{df['date'].max()}, {df['sym'].nunique()} syms.")

    a_full = gate_asym(df["pred"].to_numpy(), 3.723e-4, 1.613e-4)
    full_pnl = sym_sum(df, a_full)
    print(f"Full-fit iter_013 thr replication: {full_pnl:.4f}")

    all_records = []
    train, eval_ = split_by_date(df, train_dates=list(range(96, 108)), eval_dates=list(range(108, 120)))
    all_records += run_strategies(train, eval_, "S1_date_fwd")

    train, eval_ = split_by_date(df, train_dates=list(range(108, 120)), eval_dates=list(range(96, 108)))
    all_records += run_strategies(train, eval_, "S2_date_bwd")

    train, eval_ = split_random(df, frac_train=0.5, seed=0)
    all_records += run_strategies(train, eval_, "S3_random_50")

    out_json = os.path.join(HERE, "results_main.json")
    with open(out_json, "w") as f:
        json.dump({"iter013_full_replication": full_pnl, "records": all_records}, f, indent=2)
    print(f"\nWrote {out_json}")

    df_rec = pd.DataFrame(all_records)

    print("\n=== Summary table ===")
    cols = ["split", "strategy", "thr_up", "thr_dn", "train_pnl", "eval_pnl", "gap"]
    print(df_rec[cols].to_string(index=False))

    print("\n=== Aggregate by strategy (S1+S2 mean, date-based, most realistic) ===")
    df_date = df_rec[df_rec["split"].isin(["S1_date_fwd", "S2_date_bwd"])]
    agg = df_date.groupby("strategy").agg(
        mean_eval=("eval_pnl", "mean"),
        std_eval=("eval_pnl", "std"),
        mean_train=("train_pnl", "mean"),
        mean_gap=("gap", "mean"),
    ).sort_values("mean_eval", ascending=False)
    print(agg.to_string())


if __name__ == "__main__":
    main()
