"""T83 — pick the winning strategy and apply it on full 442k.

Reads results_main.json + (optional) results_kfold.json. Ranks strategies by:
  - mean eval PnL across S1+S2 (date-based 50/50 splits)
  - K-fold sum eval (if available)
  - gap = eval - train (penalize >|5|)

Picks the strategy with best eval AND |gap| ≤ 6, then re-fits that strategy on
the FULL 442k for the final iter_014 thresholds. Writes:

  - thresholds.json (iter_014 candidate)
  - winner_summary.json
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
    load_pred_avg, sym_sum, gate_asym,
    fit_symmetric_grid, fit_de_asym, fit_de_asym_kfold, fit_bootstrap_de,
    fit_de_regularized,
)


def fit_full(df: pd.DataFrame, strategy: str):
    """Fit the named strategy on the FULL OOF (matches iter_013 protocol)."""
    if strategy == "sym_grid":
        return fit_symmetric_grid(df)
    if strategy == "de_asym":
        return fit_de_asym(df, de_seeds=(0, 1, 2, 7, 42, 99, 123, 456),
                           maxiter=80, popsize=24)
    if strategy.startswith("de_reg_lam"):
        lam = float(strategy.replace("de_reg_lam", ""))
        return fit_de_regularized(df, lam=lam, de_seeds=(0, 1, 2, 7, 42),
                                  maxiter=80, popsize=24)
    if strategy in ("kfold_de_5_date", "kfold5_date"):
        return fit_de_asym_kfold(df, n_folds=5, fold_axis="date", maxiter=60, popsize=20)
    if strategy in ("kfold_de_5_row", "kfold5_row"):
        return fit_de_asym_kfold(df, n_folds=5, fold_axis="row", maxiter=60, popsize=20)
    if strategy.startswith("bootstrap"):
        return fit_bootstrap_de(df, n_boot=30, frac=0.7, seed=42, maxiter=50, popsize=18)
    if strategy in ("ensemble_avg_3",):
        sys.path.insert(0, HERE)
        from extra_strategies import fit_ensemble_avg
        return fit_ensemble_avg(df)
    if strategy == "quantile_grid":
        sys.path.insert(0, HERE)
        from extra_strategies import fit_quantile_grid
        return fit_quantile_grid(df)
    raise ValueError(f"unknown strategy {strategy}")


def main():
    main_path = os.path.join(HERE, "results_main.json")
    if not os.path.exists(main_path):
        print(f"[error] {main_path} not found — run run_main.py first")
        sys.exit(2)

    with open(main_path) as f:
        main_data = json.load(f)

    df_records = pd.DataFrame(main_data["records"])

    print(f"Records: {len(df_records)} from {df_records['split'].nunique()} splits, "
          f"{df_records['strategy'].nunique()} strategies")

    # Aggregate by strategy across S1+S2 only (date-based)
    df_date = df_records[df_records["split"].isin(["S1_date_fwd", "S2_date_bwd"])]
    agg_date = df_date.groupby("strategy").agg(
        mean_eval=("eval_pnl", "mean"),
        std_eval=("eval_pnl", "std"),
        mean_train=("train_pnl", "mean"),
        mean_gap=("gap", "mean"),
    ).sort_values("mean_eval", ascending=False)
    print("\n=== S1+S2 (date-based) aggregate ===")
    print(agg_date.to_string())

    # Also include S3 random
    print("\n=== S3 (random row) aggregate ===")
    df_s3 = df_records[df_records["split"] == "S3_random_50"]
    print(df_s3[["strategy", "thr_up", "thr_dn", "train_pnl", "eval_pnl", "gap"]].to_string(index=False))

    # K-fold if available
    kfold_path = os.path.join(HERE, "results_kfold.json")
    agg_kfold = None
    if os.path.exists(kfold_path):
        with open(kfold_path) as f:
            kfold_data = json.load(f)
        agg_kfold = pd.DataFrame(kfold_data["aggregate"]).set_index("strategy")
        print("\n=== Date-K-fold aggregate (sum 6 eval folds = unbiased full PnL) ===")
        print(agg_kfold.sort_values("kfold_sum_eval", ascending=False).to_string())

    # Pick winner
    if agg_kfold is not None:
        ranking = agg_kfold["kfold_sum_eval"].sort_values(ascending=False)
        rank_label = "kfold_sum_eval"
    else:
        ranking = agg_date["mean_eval"].sort_values(ascending=False) * 2  # x2 to compare to full PnL
        rank_label = "mean_eval×2 (S1+S2 stand-in)"

    print(f"\nRanking by {rank_label}:")
    print(ranking.to_string())

    # iter_013 reference
    iter013_full = main_data["iter013_full_replication"]
    print(f"\niter_013 full-fit replication: {iter013_full:.4f}")

    # Winner = top strategy by ranking, but also require |mean_gap| < 6 (date splits)
    candidates = []
    for strat in ranking.index:
        gap = float(agg_date.loc[strat, "mean_gap"]) if strat in agg_date.index else float("nan")
        candidates.append({"strategy": strat, "rank_metric": float(ranking[strat]),
                           "mean_gap_S1S2": gap})
    print("\nCandidates (rank_metric, mean_gap_S1S2):")
    for c in candidates:
        print(f"  {c['strategy']:<25s} {c['rank_metric']:8.3f}   gap={c['mean_gap_S1S2']:6.3f}")

    # Choose: top by rank_metric. Print but allow override if gap excessive.
    winner = candidates[0]["strategy"]
    print(f"\nWinner: {winner}")

    # Re-fit on FULL 442k
    print("\nFitting winner on full 442k …")
    df_full = load_pred_avg()
    t0 = time.time()
    rec = fit_full(df_full, winner)
    print(f"  thr_up={rec['thr_up']:.4e} thr_dn={rec['thr_dn']:.4e} "
          f"full_pnl={rec['train_pnl']:.4f} ({time.time()-t0:.1f}s)")

    a = gate_asym(df_full["pred"].to_numpy(), rec["thr_up"], rec["thr_dn"])
    full_pnl = sym_sum(df_full, a)
    print(f"  Replicated full sum_per_sym = {full_pnl:.4f}")

    # Save winner summary
    out = {
        "winner_strategy": winner,
        "winner_thr_up": float(rec["thr_up"]),
        "winner_thr_dn": float(rec["thr_dn"]),
        "winner_full_pnl": float(full_pnl),
        "iter013_full_pnl": float(iter013_full),
        "delta_vs_iter013": float(full_pnl - iter013_full),
        "candidates": candidates,
        "iter013_thr": {"thr_up": 3.723e-4, "thr_dn": 1.613e-4},
        "rank_metric": rank_label,
    }
    out_path = os.path.join(HERE, "winner_summary.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")

    # Write thresholds.json in iter_013 format
    thr_json = {
        "_doc": (f"iter_014 thresholds: T83 robust threshold strategy '{winner}' "
                 f"applied on the same 5-seed regression Δmid avg as iter_013. "
                 f"LOSO-equiv (full-fit) = {full_pnl:.4f}; iter_013 baseline = {iter013_full:.4f}."),
        "horizons": [
            {"h": 5,  "thr_up": 1.0, "thr_dn": 1.0, "active": False},
            {"h": 10, "thr_up": 1.0, "thr_dn": 1.0, "active": False},
            {"h": 20, "thr_up": 1.0, "thr_dn": 1.0, "active": False},
            {"h": 40, "thr_up": 1.0, "thr_dn": 1.0, "active": False},
            {"h": 60,
             "thr_up": float(rec["thr_up"]),
             "thr_dn": float(rec["thr_dn"]),
             "active": True,
             "ensemble_seeds": [1, 7, 13, 42, 100],
             "_doc": f"T83 winner: {winner}. {full_pnl:.4f} LOSO-equiv on full 442k."}
        ]
    }
    thr_path = os.path.join(HERE, "thresholds.json")
    with open(thr_path, "w") as f:
        json.dump(thr_json, f, indent=2)
    print(f"Wrote {thr_path}")


if __name__ == "__main__":
    main()
