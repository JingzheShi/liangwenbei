"""T6b: Threshold sweep on the 5-seed ensemble OOF for Scheme A.

Inputs (5 ensemble OOF parquets):
    experiments/T6_ensemble_multiseed/loso_pred_ensemble_held{0..4}.parquet
        produced by aggregate_oof.py

Decision rule (matches T5a / iter_001c / iter_001d):
    side_max = max(prob_0, prob_2)
    if side_max >= T and side_max > prob_1 + delta:
        pred = 2 if prob_2 > prob_0 else 0
    else:
        pred = 1  # flat

Sweep grid:
    T in {0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70}
    delta in {0.0, 0.05, 0.10, 0.15, 0.20, 0.25}
    -> 8 x 6 = 48 combos

Reference comparisons (single seed=42 Scheme A LOSO):
    raw argmax baseline   -22.10  (2/5 pos)
    iter_001c (0.50,0.15) +6.45   (5/5)
    iter_001d (0.45,0.05) +11.11  (4/5)

Outputs (in this dir):
    sweep_results.csv         — full grid, sorted by sum_cum_pnl
    raw_argmax_per_fold.csv   — ensemble argmax baseline per-fold
    best_per_fold.csv         — per-fold @ best (T*, delta*)
    iter001c_per_fold.csv     — per-fold @ (0.50, 0.15) on ensemble
    iter001d_per_fold.csv     — per-fold @ (0.45, 0.05) on ensemble
    results.json              — summary (used for decision)
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from src.eval.pnl import _per_horizon_metrics  # noqa: E402

T6_DIR = os.path.join(ROOT, "experiments", "T6_ensemble_multiseed")

T_GRID = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
D_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]

ITER001C = (0.50, 0.15)
ITER001D = (0.45, 0.05)

# T4 single-seed Scheme A reference numbers (from prior reports)
SINGLE_RAW_BASELINE = -22.10
SINGLE_ITER001C = 6.45
SINGLE_ITER001D = 11.11
ITER002_H10 = 21.86  # current best (T5b multi-horizon h_10)

# Decision thresholds on ensemble best:
#   < +12   -> ensemble not significant
#   >= +14  -> build iter_001e
SIGNIFICANCE_LOW = 12.0
SIGNIFICANCE_BUILD = 14.0


def progress(step: str, **extra) -> None:
    p = os.path.join(HERE, "worker-progress.json")
    payload = {
        "status": "running",
        "step": step,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **extra,
    }
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


def thresholded_pred(probs: np.ndarray, T: float, delta: float) -> np.ndarray:
    p0 = probs[:, 0]
    p1 = probs[:, 1]
    p2 = probs[:, 2]
    side_max = np.maximum(p0, p2)
    take = (side_max >= T) & (side_max > p1 + delta)
    pred_side = np.where(p2 > p0, 2, 0)
    return np.where(take, pred_side, 1).astype(np.int8)


def per_fold_metrics(df_fold: pd.DataFrame, T: float, delta: float) -> dict:
    probs = df_fold[["prob_0", "prob_1", "prob_2"]].to_numpy(np.float32)
    pred = thresholded_pred(probs, T, delta)
    m = _per_horizon_metrics(
        pred,
        df_fold["true_label_60"].to_numpy(np.int64),
        df_fold["midprice_t"].to_numpy(np.float32),
        df_fold["midprice_t60"].to_numpy(np.float32),
        fee_rate=0.0001,
    )
    return {
        "cum_pnl": float(m["cum_pnl"]),
        "single_pnl": float(m["single_pnl"]),
        "accuracy": float(m["accuracy"]),
        "n_active": int(m["n_predictions_active"]),
        "f0_5_macro": float(m["f0_5_macro"]),
    }


def per_fold_table(fold_dfs: dict, T: float, delta: float) -> list[dict]:
    rows = []
    for k in range(5):
        m = per_fold_metrics(fold_dfs[k], T, delta)
        rows.append({"sym": k, **m})
    return rows


def main():
    t0 = time.time()
    progress("loading_oof")

    try:
        import wandb
        wandb.init(
            project="liangwenbei",
            entity="cjxh21-Tsinghua University",
            name="T6b-ensemble-sweep-schemeA",
            config={
                "scheme": "A",
                "n_seeds": 5,
                "seeds": [42, 1, 7, 13, 100],
                "T_grid": T_GRID,
                "delta_grid": D_GRID,
            },
            reinit=True,
        )
        WANDB_OK = True
    except Exception as e:
        print(f"[wandb] disabled: {e}", flush=True)
        WANDB_OK = False

    # ---- Load 5 ensemble OOF folds ----
    fold_dfs = {}
    for k in range(5):
        p = os.path.join(T6_DIR, f"loso_pred_ensemble_held{k}.parquet")
        fold_dfs[k] = pd.read_parquet(p)
        print(f"  fold {k}: {len(fold_dfs[k]):,} rows  ({os.path.relpath(p, ROOT)})", flush=True)
    n_total = sum(len(v) for v in fold_dfs.values())
    print(f"total OOF rows: {n_total:,}", flush=True)

    # ---- Baseline: ensemble raw argmax ----
    print("\n=== Ensemble raw argmax per-fold ===", flush=True)
    baseline_pf = []
    for k in range(5):
        df = fold_dfs[k]
        pred = df[["prob_0", "prob_1", "prob_2"]].to_numpy(np.float32).argmax(1).astype(np.int8)
        m = _per_horizon_metrics(
            pred,
            df["true_label_60"].to_numpy(np.int64),
            df["midprice_t"].to_numpy(np.float32),
            df["midprice_t60"].to_numpy(np.float32),
            fee_rate=0.0001,
        )
        baseline_pf.append({
            "sym": k,
            "cum_pnl": float(m["cum_pnl"]),
            "n_active": int(m["n_predictions_active"]),
            "accuracy": float(m["accuracy"]),
        })
        print(f"  sym {k}: cum_pnl={m['cum_pnl']:+.4f} n_active={m['n_predictions_active']:,}", flush=True)
    baseline_sum = sum(r["cum_pnl"] for r in baseline_pf)
    baseline_pos = sum(1 for r in baseline_pf if r["cum_pnl"] > 0)
    print(f"  ENSEMBLE RAW: sum={baseline_sum:+.4f} pos={baseline_pos}/5", flush=True)
    pd.DataFrame(baseline_pf).to_csv(os.path.join(HERE, "raw_argmax_per_fold.csv"), index=False)

    # ---- Sweep grid ----
    progress("sweeping", T_grid=T_GRID, delta_grid=D_GRID)
    print(f"\n=== Sweeping {len(T_GRID)}x{len(D_GRID)} = {len(T_GRID)*len(D_GRID)} combos ===", flush=True)
    rows = []
    for T in T_GRID:
        for d in D_GRID:
            per_fold = []
            for k in range(5):
                m = per_fold_metrics(fold_dfs[k], T, d)
                per_fold.append(m)
            sum_cum = sum(r["cum_pnl"] for r in per_fold)
            mean_acc = float(np.mean([r["accuracy"] for r in per_fold]))
            sum_active = int(sum(r["n_active"] for r in per_fold))
            n_pos = int(sum(1 for r in per_fold if r["cum_pnl"] > 0))
            rows.append({
                "T": T, "delta": d,
                "sum_cum_pnl": sum_cum,
                "mean_cum_pnl": sum_cum / 5.0,
                "n_pos_folds": n_pos,
                "mean_accuracy": mean_acc,
                "sum_n_active": sum_active,
                "fold0_pnl": per_fold[0]["cum_pnl"],
                "fold1_pnl": per_fold[1]["cum_pnl"],
                "fold2_pnl": per_fold[2]["cum_pnl"],
                "fold3_pnl": per_fold[3]["cum_pnl"],
                "fold4_pnl": per_fold[4]["cum_pnl"],
            })
            print(f"  T={T:.2f} d={d:.2f} -> sum={sum_cum:+.4f} pos={n_pos}/5 active={sum_active:,}",
                  flush=True)
            if WANDB_OK:
                wandb.log({
                    "T": T, "delta": d,
                    "sum_cum_pnl": sum_cum,
                    "mean_cum_pnl": sum_cum / 5.0,
                    "n_pos_folds": n_pos,
                    "mean_accuracy": mean_acc,
                    "sum_n_active": sum_active,
                })

    sweep_df = pd.DataFrame(rows).sort_values("sum_cum_pnl", ascending=False).reset_index(drop=True)
    sweep_df.to_csv(os.path.join(HERE, "sweep_results.csv"), index=False)
    print(f"\nfull sweep saved -> sweep_results.csv ({len(sweep_df)} rows)", flush=True)

    best_row = sweep_df.iloc[0].to_dict()
    best_T = float(best_row["T"])
    best_d = float(best_row["delta"])
    best_sum = float(best_row["sum_cum_pnl"])
    best_pos = int(best_row["n_pos_folds"])
    print(f"\nBest (T*, delta*) = ({best_T}, {best_d}): sum_cum_pnl={best_sum:+.4f} pos={best_pos}/5",
          flush=True)

    # iter_001c match
    iter001c_match = sweep_df[
        (sweep_df["T"] == ITER001C[0]) & (sweep_df["delta"] == ITER001C[1])
    ].iloc[0].to_dict()
    iter001c_sum = float(iter001c_match["sum_cum_pnl"])
    print(f"On ensemble: iter_001c (T=0.50, d=0.15) sum={iter001c_sum:+.4f} pos={int(iter001c_match['n_pos_folds'])}/5",
          flush=True)

    # iter_001d match
    iter001d_match = sweep_df[
        (sweep_df["T"] == ITER001D[0]) & (sweep_df["delta"] == ITER001D[1])
    ].iloc[0].to_dict()
    iter001d_sum = float(iter001d_match["sum_cum_pnl"])
    print(f"On ensemble: iter_001d (T=0.45, d=0.05) sum={iter001d_sum:+.4f} pos={int(iter001d_match['n_pos_folds'])}/5",
          flush=True)

    # Per-fold tables
    best_pf = per_fold_table(fold_dfs, best_T, best_d)
    iter001c_pf = per_fold_table(fold_dfs, *ITER001C)
    iter001d_pf = per_fold_table(fold_dfs, *ITER001D)
    pd.DataFrame(best_pf).to_csv(os.path.join(HERE, "best_per_fold.csv"), index=False)
    pd.DataFrame(iter001c_pf).to_csv(os.path.join(HERE, "iter001c_per_fold.csv"), index=False)
    pd.DataFrame(iter001d_pf).to_csv(os.path.join(HERE, "iter001d_per_fold.csv"), index=False)

    # ---- Decision logic ----
    if best_sum >= SIGNIFICANCE_BUILD:
        decision = "build_iter_001e"
        decision_msg = (
            f"ensemble best ({best_T}, {best_d}) sum={best_sum:+.4f} >= "
            f"+{SIGNIFICANCE_BUILD}; build iter_001e (5-seed Scheme A + ({best_T},{best_d}))"
        )
    elif best_sum < SIGNIFICANCE_LOW:
        decision = "no_iter_001e"
        decision_msg = (
            f"ensemble best sum={best_sum:+.4f} < +{SIGNIFICANCE_LOW}; "
            f"ensemble does not significantly improve over single-seed iter_001d (+{SINGLE_ITER001D})"
        )
    else:
        decision = "marginal"
        decision_msg = (
            f"ensemble best sum={best_sum:+.4f} in [+{SIGNIFICANCE_LOW}, +{SIGNIFICANCE_BUILD}); "
            f"marginal vs single-seed iter_001d (+{SINGLE_ITER001D}); skip iter_001e — "
            f"iter_002 multi-horizon (h_10 +{ITER002_H10}) is stronger"
        )

    iter002_warning = (
        f"iter_002 (T5b Scheme C h_10) is currently +{ITER002_H10} which is "
        f"{'still stronger' if ITER002_H10 > best_sum else 'now weaker'} than ensemble best "
        f"({best_sum:+.4f})."
    )

    print(f"\nDECISION: {decision_msg}", flush=True)
    print(f"WARNING : {iter002_warning}", flush=True)

    out = {
        "task": "T6b: 5-seed Scheme A ensemble OOF threshold sweep",
        "scheme": "A",
        "seeds": [42, 1, 7, 13, 100],
        "n_oof_total": int(n_total),
        "T_grid": T_GRID,
        "delta_grid": D_GRID,
        "single_seed_reference": {
            "raw_argmax_sum_cum_pnl": SINGLE_RAW_BASELINE,
            "iter_001c_sum_cum_pnl": SINGLE_ITER001C,
            "iter_001d_sum_cum_pnl": SINGLE_ITER001D,
            "iter_002_h10_sum_cum_pnl": ITER002_H10,
        },
        "ensemble_baseline_argmax": {
            "sum_cum_pnl": float(baseline_sum),
            "n_pos_folds": int(baseline_pos),
            "per_fold": baseline_pf,
        },
        "ensemble_best": {
            "T": best_T,
            "delta": best_d,
            "sum_cum_pnl": best_sum,
            "n_pos_folds": int(best_pos),
            "mean_accuracy": float(best_row["mean_accuracy"]),
            "sum_n_active": int(best_row["sum_n_active"]),
            "per_fold": best_pf,
        },
        "ensemble_iter_001c_threshold": {
            "T": ITER001C[0], "delta": ITER001C[1],
            "sum_cum_pnl": iter001c_sum,
            "n_pos_folds": int(iter001c_match["n_pos_folds"]),
            "per_fold": iter001c_pf,
        },
        "ensemble_iter_001d_threshold": {
            "T": ITER001D[0], "delta": ITER001D[1],
            "sum_cum_pnl": iter001d_sum,
            "n_pos_folds": int(iter001d_match["n_pos_folds"]),
            "per_fold": iter001d_pf,
        },
        "delta_vs_iter001d_single": float(best_sum - SINGLE_ITER001D),
        "delta_vs_iter002_h10": float(best_sum - ITER002_H10),
        "decision": decision,
        "decision_msg": decision_msg,
        "iter002_warning": iter002_warning,
        "elapsed_sec": time.time() - t0,
    }
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nresults.json saved (elapsed {out['elapsed_sec']:.1f}s)", flush=True)

    if WANDB_OK:
        wandb.summary["best_T"] = best_T
        wandb.summary["best_delta"] = best_d
        wandb.summary["best_sum_cum_pnl"] = best_sum
        wandb.summary["ensemble_raw_argmax_sum_cum_pnl"] = baseline_sum
        wandb.summary["iter_001c_on_ensemble_sum"] = iter001c_sum
        wandb.summary["iter_001d_on_ensemble_sum"] = iter001d_sum
        wandb.summary["delta_vs_iter001d_single"] = best_sum - SINGLE_ITER001D
        wandb.summary["delta_vs_iter002_h10"] = best_sum - ITER002_H10
        wandb.summary["decision"] = decision
        wandb.finish()

    progress("done", decision=decision, best_T=best_T, best_delta=best_d, best_sum=best_sum)


if __name__ == "__main__":
    main()
