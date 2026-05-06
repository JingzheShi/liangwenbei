"""T5a: Re-tune (T, delta) on Scheme B's own LOSO OOF predictions.

iter_001c uses (T=0.50, delta=0.15) — taken from Scheme A's LOSO sweep.
Scheme B's raw argmax 5-fold sum is +1.33 (better than A's -22.10), so the
optimal threshold may be different. We sweep T x delta and compare against
iter_001c's transferred choice.

Decision rule:
    side_max = max(prob_0, prob_2)
    if side_max >= T and side_max > prob_1 + delta:
        pred = 2 if prob_2 > prob_0 else 0
    else:
        pred = 1  # flat

Inputs:
    experiments/T4_loso_validate/loso_pred_schemeB_held{0..4}.parquet

Outputs:
    sweep_results.csv         — full grid (T, delta, sum_cum_pnl, ...)
    best_per_fold.csv         — per-fold breakdown for best (T*, delta*)
    iter001c_per_fold.csv     — per-fold breakdown for iter_001c (T=0.5, delta=0.15)
    results.json              — summary
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

T_GRID = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
D_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
ITER001C = (0.50, 0.15)


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
    """probs: (N, 3). Returns int8 in {0,1,2}."""
    p0 = probs[:, 0]
    p1 = probs[:, 1]
    p2 = probs[:, 2]
    side_max = np.maximum(p0, p2)
    take = (side_max >= T) & (side_max > p1 + delta)
    pred_side = np.where(p2 > p0, 2, 0)
    return np.where(take, pred_side, 1).astype(np.int8)


def per_fold_metrics(
    df_fold: pd.DataFrame, T: float, delta: float
) -> dict:
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


def main():
    t0 = time.time()
    progress("loading_oof")

    # Try wandb (best-effort)
    try:
        import wandb
        wandb.init(
            project="liangwenbei",
            entity="cjxh21-Tsinghua University",
            name="T5a-rethresh-schemeB",
            config={"scheme": "B", "T_grid": T_GRID, "delta_grid": D_GRID},
            reinit=True,
        )
        WANDB_OK = True
    except Exception as e:
        print(f"[wandb] disabled: {e}", flush=True)
        WANDB_OK = False

    # ---- Load Scheme B 5-fold LOSO predictions ----
    fold_dfs = {}
    for k in range(5):
        p = os.path.join(
            ROOT, "experiments", "T4_loso_validate",
            f"loso_pred_schemeB_held{k}.parquet"
        )
        fold_dfs[k] = pd.read_parquet(p)
        print(f"  fold {k}: {len(fold_dfs[k]):,} rows", flush=True)

    n_total = sum(len(v) for v in fold_dfs.values())
    print(f"total OOF rows: {n_total:,}", flush=True)

    # ---- Baseline: raw argmax ----
    print("\n=== Baseline (raw argmax) per-fold ===", flush=True)
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
        baseline_pf.append({"sym": k, "cum_pnl": float(m["cum_pnl"]),
                            "n_active": int(m["n_predictions_active"]),
                            "accuracy": float(m["accuracy"])})
        print(f"  sym {k}: cum_pnl={m['cum_pnl']:+.4f} n_active={m['n_predictions_active']:,}", flush=True)
    baseline_sum = sum(r["cum_pnl"] for r in baseline_pf)
    print(f"  baseline 5-fold sum cum_pnl = {baseline_sum:+.4f}", flush=True)

    # ---- Sweep grid ----
    progress("sweeping", T_grid=T_GRID, delta_grid=D_GRID)
    print(f"\n=== Sweeping {len(T_GRID)} x {len(D_GRID)} = {len(T_GRID)*len(D_GRID)} combos ===", flush=True)
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
                "T": T,
                "delta": d,
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

    sweep_df = pd.DataFrame(rows).sort_values("sum_cum_pnl", ascending=False)
    sweep_df.to_csv(os.path.join(HERE, "sweep_results.csv"), index=False)
    print(f"\nfull sweep saved -> sweep_results.csv ({len(sweep_df)} rows)", flush=True)

    # ---- Best (T*, delta*) ----
    best_row = sweep_df.iloc[0].to_dict()
    best_T = float(best_row["T"])
    best_d = float(best_row["delta"])
    best_sum = float(best_row["sum_cum_pnl"])
    print(f"\nBest (T*, delta*) = ({best_T}, {best_d}): sum_cum_pnl={best_sum:+.4f}", flush=True)

    # ---- iter_001c reference (T=0.5, delta=0.15) ----
    iter001c_match = sweep_df[
        (sweep_df["T"] == ITER001C[0]) & (sweep_df["delta"] == ITER001C[1])
    ].iloc[0].to_dict()
    iter001c_sum = float(iter001c_match["sum_cum_pnl"])
    print(f"iter_001c (T=0.50, delta=0.15): sum_cum_pnl={iter001c_sum:+.4f}", flush=True)

    delta_vs_iter001c = best_sum - iter001c_sum
    print(f"\nbest - iter001c = {delta_vs_iter001c:+.4f}", flush=True)

    # ---- Per-fold breakdown for best & iter_001c ----
    print(f"\n=== Per-fold @ best (T={best_T}, delta={best_d}) ===", flush=True)
    best_pf_rows = []
    for k in range(5):
        m = per_fold_metrics(fold_dfs[k], best_T, best_d)
        best_pf_rows.append({"sym": k, **m})
        print(f"  sym {k}: cum_pnl={m['cum_pnl']:+.4f} n_active={m['n_active']:,} acc={m['accuracy']:.4f}",
              flush=True)
    pd.DataFrame(best_pf_rows).to_csv(os.path.join(HERE, "best_per_fold.csv"), index=False)

    print(f"\n=== Per-fold @ iter_001c (T={ITER001C[0]}, delta={ITER001C[1]}) ===", flush=True)
    iter001c_pf_rows = []
    for k in range(5):
        m = per_fold_metrics(fold_dfs[k], ITER001C[0], ITER001C[1])
        iter001c_pf_rows.append({"sym": k, **m})
        print(f"  sym {k}: cum_pnl={m['cum_pnl']:+.4f} n_active={m['n_active']:,} acc={m['accuracy']:.4f}",
              flush=True)
    pd.DataFrame(iter001c_pf_rows).to_csv(os.path.join(HERE, "iter001c_per_fold.csv"), index=False)

    # ---- Decision: build iter_001d? ----
    SIGNIFICANCE = 0.5  # require improvement >= 0.5 to switch
    if delta_vs_iter001c >= SIGNIFICANCE:
        decision = "build_iter001d"
        decision_msg = (f"best ({best_T}, {best_d}) beats iter_001c by {delta_vs_iter001c:+.4f} "
                        f">= {SIGNIFICANCE}; build iter_001d")
    else:
        decision = "keep_iter001c"
        decision_msg = (f"best ({best_T}, {best_d}) only beats iter_001c by {delta_vs_iter001c:+.4f} "
                        f"< {SIGNIFICANCE}; keep iter_001c, no iter_001d")
    print(f"\nDECISION: {decision_msg}", flush=True)

    out = {
        "task": "T5a re-tune threshold on Scheme B LOSO OOF",
        "scheme": "B",
        "n_oof_total": int(n_total),
        "T_grid": T_GRID,
        "delta_grid": D_GRID,
        "baseline": {
            "raw_argmax_sum_cum_pnl": float(baseline_sum),
            "per_fold": baseline_pf,
        },
        "best": {
            "T": best_T,
            "delta": best_d,
            "sum_cum_pnl": best_sum,
            "n_pos_folds": int(best_row["n_pos_folds"]),
            "mean_accuracy": float(best_row["mean_accuracy"]),
            "sum_n_active": int(best_row["sum_n_active"]),
            "per_fold": best_pf_rows,
        },
        "iter_001c_threshold": {
            "T": ITER001C[0],
            "delta": ITER001C[1],
            "sum_cum_pnl": iter001c_sum,
            "n_pos_folds": int(iter001c_match["n_pos_folds"]),
            "mean_accuracy": float(iter001c_match["mean_accuracy"]),
            "sum_n_active": int(iter001c_match["sum_n_active"]),
            "per_fold": iter001c_pf_rows,
        },
        "delta_vs_iter001c": float(delta_vs_iter001c),
        "significance_threshold": SIGNIFICANCE,
        "decision": decision,
        "decision_msg": decision_msg,
        "elapsed_sec": time.time() - t0,
    }
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nresults.json saved (elapsed {out['elapsed_sec']:.1f}s)", flush=True)

    if WANDB_OK:
        wandb.summary["best_T"] = best_T
        wandb.summary["best_delta"] = best_d
        wandb.summary["best_sum_cum_pnl"] = best_sum
        wandb.summary["iter001c_sum_cum_pnl"] = iter001c_sum
        wandb.summary["delta_vs_iter001c"] = delta_vs_iter001c
        wandb.summary["decision"] = decision
        wandb.summary["baseline_argmax_sum_cum_pnl"] = baseline_sum
        wandb.finish()

    progress("done", decision=decision, best_T=best_T, best_delta=best_d, best_sum=best_sum)


if __name__ == "__main__":
    main()
