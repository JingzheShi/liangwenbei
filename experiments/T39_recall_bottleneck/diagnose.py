"""T39 — Recall bottleneck diagnosis on iter_006 (5-seed aug_a OOF, h_60).

Steps:
  1) Load 5-seed-averaged probs per LOSO fold (via T30 _common).
  2) Oracle PnL upper bound per fold; gap to iter_006.
  3) Decompose iter_006 error into A=missed_profitable, B=wrong_active, C=correct_active.
  4) Symmetric-threshold sweep T in [0.30, 0.65] step 0.01, delta=0:
        precision_macro, recall_macro, F0.5_macro, cum_pnl, n_active per fold.
     Locate PnL-max vs F0.5-max vs iter_006 thresholds.
  5) Marginal precision/recall analysis around best-PnL T*.
  6) F0.5 macro at iter_006 asymmetric thresholds (vs iter_002 platform 0.209).
  7) Save results.json + PR-PnL figures per fold + REPORT.md.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments", "T30_threshold_optim"))
sys.path.insert(0, ROOT)
from _common import load_all_folds, gate_asymmetric, vectorized_pnl, FEE  # noqa: E402
from src.eval.pnl import _per_horizon_metrics  # noqa: E402

ITER006 = dict(
    T_up=0.4480917972708376,
    T_dn=0.39144661429951816,
    d_up=0.2597244012809361,
    d_dn=0.024449627822859837,
)

OUT_DIR = HERE


def gate_symmetric(probs: np.ndarray, T: float, delta: float = 0.0) -> np.ndarray:
    p0 = probs[:, 0]
    p1 = probs[:, 1]
    p2 = probs[:, 2]
    side_max = np.maximum(p0, p2)
    take = (side_max >= T) & (side_max > p1 + delta)
    pred_side = np.where(p2 > p0, 2, 0)
    return np.where(take, pred_side, 1).astype(np.int8)


def per_fold_metrics(pred: np.ndarray, df: pd.DataFrame) -> dict:
    m = _per_horizon_metrics(
        pred,
        df["true_label"].to_numpy(np.int64),
        df["midprice_t"].to_numpy(np.float32),
        df["midprice_th"].to_numpy(np.float32),
        fee_rate=FEE,
    )
    return {
        "cum_pnl": float(m["cum_pnl"]),
        "n_active": int(m["n_predictions_active"]),
        "accuracy": float(m["accuracy"]),
        "precision_up": float(m["precision_up"]) if not np.isnan(m["precision_up"]) else None,
        "precision_down": float(m["precision_down"]) if not np.isnan(m["precision_down"]) else None,
        "precision_macro": float(m["precision_macro"]) if not np.isnan(m["precision_macro"]) else None,
        "recall_up": float(m["recall_up"]) if not np.isnan(m["recall_up"]) else None,
        "recall_down": float(m["recall_down"]) if not np.isnan(m["recall_down"]) else None,
        "recall_macro": float(m["recall_macro"]) if not np.isnan(m["recall_macro"]) else None,
        "f0_5_macro": float(m["f0_5_macro"]) if not np.isnan(m["f0_5_macro"]) else None,
        "pred_dist": m["pred_distribution"],
    }


def main() -> None:
    print(">>> Loading 5-seed avg OOF probs across 5 folds ...")
    folds = load_all_folds()

    # ------- Step 2: Oracle upper bound + iter_006 baseline ------------
    iter006_per_fold = {}
    oracle_per_fold = {}
    decomposition_per_fold = {}
    sweep_per_fold = {}
    marginal_per_fold = {}

    for k, df in folds.items():
        probs = df[["prob_0", "prob_1", "prob_2"]].to_numpy(np.float32)
        label = df["true_label"].to_numpy(np.int64)
        mp_t = df["midprice_t"].to_numpy(np.float32)
        mp_th = df["midprice_th"].to_numpy(np.float32)

        # Oracle: pred = true_label
        oracle_pred = label.astype(np.int8)
        oracle = per_fold_metrics(oracle_pred, df)
        oracle_per_fold[k] = oracle

        # iter_006
        iter006_pred = gate_asymmetric(probs, **ITER006)
        iter006 = per_fold_metrics(iter006_pred, df)
        iter006_per_fold[k] = iter006

        # ------- Step 3: error decomposition -------
        # vector pnl per row
        per_row_oracle_pnl = vectorized_pnl(oracle_pred, label, mp_t, mp_th)
        per_row_iter006_pnl = vectorized_pnl(iter006_pred, label, mp_t, mp_th)
        per_row_idle_pnl = np.zeros_like(per_row_oracle_pnl)  # pred=1 → side=0

        # A = missed (true_label != 1, pred = 1) → loss = oracle_pnl on these rows
        mask_A = (iter006_pred == 1) & (label != 1)
        A_pnl = float(per_row_oracle_pnl[mask_A].sum())
        # B = wrong active (pred != 1 and pred != true_label) → realized iter006 PnL on these
        mask_B = (iter006_pred != 1) & (iter006_pred != label)
        B_pnl = float(per_row_iter006_pnl[mask_B].sum())
        # C = correct active (pred != 1 and pred == true_label)
        mask_C = (iter006_pred != 1) & (iter006_pred == label)
        C_pnl = float(per_row_iter006_pnl[mask_C].sum())

        # also: idle when label=1 (we correctly skipped, no PnL) — neutral
        mask_idle_correct = (iter006_pred == 1) & (label == 1)

        # We define the "gap to oracle" as oracle_total - iter006_total.
        decomposition_per_fold[k] = {
            "A_missed_profitable": A_pnl,
            "B_wrong_active": B_pnl,
            "C_correct_active": C_pnl,
            "n_A": int(mask_A.sum()),
            "n_B": int(mask_B.sum()),
            "n_C": int(mask_C.sum()),
            "n_idle_correct": int(mask_idle_correct.sum()),
        }

        # ------- Step 4: T sweep, delta=0 ------------------------------
        T_grid = np.arange(0.30, 0.65 + 1e-9, 0.01)
        rows = []
        for T in T_grid:
            pred = gate_symmetric(probs, float(T), 0.0)
            mfold = per_fold_metrics(pred, df)
            rows.append({
                "T": round(float(T), 2),
                "cum_pnl": mfold["cum_pnl"],
                "n_active": mfold["n_active"],
                "precision_macro": mfold["precision_macro"],
                "recall_macro": mfold["recall_macro"],
                "f0_5_macro": mfold["f0_5_macro"],
                "accuracy": mfold["accuracy"],
            })
        sweep_df = pd.DataFrame(rows)
        sweep_per_fold[k] = sweep_df.to_dict(orient="list")

        # locate optima
        idx_pnl_max = int(sweep_df["cum_pnl"].idxmax())
        idx_f05_max = int(sweep_df["f0_5_macro"].fillna(-1).idxmax())
        T_pnl_max = float(sweep_df.loc[idx_pnl_max, "T"])
        T_f05_max = float(sweep_df.loc[idx_f05_max, "T"])

        # ------- Step 5: marginal precision around T* -----------------
        T_star = T_pnl_max
        T_band = np.round(np.arange(T_star - 0.05, T_star + 0.05 + 1e-9, 0.01), 2)
        # build "active set" at each T
        # marginal between T_high → T_low is the new trades activated
        # we'll compute the PnL/precision of newly-active trades as T decreases
        active_masks = {}
        for T in T_band:
            pred = gate_symmetric(probs, float(T), 0.0)
            active_masks[float(T)] = pred != 1

        T_band_sorted = sorted(T_band, reverse=True)  # high T → low T (becomes more permissive)
        marg_rows = []
        prev_active = None
        for T in T_band_sorted:
            mask = active_masks[float(T)]
            if prev_active is None:
                marg_rows.append({"T": float(T), "delta_n_active": int(mask.sum()),
                                  "marg_precision": None, "marg_pnl": None})
            else:
                new_mask = mask & (~prev_active)
                # of new actives, side is whichever side had prob > the other
                pred_at_T = gate_symmetric(probs, float(T), 0.0)
                new_pred = pred_at_T[new_mask]
                new_label = label[new_mask]
                # precision = correct sided / total
                n_new = int(new_mask.sum())
                if n_new > 0:
                    correct = (new_pred == new_label).sum()
                    pnl_new = float(vectorized_pnl(new_pred, new_label, mp_t[new_mask], mp_th[new_mask]).sum())
                    marg_rows.append({"T": float(T), "delta_n_active": n_new,
                                      "marg_precision": float(correct) / float(n_new),
                                      "marg_pnl": pnl_new})
                else:
                    marg_rows.append({"T": float(T), "delta_n_active": 0,
                                      "marg_precision": None, "marg_pnl": 0.0})
            prev_active = mask
        marginal_per_fold[k] = {
            "T_star_pnl_max": T_star,
            "T_f05_max": T_f05_max,
            "band": marg_rows,
        }

        # ------- Plot PR-PnL curve --------------------------------------
        fig, ax_pnl = plt.subplots(figsize=(8, 5))
        ax_pnl.plot(sweep_df["T"], sweep_df["cum_pnl"], "b-o", ms=4, label="cum_pnl")
        ax_pnl.set_xlabel("Threshold T (delta=0)")
        ax_pnl.set_ylabel("cum_pnl", color="b")
        ax_pnl.axhline(0, color="gray", lw=0.5)
        ax_pnl.tick_params(axis="y", labelcolor="b")
        # mark iter006 cum_pnl
        ax_pnl.axhline(iter006["cum_pnl"], color="r", ls="--", lw=1,
                       label=f"iter_006 PnL={iter006['cum_pnl']:.2f}")
        ax_pnl.axhline(oracle["cum_pnl"], color="g", ls="--", lw=1,
                       label=f"oracle PnL={oracle['cum_pnl']:.2f}")
        # mark T_pnl_max and T_f05_max
        ax_pnl.axvline(T_pnl_max, color="purple", ls=":",
                       label=f"T*_PnL={T_pnl_max:.2f}")
        ax_pnl.axvline(T_f05_max, color="orange", ls=":",
                       label=f"T*_F05={T_f05_max:.2f}")
        ax2 = ax_pnl.twinx()
        ax2.plot(sweep_df["T"], sweep_df["f0_5_macro"], "k-^", ms=3, alpha=0.6, label="F0.5_macro")
        ax2.plot(sweep_df["T"], sweep_df["precision_macro"], "m-s", ms=3, alpha=0.5, label="precision")
        ax2.plot(sweep_df["T"], sweep_df["recall_macro"], "c-x", ms=3, alpha=0.5, label="recall")
        ax2.set_ylabel("F0.5 / precision / recall (macro)")
        ax2.tick_params(axis="y")
        ax_pnl.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
        plt.title(f"Fold {k}: T sweep (delta=0). iter_006 actual={iter006['cum_pnl']:.2f}, oracle={oracle['cum_pnl']:.2f}")
        plt.tight_layout()
        out_png = os.path.join(OUT_DIR, f"pr_pnl_curve_fold{k}.png")
        plt.savefig(out_png, dpi=110)
        plt.close()
        print(f"  fold {k}: oracle={oracle['cum_pnl']:+.2f} iter006={iter006['cum_pnl']:+.2f} A={A_pnl:+.2f} B={B_pnl:+.2f} C={C_pnl:+.2f}  T*_PnL={T_pnl_max:.2f}  T*_F05={T_f05_max:.2f}")

    # ----- aggregates -----
    oracle_total = sum(oracle_per_fold[k]["cum_pnl"] for k in folds)
    iter006_total = sum(iter006_per_fold[k]["cum_pnl"] for k in folds)
    A_total = sum(decomposition_per_fold[k]["A_missed_profitable"] for k in folds)
    B_total = sum(decomposition_per_fold[k]["B_wrong_active"] for k in folds)
    C_total = sum(decomposition_per_fold[k]["C_correct_active"] for k in folds)
    gap_total = oracle_total - iter006_total

    iter006_f05_per_fold = [iter006_per_fold[k]["f0_5_macro"] for k in folds]
    iter006_recall_per_fold = [iter006_per_fold[k]["recall_macro"] for k in folds]
    iter006_prec_per_fold = [iter006_per_fold[k]["precision_macro"] for k in folds]

    # T*_PnL aggregated by sum across folds with same T (not the per-fold optimum, the joint)
    T_grid = np.arange(0.30, 0.65 + 1e-9, 0.01)
    sum_pnl_per_T = []
    for T in T_grid:
        s = 0.0
        for k, df in folds.items():
            probs = df[["prob_0", "prob_1", "prob_2"]].to_numpy(np.float32)
            pred = gate_symmetric(probs, float(T), 0.0)
            s += per_fold_metrics(pred, df)["cum_pnl"]
        sum_pnl_per_T.append(s)
    sum_pnl_per_T = np.array(sum_pnl_per_T)
    idx_join = int(sum_pnl_per_T.argmax())
    T_join_optimal = float(T_grid[idx_join])
    sum_pnl_join_optimal = float(sum_pnl_per_T[idx_join])

    results = {
        "task": "T39_recall_bottleneck — recall vs precision diagnosis on iter_006",
        "iter006_thresholds": ITER006,
        "per_fold": {
            str(k): {
                "oracle": oracle_per_fold[k],
                "iter006": iter006_per_fold[k],
                "decomposition": decomposition_per_fold[k],
                "marginal_band": marginal_per_fold[k],
            }
            for k in folds
        },
        "totals": {
            "oracle_total_pnl": float(oracle_total),
            "iter006_total_pnl": float(iter006_total),
            "gap_to_oracle": float(gap_total),
            "A_missed_profitable_total": float(A_total),
            "B_wrong_active_total": float(B_total),
            "C_correct_active_total": float(C_total),
            "A_share_of_gap": float(A_total / gap_total) if gap_total > 0 else None,
            "B_share_of_gap": float(-B_total / gap_total) if gap_total > 0 else None,
            "iter006_f0_5_macro_per_fold": iter006_f05_per_fold,
            "iter006_recall_macro_per_fold": iter006_recall_per_fold,
            "iter006_precision_macro_per_fold": iter006_prec_per_fold,
            "iter006_f0_5_macro_mean": float(np.nanmean([f for f in iter006_f05_per_fold if f is not None])),
            "iter002_platform_f0_5_macro": 0.209,
            "joint_T_optimal_delta0": T_join_optimal,
            "joint_T_sum_pnl_at_optimal": sum_pnl_join_optimal,
            "joint_T_iter006_sum_pnl": float(iter006_total),
            "join_grid_pnl_per_T": [(float(T), float(p)) for T, p in zip(T_grid, sum_pnl_per_T)],
        },
        "sweep_per_fold": sweep_per_fold,
    }

    out_json = os.path.join(OUT_DIR, "results.json")
    with open(out_json, "w") as fp:
        json.dump(results, fp, indent=2, default=str)
    print(f"\nWrote {out_json}")

    # ------ console summary ------------
    print("\n=== TOTALS ===")
    print(f"Oracle total PnL = {oracle_total:+.2f}")
    print(f"iter_006 total   = {iter006_total:+.2f}")
    print(f"Gap to oracle    = {gap_total:+.2f}")
    print(f"  A (missed profitable, true!=1, pred=1)  = {A_total:+.2f}  ({A_total/gap_total*100:.1f}% of gap)")
    print(f"  B (wrong active, pred!=1 & pred!=true)  = {B_total:+.2f}  (lost {-B_total/gap_total*100:.1f}% of gap)")
    print(f"  C (correct active, pred==true!=1)       = {C_total:+.2f}  (already realized)")
    print(f"\n  joint-T best (delta=0): T={T_join_optimal:.2f} → sum_pnl={sum_pnl_join_optimal:+.2f}")
    print(f"  iter_006 (asym) sum_pnl = {iter006_total:+.2f}")
    print(f"\n  iter_006 mean F0.5_macro = {results['totals']['iter006_f0_5_macro_mean']:.3f}  (iter_002 platform = 0.209)")


if __name__ == "__main__":
    main()
