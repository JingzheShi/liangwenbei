"""T15 Step 1: Isotonic calibration on iter_002 LOSO OOF predictions.

For each horizon H and each held-out fold K:
  - Fit one-vs-rest IsotonicRegression on combined OOF of held{j != K}
  - Apply to held{K}
  - Renormalize p0_cal + p1_cal + p2_cal = 1
  - Save loso_pred_h{H}_held{K}_cal.parquet

For deployment, also fit isotonic on ALL 5 folds combined and save isotonic_h{H}.pkl
(one-vs-rest, list of 3 IsotonicRegression).

Compliance: sym-agnostic (calibrator is global, not per-sym).
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
T5B_DIR = os.path.join(ROOT, "experiments", "T5b_features_multihorizon")
HORIZONS = (5, 10, 20, 40, 60)


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


def fit_isotonic_ovr(probs: np.ndarray, true_label: np.ndarray) -> list[IsotonicRegression]:
    """One-vs-rest isotonic. probs: (N, 3); true_label: (N,) ∈ {0,1,2}."""
    isos = []
    for c in range(3):
        y_c = (true_label == c).astype(np.float32)
        x_c = probs[:, c].astype(np.float64)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(x_c, y_c)
        isos.append(iso)
    return isos


def apply_isotonic_ovr(probs: np.ndarray, isos: list[IsotonicRegression]) -> np.ndarray:
    """Apply 3-class isotonic + renormalize."""
    cal = np.empty_like(probs, dtype=np.float64)
    for c in range(3):
        cal[:, c] = isos[c].predict(probs[:, c].astype(np.float64))
    s = cal.sum(axis=1, keepdims=True)
    s = np.where(s > 1e-12, s, 1.0)
    return (cal / s).astype(np.float32)


def load_horizon_folds(H: int) -> list[pd.DataFrame]:
    dfs = []
    for k in range(5):
        p = os.path.join(T5B_DIR, f"loso_pred_h{H}_held{k}.parquet")
        dfs.append(pd.read_parquet(p))
    return dfs


def main():
    t0 = time.time()
    progress("start")

    summary = {"horizons": {}, "n_oof_total": {}}

    for H in HORIZONS:
        progress(f"calibrate_h{H}")
        print(f"\n=== Horizon h={H} ===", flush=True)
        fold_dfs = load_horizon_folds(H)
        n_total = sum(len(d) for d in fold_dfs)
        summary["n_oof_total"][f"h_{H}"] = n_total
        print(f"  total OOF rows: {n_total:,}", flush=True)

        # 1. Per-fold calibration: fit on others, apply to held
        for k in range(5):
            train_dfs = [fold_dfs[j] for j in range(5) if j != k]
            train_df = pd.concat(train_dfs, ignore_index=True)
            probs_train = train_df[["prob_0", "prob_1", "prob_2"]].to_numpy(np.float32)
            y_train = train_df["true_label"].to_numpy(np.int64)
            isos = fit_isotonic_ovr(probs_train, y_train)

            held_df = fold_dfs[k].copy()
            probs_held = held_df[["prob_0", "prob_1", "prob_2"]].to_numpy(np.float32)
            cal_held = apply_isotonic_ovr(probs_held, isos)

            held_df["prob_0_cal"] = cal_held[:, 0]
            held_df["prob_1_cal"] = cal_held[:, 1]
            held_df["prob_2_cal"] = cal_held[:, 2]
            out_p = os.path.join(HERE, f"loso_pred_h{H}_held{k}_cal.parquet")
            held_df.to_parquet(out_p)
            mean_orig_max = float(probs_held.max(axis=1).mean())
            mean_cal_max = float(cal_held.max(axis=1).mean())
            print(f"  held{k}: rows={len(held_df):,} mean_max_p orig={mean_orig_max:.4f} cal={mean_cal_max:.4f}", flush=True)

        # 2. Deployment: fit on ALL 5 folds combined
        all_df = pd.concat(fold_dfs, ignore_index=True)
        probs_all = all_df[["prob_0", "prob_1", "prob_2"]].to_numpy(np.float32)
        y_all = all_df["true_label"].to_numpy(np.int64)
        isos_deploy = fit_isotonic_ovr(probs_all, y_all)
        out_pkl = os.path.join(HERE, f"isotonic_h{H}.pkl")
        with open(out_pkl, "wb") as f:
            pickle.dump(isos_deploy, f)
        print(f"  deploy isotonic saved -> isotonic_h{H}.pkl (fit on N={len(all_df):,})", flush=True)

        # Summary stats
        cal_all = apply_isotonic_ovr(probs_all, isos_deploy)
        bce_orig = -np.mean(np.log(np.clip(probs_all[np.arange(len(y_all)), y_all], 1e-7, 1.0)))
        bce_cal = -np.mean(np.log(np.clip(cal_all[np.arange(len(y_all)), y_all], 1e-7, 1.0)))
        summary["horizons"][f"h_{H}"] = {
            "n_oof": n_total,
            "bce_orig": float(bce_orig),
            "bce_cal": float(bce_cal),
            "mean_max_p_orig": float(probs_all.max(axis=1).mean()),
            "mean_max_p_cal": float(cal_all.max(axis=1).mean()),
        }
        print(f"  BCE orig={bce_orig:.4f} cal={bce_cal:.4f}", flush=True)

    summary["elapsed_sec"] = time.time() - t0
    with open(os.path.join(HERE, "calibration_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nelapsed {summary['elapsed_sec']:.1f}s", flush=True)
    progress("calibrate_done")


if __name__ == "__main__":
    main()
