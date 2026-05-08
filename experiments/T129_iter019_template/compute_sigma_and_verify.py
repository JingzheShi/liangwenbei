"""Compute per-sym sigma_pred from iter_015 v1 stack on val 442k,
then verify the per-sym β abstain wrapper reproduces ~+41.26 LOSO-equiv.

Uses the same pred files as R_conformal_select to ensure byte-identical pipeline.
"""
from __future__ import annotations
import json
import os
import sys
import numpy as np
import pandas as pd

T87_DIR = "/root/projects/liangwenbei_workdir/experiments/T87_spo_dfl"
T75_DIR = "/root/projects/liangwenbei_workdir/experiments/T75_regression_dmid"
OUT_DIR = "/root/projects/liangwenbei_workdir/experiments/T127_iter018_v1_pkg"
SEEDS = (1, 7, 13, 42, 100)
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001

# Recommended per-sym β (4-fold CV consensus / V8 in-sample fit)
PER_SYM_BETA = {0: 0.10, 1: 0.40, 2: 0.30, 3: 0.00, 4: 0.00}
DEFAULT_BETA_FOR_OOD = float(np.mean(list(PER_SYM_BETA.values())))  # 0.16

# DE thresholds from R_conformal_select (in-sample DE on combined stack)
THR_UP = 0.00029995433796130955
THR_DN = 0.0002158528296175958


def vectorized_pnl(action, mp_t, mp_th):
    side = action.astype(np.float64) - 1.0
    abs_side = np.abs(side)
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * abs_side * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee_pnl) / denom


def load_avg(paths):
    base = pd.read_parquet(paths[0])
    n = len(base)
    p = np.zeros(n, dtype=np.float64)
    for path in paths:
        df = pd.read_parquet(path)
        p += df["pred_dmid_norm"].to_numpy(np.float64)
    p /= len(paths)
    base = base.copy()
    base["pred_dmid_norm"] = p.astype(np.float32)
    return base


def main():
    print("Loading T87 + T75 5-seed predictions ...", flush=True)
    t87_paths = [os.path.join(T87_DIR, f"pred_T87_seed{s}_main.parquet") for s in SEEDS]
    df_t87 = load_avg(t87_paths)
    t75_paths = [os.path.join(T75_DIR, f"pred_T75_seed{s}.parquet") for s in SEEDS]
    df_t75 = load_avg(t75_paths)

    # Combined stack: w_nn=1.0, w_lgb=1.5
    p_combined = (1.0 * df_t87["pred_dmid_norm"].to_numpy(np.float64)
                  + 1.5 * df_t75["pred_dmid_norm"].to_numpy(np.float64)) / 2.5

    df = df_t87.copy()
    df["pred_dmid_norm"] = p_combined.astype(np.float32)
    print(f"Total rows: {len(df)}", flush=True)

    # Compute per-sym sigma
    sigma_per_sym = {}
    for k in SYMS:
        sub = df[df["sym"] == k]
        sigma_per_sym[int(k)] = float(np.std(sub["pred_dmid_norm"].to_numpy(np.float64)))
    print("\nPer-sym sigma:", flush=True)
    for k, s in sigma_per_sym.items():
        print(f"  sym {k}: sigma={s:.6e}  band@β={PER_SYM_BETA[k]}: {PER_SYM_BETA[k]*s:+.6e}", flush=True)

    # Verify: apply the per-sym β-band wrapper, compute LOSO-equiv PnL
    per_sym_pnl = []
    per_sym_active = []
    per_sym_win = []
    per_sym_pnl_baseline = []
    for k in SYMS:
        sub = df[df["sym"] == k].reset_index(drop=True)
        pred = sub["pred_dmid_norm"].to_numpy(np.float64)
        mp_t = sub["midprice_t"].to_numpy(np.float64)
        mp_th = sub["midprice_th"].to_numpy(np.float64)

        sigma = sigma_per_sym[k]
        beta = PER_SYM_BETA[k]
        thr_up_eff = THR_UP + beta * sigma
        thr_dn_eff = THR_DN + beta * sigma

        # Wrapper action
        action = np.full(len(pred), 1, dtype=np.int8)
        action[pred > thr_up_eff] = 2
        action[pred < -thr_dn_eff] = 0
        pnl = vectorized_pnl(action, mp_t, mp_th)
        per_sym_pnl.append(float(pnl.sum()))
        active_mask = (action != 1)
        per_sym_active.append(float(active_mask.mean()))
        if active_mask.sum() > 0:
            per_sym_win.append(float((pnl[active_mask] > 0).mean()))
        else:
            per_sym_win.append(0.0)

        # Baseline (no abstain) action
        action_b = np.full(len(pred), 1, dtype=np.int8)
        action_b[pred > THR_UP] = 2
        action_b[pred < -THR_DN] = 0
        pnl_b = vectorized_pnl(action_b, mp_t, mp_th)
        per_sym_pnl_baseline.append(float(pnl_b.sum()))

    total = sum(per_sym_pnl)
    total_base = sum(per_sym_pnl_baseline)
    print(f"\n=== VERIFY: per-sym β-band wrapper ===", flush=True)
    print(f"  baseline (no abstain): total={total_base:+.4f}", flush=True)
    print(f"  per-sym β-band wrapper: total={total:+.4f}", flush=True)
    print(f"  delta: {total - total_base:+.4f}", flush=True)
    print(f"  per_sym_pnl: {[f'{v:+.4f}' for v in per_sym_pnl]}", flush=True)
    print(f"  per_sym_active: {[f'{v:.3f}' for v in per_sym_active]}", flush=True)
    print(f"  per_sym_win: {[f'{v:.3f}' for v in per_sym_win]}", flush=True)
    print(f"  min_per_sym: {min(per_sym_pnl):+.4f}", flush=True)

    # Save
    out = {
        "thr_up": THR_UP,
        "thr_dn": THR_DN,
        "per_sym_beta": PER_SYM_BETA,
        "default_beta_for_ood": DEFAULT_BETA_FOR_OOD,
        "per_sym_sigma": sigma_per_sym,
        "verify": {
            "total_loso_equiv": total,
            "total_baseline_no_abstain": total_base,
            "delta_vs_baseline": total - total_base,
            "per_sym_pnl": per_sym_pnl,
            "per_sym_pnl_baseline": per_sym_pnl_baseline,
            "per_sym_active": per_sym_active,
            "per_sym_win": per_sym_win,
            "min_per_sym": float(min(per_sym_pnl)),
            "active_rate": float(np.mean(per_sym_active)),
            "win_rate": float(np.mean(per_sym_win)),
        },
    }
    out_path = os.path.join(OUT_DIR, "verify_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
