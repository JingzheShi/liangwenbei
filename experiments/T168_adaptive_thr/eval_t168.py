#!/usr/bin/env python3
"""T168 holdout evaluation: per-batch adaptive threshold V1-V4.

Uses pre-computed T87 (NN) + T75 (LGB) prediction parquets (dates 96-119).
v2 baseline: w_nn=1.0, w_lgb=1.5, conformal band (per-sym beta*sigma).
Simulates platform-style batch_size=1024 with random shuffle (seed=42).
"""
from __future__ import annotations
import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

T87_DIR = os.path.join(ROOT, "experiments", "T87_spo_dfl")
T75_DIR = os.path.join(ROOT, "experiments", "T75_regression_dmid")

SEEDS = [1, 7, 13, 42, 100]
SYMS = [0, 1, 2, 3, 4]
FEE = 0.0001
BATCH_SIZE = 1024

# v2 thresholds
THR_UP = 0.00029995433796130955
THR_DN = 0.0002158528296175958
W_NN = 1.0
W_LGB = 1.5

# Per-sym conformal band (beta * sigma)
PER_SYM_BETA = {0: 0.10, 1: 0.40, 2: 0.30, 3: 0.0, 4: 0.0}
PER_SYM_SIGMA = {
    0: 0.0002403901,
    1: 0.0004712397,
    2: 0.0004522216,
    3: 0.0004235249,
    4: 0.0004279811,
}
DEFAULT_BETA_OOD = 0.16
DEFAULT_SIGMA_OOD = 0.0003998317
PER_SYM_BAND = {k: PER_SYM_BETA[k] * PER_SYM_SIGMA[k] for k in SYMS}


def ts():
    return datetime.now().strftime('%H:%M:%S')


def vectorized_pnl(action, mp_t, mp_th):
    side = action.astype(np.float64) - 1.0
    fee = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    return (side * (mp_th - mp_t) - fee) / (mp_t + 1.0)


def get_band_per_row(sym_arr):
    band = np.full(len(sym_arr), DEFAULT_BETA_OOD * DEFAULT_SIGMA_OOD, dtype=np.float64)
    for s, b in PER_SYM_BAND.items():
        band[sym_arr == s] = b
    return band


def load_preds():
    dfs_nn = [pd.read_parquet(os.path.join(T87_DIR, f"pred_T87_seed{s}_main.parquet")) for s in SEEDS]
    dfs_lgb = [pd.read_parquet(os.path.join(T75_DIR, f"pred_T75_seed{s}.parquet")) for s in SEEDS]

    base = dfs_lgb[0][['sym', 'date', 'session', 't', 'midprice_t', 'midprice_th']].copy()
    pred_nn = np.stack([df['pred_dmid_norm'].values for df in dfs_nn], axis=1).mean(axis=1)
    pred_lgb = np.stack([df['pred_dmid_norm'].values for df in dfs_lgb], axis=1).mean(axis=1)

    pred_dmid = (W_NN * pred_nn + W_LGB * pred_lgb) / (W_NN + W_LGB)
    sym_arr = base['sym'].values.astype(np.int32)
    mp_t = base['midprice_t'].values.astype(np.float64)
    mp_th = base['midprice_th'].values.astype(np.float64)

    print(f"[{ts()}] Loaded {len(pred_dmid)} rows, dates {base['date'].min()}-{base['date'].max()}", flush=True)
    return pred_dmid, sym_arr, mp_t, mp_th


def simulate_batches_baseline(pred_dmid, sym_arr, mp_t, mp_th, band_per_row):
    """v2 baseline: standard conformal gate, no batching (per-row gate is batch-invariant)."""
    action = np.ones(len(pred_dmid), dtype=np.int8)
    action[pred_dmid > (THR_UP + band_per_row)] = 2
    action[pred_dmid < -(THR_DN + band_per_row)] = 0
    pnl = vectorized_pnl(action, mp_t, mp_th)
    return float(pnl.sum()), int((action != 1).sum())


def simulate_batches(pred_dmid, sym_arr, mp_t, mp_th, band_per_row, variant_fn, seed=42):
    """Simulate platform-style batch_size=1024 shuffled batches, apply variant gate per batch."""
    N = len(pred_dmid)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)

    action = np.ones(N, dtype=np.int8)
    n_batches = (N + BATCH_SIZE - 1) // BATCH_SIZE

    for b in range(n_batches):
        batch_idx = idx[b * BATCH_SIZE: (b + 1) * BATCH_SIZE]
        batch_pred = pred_dmid[batch_idx]
        batch_band = band_per_row[batch_idx]
        batch_action = variant_fn(batch_pred, batch_band)
        action[batch_idx] = batch_action

    pnl = vectorized_pnl(action, mp_t, mp_th)
    return float(pnl.sum()), int((action != 1).sum())


def gate_v1(pred_dmid, band_per_row):
    """V1: multiplicative adaptive threshold."""
    GLOBAL_BASELINE_MED_ABS = 1.5e-4
    batch_med_abs = np.median(np.abs(pred_dmid))
    if batch_med_abs > 0:
        adapt_factor = batch_med_abs / GLOBAL_BASELINE_MED_ABS
        adaptive_thr_up = THR_UP / max(adapt_factor, 0.5)
        adaptive_thr_dn = THR_DN / max(adapt_factor, 0.5)
        adaptive_thr_up = min(adaptive_thr_up, THR_UP * 2.0)
        adaptive_thr_dn = min(adaptive_thr_dn, THR_DN * 2.0)
    else:
        adaptive_thr_up, adaptive_thr_dn = THR_UP, THR_DN

    out = np.ones(len(pred_dmid), dtype=np.int8)
    out[pred_dmid > (adaptive_thr_up + band_per_row)] = 2
    out[pred_dmid < -(adaptive_thr_dn + band_per_row)] = 0
    return out


def gate_v2(pred_dmid, band_per_row):
    """V2: percentile-anchored — top 25% by |pred| always trades."""
    FLOOR_PERCENTILE = 75
    pos_preds = pred_dmid[pred_dmid > 0]
    neg_preds = pred_dmid[pred_dmid < 0]

    if len(pos_preds) > 0:
        p_thr_up = np.percentile(pos_preds, FLOOR_PERCENTILE)
        adaptive_thr_up = max(THR_UP, p_thr_up)
    else:
        adaptive_thr_up = THR_UP

    if len(neg_preds) > 0:
        p_thr_dn = -np.percentile(neg_preds, 100 - FLOOR_PERCENTILE)
        adaptive_thr_dn = max(THR_DN, p_thr_dn)
    else:
        adaptive_thr_dn = THR_DN

    out = np.ones(len(pred_dmid), dtype=np.int8)
    out[pred_dmid > (adaptive_thr_up + band_per_row)] = 2
    out[pred_dmid < -(adaptive_thr_dn + band_per_row)] = 0
    return out


def gate_v3(pred_dmid, band_per_row):
    """V3: confidence-scaled predictions."""
    batch_q75_abs = np.percentile(np.abs(pred_dmid), 75)
    if batch_q75_abs > 0:
        scale = np.abs(pred_dmid) / batch_q75_abs
        scale = np.clip(scale, 0.0, 1.0)
        pred_scaled = pred_dmid * scale
    else:
        pred_scaled = pred_dmid

    out = np.ones(len(pred_dmid), dtype=np.int8)
    out[pred_scaled > (THR_UP + band_per_row)] = 2
    out[pred_scaled < -(THR_DN + band_per_row)] = 0
    return out


def gate_v4(pred_dmid, band_per_row):
    """V4: hybrid — adaptive only for extreme batches."""
    GLOBAL_BASELINE_MED_ABS = 1.5e-4
    STRONG_SIGNAL_FACTOR = 2.0
    WEAK_SIGNAL_FACTOR = 0.5

    batch_med_abs = np.median(np.abs(pred_dmid))
    ratio = batch_med_abs / GLOBAL_BASELINE_MED_ABS if GLOBAL_BASELINE_MED_ABS > 0 else 1.0

    if ratio >= STRONG_SIGNAL_FACTOR:
        adaptive_thr_up = THR_UP / min(ratio, 3.0)
        adaptive_thr_dn = THR_DN / min(ratio, 3.0)
    elif ratio <= WEAK_SIGNAL_FACTOR:
        adaptive_thr_up = THR_UP * (1.0 / max(ratio, 0.1))
        adaptive_thr_dn = THR_DN * (1.0 / max(ratio, 0.1))
        adaptive_thr_up = min(adaptive_thr_up, THR_UP * 3.0)
        adaptive_thr_dn = min(adaptive_thr_dn, THR_DN * 3.0)
    else:
        adaptive_thr_up = THR_UP
        adaptive_thr_dn = THR_DN

    out = np.ones(len(pred_dmid), dtype=np.int8)
    out[pred_dmid > (adaptive_thr_up + band_per_row)] = 2
    out[pred_dmid < -(adaptive_thr_dn + band_per_row)] = 0
    return out


def main():
    t0 = time.time()
    print(f"[{ts()}] T168 holdout evaluation starting...", flush=True)

    pred_dmid, sym_arr, mp_t, mp_th = load_preds()
    band_per_row = get_band_per_row(sym_arr)

    V2_BASELINE = 41.4939
    print(f"[{ts()}] v2 baseline (from thresholds.json): {V2_BASELINE}", flush=True)

    # Verify v2 baseline using batch-invariant gate
    v2_pnl, v2_trades = simulate_batches_baseline(pred_dmid, sym_arr, mp_t, mp_th, band_per_row)
    print(f"[{ts()}] v2 recomputed: {v2_pnl:.4f} (ref: {V2_BASELINE}), n_trades={v2_trades}", flush=True)

    # Eval each variant
    variants = {
        "V1_multiplicative": gate_v1,
        "V2_percentile_anchored": gate_v2,
        "V3_confidence_scaled": gate_v3,
        "V4_hybrid": gate_v4,
    }

    results = {}
    for vname, vfn in variants.items():
        pnl, n_trades = simulate_batches(pred_dmid, sym_arr, mp_t, mp_th, band_per_row, vfn, seed=42)
        delta = pnl - V2_BASELINE
        results[vname] = {
            "holdout_pnl": round(pnl, 4),
            "n_trades": n_trades,
            "delta": round(delta, 4),
        }
        print(f"[{ts()}] {vname}: pnl={pnl:.4f} delta={delta:+.4f} n_trades={n_trades}", flush=True)

    print(f"\n[{ts()}] Done in {time.time()-t0:.1f}s", flush=True)
    return results


if __name__ == "__main__":
    results = main()
    print("\n=== RESULTS ===")
    for vname, r in results.items():
        print(f"  {vname}: pnl={r['holdout_pnl']}, delta={r['delta']:+.4f}, n_trades={r['n_trades']}")
