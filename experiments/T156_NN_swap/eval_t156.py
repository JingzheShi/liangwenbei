#!/usr/bin/env python3
"""T156: NN swap evaluation — test diversity of T95/T97/T81 vs T87 using existing pred files.

Uses:
  - T87 NN preds (5 seeds, SPO+ DFL, MLP 256-128-64)
  - T95 GRU-C preds (5 seeds, GRU w100)
  - T97 preds (5 seeds, multi-head MLP 256-128-64, h60 output)
  - T81 preds (5 seeds, L2 regression NN)
  - T75 LGB preds (5 seeds, LOSO regression)

Evaluation:
  - Cross-correlation matrix between NN model avgs
  - 2-way: each_NN + T75 LGB with fixed iter_018 v1 weights (w_nn=1.0, w_lgb=1.5)
  - 3-way: T87 + T95/T97 + T75 LGB with DE-tuned weights (TSH-FT style)
  - conformal wrapper: per_sym_band from iter_018 v1

Does NOT retrain any model. Does NOT touch v2 LGB models.
"""
from __future__ import annotations

import json
import os
import sys
import time
import zipfile
import shutil
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

T87_DIR = os.path.join(ROOT, "experiments", "T87_spo_dfl")
T95_DIR = os.path.join(ROOT, "experiments", "T95_cnn_rnn_deeplob")
T97_DIR = os.path.join(ROOT, "experiments", "T97_multihead_nn")
T81_DIR = os.path.join(ROOT, "experiments", "T81_nn_regression_pnl")
T75_DIR = os.path.join(ROOT, "experiments", "T75_regression_dmid")
V2_PKG_DIR = os.path.join(ROOT, "experiments", "R_full_retrain", "pkg_iter019_v2")

SEEDS = (1, 7, 13, 42, 100)
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001

# iter_018 v1 thresholds
THR_UP = 0.00029995433796130955
THR_DN = 0.0002158528296175958

# iter_018 v1 conformal wrapper: per_sym_band = beta * sigma
PER_SYM_BETA = {0: 0.10, 1: 0.40, 2: 0.30, 3: 0.00, 4: 0.00}
PER_SYM_SIGMA = {0: 0.0002403901, 1: 0.0004712397, 2: 0.0004522216, 3: 0.0004235249, 4: 0.0004279811}
PER_SYM_BAND = {k: PER_SYM_BETA[k] * PER_SYM_SIGMA[k] for k in SYMS}

# v2 baseline weights
W_NN = 1.0
W_LGB = 1.5

# v2 LOSO reference (T127 iter_018 v1: T87+T75 LOSO with conformal = 41.49)
V2_LOSO_REF = 41.4939

PROGRESS_PATH = os.path.join(HERE, "worker-progress.json")


def ts():
    return datetime.now().strftime('%H:%M:%S')


def progress(step, metrics=None):
    p = {"status": "running", "step": step, "metrics": metrics or {},
         "timestamp": datetime.now().isoformat()}
    with open(PROGRESS_PATH, 'w') as f:
        json.dump(p, f, indent=2)
    print(f"[{ts()}] {step}", flush=True)


def vectorized_pnl(action, mp_t, mp_th):
    side = action.astype(np.float64) - 1.0
    abs_side = np.abs(side)
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * abs_side * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee_pnl) / denom


def conformal_action(pred_arr, sym_arr, thr_up, thr_dn, per_sym_band):
    """Apply conformal wrapper: effective thresholds per sym."""
    action = np.ones(len(pred_arr), dtype=np.int8)
    for sym in SYMS:
        mask = (sym_arr == sym)
        band = per_sym_band.get(sym, 0.0)
        eff_up = thr_up + band
        eff_dn = thr_dn + band
        p = pred_arr[mask]
        a = np.ones(mask.sum(), dtype=np.int8)
        a[p > eff_up] = 2
        a[p < -eff_dn] = 0
        action[mask] = a
    return action


def eval_config(base_df, pred_combined, label, thr_up=THR_UP, thr_dn=THR_DN,
                per_sym_band=PER_SYM_BAND):
    """Evaluate a prediction with conformal wrapper, return per-sym PnL sum."""
    sym_arr = base_df["sym"].to_numpy(np.int64)
    mp_t = base_df["midprice_t"].to_numpy(np.float64)
    mp_th = base_df["midprice_th"].to_numpy(np.float64)

    action = conformal_action(pred_combined, sym_arr, thr_up, thr_dn, per_sym_band)
    pnl = vectorized_pnl(action, mp_t, mp_th)

    per_sym = []
    for sym in SYMS:
        mask = (sym_arr == sym)
        per_sym.append(float(pnl[mask].sum()))

    total = float(sum(per_sym))
    active = float((action != 1).sum()) / len(action)
    print(f"  {label}: total={total:+.4f}  per_sym=[{', '.join(f'{x:+.3f}' for x in per_sym)}]  "
          f"active={active:.3f}", flush=True)
    return total, per_sym


def load_pred_avg(pred_dir, file_pattern, seeds=SEEDS):
    """Load and average predictions from 5 seeds. Returns (p_avg, base_df)."""
    base = None
    p = None
    for s in seeds:
        fpath = os.path.join(pred_dir, file_pattern.format(s))
        df = pd.read_parquet(fpath)
        if base is None:
            base = df
            p = np.zeros(len(df), dtype=np.float64)
        else:
            assert len(df) == len(base), f"row count mismatch {fpath}"
        p += df["pred_dmid_norm"].to_numpy(np.float64)
    p /= len(seeds)
    return p, base


def de_tune_3way(base_df, p_nn1, p_nn2, p_lgb, label):
    """DE-tune weights for 3-way ensemble (w_nn1, w_nn2, w_lgb, thr_up, thr_dn)."""
    sym_arr = base_df["sym"].to_numpy(np.int64)
    mp_t = base_df["midprice_t"].to_numpy(np.float64)
    mp_th = base_df["midprice_th"].to_numpy(np.float64)

    def objective(x):
        w1, w2, w3, tu, td = x
        s = w1 + w2 + w3
        if s < 1e-9:
            return 0.0
        pred = (w1 * p_nn1 + w2 * p_nn2 + w3 * p_lgb) / s
        action = conformal_action(pred, sym_arr, tu, td, PER_SYM_BAND)
        pnl = vectorized_pnl(action, mp_t, mp_th)
        return -float(pnl.sum())

    # Bounds: w_nn1 in [0.1,3], w_nn2 in [0.1,3], w_lgb in [0.1,3], thr_up in [0,0.005], thr_dn in [0,0.005]
    bounds = [(0.1, 3.0), (0.0, 3.0), (0.1, 3.0), (0.0, 0.005), (0.0, 0.005)]
    best_val = float('inf')
    best_x = None
    for sd in [0, 1, 7, 42]:
        result = differential_evolution(
            objective, bounds=bounds, seed=sd, maxiter=60, popsize=20,
            polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            updating="deferred", workers=1, init="sobol",
        )
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x

    w1, w2, w3, tu, td = best_x
    s = w1 + w2 + w3
    pred = (w1 * p_nn1 + w2 * p_nn2 + w3 * p_lgb) / s
    total, per_sym = eval_config(base_df, pred, label, thr_up=tu, thr_dn=td)
    print(f"    weights: w_nn1={w1:.3f} w_nn2={w2:.3f} w_lgb={w3:.3f} "
          f"thr_up={tu:.6f} thr_dn={td:.6f}", flush=True)
    return {
        "total": total, "per_sym": per_sym,
        "w_nn1": float(w1), "w_nn2": float(w2), "w_lgb": float(w3),
        "thr_up": float(tu), "thr_dn": float(td),
    }


def de_tune_2way(base_df, p_nn, p_lgb, label):
    """DE-tune thresholds only for 2-way with fixed weights (w_nn=1, w_lgb=1.5)."""
    sym_arr = base_df["sym"].to_numpy(np.int64)
    mp_t = base_df["midprice_t"].to_numpy(np.float64)
    mp_th = base_df["midprice_th"].to_numpy(np.float64)
    pred = (W_NN * p_nn + W_LGB * p_lgb) / (W_NN + W_LGB)

    def objective(x):
        tu, td = x
        action = conformal_action(pred, sym_arr, tu, td, PER_SYM_BAND)
        pnl = vectorized_pnl(action, mp_t, mp_th)
        return -float(pnl.sum())

    bounds = [(0.0, 0.005), (0.0, 0.005)]
    best_val = float('inf')
    best_x = None
    for sd in [0, 1, 7, 42]:
        result = differential_evolution(
            objective, bounds=bounds, seed=sd, maxiter=60, popsize=20,
            polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            updating="deferred", workers=1, init="sobol",
        )
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x

    tu, td = best_x
    total, per_sym = eval_config(base_df, pred, label + " (DE-thresh)", thr_up=tu, thr_dn=td)
    print(f"    thr_up={tu:.6f} thr_dn={td:.6f}", flush=True)
    return {
        "total": total, "per_sym": per_sym,
        "w_nn": W_NN, "w_lgb": W_LGB,
        "thr_up": float(tu), "thr_dn": float(td),
    }


def extract_t97_npz(pt_path, npz_path):
    """Extract T97 h60 weights → .npz compatible with T87 Predictor format."""
    import torch
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    main_idx = int(ckpt["main_idx"])
    target_scale = float(ckpt["target_scales"][main_idx])
    feat_mean = np.asarray(ckpt["feat_mean"], dtype=np.float32)
    feat_std = np.asarray(ckpt["feat_std"], dtype=np.float32)
    keep_idx = np.asarray(ckpt["keep_idx"], dtype=np.int64)
    clip_val = float(ckpt.get("clip", 10.0))
    in_dim = int(ckpt["in_dim"])
    trunk = list(ckpt["trunk"])
    use_ln = bool(ckpt.get("use_layernorm", True))

    out = {
        "in_dim": np.array([in_dim], dtype=np.int64),
        "hidden": np.array(trunk, dtype=np.int64),
        "use_layernorm": np.array([1 if use_ln else 0], dtype=np.int8),
        "target_scale": np.array([target_scale], dtype=np.float32),
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "keep_idx": keep_idx,
        "clip": np.array([clip_val], dtype=np.float32),
    }
    # trunk layers: 0=Linear, 1=LN, [2=GELU, 3=Dropout], 4=Linear, 5=LN, ...
    trunk_indices = [(0, 1), (4, 5), (8, 9)]  # (linear_idx, ln_idx)
    for i, (li, lni) in enumerate(trunk_indices):
        out[f"L{i}_W"] = sd[f"trunk.{li}.weight"].numpy().astype(np.float32)
        out[f"L{i}_b"] = sd[f"trunk.{li}.bias"].numpy().astype(np.float32)
        if use_ln:
            out[f"LN{i}_W"] = sd[f"trunk.{lni}.weight"].numpy().astype(np.float32)
            out[f"LN{i}_b"] = sd[f"trunk.{lni}.bias"].numpy().astype(np.float32)
    out["LF_W"] = sd[f"heads.{main_idx}.weight"].numpy().astype(np.float32)
    out["LF_b"] = sd[f"heads.{main_idx}.bias"].numpy().astype(np.float32)

    np.savez(npz_path, **out)
    print(f"  extracted: {npz_path}  target_scale={target_scale:.1f}", flush=True)


def build_submission_zip_t97(out_dir, zip_name, de_result_3way=None):
    """Build submission zip with T97 NN replacing T87 NN (2-way T97+LGB)."""
    import torch

    pkg_src = V2_PKG_DIR
    build_dir = os.path.join(out_dir, "pkg_t97")
    os.makedirs(build_dir, exist_ok=True)

    # Copy v2 package files
    for f in os.listdir(pkg_src):
        if f.endswith('.txt') or f.endswith('.py') or f.endswith('.json') or f == 'requirements.txt':
            shutil.copy2(os.path.join(pkg_src, f), os.path.join(build_dir, f))

    # Extract T97 .npz for each seed
    for s in SEEDS:
        pt_path = os.path.join(T97_DIR, f"model_T97_seed{s}_main.pt")
        npz_path = os.path.join(build_dir, f"nn_h60_seed{s}.npz")
        extract_t97_npz(pt_path, npz_path)

    # Update thresholds.json with new config
    with open(os.path.join(pkg_src, "thresholds.json")) as f:
        thresh = json.load(f)

    if de_result_3way:
        # 3-way not supported in current Predictor; use 2-way T97+LGB with DE-tuned thresholds
        pass

    # Use 2-way with iter_018 thresholds (keep same as v2)
    with open(os.path.join(build_dir, "thresholds.json"), "w") as f:
        json.dump(thresh, f, indent=2)

    # Create zip
    zip_path = os.path.join(out_dir, zip_name)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in os.listdir(build_dir):
            zf.write(os.path.join(build_dir, f), f)

    md5 = hashlib.md5(open(zip_path, 'rb').read()).hexdigest()
    sz = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"  zip: {zip_path}  md5={md5}  {sz:.1f}MB", flush=True)
    return zip_path, md5


def main():
    progress("Step 1: Loading predictions")
    t0 = time.time()

    # T87: pred_T87_seed{s}_main.parquet
    p_t87, base = load_pred_avg(T87_DIR, "pred_T87_seed{}_main.parquet")
    print(f"  T87 loaded: mean={p_t87.mean():+.6f} std={p_t87.std():.6f}", flush=True)

    # T95 GRU-C: pred_T95_gru_w100_C_seed{s}.parquet
    p_t95, _ = load_pred_avg(T95_DIR, "pred_T95_gru_w100_C_seed{}.parquet")
    print(f"  T95 GRU-C loaded: mean={p_t95.mean():+.6f} std={p_t95.std():.6f}", flush=True)

    # T97: pred_T97_seed{s}_main.parquet
    p_t97, _ = load_pred_avg(T97_DIR, "pred_T97_seed{}_main.parquet")
    print(f"  T97 multi-head loaded: mean={p_t97.mean():+.6f} std={p_t97.std():.6f}", flush=True)

    # T81: pred_T81_seed{s}.parquet
    p_t81, _ = load_pred_avg(T81_DIR, "pred_T81_seed{}.parquet")
    print(f"  T81 L2 NN loaded: mean={p_t81.mean():+.6f} std={p_t81.std():.6f}", flush=True)

    # T75: pred_T75_seed{s}.parquet
    p_t75, _ = load_pred_avg(T75_DIR, "pred_T75_seed{}.parquet")
    print(f"  T75 LGB loaded: mean={p_t75.mean():+.6f} std={p_t75.std():.6f}", flush=True)

    progress("Step 2: Cross-correlation matrix")
    nn_models = {"T87": p_t87, "T95_GRU-C": p_t95, "T97_multihead": p_t97, "T81_L2": p_t81, "T75_LGB": p_t75}
    names = list(nn_models.keys())
    preds = [nn_models[n] for n in names]
    n = len(names)
    corr_matrix = {}
    print("\nCross-correlation matrix:", flush=True)
    for i in range(n):
        row = {}
        for j in range(n):
            c = float(np.corrcoef(preds[i], preds[j])[0, 1])
            row[names[j]] = round(c, 4)
        corr_matrix[names[i]] = row
        print(f"  {names[i]:20s}: {' '.join(f'{row[nm]:+.3f}' for nm in names)}", flush=True)

    progress("Step 3: 2-way LOSO evaluations (fixed v2 weights)")
    print("\n=== 2-way ensembles (w_nn=1.0, w_lgb=1.5, iter_018 v1 conformal) ===", flush=True)

    sym_arr = base["sym"].to_numpy(np.int64)
    mp_t = base["midprice_t"].to_numpy(np.float64)
    mp_th = base["midprice_th"].to_numpy(np.float64)

    configs_2way = {}

    # v2 baseline: T87 + T75 LGB
    p_v2 = (W_NN * p_t87 + W_LGB * p_t75) / (W_NN + W_LGB)
    total_v2, per_sym_v2 = eval_config(base, p_v2, "v2_baseline (T87+T75)")
    configs_2way["v2_baseline_T87+LGB"] = {
        "loso": total_v2, "per_sym": per_sym_v2,
        "thr_up": THR_UP, "thr_dn": THR_DN, "w_nn": W_NN, "w_lgb": W_LGB
    }

    # T95 GRU-C + T75 LGB
    p_t95_lgb = (W_NN * p_t95 + W_LGB * p_t75) / (W_NN + W_LGB)
    total_t95, per_sym_t95 = eval_config(base, p_t95_lgb, "T95_GRU-C+T75")
    configs_2way["T95_GRUC+LGB"] = {
        "loso": total_t95, "per_sym": per_sym_t95,
        "thr_up": THR_UP, "thr_dn": THR_DN, "w_nn": W_NN, "w_lgb": W_LGB
    }

    # T97 multihead + T75 LGB
    p_t97_lgb = (W_NN * p_t97 + W_LGB * p_t75) / (W_NN + W_LGB)
    total_t97, per_sym_t97 = eval_config(base, p_t97_lgb, "T97_multihead+T75")
    configs_2way["T97_multihead+LGB"] = {
        "loso": total_t97, "per_sym": per_sym_t97,
        "thr_up": THR_UP, "thr_dn": THR_DN, "w_nn": W_NN, "w_lgb": W_LGB
    }

    # T81 L2 + T75 LGB
    p_t81_lgb = (W_NN * p_t81 + W_LGB * p_t75) / (W_NN + W_LGB)
    total_t81, per_sym_t81 = eval_config(base, p_t81_lgb, "T81_L2+T75")
    configs_2way["T81_L2+LGB"] = {
        "loso": total_t81, "per_sym": per_sym_t81,
        "thr_up": THR_UP, "thr_dn": THR_DN, "w_nn": W_NN, "w_lgb": W_LGB
    }

    print(f"\n  v2 reference LOSO (from T127 with conformal): {V2_LOSO_REF}", flush=True)
    print(f"  v2 reproduced local: {total_v2:+.4f}", flush=True)

    progress("Step 3b: DE-tune thresholds for 2-way alternatives")
    print("\n=== 2-way with DE-tuned thresholds ===", flush=True)

    de_t95_2way = de_tune_2way(base, p_t95, p_t75, "T95_GRU-C+T75")
    de_t97_2way = de_tune_2way(base, p_t97, p_t75, "T97_multihead+T75")
    de_t81_2way = de_tune_2way(base, p_t81, p_t75, "T81_L2+T75")

    configs_2way["T95_GRUC+LGB_DE"] = de_t95_2way
    configs_2way["T97_multihead+LGB_DE"] = de_t97_2way
    configs_2way["T81_L2+LGB_DE"] = de_t81_2way

    progress("Step 4: 3-way ensembles with DE-tuned weights (TSH-FT style)")
    print("\n=== 3-way ensembles (DE-tuned weights) ===", flush=True)
    configs_3way = {}

    # T87 + T95 + T75 LGB
    de_3way_t87t95 = de_tune_3way(base, p_t87, p_t95, p_t75, "T87+T95_GRU-C+T75_3way")
    configs_3way["T87+T95_GRUC+LGB_3way"] = de_3way_t87t95
    print(f"  vs v2 baseline: {de_3way_t87t95['total'] - total_v2:+.4f}", flush=True)

    # T87 + T97 + T75 LGB
    de_3way_t87t97 = de_tune_3way(base, p_t87, p_t97, p_t75, "T87+T97_multihead+T75_3way")
    configs_3way["T87+T97_multihead+LGB_3way"] = de_3way_t87t97
    print(f"  vs v2 baseline: {de_3way_t87t97['total'] - total_v2:+.4f}", flush=True)

    # T87 + T81 + T75 LGB
    de_3way_t87t81 = de_tune_3way(base, p_t87, p_t81, p_t75, "T87+T81_L2+T75_3way")
    configs_3way["T87+T81_L2+LGB_3way"] = de_3way_t87t81
    print(f"  vs v2 baseline: {de_3way_t87t81['total'] - total_v2:+.4f}", flush=True)

    progress("Step 5: Identifying best config and build decisions")
    print("\n=== SUMMARY ===", flush=True)
    print(f"  v2 baseline LOSO: {total_v2:+.4f}  (T127 ref: {V2_LOSO_REF})", flush=True)

    all_configs = {}
    all_configs.update({k: v for k, v in configs_2way.items()})
    all_configs.update({k: v for k, v in configs_3way.items()})

    # Find best
    best_key = None
    best_val = total_v2
    for k, v in all_configs.items():
        score = v.get("total") or v.get("loso", 0)
        print(f"  {k}: {score:+.4f}  delta_vs_v2={score - total_v2:+.4f}", flush=True)
        if score > best_val:
            best_val = score
            best_key = k

    # Cross-correlations of NN models with T87
    corr_t87_t95 = corr_matrix["T87"]["T95_GRU-C"]
    corr_t87_t97 = corr_matrix["T87"]["T97_multihead"]
    corr_t87_t81 = corr_matrix["T87"]["T81_L2"]

    # Build submission zip if 3-way with T97 beats v2 by >+0.5
    v2_nn_alt_zip = None
    v2_nn_alt_md5 = None
    expected_platform = None
    best_config_name = best_key or "v2_baseline_T87+LGB"

    # Check if T97 beats v2 in 3-way by >+0.5
    t97_3way_delta = de_3way_t87t97['total'] - total_v2
    t95_3way_delta = de_3way_t87t95['total'] - total_v2

    # Check for viable candidates (corr < 0.85 and loso >= 20)
    viable_t95 = (corr_t87_t95 < 0.85 and configs_2way["T95_GRUC+LGB"]["loso"] >= 20)
    viable_t97 = (corr_t87_t97 < 0.85 and configs_2way["T97_multihead+LGB"]["loso"] >= 20)

    print(f"\n  T95 GRU-C: corr_vs_T87={corr_t87_t95:.4f}, viable={viable_t95}", flush=True)
    print(f"  T97 multi-head: corr_vs_T87={corr_t87_t97:.4f}, viable={viable_t97}", flush=True)
    print(f"  T97 3-way delta vs v2: {t97_3way_delta:+.4f}", flush=True)
    print(f"  T95 3-way delta vs v2: {t95_3way_delta:+.4f}", flush=True)

    # Build T97 alt zip if:
    # (a) T97 2-way beats v2 by >+0.5, OR (b) T97 3-way beats v2 by >+0.5
    t97_2way_delta = de_t97_2way["total"] - total_v2
    should_build_t97 = (t97_2way_delta > 0.5 or t97_3way_delta > 0.5) and viable_t97

    if should_build_t97:
        progress("Step 6: Building T97 submission zip")
        print(f"\n  Building T97 submission zip (3-way delta={t97_3way_delta:+.4f})", flush=True)
        zip_name = "submission_050809_iter019_v2NN_t97_2way.zip"
        try:
            zip_path, zip_md5 = build_submission_zip_t97(HERE, zip_name)
            v2_nn_alt_zip = zip_path
            v2_nn_alt_md5 = zip_md5
            expected_platform = f"+{total_v2 + t97_2way_delta * 0.7:.1f} (estimated)"
        except Exception as e:
            print(f"  ERROR building zip: {e}", flush=True)
            import traceback; traceback.print_exc()
    else:
        print(f"\n  No candidate beats v2 by >+0.5 — no zip built", flush=True)
        print(f"  T97 2-way delta: {t97_2way_delta:+.4f}", flush=True)
        print(f"  T97 3-way delta: {t97_3way_delta:+.4f}", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Total elapsed: {elapsed:.0f}s", flush=True)

    # Build results.json
    results = {
        "task": "T156 NN swap evaluation",
        "cross_correlations": {
            "T87-T95_GRU-C": corr_t87_t95,
            "T87-T97_multihead": corr_t87_t97,
            "T87-T81_L2": corr_t87_t81,
            "T87-T75_LGB": corr_matrix["T87"]["T75_LGB"],
            "T95-T97": corr_matrix["T95_GRU-C"]["T97_multihead"],
            "T95-T81": corr_matrix["T95_GRU-C"]["T81_L2"],
            "T97-T81": corr_matrix["T97_multihead"]["T81_L2"],
        },
        "full_corr_matrix": corr_matrix,
        "configs": {
            "v2_baseline_T87+LGB": {
                "loso": total_v2, "per_sym": per_sym_v2,
                "ref_loso_T127": V2_LOSO_REF,
                "note": "T75 LGB (LOSO) + T87 NN (SPO+) + iter_018 conformal"
            },
            "T95_GRUC+LGB": {
                "loso_fixed_weights": total_t95, "loso_de_thresh": de_t95_2way["total"],
                "loso_de_thr_up": de_t95_2way["thr_up"], "loso_de_thr_dn": de_t95_2way["thr_dn"],
                "per_sym": per_sym_t95
            },
            "T97_multihead+LGB": {
                "loso_fixed_weights": total_t97, "loso_de_thresh": de_t97_2way["total"],
                "loso_de_thr_up": de_t97_2way["thr_up"], "loso_de_thr_dn": de_t97_2way["thr_dn"],
                "per_sym": per_sym_t97
            },
            "T81_L2+LGB": {
                "loso_fixed_weights": total_t81, "loso_de_thresh": de_t81_2way["total"],
                "per_sym": per_sym_t81
            },
            "T87+T95_GRUC+LGB_3way_tshft": {
                "loso": de_3way_t87t95["total"], "per_sym": de_3way_t87t95["per_sym"],
                "weights": {
                    "w_nn1_T87": de_3way_t87t95["w_nn1"],
                    "w_nn2_T95": de_3way_t87t95["w_nn2"],
                    "w_lgb": de_3way_t87t95["w_lgb"]
                },
                "thr_up": de_3way_t87t95["thr_up"], "thr_dn": de_3way_t87t95["thr_dn"],
                "delta_vs_v2_loso": t95_3way_delta
            },
            "T87+T97_multihead+LGB_3way_tshft": {
                "loso": de_3way_t87t97["total"], "per_sym": de_3way_t87t97["per_sym"],
                "weights": {
                    "w_nn1_T87": de_3way_t87t97["w_nn1"],
                    "w_nn2_T97": de_3way_t87t97["w_nn2"],
                    "w_lgb": de_3way_t87t97["w_lgb"]
                },
                "thr_up": de_3way_t87t97["thr_up"], "thr_dn": de_3way_t87t97["thr_dn"],
                "delta_vs_v2_loso": t97_3way_delta
            },
            "T87+T81_L2+LGB_3way_tshft": {
                "loso": de_3way_t87t81["total"], "per_sym": de_3way_t87t81["per_sym"],
                "weights": {
                    "w_nn1_T87": de_3way_t87t81["w_nn1"],
                    "w_nn2_T81": de_3way_t87t81["w_nn2"],
                    "w_lgb": de_3way_t87t81["w_lgb"]
                },
                "thr_up": de_3way_t87t81["thr_up"], "thr_dn": de_3way_t87t81["thr_dn"],
                "delta_vs_v2_loso": de_3way_t87t81["total"] - total_v2
            },
        },
        "viable_nn_candidates": {
            "T95_GRU-C": viable_t95,
            "T97_multihead": viable_t97,
        },
        "best_config": best_config_name,
        "best_loso": best_val,
        "best_delta_vs_v2": best_val - total_v2,
        "v2_nn_alt_zip": v2_nn_alt_zip,
        "v2_nn_alt_md5": v2_nn_alt_md5,
        "expected_platform": expected_platform,
        "elapsed_sec": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(),
    }

    results_path = os.path.join(HERE, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  wrote -> {results_path}", flush=True)

    progress("done", {"best_config": best_config_name, "best_loso": best_val, "best_delta_vs_v2": best_val - total_v2})

    print(f"\n{'='*70}", flush=True)
    print(f"RESULT: task=T156_NN_swap metrics={{v2_loso={total_v2:.4f}, "
          f"t95_3way={de_3way_t87t95['total']:.4f}, t97_3way={de_3way_t87t97['total']:.4f}, "
          f"best_delta={best_val-total_v2:+.4f}}} notes=[{best_config_name}]", flush=True)
    print('='*70, flush=True)


if __name__ == "__main__":
    main()
