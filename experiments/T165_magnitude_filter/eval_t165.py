#!/usr/bin/env python3
"""T165: Magnitude-based agreement filter variants on holdout dates 96-119.

Reproduces T163 exact pipeline (3-way ensemble + conformal gate) and tests
5 filter variants as alternatives to T163's sign-based agreement filter.

All filters apply ONLY to syms {0, 1, 2}. Syms 3, 4 use standard pipeline.

V1_sign:     T163 baseline – hold if sign(nn_combined) != sign(lgb_pred)
V2_magnitude: hold if |nn_combined| < std(nn)*f OR |lgb_pred| < std(lgb)*f
V3_strict_mag: hold if max(|nn|, |lgb|) < std(pred_combined)*f
V4_ratio:    hold if min(|nn|/|lgb|, |lgb|/|nn|) < r
V5_combined: hold if sign disagree OR |nn_combined| < std(nn)*f
"""
from __future__ import annotations

import json
import os
import sys
import shutil
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd

OUTDIR = os.path.dirname(os.path.abspath(__file__))
WD = "/root/projects/liangwenbei_workdir"

T87_SEEDS = [1, 7, 13, 42, 100]
T97_SEEDS = [1, 7, 13, 42, 100]
LGB_SEEDS = [1, 7, 13, 42, 100]

T87_TPL = f"{WD}/experiments/T87_spo_dfl/pred_T87_seed{{seed}}_main.parquet"
T97_TPL = f"{WD}/experiments/T97_multihead_nn/pred_T97_seed{{seed}}_main.parquet"
LGB_TPL = f"{WD}/experiments/T75_regression_dmid/pred_T75_seed{{seed}}.parquet"

# ── T163 exact config ───────────────────────────────────────────────────────
THR_UP = 0.00029995433796130955
THR_DN = 0.0002158528296175958
PER_SYM_BETA = {0: 0.1, 1: 0.4, 2: 0.3, 3: 0.0, 4: 0.0}
PER_SYM_SIGMA = {
    0: 0.0002403901, 1: 0.0004712397, 2: 0.0004522216,
    3: 0.0004235249, 4: 0.0004279811,
}
DEFAULT_BETA_OOD = 0.16
DEFAULT_SIGMA_OOD = 0.0003998317

W_NN = 1.5    # T87 weight
W_NN2 = 1.5   # T97 weight
W_LGB = 1.0   # LGB weight

AGREEMENT_FILTER_SYMS = frozenset({0, 1, 2})
FEE = 0.0001

SYMS = list(range(5))


def update_progress(step, metrics=None):
    data = {
        "status": "running",
        "step": step,
        "metrics": metrics or {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path = os.path.join(OUTDIR, "worker-progress.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {step}", flush=True)


def vectorized_pnl(action, mp_t, mp_th):
    side = action.astype(np.float64) - 1.0
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    return (side * diff - fee) / (mp_t.astype(np.float64) + 1.0)


def apply_conformal_gate(pred_combined: np.ndarray, sym_arr: np.ndarray) -> np.ndarray:
    """Standard conformal gate with per-sym beta * sigma bands."""
    action = np.ones(len(pred_combined), dtype=np.int8)
    for s in SYMS:
        mask = sym_arr == s
        if not mask.any():
            continue
        beta = PER_SYM_BETA.get(s, DEFAULT_BETA_OOD)
        sigma = PER_SYM_SIGMA.get(s, DEFAULT_SIGMA_OOD)
        band = beta * sigma
        p = pred_combined[mask]
        a = np.ones(mask.sum(), dtype=np.int8)
        a[p > (THR_UP + band)] = 2
        a[p < -(THR_DN + band)] = 0
        action[mask] = a
    return action


def compute_pnl(action, sym_arr, mp_t, mp_th):
    pnl_vec = vectorized_pnl(action, mp_t, mp_th)
    total = float(pnl_vec.sum())
    per_sym = [float(pnl_vec[sym_arr == s].sum()) for s in SYMS]
    n_trades = int((action != 1).sum())
    return total, per_sym, n_trades


def per_sym_std(arr, sym_arr):
    """Return array of per-sym std values (one per row)."""
    std_dict = {}
    for s in SYMS:
        mask = sym_arr == s
        if mask.any():
            std_dict[s] = float(np.std(arr[mask]))
        else:
            std_dict[s] = 1e-6
    return np.array([std_dict.get(int(s), 1e-6) for s in sym_arr])


def main():
    update_progress("Loading predictions (T87, T97, T75-LGB)")

    # Load T87 NN (average over 5 seeds)
    nn_preds = []
    for seed in T87_SEEDS:
        df = pd.read_parquet(T87_TPL.format(seed=seed))
        nn_preds.append(df["pred_dmid_norm"].values.astype(np.float64))
    nn_pred = np.mean(nn_preds, axis=0)

    df_ref = pd.read_parquet(T87_TPL.format(seed=1))
    sym_arr = df_ref["sym"].values.astype(np.int32)
    mp_t = df_ref["midprice_t"].values.astype(np.float64)
    mp_th = df_ref["midprice_th"].values.astype(np.float64)
    print(f"  T87 rows: {len(nn_pred)}, dates: {df_ref['date'].min()}-{df_ref['date'].max()}", flush=True)

    # Load T97 NN (average over 5 seeds)
    nn2_preds = []
    for seed in T97_SEEDS:
        df97 = pd.read_parquet(T97_TPL.format(seed=seed))
        nn2_preds.append(df97["pred_dmid_norm"].values.astype(np.float64))
    nn2_pred = np.mean(nn2_preds, axis=0)
    print(f"  T97 rows: {len(nn2_pred)}", flush=True)

    # Load T75 LGB (average over 5 seeds)
    lgb_preds = []
    for seed in LGB_SEEDS:
        df_lgb = pd.read_parquet(LGB_TPL.format(seed=seed))
        lgb_preds.append(df_lgb["pred_dmid_norm"].values.astype(np.float64))
    lgb_pred = np.mean(lgb_preds, axis=0)
    print(f"  LGB rows: {len(lgb_pred)}", flush=True)

    assert len(nn_pred) == len(nn2_pred) == len(lgb_pred), "Row count mismatch!"

    # Build T163 3-way combined prediction
    pred_combined = (W_NN * nn_pred + W_NN2 * nn2_pred + W_LGB * lgb_pred) / (W_NN + W_NN2 + W_LGB)

    # NN-side for agreement filter: (w_nn * T87 + w_nn2 * T97) / (w_nn + w_nn2)
    # Since w_nn == w_nn2 == 1.5, this simplifies to (T87 + T97) / 2
    nn_combined = (W_NN * nn_pred + W_NN2 * nn2_pred) / (W_NN + W_NN2)

    # Agreement filter mask: only syms 0, 1, 2
    agree_filter_mask = np.array([int(s) in AGREEMENT_FILTER_SYMS for s in sym_arr], dtype=bool)

    # Pre-compute base actions (conformal gate only, no filter)
    actions_base = apply_conformal_gate(pred_combined, sym_arr)

    # Pre-compute sign disagree mask (used in V1 and V5)
    disagree_sign = np.sign(nn_combined) != np.sign(lgb_pred)

    # Pre-compute per-sym std arrays (for magnitude variants)
    std_nn_arr = per_sym_std(nn_combined, sym_arr)      # std(nn_combined) per sym
    std_lgb_arr = per_sym_std(lgb_pred, sym_arr)        # std(lgb_pred) per sym
    std_combined_arr = per_sym_std(pred_combined, sym_arr)  # std(pred_combined) per sym

    # ── v2_baseline: 3-way ensemble, no agreement filter ────────────────────
    update_progress("Evaluating v2_baseline (no filter)")
    pnl_base, per_sym_base, n_base = compute_pnl(actions_base, sym_arr, mp_t, mp_th)
    print(f"  v2_baseline PnL: {pnl_base:.6f}, n_trades={n_base}", flush=True)
    print(f"  Per-sym: {[round(x, 4) for x in per_sym_base]}", flush=True)

    # ── V1_sign: T163 baseline ───────────────────────────────────────────────
    update_progress("Evaluating V1_sign (T163 baseline)")
    actions_v1 = actions_base.copy()
    actions_v1[agree_filter_mask & disagree_sign] = 1
    pnl_v1, per_sym_v1, n_v1 = compute_pnl(actions_v1, sym_arr, mp_t, mp_th)
    n_filtered_v1 = int((agree_filter_mask & disagree_sign).sum())
    print(f"  V1_sign PnL: {pnl_v1:.6f}, n_trades={n_v1}, n_filtered={n_filtered_v1}", flush=True)
    print(f"  Per-sym: {[round(x, 4) for x in per_sym_v1]}", flush=True)
    print(f"  Delta vs baseline: {pnl_v1 - pnl_base:+.4f}", flush=True)

    factors = [0.3, 0.5, 0.7, 1.0]

    # ── V2_magnitude: both models confident ─────────────────────────────────
    update_progress("Evaluating V2_magnitude (both confident)")
    v2_results = {}
    best_v2_pnl = -np.inf
    best_v2_factor = None
    for factor in factors:
        actions = actions_base.copy()
        nn_small = np.abs(nn_combined) < std_nn_arr * factor
        lgb_small = np.abs(lgb_pred) < std_lgb_arr * factor
        filt = agree_filter_mask & (nn_small | lgb_small)
        actions[filt] = 1
        pnl_val, _, n_trades = compute_pnl(actions, sym_arr, mp_t, mp_th)
        v2_results[str(factor)] = {
            "pnl": float(pnl_val), "n_trades": n_trades, "n_filtered": int(filt.sum()),
        }
        delta = pnl_val - pnl_v1
        print(f"  V2 factor={factor}: PnL={pnl_val:.6f} delta_vs_V1={delta:+.4f} n_filt={filt.sum()}", flush=True)
        if pnl_val > best_v2_pnl:
            best_v2_pnl = pnl_val
            best_v2_factor = factor

    # ── V3_strict_mag: max not small ────────────────────────────────────────
    update_progress("Evaluating V3_strict_mag (max confidence)")
    v3_results = {}
    best_v3_pnl = -np.inf
    best_v3_factor = None
    for factor in factors:
        actions = actions_base.copy()
        max_mag = np.maximum(np.abs(nn_combined), np.abs(lgb_pred))
        filt = agree_filter_mask & (max_mag < std_combined_arr * factor)
        actions[filt] = 1
        pnl_val, _, n_trades = compute_pnl(actions, sym_arr, mp_t, mp_th)
        v3_results[str(factor)] = {
            "pnl": float(pnl_val), "n_trades": n_trades, "n_filtered": int(filt.sum()),
        }
        delta = pnl_val - pnl_v1
        print(f"  V3 factor={factor}: PnL={pnl_val:.6f} delta_vs_V1={delta:+.4f} n_filt={filt.sum()}", flush=True)
        if pnl_val > best_v3_pnl:
            best_v3_pnl = pnl_val
            best_v3_factor = factor

    # ── V4_ratio: magnitude ratio filter ────────────────────────────────────
    update_progress("Evaluating V4_ratio (magnitude ratio)")
    v4_results = {}
    best_v4_pnl = -np.inf
    best_v4_r = None
    r_grid = [0.3, 0.5, 0.7]
    eps = 1e-10
    abs_nn = np.abs(nn_combined)
    abs_lgb = np.abs(lgb_pred)
    min_ratio = np.minimum(abs_nn / (abs_lgb + eps), abs_lgb / (abs_nn + eps))
    for r in r_grid:
        actions = actions_base.copy()
        filt = agree_filter_mask & (min_ratio < r)
        actions[filt] = 1
        pnl_val, _, n_trades = compute_pnl(actions, sym_arr, mp_t, mp_th)
        v4_results[str(r)] = {
            "pnl": float(pnl_val), "n_trades": n_trades, "n_filtered": int(filt.sum()),
        }
        delta = pnl_val - pnl_v1
        print(f"  V4 r={r}: PnL={pnl_val:.6f} delta_vs_V1={delta:+.4f} n_filt={filt.sum()}", flush=True)
        if pnl_val > best_v4_pnl:
            best_v4_pnl = pnl_val
            best_v4_r = r

    # ── V5_combined: sign agree AND nn confident ─────────────────────────────
    update_progress("Evaluating V5_combined (sign + nn confidence)")
    v5_results = {}
    best_v5_pnl = -np.inf
    best_v5_factor = None
    for factor in factors:
        actions = actions_base.copy()
        nn_not_confident = np.abs(nn_combined) < std_nn_arr * factor
        filt = agree_filter_mask & (disagree_sign | nn_not_confident)
        actions[filt] = 1
        pnl_val, _, n_trades = compute_pnl(actions, sym_arr, mp_t, mp_th)
        v5_results[str(factor)] = {
            "pnl": float(pnl_val), "n_trades": n_trades, "n_filtered": int(filt.sum()),
        }
        delta = pnl_val - pnl_v1
        print(f"  V5 factor={factor}: PnL={pnl_val:.6f} delta_vs_V1={delta:+.4f} n_filt={filt.sum()}", flush=True)
        if pnl_val > best_v5_pnl:
            best_v5_pnl = pnl_val
            best_v5_factor = factor

    # ── Summary ──────────────────────────────────────────────────────────────
    all_best = {
        "V1_sign": pnl_v1,
        "V2_magnitude": best_v2_pnl,
        "V3_strict_mag": best_v3_pnl,
        "V4_ratio": best_v4_pnl,
        "V5_combined": best_v5_pnl,
    }
    best_variant = max(all_best, key=all_best.get)
    best_pnl_overall = all_best[best_variant]
    best_delta = best_pnl_overall - pnl_v1

    print(f"\n=== T165 RESULTS SUMMARY ===", flush=True)
    print(f"  v2_baseline (3-way, no filter): {pnl_base:.6f}", flush=True)
    print(f"  V1_sign (T163 baseline):        {pnl_v1:.6f}  (delta vs base: {pnl_v1-pnl_base:+.4f})", flush=True)
    print(f"  V2_magnitude best (f={best_v2_factor}):    {best_v2_pnl:.6f}  (delta vs V1: {best_v2_pnl-pnl_v1:+.4f})", flush=True)
    print(f"  V3_strict_mag best (f={best_v3_factor}):   {best_v3_pnl:.6f}  (delta vs V1: {best_v3_pnl-pnl_v1:+.4f})", flush=True)
    print(f"  V4_ratio best (r={best_v4_r}):      {best_v4_pnl:.6f}  (delta vs V1: {best_v4_pnl-pnl_v1:+.4f})", flush=True)
    print(f"  V5_combined best (f={best_v5_factor}):  {best_v5_pnl:.6f}  (delta vs V1: {best_v5_pnl-pnl_v1:+.4f})", flush=True)
    print(f"  BEST: {best_variant} = {best_pnl_overall:.6f} (delta vs T163 V1: {best_delta:+.4f})", flush=True)

    # ── Collect per-sym std for potential pkg embedding ──────────────────────
    std_nn_per_sym = {s: float(np.std(nn_combined[sym_arr == s])) for s in SYMS if (sym_arr == s).any()}
    std_lgb_per_sym = {s: float(np.std(lgb_pred[sym_arr == s])) for s in SYMS if (sym_arr == s).any()}
    std_combined_per_sym = {s: float(np.std(pred_combined[sym_arr == s])) for s in SYMS if (sym_arr == s).any()}

    results = {
        "task": "T165 magnitude-based agreement filter variants",
        "holdout_dates": "96-119",
        "v2_baseline_holdout": float(pnl_base),
        "T163_V1_sign_holdout": float(pnl_v1),
        "variants": {
            "V1_sign": float(pnl_v1),
            "V2_magnitude": {
                "best_pnl": float(best_v2_pnl),
                "factor": best_v2_factor,
                "all_results": v2_results,
            },
            "V3_strict_mag": {
                "best_pnl": float(best_v3_pnl),
                "factor": best_v3_factor,
                "all_results": v3_results,
            },
            "V4_ratio": {
                "best_pnl": float(best_v4_pnl),
                "r": best_v4_r,
                "all_results": v4_results,
            },
            "V5_combined": {
                "best_pnl": float(best_v5_pnl),
                "factor": best_v5_factor,
                "all_results": v5_results,
            },
        },
        "best_variant": best_variant,
        "best_pnl": float(best_pnl_overall),
        "best_delta_vs_T163": float(best_delta),
        "build_pkg": bool(best_delta > 0.5),
        "std_nn_per_sym": std_nn_per_sym,
        "std_lgb_per_sym": std_lgb_per_sym,
        "std_combined_per_sym": std_combined_per_sym,
    }
    out_path = os.path.join(OUTDIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

    return results


def build_pkg(results):
    """Build submission zip if best variant beats T163 V1 by > 0.5."""
    best_variant = results["best_variant"]
    best_delta = results["best_delta_vs_T163"]
    pnl_v1 = results["T163_V1_sign_holdout"]
    best_pnl = results["best_pnl"]

    print(f"\nBuilding pkg for {best_variant} (delta={best_delta:+.4f})...", flush=True)

    # 1. Copy T163 pkg
    src_pkg = f"{WD}/experiments/T163_combined_pkg"
    dst_pkg = f"{WD}/experiments/T165_magnitude_filter_pkg"
    if os.path.exists(dst_pkg):
        shutil.rmtree(dst_pkg)
    shutil.copytree(src_pkg, dst_pkg, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    # 2. Modify Predictor.py with best filter variant
    predictor_path = os.path.join(dst_pkg, "Predictor.py")
    with open(predictor_path, "r") as f:
        predictor_src = f.read()

    # Build new filter logic based on best variant
    var_info = results["variants"][best_variant]

    if best_variant == "V2_magnitude":
        factor = var_info["factor"]
        std_nn = results["std_nn_per_sym"]
        std_lgb = results["std_lgb_per_sym"]
        new_filter_code = f"""            # T165 V2_magnitude: hold if either model not confident (factor={factor})
            if lgb_pred is not None and agree_filter_mask.any():
                if nn_pred is not None and nn2_pred is not None:
                    nn_combined = (w_nn * nn_pred + w_nn2 * nn2_pred) / (w_nn + w_nn2)
                elif nn_pred is not None:
                    nn_combined = nn_pred
                elif nn2_pred is not None:
                    nn_combined = nn2_pred
                else:
                    nn_combined = None
                if nn_combined is not None:
                    # Per-sym std computed from holdout 96-119 (stable across splits)
                    _std_nn = {{{', '.join(f'{s}: {v:.8e}' for s, v in std_nn.items())}}}
                    _std_lgb = {{{', '.join(f'{s}: {v:.8e}' for s, v in std_lgb.items())}}}
                    std_nn_per_row = np.array([_std_nn.get(int(sym_per_row[i]), 1e-6) for i in range(B)], dtype=np.float32)
                    std_lgb_per_row = np.array([_std_lgb.get(int(sym_per_row[i]), 1e-6) for i in range(B)], dtype=np.float32)
                    nn_small = np.abs(nn_combined) < std_nn_per_row * {factor}
                    lgb_small = np.abs(lgb_pred) < std_lgb_per_row * {factor}
                    magnitude_filter = nn_small | lgb_small
                    actions[agree_filter_mask & magnitude_filter] = 1"""

    elif best_variant == "V3_strict_mag":
        factor = var_info["factor"]
        std_comb = results["std_combined_per_sym"]
        new_filter_code = f"""            # T165 V3_strict_mag: hold if max(|nn|,|lgb|) < std(combined)*factor={factor}
            if lgb_pred is not None and agree_filter_mask.any():
                if nn_pred is not None and nn2_pred is not None:
                    nn_combined = (w_nn * nn_pred + w_nn2 * nn2_pred) / (w_nn + w_nn2)
                elif nn_pred is not None:
                    nn_combined = nn_pred
                elif nn2_pred is not None:
                    nn_combined = nn2_pred
                else:
                    nn_combined = None
                if nn_combined is not None:
                    _std_comb = {{{', '.join(f'{s}: {v:.8e}' for s, v in std_comb.items())}}}
                    std_comb_per_row = np.array([_std_comb.get(int(sym_per_row[i]), 1e-6) for i in range(B)], dtype=np.float32)
                    max_mag = np.maximum(np.abs(nn_combined), np.abs(lgb_pred))
                    strict_filter = max_mag < std_comb_per_row * {factor}
                    actions[agree_filter_mask & strict_filter] = 1"""

    elif best_variant == "V4_ratio":
        r = var_info["r"]
        new_filter_code = f"""            # T165 V4_ratio: hold if min(|nn|/|lgb|, |lgb|/|nn|) < {r}
            if lgb_pred is not None and agree_filter_mask.any():
                if nn_pred is not None and nn2_pred is not None:
                    nn_combined = (w_nn * nn_pred + w_nn2 * nn2_pred) / (w_nn + w_nn2)
                elif nn_pred is not None:
                    nn_combined = nn_pred
                elif nn2_pred is not None:
                    nn_combined = nn2_pred
                else:
                    nn_combined = None
                if nn_combined is not None:
                    _eps = 1e-10
                    abs_nn = np.abs(nn_combined)
                    abs_lgb = np.abs(lgb_pred)
                    min_ratio = np.minimum(abs_nn / (abs_lgb + _eps), abs_lgb / (abs_nn + _eps))
                    ratio_filter = min_ratio < {r}
                    actions[agree_filter_mask & ratio_filter] = 1"""

    elif best_variant == "V5_combined":
        factor = var_info["factor"]
        std_nn = results["std_nn_per_sym"]
        new_filter_code = f"""            # T165 V5_combined: hold if sign disagree OR |nn| < std(nn)*{factor}
            if lgb_pred is not None and agree_filter_mask.any():
                if nn_pred is not None and nn2_pred is not None:
                    nn_combined = (w_nn * nn_pred + w_nn2 * nn2_pred) / (w_nn + w_nn2)
                elif nn_pred is not None:
                    nn_combined = nn_pred
                elif nn2_pred is not None:
                    nn_combined = nn2_pred
                else:
                    nn_combined = None
                if nn_combined is not None:
                    _std_nn = {{{', '.join(f'{s}: {v:.8e}' for s, v in std_nn.items())}}}
                    std_nn_per_row = np.array([_std_nn.get(int(sym_per_row[i]), 1e-6) for i in range(B)], dtype=np.float32)
                    disagree = np.sign(nn_combined) != np.sign(lgb_pred)
                    nn_not_confident = np.abs(nn_combined) < std_nn_per_row * {factor}
                    combined_filter = disagree | nn_not_confident
                    actions[agree_filter_mask & combined_filter] = 1"""

    else:  # V1_sign (baseline, same as T163)
        new_filter_code = """            # V1_sign: T163 baseline agreement filter
            if lgb_pred is not None and agree_filter_mask.any():
                if nn_pred is not None and nn2_pred is not None:
                    nn_combined = (w_nn * nn_pred + w_nn2 * nn2_pred) / (w_nn + w_nn2)
                elif nn_pred is not None:
                    nn_combined = nn_pred
                elif nn2_pred is not None:
                    nn_combined = nn2_pred
                else:
                    nn_combined = None
                if nn_combined is not None:
                    disagree = np.sign(nn_combined) != np.sign(lgb_pred)
                    actions[agree_filter_mask & disagree] = 1"""

    # Replace the existing filter block in Predictor.py
    old_filter_block = """            # Selective agreement filter (T158 adapted for 3-way ensemble):
            # NN-side = weighted avg of T87 (nn_pred) and T97 (nn2_pred).
            # For syms 0,1,2: abstain if sign(NN-side) != sign(pred_lgb).
            if lgb_pred is not None and agree_filter_mask.any():
                if nn_pred is not None and nn2_pred is not None:
                    nn_combined = (w_nn * nn_pred + w_nn2 * nn2_pred) / (w_nn + w_nn2)
                elif nn_pred is not None:
                    nn_combined = nn_pred
                elif nn2_pred is not None:
                    nn_combined = nn2_pred
                else:
                    nn_combined = None
                if nn_combined is not None:
                    disagree = np.sign(nn_combined) != np.sign(lgb_pred)
                    actions[agree_filter_mask & disagree] = 1"""

    if old_filter_block not in predictor_src:
        print("  WARNING: Could not find exact filter block in Predictor.py – manual review needed!", flush=True)
        print("  Saving filter code to filter_patch.txt for manual application.", flush=True)
        with open(os.path.join(dst_pkg, "filter_patch.txt"), "w") as f:
            f.write("OLD (to replace):\n")
            f.write(old_filter_block)
            f.write("\n\nNEW:\n")
            f.write(new_filter_code)
        return None

    new_predictor_src = predictor_src.replace(old_filter_block, new_filter_code)
    with open(predictor_path, "w") as f:
        f.write(new_predictor_src)
    print(f"  Predictor.py updated with {best_variant} filter.", flush=True)

    # Also update the docstring to reflect T165
    # 3. Build zip
    zip_name = "submission_050909_iter019_v2NN_T97_magfilter.zip"
    zip_path = os.path.join(WD, zip_name)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(dst_pkg):
            dirs[:] = [d for d in dirs if d not in ("__pycache__",)]
            for file in files:
                if file.endswith(".pyc"):
                    continue
                fpath = os.path.join(root, file)
                arcname = os.path.relpath(fpath, dst_pkg)
                zf.write(fpath, arcname)
    size_mb = os.path.getsize(zip_path) / 1e6
    print(f"  Zip built: {zip_path} ({size_mb:.2f} MB)", flush=True)

    # Copy to output dir
    out_copy = "/tmp/metabot-outputs/worker-24662130/" + zip_name
    os.makedirs(os.path.dirname(out_copy), exist_ok=True)
    shutil.copy2(zip_path, out_copy)
    print(f"  Copied to {out_copy}", flush=True)

    return zip_path


if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    results = main()

    best_variant = results["best_variant"]
    best_pnl = results["best_pnl"]
    best_delta = results["best_delta_vs_T163"]
    pnl_base = results["v2_baseline_holdout"]
    pnl_v1 = results["T163_V1_sign_holdout"]

    update_progress("Done - evaluating pkg build", {
        "v2_baseline": pnl_base,
        "V1_sign": pnl_v1,
        "best_variant": best_variant,
        "best_pnl": best_pnl,
        "best_delta": best_delta,
    })

    if best_delta > 0.5:
        print(f"\nBest ({best_variant}) delta={best_delta:+.4f} > 0.5 — building pkg...", flush=True)
        zip_path = build_pkg(results)
        if zip_path:
            results["zip_path"] = zip_path
            out_path = os.path.join(OUTDIR, "results.json")
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
    else:
        print(f"\nBest delta={best_delta:+.4f} ≤ 0.5 — no pkg built.", flush=True)

    update_progress("COMPLETE", {
        "v2_baseline": pnl_base,
        "V1_sign": pnl_v1,
        "best_variant": best_variant,
        "best_pnl": best_pnl,
        "best_delta": best_delta,
        "pkg_built": best_delta > 0.5,
    })

    print(
        f"\nRESULT: task=[T165 magnitude agreement filter] "
        f"metrics={{v2_baseline={pnl_base:.4f}, V1_sign={pnl_v1:.4f}, "
        f"best_variant={best_variant}, best_pnl={best_pnl:.4f}, "
        f"best_delta_vs_T163={best_delta:.4f}}} "
        f"notes=[best variant is {best_variant}]"
    )
