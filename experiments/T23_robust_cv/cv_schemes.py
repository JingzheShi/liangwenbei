"""
T23: Robust cross-validation schemes + platform calibration.

Goal: Move beyond a single LOSO scalar to a richer estimate of platform PnL,
with confidence intervals and OOD-stress structure.

Schemes implemented (all reuse existing OOF predictions — NO retraining):
  A  Standard 5-fold LOSO         — sum across 5 held-out syms (current baseline)
  B  LO2SO (10 pair regrouping)   — group OOF by sym pair → 10 numbers, mean/std
  C  Time-rolling within test     — split date 96-119 into windows, per-window scores
  D  Cross sym × cross time       — 5 sym × 3 windows = 15 cells, two-way variance
  E  Bootstrap CI                 — resample (sym,date,session) with replacement → 95% CI

Calibration anchor: mmpc_demo on label_5/10/20/40/60 — local sums vs platform truth
(task 2340: label_5/10/20/40/60 = -9.079 / -13.319 / -6.649 / -16.065 / -23.132).

Outputs:
  per_cell_pnl.parquet   — long-form (model, horizon, sym, date, session, cum_pnl)
  scheme_results.json    — per-scheme per-model summaries
  calibration.json       — gap analysis + platform predictions
  report.md              — human-readable report
  results.json           — final task results
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

# ----------------------------------------------------------------------------
# Constants & platform truth (mmpc_demo on platform task 2340)
# ----------------------------------------------------------------------------
PLATFORM_MMPC_DEMO = {
    "label_5": -9.079,
    "label_10": -13.319,
    "label_20": -6.649,
    "label_40": -16.065,
    "label_60": -23.132,
}
# Platform task 2435 (iter_002 CPU mode) — 2nd calibration anchor!
PLATFORM_ITER_002 = {
    "label_5": -7.68,
    "label_10": -8.64,
    "label_20": -5.09,
    "label_40": +2.02,
    "label_60": +4.07,
}
HORIZONS = [5, 10, 20, 40, 60]

# Best (T, delta) per horizon from T5b threshold sweep — applied to OOF probs
ITER_002_THRESHOLDS = {
    5:  (0.60, 0.10),
    10: (0.55, 0.10),
    20: (0.50, 0.05),
    40: (0.50, 0.00),
    60: (0.50, 0.20),
}
# iter_003 ensemble (T11) used same threshold for h10 (0.55, 0.10)
ITER_003_THRESHOLDS = {10: (0.55, 0.10)}

FEE_RATE = 0.0001


# ----------------------------------------------------------------------------
# PnL utilities
# ----------------------------------------------------------------------------
def thresholded_pred(probs: np.ndarray, T: float, delta: float) -> np.ndarray:
    """Apply asymmetric threshold to OOF probs → 0/1/2 prediction."""
    p0 = probs[:, 0]; p1 = probs[:, 1]; p2 = probs[:, 2]
    side_max = np.maximum(p0, p2)
    take = (side_max >= T) & (side_max > p1 + delta)
    pred_side = np.where(p2 > p0, 2, 0)
    return np.where(take, pred_side, 1).astype(np.int8)


def per_row_pnl(pred: np.ndarray, mp_t: np.ndarray, mp_th: np.ndarray) -> np.ndarray:
    """Vectorized per-row PnL — same formula as src/eval/pnl.py."""
    side = pred.astype(np.float64) - 1.0
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee = FEE_RATE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee) / denom


# ----------------------------------------------------------------------------
# Per-(sym, date, session) cum_pnl loaders
# ----------------------------------------------------------------------------
def load_iter002_per_cell() -> pd.DataFrame:
    """iter_002 (T5b) — load all 5 horizons × 5 folds, apply per-horizon threshold,
    aggregate per (sym, date, session). Returns long-form DataFrame.
    """
    rows = []
    for h_idx, H in enumerate(HORIZONS):
        T, delta = ITER_002_THRESHOLDS[H]
        for k in range(5):
            p = os.path.join(
                ROOT, f"experiments/T5b_features_multihorizon/loso_pred_h{H}_held{k}.parquet"
            )
            df = pd.read_parquet(p)
            probs = df[["prob_0", "prob_1", "prob_2"]].to_numpy(np.float32)
            pred = thresholded_pred(probs, T, delta)
            mp_t = df["midprice_t"].to_numpy(np.float64)
            mp_th = df["midprice_th"].to_numpy(np.float64)
            df = df.assign(pnl=per_row_pnl(pred, mp_t, mp_th), pred_th=pred)
            grp = df.groupby(["sym", "date", "session"], observed=True).agg(
                cum_pnl=("pnl", "sum"),
                n_total=("pnl", "size"),
                n_active=("pred_th", lambda s: int((s != 1).sum())),
            ).reset_index()
            grp["model"] = "iter_002"
            grp["horizon"] = f"label_{H}"
            grp["fold_held_sym"] = k
            grp["T"] = T
            grp["delta"] = delta
            rows.append(grp)
    return pd.concat(rows, ignore_index=True)


def load_iter003_per_cell(horizon: int = 10) -> pd.DataFrame:
    """iter_003 (T11 ensemble of 5 seeds) — only h_10 ensemble OOF available."""
    T, delta = ITER_003_THRESHOLDS[horizon]
    rows = []
    for k in range(5):
        p = os.path.join(
            ROOT, f"experiments/T11_schemeC_multiseed/loso_pred_ensemble_h{horizon}_held{k}.parquet"
        )
        df = pd.read_parquet(p)
        probs = df[["prob_0", "prob_1", "prob_2"]].to_numpy(np.float32)
        pred = thresholded_pred(probs, T, delta)
        mp_t = df["midprice_t"].to_numpy(np.float64)
        mp_th = df["midprice_th"].to_numpy(np.float64)
        df = df.assign(pnl=per_row_pnl(pred, mp_t, mp_th), pred_th=pred)
        grp = df.groupby(["sym", "date", "session"], observed=True).agg(
            cum_pnl=("pnl", "sum"),
            n_total=("pnl", "size"),
            n_active=("pred_th", lambda s: int((s != 1).sum())),
        ).reset_index()
        grp["model"] = "iter_003"
        grp["horizon"] = f"label_{horizon}"
        grp["fold_held_sym"] = k
        grp["T"] = T
        grp["delta"] = delta
        rows.append(grp)
    return pd.concat(rows, ignore_index=True)


def load_mmpc_per_cell() -> pd.DataFrame:
    """mmpc_demo per-session cum_pnl — from audit_pipeline_per_session.json."""
    p = os.path.join(ROOT, "experiments/audit_pipeline_per_session.json")
    with open(p) as f:
        d = json.load(f)
    keys = d["session_keys"]  # list of [sym, date, session]
    rows = []
    for h, h_name in [(5, "label_5"), (10, "label_10"), (20, "label_20"), (40, "label_40"), (60, "label_60")]:
        pnls = d["per_session_pnl"][h_name]
        accs = d["per_session_acc"][h_name]
        flats = d["per_session_predflat"][h_name]
        for i, (sym, date, sess) in enumerate(keys):
            n_total = 1842
            n_active = int(round((1 - flats[i]) * n_total))
            rows.append({
                "sym": int(sym),
                "date": int(date),
                "session": sess,
                "cum_pnl": float(pnls[i]),
                "n_total": n_total,
                "n_active": n_active,
                "model": "mmpc_demo",
                "horizon": h_name,
                "fold_held_sym": -1,  # no LOSO — mmpc_demo is pretrained
                "T": np.nan,
                "delta": np.nan,
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Scheme implementations — all operate on per-cell cum_pnl
# ----------------------------------------------------------------------------
def scheme_A_loso(df: pd.DataFrame) -> dict:
    """Standard 5-fold LOSO: sum cum_pnl per held-out sym, sum across 5 syms."""
    by_sym = df.groupby("sym")["cum_pnl"].sum()
    return {
        "scheme": "A_LOSO_standard",
        "n_folds": int(by_sym.size),
        "per_fold": {int(k): float(v) for k, v in by_sym.items()},
        "sum": float(by_sym.sum()),
        "mean_per_fold": float(by_sym.mean()),
        "std_per_fold": float(by_sym.std(ddof=1)) if len(by_sym) > 1 else 0.0,
        "min_fold": float(by_sym.min()),
        "max_fold": float(by_sym.max()),
    }


def scheme_B_lo2so(df: pd.DataFrame) -> dict:
    """LO2SO 10-pair regrouping. Sum over 2-sym subset; sum_pair_i = sum(sym_a) + sum(sym_b).
    Total scheme A LOSO sum = (1/4) * sum(pair sums) since each sym appears in 4 pairs."""
    from itertools import combinations
    by_sym = df.groupby("sym")["cum_pnl"].sum()
    syms = sorted(by_sym.index.tolist())
    pair_sums = []
    pair_keys = []
    for a, b in combinations(syms, 2):
        pair_sums.append(by_sym[a] + by_sym[b])
        pair_keys.append((int(a), int(b)))
    arr = np.array(pair_sums)
    return {
        "scheme": "B_LO2SO_pairs",
        "n_pairs": len(arr),
        "per_pair": {f"({a},{b})": float(s) for (a, b), s in zip(pair_keys, pair_sums)},
        "mean_pair_sum": float(arr.mean()),
        "std_pair_sum": float(arr.std(ddof=1)),
        "min_pair_sum": float(arr.min()),
        "max_pair_sum": float(arr.max()),
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        # Implied LOSO 5-fold sum = sum of all pairs / 4 (each sym in C(4,1)=4 pairs)
        "implied_5fold_sum": float(arr.sum() / 4.0),
    }


def scheme_C_time_rolling(df: pd.DataFrame, n_windows: int = 3) -> dict:
    """Time-rolling: split test dates [96, 119] into n_windows; sum cum_pnl per window."""
    dates = sorted(df["date"].unique())
    L = len(dates)
    edges = np.linspace(0, L, n_windows + 1).astype(int)
    windows = []
    for i in range(n_windows):
        d_lo = dates[edges[i]]
        d_hi = dates[edges[i + 1] - 1]
        mask = (df["date"] >= d_lo) & (df["date"] <= d_hi)
        sub = df.loc[mask]
        windows.append({
            "window_idx": i,
            "date_lo": int(d_lo),
            "date_hi": int(d_hi),
            "n_dates": int(d_hi - d_lo + 1),
            "cum_pnl_sum": float(sub["cum_pnl"].sum()),
            "n_sessions": int(len(sub)),
        })
    sums = np.array([w["cum_pnl_sum"] for w in windows])
    return {
        "scheme": "C_time_rolling",
        "n_windows": n_windows,
        "windows": windows,
        "mean_window_sum": float(sums.mean()),
        "std_window_sum": float(sums.std(ddof=1)),
        "min_window_sum": float(sums.min()),
        "max_window_sum": float(sums.max()),
        "total_sum": float(sums.sum()),
    }


def scheme_D_cross_sym_time(df: pd.DataFrame, n_windows: int = 3) -> dict:
    """Cross sym × cross time: 5 syms × n_windows = cells. Per-cell sum + 2-way variance."""
    dates = sorted(df["date"].unique())
    L = len(dates)
    edges = np.linspace(0, L, n_windows + 1).astype(int)
    win_bounds = [(dates[edges[i]], dates[edges[i + 1] - 1]) for i in range(n_windows)]
    syms = sorted(df["sym"].unique())
    cells = {}
    matrix = np.zeros((len(syms), n_windows))
    for si, s in enumerate(syms):
        for wi, (d_lo, d_hi) in enumerate(win_bounds):
            mask = (df["sym"] == s) & (df["date"] >= d_lo) & (df["date"] <= d_hi)
            v = float(df.loc[mask, "cum_pnl"].sum())
            cells[f"sym{int(s)}_w{wi}"] = v
            matrix[si, wi] = v
    return {
        "scheme": "D_cross_sym_time",
        "n_syms": len(syms),
        "n_windows": n_windows,
        "win_bounds": [(int(a), int(b)) for a, b in win_bounds],
        "cells": cells,
        "row_sums_by_sym": {int(s): float(matrix[si].sum()) for si, s in enumerate(syms)},
        "col_sums_by_window": {wi: float(matrix[:, wi].sum()) for wi in range(n_windows)},
        "mean_cell": float(matrix.mean()),
        "std_cell": float(matrix.std(ddof=1)),
        "p5_cell": float(np.percentile(matrix, 5)),
        "p95_cell": float(np.percentile(matrix, 95)),
        "total_sum": float(matrix.sum()),
        "sym_variance": float(matrix.sum(axis=1).std(ddof=1)),    # variance across sym (sum over time)
        "time_variance": float(matrix.sum(axis=0).std(ddof=1)),  # variance across time (sum over sym)
    }


def scheme_E_bootstrap(
    df: pd.DataFrame, n_boot: int = 2000, seed: int = 42
) -> dict:
    """Bootstrap CI on the 240 per-session cum_pnl values.
    Resample sessions WITH replacement; compute sum each time → empirical CI of platform-equivalent total."""
    pnls = df["cum_pnl"].to_numpy(np.float64)
    rng = np.random.default_rng(seed)
    n = len(pnls)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_sums = pnls[idx].sum(axis=1)
    return {
        "scheme": "E_bootstrap",
        "n_sessions": int(n),
        "n_boot": n_boot,
        "observed_sum": float(pnls.sum()),
        "mean": float(boot_sums.mean()),
        "std": float(boot_sums.std(ddof=1)),
        "ci95_lo": float(np.percentile(boot_sums, 2.5)),
        "ci95_hi": float(np.percentile(boot_sums, 97.5)),
        "ci80_lo": float(np.percentile(boot_sums, 10)),
        "ci80_hi": float(np.percentile(boot_sums, 90)),
        "p_negative": float((boot_sums < 0).mean()),
        # cluster bootstrap: resample SYMS with replacement (more conservative — captures sym-level corr)
        "cluster_sym": _cluster_bootstrap(df, by="sym", n_boot=n_boot, rng=np.random.default_rng(seed + 1)),
        "cluster_date": _cluster_bootstrap(df, by="date", n_boot=n_boot, rng=np.random.default_rng(seed + 2)),
    }


def _cluster_bootstrap(df: pd.DataFrame, by: str, n_boot: int, rng) -> dict:
    """Resample at the cluster level (sym or date) → preserves intra-cluster correlation."""
    groups = df.groupby(by)["cum_pnl"].sum().to_dict()
    keys = list(groups.keys())
    vals = np.array([groups[k] for k in keys], dtype=np.float64)
    n = len(keys)
    idx = rng.integers(0, n, size=(n_boot, n))
    # Each bootstrap sample: sum of n resampled cluster sums (with replacement)
    boot_sums = vals[idx].sum(axis=1)
    return {
        "n_clusters": int(n),
        "observed": float(vals.sum()),
        "mean": float(boot_sums.mean()),
        "std": float(boot_sums.std(ddof=1)),
        "ci95_lo": float(np.percentile(boot_sums, 2.5)),
        "ci95_hi": float(np.percentile(boot_sums, 97.5)),
        "p_negative": float((boot_sums < 0).mean()),
    }


# ----------------------------------------------------------------------------
# Run all schemes for all (model, horizon)
# ----------------------------------------------------------------------------
def run_all_schemes(per_cell: pd.DataFrame) -> dict:
    """Group per_cell by (model, horizon) and apply all 5 schemes."""
    out = {}
    for (model, horizon), sub in per_cell.groupby(["model", "horizon"]):
        if len(sub) == 0:
            continue
        sub = sub.copy()
        out[f"{model}__{horizon}"] = {
            "model": model,
            "horizon": horizon,
            "n_cells": int(len(sub)),
            "A": scheme_A_loso(sub),
            "B": scheme_B_lo2so(sub),
            "C": scheme_C_time_rolling(sub, n_windows=3),
            "D": scheme_D_cross_sym_time(sub, n_windows=3),
            "E": scheme_E_bootstrap(sub),
        }
    return out


# ----------------------------------------------------------------------------
# Calibration: gap analysis
# ----------------------------------------------------------------------------
def calibrate_two_anchors(scheme_results: dict) -> dict:
    """2-anchor per-horizon linear regression: platform = a*local + b.
    Anchors: mmpc_demo and iter_002 (both with known platform results).

    With exactly 2 points, the fit is exact (no residual).
    Slope `a` ∈ [0, 1] would mean local roughly tracks platform with a haircut.
    Slope > 1 means small local gains amplify on platform (genuine alpha).
    Slope < 0 would mean opposite-sign — none observed.
    """
    cal = {}
    for h_name in PLATFORM_MMPC_DEMO:
        m_local = scheme_results[f"mmpc_demo__{h_name}"]["A"]["sum"]
        m_plat = PLATFORM_MMPC_DEMO[h_name]
        i_local = scheme_results[f"iter_002__{h_name}"]["A"]["sum"]
        i_plat = PLATFORM_ITER_002[h_name]
        # Linear fit: platform = a*local + b (exact for 2 points)
        denom = i_local - m_local
        if abs(denom) < 1e-9:
            slope = 0.0
            intercept = m_plat
        else:
            slope = (i_plat - m_plat) / denom
            intercept = m_plat - slope * m_local
        cal[h_name] = {
            "anchor_mmpc_local": m_local,
            "anchor_mmpc_platform": m_plat,
            "anchor_iter002_local": i_local,
            "anchor_iter002_platform": i_plat,
            "slope_a": slope,
            "intercept_b": intercept,
            "pred_at_local_zero": intercept,  # platform when local PnL = 0
            "implied_breakeven_local": -intercept / slope if abs(slope) > 1e-9 else float("inf"),
        }
    return cal


def predict_with_two_anchor(
    scheme_results: dict, cal_2anchor: dict, model: str, horizon: str
) -> dict:
    """Predict platform score using 2-anchor linear fit + bootstrap CI on local."""
    key = f"{model}__{horizon}"
    if key not in scheme_results:
        return {"error": f"no {key}"}
    if horizon not in cal_2anchor:
        return {"error": f"no calibration for {horizon}"}
    s = scheme_results[key]
    c = cal_2anchor[horizon]
    a, b = c["slope_a"], c["intercept_b"]
    local_A = s["A"]["sum"]
    plat_mean = a * local_A + b
    # CI from local bootstrap; we ignore slope/intercept uncertainty (degenerate with 2 points)
    e = s["E"]
    plat_std_session = abs(a) * e["std"]
    plat_std_clu_sym = abs(a) * e["cluster_sym"]["std"]
    return {
        "model": model,
        "horizon": horizon,
        "local_A": local_A,
        "slope_a": a,
        "intercept_b": b,
        "platform_mean": plat_mean,
        "platform_std_session": plat_std_session,
        "platform_std_cluster_sym": plat_std_clu_sym,
        "platform_ci95_session": [plat_mean - 1.96 * plat_std_session,
                                  plat_mean + 1.96 * plat_std_session],
        "platform_ci95_cluster_sym": [plat_mean - 1.96 * plat_std_clu_sym,
                                      plat_mean + 1.96 * plat_std_clu_sym],
    }


def calibrate_with_mmpc_demo(scheme_results: dict, n_boot: int = 2000) -> dict:
    """Compute platform calibration anchor from mmpc_demo.

    For each horizon, we have:
      - mmpc local sum (Scheme A) and bootstrap CI (Scheme E)
      - mmpc platform truth (PLATFORM_MMPC_DEMO[horizon])

    The "gap" = local - platform.
    Models on the SAME local test set are likely to have the same gap (additive shift assumption).
    We also report the multiplicative ratio (more conservative).

    Returns calibration data per horizon and recommended platform predictor.
    """
    cal = {}
    for h_name in PLATFORM_MMPC_DEMO:
        key = f"mmpc_demo__{h_name}"
        if key not in scheme_results:
            continue
        local = scheme_results[key]["A"]["sum"]
        platform = PLATFORM_MMPC_DEMO[h_name]
        gap = local - platform  # additive: platform = local - gap
        ratio = platform / local if abs(local) > 1e-6 else float("inf")

        # CI of the gap: from bootstrap on local sum (treating platform as known constant)
        e = scheme_results[key]["E"]
        gap_ci = (local - platform - 1.96 * e["std"], local - platform + 1.96 * e["std"])

        cal[h_name] = {
            "mmpc_local_sum_A": local,
            "mmpc_local_ci95_E": [e["ci95_lo"], e["ci95_hi"]],
            "mmpc_local_std_E": e["std"],
            "mmpc_platform_truth": platform,
            "additive_gap": gap,
            "additive_gap_ci95": list(gap_ci),
            "multiplicative_ratio": ratio,
            "implied_platform_at_local_zero_additive": -gap,  # if local=0, platform = -gap
        }
    return cal


def predict_platform_score(
    scheme_results: dict, calibration: dict, model: str, horizon: str
) -> dict:
    """Given a model's local schemes and calibration, predict platform score with CI."""
    key = f"{model}__{horizon}"
    if key not in scheme_results:
        return {"error": f"no scheme results for {key}"}
    if horizon not in calibration:
        return {"error": f"no calibration for {horizon}"}

    s = scheme_results[key]
    c = calibration[horizon]
    local_A = s["A"]["sum"]
    local_E = s["E"]
    local_E_sym = s["E"]["cluster_sym"]
    local_E_date = s["E"]["cluster_date"]
    gap = c["additive_gap"]
    ratio = c["multiplicative_ratio"]
    mmpc_local_std = c["mmpc_local_std_E"]

    # Method 1: additive gap (assumes constant local→platform shift)
    # platform = local - gap
    # variance of platform ≈ var(local_bootstrap) + var(gap_estimate)
    # gap_estimate has variance = var(mmpc_local_sum) since platform is single point
    plat_add_mean = local_A - gap
    # var combination: (model_local_std)^2 + (mmpc_local_std)^2 (independent test draws)
    plat_add_std = float(np.sqrt(local_E["std"] ** 2 + mmpc_local_std ** 2))
    plat_add_lo = plat_add_mean - 1.96 * plat_add_std
    plat_add_hi = plat_add_mean + 1.96 * plat_add_std

    # Method 2: multiplicative ratio (assumes platform = ratio * local, ratio fixed)
    plat_mul_mean = ratio * local_A
    # var(platform) = ratio^2 * var(local) (ignoring ratio uncertainty for simplicity)
    plat_mul_std = abs(ratio) * local_E["std"]
    plat_mul_lo = plat_mul_mean - 1.96 * plat_mul_std
    plat_mul_hi = plat_mul_mean + 1.96 * plat_mul_std

    # Method 3: cluster-sym bootstrap (most conservative — captures OOD-sym risk)
    # plat_cluster_sym uses sym-level resampling for local
    plat_cluster_sym_mean = local_E_sym["observed"] - gap
    plat_cluster_sym_std = float(np.sqrt(local_E_sym["std"] ** 2 + mmpc_local_std ** 2))
    plat_cluster_sym_lo = plat_cluster_sym_mean - 1.96 * plat_cluster_sym_std
    plat_cluster_sym_hi = plat_cluster_sym_mean + 1.96 * plat_cluster_sym_std

    return {
        "model": model,
        "horizon": horizon,
        "local_loso_A_sum": local_A,
        "local_E_ci95": [local_E["ci95_lo"], local_E["ci95_hi"]],
        "local_cluster_sym_ci95": [local_E_sym["ci95_lo"], local_E_sym["ci95_hi"]],
        "calibration_gap_used": gap,
        "calibration_ratio_used": ratio,
        "platform_predicted": {
            "additive_gap": {
                "mean": plat_add_mean,
                "std": plat_add_std,
                "ci95_lo": plat_add_lo,
                "ci95_hi": plat_add_hi,
            },
            "multiplicative": {
                "mean": plat_mul_mean,
                "std": plat_mul_std,
                "ci95_lo": plat_mul_lo,
                "ci95_hi": plat_mul_hi,
            },
            "cluster_sym_bootstrap": {
                "mean": plat_cluster_sym_mean,
                "std": plat_cluster_sym_std,
                "ci95_lo": plat_cluster_sym_lo,
                "ci95_hi": plat_cluster_sym_hi,
            },
        },
    }


# ----------------------------------------------------------------------------
# Max-over-horizons prediction (platform takes max of 5 horizon scores)
# ----------------------------------------------------------------------------
def predict_max_over_horizons(
    per_cell: pd.DataFrame, calibration: dict, model: str, n_boot: int = 2000, seed: int = 42,
    use_pooled_gap: bool = False,
) -> dict:
    """Paired bootstrap: resample (sym, date, session) keys ONCE, then compute all 5 horizon
    sums simultaneously. For each horizon apply gap to get platform; take max.

    This captures the correlated test-draw across horizons (they share data).
    """
    sub = per_cell[per_cell["model"] == model].copy()
    if len(sub) == 0:
        return {"error": f"no per_cell rows for {model}"}

    # Pivot: rows = (sym, date, session), cols = horizon
    pivot = sub.pivot_table(
        index=["sym", "date", "session"], columns="horizon", values="cum_pnl",
        aggfunc="sum",
    ).fillna(0.0)
    horizons_present = list(pivot.columns)
    if not horizons_present:
        return {"error": f"no horizons for {model}"}

    keys = pivot.index.tolist()
    arr = pivot.to_numpy(np.float64)  # shape (n_cells, n_horizons)

    # Gaps per horizon (from calibration). If horizon missing, use 0.
    gaps = np.array([
        calibration[h]["additive_gap"] if h in calibration else 0.0
        for h in horizons_present
    ])
    pooled_gap = float(np.mean([calibration[h]["additive_gap"] for h in calibration]))
    if use_pooled_gap:
        gaps = np.full(len(horizons_present), pooled_gap)

    # Observed (no bootstrap)
    obs_local = arr.sum(axis=0)  # per horizon
    obs_plat = obs_local - gaps  # per horizon
    obs_max = float(np.max(obs_plat))
    obs_argmax = horizons_present[int(np.argmax(obs_plat))]

    rng = np.random.default_rng(seed)
    n = len(keys)
    boot_sums = np.zeros((n_boot, len(horizons_present)))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_sums[b] = arr[idx].sum(axis=0)
    plat_boot = boot_sums - gaps  # broadcast
    max_per_boot = plat_boot.max(axis=1)
    argmax_per_boot = plat_boot.argmax(axis=1)

    # Cluster-sym bootstrap on (sym) for max-over-horizons
    sym_groups = sub.groupby(["sym", "horizon"])["cum_pnl"].sum().unstack("horizon").fillna(0.0)
    syms = sym_groups.index.tolist()
    sym_arr = sym_groups[horizons_present].to_numpy(np.float64)  # shape (5, n_horizons)
    n_sym = len(syms)
    rng2 = np.random.default_rng(seed + 100)
    cluster_max = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng2.integers(0, n_sym, size=n_sym)
        s = sym_arr[idx].sum(axis=0)
        s_plat = s - gaps
        cluster_max[b] = s_plat.max()

    return {
        "model": model,
        "horizons_present": horizons_present,
        "gaps_used": dict(zip(horizons_present, gaps.tolist())),
        "pooled_gap_alternative": pooled_gap,
        "use_pooled_gap": use_pooled_gap,
        "observed_local_per_horizon": dict(zip(horizons_present, obs_local.tolist())),
        "observed_platform_per_horizon": dict(zip(horizons_present, obs_plat.tolist())),
        "observed_max_platform": obs_max,
        "observed_argmax_horizon": obs_argmax,
        "session_bootstrap": {
            "max_mean": float(max_per_boot.mean()),
            "max_std": float(max_per_boot.std(ddof=1)),
            "max_ci95_lo": float(np.percentile(max_per_boot, 2.5)),
            "max_ci95_hi": float(np.percentile(max_per_boot, 97.5)),
            "max_ci80_lo": float(np.percentile(max_per_boot, 10)),
            "max_ci80_hi": float(np.percentile(max_per_boot, 90)),
            "p_negative_max": float((max_per_boot < 0).mean()),
            "argmax_distribution": {
                horizons_present[i]: float((argmax_per_boot == i).mean())
                for i in range(len(horizons_present))
            },
        },
        "cluster_sym_bootstrap": {
            "max_mean": float(cluster_max.mean()),
            "max_std": float(cluster_max.std(ddof=1)),
            "max_ci95_lo": float(np.percentile(cluster_max, 2.5)),
            "max_ci95_hi": float(np.percentile(cluster_max, 97.5)),
            "max_ci80_lo": float(np.percentile(cluster_max, 10)),
            "max_ci80_hi": float(np.percentile(cluster_max, 90)),
            "p_negative_max": float((cluster_max < 0).mean()),
        },
    }


# ----------------------------------------------------------------------------
# Stress test (optional)
# ----------------------------------------------------------------------------
def stress_test_iter002_h10(noise_sigmas=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0), seed: int = 42) -> dict:
    """Add Gaussian noise to OOF probabilities (then re-normalize), apply threshold,
    measure cum_pnl decay. Less decay = more robust → safer for platform OOD."""
    T, delta = ITER_002_THRESHOLDS[10]
    rng = np.random.default_rng(seed)
    out = []
    for sigma in noise_sigmas:
        total_pnl = 0.0
        n_active = 0
        for k in range(5):
            p = os.path.join(
                ROOT, f"experiments/T5b_features_multihorizon/loso_pred_h10_held{k}.parquet"
            )
            df = pd.read_parquet(p)
            probs = df[["prob_0", "prob_1", "prob_2"]].to_numpy(np.float32).astype(np.float64)
            if sigma > 0:
                # additive Gaussian noise on logits would be cleaner, but on probs is simpler;
                # we then renormalize and clip to [eps, 1-eps]
                noise = rng.normal(0, sigma, size=probs.shape)
                probs = probs + noise
                probs = np.clip(probs, 1e-4, None)
                probs = probs / probs.sum(axis=1, keepdims=True)
            pred = thresholded_pred(probs.astype(np.float32), T, delta)
            mp_t = df["midprice_t"].to_numpy(np.float64)
            mp_th = df["midprice_th"].to_numpy(np.float64)
            pnl = per_row_pnl(pred, mp_t, mp_th)
            total_pnl += float(pnl.sum())
            n_active += int((pred != 1).sum())
        out.append({"sigma": float(sigma), "sum_cum_pnl": total_pnl, "n_active": n_active})
    return {"noise_sigmas": list(noise_sigmas), "results": out}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--no-stress", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 70)
    print("T23: Robust CV + platform calibration")
    print("=" * 70)

    # 1. Load per-(sym, date, session) cum_pnl for all 3 models
    print("\n[1/4] Loading per-cell cum_pnl tables ...")
    df_iter002 = load_iter002_per_cell()
    df_iter003 = load_iter003_per_cell(horizon=10)
    df_mmpc = load_mmpc_per_cell()
    per_cell = pd.concat([df_iter002, df_iter003, df_mmpc], ignore_index=True)
    out_per_cell = os.path.join(HERE, "per_cell_pnl.parquet")
    per_cell.to_parquet(out_per_cell, index=False)
    print(f"  saved per-cell table: {len(per_cell):,} rows  → {out_per_cell}")

    # Sanity check Scheme A LOSO sums
    print("\n  Scheme A sanity check (sum cum_pnl per (model, horizon)):")
    for (m, h), sub in per_cell.groupby(["model", "horizon"]):
        s = sub["cum_pnl"].sum()
        print(f"    {m:<12s}  {h:<10s}: sum = {s:+.4f}  ({len(sub)} cells)")

    # 2. Run all schemes
    print(f"\n[2/4] Running all schemes (n_boot={args.n_boot}) ...")
    scheme_results = run_all_schemes(per_cell)
    out_schemes = os.path.join(HERE, "scheme_results.json")
    # We embed bootstrap n_boot
    for k in scheme_results:
        scheme_results[k]["E"]["n_boot_used"] = args.n_boot
    with open(out_schemes, "w") as f:
        json.dump(scheme_results, f, indent=2, default=float)
    print(f"  saved scheme results → {out_schemes}")

    # 3. Calibration
    print("\n[3/4] Calibrating with mmpc_demo platform truth ...")
    calibration = calibrate_with_mmpc_demo(scheme_results, n_boot=args.n_boot)
    print(f"  per-horizon calibration:")
    print(f"  {'horizon':<10s}  {'mmpc_local':>10s}  {'mmpc_plat':>10s}  {'gap':>9s}  {'ratio':>7s}")
    for h, c in calibration.items():
        print(f"  {h:<10s}  {c['mmpc_local_sum_A']:>+10.4f}  {c['mmpc_platform_truth']:>+10.4f}  "
              f"{c['additive_gap']:>+9.4f}  {c['multiplicative_ratio']:>+7.3f}")

    # 4. Platform predictions for iter_002 (5 horizons) and iter_003 (h_10)
    print("\n[4/4] Predicting iter_002 / iter_003 platform scores ...")
    predictions = {}
    for m in ["iter_002", "iter_003"]:
        for h_name in PLATFORM_MMPC_DEMO:
            key = f"{m}__{h_name}"
            if key in scheme_results:
                predictions[key] = predict_platform_score(
                    scheme_results, calibration, m, h_name
                )

    # Pretty print
    print(f"\n  {'model':<10s}  {'horizon':<10s}  {'local_A':>9s}  "
          f"{'plat_add':>9s}  {'add_lo':>7s}  {'add_hi':>7s}  "
          f"{'plat_mul':>9s}  {'plat_clu':>9s}  {'clu_lo':>7s}  {'clu_hi':>7s}")
    for key, p in predictions.items():
        if "error" in p:
            continue
        a = p["platform_predicted"]["additive_gap"]
        u = p["platform_predicted"]["multiplicative"]
        s = p["platform_predicted"]["cluster_sym_bootstrap"]
        print(f"  {p['model']:<10s}  {p['horizon']:<10s}  {p['local_loso_A_sum']:>+9.3f}  "
              f"{a['mean']:>+9.3f}  {a['ci95_lo']:>+7.2f}  {a['ci95_hi']:>+7.2f}  "
              f"{u['mean']:>+9.3f}  {s['mean']:>+9.3f}  {s['ci95_lo']:>+7.2f}  {s['ci95_hi']:>+7.2f}")

    # 2-anchor linear regression calibration (using both mmpc_demo and iter_002 platform truth)
    print("\n[3b/4] Two-anchor linear calibration (mmpc_demo + iter_002) ...")
    cal_2anchor = calibrate_two_anchors(scheme_results)
    print(f"  per-horizon: platform = a*local + b")
    print(f"  {'horizon':<10s}  {'a (slope)':>10s}  {'b (intercept)':>14s}  "
          f"{'pred@0':>9s}  {'breakeven_local':>16s}")
    for h_name, c in cal_2anchor.items():
        be = c["implied_breakeven_local"]
        be_str = f"{be:>+16.3f}" if abs(be) < 1000 else f"{'inf':>16s}"
        print(f"  {h_name:<10s}  {c['slope_a']:>+10.4f}  {c['intercept_b']:>+14.4f}  "
              f"{c['pred_at_local_zero']:>+9.3f}  {be_str}")

    # Predict iter_003 with 2-anchor fit
    print("\n  iter_003 platform predictions (2-anchor fit):")
    iter003_2anchor = {}
    for h_name in PLATFORM_MMPC_DEMO:
        r = predict_with_two_anchor(scheme_results, cal_2anchor, "iter_003", h_name)
        if "error" not in r:
            iter003_2anchor[h_name] = r
            print(f"    {h_name}: local={r['local_A']:+.3f} → platform={r['platform_mean']:+.3f}  "
                  f"sess CI95={r['platform_ci95_session']}  clu-sym CI95={r['platform_ci95_cluster_sym']}")

    # Iter_002 reproduction check (should give back exact platform truth)
    print("\n  iter_002 reproduction check (2-anchor fit must hit platform truth exactly):")
    for h_name in PLATFORM_MMPC_DEMO:
        r = predict_with_two_anchor(scheme_results, cal_2anchor, "iter_002", h_name)
        if "error" not in r:
            truth = PLATFORM_ITER_002[h_name]
            err = r['platform_mean'] - truth
            print(f"    {h_name}: predicted={r['platform_mean']:+.4f}  truth={truth:+.4f}  err={err:+.6f}")

    # Max-over-horizons (platform takes max of 5 horizon scores in iter_002 submission)
    print("\n[bonus] Max-over-horizons platform prediction (paired bootstrap) ...")
    max_predictions = {}
    for m in ["iter_002", "iter_003"]:
        for use_pooled in [False, True]:
            tag = "pooled" if use_pooled else "perH"
            r = predict_max_over_horizons(
                per_cell, calibration, m, n_boot=args.n_boot, use_pooled_gap=use_pooled,
            )
            max_predictions[f"{m}__max__{tag}"] = r
            if "error" in r:
                continue
            sb = r["session_bootstrap"]
            cb = r["cluster_sym_bootstrap"]
            print(f"  {m} ({tag} gap): obs_max={r['observed_max_platform']:+.3f} @ {r['observed_argmax_horizon']}; "
                  f"sess CI95=[{sb['max_ci95_lo']:+.2f}, {sb['max_ci95_hi']:+.2f}] (P(<0)={sb['p_negative_max']:.2f}); "
                  f"clusterSym CI95=[{cb['max_ci95_lo']:+.2f}, {cb['max_ci95_hi']:+.2f}] (P(<0)={cb['p_negative_max']:.2f})")
            print(f"    horizon win-rates: {sb['argmax_distribution']}")

    out_cal = os.path.join(HERE, "calibration.json")
    with open(out_cal, "w") as f:
        json.dump({
            "platform_truth_mmpc_demo": PLATFORM_MMPC_DEMO,
            "platform_truth_iter_002": PLATFORM_ITER_002,
            "calibration_1anchor_mmpc": calibration,
            "calibration_2anchor_linear": cal_2anchor,
            "predictions_1anchor": predictions,
            "predictions_iter003_2anchor": iter003_2anchor,
            "max_over_horizons": max_predictions,
        }, f, indent=2, default=float)
    print(f"\n  saved calibration → {out_cal}")

    # Stress test
    stress = None
    if not args.no_stress:
        print("\n[bonus] Stress test on iter_002 h_10 (Gaussian noise on probs) ...")
        stress = stress_test_iter002_h10()
        print(f"  {'sigma':>8s}  {'cum_pnl':>10s}  {'n_active':>10s}")
        for r in stress["results"]:
            print(f"  {r['sigma']:>8.3f}  {r['sum_cum_pnl']:>+10.4f}  {r['n_active']:>10d}")

    elapsed = time.time() - t0
    print(f"\n[done] elapsed {elapsed:.1f}s")

    # Final results
    final = {
        "task": "T23 robust CV + platform predictor",
        "elapsed_sec": elapsed,
        "n_boot": args.n_boot,
        "scheme_results_keys": list(scheme_results.keys()),
        "calibration_summary": {
            h: {"gap": c["additive_gap"], "ratio": c["multiplicative_ratio"]}
            for h, c in calibration.items()
        },
        "platform_predictions_summary": {
            k: {
                "local_A": p["local_loso_A_sum"],
                "plat_add_mean": p["platform_predicted"]["additive_gap"]["mean"],
                "plat_add_ci95": [
                    p["platform_predicted"]["additive_gap"]["ci95_lo"],
                    p["platform_predicted"]["additive_gap"]["ci95_hi"],
                ],
                "plat_mul_mean": p["platform_predicted"]["multiplicative"]["mean"],
                "plat_cluster_sym_mean": p["platform_predicted"]["cluster_sym_bootstrap"]["mean"],
                "plat_cluster_sym_ci95": [
                    p["platform_predicted"]["cluster_sym_bootstrap"]["ci95_lo"],
                    p["platform_predicted"]["cluster_sym_bootstrap"]["ci95_hi"],
                ],
            }
            for k, p in predictions.items() if "error" not in p
        },
        "max_over_horizons_summary": {
            k: {
                "observed_max": r["observed_max_platform"],
                "observed_argmax_horizon": r["observed_argmax_horizon"],
                "session_bootstrap_ci95": [
                    r["session_bootstrap"]["max_ci95_lo"],
                    r["session_bootstrap"]["max_ci95_hi"],
                ],
                "cluster_sym_ci95": [
                    r["cluster_sym_bootstrap"]["max_ci95_lo"],
                    r["cluster_sym_bootstrap"]["max_ci95_hi"],
                ],
                "p_negative_max_session": r["session_bootstrap"]["p_negative_max"],
                "p_negative_max_cluster": r["cluster_sym_bootstrap"]["p_negative_max"],
            }
            for k, r in max_predictions.items() if "error" not in r
        },
        "stress_test": stress,
    }
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(final, f, indent=2, default=float)
    print(f"\n  saved results → {os.path.join(HERE, 'results.json')}")


if __name__ == "__main__":
    main()
