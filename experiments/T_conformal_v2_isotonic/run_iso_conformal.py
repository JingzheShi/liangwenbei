"""T_conformal_v2_isotonic: combined post-hoc on iter_015 v1 stack.

Trick 1: Isotonic calibration (3 variants):
  iso_global  — single calibrator on all syms
  iso_per_sym — 5 calibrators (per sym)
  iso_abs     — calibrate |pred|, sign-preserve

Trick 2: Conformal v2:
  beta_wider     — β grid [0, 0.6] in 0.025 step (vs original [0, 0.5] step 0.05)
  beta_asym      — asymmetric β_long ≠ β_short, per-sym, OOS picked
  beta_sig_aware — β_eff = β · max(1, σ_pred(local) / σ_baseline) (per-sym)

Combined:
  iso_global → per-sym β-conformal
  iso_per_sym → per-sym β-conformal

Eval: 4-fold date CV per-sym OOS β picking, baseline-OOS-fair (no test peek).
Target: > +42 LOSO. iter_018 v1 known +41.49.
"""
from __future__ import annotations
import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import differential_evolution

EXP_DIR = "/root/projects/liangwenbei_workdir/experiments/T_conformal_v2_isotonic"
T87_DIR = "/root/projects/liangwenbei_workdir/experiments/T87_spo_dfl"
T75_DIR = "/root/projects/liangwenbei_workdir/experiments/T75_regression_dmid"
SEEDS = (1, 7, 13, 42, 100)
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001
RNG = np.random.default_rng(42)


def vectorized_pnl(action, mp_t, mp_th):
    side = action.astype(np.float64) - 1.0
    abs_side = np.abs(side)
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * abs_side * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee_pnl) / denom


def ev_gate_asym(pred, thr_up, thr_dn):
    a = np.full(len(pred), 1, dtype=np.int8)
    a[pred > thr_up] = 2
    a[pred < -thr_dn] = 0
    return a


def ev_gate_band_asym(pred, thr_up, thr_dn, band_up, band_dn):
    """Asymmetric band: long-side band ≠ short-side band."""
    a = np.full(len(pred), 1, dtype=np.int8)
    a[pred > thr_up + band_up] = 2
    a[pred < -(thr_dn + band_dn)] = 0
    return a


def load_avg(paths):
    base = pd.read_parquet(paths[0])
    n = len(base)
    p = np.zeros(n, dtype=np.float64)
    for path in paths:
        df = pd.read_parquet(path)
        if len(df) != n:
            raise RuntimeError(f"row mismatch {path}")
        p += df["pred_dmid_norm"].to_numpy(np.float64)
    p /= len(paths)
    base = base.copy()
    base["pred_dmid_norm"] = p.astype(np.float64)
    return base


def split_by_sym(df, pred_col="pred_dmid_norm"):
    out = []
    for k in SYMS:
        sub = df[df["sym"] == k].reset_index(drop=True)
        out.append({
            "sym": int(k),
            "pred": sub[pred_col].to_numpy(np.float64),
            "true_dmid": sub["true_dmid_norm"].to_numpy(np.float64),
            "mp_t": sub["midprice_t"].to_numpy(np.float64),
            "mp_th": sub["midprice_th"].to_numpy(np.float64),
            "date": sub["date"].to_numpy(np.int64),
            "n": len(sub),
        })
    return out


def kfold_date_masks(folds, k=4):
    """Split each per-sym fold into k subsets by date."""
    out = []
    for f in folds:
        date = f["date"]
        unique_dates = np.sort(np.unique(date))
        # Split dates into k consecutive chunks
        chunks = np.array_split(unique_dates, k)
        masks_per_fold = []
        for chunk in chunks:
            chunk_set = set(chunk.tolist())
            mask = np.array([d in chunk_set for d in date])
            masks_per_fold.append(mask)
        out.append(masks_per_fold)
    return out  # out[i][j] = test mask for fold i, k-fold j


def make_obj_full(folds):
    pred = [f["pred"] for f in folds]
    mp_t = [f["mp_t"] for f in folds]
    mp_th = [f["mp_th"] for f in folds]
    def fn(x):
        thr_up, thr_dn = x
        s = 0.0
        for i in range(len(folds)):
            a = ev_gate_asym(pred[i], thr_up, thr_dn)
            s += vectorized_pnl(a, mp_t[i], mp_th[i]).sum()
        return -float(s)
    return fn


def de_search(obj, bounds, seeds=(0, 1, 2, 7, 42), maxiter=80, popsize=24):
    runs = []
    for sd in seeds:
        res = differential_evolution(
            obj, bounds=bounds, seed=sd, maxiter=maxiter, popsize=popsize,
            polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            updating="deferred", workers=1, init="sobol",
        )
        runs.append({"seed": int(sd), "x": res.x.tolist(), "obj_val": float(-res.fun)})
    runs.sort(key=lambda r: r["obj_val"], reverse=True)
    return runs


def per_sym_pnl(folds, actions):
    out, actives, wins = [], [], []
    for f, a in zip(folds, actions):
        pnl = vectorized_pnl(a, f["mp_t"], f["mp_th"])
        out.append(float(pnl.sum()))
        active_mask = (a != 1)
        n_active = int(active_mask.sum())
        actives.append(n_active / len(a) if len(a) > 0 else 0.0)
        if n_active > 0:
            wins.append(float((pnl[active_mask] > 0).mean()))
        else:
            wins.append(0.0)
    return out, actives, wins


def total_summary(folds, actions, label):
    per_sym, actives, wins = per_sym_pnl(folds, actions)
    total = float(sum(per_sym))
    summary = {
        "label": label,
        "per_sym_pnl": per_sym,
        "per_sym_active_rate": actives,
        "per_sym_win_rate": wins,
        "total": total,
        "min_per_sym": float(min(per_sym)),
        "active_rate": float(np.mean(actives)),
        "win_rate": float(np.mean(wins)),
    }
    print(f"  {label}: total={total:+.4f} min_sym={min(per_sym):+.4f} "
          f"active={np.mean(actives):.3f} win={np.mean(wins):.3f}", flush=True)
    return summary


# ============ ISOTONIC CALIBRATION ============

def fit_iso_oos(pred, target, kfolds_masks):
    """Fit isotonic OOS using k-fold cross-validation.
    For each fold j, fit on ¬j and predict on j. Returns calibrated pred (full size).
    Also returns the global calibrator (fit on all data) for deployment use.
    """
    cal = np.zeros_like(pred)
    for j, m_test in enumerate(kfolds_masks):
        m_train = ~m_test
        if m_train.sum() < 100:
            cal[m_test] = pred[m_test]
            continue
        iso = IsotonicRegression(out_of_bounds="clip")
        try:
            iso.fit(pred[m_train], target[m_train])
            cal[m_test] = iso.predict(pred[m_test])
        except Exception:
            cal[m_test] = pred[m_test]
    return cal


def isotonic_global_oos(folds, kfolds_per_fold, alpha=None):
    """Concat all syms, fit one isotonic OOS via k-fold."""
    all_pred = np.concatenate([f["pred"] for f in folds])
    all_target = np.concatenate([f["true_dmid"] for f in folds])
    if alpha is not None:
        all_target = np.clip(all_target, -alpha, alpha)
    n = len(all_pred)
    # Build k-fold masks at the global level by simple date hashing per sym
    # Use existing per-fold k-fold masks by concatenation in same order
    offsets = [0]
    for f in folds:
        offsets.append(offsets[-1] + f["n"])
    K = len(kfolds_per_fold[0])
    cal = np.zeros(n)
    for j in range(K):
        m_test = np.zeros(n, dtype=bool)
        for fi, off in enumerate(offsets[:-1]):
            m_test[off:offsets[fi+1]] = kfolds_per_fold[fi][j]
        m_train = ~m_test
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(all_pred[m_train], all_target[m_train])
        cal[m_test] = iso.predict(all_pred[m_test])
    # Split back into per-sym
    out = []
    for fi, off in enumerate(offsets[:-1]):
        out.append(cal[off:offsets[fi+1]])
    return out


def isotonic_per_sym_oos(folds, kfolds_per_fold, alpha=None):
    """Fit isotonic per sym OOS via k-fold."""
    out = []
    for fi, f in enumerate(folds):
        target = f["true_dmid"].copy()
        if alpha is not None:
            target = np.clip(target, -alpha, alpha)
        cal = fit_iso_oos(f["pred"], target, kfolds_per_fold[fi])
        out.append(cal)
    return out


def isotonic_abs_oos(folds, kfolds_per_fold, alpha=None):
    """Fit isotonic on |pred| → |target|, sign-preserve."""
    out = []
    for fi, f in enumerate(folds):
        pred = f["pred"]
        target = f["true_dmid"].copy()
        if alpha is not None:
            target = np.clip(target, -alpha, alpha)
        abs_pred = np.abs(pred)
        abs_target = np.abs(target)
        cal_abs = fit_iso_oos(abs_pred, abs_target, kfolds_per_fold[fi])
        # restore sign
        cal = np.sign(pred) * cal_abs
        out.append(cal)
    return out


# ============ CONFORMAL V2 ============

def per_sym_beta_kfold(folds_pred_array, folds, kfolds_per_fold, beta_grid,
                       thr_up, thr_dn, label="per-sym β k-fold"):
    """For each sym, k-fold CV-pick β per fold, then concat actions across folds.
    folds_pred_array[i] is the (possibly calibrated) prediction array for sym i.
    """
    actions_all = []
    chosen_betas_per_sym = []
    for fi, f in enumerate(folds):
        pred = folds_pred_array[fi]
        sigma = float(np.std(pred))
        action = np.full(f["n"], 1, dtype=np.int8)
        chosen = []
        K = len(kfolds_per_fold[fi])
        for j in range(K):
            m_test = kfolds_per_fold[fi][j]
            m_train = ~m_test
            best_b, best_pnl = 0.0, -1e18
            for b in beta_grid:
                a_train = ev_gate_asym(pred[m_train],
                                        thr_up + b * sigma,
                                        thr_dn + b * sigma)
                pnl = vectorized_pnl(a_train, f["mp_t"][m_train], f["mp_th"][m_train]).sum()
                if pnl > best_pnl:
                    best_pnl = pnl
                    best_b = b
            action[m_test] = ev_gate_asym(pred[m_test],
                                           thr_up + best_b * sigma,
                                           thr_dn + best_b * sigma)
            chosen.append(best_b)
        actions_all.append(action)
        chosen_betas_per_sym.append(chosen)
    return actions_all, chosen_betas_per_sym


def per_sym_beta_asym_kfold(folds_pred_array, folds, kfolds_per_fold, beta_grid,
                             thr_up, thr_dn, label="per-sym β-asym k-fold"):
    """Asymmetric β: β_long ≠ β_short, picked jointly OOS via k-fold.
    Grid search 2D (small grid for speed).
    """
    actions_all = []
    chosen_per_sym = []
    for fi, f in enumerate(folds):
        pred = folds_pred_array[fi]
        sigma = float(np.std(pred))
        action = np.full(f["n"], 1, dtype=np.int8)
        chosen = []
        K = len(kfolds_per_fold[fi])
        for j in range(K):
            m_test = kfolds_per_fold[fi][j]
            m_train = ~m_test
            best_bL, best_bS, best_pnl = 0.0, 0.0, -1e18
            for bL in beta_grid:
                for bS in beta_grid:
                    a_train = ev_gate_band_asym(pred[m_train],
                                                  thr_up, thr_dn,
                                                  bL * sigma, bS * sigma)
                    pnl = vectorized_pnl(a_train, f["mp_t"][m_train], f["mp_th"][m_train]).sum()
                    if pnl > best_pnl:
                        best_pnl = pnl
                        best_bL, best_bS = bL, bS
            action[m_test] = ev_gate_band_asym(pred[m_test], thr_up, thr_dn,
                                                 best_bL * sigma, best_bS * sigma)
            chosen.append((best_bL, best_bS))
        actions_all.append(action)
        chosen_per_sym.append(chosen)
    return actions_all, chosen_per_sym


def per_sym_beta_sigma_aware_kfold(folds_pred_array, folds, kfolds_per_fold, beta_grid,
                                    thr_up, thr_dn, sigma_baseline=4.0e-4):
    """Sigma-aware β: band = β · max(1, σ_local / σ_baseline) · σ_local.
    σ_local uses train partition.
    """
    actions_all = []
    chosen_per_sym = []
    for fi, f in enumerate(folds):
        pred = folds_pred_array[fi]
        action = np.full(f["n"], 1, dtype=np.int8)
        chosen = []
        K = len(kfolds_per_fold[fi])
        for j in range(K):
            m_test = kfolds_per_fold[fi][j]
            m_train = ~m_test
            sigma_local = float(np.std(pred[m_train]))
            scale = max(1.0, sigma_local / sigma_baseline)
            best_b, best_pnl = 0.0, -1e18
            for b in beta_grid:
                eff_band = b * scale * sigma_local
                a_train = ev_gate_asym(pred[m_train],
                                        thr_up + eff_band,
                                        thr_dn + eff_band)
                pnl = vectorized_pnl(a_train, f["mp_t"][m_train], f["mp_th"][m_train]).sum()
                if pnl > best_pnl:
                    best_pnl = pnl
                    best_b = b
            sigma_test_train = float(np.std(pred[m_train]))  # use train to avoid peek
            scale_apply = max(1.0, sigma_test_train / sigma_baseline)
            eff_band_apply = best_b * scale_apply * sigma_test_train
            action[m_test] = ev_gate_asym(pred[m_test],
                                           thr_up + eff_band_apply,
                                           thr_dn + eff_band_apply)
            chosen.append({"beta": best_b, "scale": scale_apply, "sigma": sigma_test_train})
        actions_all.append(action)
        chosen_per_sym.append(chosen)
    return actions_all, chosen_per_sym


def main():
    print("=== Loading T87 + T75 5-seed avg → iter_015 v1 stack ===", flush=True)
    t87_paths = [os.path.join(T87_DIR, f"pred_T87_seed{s}_main.parquet") for s in SEEDS]
    df_t87 = load_avg(t87_paths)
    t75_paths = [os.path.join(T75_DIR, f"pred_T75_seed{s}.parquet") for s in SEEDS]
    df_t75 = load_avg(t75_paths)
    p_combined = (1.0 * df_t87["pred_dmid_norm"].to_numpy(np.float64)
                  + 1.5 * df_t75["pred_dmid_norm"].to_numpy(np.float64)) / 2.5
    df_stack = df_t87.copy()
    df_stack["pred_dmid_norm"] = p_combined
    print(f"stack n={len(df_stack):,}, dates {df_stack['date'].min()}-{df_stack['date'].max()}", flush=True)

    folds = split_by_sym(df_stack)
    kfolds_per_fold = kfold_date_masks(folds, k=4)

    results = {}

    # === Tune in-sample DE thresh (matches iter_015 v1 baseline) ===
    print("\n=== DE-tune thr_up/thr_dn (in-sample, full data) ===", flush=True)
    bounds = [(-3 * FEE, 3 * FEE), (-3 * FEE, 3 * FEE)]
    obj_full = make_obj_full(folds)
    runs = de_search(obj_full, bounds)
    thr_up = float(runs[0]["x"][0])
    thr_dn = float(runs[0]["x"][1])
    print(f"  thr_up={thr_up:+.6e}  thr_dn={thr_dn:+.6e}", flush=True)
    results["de_thresh"] = {"thr_up": thr_up, "thr_dn": thr_dn, "runs": runs}

    # === Baseline: iter_015 v1 (no abstain) ===
    print("\n=== Baseline: iter_015 v1 (no abstain) ===", flush=True)
    actions_b = [ev_gate_asym(f["pred"], thr_up, thr_dn) for f in folds]
    results["baseline_iter015v1"] = total_summary(folds, actions_b, "baseline iter015 v1")

    # === iter_018 v1 wrapper (locked β = {0:0.10, 1:0.40, 2:0.30, 3:0.00, 4:0.00}, σ=val full per-sym) ===
    print("\n=== iter_018 v1 (locked β, in-sample σ) ===", flush=True)
    BETA_LOCKED = {0: 0.10, 1: 0.40, 2: 0.30, 3: 0.00, 4: 0.00}
    actions_018 = []
    for f in folds:
        sigma = float(np.std(f["pred"]))
        b = BETA_LOCKED[f["sym"]]
        band = b * sigma
        actions_018.append(ev_gate_asym(f["pred"], thr_up + band, thr_dn + band))
    results["iter018_v1"] = total_summary(folds, actions_018, "iter018 v1 locked-β")

    # === Conformal v2: per-sym β, k-fold CV picked, wider grid ===
    print("\n=== Conformal v2 (per-sym β k-fold, wider grid [0,0.6] step 0.025) ===", flush=True)
    beta_grid_wide = list(np.round(np.arange(0.0, 0.625, 0.025), 4))
    actions_v2, betas_v2 = per_sym_beta_kfold(
        [f["pred"] for f in folds], folds, kfolds_per_fold, beta_grid_wide,
        thr_up, thr_dn,
    )
    print(f"  β chosen per-sym per-fold: {betas_v2}", flush=True)
    results["conformal_v2_wide"] = total_summary(folds, actions_v2, "conformal v2 wide-grid")
    results["conformal_v2_wide"]["betas_per_sym_per_fold"] = betas_v2

    # === Conformal v2: asymmetric β_long/β_short ===
    print("\n=== Conformal v2-asym (β_long ≠ β_short, k-fold) ===", flush=True)
    beta_grid_small = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]
    actions_v2asym, betas_asym = per_sym_beta_asym_kfold(
        [f["pred"] for f in folds], folds, kfolds_per_fold, beta_grid_small,
        thr_up, thr_dn,
    )
    print(f"  (βL,βS) chosen per-sym per-fold: {betas_asym}", flush=True)
    results["conformal_v2_asym"] = total_summary(folds, actions_v2asym, "conformal v2 asym")
    results["conformal_v2_asym"]["betas_per_sym_per_fold"] = [
        [list(b) for b in s] for s in betas_asym
    ]

    # === Conformal v2: sigma-aware ===
    print("\n=== Conformal v2-sig (sigma-aware β scale, k-fold) ===", flush=True)
    actions_v2sig, betas_sig = per_sym_beta_sigma_aware_kfold(
        [f["pred"] for f in folds], folds, kfolds_per_fold, beta_grid_wide,
        thr_up, thr_dn, sigma_baseline=4.0e-4,
    )
    print(f"  chosen per-sym per-fold: {betas_sig}", flush=True)
    results["conformal_v2_sig"] = total_summary(folds, actions_v2sig, "conformal v2 sig-aware")
    results["conformal_v2_sig"]["betas_per_sym_per_fold"] = betas_sig

    # === Trick 1: Isotonic (3 variants) ===
    # alpha for clipping target — match common practice (use 99th pct of |true_dmid|)
    all_target = df_stack["true_dmid_norm"].to_numpy(np.float64)
    alpha_clip = float(np.quantile(np.abs(all_target), 0.99))
    print(f"\n=== alpha_clip (99pct |true_dmid|) = {alpha_clip:.6e} ===", flush=True)

    print("\n=== Trick 1a: iso_global (1 calibrator OOS via k-fold, alpha-clipped) ===", flush=True)
    cal_global = isotonic_global_oos(folds, kfolds_per_fold, alpha=alpha_clip)
    actions_iso_g = [ev_gate_asym(c, thr_up, thr_dn) for c in cal_global]
    results["iso_global_only"] = total_summary(folds, actions_iso_g, "iso_global only (no abstain)")

    print("\n=== Trick 1b: iso_per_sym (5 calibrators OOS via k-fold) ===", flush=True)
    cal_per_sym = isotonic_per_sym_oos(folds, kfolds_per_fold, alpha=alpha_clip)
    actions_iso_p = [ev_gate_asym(c, thr_up, thr_dn) for c in cal_per_sym]
    results["iso_per_sym_only"] = total_summary(folds, actions_iso_p, "iso_per_sym only (no abstain)")

    print("\n=== Trick 1c: iso_abs (calibrate |pred|→|target|, sign preserve) ===", flush=True)
    cal_abs = isotonic_abs_oos(folds, kfolds_per_fold, alpha=alpha_clip)
    actions_iso_a = [ev_gate_asym(c, thr_up, thr_dn) for c in cal_abs]
    results["iso_abs_only"] = total_summary(folds, actions_iso_a, "iso_abs only (no abstain)")

    # === Combined: iso → conformal (wide β grid k-fold) ===
    print("\n=== Combined: iso_global → conformal v2-wide (per-sym β k-fold) ===", flush=True)
    # NOTE: thr_up/thr_dn was DE-tuned on raw stack pred, not iso. Re-tune DE on cal_global.
    folds_cal_g = [{**f, "pred": c} for f, c in zip(folds, cal_global)]
    obj_cal_g = make_obj_full(folds_cal_g)
    runs_g = de_search(obj_cal_g, bounds)
    thr_up_g, thr_dn_g = float(runs_g[0]["x"][0]), float(runs_g[0]["x"][1])
    print(f"  re-tuned DE on iso_global: thr_up={thr_up_g:+.6e} thr_dn={thr_dn_g:+.6e}", flush=True)
    actions_iso_g_v2, betas_iso_g_v2 = per_sym_beta_kfold(
        cal_global, folds, kfolds_per_fold, beta_grid_wide, thr_up_g, thr_dn_g,
    )
    print(f"  β: {betas_iso_g_v2}", flush=True)
    results["combined_iso_global_conformal_v2"] = total_summary(
        folds, actions_iso_g_v2, "iso_global + conformal v2 wide"
    )
    results["combined_iso_global_conformal_v2"]["thr_up"] = thr_up_g
    results["combined_iso_global_conformal_v2"]["thr_dn"] = thr_dn_g
    results["combined_iso_global_conformal_v2"]["betas"] = betas_iso_g_v2

    print("\n=== Combined: iso_per_sym → conformal v2-wide ===", flush=True)
    folds_cal_p = [{**f, "pred": c} for f, c in zip(folds, cal_per_sym)]
    obj_cal_p = make_obj_full(folds_cal_p)
    runs_p = de_search(obj_cal_p, bounds)
    thr_up_p, thr_dn_p = float(runs_p[0]["x"][0]), float(runs_p[0]["x"][1])
    print(f"  re-tuned DE on iso_per_sym: thr_up={thr_up_p:+.6e} thr_dn={thr_dn_p:+.6e}", flush=True)
    actions_iso_p_v2, betas_iso_p_v2 = per_sym_beta_kfold(
        cal_per_sym, folds, kfolds_per_fold, beta_grid_wide, thr_up_p, thr_dn_p,
    )
    print(f"  β: {betas_iso_p_v2}", flush=True)
    results["combined_iso_per_sym_conformal_v2"] = total_summary(
        folds, actions_iso_p_v2, "iso_per_sym + conformal v2 wide"
    )
    results["combined_iso_per_sym_conformal_v2"]["thr_up"] = thr_up_p
    results["combined_iso_per_sym_conformal_v2"]["thr_dn"] = thr_dn_p
    results["combined_iso_per_sym_conformal_v2"]["betas"] = betas_iso_p_v2

    # === Combined: iso_abs → conformal v2-wide ===
    print("\n=== Combined: iso_abs → conformal v2-wide ===", flush=True)
    folds_cal_a = [{**f, "pred": c} for f, c in zip(folds, cal_abs)]
    obj_cal_a = make_obj_full(folds_cal_a)
    runs_a = de_search(obj_cal_a, bounds)
    thr_up_a, thr_dn_a = float(runs_a[0]["x"][0]), float(runs_a[0]["x"][1])
    print(f"  re-tuned DE on iso_abs: thr_up={thr_up_a:+.6e} thr_dn={thr_dn_a:+.6e}", flush=True)
    actions_iso_a_v2, betas_iso_a_v2 = per_sym_beta_kfold(
        cal_abs, folds, kfolds_per_fold, beta_grid_wide, thr_up_a, thr_dn_a,
    )
    print(f"  β: {betas_iso_a_v2}", flush=True)
    results["combined_iso_abs_conformal_v2"] = total_summary(
        folds, actions_iso_a_v2, "iso_abs + conformal v2 wide"
    )

    # ===  Print summary ===
    print("\n========================== SUMMARY ==========================", flush=True)
    keys = [
        "baseline_iter015v1", "iter018_v1",
        "conformal_v2_wide", "conformal_v2_asym", "conformal_v2_sig",
        "iso_global_only", "iso_per_sym_only", "iso_abs_only",
        "combined_iso_global_conformal_v2",
        "combined_iso_per_sym_conformal_v2",
        "combined_iso_abs_conformal_v2",
    ]
    for k in keys:
        if k in results:
            r = results[k]
            print(f"  {k:42s}  total={r['total']:+8.4f}  "
                  f"min_sym={r['min_per_sym']:+7.4f}  active={r['active_rate']:.3f}  "
                  f"win={r['win_rate']:.3f}", flush=True)

    out_path = os.path.join(EXP_DIR, "results.json")
    with open(out_path, "w") as fp:
        json.dump(results, fp, indent=2, default=str)
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
