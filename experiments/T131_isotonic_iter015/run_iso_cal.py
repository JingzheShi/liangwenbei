"""T131: Post-hoc isotonic calibration on iter_015 v1 stack predictions.

iter_015 v1 stack:
  pred_stack = (1.0 * mean_T87_5seed + 1.5 * mean_T75_5seed) / 2.5
  Known DE-LOSO baseline on full 442k = +40.13 (reported), +39.75 (also seen)

Test plan:
  - Held-out split by date: val = dates 96-107 (12 dates), eval = dates 108-119 (12 dates)
  - Variants:
      0) RAW (no calibration), DE-LOSO on val -> apply to eval
      1) ISO_GLOBAL: fit IsotonicRegression on val (pred -> true_dmid_norm), apply to all
      2) ISO_PER_SYM: fit one isotonic per sym on val
      3) ISO_ABS: fit isotonic on |pred| -> |true|, sign-preserve
  - For each variant: DE-LOSO on val_cal -> measure PnL on eval_cal (held out)
  - Also report in-sample full 442k iso (informational, NOT held out) vs the +40.13 baseline.

Output:
  results.json  - all metrics
  iso_models.pkl  - fitted calibrators (for potential wrapper)
"""
from __future__ import annotations

import json
import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
T87_DIR = os.path.join(ROOT, "experiments", "T87_spo_dfl")
T75_DIR = os.path.join(ROOT, "experiments", "T75_regression_dmid")

SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001
SEEDS = (1, 7, 13, 42, 100)

W_NN = 1.0
W_LGB = 1.5

# Date split
VAL_DATES = list(range(96, 108))    # 96..107 inclusive = 12 dates (calibration)
EVAL_DATES = list(range(108, 120))  # 108..119 inclusive = 12 dates (held-out eval)


def vectorized_pnl(pred_action, mp_t, mp_th):
    side = pred_action.astype(np.float64) - 1.0
    abs_side = np.abs(side)
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * abs_side * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee_pnl) / denom


def ev_gate_asymmetric(pred, thr_up, thr_dn):
    a = np.full(len(pred), 1, dtype=np.int8)
    a[pred > thr_up] = 2
    a[pred < -thr_dn] = 0
    return a


def make_obj_loso_asym(pred_list, mp_t_list, mp_th_list):
    def f(x):
        thr_up, thr_dn = x
        s = 0.0
        for i in range(len(pred_list)):
            a = ev_gate_asymmetric(pred_list[i], thr_up, thr_dn)
            s += vectorized_pnl(a, mp_t_list[i], mp_th_list[i]).sum()
        return -float(s)
    return f


def de_search_asym(pred_list, mp_t_list, mp_th_list, seeds=(0, 1, 2, 7, 42)):
    obj = make_obj_loso_asym(pred_list, mp_t_list, mp_th_list)
    runs = []
    for sd in seeds:
        result = differential_evolution(
            obj, bounds=[(0.0, 0.0040), (0.0, 0.0040)],
            seed=sd, maxiter=80, popsize=24, polish=True, tol=1e-7,
            mutation=(0.5, 1.0), recombination=0.7,
            updating="deferred", workers=1, init="sobol",
        )
        runs.append({
            "seed": int(sd),
            "thr_up": float(result.x[0]),
            "thr_dn": float(result.x[1]),
            "obj_val": float(-result.fun),
        })
    runs.sort(key=lambda r: r["obj_val"], reverse=True)
    return runs[0]


def loso_pnl_at_thr(df, pred_col, thr_up, thr_dn):
    per = []
    total = 0.0
    for k in SYMS:
        sub = df[df["sym"] == k]
        p = sub[pred_col].to_numpy(np.float64)
        a = ev_gate_asymmetric(p, thr_up, thr_dn)
        v = vectorized_pnl(a, sub["midprice_t"].to_numpy(np.float64),
                           sub["midprice_th"].to_numpy(np.float64)).sum()
        per.append(float(v))
        total += float(v)
    return total, per


def split_lists(df, pred_col):
    """Return (pred_list, mp_t_list, mp_th_list) split by sym."""
    pl, ml, mhl = [], [], []
    for k in SYMS:
        sub = df[df["sym"] == k]
        pl.append(sub[pred_col].to_numpy(np.float64))
        ml.append(sub["midprice_t"].to_numpy(np.float64))
        mhl.append(sub["midprice_th"].to_numpy(np.float64))
    return pl, ml, mhl


def load_pred_avg(pred_dir, prefix, suffix=""):
    """Average predictions across seeds. Return df with merged 'pred_dmid_norm' = avg over seeds."""
    base = pd.read_parquet(os.path.join(pred_dir, f"{prefix}_seed{SEEDS[0]}{suffix}.parquet"))
    n = len(base)
    p = np.zeros(n, dtype=np.float64)
    for s in SEEDS:
        path = os.path.join(pred_dir, f"{prefix}_seed{s}{suffix}.parquet")
        df = pd.read_parquet(path)
        if len(df) != n:
            raise RuntimeError(f"row count mismatch {path}: {len(df)} vs {n}")
        if not (df["sym"].equals(base["sym"]) and df["t"].equals(base["t"]) and df["date"].equals(base["date"])):
            raise RuntimeError(f"row order mismatch {path}")
        p += df["pred_dmid_norm"].to_numpy(np.float64)
    p /= len(SEEDS)
    out = base[["sym", "date", "session", "t", "true_dmid_norm", "midprice_t", "midprice_th"]].copy()
    out["pred"] = p
    return out


def main():
    t0 = time.time()
    print("Loading T87 SPO+ NN 5-seed avg ...", flush=True)
    df_nn = load_pred_avg(T87_DIR, "pred_T87", suffix="_main")
    print(f"  T87 nn shape={df_nn.shape}, dates {df_nn['date'].min()}..{df_nn['date'].max()}", flush=True)
    print("Loading T75 LGB 5-seed avg ...", flush=True)
    df_lgb = load_pred_avg(T75_DIR, "pred_T75", suffix="")
    print(f"  T75 lgb shape={df_lgb.shape}, dates {df_lgb['date'].min()}..{df_lgb['date'].max()}", flush=True)

    # Sanity: same alignment
    assert df_nn["sym"].equals(df_lgb["sym"]), "sym alignment mismatch"
    assert df_nn["date"].equals(df_lgb["date"]), "date alignment mismatch"
    assert df_nn["t"].equals(df_lgb["t"]), "t alignment mismatch"
    assert np.allclose(df_nn["true_dmid_norm"].values, df_lgb["true_dmid_norm"].values, atol=1e-12), "y_true mismatch"

    df = df_nn.copy()
    df.rename(columns={"pred": "pred_nn"}, inplace=True)
    df["pred_lgb"] = df_lgb["pred"].values
    df["pred_raw"] = (W_NN * df["pred_nn"] + W_LGB * df["pred_lgb"]) / (W_NN + W_LGB)

    print(f"\npred_raw stats: mean={df['pred_raw'].mean():+.6e} std={df['pred_raw'].std():.6e}", flush=True)
    print(f"y_true (dmid_norm) stats: mean={df['true_dmid_norm'].mean():+.6e} std={df['true_dmid_norm'].std():.6e}", flush=True)

    # Split val / eval by date
    val_mask = df["date"].isin(VAL_DATES)
    eval_mask = df["date"].isin(EVAL_DATES)
    df_val = df[val_mask].reset_index(drop=True)
    df_eval = df[eval_mask].reset_index(drop=True)
    print(f"\nval rows={len(df_val)} dates {df_val['date'].min()}..{df_val['date'].max()}", flush=True)
    print(f"eval rows={len(df_eval)} dates {df_eval['date'].min()}..{df_eval['date'].max()}", flush=True)

    results = {
        "val_dates": VAL_DATES,
        "eval_dates": EVAL_DATES,
        "n_val": len(df_val),
        "n_eval": len(df_eval),
        "weights": {"w_nn": W_NN, "w_lgb": W_LGB},
        "variants": {},
    }

    # ---- Variant 0: RAW ----
    print("\n=== Variant 0: RAW (no calibration) ===", flush=True)
    pl, ml, mhl = split_lists(df_val, "pred_raw")
    de_raw_val = de_search_asym(pl, ml, mhl)
    print(f"  DE on val: thr_up={de_raw_val['thr_up']:.6f} thr_dn={de_raw_val['thr_dn']:.6f} val_pnl={de_raw_val['obj_val']:+.4f}",
          flush=True)
    eval_pnl_raw, eval_per_sym_raw = loso_pnl_at_thr(df_eval, "pred_raw", de_raw_val["thr_up"], de_raw_val["thr_dn"])
    print(f"  EVAL pnl @ val-thr: {eval_pnl_raw:+.4f} per_sym=[{', '.join(f'{x:+.3f}' for x in eval_per_sym_raw)}]",
          flush=True)
    # Also DE on full 442k (matches reported +40.13)
    pl_all, ml_all, mhl_all = split_lists(df, "pred_raw")
    de_raw_all = de_search_asym(pl_all, ml_all, mhl_all)
    full_pnl_raw, full_per_sym_raw = loso_pnl_at_thr(df, "pred_raw", de_raw_all["thr_up"], de_raw_all["thr_dn"])
    print(f"  DE on full 442k: thr_up={de_raw_all['thr_up']:.6f} thr_dn={de_raw_all['thr_dn']:.6f} full_pnl={full_pnl_raw:+.4f}",
          flush=True)
    print(f"     full per_sym=[{', '.join(f'{x:+.3f}' for x in full_per_sym_raw)}]", flush=True)
    results["variants"]["raw"] = {
        "de_val": de_raw_val,
        "eval_pnl_at_val_thr": eval_pnl_raw,
        "eval_per_sym": eval_per_sym_raw,
        "de_full": de_raw_all,
        "full_pnl": full_pnl_raw,
        "full_per_sym": full_per_sym_raw,
    }

    # ---- Variant 1: ISO_GLOBAL ----
    print("\n=== Variant 1: ISO_GLOBAL (single isotonic) ===", flush=True)
    iso_global = IsotonicRegression(out_of_bounds="clip")
    iso_global.fit(df_val["pred_raw"].values, df_val["true_dmid_norm"].values)
    df_val["pred_iso_global"] = iso_global.transform(df_val["pred_raw"].values)
    df_eval["pred_iso_global"] = iso_global.transform(df_eval["pred_raw"].values)
    df["pred_iso_global"] = iso_global.transform(df["pred_raw"].values)
    print(f"  iso_global: pred_iso_global val mean={df_val['pred_iso_global'].mean():+.6e} std={df_val['pred_iso_global'].std():.6e}",
          flush=True)
    pl, ml, mhl = split_lists(df_val, "pred_iso_global")
    de_iso_g = de_search_asym(pl, ml, mhl)
    print(f"  DE on val: thr_up={de_iso_g['thr_up']:.6f} thr_dn={de_iso_g['thr_dn']:.6f} val_pnl={de_iso_g['obj_val']:+.4f}",
          flush=True)
    eval_pnl_iso_g, eval_per_sym_iso_g = loso_pnl_at_thr(df_eval, "pred_iso_global",
                                                          de_iso_g["thr_up"], de_iso_g["thr_dn"])
    print(f"  EVAL pnl: {eval_pnl_iso_g:+.4f} per_sym=[{', '.join(f'{x:+.3f}' for x in eval_per_sym_iso_g)}]",
          flush=True)
    # Full 442k iso (in-sample)
    iso_global_full = IsotonicRegression(out_of_bounds="clip")
    iso_global_full.fit(df["pred_raw"].values, df["true_dmid_norm"].values)
    df["pred_iso_global_full"] = iso_global_full.transform(df["pred_raw"].values)
    pl_all, ml_all, mhl_all = split_lists(df, "pred_iso_global_full")
    de_iso_g_full = de_search_asym(pl_all, ml_all, mhl_all)
    full_pnl_iso_g, full_per_sym_iso_g = loso_pnl_at_thr(df, "pred_iso_global_full",
                                                          de_iso_g_full["thr_up"], de_iso_g_full["thr_dn"])
    print(f"  [info] in-sample full 442k iso: pnl={full_pnl_iso_g:+.4f}", flush=True)
    results["variants"]["iso_global"] = {
        "de_val": de_iso_g,
        "eval_pnl": eval_pnl_iso_g,
        "eval_per_sym": eval_per_sym_iso_g,
        "in_sample_full_pnl": full_pnl_iso_g,
    }

    # ---- Variant 2: ISO_PER_SYM ----
    print("\n=== Variant 2: ISO_PER_SYM (5 isotonics) ===", flush=True)
    iso_persym = {}
    df_val["pred_iso_per_sym"] = 0.0
    df_eval["pred_iso_per_sym"] = 0.0
    df["pred_iso_per_sym"] = 0.0
    for k in SYMS:
        v_idx = df_val[df_val["sym"] == k].index
        e_idx = df_eval[df_eval["sym"] == k].index
        f_idx = df[df["sym"] == k].index
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(df_val.loc[v_idx, "pred_raw"].values, df_val.loc[v_idx, "true_dmid_norm"].values)
        df_val.loc[v_idx, "pred_iso_per_sym"] = iso.transform(df_val.loc[v_idx, "pred_raw"].values)
        df_eval.loc[e_idx, "pred_iso_per_sym"] = iso.transform(df_eval.loc[e_idx, "pred_raw"].values)
        df.loc[f_idx, "pred_iso_per_sym"] = iso.transform(df.loc[f_idx, "pred_raw"].values)
        iso_persym[k] = iso
    pl, ml, mhl = split_lists(df_val, "pred_iso_per_sym")
    de_iso_ps = de_search_asym(pl, ml, mhl)
    print(f"  DE on val: thr_up={de_iso_ps['thr_up']:.6f} thr_dn={de_iso_ps['thr_dn']:.6f} val_pnl={de_iso_ps['obj_val']:+.4f}",
          flush=True)
    eval_pnl_iso_ps, eval_per_sym_iso_ps = loso_pnl_at_thr(df_eval, "pred_iso_per_sym",
                                                            de_iso_ps["thr_up"], de_iso_ps["thr_dn"])
    print(f"  EVAL pnl: {eval_pnl_iso_ps:+.4f} per_sym=[{', '.join(f'{x:+.3f}' for x in eval_per_sym_iso_ps)}]",
          flush=True)
    results["variants"]["iso_per_sym"] = {
        "de_val": de_iso_ps,
        "eval_pnl": eval_pnl_iso_ps,
        "eval_per_sym": eval_per_sym_iso_ps,
    }

    # ---- Variant 3: ISO_ABS (sign-preserving) ----
    print("\n=== Variant 3: ISO_ABS (sign-preserve isotonic on |pred|) ===", flush=True)
    iso_abs = IsotonicRegression(out_of_bounds="clip")
    abs_pred_val = np.abs(df_val["pred_raw"].values)
    abs_y_val = np.abs(df_val["true_dmid_norm"].values)
    iso_abs.fit(abs_pred_val, abs_y_val)
    sgn_val = np.sign(df_val["pred_raw"].values)
    sgn_eval = np.sign(df_eval["pred_raw"].values)
    df_val["pred_iso_abs"] = sgn_val * iso_abs.transform(np.abs(df_val["pred_raw"].values))
    df_eval["pred_iso_abs"] = sgn_eval * iso_abs.transform(np.abs(df_eval["pred_raw"].values))
    pl, ml, mhl = split_lists(df_val, "pred_iso_abs")
    de_iso_a = de_search_asym(pl, ml, mhl)
    print(f"  DE on val: thr_up={de_iso_a['thr_up']:.6f} thr_dn={de_iso_a['thr_dn']:.6f} val_pnl={de_iso_a['obj_val']:+.4f}",
          flush=True)
    eval_pnl_iso_a, eval_per_sym_iso_a = loso_pnl_at_thr(df_eval, "pred_iso_abs",
                                                          de_iso_a["thr_up"], de_iso_a["thr_dn"])
    print(f"  EVAL pnl: {eval_pnl_iso_a:+.4f} per_sym=[{', '.join(f'{x:+.3f}' for x in eval_per_sym_iso_a)}]",
          flush=True)
    results["variants"]["iso_abs"] = {
        "de_val": de_iso_a,
        "eval_pnl": eval_pnl_iso_a,
        "eval_per_sym": eval_per_sym_iso_a,
    }

    # ---- Summary ----
    print("\n" + "="*60, flush=True)
    print("SUMMARY: held-out eval PnL (val=dates 96-107, eval=dates 108-119)", flush=True)
    print(f"  raw:         eval={eval_pnl_raw:+.4f}", flush=True)
    print(f"  iso_global:  eval={eval_pnl_iso_g:+.4f}  Δ={eval_pnl_iso_g-eval_pnl_raw:+.4f}", flush=True)
    print(f"  iso_per_sym: eval={eval_pnl_iso_ps:+.4f}  Δ={eval_pnl_iso_ps-eval_pnl_raw:+.4f}", flush=True)
    print(f"  iso_abs:     eval={eval_pnl_iso_a:+.4f}  Δ={eval_pnl_iso_a-eval_pnl_raw:+.4f}", flush=True)
    print(f"\n  full 442k raw DE-LOSO: {full_pnl_raw:+.4f} (vs reported +40.13)", flush=True)
    print(f"  full 442k iso_global in-sample: {full_pnl_iso_g:+.4f} (informational, not held out)", flush=True)

    deltas = {
        "iso_global": eval_pnl_iso_g - eval_pnl_raw,
        "iso_per_sym": eval_pnl_iso_ps - eval_pnl_raw,
        "iso_abs": eval_pnl_iso_a - eval_pnl_raw,
    }
    best_var = max(deltas, key=lambda k: deltas[k])
    results["delta_best"] = {"variant": best_var, "delta": deltas[best_var]}
    results["wallclock_sec"] = time.time() - t0

    out_path = os.path.join(HERE, "results.json")
    with open(out_path, "w") as fp:
        json.dump(results, fp, indent=2, default=float)
    print(f"\nResults -> {out_path}", flush=True)

    # Save iso models for potential wrapper deployment
    iso_models = {
        "iso_global": iso_global,
        "iso_global_full": iso_global_full,
        "iso_per_sym": iso_persym,
        "iso_abs": iso_abs,
    }
    with open(os.path.join(HERE, "iso_models.pkl"), "wb") as fp:
        pickle.dump(iso_models, fp)
    print(f"Saved iso_models.pkl  wallclock={time.time()-t0:.1f}s", flush=True)

    return results


if __name__ == "__main__":
    main()
