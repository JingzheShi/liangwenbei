"""Conformal Selective Trading wrapper for iter_015 v1 stack (T87 NN + T75 LGB L2).

Hypothesis: Distribution-free conformal calibration on val set gives more stable
trade-selection than DE-tuned asymmetric thresholds (which over-fit local 442k).

Key design:
  - Mondrian-by-sym (per-sym calibration; no cross-sym leakage)
  - Class-conditional nonconformity using REALIZED PnL outcomes:
      * "up-conformal" calib subset = rows where true_label == +1 (actually UP)
      * "dn-conformal" calib subset = rows where true_label == -1 (actually DN)
      * score_up = -pred (small for confident long); score_dn = +pred
  - Test row: trade if predict-set is SINGLETON {up} or {down}, else FLAT
  - 2-fold cross-conformal split by date (no calib/test leakage)

Compares baseline DE-asym thresh vs conformal selective on iter_015 v1 stack.
"""
from __future__ import annotations
import json
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

EXP_DIR = "/root/projects/liangwenbei_workdir/experiments/R_conformal_select"
T87_DIR = "/root/projects/liangwenbei_workdir/experiments/T87_spo_dfl"
T75_DIR = "/root/projects/liangwenbei_workdir/experiments/T75_regression_dmid"
SEEDS = (1, 7, 13, 42, 100)
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001


def vectorized_pnl(action, mp_t, mp_th):
    side = action.astype(np.float64) - 1.0
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


def split_by_sym(df, pred_col="pred_dmid_norm"):
    out = []
    for k in SYMS:
        sub = df[df["sym"] == k].reset_index(drop=True)
        out.append({
            "sym": int(k),
            "pred": sub[pred_col].to_numpy(np.float64),
            "label": sub["true_label"].to_numpy(np.int64),
            "true_dmid": sub["true_dmid_norm"].to_numpy(np.float64),
            "mp_t": sub["midprice_t"].to_numpy(np.float64),
            "mp_th": sub["midprice_th"].to_numpy(np.float64),
            "date": sub["date"].to_numpy(np.int64),
            "n": len(sub),
        })
    return out


def make_obj_loso_asym(folds):
    pred = [f["pred"] for f in folds]
    mp_t = [f["mp_t"] for f in folds]
    mp_th = [f["mp_th"] for f in folds]
    def f(x):
        thr_up, thr_dn = x
        s = 0.0
        for i in range(len(folds)):
            a = ev_gate_asymmetric(pred[i], thr_up, thr_dn)
            s += vectorized_pnl(a, mp_t[i], mp_th[i]).sum()
        return -float(s)
    return f


def de_search(objective_fn, bounds, seeds=(0, 1, 2, 7, 42)):
    runs = []
    for sd in seeds:
        res = differential_evolution(
            objective_fn, bounds=bounds, seed=sd, maxiter=80, popsize=24,
            polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            updating="deferred", workers=1, init="sobol",
        )
        runs.append({
            "seed": int(sd),
            "thr_up": float(res.x[0]),
            "thr_dn": float(res.x[1]),
            "obj_val": float(-res.fun),
        })
    runs.sort(key=lambda r: r["obj_val"], reverse=True)
    return runs


def per_sym_pnl_from_action(folds, actions_per_sym):
    """actions_per_sym: list of arrays, one per sym, aligned to folds."""
    out = []
    actives = []
    wins = []
    for f, a in zip(folds, actions_per_sym):
        pnl = vectorized_pnl(a, f["mp_t"], f["mp_th"])
        out.append(float(pnl.sum()))
        # active rate = fraction of trades (action != 1)
        n_active = int((a != 1).sum())
        actives.append(n_active / len(a) if len(a) > 0 else 0.0)
        # win rate among active trades
        active_mask = (a != 1)
        if active_mask.sum() > 0:
            wins.append(float((pnl[active_mask] > 0).mean()))
        else:
            wins.append(0.0)
    return out, actives, wins


# --------------- Conformal selective trading ---------------

def conformal_singleton_action(
    pred_calib, true_dmid_calib, mp_t_calib, mp_th_calib,
    pred_test, alpha,
):
    """Per-sym conformal selective trading.

    Calibration class subsets:
      - "up-correct" subset: calib rows where actually going long would have given +PnL
        (i.e., true_dmid > fee_cost for long)
      - "dn-correct" subset: calib rows where actually going short would have given +PnL

    Nonconformity scores (smaller = more confident in that direction):
      - score_up(pred) = -pred           (high pred → low score → strong long)
      - score_dn(pred) = +pred           (low pred → low score → strong short)

    For test row: form 90% prediction set
      - "up" in set iff score_up_test <= q_(1-α)(score_up_calib_subset)
        ↔ pred_test >= -q_(1-α)(-pred_calib_up_correct)
        ↔ pred_test >= quantile(pred_up_correct, α)
      - "dn" in set iff pred_test <= quantile(pred_dn_correct, 1-α)
    Singleton {up} → action 2 (long); Singleton {dn} → action 0 (short); else flat (1).
    """
    # Per-trade fee in normalized units (approx 2*FEE for unit price)
    # For "up" to be profitable: true_dmid > fee_long_cost
    # fee_long ≈ FEE * (mp_th+1 + mp_t+1) / (mp_t+1) ≈ 2*FEE for prices ~0
    fee_long = FEE * np.abs((mp_th_calib + 1.0) + (mp_t_calib + 1.0)) / (mp_t_calib + 1.0)
    # PnL of long = (mp_th - mp_t)/(mp_t+1) - fee_long
    pnl_long = (mp_th_calib - mp_t_calib) / (mp_t_calib + 1.0) - fee_long
    pnl_short = -(mp_th_calib - mp_t_calib) / (mp_t_calib + 1.0) - fee_long

    up_mask = pnl_long > 0
    dn_mask = pnl_short > 0

    pred_up_correct = pred_calib[up_mask]
    pred_dn_correct = pred_calib[dn_mask]

    if len(pred_up_correct) < 10 or len(pred_dn_correct) < 10:
        # not enough calibration → flat
        return np.full(len(pred_test), 1, dtype=np.int8)

    # Quantiles for prediction set membership at confidence 1-α
    # "up" in set iff pred_test >= q_α(pred_up_correct)
    q_up = float(np.quantile(pred_up_correct, alpha))
    # "dn" in set iff pred_test <= q_(1-α)(pred_dn_correct)
    q_dn = float(np.quantile(pred_dn_correct, 1.0 - alpha))

    in_set_up = pred_test >= q_up
    in_set_dn = pred_test <= q_dn

    # Singleton check
    action = np.full(len(pred_test), 1, dtype=np.int8)  # default flat
    singleton_up = in_set_up & ~in_set_dn
    singleton_dn = in_set_dn & ~in_set_up
    action[singleton_up] = 2  # long
    action[singleton_dn] = 0  # short
    return action, q_up, q_dn


def conformal_select_2fold_per_sym(folds, alpha):
    """2-fold cross-conformal by date: split val dates into halves.
    Each row gets an action computed using the OTHER half as calibration.

    Returns: list of action arrays per sym, list of (q_up, q_dn) per fold.
    """
    actions_per_sym = []
    quantiles_per_sym = []
    for f in folds:
        pred = f["pred"]
        true_dmid = f["true_dmid"]
        mp_t = f["mp_t"]
        mp_th = f["mp_th"]
        date = f["date"]
        unique_dates = np.sort(np.unique(date))
        # split dates into 2 halves
        mid = len(unique_dates) // 2
        half_A_dates = set(unique_dates[:mid].tolist())
        half_B_dates = set(unique_dates[mid:].tolist())
        is_A = np.array([d in half_A_dates for d in date])
        is_B = ~is_A

        action = np.full(len(pred), 1, dtype=np.int8)

        # Test = A, calibrate on B
        if is_A.sum() > 0 and is_B.sum() > 0:
            res_a = conformal_singleton_action(
                pred[is_B], true_dmid[is_B], mp_t[is_B], mp_th[is_B],
                pred[is_A], alpha,
            )
            if isinstance(res_a, tuple):
                action_A, q_up_BA, q_dn_BA = res_a
                action[is_A] = action_A
            else:
                action[is_A] = res_a
                q_up_BA = q_dn_BA = float("nan")

            # Test = B, calibrate on A
            res_b = conformal_singleton_action(
                pred[is_A], true_dmid[is_A], mp_t[is_A], mp_th[is_A],
                pred[is_B], alpha,
            )
            if isinstance(res_b, tuple):
                action_B, q_up_AB, q_dn_AB = res_b
                action[is_B] = action_B
            else:
                action[is_B] = res_b
                q_up_AB = q_dn_AB = float("nan")

            quantiles_per_sym.append({
                "sym": f["sym"],
                "q_up_calibB_testA": q_up_BA,
                "q_dn_calibB_testA": q_dn_BA,
                "q_up_calibA_testB": q_up_AB,
                "q_dn_calibA_testB": q_dn_AB,
            })
        actions_per_sym.append(action)
    return actions_per_sym, quantiles_per_sym


def load_avg(paths):
    base = pd.read_parquet(paths[0])
    n = len(base)
    p = np.zeros(n, dtype=np.float64)
    for path in paths:
        df = pd.read_parquet(path)
        if len(df) != n:
            raise RuntimeError(f"row mismatch {path}: {len(df)} vs {n}")
        p += df["pred_dmid_norm"].to_numpy(np.float64)
    p /= len(paths)
    base = base.copy()
    base["pred_dmid_norm"] = p.astype(np.float32)
    return base


def evaluate_de_baseline(df, label):
    folds = split_by_sym(df)
    obj_fn = make_obj_loso_asym(folds)
    bounds = [(-3 * FEE, 3 * FEE), (-3 * FEE, 3 * FEE)]
    runs = de_search(obj_fn, bounds)
    best = runs[0]
    actions = []
    for f in folds:
        a = ev_gate_asymmetric(f["pred"], best["thr_up"], best["thr_dn"])
        actions.append(a)
    per_sym, actives, wins = per_sym_pnl_from_action(folds, actions)
    total = float(sum(per_sym))
    print(f"\n=== {label} ===", flush=True)
    print(f"  thr_up={best['thr_up']:+.6f}  thr_dn={best['thr_dn']:+.6f}", flush=True)
    print(f"  per-sym PnL: {[f'{v:+.4f}' for v in per_sym]}", flush=True)
    print(f"  per-sym active rate: {[f'{v:.3f}' for v in actives]}", flush=True)
    print(f"  per-sym win rate: {[f'{v:.3f}' for v in wins]}", flush=True)
    print(f"  TOTAL = {total:+.4f}  min_per_sym = {min(per_sym):+.4f}", flush=True)
    return {
        "label": label,
        "thr_up": best["thr_up"],
        "thr_dn": best["thr_dn"],
        "per_sym_pnl": per_sym,
        "per_sym_active_rate": actives,
        "per_sym_win_rate": wins,
        "total_loso_equiv": total,
        "min_per_sym": float(min(per_sym)),
        "active_rate": float(np.mean(actives)),
        "win_rate": float(np.mean(wins)),
    }


def evaluate_conformal(df, label, alpha):
    folds = split_by_sym(df)
    actions, quantiles = conformal_select_2fold_per_sym(folds, alpha)
    per_sym, actives, wins = per_sym_pnl_from_action(folds, actions)
    total = float(sum(per_sym))
    print(f"\n=== {label} (alpha={alpha:.3f}) ===", flush=True)
    print(f"  per-sym PnL: {[f'{v:+.4f}' for v in per_sym]}", flush=True)
    print(f"  per-sym active rate: {[f'{v:.3f}' for v in actives]}", flush=True)
    print(f"  per-sym win rate: {[f'{v:.3f}' for v in wins]}", flush=True)
    print(f"  TOTAL = {total:+.4f}  min_per_sym = {min(per_sym):+.4f}", flush=True)
    return {
        "label": label,
        "alpha": alpha,
        "per_sym_pnl": per_sym,
        "per_sym_active_rate": actives,
        "per_sym_win_rate": wins,
        "total_loso_equiv": total,
        "min_per_sym": float(min(per_sym)),
        "active_rate": float(np.mean(actives)),
        "win_rate": float(np.mean(wins)),
        "quantiles": quantiles,
    }


def main():
    print(f"Loading T87 NN preds (5-seed avg)...", flush=True)
    t87_paths = [os.path.join(T87_DIR, f"pred_T87_seed{s}_main.parquet") for s in SEEDS]
    df_t87 = load_avg(t87_paths)
    print(f"  T87: n={len(df_t87):,}", flush=True)

    print(f"Loading T75 LGB L2 preds (5-seed avg)...", flush=True)
    t75_paths = [os.path.join(T75_DIR, f"pred_T75_seed{s}.parquet") for s in SEEDS]
    df_t75 = load_avg(t75_paths)
    print(f"  T75: n={len(df_t75):,}", flush=True)

    # Sanity: align row order
    if len(df_t87) != len(df_t75):
        raise RuntimeError(f"T87/T75 row mismatch: {len(df_t87)} vs {len(df_t75)}")
    # iter_015 v1 stack: (1.0 * T87 + 1.5 * T75) / 2.5
    p_combined = (1.0 * df_t87["pred_dmid_norm"].to_numpy(np.float64)
                  + 1.5 * df_t75["pred_dmid_norm"].to_numpy(np.float64)) / 2.5
    df_stack = df_t87.copy()
    df_stack["pred_dmid_norm"] = p_combined.astype(np.float32)
    print(f"\niter_015 v1 stack pred: mean={p_combined.mean():.6e}  std={p_combined.std():.6e}", flush=True)

    results = {}

    # Baseline: DE asym thresh on iter_015 v1 stack
    results["baseline_de"] = evaluate_de_baseline(df_stack, "BASELINE iter_015 v1 stack (DE asym)")

    # Conformal sweep over alpha
    alphas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    results["conformal_sweep"] = {}
    for alpha in alphas:
        res = evaluate_conformal(df_stack, f"CONFORMAL iter_015 v1 stack", alpha)
        results["conformal_sweep"][f"alpha_{alpha:.2f}"] = res

    # pick best conformal
    best_alpha = None
    best_total = -1e18
    for k, v in results["conformal_sweep"].items():
        if v["total_loso_equiv"] > best_total:
            best_total = v["total_loso_equiv"]
            best_alpha = v["alpha"]
    results["best_conformal"] = {
        "alpha": best_alpha,
        "total_loso_equiv": best_total,
    }
    de_total = results["baseline_de"]["total_loso_equiv"]
    delta = best_total - de_total

    print("\n=== SUMMARY ===", flush=True)
    print(f"  baseline DE      = {de_total:+.4f}", flush=True)
    print(f"  best conformal α = {best_alpha:.2f}: {best_total:+.4f}", flush=True)
    print(f"  delta            = {delta:+.4f}", flush=True)
    for k, v in results["conformal_sweep"].items():
        print(f"  {k}: total={v['total_loso_equiv']:+.4f}  active={v['active_rate']:.3f}  "
              f"win={v['win_rate']:.3f}  min_sym={v['min_per_sym']:+.4f}", flush=True)

    results["summary"] = {
        "baseline_de_loso": de_total,
        "best_conformal_loso": best_total,
        "best_alpha": best_alpha,
        "delta_loso": delta,
        "baseline_active_rate": results["baseline_de"]["active_rate"],
        "baseline_win_rate": results["baseline_de"]["win_rate"],
        "conformal_active_rate": results["conformal_sweep"][f"alpha_{best_alpha:.2f}"]["active_rate"],
        "conformal_win_rate": results["conformal_sweep"][f"alpha_{best_alpha:.2f}"]["win_rate"],
    }

    out = os.path.join(EXP_DIR, "results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}", flush=True)


if __name__ == "__main__":
    main()
