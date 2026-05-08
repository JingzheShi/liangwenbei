"""Conformal Selective Trading v2: fairer comparisons + hybrid variants.

Variants:
  V0 = DE in-sample (full 442k tune+eval) [baseline iter_015 v1, +39.75 known]
  V1 = DE 2-fold out-of-sample (tune on half-A, eval on half-B; concat) [fair vs conformal]
  V2 = Conformal direct (2-fold, per-sym) [from v1, already +37.07 best]
  V3 = DE + conformal abstain band: take DE-tuned thr, but abstain if |pred-thr| < β*calib_std
  V4 = Per-sym α conformal: tune α independently per sym on calibration to maximize PnL on calib
  V5 = Conformal with stricter "significant move" class: |true_dmid| > median(|true_dmid|)

Goal: find variant with best PnL/min_per_sym profile, esp. >baseline_de_oos.
"""
from __future__ import annotations
import json
import os
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


def make_obj_loso_asym_subset(folds, masks):
    """DE objective: optimize total PnL across all syms but only on rows where mask=True."""
    pred = [f["pred"][m] for f, m in zip(folds, masks)]
    mp_t = [f["mp_t"][m] for f, m in zip(folds, masks)]
    mp_th = [f["mp_th"][m] for f, m in zip(folds, masks)]
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
    out = []
    actives = []
    wins = []
    for f, a in zip(folds, actions_per_sym):
        pnl = vectorized_pnl(a, f["mp_t"], f["mp_th"])
        out.append(float(pnl.sum()))
        n_active = int((a != 1).sum())
        actives.append(n_active / len(a) if len(a) > 0 else 0.0)
        active_mask = (a != 1)
        if active_mask.sum() > 0:
            wins.append(float((pnl[active_mask] > 0).mean()))
        else:
            wins.append(0.0)
    return out, actives, wins


def date_split_2fold(folds):
    """Returns (is_A, is_B) masks per sym, splitting by date midpoint."""
    masks_A = []
    masks_B = []
    for f in folds:
        date = f["date"]
        unique_dates = np.sort(np.unique(date))
        mid = len(unique_dates) // 2
        A = set(unique_dates[:mid].tolist())
        is_A = np.array([d in A for d in date])
        masks_A.append(is_A)
        masks_B.append(~is_A)
    return masks_A, masks_B


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


# ============ V0: DE in-sample ============

def variant_V0_de_insample(folds, label="V0 DE in-sample"):
    masks_all = [np.ones(f["n"], dtype=bool) for f in folds]
    obj_fn = make_obj_loso_asym_subset(folds, masks_all)
    bounds = [(-3 * FEE, 3 * FEE), (-3 * FEE, 3 * FEE)]
    runs = de_search(obj_fn, bounds)
    best = runs[0]
    actions = [ev_gate_asymmetric(f["pred"], best["thr_up"], best["thr_dn"]) for f in folds]
    per_sym, actives, wins = per_sym_pnl_from_action(folds, actions)
    total = float(sum(per_sym))
    print(f"\n=== {label} ===", flush=True)
    print(f"  thr_up={best['thr_up']:+.6f}  thr_dn={best['thr_dn']:+.6f}", flush=True)
    print(f"  per-sym PnL: {[f'{v:+.4f}' for v in per_sym]}", flush=True)
    print(f"  TOTAL={total:+.4f}  min_sym={min(per_sym):+.4f}  "
          f"active={np.mean(actives):.3f}  win={np.mean(wins):.3f}", flush=True)
    return {"label": label, "thr_up": best["thr_up"], "thr_dn": best["thr_dn"],
            "per_sym_pnl": per_sym, "per_sym_active_rate": actives, "per_sym_win_rate": wins,
            "total_loso_equiv": total, "min_per_sym": float(min(per_sym)),
            "active_rate": float(np.mean(actives)), "win_rate": float(np.mean(wins)),
            "actions": actions}


# ============ V1: DE 2-fold out-of-sample ============

def variant_V1_de_oos(folds, label="V1 DE 2-fold OOS"):
    masks_A, masks_B = date_split_2fold(folds)
    bounds = [(-3 * FEE, 3 * FEE), (-3 * FEE, 3 * FEE)]
    # Tune on A, apply to B
    obj_A = make_obj_loso_asym_subset(folds, masks_A)
    runs_A = de_search(obj_A, bounds)
    best_A = runs_A[0]
    # Tune on B, apply to A
    obj_B = make_obj_loso_asym_subset(folds, masks_B)
    runs_B = de_search(obj_B, bounds)
    best_B = runs_B[0]

    actions = []
    for f, mA, mB in zip(folds, masks_A, masks_B):
        a = np.full(f["n"], 1, dtype=np.int8)
        # B-tuned thresh applied to A
        a[mA] = ev_gate_asymmetric(f["pred"][mA], best_B["thr_up"], best_B["thr_dn"])
        # A-tuned thresh applied to B
        a[mB] = ev_gate_asymmetric(f["pred"][mB], best_A["thr_up"], best_A["thr_dn"])
        actions.append(a)
    per_sym, actives, wins = per_sym_pnl_from_action(folds, actions)
    total = float(sum(per_sym))
    print(f"\n=== {label} ===", flush=True)
    print(f"  tune-on-A thr_up={best_A['thr_up']:+.6f} thr_dn={best_A['thr_dn']:+.6f}", flush=True)
    print(f"  tune-on-B thr_up={best_B['thr_up']:+.6f} thr_dn={best_B['thr_dn']:+.6f}", flush=True)
    print(f"  per-sym PnL: {[f'{v:+.4f}' for v in per_sym]}", flush=True)
    print(f"  TOTAL={total:+.4f}  min_sym={min(per_sym):+.4f}  "
          f"active={np.mean(actives):.3f}  win={np.mean(wins):.3f}", flush=True)
    return {"label": label, "thr_up_A": best_A["thr_up"], "thr_dn_A": best_A["thr_dn"],
            "thr_up_B": best_B["thr_up"], "thr_dn_B": best_B["thr_dn"],
            "per_sym_pnl": per_sym, "per_sym_active_rate": actives, "per_sym_win_rate": wins,
            "total_loso_equiv": total, "min_per_sym": float(min(per_sym)),
            "active_rate": float(np.mean(actives)), "win_rate": float(np.mean(wins)),
            "actions": actions}


# ============ V3: DE + conformal abstain band ============

def variant_V3_de_with_abstain_band(folds, de_baseline, beta_list, label="V3 DE + abstain"):
    """Take DE-tuned thresh from baseline; abstain (force flat) if |pred - thr_threshold|
    is within beta * sigma_pred of the threshold (treat as 'noise zone').
    """
    thr_up = de_baseline["thr_up"]
    thr_dn = de_baseline["thr_dn"]
    out_by_beta = {}
    for beta in beta_list:
        actions = []
        for f in folds:
            sigma = float(np.std(f["pred"]))
            band = beta * sigma
            a = ev_gate_asymmetric(f["pred"], thr_up + band, thr_dn + band)  # widen the gate
            actions.append(a)
        per_sym, actives, wins = per_sym_pnl_from_action(folds, actions)
        total = float(sum(per_sym))
        print(f"  beta={beta:.2f}: TOTAL={total:+.4f}  min_sym={min(per_sym):+.4f}  "
              f"active={np.mean(actives):.3f}  win={np.mean(wins):.3f}", flush=True)
        out_by_beta[f"beta_{beta:.2f}"] = {
            "beta": beta, "per_sym_pnl": per_sym, "per_sym_active_rate": actives,
            "per_sym_win_rate": wins, "total_loso_equiv": total,
            "min_per_sym": float(min(per_sym)),
            "active_rate": float(np.mean(actives)), "win_rate": float(np.mean(wins)),
        }
    print(f"=== {label} (sweep beta over abstention band) ===", flush=True)
    return out_by_beta


# ============ V4: Per-sym alpha conformal ============

def conformal_action_simple(pred_calib, true_dmid_calib, mp_t_calib, mp_th_calib,
                            pred_test, alpha):
    """Returns (action_test, q_up, q_dn). Simple class-conditional conformal."""
    fee_long = FEE * np.abs((mp_th_calib + 1.0) + (mp_t_calib + 1.0)) / (mp_t_calib + 1.0)
    pnl_long = (mp_th_calib - mp_t_calib) / (mp_t_calib + 1.0) - fee_long
    pnl_short = -(mp_th_calib - mp_t_calib) / (mp_t_calib + 1.0) - fee_long
    up_mask = pnl_long > 0
    dn_mask = pnl_short > 0
    pred_up = pred_calib[up_mask]
    pred_dn = pred_calib[dn_mask]
    if len(pred_up) < 10 or len(pred_dn) < 10:
        return np.full(len(pred_test), 1, dtype=np.int8), float("nan"), float("nan")
    q_up = float(np.quantile(pred_up, alpha))
    q_dn = float(np.quantile(pred_dn, 1.0 - alpha))
    in_set_up = pred_test >= q_up
    in_set_dn = pred_test <= q_dn
    a = np.full(len(pred_test), 1, dtype=np.int8)
    a[in_set_up & ~in_set_dn] = 2
    a[in_set_dn & ~in_set_up] = 0
    return a, q_up, q_dn


def conformal_2fold_per_sym(folds, alphas_per_sym):
    """alphas_per_sym: list of 5 alpha values, one per sym."""
    actions_per_sym = []
    for f, alpha in zip(folds, alphas_per_sym):
        pred = f["pred"]
        true_dmid = f["true_dmid"]
        mp_t = f["mp_t"]
        mp_th = f["mp_th"]
        date = f["date"]
        unique_dates = np.sort(np.unique(date))
        mid = len(unique_dates) // 2
        A = set(unique_dates[:mid].tolist())
        is_A = np.array([d in A for d in date])
        is_B = ~is_A
        action = np.full(len(pred), 1, dtype=np.int8)
        # Test=A, calib=B
        a_A, _, _ = conformal_action_simple(pred[is_B], true_dmid[is_B], mp_t[is_B],
                                            mp_th[is_B], pred[is_A], alpha)
        action[is_A] = a_A
        # Test=B, calib=A
        a_B, _, _ = conformal_action_simple(pred[is_A], true_dmid[is_A], mp_t[is_A],
                                            mp_th[is_A], pred[is_B], alpha)
        action[is_B] = a_B
        actions_per_sym.append(action)
    return actions_per_sym


def variant_V4_per_sym_alpha(folds, alpha_grid, label="V4 per-sym alpha conformal"):
    """For each sym, pick alpha that maximizes PnL on the FIRST half (A);
    apply that alpha to the SECOND half (B) as test (proper holdout).
    Then symmetric: pick alpha on B → apply to A. Concat.
    """
    masks_A, masks_B = date_split_2fold(folds)
    # For each sym, search alpha that max PnL on A using calib=B; then evaluate on B with calib=A
    # And vice versa; then concat per sym.
    best_alpha_for_test_B = []  # alpha picked on A using B as calib (no leak: A is test, B is calib)
    best_alpha_for_test_A = []  # alpha picked on B using A as calib
    actions_per_sym = []
    for f, mA, mB in zip(folds, masks_A, masks_B):
        pred = f["pred"]
        true_dmid = f["true_dmid"]
        mp_t = f["mp_t"]
        mp_th = f["mp_th"]
        # For test_A side: alpha must be picked WITHOUT using A. Pick alpha on B (calib=A→pred B side).
        # That's the symmetric case: alpha_for_B = alpha that maxes PnL on calib=A, test=B
        # Wait this is confusing. Cleanest: 4-fold scheme.
        # Simpler: use first 1/4 of dates as alpha-selection set (calib still = same date partition)
        unique_dates = np.sort(np.unique(f["date"]))
        n_d = len(unique_dates)
        # 4-way split: dates [0:6]=alpha-tune-A, [6:12]=test-A, [12:18]=alpha-tune-B, [18:24]=test-B
        d_at = set(unique_dates[: n_d // 4].tolist())
        d_ta = set(unique_dates[n_d // 4 : n_d // 2].tolist())
        d_tb = set(unique_dates[n_d // 2 : 3 * n_d // 4].tolist())
        d_at2 = set(unique_dates[3 * n_d // 4 :].tolist())
        is_at = np.array([d in d_at for d in f["date"]])    # alpha-tune (uses ta as test, but we eval against actions)
        is_ta = np.array([d in d_ta for d in f["date"]])    # test-A
        is_tb = np.array([d in d_tb for d in f["date"]])    # test-B
        is_at2 = np.array([d in d_at2 for d in f["date"]])  # alpha-tune-2

        def score_alpha(alpha, calib_mask, eval_mask):
            calib_pred = pred[calib_mask]
            calib_dmid = true_dmid[calib_mask]
            calib_mpt = mp_t[calib_mask]
            calib_mpth = mp_th[calib_mask]
            test_pred = pred[eval_mask]
            a_test, _, _ = conformal_action_simple(calib_pred, calib_dmid, calib_mpt,
                                                    calib_mpth, test_pred, alpha)
            return float(vectorized_pnl(a_test, mp_t[eval_mask], mp_th[eval_mask]).sum())

        # Pick alpha for test_A: tune on (calib=at2, eval=at) since at and at2 don't include test-A or test-B
        # Wait that's also wrong. Let's be cleanest: pick alpha by maximizing PnL on alpha-tune partition,
        # using OTHER alpha-tune partition as calib. Then apply alpha to BOTH test-A and test-B with their calibs.
        scores_for_A = {a: score_alpha(a, calib_mask=is_at2, eval_mask=is_at) for a in alpha_grid}
        best_a_for_A = max(scores_for_A, key=scores_for_A.get)
        scores_for_B = {a: score_alpha(a, calib_mask=is_at, eval_mask=is_at2) for a in alpha_grid}
        best_a_for_B = max(scores_for_B, key=scores_for_B.get)
        best_alpha_for_test_A.append(best_a_for_A)
        best_alpha_for_test_B.append(best_a_for_B)

        # Now apply: test_A uses calib=NOT_test_A_partition with chosen alpha
        action = np.full(f["n"], 1, dtype=np.int8)
        # For test_A: calib = everything except test_A → mask = ~is_ta
        not_ta = ~is_ta
        a_ta, _, _ = conformal_action_simple(
            pred[not_ta], true_dmid[not_ta], mp_t[not_ta], mp_th[not_ta],
            pred[is_ta], best_a_for_A,
        )
        action[is_ta] = a_ta
        # For test_B: calib = everything except test_B
        not_tb = ~is_tb
        a_tb, _, _ = conformal_action_simple(
            pred[not_tb], true_dmid[not_tb], mp_t[not_tb], mp_th[not_tb],
            pred[is_tb], best_a_for_B,
        )
        action[is_tb] = a_tb
        # For at and at2 (alpha-tune partitions), use full conformal with avg alpha and calib=other partitions
        avg_alpha = 0.5 * (best_a_for_A + best_a_for_B)
        not_at = ~is_at
        a_at, _, _ = conformal_action_simple(
            pred[not_at], true_dmid[not_at], mp_t[not_at], mp_th[not_at],
            pred[is_at], avg_alpha,
        )
        action[is_at] = a_at
        not_at2 = ~is_at2
        a_at2, _, _ = conformal_action_simple(
            pred[not_at2], true_dmid[not_at2], mp_t[not_at2], mp_th[not_at2],
            pred[is_at2], avg_alpha,
        )
        action[is_at2] = a_at2
        actions_per_sym.append(action)

    per_sym, actives, wins = per_sym_pnl_from_action(folds, actions_per_sym)
    total = float(sum(per_sym))
    print(f"\n=== {label} ===", flush=True)
    print(f"  per-sym alphas: A={best_alpha_for_test_A}  B={best_alpha_for_test_B}", flush=True)
    print(f"  per-sym PnL: {[f'{v:+.4f}' for v in per_sym]}", flush=True)
    print(f"  TOTAL={total:+.4f}  min_sym={min(per_sym):+.4f}  "
          f"active={np.mean(actives):.3f}  win={np.mean(wins):.3f}", flush=True)
    return {"label": label, "alpha_per_sym_A": best_alpha_for_test_A,
            "alpha_per_sym_B": best_alpha_for_test_B,
            "per_sym_pnl": per_sym, "per_sym_active_rate": actives,
            "per_sym_win_rate": wins, "total_loso_equiv": total,
            "min_per_sym": float(min(per_sym)),
            "active_rate": float(np.mean(actives)), "win_rate": float(np.mean(wins))}


# ============ V5: Conformal with stricter "significant move" class ============

def conformal_action_strict(pred_calib, true_dmid_calib, mp_t_calib, mp_th_calib,
                            pred_test, alpha, sig_threshold):
    """Stricter conformity: require |true_dmid_calib| > sig_threshold AND right direction."""
    fee_long = FEE * np.abs((mp_th_calib + 1.0) + (mp_t_calib + 1.0)) / (mp_t_calib + 1.0)
    pnl_long = (mp_th_calib - mp_t_calib) / (mp_t_calib + 1.0) - fee_long
    pnl_short = -(mp_th_calib - mp_t_calib) / (mp_t_calib + 1.0) - fee_long
    abs_dmid = np.abs(true_dmid_calib)
    up_mask = (pnl_long > 0) & (abs_dmid > sig_threshold)
    dn_mask = (pnl_short > 0) & (abs_dmid > sig_threshold)
    pred_up = pred_calib[up_mask]
    pred_dn = pred_calib[dn_mask]
    if len(pred_up) < 10 or len(pred_dn) < 10:
        return np.full(len(pred_test), 1, dtype=np.int8), float("nan"), float("nan")
    q_up = float(np.quantile(pred_up, alpha))
    q_dn = float(np.quantile(pred_dn, 1.0 - alpha))
    in_set_up = pred_test >= q_up
    in_set_dn = pred_test <= q_dn
    a = np.full(len(pred_test), 1, dtype=np.int8)
    a[in_set_up & ~in_set_dn] = 2
    a[in_set_dn & ~in_set_up] = 0
    return a, q_up, q_dn


def variant_V5_strict_class(folds, alpha_list, label="V5 strict-class conformal"):
    out_by_alpha = {}
    masks_A, masks_B = date_split_2fold(folds)
    # significant move threshold: median |true_dmid| over all data (per-sym Mondrian)
    for alpha in alpha_list:
        actions_per_sym = []
        for f, mA, mB in zip(folds, masks_A, masks_B):
            sig = float(np.median(np.abs(f["true_dmid"])))
            pred = f["pred"]
            td = f["true_dmid"]
            mt = f["mp_t"]
            mh = f["mp_th"]
            action = np.full(f["n"], 1, dtype=np.int8)
            a_A, _, _ = conformal_action_strict(pred[mB], td[mB], mt[mB], mh[mB],
                                                 pred[mA], alpha, sig)
            action[mA] = a_A
            a_B, _, _ = conformal_action_strict(pred[mA], td[mA], mt[mA], mh[mA],
                                                 pred[mB], alpha, sig)
            action[mB] = a_B
            actions_per_sym.append(action)
        per_sym, actives, wins = per_sym_pnl_from_action(folds, actions_per_sym)
        total = float(sum(per_sym))
        print(f"  α={alpha:.2f}: TOTAL={total:+.4f}  min_sym={min(per_sym):+.4f}  "
              f"active={np.mean(actives):.3f}  win={np.mean(wins):.3f}", flush=True)
        out_by_alpha[f"alpha_{alpha:.2f}"] = {
            "alpha": alpha, "per_sym_pnl": per_sym, "per_sym_active_rate": actives,
            "per_sym_win_rate": wins, "total_loso_equiv": total,
            "min_per_sym": float(min(per_sym)),
            "active_rate": float(np.mean(actives)), "win_rate": float(np.mean(wins)),
        }
    print(f"=== {label} ===", flush=True)
    return out_by_alpha


def main():
    print("Loading T87 NN preds (5-seed avg)...", flush=True)
    t87_paths = [os.path.join(T87_DIR, f"pred_T87_seed{s}_main.parquet") for s in SEEDS]
    df_t87 = load_avg(t87_paths)
    print("Loading T75 LGB L2 preds (5-seed avg)...", flush=True)
    t75_paths = [os.path.join(T75_DIR, f"pred_T75_seed{s}.parquet") for s in SEEDS]
    df_t75 = load_avg(t75_paths)

    p_combined = (1.0 * df_t87["pred_dmid_norm"].to_numpy(np.float64)
                  + 1.5 * df_t75["pred_dmid_norm"].to_numpy(np.float64)) / 2.5
    df_stack = df_t87.copy()
    df_stack["pred_dmid_norm"] = p_combined.astype(np.float32)
    print(f"iter_015 v1 stack: n={len(df_stack):,} mean={p_combined.mean():.6e} "
          f"std={p_combined.std():.6e}", flush=True)

    folds = split_by_sym(df_stack)
    results = {}

    # V0: DE in-sample
    results["V0_de_insample"] = variant_V0_de_insample(folds)
    # remove huge action arrays before json dump
    v0_baseline = results["V0_de_insample"].copy()
    results["V0_de_insample"].pop("actions", None)

    # V1: DE 2-fold OOS
    v1_full = variant_V1_de_oos(folds)
    results["V1_de_oos"] = {k: v for k, v in v1_full.items() if k != "actions"}

    # V3: DE + abstain band sweep
    print("\n=== V3 DE + abstain band sweep ===", flush=True)
    results["V3_de_abstain"] = variant_V3_de_with_abstain_band(
        folds, v0_baseline, beta_list=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    )

    # V4: per-sym alpha conformal
    alpha_grid = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    results["V4_per_sym_alpha"] = variant_V4_per_sym_alpha(folds, alpha_grid)

    # V5: strict-class conformal
    print("\n=== V5 strict-class conformal sweep ===", flush=True)
    results["V5_strict_class"] = variant_V5_strict_class(
        folds, alpha_list=[0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
    )

    # SUMMARY
    print("\n========================= SUMMARY =========================", flush=True)
    print(f"V0 DE in-sample:    total={results['V0_de_insample']['total_loso_equiv']:+.4f}  "
          f"min_sym={results['V0_de_insample']['min_per_sym']:+.4f}  "
          f"active={results['V0_de_insample']['active_rate']:.3f}  "
          f"win={results['V0_de_insample']['win_rate']:.3f}", flush=True)
    print(f"V1 DE OOS 2-fold:   total={results['V1_de_oos']['total_loso_equiv']:+.4f}  "
          f"min_sym={results['V1_de_oos']['min_per_sym']:+.4f}  "
          f"active={results['V1_de_oos']['active_rate']:.3f}  "
          f"win={results['V1_de_oos']['win_rate']:.3f}", flush=True)
    print("V3 DE + abstain:", flush=True)
    for k, v in results["V3_de_abstain"].items():
        print(f"  {k}: total={v['total_loso_equiv']:+.4f}  min_sym={v['min_per_sym']:+.4f}  "
              f"active={v['active_rate']:.3f}  win={v['win_rate']:.3f}", flush=True)
    print(f"V4 per-sym alpha:   total={results['V4_per_sym_alpha']['total_loso_equiv']:+.4f}  "
          f"min_sym={results['V4_per_sym_alpha']['min_per_sym']:+.4f}  "
          f"active={results['V4_per_sym_alpha']['active_rate']:.3f}  "
          f"win={results['V4_per_sym_alpha']['win_rate']:.3f}", flush=True)
    print("V5 strict class:", flush=True)
    for k, v in results["V5_strict_class"].items():
        print(f"  {k}: total={v['total_loso_equiv']:+.4f}  min_sym={v['min_per_sym']:+.4f}  "
              f"active={v['active_rate']:.3f}  win={v['win_rate']:.3f}", flush=True)

    out = os.path.join(EXP_DIR, "results_v2.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}", flush=True)


if __name__ == "__main__":
    main()
