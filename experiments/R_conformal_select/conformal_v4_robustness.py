"""Robustness check: 4-fold CV per-sym β + alt date splits.
Validates V7 result (+40.46 LOSO) is not an artifact of the specific 2-fold split.
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
            "true_dmid": sub["true_dmid_norm"].to_numpy(np.float64),
            "mp_t": sub["midprice_t"].to_numpy(np.float64),
            "mp_th": sub["midprice_th"].to_numpy(np.float64),
            "date": sub["date"].to_numpy(np.int64),
            "n": len(sub),
        })
    return out


def make_obj_full(folds):
    def f(x):
        thr_up, thr_dn = x
        s = 0.0
        for ff in folds:
            a = ev_gate_asymmetric(ff["pred"], thr_up, thr_dn)
            s += vectorized_pnl(a, ff["mp_t"], ff["mp_th"]).sum()
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
        runs.append({"seed": int(sd), "thr_up": float(res.x[0]),
                     "thr_dn": float(res.x[1]), "obj_val": float(-res.fun)})
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


def kfold_per_sym_beta(folds, k, beta_grid, thr_up, thr_dn):
    """K-fold CV by date: for each fold, pick β per sym using other K-1 folds, apply to held fold."""
    actions_per_sym = []
    chosen_betas_per_sym = []
    for f in folds:
        unique_dates = np.sort(np.unique(f["date"]))
        n_d = len(unique_dates)
        # K date-folds
        date_folds = []
        for i in range(k):
            lo = i * n_d // k
            hi = (i + 1) * n_d // k
            fold_dates = set(unique_dates[lo:hi].tolist())
            mask = np.array([d in fold_dates for d in f["date"]])
            date_folds.append(mask)
        sigma = float(np.std(f["pred"]))
        action = np.full(f["n"], 1, dtype=np.int8)
        chosen_betas = []
        for fold_idx in range(k):
            test_mask = date_folds[fold_idx]
            calib_mask = ~test_mask
            # Pick β maximizing PnL on calib
            best_b = 0.0
            best_pnl = -1e18
            for b in beta_grid:
                a = ev_gate_asymmetric(f["pred"][calib_mask],
                                        thr_up + b * sigma, thr_dn + b * sigma)
                pnl = vectorized_pnl(a, f["mp_t"][calib_mask], f["mp_th"][calib_mask]).sum()
                if pnl > best_pnl:
                    best_pnl = pnl
                    best_b = b
            chosen_betas.append(best_b)
            # Apply to test
            action[test_mask] = ev_gate_asymmetric(f["pred"][test_mask],
                                                    thr_up + best_b * sigma,
                                                    thr_dn + best_b * sigma)
        actions_per_sym.append(action)
        chosen_betas_per_sym.append(chosen_betas)
    return actions_per_sym, chosen_betas_per_sym


def shifted_2fold_per_sym(folds, beta_grid, thr_up, thr_dn, shift):
    """Like V7 but with rolled date order (test robustness)."""
    actions_per_sym = []
    chosen_betas_A = []
    chosen_betas_B = []
    for f in folds:
        unique_dates = np.sort(np.unique(f["date"]))
        # roll the date order before splitting
        rolled = np.roll(unique_dates, shift)
        mid = len(rolled) // 2
        A_dates = set(rolled[:mid].tolist())
        B_dates = set(rolled[mid:].tolist())
        is_A = np.array([d in A_dates for d in f["date"]])
        is_B = ~is_A
        sigma = float(np.std(f["pred"]))
        # Pick β on A → apply to B
        best_b_for_B = 0.0
        best_pnl_B = -1e18
        for b in beta_grid:
            a = ev_gate_asymmetric(f["pred"][is_A], thr_up + b * sigma, thr_dn + b * sigma)
            pnl = vectorized_pnl(a, f["mp_t"][is_A], f["mp_th"][is_A]).sum()
            if pnl > best_pnl_B:
                best_pnl_B = pnl
                best_b_for_B = b
        # Pick β on B → apply to A
        best_b_for_A = 0.0
        best_pnl_A = -1e18
        for b in beta_grid:
            a = ev_gate_asymmetric(f["pred"][is_B], thr_up + b * sigma, thr_dn + b * sigma)
            pnl = vectorized_pnl(a, f["mp_t"][is_B], f["mp_th"][is_B]).sum()
            if pnl > best_pnl_A:
                best_pnl_A = pnl
                best_b_for_A = b
        chosen_betas_A.append(best_b_for_A)
        chosen_betas_B.append(best_b_for_B)
        action = np.full(f["n"], 1, dtype=np.int8)
        action[is_A] = ev_gate_asymmetric(f["pred"][is_A],
                                           thr_up + best_b_for_A * sigma,
                                           thr_dn + best_b_for_A * sigma)
        action[is_B] = ev_gate_asymmetric(f["pred"][is_B],
                                           thr_up + best_b_for_B * sigma,
                                           thr_dn + best_b_for_B * sigma)
        actions_per_sym.append(action)
    return actions_per_sym, chosen_betas_A, chosen_betas_B


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
    t87_paths = [os.path.join(T87_DIR, f"pred_T87_seed{s}_main.parquet") for s in SEEDS]
    df_t87 = load_avg(t87_paths)
    t75_paths = [os.path.join(T75_DIR, f"pred_T75_seed{s}.parquet") for s in SEEDS]
    df_t75 = load_avg(t75_paths)
    p_combined = (1.0 * df_t87["pred_dmid_norm"].to_numpy(np.float64)
                  + 1.5 * df_t75["pred_dmid_norm"].to_numpy(np.float64)) / 2.5
    df_stack = df_t87.copy()
    df_stack["pred_dmid_norm"] = p_combined.astype(np.float32)
    folds = split_by_sym(df_stack)

    obj_full = make_obj_full(folds)
    bounds = [(-3 * FEE, 3 * FEE), (-3 * FEE, 3 * FEE)]
    runs = de_search(obj_full, bounds)
    best = runs[0]
    thr_up = best["thr_up"]
    thr_dn = best["thr_dn"]
    print(f"In-sample DE: thr_up={thr_up:+.6f} thr_dn={thr_dn:+.6f}", flush=True)

    beta_grid = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70]

    results = {"thr_up": thr_up, "thr_dn": thr_dn}

    # Baseline (no abstain): action = ev_gate at thresh
    actions_base = [ev_gate_asymmetric(f["pred"], thr_up, thr_dn) for f in folds]
    per_sym_base, actives_base, wins_base = per_sym_pnl_from_action(folds, actions_base)
    total_base = float(sum(per_sym_base))
    print(f"\nBASELINE (no abstain): total={total_base:+.4f} min_sym={min(per_sym_base):+.4f} "
          f"active={np.mean(actives_base):.3f} win={np.mean(wins_base):.3f}", flush=True)
    results["baseline"] = {"total": total_base, "per_sym": per_sym_base,
                           "actives": actives_base, "wins": wins_base,
                           "min_sym": min(per_sym_base),
                           "active_rate": float(np.mean(actives_base)),
                           "win_rate": float(np.mean(wins_base))}

    # ----- 2-fold with shifted splits -----
    print("\n=== 2-fold per-sym β with shifted date splits ===", flush=True)
    shift_results = {}
    for shift in [0, 6, 12, 18]:
        actions, ba, bb = shifted_2fold_per_sym(folds, beta_grid, thr_up, thr_dn, shift)
        per_sym, actives, wins = per_sym_pnl_from_action(folds, actions)
        total = float(sum(per_sym))
        print(f"  shift={shift}: total={total:+.4f} min_sym={min(per_sym):+.4f} "
              f"active={np.mean(actives):.3f} win={np.mean(wins):.3f}", flush=True)
        print(f"    βA={ba} βB={bb}", flush=True)
        shift_results[f"shift_{shift}"] = {
            "total": total, "min_sym": float(min(per_sym)),
            "per_sym": per_sym, "actives": actives, "wins": wins,
            "active_rate": float(np.mean(actives)),
            "win_rate": float(np.mean(wins)),
            "betas_A": ba, "betas_B": bb,
        }
    results["shifted_2fold"] = shift_results

    # ----- K-fold CV per-sym β -----
    print("\n=== K-fold CV per-sym β ===", flush=True)
    kfold_results = {}
    for k in [3, 4, 6]:
        actions, betas = kfold_per_sym_beta(folds, k, beta_grid, thr_up, thr_dn)
        per_sym, actives, wins = per_sym_pnl_from_action(folds, actions)
        total = float(sum(per_sym))
        print(f"  k={k}: total={total:+.4f} min_sym={min(per_sym):+.4f} "
              f"active={np.mean(actives):.3f} win={np.mean(wins):.3f}", flush=True)
        for sym_idx, b in enumerate(betas):
            print(f"    sym{sym_idx} β per fold: {b}", flush=True)
        kfold_results[f"k_{k}"] = {
            "total": total, "min_sym": float(min(per_sym)),
            "per_sym": per_sym, "actives": actives, "wins": wins,
            "active_rate": float(np.mean(actives)),
            "win_rate": float(np.mean(wins)),
            "betas_per_fold": betas,
        }
    results["kfold"] = kfold_results

    # SUMMARY: which config gives most stable result above baseline?
    print("\n=== STABILITY SUMMARY ===", flush=True)
    print(f"  baseline (no abstain):  {total_base:+.4f}", flush=True)
    for k, v in shift_results.items():
        delta = v["total"] - total_base
        print(f"  shift {k}:  {v['total']:+.4f}  (Δ={delta:+.4f})  min_sym={v['min_sym']:+.4f}", flush=True)
    for k, v in kfold_results.items():
        delta = v["total"] - total_base
        print(f"  {k}: {v['total']:+.4f}  (Δ={delta:+.4f})  min_sym={v['min_sym']:+.4f}", flush=True)

    out = os.path.join(EXP_DIR, "results_v4_robustness.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}", flush=True)


if __name__ == "__main__":
    main()
