"""Conformal Selective Trading v3: per-sym beta calibration + deployable OOS combo.

Adds:
  V6 = OOS DE thresh + abstain band (truly deployable, no test peek)
  V7 = Per-sym beta calibration: pick β per sym on calib half, apply to test half
  V8 = Per-sym (alpha, beta) joint sweep

Compare against V0 (DE in-sample) and V1 (DE OOS) baselines.

Goal: find variant with best (PnL, min_per_sym, win_rate) trade-off that's
truly out-of-sample for fair platform expectation.
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


def date_split_2fold(folds):
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


def make_obj_subset(folds, masks):
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


# ============ V6: OOS DE thresh + abstain band ============

def variant_V6_oos_de_abstain(folds, beta_list, label="V6 OOS DE + abstain"):
    """For each fold, tune DE on calib, apply (thr+beta*sigma) to test."""
    masks_A, masks_B = date_split_2fold(folds)
    bounds = [(-3 * FEE, 3 * FEE), (-3 * FEE, 3 * FEE)]
    obj_A = make_obj_subset(folds, masks_A)
    runs_A = de_search(obj_A, bounds)
    best_A = runs_A[0]
    obj_B = make_obj_subset(folds, masks_B)
    runs_B = de_search(obj_B, bounds)
    best_B = runs_B[0]
    print(f"  tune-on-A thr_up={best_A['thr_up']:+.6f} thr_dn={best_A['thr_dn']:+.6f}", flush=True)
    print(f"  tune-on-B thr_up={best_B['thr_up']:+.6f} thr_dn={best_B['thr_dn']:+.6f}", flush=True)

    out_by_beta = {}
    for beta in beta_list:
        actions = []
        for f, mA, mB in zip(folds, masks_A, masks_B):
            sigma = float(np.std(f["pred"]))
            band = beta * sigma
            a = np.full(f["n"], 1, dtype=np.int8)
            # B-tuned thresh applied to A side, with abstain band
            a[mA] = ev_gate_asymmetric(f["pred"][mA],
                                        best_B["thr_up"] + band, best_B["thr_dn"] + band)
            # A-tuned thresh applied to B side
            a[mB] = ev_gate_asymmetric(f["pred"][mB],
                                        best_A["thr_up"] + band, best_A["thr_dn"] + band)
            actions.append(a)
        per_sym, actives, wins = per_sym_pnl_from_action(folds, actions)
        total = float(sum(per_sym))
        print(f"  β={beta:.2f}: total={total:+.4f} min_sym={min(per_sym):+.4f} "
              f"active={np.mean(actives):.3f} win={np.mean(wins):.3f}", flush=True)
        out_by_beta[f"beta_{beta:.2f}"] = {
            "beta": beta, "per_sym_pnl": per_sym, "per_sym_active_rate": actives,
            "per_sym_win_rate": wins, "total_loso_equiv": total,
            "min_per_sym": float(min(per_sym)),
            "active_rate": float(np.mean(actives)), "win_rate": float(np.mean(wins)),
        }
    print(f"=== {label} ===", flush=True)
    return {"thr_up_A": best_A["thr_up"], "thr_dn_A": best_A["thr_dn"],
            "thr_up_B": best_B["thr_up"], "thr_dn_B": best_B["thr_dn"],
            "by_beta": out_by_beta}


# ============ V7: Per-sym beta calibration ============

def variant_V7_per_sym_beta(folds, beta_grid, label="V7 per-sym β calibration"):
    """For each sym, tune DE on full data (in-sample, like V0), then pick β per sym
    that maximizes calibration-half PnL. Apply chosen β to test-half. Concat.

    This separates β-tuning from threshold-tuning. Threshold = V0 in-sample (matches iter_015 v1).
    """
    # First, get in-sample thresh
    masks_full = [np.ones(f["n"], dtype=bool) for f in folds]
    obj_full = make_obj_subset(folds, masks_full)
    bounds = [(-3 * FEE, 3 * FEE), (-3 * FEE, 3 * FEE)]
    runs = de_search(obj_full, bounds)
    best = runs[0]
    thr_up = best["thr_up"]
    thr_dn = best["thr_dn"]

    masks_A, masks_B = date_split_2fold(folds)

    # For each sym: find β that max PnL on A → apply to B; β that max PnL on B → apply to A
    actions_per_sym = []
    chosen_betas_A = []  # β chosen using A as eval (B as held-out)
    chosen_betas_B = []
    for f, mA, mB in zip(folds, masks_A, masks_B):
        sigma = float(np.std(f["pred"]))
        # Pick β by maximizing PnL on calib partition (A) — apply to B
        best_b_for_B = 0.0
        best_pnl_B = -1e18
        for b in beta_grid:
            a_calib = ev_gate_asymmetric(f["pred"][mA], thr_up + b * sigma, thr_dn + b * sigma)
            pnl = vectorized_pnl(a_calib, f["mp_t"][mA], f["mp_th"][mA]).sum()
            if pnl > best_pnl_B:
                best_pnl_B = pnl
                best_b_for_B = b
        # Symmetric: pick β by max PnL on B → apply to A
        best_b_for_A = 0.0
        best_pnl_A = -1e18
        for b in beta_grid:
            a_calib = ev_gate_asymmetric(f["pred"][mB], thr_up + b * sigma, thr_dn + b * sigma)
            pnl = vectorized_pnl(a_calib, f["mp_t"][mB], f["mp_th"][mB]).sum()
            if pnl > best_pnl_A:
                best_pnl_A = pnl
                best_b_for_A = b
        chosen_betas_A.append(best_b_for_A)
        chosen_betas_B.append(best_b_for_B)
        action = np.full(f["n"], 1, dtype=np.int8)
        action[mA] = ev_gate_asymmetric(f["pred"][mA],
                                         thr_up + best_b_for_A * sigma,
                                         thr_dn + best_b_for_A * sigma)
        action[mB] = ev_gate_asymmetric(f["pred"][mB],
                                         thr_up + best_b_for_B * sigma,
                                         thr_dn + best_b_for_B * sigma)
        actions_per_sym.append(action)
    per_sym, actives, wins = per_sym_pnl_from_action(folds, actions_per_sym)
    total = float(sum(per_sym))
    print(f"\n=== {label} ===", flush=True)
    print(f"  thr_up={thr_up:+.6f} thr_dn={thr_dn:+.6f}", flush=True)
    print(f"  chosen β-for-A: {chosen_betas_A}  β-for-B: {chosen_betas_B}", flush=True)
    print(f"  per-sym PnL: {[f'{v:+.4f}' for v in per_sym]}", flush=True)
    print(f"  TOTAL={total:+.4f} min_sym={min(per_sym):+.4f} "
          f"active={np.mean(actives):.3f} win={np.mean(wins):.3f}", flush=True)
    return {"label": label, "thr_up": thr_up, "thr_dn": thr_dn,
            "betas_for_A": chosen_betas_A, "betas_for_B": chosen_betas_B,
            "per_sym_pnl": per_sym, "per_sym_active_rate": actives,
            "per_sym_win_rate": wins, "total_loso_equiv": total,
            "min_per_sym": float(min(per_sym)),
            "active_rate": float(np.mean(actives)), "win_rate": float(np.mean(wins))}


# ============ V8: Per-sym fixed-β grid sweep (no calibration, just info) ============

def variant_V8_grid_per_sym(folds, beta_grid, label="V8 per-sym β grid (full-data eval)"):
    """For diagnostic: at each β, what's per-sym PnL? Helps understand which β suits each sym."""
    # Use in-sample DE thresh (matches V0)
    masks_full = [np.ones(f["n"], dtype=bool) for f in folds]
    obj_full = make_obj_subset(folds, masks_full)
    bounds = [(-3 * FEE, 3 * FEE), (-3 * FEE, 3 * FEE)]
    runs = de_search(obj_full, bounds)
    best = runs[0]
    thr_up = best["thr_up"]
    thr_dn = best["thr_dn"]
    out = {}
    for f in folds:
        sigma = float(np.std(f["pred"]))
        sym_results = {}
        for b in beta_grid:
            a = ev_gate_asymmetric(f["pred"], thr_up + b * sigma, thr_dn + b * sigma)
            pnl = float(vectorized_pnl(a, f["mp_t"], f["mp_th"]).sum())
            n_active = int((a != 1).sum())
            sym_results[f"beta_{b:.2f}"] = {
                "pnl": pnl, "active": n_active / len(a),
                "win": float((vectorized_pnl(a, f["mp_t"], f["mp_th"])[a != 1] > 0).mean())
                       if n_active > 0 else 0.0,
            }
        out[f"sym_{f['sym']}"] = sym_results
    # Report best β per sym
    best_per_sym = {}
    for sym_key, sym_data in out.items():
        best_b = max(sym_data, key=lambda k: sym_data[k]["pnl"])
        best_per_sym[sym_key] = {"best_beta": best_b, **sym_data[best_b]}
    print(f"\n=== {label} ===", flush=True)
    for k, v in best_per_sym.items():
        print(f"  {k}: best β={v['best_beta']}  pnl={v['pnl']:+.4f}  "
              f"active={v['active']:.3f}  win={v['win']:.3f}", flush=True)
    return {"thr_up": thr_up, "thr_dn": thr_dn, "per_sym_grid": out, "best_per_sym": best_per_sym}


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
    print(f"iter_015 v1 stack: n={len(df_stack):,}", flush=True)

    folds = split_by_sym(df_stack)
    results = {}

    # V6: OOS DE + abstain
    print("\n=== V6 OOS DE + abstain band sweep ===", flush=True)
    results["V6_oos_de_abstain"] = variant_V6_oos_de_abstain(
        folds, beta_list=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    )

    # V7: per-sym β calibration
    beta_grid = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70]
    results["V7_per_sym_beta"] = variant_V7_per_sym_beta(folds, beta_grid)

    # V8: per-sym β grid (diagnostic)
    results["V8_per_sym_grid"] = variant_V8_grid_per_sym(folds, beta_grid)

    # SUMMARY
    print("\n========================= SUMMARY =========================", flush=True)
    print("V6 OOS DE + abstain (deployable, no test peek):", flush=True)
    for k, v in results["V6_oos_de_abstain"]["by_beta"].items():
        print(f"  {k}: total={v['total_loso_equiv']:+.4f}  min_sym={v['min_per_sym']:+.4f}  "
              f"active={v['active_rate']:.3f}  win={v['win_rate']:.3f}", flush=True)
    print(f"V7 per-sym β calibration: total={results['V7_per_sym_beta']['total_loso_equiv']:+.4f}  "
          f"min_sym={results['V7_per_sym_beta']['min_per_sym']:+.4f}  "
          f"active={results['V7_per_sym_beta']['active_rate']:.3f}  "
          f"win={results['V7_per_sym_beta']['win_rate']:.3f}", flush=True)

    out = os.path.join(EXP_DIR, "results_v3.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}", flush=True)


if __name__ == "__main__":
    main()
