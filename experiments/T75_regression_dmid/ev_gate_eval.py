"""T75: EV-gate threshold sweep on 5-seed-averaged regression predictions.

Loads pred_T75_seed{S}.parquet for S in 1,7,13,42,100, averages predicted
Δmid_norm across seeds (per row), then computes:
  1) Single + per-sym EV-gate cum_pnl across a sweep of k ∈ {0.5, 1.0, 1.5, 2.0, 2.5, 3.0}
     (asymmetric extension also: independent k_up vs k_dn via differential evolution)
  2) Compares vs iter_012 +26.44 LOSO-equiv

Saves ev_gate_results.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001
ITER012_LOSO = 26.44


def load_avg_pred(seeds):
    base = pd.read_parquet(os.path.join(HERE, f"pred_T75_seed{seeds[0]}.parquet"))
    p_sum = base["pred_dmid_norm"].to_numpy(np.float64).copy()
    n = len(base)
    for s in seeds[1:]:
        df_s = pd.read_parquet(os.path.join(HERE, f"pred_T75_seed{s}.parquet"))
        if len(df_s) != n:
            raise RuntimeError(f"row count mismatch seed={s}: {len(df_s)} vs {n}")
        if not (df_s["sym"].equals(base["sym"]) and df_s["date"].equals(base["date"])
                and df_s["t"].equals(base["t"])):
            raise RuntimeError(f"row order mismatch seed={s}")
        p_sum += df_s["pred_dmid_norm"].to_numpy(np.float64)
    p_avg = p_sum / float(len(seeds))
    out = base.copy()
    out["pred_dmid_norm"] = p_avg.astype(np.float32)
    return out


def split_by_sym(df):
    out = []
    for k in SYMS:
        sub = df[df["sym"] == k]
        out.append({
            "sym": int(k),
            "pred": sub["pred_dmid_norm"].to_numpy(np.float64),
            "label": sub["true_label"].to_numpy(np.int64),
            "mp_t": sub["midprice_t"].to_numpy(np.float64),
            "mp_th": sub["midprice_th"].to_numpy(np.float64),
            "n": len(sub),
        })
    return out


def vectorized_pnl(pred_action, mp_t, mp_th):
    side = pred_action.astype(np.float64) - 1.0
    abs_side = np.abs(side)
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * abs_side * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee_pnl) / denom


def ev_gate_symmetric(pred, k):
    """Action 2 if pred > k * 2*FEE; 0 if pred < -k*2*FEE; else 1."""
    thr = k * 2.0 * FEE
    a = np.full(len(pred), 1, dtype=np.int8)
    a[pred > thr] = 2
    a[pred < -thr] = 0
    return a


def ev_gate_asymmetric(pred, thr_up, thr_dn):
    """Action 2 if pred > thr_up; 0 if pred < -thr_dn; else 1.
    thr_up, thr_dn are POSITIVE numbers (the dn cutoff is at -thr_dn)."""
    a = np.full(len(pred), 1, dtype=np.int8)
    a[pred > thr_up] = 2
    a[pred < -thr_dn] = 0
    return a


def per_sym_pnl_at_k(folds, k):
    per = []
    n_active = []
    for f in folds:
        a = ev_gate_symmetric(f["pred"], k)
        per.append(float(vectorized_pnl(a, f["mp_t"], f["mp_th"]).sum()))
        n_active.append(int((a != 1).sum()))
    return per, n_active


def per_sym_pnl_asym(folds, thr_up, thr_dn):
    per = []
    n_active = []
    for f in folds:
        a = ev_gate_asymmetric(f["pred"], thr_up, thr_dn)
        per.append(float(vectorized_pnl(a, f["mp_t"], f["mp_th"]).sum()))
        n_active.append(int((a != 1).sum()))
    return per, n_active


def make_obj_loso_asym(folds):
    """Maximize sum of per-sym cum_pnl (LOSO-equiv) over (thr_up, thr_dn)."""
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


def make_obj_single_asym(pred, mp_t, mp_th):
    def f(x):
        thr_up, thr_dn = x
        a = ev_gate_asymmetric(pred, thr_up, thr_dn)
        return -float(vectorized_pnl(a, mp_t, mp_th).sum())
    return f


def de_search(objective_fn, bounds, seeds=(0, 1, 2, 7, 42)):
    runs = []
    for sd in seeds:
        t0 = time.time()
        result = differential_evolution(
            objective_fn, bounds=bounds, seed=sd, maxiter=80, popsize=24,
            polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            updating="deferred", workers=1, init="sobol",
        )
        runs.append({
            "seed": int(sd),
            "thr_up": float(result.x[0]),
            "thr_dn": float(result.x[1]),
            "obj_val": float(-result.fun),
            "nfev": int(result.nfev),
            "elapsed_sec": float(time.time() - t0),
        })
    runs.sort(key=lambda r: r["obj_val"], reverse=True)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,7,13,42,100")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")]
    print(f"=== T75 EV-gate sweep seeds={seeds} ===", flush=True)

    t0 = time.time()
    df = load_avg_pred(seeds)
    print(f"  loaded {len(df):,} rows in {time.time()-t0:.1f}s", flush=True)

    folds = split_by_sym(df)
    for fk in folds:
        print(f"  sym={fk['sym']}: n={fk['n']:,}  pred mean={fk['pred'].mean():.6f} std={fk['pred'].std():.6f}", flush=True)

    pred_all = df["pred_dmid_norm"].to_numpy(np.float64)
    mp_t_all = df["midprice_t"].to_numpy(np.float64)
    mp_th_all = df["midprice_th"].to_numpy(np.float64)

    # 1) Symmetric k sweep
    print(f"\n[1/3] Symmetric k sweep (thr_up = thr_dn = k * 2*FEE):", flush=True)
    print(f"      iter_012 reference: {ITER012_LOSO:+.4f}\n", flush=True)
    sym_sweep = []
    k_grid = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0]
    print(f"  {'k':>6s}  {'thr':>10s}  {'single_total':>14s}  {'sum_per_sym':>14s}  {'n_active':>10s}  per_sym", flush=True)
    print(f"  {'-'*6}  {'-'*10}  {'-'*14}  {'-'*14}  {'-'*10}", flush=True)
    for k in k_grid:
        thr = k * 2.0 * FEE
        a_all = ev_gate_symmetric(pred_all, k)
        single_total = float(vectorized_pnl(a_all, mp_t_all, mp_th_all).sum())
        per, na = per_sym_pnl_at_k(folds, k)
        sum_per = float(sum(per))
        n_active_total = int((a_all != 1).sum())
        sym_sweep.append({
            "k": k, "thr": thr,
            "single_total": single_total,
            "sum_per_sym": sum_per,
            "per_sym": per,
            "n_active_per_sym": na,
            "n_active_total": n_active_total,
        })
        print(f"  {k:>6.2f}  {thr:>10.6f}  {single_total:>+14.4f}  {sum_per:>+14.4f}  {n_active_total:>10,}  "
              f"[{', '.join(f'{x:+.3f}' for x in per)}]", flush=True)

    # 2) DE asymmetric on single-set
    print(f"\n[2/3] DE asymmetric on single set (maximize total cum_pnl):", flush=True)
    bounds = [(0.0, 0.0040), (0.0, 0.0040)]  # thr in [0, 0.004] which is 20*FEE
    obj_single = make_obj_single_asym(pred_all, mp_t_all, mp_th_all)
    de_single = de_search(obj_single, bounds)
    best_s = de_single[0]
    thr_up_s, thr_dn_s = best_s["thr_up"], best_s["thr_dn"]
    a_de_single = ev_gate_asymmetric(pred_all, thr_up_s, thr_dn_s)
    de_single_total = float(vectorized_pnl(a_de_single, mp_t_all, mp_th_all).sum())
    de_single_per, de_single_na = per_sym_pnl_asym(folds, thr_up_s, thr_dn_s)
    print(f"  best DE-single: thr_up={thr_up_s:.6f} thr_dn={thr_dn_s:.6f}", flush=True)
    print(f"    total={de_single_total:+.4f} sum_per_sym={sum(de_single_per):+.4f} "
          f"per_sym=[{', '.join(f'{x:+.3f}' for x in de_single_per)}]", flush=True)

    # 3) DE asymmetric on LOSO-equiv (sum-of-per-sym)
    print(f"\n[3/3] DE asymmetric LOSO-equiv (maximize SUM of per-sym cum_pnl):", flush=True)
    obj_loso = make_obj_loso_asym(folds)
    de_loso = de_search(obj_loso, bounds)
    best_l = de_loso[0]
    thr_up_l, thr_dn_l = best_l["thr_up"], best_l["thr_dn"]
    de_loso_per, de_loso_na = per_sym_pnl_asym(folds, thr_up_l, thr_dn_l)
    de_loso_sum = float(sum(de_loso_per))
    a_de_loso = ev_gate_asymmetric(pred_all, thr_up_l, thr_dn_l)
    de_loso_total = float(vectorized_pnl(a_de_loso, mp_t_all, mp_th_all).sum())
    print(f"  best DE-loso: thr_up={thr_up_l:.6f} thr_dn={thr_dn_l:.6f}", flush=True)
    print(f"    sum_per_sym={de_loso_sum:+.4f} per_sym=[{', '.join(f'{x:+.3f}' for x in de_loso_per)}] "
          f"single_total={de_loso_total:+.4f}", flush=True)

    # ---- Summary ----
    best_sweep = max(sym_sweep, key=lambda r: r["sum_per_sym"])
    print(f"\n{'='*78}\n=== T75 SUMMARY ===\n{'='*78}", flush=True)
    print(f"Best symmetric k:    k={best_sweep['k']} thr={best_sweep['thr']:.6f}  "
          f"sum_per_sym={best_sweep['sum_per_sym']:+.4f}  single={best_sweep['single_total']:+.4f}",
          flush=True)
    print(f"DE single (asym):    thr_up={thr_up_s:.6f} thr_dn={thr_dn_s:.6f}  "
          f"sum_per_sym={sum(de_single_per):+.4f}  single={de_single_total:+.4f}", flush=True)
    print(f"DE LOSO-equiv (asym): thr_up={thr_up_l:.6f} thr_dn={thr_dn_l:.6f}  "
          f"sum_per_sym={de_loso_sum:+.4f}  single={de_loso_total:+.4f}", flush=True)
    print(f"vs iter_012 +{ITER012_LOSO:.2f}:", flush=True)
    print(f"  best_sym_sweep      vs iter_012: {best_sweep['sum_per_sym']-ITER012_LOSO:+.4f}", flush=True)
    print(f"  de_single_total     vs iter_012: {de_single_total-ITER012_LOSO:+.4f}", flush=True)
    print(f"  de_loso_sum         vs iter_012: {de_loso_sum-ITER012_LOSO:+.4f}", flush=True)

    out = {
        "task": "T75 EV-gate sweep on 5-seed regression Δmid avg predictions",
        "experiment": "T75_regression_dmid",
        "seeds": seeds,
        "n_test": len(df),
        "fee_rate": FEE,
        "iter_012_loso_ref": ITER012_LOSO,
        "symmetric_k_sweep": sym_sweep,
        "best_symmetric_k": best_sweep,
        "de_asym_single_set": {
            "best": best_s,
            "all_runs": de_single,
            "thr_up_at_best": thr_up_s,
            "thr_dn_at_best": thr_dn_s,
            "single_total_at_best": de_single_total,
            "per_sym_at_best": de_single_per,
            "n_active_per_sym": de_single_na,
            "sum_per_sym_at_best": float(sum(de_single_per)),
        },
        "de_asym_loso_equiv": {
            "best": best_l,
            "all_runs": de_loso,
            "thr_up_at_best": thr_up_l,
            "thr_dn_at_best": thr_dn_l,
            "sum_per_sym_at_best": de_loso_sum,
            "per_sym_at_best": de_loso_per,
            "n_active_per_sym": de_loso_na,
            "single_total_at_best": de_loso_total,
        },
        "compare_iter012": {
            "iter012_loso": ITER012_LOSO,
            "best_sweep_vs_iter012": best_sweep["sum_per_sym"] - ITER012_LOSO,
            "de_single_total_vs_iter012": de_single_total - ITER012_LOSO,
            "de_loso_sum_vs_iter012": de_loso_sum - ITER012_LOSO,
        },
    }
    out_path = os.path.join(HERE, "ev_gate_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
