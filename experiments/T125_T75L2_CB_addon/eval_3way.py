"""T125: 3-way ensemble = T87 NN + T75 LGB L2 + T89 CB RMSE on iter_015 v1 stack.

Baseline iter_015 v1 = T87 NN + T75 LGB L2 (1.0 : 1.5).
New 3-way = T87 NN + T75 LGB L2 + T89 CB RMSE  (sweep weights).

For each weight setting, run DE asymmetric thresh search and compute LOSO-equiv
(sum-per-sym) total. Reports cross-corr matrix to verify CB diversity.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
T75_DIR = os.path.join(ROOT, "experiments", "T75_regression_dmid")
T87_DIR = os.path.join(ROOT, "experiments", "T87_spo_dfl")
T89_DIR = os.path.join(ROOT, "experiments", "T89_catboost_regression")

SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001
SEEDS = (1, 7, 13, 42, 100)

ITER015_V1_LOSO = 40.13  # baseline (T87 NN + T75 LGB L2 1.0:1.5)


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


def ev_gate_symmetric(pred, k):
    thr = k * 2.0 * FEE
    a = np.full(len(pred), 1, dtype=np.int8)
    a[pred > thr] = 2
    a[pred < -thr] = 0
    return a


def split_by_sym(df_pred, base):
    """split predictions by sym using `base` for sym/mp_t/mp_th."""
    folds = []
    for k in SYMS:
        m = (base["sym"].to_numpy() == k)
        folds.append({
            "sym": int(k),
            "pred": df_pred[m].astype(np.float64),
            "mp_t": base["midprice_t"].to_numpy(np.float64)[m],
            "mp_th": base["midprice_th"].to_numpy(np.float64)[m],
            "n": int(m.sum()),
        })
    return folds


def per_sym_pnl_at_k(folds, k):
    per = []
    for f in folds:
        a = ev_gate_symmetric(f["pred"], k)
        per.append(float(vectorized_pnl(a, f["mp_t"], f["mp_th"]).sum()))
    return per


def make_obj_loso_asym(folds):
    def f(x):
        thr_up, thr_dn = x
        s = 0.0
        for fk in folds:
            a = ev_gate_asymmetric(fk["pred"], thr_up, thr_dn)
            s += vectorized_pnl(a, fk["mp_t"], fk["mp_th"]).sum()
        return -float(s)
    return f


def de_search(objective_fn, bounds, seeds=(0, 1, 2, 7, 42)):
    runs = []
    for sd in seeds:
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
        })
    runs.sort(key=lambda r: r["obj_val"], reverse=True)
    return runs


def evaluate_pred(pred_avg, base, label):
    folds = split_by_sym(pred_avg, base)
    pred_all = pred_avg.astype(np.float64)
    mp_t_all = base["midprice_t"].to_numpy(np.float64)
    mp_th_all = base["midprice_th"].to_numpy(np.float64)

    print(f"\n=== {label} ===", flush=True)
    print(f"  per-sym pred mean/std:", flush=True)
    for fk in folds:
        print(f"    sym={fk['sym']}: mean={fk['pred'].mean():+.6f} std={fk['pred'].std():.6f}",
              flush=True)

    sym_sweep = []
    print(f"  sym sweep:", flush=True)
    for k in [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]:
        per = per_sym_pnl_at_k(folds, k)
        s = float(sum(per))
        sym_sweep.append({"k": k, "sum_per_sym": s, "per_sym": per})
        print(f"    k={k:>4.2f}  thr={k*2*FEE:.5f}  sum={s:+.4f}  per=[{', '.join(f'{x:+.3f}' for x in per)}]",
              flush=True)
    best_sym = max(sym_sweep, key=lambda r: r["sum_per_sym"])

    obj = make_obj_loso_asym(folds)
    runs = de_search(obj, [(0.0, 0.0040), (0.0, 0.0040)])
    best = runs[0]
    thr_up, thr_dn = best["thr_up"], best["thr_dn"]
    a_de = ev_gate_asymmetric(pred_all, thr_up, thr_dn)
    de_total = float(vectorized_pnl(a_de, mp_t_all, mp_th_all).sum())
    de_per_sym = []
    for fk in folds:
        a = ev_gate_asymmetric(fk["pred"], thr_up, thr_dn)
        de_per_sym.append(float(vectorized_pnl(a, fk["mp_t"], fk["mp_th"]).sum()))
    de_loso_sum = float(sum(de_per_sym))

    print(f"  best sym k={best_sym['k']}: sum_per_sym={best_sym['sum_per_sym']:+.4f}", flush=True)
    print(f"  DE LOSO asym: thr_up={thr_up:.6f} thr_dn={thr_dn:.6f}  sum={de_loso_sum:+.4f}",
          flush=True)
    print(f"    per_sym=[{', '.join(f'{x:+.3f}' for x in de_per_sym)}]", flush=True)
    print(f"  vs iter_015 v1 (+{ITER015_V1_LOSO}): {de_loso_sum-ITER015_V1_LOSO:+.4f}", flush=True)

    return {
        "label": label,
        "sym_sweep": sym_sweep,
        "best_sym": best_sym,
        "de_loso": {
            "thr_up": thr_up, "thr_dn": thr_dn,
            "sum_per_sym": de_loso_sum, "per_sym": de_per_sym,
            "single_total": de_total,
        },
        "de_runs": runs,
    }


def load_avg(pred_dir, prefix, suffix=""):
    base = pd.read_parquet(os.path.join(pred_dir, f"{prefix}_seed{SEEDS[0]}{suffix}.parquet"))
    n = len(base)
    p = np.zeros(n, dtype=np.float64)
    for s in SEEDS:
        df = pd.read_parquet(os.path.join(pred_dir, f"{prefix}_seed{s}{suffix}.parquet"))
        if len(df) != n:
            raise RuntimeError(f"row count mismatch {prefix} seed={s}")
        if not (df["sym"].equals(base["sym"]) and df["t"].equals(base["t"])
                and df["date"].equals(base["date"])):
            raise RuntimeError(f"row order mismatch {prefix} seed={s}")
        p += df["pred_dmid_norm"].to_numpy(np.float64)
    p /= len(SEEDS)
    return p, base


def main():
    print("=== T125: 3-way (T87 NN + T75 LGB L2 + T89 CB RMSE) ensemble eval ===", flush=True)
    t0 = time.time()

    p_t87, base = load_avg(T87_DIR, "pred_T87", "_main")
    p_t75, base_t75 = load_avg(T75_DIR, "pred_T75")
    p_t89, base_t89 = load_avg(T89_DIR, "pred_T89")

    # sanity: rows must align with T87 base
    assert base["sym"].equals(base_t75["sym"]) and base["t"].equals(base_t75["t"])
    assert base["sym"].equals(base_t89["sym"]) and base["t"].equals(base_t89["t"])
    print(f"  loaded preds in {time.time()-t0:.1f}s  n={len(base):,}", flush=True)

    # Cross-corr
    cc_nn_lgb = float(np.corrcoef(p_t87, p_t75)[0, 1])
    cc_nn_cb  = float(np.corrcoef(p_t87, p_t89)[0, 1])
    cc_lgb_cb = float(np.corrcoef(p_t75, p_t89)[0, 1])
    print(f"\n  Cross-correlations:", flush=True)
    print(f"    T87 NN  vs T75 LGB L2 : {cc_nn_lgb:.4f}", flush=True)
    print(f"    T87 NN  vs T89 CB RMSE: {cc_nn_cb:.4f}", flush=True)
    print(f"    T75 LGB vs T89 CB RMSE: {cc_lgb_cb:.4f}", flush=True)

    results_all = []

    # Baseline: iter_015 v1 reproduction
    bl_label = "BASELINE iter_015 v1 (T87:T75 = 1.0:1.5)"
    p_bl = (1.0 * p_t87 + 1.5 * p_t75) / 2.5
    r_bl = evaluate_pred(p_bl, base, bl_label)
    r_bl["weights"] = {"t87": 1.0, "t75": 1.5, "t89": 0.0}
    results_all.append(r_bl)

    # Single-CB-only (sanity)
    r_cb = evaluate_pred(p_t89, base, "T89 CB RMSE only")
    r_cb["weights"] = {"t87": 0.0, "t75": 0.0, "t89": 1.0}
    results_all.append(r_cb)

    # 3-way: vary CB weight, keep T87:T75 ratio close to 1.0:1.5
    cb_weights = [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0]
    for w_cb in cb_weights:
        for (w_nn, w_lgb) in [(1.0, 1.5), (1.0, 1.0), (1.5, 1.0)]:
            s = w_nn + w_lgb + w_cb
            p = (w_nn * p_t87 + w_lgb * p_t75 + w_cb * p_t89) / s
            label = f"3-way  T87={w_nn} T75={w_lgb} T89={w_cb}"
            r = evaluate_pred(p, base, label)
            r["weights"] = {"t87": w_nn, "t75": w_lgb, "t89": w_cb}
            results_all.append(r)

    # Best
    best = max(results_all, key=lambda r: r["de_loso"]["sum_per_sym"])
    print(f"\n{'='*78}", flush=True)
    print(f"BEST: {best['label']}", flush=True)
    print(f"  DE LOSO {best['de_loso']['sum_per_sym']:+.4f}", flush=True)
    print(f"  vs iter_015 v1 baseline: {best['de_loso']['sum_per_sym']-r_bl['de_loso']['sum_per_sym']:+.4f}",
          flush=True)
    print('=' * 78, flush=True)

    out = {
        "results": results_all,
        "best": best,
        "baseline_iter015_v1_reproduced": r_bl["de_loso"]["sum_per_sym"],
        "baseline_iter015_v1_pm_ref": ITER015_V1_LOSO,
        "cross_corr": {
            "nn_lgb": cc_nn_lgb, "nn_cb": cc_nn_cb, "lgb_cb": cc_lgb_cb,
        },
    }
    with open(os.path.join(HERE, "ev_gate_3way_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {os.path.join(HERE, 'ev_gate_3way_results.json')}", flush=True)


if __name__ == "__main__":
    main()
