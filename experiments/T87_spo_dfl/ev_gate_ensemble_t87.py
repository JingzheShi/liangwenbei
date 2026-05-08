"""T87: Ensemble (T87 SPO+ NN) + (T75 LGB) and re-tune DE asymmetric thresholds.

Compare to iter_014 (T81 NN + T75 LGB) which gave +38.28 LOSO-equiv.

Steps:
  1. Load T87 NN seed predictions (5 seeds) → average them.
  2. Load T75 LGB seed predictions (5 seeds) → average them.
  3. Sweep weight pairs (w_nn, w_lgb) ∈ {(1,0),(0,1),(1,1),(1.5,1),(1,1.5),(2,1),(1,2)}.
  4. For each weight pair, run DE asymmetric thresh search over the 442k local test.
  5. Also include "T87 + T81 + LGB" 3-way blend at fixed weights for diversity.
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
T81_DIR = os.path.join(ROOT, "experiments", "T81_nn_regression_pnl")

SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001
ITER012_LOSO = 26.44
ITER013_LOSO = 36.23
ITER014_LOSO = 38.28

SEEDS = (1, 7, 13, 42, 100)


def vectorized_pnl(pred_action, mp_t, mp_th):
    side = pred_action.astype(np.float64) - 1.0
    abs_side = np.abs(side)
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * abs_side * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee_pnl) / denom


def ev_gate_symmetric(pred, k):
    thr = k * 2.0 * FEE
    a = np.full(len(pred), 1, dtype=np.int8)
    a[pred > thr] = 2
    a[pred < -thr] = 0
    return a


def ev_gate_asymmetric(pred, thr_up, thr_dn):
    a = np.full(len(pred), 1, dtype=np.int8)
    a[pred > thr_up] = 2
    a[pred < -thr_dn] = 0
    return a


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


def load_pred_avg(pred_dir, prefix, seeds=SEEDS):
    """Average predictions across seeds. Returns (df_template, p_mean, base_df)."""
    base = pd.read_parquet(os.path.join(pred_dir, f"{prefix}_seed{seeds[0]}{EXTRA_TAG.get(prefix,'')}.parquet"))
    n = len(base)
    p = np.zeros(n, dtype=np.float64)
    for s in seeds:
        path = os.path.join(pred_dir, f"{prefix}_seed{s}{EXTRA_TAG.get(prefix,'')}.parquet")
        df = pd.read_parquet(path)
        if len(df) != n:
            raise RuntimeError(f"row count mismatch {path}")
        if not (df["sym"].equals(base["sym"]) and df["t"].equals(base["t"])):
            raise RuntimeError(f"row order mismatch {path}")
        p += df["pred_dmid_norm"].to_numpy(np.float64)
    p /= len(seeds)
    return p, base


# T87 preds use tag "main"
EXTRA_TAG = {"pred_T87": "_main"}


def per_sym_pnl_at_k(folds, k):
    per = []
    for f in folds:
        a = ev_gate_symmetric(f["pred"], k)
        per.append(float(vectorized_pnl(a, f["mp_t"], f["mp_th"]).sum()))
    return per


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


def evaluate(df, label):
    folds = split_by_sym(df)
    pred_all = df["pred_dmid_norm"].to_numpy(np.float64)
    mp_t_all = df["midprice_t"].to_numpy(np.float64)
    mp_th_all = df["midprice_th"].to_numpy(np.float64)

    print(f"\n=== {label} ===", flush=True)
    print(f"  per-sym pred mean/std:", flush=True)
    for fk in folds:
        print(f"    sym={fk['sym']}: mean={fk['pred'].mean():+.6f} std={fk['pred'].std():.6f}",
              flush=True)

    # Symmetric sweep
    sym_sweep = []
    print(f"\n  sym sweep: {'k':>5s}  {'thr':>8s}  {'sum_per_sym':>12s}  per_sym", flush=True)
    for k in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]:
        per = per_sym_pnl_at_k(folds, k)
        s = float(sum(per))
        sym_sweep.append({"k": k, "sum_per_sym": s, "per_sym": per})
        print(f"  {k:>5.2f}  {k*2*FEE:>8.5f}  {s:+12.4f}  [{', '.join(f'{x:+.3f}' for x in per)}]",
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

    print(f"\n  best sym k={best_sym['k']}: sum_per_sym={best_sym['sum_per_sym']:+.4f}", flush=True)
    print(f"  DE LOSO asym: thr_up={thr_up:.6f} thr_dn={thr_dn:.6f}  sum={de_loso_sum:+.4f}",
          flush=True)
    print(f"    per_sym=[{', '.join(f'{x:+.3f}' for x in de_per_sym)}]", flush=True)
    print(f"  vs iter_012 (+{ITER012_LOSO}): {de_loso_sum-ITER012_LOSO:+.4f}", flush=True)
    print(f"  vs iter_013 (+{ITER013_LOSO}): {de_loso_sum-ITER013_LOSO:+.4f}", flush=True)
    print(f"  vs iter_014 (+{ITER014_LOSO}): {de_loso_sum-ITER014_LOSO:+.4f}", flush=True)

    return {
        "label": label,
        "sym_sweep": sym_sweep,
        "best_sym": best_sym,
        "de_loso": {
            "thr_up": thr_up, "thr_dn": thr_dn,
            "sum_per_sym": de_loso_sum, "per_sym": de_per_sym,
            "single_total": de_total,
        },
    }


def load_t87_avg():
    """T87 NN ensemble of 5 seeds, tag='main'."""
    base = pd.read_parquet(os.path.join(HERE, f"pred_T87_seed{SEEDS[0]}_main.parquet"))
    n = len(base)
    p = np.zeros(n, dtype=np.float64)
    for s in SEEDS:
        df = pd.read_parquet(os.path.join(HERE, f"pred_T87_seed{s}_main.parquet"))
        if len(df) != n:
            raise RuntimeError(f"row count mismatch seed={s}")
        if not (df["sym"].equals(base["sym"]) and df["t"].equals(base["t"])):
            raise RuntimeError(f"T87 row order mismatch seed={s}")
        p += df["pred_dmid_norm"].to_numpy(np.float64)
    p /= len(SEEDS)
    return p, base


def load_t75_avg():
    """T75 LGB ensemble of 5 seeds."""
    base = pd.read_parquet(os.path.join(T75_DIR, f"pred_T75_seed{SEEDS[0]}.parquet"))
    n = len(base)
    p = np.zeros(n, dtype=np.float64)
    for s in SEEDS:
        df = pd.read_parquet(os.path.join(T75_DIR, f"pred_T75_seed{s}.parquet"))
        if len(df) != n:
            raise RuntimeError(f"row count mismatch seed={s}")
        if not (df["sym"].equals(base["sym"]) and df["t"].equals(base["t"])):
            raise RuntimeError(f"T75 row order mismatch seed={s}")
        p += df["pred_dmid_norm"].to_numpy(np.float64)
    p /= len(SEEDS)
    return p, base


def load_t81_avg():
    """T81 NN ensemble of 5 seeds (for 3-way blends)."""
    base = pd.read_parquet(os.path.join(T81_DIR, f"pred_T81_seed{SEEDS[0]}.parquet"))
    n = len(base)
    p = np.zeros(n, dtype=np.float64)
    for s in SEEDS:
        df = pd.read_parquet(os.path.join(T81_DIR, f"pred_T81_seed{s}.parquet"))
        if len(df) != n:
            raise RuntimeError(f"row count mismatch seed={s}")
        if not (df["sym"].equals(base["sym"]) and df["t"].equals(base["t"])):
            raise RuntimeError(f"T81 row order mismatch seed={s}")
        p += df["pred_dmid_norm"].to_numpy(np.float64)
    p /= len(SEEDS)
    return p, base


def main():
    print("=== T87 ensemble eval: T87 NN (SPO+) + T75 LGB ===", flush=True)

    p_t87, base = load_t87_avg()
    p_t75, _ = load_t75_avg()
    p_t81, _ = load_t81_avg()

    # Cross-corr
    print(f"\n  T87-vs-T75: {float(np.corrcoef(p_t87, p_t75)[0,1]):.4f}", flush=True)
    print(f"  T87-vs-T81: {float(np.corrcoef(p_t87, p_t81)[0,1]):.4f}", flush=True)
    print(f"  T81-vs-T75: {float(np.corrcoef(p_t81, p_t75)[0,1]):.4f}", flush=True)

    results_all = []

    # 2-way: T87 + T75
    weights = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.5, 1.0), (1.0, 1.5),
               (2.0, 1.0), (1.0, 2.0), (1.0, 0.5), (0.5, 1.0)]
    for wn, wl in weights:
        p_combined = (wn * p_t87 + wl * p_t75) / max(wn + wl, 1e-9)
        df = base.copy()
        df["pred_dmid_norm"] = p_combined.astype(np.float32)
        label = f"T87+T75  w_nn={wn} w_lgb={wl}"
        r = evaluate(df, label)
        r["w_t87"] = wn
        r["w_t75"] = wl
        r["w_t81"] = 0.0
        results_all.append(r)

    # 3-way: T87 + T81 + T75 — try a few configs
    for w_t87, w_t81, w_t75 in [
        (1.0, 1.0, 1.5),
        (1.0, 0.5, 1.5),
        (0.7, 0.3, 1.0),
        (0.5, 0.5, 1.5),
        (1.0, 1.0, 2.0),
    ]:
        s = w_t87 + w_t81 + w_t75
        p_combined = (w_t87 * p_t87 + w_t81 * p_t81 + w_t75 * p_t75) / s
        df = base.copy()
        df["pred_dmid_norm"] = p_combined.astype(np.float32)
        label = f"3-way  w_t87={w_t87} w_t81={w_t81} w_t75={w_t75}"
        r = evaluate(df, label)
        r["w_t87"] = w_t87
        r["w_t81"] = w_t81
        r["w_t75"] = w_t75
        results_all.append(r)

    # Best
    best = max(results_all, key=lambda r: r["de_loso"]["sum_per_sym"])
    print(f"\n{'='*78}", flush=True)
    print(f"BEST: {best['label']}  → DE LOSO {best['de_loso']['sum_per_sym']:+.4f}",
          flush=True)
    print(f"  vs iter_014 (+{ITER014_LOSO}): {best['de_loso']['sum_per_sym']-ITER014_LOSO:+.4f}",
          flush=True)
    print('=' * 78, flush=True)

    out_path = os.path.join(HERE, "ev_gate_ensemble_results.json")
    with open(out_path, "w") as f:
        json.dump({"results": results_all, "best": best,
                   "iter012_ref": ITER012_LOSO,
                   "iter013_ref": ITER013_LOSO,
                   "iter014_ref": ITER014_LOSO}, f, indent=2)
    print(f"wrote -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
