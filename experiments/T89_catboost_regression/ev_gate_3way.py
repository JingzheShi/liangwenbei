"""T89: 3-way ensemble eval — NN (T81) + LGB (T75) + CB (T89, this dir).

Two evaluation modes:
  (a) CB-only: avg over 5 seeds (T89), sweep + DE asymmetric.
  (b) 3-way ensemble: weighted avg of NN-avg, LGB-avg, CB-avg, sweep weight grid.

Compare against iter_014 (NN+LGB only) DE-LOSO = +38.281.
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
ITER014_LOSO = 38.281

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


def avg_preds(dir_path, prefix):
    """Average pred_dmid_norm across SEEDS for given dir/prefix."""
    base = pd.read_parquet(os.path.join(dir_path, f"{prefix}{SEEDS[0]}.parquet"))
    n = len(base)
    p = np.zeros(n, dtype=np.float64)
    for s in SEEDS:
        df = pd.read_parquet(os.path.join(dir_path, f"{prefix}{s}.parquet"))
        if len(df) != n:
            raise RuntimeError(f"row count mismatch dir={dir_path} seed={s}")
        if not (df["sym"].equals(base["sym"]) and df["t"].equals(base["t"])):
            raise RuntimeError(f"row order mismatch dir={dir_path} seed={s}")
        p += df["pred_dmid_norm"].to_numpy(np.float64)
    p /= len(SEEDS)
    return base, p


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


def evaluate_full(df, label, do_de=True):
    folds = split_by_sym(df)
    pred_all = df["pred_dmid_norm"].to_numpy(np.float64)
    mp_t_all = df["midprice_t"].to_numpy(np.float64)
    mp_th_all = df["midprice_th"].to_numpy(np.float64)

    print(f"\n=== {label} ===", flush=True)
    print(f"  per-sym pred mean/std:", flush=True)
    for fk in folds:
        print(f"    sym={fk['sym']}: mean={fk['pred'].mean():+.6f} std={fk['pred'].std():.6f}",
              flush=True)

    sym_sweep = []
    print(f"\n  sym sweep: {'k':>5s}  {'thr':>8s}  {'sum_per_sym':>12s}  per_sym", flush=True)
    for k in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]:
        per = per_sym_pnl_at_k(folds, k)
        s = float(sum(per))
        sym_sweep.append({"k": k, "sum_per_sym": s, "per_sym": per})
        print(f"  {k:>5.2f}  {k*2*FEE:>8.5f}  {s:+12.4f}  [{', '.join(f'{x:+.3f}' for x in per)}]",
              flush=True)
    best_sym = max(sym_sweep, key=lambda r: r["sum_per_sym"])

    out = {
        "label": label,
        "sym_sweep": sym_sweep,
        "best_sym": best_sym,
    }

    if do_de:
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
        print(f"  vs iter_013 (+{ITER013_LOSO}): {de_loso_sum-ITER013_LOSO:+.4f}", flush=True)
        print(f"  vs iter_014 (+{ITER014_LOSO}): {de_loso_sum-ITER014_LOSO:+.4f}", flush=True)

        out["de_loso"] = {
            "thr_up": thr_up, "thr_dn": thr_dn,
            "sum_per_sym": de_loso_sum, "per_sym": de_per_sym,
            "single_total": de_total,
        }

    return out


def main():
    print("=== T89 3-way ensemble eval: NN (T81) + LGB (T75) + CB (T89) ===", flush=True)

    # Load three avg-preds
    base_nn, p_nn = avg_preds(T81_DIR, "pred_T81_seed")
    base_lgb, p_lgb = avg_preds(T75_DIR, "pred_T75_seed")
    base_cb, p_cb = avg_preds(HERE, "pred_T89_seed")

    # Sanity: row order across all three
    assert (base_nn["sym"].equals(base_lgb["sym"]) and base_nn["t"].equals(base_lgb["t"])), \
        "NN/LGB row order mismatch"
    assert (base_nn["sym"].equals(base_cb["sym"]) and base_nn["t"].equals(base_cb["t"])), \
        "NN/CB row order mismatch"

    print(f"\nCross-correlations:", flush=True)
    print(f"  corr(NN, LGB) = {np.corrcoef(p_nn, p_lgb)[0,1]:.4f}", flush=True)
    print(f"  corr(NN, CB)  = {np.corrcoef(p_nn, p_cb)[0,1]:.4f}", flush=True)
    print(f"  corr(LGB, CB) = {np.corrcoef(p_lgb, p_cb)[0,1]:.4f}", flush=True)

    base = base_nn.copy()

    # CB-only eval
    base["pred_dmid_norm"] = p_cb.astype(np.float32)
    cb_only = evaluate_full(base, "CB-only (T89 5seed avg)")

    # NN-only / LGB-only as references (sym sweep only, fast)
    base["pred_dmid_norm"] = p_nn.astype(np.float32)
    nn_only = evaluate_full(base, "NN-only (T81 5seed avg)", do_de=True)
    base["pred_dmid_norm"] = p_lgb.astype(np.float32)
    lgb_only = evaluate_full(base, "LGB-only (T75 5seed avg)", do_de=True)

    # 3-way weight grid
    weight_grid = [
        (1, 1, 1),
        (2, 1, 1), (1, 2, 1), (1, 1, 2),
        (1, 2, 2), (2, 1, 2), (2, 2, 1),
        (3, 1, 1), (1, 3, 1), (1, 1, 3),
        (1, 1, 0),  # NN+LGB (≈ iter_014)
        (1, 0, 1),  # NN+CB
        (0, 1, 1),  # LGB+CB
        (1, 1.5, 1.5),  # tilt towards GBDTs
        (2, 1.5, 1.5),  # bigger NN
        (1, 2, 2),
    ]
    results_3way = []
    for wn, wl, wc in weight_grid:
        wsum = wn + wl + wc
        p_e = (wn * p_nn + wl * p_lgb + wc * p_cb) / wsum
        base["pred_dmid_norm"] = p_e.astype(np.float32)
        label = f"3way: w_nn={wn} w_lgb={wl} w_cb={wc}"
        r = evaluate_full(base, label)
        r["w_nn"] = wn
        r["w_lgb"] = wl
        r["w_cb"] = wc
        results_3way.append(r)

    # Pick best by DE LOSO sum
    best_3way = max(results_3way, key=lambda r: r["de_loso"]["sum_per_sym"])
    print(f"\n{'='*78}", flush=True)
    print(f"BEST 3-way: {best_3way['label']}", flush=True)
    print(f"  DE LOSO sum = {best_3way['de_loso']['sum_per_sym']:+.4f}", flush=True)
    print(f"  vs iter_013 = {best_3way['de_loso']['sum_per_sym']-ITER013_LOSO:+.4f}", flush=True)
    print(f"  vs iter_014 = {best_3way['de_loso']['sum_per_sym']-ITER014_LOSO:+.4f}", flush=True)
    print(f"{'='*78}", flush=True)

    summary = {
        "cb_only": cb_only,
        "nn_only": nn_only,
        "lgb_only": lgb_only,
        "results_3way": results_3way,
        "best_3way": best_3way,
        "iter012_ref": ITER012_LOSO,
        "iter013_ref": ITER013_LOSO,
        "iter014_ref": ITER014_LOSO,
        "cross_corr": {
            "nn_lgb": float(np.corrcoef(p_nn, p_lgb)[0, 1]),
            "nn_cb": float(np.corrcoef(p_nn, p_cb)[0, 1]),
            "lgb_cb": float(np.corrcoef(p_lgb, p_cb)[0, 1]),
        },
    }
    out_path = os.path.join(HERE, "ev_gate_3way_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
