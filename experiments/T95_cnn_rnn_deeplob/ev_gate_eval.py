"""T95 EV-gate eval — supports any list of pred_*.parquet files (single or multi-seed average).

Computes:
  1) Symmetric k sweep
  2) DE asymmetric (single-set + LOSO-equiv)

Usage:
  python ev_gate_eval.py --preds pred_T95_minicnn_w100_A_seed42.parquet
  python ev_gate_eval.py --preds pred_T95_minicnn_w100_A_seed42.parquet pred_T95_minicnn_w100_A_seed1.parquet
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
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001
ITER012_LOSO = 26.44
ITER013_LOSO = 36.23
ITER014_LOSO = 38.28


def load_avg_pred(pred_paths):
    base = pd.read_parquet(pred_paths[0])
    p = base["pred_dmid_norm"].to_numpy(np.float64).copy()
    n_valid_base = ~np.isnan(p)
    # NaN handling: fill with 0 (flat decision)
    p = np.where(np.isnan(p), 0.0, p)
    cnt = np.where(n_valid_base, 1, 0).astype(np.float64)
    n = len(base)
    for path in pred_paths[1:]:
        df_s = pd.read_parquet(path)
        if len(df_s) != n:
            raise RuntimeError(f"row count mismatch {path}: {len(df_s)} vs {n}")
        if not (df_s["sym"].equals(base["sym"]) and df_s["date"].equals(base["date"])
                and df_s["t"].equals(base["t"])):
            raise RuntimeError(f"row order mismatch {path}")
        p_s = df_s["pred_dmid_norm"].to_numpy(np.float64)
        valid = ~np.isnan(p_s)
        p += np.where(valid, p_s, 0.0)
        cnt += np.where(valid, 1.0, 0.0)
    p_avg = np.where(cnt > 0, p / np.maximum(cnt, 1.0), 0.0)
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
    pred = [f["pred"] for f in folds]
    mp_t = [f["mp_t"] for f in folds]
    mp_th = [f["mp_th"] for f in folds]
    def f(x):
        s = 0.0
        for i in range(len(folds)):
            a = ev_gate_asymmetric(pred[i], x[0], x[1])
            s += vectorized_pnl(a, mp_t[i], mp_th[i]).sum()
        return -float(s)
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
    ap.add_argument("--preds", nargs="+", required=True, help="pred parquet paths")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-de", action="store_true", help="skip DE (single-set + LOSO)")
    ap.add_argument("--quick", action="store_true", help="fewer DE seeds")
    args = ap.parse_args()

    pred_paths = []
    for p in args.preds:
        if not os.path.isabs(p):
            p = os.path.join(HERE, p) if not os.path.exists(p) else p
        pred_paths.append(p)
    print(f"=== T95 EV-gate eval over {len(pred_paths)} pred files ===", flush=True)
    for p in pred_paths:
        print(f"  - {p}")

    df = load_avg_pred(pred_paths)
    print(f"\n  loaded {len(df):,} rows", flush=True)
    pred_all = df["pred_dmid_norm"].to_numpy(np.float64)
    mp_t_all = df["midprice_t"].to_numpy(np.float64)
    mp_th_all = df["midprice_th"].to_numpy(np.float64)
    print(f"  pred mean={pred_all.mean():.6e} std={pred_all.std():.6e} "
          f"min={pred_all.min():.6e} max={pred_all.max():.6e}", flush=True)

    folds = split_by_sym(df)

    # Symmetric k sweep
    print(f"\n[1] Symmetric k sweep:", flush=True)
    sym_sweep = []
    k_grid = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    for k in k_grid:
        thr = k * 2.0 * FEE
        a_all = ev_gate_symmetric(pred_all, k)
        single_total = float(vectorized_pnl(a_all, mp_t_all, mp_th_all).sum())
        per, na = per_sym_pnl_at_k(folds, k)
        sum_per = float(sum(per))
        sym_sweep.append({"k": k, "thr": thr, "single_total": single_total,
                          "sum_per_sym": sum_per, "per_sym": per,
                          "n_active_per_sym": na, "n_active_total": int((a_all != 1).sum())})
        print(f"  k={k:.2f}  thr={thr:.5f}  single={single_total:+.4f}  "
              f"sum_per_sym={sum_per:+.4f}  n_active={int((a_all != 1).sum()):,}", flush=True)

    best_sweep = max(sym_sweep, key=lambda r: r["sum_per_sym"])
    out = {
        "task": "T95 EV-gate eval",
        "pred_paths": pred_paths,
        "n_test": len(df),
        "fee_rate": FEE,
        "iter_012_loso": ITER012_LOSO,
        "iter_013_loso": ITER013_LOSO,
        "iter_014_loso": ITER014_LOSO,
        "symmetric_k_sweep": sym_sweep,
        "best_symmetric_k": best_sweep,
    }

    if not args.no_de:
        seeds = (0, 1) if args.quick else (0, 1, 2, 7, 42)
        bounds = [(0.0, 0.0040), (0.0, 0.0040)]

        print(f"\n[2] DE asym single-set total ({len(seeds)} seeds):", flush=True)
        obj_single = lambda x: -float(vectorized_pnl(
            ev_gate_asymmetric(pred_all, x[0], x[1]), mp_t_all, mp_th_all).sum())
        de_single = de_search(obj_single, bounds, seeds=seeds)
        best_s = de_single[0]
        thr_up_s, thr_dn_s = best_s["thr_up"], best_s["thr_dn"]
        a_de_single = ev_gate_asymmetric(pred_all, thr_up_s, thr_dn_s)
        de_single_total = float(vectorized_pnl(a_de_single, mp_t_all, mp_th_all).sum())
        de_single_per, de_single_na = per_sym_pnl_asym(folds, thr_up_s, thr_dn_s)
        print(f"  best DE-single: thr_up={thr_up_s:.5f} thr_dn={thr_dn_s:.5f}  "
              f"total={de_single_total:+.4f} sum_per_sym={sum(de_single_per):+.4f}", flush=True)

        print(f"\n[3] DE asym LOSO-equiv ({len(seeds)} seeds):", flush=True)
        obj_loso = make_obj_loso_asym(folds)
        de_loso = de_search(obj_loso, bounds, seeds=seeds)
        best_l = de_loso[0]
        thr_up_l, thr_dn_l = best_l["thr_up"], best_l["thr_dn"]
        de_loso_per, de_loso_na = per_sym_pnl_asym(folds, thr_up_l, thr_dn_l)
        de_loso_sum = float(sum(de_loso_per))
        a_de_loso = ev_gate_asymmetric(pred_all, thr_up_l, thr_dn_l)
        de_loso_total = float(vectorized_pnl(a_de_loso, mp_t_all, mp_th_all).sum())
        print(f"  best DE-LOSO: thr_up={thr_up_l:.5f} thr_dn={thr_dn_l:.5f}  "
              f"sum_per_sym={de_loso_sum:+.4f} per_sym={de_loso_per}", flush=True)

        out["de_asym_single_set"] = {
            "best": best_s, "all": de_single,
            "thr_up": thr_up_s, "thr_dn": thr_dn_s,
            "single_total": de_single_total,
            "per_sym": de_single_per, "n_active_per_sym": de_single_na,
            "sum_per_sym": float(sum(de_single_per)),
        }
        out["de_asym_loso_equiv"] = {
            "best": best_l, "all": de_loso,
            "thr_up": thr_up_l, "thr_dn": thr_dn_l,
            "sum_per_sym": de_loso_sum,
            "per_sym": de_loso_per, "n_active_per_sym": de_loso_na,
            "single_total": de_loso_total,
        }

    print(f"\n=== SUMMARY ===", flush=True)
    print(f"best symmetric: k={best_sweep['k']} sum_per_sym={best_sweep['sum_per_sym']:+.4f}", flush=True)
    if not args.no_de:
        print(f"de_loso: sum_per_sym={de_loso_sum:+.4f} (vs iter013={ITER013_LOSO}, iter014={ITER014_LOSO})", flush=True)
        print(f"  vs iter013: {de_loso_sum - ITER013_LOSO:+.4f}", flush=True)
        print(f"  vs iter014: {de_loso_sum - ITER014_LOSO:+.4f}", flush=True)

    if args.out is None:
        args.out = os.path.join(HERE, "ev_gate_results_" +
                                 os.path.basename(pred_paths[0]).replace(".parquet", ".json"))
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
