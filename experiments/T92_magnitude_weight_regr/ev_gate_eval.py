"""T92 EV-gate eval on 5-seed avg predictions for chosen variant.

Mirrors T75 ev_gate_eval.py exactly so results are directly comparable.
"""
from __future__ import annotations

import argparse
import glob
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
ITER013_LOSO = 36.23   # T75 5-seed DE-asym
ITER014_LOSO = 38.28   # NN+LGB ensemble (target)


def load_avg_pred(variant, seeds, tag):
    files = [os.path.join(HERE, f"pred_T92_{variant}_seed{s}_{tag}.parquet") for s in seeds]
    base = pd.read_parquet(files[0])
    p_sum = base["pred_dmid_norm"].to_numpy(np.float64).copy()
    n = len(base)
    for f in files[1:]:
        df_s = pd.read_parquet(f)
        if len(df_s) != n:
            raise RuntimeError(f"row count mismatch {f}: {len(df_s)} vs {n}")
        if not (df_s["sym"].equals(base["sym"]) and df_s["date"].equals(base["date"])
                and df_s["t"].equals(base["t"])):
            raise RuntimeError(f"row order mismatch {f}")
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
    ap.add_argument("--variant", required=True)
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--seeds", default="1,7,13,42,100")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")]
    print(f"=== T92 EV-gate sweep variant={args.variant} seeds={seeds} ===", flush=True)

    df = load_avg_pred(args.variant, seeds, args.tag)
    print(f"  loaded {len(df):,} rows  pred mean={df['pred_dmid_norm'].mean():.6f} "
          f"std={df['pred_dmid_norm'].std():.6f}", flush=True)

    folds = split_by_sym(df)
    for fk in folds:
        print(f"  sym={fk['sym']}: n={fk['n']:,}  pred mean={fk['pred'].mean():.6f} "
              f"std={fk['pred'].std():.6f}", flush=True)

    pred_all = df["pred_dmid_norm"].to_numpy(np.float64)
    mp_t_all = df["midprice_t"].to_numpy(np.float64)
    mp_th_all = df["midprice_th"].to_numpy(np.float64)

    # Symmetric k sweep
    print(f"\n[1/2] Symmetric k sweep (thr_up=thr_dn=k*2*FEE):", flush=True)
    print(f"      iter_012 +26.44 ; iter_013 (T75 5-seed DE) +{ITER013_LOSO} ; iter_014 +{ITER014_LOSO}\n",
          flush=True)
    sym_sweep = []
    k_grid = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    print(f"  {'k':>6s}  {'thr':>10s}  {'single':>12s}  {'sum_sym':>12s}  {'n_act':>10s}  per_sym", flush=True)
    for k in k_grid:
        thr = k * 2.0 * FEE
        a_all = ev_gate_symmetric(pred_all, k)
        single_total = float(vectorized_pnl(a_all, mp_t_all, mp_th_all).sum())
        per = []
        na = []
        for fk in folds:
            a = ev_gate_symmetric(fk["pred"], k)
            per.append(float(vectorized_pnl(a, fk["mp_t"], fk["mp_th"]).sum()))
            na.append(int((a != 1).sum()))
        sum_per = float(sum(per))
        n_active_total = int((a_all != 1).sum())
        sym_sweep.append({"k": k, "thr": thr, "single_total": single_total,
                          "sum_per_sym": sum_per, "per_sym": per,
                          "n_active_per_sym": na, "n_active_total": n_active_total})
        print(f"  {k:>6.2f}  {thr:>10.6f}  {single_total:>+12.4f}  {sum_per:>+12.4f}  "
              f"{n_active_total:>10,}  [{', '.join(f'{x:+.3f}' for x in per)}]", flush=True)

    best_sym = max(sym_sweep, key=lambda d: d["sum_per_sym"])
    print(f"\n  [best symmetric] k={best_sym['k']:.2f}  thr={best_sym['thr']:.6f}  "
          f"sum_per_sym={best_sym['sum_per_sym']:+.4f}  vs iter_013={best_sym['sum_per_sym']-ITER013_LOSO:+.4f}",
          flush=True)

    # Asymmetric DE
    print(f"\n[2/2] DE asymmetric (thr_up, thr_dn) on per-sym sum:", flush=True)
    obj = make_obj_loso_asym(folds)
    bounds = [(0.0, 1e-3), (0.0, 1e-3)]
    runs = de_search(obj, bounds)
    best_de = runs[0]
    per_de = []
    na_de = []
    for fk in folds:
        a = ev_gate_asymmetric(fk["pred"], best_de["thr_up"], best_de["thr_dn"])
        per_de.append(float(vectorized_pnl(a, fk["mp_t"], fk["mp_th"]).sum()))
        na_de.append(int((a != 1).sum()))
    print(f"  best DE: thr_up={best_de['thr_up']:.6f}  thr_dn={best_de['thr_dn']:.6f}  "
          f"sum_per_sym={best_de['obj_val']:+.4f}", flush=True)
    print(f"  per_sym = [{', '.join(f'{x:+.3f}' for x in per_de)}]", flush=True)
    print(f"  vs iter_012 +{ITER012_LOSO}: {best_de['obj_val']-ITER012_LOSO:+.4f}", flush=True)
    print(f"  vs iter_013 +{ITER013_LOSO}: {best_de['obj_val']-ITER013_LOSO:+.4f}", flush=True)
    print(f"  vs iter_014 +{ITER014_LOSO}: {best_de['obj_val']-ITER014_LOSO:+.4f}", flush=True)

    out = {
        "variant": args.variant,
        "seeds": seeds,
        "tag": args.tag,
        "iter013_loso": ITER013_LOSO,
        "iter014_loso": ITER014_LOSO,
        "symmetric_sweep": sym_sweep,
        "best_symmetric": best_sym,
        "de_runs": runs,
        "best_de": {**best_de, "per_sym": per_de, "n_active_per_sym": na_de},
        "vs_iter013_best_sym": best_sym["sum_per_sym"] - ITER013_LOSO,
        "vs_iter013_de": best_de["obj_val"] - ITER013_LOSO,
        "vs_iter014_de": best_de["obj_val"] - ITER014_LOSO,
    }
    out_path = os.path.join(HERE, f"ev_gate_results_{args.variant}_{args.tag}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
