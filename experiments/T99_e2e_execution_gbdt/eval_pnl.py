"""T99: EV-gate threshold sweep on multi-seed averaged predictions.

Reads pred_T99_<tag>_seed{S}.parquet for given tag and seed list, averages, runs:
  1) Symmetric k sweep
  2) DE asymmetric on (single set, total) — best (thr_up, thr_dn)
  3) DE asymmetric LOSO-equiv (sum of per-sym cum_pnl)

Saves ev_gate_<tag>.json.

Usage:
  python eval_pnl.py --tag huber_a1e-3 --seeds 1,7,13,42,100
  python eval_pnl.py --tag cb_mae --seeds 42  (single seed)
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))

SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001
ITER014_LOSO = 38.28  # T87 SPO+ DFL = iter_015 baseline reference
ITER015_LOSO = 40.09  # T87+T75 = iter_015 (current best)


def load_avg_pred(tag, seeds):
    paths = [os.path.join(HERE, f"pred_T99_{tag}_seed{s}.parquet") for s in seeds]
    base = pd.read_parquet(paths[0])
    p_sum = base["pred_dmid_norm"].to_numpy(np.float64).copy()
    n = len(base)
    for p in paths[1:]:
        df = pd.read_parquet(p)
        if len(df) != n:
            raise RuntimeError(f"row count mismatch: {p}")
        if not (df["sym"].equals(base["sym"]) and df["date"].equals(base["date"])
                and df["t"].equals(base["t"])):
            raise RuntimeError(f"row order mismatch: {p}")
        p_sum += df["pred_dmid_norm"].to_numpy(np.float64)
    p_avg = p_sum / float(len(seeds))
    out = base.copy()
    out["pred_dmid_norm"] = p_avg.astype(np.float32)
    return out


def split_by_sym(df):
    return [{"sym": int(k),
             "pred": df[df["sym"] == k]["pred_dmid_norm"].to_numpy(np.float64),
             "mp_t": df[df["sym"] == k]["midprice_t"].to_numpy(np.float64),
             "mp_th": df[df["sym"] == k]["midprice_th"].to_numpy(np.float64),
             "n": int((df["sym"] == k).sum())}
            for k in SYMS]


def vectorized_pnl(pred_action, mp_t, mp_th):
    side = pred_action.astype(np.float64) - 1.0
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee_pnl) / denom


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


def make_obj_single_asym(pred, mp_t, mp_th):
    def f(x):
        thr_up, thr_dn = x
        a = ev_gate_asymmetric(pred, thr_up, thr_dn)
        return -float(vectorized_pnl(a, mp_t, mp_th).sum())
    return f


def de_search(objective_fn, bounds, seeds=(0, 1, 2, 7, 42), maxiter=80, popsize=24):
    runs = []
    for sd in seeds:
        t0 = time.time()
        result = differential_evolution(
            objective_fn, bounds=bounds, seed=sd, maxiter=maxiter, popsize=popsize,
            polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            updating="deferred", workers=1, init="sobol",
        )
        runs.append({
            "seed": int(sd),
            "thr_up": float(result.x[0]),
            "thr_dn": float(result.x[1]),
            "obj_val": float(-result.fun),
            "elapsed_sec": float(time.time() - t0),
        })
    runs.sort(key=lambda r: r["obj_val"], reverse=True)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--bounds-hi", type=float, default=0.0040)
    ap.add_argument("--de-seeds", default="0,1,2,7,42")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")]
    de_seeds = tuple(int(x) for x in args.de_seeds.split(","))
    print(f"=== T99 EV-gate eval tag={args.tag} seeds={seeds} ===", flush=True)

    t0 = time.time()
    df = load_avg_pred(args.tag, seeds)
    print(f"  loaded {len(df):,} rows in {time.time()-t0:.1f}s", flush=True)
    folds = split_by_sym(df)
    pred_all = df["pred_dmid_norm"].to_numpy(np.float64)
    mp_t_all = df["midprice_t"].to_numpy(np.float64)
    mp_th_all = df["midprice_th"].to_numpy(np.float64)

    # Symmetric k sweep
    sym_sweep = []
    k_grid = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    print(f"\n[Symmetric k sweep]", flush=True)
    for k in k_grid:
        thr = k * 2.0 * FEE
        per = []
        for f in folds:
            a = ev_gate_asymmetric(f["pred"], thr, thr)
            per.append(float(vectorized_pnl(a, f["mp_t"], f["mp_th"]).sum()))
        a_all = ev_gate_asymmetric(pred_all, thr, thr)
        single_total = float(vectorized_pnl(a_all, mp_t_all, mp_th_all).sum())
        sum_per = float(sum(per))
        n_active = int((a_all != 1).sum())
        sym_sweep.append({"k": k, "thr": thr, "single_total": single_total,
                          "sum_per_sym": sum_per, "per_sym": per, "n_active_total": n_active})
        print(f"  k={k:>5.2f}  single={single_total:+8.3f}  sum_per_sym={sum_per:+8.3f}  "
              f"n_act={n_active:,}", flush=True)

    bounds = [(0.0, args.bounds_hi), (0.0, args.bounds_hi)]

    # DE single-set
    print(f"\n[DE asym single set]", flush=True)
    de_single = de_search(make_obj_single_asym(pred_all, mp_t_all, mp_th_all), bounds,
                          seeds=de_seeds)
    bs = de_single[0]
    a_s = ev_gate_asymmetric(pred_all, bs["thr_up"], bs["thr_dn"])
    de_single_total = float(vectorized_pnl(a_s, mp_t_all, mp_th_all).sum())
    de_single_per = []
    for f in folds:
        a = ev_gate_asymmetric(f["pred"], bs["thr_up"], bs["thr_dn"])
        de_single_per.append(float(vectorized_pnl(a, f["mp_t"], f["mp_th"]).sum()))
    print(f"  best: thr_up={bs['thr_up']:.6f} thr_dn={bs['thr_dn']:.6f}", flush=True)
    print(f"    single={de_single_total:+.4f} sum_per_sym={sum(de_single_per):+.4f} "
          f"per_sym={[f'{x:+.3f}' for x in de_single_per]}", flush=True)

    # DE LOSO-equiv
    print(f"\n[DE asym LOSO-equiv]", flush=True)
    de_loso = de_search(make_obj_loso_asym(folds), bounds, seeds=de_seeds)
    bl = de_loso[0]
    de_loso_per = []
    for f in folds:
        a = ev_gate_asymmetric(f["pred"], bl["thr_up"], bl["thr_dn"])
        de_loso_per.append(float(vectorized_pnl(a, f["mp_t"], f["mp_th"]).sum()))
    de_loso_sum = float(sum(de_loso_per))
    a_l = ev_gate_asymmetric(pred_all, bl["thr_up"], bl["thr_dn"])
    de_loso_total = float(vectorized_pnl(a_l, mp_t_all, mp_th_all).sum())
    print(f"  best: thr_up={bl['thr_up']:.6f} thr_dn={bl['thr_dn']:.6f}", flush=True)
    print(f"    sum_per_sym={de_loso_sum:+.4f} single={de_loso_total:+.4f} "
          f"per_sym={[f'{x:+.3f}' for x in de_loso_per]}", flush=True)

    best_sweep = max(sym_sweep, key=lambda r: r["sum_per_sym"])
    print(f"\n=== SUMMARY {args.tag} ===")
    print(f"  best symmetric k: k={best_sweep['k']} sum_per_sym={best_sweep['sum_per_sym']:+.4f}")
    print(f"  DE single:        thr=({bs['thr_up']:.5f},{bs['thr_dn']:.5f})  total={de_single_total:+.4f}  sum_per_sym={sum(de_single_per):+.4f}")
    print(f"  DE LOSO-equiv:    thr=({bl['thr_up']:.5f},{bl['thr_dn']:.5f})  sum_per_sym={de_loso_sum:+.4f}  single={de_loso_total:+.4f}")
    print(f"  vs T89 RMSE single +38.48:    {de_loso_sum-38.48:+.4f}")
    print(f"  vs iter_015 (T87+T75) +40.09: {de_loso_sum-40.09:+.4f}")

    out = {
        "task": f"T99 EV-gate eval tag={args.tag}",
        "tag": args.tag,
        "seeds": seeds,
        "n_test": len(df),
        "fee_rate": FEE,
        "symmetric_k_sweep": sym_sweep,
        "best_symmetric_k": best_sweep,
        "de_asym_single_set": {
            "best": bs, "all_runs": de_single,
            "thr_up": bs["thr_up"], "thr_dn": bs["thr_dn"],
            "single_total": de_single_total, "sum_per_sym": float(sum(de_single_per)),
            "per_sym": de_single_per,
        },
        "de_asym_loso_equiv": {
            "best": bl, "all_runs": de_loso,
            "thr_up": bl["thr_up"], "thr_dn": bl["thr_dn"],
            "sum_per_sym": de_loso_sum, "single_total": de_loso_total,
            "per_sym": de_loso_per,
        },
        "vs_T89_RMSE_single_3848": de_loso_sum - 38.48,
        "vs_iter015_T87_T75_4009":  de_loso_sum - 40.09,
    }
    out_path = args.out if args.out else os.path.join(HERE, f"ev_gate_{args.tag}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
