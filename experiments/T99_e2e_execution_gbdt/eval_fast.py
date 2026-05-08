"""T99: Fast EV-gate eval — symmetric k sweep + single-seed DE asym (LOSO-equiv).

For each tag, computes:
  1) Symmetric k sweep over k ∈ [0.5..3.0]
  2) DE asym (seed=42 only, maxiter=60, popsize=20) — best (thr_up, thr_dn)
  3) Per-sym PnL at the DE optimum

Saves all results in one JSON.

Usage:
  python eval_fast.py --tags huber_a0.001:1,7,13,42,100 cb_huber_a0.001:42 ...
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


def load_avg_pred(tag, seeds):
    paths = [os.path.join(HERE, f"pred_T99_{tag}_seed{s}.parquet") for s in seeds]
    base = pd.read_parquet(paths[0])
    p_sum = base["pred_dmid_norm"].to_numpy(np.float64).copy()
    for p in paths[1:]:
        df = pd.read_parquet(p)
        if not (df["sym"].equals(base["sym"]) and df["date"].equals(base["date"])
                and df["t"].equals(base["t"])):
            raise RuntimeError(f"row order mismatch {p}")
        p_sum += df["pred_dmid_norm"].to_numpy(np.float64)
    return base, p_sum / float(len(paths))


def vectorized_pnl(pred_action, mp_t, mp_th):
    side = pred_action.astype(np.float64) - 1.0
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee_pnl) / denom


def ev_gate(pred, thr_up, thr_dn):
    a = np.full(len(pred), 1, dtype=np.int8)
    a[pred > thr_up] = 2
    a[pred < -thr_dn] = 0
    return a


def per_sym_pnl(pred, sym_arr, mp_t, mp_th, thr_up, thr_dn):
    per = []
    for k in SYMS:
        m = sym_arr == k
        if m.sum() == 0:
            per.append(0.0); continue
        a = ev_gate(pred[m], thr_up, thr_dn)
        per.append(float(vectorized_pnl(a, mp_t[m], mp_th[m]).sum()))
    return per


def de_loso(pred, sym_arr, mp_t, mp_th, bounds_hi=0.0040, de_seeds=(0, 42),
            maxiter=60, popsize=20):
    fold_pred, fold_mpt, fold_mpth = [], [], []
    for k in SYMS:
        m = sym_arr == k
        fold_pred.append(pred[m])
        fold_mpt.append(mp_t[m])
        fold_mpth.append(mp_th[m])

    def f(x):
        thr_up, thr_dn = x
        s = 0.0
        for i in range(len(SYMS)):
            a = ev_gate(fold_pred[i], thr_up, thr_dn)
            s += vectorized_pnl(a, fold_mpt[i], fold_mpth[i]).sum()
        return -float(s)

    best = None
    runs = []
    for sd in de_seeds:
        result = differential_evolution(
            f, bounds=[(0.0, bounds_hi), (0.0, bounds_hi)], seed=sd,
            maxiter=maxiter, popsize=popsize, polish=True, tol=1e-7,
            mutation=(0.5, 1.0), recombination=0.7, updating="deferred",
            workers=1, init="sobol",
        )
        run = {"seed": int(sd), "thr_up": float(result.x[0]),
               "thr_dn": float(result.x[1]), "obj_val": float(-result.fun)}
        runs.append(run)
        if best is None or run["obj_val"] > best["obj_val"]:
            best = run
    return best, runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True,
                    help="tag:seedlist e.g. huber_a0.001:1,7,13,42,100")
    ap.add_argument("--bounds-hi", type=float, default=0.0040)
    ap.add_argument("--de-seeds", default="0,42")
    ap.add_argument("--maxiter", type=int, default=60)
    ap.add_argument("--popsize", type=int, default=20)
    ap.add_argument("--out", default="eval_fast_results.json")
    args = ap.parse_args()

    de_seeds = tuple(int(x) for x in args.de_seeds.split(","))
    print(f"=== T99 fast eval ({args.tags}) ===", flush=True)

    results = {}
    for spec in args.tags:
        if ":" not in spec:
            tag, seeds_str = spec, "42"
        else:
            tag, seeds_str = spec.split(":", 1)
        seeds = [int(x) for x in seeds_str.split(",")]
        t0 = time.time()
        try:
            base, p = load_avg_pred(tag, seeds)
        except Exception as e:
            print(f"  [{tag}] LOAD FAIL: {e}", flush=True)
            continue
        sym_arr = base["sym"].to_numpy()
        mp_t = base["midprice_t"].to_numpy(np.float64)
        mp_th = base["midprice_th"].to_numpy(np.float64)

        # Symmetric k sweep
        k_grid = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
        sym_sweep = []
        for k in k_grid:
            thr = k * 2.0 * FEE
            per = per_sym_pnl(p, sym_arr, mp_t, mp_th, thr, thr)
            sym_sweep.append({"k": k, "thr": thr, "sum_per_sym": float(sum(per)),
                              "per_sym": per})
        best_sweep = max(sym_sweep, key=lambda r: r["sum_per_sym"])

        # DE LOSO-equiv
        best, all_runs = de_loso(p, sym_arr, mp_t, mp_th, args.bounds_hi,
                                 de_seeds, args.maxiter, args.popsize)
        per_de = per_sym_pnl(p, sym_arr, mp_t, mp_th, best["thr_up"], best["thr_dn"])

        elapsed = time.time() - t0
        print(f"  [{tag}] seeds={seeds_str} ({elapsed:.1f}s)", flush=True)
        print(f"    best_sym k={best_sweep['k']} sum_per_sym={best_sweep['sum_per_sym']:+.4f} "
              f"per_sym={[f'{x:+.2f}' for x in best_sweep['per_sym']]}", flush=True)
        print(f"    de_loso  thr=({best['thr_up']:.5f},{best['thr_dn']:.5f}) "
              f"sum_per_sym={best['obj_val']:+.4f} "
              f"per_sym={[f'{x:+.2f}' for x in per_de]}", flush=True)

        results[tag] = {
            "tag": tag, "seeds": seeds, "n_seeds": len(seeds),
            "elapsed_sec": elapsed,
            "symmetric_sweep": sym_sweep,
            "best_symmetric": best_sweep,
            "de_loso_best": best,
            "de_loso_all_runs": all_runs,
            "de_loso_per_sym": per_de,
            "de_loso_sum_per_sym": float(sum(per_de)),
        }

    out_path = os.path.join(HERE, args.out)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}", flush=True)

    # Sorted summary
    print(f"\n=== Sorted by DE LOSO sum_per_sym ===", flush=True)
    sorted_results = sorted(results.values(), key=lambda r: -r["de_loso_sum_per_sym"])
    for r in sorted_results:
        print(f"  {r['tag']:30s} ({r['n_seeds']}seed)  "
              f"de_loso={r['de_loso_sum_per_sym']:+8.4f}  "
              f"sym_sweep_best={r['best_symmetric']['sum_per_sym']:+.4f}  "
              f"thr=({r['de_loso_best']['thr_up']:.5f},{r['de_loso_best']['thr_dn']:.5f})",
              flush=True)


if __name__ == "__main__":
    main()
