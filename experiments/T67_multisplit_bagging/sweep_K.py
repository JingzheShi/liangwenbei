"""T67 K-sweep: average first K bags and run DE 4D for each K, then plot the curve.

For K_max=20 trained bags, evaluate at K = [1, 2, 3, 5, 7, 10, 12, 15, 17, 20]
to see diminishing-return curve. Saves sweep_K_results.json.
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

T53_DIR = os.path.join(ROOT, "experiments", "T53_r34_stage3")
sys.path.insert(0, T53_DIR)
from de_thresh import (  # noqa: E402
    PROB_COLS, gate_asymmetric, vectorized_pnl,
)

SYMS = (0, 1, 2, 3, 4)


def load_first_K_avg(K: int, prefix="bag_pred_h60"):
    base = pd.read_parquet(os.path.join(HERE, f"{prefix}_k{0:02d}.parquet"))
    p_sum = base[PROB_COLS].to_numpy(np.float64).copy()
    n = len(base)
    for k in range(1, K):
        df_k = pd.read_parquet(os.path.join(HERE, f"{prefix}_k{k:02d}.parquet"))
        if len(df_k) != n:
            raise RuntimeError(f"row count mismatch k={k}")
        p_sum += df_k[PROB_COLS].to_numpy(np.float64)
    p_avg = (p_sum / float(K)).astype(np.float32)
    sym = base["sym"].to_numpy(np.int8)
    label = base["true_label"].to_numpy(np.int64)
    mp_t = base["midprice_t"].to_numpy(np.float64)
    mp_th = base["midprice_th"].to_numpy(np.float64)
    return p_avg, sym, label, mp_t, mp_th


def per_sym_metrics(probs, sym, label, mp_t, mp_th, Tu, Td, du, dd):
    pred = gate_asymmetric(probs, Tu, Td, du, dd)
    per = []
    n_active = []
    for k in SYMS:
        m = sym == k
        per.append(float(vectorized_pnl(pred[m], label[m], mp_t[m], mp_th[m]).sum()))
        n_active.append(int((pred[m] != 1).sum()))
    total = float(vectorized_pnl(pred, label, mp_t, mp_th).sum())
    return per, n_active, total


def make_obj_loso(probs, sym, label, mp_t, mp_th):
    def f(x):
        Tu, Td, du, dd = x
        pred = gate_asymmetric(probs, Tu, Td, du, dd)
        s = 0.0
        for k in SYMS:
            m = sym == k
            s += vectorized_pnl(pred[m], label[m], mp_t[m], mp_th[m]).sum()
        return -float(s)
    return f


def make_obj_single(probs, label, mp_t, mp_th):
    def f(x):
        Tu, Td, du, dd = x
        pred = gate_asymmetric(probs, Tu, Td, du, dd)
        return -float(vectorized_pnl(pred, label, mp_t, mp_th).sum())
    return f


def de_search(objective, bounds, seeds=(0, 1, 2, 7, 42)):
    runs = []
    for seed in seeds:
        result = differential_evolution(
            objective, bounds=bounds, seed=seed, maxiter=80, popsize=24,
            polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            updating="deferred", workers=1, init="sobol",
        )
        Tu, Td, du, dd = result.x
        runs.append({
            "seed": int(seed),
            "T_up": float(Tu), "T_dn": float(Td),
            "d_up": float(du), "d_dn": float(dd),
            "obj_val": float(-result.fun),
        })
    runs.sort(key=lambda r: r["obj_val"], reverse=True)
    return runs[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K-list", default="1,2,3,5,7,10,12,15,17,20")
    ap.add_argument("--bounds-mode", default="standard",
                    choices=["standard", "tight"])
    args = ap.parse_args()

    Ks = [int(x) for x in args.K_list.split(",")]
    print(f"=== T67 K-sweep K_list={Ks} ===", flush=True)

    if args.bounds_mode == "standard":
        bounds = [(0.34, 0.75), (0.34, 0.75), (0.0, 0.30), (0.0, 0.30)]
    else:
        bounds = [(0.40, 0.55), (0.40, 0.55), (0.0, 0.05), (0.0, 0.05)]

    results = []
    for K in Ks:
        t0 = time.time()
        try:
            probs, sym, label, mp_t, mp_th = load_first_K_avg(K)
        except FileNotFoundError as e:
            print(f"  K={K}: missing parquet file: {e}", flush=True)
            continue
        load_time = time.time() - t0

        # Raw argmax
        pred_raw = probs.argmax(axis=1).astype(np.int8)
        raw_total = float(vectorized_pnl(pred_raw, label, mp_t, mp_th).sum())
        raw_per_sym = []
        for k in SYMS:
            m = sym == k
            raw_per_sym.append(float(vectorized_pnl(pred_raw[m], label[m], mp_t[m], mp_th[m]).sum()))

        # DE-LOSO
        t1 = time.time()
        best_loso = de_search(make_obj_loso(probs, sym, label, mp_t, mp_th), bounds)
        loso_per, loso_n_act, loso_total_alt = per_sym_metrics(
            probs, sym, label, mp_t, mp_th,
            best_loso["T_up"], best_loso["T_dn"], best_loso["d_up"], best_loso["d_dn"],
        )
        loso_sum = float(sum(loso_per))
        de_loso_time = time.time() - t1

        # DE-SINGLE
        t2 = time.time()
        best_single = de_search(make_obj_single(probs, label, mp_t, mp_th), bounds)
        single_per, single_n_act, single_total = per_sym_metrics(
            probs, sym, label, mp_t, mp_th,
            best_single["T_up"], best_single["T_dn"], best_single["d_up"], best_single["d_dn"],
        )
        single_loso_sum = float(sum(single_per))
        de_single_time = time.time() - t2

        row = {
            "K": K,
            "raw_argmax_total": raw_total,
            "raw_per_sym": raw_per_sym,
            "raw_loso_sum": float(sum(raw_per_sym)),
            "de_loso": {
                "best": best_loso,
                "per_sym": loso_per,
                "n_active_per_sym": loso_n_act,
                "sum_per_sym": loso_sum,
                "single_set_total": loso_total_alt,
            },
            "de_single": {
                "best": best_single,
                "per_sym": single_per,
                "n_active_per_sym": single_n_act,
                "single_set_total": single_total,
                "loso_sum": single_loso_sum,
            },
            "load_time_sec": load_time,
            "de_loso_time_sec": de_loso_time,
            "de_single_time_sec": de_single_time,
        }
        results.append(row)
        print(
            f"  K={K:>3}: raw_argmax={raw_total:+.4f}  "
            f"DE-LOSO sum={loso_sum:+.4f} (single_total={loso_total_alt:+.4f})  "
            f"DE-SINGLE total={single_total:+.4f} (loso_sum={single_loso_sum:+.4f})",
            flush=True,
        )

    out_path = os.path.join(HERE, "sweep_K_results.json")
    with open(out_path, "w") as f:
        json.dump({"K_list": Ks, "rows": results}, f, indent=2)
    print(f"\nwrote -> {out_path}", flush=True)

    # Print compact summary table
    print(f"\n{'K':>4}  {'raw':>8}  {'DE-LOSO':>9}  {'DE-LOSO_st':>10}  {'DE-SGL':>8}  {'DE-SGL_lq':>10}")
    for r in results:
        print(
            f"{r['K']:>4}  {r['raw_argmax_total']:>+8.4f}  "
            f"{r['de_loso']['sum_per_sym']:>+9.4f}  "
            f"{r['de_loso']['single_set_total']:>+10.4f}  "
            f"{r['de_single']['single_set_total']:>+8.4f}  "
            f"{r['de_single']['loso_sum']:>+10.4f}",
        )

    print(f"\nbaseline iter_010 (5-seed) DE-LOSO_sum=+24.5172 single_total=+18.8896 raw_loso_sum=+18.8896",
          flush=True)


if __name__ == "__main__":
    main()
