"""T67 DE 4D thresh search on K-bag-averaged test predictions.

Loads bag_pred_h60_k{K}.parquet for k = 0..K-1, averages probs per row,
runs DE on (T_up, T_dn, d_up, d_dn) with two objectives:
  1) SINGLE-set total cum_pnl on 442k rows
  2) LOSO-equivalent SUM of per-sym cum_pnl

Saves de_results_K{K}.json.
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


def load_avg_pred(K: int, prefix="bag_pred_h60") -> pd.DataFrame:
    base = pd.read_parquet(os.path.join(HERE, f"{prefix}_k{0:02d}.parquet"))
    p_sum = base[PROB_COLS].to_numpy(np.float64).copy()
    n = len(base)
    for k in range(1, K):
        df_k = pd.read_parquet(os.path.join(HERE, f"{prefix}_k{k:02d}.parquet"))
        if len(df_k) != n:
            raise RuntimeError(f"row count mismatch k={k}: {len(df_k)} vs {n}")
        if not (df_k["sym"].equals(base["sym"]) and df_k["date"].equals(base["date"]) and df_k["t"].equals(base["t"])):
            raise RuntimeError(f"row order mismatch k={k}")
        p_sum += df_k[PROB_COLS].to_numpy(np.float64)
    p_avg = p_sum / float(K)
    out = base.drop(columns=PROB_COLS).copy()
    out["prob_0"] = p_avg[:, 0].astype(np.float32)
    out["prob_1"] = p_avg[:, 1].astype(np.float32)
    out["prob_2"] = p_avg[:, 2].astype(np.float32)
    return out


def split_by_sym(df: pd.DataFrame):
    out = []
    for k in SYMS:
        sub = df[df["sym"] == k]
        out.append({
            "sym": int(k),
            "probs": sub[PROB_COLS].to_numpy(np.float32),
            "label": sub["true_label"].to_numpy(np.int64),
            "mp_t": sub["midprice_t"].to_numpy(np.float64),
            "mp_th": sub["midprice_th"].to_numpy(np.float64),
            "n": len(sub),
        })
    return out


def per_sym_pnl_at_thresh(folds, Tu, Td, du, dd):
    per = []
    n_active = []
    for f in folds:
        pred = gate_asymmetric(f["probs"], Tu, Td, du, dd)
        per.append(float(vectorized_pnl(pred, f["label"], f["mp_t"], f["mp_th"]).sum()))
        n_active.append(int((pred != 1).sum()))
    return per, n_active


def make_objective_loso(folds):
    def f(x):
        Tu, Td, du, dd = x
        s = 0.0
        for fk in folds:
            pred = gate_asymmetric(fk["probs"], Tu, Td, du, dd)
            s += vectorized_pnl(pred, fk["label"], fk["mp_t"], fk["mp_th"]).sum()
        return -float(s)
    return f


def make_objective_single(probs, label, mp_t, mp_th):
    def f(x):
        Tu, Td, du, dd = x
        pred = gate_asymmetric(probs, Tu, Td, du, dd)
        return -float(vectorized_pnl(pred, label, mp_t, mp_th).sum())
    return f


def de_search(objective_fn, bounds, seeds=(0, 1, 2, 7, 42)):
    runs = []
    for seed in seeds:
        t0 = time.time()
        result = differential_evolution(
            objective_fn, bounds=bounds, seed=seed, maxiter=80, popsize=24,
            polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            updating="deferred", workers=1, init="sobol",
        )
        Tu, Td, du, dd = result.x
        runs.append({
            "seed": int(seed),
            "T_up": float(Tu), "T_dn": float(Td),
            "d_up": float(du), "d_dn": float(dd),
            "obj_val": float(-result.fun),
            "nfev": int(result.nfev),
            "elapsed_sec": float(time.time() - t0),
        })
    runs.sort(key=lambda r: r["obj_val"], reverse=True)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--prefix", default="bag_pred_h60")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    K = args.K
    print(f"=== T67 DE 4D thresh search K={K} ===", flush=True)

    t0 = time.time()
    df = load_avg_pred(K, prefix=args.prefix)
    print(f"  loaded {len(df):,} rows in {time.time()-t0:.1f}s", flush=True)

    folds = split_by_sym(df)
    for fk in folds:
        print(f"  sym={fk['sym']}: n={fk['n']:,}", flush=True)

    probs_all = df[PROB_COLS].to_numpy(np.float32)
    label_all = df["true_label"].to_numpy(np.int64)
    mp_t_all = df["midprice_t"].to_numpy(np.float64)
    mp_th_all = df["midprice_th"].to_numpy(np.float64)

    # Raw argmax baseline
    pred_raw = probs_all.argmax(axis=1).astype(np.int8)
    raw_total = float(vectorized_pnl(pred_raw, label_all, mp_t_all, mp_th_all).sum())
    raw_per_sym = []
    for fk in folds:
        p = fk["probs"].argmax(axis=1).astype(np.int8)
        raw_per_sym.append(float(vectorized_pnl(p, fk["label"], fk["mp_t"], fk["mp_th"]).sum()))
    print(f"\n  RAW argmax: total_cum_pnl={raw_total:+.4f}", flush=True)
    print(f"  RAW per-sym: {[round(x,3) for x in raw_per_sym]}  loso_equiv_sum={sum(raw_per_sym):+.4f}", flush=True)

    bounds = [(0.34, 0.75), (0.34, 0.75), (0.0, 0.30), (0.0, 0.30)]

    print(f"\n[1/2] SINGLE-set DE (objective = total cum_pnl on full rows) ...", flush=True)
    obj_single = make_objective_single(probs_all, label_all, mp_t_all, mp_th_all)
    de_single = de_search(obj_single, bounds)
    best_single = de_single[0]
    Tu, Td, du, dd = best_single["T_up"], best_single["T_dn"], best_single["d_up"], best_single["d_dn"]
    pred_de_single = gate_asymmetric(probs_all, Tu, Td, du, dd)
    de_single_total = float(vectorized_pnl(pred_de_single, label_all, mp_t_all, mp_th_all).sum())
    de_single_per_sym, de_single_n_active = per_sym_pnl_at_thresh(folds, Tu, Td, du, dd)
    print(f"  best DE-single: T_up={Tu:.4f} T_dn={Td:.4f} d_up={du:.4f} d_dn={dd:.4f}", flush=True)
    print(f"    -> total={de_single_total:+.4f}  per_sym={[round(x,3) for x in de_single_per_sym]} "
          f"loso_equiv_sum={sum(de_single_per_sym):+.4f}", flush=True)

    print(f"\n[2/2] LOSO-EQUIV DE (objective = SUM of per-sym cum_pnl) ...", flush=True)
    obj_loso = make_objective_loso(folds)
    de_loso = de_search(obj_loso, bounds)
    best_loso = de_loso[0]
    Tu2, Td2, du2, dd2 = best_loso["T_up"], best_loso["T_dn"], best_loso["d_up"], best_loso["d_dn"]
    de_loso_per_sym, de_loso_n_active = per_sym_pnl_at_thresh(folds, Tu2, Td2, du2, dd2)
    de_loso_sum = float(sum(de_loso_per_sym))
    pred_de_loso = gate_asymmetric(probs_all, Tu2, Td2, du2, dd2)
    de_loso_total_alt = float(vectorized_pnl(pred_de_loso, label_all, mp_t_all, mp_th_all).sum())
    print(f"  best DE-loso: T_up={Tu2:.4f} T_dn={Td2:.4f} d_up={du2:.4f} d_dn={dd2:.4f}", flush=True)
    print(f"    -> sum_per_sym={de_loso_sum:+.4f}  per_sym={[round(x,3) for x in de_loso_per_sym]} "
          f"single_set_total={de_loso_total_alt:+.4f}", flush=True)

    out = {
        "task": "T67 DE 4D thresh search on K-bag-avg predictions",
        "K": K,
        "prefix": args.prefix,
        "n_test": len(df),
        "raw_argmax": {
            "total_cum_pnl": raw_total,
            "per_sym": raw_per_sym,
            "loso_equiv_sum": float(sum(raw_per_sym)),
        },
        "de_single_set": {
            "best": best_single,
            "all_runs": de_single,
            "total_cum_pnl_at_best": de_single_total,
            "per_sym_at_best": de_single_per_sym,
            "n_active_per_sym": de_single_n_active,
            "loso_equiv_sum_at_best": float(sum(de_single_per_sym)),
        },
        "de_loso_equiv": {
            "best": best_loso,
            "all_runs": de_loso,
            "per_sym_at_best": de_loso_per_sym,
            "n_active_per_sym": de_loso_n_active,
            "sum_per_sym": de_loso_sum,
            "single_set_total_at_best": de_loso_total_alt,
        },
        "iter010_baseline": {
            "loso_equiv_sum": 24.5172,
            "delta_t67_loso_vs_iter010": de_loso_sum - 24.5172,
            "delta_t67_single_vs_iter010": de_single_total - 24.5172,
        },
    }
    tag = args.tag or f"K{K}"
    out_path = os.path.join(HERE, f"de_results_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote -> {out_path}", flush=True)

    print(f"\n{'='*78}\n=== SUMMARY (K={K}) ===\n{'='*78}", flush=True)
    print(f"RAW argmax:       single_set_total={raw_total:+.4f}  loso_equiv_sum={sum(raw_per_sym):+.4f}", flush=True)
    print(f"DE single-set:    single_set_total={de_single_total:+.4f}  loso_equiv_sum={sum(de_single_per_sym):+.4f}", flush=True)
    print(f"DE loso-equiv:    sum_per_sym={de_loso_sum:+.4f}  single_set_total={de_loso_total_alt:+.4f}", flush=True)
    print(f"vs iter_010 +24.5172: delta_loso={de_loso_sum - 24.5172:+.4f}  delta_single={de_single_total - 24.5172:+.4f}", flush=True)


if __name__ == "__main__":
    main()
