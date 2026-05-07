"""T64 DE 4D threshold search on chosen strategy's 5-seed-averaged predictions.

Loads pred_{strategy}_seed{S}.parquet for S in 1,7,13,42,100, averages probs across
seeds (per row), then computes:
  1) Raw argmax cum_pnl (single 442k + per-sym/loso-equiv)
  2) DE 4D on single 442k (objective = total cum_pnl)
  3) DE 4D treating sym0..4 as 5 folds (objective = SUM of per-sym cum_pnl)
     => LOSO-equivalent comparison vs T59 +24.52

Saves de_results_{strategy}.json.
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
    FEE, PROB_COLS, gate_asymmetric, vectorized_pnl,
)

SYMS = (0, 1, 2, 3, 4)


def load_avg_pred(strategy, seeds):
    base = pd.read_parquet(os.path.join(HERE, f"pred_{strategy}_seed{seeds[0]}.parquet"))
    p_sum = base[PROB_COLS].to_numpy(np.float64).copy()
    n = len(base)
    for s in seeds[1:]:
        df_s = pd.read_parquet(os.path.join(HERE, f"pred_{strategy}_seed{s}.parquet"))
        if len(df_s) != n:
            raise RuntimeError(f"row count mismatch seed={s}: {len(df_s)} vs {n}")
        if not (df_s["sym"].equals(base["sym"]) and df_s["date"].equals(base["date"]) and df_s["t"].equals(base["t"])):
            raise RuntimeError(f"row order mismatch seed={s}")
        p_sum += df_s[PROB_COLS].to_numpy(np.float64)
    p_avg = p_sum / float(len(seeds))
    out = base.drop(columns=PROB_COLS).copy()
    out["prob_0"] = p_avg[:, 0].astype(np.float32)
    out["prob_1"] = p_avg[:, 1].astype(np.float32)
    out["prob_2"] = p_avg[:, 2].astype(np.float32)
    return out


def split_by_sym(df):
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


def make_obj_loso(folds):
    def f(x):
        Tu, Td, du, dd = x
        s = 0.0
        for fk in folds:
            pred = gate_asymmetric(fk["probs"], Tu, Td, du, dd)
            s += vectorized_pnl(pred, fk["label"], fk["mp_t"], fk["mp_th"]).sum()
        return -float(s)
    return f


def make_obj_single(probs, label, mp_t, mp_th):
    def f(x):
        Tu, Td, du, dd = x
        pred = gate_asymmetric(probs, Tu, Td, du, dd)
        return -float(vectorized_pnl(pred, label, mp_t, mp_th).sum())
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
        Tu, Td, du, dd = result.x
        runs.append({
            "seed": int(sd),
            "T_up": float(Tu), "T_dn": float(Td),
            "d_up": float(du), "d_dn": float(dd),
            "obj_val": float(-result.fun),
            "nfev": int(result.nfev),
            "elapsed_sec": float(time.time() - t0),
        })
    runs.sort(key=lambda r: r["obj_val"], reverse=True)
    return runs


def per_sym_pnl(folds, Tu, Td, du, dd):
    per = []
    n_active = []
    for f in folds:
        pred = gate_asymmetric(f["probs"], Tu, Td, du, dd)
        per.append(float(vectorized_pnl(pred, f["label"], f["mp_t"], f["mp_th"]).sum()))
        n_active.append(int((pred != 1).sum()))
    return per, n_active


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True, choices=["V1", "V2", "V3", "V4"])
    ap.add_argument("--seeds", default="1,7,13,42,100")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")]
    strategy = args.strategy
    print(f"=== T64 DE 4D thresh strategy={strategy} seeds={seeds} ===", flush=True)

    t0 = time.time()
    df = load_avg_pred(strategy, seeds)
    print(f"  loaded {len(df):,} rows in {time.time()-t0:.1f}s", flush=True)

    folds = split_by_sym(df)
    for fk in folds:
        print(f"  sym={fk['sym']}: n={fk['n']:,}", flush=True)

    probs_all = df[PROB_COLS].to_numpy(np.float32)
    label_all = df["true_label"].to_numpy(np.int64)
    mp_t_all = df["midprice_t"].to_numpy(np.float64)
    mp_th_all = df["midprice_th"].to_numpy(np.float64)

    pred_raw = probs_all.argmax(axis=1).astype(np.int8)
    raw_total = float(vectorized_pnl(pred_raw, label_all, mp_t_all, mp_th_all).sum())
    raw_per_sym = []
    for fk in folds:
        p = fk["probs"].argmax(axis=1).astype(np.int8)
        raw_per_sym.append(float(vectorized_pnl(p, fk["label"], fk["mp_t"], fk["mp_th"]).sum()))
    print(f"\n  RAW argmax: total={raw_total:+.4f} per_sym={[round(x,3) for x in raw_per_sym]} sum={sum(raw_per_sym):+.4f}", flush=True)

    bounds = [(0.34, 0.75), (0.34, 0.75), (0.0, 0.30), (0.0, 0.30)]

    print(f"\n[1/2] DE single-set ...", flush=True)
    obj_single = make_obj_single(probs_all, label_all, mp_t_all, mp_th_all)
    de_single = de_search(obj_single, bounds)
    best_s = de_single[0]
    Tu, Td, du, dd = best_s["T_up"], best_s["T_dn"], best_s["d_up"], best_s["d_dn"]
    pred_de_single = gate_asymmetric(probs_all, Tu, Td, du, dd)
    de_single_total = float(vectorized_pnl(pred_de_single, label_all, mp_t_all, mp_th_all).sum())
    de_single_per_sym, de_single_na = per_sym_pnl(folds, Tu, Td, du, dd)
    print(f"  best DE-single: T_up={Tu:.4f} T_dn={Td:.4f} d_up={du:.4f} d_dn={dd:.4f}", flush=True)
    print(f"    total={de_single_total:+.4f}  per_sym={[round(x,3) for x in de_single_per_sym]} sum={sum(de_single_per_sym):+.4f}", flush=True)

    print(f"\n[2/2] DE loso-equiv (sum-of-per-sym) ...", flush=True)
    obj_loso = make_obj_loso(folds)
    de_loso = de_search(obj_loso, bounds)
    best_l = de_loso[0]
    Tu2, Td2, du2, dd2 = best_l["T_up"], best_l["T_dn"], best_l["d_up"], best_l["d_dn"]
    de_loso_per_sym, de_loso_na = per_sym_pnl(folds, Tu2, Td2, du2, dd2)
    de_loso_sum = float(sum(de_loso_per_sym))
    pred_de_loso = gate_asymmetric(probs_all, Tu2, Td2, du2, dd2)
    de_loso_total = float(vectorized_pnl(pred_de_loso, label_all, mp_t_all, mp_th_all).sum())
    print(f"  best DE-loso: T_up={Tu2:.4f} T_dn={Td2:.4f} d_up={du2:.4f} d_dn={dd2:.4f}", flush=True)
    print(f"    sum_per_sym={de_loso_sum:+.4f}  per_sym={[round(x,3) for x in de_loso_per_sym]} single_total={de_loso_total:+.4f}", flush=True)

    out = {
        "task": f"T64 DE 4D thresh strategy={strategy}",
        "strategy": strategy,
        "seeds": seeds,
        "n_test": len(df),
        "raw_argmax": {
            "total_cum_pnl": raw_total,
            "per_sym": raw_per_sym,
            "loso_equiv_sum": float(sum(raw_per_sym)),
        },
        "de_single_set": {
            "best": best_s,
            "all_runs": de_single,
            "total_cum_pnl_at_best": de_single_total,
            "per_sym_at_best": de_single_per_sym,
            "n_active_per_sym": de_single_na,
            "loso_equiv_sum_at_best": float(sum(de_single_per_sym)),
        },
        "de_loso_equiv": {
            "best": best_l,
            "all_runs": de_loso,
            "per_sym_at_best": de_loso_per_sym,
            "n_active_per_sym": de_loso_na,
            "sum_per_sym": de_loso_sum,
            "single_set_total_at_best": de_loso_total,
        },
        "compare_t59": {
            "t59_de_loso_sum": 24.5172,
            "t64_de_single_total_vs_t59": de_single_total - 24.5172,
            "t64_de_loso_sum_vs_t59": de_loso_sum - 24.5172,
        },
    }
    out_path = os.path.join(HERE, f"de_results_{strategy}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote -> {out_path}", flush=True)

    print(f"\n{'='*78}\n=== SUMMARY {strategy} ===\n{'='*78}", flush=True)
    print(f"RAW argmax:      single_total={raw_total:+.4f} loso_equiv_sum={sum(raw_per_sym):+.4f}", flush=True)
    print(f"DE single-set:   single_total={de_single_total:+.4f} loso_equiv_sum={sum(de_single_per_sym):+.4f}", flush=True)
    print(f"DE loso-equiv:   sum_per_sym={de_loso_sum:+.4f} single_total={de_loso_total:+.4f}", flush=True)
    print(f"vs T59 DE +24.52: t64_de_loso_diff={de_loso_sum - 24.5172:+.4f}", flush=True)


if __name__ == "__main__":
    main()
