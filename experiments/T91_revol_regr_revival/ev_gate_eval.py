"""T91 EV-gate eval — standalone (LGB-only) and ensemble (T91-LGB + T81-NN).

Outputs:
  - 5-seed-avg predictions
  - DE asymmetric threshold optimization (LOSO-equiv = sum-of-per-sym)
  - Comparison vs T75 (+36.23 LGB-only) and iter_014 (+38.28 NN+LGB)
  - Ensemble with T81 NN at the same weight ratios as iter_014's sweep
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

T81_DIR = os.path.join(ROOT, "experiments", "T81_nn_regression_pnl")
T75_DIR = os.path.join(ROOT, "experiments", "T75_regression_dmid")

SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001
ITER012_LOSO = 26.44
T75_LGB_ONLY_LOSO = 36.2281
ITER014_LOSO = 38.28102

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


def load_T91_avg(variant, seeds=SEEDS):
    base_path = os.path.join(HERE, f"pred_T91_{variant}_seed{seeds[0]}.parquet")
    base = pd.read_parquet(base_path)
    n = len(base)
    p_sum = base["pred_dmid_norm"].to_numpy(np.float64).copy()
    for s in seeds[1:]:
        p = os.path.join(HERE, f"pred_T91_{variant}_seed{s}.parquet")
        df_s = pd.read_parquet(p)
        if len(df_s) != n:
            raise RuntimeError(f"row count mismatch seed={s}")
        if not (df_s["sym"].equals(base["sym"]) and df_s["t"].equals(base["t"])
                and df_s["date"].equals(base["date"])):
            raise RuntimeError(f"row order mismatch seed={s}")
        p_sum += df_s["pred_dmid_norm"].to_numpy(np.float64)
    p_avg = p_sum / float(len(seeds))
    out = base.copy()
    out["pred_dmid_norm"] = p_avg.astype(np.float32)
    return out, p_avg


def load_T81_avg(seeds=SEEDS):
    base = pd.read_parquet(os.path.join(T81_DIR, f"pred_T81_seed{seeds[0]}.parquet"))
    n = len(base)
    p_sum = base["pred_dmid_norm"].to_numpy(np.float64).copy()
    for s in seeds[1:]:
        df_s = pd.read_parquet(os.path.join(T81_DIR, f"pred_T81_seed{s}.parquet"))
        if len(df_s) != n:
            raise RuntimeError(f"row count mismatch nn seed={s}")
        p_sum += df_s["pred_dmid_norm"].to_numpy(np.float64)
    p_avg = p_sum / float(len(seeds))
    return base, p_avg


def evaluate(df, label):
    folds = split_by_sym(df)
    pred_all = df["pred_dmid_norm"].to_numpy(np.float64)
    mp_t_all = df["midprice_t"].to_numpy(np.float64)
    mp_th_all = df["midprice_th"].to_numpy(np.float64)
    print(f"\n=== {label} ===", flush=True)
    print(f"  per-sym pred mean/std:", flush=True)
    for fk in folds:
        print(f"    sym={fk['sym']}: mean={fk['pred'].mean():+.6e} std={fk['pred'].std():.6e}",
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
    print(f"  vs T75 (+{T75_LGB_ONLY_LOSO}):  {de_loso_sum-T75_LGB_ONLY_LOSO:+.4f}", flush=True)
    print(f"  vs iter_014 (+{ITER014_LOSO:.2f}): {de_loso_sum-ITER014_LOSO:+.4f}", flush=True)

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="v1_all32")
    ap.add_argument("--seeds", default="1,7,13,42,100")
    ap.add_argument("--skip-ensemble", action="store_true")
    args = ap.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))

    print(f"=== T91 ensemble eval variant={args.variant} seeds={seeds} ===", flush=True)
    df_t91, p_t91 = load_T91_avg(args.variant, seeds=seeds)
    print(f"  loaded T91 avg pred ({len(df_t91):,} rows), pred mean={p_t91.mean():+.6e} std={p_t91.std():.6e}",
          flush=True)

    # 1) Standalone T91 LGB
    r_lgb = evaluate(df_t91, f"T91-LGB-only ({args.variant} 5seed avg)")

    out = {
        "task": f"T91 ReVol regr revival ({args.variant})",
        "variant": args.variant,
        "seeds": list(seeds),
        "n_test": int(len(df_t91)),
        "fee_rate": FEE,
        "ref": {
            "iter012_loso": ITER012_LOSO,
            "T75_LGB_only_loso": T75_LGB_ONLY_LOSO,
            "iter014_loso": ITER014_LOSO,
        },
        "lgb_only": r_lgb,
    }

    if not args.skip_ensemble:
        print("\n>>> Building NN+LGB ensemble (T81 NN + T91 LGB)", flush=True)
        df_nn_base, p_nn = load_T81_avg(seeds=seeds)
        # alignment guard
        if not (df_nn_base["sym"].equals(df_t91["sym"]) and df_nn_base["t"].equals(df_t91["t"])
                and df_nn_base["date"].equals(df_t91["date"])):
            raise RuntimeError("NN/LGB row order mismatch — order assumed identical")
        corr = float(np.corrcoef(p_nn, p_t91)[0, 1])
        print(f"  NN-T81 vs LGB-T91 cross-corr: {corr:.4f}", flush=True)

        weights = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.5, 1.0), (1.0, 1.5), (2.0, 1.0),
                   (1.0, 2.0), (1.0, 0.5), (0.5, 1.0)]
        ensemble_results = []
        for wn, wl in weights:
            p_combo = (wn * p_nn + wl * p_t91) / (wn + wl)
            df = df_t91.copy()
            df["pred_dmid_norm"] = p_combo.astype(np.float32)
            r = evaluate(df, f"ensemble w_nn={wn} w_lgb={wl}")
            r["w_nn"] = wn
            r["w_lgb"] = wl
            ensemble_results.append(r)
        best_ens = max(ensemble_results, key=lambda r: r["de_loso"]["sum_per_sym"])
        out["ensemble"] = {
            "nn_lgb_corr": corr,
            "weights_tried": [{"w_nn": w[0], "w_lgb": w[1]} for w in weights],
            "results": ensemble_results,
            "best": best_ens,
        }
        print(f"\n{'='*78}", flush=True)
        print(f"BEST ensemble: {best_ens['label']}  → DE LOSO {best_ens['de_loso']['sum_per_sym']:+.4f}",
              flush=True)
        print(f"  vs iter_014 ({ITER014_LOSO:.2f}): {best_ens['de_loso']['sum_per_sym']-ITER014_LOSO:+.4f}",
              flush=True)
        print(f"{'='*78}", flush=True)

    out_path = os.path.join(HERE, f"ev_gate_results_{args.variant}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
