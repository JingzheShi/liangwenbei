"""Mega-ensemble eval: T87 SPO+ + T89 CB + T95 GRU + (T81, T75 baselines).

T95 GRU has cross-corr 0.37 to T81 NN, 0.35 to T75 LGB — by far the lowest
cross-corr we've seen. If T95 also has low cross-corr to T87/T89, the 3-way
T87+T89+T95 may surpass current T87+T89 +41.01.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
T75_DIR = os.path.join(ROOT, "experiments", "T75_regression_dmid")
T81_DIR = os.path.join(ROOT, "experiments", "T81_nn_regression_pnl")
T87_DIR = os.path.join(ROOT, "experiments", "T87_spo_dfl")
T89_DIR = os.path.join(ROOT, "experiments", "T89_catboost_regression")

SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001
ITER014 = 38.281
ITER015_T87T75 = 40.129
ITER015_T87T89 = 41.014  # current new SOTA
SEEDS = (1, 7, 13, 42, 100)


def vectorized_pnl(pred_action, mp_t, mp_th):
    side = pred_action.astype(np.float64) - 1.0
    abs_side = np.abs(side)
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * abs_side * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee_pnl) / denom


def ev_gate_asym(pred, thr_up, thr_dn):
    a = np.full(len(pred), 1, dtype=np.int8)
    a[pred > thr_up] = 2
    a[pred < -thr_dn] = 0
    return a


def split_sym(df):
    return [{
        "sym": int(k),
        "pred": df[df["sym"] == k]["pred_dmid_norm"].to_numpy(np.float64),
        "mp_t": df[df["sym"] == k]["midprice_t"].to_numpy(np.float64),
        "mp_th": df[df["sym"] == k]["midprice_th"].to_numpy(np.float64),
    } for k in SYMS]


def make_obj(folds):
    pred = [f["pred"] for f in folds]
    mp_t = [f["mp_t"] for f in folds]
    mp_th = [f["mp_th"] for f in folds]
    def f(x):
        thr_up, thr_dn = x
        s = 0.0
        for i in range(len(folds)):
            a = ev_gate_asym(pred[i], thr_up, thr_dn)
            s += vectorized_pnl(a, mp_t[i], mp_th[i]).sum()
        return -float(s)
    return f


def de_search(obj, bounds, seeds=(0, 1, 2, 7, 42)):
    runs = []
    for sd in seeds:
        r = differential_evolution(
            obj, bounds=bounds, seed=sd, maxiter=80, popsize=24,
            polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            updating="deferred", workers=1, init="sobol",
        )
        runs.append({"seed": int(sd), "thr_up": float(r.x[0]),
                     "thr_dn": float(r.x[1]), "obj_val": float(-r.fun)})
    runs.sort(key=lambda r: r["obj_val"], reverse=True)
    return runs


def avg_preds(dir_path, prefix, suffix=""):
    base = pd.read_parquet(os.path.join(dir_path, f"{prefix}{SEEDS[0]}{suffix}.parquet"))
    p = np.zeros(len(base), dtype=np.float64)
    for s in SEEDS:
        df = pd.read_parquet(os.path.join(dir_path, f"{prefix}{s}{suffix}.parquet"))
        if not (df["sym"].equals(base["sym"]) and df["t"].equals(base["t"])):
            raise RuntimeError(f"row mismatch {prefix} seed={s}")
        p += df["pred_dmid_norm"].to_numpy(np.float64)
    p /= len(SEEDS)
    return base, p


def evaluate(p_combined, base, label):
    df = base.copy()
    df["pred_dmid_norm"] = p_combined.astype(np.float32)
    folds = split_sym(df)
    obj = make_obj(folds)
    runs = de_search(obj, [(0.0, 0.0040), (0.0, 0.0040)])
    best = runs[0]
    thr_up, thr_dn = best["thr_up"], best["thr_dn"]
    de_per = []
    for f in folds:
        a = ev_gate_asym(f["pred"], thr_up, thr_dn)
        de_per.append(float(vectorized_pnl(a, f["mp_t"], f["mp_th"]).sum()))
    de_sum = float(sum(de_per))
    print(f"  [{label:<55s}] DE={de_sum:+.4f}  thr_up={thr_up:.4e} thr_dn={thr_dn:.4e}", flush=True)
    return {"label": label, "thr_up": thr_up, "thr_dn": thr_dn, "de_sum": de_sum, "per_sym": de_per}


def main():
    print("=== Mega ensemble eval: T87 + T89 + T95 + baselines ===", flush=True)
    t0 = time.time()

    base_t81, p_t81 = avg_preds(T81_DIR, "pred_T81_seed")
    base_t75, p_t75 = avg_preds(T75_DIR, "pred_T75_seed")
    base_t87, p_t87 = avg_preds(T87_DIR, "pred_T87_seed", suffix="_main")
    base_t89, p_t89 = avg_preds(T89_DIR, "pred_T89_seed")
    base_t95, p_t95 = avg_preds(HERE, "pred_T95_gru_w100_C_seed")

    for nm, b in [("T75", base_t75), ("T87", base_t87), ("T89", base_t89), ("T95", base_t95)]:
        if not (base_t81["sym"].equals(b["sym"]) and base_t81["t"].equals(b["t"])):
            raise RuntimeError(f"row order mismatch T81 vs {nm}")

    # Cross-corrs
    print("\nCross-correlations:", flush=True)
    names = ["T81_NN", "T75_LGB", "T87_SPO+", "T89_CB", "T95_GRU"]
    preds = [p_t81, p_t75, p_t87, p_t89, p_t95]
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            c = float(np.corrcoef(preds[i], preds[j])[0, 1])
            print(f"  corr({names[i]:<10s}, {names[j]:<10s}) = {c:.4f}", flush=True)

    print("\nPred stats:", flush=True)
    for n, p in zip(names, preds):
        print(f"  {n:<10s}: mean={p.mean():+.6e} std={p.std():.6e}", flush=True)

    print("\n=== Singles ===", flush=True)
    results = []
    for n, p in zip(names, preds):
        results.append(evaluate(p, base_t81, f"{n}-only"))

    print("\n=== 2-way (key combos) ===", flush=True)
    results.append(evaluate((1.0*p_t87 + 1.5*p_t75) / 2.5, base_t81, "T87+T75 w=1:1.5 (iter_015 packaged)"))
    for w_cb in [1.0, 1.5]:
        results.append(evaluate((1.0*p_t87 + w_cb*p_t89) / (1.0+w_cb), base_t81, f"T87+T89 w=1:{w_cb}"))
    # T87+T95 — key new test
    for w_t95 in [0.3, 0.5, 0.7, 1.0]:
        results.append(evaluate((1.0*p_t87 + w_t95*p_t95) / (1.0+w_t95), base_t81, f"T87+T95 w=1:{w_t95}"))
    # T89+T95
    for w_t95 in [0.5, 0.7, 1.0]:
        results.append(evaluate((1.0*p_t89 + w_t95*p_t95) / (1.0+w_t95), base_t81, f"T89+T95 w=1:{w_t95}"))

    print("\n=== 3-way (T87 + T89 + T95) — primary new candidate ===", flush=True)
    # Search around T95 weight
    for w_nn, w_cb, w_t95 in [
        (1, 1, 0.3), (1, 1, 0.5), (1, 1, 0.7), (1, 1, 1.0),
        (1, 1.5, 0.3), (1, 1.5, 0.5), (1, 1.5, 0.7), (1, 1.5, 1.0),
        (1, 0.5, 0.5), (1, 0.5, 1.0),
        (1.5, 1, 0.5), (1.5, 1.5, 0.5),
    ]:
        wsum = w_nn + w_cb + w_t95
        p = (w_nn*p_t87 + w_cb*p_t89 + w_t95*p_t95) / wsum
        results.append(evaluate(p, base_t81, f"T87+T89+T95 w={w_nn}:{w_cb}:{w_t95}"))

    print("\n=== 3-way (T87 + T75 + T95) ===", flush=True)
    for w_lgb, w_t95 in [(1, 0.3), (1, 0.5), (1, 0.7), (1.5, 0.5), (1.5, 0.3)]:
        wsum = 1.0 + w_lgb + w_t95
        p = (1.0*p_t87 + w_lgb*p_t75 + w_t95*p_t95) / wsum
        results.append(evaluate(p, base_t81, f"T87+T75+T95 w=1:{w_lgb}:{w_t95}"))

    print("\n=== 4-way (T87 + T89 + T75 + T95) ===", flush=True)
    for w_cb, w_lgb, w_t95 in [(1, 0.5, 0.3), (1, 0.5, 0.5), (1, 1, 0.3), (1, 1, 0.5),
                                (1.5, 0.5, 0.5), (1.5, 0.5, 0.3)]:
        wsum = 1.0 + w_cb + w_lgb + w_t95
        p = (1.0*p_t87 + w_cb*p_t89 + w_lgb*p_t75 + w_t95*p_t95) / wsum
        results.append(evaluate(p, base_t81, f"T87+T89+T75+T95 w=1:{w_cb}:{w_lgb}:{w_t95}"))

    # Best
    best = max(results, key=lambda r: r["de_sum"])
    print(f"\n{'='*100}", flush=True)
    print(f"BEST: {best['label']}  DE={best['de_sum']:+.4f}", flush=True)
    print(f"  vs iter_014 (+{ITER014}): {best['de_sum']-ITER014:+.4f}", flush=True)
    print(f"  vs iter_015 packaged (T87+T75 +{ITER015_T87T75}): {best['de_sum']-ITER015_T87T75:+.4f}", flush=True)
    print(f"  vs T87+T89 (+{ITER015_T87T89}): {best['de_sum']-ITER015_T87T89:+.4f}", flush=True)
    print(f"  per_sym = {[round(x,2) for x in best['per_sym']]}", flush=True)
    print(f"{'='*100}", flush=True)
    print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)

    out = {"results": results, "best": best, "iter014": ITER014,
           "iter015_t87t75": ITER015_T87T75, "iter015_t87t89": ITER015_T87T89}
    with open(os.path.join(HERE, "ev_gate_mega_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote -> {os.path.join(HERE, 'ev_gate_mega_results.json')}", flush=True)


if __name__ == "__main__":
    main()
