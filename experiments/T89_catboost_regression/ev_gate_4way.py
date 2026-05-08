"""4-way ensemble eval — T87 SPO+ NN + T81 L2 NN + T75 LGB + T89 CB.

Tests T87 (new SOTA NN) combined with T75/T89/T81 in various weight combos.
Goal: find if iter_015 should be T87+T75 (current at +40.13) OR T87+T75+T89.
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
T89_DIR = HERE

SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001
ITER014 = 38.281
ITER015_T87T75 = 40.129  # current iter_015 candidate

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
    print(f"  [{label:<48s}] DE={de_sum:+.4f}  thr_up={thr_up:.4e} thr_dn={thr_dn:.4e}  per_sym={[round(x,2) for x in de_per]}", flush=True)
    return {"label": label, "thr_up": thr_up, "thr_dn": thr_dn, "de_sum": de_sum, "per_sym": de_per}


def main():
    print("=== 4-way ensemble eval ===", flush=True)
    t0 = time.time()

    base_t81, p_t81 = avg_preds(T81_DIR, "pred_T81_seed")
    base_t75, p_t75 = avg_preds(T75_DIR, "pred_T75_seed")
    base_t89, p_t89 = avg_preds(T89_DIR, "pred_T89_seed")
    base_t87, p_t87 = avg_preds(T87_DIR, "pred_T87_seed", suffix="_main")

    # Sanity row order
    for nm, b in [("T75", base_t75), ("T89", base_t89), ("T87", base_t87)]:
        if not (base_t81["sym"].equals(b["sym"]) and base_t81["t"].equals(b["t"])):
            raise RuntimeError(f"row order mismatch T81 vs {nm}")

    # Cross-corrs
    print("\nCross-correlations:", flush=True)
    names = ["T81_NN", "T75_LGB", "T89_CB", "T87_SPO+"]
    preds = [p_t81, p_t75, p_t89, p_t87]
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            c = float(np.corrcoef(preds[i], preds[j])[0, 1])
            print(f"  corr({names[i]:<8s}, {names[j]:<8s}) = {c:.4f}", flush=True)

    # Pred stats
    print("\nPred stats:", flush=True)
    for n, p in zip(names, preds):
        print(f"  {n:<10s}: mean={p.mean():+.6e} std={p.std():.6e}", flush=True)

    print("\n=== Singles ===", flush=True)
    results = []
    for n, p in zip(names, preds):
        results.append(evaluate(p, base_t81, f"{n}-only"))

    print("\n=== 2-way combos ===", flush=True)
    # T87+T75 (current iter_015)
    results.append(evaluate((1.0*p_t87 + 1.5*p_t75) / 2.5, base_t81, "T87+T75 w=1:1.5 (iter_015)"))
    # T87+T89
    for w_cb in [0.5, 1.0, 1.5, 2.0]:
        results.append(evaluate((1.0*p_t87 + w_cb*p_t89) / (1.0+w_cb), base_t81, f"T87+T89 w=1:{w_cb}"))
    # T75+T89 (LGB-CB high corr — should be near LGB)
    for w_cb in [0.5, 1.0]:
        results.append(evaluate((1.0*p_t75 + w_cb*p_t89) / (1.0+w_cb), base_t81, f"T75+T89 w=1:{w_cb}"))

    print("\n=== 3-way (T87 + T75 + T89) — primary candidate ===", flush=True)
    weight_grid = [
        (1, 1, 1), (1, 1.5, 1), (1, 1, 1.5), (2, 1, 1),
        (1, 1.5, 0.5), (1, 0.5, 1.5), (1, 2, 1), (1, 1, 2),
        (2, 1.5, 1), (2, 1, 1.5), (1.5, 1, 1), (1, 1.5, 1.5),
        (1, 0.5, 0.5),
    ]
    for w_nn, w_lgb, w_cb in weight_grid:
        wsum = w_nn + w_lgb + w_cb
        p = (w_nn*p_t87 + w_lgb*p_t75 + w_cb*p_t89) / wsum
        results.append(evaluate(p, base_t81, f"T87+T75+T89 w_nn={w_nn} w_lgb={w_lgb} w_cb={w_cb}"))

    print("\n=== 3-way (T81 + T75 + T89) — original ===", flush=True)
    for w_nn, w_lgb, w_cb in [(1,1,1), (1,1.5,1), (1,1,1.5), (1,1.5,0.5)]:
        wsum = w_nn + w_lgb + w_cb
        p = (w_nn*p_t81 + w_lgb*p_t75 + w_cb*p_t89) / wsum
        results.append(evaluate(p, base_t81, f"T81+T75+T89 w_nn={w_nn} w_lgb={w_lgb} w_cb={w_cb}"))

    print("\n=== 4-way (T87 + T81 + T75 + T89) ===", flush=True)
    for w_t87, w_t81, w_t75, w_t89 in [(1,0.5,1.5,0.5), (1,0.5,1,1), (1,0.5,1,0.5), (1,1,1.5,1)]:
        wsum = w_t87 + w_t81 + w_t75 + w_t89
        p = (w_t87*p_t87 + w_t81*p_t81 + w_t75*p_t75 + w_t89*p_t89) / wsum
        results.append(evaluate(p, base_t81, f"4way w_t87={w_t87} w_t81={w_t81} w_t75={w_t75} w_t89={w_t89}"))

    # Best
    best = max(results, key=lambda r: r["de_sum"])
    print(f"\n{'='*100}", flush=True)
    print(f"BEST: {best['label']}  DE={best['de_sum']:+.4f}", flush=True)
    print(f"  vs iter_014 (+{ITER014}): {best['de_sum']-ITER014:+.4f}", flush=True)
    print(f"  vs iter_015_T87T75 (+{ITER015_T87T75}): {best['de_sum']-ITER015_T87T75:+.4f}", flush=True)
    print(f"  per_sym = {[round(x,2) for x in best['per_sym']]}", flush=True)
    print(f"{'='*100}", flush=True)
    print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)

    out = {"results": results, "best": best, "iter014": ITER014, "iter015_t87t75": ITER015_T87T75}
    with open(os.path.join(HERE, "ev_gate_4way_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote -> {os.path.join(HERE, 'ev_gate_4way_results.json')}", flush=True)


if __name__ == "__main__":
    main()
