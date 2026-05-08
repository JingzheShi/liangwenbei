"""T97: Ensemble eval — multi-head NN (T97) alone, vs T75/T87/T89 blends.

Loads:
  - T97 5-seed multi-head NN preds (h=60 head)
  - T75 5-seed LGB regression preds
  - T87 5-seed SPO+ NN preds
  - T89 5-seed CatBoost regression preds
  - T81 5-seed single-head NN preds (for ablation)

Reports:
  - Per-model alone DE LOSO PnL
  - Cross-corr matrix
  - 2-way / 3-way / 4-way blends with grid search over weights, then DE thresh tune.
"""
from __future__ import annotations

import json
import os
import sys
import time
import itertools

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
ITER014_LOSO = 38.28
ITER015_LOSO = 40.13   # T87 + T75 ensemble
T8789_LOSO = 41.01     # T87 + T89 ensemble (current best from T86 audit)

SEEDS = (1, 7, 13, 42, 100)


def vectorized_pnl(pred_action, mp_t, mp_th):
    side = pred_action.astype(np.float64) - 1.0
    abs_side = np.abs(side)
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * abs_side * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee_pnl) / denom


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
            "mp_t": sub["midprice_t"].to_numpy(np.float64),
            "mp_th": sub["midprice_th"].to_numpy(np.float64),
            "n": len(sub),
        })
    return out


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
        })
    runs.sort(key=lambda r: r["obj_val"], reverse=True)
    return runs


def evaluate_pred(p, base, label):
    df = base.copy()
    df["pred_dmid_norm"] = p.astype(np.float32)
    folds = split_by_sym(df)
    obj = make_obj_loso_asym(folds)
    runs = de_search(obj, [(0.0, 0.0040), (0.0, 0.0040)])
    best = runs[0]
    thr_up, thr_dn = best["thr_up"], best["thr_dn"]
    de_per_sym = []
    for fk in folds:
        a = ev_gate_asymmetric(fk["pred"], thr_up, thr_dn)
        de_per_sym.append(float(vectorized_pnl(a, fk["mp_t"], fk["mp_th"]).sum()))
    de_loso_sum = float(sum(de_per_sym))
    print(f"  {label:50s}  thr_up={thr_up:.6f} thr_dn={thr_dn:.6f}  "
          f"sum={de_loso_sum:+.4f}  per_sym=[{','.join(f'{x:+.2f}' for x in de_per_sym)}]",
          flush=True)
    return {
        "label": label,
        "thr_up": thr_up, "thr_dn": thr_dn,
        "sum_per_sym": de_loso_sum,
        "per_sym": de_per_sym,
    }


def load_avg(dir_, prefix, seeds=SEEDS, suffix=""):
    base = pd.read_parquet(os.path.join(dir_, f"{prefix}_seed{seeds[0]}{suffix}.parquet"))
    n = len(base)
    p = np.zeros(n, dtype=np.float64)
    for s in seeds:
        df = pd.read_parquet(os.path.join(dir_, f"{prefix}_seed{s}{suffix}.parquet"))
        if len(df) != n:
            raise RuntimeError(f"row mismatch {prefix} seed={s}")
        if not (df["sym"].equals(base["sym"]) and df["t"].equals(base["t"])):
            raise RuntimeError(f"row order mismatch {prefix} seed={s}")
        p += df["pred_dmid_norm"].to_numpy(np.float64)
    p /= len(seeds)
    return p, base


def main():
    print("=== T97 ensemble eval: multi-head NN vs T81/T87/T75/T89 ===", flush=True)
    print(f"References: iter_014 +{ITER014_LOSO}, iter_015 +{ITER015_LOSO}, T87+T89 +{T8789_LOSO}", flush=True)

    print("\nLoading 5-seed averages...", flush=True)
    p_t97, base_t97 = load_avg(HERE, "pred_T97", suffix="_main")
    p_t81, base_t81 = load_avg(T81_DIR, "pred_T81")
    p_t87, _ = load_avg(T87_DIR, "pred_T87", suffix="_main")
    p_t75, _ = load_avg(T75_DIR, "pred_T75")
    p_t89, _ = load_avg(T89_DIR, "pred_T89")

    # Verify all aligned
    assert base_t97["sym"].equals(base_t81["sym"])
    assert base_t97["t"].equals(base_t81["t"])

    # Cross-corr table
    preds = {"T97": p_t97, "T81": p_t81, "T87": p_t87, "T75": p_t75, "T89": p_t89}
    print(f"\nCross-corr matrix:", flush=True)
    print(f"  {'':>5s}" + "".join(f"  {n:>7s}" for n in preds), flush=True)
    cross_corr = {}
    for n1, p1 in preds.items():
        row = {}
        line = f"  {n1:>5s}"
        for n2, p2 in preds.items():
            c = float(np.corrcoef(p1, p2)[0, 1])
            row[n2] = c
            line += f"  {c:+7.4f}"
        cross_corr[n1] = row
        print(line, flush=True)

    results = []

    # ----- Single-model alone -----
    print(f"\n[1] Single-model alone:", flush=True)
    for name, p in preds.items():
        r = evaluate_pred(p, base_t97, f"{name} alone (5-seed avg)")
        r["models"] = [name]
        r["weights"] = [1.0]
        results.append(r)

    # ----- 2-way blends with T97 -----
    print(f"\n[2] 2-way blends: T97 + (T75|T87|T89|T81)", flush=True)
    weight_grid = [(1.0, 0.5), (1.0, 1.0), (1.0, 1.5), (1.0, 2.0),
                   (0.5, 1.0), (1.5, 1.0), (2.0, 1.0)]
    for partner_name, partner_p in [("T75", p_t75), ("T87", p_t87), ("T89", p_t89), ("T81", p_t81)]:
        for w1, w2 in weight_grid:
            s = w1 + w2
            p_blend = (w1 * p_t97 + w2 * partner_p) / s
            r = evaluate_pred(p_blend, base_t97,
                              f"T97+{partner_name} w=({w1:.1f},{w2:.1f})")
            r["models"] = ["T97", partner_name]
            r["weights"] = [w1, w2]
            results.append(r)

    # ----- 3-way blends with T97 -----
    print(f"\n[3] 3-way blends with T97:", flush=True)
    for trio_name, p_a, p_b in [("T75+T89", p_t75, p_t89),
                                ("T75+T87", p_t75, p_t87),
                                ("T87+T89", p_t87, p_t89),
                                ("T81+T87", p_t81, p_t87)]:
        for w_t97, w_a, w_b in [
            (1.0, 1.0, 1.0),
            (1.0, 1.5, 1.0),
            (1.0, 1.0, 1.5),
            (1.5, 1.0, 1.0),
            (0.5, 1.0, 1.0),
            (1.0, 1.5, 1.5),
            (1.0, 0.5, 0.5),
        ]:
            s = w_t97 + w_a + w_b
            p_blend = (w_t97 * p_t97 + w_a * p_a + w_b * p_b) / s
            r = evaluate_pred(
                p_blend, base_t97,
                f"T97+{trio_name} w=({w_t97:.1f},{w_a:.1f},{w_b:.1f})")
            r["models"] = ["T97"] + trio_name.split("+")
            r["weights"] = [w_t97, w_a, w_b]
            results.append(r)

    # ----- 4-way blends -----
    print(f"\n[4] 4-way blends: T97+T87+T75+T89:", flush=True)
    for w_t97, w_t87, w_t75, w_t89 in [
        (1.0, 1.0, 1.0, 1.0),
        (1.0, 1.0, 1.5, 1.0),
        (1.0, 1.0, 1.0, 1.5),
        (1.0, 1.0, 1.5, 1.5),
        (0.5, 1.0, 1.0, 1.0),
        (1.5, 1.0, 1.0, 1.0),
        (1.0, 1.5, 1.0, 1.0),
    ]:
        s = w_t97 + w_t87 + w_t75 + w_t89
        p_blend = (w_t97 * p_t97 + w_t87 * p_t87 + w_t75 * p_t75 + w_t89 * p_t89) / s
        r = evaluate_pred(
            p_blend, base_t97,
            f"4-way T97+T87+T75+T89 w=({w_t97:.1f},{w_t87:.1f},{w_t75:.1f},{w_t89:.1f})")
        r["models"] = ["T97", "T87", "T75", "T89"]
        r["weights"] = [w_t97, w_t87, w_t75, w_t89]
        results.append(r)

    best = max(results, key=lambda r: r["sum_per_sym"])
    print(f"\n{'='*88}", flush=True)
    print(f"BEST: {best['label']}", flush=True)
    print(f"  DE LOSO sum_per_sym = {best['sum_per_sym']:+.4f}", flush=True)
    print(f"  vs iter_014 (+{ITER014_LOSO}): {best['sum_per_sym']-ITER014_LOSO:+.4f}", flush=True)
    print(f"  vs iter_015 (+{ITER015_LOSO}): {best['sum_per_sym']-ITER015_LOSO:+.4f}", flush=True)
    print(f"  vs T87+T89 (+{T8789_LOSO}): {best['sum_per_sym']-T8789_LOSO:+.4f}", flush=True)
    print('=' * 88, flush=True)

    # Top-10
    sorted_results = sorted(results, key=lambda r: -r["sum_per_sym"])[:10]
    print("\nTop-10 configs:", flush=True)
    for r in sorted_results:
        print(f"  {r['label']:50s}  {r['sum_per_sym']:+.4f}", flush=True)

    out_path = os.path.join(HERE, "ev_gate_ensemble_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "task": "T97 ensemble eval",
            "cross_corr": cross_corr,
            "iter014_ref": ITER014_LOSO,
            "iter015_ref": ITER015_LOSO,
            "t87_t89_ref": T8789_LOSO,
            "results": results,
            "best": best,
            "top10": sorted_results,
        }, f, indent=2)
    print(f"\nwrote -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
