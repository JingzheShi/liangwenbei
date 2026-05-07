"""T77: Finalize vote ensemble — finish K=4 DE and V2 DE with reduced budget.

Reuses already-computed V0/V1-uniform/K=3 DE results, completes K=4 DE and V2 DE.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

T53_DIR = os.path.join(ROOT, "experiments", "T53_r34_stage3")
sys.path.insert(0, T53_DIR)
from de_thresh import FEE, PROB_COLS, gate_asymmetric, vectorized_pnl  # noqa: E402

T70_DIR = os.path.join(ROOT, "experiments", "T70_v4_stage5")

SYMS = (0, 1, 2, 3, 4)
HORIZONS = (5, 10, 20, 40, 60)
ITER012_BASELINE = 26.4396


def load_h60_5seed_avg():
    seeds = (1, 7, 13, 42, 100)
    base = pd.read_parquet(os.path.join(T70_DIR, f"pred_T70_seed{seeds[0]}.parquet"))
    p_sum = base[PROB_COLS].to_numpy(np.float64).copy()
    n = len(base)
    for s in seeds[1:]:
        df_s = pd.read_parquet(os.path.join(T70_DIR, f"pred_T70_seed{s}.parquet"))
        p_sum += df_s[PROB_COLS].to_numpy(np.float64)
    return base, (p_sum / float(len(seeds))).astype(np.float32)


def load_horizon(H):
    p = os.path.join(HERE, f"pred_T77_h{H}_seed42.parquet")
    return pd.read_parquet(p)


def load_all():
    base, p60 = load_h60_5seed_avg()
    probs = {60: p60}
    for H in (5, 10, 20, 40):
        dfh = load_horizon(H)
        probs[H] = dfh[PROB_COLS].to_numpy(np.float32)
    sym = base["sym"].to_numpy(np.int8)
    label = base["true_label"].to_numpy(np.int64)
    mp_t = base["midprice_t"].to_numpy(np.float64)
    mp_th = base["midprice_th"].to_numpy(np.float64)
    return probs, sym, label, mp_t, mp_th


def vote_rule_v1(probs_dict, K, T_h):
    n = len(next(iter(probs_dict.values())))
    up = np.zeros(n, dtype=np.int8)
    dn = np.zeros(n, dtype=np.int8)
    for h, p in probs_dict.items():
        a = p.argmax(axis=1)
        max_p = p.max(axis=1)
        valid = max_p > T_h.get(h, 0.0)
        up += ((a == 2) & valid).astype(np.int8)
        dn += ((a == 0) & valid).astype(np.int8)
    pred = np.full(n, 1, dtype=np.int8)
    pred[up >= K] = 2
    pred[dn >= K] = 0
    pred[(up >= K) & (dn >= K)] = 1
    return pred


def vote_rule_v2(probs_dict, w_h, T):
    n = len(next(iter(probs_dict.values())))
    score = np.zeros(n, dtype=np.float64)
    for h, p in probs_dict.items():
        score += w_h.get(h, 0.0) * (p[:, 2].astype(np.float64) - p[:, 0].astype(np.float64))
    pred = np.full(n, 1, dtype=np.int8)
    pred[score > T] = 2
    pred[score < -T] = 0
    return pred


def evaluate_pred(pred, sym, label, mp_t, mp_th):
    per_sym = []
    for k in SYMS:
        m = sym == k
        per_sym.append(float(vectorized_pnl(pred[m], label[m], mp_t[m], mp_th[m]).sum()))
    total = float(vectorized_pnl(pred, label, mp_t, mp_th).sum())
    n_active = int((pred != 1).sum())
    return {"total_cum_pnl": total, "loso_equiv_sum": float(sum(per_sym)),
            "per_sym": per_sym, "n_active": n_active, "n_total": len(pred)}


def de_search_thresh(probs, sym, label, mp_t, mp_th, K, n_seeds=3, maxiter=40):
    sym_masks = [sym == k for k in SYMS]
    label_lst = [label[m] for m in sym_masks]
    mp_t_lst = [mp_t[m] for m in sym_masks]
    mp_th_lst = [mp_th[m] for m in sym_masks]

    def obj(x):
        T_h = {5: x[0], 10: x[1], 20: x[2], 40: x[3], 60: x[4]}
        pred = vote_rule_v1(probs, K, T_h)
        s = 0.0
        for i, m in enumerate(sym_masks):
            s += float(vectorized_pnl(pred[m], label_lst[i], mp_t_lst[i], mp_th_lst[i]).sum())
        return -s

    bounds = [(0.34, 0.70)] * 5
    runs = []
    for sd in range(n_seeds):
        t0 = time.time()
        res = differential_evolution(
            obj, bounds=bounds, seed=sd, maxiter=maxiter, popsize=18,
            polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            updating="deferred", workers=1, init="sobol",
        )
        runs.append({"seed": int(sd), "K": int(K),
                     "T_h": [float(v) for v in res.x],
                     "obj_val": -float(res.fun),
                     "elapsed_sec": float(time.time() - t0)})
        print(f"  K={K} seed={sd}: {-res.fun:+.4f} ({time.time()-t0:.1f}s) T_h={[round(v,3) for v in res.x]}", flush=True)
    runs.sort(key=lambda r: r["obj_val"], reverse=True)
    return runs


def de_search_v2(probs, sym, label, mp_t, mp_th, n_seeds=3, maxiter=40):
    sym_masks = [sym == k for k in SYMS]
    label_lst = [label[m] for m in sym_masks]
    mp_t_lst = [mp_t[m] for m in sym_masks]
    mp_th_lst = [mp_th[m] for m in sym_masks]

    def obj(x):
        w_h = {5: x[0], 10: x[1], 20: x[2], 40: x[3], 60: x[4]}
        T = x[5]
        pred = vote_rule_v2(probs, w_h, T)
        s = 0.0
        for i, m in enumerate(sym_masks):
            s += float(vectorized_pnl(pred[m], label_lst[i], mp_t_lst[i], mp_th_lst[i]).sum())
        return -s

    bounds = [(0.0, 2.0)] * 5 + [(0.0, 1.0)]
    runs = []
    for sd in range(n_seeds):
        t0 = time.time()
        res = differential_evolution(
            obj, bounds=bounds, seed=sd, maxiter=maxiter, popsize=18,
            polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            updating="deferred", workers=1, init="sobol",
        )
        runs.append({"seed": int(sd),
                     "x": [float(v) for v in res.x],
                     "obj_val": -float(res.fun),
                     "elapsed_sec": float(time.time() - t0)})
        print(f"  V2 seed={sd}: {-res.fun:+.4f} ({time.time()-t0:.1f}s) "
              f"w={[round(v,3) for v in res.x[:5]]} T={res.x[5]:.4f}", flush=True)
    runs.sort(key=lambda r: r["obj_val"], reverse=True)
    return runs


def main():
    print("=== T77 finalize: K=4 DE + V2 DE ===\n", flush=True)
    probs, sym, label, mp_t, mp_th = load_all()
    print(f"  loaded {len(label):,} rows\n", flush=True)

    # K=3 DE result already known (from full eval log)
    k3_de_T_h = {5: 0.47798251052007046, 10: 0.38537998706797705, 20: 0.5330342383720678,
                 40: 0.4242430270100717, 60: 0.43579704436645106}
    pred_k3 = vote_rule_v1(probs, 3, k3_de_T_h)
    res_k3 = evaluate_pred(pred_k3, sym, label, mp_t, mp_th)
    print(f"V1 K=3 DE (recompute): {res_k3['loso_equiv_sum']:+.4f}", flush=True)
    print(f"  per_sym={[round(v,3) for v in res_k3['per_sym']]} n_active={res_k3['n_active']:,}", flush=True)

    # V1 K=4 DE
    print("\n[V1 K=4 DE (3 seeds, maxiter=40)]", flush=True)
    runs_k4 = de_search_thresh(probs, sym, label, mp_t, mp_th, K=4, n_seeds=3, maxiter=40)
    best_k4 = runs_k4[0]
    T_h_k4 = dict(zip(HORIZONS, best_k4["T_h"]))
    pred_k4 = vote_rule_v1(probs, 4, T_h_k4)
    res_k4 = evaluate_pred(pred_k4, sym, label, mp_t, mp_th)
    print(f"V1 K=4 DE: {res_k4['loso_equiv_sum']:+.4f} T_h={T_h_k4}", flush=True)
    print(f"  per_sym={[round(v,3) for v in res_k4['per_sym']]} n_active={res_k4['n_active']:,}", flush=True)

    # V2 DE (weighted directional)
    print("\n[V2 weighted directional sum DE (3 seeds, maxiter=40)]", flush=True)
    runs_v2 = de_search_v2(probs, sym, label, mp_t, mp_th, n_seeds=3, maxiter=40)
    best_v2 = runs_v2[0]
    w_v2 = dict(zip(HORIZONS, best_v2["x"][:5]))
    T_v2 = best_v2["x"][5]
    pred_v2 = vote_rule_v2(probs, w_v2, T_v2)
    res_v2 = evaluate_pred(pred_v2, sym, label, mp_t, mp_th)
    print(f"V2 DE: {res_v2['loso_equiv_sum']:+.4f} w={w_v2} T={T_v2:.4f}", flush=True)
    print(f"  per_sym={[round(v,3) for v in res_v2['per_sym']]} n_active={res_v2['n_active']:,}", flush=True)

    # Combine results
    summary = {
        "iter012_baseline": ITER012_BASELINE,
        "v1_K3_DE": {
            "loso_equiv_sum": res_k3["loso_equiv_sum"],
            "T_h": k3_de_T_h,
            "per_sym": res_k3["per_sym"],
            "n_active": res_k3["n_active"],
            "vs_iter012": res_k3["loso_equiv_sum"] - ITER012_BASELINE,
        },
        "v1_K4_DE": {
            "loso_equiv_sum": res_k4["loso_equiv_sum"],
            "T_h": T_h_k4,
            "per_sym": res_k4["per_sym"],
            "n_active": res_k4["n_active"],
            "vs_iter012": res_k4["loso_equiv_sum"] - ITER012_BASELINE,
            "all_runs": runs_k4,
        },
        "v2_DE": {
            "loso_equiv_sum": res_v2["loso_equiv_sum"],
            "w_h": w_v2,
            "T": T_v2,
            "per_sym": res_v2["per_sym"],
            "n_active": res_v2["n_active"],
            "vs_iter012": res_v2["loso_equiv_sum"] - ITER012_BASELINE,
            "all_runs": runs_v2,
        },
    }
    out_path = os.path.join(HERE, "vote_de_finalize.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}", flush=True)

    print("\n=== KEY COMPARISON ===", flush=True)
    print(f"iter_012 baseline: {ITER012_BASELINE:+.4f}", flush=True)
    print(f"V1 K=3 DE: {res_k3['loso_equiv_sum']:+.4f} ({res_k3['loso_equiv_sum']-ITER012_BASELINE:+.4f})", flush=True)
    print(f"V1 K=4 DE: {res_k4['loso_equiv_sum']:+.4f} ({res_k4['loso_equiv_sum']-ITER012_BASELINE:+.4f})", flush=True)
    print(f"V2 DE:     {res_v2['loso_equiv_sum']:+.4f} ({res_v2['loso_equiv_sum']-ITER012_BASELINE:+.4f})", flush=True)


if __name__ == "__main__":
    main()
