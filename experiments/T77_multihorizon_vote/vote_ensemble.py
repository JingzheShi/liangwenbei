"""T77: Multi-horizon vote ensemble evaluation.

Inputs:
  - h_60: 5-seed-avg from T70 pred_T70_seed{1,7,13,42,100}.parquet
  - h_5/10/20/40: single seed=42 from T77 pred_T77_h{H}_seed42.parquet

Vote rules:
  V0 (raw vote >=K): for K in 3,4,5
  V1 (with per-horizon prob threshold): only count vote if max(p_h) > T_h
  V2 (weighted directional sum): sum_h w_h * (p_h_up - p_h_dn) > T => up, < -T => down
  V3 (vote + DE thresh on combined direction-prob aggregate)

Compare vs iter_012 baseline (h_60 5-seed + DE 4D thresh = +26.44 sum-per-sym).
"""
from __future__ import annotations

import json
import os
import sys
import time
from itertools import product

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


def load_horizon(H, seed=42):
    p = os.path.join(HERE, f"pred_T77_h{H}_seed{seed}.parquet")
    return pd.read_parquet(p)


def load_all():
    """Load probs for h_5/10/20/40/60. h_60 is 5-seed avg, others single seed=42.
    Returns aligned arrays + label/mp data from h_60 (target horizon for evaluation).
    """
    print("Loading h_60 5-seed avg ...", flush=True)
    df60 = load_h60_5seed_avg()
    print(f"  h_60 shape={df60.shape}", flush=True)

    probs = {}
    probs[60] = df60[PROB_COLS].to_numpy(np.float32)

    for H in (5, 10, 20, 40):
        print(f"Loading h_{H} seed=42 ...", flush=True)
        dfh = load_horizon(H, 42)
        if len(dfh) != len(df60):
            raise RuntimeError(f"row count mismatch h_{H}: {len(dfh)} vs {len(df60)}")
        if not (dfh["sym"].equals(df60["sym"]) and dfh["date"].equals(df60["date"]) and dfh["t"].equals(df60["t"])):
            raise RuntimeError(f"row order mismatch h_{H}")
        probs[H] = dfh[PROB_COLS].to_numpy(np.float32)

    sym = df60["sym"].to_numpy(np.int8)
    label = df60["true_label"].to_numpy(np.int64)
    mp_t = df60["midprice_t"].to_numpy(np.float64)
    mp_th = df60["midprice_th"].to_numpy(np.float64)
    return probs, sym, label, mp_t, mp_th


def split_by_sym(probs, sym, label, mp_t, mp_th):
    folds = []
    for k in SYMS:
        m = sym == k
        folds.append({
            "sym": int(k),
            "mask": m,
            "n": int(m.sum()),
            "probs": {h: p[m] for h, p in probs.items()},
            "label": label[m],
            "mp_t": mp_t[m],
            "mp_th": mp_th[m],
        })
    return folds


def per_sym_pnl_from_pred(folds, pred_full, sym):
    s_per = []
    n_active = []
    for fk in folds:
        m = fk["mask"]
        p = pred_full[m]
        s_per.append(float(vectorized_pnl(p, fk["label"], fk["mp_t"], fk["mp_th"]).sum()))
        n_active.append(int((p != 1).sum()))
    return s_per, n_active


# ----- Vote rules -----

def vote_rule_v0(probs_dict, K):
    """Raw vote >= K: count argmax==2 as up, argmax==0 as dn, both must >= K."""
    n = len(next(iter(probs_dict.values())))
    up = np.zeros(n, dtype=np.int8)
    dn = np.zeros(n, dtype=np.int8)
    for p in probs_dict.values():
        a = p.argmax(axis=1)
        up += (a == 2).astype(np.int8)
        dn += (a == 0).astype(np.int8)
    pred = np.full(n, 1, dtype=np.int8)
    pred[up >= K] = 2
    pred[dn >= K] = 0
    # If both up and dn >= K (rare, contradictory), prefer flat
    pred[(up >= K) & (dn >= K)] = 1
    return pred, up, dn


def vote_rule_v1(probs_dict, K, T_h):
    """Vote with per-horizon prob threshold: only count if max(p_h) > T_h."""
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
    return pred, up, dn


def vote_rule_v2(probs_dict, w_h, T):
    """Weighted directional sum: score = sum_h w_h * (p_up - p_dn). |score| > T => trade."""
    n = len(next(iter(probs_dict.values())))
    score = np.zeros(n, dtype=np.float64)
    for h, p in probs_dict.items():
        score += w_h.get(h, 0.0) * (p[:, 2].astype(np.float64) - p[:, 0].astype(np.float64))
    pred = np.full(n, 1, dtype=np.int8)
    pred[score > T] = 2
    pred[score < -T] = 0
    return pred, score


# ----- Evaluators -----

def evaluate_pred(pred, sym, label, mp_t, mp_th):
    folds = []
    per_sym_arr = []
    for k in SYMS:
        m = sym == k
        per_sym_arr.append(float(vectorized_pnl(pred[m], label[m], mp_t[m], mp_th[m]).sum()))
    total = float(vectorized_pnl(pred, label, mp_t, mp_th).sum())
    n_active = int((pred != 1).sum())
    return {
        "total_cum_pnl": total,
        "loso_equiv_sum": float(sum(per_sym_arr)),
        "per_sym": per_sym_arr,
        "n_active": n_active,
        "n_total": int(len(pred)),
    }


def evaluate_h60_only_argmax(probs, sym, label, mp_t, mp_th):
    pred = probs[60].argmax(axis=1).astype(np.int8)
    res = evaluate_pred(pred, sym, label, mp_t, mp_th)
    res["rule"] = "h60_argmax_5seed_avg"
    return res


def evaluate_h60_de_iter012_thresh(probs, sym, label, mp_t, mp_th):
    Tu, Td, du, dd = 0.4379910127008917, 0.4448964062818059, 8.099624641794145e-05, 6.68533652402048e-05
    pred = gate_asymmetric(probs[60], Tu, Td, du, dd)
    res = evaluate_pred(pred, sym, label, mp_t, mp_th)
    res["rule"] = "iter012_DE_4D_thresh"
    res["thresh"] = {"T_up": Tu, "T_dn": Td, "d_up": du, "d_dn": dd}
    return res


def evaluate_v0_K(probs, K, sym, label, mp_t, mp_th):
    pred, up, dn = vote_rule_v0(probs, K)
    res = evaluate_pred(pred, sym, label, mp_t, mp_th)
    res["rule"] = f"vote_v0_K{K}"
    res["K"] = K
    return res


def evaluate_v1_K_T(probs, K, T_h, sym, label, mp_t, mp_th):
    pred, up, dn = vote_rule_v1(probs, K, T_h)
    res = evaluate_pred(pred, sym, label, mp_t, mp_th)
    res["rule"] = f"vote_v1_K{K}"
    res["K"] = K
    res["T_h"] = T_h
    return res


def evaluate_v2(probs, w_h, T, sym, label, mp_t, mp_th):
    pred, score = vote_rule_v2(probs, w_h, T)
    res = evaluate_pred(pred, sym, label, mp_t, mp_th)
    res["rule"] = "vote_v2_weighted"
    res["w_h"] = w_h
    res["T"] = T
    return res


# ----- DE on weighted vote -----

def de_search_weighted(probs, sym, label, mp_t, mp_th):
    """Optimize (w5, w10, w20, w40, w60, T) to maximize sum-per-sym pnl."""
    fold_pnl = []
    sym_masks = [sym == k for k in SYMS]

    def obj(x):
        w5, w10, w20, w40, w60, T = x
        w_h = {5: w5, 10: w10, 20: w20, 40: w40, 60: w60}
        pred, _ = vote_rule_v2(probs, w_h, T)
        s = 0.0
        for m in sym_masks:
            s += float(vectorized_pnl(pred[m], label[m], mp_t[m], mp_th[m]).sum())
        return -s

    bounds = [(0.0, 2.0)] * 5 + [(0.0, 1.0)]
    runs = []
    for sd in (0, 1, 2, 7, 42):
        t0 = time.time()
        res = differential_evolution(
            obj, bounds=bounds, seed=sd, maxiter=80, popsize=24,
            polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            updating="deferred", workers=1, init="sobol",
        )
        runs.append({
            "seed": int(sd),
            "x": [float(v) for v in res.x],
            "obj_val": -float(res.fun),
            "elapsed_sec": float(time.time() - t0),
        })
    runs.sort(key=lambda r: r["obj_val"], reverse=True)
    return runs


def de_search_thresh_per_horizon(probs, sym, label, mp_t, mp_th, K=4):
    """Optimize T_h for each horizon for vote_v1, given K."""
    sym_masks = [sym == k for k in SYMS]

    def obj(x):
        T_h = {5: x[0], 10: x[1], 20: x[2], 40: x[3], 60: x[4]}
        pred, _, _ = vote_rule_v1(probs, K, T_h)
        s = 0.0
        for m in sym_masks:
            s += float(vectorized_pnl(pred[m], label[m], mp_t[m], mp_th[m]).sum())
        return -s

    bounds = [(0.34, 0.70)] * 5
    runs = []
    for sd in (0, 1, 2, 7, 42):
        t0 = time.time()
        res = differential_evolution(
            obj, bounds=bounds, seed=sd, maxiter=80, popsize=24,
            polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            updating="deferred", workers=1, init="sobol",
        )
        runs.append({
            "seed": int(sd), "K": int(K),
            "T_h": [float(v) for v in res.x],
            "obj_val": -float(res.fun),
            "elapsed_sec": float(time.time() - t0),
        })
    runs.sort(key=lambda r: r["obj_val"], reverse=True)
    return runs


def main():
    print("=== T77 multi-horizon vote ensemble eval ===\n", flush=True)
    probs, sym, label, mp_t, mp_th = load_all()
    print(f"\nN rows = {len(label):,}, horizons present: {sorted(probs.keys())}\n", flush=True)

    results = {}

    # Baselines
    print("[Baselines]", flush=True)
    res_argmax = evaluate_h60_only_argmax(probs, sym, label, mp_t, mp_th)
    print(f"  h60 5seed argmax: total={res_argmax['total_cum_pnl']:.4f} loso_equiv={res_argmax['loso_equiv_sum']:.4f}", flush=True)
    results["h60_argmax_5seed"] = res_argmax

    res_iter012 = evaluate_h60_de_iter012_thresh(probs, sym, label, mp_t, mp_th)
    print(f"  iter_012 DE thresh: total={res_iter012['total_cum_pnl']:.4f} loso_equiv={res_iter012['loso_equiv_sum']:.4f}", flush=True)
    results["iter012_de_thresh"] = res_iter012

    # Per-horizon h argmax baseline
    print("\n[Per-horizon argmax single seed=42 baseline]", flush=True)
    per_h = {}
    for h in HORIZONS:
        pred = probs[h].argmax(axis=1).astype(np.int8)
        r = evaluate_pred(pred, sym, label, mp_t, mp_th)
        r["rule"] = f"h{h}_argmax"
        print(f"  h{h:>2}: loso_equiv={r['loso_equiv_sum']:+.4f} (n_active={r['n_active']:,})", flush=True)
        per_h[f"h{h}"] = r
    results["per_horizon_argmax"] = per_h

    # V0: raw vote K=3,4,5
    print("\n[V0: raw vote K]", flush=True)
    for K in (3, 4, 5):
        r = evaluate_v0_K(probs, K, sym, label, mp_t, mp_th)
        print(f"  K={K}: loso_equiv={r['loso_equiv_sum']:+.4f} (n_active={r['n_active']:,})", flush=True)
        results[f"v0_K{K}"] = r

    # V1: thresh + vote, sweep T uniform across horizons
    print("\n[V1: vote + uniform per-horizon thresh]", flush=True)
    v1_sweeps = []
    for K in (3, 4):
        for T in (0.40, 0.42, 0.44, 0.46, 0.48, 0.50):
            T_h = {h: T for h in HORIZONS}
            r = evaluate_v1_K_T(probs, K, T_h, sym, label, mp_t, mp_th)
            v1_sweeps.append((K, T, r["loso_equiv_sum"], r["n_active"]))
            print(f"  K={K} T={T:.2f}: loso_equiv={r['loso_equiv_sum']:+.4f} n_active={r['n_active']:,}", flush=True)
    results["v1_uniform_sweeps"] = [
        {"K": K, "T": T, "loso_equiv_sum": float(s), "n_active": int(na)}
        for K, T, s, na in v1_sweeps
    ]

    # V1 with DE optimized per-horizon T (K=3 and K=4)
    print("\n[V1 DE per-horizon T_h optimization]", flush=True)
    for K in (3, 4):
        runs = de_search_thresh_per_horizon(probs, sym, label, mp_t, mp_th, K=K)
        best = runs[0]
        T_h = dict(zip(HORIZONS, best["T_h"]))
        r = evaluate_v1_K_T(probs, K, T_h, sym, label, mp_t, mp_th)
        print(f"  K={K} DE T_h: loso_equiv={r['loso_equiv_sum']:+.4f} T_h={T_h}", flush=True)
        results[f"v1_DE_K{K}"] = {
            "best": best,
            "all_runs": runs,
            "eval": r,
        }

    # V2: weighted directional sum, equal weights then DE
    print("\n[V2: weighted directional sum]", flush=True)
    w_eq = {h: 1.0 for h in HORIZONS}
    for T in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
        r = evaluate_v2(probs, w_eq, T, sym, label, mp_t, mp_th)
        print(f"  eq w T={T:.2f}: loso_equiv={r['loso_equiv_sum']:+.4f} n_active={r['n_active']:,}", flush=True)
    print("  ... DE optimizing (w5,w10,w20,w40,w60,T) ...", flush=True)
    de_v2 = de_search_weighted(probs, sym, label, mp_t, mp_th)
    best_v2 = de_v2[0]
    w_best = dict(zip(HORIZONS, best_v2["x"][:5]))
    T_best = best_v2["x"][5]
    r_v2 = evaluate_v2(probs, w_best, T_best, sym, label, mp_t, mp_th)
    print(f"  DE: loso_equiv={r_v2['loso_equiv_sum']:+.4f} w={w_best} T={T_best:.4f}", flush=True)
    results["v2_DE"] = {
        "best": best_v2,
        "all_runs": de_v2,
        "eval": r_v2,
    }

    # Summary
    print("\n=== SUMMARY ===", flush=True)
    summary = {
        "iter012_baseline_loso_equiv": ITER012_BASELINE,
        "h60_5seed_argmax_loso": res_argmax["loso_equiv_sum"],
        "iter012_de_loso": res_iter012["loso_equiv_sum"],
        "v0_K3": results["v0_K3"]["loso_equiv_sum"],
        "v0_K4": results["v0_K4"]["loso_equiv_sum"],
        "v0_K5": results["v0_K5"]["loso_equiv_sum"],
        "v1_DE_K3": results["v1_DE_K3"]["eval"]["loso_equiv_sum"],
        "v1_DE_K4": results["v1_DE_K4"]["eval"]["loso_equiv_sum"],
        "v2_DE": results["v2_DE"]["eval"]["loso_equiv_sum"],
    }
    for k, v in summary.items():
        if isinstance(v, float):
            delta = v - ITER012_BASELINE
            print(f"  {k:30s}: {v:+.4f}  (vs iter012: {delta:+.4f})", flush=True)
    results["_summary"] = summary

    # Save
    out_path = os.path.join(HERE, "vote_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
