"""T137: DE 4D asym threshold evaluation on 5-seed ensemble predictions.

Compares MAE / Quantile q=0.5 / Huber alpha=0.9 to T75 L2 = +36.23 LOSO-equiv.
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
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001
T75_L2_LOSO = 36.23  # DE LOSO sum_per_sym
SEEDS = [1, 7, 13, 42, 100]
VARIANTS = ["mae", "quantile", "huber"]


def load_avg_pred(variant, seeds):
    dfs = []
    for s in seeds:
        p = os.path.join(HERE, f"pred_T137_{variant}_seed{s}.parquet")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}")
        dfs.append(pd.read_parquet(p))
    base = dfs[0].copy()
    p_sum = base["pred_dmid_norm"].to_numpy(np.float64).copy()
    for df_s in dfs[1:]:
        if not (df_s["sym"].equals(base["sym"]) and df_s["date"].equals(base["date"])
                and df_s["t"].equals(base["t"])):
            raise RuntimeError("row order mismatch across seeds")
        p_sum += df_s["pred_dmid_norm"].to_numpy(np.float64)
    base["pred_dmid_norm"] = (p_sum / len(seeds)).astype(np.float32)
    return base


def vectorized_pnl(pred_action, mp_t, mp_th):
    side = pred_action.astype(np.float64) - 1.0
    abs_side = np.abs(side)
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * abs_side * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    return (side * diff - fee_pnl) / (mp_t.astype(np.float64) + 1.0)


def ev_gate_asym(pred, thr_up, thr_dn):
    a = np.full(len(pred), 1, dtype=np.int8)
    a[pred > thr_up] = 2
    a[pred < -thr_dn] = 0
    return a


def de_search_loso(folds, bounds, n_runs=5):
    pred_list = [f["pred"] for f in folds]
    mp_t_list = [f["mp_t"] for f in folds]
    mp_th_list = [f["mp_th"] for f in folds]

    def neg_loso(x):
        thr_up, thr_dn = x
        s = 0.0
        for i in range(len(folds)):
            a = ev_gate_asym(pred_list[i], thr_up, thr_dn)
            s += vectorized_pnl(a, mp_t_list[i], mp_th_list[i]).sum()
        return -float(s)

    runs = []
    for sd in range(n_runs):
        t0 = time.time()
        res = differential_evolution(
            neg_loso, bounds=bounds, seed=sd, maxiter=80, popsize=24,
            polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            updating="deferred", workers=1, init="sobol",
        )
        runs.append({"seed": sd, "thr_up": float(res.x[0]), "thr_dn": float(res.x[1]),
                     "loso_sum": float(-res.fun), "nfev": int(res.nfev),
                     "elapsed": float(time.time() - t0)})
    runs.sort(key=lambda r: r["loso_sum"], reverse=True)
    return runs


def eval_variant(variant, seeds=None):
    if seeds is None:
        seeds = SEEDS
    print(f"\n{'='*60}\n=== variant={variant} seeds={seeds} ===", flush=True)

    available = [s for s in seeds if os.path.exists(
        os.path.join(HERE, f"pred_T137_{variant}_seed{s}.parquet"))]
    if not available:
        print(f"  No predictions found for variant={variant}", flush=True)
        return None

    df = load_avg_pred(variant, available)
    n = len(df)
    pred_all = df["pred_dmid_norm"].to_numpy(np.float64)
    mp_t_all = df["midprice_t"].to_numpy(np.float64)
    mp_th_all = df["midprice_th"].to_numpy(np.float64)

    folds = []
    for sym in SYMS:
        mask = df["sym"].to_numpy() == sym
        if mask.sum() == 0:
            continue
        folds.append({
            "sym": int(sym),
            "pred": pred_all[mask],
            "mp_t": mp_t_all[mask],
            "mp_th": mp_th_all[mask],
        })

    # Symmetric sweep for reference
    k_best, best_loso_sym = 1.0, -np.inf
    for k in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        thr = k * 2.0 * FEE
        s = 0.0
        for f in folds:
            a = np.full(len(f["pred"]), 1, dtype=np.int8)
            a[f["pred"] > thr] = 2
            a[f["pred"] < -thr] = 0
            s += vectorized_pnl(a, f["mp_t"], f["mp_th"]).sum()
        if s > best_loso_sym:
            best_loso_sym = s
            k_best = k
        print(f"  k={k:.2f}  loso={s:+.4f}", flush=True)

    # DE asymmetric
    bounds = [(0.0, 0.004), (0.0, 0.004)]
    print(f"  Running DE asymmetric LOSO search ...", flush=True)
    de_runs = de_search_loso(folds, bounds, n_runs=5)
    best_de = de_runs[0]
    thr_up_best, thr_dn_best = best_de["thr_up"], best_de["thr_dn"]
    de_loso = best_de["loso_sum"]

    per_sym = []
    for f in folds:
        a = ev_gate_asym(f["pred"], thr_up_best, thr_dn_best)
        pnl = float(vectorized_pnl(a, f["mp_t"], f["mp_th"]).sum())
        n_act = int((a != 1).sum())
        per_sym.append({"sym": f["sym"], "pnl": pnl, "n_active": n_act})
        print(f"  sym={f['sym']}  pnl={pnl:+.4f}  n_active={n_act:,}", flush=True)

    # Full-set total
    a_all = ev_gate_asym(pred_all, thr_up_best, thr_dn_best)
    total_pnl = float(vectorized_pnl(a_all, mp_t_all, mp_th_all).sum())
    n_active_total = int((a_all != 1).sum())

    # Transmission (corr with T75 L2 output)
    t75_dir = os.path.join(HERE, "..", "T75_regression_dmid")
    transmission = None
    t75_available = [s for s in available if os.path.exists(
        os.path.join(t75_dir, f"pred_T75_seed{s}.parquet"))]
    if t75_available:
        t75_preds = []
        for s in t75_available:
            df75 = pd.read_parquet(os.path.join(t75_dir, f"pred_T75_seed{s}.parquet"))
            t75_preds.append(df75["pred_dmid_norm"].to_numpy(np.float64))
        t75_avg = np.mean(t75_preds, axis=0)
        transmission = float(np.corrcoef(t75_avg, pred_all)[0, 1])
        print(f"  transmission (corr with T75 L2): {transmission:.4f}", flush=True)

    vs_t75 = de_loso - T75_L2_LOSO
    print(f"\n  RESULT variant={variant}:", flush=True)
    print(f"    DE LOSO (asym): {de_loso:+.4f}  vs T75_L2 {vs_t75:+.4f}", flush=True)
    print(f"    thr_up={thr_up_best:.6f}  thr_dn={thr_dn_best:.6f}", flush=True)
    print(f"    n_seeds_used={len(available)}", flush=True)

    return {
        "variant": variant,
        "seeds_used": available,
        "n_test": n,
        "symmetric_sweep_best": {"k": k_best, "loso_sum": float(best_loso_sym)},
        "de_asym_loso": {
            "best_run": best_de,
            "all_runs": de_runs,
            "loso_sum": de_loso,
            "thr_up": thr_up_best,
            "thr_dn": thr_dn_best,
        },
        "per_sym": per_sym,
        "total_pnl": total_pnl,
        "n_active_total": n_active_total,
        "vs_t75_l2": vs_t75,
        "transmission_vs_t75_l2": transmission,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=None, choices=VARIANTS + [None],
                    help="Evaluate only this variant (default: all found)")
    ap.add_argument("--seeds", default=None, help="comma-sep seeds")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else SEEDS
    variants = [args.variant] if args.variant else VARIANTS

    all_results = {}
    for v in variants:
        r = eval_variant(v, seeds=seeds)
        if r:
            all_results[v] = r

    # Summary table
    print(f"\n{'='*70}", flush=True)
    print(f"=== T137 SUMMARY  (T75_L2 ref = {T75_L2_LOSO:+.2f}) ===", flush=True)
    print(f"{'variant':>12s}  {'DE_LOSO':>10s}  {'vs_T75_L2':>10s}  {'transm':>8s}  verdict", flush=True)
    verdicts = {}
    for v in VARIANTS:
        if v not in all_results:
            print(f"  {v:>12s}  ---", flush=True)
            continue
        r = all_results[v]
        loso = r["de_asym_loso"]["loso_sum"]
        delta = r["vs_t75_l2"]
        tr = r.get("transmission_vs_t75_l2")
        verdict = "BEAT_L2" if delta > 0.5 else ("CLOSE" if delta > -0.5 else "WORSE")
        verdicts[v] = verdict
        tr_str = f"{tr:.3f}" if tr is not None else "N/A"
        print(f"  {v:>12s}  {loso:>+10.4f}  {delta:>+10.4f}  {tr_str:>8s}  {verdict}", flush=True)

    # Write results.json
    out = {
        "task": "T137 alt robust losses sweep on T75 L2 base",
        "t75_l2_loso_ref": T75_L2_LOSO,
        "variants": all_results,
        "summary": {v: {"loso": all_results[v]["de_asym_loso"]["loso_sum"],
                         "vs_t75": all_results[v]["vs_t75_l2"],
                         "verdict": verdicts.get(v)}
                    for v in all_results},
    }
    out_path = os.path.join(HERE, "results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote -> {out_path}", flush=True)

    best_variant = max(all_results, key=lambda v: all_results[v]["de_asym_loso"]["loso_sum"],
                       default=None)
    if best_variant:
        br = all_results[best_variant]
        print(f"\nBest variant: {best_variant}  LOSO={br['de_asym_loso']['loso_sum']:+.4f}  "
              f"vs_T75_L2={br['vs_t75_l2']:+.4f}", flush=True)

    return all_results


if __name__ == "__main__":
    main()
