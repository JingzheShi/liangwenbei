"""T36: asymmetric (T_up,T_dn,d_up,d_dn) threshold optimization on top config OOFs.

Loads oof_<config_name>_held{0..4}.parquet from stage 2, runs:
  1. Coarse 4D grid sweep
  2. Differential evolution refinement around the coarse best

Saves best (sum_cum_pnl, T_up, T_dn, d_up, d_dn) per config + chosen config.
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
sys.path.insert(0, ROOT)

from src.eval.pnl import _per_horizon_metrics  # noqa: E402

N_FOLDS = 5
FEE = 0.0001


def gate_asymmetric(probs, T_up, T_dn, d_up, d_dn):
    p0 = probs[:, 0]; p1 = probs[:, 1]; p2 = probs[:, 2]
    take_up = (p2 >= T_up) & (p2 > p1 + d_up) & (p2 > p0)
    take_dn = (p0 >= T_dn) & (p0 > p1 + d_dn) & (p0 > p2)
    pred = np.full(probs.shape[0], 1, dtype=np.int8)
    pred[take_up] = 2
    pred[take_dn] = 0
    return pred


def per_fold_pnl(fold_df, pred):
    m = _per_horizon_metrics(
        pred, fold_df["true_label"].to_numpy(np.int64),
        fold_df["midprice_t"].to_numpy(np.float32),
        fold_df["midprice_th"].to_numpy(np.float32),
        fee_rate=FEE,
    )
    return float(m["cum_pnl"]), int(m["n_predictions_active"])


def sum_pnl_across_folds(folds, T_up, T_dn, d_up, d_dn):
    s, n_act, per = 0.0, 0, []
    for k, df in folds.items():
        probs = df[["prob_0", "prob_1", "prob_2"]].to_numpy(np.float32)
        pred = gate_asymmetric(probs, T_up, T_dn, d_up, d_dn)
        pnl, na = per_fold_pnl(df, pred)
        s += pnl; n_act += na
        per.append({"held": k, "cum_pnl": pnl, "n_active": na})
    return s, n_act, per


def coarse_grid(folds):
    Tu_grid = np.round(np.arange(0.40, 0.61, 0.025), 4)
    Td_grid = np.round(np.arange(0.35, 0.56, 0.025), 4)
    du_grid = np.round(np.arange(0.0, 0.31, 0.025), 4)
    dd_grid = np.round(np.arange(0.0, 0.21, 0.025), 4)
    print(f"  coarse grid: {len(Tu_grid)}x{len(Td_grid)}x{len(du_grid)}x{len(dd_grid)} = "
          f"{len(Tu_grid)*len(Td_grid)*len(du_grid)*len(dd_grid)}", flush=True)
    best = None
    for Tu in Tu_grid:
        for Td in Td_grid:
            for du in du_grid:
                for dd in dd_grid:
                    s, _, _ = sum_pnl_across_folds(folds, float(Tu), float(Td),
                                                   float(du), float(dd))
                    if best is None or s > best["sum"]:
                        best = {"sum": s, "T_up": float(Tu), "T_dn": float(Td),
                                "d_up": float(du), "d_dn": float(dd)}
    return best


def de_refine(folds, init):
    """DE refinement around coarse best."""
    bounds = [
        (max(0.34, init["T_up"] - 0.07), min(0.70, init["T_up"] + 0.07)),
        (max(0.34, init["T_dn"] - 0.07), min(0.70, init["T_dn"] + 0.07)),
        (max(0.0, init["d_up"] - 0.10), min(0.40, init["d_up"] + 0.10)),
        (max(0.0, init["d_dn"] - 0.10), min(0.40, init["d_dn"] + 0.10)),
    ]
    print(f"  DE bounds: {bounds}", flush=True)

    def neg_obj(x):
        s, _, _ = sum_pnl_across_folds(folds, x[0], x[1], x[2], x[3])
        return -s

    res = differential_evolution(
        neg_obj, bounds, seed=42, maxiter=80, popsize=15, tol=1e-5,
        polish=True, init="sobol",
    )
    return {
        "T_up": float(res.x[0]), "T_dn": float(res.x[1]),
        "d_up": float(res.x[2]), "d_dn": float(res.x[3]),
        "sum": float(-res.fun),
        "nit": int(res.nit), "nfev": int(res.nfev),
    }


def load_folds_for_config(name, n_folds=N_FOLDS):
    safe = name.replace("+", "p").replace("/", "_")
    out = {}
    for k in range(n_folds):
        p = os.path.join(HERE, f"oof_{safe}_held{k}.parquet")
        if not os.path.exists(p):
            return None
        out[k] = pd.read_parquet(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-results", default=os.path.join(HERE, "top5_results.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "asym_thresh_results.json"))
    args = ap.parse_args()

    with open(args.top_results) as f:
        top = json.load(f)

    out_rows = []
    for cfg in top:
        name = cfg["name"]
        folds = load_folds_for_config(name)
        if folds is None:
            print(f"!! missing OOFs for {name}, skipping", flush=True)
            continue
        print(f"=== asym sweep for {name} ===", flush=True)
        t0 = time.time()
        best_coarse = coarse_grid(folds)
        print(f"  coarse best sum={best_coarse['sum']:+.4f} (T_up={best_coarse['T_up']:.3f},"
              f" T_dn={best_coarse['T_dn']:.3f}, d_up={best_coarse['d_up']:.3f},"
              f" d_dn={best_coarse['d_dn']:.3f})  in {time.time()-t0:.1f}s",
              flush=True)
        t1 = time.time()
        de_best = de_refine(folds, best_coarse)
        # Compute final per_fold breakdown
        s, n_act, per_fold = sum_pnl_across_folds(
            folds, de_best["T_up"], de_best["T_dn"], de_best["d_up"], de_best["d_dn"]
        )
        print(f"  DE best sum={s:+.4f} ({de_best['T_up']:.4f},{de_best['T_dn']:.4f},"
              f"{de_best['d_up']:.4f},{de_best['d_dn']:.4f}) in {time.time()-t1:.1f}s",
              flush=True)
        out_rows.append({
            "name": name,
            "trial_number": cfg.get("trial_number"),
            "loso_sum_cum_pnl_sym": cfg.get("loso_sum_cum_pnl"),
            "sym_T": cfg.get("best_T"), "sym_d": cfg.get("best_delta"),
            "asym_de": {
                "T_up": de_best["T_up"], "T_dn": de_best["T_dn"],
                "d_up": de_best["d_up"], "d_dn": de_best["d_dn"],
                "sum_cum_pnl": s,
                "per_fold": per_fold, "n_active": n_act,
                "nit": de_best["nit"], "nfev": de_best["nfev"],
            },
        })
        with open(args.out, "w") as f:
            json.dump(out_rows, f, indent=2)

    print("\n=== summary (asym DE LOSO sum_cum_pnl) ===", flush=True)
    for r in sorted(out_rows, key=lambda x: -x["asym_de"]["sum_cum_pnl"]):
        sym = r.get("loso_sum_cum_pnl_sym", float("nan"))
        asym = r["asym_de"]["sum_cum_pnl"]
        print(f"  {r['name']}: sym={sym:+.4f}  asym={asym:+.4f}  uplift={asym-sym:+.4f}",
              flush=True)


if __name__ == "__main__":
    main()
