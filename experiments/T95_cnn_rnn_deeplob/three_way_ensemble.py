"""3-way ensemble eval: T95 (GRU) + T81 (NN) + T75 (LGB).

Sweep ensemble weights, run DE-thresh per weight combo, find best.
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
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001
ITER014_LOSO = 38.28


def load_avg(paths):
    base = pd.read_parquet(paths[0])
    p = base["pred_dmid_norm"].to_numpy(np.float64)
    p = np.where(np.isnan(p), 0.0, p)
    cnt = np.where(~np.isnan(base["pred_dmid_norm"].to_numpy(np.float64)), 1.0, 0.0)
    n = len(base)
    for path in paths[1:]:
        df = pd.read_parquet(path)
        if len(df) != n:
            raise RuntimeError(f"row count mismatch {path}")
        ps = df["pred_dmid_norm"].to_numpy(np.float64)
        m = ~np.isnan(ps)
        p += np.where(m, ps, 0.0)
        cnt += np.where(m, 1.0, 0.0)
    p_avg = np.where(cnt > 0, p / np.maximum(cnt, 1.0), 0.0)
    return base, p_avg


def vectorized_pnl(action, mp_t, mp_th):
    side = action.astype(np.float64) - 1.0
    abs_side = np.abs(side)
    diff = mp_th - mp_t
    fee_pnl = FEE * abs_side * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t + 1.0
    return (side * diff - fee_pnl) / denom


def ev_gate_asym(pred, thr_up, thr_dn):
    a = np.full(len(pred), 1, dtype=np.int8)
    a[pred > thr_up] = 2
    a[pred < -thr_dn] = 0
    return a


def split_by_sym(df, pred):
    out = []
    for k in SYMS:
        m = df["sym"] == k
        out.append({
            "pred": pred[m].astype(np.float64),
            "mp_t": df.loc[m, "midprice_t"].to_numpy(np.float64),
            "mp_th": df.loc[m, "midprice_th"].to_numpy(np.float64),
        })
    return out


def de_loso_obj(folds):
    def f(x):
        s = 0.0
        for fold in folds:
            a = ev_gate_asym(fold["pred"], x[0], x[1])
            s += vectorized_pnl(a, fold["mp_t"], fold["mp_th"]).sum()
        return -float(s)
    return f


def de_search(obj, bounds, seeds=(0, 1, 7)):
    runs = []
    for sd in seeds:
        r = differential_evolution(
            obj, bounds=bounds, seed=sd, maxiter=80, popsize=24,
            polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            updating="deferred", workers=1, init="sobol",
        )
        runs.append({"seed": sd, "thr_up": float(r.x[0]), "thr_dn": float(r.x[1]),
                     "obj_val": float(-r.fun)})
    runs.sort(key=lambda x: x["obj_val"], reverse=True)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gru-pred", default="experiments/T95_cnn_rnn_deeplob/pred_T95_gru_w100_C_seed42.parquet")
    ap.add_argument("--nn-seeds", default="1,7,13,42,100")
    ap.add_argument("--lgb-seeds", default="1,7,13,42,100")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    nn_paths = [os.path.join(ROOT, "experiments", "T81_nn_regression_pnl", f"pred_T81_seed{s}.parquet")
                for s in args.nn_seeds.split(",")]
    lgb_paths = [os.path.join(ROOT, "experiments", "T75_regression_dmid", f"pred_T75_seed{s}.parquet")
                 for s in args.lgb_seeds.split(",")]
    gru_paths = [args.gru_pred] if not args.gru_pred.startswith("[") else json.loads(args.gru_pred)

    print(f"Loading: gru={len(gru_paths)} nn={len(nn_paths)} lgb={len(lgb_paths)}", flush=True)
    base_g, p_gru = load_avg(gru_paths)
    base_n, p_nn = load_avg(nn_paths)
    base_l, p_lgb = load_avg(lgb_paths)
    assert len(p_gru) == len(p_nn) == len(p_lgb)

    # Order alignment: sort by (sym, date, session, t) just in case
    keys = list(zip(base_g["sym"], base_g["date"], base_g["session"], base_g["t"]))
    keys_n = list(zip(base_n["sym"], base_n["date"], base_n["session"], base_n["t"]))
    keys_l = list(zip(base_l["sym"], base_l["date"], base_l["session"], base_l["t"]))
    if keys != keys_n or keys != keys_l:
        # need to sort/align; use base_g as canonical
        print("  WARN: row order differs across pred files; aligning...", flush=True)
        df_g = base_g.copy(); df_g["pred_dmid_norm"] = p_gru
        df_n = base_n.copy(); df_n["pred_dmid_norm"] = p_nn
        df_l = base_l.copy(); df_l["pred_dmid_norm"] = p_lgb
        keys_cols = ["sym", "date", "session", "t"]
        df_g = df_g.sort_values(keys_cols).reset_index(drop=True)
        df_n = df_n.sort_values(keys_cols).reset_index(drop=True)
        df_l = df_l.sort_values(keys_cols).reset_index(drop=True)
        base_g = df_g
        p_gru = df_g["pred_dmid_norm"].to_numpy(np.float64)
        p_nn = df_n["pred_dmid_norm"].to_numpy(np.float64)
        p_lgb = df_l["pred_dmid_norm"].to_numpy(np.float64)

    print(f"  std: gru={p_gru.std():.6e} nn={p_nn.std():.6e} lgb={p_lgb.std():.6e}", flush=True)
    print(f"  cross-corr: gru-nn={np.corrcoef(p_gru, p_nn)[0,1]:.4f} "
          f"gru-lgb={np.corrcoef(p_gru, p_lgb)[0,1]:.4f} "
          f"nn-lgb={np.corrcoef(p_nn, p_lgb)[0,1]:.4f}", flush=True)

    # Ensemble = w_g * p_gru + w_n * p_nn + w_l * p_lgb (normalize so that scales are roughly comparable)
    # Use per-pred z-score (divide by std) before combining → weights compare apples-to-apples
    p_gru_z = p_gru / max(p_gru.std(), 1e-12)
    p_nn_z = p_nn / max(p_nn.std(), 1e-12)
    p_lgb_z = p_lgb / max(p_lgb.std(), 1e-12)

    # We want ensemble pred in original Δmid_norm scale. Scale each component back to Δmid_norm by
    # using the LGB std as canonical scale (or the avg). Actually, since DE optimizes thr in absolute units,
    # let's use a target std = mean of (nn, lgb) std's.
    canonical_std = float((p_nn.std() + p_lgb.std()) / 2.0)
    print(f"  canonical_std (nn,lgb avg)={canonical_std:.6e}", flush=True)

    weights_grid = []
    # Fix nn:lgb = 1:1.5 (iter_014), sweep gru weight 0..2.0
    for wg in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]:
        weights_grid.append({"wg": wg, "wn": 1.0, "wl": 1.5})
    # Also try equal, gru-heavy, etc
    for wg, wn, wl in [(0.5, 1.0, 1.0), (1.0, 1.0, 1.0), (1.0, 0.5, 1.0), (1.0, 1.0, 0.5),
                        (0.5, 1.5, 1.5), (0.75, 1.5, 1.5)]:
        weights_grid.append({"wg": wg, "wn": wn, "wl": wl})

    results = []
    bounds = [(0.0, 0.0040), (0.0, 0.0040)]
    for w in weights_grid:
        ens_z = w["wg"] * p_gru_z + w["wn"] * p_nn_z + w["wl"] * p_lgb_z
        # scale to canonical
        ens = ens_z / max(ens_z.std(), 1e-12) * canonical_std
        df_e = base_g.copy()
        df_e["pred_dmid_norm"] = ens
        folds = split_by_sym(df_e, ens)
        # symmetric k=1 quick
        a_k1 = ev_gate_asym(ens, 2*FEE, 2*FEE)
        per_k1 = []
        for f in folds:
            a = ev_gate_asym(f["pred"], 2*FEE, 2*FEE)
            per_k1.append(float(vectorized_pnl(a, f["mp_t"], f["mp_th"]).sum()))
        sum_k1 = sum(per_k1)
        # DE LOSO
        de = de_search(de_loso_obj(folds), bounds, seeds=(0, 1))
        best = de[0]
        # Compute per-sym at best
        a_best = ev_gate_asym(ens, best["thr_up"], best["thr_dn"])
        per = []
        for f in folds:
            a = ev_gate_asym(f["pred"], best["thr_up"], best["thr_dn"])
            per.append(float(vectorized_pnl(a, f["mp_t"], f["mp_th"]).sum()))
        loso_sum = sum(per)
        results.append({
            **w, "k1_sum": sum_k1,
            "de_thr_up": best["thr_up"], "de_thr_dn": best["thr_dn"],
            "de_loso_sum": loso_sum, "de_per_sym": per,
        })
        print(f"  w_g={w['wg']:.2f} w_n={w['wn']:.2f} w_l={w['wl']:.2f}  "
              f"k1={sum_k1:+.4f}  DE_LOSO={loso_sum:+.4f}  "
              f"thr=({best['thr_up']:.5f}, {best['thr_dn']:.5f})", flush=True)

    best = max(results, key=lambda r: r["de_loso_sum"])
    print(f"\n{'='*60}", flush=True)
    print(f"BEST: w_g={best['wg']} w_n={best['wn']} w_l={best['wl']}  "
          f"DE_LOSO={best['de_loso_sum']:+.4f}", flush=True)
    print(f"vs iter_014 (+{ITER014_LOSO:.2f}): {best['de_loso_sum'] - ITER014_LOSO:+.4f}", flush=True)

    out = {
        "task": "T95 + T81 + T75 three-way ensemble eval",
        "gru_pred": gru_paths,
        "nn_seeds": args.nn_seeds,
        "lgb_seeds": args.lgb_seeds,
        "iter_014_loso_ref": ITER014_LOSO,
        "cross_corrs": {
            "gru_nn": float(np.corrcoef(p_gru, p_nn)[0,1]),
            "gru_lgb": float(np.corrcoef(p_gru, p_lgb)[0,1]),
            "nn_lgb": float(np.corrcoef(p_nn, p_lgb)[0,1]),
        },
        "results": results,
        "best": best,
    }
    if args.out is None:
        args.out = os.path.join(HERE, "three_way_ensemble_results.json")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
