"""Fast 3-way ensemble eval: tries a small set of weight grids, single DE seed per combo."""
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
    valid_b = ~np.isnan(p)
    p = np.where(np.isnan(p), 0.0, p)
    cnt = np.where(valid_b, 1.0, 0.0)
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
        m = (df["sym"] == k).to_numpy()
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


def de_one(folds, bounds=((0.0, 0.0040), (0.0, 0.0040)), seed=0, maxiter=40, popsize=16):
    r = differential_evolution(
        de_loso_obj(folds), bounds=bounds, seed=seed, maxiter=maxiter, popsize=popsize,
        polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
        updating="deferred", workers=1, init="sobol",
    )
    return float(r.x[0]), float(r.x[1]), float(-r.fun)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gru-preds", nargs="+", required=True)
    ap.add_argument("--nn-seeds", default="1,7,13,42,100")
    ap.add_argument("--lgb-seeds", default="1,7,13,42,100")
    ap.add_argument("--out", required=True)
    ap.add_argument("--full-grid", action="store_true",
                    help="Sweep many weight combos. Otherwise just (0.5, 0.75) × (1.0, 1.5).")
    args = ap.parse_args()

    nn_paths = [os.path.join(ROOT, "experiments", "T81_nn_regression_pnl", f"pred_T81_seed{s}.parquet")
                for s in args.nn_seeds.split(",")]
    lgb_paths = [os.path.join(ROOT, "experiments", "T75_regression_dmid", f"pred_T75_seed{s}.parquet")
                 for s in args.lgb_seeds.split(",")]
    gru_paths = args.gru_preds

    print(f"Loading: gru={len(gru_paths)} nn={len(nn_paths)} lgb={len(lgb_paths)}", flush=True)
    base_g, p_gru = load_avg(gru_paths)
    base_n, p_nn = load_avg(nn_paths)
    base_l, p_lgb = load_avg(lgb_paths)

    # Sort all by canonical key (sym, date, session, t)
    cols = ["sym", "date", "session", "t"]
    df_g = base_g.copy(); df_g["pred_dmid_norm"] = p_gru
    df_n = base_n.copy(); df_n["pred_dmid_norm"] = p_nn
    df_l = base_l.copy(); df_l["pred_dmid_norm"] = p_lgb
    df_g = df_g.sort_values(cols).reset_index(drop=True)
    df_n = df_n.sort_values(cols).reset_index(drop=True)
    df_l = df_l.sort_values(cols).reset_index(drop=True)
    base_g = df_g
    p_gru = df_g["pred_dmid_norm"].to_numpy(np.float64)
    p_nn = df_n["pred_dmid_norm"].to_numpy(np.float64)
    p_lgb = df_l["pred_dmid_norm"].to_numpy(np.float64)

    print(f"  std: gru={p_gru.std():.6e} nn={p_nn.std():.6e} lgb={p_lgb.std():.6e}", flush=True)
    cc_gn = float(np.corrcoef(p_gru, p_nn)[0, 1])
    cc_gl = float(np.corrcoef(p_gru, p_lgb)[0, 1])
    cc_nl = float(np.corrcoef(p_nn, p_lgb)[0, 1])
    print(f"  cross-corr: gru-nn={cc_gn:.4f} gru-lgb={cc_gl:.4f} nn-lgb={cc_nl:.4f}", flush=True)

    p_gru_z = p_gru / max(p_gru.std(), 1e-12)
    p_nn_z = p_nn / max(p_nn.std(), 1e-12)
    p_lgb_z = p_lgb / max(p_lgb.std(), 1e-12)
    canonical_std = float((p_nn.std() + p_lgb.std()) / 2.0)

    if args.full_grid:
        wg_grid = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
        wn_grid = [1.0]
        wl_grid = [1.0, 1.5]
    else:
        wg_grid = [0.0, 0.4, 0.5, 0.6, 0.75, 1.0]
        wn_grid = [1.0]
        wl_grid = [1.5]

    results = []
    for wg in wg_grid:
        for wn in wn_grid:
            for wl in wl_grid:
                t0 = time.time()
                ens_z = wg * p_gru_z + wn * p_nn_z + wl * p_lgb_z
                ens = ens_z / max(ens_z.std(), 1e-12) * canonical_std
                df_e = base_g.copy()
                df_e["pred_dmid_norm"] = ens
                folds = split_by_sym(df_e, ens)
                thr_up, thr_dn, obj = de_one(folds, seed=0, maxiter=40, popsize=16)
                a_best = ev_gate_asym(ens, thr_up, thr_dn)
                per = []
                for f in folds:
                    a = ev_gate_asym(f["pred"], thr_up, thr_dn)
                    per.append(float(vectorized_pnl(a, f["mp_t"], f["mp_th"]).sum()))
                loso_sum = sum(per)
                el = time.time() - t0
                results.append({
                    "wg": wg, "wn": wn, "wl": wl,
                    "de_thr_up": thr_up, "de_thr_dn": thr_dn,
                    "de_loso_sum": loso_sum, "de_per_sym": per,
                    "elapsed_sec": el,
                })
                print(f"  w_g={wg:.2f} w_n={wn:.2f} w_l={wl:.2f}  "
                      f"DE_LOSO={loso_sum:+.4f}  thr=({thr_up:.5f}, {thr_dn:.5f})  ({el:.1f}s)",
                      flush=True)

    best = max(results, key=lambda r: r["de_loso_sum"])
    print(f"\nBEST: w_g={best['wg']} w_n={best['wn']} w_l={best['wl']}  "
          f"DE_LOSO={best['de_loso_sum']:+.4f}", flush=True)
    print(f"vs iter_014 (+{ITER014_LOSO:.2f}): {best['de_loso_sum'] - ITER014_LOSO:+.4f}", flush=True)

    out = {
        "task": "T95+T81+T75 fast 3-way ensemble",
        "gru_preds": gru_paths,
        "nn_seeds": args.nn_seeds, "lgb_seeds": args.lgb_seeds,
        "iter_014_loso_ref": ITER014_LOSO,
        "cross_corrs": {"gru_nn": cc_gn, "gru_lgb": cc_gl, "nn_lgb": cc_nl},
        "results": results,
        "best": best,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
