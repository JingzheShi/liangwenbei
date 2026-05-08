"""T93: Cross-correlation + ensemble evaluation.

Loads:
  * T93 candidate predictions: pred_T93_<tag>_seed<S>.parquet
  * T81 NN baseline: pred_T81_seed{1,7,13,42,100}.parquet
  * T75 LGB baseline: experiments/T75_regression_dmid/pred_T75_seed*.parquet (if avail)

For each candidate config:
  * Average predictions across its seeds
  * Compute cross-correlation with T81 (5-seed avg) and T75 (5-seed avg)
  * Compute single-config EV-gate (DE asymmetric LOSO-equiv)
  * Compute (T81+candidate) 2NN ensemble: weighted avg + DE asym
  * Compute (T81+T75+candidate) 3-way ensemble: weighted avg + DE asym
"""
from __future__ import annotations

import argparse
import glob
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


def load_pred_avg(paths):
    base = pd.read_parquet(paths[0])
    p_sum = base["pred_dmid_norm"].to_numpy(np.float64).copy()
    n = len(base)
    for p in paths[1:]:
        df = pd.read_parquet(p)
        if len(df) != n:
            raise RuntimeError(f"row count mismatch: {p}")
        if not (df["sym"].equals(base["sym"]) and df["date"].equals(base["date"])
                and df["t"].equals(base["t"])):
            raise RuntimeError(f"row order mismatch: {p}")
        p_sum += df["pred_dmid_norm"].to_numpy(np.float64)
    p_avg = p_sum / float(len(paths))
    out = base.copy()
    out["pred_dmid_norm"] = p_avg.astype(np.float32)
    return out


def split_by_sym(df):
    return [{
        "sym": int(k),
        "pred": df[df["sym"] == k]["pred_dmid_norm"].to_numpy(np.float64),
        "mp_t": df[df["sym"] == k]["midprice_t"].to_numpy(np.float64),
        "mp_th": df[df["sym"] == k]["midprice_th"].to_numpy(np.float64),
    } for k in SYMS]


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


def make_obj_loso(folds):
    pred = [f["pred"] for f in folds]
    mp_t = [f["mp_t"] for f in folds]
    mp_th = [f["mp_th"] for f in folds]
    def f(x):
        u, d = x
        s = 0.0
        for i in range(len(folds)):
            a = ev_gate_asymmetric(pred[i], u, d)
            s += vectorized_pnl(a, mp_t[i], mp_th[i]).sum()
        return -float(s)
    return f


def de_search_loso(pred_all, mp_t_all, mp_th_all, sym_all, bounds=((0., 0.005), (0., 0.005)),
                    seeds=(0, 1, 2)):
    folds = []
    for k in SYMS:
        m = sym_all == k
        folds.append({"sym": k, "pred": pred_all[m], "mp_t": mp_t_all[m], "mp_th": mp_th_all[m]})
    obj = make_obj_loso(folds)
    runs = []
    for sd in seeds:
        r = differential_evolution(obj, bounds=list(bounds), seed=sd, maxiter=60, popsize=18,
                                    polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
                                    updating="deferred", workers=1, init="sobol")
        runs.append({"seed": int(sd), "thr_up": float(r.x[0]), "thr_dn": float(r.x[1]),
                     "obj_val": float(-r.fun)})
    runs.sort(key=lambda r: r["obj_val"], reverse=True)
    best = runs[0]
    # per-sym at best
    per_sym = []
    for f in folds:
        a = ev_gate_asymmetric(f["pred"], best["thr_up"], best["thr_dn"])
        per_sym.append(float(vectorized_pnl(a, f["mp_t"], f["mp_th"]).sum()))
    return best, per_sym, runs


def list_t81_paths():
    paths = []
    for s in (1, 7, 13, 42, 100):
        p = os.path.join(T81_DIR, f"pred_T81_seed{s}.parquet")
        if os.path.exists(p):
            paths.append(p)
    return paths


def list_t75_paths():
    cand = sorted(glob.glob(os.path.join(T75_DIR, "pred_T75_seed*.parquet")))
    return cand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-tag", required=True,
                    help="T93 tag, e.g. 'ft_std' (will look for pred_T93_<tag>_seed*.parquet)")
    ap.add_argument("--seeds", default="42",
                    help="Comma-separated seed list to use for candidate (tag must match)")
    ap.add_argument("--w-cand-grid", default="0.5,0.75,1.0,1.25,1.5,2.0",
                    help="Candidate weight grid for 2NN ensemble (T81 fixed at 1.0)")
    ap.add_argument("--w-lgb-grid", default="1.0,1.5,2.0",
                    help="LGB weight grid for 3-way ensemble")
    ap.add_argument("--out", default="cross_corr_results.json")
    args = ap.parse_args()

    cand_seeds = [int(x) for x in args.seeds.split(",")]
    print(f"=== T93 cross-corr eval: tag={args.candidate_tag} seeds={cand_seeds} ===", flush=True)

    cand_paths = []
    for s in cand_seeds:
        p = os.path.join(HERE, f"pred_T93_{args.candidate_tag}_seed{s}.parquet")
        if not os.path.exists(p):
            print(f"  WARN missing {p}", flush=True)
            continue
        cand_paths.append(p)
    if not cand_paths:
        raise SystemExit("no candidate paths found")
    cand_df = load_pred_avg(cand_paths)
    print(f"  candidate: {len(cand_paths)} seeds, {len(cand_df):,} rows", flush=True)

    t81_paths = list_t81_paths()
    print(f"  T81 NN: {len(t81_paths)} seeds", flush=True)
    t81_df = load_pred_avg(t81_paths)

    t75_paths = list_t75_paths()
    print(f"  T75 LGB: {len(t75_paths)} seeds", flush=True)
    t75_df = load_pred_avg(t75_paths) if t75_paths else None

    # Align columns: assume same order (sym/date/t)
    if not (cand_df["sym"].equals(t81_df["sym"]) and cand_df["t"].equals(t81_df["t"])):
        raise RuntimeError("cand vs t81 row order mismatch")
    if t75_df is not None and not (cand_df["sym"].equals(t75_df["sym"]) and cand_df["t"].equals(t75_df["t"])):
        raise RuntimeError("cand vs t75 row order mismatch")

    cand_pred = cand_df["pred_dmid_norm"].to_numpy(np.float64)
    t81_pred = t81_df["pred_dmid_norm"].to_numpy(np.float64)
    t75_pred = t75_df["pred_dmid_norm"].to_numpy(np.float64) if t75_df is not None else None

    sym_all = cand_df["sym"].to_numpy(np.int64)
    mp_t_all = cand_df["midprice_t"].to_numpy(np.float64)
    mp_th_all = cand_df["midprice_th"].to_numpy(np.float64)

    # Cross-correlation
    cc_t81 = float(np.corrcoef(cand_pred, t81_pred)[0, 1])
    cc_t75 = float(np.corrcoef(cand_pred, t75_pred)[0, 1]) if t75_pred is not None else None
    cc_t81_t75 = float(np.corrcoef(t81_pred, t75_pred)[0, 1]) if t75_pred is not None else None
    print(f"  cross-corr cand vs T81 NN  : {cc_t81:.4f}", flush=True)
    if cc_t75 is not None:
        print(f"  cross-corr cand vs T75 LGB : {cc_t75:.4f}", flush=True)
        print(f"  cross-corr T81 vs T75      : {cc_t81_t75:.4f}", flush=True)

    # Single-config DE asymmetric LOSO
    print(f"\n[1] Single candidate DE LOSO:", flush=True)
    t0 = time.time()
    best_cand, per_sym_cand, _ = de_search_loso(cand_pred, mp_t_all, mp_th_all, sym_all)
    cand_loso = sum(per_sym_cand)
    print(f"  thr_up={best_cand['thr_up']:.6f} thr_dn={best_cand['thr_dn']:.6f}  "
          f"sum_per_sym={cand_loso:+.4f}  per_sym={per_sym_cand}  ({time.time()-t0:.1f}s)", flush=True)

    # Reference: T81 alone DE LOSO
    print(f"\n[2] T81 NN-only DE LOSO:", flush=True)
    t0 = time.time()
    best_t81, per_sym_t81, _ = de_search_loso(t81_pred, mp_t_all, mp_th_all, sym_all)
    t81_loso = sum(per_sym_t81)
    print(f"  thr_up={best_t81['thr_up']:.6f} thr_dn={best_t81['thr_dn']:.6f}  "
          f"sum_per_sym={t81_loso:+.4f}  ({time.time()-t0:.1f}s)", flush=True)

    # 2NN ensemble: weighted avg over weight grid
    print(f"\n[3] 2NN ensemble (T81 + cand):", flush=True)
    w_grid = [float(x) for x in args.w_cand_grid.split(",")]
    best_2nn = None
    for w_c in w_grid:
        ens = (1.0 * t81_pred + w_c * cand_pred) / (1.0 + w_c)
        best_e, per_sym_e, _ = de_search_loso(ens, mp_t_all, mp_th_all, sym_all)
        s = sum(per_sym_e)
        print(f"  w_t81=1.0 w_cand={w_c:.2f}  sum_per_sym={s:+.4f}  thr_up={best_e['thr_up']:.6f} thr_dn={best_e['thr_dn']:.6f}",
              flush=True)
        if (best_2nn is None) or (s > best_2nn["sum"]):
            best_2nn = {"w_t81": 1.0, "w_cand": w_c, "thr_up": best_e["thr_up"],
                         "thr_dn": best_e["thr_dn"], "sum": s, "per_sym": per_sym_e}
    print(f"  BEST 2NN: w_cand={best_2nn['w_cand']:.2f} sum_per_sym={best_2nn['sum']:+.4f}", flush=True)

    # 3-way: T81+T75+cand
    best_3way = None
    if t75_pred is not None:
        print(f"\n[4] 3-way ensemble (T81 + T75 + cand):", flush=True)
        w_lgb_grid = [float(x) for x in args.w_lgb_grid.split(",")]
        for w_l in w_lgb_grid:
            for w_c in w_grid:
                ens = (1.0 * t81_pred + w_l * t75_pred + w_c * cand_pred) / (1.0 + w_l + w_c)
                best_e, per_sym_e, _ = de_search_loso(ens, mp_t_all, mp_th_all, sym_all)
                s = sum(per_sym_e)
                print(f"  w_t81=1.0 w_lgb={w_l:.2f} w_cand={w_c:.2f}  sum_per_sym={s:+.4f}",
                      flush=True)
                if (best_3way is None) or (s > best_3way["sum"]):
                    best_3way = {"w_t81": 1.0, "w_lgb": w_l, "w_cand": w_c,
                                  "thr_up": best_e["thr_up"], "thr_dn": best_e["thr_dn"],
                                  "sum": s, "per_sym": per_sym_e}
        print(f"  BEST 3way: w_lgb={best_3way['w_lgb']:.2f} w_cand={best_3way['w_cand']:.2f} "
              f"sum_per_sym={best_3way['sum']:+.4f}", flush=True)

    # Save
    out = {
        "candidate_tag": args.candidate_tag,
        "candidate_seeds": cand_seeds,
        "n_test": len(cand_df),
        "cross_corr": {"cand_vs_T81": cc_t81, "cand_vs_T75": cc_t75, "T81_vs_T75": cc_t81_t75},
        "candidate_only_loso": {"sum_per_sym": cand_loso, "per_sym": per_sym_cand,
                                "thr_up": best_cand["thr_up"], "thr_dn": best_cand["thr_dn"]},
        "t81_only_loso": {"sum_per_sym": t81_loso, "per_sym": per_sym_t81,
                          "thr_up": best_t81["thr_up"], "thr_dn": best_t81["thr_dn"]},
        "best_2nn": best_2nn,
        "best_3way": best_3way,
    }
    out_path = os.path.join(HERE, args.out)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nwrote -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
