"""DE 4-D threshold search per horizon.

For each h ∈ {5, 10, 20, 40, 60}:
  - Load 5-seed pred_lgb_h{H}_seed{S}.parquet
  - Load 5-seed pred_cb_h{H}_seed{S}.parquet (if available)
  - Sweep weights (w_lgb, w_cb) and (thr_up, thr_dn) jointly via DE.

Outputs results.json keyed by horizon, plus per-horizon thresholds.

Note for h=60 reuse:
  - h=60 LGB pred files come from T75 (pred_T75_seed{S}.parquet)
  - h=60 CB  pred files come from T89 (pred_T89_seed{S}.parquet)
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

T75_DIR = os.path.join(ROOT, "experiments", "T75_regression_dmid")
T89_DIR = os.path.join(ROOT, "experiments", "T89_catboost_regression")

SEEDS = (1, 7, 13, 42, 100)
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001
ITER014 = 38.281
ITER015 = 40.13


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
        "pred_lgb": df[df["sym"] == k]["pred_lgb"].to_numpy(np.float64),
        "pred_cb":  df[df["sym"] == k]["pred_cb"].to_numpy(np.float64),
        "mp_t":     df[df["sym"] == k]["midprice_t"].to_numpy(np.float64),
        "mp_th":    df[df["sym"] == k]["midprice_th"].to_numpy(np.float64),
    } for k in SYMS]


def load_pred_h(h: int) -> pd.DataFrame:
    """Load and average 5-seed LGB and CB preds for horizon h."""
    if h == 60:
        lgb_dir, lgb_pref = T75_DIR, "pred_T75_seed"
        cb_dir, cb_pref = T89_DIR, "pred_T89_seed"
    else:
        lgb_dir, lgb_pref = HERE, f"pred_lgb_h{h}_seed"
        cb_dir, cb_pref = HERE, f"pred_cb_h{h}_seed"

    lgb_preds = []
    base = None
    for s in SEEDS:
        p = os.path.join(lgb_dir, f"{lgb_pref}{s}.parquet")
        if not os.path.isfile(p):
            print(f"  WARN missing {p}", flush=True)
            continue
        df = pd.read_parquet(p)
        if base is None:
            base = df[["sym", "date", "session", "t", "midprice_t", "midprice_th"]].copy()
        else:
            assert df["sym"].equals(base["sym"]) and df["t"].equals(base["t"]), \
                f"row mismatch lgb h={h} seed={s}"
        lgb_preds.append(df["pred_dmid_norm"].to_numpy(np.float64))
    p_lgb = np.mean(lgb_preds, axis=0) if lgb_preds else None

    cb_preds = []
    for s in SEEDS:
        p = os.path.join(cb_dir, f"{cb_pref}{s}.parquet")
        if not os.path.isfile(p):
            print(f"  WARN missing {p}", flush=True)
            continue
        df = pd.read_parquet(p)
        if base is None:
            base = df[["sym", "date", "session", "t", "midprice_t", "midprice_th"]].copy()
        else:
            assert df["sym"].equals(base["sym"]) and df["t"].equals(base["t"]), \
                f"row mismatch cb h={h} seed={s}"
        cb_preds.append(df["pred_dmid_norm"].to_numpy(np.float64))
    p_cb = np.mean(cb_preds, axis=0) if cb_preds else None

    if base is None:
        return None
    out = base
    out["pred_lgb"] = p_lgb if p_lgb is not None else np.zeros(len(base), dtype=np.float64)
    out["pred_cb"] = p_cb if p_cb is not None else np.zeros(len(base), dtype=np.float64)
    out["_n_lgb"] = len(lgb_preds)
    out["_n_cb"] = len(cb_preds)
    return out


def make_obj_2way(folds, w_lgb_fixed=None, w_cb_fixed=None):
    """Return scipy DE objective (minimize -PnL).

    If both weights fixed → 2-D thresh search (thr_up, thr_dn).
    Else → 4-D search (w_lgb, w_cb, thr_up, thr_dn) with weight sum normalized.
    """
    pred_lgb = [f["pred_lgb"] for f in folds]
    pred_cb  = [f["pred_cb"]  for f in folds]
    mp_t = [f["mp_t"] for f in folds]
    mp_th = [f["mp_th"] for f in folds]
    if w_lgb_fixed is not None and w_cb_fixed is not None:
        ws = w_lgb_fixed + w_cb_fixed
        if ws < 1e-9:
            ws = 1.0
        w_lgb = w_lgb_fixed / ws
        w_cb = w_cb_fixed / ws
        def f(x):
            thr_up, thr_dn = x
            s = 0.0
            for i in range(len(folds)):
                p = w_lgb * pred_lgb[i] + w_cb * pred_cb[i]
                a = ev_gate_asym(p, thr_up, thr_dn)
                s += vectorized_pnl(a, mp_t[i], mp_th[i]).sum()
            return -float(s)
        return f, 2

    def f(x):
        wl, wc, thr_up, thr_dn = x
        ws = wl + wc
        if ws < 1e-6:
            return 0.0
        wl /= ws
        wc /= ws
        s = 0.0
        for i in range(len(folds)):
            p = wl * pred_lgb[i] + wc * pred_cb[i]
            a = ev_gate_asym(p, thr_up, thr_dn)
            s += vectorized_pnl(a, mp_t[i], mp_th[i]).sum()
        return -float(s)
    return f, 4


def de_search(obj, bounds, seeds=(0, 1, 2, 7, 42), maxiter=80, popsize=24):
    runs = []
    for sd in seeds:
        r = differential_evolution(
            obj, bounds=bounds, seed=sd, maxiter=maxiter, popsize=popsize,
            polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            updating="deferred", workers=1, init="sobol",
        )
        runs.append({"seed": int(sd), "x": [float(v) for v in r.x],
                     "obj_val": float(-r.fun)})
    runs.sort(key=lambda r: r["obj_val"], reverse=True)
    return runs


def evaluate_combo(folds, w_lgb, w_cb, thr_up, thr_dn):
    de_per = []
    for f in folds:
        p = w_lgb * f["pred_lgb"] + w_cb * f["pred_cb"]
        if (w_lgb + w_cb) > 0:
            p /= (w_lgb + w_cb)
        a = ev_gate_asym(p, thr_up, thr_dn)
        de_per.append(float(vectorized_pnl(a, f["mp_t"], f["mp_th"]).sum()))
    return de_per


def evaluate_horizon(h: int) -> Dict:
    print(f"\n{'='*60}\n=== HORIZON h={h} ===\n{'='*60}", flush=True)
    df = load_pred_h(h)
    if df is None:
        print(f"  no preds available", flush=True)
        return None
    folds = split_sym(df)
    n_lgb = int(df["_n_lgb"].iloc[0])
    n_cb = int(df["_n_cb"].iloc[0])
    print(f"  n_rows={len(df):,}  n_lgb_seeds={n_lgb}  n_cb_seeds={n_cb}", flush=True)

    # Cross-corr lgb vs cb
    if n_lgb > 0 and n_cb > 0:
        c = float(np.corrcoef(df["pred_lgb"], df["pred_cb"])[0, 1])
        print(f"  corr(lgb, cb) = {c:.4f}", flush=True)

    bounds_2d = [(0.0, 0.0050), (0.0, 0.0050)]

    # Singles
    single_lgb = single_cb = None
    if n_lgb > 0:
        obj, _ = make_obj_2way(folds, w_lgb_fixed=1.0, w_cb_fixed=0.0)
        runs = de_search(obj, bounds_2d)
        best = runs[0]
        thr_up, thr_dn = best["x"]
        per_sym = evaluate_combo(folds, 1.0, 0.0, thr_up, thr_dn)
        single_lgb = {"de_sum": best["obj_val"], "thr_up": thr_up, "thr_dn": thr_dn,
                      "per_sym": per_sym, "w_lgb": 1.0, "w_cb": 0.0}
        print(f"  LGB-only:  DE={best['obj_val']:+.4f}  thr_up={thr_up:.5e} thr_dn={thr_dn:.5e}", flush=True)
    if n_cb > 0:
        obj, _ = make_obj_2way(folds, w_lgb_fixed=0.0, w_cb_fixed=1.0)
        runs = de_search(obj, bounds_2d)
        best = runs[0]
        thr_up, thr_dn = best["x"]
        per_sym = evaluate_combo(folds, 0.0, 1.0, thr_up, thr_dn)
        single_cb = {"de_sum": best["obj_val"], "thr_up": thr_up, "thr_dn": thr_dn,
                     "per_sym": per_sym, "w_lgb": 0.0, "w_cb": 1.0}
        print(f"  CB-only:   DE={best['obj_val']:+.4f}  thr_up={thr_up:.5e} thr_dn={thr_dn:.5e}", flush=True)

    # 2-way grid (a few representative weights then DE)
    combo_results = []
    if n_lgb > 0 and n_cb > 0:
        for w_lgb, w_cb in [(1, 1), (1.5, 1), (1, 1.5), (2, 1), (1, 2), (1, 0.5), (0.5, 1)]:
            obj, _ = make_obj_2way(folds, w_lgb_fixed=float(w_lgb), w_cb_fixed=float(w_cb))
            runs = de_search(obj, bounds_2d)
            best = runs[0]
            thr_up, thr_dn = best["x"]
            per_sym = evaluate_combo(folds, w_lgb, w_cb, thr_up, thr_dn)
            r = {"de_sum": best["obj_val"], "thr_up": thr_up, "thr_dn": thr_dn,
                 "per_sym": per_sym, "w_lgb": float(w_lgb), "w_cb": float(w_cb)}
            combo_results.append(r)
            print(f"  2way w_lgb={w_lgb} w_cb={w_cb}:  DE={best['obj_val']:+.4f}  thr_up={thr_up:.5e} thr_dn={thr_dn:.5e}", flush=True)

        # Joint 4D DE
        bounds_4d = [(0.0, 3.0), (0.0, 3.0), (0.0, 0.0050), (0.0, 0.0050)]
        obj4, _ = make_obj_2way(folds)
        runs4 = de_search(obj4, bounds_4d, maxiter=120, popsize=32)
        best4 = runs4[0]
        wl, wc, thr_up, thr_dn = best4["x"]
        ws = wl + wc
        if ws > 0:
            wl_n, wc_n = wl/ws, wc/ws
        else:
            wl_n, wc_n = 0.0, 0.0
        per_sym = evaluate_combo(folds, wl, wc, thr_up, thr_dn)
        joint = {"de_sum": best4["obj_val"],
                 "w_lgb": wl, "w_cb": wc,
                 "w_lgb_norm": wl_n, "w_cb_norm": wc_n,
                 "thr_up": thr_up, "thr_dn": thr_dn,
                 "per_sym": per_sym}
        print(f"  JOINT 4D:  DE={best4['obj_val']:+.4f}  w_lgb={wl:.3f} w_cb={wc:.3f} thr_up={thr_up:.5e} thr_dn={thr_dn:.5e}", flush=True)
    else:
        joint = None

    # Pick best across all combos
    pool = []
    if single_lgb: pool.append(("lgb_only", single_lgb))
    if single_cb:  pool.append(("cb_only",  single_cb))
    pool.extend([(f"2way_w{r['w_lgb']}_{r['w_cb']}", r) for r in combo_results])
    if joint: pool.append(("joint_4d", joint))
    pool.sort(key=lambda kv: kv[1]["de_sum"], reverse=True)
    best_label, best_cfg = pool[0]
    print(f"\n  BEST h={h}: {best_label}  DE={best_cfg['de_sum']:+.4f}", flush=True)

    return {
        "horizon": h,
        "n_lgb_seeds": n_lgb, "n_cb_seeds": n_cb,
        "single_lgb": single_lgb,
        "single_cb": single_cb,
        "combos_2way": combo_results,
        "joint_4d": joint,
        "best": {"label": best_label, **best_cfg},
    }


def main():
    horizons = [5, 10, 20, 40, 60]
    results = {}
    t0 = time.time()
    for h in horizons:
        results[str(h)] = evaluate_horizon(h)

    total = 0.0
    print(f"\n{'='*60}\n=== SUMMARY ===\n{'='*60}", flush=True)
    for h in horizons:
        r = results[str(h)]
        if r is None or r.get("best") is None:
            print(f"  h={h:>2}: SKIP (no preds)", flush=True)
            continue
        b = r["best"]
        print(f"  h={h:>2}: best={b['label']:<20}  DE={b['de_sum']:+.4f}  "
              f"w_lgb={b.get('w_lgb', '?'):.3f}/w_cb={b.get('w_cb', '?'):.3f}  "
              f"thr_up={b['thr_up']:.5e}  thr_dn={b['thr_dn']:.5e}",
              flush=True)
        total += b["de_sum"]
    print(f"\n  TOTAL across active horizons: DE={total:+.4f}", flush=True)
    print(f"  vs iter_014 (+{ITER014}):  {total - ITER014:+.4f}", flush=True)
    print(f"  vs iter_015 (+{ITER015}):  {total - ITER015:+.4f}", flush=True)
    print(f"\n  total time: {time.time()-t0:.1f}s", flush=True)

    out = {
        "horizons": results,
        "total_loso": float(total),
        "vs_iter014": float(total - ITER014),
        "vs_iter015": float(total - ITER015),
    }
    out_path = os.path.join(HERE, "de_thresh_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nresults -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
