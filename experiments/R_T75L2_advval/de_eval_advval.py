"""DE asymmetric thresh eval: T75 LGB L2 baseline vs trick (adv-val reweight).

Loads:
  baseline: /root/lwb_remote_pkg/preds/pred_T75_seed{42,7,13}.parquet (avg)
  trick:    /root/lwb_work_t75_advval/pred_T75L2_advval_seed{42,7,13}.parquet (avg)

For each:
  - split by sym (0..4)
  - DE asymmetric thresh search on global PnL (sum across syms) with 5 DE seeds
  - report best (thr_up, thr_dn, total_pnl, per_sym_pnl)

Output: results.json with summary fields.
"""
from __future__ import annotations
import json
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

HERE = "/root/lwb_work_t75_advval"
BASELINE_DIR = "/root/lwb_remote_pkg/preds"
SEEDS = (42, 7, 13)
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001


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
        sub = df[df["sym"] == k].reset_index(drop=True)
        out.append({
            "sym": int(k),
            "pred": sub["pred_dmid_norm"].to_numpy(np.float64),
            "label": sub["true_label"].to_numpy(np.int64),
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
        res = differential_evolution(
            objective_fn, bounds=bounds, seed=sd, maxiter=80, popsize=24,
            polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            updating="deferred", workers=1, init="sobol",
        )
        runs.append({
            "seed": int(sd),
            "thr_up": float(res.x[0]),
            "thr_dn": float(res.x[1]),
            "obj_val": float(-res.fun),
        })
    runs.sort(key=lambda r: r["obj_val"], reverse=True)
    return runs


def per_sym_pnl_at(folds, thr_up, thr_dn):
    out = []
    for f in folds:
        a = ev_gate_asymmetric(f["pred"], thr_up, thr_dn)
        out.append(float(vectorized_pnl(a, f["mp_t"], f["mp_th"]).sum()))
    return out


def load_avg(prefix_paths):
    base = pd.read_parquet(prefix_paths[0])
    n = len(base)
    p = np.zeros(n, dtype=np.float64)
    for path in prefix_paths:
        df = pd.read_parquet(path)
        if len(df) != n:
            raise RuntimeError(f"row mismatch {path}")
        p += df["pred_dmid_norm"].to_numpy(np.float64)
    p /= len(prefix_paths)
    base = base.copy()
    base["pred_dmid_norm"] = p.astype(np.float32)
    return base


def evaluate(df, label):
    folds = split_by_sym(df)
    obj_fn = make_obj_loso_asym(folds)
    bounds = [(-3 * FEE, 3 * FEE), (-3 * FEE, 3 * FEE)]
    runs = de_search(obj_fn, bounds)
    best = runs[0]
    per_sym = per_sym_pnl_at(folds, best["thr_up"], best["thr_dn"])
    total_loso = float(sum(per_sym))
    print(f"\n=== {label} ===", flush=True)
    print(f"  best thr_up={best['thr_up']:+.6f}  thr_dn={best['thr_dn']:+.6f}",
          flush=True)
    print(f"  per-sym: {[f'{v:+.4f}' for v in per_sym]}", flush=True)
    print(f"  TOTAL (LOSO-equiv) = {total_loso:+.4f}", flush=True)
    return {
        "label": label,
        "de_runs": runs,
        "best_thr_up": best["thr_up"],
        "best_thr_dn": best["thr_dn"],
        "per_sym_pnl": per_sym,
        "total_loso_equiv": total_loso,
        "min_per_sym": float(min(per_sym)),
    }


def main():
    base_paths = [os.path.join(BASELINE_DIR, f"pred_T75_seed{s}.parquet")
                  for s in SEEDS]
    trick_paths = [os.path.join(HERE, f"pred_T75L2_advval_seed{s}.parquet")
                   for s in SEEDS]

    print("Loading baseline preds (5-seed avg from canonical T75)...", flush=True)
    df_base = load_avg(base_paths)
    print(f"  baseline: n={len(df_base):,}  pred mean={df_base['pred_dmid_norm'].mean():.6e} "
          f"std={df_base['pred_dmid_norm'].std():.6e}", flush=True)

    print("Loading trick preds (advval-reweighted T75 L2)...", flush=True)
    df_trick = load_avg(trick_paths)
    print(f"  trick: n={len(df_trick):,}  pred mean={df_trick['pred_dmid_norm'].mean():.6e} "
          f"std={df_trick['pred_dmid_norm'].std():.6e}", flush=True)

    res_base = evaluate(df_base, f"BASELINE (T75 LGB L2, 3-seed avg {SEEDS})")
    res_trick = evaluate(df_trick, f"TRICK (T75 LGB L2 + advval, 3-seed avg {SEEDS})")

    delta = res_trick["total_loso_equiv"] - res_base["total_loso_equiv"]
    delta_per_sym = [t - b for t, b in zip(res_trick["per_sym_pnl"], res_base["per_sym_pnl"])]
    print(f"\n=== DELTA (trick - baseline) ===", flush=True)
    print(f"  per-sym: {[f'{v:+.4f}' for v in delta_per_sym]}", flush=True)
    print(f"  total = {delta:+.4f}  min_per_sym = {min(delta_per_sym):+.4f}",
          flush=True)

    summary = {
        "task": "T75 LGB L2 advval-reweight (3 seeds 42/7/13)",
        "seeds": list(SEEDS),
        "baseline": res_base,
        "trick": res_trick,
        "delta_total_loso": delta,
        "delta_per_sym": delta_per_sym,
        "delta_min_per_sym": float(min(delta_per_sym)),
    }
    out = os.path.join(HERE, "results.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out}", flush=True)


if __name__ == "__main__":
    main()
