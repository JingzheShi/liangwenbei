"""DE asymmetric thresh eval: 5-LOSO ensemble strategies vs full-5sym baseline.

Loads:
  full   : /root/lwb_remote_pkg/preds/pred_T75_seed{42,7,13}.parquet (3-seed avg)
  loso k : /root/lwb_work_t75_5loso/pred_T75L2_loso{k}_seed{42,7,13}.parquet (3-seed avg)
           for k in 0..4

Strategies:
  baseline   = full-5sym 3-seed avg                       (= existing T75 LGB L2)
  strategy_A = avg(5 LOSO models)                         (no full)
  strategy_B = (sum(5 LOSO) + 1*full) / 6                 (equal-weight 6-model)
  strategy_C = 0.4 * avg(5 LOSO) + 0.6 * full

For each: split by sym, DE asym-thresh search (5 DE seeds), report best.

Output: results.json
"""
from __future__ import annotations
import json
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

HERE = "/root/lwb_work_t75_5loso"
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


def split_by_sym(df, pred_col="pred_dmid_norm"):
    out = []
    for k in SYMS:
        sub = df[df["sym"] == k].reset_index(drop=True)
        out.append({
            "sym": int(k),
            "pred": sub[pred_col].to_numpy(np.float64),
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


def load_avg(paths):
    """Load N parquet files, average pred_dmid_norm column. All must align row-by-row."""
    base = pd.read_parquet(paths[0])
    n = len(base)
    p = np.zeros(n, dtype=np.float64)
    for path in paths:
        df = pd.read_parquet(path)
        if len(df) != n:
            raise RuntimeError(f"row mismatch {path}: {len(df)} vs {n}")
        p += df["pred_dmid_norm"].to_numpy(np.float64)
    p /= len(paths)
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
    print(f"  TOTAL (LOSO-equiv) = {total_loso:+.4f}  min_per_sym = {min(per_sym):+.4f}",
          flush=True)
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
    # Full 5-sym baseline (3-seed avg from canonical T75)
    full_paths = [os.path.join(BASELINE_DIR, f"pred_T75_seed{s}.parquet") for s in SEEDS]
    print("Loading FULL T75 baseline (3-seed avg)...", flush=True)
    df_full = load_avg(full_paths)
    p_full = df_full["pred_dmid_norm"].to_numpy(np.float64).copy()
    print(f"  full: n={len(df_full):,}  pred mean={p_full.mean():.6e} std={p_full.std():.6e}",
          flush=True)

    # 5 LOSO models, each 3-seed avg
    loso_avgs = []  # list of arrays, one per LOSO sym
    for k in SYMS:
        paths = [os.path.join(HERE, f"pred_T75L2_loso{k}_seed{s}.parquet") for s in SEEDS]
        print(f"Loading LOSO-{k} preds (3-seed avg)...", flush=True)
        df_k = load_avg(paths)
        # align row order with df_full
        if len(df_k) != len(df_full):
            raise RuntimeError(f"LOSO-{k} row count {len(df_k)} != full {len(df_full)}")
        loso_avgs.append(df_k["pred_dmid_norm"].to_numpy(np.float64))
        print(f"  loso{k}: pred mean={loso_avgs[-1].mean():.6e} std={loso_avgs[-1].std():.6e}",
              flush=True)

    avg_loso = np.mean(np.stack(loso_avgs, axis=0), axis=0)  # avg of 5 LOSO models
    print(f"\nAVG-of-5-LOSO: mean={avg_loso.mean():.6e} std={avg_loso.std():.6e}", flush=True)

    # Build DataFrames for each strategy. Reuse df_full skeleton; only swap pred col.
    def with_pred(p):
        d = df_full.copy()
        d["pred_dmid_norm"] = p.astype(np.float32)
        return d

    # baseline = full-5sym (already in df_full)
    # strategy_A = avg(5 LOSO)
    p_A = avg_loso
    # strategy_B = (sum(5 LOSO) + full) / 6
    p_B = (np.sum(np.stack(loso_avgs, axis=0), axis=0) + p_full) / 6.0
    # strategy_C = 0.4 * avg_loso + 0.6 * full
    p_C = 0.4 * avg_loso + 0.6 * p_full

    res_baseline = evaluate(df_full, "BASELINE (T75 full-5sym, 3-seed avg)")
    res_A = evaluate(with_pred(p_A), "STRATEGY_A (avg of 5 LOSO, 3-seed each)")
    res_B = evaluate(with_pred(p_B), "STRATEGY_B (5 LOSO + full equal weight, 6-model)")
    res_C = evaluate(with_pred(p_C), "STRATEGY_C (0.4*avg_LOSO + 0.6*full)")

    deltas = {}
    for name, res in [("strategy_A", res_A), ("strategy_B", res_B), ("strategy_C", res_C)]:
        d_total = res["total_loso_equiv"] - res_baseline["total_loso_equiv"]
        d_per = [t - b for t, b in zip(res["per_sym_pnl"], res_baseline["per_sym_pnl"])]
        d_min = res["min_per_sym"] - res_baseline["min_per_sym"]
        deltas[name] = {
            "delta_total": float(d_total),
            "delta_per_sym": d_per,
            "delta_min_per_sym": float(d_min),
            "delta_per_sym_min_componentwise": float(min(d_per)),
        }
        print(f"\n=== DELTA {name} - baseline ===", flush=True)
        print(f"  per-sym: {[f'{v:+.4f}' for v in d_per]}", flush=True)
        print(f"  total = {d_total:+.4f}  min_per_sym(strat-base) = {d_min:+.4f}  "
              f"min(delta_per_sym) = {min(d_per):+.4f}",
              flush=True)

    # Pick best by total_loso
    cands = [
        ("baseline", res_baseline["total_loso_equiv"]),
        ("strategy_A", res_A["total_loso_equiv"]),
        ("strategy_B", res_B["total_loso_equiv"]),
        ("strategy_C", res_C["total_loso_equiv"]),
    ]
    cands.sort(key=lambda x: x[1], reverse=True)
    best_name, best_total = cands[0]
    delta_best = best_total - res_baseline["total_loso_equiv"]

    summary = {
        "task": "T75 LGB L2 5-LOSO ensemble (3 seeds 42/7/13 per model)",
        "seeds": list(SEEDS),
        "syms_held_out": list(SYMS),
        "baseline": res_baseline,
        "strategy_A": res_A,
        "strategy_B": res_B,
        "strategy_C": res_C,
        "deltas": deltas,
        "best_strategy": best_name,
        "best_total_loso": float(best_total),
        "delta_best_vs_baseline": float(delta_best),
    }
    out = os.path.join(HERE, "results.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== SUMMARY ===", flush=True)
    print(f"  baseline       = {res_baseline['total_loso_equiv']:+.4f} "
          f"(min_per_sym={res_baseline['min_per_sym']:+.4f})", flush=True)
    print(f"  strategy_A     = {res_A['total_loso_equiv']:+.4f} "
          f"(min_per_sym={res_A['min_per_sym']:+.4f})", flush=True)
    print(f"  strategy_B     = {res_B['total_loso_equiv']:+.4f} "
          f"(min_per_sym={res_B['min_per_sym']:+.4f})", flush=True)
    print(f"  strategy_C     = {res_C['total_loso_equiv']:+.4f} "
          f"(min_per_sym={res_C['min_per_sym']:+.4f})", flush=True)
    print(f"  BEST = {best_name} ({best_total:+.4f}, delta={delta_best:+.4f})", flush=True)
    print(f"\nSaved {out}", flush=True)


if __name__ == "__main__":
    main()
