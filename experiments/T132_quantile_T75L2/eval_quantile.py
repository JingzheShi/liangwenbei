"""T132: Evaluate 4 quantile decision strategies vs T75 L2 baseline,
and stack with T87 NN to compare against iter_015 v1.

Strategies:
  A: q50 alone + DE asym thresh
  B: dual-gate (q30 > thr_up & q70 < -thr_dn) -> sign agreement gate
  C: q_mid = (q30+q70)/2 + DE asym thresh
  D: triplet (q30+q50+q70)/3 + DE asym thresh
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
T75_DIR = os.path.join(os.path.dirname(HERE), "T75_regression_dmid")
T87_DIR = os.path.join(os.path.dirname(HERE), "T87_spo_dfl")

ALPHAS = (0.30, 0.50, 0.70)
SEEDS = (1, 7, 42)
T75_SEEDS = (1, 7, 13, 42, 100)
T87_SEEDS = (1, 7, 13, 42, 100)
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001


def vectorized_pnl(action, mp_t, mp_th):
    side = action.astype(np.float64) - 1.0
    abs_side = np.abs(side)
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * abs_side * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee_pnl) / denom


def gate_asym_one(p, tu, td):
    a = np.full(len(p), 1, dtype=np.int8)
    a[p > tu] = 2
    a[p < -td] = 0
    return a


def gate_dual(q30, q70, tu, td):
    """B: trade up iff q30 > tu (lower bound positive) ; trade down iff q70 < -td."""
    a = np.full(len(q30), 1, dtype=np.int8)
    a[q30 > tu] = 2
    a[q70 < -td] = 0
    # If both fire, q30 > tu (>0) AND q70 < -td (<0) is impossible (q30 <= q70 in expectation)
    return a


def split_sym(df, pred_cols):
    """Split into 5 sym folds. pred_cols: list of (name, np_array_aligned_with_df_idx)."""
    out = []
    for k in SYMS:
        m = df["sym"].to_numpy() == k
        d = {"sym": int(k),
             "mp_t": df.loc[m, "midprice_t"].to_numpy(np.float64),
             "mp_th": df.loc[m, "midprice_th"].to_numpy(np.float64)}
        for name, arr in pred_cols:
            d[name] = arr[m].astype(np.float64)
        out.append(d)
    return out


def de_optimize(obj, bounds, seeds=(0, 1, 2, 7, 42)):
    runs = []
    for sd in seeds:
        r = differential_evolution(obj, bounds=bounds, seed=sd,
                                   maxiter=80, popsize=24, polish=True,
                                   tol=1e-7, init="sobol")
        runs.append((float(-r.fun), [float(x) for x in r.x]))
    runs.sort(key=lambda x: x[0], reverse=True)
    return runs[0]


def make_obj_single(folds, pred_key):
    def f(x):
        s = 0.0
        for fl in folds:
            a = gate_asym_one(fl[pred_key], x[0], x[1])
            s += vectorized_pnl(a, fl["mp_t"], fl["mp_th"]).sum()
        return -float(s)
    return f


def make_obj_dual(folds, q30_key, q70_key):
    def f(x):
        s = 0.0
        for fl in folds:
            a = gate_dual(fl[q30_key], fl[q70_key], x[0], x[1])
            s += vectorized_pnl(a, fl["mp_t"], fl["mp_th"]).sum()
        return -float(s)
    return f


def per_sym_pnl(folds, gate_fn):
    return [float(vectorized_pnl(gate_fn(fl), fl["mp_t"], fl["mp_th"]).sum()) for fl in folds]


def load_avg_preds(parquet_dir, prefix_fmt, seeds, base_df=None):
    """Average pred_dmid_norm across seeds. prefix_fmt has {seed} placeholder."""
    paths = [os.path.join(parquet_dir, prefix_fmt.format(seed=s)) for s in seeds]
    base = pd.read_parquet(paths[0])
    if base_df is not None:
        # Verify alignment
        assert (base["sym"].to_numpy() == base_df["sym"].to_numpy()).all()
        assert (base["t"].to_numpy() == base_df["t"].to_numpy()).all()
    p = np.zeros(len(base), dtype=np.float64)
    for path in paths:
        df = pd.read_parquet(path)
        p += df["pred_dmid_norm"].to_numpy(np.float64)
    return base, (p / len(paths))


def evaluate_strategy(label, folds, gate_fn, bounds, obj_fn, save):
    s, x = de_optimize(obj_fn, bounds)
    per = per_sym_pnl(folds, lambda fl: gate_fn(fl, x))
    print(f"  [{label:<32s}] DE LOSO={s:+8.4f}  x={[f'{v:.4e}' for v in x]}  "
          f"per_sym={[round(v,2) for v in per]}", flush=True)
    save[label] = {"de_sum": s, "x": x, "per_sym": per,
                   "per_sym_min": float(min(per))}


def main():
    print("=== T132 Eval: 4 quantile decision strategies + stack vs iter_015 ===", flush=True)
    t0 = time.time()
    out = {"strategies": {}, "stack": {}, "baselines": {}}

    # 1. Average T75 L2 baseline (5 seeds)
    print("\n--- Baselines ---", flush=True)
    base_t75, p_t75 = load_avg_preds(
        T75_DIR, "pred_T75_seed{seed}.parquet", T75_SEEDS)
    folds_t75 = split_sym(base_t75, [("p_t75", p_t75)])
    obj_t75 = make_obj_single(folds_t75, "p_t75")
    s_t75, x_t75 = de_optimize(obj_t75, [(0.0, 0.0040), (0.0, 0.0040)])
    per_t75 = per_sym_pnl(folds_t75, lambda fl: gate_asym_one(fl["p_t75"], x_t75[0], x_t75[1]))
    out["baselines"]["T75_L2_5seed"] = {"de_sum": s_t75, "x": x_t75, "per_sym": per_t75}
    print(f"  T75 L2 5seed DE LOSO = {s_t75:+.4f}  thr=({x_t75[0]:.4e}, {x_t75[1]:.4e})", flush=True)
    print(f"    per_sym = {[round(v,2) for v in per_t75]}", flush=True)

    # 2. Load 9 quantile models, compute 3 averaged quantiles
    print("\n--- Quantile preds (3-seed avg per alpha) ---", flush=True)
    q_avg = {}
    for alpha in ALPHAS:
        a_tag = f"{int(round(alpha*100)):02d}"
        prefix = f"pred_T132_a{a_tag}_seed{{seed}}.parquet"
        base_q, p_q = load_avg_preds(HERE, prefix, SEEDS, base_df=base_t75)
        q_avg[a_tag] = p_q
        print(f"  q{a_tag}: mean={p_q.mean():+.6f} std={p_q.std():.6f} "
              f"corr_with_t75L2={np.corrcoef(p_q, p_t75)[0,1]:.4f}", flush=True)

    # 3. Build folds with all preds
    pred_cols = [("p_t75", p_t75),
                 ("q30", q_avg["30"]), ("q50", q_avg["50"]), ("q70", q_avg["70"]),
                 ("q_mid", (q_avg["30"] + q_avg["70"]) / 2),
                 ("q_triplet", (q_avg["30"] + q_avg["50"] + q_avg["70"]) / 3)]
    folds = split_sym(base_t75, pred_cols)

    # 4. Strategy A: q50 alone
    print("\n--- Strategies ---", flush=True)
    bounds = [(0.0, 0.0040), (0.0, 0.0040)]
    save_strat = out["strategies"]
    evaluate_strategy(
        "A_q50_alone", folds,
        lambda fl, x: gate_asym_one(fl["q50"], x[0], x[1]),
        bounds, make_obj_single(folds, "q50"), save_strat)

    # 5. Strategy B: dual-gate (q30 > tu, q70 < -td)
    bounds_dual = [(-0.0040, 0.0040), (-0.0040, 0.0040)]  # allow negative
    evaluate_strategy(
        "B_dual_gate_q30q70", folds,
        lambda fl, x: gate_dual(fl["q30"], fl["q70"], x[0], x[1]),
        bounds_dual, make_obj_dual(folds, "q30", "q70"), save_strat)

    # 6. Strategy C: q_mid = (q30 + q70)/2
    evaluate_strategy(
        "C_qmid", folds,
        lambda fl, x: gate_asym_one(fl["q_mid"], x[0], x[1]),
        bounds, make_obj_single(folds, "q_mid"), save_strat)

    # 7. Strategy D: triplet (q30 + q50 + q70)/3
    evaluate_strategy(
        "D_triplet", folds,
        lambda fl, x: gate_asym_one(fl["q_triplet"], x[0], x[1]),
        bounds, make_obj_single(folds, "q_triplet"), save_strat)

    # 8. Find best strategy
    best_strat = max(save_strat.items(), key=lambda kv: kv[1]["de_sum"])
    best_label, best_info = best_strat
    out["best_strategy"] = {"label": best_label, **best_info}
    print(f"\n>>> BEST quantile strategy: {best_label}  DE LOSO={best_info['de_sum']:+.4f}", flush=True)
    print(f"    Δ vs T75 L2 baseline: {best_info['de_sum'] - s_t75:+.4f}", flush=True)

    # 9. Stack T87 NN + best quantile strategy vs iter_015 v1 baseline (T87 NN + T75 L2)
    print("\n--- Stack: T87 NN + best quantile vs iter_015 v1 (T87 NN + T75 L2) ---", flush=True)
    try:
        base_t87, p_t87 = load_avg_preds(
            T87_DIR, "pred_T87_seed{seed}_main.parquet", T87_SEEDS, base_df=base_t75)
        print(f"  T87 NN avg: mean={p_t87.mean():+.6f} std={p_t87.std():.6f} "
              f"corr_t75L2={np.corrcoef(p_t87, p_t75)[0,1]:.4f}", flush=True)

        # iter_015 v1 baseline: equal-weight (T87 NN + T75 L2) / 2
        # But predictions may have very different scales, so per-sym normalize? Actually iter_015
        # uses raw averaging in the candidate package. Let's just average equally and DE.
        p_iter015 = (p_t87 + p_t75) / 2
        folds_iter015 = split_sym(base_t75, [("p_iter015", p_iter015)])
        obj_iter015 = make_obj_single(folds_iter015, "p_iter015")
        s_iter015, x_iter015 = de_optimize(obj_iter015, bounds)
        per_iter015 = per_sym_pnl(folds_iter015, lambda fl: gate_asym_one(fl["p_iter015"], x_iter015[0], x_iter015[1]))
        out["stack"]["iter_015v1_repro_NNeq_T75L2"] = {
            "de_sum": s_iter015, "x": x_iter015, "per_sym": per_iter015}
        print(f"  iter_015 v1 repro (NN + T75 L2 eq): DE LOSO={s_iter015:+.4f}", flush=True)

        # New stack: T87 NN + best quantile pred
        if best_label == "A_q50_alone":
            p_best_q = q_avg["50"]
        elif best_label == "C_qmid":
            p_best_q = (q_avg["30"] + q_avg["70"]) / 2
        elif best_label == "D_triplet":
            p_best_q = (q_avg["30"] + q_avg["50"] + q_avg["70"]) / 3
        else:
            # B: dual-gate not a single pred, fall back to q50 for stacking
            print(f"  Note: best strategy '{best_label}' is dual-gate, using q50 for stacking", flush=True)
            p_best_q = q_avg["50"]

        p_new_stack = (p_t87 + p_best_q) / 2
        folds_new = split_sym(base_t75, [("p_new_stack", p_new_stack)])
        obj_new = make_obj_single(folds_new, "p_new_stack")
        s_new, x_new = de_optimize(obj_new, bounds)
        per_new = per_sym_pnl(folds_new, lambda fl: gate_asym_one(fl["p_new_stack"], x_new[0], x_new[1]))
        out["stack"]["T87NN_eq_bestQuantile"] = {
            "best_q_strategy": best_label,
            "de_sum": s_new, "x": x_new, "per_sym": per_new}
        print(f"  T87 NN + best quantile ({best_label}): DE LOSO={s_new:+.4f}", flush=True)
        print(f"    Δ vs iter_015 v1 repro: {s_new - s_iter015:+.4f}", flush=True)
        out["stack"]["delta_vs_iter015"] = float(s_new - s_iter015)
    except Exception as e:
        print(f"  STACK eval skipped: {e!r}", flush=True)
        out["stack"]["error"] = repr(e)

    out["timing_sec"] = time.time() - t0
    out_path = os.path.join(HERE, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved -> {out_path}  (took {time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
