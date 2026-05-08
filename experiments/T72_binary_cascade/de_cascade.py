"""T72 cascade DE 2D threshold search on (T_a, T_b).

For each row in 442k test:
  p_active = avg over 5 seeds of P_A(active|x)
  p_up     = avg over 5 seeds of P_B(up|x)        (only meaningful when active)
  pred = 1 (flat)        if p_active < T_a
  else  pred = 2 (up)    if p_up > T_b
        pred = 0 (down)  otherwise

Objective: maximize SUM-of-per-sym cum_pnl (LOSO-equivalent).
Also report single-set 442k cum_pnl.

Output: results.json
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
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

T53_DIR = os.path.join(ROOT, "experiments", "T53_r34_stage3")
sys.path.insert(0, T53_DIR)
from de_thresh import vectorized_pnl  # noqa: E402

SYMS = (0, 1, 2, 3, 4)
ITER011_LOSO = 25.94
ITER012_LOSO = 26.44


def load_avg_stage(stage: str, seeds: list[int]) -> pd.DataFrame:
    base = pd.read_parquet(os.path.join(HERE, f"pred_stage{stage}_seed{seeds[0]}.parquet"))
    col = f"prob_stage{stage}"
    p_sum = base[col].to_numpy(np.float64).copy()
    n = len(base)
    for s in seeds[1:]:
        df_s = pd.read_parquet(os.path.join(HERE, f"pred_stage{stage}_seed{s}.parquet"))
        if len(df_s) != n:
            raise RuntimeError(f"row count mismatch stage={stage} seed={s}: {len(df_s)} vs {n}")
        if not (df_s["sym"].equals(base["sym"]) and df_s["date"].equals(base["date"]) and df_s["t"].equals(base["t"])):
            raise RuntimeError(f"row order mismatch stage={stage} seed={s}")
        p_sum += df_s[col].to_numpy(np.float64)
    p_avg = (p_sum / float(len(seeds))).astype(np.float32)
    out = base.drop(columns=[col]).copy()
    out[col] = p_avg
    return out


def cascade_predict(p_active, p_up, T_a, T_b):
    """Two-stage gating. Returns int8 array with values in {0,1,2}."""
    pred = np.full(p_active.shape[0], 1, dtype=np.int8)
    active_mask = p_active >= T_a
    pred[active_mask & (p_up > T_b)] = 2
    pred[active_mask & (p_up <= T_b)] = 0
    return pred


def cascade_predict_3d(p_active, p_up, T_a, T_b_lo, T_b_hi):
    """3D: (T_a, T_b_lo, T_b_hi). Active rows go FLAT if p_up in [T_b_lo, T_b_hi]."""
    pred = np.full(p_active.shape[0], 1, dtype=np.int8)
    active_mask = p_active >= T_a
    pred[active_mask & (p_up > T_b_hi)] = 2
    pred[active_mask & (p_up < T_b_lo)] = 0
    return pred


def make_obj_loso(folds, p_a_list, p_u_list):
    def f(x):
        Ta, Tb = x
        s = 0.0
        for fk, pa, pu in zip(folds, p_a_list, p_u_list):
            pred = cascade_predict(pa, pu, Ta, Tb)
            s += vectorized_pnl(pred, fk["label"], fk["mp_t"], fk["mp_th"]).sum()
        return -float(s)
    return f


def make_obj_single(p_active, p_up, label, mp_t, mp_th):
    def f(x):
        Ta, Tb = x
        pred = cascade_predict(p_active, p_up, Ta, Tb)
        return -float(vectorized_pnl(pred, label, mp_t, mp_th).sum())
    return f


def make_obj_loso_3d(folds, p_a_list, p_u_list):
    def f(x):
        Ta, Tb_lo, Tb_hi = x
        if Tb_lo > Tb_hi:
            return 1e9  # invalid: invert -> reject
        s = 0.0
        for fk, pa, pu in zip(folds, p_a_list, p_u_list):
            pred = cascade_predict_3d(pa, pu, Ta, Tb_lo, Tb_hi)
            s += vectorized_pnl(pred, fk["label"], fk["mp_t"], fk["mp_th"]).sum()
        return -float(s)
    return f


def de_search(objective_fn, bounds, seeds=(0, 1, 2, 7, 42)):
    runs = []
    for sd in seeds:
        t0 = time.time()
        result = differential_evolution(
            objective_fn, bounds=bounds, seed=sd, maxiter=80, popsize=24,
            polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            updating="deferred", workers=1, init="sobol",
        )
        Ta, Tb = result.x
        runs.append({
            "seed": int(sd),
            "T_a": float(Ta), "T_b": float(Tb),
            "obj_val": float(-result.fun),
            "nfev": int(result.nfev),
            "elapsed_sec": float(time.time() - t0),
        })
    runs.sort(key=lambda r: r["obj_val"], reverse=True)
    return runs


def de_search_3d(objective_fn, bounds, seeds=(0, 1, 2, 7, 42)):
    runs = []
    for sd in seeds:
        t0 = time.time()
        result = differential_evolution(
            objective_fn, bounds=bounds, seed=sd, maxiter=100, popsize=30,
            polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            updating="deferred", workers=1, init="sobol",
        )
        Ta, Tb_lo, Tb_hi = result.x
        runs.append({
            "seed": int(sd),
            "T_a": float(Ta), "T_b_lo": float(Tb_lo), "T_b_hi": float(Tb_hi),
            "obj_val": float(-result.fun),
            "nfev": int(result.nfev),
            "elapsed_sec": float(time.time() - t0),
        })
    runs.sort(key=lambda r: r["obj_val"], reverse=True)
    return runs


def per_sym_pnl(folds, p_a_list, p_u_list, Ta, Tb):
    per = []
    n_active = []
    n_up = []
    n_down = []
    for fk, pa, pu in zip(folds, p_a_list, p_u_list):
        pred = cascade_predict(pa, pu, Ta, Tb)
        per.append(float(vectorized_pnl(pred, fk["label"], fk["mp_t"], fk["mp_th"]).sum()))
        n_active.append(int((pred != 1).sum()))
        n_up.append(int((pred == 2).sum()))
        n_down.append(int((pred == 0).sum()))
    return per, n_active, n_up, n_down


def split_by_sym_with_probs(df, p_active, p_up):
    folds = []
    p_a_list = []
    p_u_list = []
    for k in SYMS:
        m = (df["sym"] == k).to_numpy()
        folds.append({
            "sym": int(k),
            "label": df.loc[m, "true_label"].to_numpy(np.int64),
            "mp_t": df.loc[m, "midprice_t"].to_numpy(np.float64),
            "mp_th": df.loc[m, "midprice_th"].to_numpy(np.float64),
            "n": int(m.sum()),
        })
        p_a_list.append(p_active[m])
        p_u_list.append(p_up[m])
    return folds, p_a_list, p_u_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,7,13,42,100")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")]
    print(f"=== T72 cascade DE 2D thresh seeds={seeds} ===", flush=True)

    t0 = time.time()
    df_a = load_avg_stage("A", seeds)
    df_b = load_avg_stage("B", seeds)
    print(f"  loaded A:{len(df_a):,} B:{len(df_b):,} rows in {time.time()-t0:.1f}s", flush=True)

    if not (df_a["sym"].equals(df_b["sym"]) and df_a["date"].equals(df_b["date"])
            and df_a["t"].equals(df_b["t"])):
        raise RuntimeError("Stage A and B parquet row orders differ")
    df = df_a.copy()
    df["prob_stageB"] = df_b["prob_stageB"].to_numpy()

    p_active_all = df["prob_stageA"].to_numpy(np.float32)
    p_up_all = df["prob_stageB"].to_numpy(np.float32)
    label_all = df["true_label"].to_numpy(np.int64)
    mp_t_all = df["midprice_t"].to_numpy(np.float64)
    mp_th_all = df["midprice_th"].to_numpy(np.float64)

    print(f"  p_active stats: mean={p_active_all.mean():.4f} median={np.median(p_active_all):.4f} "
          f"min={p_active_all.min():.4f} max={p_active_all.max():.4f}", flush=True)
    print(f"  p_up     stats: mean={p_up_all.mean():.4f} median={np.median(p_up_all):.4f} "
          f"min={p_up_all.min():.4f} max={p_up_all.max():.4f}", flush=True)

    folds, p_a_list, p_u_list = split_by_sym_with_probs(df, p_active_all, p_up_all)
    for fk in folds:
        print(f"  sym={fk['sym']}: n={fk['n']:,}", flush=True)

    # Coarse argmax-equivalent: T_a=0.5, T_b=0.5
    raw_pred = cascade_predict(p_active_all, p_up_all, 0.5, 0.5)
    raw_total = float(vectorized_pnl(raw_pred, label_all, mp_t_all, mp_th_all).sum())
    raw_per_sym, raw_na, raw_nu, raw_nd = per_sym_pnl(folds, p_a_list, p_u_list, 0.5, 0.5)
    print(f"\n  RAW T_a=0.5 T_b=0.5: total={raw_total:+.4f} "
          f"per_sym={[round(x,3) for x in raw_per_sym]} sum={sum(raw_per_sym):+.4f}", flush=True)
    print(f"    n_active={raw_na} n_up={raw_nu} n_down={raw_nd}", flush=True)

    # DE 2D search
    bounds = [(0.30, 0.75), (0.30, 0.75)]

    print(f"\n[1/2] DE single-set ...", flush=True)
    obj_single = make_obj_single(p_active_all, p_up_all, label_all, mp_t_all, mp_th_all)
    de_single = de_search(obj_single, bounds)
    best_s = de_single[0]
    Ta, Tb = best_s["T_a"], best_s["T_b"]
    pred_de_single = cascade_predict(p_active_all, p_up_all, Ta, Tb)
    de_single_total = float(vectorized_pnl(pred_de_single, label_all, mp_t_all, mp_th_all).sum())
    de_single_per_sym, de_single_na, de_single_nu, de_single_nd = per_sym_pnl(
        folds, p_a_list, p_u_list, Ta, Tb)
    print(f"  best DE-single: T_a={Ta:.4f} T_b={Tb:.4f}", flush=True)
    print(f"    total={de_single_total:+.4f}  per_sym={[round(x,3) for x in de_single_per_sym]} "
          f"sum={sum(de_single_per_sym):+.4f}", flush=True)
    print(f"    n_active={de_single_na} n_up={de_single_nu} n_down={de_single_nd}", flush=True)

    print(f"\n[2/3] DE loso-equiv (sum-of-per-sym) ...", flush=True)
    obj_loso = make_obj_loso(folds, p_a_list, p_u_list)
    de_loso = de_search(obj_loso, bounds)
    best_l = de_loso[0]
    Ta2, Tb2 = best_l["T_a"], best_l["T_b"]
    de_loso_per_sym, de_loso_na, de_loso_nu, de_loso_nd = per_sym_pnl(
        folds, p_a_list, p_u_list, Ta2, Tb2)
    de_loso_sum = float(sum(de_loso_per_sym))
    pred_de_loso = cascade_predict(p_active_all, p_up_all, Ta2, Tb2)
    de_loso_total = float(vectorized_pnl(pred_de_loso, label_all, mp_t_all, mp_th_all).sum())
    print(f"  best DE-loso: T_a={Ta2:.4f} T_b={Tb2:.4f}", flush=True)
    print(f"    sum_per_sym={de_loso_sum:+.4f}  per_sym={[round(x,3) for x in de_loso_per_sym]} "
          f"single_total={de_loso_total:+.4f}", flush=True)
    print(f"    n_active={de_loso_na} n_up={de_loso_nu} n_down={de_loso_nd}", flush=True)

    # 3D DE: (T_a, T_b_lo, T_b_hi) — active rows with p_up in [T_b_lo, T_b_hi] go FLAT
    print(f"\n[3/3] DE 3D loso-equiv with FLAT-on-uncertain-direction ...", flush=True)
    bounds_3d = [(0.30, 0.75), (0.20, 0.55), (0.45, 0.80)]
    obj_loso_3d = make_obj_loso_3d(folds, p_a_list, p_u_list)
    de_loso_3d = de_search_3d(obj_loso_3d, bounds_3d)
    best_3d = de_loso_3d[0]
    Ta3, Tlo3, Thi3 = best_3d["T_a"], best_3d["T_b_lo"], best_3d["T_b_hi"]

    # per-sym for 3D
    de_3d_per_sym = []
    de_3d_na, de_3d_nu, de_3d_nd = [], [], []
    for fk, pa, pu in zip(folds, p_a_list, p_u_list):
        pred = cascade_predict_3d(pa, pu, Ta3, Tlo3, Thi3)
        de_3d_per_sym.append(float(vectorized_pnl(pred, fk["label"], fk["mp_t"], fk["mp_th"]).sum()))
        de_3d_na.append(int((pred != 1).sum()))
        de_3d_nu.append(int((pred == 2).sum()))
        de_3d_nd.append(int((pred == 0).sum()))
    de_3d_sum = float(sum(de_3d_per_sym))
    pred_3d = cascade_predict_3d(p_active_all, p_up_all, Ta3, Tlo3, Thi3)
    de_3d_total = float(vectorized_pnl(pred_3d, label_all, mp_t_all, mp_th_all).sum())
    print(f"  best DE-3D: T_a={Ta3:.4f} T_b_lo={Tlo3:.4f} T_b_hi={Thi3:.4f}", flush=True)
    print(f"    sum_per_sym={de_3d_sum:+.4f}  per_sym={[round(x,3) for x in de_3d_per_sym]} "
          f"single_total={de_3d_total:+.4f}", flush=True)
    print(f"    n_active={de_3d_na} n_up={de_3d_nu} n_down={de_3d_nd}", flush=True)

    out = {
        "task": "T72 binary cascade DE 2D thresh",
        "experiment": "T72_binary_cascade",
        "seeds": seeds,
        "n_test": int(len(df)),
        "raw_05": {
            "total_cum_pnl": raw_total,
            "per_sym": raw_per_sym,
            "loso_equiv_sum": float(sum(raw_per_sym)),
            "n_active_per_sym": raw_na,
        },
        "de_single_set": {
            "best": best_s,
            "all_runs": de_single,
            "total_cum_pnl_at_best": de_single_total,
            "per_sym_at_best": de_single_per_sym,
            "n_active_per_sym": de_single_na,
            "n_up_per_sym": de_single_nu,
            "n_down_per_sym": de_single_nd,
            "loso_equiv_sum_at_best": float(sum(de_single_per_sym)),
        },
        "de_loso_equiv": {
            "best": best_l,
            "all_runs": de_loso,
            "per_sym_at_best": de_loso_per_sym,
            "n_active_per_sym": de_loso_na,
            "n_up_per_sym": de_loso_nu,
            "n_down_per_sym": de_loso_nd,
            "sum_per_sym": de_loso_sum,
            "single_set_total_at_best": de_loso_total,
        },
        "de_loso_3d": {
            "best": best_3d,
            "all_runs": de_loso_3d,
            "per_sym_at_best": de_3d_per_sym,
            "n_active_per_sym": de_3d_na,
            "n_up_per_sym": de_3d_nu,
            "n_down_per_sym": de_3d_nd,
            "sum_per_sym": de_3d_sum,
            "single_set_total_at_best": de_3d_total,
        },
        "compare_iter011": {
            "iter011_loso": ITER011_LOSO,
            "t72_de_loso_vs_iter011": de_loso_sum - ITER011_LOSO,
            "t72_de_3d_vs_iter011": de_3d_sum - ITER011_LOSO,
        },
        "compare_iter012": {
            "iter012_loso": ITER012_LOSO,
            "t72_de_loso_vs_iter012": de_loso_sum - ITER012_LOSO,
            "t72_de_3d_vs_iter012": de_3d_sum - ITER012_LOSO,
        },
        "best_loso_equiv": max(de_loso_sum, de_3d_sum),
    }
    out_path = os.path.join(HERE, "results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote -> {out_path}", flush=True)

    print(f"\n{'='*78}\n=== SUMMARY T72 ===\n{'='*78}", flush=True)
    print(f"RAW T_a=0.5 T_b=0.5: single_total={raw_total:+.4f} loso_equiv={sum(raw_per_sym):+.4f}", flush=True)
    print(f"DE single-set:       single_total={de_single_total:+.4f} loso_equiv={sum(de_single_per_sym):+.4f}", flush=True)
    print(f"DE loso-equiv 2D:    sum_per_sym={de_loso_sum:+.4f} single_total={de_loso_total:+.4f}", flush=True)
    print(f"DE loso-equiv 3D:    sum_per_sym={de_3d_sum:+.4f} single_total={de_3d_total:+.4f}", flush=True)
    print(f"vs iter_011 +{ITER011_LOSO:.2f}: 2D_diff={de_loso_sum - ITER011_LOSO:+.4f} 3D_diff={de_3d_sum - ITER011_LOSO:+.4f}", flush=True)
    print(f"vs iter_012 +{ITER012_LOSO:.2f}: 2D_diff={de_loso_sum - ITER012_LOSO:+.4f} 3D_diff={de_3d_sum - ITER012_LOSO:+.4f}", flush=True)


if __name__ == "__main__":
    main()
