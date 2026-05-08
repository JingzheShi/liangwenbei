"""T134 ensemble eval: T134 NN (direct PnL) + T75 LGB L2 (iter_015 v1 same).

Stack: w_nn=1.0, w_lgb=1.5 (iter_015 v1 ratio).

Compare:
  - iter_015 v1 ref (5x T87 NN + 5x T75 LGB) — DE LOSO +40.13
  - 3x T87 NN  + 5x T75 LGB    (control, same 3 seeds)
  - 3x T134 NN + 5x T75 LGB    (candidate)
  - 5-mix: T134(1,7,42) + T87(13,100) + 5x T75 LGB
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

T87_PRED_DIR = os.path.join(ROOT, "experiments", "T87_spo_dfl")
T75_PRED_DIR = os.path.join(ROOT, "experiments", "T75_regression_dmid")

SYMS = (0, 1, 2, 3, 4)
SEEDS_FULL = (1, 7, 13, 42, 100)
SEEDS_T134 = (1, 7, 42)
FEE = 0.0001
W_NN = 1.0
W_LGB = 1.5
ITER_015_V1_THR_UP = 4.21e-4
ITER_015_V1_THR_DN = 1.86e-4
ITER_015_V1_LOSO = 40.13


def vectorized_pnl(action, mp_t, mp_th):
    side = action.astype(np.float64) - 1.0
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee_pnl) / denom


def gate_asym(p, tu, td):
    a = np.full(len(p), 1, dtype=np.int8)
    a[p > tu] = 2
    a[p < -td] = 0
    return a


def avg_preds(prefix, seeds, dir_, suffix=""):
    base = pd.read_parquet(os.path.join(dir_, f"{prefix}_seed{seeds[0]}{suffix}.parquet"))
    p = np.zeros(len(base), dtype=np.float64)
    for s in seeds:
        df = pd.read_parquet(os.path.join(dir_, f"{prefix}_seed{s}{suffix}.parquet"))
        p += df["pred_dmid_norm"].to_numpy(np.float64)
    return base, p / len(seeds)


def split_sym(df, pred_arr):
    folds = []
    for k in SYMS:
        m = (df["sym"].to_numpy() == k)
        folds.append({
            "sym": k,
            "pred": pred_arr[m].astype(np.float64),
            "mp_t": df["midprice_t"].to_numpy(np.float64)[m],
            "mp_th": df["midprice_th"].to_numpy(np.float64)[m],
        })
    return folds


def make_obj(folds):
    def f(x):
        s = 0.0
        for fold in folds:
            a = gate_asym(fold["pred"], x[0], x[1])
            s += vectorized_pnl(a, fold["mp_t"], fold["mp_th"]).sum()
        return -float(s)
    return f


def de_loso(folds, bounds=((0.0, 0.005), (0.0, 0.005)), seeds=(0, 1, 2, 7, 42)):
    obj = make_obj(folds)
    runs = []
    for sd in seeds:
        r = differential_evolution(obj, bounds=bounds, seed=sd, maxiter=80, popsize=24,
                                   polish=True, tol=1e-7, init="sobol")
        runs.append((float(-r.fun), float(r.x[0]), float(r.x[1])))
    runs.sort(key=lambda x: x[0], reverse=True)
    return runs[0]


def eval_at_thr(folds, tu, td):
    s = 0.0
    per = []
    for fold in folds:
        a = gate_asym(fold["pred"], tu, td)
        v = float(vectorized_pnl(a, fold["mp_t"], fold["mp_th"]).sum())
        per.append(v)
        s += v
    return s, per


def stacked_pred(nn_pred, lgb_pred):
    return (W_NN * nn_pred + W_LGB * lgb_pred) / (W_NN + W_LGB)


def evaluate_combo(label, base_df, nn_pred, lgb_pred):
    pred = stacked_pred(nn_pred, lgb_pred)
    folds = split_sym(base_df, pred)

    # At iter_015 v1 conservative thresholds
    s_v1, per_v1 = eval_at_thr(folds, ITER_015_V1_THR_UP, ITER_015_V1_THR_DN)
    # DE LOSO optimum
    s_de, tu_de, td_de = de_loso(folds)
    # per-sym at DE optimum
    _, per_de = eval_at_thr(folds, tu_de, td_de)
    cross_corr_nn_lgb = float(np.corrcoef(nn_pred, lgb_pred)[0, 1])

    print(f"\n=== {label} ===", flush=True)
    print(f"  NN-vs-LGB corr = {cross_corr_nn_lgb:.4f}", flush=True)
    print(f"  At iter_015_v1 thr ({ITER_015_V1_THR_UP:.4e}, {ITER_015_V1_THR_DN:.4e}):"
          f"  cum_pnl={s_v1:+.4f}  per_sym=[{', '.join(f'{x:+.3f}' for x in per_v1)}]",
          flush=True)
    print(f"  DE LOSO opt: thr_up={tu_de:.4e}  thr_dn={td_de:.4e}  cum_pnl={s_de:+.4f}", flush=True)
    print(f"      per_sym=[{', '.join(f'{x:+.3f}' for x in per_de)}]", flush=True)
    print(f"  vs iter_015_v1 (+{ITER_015_V1_LOSO}): {s_de - ITER_015_V1_LOSO:+.4f}", flush=True)

    return {
        "label": label,
        "cross_corr_nn_lgb": cross_corr_nn_lgb,
        "at_iter_015_v1_thr": {
            "thr_up": ITER_015_V1_THR_UP, "thr_dn": ITER_015_V1_THR_DN,
            "cum_pnl": s_v1, "per_sym": per_v1,
        },
        "de_loso": {
            "thr_up": tu_de, "thr_dn": td_de,
            "cum_pnl": s_de, "per_sym": per_de,
        },
        "delta_de_loso_vs_iter015v1": s_de - ITER_015_V1_LOSO,
    }


def main():
    print(f"=== T134 ensemble eval ({datetime.now(timezone.utc).isoformat()}) ===", flush=True)

    # Load T134 NN preds (3 seeds: 1, 7, 42, tag=main)
    t134_base, t134_pred = avg_preds("pred_T134", SEEDS_T134, HERE, suffix="_main")
    print(f"  T134 NN preds (3-seed): n={len(t134_pred)} mean={t134_pred.mean():+.6e} std={t134_pred.std():.6e}",
          flush=True)

    # Load T87 NN preds (5-seed, for iter_015 v1 ref)
    t87_base, t87_5 = avg_preds("pred_T87", SEEDS_FULL, T87_PRED_DIR, suffix="_main")
    print(f"  T87 NN preds (5-seed):  n={len(t87_5)} mean={t87_5.mean():+.6e} std={t87_5.std():.6e}",
          flush=True)
    # 3-seed T87 control (same seeds as T134)
    _, t87_3 = avg_preds("pred_T87", SEEDS_T134, T87_PRED_DIR, suffix="_main")
    # 2-seed T87 (13, 100) for mix
    _, t87_2 = avg_preds("pred_T87", (13, 100), T87_PRED_DIR, suffix="_main")

    # Load T75 LGB preds (5-seed)
    t75_base, t75_5 = avg_preds("pred_T75", SEEDS_FULL, T75_PRED_DIR)
    print(f"  T75 LGB preds (5-seed): n={len(t75_5)} mean={t75_5.mean():+.6e} std={t75_5.std():.6e}",
          flush=True)

    # Sanity row order
    for k in ("sym", "date", "t"):
        np.testing.assert_array_equal(t134_base[k].to_numpy(), t75_base[k].to_numpy(),
                                      err_msg=f"row order mismatch on {k} (T134 vs T75)")
        np.testing.assert_array_equal(t134_base[k].to_numpy(), t87_base[k].to_numpy(),
                                      err_msg=f"row order mismatch on {k} (T134 vs T87)")

    # NN-vs-NN cross-corr
    print(f"\n  T134-vs-T87(5)   : {float(np.corrcoef(t134_pred, t87_5)[0,1]):.4f}", flush=True)
    print(f"  T134-vs-T87(3)   : {float(np.corrcoef(t134_pred, t87_3)[0,1]):.4f}", flush=True)
    print(f"  T87(5)-vs-T87(3) : {float(np.corrcoef(t87_5, t87_3)[0,1]):.4f}", flush=True)

    results = []

    # Combos
    results.append(evaluate_combo("ref: 5xT87 NN + 5xT75 LGB (iter_015 v1)",
                                  t75_base, t87_5, t75_5))
    results.append(evaluate_combo("ctrl: 3xT87 NN + 5xT75 LGB (same seeds as T134)",
                                  t75_base, t87_3, t75_5))
    results.append(evaluate_combo("cand: 3xT134 NN + 5xT75 LGB (direct-PnL fine-tune)",
                                  t75_base, t134_pred, t75_5))
    # 5-mix: T134(1,7,42) blended with T87(13,100), simple weighted avg
    nn_5mix = (3.0 * t134_pred + 2.0 * t87_2) / 5.0
    results.append(evaluate_combo("mix: 3xT134+2xT87(13,100) NN + 5xT75 LGB",
                                  t75_base, nn_5mix, t75_5))
    # 50/50 ensemble of T134(3) and T87(5) at NN level
    nn_blend = 0.5 * t134_pred + 0.5 * t87_5
    results.append(evaluate_combo("blend50: 0.5*T134(3) + 0.5*T87(5) NN + 5xT75 LGB",
                                  t75_base, nn_blend, t75_5))

    best = max(results, key=lambda r: r["de_loso"]["cum_pnl"])
    print(f"\n{'='*78}", flush=True)
    print(f"BEST DE LOSO: {best['label']}  → {best['de_loso']['cum_pnl']:+.4f}", flush=True)
    print(f"  vs iter_015 v1 (+{ITER_015_V1_LOSO}): {best['delta_de_loso_vs_iter015v1']:+.4f}", flush=True)
    print('=' * 78, flush=True)

    out_path = os.path.join(HERE, "ev_gate_ensemble_results.json")
    with open(out_path, "w") as f:
        json.dump({"results": results, "best": best,
                   "iter_015_v1_ref": ITER_015_V1_LOSO,
                   "timestamp": datetime.now(timezone.utc).isoformat()},
                  f, indent=2)
    print(f"wrote -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
