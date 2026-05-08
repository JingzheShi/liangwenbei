"""Ensemble eval: T87 NN (5 seed) + T123 LGB (5 seed) — same as iter_015 v1 stacking.

w_nn=1.0, w_lgb=1.5, h=60.

Compare:
  - iter_015 v1 ref: 5 NN + 5 T75 LGB at this ratio
  - iter_017 v1: 5 NN + 5 T123 LGB (HYD+monotone+date-decay)
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

T87_PRED_DIR = os.path.join(ROOT, "experiments", "T87_spo_dfl")
T123_PRED_DIR = HERE
T75_PRED_DIR = os.path.join(ROOT, "experiments", "T75_regression_dmid")

SYMS = (0, 1, 2, 3, 4)
SEEDS = (1, 7, 13, 42, 100)
FEE = 0.0001
W_NN = 1.0
W_LGB = 1.5

# iter_015 v1 conservative thresholds (DO NOT retune per task)
ITER_015_V1_THR_UP = 4.21e-4
ITER_015_V1_THR_DN = 1.86e-4


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


def make_obj(folds):
    def f(x):
        s = 0.0
        for fold in folds:
            a = gate_asym(fold["pred"], x[0], x[1])
            s += vectorized_pnl(a, fold["mp_t"], fold["mp_th"]).sum()
        return -float(s)
    return f


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


def de_loso(folds, bounds=[(0.0, 0.005), (0.0, 0.005)], seeds=(0, 1, 2, 7, 42)):
    """LOSO DE: average DE optimum on the 5 sym splits combined (sum)."""
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


def main():
    print("=== T123 iter_017 v1 ensemble eval ===", flush=True)
    # Load T87 NN preds (5-seed ensemble) — pred_T87_seed{S}_main.parquet
    nn_base, nn_pred = avg_preds("pred_T87", SEEDS, T87_PRED_DIR, suffix="_main")
    print(f"  T87 NN preds: shape={nn_pred.shape} mean={nn_pred.mean():+.6e} std={nn_pred.std():.6e}", flush=True)

    # Load T123 LGB preds (HYD+monotone+date-decay)
    t123_base, t123_pred = avg_preds("pred_T123", SEEDS, T123_PRED_DIR)
    print(f"  T123 LGB preds: shape={t123_pred.shape} mean={t123_pred.mean():+.6e} std={t123_pred.std():.6e}",
          flush=True)

    # Load T75 LGB preds (iter_015 v1 reference)
    t75_base, t75_pred = avg_preds("pred_T75", SEEDS, T75_PRED_DIR)
    print(f"  T75 LGB preds: shape={t75_pred.shape} mean={t75_pred.mean():+.6e} std={t75_pred.std():.6e}",
          flush=True)

    # Sanity: same row order across all bases
    for k in ("sym", "date", "t"):
        np.testing.assert_array_equal(nn_base[k].to_numpy(), t123_base[k].to_numpy(),
                                      err_msg=f"order mismatch on {k}")

    # iter_017 v1 stack
    pred_iter017 = stacked_pred(nn_pred, t123_pred)
    folds_iter017 = split_sym(t123_base, pred_iter017)

    # iter_015 v1 stack (reference)
    pred_iter015v1 = stacked_pred(nn_pred, t75_pred)
    folds_iter015v1 = split_sym(t75_base, pred_iter015v1)

    # === Eval at iter_015 v1 conservative thresholds ===
    print(f"\n=== At iter_015_v1 thresholds ({ITER_015_V1_THR_UP}, {ITER_015_V1_THR_DN}) ===", flush=True)
    s17_v1thr, per17_v1thr = eval_at_thr(folds_iter017, ITER_015_V1_THR_UP, ITER_015_V1_THR_DN)
    s15_v1thr, per15_v1thr = eval_at_thr(folds_iter015v1, ITER_015_V1_THR_UP, ITER_015_V1_THR_DN)
    print(f"  iter_015_v1 (T87+T75 LGB):  cum_pnl={s15_v1thr:+.4f}  per_sym={[round(x,2) for x in per15_v1thr]}",
          flush=True)
    print(f"  iter_017_v1 (T87+T123 LGB): cum_pnl={s17_v1thr:+.4f}  per_sym={[round(x,2) for x in per17_v1thr]}",
          flush=True)
    print(f"  delta vs iter_015_v1 = {s17_v1thr - s15_v1thr:+.4f}", flush=True)

    # === DE-LOSO optimum (informational; we'll NOT use this for thresholds) ===
    print("\n=== DE-LOSO optimum (informational, NOT used for final thresholds) ===", flush=True)
    s17_de, t17_up, t17_dn = de_loso(folds_iter017)
    print(f"  iter_017_v1 DE optimum: cum_pnl={s17_de:+.4f} thr_up={t17_up:.3e} thr_dn={t17_dn:.3e}", flush=True)
    s15_de, t15_up, t15_dn = de_loso(folds_iter015v1)
    print(f"  iter_015_v1 DE optimum: cum_pnl={s15_de:+.4f} thr_up={t15_up:.3e} thr_dn={t15_dn:.3e}", flush=True)
    s17_at_de, _ = eval_at_thr(folds_iter017, t17_up, t17_dn)
    per17_at_de = []
    for fold in folds_iter017:
        a = gate_asym(fold["pred"], t17_up, t17_dn)
        per17_at_de.append(float(vectorized_pnl(a, fold["mp_t"], fold["mp_th"]).sum()))
    print(f"    per_sym at DE opt: {[round(x,2) for x in per17_at_de]}", flush=True)

    out = {
        "task": "T123 iter_017 v1 ensemble eval",
        "stack": "5xT87 NN + 5xT123 LGB (HYD+monotone+date-decay)",
        "weights": {"w_nn": W_NN, "w_lgb": W_LGB},
        "iter_015_v1_thresholds": {"thr_up": ITER_015_V1_THR_UP, "thr_dn": ITER_015_V1_THR_DN},
        "at_iter015v1_thr": {
            "iter_017_v1": {"cum_pnl": s17_v1thr, "per_sym": per17_v1thr},
            "iter_015_v1_ref": {"cum_pnl": s15_v1thr, "per_sym": per15_v1thr},
            "delta": s17_v1thr - s15_v1thr,
        },
        "de_loso_optimum": {
            "iter_017_v1": {"cum_pnl": s17_de, "thr_up": t17_up, "thr_dn": t17_dn,
                            "per_sym": per17_at_de},
            "iter_015_v1_ref": {"cum_pnl": s15_de, "thr_up": t15_up, "thr_dn": t15_dn},
        },
    }
    out_path = os.path.join(HERE, "ensemble_eval.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
