"""3-seed DE asym thresh eval for R1 date-decay trick (LGB L2 baseline).

Compares baseline vs (d=0.6,1.4) / (d=0.6,2.4) / (d=1.0,1.0).
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
PREDS = os.path.join(HERE, "preds")
SEEDS = (1, 7, 42)
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001


def vectorized_pnl(action, mp_t, mp_th):
    side = action.astype(np.float64) - 1.0
    abs_side = np.abs(side)
    diff = mp_th - mp_t
    fee_pnl = FEE * abs_side * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t + 1.0
    return (side * diff - fee_pnl) / denom


def gate_asym(p, tu, td):
    a = np.full(len(p), 1, dtype=np.int8)
    a[p > tu] = 2
    a[p < -td] = 0
    return a


def split_sym(df):
    return [{"sym": k,
             "pred": df[df["sym"] == k]["pred_dmid_norm"].to_numpy(np.float64),
             "mp_t": df[df["sym"] == k]["midprice_t"].to_numpy(np.float64),
             "mp_th": df[df["sym"] == k]["midprice_th"].to_numpy(np.float64)}
            for k in SYMS]


def make_obj(folds):
    def f(x):
        s = 0.0
        for fold in folds:
            a = gate_asym(fold["pred"], x[0], x[1])
            s += vectorized_pnl(a, fold["mp_t"], fold["mp_th"]).sum()
        return -float(s)
    return f


def de(obj, bounds=[(0.0, 0.004), (0.0, 0.004)], seeds=(0, 1, 2, 7, 42)):
    runs = []
    for sd in seeds:
        r = differential_evolution(obj, bounds=bounds, seed=sd, maxiter=80, popsize=24,
                                   polish=True, tol=1e-7, init="sobol")
        runs.append((float(-r.fun), float(r.x[0]), float(r.x[1])))
    runs.sort(key=lambda x: x[0], reverse=True)
    return runs[0]


def avg_preds(prefix):
    base = pd.read_parquet(os.path.join(PREDS, f"{prefix}_seed{SEEDS[0]}.parquet"))
    p = np.zeros(len(base), dtype=np.float64)
    for s in SEEDS:
        df = pd.read_parquet(os.path.join(PREDS, f"{prefix}_seed{s}.parquet"))
        p += df["pred_dmid_norm"].to_numpy(np.float64)
    return base, p / len(SEEDS)


def evaluate(p, base, label):
    df = base.copy()
    df["pred_dmid_norm"] = p.astype(np.float32)
    folds = split_sym(df)
    obj = make_obj(folds)
    s, tu, td = de(obj)
    per = []
    for fold in folds:
        a = gate_asym(fold["pred"], tu, td)
        per.append(float(vectorized_pnl(a, fold["mp_t"], fold["mp_th"]).sum()))
    print(f"  [{label:<35s}] DE={s:+.4f} thr_up={tu:.3e} thr_dn={td:.3e} per_sym={[round(x,2) for x in per]}", flush=True)
    return s


def main():
    print("=== R1 date-decay 3-seed DE eval ===", flush=True)
    configs = [
        ("baseline (no decay)",   "pred_R1_regression_l2_base"),
        ("d=(0.6, 1.4)",          "pred_R1_regression_l2_d0.6_1.4"),
        ("d=(0.6, 2.4)",          "pred_R1_regression_l2_d0.6_2.4"),
        ("d=(1.0, 1.0)",          "pred_R1_regression_l2_d1_1"),
    ]
    results = {}
    for name, prefix in configs:
        base, p = avg_preds(prefix)
        s = evaluate(p, base, name)
        results[name] = s
    print("\n=== Summary ===")
    base = results["baseline (no decay)"]
    for k, v in results.items():
        print(f"  {k:<28s}: {v:+.4f}  (Δ vs baseline = {v - base:+.4f})")
    with open(os.path.join(HERE, "trick_de_results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
