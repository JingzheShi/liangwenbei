"""LGB Huber 5-seed ensemble eval — does it beat T75 LGB RMSE?

If yes, also test ensembles with T87/T89/T95.
"""
from __future__ import annotations
import json, os, time
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
T75 = os.path.join(ROOT, "experiments", "T75_regression_dmid")
T87 = os.path.join(ROOT, "experiments", "T87_spo_dfl")
T89 = os.path.join(ROOT, "experiments", "T89_catboost_regression")
T95 = os.path.join(ROOT, "experiments", "T95_cnn_rnn_deeplob")

SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001
SEEDS = (1, 7, 13, 42, 100)
ITER014 = 38.281
ITER015_T87T89 = 41.014
ITER015_T87T89T95 = 42.34


def vectorized_pnl(action, mp_t, mp_th):
    side = action.astype(np.float64) - 1.0
    abs_side = np.abs(side)
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * abs_side * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee_pnl) / denom


def gate_asym(p, tu, td):
    a = np.full(len(p), 1, dtype=np.int8)
    a[p > tu] = 2
    a[p < -td] = 0
    return a


def split_sym(df):
    return [{"sym": int(k),
             "pred": df[df["sym"] == k]["pred_dmid_norm"].to_numpy(np.float64),
             "mp_t": df[df["sym"] == k]["midprice_t"].to_numpy(np.float64),
             "mp_th": df[df["sym"] == k]["midprice_th"].to_numpy(np.float64)}
            for k in SYMS]


def make_obj(folds):
    pred = [f["pred"] for f in folds]
    mp_t = [f["mp_t"] for f in folds]
    mp_th = [f["mp_th"] for f in folds]
    def f(x):
        s = 0.0
        for i in range(len(folds)):
            a = gate_asym(pred[i], x[0], x[1])
            s += vectorized_pnl(a, mp_t[i], mp_th[i]).sum()
        return -float(s)
    return f


def de(obj, bounds, seeds=(0, 1, 2, 7, 42)):
    runs = []
    for sd in seeds:
        r = differential_evolution(obj, bounds=bounds, seed=sd, maxiter=80, popsize=24,
                                   polish=True, tol=1e-7, init="sobol")
        runs.append((float(-r.fun), float(r.x[0]), float(r.x[1])))
    runs.sort(key=lambda x: x[0], reverse=True)
    return runs[0]


def avg_preds(dir_, prefix, suffix=""):
    base = pd.read_parquet(os.path.join(dir_, f"{prefix}{SEEDS[0]}{suffix}.parquet"))
    p = np.zeros(len(base), dtype=np.float64)
    for s in SEEDS:
        df = pd.read_parquet(os.path.join(dir_, f"{prefix}{s}{suffix}.parquet"))
        if not (df["sym"].equals(base["sym"]) and df["t"].equals(base["t"])):
            raise RuntimeError(f"row mismatch {prefix}{s}")
        p += df["pred_dmid_norm"].to_numpy(np.float64)
    return base, p / len(SEEDS)


def evaluate(p, base, label):
    df = base.copy()
    df["pred_dmid_norm"] = p.astype(np.float32)
    folds = split_sym(df)
    obj = make_obj(folds)
    s, tu, td = de(obj, [(0.0, 0.0040), (0.0, 0.0040)])
    per = []
    for f in folds:
        a = gate_asym(f["pred"], tu, td)
        per.append(float(vectorized_pnl(a, f["mp_t"], f["mp_th"]).sum()))
    print(f"  [{label:<55s}] DE={s:+.4f} thr_up={tu:.4e} thr_dn={td:.4e}", flush=True)
    return {"label": label, "thr_up": tu, "thr_dn": td, "de_sum": s, "per_sym": per}


def main():
    print("=== LGB Huber 5-seed ensemble eval ===", flush=True)
    t0 = time.time()
    base_h, p_huber = avg_preds(HERE, "pred_T99_huber_a0.001_seed")
    base_t75, p_t75 = avg_preds(T75, "pred_T75_seed")
    base_t87, p_t87 = avg_preds(T87, "pred_T87_seed", suffix="_main")
    base_t89, p_t89 = avg_preds(T89, "pred_T89_seed")
    base_t95, p_t95 = avg_preds(T95, "pred_T95_gru_w100_C_seed")

    for nm, b in [("T75", base_t75), ("T87", base_t87), ("T89", base_t89), ("T95", base_t95)]:
        if not (base_h["sym"].equals(b["sym"]) and base_h["t"].equals(b["t"])):
            raise RuntimeError(f"row mismatch huber vs {nm}")

    # Cross-corr
    names = ["LGB_Huber", "T75_RMSE", "T87_SPO+", "T89_CB", "T95_GRU"]
    preds = [p_huber, p_t75, p_t87, p_t89, p_t95]
    print("\nCross-correlations:", flush=True)
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            c = float(np.corrcoef(preds[i], preds[j])[0, 1])
            print(f"  corr({names[i]:<10s}, {names[j]:<10s}) = {c:.4f}", flush=True)

    print("\n=== Singles ===", flush=True)
    results = []
    for n, p in zip(names, preds):
        results.append(evaluate(p, base_h, f"{n}-only"))

    print("\n=== 2-way LGB_Huber + others ===", flush=True)
    # Replace T75 with Huber in current strong stacks
    results.append(evaluate((1.0*p_t87 + 1.5*p_huber) / 2.5, base_h, "T87+Huber w=1:1.5 (Huber replaces T75)"))
    results.append(evaluate((1.0*p_t87 + 1.0*p_huber) / 2.0, base_h, "T87+Huber w=1:1"))
    for w_huber in [0.5, 0.7, 1.0]:
        results.append(evaluate((1.0*p_t87 + w_huber*p_t89 + (1.5-w_huber*0.5)*p_huber) / (1.0 + w_huber + 1.5 - w_huber*0.5), base_h, f"T87+T89+Huber misc"))
    # Direct compare with T87+T89 (no LGB)
    results.append(evaluate((1.0*p_t87 + 1.5*p_t89) / 2.5, base_h, "T87+T89 w=1:1.5 (baseline)"))
    # Add Huber to T87+T89
    for w_huber in [0.3, 0.5, 0.7, 1.0]:
        results.append(evaluate((1.0*p_t87 + 1.5*p_t89 + w_huber*p_huber) / (2.5+w_huber), base_h, f"T87+T89+Huber w=1:1.5:{w_huber}"))
    # Add T95 too (4-way)
    for w_huber, w_t95 in [(0.3, 0.5), (0.5, 0.5), (0.5, 0.7), (0.7, 0.5), (0.7, 0.7)]:
        wsum = 1.0 + 1.5 + w_huber + w_t95
        p = (1.0*p_t87 + 1.5*p_t89 + w_huber*p_huber + w_t95*p_t95) / wsum
        results.append(evaluate(p, base_h, f"T87+T89+Huber+T95 w=1:1.5:{w_huber}:{w_t95}"))

    best = max(results, key=lambda r: r["de_sum"])
    print(f"\n{'='*100}", flush=True)
    print(f"BEST: {best['label']}  DE={best['de_sum']:+.4f}", flush=True)
    print(f"  vs iter_014 (+{ITER014}): {best['de_sum']-ITER014:+.4f}", flush=True)
    print(f"  vs T87+T89 (+{ITER015_T87T89}): {best['de_sum']-ITER015_T87T89:+.4f}", flush=True)
    print(f"  vs T87+T89+T95 (+{ITER015_T87T89T95}): {best['de_sum']-ITER015_T87T89T95:+.4f}", flush=True)
    print(f"  per_sym = {[round(x,2) for x in best['per_sym']]}", flush=True)
    print(f"{'='*100}", flush=True)
    print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)
    with open(os.path.join(HERE, "ev_lgb_huber_ensemble.json"), "w") as f:
        json.dump({"results": results, "best": best}, f, indent=2)
    print(f"wrote -> ev_lgb_huber_ensemble.json", flush=True)


if __name__ == "__main__":
    main()
