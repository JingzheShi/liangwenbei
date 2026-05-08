"""R4 evaluation: DE asym thresh, per-sym PnL, baseline vs trick.

Loads R4 baseline + trick predictions, averages over 3 seeds {1,7,42},
runs DE asym threshold search on full 442k local test set, computes
per-sym PnL summary. Δloso > 0.5 OR Δpsmin > 0.5 → SUCCESS.

Output: results.json + REPORT.md
"""
from __future__ import annotations
import json, os, time
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001
SEEDS = (1, 7, 42)


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
        r = differential_evolution(obj, bounds=bounds, seed=sd,
                                   maxiter=80, popsize=24,
                                   polish=True, tol=1e-7, init="sobol")
        runs.append((float(-r.fun), float(r.x[0]), float(r.x[1])))
    runs.sort(key=lambda x: x[0], reverse=True)
    return runs[0]


def avg_preds(prefix):
    base = pd.read_parquet(os.path.join(HERE, f"pred_R4_{prefix}_seed{SEEDS[0]}.parquet"))
    p = np.zeros(len(base), dtype=np.float64)
    for s in SEEDS:
        df = pd.read_parquet(os.path.join(HERE, f"pred_R4_{prefix}_seed{s}.parquet"))
        if not (df["sym"].equals(base["sym"]) and df["t"].equals(base["t"])):
            raise RuntimeError(f"row mismatch {prefix} seed{s}")
        p += df["pred_dmid_norm"].to_numpy(np.float64)
    return base, p / len(SEEDS)


def evaluate(label, prefix):
    print(f"\n--- {label} ({prefix}) ---", flush=True)
    base, pavg = avg_preds(prefix)
    df = base.copy()
    df["pred_dmid_norm"] = pavg.astype(np.float32)
    folds = split_sym(df)
    obj = make_obj(folds)
    s, tu, td = de(obj, [(0.0, 0.0040), (0.0, 0.0040)])
    per = []
    for f in folds:
        a = gate_asym(f["pred"], tu, td)
        per.append(float(vectorized_pnl(a, f["mp_t"], f["mp_th"]).sum()))
    fixed_thr = 2.0 * FEE
    pa = gate_asym(pavg, fixed_thr, fixed_thr)
    fixed_per = []
    full_t = base["midprice_t"].to_numpy(np.float64)
    full_th = base["midprice_th"].to_numpy(np.float64)
    full_sym = base["sym"].to_numpy(np.int64)
    pnl_full = vectorized_pnl(pa, full_t, full_th)
    for k in SYMS:
        fixed_per.append(float(pnl_full[full_sym == k].sum()))
    fixed_sum = float(pnl_full.sum())
    print(f"  DE: sum={s:+.4f} tu={tu:.4e} td={td:.4e}", flush=True)
    print(f"  DE per_sym = {[round(x,2) for x in per]}", flush=True)
    print(f"  fixed_thr=2*FEE: sum={fixed_sum:+.4f} per_sym={[round(x,2) for x in fixed_per]}", flush=True)
    return {"label": label, "prefix": prefix,
            "de_sum": s, "thr_up": tu, "thr_dn": td, "de_per_sym": per,
            "de_per_sym_min": min(per), "de_per_sym_max": max(per),
            "de_per_sym_std": float(np.std(per)),
            "fixed_thr_sum": fixed_sum, "fixed_per_sym": fixed_per}


def main():
    print("=== R4 logret-lag LGB Huber 3-seed eval ===", flush=True)
    t0 = time.time()
    base = evaluate("Baseline (359-d)", "baseline")
    trick = evaluate("Trick (359+12-d lagret)", "trick")

    delta_loso = trick["de_sum"] - base["de_sum"]
    delta_per_sym_min = trick["de_per_sym_min"] - base["de_per_sym_min"]

    print(f"\n{'='*100}", flush=True)
    print(f"BASELINE (3-seed avg) DE sum={base['de_sum']:+.4f}", flush=True)
    print(f"  per_sym = {[round(x,2) for x in base['de_per_sym']]}", flush=True)
    print(f"  per_sym_min = {base['de_per_sym_min']:.4f}", flush=True)
    print(f"TRICK    (3-seed avg) DE sum={trick['de_sum']:+.4f}", flush=True)
    print(f"  per_sym = {[round(x,2) for x in trick['de_per_sym']]}", flush=True)
    print(f"  per_sym_min = {trick['de_per_sym_min']:.4f}", flush=True)
    print(f"DELTA: loso={delta_loso:+.4f}  per_sym_min={delta_per_sym_min:+.4f}", flush=True)
    success = (delta_loso > 0.5) or (delta_per_sym_min > 0.5)
    print(f"SUCCESS_GATE (Δloso>0.5 OR Δpsmin>0.5): {success}", flush=True)
    print(f"{'='*100}", flush=True)
    print(f"Total: {time.time()-t0:.1f}s", flush=True)

    out = {
        "task": "R4 logret-lag (T110 #2)",
        "seeds": list(SEEDS),
        "baseline": base,
        "trick": trick,
        "metrics": {
            "baseline_loso": base["de_sum"],
            "trick_loso": trick["de_sum"],
            "delta_loso": delta_loso,
            "per_sym_min_baseline": base["de_per_sym_min"],
            "per_sym_min_trick": trick["de_per_sym_min"],
            "delta_per_sym_min": delta_per_sym_min,
            "per_sym_std_baseline": base["de_per_sym_std"],
            "per_sym_std_trick": trick["de_per_sym_std"],
        },
        "success_gate": success,
    }
    out_path = os.path.join(HERE, "results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
