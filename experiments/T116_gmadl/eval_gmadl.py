"""T116: Eval GMADL configs vs T99 LGB Huber baseline.

For each config (a, b) ensemble preds across the 3 seeds (1, 7, 42), then run
DE 2D asymmetric threshold search per ensemble. Compare DE-sum (sum over 5
syms with shared thresholds — this is what we call LOSO-equiv) and per-sym
breakdown.

Also compute cross-correlation between best GMADL ensemble and T99 Huber
ensemble (for stack candidate evaluation).
"""
from __future__ import annotations
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
T99 = os.path.join(ROOT, "experiments", "T99_e2e_execution_gbdt")

SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001
SEEDS = (1, 7, 42)
ITER014 = 38.281

GMADL_CONFIGS = [
    # (a, b, tag)
    (500000.0,  1.0, "gmadl_a500000_b1"),
    (500000.0,  1.5, "gmadl_a500000_b1.5"),
    (1000000.0, 1.0, "gmadl_a1e+06_b1"),
]


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


def avg_preds(dir_, prefix, suffix=""):
    """Average pred_dmid_norm across SEEDS for a given prefix/suffix."""
    base = pd.read_parquet(
        os.path.join(dir_, f"{prefix}{SEEDS[0]}{suffix}.parquet"))
    p = np.zeros(len(base), dtype=np.float64)
    for s in SEEDS:
        df = pd.read_parquet(
            os.path.join(dir_, f"{prefix}{s}{suffix}.parquet"))
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
    print(f"  [{label:<55s}] DE={s:+.4f} thr_up={tu:.4e} thr_dn={td:.4e}",
          flush=True)
    return {"label": label, "thr_up": tu, "thr_dn": td,
            "de_sum": s, "per_sym": per}


def main():
    print("=== T116 GMADL eval vs T99 LGB Huber baseline ===", flush=True)
    t0 = time.time()

    # T99 baseline: 3-seed ensemble with seeds (1,7,42)
    base_h, p_huber = avg_preds(T99, "pred_T99_huber_a0.001_seed")
    print(f"  loaded T99 huber 3-seed: n={len(base_h):,}", flush=True)

    # GMADL configs
    gmadl_preds = {}
    for a, b, tag in GMADL_CONFIGS:
        try:
            base_g, p_g = avg_preds(HERE, f"pred_T116_{tag}_seed")
        except Exception as e:
            print(f"  skip {tag}: {e!r}", flush=True)
            continue
        if not (base_g["sym"].equals(base_h["sym"])
                and base_g["t"].equals(base_h["t"])):
            raise RuntimeError(f"row mismatch huber vs gmadl {tag}")
        gmadl_preds[tag] = (a, b, p_g)
        print(f"  loaded GMADL {tag}: 3-seed avg", flush=True)

    # Cross-correlations
    print("\nCross-correlations:", flush=True)
    print(f"  corr(LGB_Huber, LGB_Huber) = 1.0000 (sanity)", flush=True)
    cross_corr = {}
    for tag, (a, b, p_g) in gmadl_preds.items():
        c = float(np.corrcoef(p_huber, p_g)[0, 1])
        cross_corr[tag] = c
        print(f"  corr(LGB_Huber, GMADL[{tag}]) = {c:.4f}", flush=True)
    # GMADL-vs-GMADL
    keys = list(gmadl_preds.keys())
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            ki, kj = keys[i], keys[j]
            c = float(np.corrcoef(gmadl_preds[ki][2], gmadl_preds[kj][2])[0, 1])
            print(f"  corr(GMADL[{ki}], GMADL[{kj}]) = {c:.4f}", flush=True)

    # Singles
    print("\n=== Singles (3-seed ensemble) ===", flush=True)
    results = []
    results.append(evaluate(p_huber, base_h, "T99_LGB_Huber baseline"))
    for tag, (a, b, p_g) in gmadl_preds.items():
        results.append(evaluate(p_g, base_h, f"GMADL {tag}"))

    # 50/50 stacks
    print("\n=== 2-way Huber + GMADL ===", flush=True)
    for tag, (a, b, p_g) in gmadl_preds.items():
        for w in [0.5, 1.0, 1.5]:
            p_stack = (1.0 * p_huber + w * p_g) / (1.0 + w)
            results.append(evaluate(p_stack, base_h, f"Huber+GMADL[{tag}] w=1:{w}"))

    baseline = results[0]
    gmadl_singles = [r for r in results[1:1 + len(gmadl_preds)]]
    best_gmadl = max(gmadl_singles, key=lambda r: r["de_sum"])
    best_overall = max(results, key=lambda r: r["de_sum"])

    delta = best_gmadl["de_sum"] - baseline["de_sum"]
    delta_overall = best_overall["de_sum"] - baseline["de_sum"]

    # Identify best GMADL tag for cross_corr
    best_tag = best_gmadl["label"].split("GMADL ")[-1]
    best_corr = cross_corr.get(best_tag, None)
    # Identify best a, b
    best_a, best_b = None, None
    for a, b, tag in GMADL_CONFIGS:
        if best_tag == tag:
            best_a, best_b = a, b
            break

    print(f"\n{'='*100}", flush=True)
    print(f"BASELINE T99 Huber 3-seed: {baseline['de_sum']:+.4f}", flush=True)
    print(f"  per_sym = {[round(x,2) for x in baseline['per_sym']]}", flush=True)
    print(f"BEST GMADL single: {best_gmadl['label']}  DE={best_gmadl['de_sum']:+.4f}",
          flush=True)
    print(f"  per_sym = {[round(x,2) for x in best_gmadl['per_sym']]}", flush=True)
    print(f"  delta vs baseline = {delta:+.4f}", flush=True)
    print(f"BEST OVERALL: {best_overall['label']}  DE={best_overall['de_sum']:+.4f}",
          flush=True)
    print(f"  per_sym = {[round(x,2) for x in best_overall['per_sym']]}", flush=True)
    print(f"  delta vs baseline = {delta_overall:+.4f}", flush=True)
    print(f"  vs iter_014 (+{ITER014}): {best_overall['de_sum']-ITER014:+.4f}",
          flush=True)
    print(f"  cross_corr_GMADL_to_Huber = {best_corr}", flush=True)
    print(f"{'='*100}", flush=True)

    out = {
        "baseline_loso": baseline["de_sum"],
        "baseline_per_sym": baseline["per_sym"],
        "gmadl_best_loso": best_gmadl["de_sum"],
        "gmadl_best_per_sym": best_gmadl["per_sym"],
        "gmadl_best_label": best_gmadl["label"],
        "best_a": best_a,
        "best_b": best_b,
        "delta": delta,
        "best_overall_label": best_overall["label"],
        "best_overall_loso": best_overall["de_sum"],
        "best_overall_per_sym": best_overall["per_sym"],
        "delta_overall": delta_overall,
        "cross_corr_to_huber": cross_corr,
        "cross_corr_to_huber_best": best_corr,
        "iter_014_baseline": ITER014,
        "vs_iter_014": best_overall["de_sum"] - ITER014,
        "all_results": results,
        "n_seeds": len(SEEDS),
        "seeds": list(SEEDS),
    }
    with open(os.path.join(HERE, "eval_gmadl.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote -> eval_gmadl.json (took {time.time()-t0:.1f}s)", flush=True)
    return out


if __name__ == "__main__":
    main()
