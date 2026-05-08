"""T122 DE LOSO evaluation: 5 variants × 3 seeds → 5 ensemble DE LOSO scores.

Variants: baseline, hyd, lag, monotone, all3.
For each variant, average pred across 3 seeds, then DE 2D asymmetric thresh
(thr_up, thr_dn) maximizing sum of per-sym cum_pnl (LOSO-equiv).

Outputs:
  eval_t122.json  — per-variant DE LOSO + per-sym + delta vs baseline
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))

VARIANTS = ("baseline", "hyd", "lag", "monotone", "all3")
SEEDS = (1, 7, 42)
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001
ITER014_LOSO = 38.28  # iter_014 baseline reference (T75 + dual-gate)
T75_BASELINE_LOSO = 36.23  # T75 5-seed DE LOSO


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
                                   maxiter=80, popsize=24, polish=True,
                                   tol=1e-7, init="sobol")
        runs.append((float(-r.fun), float(r.x[0]), float(r.x[1])))
    runs.sort(key=lambda x: x[0], reverse=True)
    return runs[0]


def avg_preds(variant, seeds):
    paths = [os.path.join(HERE, f"pred_T122_{variant}_seed{s}.parquet") for s in seeds]
    base = pd.read_parquet(paths[0])
    p = np.zeros(len(base), dtype=np.float64)
    for path in paths:
        df = pd.read_parquet(path)
        if not (df["sym"].equals(base["sym"]) and df["t"].equals(base["t"])):
            raise RuntimeError(f"row mismatch in {path}")
        p += df["pred_dmid_norm"].to_numpy(np.float64)
    return base, p / len(paths)


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
    print(f"  [{label:<28s}] DE LOSO={s:+.4f}  thr_up={tu:.4e} thr_dn={td:.4e}  "
          f"per_sym={[round(x,2) for x in per]}", flush=True)
    return {"label": label, "thr_up": tu, "thr_dn": td,
            "de_sum": s, "per_sym": per,
            "per_sym_min": float(min(per))}


def main():
    print("=== T122 DE LOSO eval (5 variants × 3-seed ensemble) ===", flush=True)
    t0 = time.time()

    # Per-seed singles (for variability assessment)
    per_seed_results = {v: {} for v in VARIANTS}
    print("\n--- Per-seed singles ---", flush=True)
    for v in VARIANTS:
        for s in SEEDS:
            try:
                df = pd.read_parquet(os.path.join(HERE, f"pred_T122_{v}_seed{s}.parquet"))
            except FileNotFoundError as e:
                print(f"  MISSING {v} seed{s}: {e}", flush=True)
                continue
            p = df["pred_dmid_norm"].to_numpy(np.float64)
            res = evaluate(p, df, f"{v}_seed{s}")
            per_seed_results[v][str(s)] = res

    # 3-seed avg ensembles
    print("\n--- 3-seed averaged ensemble (DE per variant) ---", flush=True)
    ens_results = {}
    for v in VARIANTS:
        try:
            base, p_avg = avg_preds(v, SEEDS)
        except FileNotFoundError as e:
            print(f"  MISSING ensemble for {v}: {e}", flush=True)
            continue
        ens_results[v] = evaluate(p_avg, base, f"{v}_3seed_ens")

    # Cross correlations between variants (post-ensemble)
    print("\n--- Pred correlations (post 3-seed ensemble) ---", flush=True)
    cors = {}
    base_b, p_baseline = avg_preds("baseline", SEEDS)
    for v in VARIANTS:
        if v == "baseline":
            continue
        try:
            _, p_v = avg_preds(v, SEEDS)
            c = float(np.corrcoef(p_baseline, p_v)[0, 1])
            cors[v] = c
            print(f"  corr(baseline, {v}) = {c:.4f}", flush=True)
        except FileNotFoundError:
            pass

    # Deltas vs baseline
    print(f"\n=== DELTAS (vs baseline ensemble) ===", flush=True)
    if "baseline" in ens_results:
        b_de = ens_results["baseline"]["de_sum"]
        b_per = ens_results["baseline"]["per_sym"]
        deltas = {}
        for v in VARIANTS:
            if v == "baseline" or v not in ens_results:
                continue
            v_de = ens_results[v]["de_sum"]
            v_per = ens_results[v]["per_sym"]
            d = v_de - b_de
            d_per = [v_per[i] - b_per[i] for i in range(len(SYMS))]
            deltas[v] = {"de_delta": d, "per_sym_delta": d_per,
                         "per_sym_min_delta": float(min(d_per)),
                         "passes_iter017_threshold": bool(d >= 0.5)}
            mark = "✅" if d >= 0.5 else ("⚠️" if d >= 0.3 else "❌")
            print(f"  Δ {v:>10s} = {d:+.4f} LOSO {mark}  (per_sym Δ = "
                  f"{[round(x,2) for x in d_per]})", flush=True)
    else:
        deltas = {}

    # Summary
    print(f"\n=== SUMMARY ===", flush=True)
    print(f"  {'variant':>10s}  {'DE_LOSO':>10s}  {'Δ vs base':>10s}  {'per_sym_min':>11s}  passes?", flush=True)
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*11}", flush=True)
    for v in VARIANTS:
        if v not in ens_results:
            continue
        r = ens_results[v]
        d_str = f"{deltas[v]['de_delta']:+.4f}" if v in deltas else "—"
        passes = "PASS" if v in deltas and deltas[v]["passes_iter017_threshold"] else (
            "(base)" if v == "baseline" else "fail"
        )
        print(f"  {v:>10s}  {r['de_sum']:>+10.4f}  {d_str:>10s}  "
              f"{r['per_sym_min']:>+11.4f}  {passes}", flush=True)

    out = {
        "task": "T122 DE LOSO eval — retest 3 winning tricks on T75 LGB L2 baseline",
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "per_seed": per_seed_results,
        "ensemble": ens_results,
        "correlations_to_baseline": cors,
        "deltas_vs_baseline": deltas,
        "iter014_loso_ref": ITER014_LOSO,
        "t75_baseline_loso_ref": T75_BASELINE_LOSO,
        "elapsed_sec": time.time() - t0,
    }
    out_path = os.path.join(HERE, "eval_t122.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote -> {out_path}  total={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
