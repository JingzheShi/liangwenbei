"""T117 DE LOSO eval: 3-seed baseline vs 3-seed monotone, plus pairwise ensembles.

Computes:
  - per-seed DE LOSO (sym-aware, 2D thresh search)
  - 3-seed averaged DE LOSO for {baseline, monotone}
  - per-sym break-down
  - feature importance comparison from summary JSONs
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

SEEDS = (1, 7, 42)
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001


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


def avg_preds(prefix_pattern, seeds):
    paths = [prefix_pattern.format(seed=s) for s in seeds]
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
    print(f"  [{label:<55s}] DE={s:+.4f} thr_up={tu:.4e} thr_dn={td:.4e}  per_sym={[round(x,2) for x in per]}", flush=True)
    return {"label": label, "thr_up": tu, "thr_dn": td, "de_sum": s, "per_sym": per}


def main():
    print("=== T117 baseline vs monotone DE LOSO eval (3-seed) ===", flush=True)
    t0 = time.time()

    # Load 3-seed predictions
    baseline_paths = os.path.join(T99, "pred_T99_huber_a0.001_seed{seed}.parquet")
    mono_paths = os.path.join(HERE, "pred_T117_huber_a0.001_mono_seed{seed}.parquet")

    # Per-seed individual eval first
    print("\n--- Per-seed singles ---", flush=True)
    per_seed_results = {"baseline": {}, "monotone": {}}
    for s in SEEDS:
        b_df = pd.read_parquet(baseline_paths.format(seed=s))
        m_df = pd.read_parquet(mono_paths.format(seed=s))
        if not (b_df["sym"].equals(m_df["sym"]) and b_df["t"].equals(m_df["t"])):
            raise RuntimeError(f"row mismatch seed={s}")
        b_p = b_df["pred_dmid_norm"].to_numpy(np.float64)
        m_p = m_df["pred_dmid_norm"].to_numpy(np.float64)
        b_res = evaluate(b_p, b_df, f"baseline_seed{s}")
        m_res = evaluate(m_p, m_df, f"monotone_seed{s}")
        per_seed_results["baseline"][str(s)] = b_res
        per_seed_results["monotone"][str(s)] = m_res

    # Cross-correlation between baseline and monotone for each seed
    print("\n--- Pred correlations (baseline vs monotone, per seed) ---", flush=True)
    for s in SEEDS:
        b_df = pd.read_parquet(baseline_paths.format(seed=s))
        m_df = pd.read_parquet(mono_paths.format(seed=s))
        c = float(np.corrcoef(b_df["pred_dmid_norm"], m_df["pred_dmid_norm"])[0, 1])
        print(f"  seed={s}: corr(baseline, monotone) = {c:.4f}", flush=True)

    # 3-seed avg ensemble
    print("\n--- 3-seed averaged ensembles ---", flush=True)
    base_b, p_baseline = avg_preds(baseline_paths, SEEDS)
    base_m, p_monotone = avg_preds(mono_paths, SEEDS)
    if not (base_b["sym"].equals(base_m["sym"]) and base_b["t"].equals(base_m["t"])):
        raise RuntimeError("base row mismatch baseline vs monotone")

    res_baseline = evaluate(p_baseline, base_b, "baseline_3seed_ens")
    res_monotone = evaluate(p_monotone, base_m, "monotone_3seed_ens")

    # Cross-correlation
    cc = float(np.corrcoef(p_baseline, p_monotone)[0, 1])
    print(f"\n  corr(baseline_3seed, monotone_3seed) = {cc:.4f}", flush=True)

    # Mixed ensemble (50:50)
    p_mixed = 0.5 * p_baseline + 0.5 * p_monotone
    res_mixed = evaluate(p_mixed, base_b, "baseline+monotone 50:50 mix")

    # Sweep 6-seed (3 baseline + 3 monotone)
    p_6seed = (p_baseline + p_monotone) / 2.0
    # already same as mixed; alternative weights
    for w_m in [0.3, 0.7]:
        p_alt = (1.0 - w_m) * p_baseline + w_m * p_monotone
        evaluate(p_alt, base_b, f"weighted baseline+monotone w_m={w_m}")

    # Compare to T99 5-seed published number (+40.79)
    print(f"\n=== SUMMARY ===", flush=True)
    print(f"3-seed baseline DE LOSO: {res_baseline['de_sum']:+.4f}", flush=True)
    print(f"3-seed monotone DE LOSO: {res_monotone['de_sum']:+.4f}", flush=True)
    delta = res_monotone["de_sum"] - res_baseline["de_sum"]
    print(f"Δ (monotone - baseline) = {delta:+.4f} LOSO", flush=True)
    print(f"Δ per_sym = {[round(m-b, 2) for m, b in zip(res_monotone['per_sym'], res_baseline['per_sym'])]}", flush=True)
    delta_per_sym_min = min(m - b for m, b in zip(res_monotone["per_sym"], res_baseline["per_sym"]))
    delta_per_sym_max = max(m - b for m, b in zip(res_monotone["per_sym"], res_baseline["per_sym"]))
    print(f"Δ per_sym min={delta_per_sym_min:+.4f}, max={delta_per_sym_max:+.4f}", flush=True)

    # Aggregate feature-importance across monotone seeds
    fi_total = {"split_pct": [], "gain_pct": []}
    for s in SEEDS:
        with open(os.path.join(HERE, f"summary_T117_huber_a0.001_mono_seed{s}.json")) as f:
            d = json.load(f)
        fi_total["split_pct"].append(d["feat_importance"]["monotone_split_pct"])
        fi_total["gain_pct"].append(d["feat_importance"]["monotone_gain_pct"])
    print(f"\nMonotone feat usage (split): {[round(x, 2) for x in fi_total['split_pct']]} %  mean={np.mean(fi_total['split_pct']):.2f}%", flush=True)
    print(f"Monotone feat usage (gain) : {[round(x, 2) for x in fi_total['gain_pct']]} %  mean={np.mean(fi_total['gain_pct']):.2f}%", flush=True)

    out = {
        "per_seed": per_seed_results,
        "ensemble": {
            "baseline_3seed": res_baseline,
            "monotone_3seed": res_monotone,
            "mixed_50_50": res_mixed,
            "corr_baseline_monotone": cc,
        },
        "delta": {
            "loso_total": delta,
            "loso_per_sym": [m - b for m, b in zip(res_monotone["per_sym"], res_baseline["per_sym"])],
            "loso_per_sym_min": delta_per_sym_min,
            "loso_per_sym_max": delta_per_sym_max,
        },
        "feat_importance_monotone": {
            "split_pct_per_seed": fi_total["split_pct"],
            "gain_pct_per_seed": fi_total["gain_pct"],
            "split_pct_mean": float(np.mean(fi_total["split_pct"])),
            "gain_pct_mean": float(np.mean(fi_total["gain_pct"])),
        },
        "elapsed_sec": time.time() - t0,
    }
    out_path = os.path.join(HERE, "eval_t117.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote -> {out_path}", flush=True)
    print(f"Total time: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
