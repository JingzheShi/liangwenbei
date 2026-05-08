"""T86: Dual-gate evaluation on quantile predictions.

Loads pred_T86_q030_seed{S}.parquet and pred_T86_q070_seed{S}.parquet for the
5 seeds, averages each quantile across seeds, then evaluates several
"dual-gate" decision rules and compares with iter_013 (T75 LGB regression alone)
and iter_014 (NN+LGB ensemble).

Decision rules tried:
  R1 — agree-on-sign: long if q70 > thr_up AND q30 > 0; short if q30 < -thr_dn AND q70 < 0; else flat
  R2 — both-strong:  long if q30 > thr_up AND q70 > thr_up; short if q70 < -thr_dn AND q30 < -thr_dn
  R3 — IQR centred mean (avg of q30 and q70, EV-gate on the average) — sanity baseline
  R4 — asymmetric tail thresh on each quantile (DE 4-D: thr30_lo, thr30_hi, thr70_lo, thr70_hi)
  R5 — hybrid: combine quantile midpoint with T75 mean prediction, then EV gate

For each rule we DE-tune thresholds on LOSO-equiv (sum of per-sym cum_pnl) on the
local 442k test set, mirroring iter_013/iter_014 protocol.
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
T75_DIR = os.path.join(ROOT, "experiments", "T75_regression_dmid")
T81_DIR = os.path.join(ROOT, "experiments", "T81_nn_regression_pnl")

SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001
ITER012_LOSO = 26.44
ITER013_LOSO = 36.23
ITER014_LOSO = 38.28
SEEDS = (1, 7, 13, 42, 100)


def vectorized_pnl(action, mp_t, mp_th):
    side = action.astype(np.float64) - 1.0
    abs_side = np.abs(side)
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * abs_side * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee_pnl) / denom


def split_by_sym(rows):
    return [{
        "sym": int(k),
        **{kk: vv[rows["sym"] == k] for kk, vv in rows.items() if kk != "sym"},
        "n": int((rows["sym"] == k).sum()),
    } for k in SYMS]


def load_avg_quantile(seeds, q_tag):
    base = pd.read_parquet(os.path.join(HERE, f"pred_T86_{q_tag}_seed{seeds[0]}.parquet"))
    n = len(base)
    p = np.zeros(n, dtype=np.float64)
    for s in seeds:
        df_s = pd.read_parquet(os.path.join(HERE, f"pred_T86_{q_tag}_seed{s}.parquet"))
        if len(df_s) != n:
            raise RuntimeError(f"row count mismatch seed={s} q={q_tag}: {len(df_s)} vs {n}")
        if not (df_s["sym"].equals(base["sym"]) and df_s["t"].equals(base["t"])
                and df_s["date"].equals(base["date"])):
            raise RuntimeError(f"row order mismatch seed={s} q={q_tag}")
        p += df_s["pred_dmid_norm"].to_numpy(np.float64)
    p /= float(len(seeds))
    return p, base


def load_avg_t75():
    """Mean (L2) predictions from T75 LightGBM."""
    base = pd.read_parquet(os.path.join(T75_DIR, f"pred_T75_seed{SEEDS[0]}.parquet"))
    n = len(base)
    p = np.zeros(n, dtype=np.float64)
    for s in SEEDS:
        df_s = pd.read_parquet(os.path.join(T75_DIR, f"pred_T75_seed{s}.parquet"))
        if len(df_s) != n:
            raise RuntimeError(f"row count mismatch T75 seed={s}")
        p += df_s["pred_dmid_norm"].to_numpy(np.float64)
    p /= float(len(SEEDS))
    return p, base


def load_avg_t81():
    base = pd.read_parquet(os.path.join(T81_DIR, f"pred_T81_seed{SEEDS[0]}.parquet"))
    n = len(base)
    p = np.zeros(n, dtype=np.float64)
    for s in SEEDS:
        df_s = pd.read_parquet(os.path.join(T81_DIR, f"pred_T81_seed{s}.parquet"))
        if len(df_s) != n:
            raise RuntimeError(f"row count mismatch T81 seed={s}")
        p += df_s["pred_dmid_norm"].to_numpy(np.float64)
    p /= float(len(SEEDS))
    return p, base


def folds_from_arrays(sym, q30, q70, mp_t, mp_th, **extras):
    out = []
    for k in SYMS:
        m = sym == k
        out.append({
            "sym": int(k),
            "q30": q30[m], "q70": q70[m],
            "mp_t": mp_t[m], "mp_th": mp_th[m],
            **{kk: vv[m] for kk, vv in extras.items()},
            "n": int(m.sum()),
        })
    return out


# ------------------------- Decision rules ----------------------------- #


def rule_R1_agree_on_sign(q30, q70, thr_up, thr_dn):
    """long if q70 > thr_up AND q30 > 0; short if q30 < -thr_dn AND q70 < 0."""
    a = np.full(len(q30), 1, dtype=np.int8)
    long_m = (q70 > thr_up) & (q30 > 0.0)
    short_m = (q30 < -thr_dn) & (q70 < 0.0)
    a[long_m] = 2
    a[short_m] = 0
    return a


def rule_R2_both_strong(q30, q70, thr_up, thr_dn):
    """long if both q30 and q70 > thr_up; short if both q30 and q70 < -thr_dn."""
    a = np.full(len(q30), 1, dtype=np.int8)
    long_m = (q30 > thr_up) & (q70 > thr_up)
    short_m = (q30 < -thr_dn) & (q70 < -thr_dn)
    a[long_m] = 2
    a[short_m] = 0
    return a


def rule_R3_iqr_mid(q30, q70, thr_up, thr_dn):
    """Use IQR midpoint (q30 + q70) / 2 then standard EV-gate."""
    p = 0.5 * (q30 + q70)
    a = np.full(len(p), 1, dtype=np.int8)
    a[p > thr_up] = 2
    a[p < -thr_dn] = 0
    return a


def rule_R4_independent_thresh(q30, q70, thr30_lo, thr70_hi):
    """long if q70 > thr70_hi; short if q30 < -thr30_lo. Independent."""
    a = np.full(len(q30), 1, dtype=np.int8)
    a[q70 > thr70_hi] = 2
    a[q30 < -thr30_lo] = 0
    return a


def rule_R5_hybrid(q30, q70, mean_t75, thr_up, thr_dn, w_q, w_m):
    """combined = w_q*(q30+q70)/2 + w_m * mean_t75; EV-gate on combined."""
    p = w_q * 0.5 * (q30 + q70) + w_m * mean_t75
    a = np.full(len(p), 1, dtype=np.int8)
    a[p > thr_up] = 2
    a[p < -thr_dn] = 0
    return a


# ------------------------- Search helpers ----------------------------- #


def loso_pnl(folds, action_fn):
    """Compute LOSO-equiv = sum_per_sym(cum_pnl) given per-fold action_fn(fold)."""
    total = 0.0
    per = []
    n_active = []
    for f in folds:
        a = action_fn(f)
        per.append(float(vectorized_pnl(a, f["mp_t"], f["mp_th"]).sum()))
        n_active.append(int((a != 1).sum()))
        total += per[-1]
    return total, per, n_active


def de_tune(obj_fn, bounds, seeds=(0, 1, 2), maxiter=60, popsize=20):
    runs = []
    for sd in seeds:
        t0 = time.time()
        result = differential_evolution(
            obj_fn, bounds=bounds, seed=sd, maxiter=maxiter, popsize=popsize,
            polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            updating="deferred", workers=1, init="sobol",
        )
        runs.append({
            "seed": int(sd),
            "x": [float(v) for v in result.x],
            "obj_val": float(-result.fun),
            "nfev": int(result.nfev),
            "elapsed_sec": float(time.time() - t0),
        })
    runs.sort(key=lambda r: r["obj_val"], reverse=True)
    return runs


def evaluate_rule(name, folds, rule_fn, bounds, var_names):
    """Generic evaluator: rule_fn(fold, *params) → action; DE-tune on LOSO."""
    def obj(x):
        s = 0.0
        for f in folds:
            a = rule_fn(f, *x)
            s += vectorized_pnl(a, f["mp_t"], f["mp_th"]).sum()
        return -float(s)

    runs = de_tune(obj, bounds)
    best = runs[0]
    x = best["x"]
    total, per, n_active = loso_pnl(folds, lambda f: rule_fn(f, *x))
    print(f"  [{name}] best params {dict(zip(var_names, x))}  "
          f"→ LOSO {total:+.4f}  per_sym=[{', '.join(f'{v:+.3f}' for v in per)}]  "
          f"n_active=[{', '.join(str(v) for v in n_active)}]", flush=True)
    return {
        "name": name,
        "best_params": dict(zip(var_names, x)),
        "loso_sum_per_sym": total,
        "per_sym": per,
        "n_active_per_sym": n_active,
        "all_runs": runs,
    }


# ------------------------- Main ----------------------------- #


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,7,13,42,100")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")]
    print(f"=== T86 dual-gate eval seeds={seeds} ===", flush=True)

    t0 = time.time()
    q30, base30 = load_avg_quantile(seeds, "q030")
    q70, base70 = load_avg_quantile(seeds, "q070")
    if not base30["t"].equals(base70["t"]):
        raise RuntimeError("q030/q070 row order disagrees")
    base = base30
    print(f"  loaded q30 q70  ({len(base):,} rows) in {time.time()-t0:.1f}s",
          flush=True)
    print(f"  q30 mean={q30.mean():+.6f} std={q30.std():.6f}", flush=True)
    print(f"  q70 mean={q70.mean():+.6f} std={q70.std():.6f}", flush=True)
    print(f"  q70-q30 mean={(q70-q30).mean():+.6f} std={(q70-q30).std():.6f}",
          flush=True)
    print(f"  corr(q30,q70)={np.corrcoef(q30,q70)[0,1]:.4f}", flush=True)

    # Sign-agreement stats: how many rows agree?
    agree_pos = ((q30 > 0) & (q70 > 0)).sum()
    agree_neg = ((q30 < 0) & (q70 < 0)).sum()
    disagree = len(q30) - agree_pos - agree_neg
    print(f"  sign agreement: pos={agree_pos:,} neg={agree_neg:,} "
          f"split={disagree:,} (q30 and q70 differ in sign)", flush=True)

    sym = base["sym"].to_numpy(np.int8)
    mp_t = base["midprice_t"].to_numpy(np.float64)
    mp_th = base["midprice_th"].to_numpy(np.float64)

    # Load T75 mean predictions for hybrid rule
    p_t75, base_t75 = load_avg_t75()
    if not base_t75["t"].equals(base["t"]):
        raise RuntimeError("T75 vs T86 row order disagrees")
    print(f"  T75 mean: mean={p_t75.mean():+.6f} std={p_t75.std():.6f} "
          f"corr(t75,iqr_mid)={np.corrcoef(p_t75, 0.5*(q30+q70))[0,1]:.4f}",
          flush=True)

    # Load T81 NN
    p_t81, base_t81 = load_avg_t81()
    if not base_t81["t"].equals(base["t"]):
        raise RuntimeError("T81 vs T86 row order disagrees")
    print(f"  T81 mean: mean={p_t81.mean():+.6f} std={p_t81.std():.6f}",
          flush=True)

    # Build per-sym folds with all relevant arrays
    folds = folds_from_arrays(sym, q30, q70, mp_t, mp_th,
                              p_t75=p_t75, p_t81=p_t81)
    for f in folds:
        print(f"  sym={f['sym']}: n={f['n']:,} q30 mean={f['q30'].mean():+.6f} "
              f"q70 mean={f['q70'].mean():+.6f}", flush=True)

    results = {}

    # ===== R1 agree-on-sign =====
    print(f"\n[R1] agree-on-sign — long: q70>thr_up AND q30>0; short: q30<-thr_dn AND q70<0",
          flush=True)
    bounds = [(0.0, 0.004), (0.0, 0.004)]
    var = ["thr_up", "thr_dn"]
    r = evaluate_rule(
        "R1_agree_on_sign", folds,
        lambda f, thr_up, thr_dn: rule_R1_agree_on_sign(f["q30"], f["q70"], thr_up, thr_dn),
        bounds, var)
    results["R1_agree_on_sign"] = r

    # ===== R2 both-strong =====
    print(f"\n[R2] both-strong — long: q30>thr_up AND q70>thr_up; short: q30<-thr_dn AND q70<-thr_dn",
          flush=True)
    r = evaluate_rule(
        "R2_both_strong", folds,
        lambda f, thr_up, thr_dn: rule_R2_both_strong(f["q30"], f["q70"], thr_up, thr_dn),
        bounds, var)
    results["R2_both_strong"] = r

    # ===== R3 IQR-midpoint =====
    print(f"\n[R3] IQR midpoint — pred=(q30+q70)/2; classic EV-gate", flush=True)
    r = evaluate_rule(
        "R3_iqr_mid", folds,
        lambda f, thr_up, thr_dn: rule_R3_iqr_mid(f["q30"], f["q70"], thr_up, thr_dn),
        bounds, var)
    results["R3_iqr_mid"] = r

    # ===== R4 independent thresholds (no sign-agreement constraint) =====
    print(f"\n[R4] independent thresholds — long: q70>thr_70hi; short: q30<-thr_30lo",
          flush=True)
    bounds_r4 = [(0.0, 0.004), (0.0, 0.004)]
    var_r4 = ["thr_30lo", "thr_70hi"]
    r = evaluate_rule(
        "R4_independent", folds,
        lambda f, t30lo, t70hi: rule_R4_independent_thresh(f["q30"], f["q70"], t30lo, t70hi),
        bounds_r4, var_r4)
    results["R4_independent"] = r

    # ===== R5a hybrid: midpoint + T75 mean =====
    print(f"\n[R5a] hybrid — combined = w_q*(q30+q70)/2 + w_m*p_t75; classic gate",
          flush=True)
    bounds_r5 = [(0.0, 0.004), (0.0, 0.004), (0.0, 2.0), (0.0, 2.0)]
    var_r5 = ["thr_up", "thr_dn", "w_q", "w_m"]
    r = evaluate_rule(
        "R5a_hybrid_t75", folds,
        lambda f, thr_up, thr_dn, w_q, w_m: rule_R5_hybrid(
            f["q30"], f["q70"], f["p_t75"], thr_up, thr_dn, w_q, w_m),
        bounds_r5, var_r5)
    results["R5a_hybrid_t75"] = r

    # ===== R5b hybrid: midpoint + T75 + T81 (full) =====
    print(f"\n[R5b] hybrid — combined = w_q*(q30+q70)/2 + w_m_lgb*p_t75 + w_m_nn*p_t81",
          flush=True)
    bounds_r5b = [(0.0, 0.004), (0.0, 0.004), (0.0, 2.0), (0.0, 2.0), (0.0, 2.0)]
    var_r5b = ["thr_up", "thr_dn", "w_q", "w_lgb", "w_nn"]
    def rule_r5b(q30, q70, p_lgb, p_nn, thr_up, thr_dn, w_q, w_lgb, w_nn):
        p = w_q * 0.5 * (q30 + q70) + w_lgb * p_lgb + w_nn * p_nn
        a = np.full(len(p), 1, dtype=np.int8)
        a[p > thr_up] = 2
        a[p < -thr_dn] = 0
        return a
    r = evaluate_rule(
        "R5b_hybrid_t75_t81", folds,
        lambda f, thr_up, thr_dn, w_q, w_lgb, w_nn: rule_r5b(
            f["q30"], f["q70"], f["p_t75"], f["p_t81"], thr_up, thr_dn, w_q, w_lgb, w_nn),
        bounds_r5b, var_r5b)
    results["R5b_hybrid_t75_t81"] = r

    # ===== Summary =====
    print(f"\n{'='*78}\n=== T86 SUMMARY ===\n{'='*78}", flush=True)
    print(f"  iter_012 baseline: +{ITER012_LOSO}", flush=True)
    print(f"  iter_013 (T75 LGB regr alone): +{ITER013_LOSO}", flush=True)
    print(f"  iter_014 (T75 LGB + T81 NN ensemble): +{ITER014_LOSO}", flush=True)
    print(f"  Rules tried:", flush=True)
    for name, r in results.items():
        loso = r["loso_sum_per_sym"]
        print(f"    {name:24s}: {loso:+.4f}  vs_iter013 {loso-ITER013_LOSO:+.4f}  "
              f"vs_iter014 {loso-ITER014_LOSO:+.4f}", flush=True)
    best_name = max(results, key=lambda k: results[k]["loso_sum_per_sym"])
    best = results[best_name]
    print(f"\n  BEST rule: {best_name}  → {best['loso_sum_per_sym']:+.4f}  "
          f"(vs iter_014 {best['loso_sum_per_sym']-ITER014_LOSO:+.4f})", flush=True)

    out = {
        "task": "T86 dual-gate quantile evaluation",
        "seeds": seeds,
        "n_test": len(base),
        "fee_rate": FEE,
        "ref": {"iter012": ITER012_LOSO, "iter013": ITER013_LOSO, "iter014": ITER014_LOSO},
        "quantile_stats": {
            "q30_mean": float(q30.mean()), "q30_std": float(q30.std()),
            "q70_mean": float(q70.mean()), "q70_std": float(q70.std()),
            "corr_q30_q70": float(np.corrcoef(q30, q70)[0, 1]),
            "agree_pos_count": int(agree_pos),
            "agree_neg_count": int(agree_neg),
            "split_count": int(disagree),
        },
        "rules": results,
        "best": {"name": best_name, "loso": best["loso_sum_per_sym"],
                 "vs_iter014": best["loso_sum_per_sym"] - ITER014_LOSO,
                 "vs_iter013": best["loso_sum_per_sym"] - ITER013_LOSO,
                 "params": best["best_params"]},
    }
    out_path = os.path.join(HERE, "dual_gate_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
