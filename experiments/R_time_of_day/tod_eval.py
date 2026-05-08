"""TOD evaluation: T87 + T75-style LGB ensemble at weight 1.0:1.5 (iter_015 v1).

Compares:
  baseline_ens  = (p_T87_avg + 1.5 * p_LGB_baseline_avg) / 2.5
  trick_ens     = (p_T87_avg + 1.5 * p_LGB_trick_avg)    / 2.5
where:
  p_T87_avg     = mean of T87 NN preds over 5 seeds (1,7,13,42,100)
  p_LGB_baseline_avg = mean of R4 baseline preds over 3 seeds (1,7,42)
  p_LGB_trick_avg    = mean of TOD trick preds over 3 seeds (1,7,42)

For each ensemble run DE asym thresh search per-sym aggregated ("LOSO-equiv").
Decision: trick_ens - baseline_ens > 0.5 → iter_018 candidate.

Also reports LGB-only baseline vs trick (no T87 ensemble) for sanity.
"""
from __future__ import annotations
import json, os, time
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

OUT_DIR = "/root/lwb_work_tod"
LGB_BASELINE_DIR = "/root/lwb_work"   # R4 baseline preds (reused)
LGB_TRICK_DIR = "/root/lwb_work_tod"
T87_DIR = "/root/lwb_remote_pkg/preds"
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001
LGB_SEEDS = (1, 7, 42)
T87_SEEDS = (1, 7, 13, 42, 100)
ITER015_V1_REF = 40.13


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


def split_sym(df, p):
    out = []
    sym = df["sym"].to_numpy(np.int8)
    mp_t = df["midprice_t"].to_numpy(np.float64)
    mp_th = df["midprice_th"].to_numpy(np.float64)
    for k in SYMS:
        m = sym == k
        out.append({"sym": int(k), "pred": p[m], "mp_t": mp_t[m], "mp_th": mp_th[m]})
    return out


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


def de_search(obj, bounds, seeds=(0, 1, 2, 7, 42)):
    runs = []
    for sd in seeds:
        r = differential_evolution(obj, bounds=bounds, seed=sd,
                                   maxiter=80, popsize=24,
                                   polish=True, tol=1e-7, init="sobol")
        runs.append((float(-r.fun), float(r.x[0]), float(r.x[1])))
    runs.sort(key=lambda x: x[0], reverse=True)
    return runs[0]


def avg_preds(paths, base_index_check=True):
    base = pd.read_parquet(paths[0])
    n = len(base)
    p = np.zeros(n, dtype=np.float64)
    for path in paths:
        df = pd.read_parquet(path)
        if len(df) != n:
            raise RuntimeError(f"row count mismatch {path}")
        if base_index_check:
            if not (df["sym"].equals(base["sym"]) and df["t"].equals(base["t"])):
                raise RuntimeError(f"row order mismatch {path}")
        p += df["pred_dmid_norm"].to_numpy(np.float64)
    return p / len(paths), base


def evaluate(label, p, base):
    folds = split_sym(base, p)
    obj = make_obj(folds)
    s, tu, td = de_search(obj, [(0.0, 0.0040), (0.0, 0.0040)])
    per = []
    for f in folds:
        a = gate_asym(f["pred"], tu, td)
        per.append(float(vectorized_pnl(a, f["mp_t"], f["mp_th"]).sum()))
    print(f"  [{label}] DE sum={s:+.4f}  thr_up={tu:.4e} thr_dn={td:.4e}", flush=True)
    print(f"    per_sym=[{', '.join(f'{x:+.3f}' for x in per)}]", flush=True)
    return {"label": label, "de_sum": s, "thr_up": tu, "thr_dn": td,
            "per_sym": per, "per_sym_min": min(per), "per_sym_max": max(per),
            "per_sym_std": float(np.std(per))}


def main():
    print("=== TOD eval: LGB baseline vs trick (with iter_015 v1 ensemble) ===", flush=True)
    t0 = time.time()

    # Load LGB baseline (R4 reused)
    lgb_base_paths = [os.path.join(LGB_BASELINE_DIR, f"pred_R4_baseline_seed{s}.parquet")
                      for s in LGB_SEEDS]
    p_lgb_base, base_df = avg_preds(lgb_base_paths)
    print(f"\nLGB baseline: avg over {LGB_SEEDS}  shape={p_lgb_base.shape}", flush=True)

    # Load LGB trick (TOD)
    lgb_trick_paths = [os.path.join(LGB_TRICK_DIR, f"pred_TOD_trick_seed{s}.parquet")
                       for s in LGB_SEEDS]
    p_lgb_trick, base_df_t = avg_preds(lgb_trick_paths)
    if not (base_df_t["sym"].equals(base_df["sym"]) and base_df_t["t"].equals(base_df["t"])):
        raise RuntimeError("LGB trick row order != LGB baseline row order")
    print(f"LGB trick   : avg over {LGB_SEEDS}  shape={p_lgb_trick.shape}", flush=True)

    # Load T87 NN average
    t87_paths = [os.path.join(T87_DIR, f"pred_T87_seed{s}_main.parquet")
                 for s in T87_SEEDS]
    p_t87, base_df_n = avg_preds(t87_paths)
    if not (base_df_n["sym"].equals(base_df["sym"]) and base_df_n["t"].equals(base_df["t"])):
        raise RuntimeError("T87 row order != LGB row order")
    print(f"T87 NN     : avg over {T87_SEEDS}  shape={p_t87.shape}", flush=True)

    # Cross-correlations
    print(f"\nCorrelations (full 442k):", flush=True)
    print(f"  T87 vs LGB_baseline: {float(np.corrcoef(p_t87, p_lgb_base)[0,1]):.4f}", flush=True)
    print(f"  T87 vs LGB_trick   : {float(np.corrcoef(p_t87, p_lgb_trick)[0,1]):.4f}", flush=True)
    print(f"  LGB_baseline vs trick: {float(np.corrcoef(p_lgb_base, p_lgb_trick)[0,1]):.4f}", flush=True)

    print(f"\n--- Standalone LGB eval ---", flush=True)
    res_lgb_base = evaluate("LGB-baseline-only", p_lgb_base, base_df)
    res_lgb_trick = evaluate("LGB-trick-only   ", p_lgb_trick, base_df)

    print(f"\n--- Ensemble eval (T87 w_nn=1.0, LGB w_lgb=1.5) ---", flush=True)
    p_ens_base = (1.0 * p_t87 + 1.5 * p_lgb_base) / 2.5
    p_ens_trick = (1.0 * p_t87 + 1.5 * p_lgb_trick) / 2.5
    res_ens_base = evaluate("ENS-baseline (T87+LGB_base 1:1.5)", p_ens_base, base_df)
    res_ens_trick = evaluate("ENS-trick    (T87+LGB_trick 1:1.5)", p_ens_trick, base_df)

    print(f"\n--- Ensemble eval (T87 w_nn=1.0, LGB w_lgb=1.0) ---", flush=True)
    p_ens_base_11 = (p_t87 + p_lgb_base) / 2.0
    p_ens_trick_11 = (p_t87 + p_lgb_trick) / 2.0
    res_ens_base_11 = evaluate("ENS-baseline (1:1)", p_ens_base_11, base_df)
    res_ens_trick_11 = evaluate("ENS-trick    (1:1)", p_ens_trick_11, base_df)

    delta_lgb = res_lgb_trick["de_sum"] - res_lgb_base["de_sum"]
    delta_ens = res_ens_trick["de_sum"] - res_ens_base["de_sum"]
    delta_ens_11 = res_ens_trick_11["de_sum"] - res_ens_base_11["de_sum"]

    print(f"\n{'='*100}", flush=True)
    print(f"LGB-only      delta = {delta_lgb:+.4f}  (base={res_lgb_base['de_sum']:.2f} → trick={res_lgb_trick['de_sum']:.2f})", flush=True)
    print(f"ENS 1:1.5     delta = {delta_ens:+.4f}  (base={res_ens_base['de_sum']:.2f} → trick={res_ens_trick['de_sum']:.2f})", flush=True)
    print(f"ENS 1:1.0     delta = {delta_ens_11:+.4f}  (base={res_ens_base_11['de_sum']:.2f} → trick={res_ens_trick_11['de_sum']:.2f})", flush=True)
    print(f"iter_015 v1 ref = +{ITER015_V1_REF:.2f}", flush=True)
    print(f"vs iter_015 v1 (trick ens 1:1.5): {res_ens_trick['de_sum']-ITER015_V1_REF:+.4f}", flush=True)
    success = delta_ens >= 0.5
    print(f"SUCCESS_GATE (Δens 1:1.5 >= 0.5): {success}", flush=True)
    print(f"{'='*100}", flush=True)
    print(f"Total: {time.time()-t0:.1f}s", flush=True)

    out = {
        "task": "R_time_of_day eval",
        "lgb_seeds": list(LGB_SEEDS),
        "t87_seeds": list(T87_SEEDS),
        "iter_015_v1_ref": ITER015_V1_REF,
        "lgb_baseline": res_lgb_base,
        "lgb_trick": res_lgb_trick,
        "ens_baseline_1_to_1p5": res_ens_base,
        "ens_trick_1_to_1p5": res_ens_trick,
        "ens_baseline_1_to_1": res_ens_base_11,
        "ens_trick_1_to_1": res_ens_trick_11,
        "metrics": {
            "delta_lgb_only": delta_lgb,
            "delta_ens_1_to_1p5": delta_ens,
            "delta_ens_1_to_1": delta_ens_11,
            "ens_trick_1_to_1p5_vs_iter015": res_ens_trick["de_sum"] - ITER015_V1_REF,
            "lgb_baseline_de_sum": res_lgb_base["de_sum"],
            "lgb_trick_de_sum": res_lgb_trick["de_sum"],
            "ens_baseline_1_to_1p5_de_sum": res_ens_base["de_sum"],
            "ens_trick_1_to_1p5_de_sum": res_ens_trick["de_sum"],
        },
        "success_gate_ens_1_to_1p5": bool(success),
    }
    out_path = os.path.join(OUT_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
