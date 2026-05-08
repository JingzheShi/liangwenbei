"""DE thresh + LOSO eval for R_roll_tsrv variants.

Compares:
  V0 (baseline): /root/lwb_remote_pkg/preds/pred_T75_seed{1,7,42}.parquet (avg)
  V1 (clean target): /root/lwb_work_roll_tsrv/pred_V1_seed{1,7,42}.parquet (avg)
  V2 (meta features): /root/lwb_work_roll_tsrv/pred_V2_seed{1,7,42}.parquet (avg)

For each: per-sym DE search on (thr_up, thr_dn), LOSO-equivalent sum-of-per-sym PnL.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

WORK_DIR = "/root/lwb_work_roll_tsrv"
T75_PRED_DIR = "/root/lwb_remote_pkg/preds"

SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001
SEEDS = (1, 7, 42)


def vectorized_pnl(pred_action, mp_t, mp_th):
    side = pred_action.astype(np.float64) - 1.0
    abs_side = np.abs(side)
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * abs_side * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee_pnl) / denom


def ev_gate_asym(pred, thr_up, thr_dn):
    a = np.full(len(pred), 1, dtype=np.int8)
    a[pred > thr_up] = 2
    a[pred < -thr_dn] = 0
    return a


def split_by_sym(df, p_avg):
    out = []
    for k in SYMS:
        m = df["sym"].to_numpy() == k
        out.append({
            "sym": int(k),
            "pred": p_avg[m].astype(np.float64),
            "mp_t": df["midprice_t"].to_numpy(np.float64)[m],
            "mp_th": df["midprice_th"].to_numpy(np.float64)[m],
            "n": int(m.sum()),
        })
    return out


def make_obj_loso_asym(folds):
    def f(x):
        thr_up, thr_dn = x
        s = 0.0
        for fk in folds:
            a = ev_gate_asym(fk["pred"], thr_up, thr_dn)
            s += vectorized_pnl(a, fk["mp_t"], fk["mp_th"]).sum()
        return -float(s)
    return f


def de_search(obj, bounds, seeds=(0, 1, 2, 7, 42)):
    runs = []
    for sd in seeds:
        r = differential_evolution(obj, bounds=bounds, seed=sd, maxiter=80, popsize=24,
                                   polish=True, tol=1e-7, mutation=(0.5, 1.0),
                                   recombination=0.7, updating="deferred", workers=1,
                                   init="sobol")
        runs.append({"seed": int(sd), "thr_up": float(r.x[0]),
                     "thr_dn": float(r.x[1]), "obj_val": float(-r.fun),
                     "nfev": int(r.nfev)})
    runs.sort(key=lambda r: r["obj_val"], reverse=True)
    return runs


def per_sym_at_thr(folds, thr_up, thr_dn):
    return [float(vectorized_pnl(ev_gate_asym(f["pred"], thr_up, thr_dn),
                                  f["mp_t"], f["mp_th"]).sum()) for f in folds]


def per_sym_DE(folds):
    """Per-sym DE: each sym gets its own (thr_up, thr_dn) — overfits but useful diag."""
    out = []
    for fk in folds:
        def obj(x):
            thr_up, thr_dn = x
            return -float(vectorized_pnl(ev_gate_asym(fk["pred"], thr_up, thr_dn),
                                          fk["mp_t"], fk["mp_th"]).sum())
        runs = de_search(obj, [(0.0, 0.004), (0.0, 0.004)], seeds=(0, 1, 2, 7, 42))
        b = runs[0]
        out.append({"sym": fk["sym"], "thr_up": b["thr_up"], "thr_dn": b["thr_dn"],
                    "pnl": b["obj_val"]})
    return out


def load_pred_avg(pred_dir, prefix, seeds=SEEDS, suffix=""):
    """Average predictions across seeds. Returns (df_template, p_avg)."""
    base = pd.read_parquet(os.path.join(pred_dir, f"{prefix}_seed{seeds[0]}{suffix}.parquet"))
    n = len(base)
    p = np.zeros(n, dtype=np.float64)
    for s in seeds:
        df = pd.read_parquet(os.path.join(pred_dir, f"{prefix}_seed{s}{suffix}.parquet"))
        if len(df) != n:
            raise RuntimeError(f"row count mismatch {prefix} seed={s}")
        if not (df["sym"].equals(base["sym"]) and df["t"].equals(base["t"])):
            raise RuntimeError(f"row order mismatch {prefix} seed={s}")
        p += df["pred_dmid_norm"].to_numpy(np.float64)
    p /= len(seeds)
    return base, p


def evaluate(label, base_df, p_avg):
    print(f"\n=== {label} ===", flush=True)
    folds = split_by_sym(base_df, p_avg)

    # Joint DE: single (thr_up, thr_dn) for all syms (LOSO-equiv: sum of per-sym PnL)
    obj = make_obj_loso_asym(folds)
    runs = de_search(obj, [(0.0, 0.004), (0.0, 0.004)])
    best = runs[0]
    thr_up, thr_dn = best["thr_up"], best["thr_dn"]
    de_per_sym = per_sym_at_thr(folds, thr_up, thr_dn)
    de_loso_sum = float(sum(de_per_sym))
    de_per_sym_min = float(min(de_per_sym))

    print(f"  DE joint: thr_up={thr_up:.6f} thr_dn={thr_dn:.6f}", flush=True)
    print(f"  per_sym: [{', '.join(f'{x:+.3f}' for x in de_per_sym)}]", flush=True)
    print(f"  LOSO_sum={de_loso_sum:+.4f}  per_sym_min={de_per_sym_min:+.3f}", flush=True)

    # Per-sym DE diag (UB upper bound)
    psd = per_sym_DE(folds)
    ub_sum = float(sum(p["pnl"] for p in psd))
    print(f"  PER-SYM DE (UB): sum={ub_sum:+.4f}", flush=True)

    return {"label": label, "thr_up": thr_up, "thr_dn": thr_dn,
            "de_loso_sum": de_loso_sum, "per_sym": de_per_sym,
            "per_sym_min": de_per_sym_min, "per_sym_DE_ub": ub_sum,
            "per_sym_DE_thresh": psd}


def main():
    print("=== R_roll_tsrv DE+LOSO eval ===", flush=True)
    t0 = time.time()
    results = {}

    # V0 baseline (existing T75)
    base0, p0 = load_pred_avg(T75_PRED_DIR, "pred_T75", seeds=SEEDS)
    results["V0"] = evaluate("V0 baseline T75 LGB L2 (3 seeds avg)", base0, p0)

    # V1 clean target
    try:
        base1, p1 = load_pred_avg(WORK_DIR, "pred_V1", seeds=SEEDS)
        results["V1"] = evaluate("V1 clean-slope target", base1, p1)
    except FileNotFoundError as e:
        print(f"  V1 missing: {e}", flush=True)

    # V2 meta features
    try:
        base2, p2 = load_pred_avg(WORK_DIR, "pred_V2", seeds=SEEDS)
        results["V2"] = evaluate("V2 noise meta-features", base2, p2)
    except FileNotFoundError as e:
        print(f"  V2 missing: {e}", flush=True)

    # Comparison
    print("\n=== COMPARISON ===", flush=True)
    print(f"{'variant':10s}  {'LOSO':>8s}  {'per_sym_min':>12s}  {'ΔLOSO_vs_V0':>12s}", flush=True)
    if "V0" in results:
        v0 = results["V0"]["de_loso_sum"]
        for k in ("V0", "V1", "V2"):
            if k in results:
                r = results[k]
                d = r["de_loso_sum"] - v0
                print(f"{k:10s}  {r['de_loso_sum']:+8.4f}  {r['per_sym_min']:+12.4f}  {d:+12.4f}",
                      flush=True)

    # Light blends: V2 + V0 (avg) — since V2 is augmentation of baseline
    if "V0" in results and "V2" in results:
        # blend at avg-pred level
        for w_v2 in [0.5]:
            p_blend = (1.0 - w_v2) * p0 + w_v2 * p2
            results[f"blend_V0_V2_w{w_v2}"] = evaluate(
                f"blend V0+{w_v2}*V2", base0, p_blend)

    # Save
    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "elapsed_sec": time.time() - t0,
           "seeds": list(SEEDS),
           "results": results}
    with open(os.path.join(WORK_DIR, "de_loso_results.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nsaved -> {os.path.join(WORK_DIR, 'de_loso_results.json')}", flush=True)


if __name__ == "__main__":
    main()
