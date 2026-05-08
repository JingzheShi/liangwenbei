"""T124: unified eval for R-T75L2-symaug (r2) and R-T75L2-5loso (r4).

Symaug variants: baseline, group_only, combined  (3 seeds: 1, 7, 42)
5-LOSO strategies:
  baseline   = T75 full-5sym 3-seed avg (uses symaug 'baseline' preds = same arch)
  strat_A    = avg(5 LOSO models)
  strat_B    = (sum(5 LOSO) + full) / 6   equal-weight 6-model
  strat_C    = 0.4 * avg_LOSO + 0.6 * full

Eval: DE 2D (thr_up, thr_dn) asym thresh, sum-of-per-sym PnL (LOSO-equiv).

Outputs: results.json + REPORT.md
"""
from __future__ import annotations
import json
import os
import time
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

ROOT = "/root/projects/liangwenbei_workdir/experiments"
SYMAUG_DIR = os.path.join(ROOT, "R_T75L2_symaug")
LOSO_DIR = os.path.join(ROOT, "R_T75L2_5loso")
HERE = os.path.join(ROOT, "T124_eval_r2_r4")

SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001

SYMAUG_VARIANTS = ("baseline", "group_only", "combined")
SYMAUG_SEEDS = (1, 7, 42)

LOSO_SEEDS = (7, 13, 42)
LOSO_K = (0, 1, 2, 3, 4)

T75_BASELINE_REF = 35.30  # from T122 (3-seed ens DE LOSO)


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
    return [{
        "sym": int(k),
        "pred": df.loc[df["sym"] == k, "pred_dmid_norm"].to_numpy(np.float64),
        "mp_t": df.loc[df["sym"] == k, "midprice_t"].to_numpy(np.float64),
        "mp_th": df.loc[df["sym"] == k, "midprice_th"].to_numpy(np.float64),
    } for k in SYMS]


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


def de_search(obj, bounds, seeds=(0, 1, 2, 7, 42), maxiter=50, popsize=16):
    runs = []
    for sd in seeds:
        r = differential_evolution(
            obj, bounds=bounds, seed=sd, maxiter=maxiter, popsize=popsize,
            polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            updating="deferred", workers=1, init="sobol",
        )
        runs.append((float(-r.fun), float(r.x[0]), float(r.x[1]), int(sd)))
    runs.sort(key=lambda x: x[0], reverse=True)
    return runs[0], runs


def eval_pred(base_df, p, label, de_seeds=(0, 7, 42)):
    df = base_df.copy()
    df["pred_dmid_norm"] = p.astype(np.float32)
    folds = split_sym(df)
    obj = make_obj(folds)
    best, _ = de_search(obj, [(0.0, 0.0040), (0.0, 0.0040)], seeds=de_seeds)
    s, tu, td, _ = best
    per = []
    for f in folds:
        a = gate_asym(f["pred"], tu, td)
        per.append(float(vectorized_pnl(a, f["mp_t"], f["mp_th"]).sum()))
    print(f"  [{label:<32s}] DE_LOSO={s:+.4f}  thr_up={tu:.4e} thr_dn={td:.4e}  "
          f"per_sym={[round(x,2) for x in per]}  min={min(per):+.4f}", flush=True)
    return {
        "label": label, "thr_up": tu, "thr_dn": td,
        "de_sum": float(s), "per_sym": per,
        "per_sym_min": float(min(per)),
    }


def avg_preds_symaug(variant):
    paths = [os.path.join(SYMAUG_DIR, f"pred_T75L2_{variant}_seed{s}.parquet")
             for s in SYMAUG_SEEDS]
    base = pd.read_parquet(paths[0])
    p = base["pred_dmid_norm"].to_numpy(np.float64).copy()
    for path in paths[1:]:
        df = pd.read_parquet(path)
        if not (df["sym"].equals(base["sym"]) and df["t"].equals(base["t"])
                and df["date"].equals(base["date"])):
            raise RuntimeError(f"row mismatch in {path}")
        p += df["pred_dmid_norm"].to_numpy(np.float64)
    return base, p / float(len(paths))


def avg_preds_loso(k):
    paths = [os.path.join(LOSO_DIR, f"pred_T75L2_loso{k}_seed{s}.parquet")
             for s in LOSO_SEEDS]
    base = pd.read_parquet(paths[0])
    p = base["pred_dmid_norm"].to_numpy(np.float64).copy()
    for path in paths[1:]:
        df = pd.read_parquet(path)
        if not (df["sym"].equals(base["sym"]) and df["t"].equals(base["t"])
                and df["date"].equals(base["date"])):
            raise RuntimeError(f"row mismatch in {path}")
        p += df["pred_dmid_norm"].to_numpy(np.float64)
    return base, p / float(len(paths))


def main():
    print("=== T124 unified eval (r2 sym-aug + r4 5-LOSO) ===", flush=True)
    t0 = time.time()
    out = {"task": "T124 eval r2 sym-aug + r4 5-LOSO",
           "t75_baseline_ref": T75_BASELINE_REF}

    # ---------------------------------------------------------- SYMAUG ----
    print("\n--- SYMAUG variants (3-seed ensembles) ---", flush=True)
    symaug_ens = {}
    base_for_loso = None
    p_full_baseline = None
    for v in SYMAUG_VARIANTS:
        base, p_avg = avg_preds_symaug(v)
        if v == "baseline":
            base_for_loso = base
            p_full_baseline = p_avg.copy()
        symaug_ens[v] = eval_pred(base, p_avg, f"symaug_{v}_3seed_ens")

    # Symaug deltas vs baseline
    base_de = symaug_ens["baseline"]["de_sum"]
    base_per = symaug_ens["baseline"]["per_sym"]
    symaug_deltas = {}
    print("\n--- SYMAUG deltas vs baseline ---", flush=True)
    for v in SYMAUG_VARIANTS:
        if v == "baseline":
            continue
        d_de = symaug_ens[v]["de_sum"] - base_de
        d_per = [symaug_ens[v]["per_sym"][i] - base_per[i] for i in range(len(SYMS))]
        d_min = symaug_ens[v]["per_sym_min"] - symaug_ens["baseline"]["per_sym_min"]
        symaug_deltas[v] = {
            "delta_de_loso": float(d_de),
            "delta_per_sym": d_per,
            "delta_per_sym_min": float(d_min),
            "delta_min_componentwise": float(min(d_per)),
            "passes_iter017_threshold": bool(d_de >= 0.5),
        }
        mark = "PASS" if d_de >= 0.5 else ("warn" if d_de >= 0.3 else "fail")
        print(f"  {v:>14s}: ΔDE={d_de:+.4f} [{mark}]  Δper_sym={[round(x,2) for x in d_per]}  "
              f"Δmin={d_min:+.4f}", flush=True)
    out["symaug_ensemble"] = symaug_ens
    out["symaug_deltas"] = symaug_deltas

    # Per-seed singles for symaug (faster: 2 DE seeds only)
    print("\n--- SYMAUG per-seed singles (2 DE seeds) ---", flush=True)
    symaug_singles = {v: {} for v in SYMAUG_VARIANTS}
    for v in SYMAUG_VARIANTS:
        for s in SYMAUG_SEEDS:
            df = pd.read_parquet(
                os.path.join(SYMAUG_DIR, f"pred_T75L2_{v}_seed{s}.parquet"))
            p = df["pred_dmid_norm"].to_numpy(np.float64)
            r = eval_pred(df, p, f"symaug_{v}_seed{s}", de_seeds=(0,))
            symaug_singles[v][str(s)] = r
    out["symaug_singles"] = symaug_singles

    # ----------------------------------------------------------- 5-LOSO ----
    print("\n--- 5-LOSO: avg per LOSO sym held out ---", flush=True)
    loso_avgs = []
    for k in LOSO_K:
        base_k, p_k = avg_preds_loso(k)
        # ensure row alignment with full baseline
        if (not base_k["sym"].equals(base_for_loso["sym"])
                or not base_k["t"].equals(base_for_loso["t"])
                or not base_k["date"].equals(base_for_loso["date"])):
            raise RuntimeError(f"LOSO-{k} row mismatch with full baseline")
        loso_avgs.append(p_k)
        print(f"  LOSO-{k}: pred mean={p_k.mean():+.6e} std={p_k.std():.6e}", flush=True)

    # Strategies
    p_full = p_full_baseline
    p_A = np.mean(np.stack(loso_avgs, axis=0), axis=0)
    p_B = (np.sum(np.stack(loso_avgs, axis=0), axis=0) + p_full) / 6.0
    p_C = 0.4 * p_A + 0.6 * p_full

    print("\n--- 5-LOSO strategy DE eval ---", flush=True)
    res_baseline_loso = eval_pred(base_for_loso, p_full, "loso_baseline_full")
    res_A = eval_pred(base_for_loso, p_A, "loso_strat_A_avg5LOSO")
    res_B = eval_pred(base_for_loso, p_B, "loso_strat_B_6model_eq")
    res_C = eval_pred(base_for_loso, p_C, "loso_strat_C_0.4_0.6")

    loso_results = {
        "baseline": res_baseline_loso,
        "strat_A": res_A,
        "strat_B": res_B,
        "strat_C": res_C,
    }
    out["loso_results"] = loso_results

    # 5-LOSO deltas
    base_de = res_baseline_loso["de_sum"]
    base_per = res_baseline_loso["per_sym"]
    base_min = res_baseline_loso["per_sym_min"]
    loso_deltas = {}
    print("\n--- 5-LOSO deltas vs baseline (full-5sym) ---", flush=True)
    for name in ("strat_A", "strat_B", "strat_C"):
        r = loso_results[name]
        d_de = r["de_sum"] - base_de
        d_per = [r["per_sym"][i] - base_per[i] for i in range(len(SYMS))]
        d_min = r["per_sym_min"] - base_min
        loso_deltas[name] = {
            "delta_de_loso": float(d_de),
            "delta_per_sym": d_per,
            "delta_per_sym_min": float(d_min),
            "passes_iter017_threshold": bool(d_de >= 0.5),
        }
        mark = "PASS" if d_de >= 0.5 else ("warn" if d_de >= 0.3 else "fail")
        print(f"  {name:>10s}: ΔDE={d_de:+.4f} [{mark}]  Δper_sym={[round(x,2) for x in d_per]}  "
              f"Δmin={d_min:+.4f}", flush=True)
    out["loso_deltas"] = loso_deltas

    # ---------------------------------------------------- WINS for iter017 v2
    wins = []
    for v, d in symaug_deltas.items():
        if d["passes_iter017_threshold"]:
            wins.append({"src": "symaug", "variant": v,
                         "delta_de_loso": d["delta_de_loso"],
                         "delta_per_sym_min": d["delta_per_sym_min"]})
    for n, d in loso_deltas.items():
        if d["passes_iter017_threshold"]:
            wins.append({"src": "5loso", "strategy": n,
                         "delta_de_loso": d["delta_de_loso"],
                         "delta_per_sym_min": d["delta_per_sym_min"]})
    out["wins_for_iter017_v2"] = wins

    print("\n=== SUMMARY ===", flush=True)
    print(f"  symaug baseline = {symaug_ens['baseline']['de_sum']:+.4f}", flush=True)
    print(f"  symaug group    = {symaug_ens['group_only']['de_sum']:+.4f}", flush=True)
    print(f"  symaug combined = {symaug_ens['combined']['de_sum']:+.4f}", flush=True)
    print(f"  loso baseline   = {res_baseline_loso['de_sum']:+.4f}  "
          f"(min_per_sym={res_baseline_loso['per_sym_min']:+.4f})", flush=True)
    print(f"  loso strat_A    = {res_A['de_sum']:+.4f}  "
          f"(min_per_sym={res_A['per_sym_min']:+.4f})", flush=True)
    print(f"  loso strat_B    = {res_B['de_sum']:+.4f}  "
          f"(min_per_sym={res_B['per_sym_min']:+.4f})", flush=True)
    print(f"  loso strat_C    = {res_C['de_sum']:+.4f}  "
          f"(min_per_sym={res_C['per_sym_min']:+.4f})", flush=True)
    print(f"  WINS for iter017 v2 (>=+0.5 over their baseline): {len(wins)}", flush=True)
    for w in wins:
        print(f"    {w}", flush=True)

    out["elapsed_sec"] = time.time() - t0
    out_path = os.path.join(HERE, "results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote -> {out_path}  total={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
