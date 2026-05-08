"""T138 DE LOSO evaluation: mixup variants vs T75 baseline.

For each α variant: average pred across 2 seeds, run DE LOSO.
Also evaluate 6-seed ensemble (all α/seed combos pooled).
Compare to T75 baseline (5-seed ensemble).
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
T75_DIR = os.path.join(ROOT, "experiments", "T75_regression_dmid")

ALPHAS = [0.2, 0.4, 1.0]
SEEDS_PER_ALPHA = [42, 1]
T75_SEEDS = [1, 7, 42, 13, 100]
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001
T75_BASELINE_REF = 37.05
ITER018_REF = 41.49


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


def split_sym(df, pred):
    result = []
    for k in SYMS:
        mask = df["sym"].to_numpy() == k
        result.append({
            "sym": k,
            "pred": pred[mask],
            "mp_t": df["midprice_t"].to_numpy(np.float64)[mask],
            "mp_th": df["midprice_th"].to_numpy(np.float64)[mask],
        })
    return result


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


def de_optimize(obj, bounds=[(0.0, 0.004), (0.0, 0.004)], seeds=(0, 1, 2, 7, 42)):
    runs = []
    for sd in seeds:
        r = differential_evolution(obj, bounds=bounds, seed=sd,
                                   maxiter=80, popsize=24, polish=True,
                                   tol=1e-7, init="sobol")
        runs.append((float(-r.fun), float(r.x[0]), float(r.x[1])))
    runs.sort(key=lambda x: x[0], reverse=True)
    return runs[0]


def evaluate_preds(base_df, pred_arr, label):
    folds = split_sym(base_df, pred_arr)
    obj = make_obj(folds)
    s, tu, td = de_optimize(obj)
    per = []
    for f in folds:
        a = gate_asym(f["pred"], tu, td)
        per.append(float(vectorized_pnl(a, f["mp_t"], f["mp_th"]).sum()))
    print(f"  [{label:<32s}] LOSO={s:+.4f}  thr=({tu:.4e},{td:.4e})  "
          f"per={[round(x,2) for x in per]}", flush=True)
    return {"label": label, "thr_up": tu, "thr_dn": td,
            "de_sum": s, "per_sym": per}


def load_pred(path):
    return pd.read_parquet(path)


def avg_pred_files(paths):
    base = load_pred(paths[0])
    p = base["pred_dmid_norm"].to_numpy(np.float64).copy()
    for path in paths[1:]:
        df = load_pred(path)
        p += df["pred_dmid_norm"].to_numpy(np.float64)
    return base, p / len(paths)


def main():
    print("=== T138 DE LOSO evaluation ===", flush=True)
    t0 = time.time()
    results = {}

    # T75 baseline (5-seed ensemble)
    print("\n--- T75 baseline (5-seed ensemble) ---", flush=True)
    t75_paths = [os.path.join(T75_DIR, f"pred_T75_seed{s}.parquet") for s in T75_SEEDS]
    t75_paths_exist = [p for p in t75_paths if os.path.exists(p)]
    if t75_paths_exist:
        base_df, t75_pred = avg_pred_files(t75_paths_exist)
        t75_res = evaluate_preds(base_df, t75_pred, f"T75_5seed")
        results["t75_baseline"] = t75_res
    else:
        print("  T75 baseline preds not found!", flush=True)
        t75_res = {"de_sum": T75_BASELINE_REF}

    t75_loso = t75_res["de_sum"]

    # Per-alpha 2-seed ensembles
    print("\n--- T138 per-α 2-seed ensembles ---", flush=True)
    alpha_results = {}
    for alpha in ALPHAS:
        alpha_str = f"{alpha:.1f}".replace(".", "p")
        per_seed_pnl = []
        valid_paths = []

        for seed in SEEDS_PER_ALPHA:
            label = f"T138_a{alpha_str}_seed{seed}"
            path = os.path.join(HERE, f"pred_{label}.parquet")
            if os.path.exists(path):
                valid_paths.append(path)
                df_s = load_pred(path)
                res_s = evaluate_preds(df_s, df_s["pred_dmid_norm"].to_numpy(np.float64),
                                       f"a{alpha_str}_s{seed}")
                per_seed_pnl.append(res_s["de_sum"])
            else:
                print(f"  MISSING {path}", flush=True)

        if len(valid_paths) >= 1:
            base_df, avg_p = avg_pred_files(valid_paths)
            ens_res = evaluate_preds(base_df, avg_p, f"a{alpha_str}_{len(valid_paths)}seed_ens")
            alpha_results[f"alpha_{alpha:.1f}"] = {
                "seeds": SEEDS_PER_ALPHA[:len(valid_paths)],
                "per_seed_pnl": per_seed_pnl,
                "ensemble_2seed_loso": ens_res["de_sum"],
                "thr_up": ens_res["thr_up"],
                "thr_dn": ens_res["thr_dn"],
                "per_sym": ens_res["per_sym"],
                "delta_vs_t75": ens_res["de_sum"] - t75_loso,
            }
        else:
            print(f"  No valid predictions for α={alpha}", flush=True)

    # 6-seed ensemble (all 6 runs pooled)
    print("\n--- 6-seed mega-ensemble (all α×seed combos) ---", flush=True)
    all_paths = []
    for alpha in ALPHAS:
        alpha_str = f"{alpha:.1f}".replace(".", "p")
        for seed in SEEDS_PER_ALPHA:
            p = os.path.join(HERE, f"pred_T138_a{alpha_str}_seed{seed}.parquet")
            if os.path.exists(p):
                all_paths.append(p)
    if len(all_paths) >= 2:
        base_df, mega_p = avg_pred_files(all_paths)
        mega_res = evaluate_preds(base_df, mega_p, f"T138_6seed_mega_ens")
        results["t138_6seed_mega"] = {
            "n_models": len(all_paths),
            "de_sum": mega_res["de_sum"],
            "delta_vs_t75": mega_res["de_sum"] - t75_loso,
            "per_sym": mega_res["per_sym"],
        }

    # Summary
    print("\n=== T138 SUMMARY ===", flush=True)
    print(f"  T75 baseline:    LOSO = {t75_loso:+.4f}", flush=True)
    best_alpha = None
    best_loso = -999.0
    for k, v in alpha_results.items():
        loso = v["ensemble_2seed_loso"]
        delta = v["delta_vs_t75"]
        mark = "✅" if delta >= 0.5 else ("⚠️" if delta >= 0.2 else "❌")
        print(f"  {k:<14s}:  LOSO = {loso:+.4f}  Δ={delta:+.4f}  {mark}", flush=True)
        if loso > best_loso:
            best_loso = loso
            best_alpha = k

    if "t138_6seed_mega" in results:
        m = results["t138_6seed_mega"]
        print(f"  6-seed mega ens: LOSO = {m['de_sum']:+.4f}  Δ={m['delta_vs_t75']:+.4f}", flush=True)

    best_delta = best_loso - t75_loso if best_alpha else 0.0
    verdict = ("mixup helps (+>0.5 delta)" if best_delta >= 0.5
               else "mixup neutral (0.2-0.5)" if best_delta >= 0.2
               else "mixup hurts or no gain (<0.2)")
    print(f"\n  Best variant: {best_alpha}  LOSO={best_loso:+.4f}  Δ={best_delta:+.4f}", flush=True)
    print(f"  Verdict: {verdict}", flush=True)

    out = {
        "task": "T138 mixup augmentation on T75 L2 LGB",
        "t75_baseline_loso": t75_loso,
        "iter018_loso": ITER018_REF,
        "variants": alpha_results,
        "t138_6seed_mega": results.get("t138_6seed_mega", {}),
        "best_variant": best_alpha,
        "best_loso": best_loso,
        "delta_vs_t75": best_delta,
        "verdict": verdict,
        "elapsed_sec": time.time() - t0,
    }
    out_path = os.path.join(HERE, "results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote -> {out_path}  total={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
