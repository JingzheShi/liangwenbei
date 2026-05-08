"""T144: Quantile + Huber multi-loss ensemble diversity vs MAE-only.

Eval configs B, F1-F5 using DE 2D asym threshold LOSO.
DE budget: maxiter=50, popsize=16, n_runs=3 (reduced vs T137 to avoid OOM).
"""
from __future__ import annotations

import gc
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

WORKDIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(WORKDIR, "..")
T87_DIR = os.path.join(BASE, "T87_spo_dfl")
T75_DIR = os.path.join(BASE, "T75_regression_dmid")
T137_DIR = os.path.join(BASE, "T137_alt_robust_losses")

SEEDS = [1, 7, 13, 42, 100]
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001

# T127 conformal wrapper params
T127_THR_UP = 0.00029995433796130955
T127_THR_DN = 0.0002158528296175958
T127_PER_SYM_BETA = {0: 0.10, 1: 0.40, 2: 0.30, 3: 0.00, 4: 0.00}
T127_PER_SYM_SIGMA = {
    0: 0.0002403901, 1: 0.0004712397, 2: 0.0004522216,
    3: 0.0004235249, 4: 0.0004279811,
}
T127_PER_SYM_BAND = {s: T127_PER_SYM_BETA[s] * T127_PER_SYM_SIGMA[s] for s in SYMS}

T87_MAE_KNOWN = 41.20   # from T137 experiment
ITER018_V1_BASELINE = 40.13


def load_pred_avg(paths):
    """Load and average pred_dmid_norm across parquet files."""
    dfs = [pd.read_parquet(p) for p in paths]
    base = dfs[0]
    psum = base["pred_dmid_norm"].to_numpy(np.float64).copy()
    for df in dfs[1:]:
        if not (df["sym"].equals(base["sym"]) and df["date"].equals(base["date"])
                and df["t"].equals(base["t"])):
            raise RuntimeError(f"Row mismatch: {paths}")
        psum += df["pred_dmid_norm"].to_numpy(np.float64)
    return psum / len(dfs), base


def vectorized_pnl(pred_action, mp_t, mp_th):
    side = pred_action.astype(np.float64) - 1.0
    abs_side = np.abs(side)
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * abs_side * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    return (side * diff - fee_pnl) / (mp_t.astype(np.float64) + 1.0)


def ev_gate_asym(pred, thr_up, thr_dn):
    a = np.full(len(pred), 1, dtype=np.int8)
    a[pred > thr_up] = 2
    a[pred < -thr_dn] = 0
    return a


def ev_gate_conformal(pred, sym_arr, thr_up, thr_dn):
    """Apply per-sym band on top of base thresholds."""
    a = np.full(len(pred), 1, dtype=np.int8)
    for s in SYMS:
        band = T127_PER_SYM_BAND[s]
        mask = sym_arr == s
        a[mask & (pred > thr_up + band)] = 2
        a[mask & (pred < -(thr_dn + band))] = 0
    return a


def de_search_loso(folds, bounds, n_runs=3, maxiter=50, popsize=16):
    pred_list = [f["pred"] for f in folds]
    mp_t_list = [f["mp_t"] for f in folds]
    mp_th_list = [f["mp_th"] for f in folds]

    def neg_loso(x):
        thr_up, thr_dn = x
        s = 0.0
        for i in range(len(folds)):
            a = ev_gate_asym(pred_list[i], thr_up, thr_dn)
            s += vectorized_pnl(a, mp_t_list[i], mp_th_list[i]).sum()
        return -float(s)

    runs = []
    for sd in range(n_runs):
        t0 = time.time()
        res = differential_evolution(
            neg_loso, bounds=bounds, seed=sd, maxiter=maxiter, popsize=popsize,
            polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            updating="deferred", workers=1, init="sobol",
        )
        runs.append({
            "seed": sd, "thr_up": float(res.x[0]), "thr_dn": float(res.x[1]),
            "loso_sum": float(-res.fun), "nfev": int(res.nfev),
            "elapsed": float(time.time() - t0),
        })
    runs.sort(key=lambda r: r["loso_sum"], reverse=True)
    return runs


def eval_config(name, pred_all, base_df, sym_arr, mp_t_all, mp_th_all):
    print(f"\n{'='*60}\n=== config={name} ===", flush=True)

    folds = []
    for sym in SYMS:
        mask = sym_arr == sym
        if mask.sum() == 0:
            continue
        folds.append({
            "sym": int(sym),
            "pred": pred_all[mask],
            "mp_t": mp_t_all[mask],
            "mp_th": mp_th_all[mask],
        })

    bounds = [(0.0, 0.004), (0.0, 0.004)]
    print(f"  Running DE LOSO (maxiter=50, popsize=16, n_runs=3) ...", flush=True)
    de_runs = de_search_loso(folds, bounds, n_runs=3, maxiter=50, popsize=16)
    best_de = de_runs[0]
    thr_up_best = best_de["thr_up"]
    thr_dn_best = best_de["thr_dn"]
    de_loso = best_de["loso_sum"]

    per_sym = []
    for f in folds:
        a = ev_gate_asym(f["pred"], thr_up_best, thr_dn_best)
        pnl = float(vectorized_pnl(a, f["mp_t"], f["mp_th"]).sum())
        n_act = int((a != 1).sum())
        per_sym.append({"sym": f["sym"], "pnl": pnl, "n_active": n_act})
        print(f"  sym={f['sym']}  pnl={pnl:+.4f}  n_active={n_act:,}", flush=True)

    print(f"\n  RESULT config={name}: DE_LOSO={de_loso:+.4f}  "
          f"thr_up={thr_up_best:.6f}  thr_dn={thr_dn_best:.6f}", flush=True)

    result = {
        "de_loso": de_loso,
        "thr_up": thr_up_best,
        "thr_dn": thr_dn_best,
        "per_sym": per_sym,
        "de_runs": de_runs,
        "vs_iter018_v1": de_loso - ITER018_V1_BASELINE,
        "vs_T87_MAE": de_loso - T87_MAE_KNOWN,
    }
    return result


def eval_conformal(name, pred_all, sym_arr, mp_t_all, mp_th_all, thr_up, thr_dn):
    """Evaluate best config with T127 conformal wrapper."""
    print(f"\n=== conformal eval for {name} ===", flush=True)
    a_all = ev_gate_conformal(pred_all, sym_arr, thr_up, thr_dn)
    total_pnl = float(vectorized_pnl(a_all, mp_t_all, mp_th_all).sum())
    n_active = int((a_all != 1).sum())
    print(f"  conformal LOSO={total_pnl:+.4f}  n_active={n_active:,}", flush=True)

    per_sym = []
    for sym in SYMS:
        mask = sym_arr == sym
        if mask.sum() == 0:
            continue
        pnl = float(vectorized_pnl(a_all[mask], mp_t_all[mask], mp_th_all[mask]).sum())
        per_sym.append({"sym": int(sym), "pnl": pnl})
        print(f"  sym={sym}  pnl={pnl:+.4f}  band={T127_PER_SYM_BAND[sym]:.2e}", flush=True)

    return {"loso": total_pnl, "per_sym": per_sym, "n_active": n_active}


def main():
    # --- Load all base predictions ---
    print("Loading predictions...", flush=True)
    t87_paths = [os.path.join(T87_DIR, f"pred_T87_seed{s}_main.parquet") for s in SEEDS]
    t75_paths = [os.path.join(T75_DIR, f"pred_T75_seed{s}.parquet") for s in SEEDS]
    mae_paths = [os.path.join(T137_DIR, f"pred_T137_mae_seed{s}.parquet") for s in SEEDS]
    quant_paths = [os.path.join(T137_DIR, f"pred_T137_quantile_seed{s}.parquet") for s in SEEDS]
    huber_paths = [os.path.join(T137_DIR, f"pred_T137_huber_seed{s}.parquet") for s in SEEDS]

    p_t87, base_df = load_pred_avg(t87_paths)
    p_t75, _ = load_pred_avg(t75_paths)
    p_mae, _ = load_pred_avg(mae_paths)
    p_quant, _ = load_pred_avg(quant_paths)
    p_huber, _ = load_pred_avg(huber_paths)
    print(f"  All loaded. N={len(p_t87):,}", flush=True)

    sym_arr = base_df["sym"].to_numpy()
    mp_t_all = base_df["midprice_t"].to_numpy(np.float64)
    mp_th_all = base_df["midprice_th"].to_numpy(np.float64)

    # --- Cross-correlations ---
    print("\nCross-correlations:", flush=True)
    names = ["T87", "L2", "MAE", "Quantile", "Huber"]
    preds_list = [p_t87, p_t75, p_mae, p_quant, p_huber]
    cross_corr = {}
    for i, ni in enumerate(names):
        for j in range(i + 1, len(names)):
            c = float(np.corrcoef(preds_list[i], preds_list[j])[0, 1])
            key = f"{ni}-{names[j]}"
            cross_corr[key] = round(c, 6)
            print(f"  corr({ni}, {names[j]}) = {c:.4f}", flush=True)

    # --- Build ensemble preds for each config ---
    configs_def = {
        "B_T87_MAE":     (p_t87 * 1.0 + p_mae * 1.5) / 2.5,
        "F1_4way_light": (p_t87 * 1.0 + p_t75 * 1.0 + p_mae * 1.0 + p_quant * 0.5) / 3.5,
        "F2_3way_no_L2": (p_t87 * 1.0 + p_mae * 1.5 + p_quant * 0.7) / 3.2,
        "F3_T87_MAE_Huber": (p_t87 * 1.0 + p_mae * 1.5 + p_huber * 0.5) / 3.0,
        "F4_5way_equal": (p_t87 * 1.0 + p_t75 * 1.0 + p_mae * 1.0 + p_quant * 1.0 + p_huber * 1.0) / 5.0,
        "F5_5way_mae_heavy": (p_t87 * 1.0 + p_t75 * 0.7 + p_mae * 1.5 + p_quant * 0.7 + p_huber * 0.7) / 4.6,
    }

    # Free base arrays after building configs
    del p_t87, p_t75, p_mae, p_quant, p_huber
    gc.collect()

    # --- Evaluate each config ---
    config_results = {}
    for cfg_name, pred_ens in configs_def.items():
        r = eval_config(cfg_name, pred_ens, base_df, sym_arr, mp_t_all, mp_th_all)
        config_results[cfg_name] = r
        del pred_ens
        gc.collect()

    # --- Summary ---
    print(f"\n{'='*70}", flush=True)
    print(f"=== T144 SUMMARY ===", flush=True)
    print(f"{'config':>22s}  {'DE_LOSO':>10s}  {'vs_baseline':>12s}  {'vs_T87_MAE':>11s}", flush=True)
    for cfg_name, r in config_results.items():
        print(f"  {cfg_name:>22s}  {r['de_loso']:>+10.4f}  "
              f"{r['vs_iter018_v1']:>+12.4f}  {r['vs_T87_MAE']:>+11.4f}", flush=True)

    best_cfg = max(config_results, key=lambda k: config_results[k]["de_loso"])
    best_loso = config_results[best_cfg]["de_loso"]
    print(f"\nBest: {best_cfg} = {best_loso:+.4f}", flush=True)

    # --- Conformal wrapper (if best > +41.20) ---
    conformal_result = None
    if best_loso > T87_MAE_KNOWN:
        print(f"\nBest ({best_loso:+.4f}) > T87+MAE ({T87_MAE_KNOWN:+.2f}) → running conformal eval", flush=True)
        # Rebuild best config pred
        configs_def2 = {
            "B_T87_MAE":     None,
            "F1_4way_light": None,
            "F2_3way_no_L2": None,
            "F3_T87_MAE_Huber": None,
            "F4_5way_equal": None,
            "F5_5way_mae_heavy": None,
        }
        # Re-load base preds for best config only
        print("  Re-loading preds for conformal eval...", flush=True)
        p_t87_c, _ = load_pred_avg(t87_paths)
        p_t75_c, _ = load_pred_avg(t75_paths)
        p_mae_c, _ = load_pred_avg(mae_paths)
        p_quant_c, _ = load_pred_avg(quant_paths)
        p_huber_c, _ = load_pred_avg(huber_paths)

        rebuild_map = {
            "B_T87_MAE":     (p_t87_c * 1.0 + p_mae_c * 1.5) / 2.5,
            "F1_4way_light": (p_t87_c * 1.0 + p_t75_c * 1.0 + p_mae_c * 1.0 + p_quant_c * 0.5) / 3.5,
            "F2_3way_no_L2": (p_t87_c * 1.0 + p_mae_c * 1.5 + p_quant_c * 0.7) / 3.2,
            "F3_T87_MAE_Huber": (p_t87_c * 1.0 + p_mae_c * 1.5 + p_huber_c * 0.5) / 3.0,
            "F4_5way_equal": (p_t87_c * 1.0 + p_t75_c * 1.0 + p_mae_c * 1.0 + p_quant_c * 1.0 + p_huber_c * 1.0) / 5.0,
            "F5_5way_mae_heavy": (p_t87_c * 1.0 + p_t75_c * 0.7 + p_mae_c * 1.5 + p_quant_c * 0.7 + p_huber_c * 0.7) / 4.6,
        }
        best_pred_ens = rebuild_map[best_cfg]
        del p_t87_c, p_t75_c, p_mae_c, p_quant_c, p_huber_c
        gc.collect()

        thr_up_best = config_results[best_cfg]["thr_up"]
        thr_dn_best = config_results[best_cfg]["thr_dn"]
        conformal_result = eval_conformal(
            best_cfg, best_pred_ens, sym_arr, mp_t_all, mp_th_all,
            thr_up_best, thr_dn_best,
        )
        del best_pred_ens
        gc.collect()

    # --- Verdict ---
    delta_vs_mae = best_loso - T87_MAE_KNOWN
    delta_vs_baseline = best_loso - ITER018_V1_BASELINE
    if best_loso > 41.5:
        verdict = f"SHIP as iter_019 v4 — multi-loss diversity beats MAE-only by +{delta_vs_mae:.2f}"
    elif best_loso > T87_MAE_KNOWN:
        verdict = f"MARGINAL improvement over T87+MAE (+{delta_vs_mae:.2f}) — recommend if DE confirms consistently"
    else:
        verdict = f"MAE-only (T87+MAE) remains winner — no v4 needed (best={best_loso:+.4f}, delta={delta_vs_mae:+.4f})"

    # --- Write results.json ---
    out = {
        "task": "T144 Quantile/Huber multi-loss ensemble",
        "cross_correlations": cross_corr,
        "configs": {},
        "best_config": best_cfg,
        "best_loso_pnl": round(best_loso, 4),
        "delta_vs_iter018_v1_baseline": round(delta_vs_baseline, 4),
        "delta_vs_T87_MAE": round(delta_vs_mae, 4),
        "verdict": verdict,
    }

    weights_map = {
        "B_T87_MAE": "1:1.5",
        "F1_4way_light": "1:1:1:0.5",
        "F2_3way_no_L2": "1:1.5:0.7",
        "F3_T87_MAE_Huber": "1:1.5:0.5",
        "F4_5way_equal": "1:1:1:1:1",
        "F5_5way_mae_heavy": "1:0.7:1.5:0.7:0.7",
    }
    for cfg_name, r in config_results.items():
        out["configs"][cfg_name] = {
            "loso_pnl": round(r["de_loso"], 4),
            "thr_up": round(r["thr_up"], 8),
            "thr_dn": round(r["thr_dn"], 8),
            "vs_iter018_v1": round(r["vs_iter018_v1"], 4),
            "vs_T87_MAE": round(r["vs_T87_MAE"], 4),
            "weights": weights_map.get(cfg_name, ""),
            "per_sym": r["per_sym"],
        }

    if conformal_result is not None:
        out["conformal_wrapper"] = {
            "applied_to": best_cfg,
            "loso_pnl": round(conformal_result["loso"], 4),
            "n_active": conformal_result["n_active"],
            "per_sym": conformal_result["per_sym"],
            "per_sym_band": {str(k): round(v, 10) for k, v in T127_PER_SYM_BAND.items()},
        }

    out_path = os.path.join(WORKDIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote -> {out_path}", flush=True)

    print(f"\n{'='*70}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)
    print(f"\nRESULT: task=[T144 Quantile/Huber ensemble diversity] "
          f"metrics={{best_loso_pnl={best_loso:.2f}, delta_vs_mae={delta_vs_mae:+.2f}, best_config={best_cfg}}} "
          f"notes=[{verdict}]")


if __name__ == "__main__":
    main()
