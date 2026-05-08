"""T143: 3-way ensemble eval (T87 NN + T75 L2 + T137 MAE) with DE 4D asym thresholds.

Configs:
  A: (1.0*T87 + 1.5*T75_L2) / 2.5  [baseline]
  B: (1.0*T87 + 1.5*MAE) / 2.5      [mae replaces L2]
  C: (1.0*T87 + 1.5*T75_L2 + 0.7*MAE) / 3.2  [3-way light]
  D: (1.0*T87 + 1.0*T75_L2 + 1.0*MAE) / 3.0  [3-way equal]
  E: (1.0*T87 + 0.7*T75_L2 + 1.5*MAE) / 3.2  [3-way mae heavy]
"""
from __future__ import annotations
import json, os, time
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
# HERE = .../experiments/T143_mae_ensemble_v3 → parent of parent is project root
EXPERIMENTS_DIR = os.path.dirname(HERE)
T87_DIR = os.path.join(EXPERIMENTS_DIR, "T87_spo_dfl")
T75_DIR = os.path.join(EXPERIMENTS_DIR, "T75_regression_dmid")
MAE_DIR = os.path.join(EXPERIMENTS_DIR, "T137_alt_robust_losses")

FEE = 0.0001
SEEDS = [1, 7, 13, 42, 100]
SYMS = [0, 1, 2, 3, 4]

# iter_018 v1 conformal wrapper params (from T127 results.json)
CONFORMAL = {
    "per_sym_beta": {0: 0.10, 1: 0.40, 2: 0.30, 3: 0.00, 4: 0.00},
    "per_sym_sigma": {
        0: 0.0002403901,
        1: 0.0004712397,
        2: 0.0004522216,
        3: 0.0004235249,
        4: 0.0004279811,
    },
    "default_beta_for_ood": 0.16,
    "default_sigma_for_ood": 0.0003998317,
}


def load_preds_avg(fpaths: list[str], col="pred_dmid_norm") -> pd.DataFrame:
    dfs = [pd.read_parquet(p) for p in fpaths]
    base = dfs[0].copy()
    arr = base[col].to_numpy(np.float64).copy()
    for d in dfs[1:]:
        arr += d[col].to_numpy(np.float64)
    base[col] = (arr / len(dfs)).astype(np.float32)
    return base


def load_t87() -> pd.DataFrame:
    paths = [os.path.join(T87_DIR, f"pred_T87_seed{s}_main.parquet") for s in SEEDS
             if os.path.exists(os.path.join(T87_DIR, f"pred_T87_seed{s}_main.parquet"))]
    print(f"  T87 seeds found: {[os.path.basename(p) for p in paths]}")
    return load_preds_avg(paths)


def load_t75() -> pd.DataFrame:
    paths = [os.path.join(T75_DIR, f"pred_T75_seed{s}.parquet") for s in SEEDS
             if os.path.exists(os.path.join(T75_DIR, f"pred_T75_seed{s}.parquet"))]
    print(f"  T75 L2 seeds found: {[os.path.basename(p) for p in paths]}")
    return load_preds_avg(paths)


def load_mae() -> pd.DataFrame:
    paths = [os.path.join(MAE_DIR, f"pred_T137_mae_seed{s}.parquet") for s in SEEDS
             if os.path.exists(os.path.join(MAE_DIR, f"pred_T137_mae_seed{s}.parquet"))]
    print(f"  MAE seeds found: {[os.path.basename(p) for p in paths]}")
    return load_preds_avg(paths)


def vectorized_pnl(actions, mp_t, mp_th):
    side = actions.astype(np.float64) - 1.0
    abs_side = np.abs(side)
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee = FEE * abs_side * np.abs((mp_th.astype(np.float64) + 1.0) + (mp_t.astype(np.float64) + 1.0))
    return (side * diff - fee) / (mp_t.astype(np.float64) + 1.0)


def ev_gate_asym(pred, thr_up, thr_dn):
    a = np.full(len(pred), 1, dtype=np.int8)
    a[pred > thr_up] = 2
    a[pred < -thr_dn] = 0
    return a


def de_search(folds, bounds, n_runs=5):
    pred_list = [f["pred"] for f in folds]
    mp_t_list = [f["mp_t"] for f in folds]
    mp_th_list = [f["mp_th"] for f in folds]

    def neg_loso(x):
        thr_up, thr_dn = x
        s = 0.0
        for pred, mp_t, mp_th in zip(pred_list, mp_t_list, mp_th_list):
            a = ev_gate_asym(pred, thr_up, thr_dn)
            s += vectorized_pnl(a, mp_t, mp_th).sum()
        return -float(s)

    runs = []
    for sd in range(n_runs):
        res = differential_evolution(
            neg_loso, bounds=bounds, seed=sd, maxiter=80, popsize=24,
            polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            updating="deferred", workers=1, init="sobol",
        )
        runs.append({"seed": sd, "thr_up": float(res.x[0]), "thr_dn": float(res.x[1]),
                     "loso_sum": float(-res.fun)})
    runs.sort(key=lambda r: r["loso_sum"], reverse=True)
    return runs


def eval_config(name, pred_arr, df_base, folds_meta):
    """Run DE LOSO search for a given pred_arr, return result dict."""
    print(f"\n--- Config {name} ---", flush=True)
    folds = []
    for sym in SYMS:
        mask = folds_meta["sym"] == sym
        if mask.sum() == 0:
            continue
        folds.append({
            "sym": sym,
            "pred": pred_arr[mask],
            "mp_t": folds_meta["mp_t"][mask],
            "mp_th": folds_meta["mp_th"][mask],
        })

    bounds = [(0.0, 0.004), (0.0, 0.004)]
    de_runs = de_search(folds, bounds, n_runs=5)
    best = de_runs[0]
    thr_up, thr_dn = best["thr_up"], best["thr_dn"]
    loso = best["loso_sum"]

    per_sym = []
    for f in folds:
        a = ev_gate_asym(f["pred"], thr_up, thr_dn)
        pnl = float(vectorized_pnl(a, f["mp_t"], f["mp_th"]).sum())
        n_act = int((a != 1).sum())
        per_sym.append({"sym": f["sym"], "pnl": round(pnl, 6), "n_active": n_act})
        print(f"  sym={f['sym']}  pnl={pnl:+.4f}  n_active={n_act:,}", flush=True)

    print(f"  LOSO={loso:+.4f}  thr_up={thr_up:.6f}  thr_dn={thr_dn:.6f}", flush=True)
    return {"no_conformal": round(loso, 6), "thr_up": thr_up, "thr_dn": thr_dn, "per_sym": per_sym}


def apply_conformal_and_eval(pred_arr, thr_up, thr_dn, folds_meta):
    """Apply iter_018 v1 conformal wrapper, return LOSO PnL."""
    sigma_dict = CONFORMAL["per_sym_sigma"]
    beta_dict = CONFORMAL["per_sym_beta"]
    total_pnl = 0.0
    per_sym = []

    for sym in SYMS:
        mask = folds_meta["sym"] == sym
        if mask.sum() == 0:
            continue
        pred_sym = pred_arr[mask]
        mp_t = folds_meta["mp_t"][mask]
        mp_th = folds_meta["mp_th"][mask]

        sigma = sigma_dict.get(sym, CONFORMAL["default_sigma_for_ood"])
        beta = beta_dict.get(sym, CONFORMAL["default_beta_for_ood"])
        band = beta * sigma

        # Conformal: abstain if |pred| < band
        a = ev_gate_asym(pred_sym, thr_up, thr_dn)
        abstain_mask = np.abs(pred_sym) < band
        a[abstain_mask] = 1  # abstain

        pnl = float(vectorized_pnl(a, mp_t, mp_th).sum())
        total_pnl += pnl
        per_sym.append({"sym": sym, "pnl": round(pnl, 6), "n_active": int((a != 1).sum())})
        print(f"  [conformal] sym={sym}  pnl={pnl:+.4f}  band={band:.6f}  n_active={int((a!=1).sum()):,}", flush=True)

    print(f"  [conformal] LOSO={total_pnl:+.4f}", flush=True)
    return {"conformal": round(total_pnl, 6), "per_sym_conformal": per_sym}


def main():
    print("Loading predictions...", flush=True)
    df_t87 = load_t87()
    df_t75 = load_t75()
    df_mae = load_mae()

    # Verify row alignment
    for df in [df_t75, df_mae]:
        assert df["sym"].equals(df_t87["sym"]), "sym mismatch"
        assert df["t"].equals(df_t87["t"]), "t mismatch"

    p87 = df_t87["pred_dmid_norm"].to_numpy(np.float64)
    p75 = df_t75["pred_dmid_norm"].to_numpy(np.float64)
    pmae = df_mae["pred_dmid_norm"].to_numpy(np.float64)
    sym_arr = df_t87["sym"].to_numpy()
    mp_t = df_t87["midprice_t"].to_numpy(np.float64)
    mp_th = df_t87["midprice_th"].to_numpy(np.float64)

    folds_meta = {"sym": sym_arr, "mp_t": mp_t, "mp_th": mp_th}
    print(f"  n={len(p87):,}  syms={np.unique(sym_arr)}", flush=True)

    # Define 5 configs
    configs = {
        "A_baseline_l2":      (1.0*p87 + 1.5*p75) / 2.5,
        "B_replace_l2_mae":   (1.0*p87 + 1.5*pmae) / 2.5,
        "C_3way_light":       (1.0*p87 + 1.5*p75 + 0.7*pmae) / 3.2,
        "D_3way_equal":       (1.0*p87 + 1.0*p75 + 1.0*pmae) / 3.0,
        "E_3way_mae_heavy":   (1.0*p87 + 0.7*p75 + 1.5*pmae) / 3.2,
    }

    results_no_conformal = {}
    for name, pred in configs.items():
        r = eval_config(name, pred, None, folds_meta)
        results_no_conformal[name] = r

    # Find best no-conformal config
    best_name = max(results_no_conformal, key=lambda k: results_no_conformal[k]["no_conformal"])
    best_r = results_no_conformal[best_name]
    print(f"\n=== Best no-conformal: {best_name}  LOSO={best_r['no_conformal']:+.4f} ===", flush=True)

    # Get pred array for best
    best_pred = configs[best_name]
    thr_up = best_r["thr_up"]
    thr_dn = best_r["thr_dn"]

    # Eval all with conformal (using best thresholds from each config's own DE)
    print("\n--- Applying conformal wrapper to all configs ---", flush=True)
    results_conformal = {}
    for name, pred in configs.items():
        tu = results_no_conformal[name]["thr_up"]
        td = results_no_conformal[name]["thr_dn"]
        print(f"\n  Config {name} (thr_up={tu:.6f}, thr_dn={td:.6f}):", flush=True)
        rc = apply_conformal_and_eval(pred, tu, td, folds_meta)
        results_conformal[name] = rc

    # Build combined result dict
    all_configs = {}
    for name in configs:
        all_configs[name] = {
            "no_conformal": results_no_conformal[name]["no_conformal"],
            "conformal": results_conformal[name]["conformal"],
            "thr_up": results_no_conformal[name]["thr_up"],
            "thr_dn": results_no_conformal[name]["thr_dn"],
            "per_sym": results_no_conformal[name]["per_sym"],
        }

    best_conformal_name = max(all_configs, key=lambda k: all_configs[k]["conformal"])
    best_conformal_loso = all_configs[best_conformal_name]["conformal"]

    # iter_018 v1 platform score for delta calc
    ITER018_V1_PLATFORM = 28.93

    print(f"\n{'='*70}", flush=True)
    print("=== SUMMARY ===", flush=True)
    print(f"{'config':>25s}  {'no_conf':>10s}  {'conformal':>10s}", flush=True)
    for name in configs:
        r = all_configs[name]
        print(f"  {name:>25s}  {r['no_conformal']:>+10.4f}  {r['conformal']:>+10.4f}", flush=True)
    print(f"\n  Best (conformal): {best_conformal_name}  LOSO={best_conformal_loso:+.4f}", flush=True)

    out = {
        "task": "T143: MAE ensemble eval + iter_019 v3_mae packaging",
        "configs": all_configs,
        "best_config": best_conformal_name,
        "best_loso_pnl": best_conformal_loso,
        "delta_vs_iter018_v1": round(best_conformal_loso - 41.49, 4),
        "v3_zip": None,
        "v3_md5": None,
        "v3_size_mb": None,
        "v3_files": None,
        "smoke_test_ok": None,
        "expected_platform_v3": f"iter_018_v1 +{ITER018_V1_PLATFORM} + (delta from MAE config * transmission ~0.75)",
    }

    out_path = os.path.join(HERE, "results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote partial results -> {out_path}", flush=True)

    # Return best thresholds for Part B
    best_thr_up = all_configs[best_conformal_name]["thr_up"]
    best_thr_dn = all_configs[best_conformal_name]["thr_dn"]
    print(f"\nBest config for packaging: {best_conformal_name}", flush=True)
    print(f"  thr_up={best_thr_up:.8f}  thr_dn={best_thr_dn:.8f}", flush=True)

    return out, best_conformal_name, best_thr_up, best_thr_dn


if __name__ == "__main__":
    main()
