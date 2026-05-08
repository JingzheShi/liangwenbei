"""T128: conformal_v2 — explore better β strategies for iter_015 v1 stack abstain.

Baseline to beat: iter_018 v1 in-sample β = {0:0.1, 1:0.4, 2:0.3, 3:0, 4:0} → +41.4939 LOSO.
Existing 4-fold CV (V4_robustness): +41.26 OOF.

Variants:
  V1: finer β grid (0..0.50 step 0.05), 4-fold time-block CV
  V2: empirical win-rate calibration (smallest β such that calib win-rate ≥ target)
  V3: asymmetric β (β_long, β_short per sym), OOF 4-fold CV grid search
  V4: sigma-aware β (β_eff = β · max(1, σ_local / σ_baseline)) — research only (rolling window)

All variants use OOF CV-tuned β. Threshold thr_up/thr_dn fixed at iter_015 v1 DE optimum.
"""
from __future__ import annotations
import json
import os
import time
import numpy as np
import pandas as pd

EXP_DIR = "/root/projects/liangwenbei_workdir/experiments/T128_conformal_v2"
T87_DIR = "/root/projects/liangwenbei_workdir/experiments/T87_spo_dfl"
T75_DIR = "/root/projects/liangwenbei_workdir/experiments/T75_regression_dmid"
SEEDS = (1, 7, 13, 42, 100)
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001

# DE-tuned baseline thresholds (from R_conformal_select)
THR_UP = 0.00029995433796130955
THR_DN = 0.0002158528296175958

# iter_018 v1 baseline (in-sample fit β)
ITER018_BASELINE = 41.4939
NO_ABSTAIN_BASELINE = 39.7467


def vectorized_pnl(action, mp_t, mp_th):
    side = action.astype(np.float64) - 1.0
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee_pnl) / denom


def gate_symmetric_band(pred, thr_up, thr_dn, band_up, band_dn):
    a = np.full(len(pred), 1, dtype=np.int8)
    a[pred > thr_up + band_up] = 2
    a[pred < -(thr_dn + band_dn)] = 0
    return a


def split_by_sym(df, pred_col="pred_dmid_norm"):
    out = []
    for k in SYMS:
        sub = df[df["sym"] == k].reset_index(drop=True)
        # ensure date-sorted within sym
        order = np.argsort(sub["date"].to_numpy(), kind="mergesort")
        out.append({
            "sym": int(k),
            "pred": sub[pred_col].to_numpy(np.float64)[order],
            "true_dmid": sub["true_dmid_norm"].to_numpy(np.float64)[order],
            "mp_t": sub["midprice_t"].to_numpy(np.float64)[order],
            "mp_th": sub["midprice_th"].to_numpy(np.float64)[order],
            "date": sub["date"].to_numpy(np.int64)[order],
            "n": len(sub),
        })
    return out


def load_avg(paths):
    base = pd.read_parquet(paths[0])
    p = np.zeros(len(base), dtype=np.float64)
    for path in paths:
        df = pd.read_parquet(path)
        p += df["pred_dmid_norm"].to_numpy(np.float64)
    p /= len(paths)
    base = base.copy()
    base["pred_dmid_norm"] = p.astype(np.float32)
    return base


def per_sym_metrics(folds, actions_per_sym):
    out = []
    actives, wins = [], []
    for f, a in zip(folds, actions_per_sym):
        pnl = vectorized_pnl(a, f["mp_t"], f["mp_th"])
        out.append(float(pnl.sum()))
        n_active = int((a != 1).sum())
        actives.append(n_active / len(a) if len(a) > 0 else 0.0)
        active_mask = (a != 1)
        wins.append(float((pnl[active_mask] > 0).mean()) if active_mask.sum() > 0 else 0.0)
    return out, actives, wins


def k_date_folds(date_arr, k):
    """Return list of boolean masks (len k) for k contiguous date folds."""
    unique_dates = np.sort(np.unique(date_arr))
    n_d = len(unique_dates)
    out = []
    for i in range(k):
        lo = i * n_d // k
        hi = (i + 1) * n_d // k
        fold_dates = set(unique_dates[lo:hi].tolist())
        mask = np.array([d in fold_dates for d in date_arr])
        out.append(mask)
    return out


# =====================================================================
# Variant 1: Finer β grid, symmetric (single β per sym)
# =====================================================================
def variant1_fine_grid(folds, beta_grid, k):
    actions_per_sym = []
    chosen_betas = []
    for f in folds:
        sigma = float(np.std(f["pred"]))
        date_folds = k_date_folds(f["date"], k)
        action = np.full(f["n"], 1, dtype=np.int8)
        betas_per_fold = []
        for fold_idx in range(k):
            test_mask = date_folds[fold_idx]
            calib_mask = ~test_mask
            best_b, best_pnl = 0.0, -1e18
            for b in beta_grid:
                a = gate_symmetric_band(f["pred"][calib_mask], THR_UP, THR_DN,
                                        b * sigma, b * sigma)
                pnl = vectorized_pnl(a, f["mp_t"][calib_mask], f["mp_th"][calib_mask]).sum()
                if pnl > best_pnl:
                    best_pnl = pnl
                    best_b = b
            betas_per_fold.append(best_b)
            action[test_mask] = gate_symmetric_band(
                f["pred"][test_mask], THR_UP, THR_DN, best_b * sigma, best_b * sigma)
        chosen_betas.append(betas_per_fold)
        actions_per_sym.append(action)
    return actions_per_sym, chosen_betas


# =====================================================================
# Variant 2: Win-rate based calibration
# Find smallest β such that calib win-rate ≥ target_winrate.
# =====================================================================
def variant2_winrate(folds, beta_grid, k, target_winrate):
    actions_per_sym = []
    chosen_betas = []
    for f in folds:
        sigma = float(np.std(f["pred"]))
        date_folds = k_date_folds(f["date"], k)
        action = np.full(f["n"], 1, dtype=np.int8)
        betas_per_fold = []
        for fold_idx in range(k):
            test_mask = date_folds[fold_idx]
            calib_mask = ~test_mask
            chosen = beta_grid[-1]  # default to most aggressive abstain
            best_pnl_at_target = -1e18
            best_b_fallback = 0.0
            best_pnl_fallback = -1e18
            for b in beta_grid:
                a = gate_symmetric_band(f["pred"][calib_mask], THR_UP, THR_DN,
                                        b * sigma, b * sigma)
                pnl = vectorized_pnl(a, f["mp_t"][calib_mask], f["mp_th"][calib_mask])
                active_mask = (a != 1)
                if active_mask.sum() == 0:
                    continue
                wr = float((pnl[active_mask] > 0).mean())
                tot = float(pnl.sum())
                if tot > best_pnl_fallback:
                    best_pnl_fallback = tot
                    best_b_fallback = b
                if wr >= target_winrate and tot > best_pnl_at_target:
                    best_pnl_at_target = tot
                    chosen = b
            if best_pnl_at_target == -1e18:
                # No β hits target — use best by PnL
                chosen = best_b_fallback
            betas_per_fold.append(chosen)
            action[test_mask] = gate_symmetric_band(
                f["pred"][test_mask], THR_UP, THR_DN, chosen * sigma, chosen * sigma)
        chosen_betas.append(betas_per_fold)
        actions_per_sym.append(action)
    return actions_per_sym, chosen_betas


# =====================================================================
# Variant 3: Asymmetric β (long vs short per sym)
# =====================================================================
def variant3_asymmetric(folds, beta_grid, k):
    actions_per_sym = []
    chosen_betas_long = []
    chosen_betas_short = []
    for f in folds:
        sigma = float(np.std(f["pred"]))
        date_folds = k_date_folds(f["date"], k)
        action = np.full(f["n"], 1, dtype=np.int8)
        bL_per_fold, bS_per_fold = [], []
        for fold_idx in range(k):
            test_mask = date_folds[fold_idx]
            calib_mask = ~test_mask
            # Greedy: search β_long first holding β_short=0, then β_short holding β_long fixed.
            # For 5x5 grid with ~11 values each, full search is 121 evals × 5 folds × 5 syms = 3025.
            # Affordable. Do full grid.
            best_bL, best_bS, best_pnl = 0.0, 0.0, -1e18
            for bL in beta_grid:
                for bS in beta_grid:
                    a = gate_symmetric_band(f["pred"][calib_mask], THR_UP, THR_DN,
                                            bL * sigma, bS * sigma)
                    pnl = vectorized_pnl(a, f["mp_t"][calib_mask], f["mp_th"][calib_mask]).sum()
                    if pnl > best_pnl:
                        best_pnl = pnl
                        best_bL, best_bS = bL, bS
            bL_per_fold.append(best_bL)
            bS_per_fold.append(best_bS)
            action[test_mask] = gate_symmetric_band(
                f["pred"][test_mask], THR_UP, THR_DN, best_bL * sigma, best_bS * sigma)
        chosen_betas_long.append(bL_per_fold)
        chosen_betas_short.append(bS_per_fold)
        actions_per_sym.append(action)
    return actions_per_sym, chosen_betas_long, chosen_betas_short


# =====================================================================
# Variant 4: Sigma-aware β
# β_eff(row) = β · max(1, σ_local(row) / σ_baseline)
# σ_local = rolling 200-bar std of pred (within sym, by date order)
# σ_baseline = global per-sym σ_pred
# Note: This requires looking at neighbors of test rows → research-only since
# Predictor.py is stateless. But we can simulate it OOF by computing sigma
# across the FULL data first (calib + test) then applying.
# =====================================================================
def rolling_std(x, win):
    """Centered rolling std using pd."""
    s = pd.Series(x)
    return s.rolling(win, min_periods=10, center=True).std().fillna(s.std()).to_numpy()


def variant4_sigma_aware(folds, beta_grid, k, win=200):
    actions_per_sym = []
    chosen_betas = []
    for f in folds:
        sigma_baseline = float(np.std(f["pred"]))
        sigma_local = rolling_std(f["pred"], win)
        # Multiplier: max(1, sigma_local / sigma_baseline)
        mult = np.maximum(1.0, sigma_local / max(sigma_baseline, 1e-9))
        date_folds = k_date_folds(f["date"], k)
        action = np.full(f["n"], 1, dtype=np.int8)
        betas_per_fold = []
        for fold_idx in range(k):
            test_mask = date_folds[fold_idx]
            calib_mask = ~test_mask
            best_b, best_pnl = 0.0, -1e18
            for b in beta_grid:
                # Per-row band = b * sigma_baseline * mult[row]
                bands_calib = b * sigma_baseline * mult[calib_mask]
                pred_c = f["pred"][calib_mask]
                a = np.full(len(pred_c), 1, dtype=np.int8)
                a[pred_c > THR_UP + bands_calib] = 2
                a[pred_c < -(THR_DN + bands_calib)] = 0
                pnl = vectorized_pnl(a, f["mp_t"][calib_mask], f["mp_th"][calib_mask]).sum()
                if pnl > best_pnl:
                    best_pnl = pnl
                    best_b = b
            betas_per_fold.append(best_b)
            bands_test = best_b * sigma_baseline * mult[test_mask]
            pred_t = f["pred"][test_mask]
            a = np.full(len(pred_t), 1, dtype=np.int8)
            a[pred_t > THR_UP + bands_test] = 2
            a[pred_t < -(THR_DN + bands_test)] = 0
            action[test_mask] = a
        chosen_betas.append(betas_per_fold)
        actions_per_sym.append(action)
    return actions_per_sym, chosen_betas


def evaluate(folds, actions, label):
    per_sym, actives, wins = per_sym_metrics(folds, actions)
    total = float(sum(per_sym))
    return {
        "label": label,
        "total": total,
        "per_sym": per_sym,
        "actives": actives,
        "wins": wins,
        "min_sym": float(min(per_sym)),
        "active_rate": float(np.mean(actives)),
        "win_rate": float(np.mean(wins)),
    }


def main():
    t0 = time.time()
    print("Loading T87 + T75 5-seed avg preds…", flush=True)
    t87_paths = [os.path.join(T87_DIR, f"pred_T87_seed{s}_main.parquet") for s in SEEDS]
    df_t87 = load_avg(t87_paths)
    t75_paths = [os.path.join(T75_DIR, f"pred_T75_seed{s}.parquet") for s in SEEDS]
    df_t75 = load_avg(t75_paths)
    p_combined = (1.0 * df_t87["pred_dmid_norm"].to_numpy(np.float64)
                  + 1.5 * df_t75["pred_dmid_norm"].to_numpy(np.float64)) / 2.5
    df_stack = df_t87.copy()
    df_stack["pred_dmid_norm"] = p_combined.astype(np.float32)
    folds = split_by_sym(df_stack)
    print(f"Loaded in {time.time()-t0:.1f}s. Per-sym n: {[f['n'] for f in folds]}", flush=True)

    # Baseline: no abstain (β=0)
    actions_base = [gate_symmetric_band(f["pred"], THR_UP, THR_DN, 0.0, 0.0) for f in folds]
    base_eval = evaluate(folds, actions_base, "baseline_no_abstain")
    print(f"\nBaseline no-abstain: total={base_eval['total']:+.4f} min_sym={base_eval['min_sym']:+.4f} "
          f"active={base_eval['active_rate']:.3f} win={base_eval['win_rate']:.3f}", flush=True)

    # Reference: iter_018 v1 in-sample β (for comparison only)
    iter018_beta = {0: 0.10, 1: 0.40, 2: 0.30, 3: 0.00, 4: 0.00}
    actions_ref = []
    for f in folds:
        sigma = float(np.std(f["pred"]))
        b = iter018_beta[f["sym"]]
        actions_ref.append(gate_symmetric_band(f["pred"], THR_UP, THR_DN, b*sigma, b*sigma))
    ref_eval = evaluate(folds, actions_ref, "iter018_v1_insample")
    print(f"iter_018 v1 in-sample β: total={ref_eval['total']:+.4f} min_sym={ref_eval['min_sym']:+.4f} "
          f"active={ref_eval['active_rate']:.3f} win={ref_eval['win_rate']:.3f}", flush=True)

    K = 4
    # Variant 1: finer grid
    print(f"\n=== V1: finer β grid (0..0.50 step 0.05), {K}-fold CV ===", flush=True)
    grid_v1 = [round(x, 3) for x in np.arange(0.0, 0.55, 0.05).tolist()]
    a1, betas1 = variant1_fine_grid(folds, grid_v1, K)
    e1 = evaluate(folds, a1, "v1_fine_grid")
    e1["betas_per_fold"] = betas1
    e1["beta_grid"] = grid_v1
    print(f"V1 fine grid: total={e1['total']:+.4f} min_sym={e1['min_sym']:+.4f} "
          f"active={e1['active_rate']:.3f} win={e1['win_rate']:.3f}", flush=True)
    for sym_idx, b in enumerate(betas1):
        print(f"  sym{sym_idx} β per fold: {b}", flush=True)

    # Variant 2: win-rate calibration
    print(f"\n=== V2: empirical win-rate calibration ===", flush=True)
    v2_results = {}
    for tgt in [0.48, 0.50, 0.52]:
        a2, betas2 = variant2_winrate(folds, grid_v1, K, target_winrate=tgt)
        e2 = evaluate(folds, a2, f"v2_winrate_t{tgt}")
        e2["betas_per_fold"] = betas2
        e2["target_winrate"] = tgt
        v2_results[f"target_{tgt}"] = e2
        print(f"V2 wr≥{tgt}: total={e2['total']:+.4f} min_sym={e2['min_sym']:+.4f} "
              f"active={e2['active_rate']:.3f} win={e2['win_rate']:.3f}", flush=True)
        for sym_idx, b in enumerate(betas2):
            print(f"  sym{sym_idx} β per fold: {b}", flush=True)

    # Variant 3: asymmetric
    print(f"\n=== V3: asymmetric β (long vs short) ===", flush=True)
    grid_v3 = [round(x, 3) for x in np.arange(0.0, 0.55, 0.10).tolist()]  # coarser to keep runtime down
    print(f"V3 grid: {grid_v3} ({len(grid_v3)**2} combos per fold)", flush=True)
    a3, bL3, bS3 = variant3_asymmetric(folds, grid_v3, K)
    e3 = evaluate(folds, a3, "v3_asymmetric")
    e3["betas_long_per_fold"] = bL3
    e3["betas_short_per_fold"] = bS3
    e3["beta_grid"] = grid_v3
    print(f"V3 asymmetric: total={e3['total']:+.4f} min_sym={e3['min_sym']:+.4f} "
          f"active={e3['active_rate']:.3f} win={e3['win_rate']:.3f}", flush=True)
    for sym_idx in range(5):
        print(f"  sym{sym_idx} βL per fold: {bL3[sym_idx]}  βS per fold: {bS3[sym_idx]}", flush=True)

    # Variant 4: sigma-aware
    print(f"\n=== V4: sigma-aware β ===", flush=True)
    a4, betas4 = variant4_sigma_aware(folds, grid_v1, K, win=200)
    e4 = evaluate(folds, a4, "v4_sigma_aware")
    e4["betas_per_fold"] = betas4
    e4["beta_grid"] = grid_v1
    e4["sigma_window"] = 200
    print(f"V4 sigma-aware: total={e4['total']:+.4f} min_sym={e4['min_sym']:+.4f} "
          f"active={e4['active_rate']:.3f} win={e4['win_rate']:.3f}", flush=True)
    for sym_idx, b in enumerate(betas4):
        print(f"  sym{sym_idx} β per fold: {b}", flush=True)

    # Summary
    summary = {
        "baseline_no_abstain": base_eval["total"],
        "iter018_v1_insample": ref_eval["total"],
        "v1_fine_grid": e1["total"],
        "v2_winrate": {k: v["total"] for k, v in v2_results.items()},
        "v3_asymmetric": e3["total"],
        "v4_sigma_aware": e4["total"],
    }
    print("\n=== SUMMARY (totals) ===", flush=True)
    for k, v in summary.items():
        print(f"  {k}: {v}", flush=True)

    deltas = {
        "v1_vs_iter018": e1["total"] - ref_eval["total"],
        "v3_vs_iter018": e3["total"] - ref_eval["total"],
        "v4_vs_iter018": e4["total"] - ref_eval["total"],
        "v2_best_vs_iter018": max(v["total"] for v in v2_results.values()) - ref_eval["total"],
    }
    print("\n=== DELTAS vs iter_018 v1 ===", flush=True)
    for k, v in deltas.items():
        print(f"  {k}: {v:+.4f}", flush=True)

    out = {
        "thr_up": THR_UP, "thr_dn": THR_DN,
        "baseline_no_abstain": base_eval,
        "iter018_v1_insample": ref_eval,
        "v1_fine_grid": e1,
        "v2_winrate": v2_results,
        "v3_asymmetric": e3,
        "v4_sigma_aware": e4,
        "summary": summary,
        "deltas_vs_iter018": deltas,
        "elapsed_sec": float(time.time() - t0),
    }
    with open(os.path.join(EXP_DIR, "results.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\nDone in {time.time()-t0:.1f}s. Results -> results.json", flush=True)


if __name__ == "__main__":
    main()
