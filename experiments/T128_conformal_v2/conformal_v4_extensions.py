"""T128 followup: V4 (sigma-aware) is the only winner. Explore extensions:

- V4a: one-sided (causal) rolling window — checks if centered-window lookahead matters
- V4b: V4 with finer β grid (0..0.50 step 0.025)
- V4c: V4 + asymmetric β (β_long, β_short per sym)
- V4d: V4 with different window sizes {50, 100, 200, 500, 1000}
- V4e: V4 with sigma_local clipping cap (avoid extreme blowups)
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
THR_UP = 0.00029995433796130955
THR_DN = 0.0002158528296175958
ITER018_BASELINE = 41.4939


def vectorized_pnl(action, mp_t, mp_th):
    side = action.astype(np.float64) - 1.0
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee_pnl) / denom


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


def split_by_sym(df):
    out = []
    for k in SYMS:
        sub = df[df["sym"] == k].reset_index(drop=True)
        order = np.argsort(sub["date"].to_numpy(), kind="mergesort")
        out.append({
            "sym": int(k),
            "pred": sub["pred_dmid_norm"].to_numpy(np.float64)[order],
            "mp_t": sub["midprice_t"].to_numpy(np.float64)[order],
            "mp_th": sub["midprice_th"].to_numpy(np.float64)[order],
            "date": sub["date"].to_numpy(np.int64)[order],
            "n": len(sub),
        })
    return out


def k_date_folds(date_arr, k):
    unique_dates = np.sort(np.unique(date_arr))
    n_d = len(unique_dates)
    out = []
    for i in range(k):
        lo = i * n_d // k
        hi = (i + 1) * n_d // k
        fold_dates = set(unique_dates[lo:hi].tolist())
        out.append(np.array([d in fold_dates for d in date_arr]))
    return out


def per_sym_metrics(folds, actions_per_sym):
    out, actives, wins = [], [], []
    for f, a in zip(folds, actions_per_sym):
        pnl = vectorized_pnl(a, f["mp_t"], f["mp_th"])
        out.append(float(pnl.sum()))
        n_active = int((a != 1).sum())
        actives.append(n_active / len(a) if len(a) > 0 else 0.0)
        active_mask = (a != 1)
        wins.append(float((pnl[active_mask] > 0).mean()) if active_mask.sum() > 0 else 0.0)
    return out, actives, wins


def evaluate(folds, actions, label):
    per_sym, actives, wins = per_sym_metrics(folds, actions)
    return {
        "label": label,
        "total": float(sum(per_sym)),
        "per_sym": per_sym,
        "actives": actives,
        "wins": wins,
        "min_sym": float(min(per_sym)),
        "active_rate": float(np.mean(actives)),
        "win_rate": float(np.mean(wins)),
    }


def rolling_std_centered(x, win):
    s = pd.Series(x)
    return s.rolling(win, min_periods=10, center=True).std().fillna(s.std()).to_numpy()


def rolling_std_causal(x, win):
    s = pd.Series(x)
    return s.rolling(win, min_periods=10).std().fillna(s.std()).to_numpy()


def gate_with_per_row_band(pred, thr_up, thr_dn, band_up, band_dn):
    a = np.full(len(pred), 1, dtype=np.int8)
    a[pred > thr_up + band_up] = 2
    a[pred < -(thr_dn + band_dn)] = 0
    return a


def variant_sigma_aware(folds, beta_grid, k, sigma_local_per_sym, asymmetric=False, clip_mult=None):
    actions_per_sym = []
    chosen_betas = []  # list of (sym -> per-fold β) or per-fold (bL, bS)
    for f, sigma_local in zip(folds, sigma_local_per_sym):
        sigma_baseline = float(np.std(f["pred"]))
        mult = np.maximum(1.0, sigma_local / max(sigma_baseline, 1e-9))
        if clip_mult is not None:
            mult = np.minimum(mult, clip_mult)
        date_folds = k_date_folds(f["date"], k)
        action = np.full(f["n"], 1, dtype=np.int8)
        if not asymmetric:
            betas_per_fold = []
            for fold_idx in range(k):
                test_mask = date_folds[fold_idx]
                calib_mask = ~test_mask
                best_b, best_pnl = 0.0, -1e18
                for b in beta_grid:
                    bands_calib = b * sigma_baseline * mult[calib_mask]
                    a = gate_with_per_row_band(f["pred"][calib_mask], THR_UP, THR_DN,
                                                bands_calib, bands_calib)
                    pnl = vectorized_pnl(a, f["mp_t"][calib_mask], f["mp_th"][calib_mask]).sum()
                    if pnl > best_pnl:
                        best_pnl = pnl
                        best_b = b
                betas_per_fold.append(best_b)
                bands_test = best_b * sigma_baseline * mult[test_mask]
                action[test_mask] = gate_with_per_row_band(
                    f["pred"][test_mask], THR_UP, THR_DN, bands_test, bands_test)
            chosen_betas.append(betas_per_fold)
        else:
            bLbS_per_fold = []
            for fold_idx in range(k):
                test_mask = date_folds[fold_idx]
                calib_mask = ~test_mask
                best_bL, best_bS, best_pnl = 0.0, 0.0, -1e18
                for bL in beta_grid:
                    for bS in beta_grid:
                        bands_L = bL * sigma_baseline * mult[calib_mask]
                        bands_S = bS * sigma_baseline * mult[calib_mask]
                        a = gate_with_per_row_band(
                            f["pred"][calib_mask], THR_UP, THR_DN, bands_L, bands_S)
                        pnl = vectorized_pnl(a, f["mp_t"][calib_mask], f["mp_th"][calib_mask]).sum()
                        if pnl > best_pnl:
                            best_pnl = pnl
                            best_bL, best_bS = bL, bS
                bLbS_per_fold.append((best_bL, best_bS))
                bands_L_t = best_bL * sigma_baseline * mult[test_mask]
                bands_S_t = best_bS * sigma_baseline * mult[test_mask]
                action[test_mask] = gate_with_per_row_band(
                    f["pred"][test_mask], THR_UP, THR_DN, bands_L_t, bands_S_t)
            chosen_betas.append(bLbS_per_fold)
        actions_per_sym.append(action)
    return actions_per_sym, chosen_betas


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
    print(f"Loaded in {time.time()-t0:.1f}s.", flush=True)

    K = 4
    fine_grid = [round(x, 3) for x in np.arange(0.0, 0.55, 0.025).tolist()]
    coarse_asym_grid = [round(x, 3) for x in np.arange(0.0, 0.55, 0.05).tolist()]

    results = {"thr_up": THR_UP, "thr_dn": THR_DN, "iter018_baseline": ITER018_BASELINE}

    # V4a: causal (one-sided) rolling window, win=200
    print(f"\n=== V4a: causal rolling σ (win=200), grid={fine_grid[:5]}…{fine_grid[-3:]} ===", flush=True)
    sigma_causal = [rolling_std_causal(f["pred"], 200) for f in folds]
    a4a, betas4a = variant_sigma_aware(folds, fine_grid, K, sigma_causal, asymmetric=False)
    e4a = evaluate(folds, a4a, "v4a_causal_w200")
    e4a["betas_per_fold"] = betas4a
    results["v4a"] = e4a
    print(f"V4a: total={e4a['total']:+.4f} min_sym={e4a['min_sym']:+.4f} "
          f"active={e4a['active_rate']:.3f} win={e4a['win_rate']:.3f} "
          f"(Δ vs iter_018: {e4a['total']-ITER018_BASELINE:+.4f})", flush=True)
    for sym_idx, b in enumerate(betas4a):
        print(f"  sym{sym_idx} β: {b}", flush=True)

    # V4b: centered, finer grid (0..0.50 step 0.025)
    print(f"\n=== V4b: centered σ (win=200), fine grid step 0.025 ===", flush=True)
    sigma_cent = [rolling_std_centered(f["pred"], 200) for f in folds]
    a4b, betas4b = variant_sigma_aware(folds, fine_grid, K, sigma_cent, asymmetric=False)
    e4b = evaluate(folds, a4b, "v4b_centered_w200_fine")
    e4b["betas_per_fold"] = betas4b
    results["v4b"] = e4b
    print(f"V4b: total={e4b['total']:+.4f} min_sym={e4b['min_sym']:+.4f} "
          f"active={e4b['active_rate']:.3f} win={e4b['win_rate']:.3f} "
          f"(Δ vs iter_018: {e4b['total']-ITER018_BASELINE:+.4f})", flush=True)
    for sym_idx, b in enumerate(betas4b):
        print(f"  sym{sym_idx} β: {b}", flush=True)

    # V4c: centered, asymmetric (β_long, β_short)
    print(f"\n=== V4c: centered σ (win=200), asymmetric (βL, βS) ===", flush=True)
    a4c, betas4c = variant_sigma_aware(folds, coarse_asym_grid, K, sigma_cent, asymmetric=True)
    e4c = evaluate(folds, a4c, "v4c_centered_asymmetric")
    e4c["betas_per_fold"] = [[(float(x[0]), float(x[1])) for x in b] for b in betas4c]
    results["v4c"] = e4c
    print(f"V4c: total={e4c['total']:+.4f} min_sym={e4c['min_sym']:+.4f} "
          f"active={e4c['active_rate']:.3f} win={e4c['win_rate']:.3f} "
          f"(Δ vs iter_018: {e4c['total']-ITER018_BASELINE:+.4f})", flush=True)
    for sym_idx, b in enumerate(betas4c):
        print(f"  sym{sym_idx} (βL,βS): {b}", flush=True)

    # V4d: window scan
    print(f"\n=== V4d: window scan ===", flush=True)
    win_scan = {}
    for win in [50, 100, 200, 500, 1000]:
        sigma_w = [rolling_std_centered(f["pred"], win) for f in folds]
        a, b = variant_sigma_aware(folds, fine_grid, K, sigma_w, asymmetric=False)
        e = evaluate(folds, a, f"v4d_w{win}")
        e["betas_per_fold"] = b
        win_scan[f"w{win}"] = e
        print(f"  win={win}: total={e['total']:+.4f} min_sym={e['min_sym']:+.4f} "
              f"active={e['active_rate']:.3f} win={e['win_rate']:.3f} "
              f"(Δ vs iter_018: {e['total']-ITER018_BASELINE:+.4f})", flush=True)
    results["v4d_window_scan"] = win_scan

    # V4e: σ-aware with mult clip cap (avoid extreme outliers blowing up the band)
    print(f"\n=== V4e: clipped multiplier (cap mult ≤ K) ===", flush=True)
    clip_scan = {}
    for cap in [1.5, 2.0, 3.0, 5.0]:
        a, b = variant_sigma_aware(folds, fine_grid, K, sigma_cent,
                                   asymmetric=False, clip_mult=cap)
        e = evaluate(folds, a, f"v4e_cap{cap}")
        e["betas_per_fold"] = b
        clip_scan[f"cap_{cap}"] = e
        print(f"  cap={cap}: total={e['total']:+.4f} min_sym={e['min_sym']:+.4f} "
              f"active={e['active_rate']:.3f} win={e['win_rate']:.3f} "
              f"(Δ vs iter_018: {e['total']-ITER018_BASELINE:+.4f})", flush=True)
    results["v4e_clip_scan"] = clip_scan

    # V4f: per-row σ_local from a CALIB-only fit (deployable proxy):
    # Compute σ_local on each fold's calib data globally, look up by (date) bucket.
    # Approximation for stateless deployment.
    # Skipped — out of time scope. Document as future work.

    # Summary
    print("\n=== SUMMARY ===", flush=True)
    print(f"  iter_018 baseline: {ITER018_BASELINE}", flush=True)
    print(f"  V4a causal w200: {e4a['total']:+.4f} (Δ {e4a['total']-ITER018_BASELINE:+.4f})", flush=True)
    print(f"  V4b centered w200 fine: {e4b['total']:+.4f} (Δ {e4b['total']-ITER018_BASELINE:+.4f})", flush=True)
    print(f"  V4c centered asym: {e4c['total']:+.4f} (Δ {e4c['total']-ITER018_BASELINE:+.4f})", flush=True)
    print(f"  V4d best window: ", flush=True)
    best_d = max(win_scan.items(), key=lambda kv: kv[1]['total'])
    print(f"    {best_d[0]}: {best_d[1]['total']:+.4f} (Δ {best_d[1]['total']-ITER018_BASELINE:+.4f})", flush=True)
    best_e = max(clip_scan.items(), key=lambda kv: kv[1]['total'])
    print(f"  V4e best cap: {best_e[0]}: {best_e[1]['total']:+.4f} "
          f"(Δ {best_e[1]['total']-ITER018_BASELINE:+.4f})", flush=True)

    with open(os.path.join(EXP_DIR, "results_v4_extensions.json"), "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\nDone in {time.time()-t0:.1f}s. Results -> results_v4_extensions.json", flush=True)


if __name__ == "__main__":
    main()
