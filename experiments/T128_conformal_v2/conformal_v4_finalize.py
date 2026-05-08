"""T128 finalize: V4d w50 wins +0.42 LOSO. Verify stability across CV schemes + causal alternative."""
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


def variant_sigma_aware(folds, beta_grid, k, sigma_local_per_sym):
    actions_per_sym = []
    chosen_betas = []
    for f, sigma_local in zip(folds, sigma_local_per_sym):
        sigma_baseline = float(np.std(f["pred"]))
        mult = np.maximum(1.0, sigma_local / max(sigma_baseline, 1e-9))
        date_folds = k_date_folds(f["date"], k)
        action = np.full(f["n"], 1, dtype=np.int8)
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

    fine_grid = [round(x, 3) for x in np.arange(0.0, 0.55, 0.025).tolist()]
    results = {"thr_up": THR_UP, "thr_dn": THR_DN, "iter018_baseline": ITER018_BASELINE}

    # Scan: window x CV-fold count (centered)
    print(f"\n=== Window x CV-fold sensitivity (centered σ) ===", flush=True)
    grid_results = {}
    for K in [3, 4, 5, 6]:
        for win in [20, 30, 50, 75, 100, 150, 200, 300, 500]:
            sigma_w = [rolling_std_centered(f["pred"], win) for f in folds]
            a, b = variant_sigma_aware(folds, fine_grid, K, sigma_w)
            e = evaluate(folds, a, f"k{K}_w{win}")
            grid_results[f"k{K}_w{win}"] = {
                "total": e["total"], "min_sym": e["min_sym"],
                "active_rate": e["active_rate"], "win_rate": e["win_rate"],
                "delta_vs_iter018": e["total"] - ITER018_BASELINE,
                "betas_per_fold": b,
            }
            print(f"  K={K} win={win}: total={e['total']:+.4f} min_sym={e['min_sym']:+.4f} "
                  f"act={e['active_rate']:.3f} win={e['win_rate']:.3f} "
                  f"(Δ {e['total']-ITER018_BASELINE:+.4f})", flush=True)
    results["window_kfold_scan_centered"] = grid_results

    # Scan: causal versions for deployable check
    print(f"\n=== Causal σ — for deployability test ===", flush=True)
    causal_results = {}
    for K in [4]:
        for win in [50, 100, 200, 500]:
            sigma_w = [rolling_std_causal(f["pred"], win) for f in folds]
            a, b = variant_sigma_aware(folds, fine_grid, K, sigma_w)
            e = evaluate(folds, a, f"causal_k{K}_w{win}")
            causal_results[f"k{K}_w{win}"] = {
                "total": e["total"], "min_sym": e["min_sym"],
                "active_rate": e["active_rate"], "win_rate": e["win_rate"],
                "delta_vs_iter018": e["total"] - ITER018_BASELINE,
                "betas_per_fold": b,
            }
            print(f"  causal K={K} win={win}: total={e['total']:+.4f} min_sym={e['min_sym']:+.4f} "
                  f"act={e['active_rate']:.3f} win={e['win_rate']:.3f} "
                  f"(Δ {e['total']-ITER018_BASELINE:+.4f})", flush=True)
    results["causal_scan"] = causal_results

    # Find best across centered scan
    best_key, best_val = max(grid_results.items(), key=lambda kv: kv[1]["total"])
    print(f"\n=== BEST CENTERED ===", flush=True)
    print(f"  {best_key}: total={best_val['total']:+.4f} min_sym={best_val['min_sym']:+.4f} "
          f"(Δ {best_val['delta_vs_iter018']:+.4f})", flush=True)
    print(f"  betas: {best_val['betas_per_fold']}", flush=True)

    best_causal_key, best_causal_val = max(causal_results.items(), key=lambda kv: kv[1]["total"])
    print(f"\n=== BEST CAUSAL ===", flush=True)
    print(f"  {best_causal_key}: total={best_causal_val['total']:+.4f} "
          f"min_sym={best_causal_val['min_sym']:+.4f} "
          f"(Δ {best_causal_val['delta_vs_iter018']:+.4f})", flush=True)

    # Cross-CV stability for the best window: how stable across K values?
    best_win = int(best_key.split("_w")[1])
    print(f"\n=== Stability check: centered w={best_win}, K∈[3,4,5,6] ===", flush=True)
    stability = []
    for K in [3, 4, 5, 6]:
        v = grid_results[f"k{K}_w{best_win}"]
        stability.append((K, v["total"], v["delta_vs_iter018"]))
        print(f"  K={K}: {v['total']:+.4f} (Δ {v['delta_vs_iter018']:+.4f})", flush=True)
    avg_total = np.mean([s[1] for s in stability])
    avg_delta = np.mean([s[2] for s in stability])
    print(f"  avg over K: {avg_total:+.4f} (Δ {avg_delta:+.4f})", flush=True)
    results["best_window_stability"] = {
        "best_win": best_win, "stability": stability,
        "avg_total": float(avg_total), "avg_delta": float(avg_delta),
    }

    with open(os.path.join(EXP_DIR, "results_finalize.json"), "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\nDone in {time.time()-t0:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
