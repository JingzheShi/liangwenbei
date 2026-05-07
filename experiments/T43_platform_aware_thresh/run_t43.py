"""T43 — Platform-aware threshold optimization.

Hypothesis: iter_006's DE 4D thresholds were tuned to maximize sum cum_pnl
across 5 LOSO folds, but fold #3 (sym=3 OOD) forced overly tight gates to
avoid negative PnL on that nearly-untraded fold. The platform's sym 0-4
should all be in-distribution, so dropping/down-weighting the sym=3 fold
during threshold tuning may unlock substantially more activity (recall) and
PnL.

This script:
  1. Loads 5-seed iter_005b aug_a h_60 OOF (seed 42 from T26, seeds 1/7/13/100
     from T27), averages probs per fold.
  2. Runs DE 4D thresh search under 4 objective variants:
       V0  sum over all 5 folds                (= iter_006 baseline)
       V1  sum over folds {0,1,2,4}            (drop sym=3)
       V2  weighted sum: fold 3 weight 0.2, others 1.0
       V3  fit on {0,1,2,4} only, but report 5-fold and 4-fold sums
     (V1 and V3 are mathematically equivalent because both ignore fold 3 in
     the optimization objective and only differ in reporting; we keep both
     for sanity but show the same opt result.)
  3. Grid sweep symmetric (T_up=T_dn=T, d_up=d_dn=delta) for recall analysis.
  4. Writes results.json + REPORT.md + heatmap.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.eval.pnl import _per_horizon_metrics  # noqa: E402

T26_DIR = os.path.join(ROOT, "experiments", "T26_domain_randomization")
T27_DIR = os.path.join(ROOT, "experiments", "T27_iter005")

SEEDS = [42, 1, 7, 13, 100]
N_FOLDS = 5
H = 60
FEE = 0.0001
PROB_COLS = ["prob_0", "prob_1", "prob_2"]


def _seed_path(seed: int, k: int) -> str:
    if seed == 42:
        return os.path.join(T26_DIR, f"loso_pred_h{H}_aug_a_held{k}.parquet")
    return os.path.join(T27_DIR, f"loso_pred_h{H}_aug_a_seed{seed}_held{k}.parquet")


def load_fold(k: int) -> pd.DataFrame:
    base = pd.read_parquet(_seed_path(SEEDS[0], k))
    p_sum = base[PROB_COLS].to_numpy(np.float64).copy()
    n = len(base)
    for s in SEEDS[1:]:
        df_s = pd.read_parquet(_seed_path(s, k))
        if len(df_s) != n:
            raise RuntimeError(f"row count mismatch fold={k} seed={s}: {len(df_s)} vs {n}")
        p_sum += df_s[PROB_COLS].to_numpy(np.float64)
    p_avg = p_sum / float(len(SEEDS))
    out = base.drop(columns=PROB_COLS).copy()
    out["prob_0"] = p_avg[:, 0].astype(np.float32)
    out["prob_1"] = p_avg[:, 1].astype(np.float32)
    out["prob_2"] = p_avg[:, 2].astype(np.float32)
    return out


@dataclass
class FoldArrays:
    fold: int
    probs: np.ndarray
    label: np.ndarray
    mp_t: np.ndarray
    mp_th: np.ndarray
    n: int


def to_arrays(folds: Dict[int, pd.DataFrame]) -> List[FoldArrays]:
    out = []
    for k in range(N_FOLDS):
        df = folds[k]
        out.append(FoldArrays(
            fold=k,
            probs=df[PROB_COLS].to_numpy(np.float32),
            label=df["true_label"].to_numpy(np.int64),
            mp_t=df["midprice_t"].to_numpy(np.float64),
            mp_th=df["midprice_th"].to_numpy(np.float64),
            n=len(df),
        ))
    return out


def gate_asym(probs: np.ndarray, T_up: float, T_dn: float, d_up: float, d_dn: float) -> np.ndarray:
    p0 = probs[:, 0]; p1 = probs[:, 1]; p2 = probs[:, 2]
    take_up = (p2 >= T_up) & (p2 > p1 + d_up) & (p2 > p0)
    take_dn = (p0 >= T_dn) & (p0 > p1 + d_dn) & (p0 > p2)
    pred = np.full(probs.shape[0], 1, dtype=np.int8)
    pred[take_up] = 2
    pred[take_dn] = 0
    return pred


def gate_sym(probs: np.ndarray, T: float, delta: float) -> np.ndarray:
    p0 = probs[:, 0]; p1 = probs[:, 1]; p2 = probs[:, 2]
    side_max = np.maximum(p0, p2)
    take = (side_max >= T) & (side_max > p1 + delta)
    pred_side = np.where(p2 > p0, 2, 0)
    return np.where(take, pred_side, 1).astype(np.int8)


def fold_pnl(fa: FoldArrays, pred: np.ndarray) -> dict:
    m = _per_horizon_metrics(pred, fa.label, fa.mp_t.astype(np.float32),
                              fa.mp_th.astype(np.float32), fee_rate=FEE)
    return {
        "cum_pnl": float(m["cum_pnl"]),
        "single_pnl": float(m["single_pnl"]),
        "accuracy": float(m["accuracy"]),
        "n_active": int(m["n_predictions_active"]),
        "n_total": int(fa.n),
        "active_rate": float(m["n_predictions_active"]) / float(fa.n),
        "f05_macro": float(m.get("f0.5_macro", float("nan"))),
        "f1_macro": float(m.get("f1_macro", float("nan"))),
    }


def all_folds_metrics(folds: List[FoldArrays], T_up, T_dn, d_up, d_dn) -> dict:
    per = []
    for fa in folds:
        pred = gate_asym(fa.probs, T_up, T_dn, d_up, d_dn)
        per.append(fold_pnl(fa, pred))
    cum = [r["cum_pnl"] for r in per]
    return {
        "T_up": float(T_up), "T_dn": float(T_dn),
        "d_up": float(d_up), "d_dn": float(d_dn),
        "sum_5fold": float(sum(cum)),
        "sum_4fold_no3": float(cum[0] + cum[1] + cum[2] + cum[4]),
        "per_fold_pnl": [float(c) for c in cum],
        "per_fold_active": [r["n_active"] for r in per],
        "per_fold_active_rate": [r["active_rate"] for r in per],
        "per_fold_f05": [r["f05_macro"] for r in per],
        "per_fold_f1": [r["f1_macro"] for r in per],
        "n_pos_folds_5": int(sum(1 for c in cum if c > 0)),
        "n_pos_folds_4": int(sum(1 for c in (cum[0], cum[1], cum[2], cum[4]) if c > 0)),
        "mean_active_rate_4fold": float(np.mean([per[k]["active_rate"] for k in (0,1,2,4)])),
    }


def make_objective(folds: List[FoldArrays], variant: str):
    """Return (objective fn, opt-fold-list). Lower is better (we minimize -pnl)."""
    if variant == "V0":
        opt_idx = [0, 1, 2, 3, 4]
        weights = [1.0, 1.0, 1.0, 1.0, 1.0]
    elif variant == "V1":
        opt_idx = [0, 1, 2, 4]
        weights = [1.0, 1.0, 1.0, 1.0]
    elif variant == "V2":
        opt_idx = [0, 1, 2, 3, 4]
        weights = [1.0, 1.0, 1.0, 0.2, 1.0]
    elif variant == "V3":
        opt_idx = [0, 1, 2, 4]
        weights = [1.0, 1.0, 1.0, 1.0]
    else:
        raise ValueError(variant)
    sel = [folds[i] for i in opt_idx]

    def obj(x):
        T_up, T_dn, d_up, d_dn = float(x[0]), float(x[1]), float(x[2]), float(x[3])
        if T_up <= 0.34 or T_up > 0.85 or T_dn <= 0.34 or T_dn > 0.85:
            return 1e6
        s = 0.0
        for w, fa in zip(weights, sel):
            pred = gate_asym(fa.probs, T_up, T_dn, d_up, d_dn)
            side = pred.astype(np.float64) - 1.0
            abs_side = np.abs(side)
            diff = fa.mp_th - fa.mp_t
            fee = FEE * abs_side * np.abs((fa.mp_th + 1.0) + (fa.mp_t + 1.0))
            denom = fa.mp_t + 1.0
            pnl = (side * diff - fee) / denom
            s += w * float(pnl.sum())
        return -s

    return obj, opt_idx, weights


def run_de(folds: List[FoldArrays], variant: str, seeds=(0, 7, 42)) -> dict:
    bounds = [(0.34, 0.85), (0.34, 0.85), (0.0, 0.30), (0.0, 0.30)]
    obj, opt_idx, weights = make_objective(folds, variant)
    runs = []
    for s in seeds:
        t0 = time.time()
        res = differential_evolution(
            obj, bounds=bounds, seed=s,
            maxiter=120, popsize=20, tol=1e-7,
            mutation=(0.5, 1.0), recombination=0.7,
            polish=True, init="sobol",
        )
        T_up, T_dn, d_up, d_dn = res.x
        rep = all_folds_metrics(folds, T_up, T_dn, d_up, d_dn)
        rep["de_seed"] = int(s)
        rep["nfev"] = int(res.nfev)
        rep["elapsed_sec"] = time.time() - t0
        runs.append(rep)
    runs.sort(key=lambda r: -r["sum_5fold"] if variant in ("V0", "V2") else -r["sum_4fold_no3"])
    best = runs[0]
    return {
        "variant": variant,
        "opt_fold_idx": opt_idx,
        "weights": weights,
        "best": best,
        "all_runs": runs,
    }


def grid_sweep(folds: List[FoldArrays]) -> List[dict]:
    Ts = [0.30, 0.32, 0.35, 0.38, 0.40, 0.42, 0.45, 0.48, 0.50]
    deltas = [0.0, 0.02, 0.05, 0.10]
    rows = []
    for T in Ts:
        for d in deltas:
            rep = all_folds_metrics(folds, T, T, d, d)
            rep["mode"] = "symmetric"
            rep["T"] = T; rep["delta"] = d
            rows.append(rep)
    return rows


def main():
    print("Loading 5-seed OOF (T26 seed42 + T27 seeds 1,7,13,100)...")
    t0 = time.time()
    folds_df = {k: load_fold(k) for k in range(N_FOLDS)}
    folds = to_arrays(folds_df)
    print(f"  loaded in {time.time()-t0:.1f}s; per-fold n: {[fa.n for fa in folds]}")

    iter006 = (0.4480917972708376, 0.39144661429951816,
               0.2597244012809361, 0.024449627822859837)
    base = all_folds_metrics(folds, *iter006)
    print(f"\niter_006 (current) thresh check: 5fold={base['sum_5fold']:.3f}  4fold={base['sum_4fold_no3']:.3f}")
    print(f"  per-fold pnl: {[round(x,3) for x in base['per_fold_pnl']]}")
    print(f"  per-fold active rate: {[round(x,4) for x in base['per_fold_active_rate']]}")

    de_results = {}
    for variant in ("V0", "V1", "V2", "V3"):
        print(f"\nRunning DE variant={variant} ...")
        t0 = time.time()
        de_results[variant] = run_de(folds, variant, seeds=(0, 7, 42))
        b = de_results[variant]["best"]
        print(f"  best: T_up={b['T_up']:.4f} T_dn={b['T_dn']:.4f} "
              f"d_up={b['d_up']:.4f} d_dn={b['d_dn']:.4f}")
        print(f"    5fold={b['sum_5fold']:.3f}  4fold={b['sum_4fold_no3']:.3f}")
        print(f"    per_fold={[round(x,3) for x in b['per_fold_pnl']]}")
        print(f"    active_rate_per_fold={[round(x,4) for x in b['per_fold_active_rate']]}")
        print(f"  elapsed {time.time()-t0:.1f}s")

    print("\nGrid sweep (symmetric T,delta)...")
    grid = grid_sweep(folds)
    grid.sort(key=lambda r: -r["sum_4fold_no3"])
    for row in grid[:10]:
        print(f"  T={row['T']:.2f} d={row['delta']:.2f}  "
              f"5fold={row['sum_5fold']:.3f}  4fold={row['sum_4fold_no3']:.3f}  "
              f"actr_4f={row['mean_active_rate_4fold']:.3f}")

    out = {
        "task": "platform_aware_thresh",
        "iter_006_baseline": {"T_up": iter006[0], "T_dn": iter006[1],
                              "d_up": iter006[2], "d_dn": iter006[3],
                              "metrics": base},
        "de_variants": de_results,
        "grid_sweep": grid,
    }
    out_path = os.path.join(HERE, "results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else (int(o) if isinstance(o, np.integer) else str(o)))
    print(f"\nWrote {out_path}")
    return out


if __name__ == "__main__":
    main()
