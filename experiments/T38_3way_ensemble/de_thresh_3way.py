"""T38: 3-way OOF ensemble (iter_005b + T37 stepA + T37 stepB) + DE 4D thresh.

Combines:
  M1 = iter_005b (T26 seed42 + T27 seeds 1,7,13,100), aug_a [0.80, 1.20] @ h_60
  M2 = T37 stepA  (5 seeds), aug_a [0.75, 1.25] @ h_60
  M3 = T37 stepB  (5 seeds), aug_a [0.75, 1.25] + revol_wmp1_sigma_hat @ h_60

For each fold k:
  M1_p = mean over 5 seeds (T26 seed42 + T27 1,7,13,100)
  M2_p = mean over 5 seeds (stepA)
  M3_p = mean over 5 seeds (stepB)

Combinations searched (all feed into the same DE 4D thresh search):
  M1                       (sanity, should reproduce iter_006 +13.61)
  M2 (stepA only)          (sanity, should reproduce T37 stepA +12.94)
  M3 (stepB only)          (sanity, should reproduce T37 stepB +12.27)
  (M1 + M2) / 2
  (M1 + M3) / 2
  (M2 + M3) / 2
  (M1 + M2 + M3) / 3       (the headline 3-way)
  weighted: M1 0.5 + M2 0.25 + M3 0.25
  weighted: M1 0.5 + M2 0.5 + M3 0.0   (equiv to (M1+M2)/2 already)
  weighted: M1 0.6 + M2 0.2 + M3 0.2
  weighted: M1 0.4 + M2 0.4 + M3 0.2
  weighted: M1 0.4 + M2 0.2 + M3 0.4

Outputs:
  results.json (per-combo: DE best + per-fold)
  de_3way_results.json (the headline 3-way result with full DE detail)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.eval.pnl import _per_horizon_metrics  # noqa: E402

SEEDS = [42, 1, 7, 13, 100]
N_FOLDS = 5
FEE = 0.0001
PROB_COLS = ["prob_0", "prob_1", "prob_2"]

T26_DIR = os.path.join(ROOT, "experiments", "T26_domain_randomization")
T27_DIR = os.path.join(ROOT, "experiments", "T27_iter005")
T37_DIR = os.path.join(ROOT, "experiments", "T37_iter007_aug_wide_revol")


def load_iter005b_seed_avg(k: int) -> Tuple[np.ndarray, pd.DataFrame]:
    """seed42 from T26 + seeds {1,7,13,100} from T27."""
    base = pd.read_parquet(os.path.join(T26_DIR, f"loso_pred_h60_aug_a_held{k}.parquet"))
    p_sum = base[PROB_COLS].to_numpy(np.float64).copy()
    n = len(base)
    for s in [1, 7, 13, 100]:
        df_s = pd.read_parquet(os.path.join(T27_DIR, f"loso_pred_h60_aug_a_seed{s}_held{k}.parquet"))
        if len(df_s) != n:
            raise RuntimeError(f"row count mismatch iter005b fold={k} seed={s}: {len(df_s)} vs {n}")
        # sanity: labels and midprices must match
        if not (df_s["true_label"].values == base["true_label"].values).all():
            raise RuntimeError(f"label mismatch iter005b fold={k} seed={s}")
        p_sum += df_s[PROB_COLS].to_numpy(np.float64)
    p_avg = p_sum / 5.0
    return p_avg.astype(np.float32), base


def load_t37_seed_avg(tag: str, k: int) -> np.ndarray:
    base_path = os.path.join(T37_DIR, f"{tag}_pred_seed{SEEDS[0]}_held{k}.parquet")
    base = pd.read_parquet(base_path)
    p_sum = base[PROB_COLS].to_numpy(np.float64).copy()
    n = len(base)
    for s in SEEDS[1:]:
        df_s = pd.read_parquet(os.path.join(T37_DIR, f"{tag}_pred_seed{s}_held{k}.parquet"))
        if len(df_s) != n:
            raise RuntimeError(f"row count mismatch {tag} fold={k} seed={s}: {len(df_s)} vs {n}")
        p_sum += df_s[PROB_COLS].to_numpy(np.float64)
    p_avg = p_sum / float(len(SEEDS))
    return p_avg.astype(np.float32)


def gate_asymmetric(probs: np.ndarray, T_up: float, T_dn: float,
                    d_up: float, d_dn: float) -> np.ndarray:
    p0 = probs[:, 0]; p1 = probs[:, 1]; p2 = probs[:, 2]
    take_up = (p2 >= T_up) & (p2 > p1 + d_up) & (p2 > p0)
    take_dn = (p0 >= T_dn) & (p0 > p1 + d_dn) & (p0 > p2)
    pred = np.full(probs.shape[0], 1, dtype=np.int8)
    pred[take_up] = 2
    pred[take_dn] = 0
    return pred


def vectorized_pnl(pred: np.ndarray, label: np.ndarray,
                   mp_t: np.ndarray, mp_th: np.ndarray) -> np.ndarray:
    side = pred.astype(np.float64) - 1.0
    abs_side = np.abs(side)
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee = FEE * abs_side * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee) / denom


@dataclass
class FoldArrays:
    fold: int
    probs: np.ndarray
    label: np.ndarray
    mp_t: np.ndarray
    mp_th: np.ndarray


def make_objective(fas):
    label = [fa.label for fa in fas]
    mp_t = [fa.mp_t for fa in fas]
    mp_th = [fa.mp_th for fa in fas]
    probs = [fa.probs for fa in fas]

    def f(x):
        Tu, Td, du, dd = x
        s = 0.0
        for k in range(N_FOLDS):
            pred = gate_asymmetric(probs[k], Tu, Td, du, dd)
            s += vectorized_pnl(pred, label[k], mp_t[k], mp_th[k]).sum()
        return -float(s)
    return f


def coarse_sweep(fas):
    T_up = np.round(np.arange(0.40, 0.6501, 0.025), 4)
    T_dn = np.round(np.arange(0.40, 0.6501, 0.025), 4)
    d_up = np.round(np.arange(0.00, 0.2001, 0.025), 4)
    d_dn = np.round(np.arange(0.00, 0.2001, 0.025), 4)
    rows = []
    label = [fa.label for fa in fas]
    mp_t = [fa.mp_t for fa in fas]
    mp_th = [fa.mp_th for fa in fas]
    probs = [fa.probs for fa in fas]
    for Tu in T_up:
        for Td in T_dn:
            for du in d_up:
                for dd in d_dn:
                    per = []
                    for k in range(N_FOLDS):
                        pred = gate_asymmetric(probs[k], float(Tu), float(Td), float(du), float(dd))
                        per.append(float(vectorized_pnl(pred, label[k], mp_t[k], mp_th[k]).sum()))
                    rows.append({
                        "T_up": float(Tu), "T_dn": float(Td),
                        "d_up": float(du), "d_dn": float(dd),
                        "sum_cum_pnl": float(sum(per)),
                        "n_pos_folds": int(sum(1 for x in per if x > 0)),
                        "per_fold": per,
                    })
    df = pd.DataFrame(rows).sort_values("sum_cum_pnl", ascending=False).reset_index(drop=True)
    return df


def de_search(fas, bounds, seeds=(0, 1, 7), maxiter=60, popsize=20):
    f = make_objective(fas)
    runs = []
    for seed in seeds:
        t_run = time.time()
        result = differential_evolution(
            f, bounds=bounds, seed=seed, maxiter=maxiter, popsize=popsize,
            polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            updating="deferred", workers=1, init="sobol",
        )
        sum_pnl = -result.fun
        Tu, Td, du, dd = result.x
        per = []
        n_active = []
        for k in range(N_FOLDS):
            pred = gate_asymmetric(fas[k].probs, Tu, Td, du, dd)
            per.append(float(vectorized_pnl(pred, fas[k].label, fas[k].mp_t, fas[k].mp_th).sum()))
            n_active.append(int((pred != 1).sum()))
        runs.append({
            "seed": int(seed),
            "T_up": float(Tu), "T_dn": float(Td),
            "d_up": float(du), "d_dn": float(dd),
            "sum_cum_pnl": float(sum_pnl),
            "per_fold_pnl": per,
            "n_active_per_fold": n_active,
            "n_pos_folds": int(sum(1 for x in per if x > 0)),
            "std_cum_pnl": float(np.std(per, ddof=0)),
            "nfev": int(result.nfev),
            "elapsed_sec": float(time.time() - t_run),
        })
        print(f"    DE seed={seed}: T_up={Tu:.4f} T_dn={Td:.4f} d_up={du:.4f} d_dn={dd:.4f} "
              f"-> sum={sum_pnl:+.4f}  per_fold={[round(x,3) for x in per]} "
              f"({result.nfev} evals, {time.time()-t_run:.1f}s)", flush=True)
    runs.sort(key=lambda r: r["sum_cum_pnl"], reverse=True)
    return runs


def neighborhood_check(fas, Tu0, Td0, du0, dd0, half_T=0.04, half_d=0.04, step=0.01):
    Tu_grid = np.round(np.arange(max(0.34, Tu0 - half_T), Tu0 + half_T + 1e-9, step), 4)
    Td_grid = np.round(np.arange(max(0.34, Td0 - half_T), Td0 + half_T + 1e-9, step), 4)
    du_grid = np.round(np.arange(max(0.0, du0 - half_d), min(0.30, du0 + half_d) + 1e-9, step), 4)
    dd_grid = np.round(np.arange(max(0.0, dd0 - half_d), min(0.30, dd0 + half_d) + 1e-9, step), 4)
    rows = []
    label = [fa.label for fa in fas]
    mp_t = [fa.mp_t for fa in fas]
    mp_th = [fa.mp_th for fa in fas]
    probs = [fa.probs for fa in fas]
    for Tu in Tu_grid:
        for Td in Td_grid:
            for du in du_grid:
                for dd in dd_grid:
                    per = []
                    for k in range(N_FOLDS):
                        pred = gate_asymmetric(probs[k], float(Tu), float(Td), float(du), float(dd))
                        per.append(float(vectorized_pnl(pred, label[k], mp_t[k], mp_th[k]).sum()))
                    rows.append({
                        "T_up": float(Tu), "T_dn": float(Td),
                        "d_up": float(du), "d_dn": float(dd),
                        "sum_cum_pnl": float(sum(per)),
                        "n_pos_folds": int(sum(1 for x in per if x > 0)),
                    })
    return pd.DataFrame(rows).sort_values("sum_cum_pnl", ascending=False).reset_index(drop=True)


def build_fas(probs_per_fold: List[np.ndarray], meta_per_fold: List[pd.DataFrame]) -> List[FoldArrays]:
    out = []
    for k in range(N_FOLDS):
        df = meta_per_fold[k]
        out.append(FoldArrays(
            fold=k,
            probs=probs_per_fold[k],
            label=df["true_label"].to_numpy(np.int64),
            mp_t=df["midprice_t"].to_numpy(np.float64),
            mp_th=df["midprice_th"].to_numpy(np.float64),
        ))
    return out


def evaluate_combo(name: str, fas: List[FoldArrays], full_de=True) -> dict:
    print(f"\n=== {name} ===", flush=True)
    raw_pf = []
    for fa in fas:
        pred = fa.probs.argmax(axis=1).astype(np.int8)
        raw_pf.append(float(vectorized_pnl(pred, fa.label, fa.mp_t, fa.mp_th).sum()))
    print(f"  raw argmax sum={sum(raw_pf):+.4f} per_fold={[round(x,3) for x in raw_pf]}", flush=True)

    bounds = [(0.34, 0.75), (0.34, 0.75), (0.0, 0.30), (0.0, 0.30)]
    t1 = time.time()
    de_runs = de_search(fas, bounds)
    best = de_runs[0]
    print(f"  DE best: sum={best['sum_cum_pnl']:+.4f} at "
          f"(T_up={best['T_up']:.4f}, T_dn={best['T_dn']:.4f}, "
          f"d_up={best['d_up']:.4f}, d_dn={best['d_dn']:.4f})  "
          f"per_fold={[round(x,3) for x in best['per_fold_pnl']]} pos={best['n_pos_folds']}/5  "
          f"({time.time()-t1:.1f}s)", flush=True)

    out = {
        "combo": name,
        "raw_argmax_sum": float(sum(raw_pf)),
        "raw_argmax_per_fold": raw_pf,
        "de_best": best,
        "de_runs": de_runs,
    }
    if full_de:
        nb = neighborhood_check(fas, best["T_up"], best["T_dn"], best["d_up"], best["d_dn"])
        n_robust = int((nb["sum_cum_pnl"] >= best["sum_cum_pnl"] - 0.5).sum())
        out["neighborhood_top10"] = nb.head(10).to_dict(orient="records")
        out["neighborhood_n_within_0.5"] = n_robust
        out["neighborhood_max"] = float(nb["sum_cum_pnl"].iloc[0])
        print(f"  neighborhood top sum={out['neighborhood_max']:+.4f}, configs within 0.5: {n_robust}/{len(nb)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="Skip neighborhood check and reduce coarse sweep for faster iteration")
    args = ap.parse_args()

    t0 = time.time()
    print(f"=== T38 3-way ensemble + DE thresh ===", flush=True)
    print(f"  loading iter_005b 5-seed avg ...", flush=True)
    M1 = []
    meta = []
    for k in range(N_FOLDS):
        p, df = load_iter005b_seed_avg(k)
        M1.append(p)
        meta.append(df)
    print(f"    iter_005b done  ({time.time()-t0:.1f}s)", flush=True)

    print(f"  loading T37 stepA 5-seed avg ...", flush=True)
    M2 = [load_t37_seed_avg("stepA", k) for k in range(N_FOLDS)]
    print(f"    stepA done  ({time.time()-t0:.1f}s)", flush=True)

    print(f"  loading T37 stepB 5-seed avg ...", flush=True)
    M3 = [load_t37_seed_avg("stepB", k) for k in range(N_FOLDS)]
    print(f"    stepB done  ({time.time()-t0:.1f}s)", flush=True)

    # build combos
    combos = {}
    combos["M1_iter005b"] = M1
    combos["M2_stepA"] = M2
    combos["M3_stepB"] = M3
    combos["M12_avg"] = [(M1[k] + M2[k]) / 2.0 for k in range(N_FOLDS)]
    combos["M13_avg"] = [(M1[k] + M3[k]) / 2.0 for k in range(N_FOLDS)]
    combos["M23_avg"] = [(M2[k] + M3[k]) / 2.0 for k in range(N_FOLDS)]
    combos["M123_avg"] = [(M1[k] + M2[k] + M3[k]) / 3.0 for k in range(N_FOLDS)]
    # weighted variants
    combos["M123_w_5_25_25"] = [(0.50 * M1[k] + 0.25 * M2[k] + 0.25 * M3[k]) for k in range(N_FOLDS)]
    combos["M123_w_6_2_2"] = [(0.60 * M1[k] + 0.20 * M2[k] + 0.20 * M3[k]) for k in range(N_FOLDS)]
    combos["M123_w_4_4_2"] = [(0.40 * M1[k] + 0.40 * M2[k] + 0.20 * M3[k]) for k in range(N_FOLDS)]
    combos["M123_w_4_2_4"] = [(0.40 * M1[k] + 0.20 * M2[k] + 0.40 * M3[k]) for k in range(N_FOLDS)]
    combos["M123_w_7_15_15"] = [(0.70 * M1[k] + 0.15 * M2[k] + 0.15 * M3[k]) for k in range(N_FOLDS)]
    combos["M12_w_7_3"] = [(0.70 * M1[k] + 0.30 * M2[k]) for k in range(N_FOLDS)]
    combos["M12_w_6_4"] = [(0.60 * M1[k] + 0.40 * M2[k]) for k in range(N_FOLDS)]

    # eval each
    all_results = {}
    for name, probs_per_fold in combos.items():
        fas = build_fas(probs_per_fold, meta)
        all_results[name] = evaluate_combo(name, fas, full_de=not args.quick)

    # ranking
    ranking = sorted(
        [(name, r["de_best"]["sum_cum_pnl"], r["de_best"]["n_pos_folds"], r["de_best"]["std_cum_pnl"])
         for name, r in all_results.items()],
        key=lambda x: x[1], reverse=True,
    )
    print(f"\n=== RANKING ===", flush=True)
    for name, s, pos, std in ranking:
        print(f"  {name:24s}  sum={s:+.4f}  pos={pos}/5  std={std:.3f}", flush=True)

    out = {
        "task": "T38 3-way ensemble + DE 4D thresh search",
        "iter_006_baseline_sum": 13.6062,
        "T37_stepA_baseline_sum": 12.94,
        "T37_stepB_baseline_sum": 12.27,
        "ranking": [{"combo": n, "sum": s, "n_pos_folds": p, "std": st} for n, s, p, st in ranking],
        "results_by_combo": all_results,
        "elapsed_sec": float(time.time() - t0),
    }
    out_path = os.path.join(HERE, "results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nDone in {time.time()-t0:.1f}s. -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
