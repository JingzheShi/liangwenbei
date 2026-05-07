"""DE-optimized asymmetric (T_up, T_dn, d_up, d_dn) gate on T45 5-seed Group DRO
ensemble OOF predictions.

Loads dro5seed_pred_seed{S}_held{K}.parquet for S in {42,1,7,13,100}, K in {0..4},
averages per-seed softmax probs, then runs:
  1. raw argmax baseline (no threshold)
  2. iter_006 DE optimum unchanged (Tu=0.448, Td=0.391, du=0.260, dd=0.024)
  3. fresh DE search on the 5-seed avg
  4. coarse grid sweep around DE optimum

Produces dro5seed_thresh_results.json with per-fold PnL, comparison vs +13.61.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

T30_DIR = os.path.join(ROOT, "experiments", "T30_threshold_optim")
sys.path.insert(0, T30_DIR)
from _common import gate_asymmetric, vectorized_pnl  # noqa: E402

PROB_COLS = ["prob_0", "prob_1", "prob_2"]
SEEDS = [42, 1, 7, 13, 100]
N_FOLDS = 5
TAG = "dro5seed"


def load_5seed_avg_folds():
    """Load 5-seed ensemble (avg probs) per fold."""
    folds = {}
    for k in range(N_FOLDS):
        base_path = os.path.join(HERE, f"{TAG}_pred_seed{SEEDS[0]}_held{k}.parquet")
        base = pd.read_parquet(base_path)
        n = len(base)
        p_sum = base[PROB_COLS].to_numpy(np.float64).copy()
        for s in SEEDS[1:]:
            p = os.path.join(HERE, f"{TAG}_pred_seed{s}_held{k}.parquet")
            df_s = pd.read_parquet(p)
            if len(df_s) != n:
                raise RuntimeError(f"row mismatch fold={k} seed={s}: {len(df_s)} vs {n}")
            p_sum += df_s[PROB_COLS].to_numpy(np.float64)
        p_avg = p_sum / float(len(SEEDS))
        out = base.drop(columns=PROB_COLS).copy()
        out["prob_0"] = p_avg[:, 0].astype(np.float32)
        out["prob_1"] = p_avg[:, 1].astype(np.float32)
        out["prob_2"] = p_avg[:, 2].astype(np.float32)
        folds[k] = out
    return folds


def fold_arrays(folds):
    out = []
    for k in range(N_FOLDS):
        df = folds[k]
        out.append({
            "probs": df[PROB_COLS].to_numpy(np.float32),
            "label": df["true_label"].to_numpy(np.int64),
            "mp_t": df["midprice_t"].to_numpy(np.float64),
            "mp_th": df["midprice_th"].to_numpy(np.float64),
        })
    return out


def eval_thresh(fas, Tu, Td, du, dd):
    per = []
    for fa in fas:
        pred = gate_asymmetric(fa["probs"], float(Tu), float(Td),
                               float(du), float(dd))
        per.append(float(vectorized_pnl(pred, fa["label"], fa["mp_t"], fa["mp_th"]).sum()))
    return per


def raw_argmax_pnl(fas):
    per = []
    for fa in fas:
        pred = fa["probs"].argmax(axis=1).astype(np.int8)
        per.append(float(vectorized_pnl(pred, fa["label"], fa["mp_t"], fa["mp_th"]).sum()))
    return per


def make_objective(fas):
    label = [fa["label"] for fa in fas]
    mp_t = [fa["mp_t"] for fa in fas]
    mp_th = [fa["mp_th"] for fa in fas]
    probs = [fa["probs"] for fa in fas]

    def f(x):
        Tu, Td, du, dd = x
        s = 0.0
        for k in range(N_FOLDS):
            pred = gate_asymmetric(probs[k], Tu, Td, du, dd)
            s += vectorized_pnl(pred, label[k], mp_t[k], mp_th[k]).sum()
        return -float(s)
    return f


def run_de(fas, n_runs=5, bounds=None):
    if bounds is None:
        bounds = [(0.34, 0.75), (0.34, 0.75), (0.0, 0.30), (0.0, 0.30)]
    f = make_objective(fas)
    runs = []
    for seed in (0, 1, 2, 7, 42)[:n_runs]:
        t_run = time.time()
        result = differential_evolution(
            f, bounds=bounds, seed=seed, maxiter=80, popsize=24,
            polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            updating="deferred", workers=1, init="sobol",
        )
        sum_pnl = -result.fun
        Tu, Td, du, dd = result.x
        per = eval_thresh(fas, Tu, Td, du, dd)
        n_active = []
        for k in range(N_FOLDS):
            pred = gate_asymmetric(fas[k]["probs"], Tu, Td, du, dd)
            n_active.append(int((pred != 1).sum()))
        runs.append({
            "seed": seed,
            "T_up": float(Tu), "T_dn": float(Td),
            "d_up": float(du), "d_dn": float(dd),
            "sum_cum_pnl": float(sum_pnl),
            "per_fold_pnl": per,
            "n_active_per_fold": n_active,
            "n_pos_folds": int(sum(1 for x in per if x > 0)),
            "std_cum_pnl": float(np.std(per, ddof=0)),
            "nfev": int(result.nfev),
            "elapsed_sec": time.time() - t_run,
        })
        print(f"  DE seed={seed}: Tu={Tu:.4f} Td={Td:.4f} du={du:.4f} dd={dd:.4f} "
              f"sum={sum_pnl:+.4f} per={[round(x,3) for x in per]} ({result.nfev} ev, {runs[-1]['elapsed_sec']:.1f}s)",
              flush=True)
    runs.sort(key=lambda r: r["sum_cum_pnl"], reverse=True)
    return runs


def main():
    t0 = time.time()
    print(f"Loading 5-seed avg OOF (tag={TAG}, seeds={SEEDS}) ...", flush=True)
    folds = load_5seed_avg_folds()
    fas = fold_arrays(folds)
    print(f"  per-fold n={[len(f['label']) for f in fas]}", flush=True)

    # 0. raw argmax baseline
    raw = raw_argmax_pnl(fas)
    print(f"\n[raw argmax (5-seed avg)]  per-fold={[f'{x:+.3f}' for x in raw]}  "
          f"sum={sum(raw):+.4f}", flush=True)

    # 1. iter_006 DE optimum unchanged
    Tu0, Td0, du0, dd0 = 0.448, 0.391, 0.260, 0.024
    per_iter006 = eval_thresh(fas, Tu0, Td0, du0, dd0)
    print(f"\n[iter_006 DE optimum (Tu={Tu0}, Td={Td0}, du={du0}, dd={dd0})]")
    print(f"  per-fold={[f'{x:+.3f}' for x in per_iter006]}  "
          f"sum={sum(per_iter006):+.4f}")
    print(f"  vs +13.61: {sum(per_iter006) - 13.61:+.4f}", flush=True)

    # 2. fresh 4D DE on 5-seed avg
    print(f"\n=== Fresh 4D DE search (5 runs) ===", flush=True)
    de_runs = run_de(fas, n_runs=5)
    best = de_runs[0]
    print(f"\nBest DE: Tu={best['T_up']:.4f} Td={best['T_dn']:.4f} "
          f"du={best['d_up']:.4f} dd={best['d_dn']:.4f} sum={best['sum_cum_pnl']:+.4f} "
          f"pos={best['n_pos_folds']}/5 std={best['std_cum_pnl']:.3f}")
    print(f"vs +13.61: {best['sum_cum_pnl'] - 13.61:+.4f}", flush=True)

    out = {
        "task": "T45 Group DRO 5-seed ensemble + 4D DE asymmetric thresh",
        "tag": TAG,
        "seeds": SEEDS,
        "n_folds": N_FOLDS,
        "raw_argmax_per_fold": raw,
        "raw_argmax_sum": float(sum(raw)),
        "iter006_DE_unchanged": {
            "thresholds": {"T_up": Tu0, "T_dn": Td0, "d_up": du0, "d_dn": dd0},
            "per_fold": per_iter006,
            "sum_cum_pnl": float(sum(per_iter006)),
            "vs_13_61": float(sum(per_iter006)) - 13.61,
        },
        "fresh_DE_runs": de_runs,
        "fresh_DE_best": best,
        "ensemble_baseline_to_beat": 13.61,
        "comparison": {
            "best_de_vs_13_61": best["sum_cum_pnl"] - 13.61,
            "iter006_unchanged_vs_13_61": float(sum(per_iter006)) - 13.61,
        },
        "elapsed_sec": time.time() - t0,
    }
    out_path = os.path.join(HERE, "dro5seed_thresh_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults -> {out_path}", flush=True)
    print(f"\nFINAL: best 5-seed sum = {best['sum_cum_pnl']:+.4f} "
          f"(vs +13.61: {best['sum_cum_pnl'] - 13.61:+.4f})", flush=True)


if __name__ == "__main__":
    main()
