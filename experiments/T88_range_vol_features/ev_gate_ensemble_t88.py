"""T88: Ensemble NN + LGB (T88 versions, schemeP + 12 range-vol features) → DE LOSO.

Mirrors T81 ev_gate_ensemble_nn_lgb.py.
Goal: compare against iter_014 baseline (+38.28 LOSO-equiv).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))

SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001
ITER012_LOSO = 26.44
ITER013_LOSO = 36.23
ITER014_LOSO = 38.28
SEEDS = (1, 7, 13, 42, 100)


def vectorized_pnl(pred_action, mp_t, mp_th):
    side = pred_action.astype(np.float64) - 1.0
    abs_side = np.abs(side)
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * abs_side * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee_pnl) / denom


def ev_gate_symmetric(pred, k):
    thr = k * 2.0 * FEE
    a = np.full(len(pred), 1, dtype=np.int8)
    a[pred > thr] = 2; a[pred < -thr] = 0
    return a


def ev_gate_asymmetric(pred, thr_up, thr_dn):
    a = np.full(len(pred), 1, dtype=np.int8)
    a[pred > thr_up] = 2; a[pred < -thr_dn] = 0
    return a


def split_by_sym(df):
    out = []
    for k in SYMS:
        sub = df[df["sym"] == k]
        out.append({"sym": int(k),
                    "pred": sub["pred_dmid_norm"].to_numpy(np.float64),
                    "mp_t": sub["midprice_t"].to_numpy(np.float64),
                    "mp_th": sub["midprice_th"].to_numpy(np.float64),
                    "n": len(sub)})
    return out


def load_avg_pred(weight_nn, weight_lgb):
    base = pd.read_parquet(os.path.join(HERE, f"pred_T88nn_seed{SEEDS[0]}.parquet"))
    n = len(base)
    p_nn = np.zeros(n, dtype=np.float64)
    p_lgb = np.zeros(n, dtype=np.float64)
    for s in SEEDS:
        df_nn = pd.read_parquet(os.path.join(HERE, f"pred_T88nn_seed{s}.parquet"))
        df_lgb = pd.read_parquet(os.path.join(HERE, f"pred_T88lgb_seed{s}.parquet"))
        if not (df_nn["sym"].equals(base["sym"]) and df_nn["t"].equals(base["t"])):
            raise RuntimeError(f"NN row mismatch seed={s}")
        if not (df_lgb["sym"].equals(base["sym"]) and df_lgb["t"].equals(base["t"])):
            raise RuntimeError(f"LGB row mismatch seed={s}")
        p_nn += df_nn["pred_dmid_norm"].to_numpy(np.float64)
        p_lgb += df_lgb["pred_dmid_norm"].to_numpy(np.float64)
    p_nn /= len(SEEDS); p_lgb /= len(SEEDS)
    p_combined = (weight_nn * p_nn + weight_lgb * p_lgb) / (weight_nn + weight_lgb)
    out = base.copy()
    out["pred_dmid_norm"] = p_combined.astype(np.float32)
    out["pred_nn"] = p_nn.astype(np.float32)
    out["pred_lgb"] = p_lgb.astype(np.float32)
    return out, p_nn, p_lgb


def per_sym_pnl_at_k(folds, k):
    return [float(vectorized_pnl(ev_gate_symmetric(f["pred"], k),
                                  f["mp_t"], f["mp_th"]).sum()) for f in folds]


def make_obj_loso_asym(folds):
    def f(x):
        thr_up, thr_dn = x
        s = 0.0
        for fk in folds:
            a = ev_gate_asymmetric(fk["pred"], thr_up, thr_dn)
            s += vectorized_pnl(a, fk["mp_t"], fk["mp_th"]).sum()
        return -float(s)
    return f


def de_search(obj_fn, bounds, seeds=(0, 1, 2, 7, 42)):
    runs = []
    for sd in seeds:
        r = differential_evolution(obj_fn, bounds=bounds, seed=sd, maxiter=80, popsize=24,
                                    polish=True, tol=1e-7, mutation=(0.5, 1.0),
                                    recombination=0.7, updating="deferred", workers=1, init="sobol")
        runs.append({"seed": int(sd), "thr_up": float(r.x[0]), "thr_dn": float(r.x[1]),
                     "obj_val": float(-r.fun), "nfev": int(r.nfev)})
    runs.sort(key=lambda r: r["obj_val"], reverse=True)
    return runs


def evaluate_full(df, label):
    folds = split_by_sym(df)
    pred_all = df["pred_dmid_norm"].to_numpy(np.float64)
    mp_t_all = df["midprice_t"].to_numpy(np.float64)
    mp_th_all = df["midprice_th"].to_numpy(np.float64)

    print(f"\n=== {label} ===", flush=True)
    sym_sweep = []
    for k in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]:
        per = per_sym_pnl_at_k(folds, k)
        s = float(sum(per))
        sym_sweep.append({"k": k, "sum_per_sym": s, "per_sym": per})
        print(f"  k={k:>4.2f} sum={s:+8.4f}  per_sym=[{', '.join(f'{x:+.3f}' for x in per)}]",
              flush=True)
    best_sym = max(sym_sweep, key=lambda r: r["sum_per_sym"])

    obj = make_obj_loso_asym(folds)
    runs = de_search(obj, [(0.0, 0.0040), (0.0, 0.0040)])
    best = runs[0]
    thr_up, thr_dn = best["thr_up"], best["thr_dn"]
    de_per_sym = [float(vectorized_pnl(ev_gate_asymmetric(fk["pred"], thr_up, thr_dn),
                                        fk["mp_t"], fk["mp_th"]).sum()) for fk in folds]
    de_loso_sum = float(sum(de_per_sym))
    de_total = float(vectorized_pnl(ev_gate_asymmetric(pred_all, thr_up, thr_dn),
                                     mp_t_all, mp_th_all).sum())

    print(f"  best sym k={best_sym['k']}: {best_sym['sum_per_sym']:+.4f}", flush=True)
    print(f"  DE LOSO asym thr_up={thr_up:.6f} thr_dn={thr_dn:.6f}: {de_loso_sum:+.4f}",
          flush=True)
    print(f"  per_sym=[{', '.join(f'{x:+.3f}' for x in de_per_sym)}]", flush=True)
    print(f"  vs iter_012 (+{ITER012_LOSO}): {de_loso_sum-ITER012_LOSO:+.4f}", flush=True)
    print(f"  vs iter_013 (+{ITER013_LOSO}): {de_loso_sum-ITER013_LOSO:+.4f}", flush=True)
    print(f"  vs iter_014 (+{ITER014_LOSO}): {de_loso_sum-ITER014_LOSO:+.4f}", flush=True)

    return {"label": label, "sym_sweep": sym_sweep, "best_sym": best_sym,
            "de_loso": {"thr_up": thr_up, "thr_dn": thr_dn,
                        "sum_per_sym": de_loso_sum, "per_sym": de_per_sym,
                        "single_total": de_total}}


def main():
    print("=== T88 ensemble eval: NN (T88) + LGB (T88) ===", flush=True)
    weights = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.5, 1.0), (1.0, 1.5), (2.0, 1.0), (1.0, 2.0)]
    results_all = []
    for wn, wl in weights:
        df, p_nn, p_lgb = load_avg_pred(wn, wl)
        if (wn, wl) == weights[0]:
            corr = float(np.corrcoef(p_nn, p_lgb)[0, 1])
            print(f"\nNN vs LGB cross-corr: {corr:.4f}", flush=True)
        label = f"w_nn={wn} w_lgb={wl}"
        r = evaluate_full(df, label)
        r["w_nn"] = wn; r["w_lgb"] = wl
        results_all.append(r)
    best = max(results_all, key=lambda r: r["de_loso"]["sum_per_sym"])
    print(f"\n{'='*78}\nBEST: {best['label']} → DE LOSO {best['de_loso']['sum_per_sym']:+.4f} "
          f"(vs iter_014 {best['de_loso']['sum_per_sym']-ITER014_LOSO:+.4f})\n{'='*78}", flush=True)
    out = {"weights_tried": [{"w_nn": w[0], "w_lgb": w[1]} for w in weights],
           "results": results_all, "best": best,
           "iter012_ref": ITER012_LOSO, "iter013_ref": ITER013_LOSO,
           "iter014_ref": ITER014_LOSO}
    out_path = os.path.join(HERE, "ev_gate_ensemble_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
