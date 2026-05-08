"""R_T75L2_pairwise DE asym thresh search per variant.

For each variant in {baseline, trick}:
  - Average pred_dmid_norm over 3 seeds (1, 7, 42)
  - Run DE LOSO (per-sym sum) thresh search
  - Report DE LOSO, thr_up, thr_dn, per-sym PnL
  - Compute delta vs baseline
"""
from __future__ import annotations

import json
import os
import time
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

OUT_DIR = "/root/lwb_work_t75_pairwise"
SYMS = (0, 1, 2, 3, 4)
SEEDS = (1, 7, 42)
FEE = 0.0001
VARIANTS = ("baseline", "trick")


def vectorized_pnl(pred_action, mp_t, mp_th):
    side = pred_action.astype(np.float64) - 1.0
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee_pnl) / denom


def ev_gate(pred, thr_up, thr_dn):
    a = np.full(len(pred), 1, dtype=np.int8)
    a[pred > thr_up] = 2
    a[pred < -thr_dn] = 0
    return a


def load_avg_pred(variant):
    paths = [os.path.join(OUT_DIR, f"pred_T75L2_{variant}_seed{s}.parquet")
             for s in SEEDS]
    base = pd.read_parquet(paths[0])
    p_sum = base["pred_dmid_norm"].to_numpy(np.float64).copy()
    for p in paths[1:]:
        df = pd.read_parquet(p)
        if not (df["sym"].equals(base["sym"]) and df["date"].equals(base["date"])
                and df["t"].equals(base["t"])):
            raise RuntimeError(f"row order mismatch {p}")
        p_sum += df["pred_dmid_norm"].to_numpy(np.float64)
    return base, p_sum / float(len(paths))


def per_sym_pnl(pred, sym_arr, mp_t, mp_th, thr_up, thr_dn):
    per = []
    for k in SYMS:
        m = sym_arr == k
        if m.sum() == 0:
            per.append(0.0); continue
        a = ev_gate(pred[m], thr_up, thr_dn)
        per.append(float(vectorized_pnl(a, mp_t[m], mp_th[m]).sum()))
    return per


def de_loso(pred, sym_arr, mp_t, mp_th, de_seeds=(0, 42), maxiter=60, popsize=20):
    fold_pred, fold_mpt, fold_mpth = [], [], []
    for k in SYMS:
        m = sym_arr == k
        fold_pred.append(pred[m])
        fold_mpt.append(mp_t[m])
        fold_mpth.append(mp_th[m])

    def f(x):
        thr_up, thr_dn = x
        s = 0.0
        for i in range(len(SYMS)):
            a = ev_gate(fold_pred[i], thr_up, thr_dn)
            s += vectorized_pnl(a, fold_mpt[i], fold_mpth[i]).sum()
        return -float(s)

    best, runs = None, []
    for sd in de_seeds:
        result = differential_evolution(
            f, bounds=[(0.0, 0.0040), (0.0, 0.0040)], seed=sd,
            maxiter=maxiter, popsize=popsize, polish=True, tol=1e-7,
            mutation=(0.5, 1.0), recombination=0.7, updating="deferred",
            workers=1, init="sobol",
        )
        run = {"seed": int(sd), "thr_up": float(result.x[0]),
               "thr_dn": float(result.x[1]), "obj_val": float(-result.fun)}
        runs.append(run)
        if best is None or run["obj_val"] > best["obj_val"]:
            best = run
    return best, runs


def k1_eval(pred, sym_arr, mp_t, mp_th):
    fee_thr = 2.0 * FEE
    a = ev_gate(pred, fee_thr, fee_thr)
    cum = float(vectorized_pnl(a, mp_t, mp_th).sum())
    per = per_sym_pnl(pred, sym_arr, mp_t, mp_th, fee_thr, fee_thr)
    return cum, per


def main():
    print("=== R_T75L2_pairwise DE thresh search ===", flush=True)
    out = {}
    for variant in VARIANTS:
        print(f"\n--- variant={variant} ---", flush=True)
        try:
            base, pred = load_avg_pred(variant)
        except Exception as e:
            print(f"  skip variant={variant}: {e}", flush=True)
            continue
        sym_arr = base["sym"].to_numpy(np.int64)
        mp_t = base["midprice_t"].to_numpy(np.float64)
        mp_th = base["midprice_th"].to_numpy(np.float64)

        k1_cum, k1_per = k1_eval(pred, sym_arr, mp_t, mp_th)
        print(f"  k=1 cum_pnl={k1_cum:+.4f}  per_sym={[f'{x:+.2f}' for x in k1_per]}",
              flush=True)

        t0 = time.time()
        best, runs = de_loso(pred, sym_arr, mp_t, mp_th)
        de_cum = best["obj_val"]
        thr_up, thr_dn = best["thr_up"], best["thr_dn"]
        de_per = per_sym_pnl(pred, sym_arr, mp_t, mp_th, thr_up, thr_dn)
        print(f"  DE LOSO obj={de_cum:+.4f}  thr_up={thr_up:.6f} thr_dn={thr_dn:.6f}",
              flush=True)
        print(f"  DE per_sym={[f'{x:+.2f}' for x in de_per]}  ({time.time()-t0:.1f}s)",
              flush=True)

        out[variant] = {
            "variant": variant,
            "n_seeds": len(SEEDS), "seeds": list(SEEDS),
            "k1_cum_pnl": k1_cum, "k1_per_sym": k1_per,
            "de_loso_cum_pnl": de_cum,
            "de_thr_up": thr_up, "de_thr_dn": thr_dn,
            "de_per_sym": de_per,
            "de_runs": runs,
        }

    # Compute deltas
    if "baseline" in out and "trick" in out:
        b = out["baseline"]
        t = out["trick"]
        delta_de = t["de_loso_cum_pnl"] - b["de_loso_cum_pnl"]
        delta_per_sym = [t["de_per_sym"][i] - b["de_per_sym"][i] for i in range(5)]
        out["delta"] = {
            "de_loso_cum_pnl": delta_de,
            "k1_cum_pnl": t["k1_cum_pnl"] - b["k1_cum_pnl"],
            "de_per_sym_delta": delta_per_sym,
            "de_per_sym_min_delta": min(delta_per_sym),
        }
        print(f"\n=== DELTA: trick - baseline ===")
        print(f"  DE LOSO: {delta_de:+.4f}")
        print(f"  k=1:     {t['k1_cum_pnl'] - b['k1_cum_pnl']:+.4f}")
        print(f"  per_sym: {[f'{x:+.2f}' for x in delta_per_sym]}")

    out_path = os.path.join(OUT_DIR, "de_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
