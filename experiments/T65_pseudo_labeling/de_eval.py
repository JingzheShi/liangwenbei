"""T65 DE 4D threshold search on pseudo-trained 5-seed-averaged test predictions.

Mirrors T59/de_eval.py — loads pseudo_pred_h60_seed{S}.parquet for the 5 seeds,
averages probs, computes:
  1) Raw argmax cum_pnl on full 442k test set + per-sym (LOSO-equivalent)
  2) DE 4D on a SINGLE 442k test set (objective = total cum_pnl)
  3) DE 4D treating sym0..4 as 5 "folds" (objective = SUM of per-sym cum_pnl)
     => This is the comparison vs iter_010 +24.52
  4) Same eval restricted to non-pseudo rows (held-out subset)
     => Fairer comparison since pseudo rows were used during training.

Saves de_results.json.
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

T53_DIR = os.path.join(ROOT, "experiments", "T53_r34_stage3")
sys.path.insert(0, T53_DIR)
from de_thresh import (  # noqa: E402
    FEE, PROB_COLS, gate_asymmetric, vectorized_pnl,
)

SYMS = (0, 1, 2, 3, 4)


def load_avg_pred(seeds, prefix="pseudo_pred_h60") -> pd.DataFrame:
    base = pd.read_parquet(os.path.join(HERE, f"{prefix}_seed{seeds[0]}.parquet"))
    p_sum = base[PROB_COLS].to_numpy(np.float64).copy()
    n = len(base)
    for s in seeds[1:]:
        df_s = pd.read_parquet(os.path.join(HERE, f"{prefix}_seed{s}.parquet"))
        if len(df_s) != n:
            raise RuntimeError(f"row count mismatch seed={s}: {len(df_s)} vs {n}")
        if not (df_s["sym"].equals(base["sym"]) and df_s["date"].equals(base["date"]) and df_s["t"].equals(base["t"])):
            raise RuntimeError(f"row order mismatch seed={s}")
        p_sum += df_s[PROB_COLS].to_numpy(np.float64)
    p_avg = p_sum / float(len(seeds))
    out = base.drop(columns=PROB_COLS).copy()
    out["prob_0"] = p_avg[:, 0].astype(np.float32)
    out["prob_1"] = p_avg[:, 1].astype(np.float32)
    out["prob_2"] = p_avg[:, 2].astype(np.float32)
    return out


def split_by_sym(df: pd.DataFrame):
    out = []
    for k in SYMS:
        sub = df[df["sym"] == k]
        out.append({
            "sym": int(k),
            "probs": sub[PROB_COLS].to_numpy(np.float32),
            "label": sub["true_label"].to_numpy(np.int64),
            "mp_t": sub["midprice_t"].to_numpy(np.float64),
            "mp_th": sub["midprice_th"].to_numpy(np.float64),
            "n": len(sub),
        })
    return out


def make_objective_loso(folds):
    def f(x):
        Tu, Td, du, dd = x
        s = 0.0
        for fk in folds:
            pred = gate_asymmetric(fk["probs"], Tu, Td, du, dd)
            s += vectorized_pnl(pred, fk["label"], fk["mp_t"], fk["mp_th"]).sum()
        return -float(s)
    return f


def make_objective_single(probs, label, mp_t, mp_th):
    def f(x):
        Tu, Td, du, dd = x
        pred = gate_asymmetric(probs, Tu, Td, du, dd)
        return -float(vectorized_pnl(pred, label, mp_t, mp_th).sum())
    return f


def de_search(objective_fn, bounds, seeds=(0, 1, 2, 7, 42)):
    runs = []
    for seed in seeds:
        t0 = time.time()
        result = differential_evolution(
            objective_fn, bounds=bounds, seed=seed, maxiter=80, popsize=24,
            polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            updating="deferred", workers=1, init="sobol",
        )
        Tu, Td, du, dd = result.x
        runs.append({
            "seed": int(seed),
            "T_up": float(Tu), "T_dn": float(Td),
            "d_up": float(du), "d_dn": float(dd),
            "obj_val": float(-result.fun),
            "nfev": int(result.nfev),
            "elapsed_sec": float(time.time() - t0),
        })
    runs.sort(key=lambda r: r["obj_val"], reverse=True)
    return runs


def per_sym_pnl_at_thresh(folds, Tu, Td, du, dd):
    per = []
    n_active_per_sym = []
    for f in folds:
        pred = gate_asymmetric(f["probs"], Tu, Td, du, dd)
        per.append(float(vectorized_pnl(pred, f["label"], f["mp_t"], f["mp_th"]).sum()))
        n_active_per_sym.append(int((pred != 1).sum()))
    return per, n_active_per_sym


def main():
    seeds = [1, 7, 13, 42, 100]
    print(f"=== T65 DE 4D thresh search seeds={seeds} ===", flush=True)

    t0 = time.time()
    df = load_avg_pred(seeds)
    print(f"  loaded {len(df):,} rows in {time.time()-t0:.1f}s", flush=True)

    folds_full = split_by_sym(df)
    for fk in folds_full:
        print(f"  sym={fk['sym']}: n={fk['n']:,}", flush=True)

    probs_all = df[PROB_COLS].to_numpy(np.float32)
    label_all = df["true_label"].to_numpy(np.int64)
    mp_t_all = df["midprice_t"].to_numpy(np.float64)
    mp_th_all = df["midprice_th"].to_numpy(np.float64)

    # Raw argmax baseline (full)
    raw_per_sym = []
    for fk in folds_full:
        p = fk["probs"].argmax(axis=1).astype(np.int8)
        raw_per_sym.append(float(vectorized_pnl(p, fk["label"], fk["mp_t"], fk["mp_th"]).sum()))
    pred_raw = probs_all.argmax(axis=1).astype(np.int8)
    raw_total = float(vectorized_pnl(pred_raw, label_all, mp_t_all, mp_th_all).sum())
    print(f"\n  RAW argmax (full 442k): total_cum_pnl={raw_total:+.4f}", flush=True)
    print(f"  RAW per-sym: {[round(x,3) for x in raw_per_sym]}  loso_equiv_sum={sum(raw_per_sym):+.4f}", flush=True)

    # Raw argmax baseline (held-out: not pseudo)
    held = df[~df["is_pseudo"]].copy()
    print(f"\n  Held-out (non-pseudo): n={len(held):,} ({100*len(held)/len(df):.2f}%)", flush=True)
    folds_held = split_by_sym(held)
    raw_per_sym_held = []
    for fk in folds_held:
        p = fk["probs"].argmax(axis=1).astype(np.int8)
        raw_per_sym_held.append(float(vectorized_pnl(p, fk["label"], fk["mp_t"], fk["mp_th"]).sum()))
    print(f"  RAW per-sym (held): {[round(x,3) for x in raw_per_sym_held]}  loso_equiv_sum={sum(raw_per_sym_held):+.4f}", flush=True)

    bounds = [(0.34, 0.75), (0.34, 0.75), (0.0, 0.30), (0.0, 0.30)]

    # --- DE LOSO-EQUIV on FULL test ---
    print(f"\n[1/2] DE LOSO (full 442k, sum per-sym) ...", flush=True)
    obj_loso = make_objective_loso(folds_full)
    de_loso = de_search(obj_loso, bounds)
    best_loso = de_loso[0]
    Tu, Td, du, dd = best_loso["T_up"], best_loso["T_dn"], best_loso["d_up"], best_loso["d_dn"]
    de_loso_per_sym, de_loso_n_active = per_sym_pnl_at_thresh(folds_full, Tu, Td, du, dd)
    de_loso_sum = float(sum(de_loso_per_sym))
    print(f"  best DE-loso: T_up={Tu:.4f} T_dn={Td:.4f} d_up={du:.4f} d_dn={dd:.4f} "
          f"-> sum_per_sym={de_loso_sum:+.4f}  per_sym={[round(x,3) for x in de_loso_per_sym]}", flush=True)

    # --- DE LOSO on HELD-OUT (non-pseudo) ---
    print(f"\n[2/2] DE LOSO (held-out non-pseudo, sum per-sym) ...", flush=True)
    obj_loso_held = make_objective_loso(folds_held)
    de_loso_held = de_search(obj_loso_held, bounds)
    best_loso_held = de_loso_held[0]
    Tu2, Td2, du2, dd2 = best_loso_held["T_up"], best_loso_held["T_dn"], best_loso_held["d_up"], best_loso_held["d_dn"]
    de_loso_held_per_sym, de_loso_held_n_active = per_sym_pnl_at_thresh(folds_held, Tu2, Td2, du2, dd2)
    de_loso_held_sum = float(sum(de_loso_held_per_sym))
    print(f"  best DE-loso(held): T_up={Tu2:.4f} T_dn={Td2:.4f} d_up={du2:.4f} d_dn={dd2:.4f} "
          f"-> sum_per_sym={de_loso_held_sum:+.4f}  per_sym={[round(x,3) for x in de_loso_held_per_sym]}", flush=True)

    # Apply best-thresh from full-test DE to held-out (deployment-realistic)
    # The "production" decision: optimize threshold on the full test, apply on held
    held_per_sym_at_full_thresh, held_n_active_at_full_thresh = per_sym_pnl_at_thresh(
        folds_held, Tu, Td, du, dd
    )
    held_sum_at_full_thresh = float(sum(held_per_sym_at_full_thresh))
    print(f"\n  Held-out cum_pnl using full-test-DE thresh: {held_sum_at_full_thresh:+.4f}", flush=True)
    print(f"  Held-out per-sym at full thresh: {[round(x,3) for x in held_per_sym_at_full_thresh]}", flush=True)

    # Apply best-thresh from held DE to full test (anti-overfit check)
    full_per_sym_at_held_thresh, _ = per_sym_pnl_at_thresh(
        folds_full, Tu2, Td2, du2, dd2
    )
    full_sum_at_held_thresh = float(sum(full_per_sym_at_held_thresh))
    print(f"  Full-test cum_pnl using held-DE thresh: {full_sum_at_held_thresh:+.4f}", flush=True)

    iter010 = 24.52  # iter_010 reference (DE LOSO-equiv on full 442k test)

    out = {
        "task": "T65 DE 4D thresh on pseudo-trained 5-seed-avg",
        "seeds": seeds,
        "n_test_total": len(df),
        "n_test_held": len(held),
        "n_test_pseudo": int(df["is_pseudo"].sum()),
        "raw_argmax_full": {
            "total_cum_pnl": raw_total,
            "per_sym": raw_per_sym,
            "loso_equiv_sum": float(sum(raw_per_sym)),
        },
        "raw_argmax_held": {
            "per_sym": raw_per_sym_held,
            "loso_equiv_sum": float(sum(raw_per_sym_held)),
        },
        "de_loso_full": {
            "best": best_loso,
            "all_runs": de_loso,
            "per_sym": de_loso_per_sym,
            "n_active_per_sym": de_loso_n_active,
            "sum": de_loso_sum,
        },
        "de_loso_held": {
            "best": best_loso_held,
            "all_runs": de_loso_held,
            "per_sym": de_loso_held_per_sym,
            "n_active_per_sym": de_loso_held_n_active,
            "sum": de_loso_held_sum,
        },
        "held_at_full_thresh": {
            "per_sym": held_per_sym_at_full_thresh,
            "sum": held_sum_at_full_thresh,
            "n_active_per_sym": held_n_active_at_full_thresh,
            "thresh": [Tu, Td, du, dd],
        },
        "full_at_held_thresh": {
            "per_sym": full_per_sym_at_held_thresh,
            "sum": full_sum_at_held_thresh,
            "thresh": [Tu2, Td2, du2, dd2],
        },
        "compare_iter010": {
            "iter010_de_loso_sum": iter010,
            "t65_de_loso_full_vs_iter010": de_loso_sum - iter010,
            "t65_de_loso_held_vs_iter010": de_loso_held_sum - iter010,
        },
    }
    out_path = os.path.join(HERE, "de_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote -> {out_path}", flush=True)

    print(f"\n{'='*78}\n=== SUMMARY ===\n{'='*78}", flush=True)
    print(f"RAW argmax(full)  : loso_equiv_sum={sum(raw_per_sym):+.4f}", flush=True)
    print(f"RAW argmax(held)  : loso_equiv_sum={sum(raw_per_sym_held):+.4f}", flush=True)
    print(f"DE LOSO  (full)   : sum_per_sym={de_loso_sum:+.4f}", flush=True)
    print(f"DE LOSO  (held)   : sum_per_sym={de_loso_held_sum:+.4f}", flush=True)
    print(f"vs iter_010 (24.52): full_diff={de_loso_sum - iter010:+.4f}  held_diff={de_loso_held_sum - iter010:+.4f}", flush=True)


if __name__ == "__main__":
    main()
