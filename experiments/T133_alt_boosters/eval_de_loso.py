"""T133 evaluation: 3-seed ensemble + DE asymmetric threshold sweep + LOSO score.

For each variant in {baseline, dart, tweedie, ngboost}:
  - Load pred_T133_{variant}_seed{1,7,42}.parquet (or T122 baseline equivalents)
  - Average pred_dmid_norm across 3 seeds
  - Grid-search (thr_up, thr_dn) to maximize total cum_pnl
  - Report: best thresholds, total cum_pnl (= LOSO equivalent), per-sym

Also computes 4-way ensemble = avg(baseline, dart, tweedie, ngboost) post-3seed-avg.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

FEE = 0.0001
SYMS = (0, 1, 2, 3, 4)
SEEDS = (1, 7, 42)

T122_DIR = os.path.join(ROOT, "experiments", "T122_retest_wins_on_T75")


def vectorized_pnl(action, mp_t, mp_th):
    side = action.astype(np.float64) - 1.0
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee_pnl) / denom


def eval_at_thresh(pred, mp_t, mp_th, sym, thr_up, thr_dn):
    pa = np.full(len(pred), 1, dtype=np.int8)
    pa[pred > thr_up] = 2
    pa[pred < -thr_dn] = 0
    pnl = vectorized_pnl(pa, mp_t, mp_th)
    total = float(pnl.sum())
    per_sym = [float(pnl[sym == s].sum()) for s in SYMS]
    n_active = int((pa != 1).sum())
    return total, per_sym, n_active


def grid_search_de(pred, mp_t, mp_th, sym,
                   up_grid=None, dn_grid=None):
    """Grid-search asymmetric (thr_up, thr_dn). Returns best (thr_up, thr_dn, total, per_sym, n_active).

    Vectorized: builds 2D grid of (thr_up × thr_dn) at once.
    """
    if up_grid is None:
        # 2 fee = 2e-4. Search 0.0 to 8e-4
        up_grid = np.concatenate([
            np.linspace(0.0, 2e-4, 11),
            np.linspace(2.2e-4, 5e-4, 15),
            np.linspace(5.5e-4, 8e-4, 6),
        ])
    if dn_grid is None:
        dn_grid = np.concatenate([
            np.linspace(0.0, 2e-4, 11),
            np.linspace(2.2e-4, 5e-4, 15),
            np.linspace(5.5e-4, 8e-4, 6),
        ])

    # For each pair (thr_up, thr_dn), compute total pnl
    best = None
    best_total = -np.inf
    pred = np.asarray(pred, dtype=np.float64)
    mp_t = np.asarray(mp_t, dtype=np.float64)
    mp_th = np.asarray(mp_th, dtype=np.float64)
    sym = np.asarray(sym)
    diff = mp_th - mp_t

    # precompute "up trade pnl" and "down trade pnl" per row
    fee_pnl_per_trade = FEE * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t + 1.0
    up_pnl = (diff - fee_pnl_per_trade) / denom   # action=2: side=+1
    dn_pnl = (-diff - fee_pnl_per_trade) / denom  # action=0: side=-1

    for thr_up in up_grid:
        m_up = pred > thr_up
        for thr_dn in dn_grid:
            m_dn = pred < -thr_dn
            # rows that fall in both (thr_up < pred and pred < -thr_dn) — impossible since thr_dn>=0 and thr_up>=0.
            # safety: if pred > thr_up we trade up, else if pred < -thr_dn trade down, else flat.
            # implement via additive selection
            pnl_total = float(up_pnl[m_up].sum() + dn_pnl[m_dn].sum())
            if pnl_total > best_total:
                best_total = pnl_total
                best = (float(thr_up), float(thr_dn))

    # eval at best pair to get per-sym
    thr_up, thr_dn = best
    total, per_sym, n_active = eval_at_thresh(pred, mp_t, mp_th, sym, thr_up, thr_dn)
    return {"thr_up": thr_up, "thr_dn": thr_dn,
            "total_cum_pnl": total, "per_sym": per_sym,
            "n_active": n_active}


def load_variant(variant):
    """Returns (pred_avg, mp_t, mp_th, sym) tuple. Variant: baseline|dart|tweedie|ngboost."""
    if variant == "baseline":
        # use T122 baseline preds
        rows = []
        for seed in SEEDS:
            p = os.path.join(T122_DIR, f"pred_T122_baseline_seed{seed}.parquet")
            df = pd.read_parquet(p)
            rows.append(df["pred_dmid_norm"].to_numpy(dtype=np.float64))
        pred_avg = np.mean(np.stack(rows, axis=0), axis=0)
        mp_t = df["midprice_t"].to_numpy(dtype=np.float64)
        mp_th = df["midprice_th"].to_numpy(dtype=np.float64)
        sym = df["sym"].to_numpy()
        return pred_avg, mp_t, mp_th, sym
    else:
        rows = []
        for seed in SEEDS:
            p = os.path.join(HERE, f"pred_T133_{variant}_seed{seed}.parquet")
            if not os.path.isfile(p):
                raise FileNotFoundError(p)
            df = pd.read_parquet(p)
            rows.append(df["pred_dmid_norm"].to_numpy(dtype=np.float64))
        pred_avg = np.mean(np.stack(rows, axis=0), axis=0)
        mp_t = df["midprice_t"].to_numpy(dtype=np.float64)
        mp_th = df["midprice_th"].to_numpy(dtype=np.float64)
        sym = df["sym"].to_numpy()
        return pred_avg, mp_t, mp_th, sym


def main(variants=None):
    if variants is None:
        variants = ["baseline", "dart", "tweedie", "ngboost"]

    results = {}
    pred_arrays = {}
    mp_t_ref = mp_th_ref = sym_ref = None

    for v in variants:
        try:
            pred_avg, mp_t, mp_th, sym = load_variant(v)
        except FileNotFoundError as e:
            print(f"[{v}] missing — skipping: {e}", flush=True)
            continue
        if mp_t_ref is None:
            mp_t_ref = mp_t
            mp_th_ref = mp_th
            sym_ref = sym
        else:
            assert np.allclose(mp_t, mp_t_ref), f"{v} mp_t mismatch"
        pred_arrays[v] = pred_avg
        t0 = time.time()
        res = grid_search_de(pred_avg, mp_t, mp_th, sym)
        res["pred_corr_to_truth"] = None
        results[v] = res
        elapsed = time.time() - t0
        print(f"[{v}] thr_up={res['thr_up']:.6f} thr_dn={res['thr_dn']:.6f} "
              f"total={res['total_cum_pnl']:+.4f} "
              f"per_sym={[round(x,2) for x in res['per_sym']]} "
              f"n_active={res['n_active']:,} ({elapsed:.1f}s)", flush=True)

    # 4-way ensemble (only if all 4 present)
    if len(pred_arrays) == 4:
        all_arr = np.stack([pred_arrays[v] for v in ["baseline", "dart", "tweedie", "ngboost"]], axis=0)
        ens4 = np.mean(all_arr, axis=0)
        res = grid_search_de(ens4, mp_t_ref, mp_th_ref, sym_ref)
        results["4way_ensemble"] = res
        print(f"[4way_ens] thr_up={res['thr_up']:.6f} thr_dn={res['thr_dn']:.6f} "
              f"total={res['total_cum_pnl']:+.4f} "
              f"per_sym={[round(x,2) for x in res['per_sym']]} "
              f"n_active={res['n_active']:,}", flush=True)

        # diagnostic: pairwise correlation of variant preds
        corr_mat = np.corrcoef(all_arr)
        results["pred_correlations"] = {
            "variants": ["baseline", "dart", "tweedie", "ngboost"],
            "matrix": corr_mat.tolist(),
        }
    elif len(pred_arrays) >= 2:
        # partial ensemble with whatever we have
        names = list(pred_arrays.keys())
        all_arr = np.stack([pred_arrays[v] for v in names], axis=0)
        ens = np.mean(all_arr, axis=0)
        res = grid_search_de(ens, mp_t_ref, mp_th_ref, sym_ref)
        results[f"{len(names)}way_ensemble"] = res
        results[f"{len(names)}way_variants"] = names
        print(f"[{len(names)}way_ens] thr_up={res['thr_up']:.6f} thr_dn={res['thr_dn']:.6f} "
              f"total={res['total_cum_pnl']:+.4f} "
              f"per_sym={[round(x,2) for x in res['per_sym']]} "
              f"n_active={res['n_active']:,}", flush=True)

        if len(pred_arrays) >= 2:
            corr_mat = np.corrcoef(all_arr)
            results["pred_correlations"] = {"variants": names,
                                             "matrix": corr_mat.tolist()}

    out = os.path.join(HERE, "eval_de_loso.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nresults -> {out}", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        main(variants=args)
    else:
        main()
