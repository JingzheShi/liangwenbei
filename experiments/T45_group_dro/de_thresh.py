"""Apply iter_006 DE-tuned asymmetric thresholds to T45 single-seed Group DRO
predictions, plus a small neighborhood sweep to find the per-T45 optimum.

Compares to baselines:
  * T44 single seed sum cum_pnl (no DRO, raw argmax) ~ -5.65
  * iter_006 5-seed ensemble + asym thresh = +13.61 (the bar to beat)
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
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

T30_DIR = os.path.join(ROOT, "experiments", "T30_threshold_optim")
sys.path.insert(0, T30_DIR)
from _common import gate_asymmetric, vectorized_pnl  # noqa: E402

PROB_COLS = ["prob_0", "prob_1", "prob_2"]
N_FOLDS = 5
TAG = "tau_0_5_K50_loso"


def load_t45_folds():
    folds = {}
    for k in range(N_FOLDS):
        p = os.path.join(HERE, f"{TAG}_pred_seed42_held{k}.parquet")
        folds[k] = pd.read_parquet(p)
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
    """Argmax baseline (no thresholding)."""
    per = []
    for fa in fas:
        pred = fa["probs"].argmax(axis=1).astype(np.int8)
        per.append(float(vectorized_pnl(pred, fa["label"], fa["mp_t"], fa["mp_th"]).sum()))
    return per


def main():
    t0 = time.time()
    folds = load_t45_folds()
    fas = fold_arrays(folds)
    print(f"loaded T45 LOSO predictions tag={TAG}: "
          f"per-fold n={[len(f['label']) for f in fas]}", flush=True)

    # 0. Raw argmax baseline (sanity, should match training script ~+3.83)
    raw = raw_argmax_pnl(fas)
    print(f"\n[raw argmax]  per-fold={[f'{x:+.3f}' for x in raw]}  "
          f"sum={sum(raw):+.4f}", flush=True)

    # 1. Apply iter_006 DE optimum unchanged
    Tu0, Td0, du0, dd0 = 0.448, 0.391, 0.260, 0.024
    per_iter006 = eval_thresh(fas, Tu0, Td0, du0, dd0)
    print(f"\n[iter_006 DE optimum (Tu={Tu0}, Td={Td0}, du={du0}, dd={dd0})]")
    print(f"  per-fold={[f'{x:+.3f}' for x in per_iter006]}  "
          f"sum={sum(per_iter006):+.4f}")
    print(f"  vs +13.61 ensemble baseline: "
          f"{'+' if sum(per_iter006) > 13.61 else ''}{sum(per_iter006) - 13.61:+.3f}")

    # 2. Coarse sweep (single-seed prob distribution may differ from ensemble)
    Tu_grid = np.round(np.arange(0.34, 0.52 + 1e-9, 0.02), 4)
    Td_grid = np.round(np.arange(0.34, 0.50 + 1e-9, 0.02), 4)
    du_grid = np.round(np.arange(0.00, 0.30 + 1e-9, 0.04), 4)
    dd_grid = np.round(np.arange(0.00, 0.20 + 1e-9, 0.04), 4)
    n = len(Tu_grid) * len(Td_grid) * len(du_grid) * len(dd_grid)
    print(f"\nCoarse grid: |Tu|={len(Tu_grid)} |Td|={len(Td_grid)} "
          f"|du|={len(du_grid)} |dd|={len(dd_grid)} = {n}")

    rows = []
    label = [fa["label"] for fa in fas]
    mp_t = [fa["mp_t"] for fa in fas]
    mp_th = [fa["mp_th"] for fa in fas]
    probs = [fa["probs"] for fa in fas]
    for Tu in Tu_grid:
        for Td in Td_grid:
            for du in du_grid:
                for dd in dd_grid:
                    per = []
                    for k in range(N_FOLDS):
                        pred = gate_asymmetric(probs[k], float(Tu), float(Td),
                                               float(du), float(dd))
                        per.append(float(vectorized_pnl(
                            pred, label[k], mp_t[k], mp_th[k]).sum()))
                    sum_p = sum(per)
                    rows.append({
                        "T_up": float(Tu), "T_dn": float(Td),
                        "d_up": float(du), "d_dn": float(dd),
                        "sum_cum_pnl": sum_p,
                        "n_pos_folds": int(sum(1 for x in per if x > 0)),
                        **{f"fold{k}_pnl": per[k] for k in range(N_FOLDS)},
                    })

    df = pd.DataFrame(rows).sort_values("sum_cum_pnl", ascending=False).reset_index(drop=True)
    csv_path = os.path.join(HERE, "de_thresh_grid.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nDone sweep in {time.time()-t0:.1f}s -> {csv_path}")
    print(f"\nTop 20:")
    print(df.head(20).to_string(index=False))

    best = df.iloc[0]
    out = {
        "task": "T45 Group DRO + DE thresh (single-seed=42)",
        "tag": TAG,
        "raw_argmax_per_fold": raw,
        "raw_argmax_sum": float(sum(raw)),
        "iter006_DE_unchanged": {
            "thresholds": {"T_up": Tu0, "T_dn": Td0, "d_up": du0, "d_dn": dd0},
            "per_fold": per_iter006,
            "sum_cum_pnl": float(sum(per_iter006)),
        },
        "best_t45_grid": {
            "thresholds": {
                "T_up": float(best["T_up"]), "T_dn": float(best["T_dn"]),
                "d_up": float(best["d_up"]), "d_dn": float(best["d_dn"]),
            },
            "per_fold": [float(best[f"fold{k}_pnl"]) for k in range(N_FOLDS)],
            "sum_cum_pnl": float(best["sum_cum_pnl"]),
            "n_pos_folds": int(best["n_pos_folds"]),
        },
        "ensemble_baseline_to_beat": 13.61,
        "comparison": {
            "best_vs_13_61": float(best["sum_cum_pnl"]) - 13.61,
            "iter006_unchanged_vs_13_61": float(sum(per_iter006)) - 13.61,
        },
        "elapsed_sec": time.time() - t0,
    }
    out_path = os.path.join(HERE, "de_thresh_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults -> {out_path}", flush=True)
    print(f"\nFINAL: best T45 sum_cum_pnl = {out['best_t45_grid']['sum_cum_pnl']:+.4f} "
          f"(vs +13.61 ensemble: {out['comparison']['best_vs_13_61']:+.4f})")


if __name__ == "__main__":
    main()
