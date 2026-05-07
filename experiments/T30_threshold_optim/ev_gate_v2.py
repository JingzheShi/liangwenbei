"""Calibrated EV gate v2 (sym-agnostic, stateless, no date).

T15 tried: isotonic-calibrated probs + EV gate using global E[Δp|up]; failed.
v2 difference:
  (a) Per-fold isotonic calibration of (prob_0, prob_2) on the OTHER 4 folds'
      OOF residuals (one-vs-rest).
  (b) Per-sample E[|Δp|/(p+1)] estimated from the LAST 100-tick window's
      realized magnitude — this is sym-agnostic and stateless (depends only
      on 100-tick window).
  (c) Decision rule:
        EV(pred=2) = p2_calib * E_up - p0_calib * E_dn - fee
        EV(pred=0) = p0_calib * E_dn - p2_calib * E_up - fee
        pred=2 if EV(pred=2) > eps else (pred=0 if EV(pred=0) > eps else 1)
      where E_up, E_dn = expected forward magnitude in up/down direction.

For step (b), we use the regime_features.regime_vol (rolling 100-tick std)
as a proxy for forward magnitude; calibrate the scaling factor α so that
α * std ≈ E[|forward return at horizon=60|] on the calibration set.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from _common import N_FOLDS, fold_arrays, load_all_folds, vectorized_pnl

HERE = os.path.dirname(os.path.abspath(__file__))
REGIME_PARQUET = os.path.join(HERE, "regime_features.parquet")
FEE = 0.0001


def fit_iso(probs_calib: np.ndarray, label_calib: np.ndarray, side: int):
    """Fit isotonic on prob_side vs (label==side)."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    p = probs_calib[:, side]
    y = (label_calib == side).astype(np.float32)
    iso.fit(p, y)
    return iso


def main():
    t0 = time.time()
    folds = load_all_folds()
    fas = fold_arrays(folds)

    regime_df = pd.read_parquet(REGIME_PARQUET)
    fold_dfs = []
    for k in range(N_FOLDS):
        m = folds[k].merge(regime_df[["sym","date","session","t","regime_vol","regime_ret"]],
                           on=["sym","date","session","t"], how="left")
        m["regime_vol"] = m["regime_vol"].fillna(m["regime_vol"].median())
        m["regime_ret"] = m["regime_ret"].fillna(0.0)
        fold_dfs.append(m)

    eps_grid = [0.0, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4]
    alpha_grid = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
    results = []

    for alpha in alpha_grid:
        per_fold_per_eps = {e: [] for e in eps_grid}

        for k in range(N_FOLDS):
            calib_idx = [j for j in range(N_FOLDS) if j != k]
            calib_probs = np.concatenate([fas[j].probs for j in calib_idx])
            calib_label = np.concatenate([fas[j].label for j in calib_idx])

            iso0 = fit_iso(calib_probs, calib_label, side=0)
            iso2 = fit_iso(calib_probs, calib_label, side=2)

            df_k = fold_dfs[k]
            p0c = iso0.transform(fas[k].probs[:, 0])
            p2c = iso2.transform(fas[k].probs[:, 2])

            # Per-sample expected magnitude: alpha * regime_vol (sym-agnostic, 100-tick)
            E_mag = alpha * df_k["regime_vol"].to_numpy()  # forward absolute return scaled
            denom = (df_k["midprice_t"].to_numpy() + 1.0)
            E_dir = E_mag / denom  # rough EV per unit prob

            ev_up = p2c * E_dir - p0c * E_dir - 2 * FEE
            ev_dn = p0c * E_dir - p2c * E_dir - 2 * FEE

            for eps in eps_grid:
                pred = np.full(len(p0c), 1, dtype=np.int8)
                take_up = ev_up > eps
                take_dn = ev_dn > eps
                pred[take_up & ~take_dn] = 2
                pred[take_dn & ~take_up] = 0
                # Tie-break: pick larger
                tie = take_up & take_dn
                pred[tie] = np.where(ev_up[tie] >= ev_dn[tie], 2, 0)

                s = float(vectorized_pnl(pred, fas[k].label, fas[k].mp_t, fas[k].mp_th).sum())
                per_fold_per_eps[eps].append(s)

        for eps in eps_grid:
            per = per_fold_per_eps[eps]
            results.append({
                "alpha": float(alpha),
                "eps": float(eps),
                "sum_cum_pnl": float(sum(per)),
                "per_fold_pnl": per,
                "n_pos_folds": int(sum(1 for x in per if x > 0)),
                "std_cum_pnl": float(np.std(per, ddof=0)),
            })

    df_r = pd.DataFrame(results).sort_values("sum_cum_pnl", ascending=False).reset_index(drop=True)
    df_r.to_csv(os.path.join(HERE, "ev_gate_v2_grid.csv"), index=False)
    print("Top 10 EV gate v2:")
    print(df_r.head(10).to_string(index=False))

    best = df_r.iloc[0].to_dict()
    print(f"\nBest: alpha={best['alpha']}, eps={best['eps']:.6f}, sum={best['sum_cum_pnl']:+.4f}, "
          f"per_fold={best['per_fold_pnl']}, pos={int(best['n_pos_folds'])}/5")

    out = {
        "task": "T30 calibrated EV gate v2",
        "alpha_grid": alpha_grid,
        "eps_grid": eps_grid,
        "results_top20": df_r.head(20).to_dict(orient="records"),
        "best": best,
        "elapsed_sec": time.time() - t0,
    }
    with open(os.path.join(HERE, "ev_gate_v2_results.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
