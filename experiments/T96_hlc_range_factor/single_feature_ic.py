"""T96: Single-feature IC and LightGBM gain-rank tests on V4 split.

For each new feature:
  - Pearson IC vs y_regr_60 on (train→val of V4 split)
  - Spearman IC
  - Single-feature LightGBM (depth-3, 100 rounds): val MSE / val pnl
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
T68_CACHE = os.path.join(ROOT, "experiments", "T68_stage5_features", "cache")
CACHE = os.path.join(HERE, "cache")

import lightgbm as lgb

FEE = 0.0001


def regr_target(mp_t, mp_th):
    return ((mp_th.astype(np.float64) - mp_t.astype(np.float64))
            / (mp_t.astype(np.float64) + 1.0)).astype(np.float32)


def main():
    print("=== T96 single-feature IC + gain-rank ===\n", flush=True)
    t0 = time.time()
    P = np.load(os.path.join(T68_CACHE, "schemeP_train.npz"))
    E = np.load(os.path.join(CACHE, "T96_train.npz"))
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    # Verify alignment
    assert np.array_equal(P["sym"].astype(np.int32), E["sym"].astype(np.int32)), "sym mismatch"
    assert np.array_equal(P["date"].astype(np.int32), E["date"].astype(np.int32)), "date mismatch"
    assert np.array_equal(P["sess_idx"].astype(np.int32), E["sess"].astype(np.int32)), "sess mismatch"
    assert np.array_equal(P["t"].astype(np.int32), E["t"].astype(np.int32)), "t mismatch"
    print("  alignment OK\n", flush=True)

    with open(os.path.join(CACHE, "T96_feat_names.txt")) as f:
        feat_names = [ln.strip() for ln in f]

    X_extra = E["X_extra"]
    date = P["date"]
    y_regr = regr_target(P["mp_t"], P["mp_t60"])
    y_cls = P["y60"]

    # V4 split
    mva = date >= 76
    mtr = ~mva
    print(f"train n={mtr.sum():,}  val n={mva.sum():,}\n", flush=True)

    # 1) Pearson + Spearman IC on the full training set (using val for held-out check)
    results = []
    print(f"{'feature':<32s} {'mean(tr)':>12s} {'std(tr)':>10s} {'IC(va)':>9s} {'IC_sp(va)':>10s} {'sgl_lgb_va_mse':>15s} {'sgl_lgb_va_corr':>16s}", flush=True)
    print("-" * 100, flush=True)

    y_va = y_regr[mva]
    for i, fn in enumerate(feat_names):
        col_tr = X_extra[mtr, i].astype(np.float64)
        col_va = X_extra[mva, i].astype(np.float64)

        mean_tr = col_tr.mean()
        std_tr = col_tr.std()
        if std_tr < 1e-12:
            ic_va = 0.0
            ic_sp = 0.0
        else:
            ic_va = float(np.corrcoef(col_va, y_va)[0, 1])
            # Spearman approximation (use rankdata)
            from scipy.stats import rankdata
            ic_sp = float(np.corrcoef(rankdata(col_va), rankdata(y_va))[0, 1])

        # Quick single-feature LightGBM (depth 3, 50 trees)
        dtrain = lgb.Dataset(X_extra[mtr, i:i+1].astype(np.float32),
                             label=y_regr[mtr], free_raw_data=False)
        dval = lgb.Dataset(X_extra[mva, i:i+1].astype(np.float32),
                           label=y_va, reference=dtrain, free_raw_data=False)
        params = {
            "objective": "regression_l2",
            "learning_rate": 0.05,
            "num_leaves": 7,  # depth ~3
            "min_data_in_leaf": 200,
            "feature_fraction": 1.0,
            "bagging_fraction": 1.0,
            "lambda_l2": 1.0,
            "num_threads": 8,
            "seed": 42,
            "verbose": -1,
            "metric": "l2",
        }
        booster = lgb.train(params, dtrain, num_boost_round=50,
                            valid_sets=[dval], valid_names=["val"], callbacks=[])
        va_pred = booster.predict(X_extra[mva, i:i+1].astype(np.float32))
        sgl_mse = float(((va_pred - y_va) ** 2).mean())
        sgl_corr = float(np.corrcoef(va_pred, y_va)[0, 1])

        print(f"{fn:<32s} {mean_tr:>+12.4e} {std_tr:>10.3e} "
              f"{ic_va:>+9.5f} {ic_sp:>+10.5f} {sgl_mse:>15.7e} {sgl_corr:>+16.5f}",
              flush=True)
        results.append({
            "feature": fn,
            "mean_tr": float(mean_tr),
            "std_tr": float(std_tr),
            "ic_va_pearson": float(ic_va),
            "ic_va_spearman": float(ic_sp),
            "sgl_lgb_va_mse": sgl_mse,
            "sgl_lgb_va_corr": sgl_corr,
        })

    out_path = os.path.join(HERE, "single_feature_ic.json")
    with open(out_path, "w") as f:
        json.dump({"feat_results": results}, f, indent=2)
    print(f"\nwrote {out_path}", flush=True)

    # Sort by abs(IC) and display top
    print("\n=== Top features by |IC_va_pearson| ===", flush=True)
    results.sort(key=lambda r: abs(r["ic_va_pearson"]), reverse=True)
    for r in results[:10]:
        print(f"  {r['feature']:<32s} ic={r['ic_va_pearson']:+.5f} "
              f"ic_sp={r['ic_va_spearman']:+.5f} corr={r['sgl_lgb_va_corr']:+.5f}",
              flush=True)


if __name__ == "__main__":
    main()
