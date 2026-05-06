"""T12 alt: LightGBM meta-learner for multi-horizon stacking.

Same data layout as train_meta.py, but meta = small LightGBM (num_leaves=15,
num_boost_round=200, max_depth=4). LGB can learn the nonlinear interactions
between horizons that linear models smooth away.

Outputs:
    meta_oof_h10_lgbm.parquet
    fold_summaries_lgbm.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, Any

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss

HERE = os.path.dirname(os.path.abspath(__file__))
T5B = os.path.abspath(os.path.join(HERE, "..", "T5b_features_multihorizon"))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from src.eval.pnl import _per_horizon_metrics  # noqa: E402

HORIZONS = [5, 10, 20, 40, 60]
KEY_COLS = ["sym", "date", "session", "t"]
TARGET_HORIZON = 10

# Sweep over a few small LightGBM configs
LGBM_CONFIGS = [
    {"num_leaves": 15, "max_depth": 4, "num_boost_round": 200,
     "learning_rate": 0.05, "min_data_in_leaf": 50, "name": "lgbm_15_4_200"},
    {"num_leaves": 31, "max_depth": 5, "num_boost_round": 300,
     "learning_rate": 0.03, "min_data_in_leaf": 100, "name": "lgbm_31_5_300"},
    {"num_leaves": 7, "max_depth": 3, "num_boost_round": 150,
     "learning_rate": 0.05, "min_data_in_leaf": 100, "name": "lgbm_7_3_150"},
]


def load_oof_for_fold(k: int) -> pd.DataFrame:
    base = None
    for H in HORIZONS:
        path = os.path.join(T5B, f"loso_pred_h{H}_held{k}.parquet")
        df = pd.read_parquet(path)
        df = df.rename(columns={
            "true_label": f"true_label_{H}",
            "prob_0": f"p0_h{H}",
            "prob_1": f"p1_h{H}",
            "prob_2": f"p2_h{H}",
            "midprice_th": f"midprice_th_{H}",
        })
        df = df.drop(columns=["pred_label"])
        if base is None:
            base = df
        else:
            df = df.drop(columns=["midprice_t"])
            base = base.merge(df, on=KEY_COLS, how="inner")
    return base


def fit_lgbm(X_tr, y_tr, X_va, y_va, cfg) -> tuple:
    train_set = lgb.Dataset(X_tr, label=y_tr)
    val_set = lgb.Dataset(X_va, label=y_va, reference=train_set)
    params = {
        "objective": "multiclass", "num_class": 3,
        "metric": "multi_logloss",
        "num_leaves": cfg["num_leaves"], "max_depth": cfg["max_depth"],
        "learning_rate": cfg["learning_rate"],
        "min_data_in_leaf": cfg["min_data_in_leaf"],
        "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 5,
        "verbose": -1, "num_threads": 4,
    }
    booster = lgb.train(
        params, train_set, num_boost_round=cfg["num_boost_round"],
        valid_sets=[val_set], valid_names=["val"],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    p_va = booster.predict(X_va, num_iteration=booster.best_iteration)
    return booster, p_va


def main():
    t0 = time.time()
    print("=== Loading OOF for 5 folds ===", flush=True)
    fold_dfs = {}
    for k in range(5):
        fold_dfs[k] = load_oof_for_fold(k)

    full_df = pd.concat([fold_dfs[k] for k in range(5)], ignore_index=True)
    feat_cols = [f"p{c}_h{H}" for H in HORIZONS for c in [0, 1, 2]]
    X_all = full_df[feat_cols].to_numpy(np.float32)
    y_all = full_df[f"true_label_{TARGET_HORIZON}"].to_numpy(np.int64)
    sym_all = full_df["sym"].to_numpy(np.int64)
    print(f"X shape: {X_all.shape}, target dist: "
          f"{np.bincount(y_all).tolist()}", flush=True)

    print("\n=== Per-fold LightGBM meta training ===", flush=True)
    meta_oof_rows = []
    fold_summaries = []
    boosters_by_fold: Dict[int, Any] = {}
    for k in range(5):
        train_mask = sym_all != k
        val_mask = sym_all == k
        X_tr, y_tr = X_all[train_mask], y_all[train_mask]
        X_va, y_va = X_all[val_mask], y_all[val_mask]
        df_va = full_df[val_mask].reset_index(drop=True)

        results = []
        for cfg in LGBM_CONFIGS:
            booster, p_va = fit_lgbm(X_tr, y_tr, X_va, y_va, cfg)
            ll = log_loss(y_va, p_va, labels=[0, 1, 2])
            results.append({"name": cfg["name"], "cfg": cfg,
                            "logloss": ll, "booster": booster, "probs": p_va,
                            "best_iter": int(booster.best_iteration)})

        results.sort(key=lambda r: r["logloss"])
        best = results[0]
        print(f"\n  fold{k} (val sym={k}): n_tr={len(y_tr):,} n_va={len(y_va):,}",
              flush=True)
        for r in results:
            print(f"    {r['name']:<18s} val_logloss={r['logloss']:.4f}  "
                  f"best_iter={r['best_iter']}", flush=True)
        print(f"  -> best: {best['name']} logloss={best['logloss']:.4f}",
              flush=True)

        df_va["meta_p0"] = best["probs"][:, 0]
        df_va["meta_p1"] = best["probs"][:, 1]
        df_va["meta_p2"] = best["probs"][:, 2]
        df_va["meta_pred"] = best["probs"].argmax(axis=1).astype(np.int64)
        df_va["meta_name"] = best["name"]

        keep = KEY_COLS + [
            f"true_label_{TARGET_HORIZON}", "midprice_t",
            f"midprice_th_{TARGET_HORIZON}",
            "meta_p0", "meta_p1", "meta_p2", "meta_pred", "meta_name",
        ]
        meta_oof_rows.append(df_va[keep])
        boosters_by_fold[k] = best["booster"]
        fold_summaries.append({
            "fold": k, "best_name": best["name"],
            "best_logloss": float(best["logloss"]),
            "best_iter": int(best["best_iter"]),
            "n_train": int(len(y_tr)), "n_val": int(len(y_va)),
            "all_results": [
                {"name": r["name"], "logloss": float(r["logloss"]),
                 "best_iter": int(r["best_iter"])}
                for r in results
            ],
        })

        # Save best booster
        booster_path = os.path.join(HERE, f"meta_lgbm_held{k}.txt")
        best["booster"].save_model(booster_path,
                                    num_iteration=best["booster"].best_iteration)
        print(f"  -> saved {booster_path}", flush=True)

    meta_oof = pd.concat(meta_oof_rows, ignore_index=True)
    out_path = os.path.join(HERE, "meta_oof_h10_lgbm.parquet")
    meta_oof.to_parquet(out_path, index=False)
    print(f"\nmeta OOF saved -> {out_path} ({len(meta_oof):,} rows)", flush=True)

    # Raw argmax on meta OOF
    print("\n=== Meta-LGBM raw-argmax baseline (no threshold) per fold ===",
          flush=True)
    sum_argmax = 0.0
    for k in range(5):
        sub = meta_oof[meta_oof.sym == k]
        probs_va = sub[["meta_p0", "meta_p1", "meta_p2"]].to_numpy(np.float32)
        m = _per_horizon_metrics(
            probs_va.argmax(1).astype(np.int64),
            sub[f"true_label_{TARGET_HORIZON}"].to_numpy(np.int64),
            sub["midprice_t"].to_numpy(np.float32),
            sub[f"midprice_th_{TARGET_HORIZON}"].to_numpy(np.float32),
            fee_rate=0.0001,
        )
        sum_argmax += m["cum_pnl"]
        print(f"  fold{k}: cum_pnl={m['cum_pnl']:+.4f}  "
              f"n_active={int(m['n_predictions_active'])}  "
              f"acc={m['accuracy']:.4f}", flush=True)
    print(f"  SUM argmax: {sum_argmax:+.4f}", flush=True)

    summary = {
        "task": "T12 LightGBM meta-learner for multi-horizon stacking",
        "target_horizon": TARGET_HORIZON,
        "feat_cols": feat_cols,
        "fold_summaries": fold_summaries,
        "raw_argmax_sum": float(sum_argmax),
        "elapsed_sec": time.time() - t0,
    }
    with open(os.path.join(HERE, "fold_summaries_lgbm.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nfold_summaries_lgbm.json saved", flush=True)
    print(f"\nelapsed: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
