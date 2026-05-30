"""Compute LGB gain-based feature importance on schemeP train split (359-d L8 input).

Train a single LGB regression on date 0-79 (train only) with H=60 target,
save gain importance to feat_imp_gain_359d.npy.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np
import lightgbm as lgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data_loader import get_keep_idx, get_feat_names

CACHE = Path('/root/projects/liangwenbei_workdir/ablation_runs/cache_log1p')
H = 60


def class_balanced_weight(y, num_class=3):
    counts = np.bincount(y.astype(np.int64), minlength=num_class).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    cw = len(y) / (num_class * counts)
    return cw[y.astype(np.int64)].astype(np.float32)


def main():
    t0 = time.time()
    keep_idx = get_keep_idx("L8")
    feat_names = get_feat_names(keep_idx)
    n_feat = len(keep_idx)
    print(f"L8 keep_idx: n_feat={n_feat}", flush=True)

    print("loading schemeP_train.npz ...", flush=True)
    d = np.load(CACHE / 'schemeP_train.npz')
    X = d['X'][:, keep_idx].astype(np.float32, copy=False)
    mp_t = d['mp_t'].astype(np.float64)
    mp_th = d[f'mp_t{H}'].astype(np.float64)
    y_cls = d[f'y{H}'].astype(np.int64)
    d.close()
    y_regr = ((mp_th - mp_t) / (mp_t + 1.0)).astype(np.float32)
    sw = class_balanced_weight(y_cls, num_class=3)
    print(f"  X={X.shape}  y_regr mean={y_regr.mean():.4e} std={y_regr.std():.4e}", flush=True)

    # HP config 0 (same as final submission base config)
    params = {
        "objective": "regression_l2",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l2": 1.0,
        "min_data_in_leaf": 100,
        "num_threads": 18,
        "device": "gpu",
        "gpu_use_dp": False,
        "seed": 42,
        "verbose": -1,
    }
    dtrain = lgb.Dataset(X, label=y_regr, weight=sw,
                         feature_name=feat_names, free_raw_data=True)
    del X, y_regr, y_cls, sw

    print("training LGB (GPU, 330 rounds) ...", flush=True)
    t_tr = time.time()
    booster = lgb.train(
        params, dtrain,
        num_boost_round=330,
        callbacks=[lgb.log_evaluation(period=50)],
    )
    print(f"LGB done in {time.time()-t_tr:.1f}s", flush=True)

    gain = booster.feature_importance(importance_type='gain')
    split = booster.feature_importance(importance_type='split')
    assert gain.shape == (n_feat,)
    np.save(HERE / 'feat_imp_gain_359d.npy', gain)
    np.save(HERE / 'feat_imp_split_359d.npy', split)
    print(f"saved feat_imp_gain_359d.npy  (n={n_feat})", flush=True)

    # Save feat-name -> importance mapping for inspection
    with open(HERE / 'feat_imp_359d.json', 'w') as f:
        json.dump({
            "feat_names": feat_names,
            "gain": gain.tolist(),
            "split": split.tolist(),
        }, f, indent=2)

    # Sanity print top-15 and bottom-15
    order_desc = np.argsort(gain)[::-1]
    order_asc = np.argsort(gain)
    print("\n=== Top 15 gain ===", flush=True)
    for r, i in enumerate(order_desc[:15]):
        print(f"  {r+1:2d}. {feat_names[i]:30s} gain={gain[i]:.3e}  split={split[i]}", flush=True)
    print("\n=== Bottom 15 gain ===", flush=True)
    for r, i in enumerate(order_asc[:15]):
        print(f"  {r+1:2d}. {feat_names[i]:30s} gain={gain[i]:.3e}  split={split[i]}", flush=True)

    # Construct keep_idx_259d (top 259 by gain)
    drop_idx = np.sort(order_asc[:100])
    keep_259 = np.sort(order_asc[100:])
    assert len(keep_259) == 259
    np.save(HERE / 'keep_idx_259d.npy', keep_259)
    np.save(HERE / 'drop_idx_100.npy', drop_idx)
    print(f"\nsaved keep_idx_259d.npy ({len(keep_259)})  drop_idx_100.npy ({len(drop_idx)})", flush=True)
    print(f"\nDropped (bottom 100) feature names:", flush=True)
    for i in drop_idx:
        print(f"  {feat_names[i]:30s} gain={gain[i]:.3e}", flush=True)
    print(f"\nTotal: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
