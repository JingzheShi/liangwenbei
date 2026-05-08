"""Quick LGB screening to pick top-K pairwise imbalances.

Train one LGB on (schemeP-359 + 190 pairs + 7 globals) with a single seed
for ~150 rounds. Rank pairwise features by gain importance and select top K.

Saves cache/top_pairs_idx.npy (K-length int64 indices into 190 pairs)
and cache/top_pairs_names.txt for the train.py to consume.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

T26_DIR = os.path.join(ROOT, "experiments", "T26_domain_randomization")
sys.path.insert(0, T26_DIR)
from build_aug import class_balanced_weight  # noqa: E402

T68_CACHE = os.path.join(ROOT, "experiments", "T68_stage5_features", "cache")
HERE_CACHE = os.path.join(HERE, "cache")

# Same DROP list as R3
T59_FAIL = ["dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
            "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
            "qrank_W100_cumspread", "kyle_lam_W50", "kyle_lam_W100",
            "roll_eff_spr_ratio_W100"]
STAGE5_FAIL = ["liq_asym_top5_W5"]
DROP_NAMES = T59_FAIL + STAGE5_FAIL

NUM_CLASS = 3
TOPK = 30


def regr_target(mp_t, mp_th):
    return ((mp_th.astype(np.float64) - mp_t.astype(np.float64))
            / (mp_t.astype(np.float64) + 1.0)).astype(np.float32)


def main():
    print("=== screening top-K pairwise features (single LGB, seed=0) ===", flush=True)

    train = dict(np.load(os.path.join(T68_CACHE, "schemeP_train.npz")))
    with open(os.path.join(T68_CACHE, "schemeP_feat_names.txt")) as f:
        all_names = [l.strip() for l in f if l.strip()]
    name_to_idx = {n: i for i, n in enumerate(all_names)}
    drop_idx = {name_to_idx[n] for n in DROP_NAMES if n in name_to_idx}
    keep_idx = np.array([i for i in range(len(all_names)) if i not in drop_idx],
                        dtype=np.int64)
    base_names = [all_names[i] for i in keep_idx]
    print(f"  schemeP kept {len(keep_idx)} of {len(all_names)} feats", flush=True)

    pair_train = np.load(os.path.join(HERE_CACHE, "pairwise_train.npy"))
    with open(os.path.join(HERE_CACHE, "pairwise_feat_names.txt")) as f:
        pair_names = [l.strip() for l in f if l.strip()]
    print(f"  pairwise: {pair_train.shape}", flush=True)

    glob = np.load(os.path.join(HERE_CACHE, "globals_consts.npz"))
    glob_vals = glob["values"].astype(np.float32)
    glob_names = list(glob["names"])
    print(f"  globals: {len(glob_names)} consts", flush=True)

    date_tr = train["date"]
    m_va = date_tr >= 76
    m_t = ~m_va
    n_tr = int(m_t.sum())
    n_va = int(m_va.sum())

    X_base_tr = train["X"][m_t][:, keep_idx].astype(np.float32, copy=False)
    X_base_va = train["X"][m_va][:, keep_idx].astype(np.float32, copy=False)
    pair_tr = pair_train[m_t]
    pair_va = pair_train[m_va]

    glob_tr = np.broadcast_to(glob_vals, (n_tr, len(glob_vals))).copy()
    glob_va = np.broadcast_to(glob_vals, (n_va, len(glob_vals))).copy()

    X_tr = np.concatenate([X_base_tr, pair_tr, glob_tr], axis=1)
    X_va = np.concatenate([X_base_va, pair_va, glob_va], axis=1)
    feat_names = base_names + pair_names + glob_names
    assert X_tr.shape[1] == len(feat_names)
    print(f"  X_tr {X_tr.shape}  X_va {X_va.shape}", flush=True)

    y_cls_tr = train["y60"][m_t].astype(np.int64)
    y_regr_tr = regr_target(train["mp_t"][m_t], train["mp_t60"][m_t])
    y_cls_va = train["y60"][m_va].astype(np.int64)
    y_regr_va = regr_target(train["mp_t"][m_va], train["mp_t60"][m_va])

    sw_tr = class_balanced_weight(y_cls_tr, num_class=NUM_CLASS)
    sw_va = class_balanced_weight(y_cls_va, num_class=NUM_CLASS)

    dtrain = lgb.Dataset(X_tr, label=y_regr_tr, weight=sw_tr,
                         feature_name=feat_names, free_raw_data=False)
    dval = lgb.Dataset(X_va, label=y_regr_va, weight=sw_va,
                       feature_name=feat_names, reference=dtrain, free_raw_data=False)

    params = {
        "objective": "huber", "alpha": 1e-3, "metric": "l1",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "min_data_in_leaf": 100,
        "feature_fraction": 1.0,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l2": 1.0,
        "num_threads": 18,
        "seed": 0,
        "feature_fraction_seed": 1,
        "bagging_seed": 2,
        "data_random_seed": 3,
        "device": "gpu", "gpu_use_dp": False,
        "verbose": -1,
    }

    t0 = time.time()
    booster = lgb.train(params=params, train_set=dtrain,
                        num_boost_round=500,
                        valid_sets=[dval], valid_names=["val"],
                        callbacks=[lgb.early_stopping(40, verbose=False),
                                   lgb.log_evaluation(period=50)])
    print(f"  trained in {time.time()-t0:.1f}s, best={booster.best_iteration}", flush=True)

    importance = booster.feature_importance(importance_type="gain")
    pair_offset = len(base_names)
    n_pairs = len(pair_names)
    pair_imp = importance[pair_offset:pair_offset + n_pairs]
    glob_imp = importance[pair_offset + n_pairs:]

    print(f"\n  globals gain importance:")
    for n, v in sorted(zip(glob_names, glob_imp), key=lambda x: -x[1]):
        print(f"    {n}: {v:.1f}")

    # Top-K pairs by gain
    order = np.argsort(-pair_imp)
    top_idx = order[:TOPK]
    print(f"\n  top-{TOPK} pairs by gain:")
    top_pairs = []
    for i, idx in enumerate(top_idx):
        info = {"rank": i + 1, "pair_idx": int(idx), "name": pair_names[idx],
                "gain": float(pair_imp[idx])}
        top_pairs.append(info)
        print(f"    [{i+1:2d}] {pair_names[idx]:32s} gain={pair_imp[idx]:.1f}")

    np.save(os.path.join(HERE_CACHE, "top_pairs_idx.npy"),
            top_idx.astype(np.int64))
    with open(os.path.join(HERE_CACHE, "top_pairs_names.txt"), "w") as f:
        f.write("\n".join([pair_names[i] for i in top_idx]) + "\n")
    with open(os.path.join(HERE_CACHE, "screening_results.json"), "w") as f:
        json.dump({"top_pairs": top_pairs,
                   "globals_gain": dict(zip([str(n) for n in glob_names],
                                            [float(v) for v in glob_imp])),
                   "best_iter": int(booster.best_iteration)},
                  f, indent=2)
    print(f"\n  saved top_pairs_idx.npy + top_pairs_names.txt + screening_results.json")


if __name__ == "__main__":
    main()
