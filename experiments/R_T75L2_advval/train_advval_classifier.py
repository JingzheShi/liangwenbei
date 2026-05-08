"""Adversarial validation classifier for T75 LGB L2.

Train binary classifier: label=0 if date<=79 (past=train cache), label=1 if
date>=80 (future=val+test caches). Use 5-fold date-block CV on past samples to
get OOF proba; future samples are fully used for training in every fold.

Drop date column (and sym) to avoid trivial discrimination.

Save:
  - oof_proba_past.npy  (n_past,)  raw OOF proba [0,1]
  - sw_advval.npy       (n_past,)  isotonic-calibrated, mean-1 normalized weight
  - cls_summary.json    AUC, fold AUCs, top-feature importance
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression

HERE = "/root/lwb_work_t75_advval"
CACHE_DIR = "/root/lwb_remote_pkg/cache"
N_FOLDS = 5

# Same drop set as T75 (10 + 1 fail features)
T59_FAIL_NAMES = [
    "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
    "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
    "qrank_W100_cumspread",
    "kyle_lam_W50", "kyle_lam_W100",
    "roll_eff_spr_ratio_W100",
]
STAGE5_FAIL_NAMES = ["liq_asym_top5_W5"]
DROP_NAMES = T59_FAIL_NAMES + STAGE5_FAIL_NAMES


def progress(step, **extra):
    p = os.path.join(HERE, "advval_progress.json")
    payload = {"status": "running", "step": step,
               "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               **extra}
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


def load_split(split):
    p = os.path.join(CACHE_DIR, f"schemeP_{split}.npz")
    print(f"  loading {p} ...", flush=True)
    d = np.load(p)
    return {k: d[k] for k in d.files}


def main():
    os.makedirs(HERE, exist_ok=True)
    progress("loading_caches")

    t0 = time.time()
    train_full = load_split("train")
    val_full = load_split("val")
    test_full = load_split("test")
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    feat_names_path = os.path.join(CACHE_DIR, "schemeP_feat_names.txt")
    with open(feat_names_path) as f:
        all_feat_names = [line.strip() for line in f]
    total_dim = train_full["X"].shape[1]
    assert len(all_feat_names) == total_dim
    name_to_idx = {n: i for i, n in enumerate(all_feat_names)}
    drop_idx = set()
    for n in DROP_NAMES:
        if n in name_to_idx:
            drop_idx.add(name_to_idx[n])
    keep_idx = np.array(
        [i for i in range(total_dim) if i not in drop_idx], dtype=np.int64,
    )
    feat_dim = len(keep_idx)
    print(f"  feat_dim={feat_dim} (dropped {len(drop_idx)})", flush=True)
    feat_names = [all_feat_names[i] for i in keep_idx]
    forbidden = {"date", "sym", "time"}
    leak = forbidden & set(feat_names)
    assert not leak, f"forbidden in features: {leak}"

    # ---- assemble past + future ----
    X_past = train_full["X"][:, keep_idx].astype(np.float32, copy=False)
    date_past = train_full["date"].astype(np.int16)
    n_past = len(X_past)

    X_val = val_full["X"][:, keep_idx].astype(np.float32, copy=False)
    X_test = test_full["X"][:, keep_idx].astype(np.float32, copy=False)
    X_future = np.concatenate([X_val, X_test], axis=0)
    n_future = len(X_future)

    print(f"  n_past={n_past:,}  n_future={n_future:,}  feat_dim={feat_dim}", flush=True)

    # ---- 5-fold date-block CV on past ----
    # Past dates 0-79 (80 dates). Round-robin date assignment to folds for
    # balanced size (each fold has 16 dates, ~294k samples).
    unique_dates = sorted(np.unique(date_past).tolist())
    print(f"  past unique_dates={len(unique_dates)}: {unique_dates[:5]}..{unique_dates[-5:]}", flush=True)
    rng = np.random.default_rng(0)
    shuffled_dates = unique_dates.copy()
    rng.shuffle(shuffled_dates)
    fold_to_dates = {f: [] for f in range(N_FOLDS)}
    for i, d in enumerate(shuffled_dates):
        fold_to_dates[i % N_FOLDS].append(d)
    for f in range(N_FOLDS):
        print(f"    fold {f}: {len(fold_to_dates[f])} dates", flush=True)

    oof_proba = np.zeros(n_past, dtype=np.float32)
    fold_aucs = []
    feat_imp_sum = np.zeros(feat_dim, dtype=np.float64)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l2": 1.0,
        "num_threads": 18,
        "verbose": -1,
        "seed": 42,
        "device": "gpu",
        "gpu_use_dp": False,
    }

    for fold in range(N_FOLDS):
        progress(f"cv_fold_{fold}")
        held_dates = set(fold_to_dates[fold])
        m_held = np.array([int(d) in held_dates for d in date_past], dtype=bool)
        m_train_past = ~m_held
        # Train: past minus held + ALL future
        X_clf_train = np.concatenate(
            [X_past[m_train_past], X_future], axis=0)
        y_clf_train = np.concatenate(
            [np.zeros(int(m_train_past.sum()), dtype=np.float32),
             np.ones(n_future, dtype=np.float32)],
            axis=0)
        # Subsample held set as eval (for early stopping). Use random 20% of held.
        held_idx = np.where(m_held)[0]
        eval_idx = rng.choice(held_idx, size=min(50_000, len(held_idx)), replace=False)
        # eval = held past (label=0) + sample of future (label=1)
        future_eval_idx = rng.choice(n_future, size=min(50_000, n_future), replace=False)
        X_eval = np.concatenate(
            [X_past[eval_idx], X_future[future_eval_idx]], axis=0)
        y_eval = np.concatenate(
            [np.zeros(len(eval_idx), dtype=np.float32),
             np.ones(len(future_eval_idx), dtype=np.float32)], axis=0)
        print(f"  fold {fold}: train={len(X_clf_train):,} eval={len(X_eval):,}",
              flush=True)

        dtrain = lgb.Dataset(X_clf_train, label=y_clf_train,
                             feature_name=feat_names, free_raw_data=False)
        deval = lgb.Dataset(X_eval, label=y_eval,
                            feature_name=feat_names, reference=dtrain,
                            free_raw_data=False)
        t_s = time.time()
        booster = lgb.train(
            params, dtrain,
            num_boost_round=400,
            valid_sets=[deval],
            valid_names=["eval"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(period=50),
            ],
        )
        print(f"    fold {fold} trained in {time.time()-t_s:.1f}s, best={booster.best_iteration}", flush=True)

        # Predict on FULL held past (not just eval subset)
        held_pred = booster.predict(X_past[m_held], num_iteration=booster.best_iteration)
        oof_proba[m_held] = held_pred.astype(np.float32)
        # AUC on held (need positives) - use eval set
        eval_pred = booster.predict(X_eval, num_iteration=booster.best_iteration)
        try:
            auc = roc_auc_score(y_eval, eval_pred)
        except Exception as e:
            auc = float("nan")
        fold_aucs.append(float(auc))
        print(f"    fold {fold} eval_auc={auc:.4f}", flush=True)

        # accumulate feature importance
        imp = booster.feature_importance(importance_type="gain")
        feat_imp_sum += imp.astype(np.float64)

        del booster, dtrain, deval, X_clf_train, y_clf_train, X_eval, y_eval

    # ---- combined AUC: predict held proba on full past, label past=0; combine with future label=1 holding constant ----
    # Approximate global AUC: oof_proba on past vs proba_on_future from final-fold model. We instead skip this, fold AUC mean is enough.
    print(f"\nFold AUCs: {fold_aucs}  mean={np.mean(fold_aucs):.4f}", flush=True)

    # ---- Calibrate weights via isotonic regression ----
    # Goal: turn proba into a sample weight such that high-proba (test-like) past
    # samples get more weight. Use isotonic on (proba, target) where target=1
    # for future samples (proxy). But since OOF is only past, simpler: take
    # proba_past directly, clip, and normalize to mean=1.
    p_clip = np.clip(oof_proba, 0.01, 0.99)
    # Logit-style sw: w_i = p_i / (1 - p_i); clip to [0.1, 10]
    # But this can be very heavy-tailed. Use mild scheme:
    #   w_i = p_i / (1 - p_i), then clip to [0.25, 4.0], normalize mean=1
    sw_raw = p_clip / (1.0 - p_clip)
    sw_clip = np.clip(sw_raw, 0.25, 4.0)
    sw_norm = sw_clip / sw_clip.mean()  # mean=1

    # Also compute isotonic-fit version: target=mean of OOF (since past has y=0).
    # We instead use isotonic to map proba->weight monotonically with strong upper-cap.
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.25, y_max=4.0)
    # Fit isotonic on (proba, sw_clip) then transform on proba.
    # This doesn't change much but smooths the mapping.
    sort_idx = np.argsort(p_clip)
    iso.fit(p_clip[sort_idx], sw_clip[sort_idx])
    sw_iso = iso.transform(p_clip)
    sw_iso_norm = sw_iso / sw_iso.mean()

    # Save both - prefer sw_iso_norm
    np.save(os.path.join(HERE, "oof_proba_past.npy"), oof_proba)
    np.save(os.path.join(HERE, "sw_advval.npy"), sw_iso_norm.astype(np.float32))
    np.save(os.path.join(HERE, "sw_advval_logit.npy"), sw_norm.astype(np.float32))

    # Top features
    feat_imp_avg = feat_imp_sum / N_FOLDS
    top_idx = np.argsort(feat_imp_avg)[::-1][:20]
    top_feats = [(feat_names[i], float(feat_imp_avg[i])) for i in top_idx]

    summary = {
        "task": "advval_classifier_T75",
        "n_past": int(n_past),
        "n_future": int(n_future),
        "feat_dim": int(feat_dim),
        "n_folds": N_FOLDS,
        "fold_aucs": fold_aucs,
        "mean_auc": float(np.mean(fold_aucs)),
        "oof_proba_stats": {
            "mean": float(oof_proba.mean()),
            "std": float(oof_proba.std()),
            "p10": float(np.percentile(oof_proba, 10)),
            "p50": float(np.percentile(oof_proba, 50)),
            "p90": float(np.percentile(oof_proba, 90)),
        },
        "sw_advval_stats": {
            "mean": float(sw_iso_norm.mean()),
            "std": float(sw_iso_norm.std()),
            "p10": float(np.percentile(sw_iso_norm, 10)),
            "p50": float(np.percentile(sw_iso_norm, 50)),
            "p90": float(np.percentile(sw_iso_norm, 90)),
            "min": float(sw_iso_norm.min()),
            "max": float(sw_iso_norm.max()),
        },
        "top_features_gain": top_feats,
    }
    with open(os.path.join(HERE, "advval_cls_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsummary saved", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    progress("done", mean_auc=float(np.mean(fold_aucs)))


if __name__ == "__main__":
    main()
