"""T66 Step 1: Adversarial validation.

Train a binary LightGBM classifier on train+val+test rows of schemeN cache to
predict is_test (date >= 96). Measure AUC on a held-out random split.

If AUC > 0.55, distribution shift is non-trivial. Beyond that, fit on all rows
and emit P(is_test|x) for every train row for use as a sample-weight prior
in downstream models (T66 Step 3).

Outputs:
  adversarial_auc.json  — AUC + distribution
  adv_prob_train.npy    — P(is_test|x) per train row
  adv_prob_val.npy      — P(is_test|x) per val row
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

import lightgbm as lgb  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "experiments", "T59_fullsym_train", "cache")
T59_DIR = os.path.join(ROOT, "experiments", "T59_fullsym_train")


def progress(step, **extra):
    p = os.path.join(HERE, "worker-progress.json")
    payload = {
        "status": "running",
        "step": step,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **extra,
    }
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


def load_split(split):
    p = os.path.join(CACHE_DIR, f"schemeN_{split}.npz")
    print(f"  loading {p} ...", flush=True)
    d = np.load(p)
    return {k: d[k] for k in d.files}


def get_slicer():
    """Reproduce T59's 340-d slicer (drop fail extras + drop tail 3)."""
    base_dim = 226
    extras_path = os.path.join(CACHE_DIR, "schemeN_extra_feat_names.txt")
    with open(extras_path) as f:
        extra_names = [line.strip() for line in f]
    n_extras = len(extra_names)
    total_dim = base_dim + n_extras  # 353
    report_path = os.path.join(T59_DIR, "sym_invariance_report.json")
    with open(report_path) as f:
        report = json.load(f)
    fail_names = report["summary"]["fail_names"]
    fail_local = [extra_names.index(n) for n in fail_names]
    fail_global = [base_dim + i for i in fail_local]
    drop = set(fail_global)
    for i in range(3):
        drop.add(base_dim - 1 - i)
    keep_idx = np.array(
        [i for i in range(total_dim) if i not in drop],
        dtype=np.int64,
    )
    return keep_idx


def main():
    progress("loading_caches")
    t0 = time.time()
    train = load_split("train")
    val = load_split("val")
    test = load_split("test")
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    keep_idx = get_slicer()
    print(f"  feat_dim={len(keep_idx)} (340-d sym-invariant subset)", flush=True)

    # Concat train+val+test
    X_all = np.concatenate(
        [train["X"][:, keep_idx], val["X"][:, keep_idx], test["X"][:, keep_idx]],
        axis=0,
    ).astype(np.float32, copy=False)
    n_train, n_val, n_test = len(train["X"]), len(val["X"]), len(test["X"])
    is_test = np.concatenate(
        [
            np.zeros(n_train, dtype=np.int8),
            np.zeros(n_val, dtype=np.int8),
            np.ones(n_test, dtype=np.int8),
        ]
    )
    date_all = np.concatenate(
        [train["date"], val["date"], test["date"]]
    )
    sym_all = np.concatenate(
        [train["sym"], val["sym"], test["sym"]]
    )
    print(f"  X_all shape: {X_all.shape}, is_test mean: {is_test.mean():.4f}", flush=True)

    # Random 80/20 shuffle split (stratified by is_test)
    rng = np.random.default_rng(0)
    idx_pos = np.where(is_test == 1)[0]
    idx_neg = np.where(is_test == 0)[0]
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)
    cut_pos = int(0.8 * len(idx_pos))
    cut_neg = int(0.8 * len(idx_neg))
    idx_train = np.concatenate([idx_pos[:cut_pos], idx_neg[:cut_neg]])
    idx_eval = np.concatenate([idx_pos[cut_pos:], idx_neg[cut_neg:]])
    rng.shuffle(idx_train)
    rng.shuffle(idx_eval)
    print(f"  cv split: train={len(idx_train):,}  eval={len(idx_eval):,}", flush=True)

    progress("training_adv_classifier_cv")
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "lambda_l2": 1.0,
        "num_threads": 18,
        "seed": 42,
        "verbose": -1,
        "device": "gpu",
        "gpu_use_dp": False,
    }

    dtrain = lgb.Dataset(X_all[idx_train], label=is_test[idx_train], free_raw_data=False)
    deval = lgb.Dataset(X_all[idx_eval], label=is_test[idx_eval], reference=dtrain, free_raw_data=False)

    t1 = time.time()
    booster = lgb.train(
        params, dtrain,
        num_boost_round=400,
        valid_sets=[dtrain, deval],
        valid_names=["train", "eval"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )
    train_time = time.time() - t1
    print(f"  trained in {train_time:.1f}s, best_iter={booster.best_iteration}", flush=True)

    # Eval AUC
    eval_prob = booster.predict(X_all[idx_eval])
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(is_test[idx_eval], eval_prob))
    print(f"\n  HELD-OUT AUC: {auc:.4f}", flush=True)
    if auc > 0.7:
        verdict = "HIGH SHIFT (AUC>0.7) — adversarial reweighting recommended"
    elif auc > 0.55:
        verdict = "MILD SHIFT (0.55<AUC<0.7) — adversarial weight may help"
    else:
        verdict = f"NO SHIFT (AUC≈{auc:.2f}) — distributions are similar"
    print(f"  verdict: {verdict}", flush=True)

    # Per-date AUC analysis (which dates are most "test-like"?)
    eval_idx_by_date = {}
    for d in np.unique(date_all[idx_eval]):
        m = date_all[idx_eval] == d
        if m.sum() < 200:
            continue
        labels = is_test[idx_eval][m]
        if len(np.unique(labels)) < 2:
            avg_p = float(eval_prob[m].mean())
            eval_idx_by_date[int(d)] = {"avg_p_test": avg_p, "n": int(m.sum()), "auc": None}
            continue
        try:
            sub_auc = float(roc_auc_score(labels, eval_prob[m]))
        except Exception:
            sub_auc = None
        eval_idx_by_date[int(d)] = {
            "avg_p_test": float(eval_prob[m].mean()),
            "n": int(m.sum()),
            "auc": sub_auc,
        }

    # Top features
    importance = booster.feature_importance(importance_type="gain")
    feat_names_path = os.path.join(CACHE_DIR, "schemeN_feat_names.txt")
    with open(feat_names_path) as f:
        all_feat_names = [line.strip() for line in f]
    feat_names = [all_feat_names[i] for i in keep_idx]
    top_idx = np.argsort(-importance)[:20]
    top_feats = [{"feat": feat_names[i], "gain": float(importance[i])} for i in top_idx]
    print(f"  top-20 distinguishing features:", flush=True)
    for tf in top_feats[:10]:
        print(f"    {tf['feat']:40s}  gain={tf['gain']:.0f}", flush=True)

    # Train final model on ALL rows (so we can score every train row)
    progress("training_adv_classifier_full")
    t2 = time.time()
    dfull = lgb.Dataset(X_all, label=is_test, free_raw_data=False)
    final_iters = max(int(booster.best_iteration), 50)
    booster_full = lgb.train(
        params, dfull,
        num_boost_round=final_iters,
        callbacks=[lgb.log_evaluation(period=50)],
    )
    print(f"  full booster trained in {time.time()-t2:.1f}s ({final_iters} iters)", flush=True)

    # Score train + val
    p_train = booster_full.predict(X_all[:n_train]).astype(np.float32)
    p_val = booster_full.predict(X_all[n_train : n_train + n_val]).astype(np.float32)
    p_test = booster_full.predict(X_all[n_train + n_val :]).astype(np.float32)

    print(f"\n  P(is_test|x) stats:")
    print(f"    train: mean={p_train.mean():.4f}  median={np.median(p_train):.4f}  q90={np.quantile(p_train,0.9):.4f}  q99={np.quantile(p_train,0.99):.4f}", flush=True)
    print(f"    val:   mean={p_val.mean():.4f}  median={np.median(p_val):.4f}  q90={np.quantile(p_val,0.9):.4f}", flush=True)
    print(f"    test:  mean={p_test.mean():.4f}  median={np.median(p_test):.4f}", flush=True)

    # Save adv probs
    np.save(os.path.join(HERE, "adv_prob_train.npy"), p_train)
    np.save(os.path.join(HERE, "adv_prob_val.npy"), p_val)
    np.save(os.path.join(HERE, "adv_prob_test.npy"), p_test)

    # Save model
    booster_full.save_model(os.path.join(HERE, "adv_classifier.txt"))

    out = {
        "task": "T66 Step 1 - adversarial validation",
        "feat_dim": int(len(keep_idx)),
        "n_train": int(n_train),
        "n_val": int(n_val),
        "n_test": int(n_test),
        "is_test_pos_rate": float(is_test.mean()),
        "cv_split": {"train": int(len(idx_train)), "eval": int(len(idx_eval))},
        "auc_holdout": auc,
        "verdict": verdict,
        "best_iteration": int(booster.best_iteration),
        "p_train_stats": {
            "mean": float(p_train.mean()),
            "median": float(np.median(p_train)),
            "q90": float(np.quantile(p_train, 0.9)),
            "q99": float(np.quantile(p_train, 0.99)),
            "min": float(p_train.min()),
            "max": float(p_train.max()),
        },
        "p_val_stats": {
            "mean": float(p_val.mean()),
            "median": float(np.median(p_val)),
        },
        "p_test_stats": {
            "mean": float(p_test.mean()),
            "median": float(np.median(p_test)),
        },
        "per_date_eval": eval_idx_by_date,
        "top_features": top_feats,
    }
    with open(os.path.join(HERE, "adversarial_auc.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote -> adversarial_auc.json", flush=True)
    progress("step1_done", auc=auc)


if __name__ == "__main__":
    main()
