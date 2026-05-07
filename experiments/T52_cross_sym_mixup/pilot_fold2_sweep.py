"""T52 pilot: sweep (mixup_ratio, alpha) on fold 2 (held=2, the worst fold).

Runs 9 combos: ratio in {0.3, 0.5, 0.7} x alpha in {0.2, 0.4, 0.6}.
Each: train on {sym!=2}, evaluate held_out_test on {sym==2}, report
raw argmax cum_pnl.

Two extra control runs:
    - mixup_ratio=0, aug_a only (T44-equivalent baseline on schemeC 226-d)
    - mixup_ratio=0.5, alpha=0.4, no_aug_a (mixup-only ablation)
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

import lightgbm as lgb  # noqa: E402

T26_DIR = os.path.join(ROOT, "experiments", "T26_domain_randomization")
sys.path.insert(0, T26_DIR)
from build_aug import class_balanced_weight  # noqa: E402

sys.path.insert(0, HERE)
from build_mixup import build_train_with_mixup  # noqa: E402

from src.eval.pnl import _per_horizon_metrics, sanitize_for_json  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "experiments", "T5b_features_multihorizon", "cache")
NUM_CLASS = 3
H = 60
HELD = 2  # worst fold per T44/iter_006

GRID = []
for mr in (0.3, 0.5, 0.7):
    for al in (0.2, 0.4, 0.6):
        GRID.append({"mixup_ratio": mr, "alpha": al, "use_aug_a": True, "tag": f"mr{mr}_a{al}_auga"})
# extras: aug_a only baseline (mixup_ratio=0)
GRID.append({"mixup_ratio": 0.0, "alpha": 0.0, "use_aug_a": True, "tag": "mr0_auga_only"})
# extras: mixup-only (no aug_a) at mid setting
GRID.append({"mixup_ratio": 0.5, "alpha": 0.4, "use_aug_a": False, "tag": "mr0p5_a0p4_noaa"})


def progress(step: str, **extra) -> None:
    p = os.path.join(HERE, "worker-progress.json")
    payload = {
        "status": "running",
        "step": step,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **extra,
    }
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


def predict_proba(booster, X, batch=200_000):
    chunks = []
    for s in range(0, len(X), batch):
        chunks.append(booster.predict(X[s : s + batch]).astype(np.float32))
    return np.concatenate(chunks, axis=0)


def main():
    print(f"=== T52 PILOT FOLD-{HELD} sweep h={H}: {len(GRID)} combos ===", flush=True)
    progress("loading_caches")

    t0 = time.time()
    train_full = {k: v for k, v in np.load(os.path.join(CACHE_DIR, "schemeC_train.npz")).items()}
    val_full = {k: v for k, v in np.load(os.path.join(CACHE_DIR, "schemeC_val.npz")).items()}
    test_full = {k: v for k, v in np.load(os.path.join(CACHE_DIR, "schemeC_test.npz")).items()}
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    feat_names_path = os.path.join(CACHE_DIR, "schemeC_feat_names.txt")
    with open(feat_names_path) as f:
        feat_names = [l.strip() for l in f]
    forbidden = {"date", "sym", "time"}
    leak = forbidden & set(feat_names)
    assert not leak, f"FORBIDDEN feature leak: {leak}"
    feat_dim = len(feat_names)
    print(f"  feat_dim={feat_dim}", flush=True)

    m_tr = train_full["sym"] != HELD
    m_va = val_full["sym"] != HELD
    m_te = test_full["sym"] == HELD

    X_tr_orig = train_full["X"][m_tr].astype(np.float32, copy=False)
    y_tr_orig = train_full[f"y{H}"][m_tr].astype(np.int64)
    sym_tr_fold = train_full["sym"][m_tr]
    X_va = val_full["X"][m_va].astype(np.float32, copy=False)
    y_va = val_full[f"y{H}"][m_va].astype(np.int64)
    X_te = test_full["X"][m_te].astype(np.float32, copy=False)
    y_te = test_full[f"y{H}"][m_te].astype(np.int64)
    mp_t_te_k = test_full["mp_t"][m_te]
    mp_th_te_k = test_full[f"mp_t{H}"][m_te]

    print(f"n_train_orig={len(X_tr_orig):,} n_val={len(X_va):,} n_test={len(X_te):,}", flush=True)
    sw_va = class_balanced_weight(y_va, num_class=NUM_CLASS)

    SEED = 42
    base_params = {
        "objective": "multiclass",
        "num_class": NUM_CLASS,
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l2": 1.0,
        "num_threads": 18,
        "seed": SEED,
        "feature_fraction_seed": SEED + 1,
        "bagging_seed": SEED + 2,
        "data_random_seed": SEED + 3,
        "verbose": -1,
        "device": "gpu",
        "gpu_use_dp": False,
    }

    results = []
    for i, cfg in enumerate(GRID, 1):
        tag = cfg["tag"]
        print(f"\n--- [{i}/{len(GRID)}] tag={tag} ---", flush=True)
        progress(f"sweep_{i}/{len(GRID)}", tag=tag)

        rng = np.random.default_rng(SEED * 7919 + 1 + HELD * 31)
        t_aug = time.time()
        X_tr, y_tr, sw_tr = build_train_with_mixup(
            X_tr_orig, y_tr_orig, sym_tr_fold,
            rng=rng,
            use_aug_a=cfg["use_aug_a"],
            aug_a_ratio=1.0,
            mixup_ratio=cfg["mixup_ratio"],
            alpha=cfg["alpha"],
            num_class=NUM_CLASS,
        )
        print(f"  n_used={len(X_tr):,} aug_time={time.time()-t_aug:.1f}s", flush=True)

        dtrain = lgb.Dataset(
            X_tr, label=y_tr, weight=sw_tr,
            feature_name=feat_names, free_raw_data=False,
        )
        dval = lgb.Dataset(
            X_va, label=y_va, weight=sw_va,
            feature_name=feat_names, reference=dtrain, free_raw_data=False,
        )
        t_start = time.time()
        booster = lgb.train(
            base_params, dtrain,
            num_boost_round=600,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=40, verbose=False),
                lgb.log_evaluation(period=400),
            ],
        )
        train_time = time.time() - t_start
        print(f"  trained in {train_time:.1f}s, best_iter={booster.best_iteration}",
              flush=True)

        te_prob = predict_proba(booster, X_te)
        te_pred = te_prob.argmax(axis=1).astype(np.int8)
        m = _per_horizon_metrics(te_pred, y_te, mp_t_te_k, mp_th_te_k, fee_rate=0.0001)
        cum_pnl = float(m["cum_pnl"])
        acc = float(m["accuracy"])
        f05 = float(m["f0_5_macro"])
        n_active = int(m["n_predictions_active"])
        print(f"  HELD={HELD} cum_pnl={cum_pnl:+.4f} acc={acc:.4f} f0.5={f05:.4f} "
              f"active={n_active}", flush=True)

        rec = {
            **cfg,
            "best_iter": int(booster.best_iteration),
            "train_time": float(train_time),
            "n_train_used": int(len(X_tr)),
            "test_cum_pnl": cum_pnl,
            "test_acc": acc,
            "test_f0_5_macro": f05,
            "test_n_active": n_active,
        }
        results.append(rec)
        del X_tr, y_tr, sw_tr, dtrain, dval, booster

        # write incrementally
        with open(os.path.join(HERE, "pilot_fold2_results.json"), "w") as f:
            json.dump(sanitize_for_json({"horizon": H, "held": HELD, "results": results}),
                      f, indent=2)

    # Print sorted summary
    print(f"\n{'='*78}\n=== PILOT FOLD-{HELD} SUMMARY (sorted by cum_pnl desc) ===\n{'='*78}",
          flush=True)
    df = pd.DataFrame(results).sort_values("test_cum_pnl", ascending=False)
    print(df[["tag", "mixup_ratio", "alpha", "use_aug_a", "test_cum_pnl",
              "test_acc", "test_f0_5_macro", "test_n_active", "best_iter"]].to_string(index=False))

    progress("pilot_done", n_combos=len(results))
    print(f"\n[done] -> pilot_fold2_results.json", flush=True)


if __name__ == "__main__":
    main()
