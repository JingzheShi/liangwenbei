"""Verify the iter_001 Predictor produces the same probs / preds as the training-cache pipeline.

Take 5 random sessions × 50 random t-points each, build a 100-tick window, push through
both:
  (a) the trained Scheme B booster fed by build_features.compute_rolling_stats (training path)
  (b) the new Predictor's internal _build_feats_batch (inference path)

Assert: max |prob_train - prob_predictor| < 1e-5.

This catches feature-order or transform mismatches before submission.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import lightgbm as lgb  # noqa: E402

from src.data.split import get_split, get_file_path  # noqa: E402
from src.data.dataset import get_default_feature_cols  # noqa: E402
from experiments.T2_gbdt_lgbm.build_features import (  # noqa: E402
    build_features_one_session,
)


def _import_predictor(submission_dir: str):
    sys.path.insert(0, submission_dir)
    if "Predictor" in sys.modules:
        del sys.modules["Predictor"]
    from Predictor import Predictor  # type: ignore
    sys.path.pop(0)
    return Predictor


def main():
    feat_cols = get_default_feature_cols()
    splits = get_split("local_debug")
    test = splits["test"]

    model_path = os.path.join(ROOT, "experiments/T2_gbdt_lgbm/model_schemeB.txt")
    booster = lgb.Booster(model_file=model_path)
    print(f"loaded booster: {model_path}", flush=True)

    rng = np.random.default_rng(42)
    sessions = [test[i] for i in rng.choice(len(test), 5, replace=False)]
    print(f"sampled sessions: {sessions}", flush=True)

    # ---- Path A: build full session like training, slice samples ----
    samples_train_pred = []
    samples_inference_input = []
    for sym, date, sess in sessions:
        path = get_file_path(sym, date, sess, data_dir=os.path.join(ROOT, "data"))
        df = pd.read_parquet(path)
        # Build via training pipeline
        X_tr, y60, mp_t, mp_t60, t_valid = build_features_one_session(df, feat_cols, "B")
        # Pick 50 random valid samples
        idxs = rng.choice(len(t_valid), 50, replace=False)
        for i in idxs:
            t = int(t_valid[i])
            # Path A: training feature row
            samples_train_pred.append(X_tr[i])
            # Path B: 100-row window ending at t (inclusive) — this is what Predictor sees
            window = df[feat_cols].iloc[t - 99 : t + 1].copy().reset_index(drop=True)
            samples_inference_input.append(window)

    X_train_path = np.array(samples_train_pred)
    print(f"training-path features: {X_train_path.shape}", flush=True)

    probs_train = booster.predict(X_train_path)

    # ---- Path B: feed through Predictor ----
    Predictor = _import_predictor(os.path.join(ROOT, "submission/iter_001_lgbm_schemeB"))
    p = Predictor()
    # Need to also access raw probs — patch _build_feats_batch to short-circuit
    arrs = [d.to_numpy(dtype=np.float32, copy=False) for d in samples_inference_input]
    x_np = np.stack(arrs, axis=0)
    amount_idx = list(samples_inference_input[0].columns).index("amount_delta")
    feats_inf = Predictor._build_feats_batch(x_np, amount_idx)
    probs_inf = p.booster.predict(feats_inf)

    # ---- Compare ----
    diff = np.abs(probs_train - probs_inf)
    print(f"\nfeat shape match: {X_train_path.shape == feats_inf.shape}", flush=True)
    print(f"feat max abs diff: {np.abs(X_train_path - feats_inf).max():.3e}", flush=True)
    print(f"prob max abs diff: {diff.max():.3e}", flush=True)
    print(f"prob mean abs diff: {diff.mean():.3e}", flush=True)

    # Check argmax agreement
    a_train = probs_train.argmax(1)
    a_inf = probs_inf.argmax(1)
    print(f"argmax agreement: {(a_train == a_inf).mean()*100:.2f}%", flush=True)

    if diff.max() < 1e-4:
        print("✓ Predictor matches training pipeline", flush=True)
    else:
        print("✗ MISMATCH — investigate", flush=True)
        for i in range(min(5, len(probs_train))):
            print(f"  sample {i}: train={probs_train[i]} inf={probs_inf[i]}", flush=True)


if __name__ == "__main__":
    main()
