"""Predictor for iter_001f: LightGBM Scheme D1 (window z-score, sym-agnostic).

Feature design (308 dims per sample):
    - raw last-tick (154): values at t (with amount_delta log1p sign-preserving)
    - window-zscore last-tick (154): (x[t] - mean(x[t-99..t])) / (std + 1e-8)

The window z-score is sym-agnostic adaptive normalization (no per-sym statistics,
no train-time globals), so the model generalizes better to OOD sym IDs.

Decision rule (post-prediction confidence gating, T7 sweep):
    side_max = max(prob_0, prob_2)
    if side_max >= T and side_max > prob_1 + DELTA:
        pred = argmax(prob_0, prob_2)
    else:
        pred = 1   # flat / no trade

T7 LOSO 5-fold sum_cum_pnl summary (442k OOF rows):
    raw argmax            : -5.23  (2/5 folds positive — model overconfident on OOD sym)
    T=0.45 delta=0.05  (f): +13.91 (5/5 folds positive)  <- iter_001f (best)
    iter_001d (Scheme B 0.45/0.05): +11.11

Compliance with platform contract (CRITICAL_CONSTRAINTS.md §1):
  - sym / date never enter the feature vector
  - No cross-call state on `self`; window stats computed inside each predict()
  - Sym-agnostic: window z-score uses only the current 100-tick window (no per-sym statistics)
"""
from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd
import lightgbm as lgb


WINDOW = 100
EPS = 1e-8

THRESHOLD_T = 0.45
THRESHOLD_DELTA = 0.05


class Predictor:
    def __init__(self) -> None:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.txt")
        self.booster = lgb.Booster(model_file=model_path)

    @staticmethod
    def _build_feats_batch(arr: np.ndarray, amount_delta_idx: int) -> np.ndarray:
        """arr: (B, 100, F) raw float32 features. Returns (B, 2F) float32.

        For each sample independently:
            raw = arr[:, -1, :]                              # last-tick raw
            mean = arr.mean(axis=1)
            std  = arr.std(axis=1) + EPS
            zs   = (raw - mean) / std                        # last-tick z-score
            feat = concat([raw, zs], axis=-1)
        amount_delta is log1p'd before z-score (sign-preserving).
        """
        if amount_delta_idx >= 0:
            col = arr[:, :, amount_delta_idx]
            arr = arr.copy()
            arr[:, :, amount_delta_idx] = np.sign(col) * np.log1p(np.abs(col))

        last = arr[:, -1, :]                                 # (B, F)
        mean = arr.mean(axis=1, dtype=np.float32)            # (B, F)
        std = arr.std(axis=1, dtype=np.float32) + EPS        # (B, F)
        zs = ((last - mean) / std).astype(np.float32, copy=False)  # (B, F)
        return np.concatenate([last.astype(np.float32, copy=False), zs], axis=1)

    @staticmethod
    def _threshold_predict(probs: np.ndarray) -> np.ndarray:
        p0 = probs[:, 0]
        p1 = probs[:, 1]
        p2 = probs[:, 2]
        side_max = np.maximum(p0, p2)
        take = (side_max >= THRESHOLD_T) & (side_max > p1 + THRESHOLD_DELTA)
        side_pred = np.where(p2 > p0, 2, 0)
        return np.where(take, side_pred, 1).astype(np.int64)

    def predict(self, batches: List[pd.DataFrame]) -> List[List[int]]:
        if not batches:
            return []

        first = batches[0]
        cols = list(first.columns)
        try:
            amount_delta_idx = cols.index("amount_delta")
        except ValueError:
            amount_delta_idx = -1

        arrs = [df.to_numpy(dtype=np.float32, copy=False) for df in batches]
        x_np = np.stack(arrs, axis=0)                        # (B, 100, F)

        feats = self._build_feats_batch(x_np, amount_delta_idx)
        probs = self.booster.predict(feats)
        pred60 = self._threshold_predict(probs)

        B = pred60.shape[0]
        out = np.ones((B, 5), dtype=np.int64)
        out[:, 4] = pred60
        return out.tolist()


if __name__ == "__main__":
    import json
    cfg = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")))
    feats = cfg["feature"]
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.standard_normal((100, len(feats))).astype(np.float32), columns=feats)
    p = Predictor()
    print("smoke test predict ->", p.predict([df, df, df]))
