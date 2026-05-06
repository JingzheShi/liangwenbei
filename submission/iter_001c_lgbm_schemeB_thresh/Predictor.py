"""Predictor for iter_001c: LightGBM Scheme B with confidence thresholding.

Same model as iter_001_lgbm_schemeB, but with a threshold post-processor:
    if max(prob_0, prob_2) > T and max(prob_0, prob_2) > prob_1 + DELTA:
        pred = argmax(prob_0, prob_2)
    else:
        pred = 1 (flat / no trade)

(T, DELTA) tuned on Leave-One-Sym-Out CV (T4) over Scheme A LOSO OOF predictions:
    raw argmax cum_pnl  : -22.10  (5-fold sum, ~mmpc_demo level)
    T=0.50 delta=0.15   :  +6.45  (4/5 folds positive)

Theory: T2 GBDT is brittle on cross-sym OOD; default argmax (1/3 threshold) makes too many
low-confidence trades that bleed fees on unfamiliar sym distributions. Forcing high-confidence
gating dramatically reduces brittle trades while preserving the most-confident edges.

Compliance with platform contract (CRITICAL_CONSTRAINTS.md §1):
  - sym / date never enter the feature vector
  - No cross-call state held on `self`
  - Model is sym-agnostic by construction
"""
from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd
import lightgbm as lgb


WINDOW = 100
ROLL_WS = (5, 20, 60)

# Confidence-gating thresholds (best from LOSO Scheme A; transfers reasonably to Scheme B
# given underlying probability calibration is similar).
THRESHOLD_T = 0.50
THRESHOLD_DELTA = 0.15


class Predictor:
    def __init__(self) -> None:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.txt")
        self.booster = lgb.Booster(model_file=model_path)

    @staticmethod
    def _build_feats_batch(arr: np.ndarray, amount_delta_idx: int) -> np.ndarray:
        if amount_delta_idx >= 0:
            col = arr[:, :, amount_delta_idx]
            arr = arr.copy()
            arr[:, :, amount_delta_idx] = np.sign(col) * np.log1p(np.abs(col))

        parts = [arr[:, -1, :]]
        for W in ROLL_WS:
            sl = arr[:, -W:, :]
            parts.append(sl.mean(axis=1, dtype=np.float32))
            parts.append(sl.std(axis=1, dtype=np.float32))
            parts.append(sl.min(axis=1).astype(np.float32, copy=False))
            parts.append(sl.max(axis=1).astype(np.float32, copy=False))
        return np.concatenate(parts, axis=1)

    @staticmethod
    def _threshold_predict(probs: np.ndarray) -> np.ndarray:
        """Apply confidence-gating: only trade when one side is sufficiently dominant."""
        p0 = probs[:, 0]
        p1 = probs[:, 1]
        p2 = probs[:, 2]
        side_max = np.maximum(p0, p2)
        take = (side_max > THRESHOLD_T) & (side_max > p1 + THRESHOLD_DELTA)
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
        x_np = np.stack(arrs, axis=0)

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
