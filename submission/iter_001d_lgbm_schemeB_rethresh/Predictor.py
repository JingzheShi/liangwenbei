"""Predictor for iter_001d: LightGBM Scheme B with re-tuned confidence thresholding.

Same model as iter_001c_lgbm_schemeB_thresh, but (T, DELTA) re-tuned on Scheme B's
own LOSO OOF predictions instead of transferring from Scheme A.

Decision rule:
    side_max = max(prob_0, prob_2)
    if side_max >= T and side_max > prob_1 + DELTA:
        pred = argmax(prob_0, prob_2)
    else:
        pred = 1 (flat / no trade)

T5a sweep on Scheme B 5-fold LOSO OOF (442k rows):
    raw argmax            : sum_cum_pnl = +1.33  (3/5 folds positive)
    T=0.50 delta=0.15 (c) : sum_cum_pnl = +8.15  (5/5 folds positive)  <- iter_001c
    T=0.45 delta=0.05 (d) : sum_cum_pnl = +11.11 (4/5 folds positive)  <- iter_001d (best)

Trade-off vs iter_001c: iter_001d trades more (sum_n_active 83k vs 33k) and gives up
sym=2's positive +0.93 (becomes -0.98) but more than makes up for it on sym=1, 4
(+0.57 and +3.47 net gains). Best aggregate sum.

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

# Confidence-gating thresholds (re-tuned on LOSO Scheme B OOF, T5a).
THRESHOLD_T = 0.45
THRESHOLD_DELTA = 0.05


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
