"""Predictor for iter_001: LightGBM Scheme B (last-tick + rolling{mean,std,min,max} over W=5,20,60).

Trained only on label_60 (best official horizon for T2). Other 4 horizons output 1 (flat / no
trade) — conservative; we lose those horizon scores but avoid bleeding fees.

Compliance with platform contract (CRITICAL_CONSTRAINTS.md §1):
  - sym / date never enter the feature vector (config["feature"] is the 154 raw cols only)
  - No cross-call state held on `self` (each predict() is independent)
  - Model is sym-agnostic by construction (no sym embed, no per-sym norm)
"""
from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd
import lightgbm as lgb


WINDOW = 100              # input ticks per sample (matches platform contract)
ROLL_WS = (5, 20, 60)     # rolling-window sizes (must match training)


class Predictor:
    def __init__(self) -> None:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.txt")
        self.booster = lgb.Booster(model_file=model_path)

    @staticmethod
    def _build_feats_batch(arr: np.ndarray, amount_delta_idx: int) -> np.ndarray:
        """arr: (B, T, F) float32.  Returns (B, F * 13) = (B, 2002).

        Order (must match training feat_names):
            last_tick (F),
            roll5_{mean,std,min,max} (F each),
            roll20_{mean,std,min,max} (F each),
            roll60_{mean,std,min,max} (F each).

        Stats use ddof=0 (numpy default), matching build_features.compute_rolling_stats.
        """
        if amount_delta_idx >= 0:
            # log1p with sign preservation (matches training)
            col = arr[:, :, amount_delta_idx]
            arr = arr.copy()
            arr[:, :, amount_delta_idx] = np.sign(col) * np.log1p(np.abs(col))

        parts = [arr[:, -1, :]]                        # last tick: (B, F)
        for W in ROLL_WS:
            sl = arr[:, -W:, :]                        # (B, W, F)
            parts.append(sl.mean(axis=1, dtype=np.float32))
            parts.append(sl.std(axis=1, dtype=np.float32))   # ddof=0
            parts.append(sl.min(axis=1).astype(np.float32, copy=False))
            parts.append(sl.max(axis=1).astype(np.float32, copy=False))
        return np.concatenate(parts, axis=1)           # (B, F*13)

    def predict(self, batches: List[pd.DataFrame]) -> List[List[int]]:
        """Each df: shape (100, 154). Returns [[p5, p10, p20, p40, p60], ...]."""
        if not batches:
            return []

        first = batches[0]
        cols = list(first.columns)
        try:
            amount_delta_idx = cols.index("amount_delta")
        except ValueError:
            amount_delta_idx = -1

        arrs = [df.to_numpy(dtype=np.float32, copy=False) for df in batches]
        x_np = np.stack(arrs, axis=0)                  # (B, T, F)

        feats = self._build_feats_batch(x_np, amount_delta_idx)   # (B, 2002)
        probs = self.booster.predict(feats)                       # (B, 3)
        pred60 = probs.argmax(axis=1).astype(np.int64)            # (B,)

        # Other horizons → 1 (flat, no trade) per design doc.
        B = pred60.shape[0]
        out = np.ones((B, 5), dtype=np.int64)
        out[:, 4] = pred60                                        # label_60 is index 4
        return out.tolist()


if __name__ == "__main__":
    import json
    cfg = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")))
    feats = cfg["feature"]
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.standard_normal((100, len(feats))).astype(np.float32), columns=feats)
    p = Predictor()
    print("smoke test predict ->", p.predict([df, df, df]))
