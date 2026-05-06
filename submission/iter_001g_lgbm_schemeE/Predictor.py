"""Predictor for iter_001g: LightGBM Scheme E.

Features (164 total):
  - 154 baseline last-tick (T2 Scheme A)
  - F1 (2): amount_delta z-score (last + q95) within past-100-tick window
  - F2 (2): spread-normalized Δmidprice (last + max abs)
  - F3 (1): |current Δmid| / realized_vol (current move in window-relative units)
  - F4 (5): order-flow imbalance ratios at last tick

All F-features sym-agnostic — no per-sym normalization or sym ID input.

Decision rule (re-tuned on Scheme E LOSO OOF in T9):
    side_max = max(prob_0, prob_2)
    if side_max >= T and side_max > prob_1 + DELTA:
        pred = argmax(prob_0, prob_2)
    else:
        pred = 1 (flat)

Compliance with platform contract (CRITICAL_CONSTRAINTS.md §1):
  - sym / date never enter the feature vector
  - No cross-call state held on `self`
  - Model is sym-agnostic by construction (within-window stats, scale-free ratios)
"""
from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd
import lightgbm as lgb


WINDOW = 100
EPS = 1e-8

# Will be overridden from results — placeholder until training finishes.
THRESHOLD_T = 0.45
THRESHOLD_DELTA = 0.05


class Predictor:
    def __init__(self) -> None:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.txt")
        self.booster = lgb.Booster(model_file=model_path)

    @staticmethod
    def _build_feats_batch(arr: np.ndarray, idx_map: dict) -> np.ndarray:
        """arr: (B, 100, 154) raw feature window. Returns (B, 164) float32 features.

        Column layout matches build_features.py: 154 raw (with amount_delta log1p'd)
        + F1 (2) + F2 (2) + F3 (1) + F4 (5).
        """
        B = arr.shape[0]
        # Raw last-tick view of all 154 cols (we'll log1p amount_delta in place)
        # Make a working copy because we'll overwrite amount_delta.
        x = arr.copy().astype(np.float32, copy=False)

        ai = idx_map.get("amount_delta", -1)
        # F1 from raw amount_delta BEFORE log1p
        if ai >= 0:
            raw_amount = x[:, :, ai]  # (B, 100)
            log_amt = np.sign(raw_amount) * np.log1p(np.abs(raw_amount))
            mean = log_amt.mean(axis=1, keepdims=True)
            std = log_amt.std(axis=1, keepdims=True) + EPS
            z_full = (log_amt - mean) / std  # (B, 100)
            f1_last = z_full[:, -1].astype(np.float32)
            f1_q95 = np.quantile(z_full, 0.95, axis=1).astype(np.float32)
            # Now log1p amount_delta in place (matches training)
            x[:, :, ai] = log_amt
        else:
            f1_last = np.zeros(B, dtype=np.float32)
            f1_q95 = np.zeros(B, dtype=np.float32)

        # F2 + F3 from raw bid1 / ask1 (NOT modified by log1p)
        bi = idx_map.get("bid1", -1)
        ki = idx_map.get("ask1", -1)
        si = idx_map.get("spread1", -1)
        if bi >= 0 and ki >= 0 and si >= 0:
            bid1 = arr[:, :, bi].astype(np.float32, copy=False)
            ask1 = arr[:, :, ki].astype(np.float32, copy=False)
            spread1 = arr[:, :, si].astype(np.float32, copy=False)
            spread_actual = spread1 + 1.0  # ask1 - bid1
            mid_actual = (bid1 + ask1) / 2.0
            # Δmid: prepend mid[:, :1] so first diff is 0
            delta_mid = np.diff(mid_actual, axis=1, prepend=mid_actual[:, :1]).astype(np.float32)
            norm_move = delta_mid / (np.abs(spread_actual) + EPS)
            f2_last = norm_move[:, -1].astype(np.float32)
            f2_max = np.abs(norm_move).max(axis=1).astype(np.float32)
            # F3
            realized_vol = delta_mid.std(axis=1) + EPS
            cur_abs = np.abs(delta_mid[:, -1])
            f3_last = (cur_abs / realized_vol).astype(np.float32)
        else:
            f2_last = np.zeros(B, dtype=np.float32)
            f2_max = np.zeros(B, dtype=np.float32)
            f3_last = np.zeros(B, dtype=np.float32)

        # F4 ratios at last tick
        def _last(name: str) -> np.ndarray:
            i = idx_map.get(name, -1)
            if i < 0:
                return np.zeros(B, dtype=np.float32)
            return arr[:, -1, i].astype(np.float32, copy=False)

        lb = _last("lb_intst"); la = _last("la_intst")
        mb = _last("mb_intst"); ma = _last("ma_intst")
        cb = _last("cb_intst"); ca = _last("ca_intst")
        f4_mb_over_lb = mb / (lb + mb + EPS)
        f4_ma_over_la = ma / (la + ma + EPS)
        f4_cancel_buy = cb / (lb + mb + cb + EPS)
        f4_cancel_sell = ca / (la + ma + ca + EPS)
        f4_book_pressure = (mb + lb) / (ma + la + mb + lb + EPS)

        # Concatenate: last-tick of 154 + F1 + F2 + F3 + F4 = 164
        last = x[:, -1, :]  # (B, 154)
        f1_block = np.stack([f1_last, f1_q95], axis=1)
        f2_block = np.stack([f2_last, f2_max], axis=1)
        f3_block = f3_last[:, None]
        f4_block = np.stack(
            [f4_mb_over_lb, f4_ma_over_la, f4_cancel_buy, f4_cancel_sell, f4_book_pressure],
            axis=1,
        )
        feats = np.concatenate([last, f1_block, f2_block, f3_block, f4_block], axis=1)
        return feats.astype(np.float32, copy=False)

    @staticmethod
    def _threshold_predict(probs: np.ndarray) -> np.ndarray:
        p0 = probs[:, 0]; p1 = probs[:, 1]; p2 = probs[:, 2]
        side_max = np.maximum(p0, p2)
        take = (side_max >= THRESHOLD_T) & (side_max > p1 + THRESHOLD_DELTA)
        side_pred = np.where(p2 > p0, 2, 0)
        return np.where(take, side_pred, 1).astype(np.int64)

    def predict(self, batches: List[pd.DataFrame]) -> List[List[int]]:
        if not batches:
            return []
        first = batches[0]
        cols = list(first.columns)
        # Build name → column-index map once per call (fast, no state across calls)
        idx_map = {c: i for i, c in enumerate(cols)}

        arrs = [df.to_numpy(dtype=np.float32, copy=False) for df in batches]
        x_np = np.stack(arrs, axis=0)  # (B, 100, 154)

        feats = self._build_feats_batch(x_np, idx_map)
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
    print("smoke test predict ->", p.predict([df, df, df])[:2])
