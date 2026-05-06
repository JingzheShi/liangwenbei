"""Predictor for iter_002: LightGBM Scheme C1 (154 raw + 72 T3 features) multi-horizon.

Compliance with platform contract (CRITICAL_CONSTRAINTS.md §1):
  - sym / date never enter the feature vector
  - No cross-call state held on `self` (per-call window only)
  - Model is sym-agnostic by construction
  - T3 features (MLOFI / WMP / RV / EWMA / time) computed fresh per 100-row window

config.feature includes 'time' so the platform forwards it for time-encoding.
If 'time' parses fail (e.g. sanity_check passes random floats), time-encoding
falls back to zeros so predict() never crashes.

Bundle:
  thresholds.json   per-horizon (T, delta) and active flag
  model_h{H}.txt    LightGBM booster, 226-dim input (one per active horizon)
"""
from __future__ import annotations

import importlib.util
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import lightgbm as lgb

WINDOW = 100
HORIZON_LIST = (5, 10, 20, 40, 60)
HORIZON_TO_IDX = {h: i for i, h in enumerate(HORIZON_LIST)}


def _load_t3_module(here: str):
    spec = importlib.util.spec_from_file_location(
        "iter_002_t3_compute", os.path.join(here, "compute.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Predictor:
    def __init__(self) -> None:
        here = os.path.dirname(os.path.abspath(__file__))
        self._t3 = _load_t3_module(here)
        self._t3_feat_cols: List[str] = list(self._t3.feature_v1_columns())

        cfg_path = os.path.join(here, "config.json")
        with open(cfg_path) as f:
            cfg_main = json.load(f)
        self._all_input_cols: List[str] = list(cfg_main["feature"])
        # Raw model-input cols = 154 from default LOB feature set (drop 'time')
        self._raw_feat_cols: List[str] = [c for c in self._all_input_cols if c != "time"]
        try:
            self._amount_delta_idx = self._raw_feat_cols.index("amount_delta")
        except ValueError:
            self._amount_delta_idx = -1

        thresholds_path = os.path.join(here, "thresholds.json")
        with open(thresholds_path) as f:
            tcfg = json.load(f)
        self._horizons: List[Dict] = tcfg["horizons"]

        # Load active boosters
        self._boosters: Dict[int, lgb.Booster] = {}
        for hcfg in self._horizons:
            if not hcfg.get("active", True):
                continue
            H = int(hcfg["h"])
            mp = os.path.join(here, f"model_h{H}.txt")
            if os.path.isfile(mp):
                self._boosters[H] = lgb.Booster(model_file=mp)

    @staticmethod
    def _threshold_predict(
        probs: np.ndarray, T_thr: float, delta: float
    ) -> np.ndarray:
        p0 = probs[:, 0]; p1 = probs[:, 1]; p2 = probs[:, 2]
        side_max = np.maximum(p0, p2)
        take = (side_max >= T_thr) & (side_max > p1 + delta)
        side_pred = np.where(p2 > p0, 2, 0)
        return np.where(take, side_pred, 1).astype(np.int64)

    def _compute_t3_safe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute the 72 T3 features. Fall back to zeros for time encoding if parse fails."""
        mlofi = self._t3.compute_mlofi(df)
        wmp = self._t3.compute_wmp(df)
        rv = self._t3.compute_rv(wmp["wmp_lvl1"])
        ewma = self._t3.compute_ewma_intst(df)
        try:
            timef = self._t3.compute_time_encoding(df)
        except Exception:
            timef = pd.DataFrame({
                "time_minutes_since_session_start": np.zeros(len(df), dtype=np.int32),
                "time_session_progress": np.zeros(len(df), dtype=np.float32),
                "time_is_pm": np.zeros(len(df), dtype=np.int8),
            }, index=df.index)
        return pd.concat([mlofi, wmp, rv, ewma, timef], axis=1)

    def _compute_window_features(self, df: pd.DataFrame) -> np.ndarray:
        """Build 226-dim last-tick feature vector for one 100-row window."""
        t3 = self._compute_t3_safe(df)
        raw_last = df[self._raw_feat_cols].iloc[-1].to_numpy(dtype=np.float32, copy=True)
        if self._amount_delta_idx >= 0:
            v = raw_last[self._amount_delta_idx]
            raw_last[self._amount_delta_idx] = np.sign(v) * np.log1p(abs(v))
        t3_last = t3[self._t3_feat_cols].iloc[-1].to_numpy(dtype=np.float32, copy=False)
        return np.concatenate([raw_last, t3_last]).astype(np.float32, copy=False)

    def _compute_batch_features(self, batches: List[pd.DataFrame]) -> np.ndarray:
        feats = np.empty(
            (len(batches), len(self._raw_feat_cols) + len(self._t3_feat_cols)),
            dtype=np.float32,
        )
        for i, df in enumerate(batches):
            feats[i] = self._compute_window_features(df)
        return feats

    def predict(self, batches: List[pd.DataFrame]) -> List[List[int]]:
        if not batches:
            return []
        feats = self._compute_batch_features(batches)
        B = feats.shape[0]
        out = np.ones((B, 5), dtype=np.int64)
        for hcfg in self._horizons:
            if not hcfg.get("active", True):
                continue
            H = int(hcfg["h"])
            booster = self._boosters.get(H)
            if booster is None:
                continue
            probs = booster.predict(feats)
            T_thr = float(hcfg.get("T", 0.50))
            delta = float(hcfg.get("delta", 0.15))
            preds = self._threshold_predict(probs, T_thr, delta)
            col_idx = HORIZON_TO_IDX[H]
            out[:, col_idx] = preds
        return out.tolist()


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    cfg = json.load(open(os.path.join(here, "config.json")))
    feats = cfg["feature"]
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        rng.standard_normal((100, len(feats))).astype(np.float32),
        columns=feats,
    )
    p = Predictor()
    out = p.predict([df, df, df])
    print("smoke test predict ->", out[0], "len=", len(out))
