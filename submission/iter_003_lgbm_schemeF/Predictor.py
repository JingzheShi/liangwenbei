"""Predictor for iter_003: LightGBM Scheme F multi-horizon.

Feature vector (per 100-row window): 154 raw + 154 zscore100 + 69 T3 (no time
encoding) = 377 dims.

Compliance with platform contract (CRITICAL_CONSTRAINTS.md §1):
  - sym / date never enter the feature vector
  - No cross-call state held on `self` (per-call window only)
  - Model is sym-agnostic by construction
  - Z-score is computed against the **current 100-row window** only — no global
    or per-sym statistics. Truly adaptive / OOD-robust.
  - T3 features (MLOFI / WMP / RV / EWMA) computed fresh per 100-row window.

We DROPPED the 3 T3 time-encoding features so we don't need `time` in
config.feature (LocalEvaluator coerces all feature columns to float32, which
breaks on datetime.time).

Bundle:
  config.json       feature list (154 raw cols)
  thresholds.json   per-horizon (T, delta) and active flag
  model_h{H}.txt    LightGBM booster, 377-d input (one per active horizon)
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
EPS = 1e-8


def _load_t3_module(here: str):
    spec = importlib.util.spec_from_file_location(
        "iter_003_t3_compute", os.path.join(here, "compute.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Predictor:
    def __init__(self) -> None:
        here = os.path.dirname(os.path.abspath(__file__))
        self._t3 = _load_t3_module(here)
        # Drop the 3 time-encoding cols at the tail; keep the first 69 of 72 T3
        all_t3_cols = list(self._t3.feature_v1_columns())
        self._t3_feat_cols: List[str] = [c for c in all_t3_cols if not c.startswith("time_")]

        cfg_path = os.path.join(here, "config.json")
        with open(cfg_path) as f:
            cfg_main = json.load(f)
        self._raw_feat_cols: List[str] = list(cfg_main["feature"])
        try:
            self._amount_delta_idx = self._raw_feat_cols.index("amount_delta")
        except ValueError:
            self._amount_delta_idx = -1

        thresholds_path = os.path.join(here, "thresholds.json")
        with open(thresholds_path) as f:
            tcfg = json.load(f)
        self._horizons: List[Dict] = tcfg["horizons"]

        self._boosters: Dict[int, lgb.Booster] = {}
        for hcfg in self._horizons:
            if not hcfg.get("active", True):
                continue
            H = int(hcfg["h"])
            mp = os.path.join(here, f"model_h{H}.txt")
            if os.path.isfile(mp):
                self._boosters[H] = lgb.Booster(model_file=mp)

    @staticmethod
    def _threshold_predict(probs: np.ndarray, T_thr: float, delta: float) -> np.ndarray:
        p0 = probs[:, 0]; p1 = probs[:, 1]; p2 = probs[:, 2]
        side_max = np.maximum(p0, p2)
        take = (side_max >= T_thr) & (side_max > p1 + delta)
        side_pred = np.where(p2 > p0, 2, 0)
        return np.where(take, side_pred, 1).astype(np.int64)

    def _compute_t3_no_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute the 69 non-time T3 features (MLOFI + WMP + RV + EWMA)."""
        mlofi = self._t3.compute_mlofi(df)
        wmp = self._t3.compute_wmp(df)
        rv = self._t3.compute_rv(wmp["wmp_lvl1"])
        ewma = self._t3.compute_ewma_intst(df)
        return pd.concat([mlofi, wmp, rv, ewma], axis=1)

    def _compute_window_features(self, df: pd.DataFrame) -> np.ndarray:
        """Build 377-d last-tick feature vector for one 100-row window.

        Layout: [raw_154, zscore100_154, t3_69]
        """
        raw_full = df[self._raw_feat_cols].to_numpy(dtype=np.float32, copy=True)  # (100, 154)
        if self._amount_delta_idx >= 0:
            col = raw_full[:, self._amount_delta_idx]
            raw_full[:, self._amount_delta_idx] = np.sign(col) * np.log1p(np.abs(col))

        # Last-tick raw
        raw_last = raw_full[-1, :]  # (154,)

        # Window z-score (sym-agnostic): mean/std over the full 100-row window
        mean_w = raw_full.mean(axis=0)  # (154,)
        std_w = raw_full.std(axis=0) + EPS  # (154,)
        z_last = (raw_last - mean_w) / std_w  # (154,)

        # T3 last-tick (computed online from this window)
        t3_df = self._compute_t3_no_time(df)
        t3_last = t3_df[self._t3_feat_cols].iloc[-1].to_numpy(dtype=np.float32, copy=False)

        out = np.concatenate([raw_last, z_last, t3_last]).astype(np.float32, copy=False)
        if not np.isfinite(out).all():
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    def _compute_batch_features(self, batches: List[pd.DataFrame]) -> np.ndarray:
        feats = np.empty(
            (len(batches), 2 * len(self._raw_feat_cols) + len(self._t3_feat_cols)),
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
            out[:, HORIZON_TO_IDX[H]] = preds
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
