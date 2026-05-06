"""Predictor for iter_005: LightGBM + XGBoost ensemble at h=10, single-model
LightGBM at h=5/h=20/h=40/h=60. Scheme C 223-d features.

Compliance with platform contract (CRITICAL_CONSTRAINTS.md §1):
  - sym / date never enter the feature vector
  - No cross-call state on `self` (per-call window only)
  - Model is sym-agnostic by construction
  - T3 features (MLOFI / WMP / RV / EWMA) computed fresh per 100-row window

Difference from iter_003:
  h=10 horizon now uses an ensemble of LightGBM boosters + XGBoost boosters.
  thresholds.json["horizons"] entry for h=10 carries:
    `lgbm_seeds`: list of LightGBM seed IDs to load model_h10_seed{S}.txt
    `xgb_seeds`: list of XGBoost seed IDs to load xgb_model_h10_seed{S}.ubj
  Probability is the simple arithmetic mean across all boosters.
  Other horizons (5, 20, 40, 60) reuse the iter_002 single LightGBM models.

Bundle:
  thresholds.json              per-horizon (T, delta), active flag,
                              for h=10 also `lgbm_seeds` and/or `xgb_seeds`
  model_h{H}.txt              single LightGBM booster (h=5/20/40/60)
  model_h10_seed{S}.txt       LightGBM boosters for h=10 ensemble
  xgb_model_h10_seed{S}.ubj   XGBoost boosters for h=10 ensemble
"""
from __future__ import annotations

import importlib.util
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

WINDOW = 100
HORIZON_LIST = (5, 10, 20, 40, 60)
HORIZON_TO_IDX = {h: i for i, h in enumerate(HORIZON_LIST)}


def _load_t3_module(here: str):
    spec = importlib.util.spec_from_file_location(
        "iter_005_t3_compute", os.path.join(here, "compute.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Predictor:
    def __init__(self) -> None:
        here = os.path.dirname(os.path.abspath(__file__))
        self._t3 = _load_t3_module(here)
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

        # boosters[H] = list of (kind, model)
        # kind in {"lgbm", "xgb"}
        self._boosters: Dict[int, List] = {}
        for hcfg in self._horizons:
            if not hcfg.get("active", True):
                continue
            H = int(hcfg["h"])
            ms = []
            lgbm_seeds = hcfg.get("lgbm_seeds")
            xgb_seeds = hcfg.get("xgb_seeds")
            legacy_seeds = hcfg.get("seeds")  # iter_003-style
            if lgbm_seeds:
                for s in lgbm_seeds:
                    mp = os.path.join(here, f"model_h{H}_seed{s}.txt")
                    if os.path.isfile(mp):
                        ms.append(("lgbm", lgb.Booster(model_file=mp)))
            elif legacy_seeds:
                for s in legacy_seeds:
                    mp = os.path.join(here, f"model_h{H}_seed{s}.txt")
                    if os.path.isfile(mp):
                        ms.append(("lgbm", lgb.Booster(model_file=mp)))
            else:
                # single-model fallback (h=5/20/40/60)
                mp = os.path.join(here, f"model_h{H}.txt")
                if os.path.isfile(mp):
                    ms.append(("lgbm", lgb.Booster(model_file=mp)))
            if xgb_seeds:
                for s in xgb_seeds:
                    mp = os.path.join(here, f"xgb_model_h{H}_seed{s}.ubj")
                    if os.path.isfile(mp):
                        booster = xgb.Booster()
                        booster.load_model(mp)
                        ms.append(("xgb", booster))
            if ms:
                self._boosters[H] = ms

    @staticmethod
    def _threshold_predict(probs: np.ndarray, T_thr: float, delta: float) -> np.ndarray:
        p0 = probs[:, 0]; p1 = probs[:, 1]; p2 = probs[:, 2]
        side_max = np.maximum(p0, p2)
        take = (side_max >= T_thr) & (side_max > p1 + delta)
        side_pred = np.where(p2 > p0, 2, 0)
        return np.where(take, side_pred, 1).astype(np.int64)

    def _compute_t3_no_time(self, df: pd.DataFrame) -> pd.DataFrame:
        mlofi = self._t3.compute_mlofi(df)
        wmp = self._t3.compute_wmp(df)
        rv = self._t3.compute_rv(wmp["wmp_lvl1"])
        ewma = self._t3.compute_ewma_intst(df)
        return pd.concat([mlofi, wmp, rv, ewma], axis=1)

    def _compute_window_features(self, df: pd.DataFrame) -> np.ndarray:
        t3 = self._compute_t3_no_time(df)
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
        dmat = None  # build lazily only if any xgb model is used
        for hcfg in self._horizons:
            if not hcfg.get("active", True):
                continue
            H = int(hcfg["h"])
            ms = self._boosters.get(H)
            if not ms:
                continue
            prob_sum = np.zeros((B, 3), dtype=np.float32)
            for kind, model in ms:
                if kind == "lgbm":
                    prob_sum += model.predict(feats).astype(np.float32)
                else:  # xgb
                    if dmat is None:
                        dmat = xgb.DMatrix(feats)
                    prob_sum += model.predict(dmat).astype(np.float32)
            probs = prob_sum / float(len(ms))
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
