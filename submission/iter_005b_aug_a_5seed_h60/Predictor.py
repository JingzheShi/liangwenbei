"""Predictor for iter_005b: LightGBM Scheme C 223-d, h_60 = 5-seed aug_a ensemble.

Same as iter_002 Predictor except for h_60: instead of one booster, load N
boosters (model_h60_seed{S}.txt) and average their probabilities before
applying the (T, delta) threshold rule.

For other horizons (h_5/10/20 inactive, h_40 active), behavior is identical
to iter_002.

Compliance with platform contract (CRITICAL_CONSTRAINTS.md §1):
  - sym / date never enter the feature vector
  - No cross-call state held on `self` (per-call window only)
  - sym-agnostic by construction
  - Aug applied only at training time; inference is plain LightGBM
"""
from __future__ import annotations

import glob
import importlib.util
import json
import os
import re
from typing import Dict, List

import numpy as np
import pandas as pd
import lightgbm as lgb

WINDOW = 100
HORIZON_LIST = (5, 10, 20, 40, 60)
HORIZON_TO_IDX = {h: i for i, h in enumerate(HORIZON_LIST)}


def _load_t3_module(here: str):
    spec = importlib.util.spec_from_file_location(
        "iter_005b_t3_compute", os.path.join(here, "compute.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _find_horizon_models(here: str, H: int) -> List[str]:
    """Return list of model files for horizon H, sorted for stable ensembling.

    Looks first for model_h{H}_seed*.txt (ensemble), falls back to model_h{H}.txt.
    """
    pat = os.path.join(here, f"model_h{H}_seed*.txt")
    seed_paths = sorted(glob.glob(pat))
    if seed_paths:
        return seed_paths
    single = os.path.join(here, f"model_h{H}.txt")
    return [single] if os.path.isfile(single) else []


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

        # For each active horizon, load 1+ boosters (ensemble if multiple).
        self._booster_lists: Dict[int, List[lgb.Booster]] = {}
        for hcfg in self._horizons:
            if not hcfg.get("active", True):
                continue
            H = int(hcfg["h"])
            paths = _find_horizon_models(here, H)
            if not paths:
                continue
            self._booster_lists[H] = [lgb.Booster(model_file=p) for p in paths]

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

    def _ensemble_predict(self, boosters: List[lgb.Booster], X: np.ndarray) -> np.ndarray:
        if len(boosters) == 1:
            return boosters[0].predict(X)
        # Mean of softmax probabilities
        acc = None
        for b in boosters:
            p = b.predict(X).astype(np.float32)
            acc = p if acc is None else acc + p
        return acc / float(len(boosters))

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
            boosters = self._booster_lists.get(H)
            if not boosters:
                continue
            probs = self._ensemble_predict(boosters, feats)
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
