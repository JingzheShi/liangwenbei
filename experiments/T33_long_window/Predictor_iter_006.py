"""Predictor for iter_006: Scheme J (583-d) h_60 N-seed ensemble + h_40 (223-d Scheme C).

Per call (predict on a list of 100×154 DataFrames):
  * Compute T3 derived features (without time encoding) → 69-d
  * Compute long-window features → 360-d
  * Concatenate: [154 raw][69 T3 no-time][360 long-window] = 583-d Scheme J
  * For h_40 (active): use first 223 cols (Scheme C 223-d) with iter_002 model
  * For h_60 (active): use all 583 cols, average probs across N seed boosters
  * h_5/h_10/h_20: inactive (predict 1)

Compliance with platform contract (CRITICAL_CONSTRAINTS.md §1):
  - sym / date never used; not in feature vector
  - No cross-call state on `self`; window-only computation
  - Stateless per-call → safe under shuffled batch ordering
  - Sym-agnostic → safe with training-out sym IDs
"""
from __future__ import annotations

import glob
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


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _find_horizon_models(here: str, H: int) -> List[str]:
    pat = os.path.join(here, f"model_h{H}_seed*.txt")
    seed_paths = sorted(glob.glob(pat))
    if seed_paths:
        return seed_paths
    single = os.path.join(here, f"model_h{H}.txt")
    return [single] if os.path.isfile(single) else []


class Predictor:
    def __init__(self) -> None:
        here = os.path.dirname(os.path.abspath(__file__))
        # compute_schemeJ.py provides: compute_mlofi/wmp/rv/ewma_intst,
        # feature_v1_columns_no_time, compute_long_window
        self._j = _load_module("iter006_schemeJ", os.path.join(here, "compute_schemeJ.py"))

        cfg_path = os.path.join(here, "config.json")
        with open(cfg_path) as f:
            cfg_main = json.load(f)
        self._raw_feat_cols: List[str] = list(cfg_main["feature"])
        try:
            self._amount_delta_idx = self._raw_feat_cols.index("amount_delta")
        except ValueError:
            self._amount_delta_idx = -1

        self._t3_feat_cols: List[str] = list(self._j.feature_v1_columns_no_time())
        self._n_base = len(self._raw_feat_cols) + len(self._t3_feat_cols)  # 223
        # 583-d Scheme J full names not strictly needed at inference (we use slicing)

        thresholds_path = os.path.join(here, "thresholds.json")
        with open(thresholds_path) as f:
            tcfg = json.load(f)
        self._horizons: List[Dict] = tcfg["horizons"]

        # Load active boosters per horizon (1+ each)
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
        mlofi = self._j.compute_mlofi(df)
        wmp = self._j.compute_wmp(df)
        rv = self._j.compute_rv(wmp["wmp_lvl1"])
        ewma = self._j.compute_ewma_intst(df)
        return pd.concat([mlofi, wmp, rv, ewma], axis=1)

    def _compute_window_full(self, df: pd.DataFrame) -> np.ndarray:
        """Returns 583-d feature vector for the LAST tick of df."""
        t3 = self._compute_t3_no_time(df)
        # Long-window features (T, 360); take last row
        long_feats = self._j.compute_long_window(df, t3)
        long_last = long_feats[-1].astype(np.float32, copy=False)
        # raw last (with log1p amount_delta)
        raw_last = df[self._raw_feat_cols].iloc[-1].to_numpy(dtype=np.float32, copy=True)
        if self._amount_delta_idx >= 0:
            v = raw_last[self._amount_delta_idx]
            raw_last[self._amount_delta_idx] = np.sign(v) * np.log1p(abs(v))
        # T3 last (no time)
        t3_last = t3[self._t3_feat_cols].iloc[-1].to_numpy(dtype=np.float32, copy=False)
        return np.concatenate([raw_last, t3_last, long_last]).astype(np.float32, copy=False)

    def _compute_batch_features(self, batches: List[pd.DataFrame]) -> np.ndarray:
        # 154 + 69 + 360 = 583
        n_feat = self._n_base + 360
        feats = np.empty((len(batches), n_feat), dtype=np.float32)
        for i, df in enumerate(batches):
            feats[i] = self._compute_window_full(df)
        return feats

    def _ensemble_predict(self, boosters: List[lgb.Booster], X: np.ndarray) -> np.ndarray:
        if len(boosters) == 1:
            return boosters[0].predict(X)
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

        # Pre-slice for horizons that use 223-d Scheme C
        feats_223 = feats[:, : self._n_base]

        for hcfg in self._horizons:
            if not hcfg.get("active", True):
                continue
            H = int(hcfg["h"])
            boosters = self._booster_lists.get(H)
            if not boosters:
                continue
            n_feat_h = boosters[0].num_feature()
            if n_feat_h == self._n_base:
                X_h = feats_223
            elif n_feat_h == feats.shape[1]:
                X_h = feats
            else:
                # Fallback: if model expects something else, raise
                raise RuntimeError(
                    f"booster for h={H} expects {n_feat_h} features but pipeline "
                    f"produces {self._n_base}/Scheme-C or {feats.shape[1]}/Scheme-J"
                )
            probs = self._ensemble_predict(boosters, X_h)
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
