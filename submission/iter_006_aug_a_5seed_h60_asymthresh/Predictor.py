"""Predictor for iter_006: iter_005b base (5-seed aug_a ensemble for h_60)
+ asymmetric (T_up, T_dn, d_up, d_dn) gating from T30 DE optimization.

Identical to Predictor_iter_005b.py except for the threshold rule, which now
accepts per-side thresholds. Backward-compatible with iter_005b style configs
that use {"T": x, "delta": y} (read as symmetric T_up=T_dn=T, d_up=d_dn=δ).

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
from typing import Dict, List

import numpy as np
import pandas as pd
import lightgbm as lgb

WINDOW = 100
HORIZON_LIST = (5, 10, 20, 40, 60)
HORIZON_TO_IDX = {h: i for i, h in enumerate(HORIZON_LIST)}


def _load_t3_module(here: str):
    spec = importlib.util.spec_from_file_location(
        "iter_006_t3_compute", os.path.join(here, "compute.py")
    )
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
    def _threshold_predict(probs: np.ndarray, hcfg: Dict) -> np.ndarray:
        """Asymmetric gate: pred=2 if p2>=T_up & p2>p1+d_up & p2>p0;
                            pred=0 if p0>=T_dn & p0>p1+d_dn & p0>p2;
                            else 1.
        Falls back to symmetric (T, delta) if T_up/T_dn not present.
        """
        p0 = probs[:, 0]; p1 = probs[:, 1]; p2 = probs[:, 2]
        if "T_up" in hcfg or "T_dn" in hcfg:
            T_up = float(hcfg.get("T_up", hcfg.get("T", 0.50)))
            T_dn = float(hcfg.get("T_dn", hcfg.get("T", 0.50)))
            d_up = float(hcfg.get("d_up", hcfg.get("delta", 0.00)))
            d_dn = float(hcfg.get("d_dn", hcfg.get("delta", 0.00)))
            take_up = (p2 >= T_up) & (p2 > p1 + d_up) & (p2 > p0)
            take_dn = (p0 >= T_dn) & (p0 > p1 + d_dn) & (p0 > p2)
            pred = np.full(probs.shape[0], 1, dtype=np.int64)
            pred[take_up] = 2
            pred[take_dn] = 0
            return pred
        # Symmetric fallback (matches iter_005b)
        T = float(hcfg.get("T", 0.50))
        delta = float(hcfg.get("delta", 0.00))
        side_max = np.maximum(p0, p2)
        take = (side_max >= T) & (side_max > p1 + delta)
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
            preds = self._threshold_predict(probs, hcfg)
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
