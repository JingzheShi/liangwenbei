"""Predictor for iter_006: Scheme K (Scheme C 223 + ReVol 12 + SG 20 = 255 d)
                                 LightGBM, h_60 = 5-seed aug_a ensemble.

Differences vs iter_005b:
  - Feature vector grows by 32 dims (ReVol + Savitzky-Golay).
  - At inference, we compute these 32 extras from the raw 100-tick window
    using the bundled `revol.py` and `savgol.py` (causal, sym-agnostic).
  - h_5 / h_10 / h_20 inactive (label=1).
  - h_40 model from iter_002 expects 223-d, so we feed only the first 223
    of the new 255-d vector to the h_40 booster.
  - h_60 ensemble of 5 boosters trained on the 255-d Scheme K vector.

Compliance with platform contract (CRITICAL_CONSTRAINTS.md §1):
  - sym / date never enter the feature vector
  - No cross-call state (window-only)
  - sym-agnostic
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

# These match T35 build_features.py.
REVOL_SIGNALS = ["midprice", "wmp_lvl1"]
SG_SIGNALS = ["midprice", "wmp_lvl1", "imbalance", "mlofi_W60_lvl1", "spread1"]


def _load_module(here: str, mod_name: str, fname: str):
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(here, fname))
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
        self._here = here
        self._t3 = _load_module(here, "iter_006_t3_compute", "compute.py")
        self._revol = _load_module(here, "iter_006_revol", "revol.py")
        self._savgol = _load_module(here, "iter_006_savgol", "savgol.py")

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

        # T35 modules emit 6 ReVol features per signal × 2 signals = 12;
        # 4 SG features per signal × 5 signals = 20. Total 32.
        self._n_revol_feats = 6 * len(REVOL_SIGNALS)
        self._n_sg_feats = 4 * len(SG_SIGNALS)
        self._n_extra = self._n_revol_feats + self._n_sg_feats
        self._n_base = len(self._raw_feat_cols) + len(self._t3_feat_cols)  # 154+69=223
        self._n_total = self._n_base + self._n_extra                       # 255

        thresholds_path = os.path.join(here, "thresholds.json")
        with open(thresholds_path) as f:
            tcfg = json.load(f)
        self._horizons: List[Dict] = tcfg["horizons"]

        # For each active horizon, load 1+ boosters.
        # h_40 uses old 223-d models from iter_002 → feed [:n_base].
        # h_60 uses new 255-d Scheme K models → feed full vector.
        self._booster_lists: Dict[int, List[lgb.Booster]] = {}
        self._n_features_for: Dict[int, int] = {}
        for hcfg in self._horizons:
            if not hcfg.get("active", True):
                continue
            H = int(hcfg["h"])
            paths = _find_horizon_models(here, H)
            if not paths:
                continue
            self._booster_lists[H] = [lgb.Booster(model_file=p) for p in paths]
            # Probe model expected dim to slice features correctly.
            self._n_features_for[H] = self._booster_lists[H][0].num_feature()

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
        # Base 223: raw 154 + T3 69
        t3 = self._compute_t3_no_time(df)
        raw_last = df[self._raw_feat_cols].iloc[-1].to_numpy(dtype=np.float32, copy=True)
        if self._amount_delta_idx >= 0:
            v = raw_last[self._amount_delta_idx]
            raw_last[self._amount_delta_idx] = np.sign(v) * np.log1p(abs(v))
        t3_last = t3[self._t3_feat_cols].iloc[-1].to_numpy(dtype=np.float32, copy=False)
        base = np.concatenate([raw_last, t3_last]).astype(np.float32, copy=False)

        # Extra 32: ReVol + SG. ReVol needs `midprice`,`wmp_lvl1`; both are in the
        # raw input + T3 output. We assemble a tiny DF to feed the modules.
        # `midprice`, `imbalance`, `spread1` are in raw input; `wmp_lvl1` and
        # `mlofi_W60_lvl1` come from T3 output.
        sig_df = df[["midprice", "imbalance", "spread1"]].copy()
        sig_df["wmp_lvl1"] = t3["wmp_lvl1"].to_numpy()
        sig_df["mlofi_W60_lvl1"] = t3["mlofi_W60_lvl1"].to_numpy()
        revol_feats = self._revol.compute_revol_window(sig_df, REVOL_SIGNALS)  # (12,)
        sg_feats = self._savgol.compute_sg_window(sig_df, SG_SIGNALS)          # (20,)
        return np.concatenate([base, revol_feats, sg_feats]).astype(np.float32, copy=False)

    def _compute_batch_features(self, batches: List[pd.DataFrame]) -> np.ndarray:
        feats = np.empty((len(batches), self._n_total), dtype=np.float32)
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
            n_feat_h = self._n_features_for[H]
            X_h = feats if n_feat_h == self._n_total else feats[:, :n_feat_h]
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
