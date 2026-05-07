"""Predictor for iter_007: T51 R34 Stage 1+2 features (327-d) with 5-seed full-data
LightGBM ensemble for h_60, plus T30 asymmetric (T_up,T_dn,d_up,d_dn) DE-optimized
gating tuned on T51 5-seed LOSO OOF.

Feature pipeline per inference call:
  raw 154 cols (config.feature, last row)
  + T3 schemeC features (no time): mlofi(30) + wmp(11) + rv(4) + ewma_intst(24) = 69
  + Stage 1 R34 (T44): dual_z(37) + signed_rv(3) + kyle_inv(2) + ewma_ofi(12) = 54
  + Stage 2 R34 (T51): qrank(20) + rskew(3) + gofi(30) + kyle_lam(2) + vol_burst(4) = 59
  --------------------------------------------------------------------------------
  total 336, then drop 9 FAIL extras → 327 final features (matches model.feature_name())

Compliance with platform contract (CRITICAL_CONSTRAINTS.md §1):
  - sym / date never enter the feature vector
  - No cross-call state held on `self` (each predict() call is independent)
  - sym-agnostic: features only use 100-tick LOB window, no sym-specific normalization
  - W <= 100 for every rolling: all features computable from 100-tick window
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

FAIL_NAMES = (
    "dualz_ask_diff1",
    "dualz_bid_diff5",
    "dualz_ask_diff5",
    "qrank_W100_spread1",
    "qrank_W100_spread5",
    "qrank_W100_spread10",
    "qrank_W100_cumspread",
    "kyle_lam_W50",
    "kyle_lam_W100",
)


def _load_module(here: str, fname: str, mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(here, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Predictor:
    def __init__(self) -> None:
        here = os.path.dirname(os.path.abspath(__file__))
        self._t3 = _load_module(here, "compute.py", "iter_007_t3_compute")
        self._s1 = _load_module(here, "r34_features.py", "iter_007_r34_s1")
        self._s2 = _load_module(here, "r34_stage2_features.py", "iter_007_r34_s2")

        all_t3_cols = list(self._t3.feature_v1_columns())
        self._t3_no_time_cols: List[str] = [c for c in all_t3_cols if not c.startswith("time_")]

        cfg_path = os.path.join(here, "config.json")
        with open(cfg_path) as f:
            cfg_main = json.load(f)
        self._raw_feat_cols: List[str] = list(cfg_main["feature"])
        try:
            self._amount_delta_idx = self._raw_feat_cols.index("amount_delta")
        except ValueError:
            self._amount_delta_idx = -1

        s1_names = list(self._s1.all_feature_names())
        s2_names = list(self._s2.all_feature_names())
        full_extra_names = s1_names + s2_names
        self._extra_keep_idx = np.array(
            [i for i, n in enumerate(full_extra_names) if n not in FAIL_NAMES],
            dtype=np.int64,
        )

        thresholds_path = os.path.join(here, "thresholds.json")
        with open(thresholds_path) as f:
            tcfg = json.load(f)
        self._horizons: List[Dict] = tcfg["horizons"]

        self._booster_lists: Dict[int, List[lgb.Booster]] = {}
        for hcfg in self._horizons:
            if not hcfg.get("active", True):
                continue
            H = int(hcfg["h"])
            seeds = hcfg.get("ensemble_seeds")
            paths: List[str] = []
            if seeds:
                for s in seeds:
                    p = os.path.join(here, f"model_h{H}_seed{s}.txt")
                    if os.path.isfile(p):
                        paths.append(p)
            if not paths:
                p = os.path.join(here, f"model_h{H}.txt")
                if os.path.isfile(p):
                    paths.append(p)
            if paths:
                self._booster_lists[H] = [lgb.Booster(model_file=p) for p in paths]

    @staticmethod
    def _threshold_predict(probs: np.ndarray, hcfg: Dict) -> np.ndarray:
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
        T = len(df)
        last = T - 1

        raw_last = df[self._raw_feat_cols].iloc[-1].to_numpy(dtype=np.float32, copy=True)
        if self._amount_delta_idx >= 0:
            v = raw_last[self._amount_delta_idx]
            raw_last[self._amount_delta_idx] = np.sign(v) * np.log1p(abs(v))

        # T3 schemeC features: mlofi(30), wmp(11), rv(4), ewma_intst(24) = 69
        t3 = self._compute_t3_no_time(df)
        t3_last = t3[self._t3_no_time_cols].iloc[-1].to_numpy(dtype=np.float32, copy=False)

        # R34 stage1/2 expect df["midprice"] (= midprice1 = (bid1+ask1)/2) and
        # df["mlofi_W20_lvl_k"] (computed by T3 above). Platform input has only the 154
        # raw cols, so we attach the needed derived columns onto a working df.
        derived = {"midprice": df["midprice1"].to_numpy(dtype=np.float64, copy=False)}
        for k in (1, 5, 10):
            col = f"mlofi_W20_lvl{k}"
            derived[col] = t3[col].to_numpy(dtype=np.float64, copy=False)
        df_ext = df.assign(**derived)

        s1 = self._s1.compute_all_session(df_ext, last, last)
        s2 = self._s2.compute_all_session(df_ext, last, last)
        extras = np.concatenate([s1, s2], axis=1).reshape(-1)
        extras = np.where(np.isfinite(extras), extras, 0.0).astype(np.float32, copy=False)
        extras_kept = extras[self._extra_keep_idx]

        return np.concatenate([raw_last, t3_last, extras_kept]).astype(np.float32, copy=False)

    def _compute_batch_features(self, batches: List[pd.DataFrame]) -> np.ndarray:
        n_dim = len(self._raw_feat_cols) + len(self._t3_no_time_cols) + len(self._extra_keep_idx)
        feats = np.empty((len(batches), n_dim), dtype=np.float32)
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
