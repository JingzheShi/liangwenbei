"""Predictor for iter_010: same model + thresholds as iter_009 (T59 5-seed full-sym
LightGBM + DE 4D thresh on 442k full test, LOSO-equivalent +24.52), but with
**batch-vectorized feature extraction** to fit within the platform inference budget.

Feature pipeline per inference call:
  raw 154 cols (config.feature, last row, with amount_delta log1p sign-preserving)
  + T3 schemeC features (no time): mlofi(30) + wmp(11) + rv(4) + ewma_intst(24) = 69
  + Stage 1 R34 (T44): dual_z(37) + signed_rv(3) + kyle_inv(2) + ewma_ofi(12) = 54
  + Stage 2 R34 (T51): qrank(20) + rskew(3) + gofi(30) + kyle_lam(2) + vol_burst(4) = 59
  + Stage 3 R34 (T53): ewma_resid(1) + rv_ratio(3) + jshare(4) + cancel_imb(3) + roll_eff_spr(3) = 14
  --------------------------------------------------------------------------------
  total 350, then drop 10 KS-fail extras -> 340 final features.

Compliance with platform contract (CRITICAL_CONSTRAINTS.md):
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

# 10 KS-fail features dropped (3 stage1 + 6 stage2 + 1 stage3)
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
    "roll_eff_spr_ratio_W100",
)


def _load_module(here: str, fname: str, mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(here, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Predictor:
    def __init__(self) -> None:
        here = os.path.dirname(os.path.abspath(__file__))
        self._ffb = _load_module(here, "fast_features_batch.py", "iter_010_ffb")

        cfg_path = os.path.join(here, "config.json")
        with open(cfg_path) as f:
            cfg_main = json.load(f)
        self._raw_feat_cols: List[str] = list(cfg_main["feature"])
        self._raw_col_to_idx = {c: i for i, c in enumerate(self._raw_feat_cols)}
        try:
            self._amount_delta_idx = self._raw_feat_cols.index("amount_delta")
        except ValueError:
            self._amount_delta_idx = -1

        # Feature names for the extras block (T3 + S1 + S2 + S3 = 196), drop FAIL → keep
        self._extra_names: List[str] = list(self._ffb.all_feature_names())
        self._extra_keep_idx = np.array(
            [i for i, n in enumerate(self._extra_names) if n not in FAIL_NAMES],
            dtype=np.int64,
        )

        thresholds_path = os.path.join(here, "thresholds.json")
        with open(thresholds_path) as f:
            tcfg = json.load(f)
        self._horizons: List[Dict] = tcfg["horizons"]

        # Cached column_idx dict for fast lookup in compute_batch_features.
        # The batch path needs raw + 'midprice' alias; we add 'midprice' index = midprice1.
        self._col_idx: Dict[str, int] = dict(self._raw_col_to_idx)
        if "midprice1" in self._raw_col_to_idx:
            self._col_idx["midprice"] = self._raw_col_to_idx["midprice1"]

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

    def _compute_batch_features(self, batches: List[pd.DataFrame]) -> np.ndarray:
        """Stack N windows into (N, 100, K) tensor and run the batch-vectorized extractor.

        Returns (N, 340) float32 — raw_last (154, with amount_delta log-transform)
        + T3-no-time (69) + Stage1 (54) + Stage2 (59) + Stage3 (14), with 10 FAIL extras dropped.
        """
        N = len(batches)
        K = len(self._raw_feat_cols)
        X3d = np.empty((N, WINDOW, K), dtype=np.float64)
        for n, df in enumerate(batches):
            X3d[n] = df[self._raw_feat_cols].to_numpy(dtype=np.float64, copy=False)

        # raw_last (N, K) with amount_delta log1p sign-preserving
        raw_last = X3d[:, -1, :].astype(np.float32, copy=True)
        if self._amount_delta_idx >= 0:
            v = raw_last[:, self._amount_delta_idx]
            raw_last[:, self._amount_delta_idx] = np.sign(v) * np.log1p(np.abs(v))

        # T3 + S1 + S2 + S3 batch
        extras_full = self._ffb.compute_batch_features(X3d, self._col_idx)  # (N, 196) float64
        extras_full = np.where(np.isfinite(extras_full), extras_full, 0.0).astype(np.float32, copy=False)
        extras_kept = extras_full[:, self._extra_keep_idx]

        return np.concatenate([raw_last, extras_kept], axis=1)

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
