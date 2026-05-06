"""Predictor for iter_006: NN + LightGBM ensemble at h=10, single-model
LightGBM at h=5/h=20/h=40/h=60.

Compliance with platform contract (CRITICAL_CONSTRAINTS.md §1):
  - sym / date never enter the feature vector
  - No cross-call state on `self` (per-call window only)
  - Models are sym-agnostic by construction (DeepLOB conv on 100x154; GBDT on
    154-raw + T3 derived; no sym embedding)
  - T3 features computed fresh per 100-row window for LightGBM input

h=10 ensemble strategy:
  - 5 NN models (DeepLOBReg trained per LOSO fold)
  - 1 LightGBM seed=42 model (Scheme C 223-d, full-retrained)
  - Final prob = average over (NN_avg, LGB_prob) 50/50 (matches OOF combination)

Other horizons (5, 20, 40, 60) use the iter_002/iter_003 single-model LightGBM
verbatim (with Scheme C 223-d features).

Bundle:
  config.json
  thresholds.json
  compute.py             — T3 feature compute (copied from iter_002/iter_003)
  model.py               — DeepLOBReg architecture
  model_h10_held{0..4}.pt  — 5 LOSO NN models
  model_h10_lgbm.txt     — single LightGBM h=10 (seed=42 final)
  model_h{5,20,40,60}.txt — single LightGBM other horizons (from iter_002)
  requirements.txt
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import lightgbm as lgb
import torch

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


class Predictor:
    def __init__(self) -> None:
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)

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

        # Device for NN
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- Load NN ensemble for h_10 (DeepLOBReg) ----
        from model import DeepLOBReg  # local
        self._nn_models: List[torch.nn.Module] = []
        self._nn_norm: Dict[str, np.ndarray] = {}
        for k in (0, 1, 2, 3, 4):
            pt_path = os.path.join(here, f"model_h10_held{k}.pt")
            if not os.path.isfile(pt_path):
                continue
            try:
                ckpt = torch.load(pt_path, map_location=self.device, weights_only=False)
            except TypeError:
                ckpt = torch.load(pt_path, map_location=self.device)
            meta = dict(ckpt.get("meta") or {})
            nums = meta.get("num_classes_per_head", [3, 3, 3, 3, 3])
            m = DeepLOBReg(list(nums), seq_len=int(meta.get("T", 100)),
                           num_features=len(meta.get("feature_columns", self._raw_feat_cols)))
            # Materialize LazyLinear with a dummy forward
            with torch.no_grad():
                dummy = torch.zeros(2, 1, 100, len(meta.get("feature_columns", self._raw_feat_cols)))
                m(dummy)
            m.load_state_dict(ckpt["model_state"], strict=True)
            m.to(self.device).eval()
            self._nn_models.append(m)
            # Stats: same across all folds (single global stats)
            if not self._nn_norm:
                self._nn_norm["mean"] = np.asarray(meta["mean"], dtype=np.float32)
                self._nn_norm["std"] = np.asarray(meta["std"], dtype=np.float32)
                # Index of log1p cols in the *NN feature_columns* (same as raw_feat_cols here)
                fc = list(meta.get("feature_columns", self._raw_feat_cols))
                log1p_cols = list(meta.get("log1p_cols", ["amount_delta"]))
                self._nn_norm["log1p_idx"] = np.array(
                    [fc.index(c) for c in log1p_cols if c in fc], dtype=np.int64
                )
                self._nn_feat_cols = fc

        # ---- Load LightGBM models ----
        self._boosters: Dict[int, List[lgb.Booster]] = {}
        for hcfg in self._horizons:
            if not hcfg.get("active", True):
                continue
            H = int(hcfg["h"])
            ms: List[lgb.Booster] = []
            mp = os.path.join(here, f"model_h{H}.txt") if H != 10 else os.path.join(here, "model_h10_lgbm.txt")
            if os.path.isfile(mp):
                ms.append(lgb.Booster(model_file=mp))
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

    def _compute_window_features_lgb(self, df: pd.DataFrame) -> np.ndarray:
        """Last-tick LGBM features (raw + T3): 154 + 69 = 223-d."""
        t3 = self._compute_t3_no_time(df)
        raw_last = df[self._raw_feat_cols].iloc[-1].to_numpy(dtype=np.float32, copy=True)
        if self._amount_delta_idx >= 0:
            v = raw_last[self._amount_delta_idx]
            raw_last[self._amount_delta_idx] = np.sign(v) * np.log1p(abs(v))
        t3_last = t3[self._t3_feat_cols].iloc[-1].to_numpy(dtype=np.float32, copy=False)
        return np.concatenate([raw_last, t3_last]).astype(np.float32, copy=False)

    def _compute_batch_features_lgb(self, batches: List[pd.DataFrame]) -> np.ndarray:
        feats = np.empty(
            (len(batches), len(self._raw_feat_cols) + len(self._t3_feat_cols)),
            dtype=np.float32,
        )
        for i, df in enumerate(batches):
            feats[i] = self._compute_window_features_lgb(df)
        return feats

    def _compute_batch_features_nn(self, batches: List[pd.DataFrame]) -> np.ndarray:
        """100x154 normalized time series for NN input."""
        B = len(batches)
        F = len(self._nn_feat_cols)
        out = np.empty((B, WINDOW, F), dtype=np.float32)
        mean = self._nn_norm["mean"]
        std = self._nn_norm["std"]
        log1p_idx = self._nn_norm["log1p_idx"]
        for i, df in enumerate(batches):
            x = df[self._nn_feat_cols].to_numpy(dtype=np.float32, copy=True)
            if x.shape[0] < WINDOW:
                # left-pad with first row
                pad = np.repeat(x[:1], WINDOW - x.shape[0], axis=0)
                x = np.concatenate([pad, x], axis=0)
            elif x.shape[0] > WINDOW:
                x = x[-WINDOW:]
            x[:, log1p_idx] = np.log1p(x[:, log1p_idx])
            x -= mean
            x /= std
            bad = ~np.isfinite(x)
            if bad.any():
                x[bad] = 0.0
            out[i] = x
        return out

    @torch.no_grad()
    def _nn_predict_h10(self, batches: List[pd.DataFrame]) -> np.ndarray:
        """Average prob_0/1/2 across loaded NN models, return (B, 3)."""
        if not self._nn_models:
            B = len(batches)
            # No NN — return uniform
            return np.full((B, 3), 1.0 / 3.0, dtype=np.float32)
        x_np = self._compute_batch_features_nn(batches)
        x = torch.from_numpy(x_np).unsqueeze(1).to(self.device, dtype=torch.float32)  # (B, 1, T, F)
        prob_sum = torch.zeros(x.shape[0], 3, device=self.device, dtype=torch.float32)
        h10_idx = HORIZON_TO_IDX[10]
        for m in self._nn_models:
            heads = m(x)  # tuple
            prob_sum += torch.softmax(heads[h10_idx], dim=1)
        prob_avg = (prob_sum / float(len(self._nn_models))).cpu().numpy().astype(np.float32)
        return prob_avg

    def predict(self, batches: List[pd.DataFrame]) -> List[List[int]]:
        if not batches:
            return []
        feats_lgb = self._compute_batch_features_lgb(batches)
        B = feats_lgb.shape[0]
        out = np.ones((B, 5), dtype=np.int64)

        # NN h=10 probs (computed once)
        nn_probs_h10 = self._nn_predict_h10(batches) if 10 in self._boosters else None

        for hcfg in self._horizons:
            if not hcfg.get("active", True):
                continue
            H = int(hcfg["h"])
            ms = self._boosters.get(H)
            if not ms:
                continue
            prob_sum = np.zeros((B, 3), dtype=np.float32)
            for booster in ms:
                prob_sum += booster.predict(feats_lgb).astype(np.float32)
            lgb_probs = prob_sum / float(len(ms))

            if H == 10 and nn_probs_h10 is not None:
                # 50/50 NN-avg vs LGBM
                probs = 0.5 * nn_probs_h10 + 0.5 * lgb_probs
            else:
                probs = lgb_probs
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
