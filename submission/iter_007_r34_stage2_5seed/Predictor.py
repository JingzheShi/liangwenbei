"""Predictor for iter_007: T51 R34 stage 2 features (327-d), 5-seed aug_a ensemble for h_60
+ DE 4D asymmetric (T_up,T_dn,d_up,d_dn) gating.

h_5/h_10/h_20: inactive (matches iter_006).
h_40: kept from iter_006 (single iter_002 model, trained on 223-d base = 154 raw + 69 T3-no-time).
h_60: 5-seed ensemble of T51 stage 2 (327-d) + DE 4D asymmetric thresh.

Compliance with platform contract (CRITICAL_CONSTRAINTS.md §1):
  - sym / date / time never enter the feature vector
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


def _load_module(here: str, name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(here, fname))
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
        # T3 (72 cols incl. time_)
        self._t3 = _load_module(here, "iter_007_t3_compute", "compute.py")
        all_t3_cols = list(self._t3.feature_v1_columns())
        self._t3_feat_cols: List[str] = [c for c in all_t3_cols if not c.startswith("time_")]
        # Stage1 (54) and Stage2 (59) both session-scoped, computed at the last tick.
        self._stage1 = _load_module(here, "iter_007_stage1", "r34_features.py")
        self._stage2 = _load_module(here, "iter_007_stage2", "r34_stage2_features.py")
        self._stage1_names: List[str] = list(self._stage1.all_feature_names())
        self._stage2_names: List[str] = list(self._stage2.all_feature_names())

        cfg_path = os.path.join(here, "config.json")
        with open(cfg_path) as f:
            cfg_main = json.load(f)
        self._raw_feat_cols: List[str] = list(cfg_main["feature"])
        try:
            self._amount_delta_idx = self._raw_feat_cols.index("amount_delta")
        except ValueError:
            self._amount_delta_idx = -1

        # Drop indices: 9 FAIL extras (in stage1+stage2 namespace) — same as training.
        # stage1: 34, 35, 36 (dualz_ask_diff1, dualz_bid_diff5, dualz_ask_diff5)
        # stage2: 0, 1, 2, 3 (qrank_W100_spread1/spread5/spread10/cumspread), 53, 54 (kyle_lam_W50, kyle_lam_W100)
        s1_drop = ["dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5"]
        s2_drop = [
            "qrank_W100_spread1", "qrank_W100_spread5",
            "qrank_W100_spread10", "qrank_W100_cumspread",
            "kyle_lam_W50", "kyle_lam_W100",
        ]
        self._stage1_keep_idx = np.array(
            [i for i, n in enumerate(self._stage1_names) if n not in set(s1_drop)],
            dtype=np.int64,
        )
        self._stage2_keep_idx = np.array(
            [i for i, n in enumerate(self._stage2_names) if n not in set(s2_drop)],
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
            paths = _find_horizon_models(here, H)
            if not paths:
                continue
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

    def _compute_window_features(self, df: pd.DataFrame):
        """Return (feat_223, feat_327) for one 100-tick window.
        feat_223 used by h_40 model (legacy iter_006 base);
        feat_327 used by h_60 model (T51 stage 2).
        """
        n = len(df)
        valid_lo = n - 1
        valid_hi = n - 1

        # Base 223: 154 raw + 69 t3-no-time
        t3 = self._compute_t3_no_time(df)
        raw_last = df[self._raw_feat_cols].iloc[-1].to_numpy(dtype=np.float32, copy=True)
        if self._amount_delta_idx >= 0:
            v = raw_last[self._amount_delta_idx]
            raw_last[self._amount_delta_idx] = np.sign(v) * np.log1p(abs(v))
        t3_last = t3[self._t3_feat_cols].iloc[-1].to_numpy(dtype=np.float32, copy=False)
        feat_223 = np.concatenate([raw_last, t3_last]).astype(np.float32, copy=False)

        # Stage1 / Stage2 modules need 'midprice' (platform does not pass it) and
        # 'mlofi_W20_lvl{1,5,10}' (T3-derived, used by stage1 EWMA-OFI).
        # midprice = (bid1 + ask1) / 2 verified bit-exact on training parquets.
        ext_cols = {}
        if "midprice" not in df.columns:
            ext_cols["midprice"] = (df["bid1"].to_numpy() + df["ask1"].to_numpy()) * 0.5
        for k in (1, 5, 10):
            name = f"mlofi_W20_lvl{k}"
            if name not in df.columns:
                ext_cols[name] = t3[name].to_numpy()
        if ext_cols:
            df = df.assign(**ext_cols)

        # Stage1 (54) + Stage2 (59) at the last tick of the window.
        s1_full = self._stage1.compute_all_session(df, valid_lo, valid_hi)  # (1, 54)
        s2_full = self._stage2.compute_all_session(df, valid_lo, valid_hi)  # (1, 59)
        s1_kept = s1_full[0, self._stage1_keep_idx]  # (51,)
        s2_kept = s2_full[0, self._stage2_keep_idx]  # (53,)
        # Sanitize (training also replaces non-finite with 0)
        s1_kept = np.where(np.isfinite(s1_kept), s1_kept, 0.0).astype(np.float32, copy=False)
        s2_kept = np.where(np.isfinite(s2_kept), s2_kept, 0.0).astype(np.float32, copy=False)
        feat_327 = np.concatenate([feat_223, s1_kept, s2_kept]).astype(np.float32, copy=False)
        return feat_223, feat_327

    def _compute_batch_features(self, batches: List[pd.DataFrame]):
        B = len(batches)
        F223 = len(self._raw_feat_cols) + len(self._t3_feat_cols)
        F327 = F223 + len(self._stage1_keep_idx) + len(self._stage2_keep_idx)
        feats_223 = np.empty((B, F223), dtype=np.float32)
        feats_327 = np.empty((B, F327), dtype=np.float32)
        for i, df in enumerate(batches):
            f223, f327 = self._compute_window_features(df)
            feats_223[i] = f223
            feats_327[i] = f327
        return feats_223, feats_327

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
        feats_223, feats_327 = self._compute_batch_features(batches)
        B = feats_223.shape[0]
        out = np.ones((B, 5), dtype=np.int64)
        for hcfg in self._horizons:
            if not hcfg.get("active", True):
                continue
            H = int(hcfg["h"])
            boosters = self._booster_lists.get(H)
            if not boosters:
                continue
            X = feats_327 if H == 60 else feats_223
            probs = self._ensemble_predict(boosters, X)
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
