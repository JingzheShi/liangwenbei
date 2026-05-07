"""Predictor for iter_013: T74 V4 walk-forward + Stage 5 + 6 time features.

Identical pipeline to iter_012, with 6 time features appended at the end of
the feature vector (so model_h60_seed*.txt is a 365-d model = 359 schemeP-kept
+ 6 time).

Time features (sym-invariant, derived from the `time` column of the *last* row
of each window):

  1. sec_norm        : t / 2000   (session-relative)
  2. sin_sess_pos    : sin(2*pi * t / 2000)
  3. cos_sess_pos    : cos(2*pi * t / 2000)
  4. is_open_30min   : 1.0 if t < 600 else 0.0
  5. is_close_30min  : 1.0 if t > 1400 else 0.0
  6. is_pm           : 1.0 if PM session else 0.0

`t` is the tick index (0..2000) within the AM/PM session, recovered from
`time` (a `datetime.time` object passed through config.feature). The platform
keeps `time` as the actual timestamp at 3-second cadence (CRITICAL_CONSTRAINTS
§2). AM session window is 09:40:00–11:20:00, PM is 13:10:00–14:50:00.

Compliance with platform contract (CRITICAL_CONSTRAINTS.md):
  - sym / date never enter the feature vector
  - No cross-call state held on `self`
  - Every predict() call independent (time features depend only on the current
    window's last-row timestamp; no previous-row lookup)
  - sym-agnostic: time features are identical across syms for the same time
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

RAW_COLS_TRAIN_ORDER = (
    "open", "high", "low", "close", "volume_delta", "amount_delta",
    *(f"bid{k}" for k in range(1, 11)),
    *(f"bsize{k}" for k in range(1, 11)),
    *(f"ask{k}" for k in range(1, 11)),
    *(f"asize{k}" for k in range(1, 11)),
    "avgbid", "avgask", "totalbsize", "totalasize",
    "lb_intst", "la_intst", "mb_intst", "ma_intst", "cb_intst", "ca_intst",
    "lb_ind", "la_ind", "mb_ind", "ma_ind", "cb_ind", "ca_ind",
    "lb_acc", "la_acc", "mb_acc", "ma_acc", "cb_acc", "ca_acc",
    *(f"midprice{k}" for k in range(1, 11)),
    *(f"spread{k}" for k in range(1, 11)),
    *(f"bid_diff{k}" for k in range(1, 11)),
    *(f"ask_diff{k}" for k in range(1, 11)),
    "bid_mean", "ask_mean", "bsize_mean", "asize_mean", "cumspread", "imbalance",
    *(f"bid_rate{k}" for k in range(1, 11)),
    *(f"ask_rate{k}" for k in range(1, 11)),
    *(f"bsize_rate{k}" for k in range(1, 11)),
    *(f"asize_rate{k}" for k in range(1, 11)),
)
assert len(RAW_COLS_TRAIN_ORDER) == 154

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
    "liq_asym_top5_W5",
)

# Session boundaries (Chinese A-share, with 10-min crop on each side)
AM_START_SEC = 9 * 3600 + 40 * 60   # 34800
PM_START_SEC = 13 * 3600 + 10 * 60  # 47400
MID_BREAK_SEC = 12 * 3600 + 15 * 60  # 43500 (between 11:20 and 13:10)
T_MAX = 2000.0


def _load_module(here: str, fname: str, mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(here, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _time_to_t_sess(time_val) -> tuple:
    """Convert a `datetime.time`-like object to (t, sess_idx).

    Robust to NaT / None: returns (1000, 0) (mid-AM) so feature stays finite.
    """
    if time_val is None:
        return 1000, 0
    try:
        if pd.isna(time_val):
            return 1000, 0
    except (TypeError, ValueError):
        pass

    # Accept datetime.time, datetime.datetime, pandas Timestamp, or string
    if hasattr(time_val, "hour") and hasattr(time_val, "minute"):
        h = int(time_val.hour); m = int(time_val.minute)
        s = int(getattr(time_val, "second", 0) or 0)
    elif isinstance(time_val, str):
        # Expect "HH:MM:SS" or "HH:MM"
        parts = time_val.split(":")
        h = int(parts[0]); m = int(parts[1])
        s = int(parts[2]) if len(parts) > 2 else 0
    else:
        return 1000, 0

    sec_of_day = h * 3600 + m * 60 + s
    if sec_of_day < MID_BREAK_SEC:
        sess_idx = 0
        t_int = (sec_of_day - AM_START_SEC) // 3
    else:
        sess_idx = 1
        t_int = (sec_of_day - PM_START_SEC) // 3
    # Clip into [0, 2000] just in case the platform sends a slightly-shifted timestamp
    if t_int < 0:
        t_int = 0
    elif t_int > 2000:
        t_int = 2000
    return int(t_int), int(sess_idx)


def _compute_time_features(times) -> np.ndarray:
    """Return (N, 6) float32. `times` is a list/Series of N datetime.time-likes."""
    n = len(times)
    out = np.zeros((n, 6), dtype=np.float32)
    for i, tv in enumerate(times):
        t_int, sess_idx = _time_to_t_sess(tv)
        sec_norm = t_int / T_MAX
        angle = 2.0 * np.pi * sec_norm
        out[i, 0] = sec_norm
        out[i, 1] = np.sin(angle)
        out[i, 2] = np.cos(angle)
        out[i, 3] = 1.0 if t_int < 600 else 0.0
        out[i, 4] = 1.0 if t_int > 1400 else 0.0
        out[i, 5] = float(sess_idx)
    return out


class Predictor:
    def __init__(self) -> None:
        here = os.path.dirname(os.path.abspath(__file__))
        self._ffb = _load_module(here, "fast_features_batch.py", "iter_013_ffb")

        self._raw_feat_cols: List[str] = list(RAW_COLS_TRAIN_ORDER)
        self._raw_col_to_idx = {c: i for i, c in enumerate(self._raw_feat_cols)}
        self._amount_delta_idx = self._raw_feat_cols.index("amount_delta")

        self._extra_names: List[str] = list(self._ffb.all_feature_names())
        self._extra_keep_idx = np.array(
            [i for i, n in enumerate(self._extra_names) if n not in FAIL_NAMES],
            dtype=np.int64,
        )

        thresholds_path = os.path.join(here, "thresholds.json")
        with open(thresholds_path) as f:
            tcfg = json.load(f)
        self._horizons: List[Dict] = tcfg["horizons"]

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
        """Stack N windows into (N, 100, K) and run batch extractor; append time features.

        Returns (N, 365) float32 = 154 raw_last + 196 baseline extras (10 dropped)
        + 20 stage5 (1 dropped) + 6 time = 154 + 186 + 19 + 6 = 365.
        """
        N = len(batches)
        K = len(self._raw_feat_cols)
        X3d = np.empty((N, WINDOW, K), dtype=np.float64)
        last_times = []
        for n, df in enumerate(batches):
            X3d[n] = df[self._raw_feat_cols].to_numpy(dtype=np.float64, copy=False)
            last_times.append(df["time"].iloc[-1])

        raw_last = X3d[:, -1, :].astype(np.float32, copy=True)
        if self._amount_delta_idx >= 0:
            v = raw_last[:, self._amount_delta_idx]
            raw_last[:, self._amount_delta_idx] = np.sign(v) * np.log1p(np.abs(v))

        extras_full = self._ffb.compute_batch_features(X3d, self._col_idx)
        extras_full = np.where(np.isfinite(extras_full), extras_full, 0.0).astype(np.float32, copy=False)
        extras_kept = extras_full[:, self._extra_keep_idx]

        time_feats = _compute_time_features(last_times)

        return np.concatenate([raw_last, extras_kept, time_feats], axis=1)

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
    n_numeric = len(feats) - (1 if "time" in feats else 0)
    df = pd.DataFrame(
        rng.standard_normal((100, n_numeric)).astype(np.float32),
        columns=[c for c in feats if c != "time"],
    )
    if "time" in feats:
        import datetime as _dt
        df["time"] = [_dt.time(10, 0, 0)] * 100
        df = df[feats]  # reorder
    p = Predictor()
    out = p.predict([df, df, df])
    print("smoke test predict ->", out[0], "len=", len(out))
