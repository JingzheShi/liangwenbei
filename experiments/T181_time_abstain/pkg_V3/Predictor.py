"""Predictor for T181 V3 close-only time-window abstain wrapper on top of T170.

T170 = iter_018 v1 (T127): iter_015 v1 (T87 NN + T75 LGB) + per-sym conformal abstain.

T181 adds an OUT-OF-MODEL time abstain layer:
  - Read 'time' column from raw input DataFrame (last tick of each window)
  - Parse 'HH:MM:SS' string to (sess_idx, session_minute)
  - If session_minute falls in an abstain window -> action = 1 (hold) for ALL horizons
  - Otherwise: standard T170 pipeline (gate_asym + conformal band)

V1 conservative: abstain first 10 min AND last 10 min of each session:
  AM: 9:40-9:50 (session_minute in [0, 10])  and  11:10-11:20 (session_minute in [90, 100])
  PM: 13:10-13:20 (session_minute in [0, 10]) and  14:40-14:50 (session_minute in [90, 100])

CRITICAL COMPLIANCE (CRITICAL_CONSTRAINTS.md):
  - 'time' used ONLY for out-of-model abstain mask. NOT fed to LGB/NN forward.
  - 'date' never used (still zero'd by platform, still not used here).
  - No cross-call state on self.
  - sym used only for per-sym beta lookup (same as T170).
  - W <= 100 for all rolling features.
"""
from __future__ import annotations

import importlib.util
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb

# ==== TIME ABSTAIN CONFIG (V1: conservative) ====
# Each window: (sess_idx, lo_session_minute, hi_session_minute)  [inclusive]
# session_minute = seconds_from_session_start / 60.0
# AM session start: 09:40:00   PM session start: 13:10:00
ABSTAIN_WINDOWS: List[Tuple[int, float, float]] = [
    (0, 95.0, 100.0),  # AM last 5 min  (11:15-11:20)
    (1, 95.0, 100.0),  # PM last 5 min  (14:45-14:50)
]

_AM_START_SEC = 9 * 3600 + 40 * 60   # 34800
_PM_START_SEC = 13 * 3600 + 10 * 60  # 47400


def _parse_session_minute(time_val: object) -> Tuple[int, float]:
    """Parse 'HH:MM:SS' string -> (sess_idx, session_minute_float).
    Returns (None, None) on any parse failure (graceful no-abstain fallback)."""
    try:
        s = str(time_val).strip()
        h = int(s[0:2])
        m = int(s[3:5])
        sc = int(s[6:8])
        total = h * 3600 + m * 60 + sc
        if h < 12:
            return 0, (total - _AM_START_SEC) / 60.0
        else:
            return 1, (total - _PM_START_SEC) / 60.0
    except Exception:
        return None, None


def _build_abstain_mask(batches: List[pd.DataFrame], windows: List[Tuple]) -> np.ndarray:
    """Return bool array (B,): True where sample should be hard-abstained (action=1)."""
    B = len(batches)
    mask = np.zeros(B, dtype=bool)
    if not windows:
        return mask
    for i, df in enumerate(batches):
        if "time" not in df.columns:
            continue
        try:
            time_val = df["time"].iloc[-1]
        except Exception:
            continue
        sess, mins = _parse_session_minute(time_val)
        if sess is None:
            continue
        for (w_sess, lo, hi) in windows:
            if sess == w_sess and lo <= mins <= hi:
                mask[i] = True
                break
    return mask


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
    "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
    "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
    "qrank_W100_cumspread",
    "kyle_lam_W50", "kyle_lam_W100",
    "roll_eff_spr_ratio_W100",
    "liq_asym_top5_W5",
)


def _gelu_tanh(x: np.ndarray) -> np.ndarray:
    c = 0.7978845608028654
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x * x * x)))


def _layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


class _MLPNumpy:
    def __init__(self, npz_path: str):
        d = np.load(npz_path, allow_pickle=False)
        self.in_dim = int(d["in_dim"][0])
        self.hidden = list(d["hidden"].tolist())
        self.use_layernorm = bool(d["use_layernorm"][0])
        self.target_scale = float(d["target_scale"][0])
        self.feat_mean = d["feat_mean"].astype(np.float32)
        self.feat_std = d["feat_std"].astype(np.float32)
        self.keep_idx = d["keep_idx"].astype(np.int64)
        self.clip = float(d["clip"][0])
        self.W = []
        self.b = []
        self.LN_W = []
        self.LN_b = []
        for i in range(len(self.hidden)):
            self.W.append(d[f"L{i}_W"].astype(np.float32))
            self.b.append(d[f"L{i}_b"].astype(np.float32))
            if self.use_layernorm:
                self.LN_W.append(d[f"LN{i}_W"].astype(np.float32))
                self.LN_b.append(d[f"LN{i}_b"].astype(np.float32))
        self.WF = d["LF_W"].astype(np.float32)
        self.bF = d["LF_b"].astype(np.float32)

    def predict(self, X_in: np.ndarray) -> np.ndarray:
        Xs = X_in
        if Xs.shape[1] != self.in_dim:
            Xs = Xs[:, self.keep_idx]
        Xs = Xs.astype(np.float32, copy=True)
        Xs = (Xs - self.feat_mean) / self.feat_std
        Xs = np.where(np.isnan(Xs), 0.0, Xs)
        Xs = np.clip(Xs, -self.clip, self.clip).astype(np.float32, copy=False)
        h = Xs
        for i in range(len(self.hidden)):
            h = h @ self.W[i].T + self.b[i]
            if self.use_layernorm:
                h = _layernorm(h, self.LN_W[i], self.LN_b[i])
            h = _gelu_tanh(h)
        out = h @ self.WF.T + self.bF
        out = out.squeeze(-1) / self.target_scale
        return out.astype(np.float32, copy=False)


def _load_module(here: str, fname: str, mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(here, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Predictor:
    def __init__(self) -> None:
        here = os.path.dirname(os.path.abspath(__file__))
        self._ffb = _load_module(here, "fast_features_batch.py", "iter_018_ffb")

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

        cw = tcfg.get("conformal_wrapper", {"enabled": False})
        self._cw_enabled = bool(cw.get("enabled", False))
        self._cw_band: Dict[int, float] = {}
        self._cw_default_band = 0.0
        if self._cw_enabled:
            psb = cw.get("per_sym_beta", {})
            pss = cw.get("per_sym_sigma", {})
            for k_str, b in psb.items():
                k = int(k_str)
                s = float(pss.get(k_str, 0.0))
                self._cw_band[k] = float(b) * s
            self._cw_default_band = float(cw.get("default_beta_for_ood", 0.16)) * \
                                     float(cw.get("default_sigma_for_ood", 4.0e-4))

        self._col_idx: Dict[str, int] = dict(self._raw_col_to_idx)
        if "midprice1" in self._raw_col_to_idx:
            self._col_idx["midprice"] = self._raw_col_to_idx["midprice1"]

        self._lgb_lists: Dict[int, List[lgb.Booster]] = {}
        self._nn_lists: Dict[int, List[_MLPNumpy]] = {}
        self._weights: Dict[int, Tuple[float, float]] = {}
        for hcfg in self._horizons:
            if not hcfg.get("active", True):
                continue
            H = int(hcfg["h"])
            seeds = hcfg.get("ensemble_seeds", [])
            lgb_paths: List[str] = []
            nn_paths: List[str] = []
            for s in seeds:
                lp = os.path.join(here, f"model_h{H}_seed{s}.txt")
                np_ = os.path.join(here, f"nn_h{H}_seed{s}.npz")
                if os.path.isfile(lp):
                    lgb_paths.append(lp)
                if os.path.isfile(np_):
                    nn_paths.append(np_)
            if lgb_paths:
                self._lgb_lists[H] = [lgb.Booster(model_file=p) for p in lgb_paths]
            if nn_paths:
                self._nn_lists[H] = [_MLPNumpy(p) for p in nn_paths]
            self._weights[H] = (float(hcfg.get("w_nn", 1.0)),
                                float(hcfg.get("w_lgb", 1.0)))

    def _gate_with_band(self, pred_dmid: np.ndarray, hcfg: Dict,
                        band_per_row: np.ndarray) -> np.ndarray:
        thr_up = float(hcfg.get("thr_up", 2.0e-4))
        thr_dn = float(hcfg.get("thr_dn", 2.0e-4))
        out = np.full(pred_dmid.shape[0], 1, dtype=np.int64)
        out[pred_dmid > (thr_up + band_per_row)] = 2
        out[pred_dmid < -(thr_dn + band_per_row)] = 0
        return out

    @staticmethod
    def _ev_gate_predict(pred_dmid: np.ndarray, hcfg: Dict) -> np.ndarray:
        thr_up = float(hcfg.get("thr_up", 2.0e-4))
        thr_dn = float(hcfg.get("thr_dn", 2.0e-4))
        out = np.full(pred_dmid.shape[0], 1, dtype=np.int64)
        out[pred_dmid > thr_up] = 2
        out[pred_dmid < -thr_dn] = 0
        return out

    def _compute_batch_features(self, batches: List[pd.DataFrame]) -> np.ndarray:
        N = len(batches)
        K = len(self._raw_feat_cols)
        X3d = np.empty((N, WINDOW, K), dtype=np.float64)
        for n, df in enumerate(batches):
            X3d[n] = df[self._raw_feat_cols].to_numpy(dtype=np.float64, copy=False)

        raw_last = X3d[:, -1, :].astype(np.float32, copy=True)
        if self._amount_delta_idx >= 0:
            v = raw_last[:, self._amount_delta_idx]
            raw_last[:, self._amount_delta_idx] = np.sign(v) * np.log1p(np.abs(v))

        extras_full = self._ffb.compute_batch_features(X3d, self._col_idx)
        extras_full = np.where(np.isfinite(extras_full), extras_full, 0.0).astype(np.float32, copy=False)
        extras_kept = extras_full[:, self._extra_keep_idx]

        return np.concatenate([raw_last, extras_kept], axis=1)

    def _extract_band_per_row(self, batches: List[pd.DataFrame]) -> np.ndarray:
        B = len(batches)
        out = np.full(B, self._cw_default_band, dtype=np.float64)
        for i, df in enumerate(batches):
            if "sym" not in df.columns:
                continue
            try:
                s = int(df["sym"].iloc[-1])
            except (ValueError, TypeError, IndexError):
                continue
            if s in self._cw_band:
                out[i] = self._cw_band[s]
        return out

    @staticmethod
    def _ensemble_predict_lgb(boosters: List[lgb.Booster], X: np.ndarray) -> np.ndarray:
        if len(boosters) == 1:
            return boosters[0].predict(X).astype(np.float32, copy=False)
        acc = None
        for b in boosters:
            p = b.predict(X).astype(np.float32)
            acc = p if acc is None else acc + p
        return acc / float(len(boosters))

    @staticmethod
    def _ensemble_predict_nn(nns: List[_MLPNumpy], X: np.ndarray) -> np.ndarray:
        if len(nns) == 1:
            return nns[0].predict(X)
        acc = None
        for n in nns:
            p = n.predict(X)
            acc = p if acc is None else acc + p
        return acc / float(len(nns))

    def predict(self, batches: List[pd.DataFrame]) -> List[List[int]]:
        if not batches:
            return []

        # Time abstain mask (computed BEFORE model inference, uses only 'time' column)
        abstain_mask = _build_abstain_mask(batches, ABSTAIN_WINDOWS)

        feats = self._compute_batch_features(batches)
        B = feats.shape[0]
        out = np.ones((B, 5), dtype=np.int64)

        if self._cw_enabled:
            band_per_row = self._extract_band_per_row(batches)
        else:
            band_per_row = np.zeros(B, dtype=np.float64)

        for hcfg in self._horizons:
            if not hcfg.get("active", True):
                continue
            H = int(hcfg["h"])
            lgbs = self._lgb_lists.get(H)
            nns = self._nn_lists.get(H)
            w_nn, w_lgb = self._weights.get(H, (1.0, 1.0))

            preds = []
            ws = []
            if lgbs and w_lgb > 0:
                preds.append(self._ensemble_predict_lgb(lgbs, feats))
                ws.append(w_lgb)
            if nns and w_nn > 0:
                preds.append(self._ensemble_predict_nn(nns, feats))
                ws.append(w_nn)
            if not preds:
                continue
            stacked = np.stack(preds, axis=0)
            ws_arr = np.array(ws, dtype=np.float32).reshape(-1, 1)
            pred_dmid = (stacked * ws_arr).sum(axis=0) / ws_arr.sum()

            if self._cw_enabled:
                actions = self._gate_with_band(pred_dmid, hcfg, band_per_row)
            else:
                actions = self._ev_gate_predict(pred_dmid, hcfg)
            out[:, HORIZON_TO_IDX[H]] = actions

        # Hard override: abstained rows -> action=1 for all horizons
        if abstain_mask.any():
            out[abstain_mask, :] = 1

        return out.tolist()


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    cfg = json.load(open(os.path.join(here, "config.json")))
    feats = cfg["feature"]
    rng = np.random.default_rng(0)
    # Use string time column for smoke test
    df_data = rng.standard_normal((100, len(feats))).astype(np.float32)
    df = pd.DataFrame(df_data, columns=feats)
    df["sym"] = 1
    df["time"] = "10:30:00"  # mid-session, should NOT abstain
    p = Predictor()
    out = p.predict([df, df, df])
    print("smoke test predict ->", out[0], "len=", len(out))
    df_abstain = df.copy()
    df_abstain["time"] = "11:18:00"  # AM last 5 min, SHOULD abstain
    out2 = p.predict([df_abstain])
    print("abstained (11:18 AM) ->", out2[0], "(expect [1,1,1,1,1])")
    df_ood = df.copy()
    df_ood["sym"] = 99
    out_ood = p.predict([df_ood])
    print("OOD sym=99 ->", out_ood[0])
