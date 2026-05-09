"""Predictor for iter_019 v11 (T154): v2 + pruned LOB (259) + 5 TOD = 264 for LGB.

iter_019 v11 = iter_019 v2 compound:
  - 259 LOB features (T147 adversarial-filtered from 359, same as v10)
  - 5 TOD features appended (same as v9b)
  → LGB input: 264-d (259 LOB + 5 TOD)
  → NN input:  359-d (unchanged, byte-identical to iter_015 v1 / v2)

  pred_lgb = mean over 5 LightGBM regression boosters (264-d input)
  pred_nn  = mean over 5 small MLPs (359-d input, byte-identical to iter_015 v1 / v2)
  pred     = (w_nn * pred_nn + w_lgb * pred_lgb) / (w_nn + w_lgb), w_nn=1.0, w_lgb=1.5

TOD features (appended last, positions 259-263 in LGB input):
  1. tod_sin       sin(2π * mins_full / 200)
  2. tod_cos       cos(2π * mins_full / 200)
  3. is_first_20   1.0 if mins_into_session <= 20
  4. is_last_20    1.0 if mins_into_session >= 80
  5. is_lunch_prox 1.0 if 90 <= mins_full <= 110

Conformal wrapper (iter_018): per-sym beta × sigma band on top of DE-tuned thresholds.
Thresholds: v2 clean (w_lgb=1.5, per_sym_beta_1=0.40).

Compliance with CRITICAL_CONSTRAINTS.md:
  - sym read only for per-sym beta lookup (not fed to model.forward)
  - time read only for TOD feature extraction (not fed to model; stateless)
  - date never used
  - No cross-call state held on `self`
  - sym-agnostic FORWARD: NN/LGB features only use 100-tick LOB window
  - W <= 100 for every rolling feature
  - Stateless / shuffle-invariant
"""
from __future__ import annotations

import datetime
import importlib.util
import json
import os
from typing import Dict, List, Tuple

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
    "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
    "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
    "qrank_W100_cumspread",
    "kyle_lam_W50", "kyle_lam_W100",
    "roll_eff_spr_ratio_W100",
    "liq_asym_top5_W5",
)

# AM session: 9:40-11:20, PM session: 13:10-14:50 (each 100 minutes)
_AM_START_SEC = 9 * 3600 + 40 * 60   # 34800
_PM_START_SEC = 13 * 3600 + 10 * 60  # 47400
_PM_THRESH_SEC = 13 * 3600            # 46800


def _gelu_tanh(x: np.ndarray) -> np.ndarray:
    c = 0.7978845608028654  # sqrt(2/pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x * x * x)))


def _layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


class _MLPNumpy:
    """MLP inference in numpy: Linear -> LayerNorm -> GELU -> ... -> Linear -> / target_scale."""

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


def _time_to_secs(t_val) -> int:
    """Convert platform time value to total seconds since midnight."""
    if isinstance(t_val, datetime.time):
        return t_val.hour * 3600 + t_val.minute * 60 + t_val.second
    if isinstance(t_val, (int, float)):
        return int(t_val)
    s = str(t_val).strip()
    parts = s.split(":")
    if len(parts) >= 2:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + (int(parts[2]) if len(parts) > 2 else 0)
    return int(float(s))


class Predictor:
    def __init__(self) -> None:
        here = os.path.dirname(os.path.abspath(__file__))
        self._ffb = _load_module(here, "fast_features_batch.py", "iter_019v11_ffb")

        self._raw_feat_cols: List[str] = list(RAW_COLS_TRAIN_ORDER)
        self._raw_col_to_idx = {c: i for i, c in enumerate(self._raw_feat_cols)}
        self._amount_delta_idx = self._raw_feat_cols.index("amount_delta")

        self._extra_names: List[str] = list(self._ffb.all_feature_names())
        self._extra_keep_idx = np.array(
            [i for i, n in enumerate(self._extra_names) if n not in FAIL_NAMES],
            dtype=np.int64,
        )

        # Pruning: keep 259 out of 359 LOB features (T147 adversarial filtering)
        pruned_json_path = os.path.join(here, "pruned_features.json")
        with open(pruned_json_path) as f:
            _pruned_info = json.load(f)
        self._lob_kept_idx: np.ndarray = np.array(
            _pruned_info["pruned_idx_in_359"], dtype=np.int64
        )  # 259 indices into the 359-feature space

        thresholds_path = os.path.join(here, "thresholds.json")
        with open(thresholds_path) as f:
            tcfg = json.load(f)
        self._horizons: List[Dict] = tcfg["horizons"]

        # Conformal wrapper config
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

        # Load per-horizon LGB (264-d) and NN (359-d) ensembles
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

    def _compute_tod_features(self, batches: List[pd.DataFrame]) -> np.ndarray:
        """Compute 5 TOD features for each window using the last tick's time."""
        N = len(batches)
        tod = np.zeros((N, 5), dtype=np.float32)
        for i, df in enumerate(batches):
            try:
                t_val = df["time"].iloc[-1]
                total_secs = _time_to_secs(t_val)
            except Exception:
                total_secs = _AM_START_SEC  # safe fallback: start of AM

            if total_secs < _PM_THRESH_SEC:
                sess_idx = 0.0
                secs_into = float(total_secs - _AM_START_SEC)
            else:
                sess_idx = 1.0
                secs_into = float(total_secs - _PM_START_SEC)

            mins_into = max(0.0, secs_into / 60.0)
            mins_full = mins_into + 100.0 * sess_idx

            tod[i, 0] = np.sin(2.0 * np.pi * mins_full / 200.0)
            tod[i, 1] = np.cos(2.0 * np.pi * mins_full / 200.0)
            tod[i, 2] = 1.0 if mins_into <= 20.0 else 0.0
            tod[i, 3] = 1.0 if mins_into >= 80.0 else 0.0
            tod[i, 4] = 1.0 if 90.0 <= mins_full <= 110.0 else 0.0
        return tod

    def _compute_batch_features(self, batches: List[pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute features for LGB (264-d) and NN (359-d).

        Returns:
          feats_lgb: (N, 264) — 259 pruned LOB + 5 TOD
          feats_nn:  (N, 359) — 154 raw + 205 extras (same as v2)
        """
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

        feats_359 = np.concatenate([raw_last, extras_kept], axis=1)  # (N, 359)

        # Prune 359 → 259 LOB features
        feats_259 = feats_359[:, self._lob_kept_idx]  # (N, 259)

        # Append 5 TOD features → 264 for LGB
        tod = self._compute_tod_features(batches)  # (N, 5)
        feats_264 = np.concatenate([feats_259, tod], axis=1)  # (N, 264)

        return feats_264, feats_359

    def _extract_band_per_row(self, batches: List[pd.DataFrame]) -> np.ndarray:
        """Look up beta * sigma per row using sym from the LAST tick of each window."""
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
        feats_lgb, feats_nn = self._compute_batch_features(batches)
        B = feats_lgb.shape[0]
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
                preds.append(self._ensemble_predict_lgb(lgbs, feats_lgb))
                ws.append(w_lgb)
            if nns and w_nn > 0:
                preds.append(self._ensemble_predict_nn(nns, feats_nn))
                ws.append(w_nn)
            if not preds:
                continue
            stacked = np.stack(preds, axis=0)  # (M, B)
            ws_arr = np.array(ws, dtype=np.float32).reshape(-1, 1)
            pred_dmid = (stacked * ws_arr).sum(axis=0) / ws_arr.sum()

            if self._cw_enabled:
                actions = self._gate_with_band(pred_dmid, hcfg, band_per_row)
            else:
                actions = self._ev_gate_predict(pred_dmid, hcfg)
            out[:, HORIZON_TO_IDX[H]] = actions
        return out.tolist()


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    cfg = json.load(open(os.path.join(here, "config.json")))
    feats = cfg["feature"]
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        rng.standard_normal((100, len([f for f in feats if f not in ("sym", "time")]))).astype(np.float32),
        columns=[f for f in feats if f not in ("sym", "time")],
    )
    df["sym"] = 1
    import datetime as _dt
    df["time"] = [_dt.time(9, 40 + (i * 3) // 60, (i * 3) % 60) for i in range(100)]
    p = Predictor()
    out = p.predict([df, df, df])
    print("smoke test predict ->", out[0], "len=", len(out))
    df_ood = df.copy()
    df_ood["sym"] = 99
    out_ood = p.predict([df_ood])
    print("OOD sym=99 ->", out_ood[0])
    df_no_sym = df.drop(columns=["sym"])
    out_no = p.predict([df_no_sym])
    print("missing sym ->", out_no[0])
    df_pm = df.copy()
    df_pm["time"] = [_dt.time(13, 10 + (i * 3) // 60, (i * 3) % 60) for i in range(100)]
    out_pm = p.predict([df_pm])
    print("PM session ->", out_pm[0])
