"""Predictor for iter_015 v2: 4-way ensemble of T87 SPO+ NN + T89 CB + T99 LGB Huber + T95 GRU.

w_t87:w_t89:w_huber:w_gru = 1.0:1.5:0.7:0.7

Components, all stateless and sym-agnostic:
  pred_nn    = mean over 5 SPO+ MLPs   (T87, numpy [359 -> 256 -> 128 -> 64 -> 1])
  pred_cb    = mean over 5 CatBoost regressors (T89)
  pred_huber = mean over 5 LightGBM Huber regressors (T99)
  pred_gru   = mean over 5 single-layer numpy GRUs (T95, raw 20-d top-5 LOB window)
  pred       = (1.0*pred_nn + 1.5*pred_cb + 0.7*pred_huber + 0.7*pred_gru)
              / (1.0 + 1.5 + 0.7 + 0.7)

Decision: asymmetric EV gate (h=60 horizon)
  pred > thr_up => 2 (long), pred < -thr_dn => 0 (short), else 1 (flat)

Compliance with CRITICAL_CONSTRAINTS.md:
  - sym/date never enter the feature vector or the GRU input
  - No per-call state on `self` beyond loaded models (each predict() is independent)
  - Per-window normalisation for GRU is window-internal (no train-time per-sym stats)
  - sym=99-safe: every model is sym-agnostic
"""
from __future__ import annotations

import importlib.util
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb

# CatBoost is in requirements.txt; importing here at module load time keeps the
# Predictor class clean.
from catboost import CatBoostRegressor

WINDOW = 100

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

# T95 GRU raw-LOB column order (top-5 interleaved: bid,bsize,ask,asize per level)
GRU_RAW_COLS = []
for _i in range(1, 6):
    GRU_RAW_COLS.extend([f"bid{_i}", f"bsize{_i}", f"ask{_i}", f"asize{_i}"])
GRU_RAW_COLS = tuple(GRU_RAW_COLS)
assert len(GRU_RAW_COLS) == 20


def _gelu_tanh(x: np.ndarray) -> np.ndarray:
    c = 0.7978845608028654  # sqrt(2/pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x * x * x)))


def _layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid
    pos = x >= 0
    out = np.empty_like(x)
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


class _MLPNumpy:
    """T87 MLP inference in numpy: standardize -> [Linear -> LayerNorm -> GELU]xN -> Linear -> /target_scale."""

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


class _GRUNumpy:
    """T95 single-layer GRU inference in numpy.

    Pipeline:
      raw window (B, W, 20) -> per-window standardize -> clip(+-clip) -> LayerNorm
                            -> GRU (hidden=64) -> last-step -> Linear -> / target_scale
    """

    def __init__(self, npz_path: str):
        d = np.load(npz_path, allow_pickle=True)
        self.in_dim = int(d["in_dim"][0])
        self.hidden = int(d["hidden"][0])
        self.window = int(d["window"][0])
        self.target_scale = float(d["target_scale"][0])
        self.clip = float(d["clip"][0])
        self.norm_mode = str(d["norm_mode"][0])
        self.in_norm_W = d["in_norm_W"].astype(np.float32)
        self.in_norm_b = d["in_norm_b"].astype(np.float32)
        self.Wih = d["gru_Wih"].astype(np.float32)        # (3H, F)
        self.Whh = d["gru_Whh"].astype(np.float32)        # (3H, H)
        self.bih = d["gru_bih"].astype(np.float32)        # (3H,)
        self.bhh = d["gru_bhh"].astype(np.float32)        # (3H,)
        self.fc_W = d["fc_W"].astype(np.float32)          # (1, H)
        self.fc_b = d["fc_b"].astype(np.float32)          # (1,)

    def predict(self, X3d: np.ndarray) -> np.ndarray:
        """X3d: (B, W=100, F=20) raw LOB window. Returns (B,) Δmid_norm prediction."""
        B, W, F = X3d.shape
        assert W == self.window and F == self.in_dim, f"GRU input shape mismatch: got {X3d.shape}"
        # Per-window standardise (norm_mode == "window")
        m = X3d.mean(axis=1, keepdims=True)
        s = X3d.std(axis=1, keepdims=True)
        xs = (X3d - m) / np.maximum(s, 1e-8)
        xs = np.clip(xs, -self.clip, self.clip).astype(np.float32, copy=False)
        # LayerNorm over feature dim
        xn = _layernorm(xs, self.in_norm_W, self.in_norm_b)
        # Pre-project input for all timesteps:  (B, W, 3H)
        x_proj = xn @ self.Wih.T + self.bih
        H = self.hidden
        h = np.zeros((B, H), dtype=np.float32)
        for t in range(W):
            gi = x_proj[:, t, :]
            gh = h @ self.Whh.T + self.bhh  # (B, 3H)
            r = _sigmoid(gi[:, :H] + gh[:, :H])
            z = _sigmoid(gi[:, H:2 * H] + gh[:, H:2 * H])
            n = np.tanh(gi[:, 2 * H:] + r * gh[:, 2 * H:])
            h = (1.0 - z) * n + z * h
        # Final FC + descale
        out = h @ self.fc_W.T + self.fc_b
        return (out.squeeze(-1) / self.target_scale).astype(np.float32, copy=False)


def _load_module(here: str, fname: str, mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(here, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SEEDS = (1, 7, 13, 42, 100)


class Predictor:
    def __init__(self) -> None:
        here = os.path.dirname(os.path.abspath(__file__))
        self._ffb = _load_module(here, "fast_features_batch.py", "iter_015v2_ffb")

        self._raw_feat_cols: List[str] = list(RAW_COLS_TRAIN_ORDER)
        self._raw_col_to_idx = {c: i for i, c in enumerate(self._raw_feat_cols)}
        self._amount_delta_idx = self._raw_feat_cols.index("amount_delta")

        self._extra_names: List[str] = list(self._ffb.all_feature_names())
        self._extra_keep_idx = np.array(
            [i for i, n in enumerate(self._extra_names) if n not in FAIL_NAMES],
            dtype=np.int64,
        )

        # GRU raw-cache columns (interleaved top-5 LOB)
        self._gru_col_idx = np.array(
            [self._raw_col_to_idx[c] for c in GRU_RAW_COLS], dtype=np.int64
        )

        self._col_idx: Dict[str, int] = dict(self._raw_col_to_idx)
        if "midprice1" in self._raw_col_to_idx:
            self._col_idx["midprice"] = self._raw_col_to_idx["midprice1"]

        # Thresholds + weights
        with open(os.path.join(here, "thresholds.json")) as f:
            tcfg = json.load(f)
        h_cfg = tcfg["horizons"][0]  # h=60 only
        self.thr_up = float(h_cfg["thr_up"])
        self.thr_dn = float(h_cfg["thr_dn"])
        self.w_t87 = float(h_cfg["w_t87"])
        self.w_t89 = float(h_cfg["w_t89"])
        self.w_huber = float(h_cfg["w_huber"])
        self.w_gru = float(h_cfg["w_gru"])

        # Load 5x SPO+ MLPs (T87)
        self._nn_pool = []
        for s in SEEDS:
            p = os.path.join(here, f"nn_h60_seed{s}.npz")
            if os.path.isfile(p):
                self._nn_pool.append(_MLPNumpy(p))

        # Load 5x CatBoost (T89)
        self._cb_pool = []
        for s in SEEDS:
            p = os.path.join(here, f"model_T89_seed{s}.cbm")
            if os.path.isfile(p):
                cb = CatBoostRegressor()
                cb.load_model(p)
                self._cb_pool.append(cb)

        # Load 5x LGB Huber (T99)
        self._huber_pool = []
        for s in SEEDS:
            p = os.path.join(here, f"model_T99_huber_a0.001_seed{s}.txt")
            if os.path.isfile(p):
                self._huber_pool.append(lgb.Booster(model_file=p))

        # Load 5x GRU (T95)
        self._gru_pool = []
        for s in SEEDS:
            p = os.path.join(here, f"gru_h60_seed{s}.npz")
            if os.path.isfile(p):
                self._gru_pool.append(_GRUNumpy(p))

    @staticmethod
    def _ensemble_avg(preds: List[np.ndarray]) -> np.ndarray:
        if len(preds) == 1:
            return preds[0].astype(np.float32, copy=False)
        acc = preds[0].astype(np.float64, copy=True)
        for p in preds[1:]:
            acc += p.astype(np.float64)
        return (acc / float(len(preds))).astype(np.float32, copy=False)

    def _compute_features(self, batches: List[pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """Returns:
            feats_359  - (N, 359) tabular feature vector for T87/T89/Huber
            x_gru      - (N, 100, 20) raw LOB window for T95
        """
        N = len(batches)
        K = len(self._raw_feat_cols)
        X3d = np.empty((N, WINDOW, K), dtype=np.float64)
        for n, df in enumerate(batches):
            X3d[n] = df[self._raw_feat_cols].to_numpy(dtype=np.float64, copy=False)

        # Last-tick raw vector with amount_delta log1p (matches training)
        raw_last = X3d[:, -1, :].astype(np.float32, copy=True)
        if self._amount_delta_idx >= 0:
            v = raw_last[:, self._amount_delta_idx]
            raw_last[:, self._amount_delta_idx] = np.sign(v) * np.log1p(np.abs(v))

        extras_full = self._ffb.compute_batch_features(X3d, self._col_idx)
        extras_full = np.where(np.isfinite(extras_full), extras_full, 0.0).astype(np.float32, copy=False)
        extras_kept = extras_full[:, self._extra_keep_idx]
        feats_359 = np.concatenate([raw_last, extras_kept], axis=1)

        # GRU raw 20-col window
        x_gru = X3d[:, :, self._gru_col_idx].astype(np.float32, copy=False)
        return feats_359, x_gru

    def predict(self, batches: List[pd.DataFrame]) -> List[List[int]]:
        if not batches:
            return []
        feats, x_gru = self._compute_features(batches)
        B = feats.shape[0]

        preds_w: List[Tuple[np.ndarray, float]] = []

        if self._nn_pool and self.w_t87 > 0:
            preds_w.append(
                (self._ensemble_avg([nn.predict(feats) for nn in self._nn_pool]), self.w_t87)
            )
        if self._cb_pool and self.w_t89 > 0:
            preds_w.append(
                (self._ensemble_avg([cb.predict(feats).astype(np.float32) for cb in self._cb_pool]),
                 self.w_t89)
            )
        if self._huber_pool and self.w_huber > 0:
            preds_w.append(
                (self._ensemble_avg([m.predict(feats).astype(np.float32) for m in self._huber_pool]),
                 self.w_huber)
            )
        if self._gru_pool and self.w_gru > 0:
            preds_w.append(
                (self._ensemble_avg([g.predict(x_gru) for g in self._gru_pool]), self.w_gru)
            )

        if not preds_w:
            # Fallback: all flat
            return [[1, 1, 1, 1, 1] for _ in range(B)]

        wsum = sum(w for _, w in preds_w)
        pred = np.zeros(B, dtype=np.float64)
        for p, w in preds_w:
            pred += w * p.astype(np.float64)
        pred /= wsum
        pred_f32 = pred.astype(np.float32, copy=False)

        actions = np.full(B, 1, dtype=np.int64)
        actions[pred_f32 > self.thr_up] = 2
        actions[pred_f32 < -self.thr_dn] = 0

        out = np.ones((B, 5), dtype=np.int64)
        out[:, 4] = actions  # h=60 lives at index 4
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
