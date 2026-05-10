"""Predictor for T173 multi-horizon submission (iter_019 v2 + iter_016 short horizons).

h=60 ("nn_lgb_2way"):
  pred_lgb = mean over 5 LightGBM boosters (iter_019 v2 full-retrain)
  pred_nn  = mean over 5 small MLPs (iter_019 v2 full-retrain, T87-style SPO+)
  pred     = (w_nn * pred_nn + w_lgb * pred_lgb) / (w_nn + w_lgb), w_nn=1.0, w_lgb=1.5
  + per-sym beta-calibrated conformal abstain band (from R_conformal_select)

h=5/10/20/40 ("lgb_cb_2way"):
  pred_lgb = mean over 5 LightGBM boosters (T98 iter_016 per-horizon regression)
  pred_cb  = mean over 5 CatBoost regressors (T98 iter_016 per-horizon regression)
  pred     = (w_lgb * pred_lgb + w_cb * pred_cb) / (w_lgb + w_cb)
  Plain DE-tuned asymmetric EV gate (no conformal band).

Decision per horizon:
  pred > thr_up  -> 2 (long)
  pred < -thr_dn -> 0 (short)
  else           -> 1 (flat)

Compliance with CRITICAL_CONSTRAINTS.md:
  - sym read from input ONLY for per-sym beta lookup in conformal band (allowed)
  - date never used
  - No cross-call state (predict() is fully independent per call)
  - All models are sym-agnostic (no sym features fed to model.forward)
  - Every rolling feature W <= 100
  - Shuffle-invariant: no row order dependency
"""
from __future__ import annotations

import importlib.util
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor

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
    """MLP inference in numpy: Linear -> LayerNorm -> GELU -> ... -> Linear -> /target_scale."""

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
        self._ffb = _load_module(here, "fast_features_batch.py", "t173_ffb")

        self._raw_feat_cols: List[str] = list(RAW_COLS_TRAIN_ORDER)
        self._raw_col_to_idx = {c: i for i, c in enumerate(self._raw_feat_cols)}
        self._amount_delta_idx = self._raw_feat_cols.index("amount_delta")

        self._extra_names: List[str] = list(self._ffb.all_feature_names())
        self._extra_keep_idx = np.array(
            [i for i, n in enumerate(self._extra_names) if n not in FAIL_NAMES],
            dtype=np.int64,
        )

        self._col_idx: Dict[str, int] = dict(self._raw_col_to_idx)
        if "midprice1" in self._raw_col_to_idx:
            self._col_idx["midprice"] = self._raw_col_to_idx["midprice1"]

        thresholds_path = os.path.join(here, "thresholds.json")
        with open(thresholds_path) as f:
            tcfg = json.load(f)
        self._horizons: List[Dict] = tcfg["horizons"]

        cw = tcfg.get("conformal_wrapper", {"enabled": False})
        self._cw_enabled = bool(cw.get("enabled", False))
        self._cw_horizons = set(int(h) for h in cw.get("apply_to_horizons", []))
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

        # Per-horizon model pools
        self._lgb_lists: Dict[int, List[lgb.Booster]] = {}
        self._nn_lists: Dict[int, List[_MLPNumpy]] = {}
        self._cb_lists: Dict[int, List[CatBoostRegressor]] = {}

        for hcfg in self._horizons:
            if not hcfg.get("active", True):
                continue
            H = int(hcfg["h"])
            seeds = hcfg.get("ensemble_seeds", [1, 7, 13, 42, 100])
            kind = hcfg.get("kind", "nn_lgb_2way")

            if kind == "lgb_cb_2way":
                lgb_pool = []
                cb_pool = []
                for s in seeds:
                    lp = os.path.join(here, f"model_lgb_h{H}_seed{s}.txt")
                    cp = os.path.join(here, f"model_cb_h{H}_seed{s}.cbm")
                    if os.path.isfile(lp):
                        lgb_pool.append(lgb.Booster(model_file=lp))
                    if os.path.isfile(cp):
                        m = CatBoostRegressor()
                        m.load_model(cp)
                        cb_pool.append(m)
                self._lgb_lists[H] = lgb_pool
                self._cb_lists[H] = cb_pool

            elif kind == "nn_lgb_2way":
                lgb_pool = []
                nn_pool = []
                for s in seeds:
                    lp = os.path.join(here, f"model_h{H}_seed{s}.txt")
                    np_ = os.path.join(here, f"nn_h{H}_seed{s}.npz")
                    if os.path.isfile(lp):
                        lgb_pool.append(lgb.Booster(model_file=lp))
                    if os.path.isfile(np_):
                        nn_pool.append(_MLPNumpy(np_))
                self._lgb_lists[H] = lgb_pool
                self._nn_lists[H] = nn_pool

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
    def _ensemble_lgb(boosters: List[lgb.Booster], X: np.ndarray) -> np.ndarray:
        acc = None
        for b in boosters:
            p = b.predict(X).astype(np.float32)
            acc = p if acc is None else acc + p
        return acc / float(len(boosters))

    @staticmethod
    def _ensemble_nn(nns: List[_MLPNumpy], X: np.ndarray) -> np.ndarray:
        acc = None
        for n in nns:
            p = n.predict(X)
            acc = p if acc is None else acc + p
        return acc / float(len(nns))

    @staticmethod
    def _ensemble_cb(models: List[CatBoostRegressor], X: np.ndarray) -> np.ndarray:
        acc = None
        for m in models:
            p = m.predict(X).astype(np.float32)
            acc = p if acc is None else acc + p
        return acc / float(len(models))

    @staticmethod
    def _ev_gate(pred: np.ndarray, thr_up: float, thr_dn: float) -> np.ndarray:
        out = np.full(pred.shape[0], 1, dtype=np.int64)
        out[pred > thr_up] = 2
        out[pred < -thr_dn] = 0
        return out

    def _ev_gate_with_band(self, pred: np.ndarray, thr_up: float, thr_dn: float,
                           band: np.ndarray) -> np.ndarray:
        out = np.full(pred.shape[0], 1, dtype=np.int64)
        out[pred > (thr_up + band)] = 2
        out[pred < -(thr_dn + band)] = 0
        return out

    def predict(self, batches: List[pd.DataFrame]) -> List[List[int]]:
        if not batches:
            return []
        feats = self._compute_batch_features(batches)
        B = feats.shape[0]
        out = np.ones((B, 5), dtype=np.int64)

        band_per_row = self._extract_band_per_row(batches) if self._cw_enabled else np.zeros(B)

        for hcfg in self._horizons:
            if not hcfg.get("active", True):
                continue
            H = int(hcfg["h"])
            kind = hcfg.get("kind", "nn_lgb_2way")
            thr_up = float(hcfg["thr_up"])
            thr_dn = float(hcfg["thr_dn"])

            if kind == "lgb_cb_2way":
                lgbs = self._lgb_lists.get(H, [])
                cbs = self._cb_lists.get(H, [])
                w_lgb = float(hcfg.get("w_lgb", 1.0))
                w_cb = float(hcfg.get("w_cb", 1.0))
                preds_w: List[Tuple[np.ndarray, float]] = []
                if lgbs and w_lgb > 0:
                    preds_w.append((self._ensemble_lgb(lgbs, feats), w_lgb))
                if cbs and w_cb > 0:
                    preds_w.append((self._ensemble_cb(cbs, feats), w_cb))
                if not preds_w:
                    continue
                wsum = sum(w for _, w in preds_w)
                pred = np.zeros(B, dtype=np.float64)
                for p, w in preds_w:
                    pred += w * p.astype(np.float64)
                pred /= wsum
                actions = self._ev_gate(pred.astype(np.float32), thr_up, thr_dn)

            elif kind == "nn_lgb_2way":
                lgbs = self._lgb_lists.get(H, [])
                nns = self._nn_lists.get(H, [])
                w_lgb = float(hcfg.get("w_lgb", 1.5))
                w_nn = float(hcfg.get("w_nn", 1.0))
                preds_w = []
                if lgbs and w_lgb > 0:
                    preds_w.append((self._ensemble_lgb(lgbs, feats), w_lgb))
                if nns and w_nn > 0:
                    preds_w.append((self._ensemble_nn(nns, feats), w_nn))
                if not preds_w:
                    continue
                wsum = sum(w for _, w in preds_w)
                pred = np.zeros(B, dtype=np.float64)
                for p, w in preds_w:
                    pred += w * p.astype(np.float64)
                pred /= wsum
                if self._cw_enabled and H in self._cw_horizons:
                    actions = self._ev_gate_with_band(pred.astype(np.float32), thr_up, thr_dn, band_per_row)
                else:
                    actions = self._ev_gate(pred.astype(np.float32), thr_up, thr_dn)

            else:
                continue

            out[:, HORIZON_TO_IDX[H]] = actions

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
    df["sym"] = 1
    print("Loading Predictor...")
    p = Predictor()
    print("Smoke test with 3 batches...")
    out = p.predict([df, df, df])
    print("predict ->", out[0], "len=", len(out))
    assert len(out) == 3, "Expected 3 predictions"
    assert len(out[0]) == 5, "Expected 5 horizons per row"
    all_valid = all(a in {0, 1, 2} for row in out for a in row)
    assert all_valid, "All actions must be in {0, 1, 2}"
    print("OOD sym test...")
    df_ood = df.copy()
    df_ood["sym"] = 99
    out_ood = p.predict([df_ood])
    print("OOD sym=99 ->", out_ood[0])
    print("Missing sym test...")
    df_no_sym = df[[c for c in df.columns if c != "sym"]]
    out_no = p.predict([df_no_sym])
    print("missing sym ->", out_no[0])
    print("ALL SMOKE TESTS PASSED")
