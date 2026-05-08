"""Predictor for iter_016 ablation — GRU removed, conservative thresh.

Ablation rationale:
  iter_016 v3 platform (+25.08) underperformed iter_015 v1 (+28.16) despite
  +4.6 LOSO lift on local 442k. Reverse local-vs-platform signal indicates
  the v3 additions overfit local. The two prime suspects are:
    1. T95 GRU (cross-corr 0.34 = likely fitting local noise)
    2. h=60 thresholds re-tuned aggressively on local 442k
  This ablation removes both.

Per-horizon stacks:

  h=5/10/20/40 (DISABLED — always action=1 / flat):
    The short-horizon DE 4-D thresh search was a key over-tuning vector.
    With active=false the whole h=5/10/20/40 branch is skipped and every
    sample emits action 1 (flat) for those horizons.

  h=60 ("threeway_no_gru"):
    pred_h = (1.0 * mean_T87_NN + 0.7 * mean_T99_CB_Huber
              + 1.0 * mean_T99_LGB_Huber) / 2.7
    Same weight ratio as v3's non-GRU streams; GRU dropped entirely.
    thr_up=4.21e-4 / thr_dn=1.86e-4 — conservative iter_015 v1-style
    thresholds (NOT the v3 aggressive 2.88e-4 / 2.19e-4 from local DE).

Decision per horizon (asymmetric EV gate):
    pred_h > thr_up_h        -> 2 (long)
    pred_h < -thr_dn_h       -> 0 (short)
    else                     -> 1 (flat)

Compliance with CRITICAL_CONSTRAINTS.md:
  - sym / date never enter the feature vector or any model's input
  - No cross-call state held on `self` beyond loaded models (predict() is
    fully independent across calls; safe under shuffled test order)
  - Every component is sym-agnostic; unseen sym IDs are safe
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

SEEDS = (1, 7, 13, 42, 100)


def _gelu_tanh(x: np.ndarray) -> np.ndarray:
    c = 0.7978845608028654
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x * x * x)))


def _layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


class _MLPNumpy:
    """T87 SPO+ MLP inference in numpy."""

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
        self._ffb = _load_module(here, "fast_features_batch.py", "iter_016_ablation_ffb")

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

        with open(os.path.join(here, "thresholds.json")) as f:
            tcfg = json.load(f)
        self._horizons: List[Dict] = tcfg["horizons"]

        self._h_short: Dict[int, Dict] = {}
        self._h_long: Dict = None

        for hcfg in self._horizons:
            if not hcfg.get("active", True):
                continue
            H = int(hcfg["h"])
            kind = hcfg.get("kind", "lgb_cb_2way")
            seeds = hcfg.get("ensemble_seeds", list(SEEDS))

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
                self._h_short[H] = {
                    "lgb": lgb_pool,
                    "cb": cb_pool,
                    "w_lgb": float(hcfg.get("w_lgb", 1.0)),
                    "w_cb": float(hcfg.get("w_cb", 1.0)),
                    "thr_up": float(hcfg.get("thr_up", 1.0)),
                    "thr_dn": float(hcfg.get("thr_dn", 1.0)),
                }
            elif kind == "threeway_no_gru":
                nn_pool = []
                cb_pool = []
                huber_pool = []
                for s in seeds:
                    nn_p = os.path.join(here, f"nn_h60_seed{s}.npz")
                    cb_p = os.path.join(here, f"model_cb_huber_seed{s}.cbm")
                    hu_p = os.path.join(here, f"model_T99_huber_a0.001_seed{s}.txt")
                    if os.path.isfile(nn_p):
                        nn_pool.append(_MLPNumpy(nn_p))
                    if os.path.isfile(cb_p):
                        m = CatBoostRegressor()
                        m.load_model(cb_p)
                        cb_pool.append(m)
                    if os.path.isfile(hu_p):
                        huber_pool.append(lgb.Booster(model_file=hu_p))

                self._h_long = {
                    "horizon": H,
                    "nn": nn_pool,
                    "cb": cb_pool,
                    "huber": huber_pool,
                    "w_t87": float(hcfg.get("w_t87", 1.0)),
                    "w_t89": float(hcfg.get("w_t89", 0.7)),
                    "w_huber": float(hcfg.get("w_huber", 1.0)),
                    "thr_up": float(hcfg.get("thr_up", 1.0)),
                    "thr_dn": float(hcfg.get("thr_dn", 1.0)),
                }
            else:
                raise ValueError(f"unknown horizon kind: {kind}")

    @staticmethod
    def _ensemble_avg(preds: List[np.ndarray]) -> np.ndarray:
        if not preds:
            return None
        if len(preds) == 1:
            return preds[0].astype(np.float32, copy=False)
        acc = preds[0].astype(np.float64, copy=True)
        for p in preds[1:]:
            acc += p.astype(np.float64)
        return (acc / float(len(preds))).astype(np.float32, copy=False)

    @staticmethod
    def _ev_gate(pred: np.ndarray, thr_up: float, thr_dn: float) -> np.ndarray:
        a = np.full(pred.shape[0], 1, dtype=np.int64)
        a[pred > thr_up] = 2
        a[pred < -thr_dn] = 0
        return a

    def _compute_features(self, batches: List[pd.DataFrame]) -> np.ndarray:
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
        feats_359 = np.concatenate([raw_last, extras_kept], axis=1)
        return feats_359

    def predict(self, batches: List[pd.DataFrame]) -> List[List[int]]:
        if not batches:
            return []
        feats = self._compute_features(batches)
        B = feats.shape[0]
        out = np.ones((B, 5), dtype=np.int64)

        for H, mp in self._h_short.items():
            preds_w: List[Tuple[np.ndarray, float]] = []
            if mp["lgb"] and mp["w_lgb"] > 0:
                preds_w.append(
                    (self._ensemble_avg([m.predict(feats).astype(np.float32) for m in mp["lgb"]]),
                     mp["w_lgb"])
                )
            if mp["cb"] and mp["w_cb"] > 0:
                preds_w.append(
                    (self._ensemble_avg([m.predict(feats).astype(np.float32) for m in mp["cb"]]),
                     mp["w_cb"])
                )
            if not preds_w:
                continue
            wsum = sum(w for _, w in preds_w)
            pred = np.zeros(B, dtype=np.float64)
            for p, w in preds_w:
                pred += w * p.astype(np.float64)
            pred /= wsum
            actions = self._ev_gate(pred.astype(np.float32, copy=False),
                                    mp["thr_up"], mp["thr_dn"])
            out[:, HORIZON_TO_IDX[H]] = actions

        if self._h_long is not None:
            mp = self._h_long
            preds_w: List[Tuple[np.ndarray, float]] = []
            if mp["nn"] and mp["w_t87"] > 0:
                preds_w.append(
                    (self._ensemble_avg([m.predict(feats) for m in mp["nn"]]),
                     mp["w_t87"])
                )
            if mp["cb"] and mp["w_t89"] > 0:
                preds_w.append(
                    (self._ensemble_avg([m.predict(feats).astype(np.float32) for m in mp["cb"]]),
                     mp["w_t89"])
                )
            if mp["huber"] and mp["w_huber"] > 0:
                preds_w.append(
                    (self._ensemble_avg([m.predict(feats).astype(np.float32) for m in mp["huber"]]),
                     mp["w_huber"])
                )
            if preds_w:
                wsum = sum(w for _, w in preds_w)
                pred = np.zeros(B, dtype=np.float64)
                for p, w in preds_w:
                    pred += w * p.astype(np.float64)
                pred /= wsum
                actions = self._ev_gate(pred.astype(np.float32, copy=False),
                                        mp["thr_up"], mp["thr_dn"])
                out[:, HORIZON_TO_IDX[mp["horizon"]]] = actions

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
