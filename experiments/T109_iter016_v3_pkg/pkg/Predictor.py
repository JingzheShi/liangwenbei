"""Predictor for iter_016 v3 CHIS — GPU torch + CB_Huber upgrade.

Identical pipeline to T108 / iter_016 v2, except the h=60 CatBoost stream
is swapped from T89 (RMSE) to T99 CB_Huber alpha=1e-3 — the new SOTA
component identified in T99 ensemble search. Weights / thresholds for
h=60 are re-tuned accordingly. The T95 GRU pool is still implemented in
PyTorch (CUDA when available, numpy fallback).

Per-horizon stacks:

  h=5/10/20/40 ("lgb_cb_2way"):
    pred_h = (w_lgb * mean_lgb_h + w_cb * mean_cb_h) / (w_lgb + w_cb)

  h=60 ("fourway_iter015v2"):  -- weights from T99 winner refine
    pred_h = (1.0 * mean_T87_NN + 0.7 * mean_T99_CB_Huber
              + 1.0 * mean_T99_LGB_Huber + 1.0 * mean_T95_GRU) / 3.7

Decision per horizon (asymmetric EV gate):
    pred_h > thr_up_h        -> 2 (long)
    pred_h < -thr_dn_h       -> 0 (short)
    else                     -> 1 (flat)

Compliance with CRITICAL_CONSTRAINTS.md:
  - sym / date never enter the feature vector or any model's input
  - No cross-call state held on `self` beyond loaded models (predict() is
    fully independent across calls; safe under shuffled test order)
  - Per-window normalisation for the GRU is window-internal (no train-time
    per-sym stats)
  - Every component is sym-agnostic, so unseen sym IDs are safe
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

GRU_RAW_COLS = []
for _i in range(1, 6):
    GRU_RAW_COLS.extend([f"bid{_i}", f"bsize{_i}", f"ask{_i}", f"asize{_i}"])
GRU_RAW_COLS = tuple(GRU_RAW_COLS)
assert len(GRU_RAW_COLS) == 20

SEEDS = (1, 7, 13, 42, 100)


def _gelu_tanh(x: np.ndarray) -> np.ndarray:
    c = 0.7978845608028654
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x * x * x)))


def _layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


def _sigmoid(x: np.ndarray) -> np.ndarray:
    pos = x >= 0
    out = np.empty_like(x)
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


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


class _GRUNumpy:
    """T95 single-layer GRU inference in numpy (fallback when torch unavailable)."""

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
        self.Wih = d["gru_Wih"].astype(np.float32)
        self.Whh = d["gru_Whh"].astype(np.float32)
        self.bih = d["gru_bih"].astype(np.float32)
        self.bhh = d["gru_bhh"].astype(np.float32)
        self.fc_W = d["fc_W"].astype(np.float32)
        self.fc_b = d["fc_b"].astype(np.float32)

    def predict(self, X3d: np.ndarray) -> np.ndarray:
        B, W, F = X3d.shape
        assert W == self.window and F == self.in_dim, f"GRU input shape mismatch: got {X3d.shape}"
        m = X3d.mean(axis=1, keepdims=True)
        s = X3d.std(axis=1, keepdims=True)
        xs = (X3d - m) / np.maximum(s, 1e-8)
        xs = np.clip(xs, -self.clip, self.clip).astype(np.float32, copy=False)
        xn = _layernorm(xs, self.in_norm_W, self.in_norm_b)
        x_proj = xn @ self.Wih.T + self.bih
        H = self.hidden
        h = np.zeros((B, H), dtype=np.float32)
        for t in range(W):
            gi = x_proj[:, t, :]
            gh = h @ self.Whh.T + self.bhh
            r = _sigmoid(gi[:, :H] + gh[:, :H])
            z = _sigmoid(gi[:, H:2 * H] + gh[:, H:2 * H])
            n = np.tanh(gi[:, 2 * H:] + r * gh[:, 2 * H:])
            h = (1.0 - z) * n + z * h
        out = h @ self.fc_W.T + self.fc_b
        return (out.squeeze(-1) / self.target_scale).astype(np.float32, copy=False)


def _try_import_torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except Exception:
        return None, None


class _GRUEnsembleTorch:
    """Ensemble of 5 single-layer GRUs implemented in torch.

    All five seeded models share the same `clip` constant and `target_scale`
    (verified across seeds 1/7/13/42/100), so per-window normalisation is
    computed once on the chosen device and shared across all five forward
    passes. Falls back transparently to torch CPU if CUDA fails.
    """

    def __init__(self, npz_paths: List[str], device_str: str):
        torch, nn = _try_import_torch()
        assert torch is not None, "torch must be importable to construct _GRUEnsembleTorch"
        self.torch = torch
        self.nn = nn
        self.device = torch.device(device_str)
        self.models = []
        in_dim = hidden = window = None
        clip = target_scale = None
        for p in npz_paths:
            d = np.load(p, allow_pickle=True)
            cur_in = int(d["in_dim"][0])
            cur_h = int(d["hidden"][0])
            cur_w = int(d["window"][0])
            cur_clip = float(d["clip"][0])
            cur_ts = float(d["target_scale"][0])
            if in_dim is None:
                in_dim, hidden, window, clip, target_scale = cur_in, cur_h, cur_w, cur_clip, cur_ts
            else:
                assert (cur_in, cur_h, cur_w) == (in_dim, hidden, window), \
                    f"GRU shape mismatch in npz {p}"
                assert abs(cur_clip - clip) < 1e-6 and abs(cur_ts - target_scale) < 1e-3, \
                    f"GRU clip/target_scale mismatch in npz {p}"
            m = self._build_model(d, in_dim, hidden)
            m.eval()
            m.to(self.device)
            self.models.append(m)
        self.in_dim = in_dim
        self.hidden = hidden
        self.window = window
        self.clip = clip
        self.target_scale = target_scale

    def _build_model(self, d, in_dim, hidden):
        torch = self.torch
        nn = self.nn

        class _M(nn.Module):
            def __init__(self):
                super().__init__()
                self.in_norm = nn.LayerNorm(in_dim, eps=1e-5)
                self.gru = nn.GRU(in_dim, hidden, num_layers=1, batch_first=True)
                self.fc = nn.Linear(hidden, 1)

        m = _M()
        with torch.no_grad():
            m.in_norm.weight.copy_(torch.from_numpy(d["in_norm_W"]).float())
            m.in_norm.bias.copy_(torch.from_numpy(d["in_norm_b"]).float())
            m.gru.weight_ih_l0.copy_(torch.from_numpy(d["gru_Wih"]).float())
            m.gru.weight_hh_l0.copy_(torch.from_numpy(d["gru_Whh"]).float())
            m.gru.bias_ih_l0.copy_(torch.from_numpy(d["gru_bih"]).float())
            m.gru.bias_hh_l0.copy_(torch.from_numpy(d["gru_bhh"]).float())
            m.fc.weight.copy_(torch.from_numpy(d["fc_W"]).float())
            m.fc.bias.copy_(torch.from_numpy(d["fc_b"]).float())
        return m

    def predict_mean(self, x_np: np.ndarray) -> np.ndarray:
        """Compute mean of the 5-seed ensemble on the supplied (B, W, F) batch."""
        torch = self.torch
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(x_np)).to(self.device).float()
            m = x.mean(dim=1, keepdim=True)
            s = x.std(dim=1, keepdim=True, unbiased=False)
            xs = (x - m) / torch.clamp(s, min=1e-8)
            xs = torch.clamp(xs, -self.clip, self.clip)
            acc = None
            for mdl in self.models:
                xn = mdl.in_norm(xs)
                out, _ = mdl.gru(xn)
                last = out[:, -1, :]
                y = mdl.fc(last).squeeze(-1) / self.target_scale
                acc = y if acc is None else acc + y
            acc = acc / float(len(self.models))
            return acc.detach().cpu().numpy().astype(np.float32)


def _load_module(here: str, fname: str, mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(here, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Predictor:
    def __init__(self) -> None:
        here = os.path.dirname(os.path.abspath(__file__))
        self._ffb = _load_module(here, "fast_features_batch.py", "iter_016_chis_ffb")

        self._raw_feat_cols: List[str] = list(RAW_COLS_TRAIN_ORDER)
        self._raw_col_to_idx = {c: i for i, c in enumerate(self._raw_feat_cols)}
        self._amount_delta_idx = self._raw_feat_cols.index("amount_delta")

        self._extra_names: List[str] = list(self._ffb.all_feature_names())
        self._extra_keep_idx = np.array(
            [i for i, n in enumerate(self._extra_names) if n not in FAIL_NAMES],
            dtype=np.int64,
        )

        self._gru_col_idx = np.array(
            [self._raw_col_to_idx[c] for c in GRU_RAW_COLS], dtype=np.int64
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
            elif kind == "fourway_iter015v2":
                nn_pool = []
                cb_pool = []
                huber_pool = []
                gru_paths = []
                gru_pool_np = []
                for s in seeds:
                    nn_p = os.path.join(here, f"nn_h60_seed{s}.npz")
                    cb_p = os.path.join(here, f"model_cb_huber_seed{s}.cbm")
                    hu_p = os.path.join(here, f"model_T99_huber_a0.001_seed{s}.txt")
                    gr_p = os.path.join(here, f"gru_h60_seed{s}.npz")
                    if os.path.isfile(nn_p):
                        nn_pool.append(_MLPNumpy(nn_p))
                    if os.path.isfile(cb_p):
                        m = CatBoostRegressor()
                        m.load_model(cb_p)
                        cb_pool.append(m)
                    if os.path.isfile(hu_p):
                        huber_pool.append(lgb.Booster(model_file=hu_p))
                    if os.path.isfile(gr_p):
                        gru_paths.append(gr_p)
                        gru_pool_np.append(_GRUNumpy(gr_p))

                # Try torch GPU first, then torch CPU, then numpy fallback.
                gru_torch = None
                gru_backend = "numpy"
                torch_mod, _ = _try_import_torch()
                if torch_mod is not None and gru_paths:
                    try:
                        if torch_mod.cuda.is_available():
                            gru_torch = _GRUEnsembleTorch(gru_paths, "cuda")
                            gru_backend = "torch_cuda"
                        else:
                            gru_torch = _GRUEnsembleTorch(gru_paths, "cpu")
                            gru_backend = "torch_cpu"
                    except Exception:
                        gru_torch = None
                        gru_backend = "numpy"

                self._h_long = {
                    "horizon": H,
                    "nn": nn_pool,
                    "cb": cb_pool,
                    "huber": huber_pool,
                    "gru_np": gru_pool_np,
                    "gru_torch": gru_torch,
                    "gru_backend": gru_backend,
                    "w_t87": float(hcfg.get("w_t87", 1.0)),
                    "w_t89": float(hcfg.get("w_t89", 0.7)),
                    "w_huber": float(hcfg.get("w_huber", 1.0)),
                    "w_gru": float(hcfg.get("w_gru", 1.0)),
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

    def _compute_features(self, batches: List[pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """Returns:
            feats_359 - (N, 359) tabular feature vector for LGB/CB/T87/T89/Huber
            x_gru     - (N, 100, 20) raw LOB window for T95 GRU
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
        feats_359 = np.concatenate([raw_last, extras_kept], axis=1)

        x_gru = X3d[:, :, self._gru_col_idx].astype(np.float32, copy=False)
        return feats_359, x_gru

    def predict(self, batches: List[pd.DataFrame]) -> List[List[int]]:
        if not batches:
            return []
        feats, x_gru = self._compute_features(batches)
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
            if mp["w_gru"] > 0:
                if mp["gru_torch"] is not None:
                    p_gru = mp["gru_torch"].predict_mean(x_gru)
                    preds_w.append((p_gru, mp["w_gru"]))
                elif mp["gru_np"]:
                    preds_w.append(
                        (self._ensemble_avg([g.predict(x_gru) for g in mp["gru_np"]]),
                         mp["w_gru"])
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
    backend = p._h_long["gru_backend"] if p._h_long is not None else "n/a"
    print(f"smoke test: GRU backend={backend}")
    out = p.predict([df, df, df])
    print("smoke test predict ->", out[0], "len=", len(out))
