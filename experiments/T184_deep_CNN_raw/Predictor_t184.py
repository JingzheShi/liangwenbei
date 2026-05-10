"""T184 Predictor: v2 LGB + Raw-LOB TickTransformer NN.

Replaces T87 MLP with TickTransformer trained on raw 100-tick LOB window.
v2 LGB (5 seeds) kept intact.

Input pipeline per predict() call:
  - LGB path: batch → schemeP features (fast_features_batch.py) → LGB predict
  - Transformer path: batch → raw v2 LOB window (100×31) → window-norm → Transformer

Both paths combined: (w_lgb * pred_lgb + w_nn * pred_nn) / (w_lgb + w_nn)

CRITICAL_CONSTRAINTS compliance:
  - No sym in model forward (sym only used for conformal band lookup, not fed to model)
  - No date anywhere
  - No cross-call state (all processing stateless per predict() call)
  - Sym-agnostic: global window-norm statistics from training, Transformer is sym-agnostic
"""
from __future__ import annotations

import importlib.util
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
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

# Top-5 LOB columns needed for raw Transformer input (v2 features)
_TOP5_COLS = (
    "bid1", "bsize1", "ask1", "asize1",
    "bid2", "bsize2", "ask2", "asize2",
    "bid3", "bsize3", "ask3", "asize3",
    "bid4", "bsize4", "ask4", "asize4",
    "bid5", "bsize5", "ask5", "asize5",
)


class _TickTransformerTorch(torch.nn.Module):
    """TickTransformer for CPU inference loaded from .pt checkpoint."""

    def __init__(self, ckpt_path: str):
        super().__init__()
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.n_features = ck["n_features"]
        self.d_model = ck["d_model"]
        self.n_layers = ck["n_layers"]
        self.nhead = ck["nhead"]
        self.window = ck["window"]
        self.target_scale = float(ck["target_scale"])
        ff_dim = ck.get("ff_dim", 256)

        self.proj = torch.nn.Linear(self.n_features, self.d_model)
        self.pe = torch.nn.Embedding(self.window, self.d_model)
        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.nhead,
            dim_feedforward=ff_dim,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(enc_layer, num_layers=self.n_layers)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(self.d_model, self.d_model // 2),
            torch.nn.GELU(),
            torch.nn.Linear(self.d_model // 2, 1),
        )
        self.load_state_dict(ck["state_dict"])
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, F) window-normalized
        B, W, F = x.shape
        pos = torch.arange(W, device=x.device)
        tokens = self.proj(x) + self.pe(pos).unsqueeze(0)
        tokens = self.encoder(tokens)
        last = tokens[:, -1, :]
        return (self.head(last).squeeze(-1) / self.target_scale)


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
        self.W = []; self.b = []; self.LN_W = []; self.LN_b = []
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
        return (out.squeeze(-1) / self.target_scale).astype(np.float32, copy=False)


def _load_module(here: str, fname: str, mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(here, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_v2_features_from_raw(x3d: np.ndarray, top5_idx: np.ndarray) -> np.ndarray:
    """Build v2 LOB features (31 dims) from raw batch array.

    x3d: (N, W, K) full raw LOB array, K>=154
    top5_idx: indices of top5 cols (bid1,bsize1,ask1,asize1,...,bid5,bsize5,ask5,asize5)
    Returns: (N, W, 31)
    """
    N, W, _ = x3d.shape
    raw = x3d[:, :, top5_idx].astype(np.float32)  # (N, W, 20)
    bid1 = raw[:, :, 0]; bsize1 = raw[:, :, 1]; ask1 = raw[:, :, 2]; asize1 = raw[:, :, 3]
    bid2 = raw[:, :, 4]; bsize2 = raw[:, :, 5]; ask2 = raw[:, :, 6]; asize2 = raw[:, :, 7]
    bid3 = raw[:, :, 8]; bsize3 = raw[:, :, 9]; ask3 = raw[:, :, 10]; asize3 = raw[:, :, 11]
    bid4 = raw[:, :, 12]; bsize4 = raw[:, :, 13]; ask4 = raw[:, :, 14]; asize4 = raw[:, :, 15]
    bid5 = raw[:, :, 16]; bsize5 = raw[:, :, 17]; ask5 = raw[:, :, 18]; asize5 = raw[:, :, 19]

    mid = (bid1 + ask1) / 2.0
    mid_ret = np.concatenate([np.zeros((N, 1), dtype=np.float32),
                               (mid[:, 1:] - mid[:, :-1])], axis=1)
    spread = ask1 - bid1
    eps = 1e-9
    imb1 = (bsize1 - asize1) / (bsize1 + asize1 + eps)
    imb2 = (bsize2 - asize2) / (bsize2 + asize2 + eps)
    imb3 = (bsize3 - asize3) / (bsize3 + asize3 + eps)
    imb4 = (bsize4 - asize4) / (bsize4 + asize4 + eps)
    imb5 = (bsize5 - asize5) / (bsize5 + asize5 + eps)
    tot_b = bsize1 + bsize2 + bsize3 + bsize4 + bsize5
    tot_a = asize1 + asize2 + asize3 + asize4 + asize5
    log_tot_b = np.log1p(np.maximum(tot_b, 0.0))
    log_tot_a = np.log1p(np.maximum(tot_a, 0.0))
    bsize_ratio = bsize1 / (bsize1 + asize1 + eps)

    # Stack extra features: (N, W, 11)
    extra = np.stack([mid, mid_ret, spread, imb1, imb2, imb3, imb4, imb5,
                       log_tot_b, log_tot_a, bsize_ratio], axis=2)
    return np.concatenate([raw, extra], axis=2)  # (N, W, 31)


def _window_normalize(x: np.ndarray, clip: float = 10.0) -> np.ndarray:
    """x: (N, W, F). Per-window per-feature standardize."""
    m = x.mean(axis=1, keepdims=True)
    s = x.std(axis=1, keepdims=True)
    out = (x - m) / np.maximum(s, 1e-8)
    return np.clip(out, -clip, clip).astype(np.float32)


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

        # Top-5 LOB column indices for Transformer input
        self._top5_idx = np.array(
            [self._raw_col_to_idx[c] for c in _TOP5_COLS], dtype=np.int64
        )

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

        # Raw transformer config
        rt_cfg = tcfg.get("raw_transformer", {})
        self._rt_enabled = bool(rt_cfg.get("enabled", False))
        self._rt_seeds = list(rt_cfg.get("seeds", []))

        # Load per-horizon LGB and NN ensembles
        self._lgb_lists: Dict[int, List[lgb.Booster]] = {}
        self._nn_lists: Dict[int, List] = {}  # T87 MLP (for fallback)
        self._weights: Dict[int, Tuple[float, float]] = {}
        for hcfg in self._horizons:
            if not hcfg.get("active", True):
                continue
            H = int(hcfg["h"])
            seeds = hcfg.get("ensemble_seeds", [])
            lgb_paths: List[str] = []
            for s in seeds:
                lp = os.path.join(here, f"model_h{H}_seed{s}.txt")
                if os.path.isfile(lp):
                    lgb_paths.append(lp)
            if lgb_paths:
                self._lgb_lists[H] = [lgb.Booster(model_file=p) for p in lgb_paths]
            self._weights[H] = (float(hcfg.get("w_nn", 1.0)),
                                float(hcfg.get("w_lgb", 1.0)))

        # Load Transformer NN models (torch CPU)
        self._tr_models: List[_TickTransformerTorch] = []
        if self._rt_enabled:
            for s in self._rt_seeds:
                pt_path = os.path.join(here, f"nn_raw_h60_seed{s}.pt")
                if os.path.isfile(pt_path):
                    m = _TickTransformerTorch(pt_path)
                    self._tr_models.append(m)
        if not self._tr_models:
            raise RuntimeError("No T184 Transformer models found! Expected nn_raw_h60_seed*.pt")

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

    def _compute_batch_features(self, batches: List[pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (schemeP_feats, X3d_raw) where schemeP_feats is (N, 359) and X3d_raw is (N, 100, 154)."""
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

        schemeP_feats = np.concatenate([raw_last, extras_kept], axis=1)
        return schemeP_feats, X3d.astype(np.float32)

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

    def _predict_transformer(self, x3d_raw: np.ndarray) -> np.ndarray:
        """x3d_raw: (N, W, K). Returns (N,) predictions from ensemble."""
        # Build v2 features (31 dims) from top5 LOB cols
        v2_window = _build_v2_features_from_raw(x3d_raw, self._top5_idx)  # (N, W, 31)
        # Window normalize
        v2_norm = _window_normalize(v2_window)  # (N, W, 31)
        x_t = torch.from_numpy(v2_norm)  # (N, W, 31)
        pred_acc = np.zeros(x_t.shape[0], dtype=np.float32)
        with torch.no_grad():
            for m in self._tr_models:
                p = m(x_t).numpy()
                pred_acc += p
        return (pred_acc / len(self._tr_models)).astype(np.float32)

    def predict(self, batches: List[pd.DataFrame]) -> List[List[int]]:
        if not batches:
            return []
        schemeP_feats, X3d_raw = self._compute_batch_features(batches)
        B = schemeP_feats.shape[0]
        out = np.ones((B, 5), dtype=np.int64)

        if self._cw_enabled:
            band_per_row = self._extract_band_per_row(batches)
        else:
            band_per_row = np.zeros(B, dtype=np.float64)

        # Transformer predictions (for all active horizons using nn)
        pred_tr = self._predict_transformer(X3d_raw)  # (B,)

        for hcfg in self._horizons:
            if not hcfg.get("active", True):
                continue
            H = int(hcfg["h"])
            lgbs = self._lgb_lists.get(H)
            w_nn, w_lgb = self._weights.get(H, (1.0, 1.0))

            preds = []
            ws = []
            if lgbs and w_lgb > 0:
                preds.append(self._ensemble_predict_lgb(lgbs, schemeP_feats))
                ws.append(w_lgb)
            if w_nn > 0 and self._tr_models:
                preds.append(pred_tr)
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
