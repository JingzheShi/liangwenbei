"""Predictor T169 V2_sigmoid = v2 + per-row AR(5) TTT confidence scaling (T127): iter_015 v1 stack + per-sym beta abstain wrapper.

iter_018 = iter_015 v1 (BYTE-IDENTICAL T87 NN + T75 LGB) + post-hoc conformal-style
abstain band layer over the existing DE-tuned thresholds.

  pred_lgb = mean over 5 LightGBM regression boosters (T75, byte-identical to iter_015 v1)
  pred_nn  = mean over 5 small MLPs ([359 -> 256 -> 128 -> 64 -> 1], T87 SPO+ DFL,
             byte-identical to iter_015 v1)
  pred     = (w_nn * pred_nn + w_lgb * pred_lgb) / (w_nn + w_lgb), w_nn=1.0, w_lgb=1.5

NEW in iter_018: per-sym calibrated abstain band on top of asymmetric EV gate:
  effective_thr_up = thr_up + beta_sym * sigma_sym
  effective_thr_dn = thr_dn + beta_sym * sigma_sym
  pred > effective_thr_up  -> action 2 (long)
  pred < -effective_thr_dn -> action 0 (short)
  else                      -> action 1 (flat)

beta values from R_conformal_select 4-fold CV consensus / V8 in-sample fit:
  sym 0: beta=0.10  sym 1: beta=0.40  sym 2: beta=0.30  sym 3: beta=0.00  sym 4: beta=0.00
sigma values precomputed from std(combined_pred) per sym on local val 442k.
For OOD sym IDs (not in 0..4), fall back to mean beta (0.16) and mean sigma.

NN inference is in pure numpy (no torch dependency).

Compliance with CRITICAL_CONSTRAINTS.md:
  - sym is read from input DataFrame ONLY for per-sym beta lookup (allowed: not fed
    to model.forward; sym ID range 0..4 per platform spec, fallback for OOD IDs)
  - date never used
  - No cross-call state held on `self` (each predict() call is independent)
  - sym-agnostic FORWARD: NN/LGB features only use 100-tick LOB window; standardization
    statistics are GLOBAL (computed from train data, no per-sym)
  - W <= 100 for every rolling feature
  - Stateless / shuffle-invariant: no row order dependency, no buffers
"""
from __future__ import annotations

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

        # Load per-horizon LGB and NN ensembles (byte-identical to iter_015 v1)
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
        """Look up beta * sigma per row using sym from the LAST tick of each window.
        Falls back to the OOD-default band if sym not in 0..4 or 'sym' column missing."""
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


    # ── T169: AR(5) in-row TTT ────────────────────────────────────────────────

    @staticmethod
    def _compute_ar5_confidence(batches):
        """
        Vectorized per-row AR(5) confidence (local R²) from 100-tick mid-price window.
        Self-contained per row; no cross-row or cross-call state.
        """
        import numpy as _np
        N = len(batches)
        T = 100

        mid_matrix = _np.empty((N, T), dtype=_np.float64)
        for i, df in enumerate(batches):
            v = df["midprice1"].values
            mid_matrix[i] = v[:T].astype(_np.float64)

        price = mid_matrix + 1.0
        price = _np.where(price > 1e-8, price, 1e-8)
        log_r = _np.diff(_np.log(price), axis=1)  # (N, 99)

        y_train = log_r[:, 5:95]  # (N, 90)
        X_train = _np.stack([
            log_r[:, 4:94], log_r[:, 3:93], log_r[:, 2:92],
            log_r[:, 1:91], log_r[:, 0:90],
        ], axis=2)  # (N, 90, 5)

        eps = 1e-12
        XtX = X_train.transpose(0, 2, 1) @ X_train  # (N, 5, 5)
        XtX += eps * _np.eye(5)[None, :, :]
        Xty = (X_train.transpose(0, 2, 1) @ y_train[:, :, None]).squeeze(-1)  # (N, 5)
        coef = _np.linalg.solve(XtX, Xty[:, :, None]).squeeze(-1)  # (N, 5)

        y_test = log_r[:, 94:99]  # (N, 5)
        X_test = _np.stack([
            log_r[:, 93:98], log_r[:, 92:97], log_r[:, 91:96],
            log_r[:, 90:95], log_r[:, 89:94],
        ], axis=2)  # (N, 5, 5)

        y_pred = (X_test @ coef[:, :, None]).squeeze(-1)  # (N, 5)
        ss_res = ((y_test - y_pred) ** 2).sum(axis=1)
        y_test_mean = y_test.mean(axis=1, keepdims=True)
        ss_tot = ((y_test - y_test_mean) ** 2).sum(axis=1)
        ss_tot = _np.where(ss_tot < 1e-20, 1e-20, ss_tot)

        local_R2 = 1.0 - ss_res / ss_tot
        return _np.clip(local_R2, -5.0, 1.0)

    @staticmethod
    def _ar5_scale(pred_dmid, local_R2):
        """V2: continuous sigmoid — factor = sigmoid(R2 * 5 - 1)."""
        import numpy as _np
        factor = 1.0 / (1.0 + _np.exp(-(local_R2 * 5.0 - 1.0)))
        return pred_dmid * factor

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
        feats = self._compute_batch_features(batches)
        B = feats.shape[0]
        out = np.ones((B, 5), dtype=np.int64)

        if self._cw_enabled:
            band_per_row = self._extract_band_per_row(batches)
        else:
            band_per_row = np.zeros(B, dtype=np.float64)

        _ar5_local_R2 = self._compute_ar5_confidence(batches)
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
            stacked = np.stack(preds, axis=0)  # (M, B)
            ws_arr = np.array(ws, dtype=np.float32).reshape(-1, 1)
            pred_dmid = (stacked * ws_arr).sum(axis=0) / ws_arr.sum()

            pred_dmid = self._ar5_scale(pred_dmid, _ar5_local_R2)
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
