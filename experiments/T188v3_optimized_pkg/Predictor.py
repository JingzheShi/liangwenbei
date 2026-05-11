"""Predictor for T188v3: T188v2 (50 NN + 50 LGB) with optimized NN ensemble inference.

T188v2 baseline: iter_018 v1 stack + per-sym beta abstain wrapper.
T188v3 change: _MLPNumpy sequential loop (50×) replaced by _BatchedMLPEnsemble — all 50
  NNs run in one batched torch bmm forward pass (CUDA if available, else CPU).
  Outputs match T188v2 within float32 precision (~1e-6 typical, 1e-4 max).

  pred_lgb = mean over 50 LightGBM regression boosters (T75 family)
  pred_nn  = mean over 50 small MLPs ([359 -> 256 -> 128 -> 64 -> 1], T87 SPO+ DFL)
  pred     = (w_nn * pred_nn + w_lgb * pred_lgb) / (w_nn + w_lgb), w_nn=1.0, w_lgb=1.5

per-sym calibrated abstain band on top of asymmetric EV gate:
  effective_thr_up = thr_up + beta_sym * sigma_sym
  effective_thr_dn = thr_dn + beta_sym * sigma_sym
  pred > effective_thr_up  -> action 2 (long)
  pred < -effective_thr_dn -> action 0 (short)
  else                      -> action 1 (flat)

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
import torch
import torch.nn.functional as F

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


class _BatchedMLPEnsemble:
    """All N MLPs of the same architecture run in one batched forward pass.

    Stacks weight matrices into (N, out, in) tensors and uses torch.bmm so all N
    networks execute in a single kernel call per layer. Auto-selects CUDA > CPU.

    Requirements:
      - All NNs share the same in_dim, hidden dims, and keep_idx.
      - Each may have different feat_mean / feat_std / weights.
    """

    def __init__(self, npz_paths: List[str], device: torch.device) -> None:
        self._device = device
        N = len(npz_paths)
        self.N = N

        d0 = np.load(npz_paths[0], allow_pickle=False)
        self.in_dim: int = int(d0["in_dim"][0])
        self.hidden: List[int] = list(d0["hidden"].tolist())
        self.use_layernorm: bool = bool(d0["use_layernorm"][0])
        self.keep_idx: np.ndarray = d0["keep_idx"].astype(np.int64)

        feat_means, feat_stds = [], []
        clips: List[float] = []
        target_scales: List[float] = []
        lW: List[List[np.ndarray]] = [[] for _ in self.hidden]
        lb: List[List[np.ndarray]] = [[] for _ in self.hidden]
        lLNW: List[List[np.ndarray]] = [[] for _ in self.hidden]
        lLNb: List[List[np.ndarray]] = [[] for _ in self.hidden]
        WF_list: List[np.ndarray] = []
        bF_list: List[np.ndarray] = []

        for path in npz_paths:
            d = np.load(path, allow_pickle=False)
            feat_means.append(d["feat_mean"].astype(np.float32))
            feat_stds.append(d["feat_std"].astype(np.float32))
            clips.append(float(d["clip"][0]))
            target_scales.append(float(d["target_scale"][0]))
            for i in range(len(self.hidden)):
                lW[i].append(d[f"L{i}_W"].astype(np.float32))
                lb[i].append(d[f"L{i}_b"].astype(np.float32))
                if self.use_layernorm:
                    lLNW[i].append(d[f"LN{i}_W"].astype(np.float32))
                    lLNb[i].append(d[f"LN{i}_b"].astype(np.float32))
            WF_list.append(d["LF_W"].astype(np.float32))  # (1, hidden_last)
            bF_list.append(d["LF_b"].astype(np.float32))  # (1,)

        def to_t(lst: List[np.ndarray]) -> torch.Tensor:
            return torch.from_numpy(np.stack(lst, 0)).to(device)

        self.feat_mean: torch.Tensor = to_t(feat_means)   # (N, in_dim)
        self.feat_std: torch.Tensor = to_t(feat_stds)     # (N, in_dim)
        self.clip: float = float(max(clips))
        self.target_scale: torch.Tensor = torch.tensor(
            target_scales, dtype=torch.float32, device=device
        ).unsqueeze(1)                                     # (N, 1)

        # W[i]: (N, out_dim, in_dim), b[i]: (N, out_dim)
        self.W: List[torch.Tensor] = [to_t(lW[i]) for i in range(len(self.hidden))]
        self.b: List[torch.Tensor] = [to_t(lb[i]) for i in range(len(self.hidden))]
        if self.use_layernorm:
            self.LNW: List[torch.Tensor] = [to_t(lLNW[i]) for i in range(len(self.hidden))]
            self.LNb: List[torch.Tensor] = [to_t(lLNb[i]) for i in range(len(self.hidden))]

        self.WF: torch.Tensor = to_t(WF_list)  # (N, 1, hidden_last)
        self.bF: torch.Tensor = to_t(bF_list)  # (N, 1)

    @torch.no_grad()
    def predict_mean(self, X_in: np.ndarray) -> np.ndarray:
        """Batched forward for all N NNs; returns mean over N, shape (B,)."""
        if X_in.shape[1] != self.in_dim:
            Xs = X_in[:, self.keep_idx].astype(np.float32)
        else:
            Xs = X_in.astype(np.float32, copy=True)

        # h: (B, in_dim) -> (N, B, in_dim)
        h = torch.from_numpy(Xs).to(self._device)
        h = h.unsqueeze(0).expand(self.N, -1, -1).contiguous()

        # Per-NN normalization: feat_mean/std (N, in_dim) -> (N, 1, in_dim) for broadcast
        h = (h - self.feat_mean.unsqueeze(1)) / self.feat_std.unsqueeze(1)
        h = torch.where(torch.isnan(h), torch.zeros_like(h), h)
        h = torch.clamp(h, -self.clip, self.clip)

        for i in range(len(self.hidden)):
            # W[i]: (N, out_dim, in_dim); bmm(h, W.T): (N, B, out_dim)
            h = torch.bmm(h, self.W[i].transpose(-1, -2)) + self.b[i].unsqueeze(1)
            if self.use_layernorm:
                # Manual layer norm: per-NN gamma/beta can't use F.layer_norm directly
                mean = h.mean(dim=-1, keepdim=True)
                var = h.var(dim=-1, keepdim=True, unbiased=False)
                h = (h - mean) / torch.sqrt(var + 1e-5)
                h = h * self.LNW[i].unsqueeze(1) + self.LNb[i].unsqueeze(1)
            h = F.gelu(h, approximate="tanh")

        # WF: (N, 1, hidden_last); bmm(h, WF.T): (N, B, 1) -> squeeze -> (N, B)
        out = torch.bmm(h, self.WF.transpose(-1, -2)).squeeze(-1)
        out = out + self.bF          # bF (N, 1) broadcasts over B
        out = out / self.target_scale  # (N, 1) broadcasts over B
        out = out.mean(0)            # mean over 50 NNs -> (B,)
        return out.cpu().numpy().astype(np.float32)


def _load_module(here: str, fname: str, mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(here, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Predictor:
    def __init__(self) -> None:
        here = os.path.dirname(os.path.abspath(__file__))
        self._ffb = _load_module(here, "fast_features_batch.py", "iter_018_ffb")

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        # Load per-horizon LGB ensembles (CPU sequential, already 84ms)
        # and batched NN ensembles (torch bmm, all 50 NNs in one pass)
        self._lgb_lists: Dict[int, List[lgb.Booster]] = {}
        self._nn_batched: Dict[int, _BatchedMLPEnsemble] = {}
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
                self._nn_batched[H] = _BatchedMLPEnsemble(nn_paths, self._device)
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

        for hcfg in self._horizons:
            if not hcfg.get("active", True):
                continue
            H = int(hcfg["h"])
            lgbs = self._lgb_lists.get(H)
            nn_ens = self._nn_batched.get(H)
            w_nn, w_lgb = self._weights.get(H, (1.0, 1.0))

            preds = []
            ws = []
            if lgbs and w_lgb > 0:
                preds.append(self._ensemble_predict_lgb(lgbs, feats))
                ws.append(w_lgb)
            if nn_ens is not None and w_nn > 0:
                preds.append(nn_ens.predict_mean(feats))
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
        rng.standard_normal((100, len(feats))).astype(np.float32),
        columns=feats,
    )
    df["sym"] = 1
    p = Predictor()
    print(f"Device: {p._device}")
    out = p.predict([df, df, df])
    print("smoke test predict ->", out[0], "len=", len(out))
    df_ood = df.copy()
    df_ood["sym"] = 99
    out_ood = p.predict([df_ood])
    print("OOD sym=99 ->", out_ood[0])
    df_no_sym = df.drop(columns=["sym"])
    out_no = p.predict([df_no_sym])
    print("missing sym ->", out_no[0])
