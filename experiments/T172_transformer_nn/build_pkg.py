"""Build T172 submission package: v2 + GroupTransformer ensemble.

Copies v2 pkg, adds nn_tr_h60_seed*.npz, updates Predictor.py and thresholds.json.
Output: experiments/T172_transformer_nn/pkg_v2T/
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
V2_PKG = os.path.join(ROOT, "experiments", "R_full_retrain", "pkg_iter019_v2")
PKG_OUT = os.path.join(HERE, "pkg_v2T")

TR_NPZ_DIR = os.environ.get("TR_NPZ_DIR", HERE)
W_TR = float(os.environ.get("W_TR", "1.0"))

SEEDS = [1, 7, 13, 42, 100]
HORIZON = 60


def main():
    # Verify all transformer npz exist
    npz_files = []
    for s in SEEDS:
        p = os.path.join(TR_NPZ_DIR, f"nn_tr_h{HORIZON}_seed{s}.npz")
        if not os.path.exists(p):
            print(f"ERROR: Missing {p}")
            sys.exit(1)
        npz_files.append((s, p))
    print(f"Found {len(npz_files)} GroupTransformer npz files")

    # Create pkg dir
    if os.path.exists(PKG_OUT):
        shutil.rmtree(PKG_OUT)
    shutil.copytree(V2_PKG, PKG_OUT)
    print(f"Copied v2 pkg to {PKG_OUT}")

    # Copy transformer npz
    for s, p in npz_files:
        dst = os.path.join(PKG_OUT, f"nn_tr_h{HORIZON}_seed{s}.npz")
        shutil.copy2(p, dst)
        print(f"  copied nn_tr_h{HORIZON}_seed{s}.npz")

    # Update thresholds.json to add w_tr weight
    thr_path = os.path.join(PKG_OUT, "thresholds.json")
    with open(thr_path) as f:
        tcfg = json.load(f)
    for hcfg in tcfg["horizons"]:
        if hcfg.get("h") == HORIZON:
            hcfg["w_tr"] = W_TR
            hcfg["transformer_seeds"] = SEEDS
    with open(thr_path, "w") as f:
        json.dump(tcfg, f, indent=2)
    print(f"Updated thresholds.json: w_tr={W_TR}")

    # Copy GroupTransformer numpy inference
    shutil.copy2(os.path.join(HERE, "ft_transformer_numpy.py"),
                 os.path.join(PKG_OUT, "group_transformer_numpy.py"))
    print("Copied group_transformer_numpy.py")

    # Write new Predictor.py
    predictor_path = os.path.join(PKG_OUT, "Predictor.py")
    write_predictor(predictor_path, W_TR, SEEDS, HORIZON)
    print("Wrote new Predictor.py")

    # Smoke test
    print("\nSmoke testing Predictor...")
    sys.path.insert(0, PKG_OUT)
    # Clear any cached modules
    for mod in list(sys.modules.keys()):
        if "Predictor" in mod or "fast_features" in mod or "group_transformer" in mod:
            del sys.modules[mod]

    import importlib
    import importlib.util
    spec = importlib.util.spec_from_file_location("Predictor", predictor_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import numpy as np
    import pandas as pd
    cfg = json.load(open(os.path.join(PKG_OUT, "config.json")))
    feats = cfg["feature"]
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.standard_normal((100, len(feats))).astype(np.float32), columns=feats)
    df["sym"] = 1
    p = mod.Predictor()
    out = p.predict([df, df, df])
    print(f"  smoke test: {out[0]}  len={len(out)}")
    assert len(out) == 3
    assert all(len(r) == 5 for r in out)
    print("  PASSED")

    # Build zip
    zip_name = f"submission_050909_iter019_v2T_transformer.zip"
    zip_path = os.path.join(HERE, zip_name)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fn in os.listdir(PKG_OUT):
            fp = os.path.join(PKG_OUT, fn)
            if os.path.isfile(fp) and not fn.startswith("__"):
                zf.write(fp, fn)
    size_mb = os.path.getsize(zip_path) / 1024 / 1024
    print(f"\nBuilt {zip_path} ({size_mb:.1f} MB)")
    return zip_path


def write_predictor(path, w_tr, seeds, H):
    code = f'''"""Predictor for T172: v2 (T87 MLP + T75 LGB) + GroupTransformer ensemble.

pred_lgb  = mean over 5 LightGBM regression boosters (T75, byte-identical to v2)
pred_nn   = mean over 5 small MLPs (T87, byte-identical to v2)
pred_tr   = mean over 5 GroupTransformer models (T172, feature-grouping + attention)
pred      = (w_nn*pred_nn + w_lgb*pred_lgb + w_tr*pred_tr) / (w_nn + w_lgb + w_tr)

Inference is pure numpy (no torch dependency).

CRITICAL_CONSTRAINTS compliance:
  - date never used
  - No cross-call state (each predict() call is independent)
  - sym only used for conformal band lookup (not fed to model forward)
  - sym-agnostic: global standardization, no per-sym models
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
HORIZON_TO_IDX = {{h: i for i, h in enumerate(HORIZON_LIST)}}

RAW_COLS_TRAIN_ORDER = (
    "open", "high", "low", "close", "volume_delta", "amount_delta",
    *(f"bid{{k}}" for k in range(1, 11)),
    *(f"bsize{{k}}" for k in range(1, 11)),
    *(f"ask{{k}}" for k in range(1, 11)),
    *(f"asize{{k}}" for k in range(1, 11)),
    "avgbid", "avgask", "totalbsize", "totalasize",
    "lb_intst", "la_intst", "mb_intst", "ma_intst", "cb_intst", "ca_intst",
    "lb_ind", "la_ind", "mb_ind", "ma_ind", "cb_ind", "ca_ind",
    "lb_acc", "la_acc", "mb_acc", "ma_acc", "cb_acc", "ca_acc",
    *(f"midprice{{k}}" for k in range(1, 11)),
    *(f"spread{{k}}" for k in range(1, 11)),
    *(f"bid_diff{{k}}" for k in range(1, 11)),
    *(f"ask_diff{{k}}" for k in range(1, 11)),
    "bid_mean", "ask_mean", "bsize_mean", "asize_mean", "cumspread", "imbalance",
    *(f"bid_rate{{k}}" for k in range(1, 11)),
    *(f"ask_rate{{k}}" for k in range(1, 11)),
    *(f"bsize_rate{{k}}" for k in range(1, 11)),
    *(f"asize_rate{{k}}" for k in range(1, 11)),
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


def _gelu_tanh(x):
    c = 0.7978845608028654
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x * x * x)))


def _layernorm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


class _MLPNumpy:
    def __init__(self, npz_path):
        d = np.load(npz_path, allow_pickle=False)
        self.in_dim = int(d["in_dim"][0])
        self.hidden = list(d["hidden"].tolist())
        self.use_layernorm = bool(d["use_layernorm"][0])
        self.target_scale = float(d["target_scale"][0])
        self.feat_mean = d["feat_mean"].astype(np.float32)
        self.feat_std = d["feat_std"].astype(np.float32)
        self.keep_idx = d["keep_idx"].astype(np.int64)
        self.clip = float(d["clip"][0])
        self.W, self.b, self.LN_W, self.LN_b = [], [], [], []
        for i in range(len(self.hidden)):
            self.W.append(d[f"L{{i}}_W"].astype(np.float32))
            self.b.append(d[f"L{{i}}_b"].astype(np.float32))
            if self.use_layernorm:
                self.LN_W.append(d[f"LN{{i}}_W"].astype(np.float32))
                self.LN_b.append(d[f"LN{{i}}_b"].astype(np.float32))
        self.WF = d["LF_W"].astype(np.float32)
        self.bF = d["LF_b"].astype(np.float32)

    def predict(self, X_in):
        Xs = X_in
        if Xs.shape[1] != self.in_dim:
            Xs = Xs[:, self.keep_idx]
        Xs = Xs.astype(np.float32, copy=True)
        Xs = (Xs - self.feat_mean) / self.feat_std
        Xs = np.where(np.isnan(Xs), 0.0, Xs)
        Xs = np.clip(Xs, -self.clip, self.clip).astype(np.float32)
        h = Xs
        for i in range(len(self.hidden)):
            h = h @ self.W[i].T + self.b[i]
            if self.use_layernorm:
                h = _layernorm(h, self.LN_W[i], self.LN_b[i])
            h = _gelu_tanh(h)
        return (h @ self.WF.T + self.bF).squeeze(-1) / self.target_scale


def _mha(x, in_w, in_b, out_w, out_b, n_heads):
    B, L, d = x.shape
    dh = d // n_heads
    qkv = (x.reshape(B * L, d) @ in_w.T + in_b).reshape(B, L, 3, d)
    Q = qkv[:,:,0].reshape(B, L, n_heads, dh).transpose(0,2,1,3)
    K = qkv[:,:,1].reshape(B, L, n_heads, dh).transpose(0,2,1,3)
    V = qkv[:,:,2].reshape(B, L, n_heads, dh).transpose(0,2,1,3)
    sc = np.float32(1.0 / np.sqrt(dh))
    s = (Q @ K.transpose(0,1,3,2)) * sc
    s -= s.max(-1, keepdims=True)
    a = np.exp(s); a /= a.sum(-1, keepdims=True)
    out = (a @ V).transpose(0,2,1,3).reshape(B * L, d)
    return (out @ out_w.T + out_b).reshape(B, L, d)


class _GroupTransformerNumpy:
    def __init__(self, npz_path):
        d = np.load(npz_path, allow_pickle=True)
        self.feat_dim = int(d["n_features"][0])
        self.n_tokens = int(d["n_tokens"][0])
        self.d_token = int(d["d_token"][0])
        self.n_heads = int(d["n_heads"][0])
        self.n_layers = int(d["n_layers"][0])
        self.target_scale = float(d["target_scale"][0])
        self.clip = float(d["clip"][0])
        self.feat_mean = d["feat_mean"].astype(np.float32)
        self.feat_std = d["feat_std"].astype(np.float32)
        self.keep_idx = d["keep_idx"].astype(np.int64)
        self.proj_w = d["proj_w"].astype(np.float32)
        self.proj_b = d["proj_b"].astype(np.float32)
        self.proj_norm_w = d["proj_norm_w"].astype(np.float32)
        self.proj_norm_b = d["proj_norm_b"].astype(np.float32)
        self.cls_token = d["cls_token"].astype(np.float32)
        self.head_ln_w = d["head_ln_w"].astype(np.float32)
        self.head_ln_b = d["head_ln_b"].astype(np.float32)
        self.head_fc1_w = d["head_fc1_w"].astype(np.float32)
        self.head_fc1_b = d["head_fc1_b"].astype(np.float32)
        self.head_fc2_w = d["head_fc2_w"].astype(np.float32)
        self.head_fc2_b = d["head_fc2_b"].astype(np.float32)
        self.layers = []
        for i in range(self.n_layers):
            self.layers.append({{k: d[f"L{{i}}_{{k}}"].astype(np.float32)
                                 for k in ["norm1_w","norm1_b","norm2_w","norm2_b",
                                           "attn_in_proj_w","attn_in_proj_b",
                                           "attn_out_w","attn_out_b",
                                           "ffn1_w","ffn1_b","ffn2_w","ffn2_b"]}})

    def predict(self, X_in):
        Xs = X_in.astype(np.float32, copy=False)
        if Xs.shape[1] != self.feat_dim:
            Xs = Xs[:, self.keep_idx]
        Xs = np.clip(np.where(np.isnan((Xs - self.feat_mean) / self.feat_std), 0.0,
                               (Xs - self.feat_mean) / self.feat_std), -self.clip, self.clip).astype(np.float32)
        B = Xs.shape[0]
        tokens = (Xs @ self.proj_w.T + self.proj_b).reshape(B, self.n_tokens, self.d_token)
        tokens = _layernorm(tokens, self.proj_norm_w, self.proj_norm_b)
        cls = np.broadcast_to(self.cls_token, (B, 1, self.d_token)).copy().astype(np.float32)
        seq = np.concatenate([cls, tokens], axis=1)
        for L in self.layers:
            n = _layernorm(seq, L["norm1_w"], L["norm1_b"])
            seq = seq + _mha(n, L["attn_in_proj_w"], L["attn_in_proj_b"],
                             L["attn_out_w"], L["attn_out_b"], self.n_heads)
            n = _layernorm(seq, L["norm2_w"], L["norm2_b"])
            f = _gelu_tanh(n.reshape(-1, self.d_token) @ L["ffn1_w"].T + L["ffn1_b"])
            seq = seq + (f @ L["ffn2_w"].T + L["ffn2_b"]).reshape(B, -1, self.d_token)
        cls_out = _layernorm(seq[:, 0], self.head_ln_w, self.head_ln_b)
        cls_out = _gelu_tanh(cls_out @ self.head_fc1_w.T + self.head_fc1_b)
        return ((cls_out @ self.head_fc2_w.T + self.head_fc2_b).squeeze(-1) / self.target_scale).astype(np.float32)


def _load_module(here, fname, mod_name):
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(here, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Predictor:
    def __init__(self):
        here = os.path.dirname(os.path.abspath(__file__))
        self._ffb = _load_module(here, "fast_features_batch.py", "iter019_ffb")
        self._raw_feat_cols = list(RAW_COLS_TRAIN_ORDER)
        self._raw_col_to_idx = {{c: i for i, c in enumerate(self._raw_feat_cols)}}
        self._amount_delta_idx = self._raw_feat_cols.index("amount_delta")
        self._extra_names = list(self._ffb.all_feature_names())
        self._extra_keep_idx = np.array(
            [i for i, n in enumerate(self._extra_names) if n not in FAIL_NAMES], dtype=np.int64)

        thresholds_path = os.path.join(here, "thresholds.json")
        with open(thresholds_path) as f:
            tcfg = json.load(f)
        self._horizons = tcfg["horizons"]

        cw = tcfg.get("conformal_wrapper", {{"enabled": False}})
        self._cw_enabled = bool(cw.get("enabled", False))
        self._cw_band: Dict[int, float] = {{}}
        self._cw_default_band = 0.0
        if self._cw_enabled:
            psb = cw.get("per_sym_beta", {{}})
            pss = cw.get("per_sym_sigma", {{}})
            for k_str, b in psb.items():
                k = int(k_str)
                s = float(pss.get(k_str, 0.0))
                self._cw_band[k] = float(b) * s
            self._cw_default_band = float(cw.get("default_beta_for_ood", 0.16)) * \\
                                     float(cw.get("default_sigma_for_ood", 4.0e-4))

        self._col_idx = dict(self._raw_col_to_idx)
        if "midprice1" in self._raw_col_to_idx:
            self._col_idx["midprice"] = self._raw_col_to_idx["midprice1"]

        self._lgb_lists: Dict[int, List[lgb.Booster]] = {{}}
        self._nn_lists: Dict[int, List[_MLPNumpy]] = {{}}
        self._tr_lists: Dict[int, List[_GroupTransformerNumpy]] = {{}}
        self._weights: Dict[int, Tuple[float, float, float]] = {{}}

        for hcfg in self._horizons:
            if not hcfg.get("active", True):
                continue
            H = int(hcfg["h"])
            seeds = hcfg.get("ensemble_seeds", [])
            lgb_paths, nn_paths, tr_paths = [], [], []
            for s in seeds:
                lp = os.path.join(here, f"model_h{{H}}_seed{{s}}.txt")
                np_ = os.path.join(here, f"nn_h{{H}}_seed{{s}}.npz")
                tp = os.path.join(here, f"nn_tr_h{{H}}_seed{{s}}.npz")
                if os.path.isfile(lp): lgb_paths.append(lp)
                if os.path.isfile(np_): nn_paths.append(np_)
                if os.path.isfile(tp): tr_paths.append(tp)
            if lgb_paths:
                self._lgb_lists[H] = [lgb.Booster(model_file=p) for p in lgb_paths]
            if nn_paths:
                self._nn_lists[H] = [_MLPNumpy(p) for p in nn_paths]
            if tr_paths:
                self._tr_lists[H] = [_GroupTransformerNumpy(p) for p in tr_paths]
            self._weights[H] = (float(hcfg.get("w_nn", 1.0)),
                                float(hcfg.get("w_lgb", 1.0)),
                                float(hcfg.get("w_tr", 0.0)))

    def _gate_with_band(self, pred, hcfg, band):
        thr_up = float(hcfg.get("thr_up", 2e-4))
        thr_dn = float(hcfg.get("thr_dn", 2e-4))
        out = np.full(pred.shape[0], 1, dtype=np.int64)
        out[pred > (thr_up + band)] = 2
        out[pred < -(thr_dn + band)] = 0
        return out

    @staticmethod
    def _ev_gate(pred, hcfg):
        thr_up = float(hcfg.get("thr_up", 2e-4))
        thr_dn = float(hcfg.get("thr_dn", 2e-4))
        out = np.full(pred.shape[0], 1, dtype=np.int64)
        out[pred > thr_up] = 2
        out[pred < -thr_dn] = 0
        return out

    def _compute_batch_features(self, batches):
        N = len(batches)
        K = len(self._raw_feat_cols)
        X3d = np.empty((N, WINDOW, K), dtype=np.float64)
        for n, df in enumerate(batches):
            X3d[n] = df[self._raw_feat_cols].to_numpy(dtype=np.float64, copy=False)
        raw_last = X3d[:, -1, :].astype(np.float32, copy=True)
        v = raw_last[:, self._amount_delta_idx]
        raw_last[:, self._amount_delta_idx] = np.sign(v) * np.log1p(np.abs(v))
        extras_full = self._ffb.compute_batch_features(X3d, self._col_idx)
        extras_full = np.where(np.isfinite(extras_full), extras_full, 0.0).astype(np.float32)
        extras_kept = extras_full[:, self._extra_keep_idx]
        return np.concatenate([raw_last, extras_kept], axis=1)

    def _extract_band_per_row(self, batches):
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
    def _ensemble_mean(models, X, method="predict"):
        if len(models) == 1:
            return models[0].predict(X).astype(np.float32)
        acc = None
        for m in models:
            p = m.predict(X).astype(np.float32)
            acc = p if acc is None else acc + p
        return acc / float(len(models))

    def predict(self, batches):
        if not batches:
            return []
        feats = self._compute_batch_features(batches)
        B = feats.shape[0]
        out = np.ones((B, 5), dtype=np.int64)

        band_per_row = (self._extract_band_per_row(batches)
                        if self._cw_enabled else np.zeros(B, dtype=np.float64))

        for hcfg in self._horizons:
            if not hcfg.get("active", True):
                continue
            H = int(hcfg["h"])
            lgbs = self._lgb_lists.get(H)
            nns = self._nn_lists.get(H)
            trs = self._tr_lists.get(H)
            w_nn, w_lgb, w_tr = self._weights.get(H, (1.0, 1.0, 0.0))

            preds, ws = [], []
            if lgbs and w_lgb > 0:
                preds.append(self._ensemble_mean(lgbs, feats)); ws.append(w_lgb)
            if nns and w_nn > 0:
                preds.append(self._ensemble_mean(nns, feats)); ws.append(w_nn)
            if trs and w_tr > 0:
                preds.append(self._ensemble_mean(trs, feats)); ws.append(w_tr)
            if not preds:
                continue

            stacked = np.stack(preds, axis=0)
            ws_arr = np.array(ws, dtype=np.float32).reshape(-1, 1)
            pred_dmid = (stacked * ws_arr).sum(0) / ws_arr.sum()

            if self._cw_enabled:
                actions = self._gate_with_band(pred_dmid, hcfg, band_per_row)
            else:
                actions = self._ev_gate(pred_dmid, hcfg)
            out[:, HORIZON_TO_IDX[H]] = actions

        return out.tolist()


if __name__ == "__main__":
    import json, numpy as np, pandas as pd
    here = os.path.dirname(os.path.abspath(__file__))
    cfg = json.load(open(os.path.join(here, "config.json")))
    feats = cfg["feature"]
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.standard_normal((100, len(feats))).astype(np.float32), columns=feats)
    df["sym"] = 1
    p = Predictor()
    out = p.predict([df, df, df])
    print("smoke test:", out[0], "len:", len(out))
'''
    with open(path, "w") as f:
        f.write(code)


if __name__ == "__main__":
    main()
