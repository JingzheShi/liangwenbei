"""Predictor iter_019 v2N_Tr: T87 NN + T172 Transformer + T75 LGB 3-way ensemble.

Weights (w_nn, w_tr, w_lgb) and thresholds read from thresholds.json.
Per-sym beta conformal abstain band (same as v2 T150 tuning).
Transformer inference uses torch CPU (much faster than numpy for large batches).

CRITICAL_CONSTRAINTS compliance:
  - date never used; sym only for beta lookup; no cross-call state; W<=100 always
"""
from __future__ import annotations
import importlib.util, json, os
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


def _gelu(x: np.ndarray) -> np.ndarray:
    c = 0.7978845608028654
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x * x * x)))


def _ln(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    m = x.mean(axis=-1, keepdims=True)
    v = x.var(axis=-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


class _MLPNumpy:
    def __init__(self, p: str):
        d = np.load(p, allow_pickle=False)
        self.in_dim = int(d["in_dim"][0])
        self.hidden = list(d["hidden"].tolist())
        self.uln = bool(d["use_layernorm"][0])
        self.ts = float(d["target_scale"][0])
        self.fm = d["feat_mean"].astype(np.float32)
        self.fs = d["feat_std"].astype(np.float32)
        self.ki = d["keep_idx"].astype(np.int64)
        self.clip = float(d["clip"][0])
        self.W, self.b, self.LW, self.Lb = [], [], [], []
        for i in range(len(self.hidden)):
            self.W.append(d[f"L{i}_W"].astype(np.float32))
            self.b.append(d[f"L{i}_b"].astype(np.float32))
            if self.uln:
                self.LW.append(d[f"LN{i}_W"].astype(np.float32))
                self.Lb.append(d[f"LN{i}_b"].astype(np.float32))
        self.WF = d["LF_W"].astype(np.float32)
        self.bF = d["LF_b"].astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = X if X.shape[1] == self.in_dim else X[:, self.ki]
        Xs = np.clip(np.nan_to_num((Xs.astype(np.float32) - self.fm) / self.fs), -self.clip, self.clip)
        h = Xs
        for i in range(len(self.hidden)):
            h = h @ self.W[i].T + self.b[i]
            if self.uln:
                h = _ln(h, self.LW[i], self.Lb[i])
            h = _gelu(h)
        return ((h @ self.WF.T + self.bF).squeeze(-1) / self.ts).astype(np.float32)


class _TransformerTorch:
    """GroupTransformer inference using torch CPU (faster than numpy for large batches).
    Loaded from npz; uses no GPU — CPU-only torch for portability."""

    def __init__(self, p: str):
        d = np.load(p, allow_pickle=True)
        self.nl = int(d["n_layers"][0])
        self.nh = int(d["n_heads"][0])
        self.nt = int(d["n_tokens"][0])
        self.dt = int(d["d_token"][0])
        self.dh = self.dt // self.nh
        self.clip = float(d["clip"][0])
        self.ts = float(d["target_scale"][0])

        def t(k): return torch.from_numpy(d[k].astype(np.float32))
        self.fm  = t("feat_mean"); self.fs = t("feat_std")
        self.ki  = d["keep_idx"].astype(np.int64)
        self.pw  = t("proj_w"); self.pb = t("proj_b")
        self.pnw = t("proj_norm_w"); self.pnb = t("proj_norm_b")
        self.cls = t("cls_token")
        self.L = []
        for i in range(self.nl):
            self.L.append({
                "aiw": t(f"L{i}_attn_in_proj_w"),
                "aib": t(f"L{i}_attn_in_proj_b"),
                "aow": t(f"L{i}_attn_out_w"),
                "aob": t(f"L{i}_attn_out_b"),
                "f1w": t(f"L{i}_ffn1_w"), "f1b": t(f"L{i}_ffn1_b"),
                "f2w": t(f"L{i}_ffn2_w"), "f2b": t(f"L{i}_ffn2_b"),
                "n1w": t(f"L{i}_norm1_w"), "n1b": t(f"L{i}_norm1_b"),
                "n2w": t(f"L{i}_norm2_w"), "n2b": t(f"L{i}_norm2_b"),
            })
        self.hf1w = t("head_fc1_w"); self.hf1b = t("head_fc1_b")
        self.hlw  = t("head_ln_w");  self.hlb  = t("head_ln_b")
        self.hf2w = t("head_fc2_w"); self.hf2b = t("head_fc2_b")

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        B = X.shape[0]
        D = self.dt; H = self.nh; DH = self.dh; S = self.nt + 1; FF = 256
        Xs = X[:, self.ki] if X.shape[1] != len(self.ki) else X
        x = torch.from_numpy(Xs.astype(np.float32, copy=False))
        x = ((x - self.fm) / self.fs).nan_to_num_(0.0).clamp_(-self.clip, self.clip)
        tok = F.layer_norm(
            F.linear(x, self.pw, self.pb).reshape(B, self.nt, D),
            (D,), self.pnw, self.pnb,
        )
        cls = self.cls.expand(B, -1, -1)
        tok = torch.cat([cls, tok], dim=1)  # (B, S, D)
        for lyr in self.L:
            h = F.layer_norm(tok, (D,), lyr["n1w"], lyr["n1b"])
            qkv = F.linear(h.reshape(B * S, D), lyr["aiw"], lyr["aib"]).reshape(B, S, 3 * D)
            BH = B * H
            Q = qkv[:, :, :D].reshape(B, S, H, DH).permute(0, 2, 1, 3).reshape(BH, S, DH)
            K = qkv[:, :, D:2*D].reshape(B, S, H, DH).permute(0, 2, 1, 3).reshape(BH, S, DH)
            V = qkv[:, :, 2*D:].reshape(B, S, H, DH).permute(0, 2, 1, 3).reshape(BH, S, DH)
            scores = torch.bmm(Q, K.transpose(1, 2)) / (DH ** 0.5)
            attn_w = torch.softmax(scores, dim=-1)
            out = torch.bmm(attn_w, V).reshape(B, H, S, DH).permute(0, 2, 1, 3)
            attn_out = F.linear(out.reshape(B * S, D), lyr["aow"], lyr["aob"]).reshape(B, S, D)
            tok = tok + attn_out
            h = F.layer_norm(tok, (D,), lyr["n2w"], lyr["n2b"])
            h = F.gelu(F.linear(h.reshape(B * S, D), lyr["f1w"], lyr["f1b"])).reshape(B, S, FF)
            tok = tok + F.linear(h.reshape(B * S, FF), lyr["f2w"], lyr["f2b"]).reshape(B, S, D)
        h = F.gelu(F.linear(tok[:, 0, :], self.hf1w, self.hf1b))
        h = F.layer_norm(h, (D,), self.hlw, self.hlb)
        return (F.linear(h, self.hf2w, self.hf2b).squeeze(-1) / self.ts).numpy().astype(np.float32)


def _load_module(here: str, fname: str, mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(here, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Predictor:
    def __init__(self) -> None:
        here = os.path.dirname(os.path.abspath(__file__))
        self._ffb = _load_module(here, "fast_features_batch.py", "iter_018_ffb")
        self._raw: List[str] = list(RAW_COLS_TRAIN_ORDER)
        self._rcol = {c: i for i, c in enumerate(self._raw)}
        self._adi = self._raw.index("amount_delta")
        extra = list(self._ffb.all_feature_names())
        self._eki = np.array([i for i, n in enumerate(extra) if n not in FAIL_NAMES], dtype=np.int64)

        with open(os.path.join(here, "thresholds.json")) as f:
            tcfg = json.load(f)
        self._horizons: List[Dict] = tcfg["horizons"]

        cw = tcfg.get("conformal_wrapper", {"enabled": False})
        self._cwe = bool(cw.get("enabled", False))
        self._cwb: Dict[int, float] = {}
        self._cwbd = 0.0
        if self._cwe:
            psb = cw.get("per_sym_beta", {})
            pss = cw.get("per_sym_sigma", {})
            for k, b in psb.items():
                self._cwb[int(k)] = float(b) * float(pss.get(k, 0.0))
            self._cwbd = float(cw.get("default_beta_for_ood", 0.16)) * \
                         float(cw.get("default_sigma_for_ood", 4e-4))

        self._ci = dict(self._rcol)
        if "midprice1" in self._rcol:
            self._ci["midprice"] = self._rcol["midprice1"]

        self._lgbs: Dict[int, List[lgb.Booster]] = {}
        self._nns:  Dict[int, List[_MLPNumpy]] = {}
        self._trs:  Dict[int, List[_TransformerTorch]] = {}
        self._weights: Dict[int, Tuple[float, float, float]] = {}

        for hcfg in self._horizons:
            if not hcfg.get("active", True):
                continue
            H = int(hcfg["h"])
            seeds = hcfg.get("ensemble_seeds", [])
            lp, np_, tp = [], [], []
            for s in seeds:
                l = os.path.join(here, f"model_h{H}_seed{s}.txt")
                n = os.path.join(here, f"nn_h{H}_seed{s}.npz")
                t = os.path.join(here, f"nn_tr_h{H}_seed{s}.npz")
                if os.path.isfile(l):  lp.append(l)
                if os.path.isfile(n):  np_.append(n)
                if os.path.isfile(t):  tp.append(t)
            if lp:  self._lgbs[H] = [lgb.Booster(model_file=p) for p in lp]
            if np_: self._nns[H]  = [_MLPNumpy(p) for p in np_]
            if tp:  self._trs[H]  = [_TransformerTorch(p) for p in tp]
            self._weights[H] = (
                float(hcfg.get("w_nn",  1.0)),
                float(hcfg.get("w_lgb", 1.0)),
                float(hcfg.get("w_tr",  0.0)),
            )

    def _feats(self, batches: List[pd.DataFrame]) -> np.ndarray:
        N = len(batches); K = len(self._raw)
        X3 = np.empty((N, WINDOW, K), dtype=np.float64)
        for n, df in enumerate(batches):
            X3[n] = df[self._raw].to_numpy(dtype=np.float64, copy=False)
        rl = X3[:, -1, :].astype(np.float32, copy=True)
        if self._adi >= 0:
            v = rl[:, self._adi]
            rl[:, self._adi] = np.sign(v) * np.log1p(np.abs(v))
        ex = self._ffb.compute_batch_features(X3, self._ci)
        ex = np.where(np.isfinite(ex), ex, 0.0).astype(np.float32)
        return np.concatenate([rl, ex[:, self._eki]], axis=1)

    def _band(self, batches: List[pd.DataFrame]) -> np.ndarray:
        out = np.full(len(batches), self._cwbd, dtype=np.float64)
        for i, df in enumerate(batches):
            if "sym" not in df.columns:
                continue
            try:
                s = int(df["sym"].iloc[-1])
            except (ValueError, TypeError, IndexError):
                continue
            if s in self._cwb:
                out[i] = self._cwb[s]
        return out

    def predict(self, batches: List[pd.DataFrame]) -> List[List[int]]:
        if not batches:
            return []
        F = self._feats(batches)
        B = F.shape[0]
        out = np.ones((B, 5), dtype=np.int64)
        band = self._band(batches) if self._cwe else np.zeros(B, dtype=np.float64)

        for hcfg in self._horizons:
            if not hcfg.get("active", True):
                continue
            H = int(hcfg["h"])
            w_nn, w_lgb, w_tr = self._weights.get(H, (1.0, 1.0, 0.0))
            preds, ws = [], []

            if (L := self._lgbs.get(H)) and w_lgb > 0:
                acc = None
                for b in L:
                    p = b.predict(F).astype(np.float32)
                    acc = p if acc is None else acc + p
                preds.append(acc / len(L))
                ws.append(w_lgb)

            if (N := self._nns.get(H)) and w_nn > 0:
                acc = None
                for n in N:
                    p = n.predict(F)
                    acc = p if acc is None else acc + p
                preds.append(acc / len(N))
                ws.append(w_nn)

            if (T := self._trs.get(H)) and w_tr > 0:
                acc = None
                for t in T:
                    p = t.predict(F)
                    acc = p if acc is None else acc + p
                preds.append(acc / len(T))
                ws.append(w_tr)

            if not preds:
                continue

            wa = np.array(ws, dtype=np.float32).reshape(-1, 1)
            pred_dmid = (np.stack(preds, axis=0) * wa).sum(axis=0) / wa.sum()

            thr_up = float(hcfg.get("thr_up", 2e-4))
            thr_dn = float(hcfg.get("thr_dn", 2e-4))
            act = np.full(B, 1, dtype=np.int64)
            act[pred_dmid >  (thr_up + band)] = 2
            act[pred_dmid < -(thr_dn + band)] = 0
            out[:, HORIZON_TO_IDX[H]] = act

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
    df2 = df.copy(); df2["sym"] = 99
    print("OOD sym=99 ->", p.predict([df2])[0])
