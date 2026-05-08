"""T93: NN architecture / preprocessing exploration for ensemble diversity vs T81 MLP.

Architectures:
  * mlp     -- T81-equivalent baseline [256,128,64], LayerNorm, GELU, Dropout
  * ft      -- FT-Transformer: per-feature linear tokenizer + CLS + transformer encoder
  * mixer   -- 1D MLP-Mixer over feature axis (no attention)
  * autoint -- multi-head self-attn (no FFN), residual stack on token embeddings
  * resmlp  -- pre-LN residual MLP (deeper than T81), different inductive bias
  * gmlp    -- gated MLP with spatial gating over feature tokens (lightweight)

Preprocessing modes (--prep):
  * standard        -- (x - mean) / std, clip ±10  (T81 default)
  * rankgauss       -- sklearn QuantileTransformer output='normal' (clip ±10)
  * winsorize_p1    -- clip per-feat to (1%, 99%) percentile, then standardize
  * winsorize_p5    -- clip per-feat to (5%, 95%) percentile, then standardize

Identical to T81 elsewhere:
  * V4 split (train date 0-75, val 76-79)
  * 359 features (drop 11 fail names)
  * y_regr = (mp_th - mp_t) / (mp_t + 1)  (Δmid_norm)
  * weighted L2 with class-balanced sample weights
  * aug_a concat (per-(sample, feat) U[0.8, 1.2])
  * target_scale = 1 / std(y_regr_train) auto

CRITICAL_CONSTRAINTS compliance: forward() takes only X; global standardization stats;
no sym/date/time features.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

T26_DIR = os.path.join(ROOT, "experiments", "T26_domain_randomization")
sys.path.insert(0, T26_DIR)
from build_aug import class_balanced_weight  # noqa: E402

T68_DIR = os.path.join(ROOT, "experiments", "T68_stage5_features")
CACHE_DIR = os.path.join(T68_DIR, "cache")

NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001
SEED_LIST = (42, 1, 7, 13, 100)

T59_FAIL_NAMES = [
    "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
    "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
    "qrank_W100_cumspread",
    "kyle_lam_W50", "kyle_lam_W100",
    "roll_eff_spr_ratio_W100",
]
STAGE5_FAIL_NAMES = ["liq_asym_top5_W5"]
DROP_NAMES = T59_FAIL_NAMES + STAGE5_FAIL_NAMES


def progress(name, step, **extra):
    p = os.path.join(HERE, f"worker-progress_{name}.json")
    payload = {"status": "running", "step": step,
               "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               **extra}
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


def load_split(split):
    p = os.path.join(CACHE_DIR, f"schemeP_{split}.npz")
    print(f"  loading {p} ...", flush=True)
    d = np.load(p)
    return {k: d[k] for k in d.files}


def regr_target(mp_t, mp_th):
    return ((mp_th.astype(np.float64) - mp_t.astype(np.float64))
            / (mp_t.astype(np.float64) + 1.0)).astype(np.float32)


# ============================================================
# Preprocessing
# ============================================================

def fit_standard(X, mode):
    """X is (n, d), float32, may contain NaN. Returns dict of stats."""
    if mode == "standard":
        m = np.nanmean(X, axis=0).astype(np.float32)
        s = np.maximum(np.nanstd(X, axis=0).astype(np.float32), 1e-6)
        return {"mode": "standard", "mean": m, "std": s, "clip": 10.0}
    if mode in ("winsorize_p1", "winsorize_p5"):
        q_lo = 0.01 if mode == "winsorize_p1" else 0.05
        q_hi = 1.0 - q_lo
        # nanquantile on each column
        lo = np.nanquantile(X, q_lo, axis=0).astype(np.float32)
        hi = np.nanquantile(X, q_hi, axis=0).astype(np.float32)
        # apply winsor then compute mean/std on winsorized
        Xw = np.clip(X, lo, hi)
        m = np.nanmean(Xw, axis=0).astype(np.float32)
        s = np.maximum(np.nanstd(Xw, axis=0).astype(np.float32), 1e-6)
        return {"mode": mode, "lo": lo, "hi": hi, "mean": m, "std": s, "clip": 10.0}
    if mode == "rankgauss":
        # Per-feature: rank-based gaussian transform. Manual implementation
        # avoids sklearn dependency on cache size and is faster.
        # For each column: argsort values, replace with normal CDF inverse of (rank+0.5)/n.
        n, d = X.shape
        # First impute NaN with column median (rank-aware)
        X_imp = X.copy()
        for j in range(d):
            col = X_imp[:, j]
            mask = np.isnan(col)
            if mask.any():
                col[mask] = np.nanmedian(col)
                X_imp[:, j] = col
        # We'll store the sorted reference values per column and the corresponding
        # gaussian quantile, so test transform can interpolate.
        n_quantiles = min(n, 2000)  # subsample for storage
        # Use uniform sample of n_quantiles ranks
        idx_q = np.linspace(0, n - 1, n_quantiles).astype(np.int64)
        sorted_X = np.sort(X_imp, axis=0)  # (n, d)
        ref_values = sorted_X[idx_q]  # (n_quantiles, d)
        # Gaussian quantiles for the same ranks
        from scipy.special import erfinv
        u = (idx_q + 0.5) / n  # (n_quantiles,)
        gauss_q = (np.sqrt(2.0) * erfinv(2.0 * u - 1.0)).astype(np.float32)
        # Clip to ±5 to avoid infinities at extremes
        gauss_q = np.clip(gauss_q, -5.0, 5.0)
        return {
            "mode": "rankgauss", "ref_values": ref_values.astype(np.float32),
            "gauss_q": gauss_q.astype(np.float32),
            "median": np.array([np.nanmedian(X[:, j]) for j in range(d)], dtype=np.float32),
            "clip": 5.0,
        }
    raise ValueError(mode)


def apply_prep(X, stats):
    """X (n, d). Returns standardized & clipped float32."""
    mode = stats["mode"]
    if mode == "standard":
        m = stats["mean"]; s = stats["std"]
        Xs = (X - m) / s
        Xs = np.where(np.isnan(Xs), 0.0, Xs)
        Xs = np.clip(Xs, -stats["clip"], stats["clip"])
        return Xs.astype(np.float32)
    if mode in ("winsorize_p1", "winsorize_p5"):
        Xc = np.clip(X, stats["lo"], stats["hi"])
        m = stats["mean"]; s = stats["std"]
        Xs = (Xc - m) / s
        Xs = np.where(np.isnan(Xs), 0.0, Xs)
        Xs = np.clip(Xs, -stats["clip"], stats["clip"])
        return Xs.astype(np.float32)
    if mode == "rankgauss":
        # impute NaN with median
        Xi = X.copy()
        med = stats["median"]
        nan_mask = np.isnan(Xi)
        if nan_mask.any():
            for j in range(Xi.shape[1]):
                col_m = nan_mask[:, j]
                if col_m.any():
                    Xi[col_m, j] = med[j]
        ref = stats["ref_values"]  # (Q, d)
        gq = stats["gauss_q"]      # (Q,)
        # Vectorized per-column linear interp using np.searchsorted
        n, d = Xi.shape
        out = np.empty_like(Xi, dtype=np.float32)
        for j in range(d):
            xp = ref[:, j]
            # Ensure monotonically increasing; if duplicates, np.interp still ok.
            out[:, j] = np.interp(Xi[:, j], xp, gq).astype(np.float32)
        out = np.clip(out, -stats["clip"], stats["clip"])
        return out.astype(np.float32)
    raise ValueError(mode)


# ============================================================
# Architectures
# ============================================================

class MLPRegr(nn.Module):
    """T81 baseline: [in -> hidden -> 1] with LayerNorm + GELU + Dropout."""
    def __init__(self, in_dim, hidden=(256, 128, 64), dropout=0.1):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
        self._init()
    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.net(x).squeeze(-1)


class FTTransformer(nn.Module):
    """FT-Transformer (Gorishniy 2021).
    Each feature -> its own learned linear embedding [B, n, d_token]. Add [CLS], pass
    through pre-norm transformer encoder, predict from [CLS].
    """
    def __init__(self, n_features, d_token=32, n_heads=4, depth=2, dropout=0.1, ffn_mult=2):
        super().__init__()
        self.n_features = n_features
        self.d_token = d_token
        # per-feature linear: weight (n, d), bias (n, d) -- equivalent to applying
        # a separate Linear(1, d) to each feature scalar.
        self.tok_w = nn.Parameter(torch.randn(n_features, d_token) * (1.0 / math.sqrt(d_token)))
        self.tok_b = nn.Parameter(torch.zeros(n_features, d_token))
        self.cls = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads,
            dim_feedforward=int(d_token * ffn_mult),
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, 1),
        )
    def forward(self, x):
        # x: (B, n)
        emb = x.unsqueeze(-1) * self.tok_w + self.tok_b  # (B, n, d)
        cls = self.cls.expand(x.size(0), -1, -1)
        seq = torch.cat([cls, emb], dim=1)  # (B, n+1, d)
        out = self.encoder(seq)
        return self.head(out[:, 0]).squeeze(-1)


class MixerBlock(nn.Module):
    def __init__(self, n_tokens, d_token, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        h_t = max(int(n_tokens * mlp_ratio), 16)
        self.token_mlp = nn.Sequential(
            nn.Linear(n_tokens, h_t), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h_t, n_tokens), nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_token)
        h_c = int(d_token * mlp_ratio)
        self.channel_mlp = nn.Sequential(
            nn.Linear(d_token, h_c), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h_c, d_token), nn.Dropout(dropout),
        )
    def forward(self, x):
        # x: (B, n, d)
        y = self.norm1(x)
        y = y.transpose(1, 2)  # (B, d, n)
        y = self.token_mlp(y)
        y = y.transpose(1, 2)  # (B, n, d)
        x = x + y
        x = x + self.channel_mlp(self.norm2(x))
        return x


class TabMixer(nn.Module):
    """1D MLP-Mixer over feature tokens. n_features per-feat embedding then alternating
    token-mixing MLP / channel-mixing MLP. Final mean over token axis -> head.
    """
    def __init__(self, n_features, d_token=16, depth=4, dropout=0.1, mlp_ratio=2.0):
        super().__init__()
        self.tok_w = nn.Parameter(torch.randn(n_features, d_token) * (1.0 / math.sqrt(d_token)))
        self.tok_b = nn.Parameter(torch.zeros(n_features, d_token))
        self.blocks = nn.ModuleList([
            MixerBlock(n_features, d_token, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(
            nn.Linear(d_token, d_token), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_token, 1),
        )
    def forward(self, x):
        emb = x.unsqueeze(-1) * self.tok_w + self.tok_b  # (B, n, d)
        for blk in self.blocks:
            emb = blk(emb)
        emb = self.norm(emb).mean(dim=1)
        return self.head(emb).squeeze(-1)


class AutoInt(nn.Module):
    """AutoInt-style: per-feat token + stack of multi-head self-attention residual
    layers (no FFN). Lighter than full FT-Transformer; classic tabular attention."""
    def __init__(self, n_features, d_token=16, n_heads=2, depth=2, dropout=0.1):
        super().__init__()
        self.tok_w = nn.Parameter(torch.randn(n_features, d_token) * (1.0 / math.sqrt(d_token)))
        self.tok_b = nn.Parameter(torch.zeros(n_features, d_token))
        self.attns = nn.ModuleList([
            nn.MultiheadAttention(d_token, n_heads, dropout=dropout, batch_first=True)
            for _ in range(depth)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_token) for _ in range(depth)])
        self.dropout = nn.Dropout(dropout)
        # Pool over feature axis -> head
        self.head = nn.Sequential(
            nn.LayerNorm(d_token * n_features),
            nn.Linear(d_token * n_features, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
    def forward(self, x):
        emb = x.unsqueeze(-1) * self.tok_w + self.tok_b  # (B, n, d)
        for attn, norm in zip(self.attns, self.norms):
            y = norm(emb)
            y, _ = attn(y, y, y, need_weights=False)
            emb = emb + self.dropout(y)
        out = emb.flatten(1)  # (B, n*d)
        return self.head(out).squeeze(-1)


class ResMLPBlock(nn.Module):
    def __init__(self, d, dropout=0.1, mlp_ratio=2.0):
        super().__init__()
        h = int(d * mlp_ratio)
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, h)
        self.fc2 = nn.Linear(h, d)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        y = self.norm(x)
        y = self.fc2(self.drop(self.act(self.fc1(y))))
        return x + self.drop(y)


class ResMLP(nn.Module):
    """Pre-norm residual MLP, deeper than T81. Different inductive bias via skip
    connections (no transformer, no token axis)."""
    def __init__(self, in_dim, d=256, depth=4, dropout=0.1, mlp_ratio=2.0):
        super().__init__()
        self.proj = nn.Linear(in_dim, d)
        self.blocks = nn.ModuleList([
            ResMLPBlock(d, dropout=dropout, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)
    def forward(self, x):
        h = self.proj(x)
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.norm(h)).squeeze(-1)


class GatedMLP(nn.Module):
    """gMLP-lite: per-feat token, then SGU-style spatial gating over feature axis.
    Cheaper than self-attn, different inductive bias from FT-Transformer."""
    def __init__(self, n_features, d_token=16, depth=2, dropout=0.1, mlp_ratio=2.0):
        super().__init__()
        self.tok_w = nn.Parameter(torch.randn(n_features, d_token) * (1.0 / math.sqrt(d_token)))
        self.tok_b = nn.Parameter(torch.zeros(n_features, d_token))
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            blk = nn.ModuleDict({
                "norm": nn.LayerNorm(d_token),
                "fc_in": nn.Linear(d_token, int(d_token * mlp_ratio * 2)),
                "spatial": nn.Linear(n_features, n_features),
                "fc_out": nn.Linear(int(d_token * mlp_ratio), d_token),
                "drop": nn.Dropout(dropout),
            })
            self.blocks.append(blk)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(
            nn.Linear(d_token, d_token), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_token, 1),
        )
        self._n = n_features
    def forward(self, x):
        emb = x.unsqueeze(-1) * self.tok_w + self.tok_b  # (B, n, d)
        for blk in self.blocks:
            y = blk["norm"](emb)
            y = blk["fc_in"](y)  # (B, n, 2*h)
            u, v = y.chunk(2, dim=-1)  # (B, n, h) each
            v = blk["spatial"](v.transpose(1, 2)).transpose(1, 2)  # mix tokens
            y = u * v  # gating
            y = blk["fc_out"](y)
            emb = emb + blk["drop"](y)
        emb = self.norm(emb).mean(dim=1)
        return self.head(emb).squeeze(-1)


def build_model(arch, in_dim, args):
    if arch == "mlp":
        hidden = tuple(int(x) for x in args.mlp_hidden.split(","))
        return MLPRegr(in_dim, hidden=hidden, dropout=args.dropout)
    if arch == "ft":
        return FTTransformer(in_dim, d_token=args.d_token, n_heads=args.n_heads,
                             depth=args.depth, dropout=args.dropout, ffn_mult=args.ffn_mult)
    if arch == "mixer":
        return TabMixer(in_dim, d_token=args.d_token, depth=args.depth,
                        dropout=args.dropout, mlp_ratio=args.ffn_mult)
    if arch == "autoint":
        return AutoInt(in_dim, d_token=args.d_token, n_heads=args.n_heads,
                       depth=args.depth, dropout=args.dropout)
    if arch == "resmlp":
        return ResMLP(in_dim, d=args.resmlp_d, depth=args.depth,
                      dropout=args.dropout, mlp_ratio=args.ffn_mult)
    if arch == "gmlp":
        return GatedMLP(in_dim, d_token=args.d_token, depth=args.depth,
                        dropout=args.dropout, mlp_ratio=args.ffn_mult)
    raise ValueError(arch)


# ============================================================
# Train loop
# ============================================================

def predict_chunked(model, X_t, batch=8192, device="cuda", on_gpu=False, amp=False):
    model.eval()
    out = np.zeros(len(X_t), dtype=np.float32)
    amp_dtype = torch.bfloat16
    with torch.no_grad():
        for s in range(0, len(X_t), batch):
            if on_gpu:
                xb = X_t[s:s+batch]
            else:
                xb = X_t[s:s+batch].to(device, non_blocking=True)
            if amp:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    yb = model(xb).float().detach().cpu().numpy()
            else:
                yb = model(xb).detach().cpu().numpy()
            out[s:s+batch] = yb
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="exp")
    ap.add_argument("--arch", default="ft", choices=["mlp", "ft", "mixer", "autoint", "resmlp", "gmlp"])
    ap.add_argument("--prep", default="standard",
                    choices=["standard", "rankgauss", "winsorize_p1", "winsorize_p5"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--horizon", type=int, default=60)
    # MLP args
    ap.add_argument("--mlp-hidden", default="256,128,64")
    # Token-based args
    ap.add_argument("--d-token", type=int, default=32)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--ffn-mult", type=float, default=2.0)
    ap.add_argument("--resmlp-d", type=int, default=256)
    # Regularization / training
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--target-scale", type=float, default=0.0)
    ap.add_argument("--use-sample-weight", action="store_true", default=True)
    ap.add_argument("--no-sample-weight", dest="use_sample_weight", action="store_false")
    ap.add_argument("--aug-mode", default="concat", choices=["concat", "online", "none"])
    ap.add_argument("--aug-lo", type=float, default=0.80)
    ap.add_argument("--aug-hi", type=float, default=1.20)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--cuda", default="0")
    ap.add_argument("--gpu-data", action="store_true", default=True,
                    help="Move all tensors to GPU at start (faster, ~4GB VRAM)")
    ap.add_argument("--cpu-data", dest="gpu_data", action="store_false")
    ap.add_argument("--amp", action="store_true", default=True,
                    help="Use bfloat16 autocast (Ampere+)")
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    H = args.horizon
    seed = args.seed
    print(f"=== T93 {args.arch}/{args.prep}/seed{seed} tag={args.tag} ===", flush=True)
    print(f"  device={device}", flush=True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    progress(args.tag, "loading_caches", arch=args.arch, prep=args.prep, seed=seed)
    t0 = time.time()
    train_full = load_split("train")
    test_full = load_split("test")
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    feat_names_path = os.path.join(CACHE_DIR, "schemeP_feat_names.txt")
    with open(feat_names_path) as f:
        all_feat_names = [line.strip() for line in f]
    total_dim = train_full["X"].shape[1]
    assert len(all_feat_names) == total_dim, (len(all_feat_names), total_dim)

    name_to_idx = {n: i for i, n in enumerate(all_feat_names)}
    drop_idx = set()
    for n in DROP_NAMES:
        if n in name_to_idx:
            drop_idx.add(name_to_idx[n])
    keep_idx = np.array(
        [i for i in range(total_dim) if i not in drop_idx], dtype=np.int64,
    )
    feat_dim = len(keep_idx)
    print(f"  feat_dim={feat_dim}", flush=True)

    forbidden = {"date", "sym", "time"}
    feat_names = [all_feat_names[i] for i in keep_idx]
    assert not (forbidden & set(feat_names))

    date_tr = train_full["date"]
    m_va = date_tr >= 76
    m_t = ~m_va

    X_tr = train_full["X"][m_t][:, keep_idx]
    y_cls_tr = train_full["y60"][m_t].astype(np.int64)
    y_regr_tr = regr_target(train_full["mp_t"][m_t], train_full["mp_t60"][m_t])

    X_va = train_full["X"][m_va][:, keep_idx]
    y_regr_va = regr_target(train_full["mp_t"][m_va], train_full["mp_t60"][m_va])

    print(f"  V4 split: n_train={len(X_tr):,} n_val={len(X_va):,}", flush=True)

    # Fit prep on train, apply to val/test/aug
    progress(args.tag, "fitting_prep", prep=args.prep)
    t_p = time.time()
    stats = fit_standard(X_tr, args.prep)
    print(f"  prep '{args.prep}' fit in {time.time()-t_p:.1f}s", flush=True)

    # Test slice
    X_te = test_full["X"][:, keep_idx]
    y_cls_te = test_full[f"y{H}"].astype(np.int64)
    y_regr_te = regr_target(test_full["mp_t"], test_full[f"mp_t{H}"])
    mp_t_te = test_full["mp_t"]
    mp_th_te = test_full[f"mp_t{H}"]

    progress(args.tag, "applying_prep_val_test")
    t_p = time.time()
    X_tr_s = apply_prep(X_tr, stats)
    X_va_s = apply_prep(X_va, stats)
    X_te_s = apply_prep(X_te, stats)
    print(f"  prep apply in {time.time()-t_p:.1f}s", flush=True)

    # For aug_a, we need to apply on raw X_tr (NaN-imputed) then re-prep.
    # For 'standard' / 'winsorize' this is trivial. For 'rankgauss' the augmented
    # features are reshuffled in a non-monotonic way, so we still re-apply prep
    # (the rank transform is stable: scaled features still close in rank).
    # For simplicity & T75/T81 parity, we apply aug on raw imputed X then re-prep.
    if args.aug_mode == "concat":
        # Impute NaN in raw X_tr (use feat-mean for std/winsor; median for rankgauss)
        if args.prep == "rankgauss":
            imp = stats["median"]
        else:
            imp = stats["mean"]
        X_imp = X_tr.copy()
        nan_m = np.isnan(X_imp)
        if nan_m.any():
            for j in np.where(nan_m.any(axis=0))[0]:
                col_m = nan_m[:, j]
                X_imp[col_m, j] = imp[j]
        seed_rng = np.random.default_rng(seed * 7919 + 1)
        scales = seed_rng.uniform(args.aug_lo, args.aug_hi, size=X_imp.shape).astype(np.float32)
        X_aug_raw = X_imp * scales
        # Re-apply prep to the augmented batch (no NaN now)
        X_aug_s = apply_prep(X_aug_raw, stats)
        X_full_s = np.concatenate([X_tr_s, X_aug_s], axis=0)
        y_full = np.concatenate([y_regr_tr, y_regr_tr], axis=0)
        y_cls_full = np.concatenate([y_cls_tr, y_cls_tr], axis=0)
        print(f"  aug=concat: X_full_s={X_full_s.shape}", flush=True)
    else:
        X_full_s = X_tr_s
        y_full = y_regr_tr
        y_cls_full = y_cls_tr

    # Target scale
    if args.target_scale > 0:
        target_scale = float(args.target_scale)
    else:
        target_scale = 1.0 / float(max(y_regr_tr.std(), 1e-8))
    print(f"  target_scale={target_scale:.4f}", flush=True)
    y_full_s = (y_full * target_scale).astype(np.float32)

    # Tensors
    X_tr_t = torch.from_numpy(X_full_s)
    y_tr_t = torch.from_numpy(y_full_s).float()
    if args.use_sample_weight:
        sw = class_balanced_weight(y_cls_full, num_class=NUM_CLASS)
    else:
        sw = np.ones(len(X_full_s), dtype=np.float32)
    sw_t = torch.from_numpy(sw).float()

    X_va_t = torch.from_numpy(X_va_s)
    X_te_t = torch.from_numpy(X_te_s)

    # Move tensors to GPU at start to avoid per-batch CPU->GPU transfers.
    if args.gpu_data and torch.cuda.is_available():
        print(f"  moving tensors to GPU ...", flush=True)
        X_tr_t = X_tr_t.to(device)
        y_tr_t = y_tr_t.to(device)
        sw_t = sw_t.to(device)
        X_va_t = X_va_t.to(device)
        X_te_t = X_te_t.to(device)

    # WandB
    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = f"T93-{args.arch}-{args.prep}-seed{seed}-{datetime.now().strftime('%H%M%S')}"
            wandb.init(
                project="liangwenbei",
                entity="cjxh21-Tsinghua University",
                name=run_name,
                config={"task": "T93_nn_arch_transformer",
                        **{k: v for k, v in vars(args).items()}},
                tags=["T93", args.arch, args.prep, f"seed{seed}"],
                reinit=True,
            )
        except Exception as e:
            print(f"  wandb init failed ({e!r}); skip", flush=True)
            use_wandb = False

    # Build model
    model = build_model(args.arch, feat_dim, args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model={args.arch} params={n_params:,}", flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    n_train = len(X_tr_t)
    best_val_mse = float("inf"); best_epoch = -1; best_state = None; bad = 0
    epoch_logs = []

    progress(args.tag, "training", n_train=n_train, n_val=len(X_va_t))
    on_gpu = args.gpu_data and torch.cuda.is_available()
    use_amp = args.amp and torch.cuda.is_available()
    amp_dtype = torch.bfloat16
    t_train = time.time()
    for epoch in range(args.epochs):
        t_ep = time.time()
        model.train()
        perm = torch.randperm(n_train, device=("cuda" if on_gpu else "cpu"))
        running_loss = 0.0; running_n = 0
        for s in range(0, n_train, args.batch_size):
            idx = perm[s:s+args.batch_size]
            if on_gpu:
                xb = X_tr_t[idx]
                yb = y_tr_t[idx]
                wb = sw_t[idx]
            else:
                xb = X_tr_t[idx].to(device, non_blocking=True)
                yb = y_tr_t[idx].to(device, non_blocking=True)
                wb = sw_t[idx].to(device, non_blocking=True)

            if use_amp:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    pred = model(xb)
                    mse_per = (pred - yb) ** 2
                    loss = (mse_per * wb).sum() / wb.sum().clamp_min(1.0)
            else:
                pred = model(xb)
                mse_per = (pred - yb) ** 2
                loss = (mse_per * wb).sum() / wb.sum().clamp_min(1.0)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()
            running_loss += float(loss.item()) * xb.size(0)
            running_n += xb.size(0)
        sched.step()
        train_loss_avg = running_loss / running_n

        va_pred_s = predict_chunked(model, X_va_t, batch=8192, device=device, on_gpu=on_gpu, amp=use_amp)
        va_pred = va_pred_s / target_scale
        val_mse = float(((va_pred - y_regr_va) ** 2).mean())
        val_corr = float(np.corrcoef(va_pred, y_regr_va)[0, 1]) if va_pred.std() > 0 else 0.0
        ep_t = time.time() - t_ep
        cur_lr = optim.param_groups[0]["lr"]
        epoch_logs.append({"epoch": epoch, "train_loss": train_loss_avg, "val_mse": val_mse,
                           "val_corr": val_corr, "lr": cur_lr, "time": ep_t})
        print(f"  ep {epoch:3d}  train_loss={train_loss_avg:.5e}  "
              f"val_mse={val_mse:.5e}  val_corr={val_corr:.4f}  lr={cur_lr:.2e}  ({ep_t:.1f}s)",
              flush=True)
        if use_wandb:
            try:
                wandb.log({"epoch": epoch, "train_loss": train_loss_avg, "val_mse": val_mse,
                           "val_corr": val_corr, "lr": cur_lr, "time": ep_t})
            except Exception:
                pass
        if val_mse < best_val_mse - 1e-10:
            best_val_mse = val_mse; best_epoch = epoch
            best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  early stop at ep {epoch}", flush=True); break

    train_time = time.time() - t_train
    print(f"  train done in {train_time:.1f}s; best_epoch={best_epoch} best_val_mse={best_val_mse:.5e}", flush=True)
    assert best_state is not None
    model.load_state_dict(best_state)

    # Final test predictions
    te_pred = predict_chunked(model, X_te_t, batch=8192, device=device, on_gpu=on_gpu, amp=use_amp) / target_scale
    te_corr = float(np.corrcoef(te_pred, y_regr_te)[0, 1]) if te_pred.std() > 0 else 0.0
    te_mse = float(((te_pred - y_regr_te) ** 2).mean())

    # Quick raw EV-gate sanity at k=1
    fee_thr = 2.0 * FEE
    pred_action = np.full(len(te_pred), 1, dtype=np.int8)
    pred_action[te_pred > fee_thr] = 2
    pred_action[te_pred < -fee_thr] = 0
    side = pred_action.astype(np.float64) - 1.0
    diff = mp_th_te.astype(np.float64) - mp_t_te.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th_te + 1.0) + (mp_t_te + 1.0))
    denom = mp_t_te.astype(np.float64) + 1.0
    pnl = (side * diff - fee_pnl) / denom
    cum_pnl = float(pnl.sum())
    n_active = int((pred_action != 1).sum())
    print(f"  TEST k=1 thr={fee_thr:.5f}: cum_pnl={cum_pnl:+.4f} n_active={n_active:,} corr={te_corr:.4f}",
          flush=True)

    # Save predictions parquet
    sess_map = {0: "am", 1: "pm"}
    te_df = pd.DataFrame({
        "sym": test_full["sym"].astype(np.int8),
        "date": test_full["date"].astype(np.int16),
        "session": np.array([sess_map[int(s)] for s in test_full["sess_idx"]], dtype=object),
        "t": test_full["t"].astype(np.int16),
        "true_label": y_cls_te.astype(np.int8),
        "true_dmid_norm": y_regr_te.astype(np.float32),
        "pred_dmid_norm": te_pred.astype(np.float32),
        "midprice_t": mp_t_te.astype(np.float32),
        "midprice_th": mp_th_te.astype(np.float32),
    })
    pred_path = os.path.join(HERE, f"pred_T93_{args.tag}_seed{seed}.parquet")
    te_df.to_parquet(pred_path, index=False)
    print(f"  saved {pred_path}", flush=True)

    # Save model + summary
    summary = {
        "tag": args.tag,
        "arch": args.arch, "prep": args.prep,
        "seed": seed, "horizon": H,
        "n_features": feat_dim,
        "n_params": int(n_params),
        "best_epoch": best_epoch, "best_val_mse": best_val_mse,
        "test": {"mse": te_mse, "corr": te_corr,
                 "ev_gate_k1_cum_pnl": cum_pnl, "ev_gate_k1_n_active": n_active},
        "train_time_sec": float(train_time),
        "epoch_logs": epoch_logs,
        "params": vars(args),
        "target_scale": float(target_scale),
    }
    out_path = os.path.join(HERE, f"summary_T93_{args.tag}_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"  summary -> {out_path}", flush=True)

    if use_wandb:
        try:
            wandb.log({"best_epoch": best_epoch, "best_val_mse": best_val_mse,
                       "test_mse": te_mse, "test_corr": te_corr,
                       "test_ev_gate_k1_cum_pnl": cum_pnl,
                       "test_ev_gate_k1_n_active": n_active,
                       "train_time_sec": float(train_time)})
            wandb.finish()
        except Exception:
            pass

    progress(args.tag, "done", arch=args.arch, prep=args.prep,
             best_epoch=best_epoch, val_mse=best_val_mse,
             test_corr=te_corr, ev_gate_k1=cum_pnl)


if __name__ == "__main__":
    main()
