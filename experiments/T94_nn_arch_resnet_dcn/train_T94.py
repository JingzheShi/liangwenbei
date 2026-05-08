"""T94: NN architecture exploration + regularization ablation for iter_015 diversifier.

Reuses T81's pipeline (load schemeP, V4 walk-forward, drop 11 fail features, global standardize,
NaN→mean impute, clip ±10, target_scale=1/σ_y, EV-gate k=1 sanity).

Architectures (--arch):
  - mlp_small : T81 baseline [256,128,64], dropout=0.10, GELU+LN  (sanity replica)
  - mlp_wide  : [1024,512,256,128]
  - mlp_deep  : [512,512,512,256,128]
  - resnet    : [in→512] → ResBlock×N (LN + GELU + Dropout, residual)
  - dcn_v2    : Deep & Cross Network V2 — cross_layers (degree-N feature interactions) + deep MLP
  - wide_deep : linear branch + deep MLP, summed
  - geglu     : GeGLU activation MLP
  - snn       : SELU + AlphaDropout
  - emb_mlp   : per-feature discretized embedding + MLP (NEW)

Regularization (--reg):
  - none
  - mixup     : α-mixup on (x,y)
  - swapnoise : SwapNoise — random column-wise swap with another sample (p=0.15)
  - input_dropout : zero-out input features at rate p
  - swa       : Stochastic Weight Averaging (last K epochs)

Each combo logs:
  - val MSE, val corr
  - test MSE/corr
  - test EV-gate k=1 cum_pnl (single set, no DE)
  - saves model_{tag}_seed{S}.pt and pred_{tag}_seed{S}.parquet

CRITICAL_CONSTRAINTS: Model.forward only takes x (no sym/date/time), inference stateless,
global stats only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from copy import deepcopy
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

T59_FAIL_NAMES = [
    "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
    "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
    "qrank_W100_cumspread",
    "kyle_lam_W50", "kyle_lam_W100",
    "roll_eff_spr_ratio_W100",
]
STAGE5_FAIL_NAMES = ["liq_asym_top5_W5"]
DROP_NAMES = T59_FAIL_NAMES + STAGE5_FAIL_NAMES


def progress(step, **extra):
    p = os.path.join(HERE, "worker-progress.json")
    payload = {"status": "running", "step": step,
               "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               **extra}
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


def load_split(split):
    p = os.path.join(CACHE_DIR, f"schemeP_{split}.npz")
    d = np.load(p)
    return {k: d[k] for k in d.files}


def regr_target(mp_t, mp_th):
    return ((mp_th.astype(np.float64) - mp_t.astype(np.float64))
            / (mp_t.astype(np.float64) + 1.0)).astype(np.float32)


# ----------- Architectures -----------

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, dropout=0.1, use_layernorm=True, act="gelu"):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            if use_layernorm:
                layers.append(nn.LayerNorm(h))
            if act == "gelu":
                layers.append(nn.GELU())
            elif act == "relu":
                layers.append(nn.ReLU())
            elif act == "selu":
                layers.append(nn.SELU())
            else:
                raise ValueError(act)
            layers.append(nn.Dropout(dropout) if act != "selu" else nn.AlphaDropout(dropout))
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        self._init(act)

    def _init(self, act):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if act == "selu":
                    # SELU needs lecun_normal init for self-norm property
                    nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                    with torch.no_grad():
                        m.weight.data *= (1.0 / np.sqrt(2.0))
                else:
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ResBlock(nn.Module):
    """Pre-LN residual block: x -> LN -> Linear -> GELU -> Dropout -> Linear -> Dropout, +x."""

    def __init__(self, dim, dropout=0.15):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.ln(x)
        h = F.gelu(self.fc1(h))
        h = self.drop(h)
        h = self.fc2(h)
        h = self.drop(h)
        return x + h


class TabResNet(nn.Module):
    """[in→hidden] (LN+GELU) → ResBlock×N → LN → Linear(1)."""

    def __init__(self, in_dim, hidden=512, n_blocks=3, dropout=0.15):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.proj(x)
        for b in self.blocks:
            h = b(h)
        return self.head(h).squeeze(-1)


class CrossLayer(nn.Module):
    """DCN-V2 cross layer: x_{l+1} = x0 * (W x_l + b) + x_l."""

    def __init__(self, in_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, in_dim)

    def forward(self, x0, xl):
        return x0 * self.W(xl) + xl


class DCNv2(nn.Module):
    """Cross network on x0 (n_cross layers) in parallel with deep MLP. Concat → Linear(1)."""

    def __init__(self, in_dim, n_cross=3, deep_hidden=(256, 128, 64), dropout=0.15):
        super().__init__()
        self.crosses = nn.ModuleList([CrossLayer(in_dim) for _ in range(n_cross)])
        deep_layers = []
        d = in_dim
        for h in deep_hidden:
            deep_layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        self.deep = nn.Sequential(*deep_layers)
        self.head = nn.Linear(in_dim + d, 1)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        xl = x
        for c in self.crosses:
            xl = c(x, xl)
        deep = self.deep(x)
        h = torch.cat([xl, deep], dim=-1)
        return self.head(h).squeeze(-1)


class WideDeep(nn.Module):
    """Wide linear + Deep MLP, summed at output."""

    def __init__(self, in_dim, hidden=(512, 256, 128), dropout=0.15):
        super().__init__()
        self.wide = nn.Linear(in_dim, 1)
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.deep = nn.Sequential(*layers)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return (self.wide(x) + self.deep(x)).squeeze(-1)


class GeGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        a, b = self.proj(x).chunk(2, dim=-1)
        return a * F.gelu(b)


class GeGLUMLP(nn.Module):
    def __init__(self, in_dim, hidden=(512, 256, 128), dropout=0.1):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers.append(GeGLU(d, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_model(arch, in_dim, args):
    if arch == "mlp_small":
        return MLP(in_dim, [256, 128, 64], dropout=args.dropout, act="gelu")
    if arch == "mlp_med":
        return MLP(in_dim, [512, 256, 128], dropout=args.dropout, act="gelu")
    if arch == "mlp_wide":
        return MLP(in_dim, [1024, 512, 256, 128], dropout=args.dropout, act="gelu")
    if arch == "mlp_xwide":
        return MLP(in_dim, [2048, 1024, 512, 256], dropout=args.dropout, act="gelu")
    if arch == "mlp_deep":
        return MLP(in_dim, [512, 512, 512, 256, 128], dropout=args.dropout, act="gelu")
    if arch == "resnet":
        return TabResNet(in_dim, hidden=args.resnet_hidden, n_blocks=args.resnet_blocks,
                         dropout=args.dropout)
    if arch == "dcn_v2":
        return DCNv2(in_dim, n_cross=args.n_cross,
                     deep_hidden=tuple(int(x) for x in args.dcn_deep.split(",")),
                     dropout=args.dropout)
    if arch == "wide_deep":
        return WideDeep(in_dim, hidden=tuple(int(x) for x in args.wd_hidden.split(",")),
                        dropout=args.dropout)
    if arch == "geglu":
        return GeGLUMLP(in_dim, hidden=tuple(int(x) for x in args.geglu_hidden.split(",")),
                        dropout=args.dropout)
    if arch == "snn":
        return MLP(in_dim, [512, 256, 128], dropout=args.dropout, use_layernorm=False, act="selu")
    raise ValueError(arch)


# ----------- Regularization helpers -----------

def mixup_batch(x, y, w, alpha=0.2):
    if alpha <= 0:
        return x, y, w
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    x_m = lam * x + (1 - lam) * x[perm]
    y_m = lam * y + (1 - lam) * y[perm]
    w_m = lam * w + (1 - lam) * w[perm]
    return x_m, y_m, w_m


def swap_noise(x, p=0.15):
    """Replace each cell with a value from a random other row in the batch with prob p."""
    B, D = x.shape
    mask = (torch.rand(B, D, device=x.device) < p)
    perm_rows = torch.randint(0, B, (B, D), device=x.device)
    sample = x[perm_rows, torch.arange(D, device=x.device).expand(B, D)]
    return torch.where(mask, sample, x)


def input_dropout(x, p=0.1):
    """Zero out random features (input-level dropout, fixed-rate, no scaling)."""
    if p <= 0:
        return x
    mask = (torch.rand_like(x) > p).float()
    return x * mask


# ----------- Predict / EV-gate -----------

def predict_chunked(model, X, batch=16384, device="cuda"):
    model.eval()
    out = np.zeros(len(X), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(X), batch):
            xb = X[s:s+batch].to(device, non_blocking=True)
            yb = model(xb).detach().cpu().numpy()
            out[s:s+batch] = yb
    return out


def ev_gate_k1_pnl(pred, mp_t, mp_th):
    fee_thr = 2.0 * FEE
    a = np.full(len(pred), 1, dtype=np.int8)
    a[pred > fee_thr] = 2
    a[pred < -fee_thr] = 0
    side = a.astype(np.float64) - 1.0
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    pnl = (side * diff - fee_pnl) / denom
    return float(pnl.sum()), int((a != 1).sum())


def per_sym_ev_gate_k1(pred, mp_t, mp_th, sym):
    """Returns (sum_of_per_sym_cum_pnl, list_of_per_sym_pnl, list_of_n_active)."""
    syms = []
    pnls = []
    nas = []
    for k in SYMS:
        m = sym == k
        if m.sum() == 0:
            pnls.append(0.0)
            nas.append(0)
            continue
        p_k, n_k = ev_gate_k1_pnl(pred[m], mp_t[m], mp_th[m])
        pnls.append(p_k)
        nas.append(n_k)
    return float(sum(pnls)), pnls, nas


# ----------- Main -----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="tag for output files")
    ap.add_argument("--arch", required=True,
                    choices=["mlp_small", "mlp_med", "mlp_wide", "mlp_xwide", "mlp_deep",
                             "resnet", "dcn_v2", "wide_deep", "geglu", "snn"])
    ap.add_argument("--reg", default="none",
                    choices=["none", "mixup", "swapnoise", "input_dropout", "swa", "mixup_swa"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--target-scale", type=float, default=0.0)
    ap.add_argument("--use-sample-weight", action="store_true", default=True)
    ap.add_argument("--no-sample-weight", dest="use_sample_weight", action="store_false")
    ap.add_argument("--aug-mode", default="concat", choices=["concat", "online", "none"])
    ap.add_argument("--aug-lo", type=float, default=0.80)
    ap.add_argument("--aug-hi", type=float, default=1.20)
    ap.add_argument("--mixup-alpha", type=float, default=0.4)
    ap.add_argument("--swap-p", type=float, default=0.15)
    ap.add_argument("--input-dropout-p", type=float, default=0.1)
    ap.add_argument("--swa-start", type=int, default=15, help="epoch to start SWA")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--cuda", default="0")
    # arch-specific
    ap.add_argument("--resnet-hidden", type=int, default=512)
    ap.add_argument("--resnet-blocks", type=int, default=3)
    ap.add_argument("--n-cross", type=int, default=3)
    ap.add_argument("--dcn-deep", default="256,128,64")
    ap.add_argument("--wd-hidden", default="512,256,128")
    ap.add_argument("--geglu-hidden", default="512,256,128")
    ap.add_argument("--grad-clip", type=float, default=5.0)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = args.seed
    H = args.horizon
    tag = args.tag
    print(f"=== T94 tag={tag} arch={args.arch} reg={args.reg} seed={seed} h={H} ===", flush=True)
    print(f"  device={device}", flush=True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    progress("loading_caches", tag=tag)
    t0 = time.time()
    train_full = load_split("train")
    test_full = load_split("test")
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    feat_names_path = os.path.join(CACHE_DIR, "schemeP_feat_names.txt")
    with open(feat_names_path) as f:
        all_feat_names = [line.strip() for line in f]
    total_dim = train_full["X"].shape[1]

    name_to_idx = {n: i for i, n in enumerate(all_feat_names)}
    drop_idx = set()
    for n in DROP_NAMES:
        if n in name_to_idx:
            drop_idx.add(name_to_idx[n])
    keep_idx = np.array([i for i in range(total_dim) if i not in drop_idx], dtype=np.int64)
    feat_dim = len(keep_idx)
    feat_names = [all_feat_names[i] for i in keep_idx]
    print(f"  feat_dim={feat_dim}", flush=True)

    forbidden = {"date", "sym", "time"}
    assert not (forbidden & set(feat_names))

    # V4 walk-forward
    date_tr = train_full["date"]
    m_va = date_tr >= 76
    m_t = ~m_va

    X_tr = train_full["X"][m_t][:, keep_idx]
    y_cls_tr = train_full["y60"][m_t].astype(np.int64)
    y_regr_tr = regr_target(train_full["mp_t"][m_t], train_full["mp_t60"][m_t])
    mp_t_tr = train_full["mp_t"][m_t]
    mp_th_tr = train_full["mp_t60"][m_t]

    X_va = train_full["X"][m_va][:, keep_idx]
    y_cls_va = train_full["y60"][m_va].astype(np.int64)
    y_regr_va = regr_target(train_full["mp_t"][m_va], train_full["mp_t60"][m_va])
    mp_t_va = train_full["mp_t"][m_va]
    mp_th_va = train_full["mp_t60"][m_va]

    # Test
    X_te = test_full["X"][:, keep_idx]
    y_cls_te = test_full[f"y{H}"].astype(np.int64)
    y_regr_te = regr_target(test_full["mp_t"], test_full[f"mp_t{H}"])
    mp_t_te = test_full["mp_t"]
    mp_th_te = test_full[f"mp_t{H}"]
    sym_te = test_full["sym"]

    print(f"  V4 split: n_train={len(X_tr):,} n_val={len(X_va):,} n_test={len(X_te):,}", flush=True)

    # Stats (global, no per-sym)
    feat_mean = np.nanmean(X_tr, axis=0).astype(np.float32)
    feat_std = np.nanstd(X_tr, axis=0).astype(np.float32)
    feat_std = np.maximum(feat_std, 1e-6)
    CLIP = 10.0

    def standardize(X):
        Xs = (X - feat_mean) / feat_std
        Xs = np.where(np.isnan(Xs), 0.0, Xs)
        return np.clip(Xs, -CLIP, CLIP).astype(np.float32)

    X_va_std = torch.from_numpy(standardize(X_va))
    X_te_std = torch.from_numpy(standardize(X_te))

    # Pre-impute train (so aug_a multiplies clean values)
    X_tr_imputed = X_tr.copy()
    nan_mask = np.isnan(X_tr_imputed)
    if nan_mask.any():
        for d_idx in np.where(nan_mask.any(axis=0))[0]:
            col_nan = np.isnan(X_tr_imputed[:, d_idx])
            X_tr_imputed[col_nan, d_idx] = feat_mean[d_idx]

    if args.target_scale > 0:
        target_scale = float(args.target_scale)
    else:
        target_scale = 1.0 / float(max(y_regr_tr.std(), 1e-8))
    print(f"  target_scale={target_scale:.4f}", flush=True)
    y_regr_tr_s = (y_regr_tr * target_scale).astype(np.float32)

    # aug
    if args.aug_mode == "concat":
        seed_rng = np.random.default_rng(seed * 7919 + 1)
        scales = seed_rng.uniform(args.aug_lo, args.aug_hi, size=X_tr_imputed.shape).astype(np.float32)
        X_aug = X_tr_imputed * scales
        X_tr_full = np.concatenate([X_tr_imputed, X_aug], axis=0)
        y_tr_full = np.concatenate([y_regr_tr_s, y_regr_tr_s], axis=0)
        y_cls_tr_full = np.concatenate([y_cls_tr, y_cls_tr], axis=0)
        print(f"  aug=concat: X_tr_full={X_tr_full.shape}", flush=True)
    else:
        X_tr_full = X_tr_imputed
        y_tr_full = y_regr_tr_s
        y_cls_tr_full = y_cls_tr

    X_tr_raw = torch.from_numpy(X_tr_full.astype(np.float32))
    y_tr_t = torch.from_numpy(y_tr_full).float()
    if args.use_sample_weight:
        sw = class_balanced_weight(y_cls_tr_full, num_class=NUM_CLASS)
    else:
        sw = np.ones(len(X_tr_full), dtype=np.float32)
    sw_t = torch.from_numpy(sw).float()

    # WandB
    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = f"T94-{tag}-seed{seed}"
            wandb.init(
                project="liangwenbei",
                entity="cjxh21-Tsinghua University",
                name=run_name,
                config={"model": args.arch, "reg": args.reg, "task": "T94",
                        "n_features": feat_dim, "horizon": H, "seed": seed,
                        **{k: v for k, v in vars(args).items()}},
                tags=["T94", args.arch, args.reg, f"seed{seed}"],
            )
        except Exception as e:
            print(f"  wandb init failed: {e!r}", flush=True)
            use_wandb = False

    # Model
    model = build_model(args.arch, feat_dim, args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model={args.arch}  n_params={n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    fm_t = torch.from_numpy(feat_mean).to(device)
    fs_t = torch.from_numpy(feat_std).to(device)

    best_val_mse = float("inf")
    best_epoch = -1
    best_state = None
    bad_epochs = 0

    # SWA state
    do_swa = args.reg in ("swa", "mixup_swa")
    swa_n = 0
    swa_state = None

    n_train = len(X_tr_raw)
    epoch_logs = []

    progress("training", tag=tag, n_train=n_train)
    t_train_start = time.time()
    for epoch in range(args.epochs):
        t_ep = time.time()
        model.train()
        perm = torch.randperm(n_train)
        running_loss = 0.0
        running_n = 0

        for s in range(0, n_train, args.batch_size):
            idx = perm[s:s+args.batch_size]
            xb_raw = X_tr_raw[idx].to(device, non_blocking=True)
            yb = y_tr_t[idx].to(device, non_blocking=True)
            wb = sw_t[idx].to(device, non_blocking=True)

            if args.aug_mode == "online":
                scales = torch.empty_like(xb_raw).uniform_(args.aug_lo, args.aug_hi)
                xb_raw = xb_raw * scales

            xb = (xb_raw - fm_t) / fs_t
            xb = torch.clamp(xb, -CLIP, CLIP)

            # apply regularization (post-standardization)
            if args.reg in ("swapnoise",):
                xb = swap_noise(xb, p=args.swap_p)
            if args.reg in ("input_dropout",):
                xb = input_dropout(xb, p=args.input_dropout_p)
            if args.reg in ("mixup", "mixup_swa"):
                xb, yb, wb = mixup_batch(xb, yb, wb, alpha=args.mixup_alpha)

            pred = model(xb)
            mse_per = (pred - yb) ** 2
            loss = (mse_per * wb).sum() / wb.sum().clamp_min(1.0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            running_loss += float(loss.item()) * xb.size(0)
            running_n += xb.size(0)

        scheduler.step()
        train_loss = running_loss / running_n

        va_pred_s = predict_chunked(model, X_va_std, device=device)
        va_pred = va_pred_s / target_scale
        val_mse = float(((va_pred - y_regr_va) ** 2).mean())
        val_corr = float(np.corrcoef(va_pred, y_regr_va)[0, 1]) if va_pred.std() > 0 else 0.0

        ep_time = time.time() - t_ep
        cur_lr = optimizer.param_groups[0]["lr"]
        log = {"epoch": epoch, "train_loss": train_loss, "val_mse": val_mse,
               "val_corr": val_corr, "lr": cur_lr, "time": ep_time}
        epoch_logs.append(log)
        print(f"  ep {epoch:3d}  train={train_loss:.6e}  val_mse={val_mse:.6e}  "
              f"val_corr={val_corr:.4f}  lr={cur_lr:.2e}  ({ep_time:.1f}s)", flush=True)

        if use_wandb:
            wandb.log({**log, "epoch": epoch})

        if val_mse < best_val_mse - 1e-10:
            best_val_mse = val_mse
            best_epoch = epoch
            best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience and not do_swa:
                print(f"  early stop at epoch {epoch}", flush=True)
                break

        # SWA accumulation
        if do_swa and epoch >= args.swa_start:
            current = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
            if swa_state is None:
                swa_state = current
                swa_n = 1
            else:
                swa_n += 1
                for k in swa_state:
                    swa_state[k].mul_((swa_n - 1) / swa_n).add_(current[k] * (1.0 / swa_n))

    train_time = time.time() - t_train_start
    print(f"  train done in {train_time:.1f}s; best_epoch={best_epoch} val={best_val_mse:.6e}",
          flush=True)
    assert best_state is not None

    # If SWA: evaluate SWA, prefer it if better val
    final_state = best_state
    final_tag = "best_val"
    if do_swa and swa_state is not None:
        model.load_state_dict(swa_state)
        swa_pred_s = predict_chunked(model, X_va_std, device=device)
        swa_pred = swa_pred_s / target_scale
        swa_val_mse = float(((swa_pred - y_regr_va) ** 2).mean())
        swa_val_corr = float(np.corrcoef(swa_pred, y_regr_va)[0, 1]) if swa_pred.std() > 0 else 0.0
        print(f"  SWA val_mse={swa_val_mse:.6e} val_corr={swa_val_corr:.4f} (n_avg={swa_n})", flush=True)
        if swa_val_mse < best_val_mse:
            final_state = swa_state
            final_tag = "swa"
            print(f"  using SWA weights", flush=True)
        else:
            print(f"  SWA worse than best, using best_val", flush=True)

    model.load_state_dict(final_state)

    # Save model
    model_path = os.path.join(HERE, f"model_{tag}_seed{seed}.pt")
    torch.save({
        "state_dict": final_state, "feat_mean": feat_mean, "feat_std": feat_std,
        "keep_idx": keep_idx, "feat_names": feat_names, "in_dim": feat_dim,
        "target_scale": float(target_scale), "clip": float(CLIP),
        "best_epoch": best_epoch, "best_val_mse": best_val_mse,
        "args": vars(args), "final_tag": final_tag,
    }, model_path)

    # Predict
    va_pred = predict_chunked(model, X_va_std, device=device) / target_scale
    te_pred = predict_chunked(model, X_te_std, device=device) / target_scale

    va_mse = float(((va_pred - y_regr_va) ** 2).mean())
    va_corr = float(np.corrcoef(va_pred, y_regr_va)[0, 1]) if va_pred.std() > 0 else 0.0
    te_mse = float(((te_pred - y_regr_te) ** 2).mean())
    te_corr = float(np.corrcoef(te_pred, y_regr_te)[0, 1]) if te_pred.std() > 0 else 0.0

    # EV-gate (single-set, k=1) on test
    cum_pnl, n_active = ev_gate_k1_pnl(te_pred, mp_t_te, mp_th_te)
    # Also LOSO-equiv k=1 (sum of per-sym)
    loso_k1, per_sym_pnls, per_sym_na = per_sym_ev_gate_k1(te_pred, mp_t_te, mp_th_te, sym_te)

    print(f"  VAL: mse={va_mse:.7f} corr={va_corr:.4f}", flush=True)
    print(f"  TEST: mse={te_mse:.7f} corr={te_corr:.4f}", flush=True)
    print(f"  TEST EV-gate k=1: single={cum_pnl:+.4f} n_active={n_active:,}", flush=True)
    print(f"  TEST EV-gate k=1 LOSO-equiv: {loso_k1:+.4f}  per_sym=[{', '.join(f'{x:+.3f}' for x in per_sym_pnls)}]", flush=True)

    # Save preds
    sess_map = {0: "am", 1: "pm"}
    te_df = pd.DataFrame({
        "sym": sym_te.astype(np.int8),
        "date": test_full["date"].astype(np.int16),
        "session": np.array([sess_map[int(s)] for s in test_full["sess_idx"]], dtype=object),
        "t": test_full["t"].astype(np.int16),
        "true_label": y_cls_te.astype(np.int8),
        "true_dmid_norm": y_regr_te.astype(np.float32),
        "pred_dmid_norm": te_pred.astype(np.float32),
        "midprice_t": mp_t_te.astype(np.float32),
        "midprice_th": mp_th_te.astype(np.float32),
    })
    pred_path = os.path.join(HERE, f"pred_{tag}_seed{seed}.parquet")
    te_df.to_parquet(pred_path, index=False)

    summary = {
        "task": f"T94 {tag} seed={seed}",
        "tag": tag, "arch": args.arch, "reg": args.reg, "seed": seed,
        "n_features": feat_dim, "n_params": int(n_params),
        "epochs_run": epoch_logs[-1]["epoch"] + 1 if epoch_logs else 0,
        "best_epoch": best_epoch, "best_val_mse": best_val_mse,
        "final_state": final_tag, "train_time_sec": float(train_time),
        "val": {"mse": va_mse, "corr": va_corr},
        "test": {"mse": te_mse, "corr": te_corr,
                 "ev_gate_k1_cum_pnl": cum_pnl, "ev_gate_k1_n_active": n_active,
                 "ev_gate_k1_loso_equiv": loso_k1,
                 "per_sym_pnl": per_sym_pnls, "per_sym_n_active": per_sym_na},
        "params": vars(args),
        "epoch_logs": epoch_logs,
    }
    out_path = os.path.join(HERE, f"summary_{tag}_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    if use_wandb:
        wandb.log({"best_epoch": best_epoch, "best_val_mse": best_val_mse,
                   "test_mse": te_mse, "test_corr": te_corr,
                   "test_ev_gate_k1_cum_pnl": cum_pnl,
                   "test_ev_gate_k1_loso_equiv": loso_k1,
                   "train_time_sec": float(train_time)})
        wandb.finish()
    progress("done", tag=tag, best_epoch=best_epoch, val_mse=best_val_mse,
             test_corr=te_corr, ev_gate_k1=cum_pnl, ev_gate_k1_loso_equiv=loso_k1)


if __name__ == "__main__":
    main()
