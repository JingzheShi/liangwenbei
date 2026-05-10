"""T172: Group-Transformer on full data 0-119, 5 seeds.

Architecture:
  - Input projection: Linear(feat_dim, K*d_token) → reshape to (B, K, d_token)
    Creates K=32 "super-tokens" via learned feature grouping
  - CLS token prepended: sequence length = K+1
  - 2-layer TransformerEncoder: nhead=4, dim_ff=256, dropout=0.1, pre-norm
  - CLS token → head: LayerNorm + Linear + GELU + Linear

Speed advantage vs FT-Transformer: O(K²) instead of O(feat_dim²) attention.
K=32 vs L=360: 126x faster attention. ~8s/epoch vs 249s/epoch.

M7 trick: train on ALL data (dates 0-119: train+val+test), fixed epochs, no early stop.
Export: weights to .npz for numpy-only inference.
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
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.environ.get("CACHE_DIR", "/root/lwb_remote_pkg/cache")

T59_FAIL_NAMES = [
    "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
    "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
    "qrank_W100_cumspread",
    "kyle_lam_W50", "kyle_lam_W100",
    "roll_eff_spr_ratio_W100",
]
STAGE5_FAIL_NAMES = ["liq_asym_top5_W5"]
DROP_NAMES = T59_FAIL_NAMES + STAGE5_FAIL_NAMES
NUM_CLASS = 3
FEE = 0.0001


def progress(step, **extra):
    p = os.path.join(HERE, "worker-progress.json")
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


def class_balanced_weight(y_cls, num_class=3):
    w = np.ones(len(y_cls), dtype=np.float32)
    for c in range(num_class):
        mask = y_cls == c
        n_c = mask.sum()
        if n_c > 0:
            w[mask] = len(y_cls) / (num_class * n_c)
    return w.astype(np.float32)


class GroupTransformer(nn.Module):
    """Feature-Grouping Transformer for tabular regression.

    Projects feat_dim flat features → K super-tokens via a single linear,
    prepends a CLS token, then runs a standard pre-norm TransformerEncoder.
    Final prediction from CLS. Attention cost is O(K²) not O(feat_dim²).
    """
    def __init__(self, feat_dim, n_tokens=32, d_token=64, n_heads=4, depth=2,
                 dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_token = d_token
        self.proj = nn.Linear(feat_dim, n_tokens * d_token)
        self.proj_norm = nn.LayerNorm(d_token)
        self.cls = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads,
            dim_feedforward=dim_feedforward,
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
        B = x.size(0)
        tokens = self.proj(x).view(B, self.n_tokens, self.d_token)
        tokens = self.proj_norm(tokens)
        cls = self.cls.expand(B, -1, -1)
        seq = torch.cat([cls, tokens], dim=1)       # (B, K+1, d)
        out = self.encoder(seq)
        return self.head(out[:, 0]).squeeze(-1)      # predict from CLS


def save_npz(model: GroupTransformer, feat_mean, feat_std, feat_dim, keep_idx,
             target_scale, clip, n_heads, out_path):
    """Export GroupTransformer weights to npz for numpy inference."""
    arrays = {
        "arch": np.array(["GroupTransformer"], dtype=object),
        "n_features": np.array([feat_dim], dtype=np.int64),
        "n_tokens": np.array([model.n_tokens], dtype=np.int64),
        "d_token": np.array([model.d_token], dtype=np.int64),
        "n_heads": np.array([n_heads], dtype=np.int64),
        "n_layers": np.array([len(model.encoder.layers)], dtype=np.int64),
        "target_scale": np.array([target_scale], dtype=np.float32),
        "feat_mean": feat_mean.astype(np.float32),
        "feat_std": feat_std.astype(np.float32),
        "keep_idx": keep_idx.astype(np.int64),
        "clip": np.array([clip], dtype=np.float32),
        # Input projection
        "proj_w": model.proj.weight.detach().cpu().numpy().astype(np.float32),
        "proj_b": model.proj.bias.detach().cpu().numpy().astype(np.float32),
        "proj_norm_w": model.proj_norm.weight.detach().cpu().numpy().astype(np.float32),
        "proj_norm_b": model.proj_norm.bias.detach().cpu().numpy().astype(np.float32),
        "cls_token": model.cls.detach().cpu().numpy().astype(np.float32),
        # Head
        "head_ln_w": model.head[0].weight.detach().cpu().numpy().astype(np.float32),
        "head_ln_b": model.head[0].bias.detach().cpu().numpy().astype(np.float32),
        "head_fc1_w": model.head[1].weight.detach().cpu().numpy().astype(np.float32),
        "head_fc1_b": model.head[1].bias.detach().cpu().numpy().astype(np.float32),
        "head_fc2_w": model.head[4].weight.detach().cpu().numpy().astype(np.float32),
        "head_fc2_b": model.head[4].bias.detach().cpu().numpy().astype(np.float32),
    }
    for i, layer in enumerate(model.encoder.layers):
        arrays[f"L{i}_norm1_w"] = layer.norm1.weight.detach().cpu().numpy().astype(np.float32)
        arrays[f"L{i}_norm1_b"] = layer.norm1.bias.detach().cpu().numpy().astype(np.float32)
        arrays[f"L{i}_norm2_w"] = layer.norm2.weight.detach().cpu().numpy().astype(np.float32)
        arrays[f"L{i}_norm2_b"] = layer.norm2.bias.detach().cpu().numpy().astype(np.float32)
        arrays[f"L{i}_attn_in_proj_w"] = layer.self_attn.in_proj_weight.detach().cpu().numpy().astype(np.float32)
        arrays[f"L{i}_attn_in_proj_b"] = layer.self_attn.in_proj_bias.detach().cpu().numpy().astype(np.float32)
        arrays[f"L{i}_attn_out_w"] = layer.self_attn.out_proj.weight.detach().cpu().numpy().astype(np.float32)
        arrays[f"L{i}_attn_out_b"] = layer.self_attn.out_proj.bias.detach().cpu().numpy().astype(np.float32)
        arrays[f"L{i}_ffn1_w"] = layer.linear1.weight.detach().cpu().numpy().astype(np.float32)
        arrays[f"L{i}_ffn1_b"] = layer.linear1.bias.detach().cpu().numpy().astype(np.float32)
        arrays[f"L{i}_ffn2_w"] = layer.linear2.weight.detach().cpu().numpy().astype(np.float32)
        arrays[f"L{i}_ffn2_b"] = layer.linear2.bias.detach().cpu().numpy().astype(np.float32)
    np.savez(out_path, **arrays)
    sz = os.path.getsize(out_path + ".npz") / 1024
    print(f"  saved {out_path}.npz ({sz:.0f} KB)", flush=True)


def predict_chunked(model, X_t, batch=4096, device="cuda", amp=False):
    model.eval()
    out = np.zeros(len(X_t), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(X_t), batch):
            xb = X_t[s:s+batch].to(device, non_blocking=True)
            if amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    yb = model(xb).float()
            else:
                yb = model(xb)
            out[s:s+batch] = yb.detach().cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--n-tokens", type=int, default=32)
    ap.add_argument("--d-token", type=int, default=64)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--ffn-mult", type=float, default=4.0)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--clip", type=float, default=10.0)
    ap.add_argument("--aug-lo", type=float, default=0.80)
    ap.add_argument("--aug-hi", type=float, default=1.20)
    ap.add_argument("--cuda", default="0")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out-dir", default=HERE)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H = args.horizon
    seed = args.seed
    dim_ff = int(args.d_token * args.ffn_mult)
    out_dir = args.out_dir
    print(f"=== T172 GroupTransformer seed={seed} h={H} K={args.n_tokens} d={args.d_token} ===", flush=True)
    print(f"  device={device}  epochs={args.epochs}  batch={args.batch_size}", flush=True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Load precomputed data (fast: avoids ~360s loading+aug per seed)
    meta_path = os.path.join(out_dir, "precomputed_meta.json")
    if not os.path.exists(meta_path):
        print("  ERROR: precomputed data not found. Run precompute_data.py first.", flush=True)
        sys.exit(1)

    progress("loading_precomputed", seed=seed)
    t0 = time.time()
    with open(meta_path) as f:
        meta = json.load(f)
    feat_dim = meta["feat_dim"]
    target_scale = meta["target_scale"]
    clip = meta["clip"]

    X_full = np.load(os.path.join(out_dir, "X_full.npy"))
    y_full = np.load(os.path.join(out_dir, "y_full.npy"))
    sw_full = np.load(os.path.join(out_dir, "sw_full.npy"))
    feat_mean = np.load(os.path.join(out_dir, "feat_mean.npy"))
    feat_std = np.load(os.path.join(out_dir, "feat_std.npy"))
    keep_idx = np.load(os.path.join(out_dir, "keep_idx.npy"))
    print(f"  loaded precomputed in {time.time()-t0:.1f}s: X={X_full.shape} target_scale={target_scale:.4f}", flush=True)

    # CPU tensors (too large for GPU: 4.4M × 359 × 4 bytes = 6.4 GB)
    X_t = torch.from_numpy(X_full)
    y_t = torch.from_numpy(y_full).float()
    sw_t = torch.from_numpy(sw_full).float()

    # WandB
    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wm
            wandb = wm
            wandb.init(
                project="liangwenbei",
                entity="cjxh21-Tsinghua University",
                name=f"T172-GroupTr-K{args.n_tokens}-seed{seed}-{datetime.now().strftime('%H%M%S')}",
                config={"task": "T172", "arch": "GroupTransformer", "seed": seed,
                        "n_tokens": args.n_tokens, "d_token": args.d_token,
                        "n_heads": args.n_heads, "depth": args.depth,
                        "epochs": args.epochs, "lr": args.lr,
                        "batch_size": args.batch_size},
                tags=["T172", "GroupTransformer", f"seed{seed}", "full_retrain"],
                reinit=True,
            )
        except Exception as e:
            print(f"  wandb failed: {e}; skip", flush=True)
            use_wandb = False

    # Build model
    model = GroupTransformer(feat_dim, n_tokens=args.n_tokens, d_token=args.d_token,
                             n_heads=args.n_heads, depth=args.depth,
                             dim_feedforward=dim_ff, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params={n_params:,}", flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    use_amp = torch.cuda.is_available()

    n_train = len(X_t)
    t_train = time.time()
    progress("training", seed=seed, n_train=n_train)

    for epoch in range(args.epochs):
        t_ep = time.time()
        model.train()
        perm = torch.randperm(n_train, device="cpu")
        running_loss = 0.0; running_n = 0
        for s in range(0, n_train, args.batch_size):
            idx = perm[s:s+args.batch_size]
            xb = X_t[idx].to(device, non_blocking=True)
            yb = y_t[idx].to(device, non_blocking=True)
            wb = sw_t[idx].to(device, non_blocking=True)
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pred = model(xb)
                    loss = ((pred - yb) ** 2 * wb).sum() / wb.sum().clamp_min(1.0)
            else:
                pred = model(xb)
                loss = ((pred - yb) ** 2 * wb).sum() / wb.sum().clamp_min(1.0)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            running_loss += float(loss.item()) * xb.size(0)
            running_n += xb.size(0)
        sched.step()
        avg_loss = running_loss / running_n
        ep_t = time.time() - t_ep
        print(f"  ep {epoch:3d}  loss={avg_loss:.5e}  lr={optim.param_groups[0]['lr']:.2e}  ({ep_t:.1f}s)",
              flush=True)
        progress("training", seed=seed, epoch=epoch, loss=avg_loss, epoch_time=ep_t)
        if use_wandb and wandb:
            try:
                wandb.log({"epoch": epoch, "train_loss": avg_loss,
                           "lr": optim.param_groups[0]["lr"], "epoch_time": ep_t})
            except Exception:
                pass

    train_time = time.time() - t_train
    print(f"  training done in {train_time:.1f}s", flush=True)

    # Save npz
    out_prefix = os.path.join(out_dir, f"nn_tr_h{H}_seed{seed}")
    save_npz(model, feat_mean, feat_std, feat_dim, keep_idx,
             target_scale, clip, args.n_heads, out_prefix)

    # Sample predictions sanity
    model.eval()
    with torch.no_grad():
        xsample = X_t[:2048].to(device)
        psample = (model(xsample.float()).detach().cpu().numpy() / target_scale)
    print(f"  sample: mean={psample.mean():.5e} std={psample.std():.5e}", flush=True)

    summary = {
        "task": "T172", "arch": "GroupTransformer",
        "seed": seed, "horizon": H,
        "n_features": feat_dim, "n_params": int(n_params),
        "n_tokens": args.n_tokens, "d_token": args.d_token,
        "n_heads": args.n_heads, "depth": args.depth, "dim_ff": dim_ff,
        "epochs": args.epochs, "train_time_sec": float(train_time),
        "target_scale": float(target_scale),
    }
    with open(os.path.join(out_dir, f"summary_T172_seed{seed}.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    if use_wandb and wandb:
        try:
            wandb.log({"train_time_sec": float(train_time)})
            wandb.finish()
        except Exception:
            pass

    print(f"DONE seed={seed}", flush=True)


if __name__ == "__main__":
    main()
