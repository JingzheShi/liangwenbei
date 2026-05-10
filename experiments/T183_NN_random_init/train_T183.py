"""T183: T87-architecture NN with RANDOM Xavier init (no T81 warm-start), M7 retrain.

Goal: test whether T81 warm-start was a lucky anchor or not.
Key difference from T170:
  - Same architecture as T87/T170 (MLP 256-128-64 + LayerNorm + GELU)
  - Same training config: full 0-119 data, 11 epochs fixed, lr=3e-5, lambda_spo=30
  - DIFFERENT init: Xavier random init instead of T81 warm-start weights
  - Still uses T81 checkpoint for: feat_mean, feat_std, keep_idx, target_scale, arch dims
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
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
T81_DIR = os.path.join(ROOT, "experiments", "T81_nn_regression_pnl")

NUM_CLASS = 3
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
    print(f"  loading {p} ...", flush=True)
    d = np.load(p)
    return {k: d[k] for k in d.files}


def regr_target(mp_t, mp_th):
    return ((mp_th.astype(np.float64) - mp_t.astype(np.float64))
            / (mp_t.astype(np.float64) + 1.0)).astype(np.float32)


def fee_eff(mp_t, mp_th):
    return (FEE * ((mp_th.astype(np.float64) + 1.0) + (mp_t.astype(np.float64) + 1.0))
            / (mp_t.astype(np.float64) + 1.0)).astype(np.float32)


class MLPRegr(nn.Module):
    def __init__(self, in_dim, hidden=(256, 128, 64), dropout=0.10, use_layernorm=True):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            if use_layernorm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def predict_chunked(model, X, batch=16384, device="cuda"):
    model.eval()
    out = np.zeros(len(X), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(X), batch):
            xb = X[s:s+batch].to(device, non_blocking=True)
            yb = model(xb).detach().cpu().numpy()
            out[s:s+batch] = yb
    return out


def spo_plus_loss(pred_scaled, y_scaled, fee_scaled):
    z_star = torch.where(
        y_scaled > fee_scaled, torch.ones_like(y_scaled),
        torch.where(y_scaled < -fee_scaled, -torch.ones_like(y_scaled),
                    torch.zeros_like(y_scaled)),
    )
    abs_z_star = torch.abs(z_star)
    spread = 2.0 * pred_scaled - y_scaled
    relu_max = F.relu(torch.abs(spread) - fee_scaled)
    ell = relu_max - z_star * spread + fee_scaled * abs_z_star
    return ell


def evaluate_pnl(pred_unscaled, mp_t, mp_th, thr_up=2*FEE, thr_dn=2*FEE):
    pred_action = np.full(len(pred_unscaled), 1, dtype=np.int8)
    pred_action[pred_unscaled > thr_up] = 2
    pred_action[pred_unscaled < -thr_dn] = 0
    side = pred_action.astype(np.float64) - 1.0
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    pnl = (side * diff - fee_pnl) / denom
    return float(pnl.sum()), int((pred_action != 1).sum())


def extract_npz(pt_path, npz_path, ckpt=None):
    if ckpt is None:
        ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    hidden = ckpt["hidden"]
    in_dim = ckpt["in_dim"]
    use_layernorm = ckpt.get("use_layernorm", True)
    target_scale = float(ckpt.get("target_scale", 1.0))
    feat_mean = np.asarray(ckpt["feat_mean"], dtype=np.float32)
    feat_std = np.asarray(ckpt["feat_std"], dtype=np.float32)
    keep_idx = np.asarray(ckpt["keep_idx"], dtype=np.int64)
    clip_val = float(ckpt.get("clip", 10.0))

    out = {
        "in_dim": np.array([in_dim], dtype=np.int64),
        "hidden": np.array(list(hidden), dtype=np.int64),
        "use_layernorm": np.array([1 if use_layernorm else 0], dtype=np.int8),
        "target_scale": np.array([target_scale], dtype=np.float32),
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "keep_idx": keep_idx,
        "clip": np.array([clip_val], dtype=np.float32),
    }
    layer_idx = 0
    for i in range(len(hidden)):
        out[f"L{i}_W"] = sd[f"net.{layer_idx}.weight"].numpy().astype(np.float32)
        out[f"L{i}_b"] = sd[f"net.{layer_idx}.bias"].numpy().astype(np.float32)
        layer_idx += 1
        if use_layernorm:
            out[f"LN{i}_W"] = sd[f"net.{layer_idx}.weight"].numpy().astype(np.float32)
            out[f"LN{i}_b"] = sd[f"net.{layer_idx}.bias"].numpy().astype(np.float32)
            layer_idx += 1
        layer_idx += 2  # GELU + Dropout
    out["LF_W"] = sd[f"net.{layer_idx}.weight"].numpy().astype(np.float32)
    out["LF_b"] = sd[f"net.{layer_idx}.bias"].numpy().astype(np.float32)
    np.savez(npz_path, **out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=11,
                    help="Fixed epochs (no early stop), same as T170")
    ap.add_argument("--lambda-spo", type=float, default=30.0)
    ap.add_argument("--aug-lo", type=float, default=0.80)
    ap.add_argument("--aug-hi", type=float, default=1.20)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--cuda", default="0")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    H = args.horizon
    seed = args.seed
    print(f"=== T183 RANDOM INIT M7 full-retrain seed={seed} h={H} lambda_spo={args.lambda_spo} "
          f"epochs={args.epochs} ===", flush=True)
    print(f"  device={device}", flush=True)
    print(f"  NOTE: Xavier random init (NO T81 warm-start weights)", flush=True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Load T81 for architecture config + feature stats ONLY (not weights)
    progress("loading_t81_config", seed=seed)
    t81_pt = os.path.join(T81_DIR, f"model_T81_seed{seed}.pt")
    print(f"  loading T81 config from {t81_pt} (arch + feat stats only, NOT weights)", flush=True)
    ckpt = torch.load(t81_pt, map_location="cpu", weights_only=False)
    feat_mean = np.asarray(ckpt["feat_mean"], dtype=np.float32)
    feat_std = np.asarray(ckpt["feat_std"], dtype=np.float32)
    keep_idx = np.asarray(ckpt["keep_idx"], dtype=np.int64)
    target_scale = float(ckpt["target_scale"])
    hidden = tuple(int(x) for x in ckpt["hidden"])
    dropout = float(ckpt["dropout"])
    use_layernorm = bool(ckpt.get("use_layernorm", True))
    feat_dim = int(ckpt["in_dim"])
    clip_val = float(ckpt.get("clip", 10.0))
    print(f"  T81 config: hidden={hidden} dropout={dropout} use_ln={use_layernorm} "
          f"target_scale={target_scale:.4f} feat_dim={feat_dim}", flush=True)

    feat_names_path = os.path.join(CACHE_DIR, "schemeP_feat_names.txt")
    with open(feat_names_path) as f:
        all_feat_names = [line.strip() for line in f]
    feat_names = [all_feat_names[i] for i in keep_idx]
    forbidden = {"date", "sym", "time"}
    assert not (forbidden & set(feat_names)), f"FORBIDDEN feat leak: {forbidden & set(feat_names)}"

    # M7: load ALL three splits — train(0-79) + val(80-95) + test(96-119)
    progress("loading_all_data", seed=seed)
    t0 = time.time()
    train_d = load_split("train")
    val_d = load_split("val")
    test_d = load_split("test")
    print(f"loaded all splits in {time.time()-t0:.1f}s", flush=True)
    print(f"  train: {len(train_d['date']):,} rows (date {train_d['date'].min()}-{train_d['date'].max()})",
          flush=True)
    print(f"  val:   {len(val_d['date']):,} rows (date {val_d['date'].min()}-{val_d['date'].max()})",
          flush=True)
    print(f"  test:  {len(test_d['date']):,} rows (date {test_d['date'].min()}-{test_d['date'].max()})",
          flush=True)

    # Concat all data
    X_all = np.concatenate([train_d["X"], val_d["X"], test_d["X"]], axis=0)[:, keep_idx]
    mp_t_all = np.concatenate([train_d["mp_t"], val_d["mp_t"], test_d["mp_t"]])
    mp_th_all = np.concatenate([train_d[f"mp_t{H}"], val_d[f"mp_t{H}"], test_d[f"mp_t{H}"]])
    y_cls_all = np.concatenate([train_d[f"y{H}"], val_d[f"y{H}"], test_d[f"y{H}"]]).astype(np.int64)

    y_regr_all = regr_target(mp_t_all, mp_th_all)
    fee_eff_all = fee_eff(mp_t_all, mp_th_all)
    print(f"  total rows: {len(X_all):,}", flush=True)

    del train_d, val_d, test_d

    # NaN imputation
    nan_mask = np.isnan(X_all)
    if nan_mask.any():
        for d_idx in np.where(nan_mask.any(axis=0))[0]:
            col_nan = np.isnan(X_all[:, d_idx])
            X_all[col_nan, d_idx] = feat_mean[d_idx]
        print(f"  imputed {int(nan_mask.sum()):,} NaN cells", flush=True)

    def standardize(X):
        Xs = (X - feat_mean) / np.maximum(feat_std, 1e-6)
        Xs = np.where(np.isnan(Xs), 0.0, Xs)
        Xs = np.clip(Xs, -clip_val, clip_val)
        return Xs.astype(np.float32)

    # Aug: concat scheme
    seed_rng = np.random.default_rng(seed * 7919 + 137)
    n_orig = len(X_all)
    X_full = np.empty((2 * n_orig, X_all.shape[1]), dtype=np.float32)
    X_full[:n_orig] = X_all
    for i in range(0, n_orig, 100_000):
        j = min(i + 100_000, n_orig)
        scales = seed_rng.uniform(args.aug_lo, args.aug_hi,
                                  size=(j - i, X_all.shape[1])).astype(np.float32)
        X_full[n_orig + i:n_orig + j] = X_all[i:j] * scales
    del X_all
    y_regr_full = np.concatenate([y_regr_all, y_regr_all])
    y_cls_full = np.concatenate([y_cls_all, y_cls_all])
    fee_eff_full = np.concatenate([fee_eff_all, fee_eff_all])
    print(f"  aug=concat: X_full={X_full.shape}", flush=True)

    y_scaled_full = (y_regr_full * target_scale).astype(np.float32)
    fee_scaled_full = (fee_eff_full * target_scale).astype(np.float32)
    sw_full = class_balanced_weight(y_cls_full, num_class=NUM_CLASS).astype(np.float32)

    X_tr_raw = torch.from_numpy(X_full.astype(np.float32))
    y_scaled_t = torch.from_numpy(y_scaled_full)
    fee_scaled_t = torch.from_numpy(fee_scaled_full)
    sw_t = torch.from_numpy(sw_full)

    # Build model with XAVIER RANDOM INIT (no T81 weight loading)
    model = MLPRegr(feat_dim, hidden=hidden, dropout=dropout,
                    use_layernorm=use_layernorm).to(device)
    model.apply(xavier_init)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: hidden={hidden} dropout={dropout} params={n_params:,}", flush=True)
    print(f"  init: Xavier normal (NO warm-start from T81)", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    fm_t = torch.from_numpy(feat_mean).to(device)
    fs_t = torch.from_numpy(np.maximum(feat_std, 1e-6)).to(device)

    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = f"T183-RandInit-seed{seed}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            wandb.init(
                project="liangwenbei",
                entity="cjxh21-Tsinghua University",
                name=run_name,
                config={
                    "task": "T183_NN_random_init",
                    "warm_start": "NONE_xavier_random",
                    "lambda_spo": args.lambda_spo,
                    "lr": args.lr,
                    "epochs": args.epochs,
                    "seed": seed,
                    "horizon": H,
                    "M7_full_retrain": True,
                    "data_dates": "0-119",
                    "init": "xavier_normal",
                    **{k: v for k, v in vars(args).items()},
                },
                tags=["T183", "M7", "SPO+", "full-retrain", "random-init", f"seed{seed}"],
            )
        except Exception as e:
            print(f"  WandB init failed ({e!r}); skipping", flush=True)
            use_wandb = False

    # Training loop — fixed epochs, no early stopping
    n_train = len(X_tr_raw)
    epoch_logs = []
    progress("training", seed=seed, n_train=n_train, epochs=args.epochs)
    t_train_start = time.time()
    for epoch in range(args.epochs):
        t_ep = time.time()
        model.train()
        perm = torch.randperm(n_train)
        running_l2 = 0.0
        running_spo = 0.0
        running_n = 0

        for s in range(0, n_train, args.batch_size):
            idx = perm[s:s+args.batch_size]
            xb_raw = X_tr_raw[idx].to(device, non_blocking=True)
            yb = y_scaled_t[idx].to(device, non_blocking=True)
            fb = fee_scaled_t[idx].to(device, non_blocking=True)
            wb = sw_t[idx].to(device, non_blocking=True)

            xb = (xb_raw - fm_t) / fs_t
            xb = torch.clamp(xb, -clip_val, clip_val)

            pred = model(xb)
            l2_per = (pred - yb) ** 2
            spo_per = spo_plus_loss(pred, yb, fb)

            l2_loss = (l2_per * wb).sum() / wb.sum().clamp_min(1.0)
            spo_loss = (spo_per * wb).sum() / wb.sum().clamp_min(1.0)
            loss = l2_loss + args.lambda_spo * spo_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            bs = xb.size(0)
            running_l2 += float(l2_loss.item()) * bs
            running_spo += float(spo_loss.item()) * bs
            running_n += bs

        scheduler.step()
        train_l2 = running_l2 / running_n
        train_spo = running_spo / running_n
        ep_time = time.time() - t_ep
        cur_lr = optimizer.param_groups[0]["lr"]

        log = {"epoch": epoch, "train_l2": train_l2, "train_spo": train_spo,
               "lr": cur_lr, "time": ep_time}
        epoch_logs.append(log)
        print(f"  ep {epoch:3d}  l2={train_l2:.6e}  spo={train_spo:.6e}  "
              f"lr={cur_lr:.2e}  ({ep_time:.1f}s)", flush=True)
        if use_wandb:
            wandb.log(log)

        progress("training", seed=seed, epoch=epoch, train_l2=float(train_l2))

    train_time = time.time() - t_train_start
    print(f"\n  train done in {train_time:.1f}s ({args.epochs} epochs, no early stop)", flush=True)

    # Save model
    final_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
    pt_path = os.path.join(HERE, f"nn_h60_seed{seed}.pt")
    save_ckpt = {
        "state_dict": final_state,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "keep_idx": keep_idx,
        "feat_names": feat_names,
        "hidden": hidden,
        "dropout": dropout,
        "use_layernorm": use_layernorm,
        "in_dim": feat_dim,
        "target_scale": float(target_scale),
        "clip": float(clip_val),
        "epochs_fixed": args.epochs,
        "lambda_spo": args.lambda_spo,
        "warm_start_from": "NONE_xavier_random",
        "M7_full_retrain": True,
        "data_dates": "0-119",
    }
    torch.save(save_ckpt, pt_path)
    print(f"  saved {pt_path}", flush=True)

    # Extract .npz for Predictor
    npz_path = os.path.join(HERE, f"nn_h60_seed{seed}.npz")
    extract_npz(pt_path, npz_path, ckpt=save_ckpt)
    print(f"  saved {npz_path}", flush=True)

    summary = {
        "task": f"T183 random-init M7 full-retrain seed={seed}",
        "seed": seed,
        "horizon": H,
        "lambda_spo": args.lambda_spo,
        "M7_full_retrain": True,
        "data_dates": "0-119",
        "warm_start": "NONE_xavier_random",
        "n_features": feat_dim,
        "hidden": list(hidden),
        "dropout": dropout,
        "lr": args.lr,
        "epochs_fixed": args.epochs,
        "n_params": int(n_params),
        "train_time_sec": float(train_time),
        "epoch_logs": epoch_logs,
        "params": vars(args),
    }
    out_path = os.path.join(HERE, f"summary_T183_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"summary -> {out_path}", flush=True)

    if use_wandb:
        wandb.log({"epochs_fixed": args.epochs, "train_time_sec": train_time})
        wandb.finish()

    progress("done", seed=seed, epochs_fixed=args.epochs, train_time_sec=float(train_time))
    print(f"\nDONE seed={seed}", flush=True)


if __name__ == "__main__":
    main()
