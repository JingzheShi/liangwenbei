"""T134: Direct PnL + Sharpe loss NN — warm-start from T87 SPO+ MLP.

Loss design (per-sample, in unscaled c-space):
    pos = tanh(alpha * pred_unscaled)        # smooth differentiable position
    pnl = pos * c - fee_per_sample * |pos|   # per-sample PnL
    sharpe = mean(pnl) / (std(pnl) + 1e-6)
    loss = -mean(pnl) - sharpe_lambda * sharpe

Where:
    pred_unscaled = pred_scaled / target_scale  (c-units, ~ Δmid_norm)
    c             = (mp_th - mp_t) / (mp_t + 1)
    fee_per_sample = FEE * ((mp_th+1)+(mp_t+1)) / (mp_t+1)   ≈ 2·FEE

Warm-start from T87 SPO+ checkpoint (model_T87_seed{S}_main.pt).
Fine-tune at small LR for 5-10 epochs.

Output: model_T134_seed{S}.pt, pred_T134_seed{S}.parquet
"""
from __future__ import annotations

import argparse
import json
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
T87_DIR = os.path.join(ROOT, "experiments", "T87_spo_dfl")

NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)
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


def predict_chunked(model, X, batch=16384, device="cuda"):
    model.eval()
    out = np.zeros(len(X), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(X), batch):
            xb = X[s:s+batch].to(device, non_blocking=True)
            yb = model(xb).detach().cpu().numpy()
            out[s:s+batch] = yb
    return out


def evaluate_pnl(pred_unscaled, mp_t, mp_th, thr_up=2*FEE, thr_dn=2*FEE):
    """EV-gate cum_pnl evaluation with hard 3-class action."""
    pred_action = np.full(len(pred_unscaled), 1, dtype=np.int8)
    pred_action[pred_unscaled > thr_up] = 2
    pred_action[pred_unscaled < -thr_dn] = 0
    side = pred_action.astype(np.float64) - 1.0
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    pnl = (side * diff - fee_pnl) / denom
    return float(pnl.sum()), int((pred_action != 1).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=200.0,
                    help="Position scaling: pos = tanh(alpha * pred_unscaled). "
                         "alpha~1/(2*FEE)=5000 → too sharp; 200 → tanh saturates near "
                         "pred~5e-3 (~25*FEE), reasonable for our pred range.")
    ap.add_argument("--sharpe-lambda", type=float, default=0.1)
    ap.add_argument("--use-aug", action="store_true", default=True)
    ap.add_argument("--no-aug", dest="use_aug", action="store_false")
    ap.add_argument("--aug-lo", type=float, default=0.80)
    ap.add_argument("--aug-hi", type=float, default=1.20)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--cuda", default="0")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    H = args.horizon
    seed = args.seed
    tag = args.tag if args.tag else f"a{args.alpha:g}_sl{args.sharpe_lambda:g}"
    print(f"=== T134 Direct PnL+Sharpe seed={seed} h={H} alpha={args.alpha} "
          f"sharpe_lambda={args.sharpe_lambda} tag={tag} ===", flush=True)
    print(f"  device={device}", flush=True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    progress("loading_t87_init", seed=seed)
    t87_pt = os.path.join(T87_DIR, f"model_T87_seed{seed}_main.pt")
    print(f"  warm-start from {t87_pt}", flush=True)
    ckpt = torch.load(t87_pt, map_location="cpu", weights_only=False)
    feat_mean = np.asarray(ckpt["feat_mean"], dtype=np.float32)
    feat_std = np.asarray(ckpt["feat_std"], dtype=np.float32)
    keep_idx = np.asarray(ckpt["keep_idx"], dtype=np.int64)
    target_scale = float(ckpt["target_scale"])
    hidden = tuple(int(x) for x in ckpt["hidden"])
    dropout = float(ckpt["dropout"])
    use_layernorm = bool(ckpt.get("use_layernorm", True))
    feat_dim = int(ckpt["in_dim"])
    clip_val = float(ckpt.get("clip", 10.0))
    print(f"  T87 ckpt: hidden={hidden} dropout={dropout} use_ln={use_layernorm} "
          f"target_scale={target_scale:.4f} feat_dim={feat_dim} clip={clip_val}",
          flush=True)

    progress("loading_caches", seed=seed)
    t0 = time.time()
    train_full = load_split("train")
    test_full = load_split("test")
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    feat_names_path = os.path.join(CACHE_DIR, "schemeP_feat_names.txt")
    with open(feat_names_path) as f:
        all_feat_names = [line.strip() for line in f]
    feat_names = [all_feat_names[i] for i in keep_idx]
    forbidden = {"date", "sym", "time"}
    assert not (forbidden & set(feat_names)), f"FORBIDDEN feat leak"

    # V4 walk-forward (matches T87/T81 exactly)
    date_tr = train_full["date"]
    m_va = date_tr >= 76
    m_t = ~m_va

    X_tr = train_full["X"][m_t][:, keep_idx]
    y_cls_tr = train_full["y60"][m_t].astype(np.int64)
    y_regr_tr = regr_target(train_full["mp_t"][m_t], train_full["mp_t60"][m_t])
    mp_t_tr = train_full["mp_t"][m_t]
    mp_th_tr = train_full["mp_t60"][m_t]
    fee_eff_tr = fee_eff(mp_t_tr, mp_th_tr)
    # delta_mid in raw c-units (regr_target IS c)
    c_tr = y_regr_tr.copy()  # already c

    X_va = train_full["X"][m_va][:, keep_idx]
    y_cls_va = train_full["y60"][m_va].astype(np.int64)
    y_regr_va = regr_target(train_full["mp_t"][m_va], train_full["mp_t60"][m_va])
    mp_t_va = train_full["mp_t"][m_va]
    mp_th_va = train_full["mp_t60"][m_va]

    X_te = test_full["X"][:, keep_idx]
    y_cls_te = test_full[f"y{H}"].astype(np.int64)
    y_regr_te = regr_target(test_full["mp_t"], test_full[f"mp_t{H}"])
    mp_t_te = test_full["mp_t"]
    mp_th_te = test_full[f"mp_t{H}"]
    sym_te = test_full["sym"]

    print(f"  V4 split: n_train={len(X_tr):,} n_val={len(X_va):,} n_test={len(X_te):,}", flush=True)

    # NaN imputation
    nan_mask_tr = np.isnan(X_tr)
    if nan_mask_tr.any():
        for d_idx in np.where(nan_mask_tr.any(axis=0))[0]:
            col_nan = np.isnan(X_tr[:, d_idx])
            X_tr[col_nan, d_idx] = feat_mean[d_idx]

    def standardize(X):
        Xs = (X - feat_mean) / np.maximum(feat_std, 1e-6)
        Xs = np.where(np.isnan(Xs), 0.0, Xs)
        Xs = np.clip(Xs, -clip_val, clip_val)
        return Xs.astype(np.float32)

    X_va_std = torch.from_numpy(standardize(X_va))
    X_te_std = torch.from_numpy(standardize(X_te))
    del train_full

    # Augmentation: same scheme as T87 (concat scale aug)
    if args.use_aug:
        seed_rng = np.random.default_rng(seed * 7919 + 137)
        n_orig = len(X_tr)
        X_tr_full = np.empty((2 * n_orig, X_tr.shape[1]), dtype=np.float32)
        X_tr_full[:n_orig] = X_tr
        for i in range(0, n_orig, 100_000):
            j = min(i + 100_000, n_orig)
            scales = seed_rng.uniform(args.aug_lo, args.aug_hi,
                                      size=(j - i, X_tr.shape[1])).astype(np.float32)
            X_tr_full[n_orig + i:n_orig + j] = X_tr[i:j] * scales
        del X_tr
        c_tr_full = np.concatenate([c_tr, c_tr], axis=0)
        y_cls_tr_full = np.concatenate([y_cls_tr, y_cls_tr], axis=0)
        fee_eff_tr_full = np.concatenate([fee_eff_tr, fee_eff_tr], axis=0)
        print(f"  aug=concat: X_tr_full={X_tr_full.shape}", flush=True)
    else:
        X_tr_full = X_tr
        c_tr_full = c_tr
        y_cls_tr_full = y_cls_tr
        fee_eff_tr_full = fee_eff_tr

    sw_tr_full = class_balanced_weight(y_cls_tr_full, num_class=NUM_CLASS).astype(np.float32)
    print(f"  sw_tr_full mean={sw_tr_full.mean():.4f}", flush=True)

    # Tensors
    X_tr_raw = torch.from_numpy(X_tr_full.astype(np.float32))
    c_tr_t = torch.from_numpy(c_tr_full.astype(np.float32))
    fee_tr_t = torch.from_numpy(fee_eff_tr_full.astype(np.float32))
    sw_tr_t = torch.from_numpy(sw_tr_full)

    # Build model + load T87 weights
    model = MLPRegr(feat_dim, hidden=hidden, dropout=dropout,
                    use_layernorm=use_layernorm).to(device)
    model.load_state_dict(ckpt["state_dict"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: hidden={hidden} dropout={dropout} params={n_params:,}", flush=True)

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
            run_name = f"T134-pnl-seed{seed}-{tag}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            wandb.init(
                project="liangwenbei",
                entity="cjxh21-Tsinghua University",
                name=run_name,
                config={
                    "task": "T134_direct_pnl",
                    "warm_start": "T87",
                    "alpha": args.alpha,
                    "sharpe_lambda": args.sharpe_lambda,
                    "lr": args.lr,
                    "epochs": args.epochs,
                    "seed": seed,
                    "horizon": H,
                    **{k: v for k, v in vars(args).items()},
                },
                tags=["T134", "direct_pnl", "sharpe", f"seed{seed}", f"h{H}"],
            )
        except Exception as e:
            print(f"  WandB init failed ({e!r}); skipping", flush=True)
            use_wandb = False

    # Initial eval (== T87 baseline)
    va_pred_init = predict_chunked(model, X_va_std, batch=16384, device=device) / target_scale
    te_pred_init = predict_chunked(model, X_te_std, batch=16384, device=device) / target_scale
    va_corr_init = float(np.corrcoef(va_pred_init, y_regr_va)[0, 1])
    te_corr_init = float(np.corrcoef(te_pred_init, y_regr_te)[0, 1])
    val_pnl_init, val_n_act_init = evaluate_pnl(va_pred_init, mp_t_va, mp_th_va)
    cum_init, n_act_init = evaluate_pnl(te_pred_init, mp_t_te, mp_th_te)
    print(f"\n  T87 init eval:  va_corr={va_corr_init:.4f}  te_corr={te_corr_init:.4f}  "
          f"val_pnl={val_pnl_init:+.4f}  test_pnl={cum_init:+.4f}", flush=True)

    best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
    best_val_pnl = val_pnl_init
    best_te_pnl = cum_init
    best_epoch = -1

    # Training loop
    n_train = len(X_tr_raw)
    epoch_logs = []
    bad_epochs = 0
    progress("training", seed=seed, n_train=n_train)
    t_train_start = time.time()
    for epoch in range(args.epochs):
        t_ep = time.time()
        model.train()
        perm = torch.randperm(n_train)
        running_pnl = 0.0
        running_sharpe = 0.0
        running_n = 0

        for s in range(0, n_train, args.batch_size):
            idx = perm[s:s+args.batch_size]
            xb_raw = X_tr_raw[idx].to(device, non_blocking=True)
            cb = c_tr_t[idx].to(device, non_blocking=True)
            fb = fee_tr_t[idx].to(device, non_blocking=True)
            wb = sw_tr_t[idx].to(device, non_blocking=True)

            xb = (xb_raw - fm_t) / fs_t
            xb = torch.clamp(xb, -clip_val, clip_val)

            pred_scaled = model(xb)
            pred_unscaled = pred_scaled / target_scale
            pos = torch.tanh(args.alpha * pred_unscaled)
            pnl_per = pos * cb - fb * torch.abs(pos)

            # weighted PnL terms
            mean_pnl = (pnl_per * wb).sum() / wb.sum().clamp_min(1.0)
            # variance via weighted second moment
            sq = ((pnl_per - mean_pnl) ** 2 * wb).sum() / wb.sum().clamp_min(1.0)
            std_pnl = torch.sqrt(sq + 1e-12)
            sharpe = mean_pnl / (std_pnl + 1e-6)
            loss = -mean_pnl - args.sharpe_lambda * sharpe

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            bs = xb.size(0)
            running_pnl += float(mean_pnl.item()) * bs
            running_sharpe += float(sharpe.item()) * bs
            running_n += bs

        scheduler.step()
        train_pnl = running_pnl / running_n
        train_sharpe = running_sharpe / running_n

        va_pred_s = predict_chunked(model, X_va_std, batch=16384, device=device)
        va_pred = va_pred_s / target_scale
        val_corr = float(np.corrcoef(va_pred, y_regr_va)[0, 1]) if va_pred.std() > 0 else 0.0
        val_pnl, val_n_act = evaluate_pnl(va_pred, mp_t_va, mp_th_va)

        te_pred_s = predict_chunked(model, X_te_std, batch=16384, device=device)
        te_pred = te_pred_s / target_scale
        te_corr = float(np.corrcoef(te_pred, y_regr_te)[0, 1]) if te_pred.std() > 0 else 0.0
        te_pnl, te_n_act = evaluate_pnl(te_pred, mp_t_te, mp_th_te)

        ep_time = time.time() - t_ep
        cur_lr = optimizer.param_groups[0]["lr"]
        log = {"epoch": epoch, "train_pnl": train_pnl, "train_sharpe": train_sharpe,
               "val_corr": val_corr, "val_pnl": val_pnl, "val_n_act": val_n_act,
               "te_corr": te_corr, "te_pnl": te_pnl, "te_n_act": te_n_act,
               "lr": cur_lr, "time": ep_time}
        epoch_logs.append(log)
        print(f"  ep {epoch:3d}  train_pnl={train_pnl:+.6e}  train_sharpe={train_sharpe:+.4f}  "
              f"val_corr={val_corr:.4f} val_pnl={val_pnl:+.3f}  te_pnl={te_pnl:+.3f}  "
              f"lr={cur_lr:.2e}  ({ep_time:.1f}s)", flush=True)
        if use_wandb:
            wandb.log(log)

        if val_pnl > best_val_pnl:
            best_val_pnl = val_pnl
            best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
            best_te_pnl = te_pnl
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"  early stop at epoch {epoch}", flush=True)
                break

    train_time = time.time() - t_train_start
    print(f"\n  train done in {train_time:.1f}s; best_epoch={best_epoch} "
          f"best_val_pnl={best_val_pnl:+.3f}", flush=True)
    model.load_state_dict(best_state)

    model_path = os.path.join(HERE, f"model_T134_seed{seed}.pt")
    torch.save({
        "state_dict": best_state,
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
        "best_epoch": best_epoch,
        "alpha": args.alpha,
        "sharpe_lambda": args.sharpe_lambda,
        "warm_start_from": "T87",
    }, model_path)
    print(f"  saved {model_path}", flush=True)

    va_pred = predict_chunked(model, X_va_std, batch=16384, device=device) / target_scale
    te_pred = predict_chunked(model, X_te_std, batch=16384, device=device) / target_scale
    va_corr = float(np.corrcoef(va_pred, y_regr_va)[0, 1])
    te_corr = float(np.corrcoef(te_pred, y_regr_te)[0, 1])
    va_pnl, va_n_act = evaluate_pnl(va_pred, mp_t_va, mp_th_va)
    te_pnl, te_n_act = evaluate_pnl(te_pred, mp_t_te, mp_th_te)
    print(f"\n  FINAL val:  corr={va_corr:.4f} cum_pnl={va_pnl:+.4f} n_act={va_n_act:,}", flush=True)
    print(f"  FINAL test: corr={te_corr:.4f} cum_pnl={te_pnl:+.4f} n_act={te_n_act:,}", flush=True)
    print(f"  vs T87 init: te_pnl change={te_pnl - cum_init:+.3f}", flush=True)

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
    pred_path = os.path.join(HERE, f"pred_T134_seed{seed}.parquet")
    te_df.to_parquet(pred_path, index=False)
    print(f"  saved {pred_path} ({len(te_df):,} rows)", flush=True)

    summary = {
        "task": f"T134 direct PnL+Sharpe seed={seed}",
        "seed": seed, "horizon": H, "tag": tag,
        "alpha": args.alpha, "sharpe_lambda": args.sharpe_lambda,
        "warm_start_t87": True,
        "n_features": feat_dim, "hidden": list(hidden), "dropout": dropout,
        "lr": args.lr, "epochs_run": len(epoch_logs),
        "best_epoch": best_epoch, "n_params": int(n_params),
        "train_time_sec": float(train_time),
        "init": {"te_corr": te_corr_init, "te_pnl": cum_init, "te_n_act": n_act_init,
                 "val_pnl": val_pnl_init},
        "val": {"corr": va_corr, "ev_gate_k1_cum_pnl": va_pnl, "ev_gate_k1_n_active": va_n_act},
        "test": {"corr": te_corr, "ev_gate_k1_cum_pnl": te_pnl, "ev_gate_k1_n_active": te_n_act},
        "delta_te_pnl_vs_t87": te_pnl - cum_init,
        "epoch_logs": epoch_logs,
        "params": vars(args),
    }
    out_path = os.path.join(HERE, f"summary_T134_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"summary -> {out_path}", flush=True)

    if use_wandb:
        wandb.log({
            "best_epoch": best_epoch,
            "best_val_pnl": best_val_pnl,
            "final_te_corr": te_corr,
            "final_te_pnl": te_pnl,
            "delta_vs_t87_init": te_pnl - cum_init,
        })
        wandb.finish()

    progress("done", seed=seed, best_epoch=best_epoch, te_pnl=te_pnl,
             delta=te_pnl - cum_init)


if __name__ == "__main__":
    main()
