"""T88 NN regression: schemeP + 12 range-vol features.

Mirrors T81 train_nn_regr.py exactly with one change: load schemeP cache + T88
cache and concatenate features.
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
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

T26_DIR = os.path.join(ROOT, "experiments", "T26_domain_randomization")
sys.path.insert(0, T26_DIR)
from build_aug import class_balanced_weight  # noqa: E402

T68_DIR = os.path.join(ROOT, "experiments", "T68_stage5_features")
SCHEMEP_CACHE = os.path.join(T68_DIR, "cache")
T88_CACHE = os.path.join(HERE, "cache")

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


def load_split_concat(split: str) -> dict:
    p_sp = os.path.join(SCHEMEP_CACHE, f"schemeP_{split}.npz")
    p_t88 = os.path.join(T88_CACHE, f"t88_range_vol_{split}.npz")
    print(f"  loading {p_sp} + {p_t88}", flush=True)
    sp = np.load(p_sp); t88 = np.load(p_t88)
    X_full = np.concatenate([sp["X"], t88["X_t88"].astype(np.float32)], axis=1)
    out = {k: sp[k] for k in sp.files if k != "X"}
    out["X"] = X_full
    return out


def regr_target(mp_t, mp_th):
    return ((mp_th.astype(np.float64) - mp_t.astype(np.float64))
            / (mp_t.astype(np.float64) + 1.0)).astype(np.float32)


class MLPRegr(nn.Module):
    def __init__(self, in_dim: int, hidden=(256, 128, 64), dropout: float = 0.10,
                 use_layernorm: bool = True):
        super().__init__()
        layers = []; d = in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            if use_layernorm: layers.append(nn.LayerNorm(h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x): return self.net(x).squeeze(-1)


def predict_chunked(model, X, batch=16384, device="cuda"):
    model.eval()
    out = np.zeros(len(X), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(X), batch):
            xb = X[s:s+batch].to(device, non_blocking=True)
            out[s:s+batch] = model(xb).detach().cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--hidden", default="256,128,64")
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--target-scale", type=float, default=0.0)
    ap.add_argument("--aug-mode", default="concat", choices=["concat", "online", "none"])
    ap.add_argument("--aug-lo", type=float, default=0.80)
    ap.add_argument("--aug-hi", type=float, default=1.20)
    ap.add_argument("--use-layernorm", action="store_true", default=True)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--cuda", default="0")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H = args.horizon; seed = args.seed
    print(f"=== T88 NN regression seed={seed} h={H} device={device} ===", flush=True)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)

    progress("loading_caches", seed=seed)
    t0 = time.time()
    train_full = load_split_concat("train")
    test_full = load_split_concat("test")
    print(f"loaded in {time.time()-t0:.1f}s | X shape: {train_full['X'].shape}", flush=True)

    sp_names_path = os.path.join(SCHEMEP_CACHE, "schemeP_feat_names.txt")
    with open(sp_names_path) as f: sp_names = [l.strip() for l in f]
    t88_names_path = os.path.join(T88_CACHE, "t88_feat_names.txt")
    with open(t88_names_path) as f: t88_names = [l.strip() for l in f]
    all_feat_names = sp_names + t88_names
    total_dim = train_full["X"].shape[1]
    assert len(all_feat_names) == total_dim and total_dim == 382

    name_to_idx = {n: i for i, n in enumerate(all_feat_names)}
    drop_idx = set(name_to_idx[n] for n in DROP_NAMES if n in name_to_idx)
    keep_idx = np.array([i for i in range(total_dim) if i not in drop_idx], dtype=np.int64)
    feat_dim = len(keep_idx)
    feat_names = [all_feat_names[i] for i in keep_idx]
    print(f"  feat_dim={feat_dim} (dropped {len(drop_idx)}); + 12 t88", flush=True)

    # V4 walk-forward
    date_tr = train_full["date"]
    m_va = date_tr >= 76; m_t = ~m_va
    X_tr = train_full["X"][m_t][:, keep_idx]
    y_cls_tr = train_full["y60"][m_t].astype(np.int64)
    y_regr_tr = regr_target(train_full["mp_t"][m_t], train_full["mp_t60"][m_t])
    X_va = train_full["X"][m_va][:, keep_idx]
    y_cls_va = train_full["y60"][m_va].astype(np.int64)
    y_regr_va = regr_target(train_full["mp_t"][m_va], train_full["mp_t60"][m_va])
    mp_t_va = train_full["mp_t"][m_va]; mp_th_va = train_full["mp_t60"][m_va]
    print(f"  V4: n_train={len(X_tr):,} n_val={len(X_va):,}", flush=True)

    # Standardization (global)
    feat_mean = np.nanmean(X_tr, axis=0).astype(np.float32)
    feat_std = np.maximum(np.nanstd(X_tr, axis=0).astype(np.float32), 1e-6)
    CLIP = 10.0

    def standardize(X):
        Xs = (X - feat_mean) / feat_std
        Xs = np.where(np.isnan(Xs), 0.0, Xs)
        return np.clip(Xs, -CLIP, CLIP).astype(np.float32)

    X_te = test_full["X"][:, keep_idx]
    y_cls_te = test_full[f"y{H}"].astype(np.int64)
    y_regr_te = regr_target(test_full["mp_t"], test_full[f"mp_t{H}"])
    mp_t_te = test_full["mp_t"]; mp_th_te = test_full[f"mp_t{H}"]

    X_va_std = torch.from_numpy(standardize(X_va))
    X_te_std = torch.from_numpy(standardize(X_te))

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

    if args.aug_mode == "concat":
        seed_rng = np.random.default_rng(seed * 7919 + 1)
        scales = seed_rng.uniform(args.aug_lo, args.aug_hi, size=X_tr_imputed.shape).astype(np.float32)
        X_aug = X_tr_imputed * scales
        X_tr_full = np.concatenate([X_tr_imputed, X_aug], axis=0)
        y_tr_full = np.concatenate([y_regr_tr_s, y_regr_tr_s], axis=0)
        y_cls_tr_full = np.concatenate([y_cls_tr, y_cls_tr], axis=0)
        del X_aug
    else:
        X_tr_full = X_tr_imputed; y_tr_full = y_regr_tr_s; y_cls_tr_full = y_cls_tr

    sw_tr_full = class_balanced_weight(y_cls_tr_full, num_class=NUM_CLASS)
    X_tr_raw = torch.from_numpy(X_tr_full.astype(np.float32))
    y_tr_t = torch.from_numpy(y_tr_full).float()
    sw_tr_t = torch.from_numpy(sw_tr_full).float()
    print(f"  X_tr_raw={X_tr_raw.shape}", flush=True)

    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            wandb.init(project="liangwenbei", entity="cjxh21-Tsinghua University",
                       name=f"T88-NN-seed{seed}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                       config={"task": "T88_range_vol", "seed": seed, "n_features": feat_dim,
                               "horizon": H, "model": "MLP-LayerNorm-GELU",
                               **{k: v for k, v in vars(args).items()}},
                       tags=["T88", "range_vol", "NN", f"h{H}"])
        except Exception as e:
            print(f"  WandB init failed: {e!r}"); use_wandb = False

    hidden = tuple(int(x) for x in args.hidden.split(","))
    model = MLPRegr(feat_dim, hidden=hidden, dropout=args.dropout,
                    use_layernorm=args.use_layernorm).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: in={feat_dim} hidden={hidden} dropout={args.dropout} params={n_params:,}",
          flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    fm_t = torch.from_numpy(feat_mean).to(device)
    fs_t = torch.from_numpy(feat_std).to(device)

    best_val_mse = float("inf"); best_epoch = -1; best_state = None; bad = 0
    n_train = len(X_tr_raw); epoch_logs = []
    progress("training", seed=seed, n_train=n_train, n_val=len(X_va))

    t_train_start = time.time()
    for epoch in range(args.epochs):
        t_ep = time.time(); model.train()
        perm = torch.randperm(n_train)
        running_loss = 0.0; running_n = 0
        for s in range(0, n_train, args.batch_size):
            idx = perm[s:s+args.batch_size]
            xb_raw = X_tr_raw[idx].to(device, non_blocking=True)
            yb = y_tr_t[idx].to(device, non_blocking=True)
            wb = sw_tr_t[idx].to(device, non_blocking=True)
            if args.aug_mode == "online":
                scales = torch.empty_like(xb_raw).uniform_(args.aug_lo, args.aug_hi)
                xb_raw = xb_raw * scales
            xb = (xb_raw - fm_t) / fs_t
            xb = torch.clamp(xb, -CLIP, CLIP)
            pred = model(xb)
            mse_per = (pred - yb) ** 2
            loss = (mse_per * wb).sum() / wb.sum().clamp_min(1.0)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            bs = xb.size(0)
            running_loss += float(loss.item()) * bs; running_n += bs
        scheduler.step()
        train_loss_avg = running_loss / running_n

        # Val
        va_pred = predict_chunked(model, X_va_std, batch=16384, device=device) / target_scale
        val_mse = float(((va_pred - y_regr_va) ** 2).mean())
        val_corr = float(np.corrcoef(va_pred, y_regr_va)[0, 1]) if va_pred.std() > 0 else 0.0
        ep_time = time.time() - t_ep
        log = {"epoch": epoch, "train_loss": train_loss_avg, "val_mse": val_mse,
               "val_corr": val_corr, "lr": optimizer.param_groups[0]["lr"], "time": ep_time}
        epoch_logs.append(log)
        print(f"  ep {epoch:3d} train_loss={train_loss_avg:.4e} val_mse={val_mse:.4e} "
              f"val_corr={val_corr:.4f} ({ep_time:.1f}s)", flush=True)
        if use_wandb: wandb.log(log)

        if val_mse < best_val_mse - 1e-10:
            best_val_mse = val_mse; best_epoch = epoch
            best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  early stop at {epoch}", flush=True); break

    train_time = time.time() - t_train_start
    print(f"  trained in {train_time:.1f}s best_epoch={best_epoch} best_val_mse={best_val_mse:.4e}",
          flush=True)
    model.load_state_dict(best_state)

    model_path = os.path.join(HERE, f"model_T88nn_seed{seed}.pt")
    torch.save({
        "state_dict": best_state, "feat_mean": feat_mean, "feat_std": feat_std,
        "keep_idx": keep_idx, "feat_names": feat_names,
        "hidden": hidden, "dropout": args.dropout, "use_layernorm": args.use_layernorm,
        "in_dim": feat_dim, "target_scale": float(target_scale), "clip": float(CLIP),
        "best_epoch": best_epoch, "best_val_mse": best_val_mse,
    }, model_path)

    va_pred = predict_chunked(model, X_va_std, batch=16384, device=device) / target_scale
    te_pred = predict_chunked(model, X_te_std, batch=16384, device=device) / target_scale
    va_mse = float(((va_pred - y_regr_va) ** 2).mean())
    va_corr = float(np.corrcoef(va_pred, y_regr_va)[0, 1])
    te_mse = float(((te_pred - y_regr_te) ** 2).mean())
    te_corr = float(np.corrcoef(te_pred, y_regr_te)[0, 1])
    print(f"  VAL mse={va_mse:.7f} corr={va_corr:.4f}", flush=True)
    print(f"  TEST mse={te_mse:.7f} corr={te_corr:.4f}", flush=True)

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
    pred_path = os.path.join(HERE, f"pred_T88nn_seed{seed}.parquet")
    te_df.to_parquet(pred_path, index=False)

    fee_thr = 2.0 * FEE
    pa = np.full(len(te_pred), 1, dtype=np.int8)
    pa[te_pred > fee_thr] = 2; pa[te_pred < -fee_thr] = 0
    side = pa.astype(np.float64) - 1.0
    diff = mp_th_te.astype(np.float64) - mp_t_te.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th_te + 1.0) + (mp_t_te + 1.0))
    pnl = (side * diff - fee_pnl) / (mp_t_te.astype(np.float64) + 1.0)
    cum_pnl = float(pnl.sum()); n_active = int((pa != 1).sum())
    print(f"  TEST EV-gate k=1: cum_pnl={cum_pnl:+.4f} n_active={n_active:,}", flush=True)

    summary = {
        "task": f"T88 NN regression seed={seed}", "seed": seed, "horizon": H,
        "n_features": feat_dim, "best_epoch": best_epoch, "n_params": int(n_params),
        "train_time_sec": float(train_time),
        "val": {"mse": va_mse, "corr": va_corr},
        "test": {"mse": te_mse, "corr": te_corr,
                 "ev_gate_k1_cum_pnl": cum_pnl, "ev_gate_k1_n_active": n_active},
        "params": vars(args),
    }
    with open(os.path.join(HERE, f"summary_T88nn_seed{seed}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if use_wandb:
        wandb.log({"best_epoch": best_epoch, "best_val_mse": best_val_mse,
                   "test_mse": te_mse, "test_corr": te_corr,
                   "test_ev_gate_k1_cum_pnl": cum_pnl})
        wandb.finish()
    progress("done", seed=seed)


if __name__ == "__main__":
    main()
