"""T185: Hybrid NN (schemeP MLP + raw LOB Transformer) with val+early stopping.

CRITICAL_CONSTRAINTS compliance:
  - No date features, no sym embedding
  - Stateless predict() (no cross-call state)
  - Sym-agnostic (global normalization, no per-sym paths)

Architecture:
  Branch A: MLP on static schemeP 359-d features
  Branch B: Transformer on raw LOB 100-tick × 31-feature window
  Head: concat → Linear → output

Data:
  schemeP_train.npz (dates 0-79)  + lob_train_v2.npz → train
  schemeP_test.npz  (dates 96-119) + lob_test_v2.npz  → val/holdout

Phase 1: train on 0-79, val on 96-119, early stop patience=5
Phase 2: retrain on 0-79 only, epochs = ceil(best_epoch_avg * 1.1)
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
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SP_CACHE = os.path.join(ROOT, "experiments", "T68_stage5_features", "cache")
LOB_CACHE = os.path.join(ROOT, "experiments", "T95_cnn_rnn_deeplob", "cache")
T170_PKG = os.path.join(ROOT, "experiments", "T170_T87M7_clean_pkg")
FEE = 0.0001
WINDOW = 100


def progress(step, **extra):
    p = os.path.join(HERE, "worker-progress.json")
    payload = {"status": "running", "step": step,
               "timestamp": datetime.now(timezone.utc).isoformat(), **extra}
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


def regr_target(mp_t, mp_th):
    return ((mp_th.astype(np.float64) - mp_t.astype(np.float64))
            / (mp_t.astype(np.float64) + 1.0)).astype(np.float32)


class HybridDataset(Dataset):
    """Joint (schemeP static features, LOB window) dataset.

    lob_arr: (n_sess, T_max, n_raw)
    sess_ids: (n_samples,) → indexes into lob_arr
    ts: (n_samples,)
    X_static: (n_samples, n_static) - already keep_idx applied
    y: (n_samples,)
    sw: (n_samples,) - sample weights
    """
    def __init__(self, lob_arr, sess_ids, ts, X_static, y, sw):
        self.lob_arr = lob_arr
        self.sess_ids = sess_ids
        self.ts = ts
        self.X_static = X_static
        self.y = y
        self.sw = sw

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        s = int(self.sess_ids[i])
        ti = int(self.ts[i])
        # LOB window: (WINDOW, n_raw)
        win = self.lob_arr[s, ti - WINDOW + 1: ti + 1].astype(np.float32)
        return self.X_static[i], win, self.y[i], self.sw[i]


def window_normalize(x_raw: torch.Tensor, clip: float = 10.0) -> torch.Tensor:
    """Per-window per-feature standardize. x_raw: (B, W, F)."""
    m = x_raw.mean(dim=1, keepdim=True)
    s = x_raw.std(dim=1, keepdim=True)
    return ((x_raw - m) / s.clamp_min(1e-8)).clamp(-clip, clip)


class HybridNet(nn.Module):
    def __init__(self, n_static=359, n_raw=31, n_ticks=100,
                 d_static=64, d_seq=64, dropout=0.1):
        super().__init__()
        # Branch A: MLP on static features
        self.mlp = nn.Sequential(
            nn.Linear(n_static, 256), nn.GELU(), nn.LayerNorm(256), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.GELU(), nn.LayerNorm(128), nn.Dropout(dropout),
            nn.Linear(128, d_static),
        )
        # Branch B: Transformer on raw LOB window
        self.proj = nn.Linear(n_raw, d_seq)
        self.pos_enc = nn.Parameter(torch.randn(1, n_ticks, d_seq) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_seq, nhead=4, dim_feedforward=128, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        # Head
        self.head = nn.Sequential(
            nn.Linear(d_static + d_seq, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x_static, x_raw):
        # x_static: (B, n_static), x_raw: (B, W, n_raw) already window-normed
        a = self.mlp(x_static)
        b = self.transformer(self.proj(x_raw) + self.pos_enc).mean(dim=1)
        return self.head(torch.cat([a, b], dim=-1)).squeeze(-1)


def class_balanced_weight(y_cls, n=3):
    counts = np.bincount(y_cls, minlength=n).astype(np.float64)
    w = 1.0 / np.maximum(counts, 1.0)
    return (w / w.mean())[y_cls].astype(np.float32)


def eval_pnl(pred_raw, mp_t, mp_th, fee_thr=None):
    """Quick EV-gate PnL from regression predictions."""
    if fee_thr is None:
        fee_thr = 2.0 * FEE
    side = np.where(pred_raw > fee_thr, 1.0,
           np.where(pred_raw < -fee_thr, -1.0, 0.0))
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    pnl_per = (side * diff - fee) / denom
    total = float(pnl_per.sum())
    n_trades = int((side != 0).sum())
    return total, n_trades


def val_mse(model, loader, device, target_scale, scaler=None):
    model.eval()
    total_loss, total_n = 0.0, 0
    with torch.no_grad():
        for x_st, win, y, sw in loader:
            x_st = x_st.to(device, non_blocking=True)
            win = win.to(device, non_blocking=True)
            y_s = (y * target_scale).to(device, non_blocking=True)
            xb = window_normalize(win)
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    pred = model(x_st, xb)
            else:
                pred = model(x_st, xb)
            loss = ((pred - y_s) ** 2).mean()
            total_loss += float(loss.item()) * len(y)
            total_n += len(y)
    return total_loss / max(total_n, 1)


def predict_all(model, loader, device, target_scale, scaler=None):
    model.eval()
    preds = []
    with torch.no_grad():
        for x_st, win, y, sw in loader:
            x_st = x_st.to(device, non_blocking=True)
            win = win.to(device, non_blocking=True)
            xb = window_normalize(win)
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    pred = model(x_st, xb)
            else:
                pred = model(x_st, xb)
            preds.append(pred.cpu().numpy())
    raw = np.concatenate(preds, axis=0) / target_scale
    return raw


def train_one_seed(seed, ds_train, ds_val, device, args,
                   phase="phase1", max_epochs=None, target_scale=None):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    lob_sample = ds_train.lob_arr[0, :, :]
    n_raw = lob_sample.shape[-1]
    n_static = ds_train.X_static.shape[-1]

    model = HybridNet(n_static=n_static, n_raw=n_raw, n_ticks=WINDOW,
                      d_static=args.d_static, d_seq=args.d_seq,
                      dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [seed={seed} phase={phase}] n_params={n_params:,}", flush=True)

    lr = args.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.wd)
    epochs = max_epochs if max_epochs is not None else args.max_epochs
    # Cosine anneal to min_lr (not 0!)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=args.min_lr)

    use_amp = args.amp and device.type == "cuda"
    grad_scaler = torch.cuda.amp.GradScaler() if use_amp else None

    train_loader = DataLoader(
        ds_train, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=False, persistent_workers=(args.num_workers > 0))

    if ds_val is not None:
        val_loader = DataLoader(
            ds_val, batch_size=args.batch * 2, shuffle=False,
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0))

    best_val = float("inf")
    best_epoch = 0
    no_improve = 0
    best_state = None

    for epoch in range(epochs):
        t_ep = time.time()
        model.train()
        rl, n_seen = 0.0, 0

        for x_st, win, y, sw in train_loader:
            x_st = x_st.to(device, non_blocking=True)
            win = win.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            sw = sw.to(device, non_blocking=True)

            y_s = y * target_scale

            # Augmentation on raw window
            if args.aug_scale > 0:
                scl = torch.empty(win.size(0), 1, win.size(2), device=device).uniform_(
                    1 - args.aug_scale, 1 + args.aug_scale)
                win = win * scl

            xb = window_normalize(win)

            if args.aug_noise > 0:
                xb = xb + torch.randn_like(xb) * args.aug_noise

            optimizer.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast():
                    pred = model(x_st, xb)
                    loss = ((pred - y_s) ** 2 * sw).sum() / sw.sum().clamp_min(1.0)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                pred = model(x_st, xb)
                loss = ((pred - y_s) ** 2 * sw).sum() / sw.sum().clamp_min(1.0)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            rl += float(loss.item()) * len(y)
            n_seen += len(y)

        scheduler.step()
        train_loss = rl / max(n_seen, 1)
        cur_lr = optimizer.param_groups[0]["lr"]
        ep_time = time.time() - t_ep

        val_loss_str = ""
        if ds_val is not None and phase == "phase1":
            vl = val_mse(model, val_loader, device, target_scale, grad_scaler)
            val_loss_str = f"  val_mse={vl:.6e}"
            if vl < best_val:
                best_val = vl
                best_epoch = epoch
                no_improve = 0
                # save state
                import copy
                best_state = copy.deepcopy(model.state_dict())
            else:
                no_improve += 1

        print(f"  ep {epoch:3d}/{epochs}  train_loss={train_loss:.6e}  "
              f"lr={cur_lr:.2e}{val_loss_str}  ({ep_time:.1f}s)", flush=True)

        if phase == "phase1" and no_improve >= args.patience:
            print(f"  Early stop at epoch {epoch} (best={best_epoch})", flush=True)
            break

    if phase == "phase1" and best_state is not None:
        model.load_state_dict(best_state)
        return model, best_epoch, best_val
    return model, epoch, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 7, 42])
    ap.add_argument("--max-epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--d-static", type=int, default=64)
    ap.add_argument("--d-seq", type=int, default=64)
    ap.add_argument("--aug-scale", type=float, default=0.02)
    ap.add_argument("--aug-noise", type=float, default=0.01)
    ap.add_argument("--cuda", default="0")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--skip-phase2", action="store_true")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== T185 HybridNet (schemeP + LOB Transformer) ===", flush=True)
    print(f"  device={device}  seeds={args.seeds}", flush=True)

    # --- WandB ---
    use_wandb = not args.no_wandb
    try:
        import wandb as wm
        if use_wandb:
            wm.init(
                project="liangwenbei",
                entity="cjxh21-Tsinghua University",
                name=f"T185-HybridNet-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config=vars(args),
                tags=["T185", "hybrid", "transformer", "val-earlystop"],
            )
    except Exception as e:
        print(f"  WandB init failed: {e}", flush=True)
        use_wandb = False

    progress("loading_data")
    t0 = time.time()

    # Load schemeP keep_idx from T170 reference model
    ref_nn = np.load(os.path.join(T170_PKG, "nn_h60_seed1.npz"))
    keep_idx = ref_nn["keep_idx"].astype(np.int64)
    print(f"  keep_idx: {len(keep_idx)} features", flush=True)

    # Load schemeP train (dates 0-79)
    sp_tr = np.load(os.path.join(SP_CACHE, "schemeP_train.npz"))
    X_tr = sp_tr["X"][:, keep_idx].astype(np.float32)
    mp_t_tr = sp_tr["mp_t"]
    mp_th_tr = sp_tr["mp_t60"]
    y_cls_tr = sp_tr["y60"].astype(np.int64)
    y_tr = regr_target(mp_t_tr, mp_th_tr)
    sw_tr = class_balanced_weight(y_cls_tr)

    # Load LOB train (dates 0-79) — row-aligned with schemeP_train
    lob_tr = np.load(os.path.join(LOB_CACHE, "lob_train_v2.npz"))
    lob_arr_tr = lob_tr["lob_arr"]  # (n_sess, T_max, 31)
    sess_id_tr = lob_tr["sess_id_for_sample"]
    t_tr = lob_tr["t_for_sample"]

    # Load schemeP test (dates 96-119)
    sp_te = np.load(os.path.join(SP_CACHE, "schemeP_test.npz"))
    X_te = sp_te["X"][:, keep_idx].astype(np.float32)
    mp_t_te = sp_te["mp_t"]
    mp_th_te = sp_te["mp_t60"]
    y_cls_te = sp_te["y60"].astype(np.int64)
    y_te = regr_target(mp_t_te, mp_th_te)
    sw_te = class_balanced_weight(y_cls_te)

    # Load LOB test (dates 96-119)
    lob_te = np.load(os.path.join(LOB_CACHE, "lob_test_v2.npz"))
    lob_arr_te = lob_te["lob_arr"]  # (n_sess, T_max, 31)
    sess_id_te = lob_te["sess_id_for_sample"]
    t_te = lob_te["t_for_sample"]

    n_raw = lob_arr_tr.shape[-1]
    n_static = X_tr.shape[1]
    target_scale = float(1.0 / max(y_tr.std(), 1e-8))
    print(f"  n_train={len(X_tr):,}  n_test={len(X_te):,}", flush=True)
    print(f"  n_static={n_static}  n_raw={n_raw}  target_scale={target_scale:.4f}", flush=True)
    print(f"  Data loaded in {time.time()-t0:.1f}s", flush=True)

    # Fill NaN in static features before normalization
    np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    print(f"  NaN filled in static features", flush=True)

    # Normalize static features (global, fit on train)
    feat_mean = X_tr.mean(axis=0)
    feat_std = X_tr.std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0
    X_tr_n = np.clip((X_tr - feat_mean) / feat_std, -10, 10).astype(np.float32)
    X_te_n = np.clip((X_te - feat_mean) / feat_std, -10, 10).astype(np.float32)
    # Safety: fill any residual NaN after normalization
    np.nan_to_num(X_tr_n, nan=0.0, copy=False)
    np.nan_to_num(X_te_n, nan=0.0, copy=False)

    y_tr_f = y_tr.astype(np.float32)
    y_te_f = y_te.astype(np.float32)

    ds_train = HybridDataset(lob_arr_tr, sess_id_tr, t_tr, X_tr_n, y_tr_f, sw_tr)
    ds_test = HybridDataset(lob_arr_te, sess_id_te, t_te, X_te_n, y_te_f, sw_te)

    # ============================================================
    # PHASE 1: Train on 0-79, val on 96-119, early stop
    # ============================================================
    progress("phase1_training")
    print("\n=== PHASE 1: Train on dates 0-79, val on 96-119 ===", flush=True)
    p1_models = {}
    p1_best_epochs = {}
    p1_val_mses = {}

    for seed in args.seeds:
        print(f"\n--- Phase 1, seed={seed} ---", flush=True)
        t_s = time.time()
        model, best_ep, best_mse = train_one_seed(
            seed, ds_train, ds_test, device, args,
            phase="phase1", target_scale=target_scale)
        elapsed = time.time() - t_s
        p1_models[seed] = model
        p1_best_epochs[seed] = best_ep
        p1_val_mses[seed] = float(best_mse) if best_mse else None
        print(f"  Phase1 seed={seed}: best_epoch={best_ep}, best_val_mse={best_mse}, elapsed={elapsed:.0f}s",
              flush=True)
        if use_wandb:
            import wandb as wm
            wm.log({"phase1/seed": seed, "phase1/best_epoch": best_ep,
                    "phase1/best_val_mse": best_mse})

        # Save phase1 model
        pt_path = os.path.join(HERE, f"model_p1_seed{seed}.pt")
        torch.save(model.state_dict(), pt_path)

        progress("phase1_eval", seed=seed, best_epoch=best_ep)

    # Phase 1 holdout evaluation (ensemble)
    print("\n=== Phase 1 Holdout Evaluation ===", flush=True)
    val_loader = DataLoader(ds_test, batch_size=args.batch * 2, shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=(device.type == "cuda"))

    p1_preds_per_seed = {}
    for seed in args.seeds:
        model = p1_models[seed]
        preds = predict_all(model, val_loader, device, target_scale)
        p1_preds_per_seed[seed] = preds

    p1_ensemble_pred = np.mean(list(p1_preds_per_seed.values()), axis=0)
    p1_pnl, p1_trades = eval_pnl(p1_ensemble_pred, mp_t_te, mp_th_te)
    print(f"  Phase1 ensemble holdout PnL={p1_pnl:.4f}  trades={p1_trades}", flush=True)

    # Per-seed standalone PnL
    p1_standalone_pnls = {}
    for seed in args.seeds:
        pnl, trades = eval_pnl(p1_preds_per_seed[seed], mp_t_te, mp_th_te)
        p1_standalone_pnls[seed] = pnl
        print(f"  Phase1 seed={seed}: standalone PnL={pnl:.4f} trades={trades}", flush=True)

    if use_wandb:
        import wandb as wm
        wm.log({"phase1/ensemble_holdout_pnl": p1_pnl,
                "phase1/ensemble_trades": p1_trades})

    # ============================================================
    # PHASE 2: Retrain on dates 0-79 with ceil(best_avg_epoch * 1.1) epochs
    # ============================================================
    avg_best_ep = np.mean(list(p1_best_epochs.values()))
    p2_epochs = max(int(math.ceil(avg_best_ep * 1.1)), 5)
    print(f"\n=== PHASE 2: Retrain on dates 0-79, epochs={p2_epochs} ===", flush=True)
    print(f"  (avg best_epoch={avg_best_ep:.1f} × 1.1 = {p2_epochs})", flush=True)
    progress("phase2_training", p2_epochs=p2_epochs)

    p2_models = {}
    p2_preds = {}

    if not args.skip_phase2:
        for seed in args.seeds:
            print(f"\n--- Phase 2, seed={seed} ---", flush=True)
            t_s = time.time()
            model, ep, _ = train_one_seed(
                seed, ds_train, None, device, args,
                phase="phase2", max_epochs=p2_epochs, target_scale=target_scale)
            elapsed = time.time() - t_s
            p2_models[seed] = model
            print(f"  Phase2 seed={seed}: done in {elapsed:.0f}s", flush=True)

            # Save phase2 model
            pt_path = os.path.join(HERE, f"model_p2_seed{seed}.pt")
            torch.save(model.state_dict(), pt_path)

            # Get holdout pred (in-sample for Phase2)
            preds = predict_all(model, val_loader, device, target_scale)
            p2_preds[seed] = preds

        p2_ensemble_pred = np.mean(list(p2_preds.values()), axis=0)
        p2_pnl, p2_trades = eval_pnl(p2_ensemble_pred, mp_t_te, mp_th_te)
        print(f"\n  Phase2 ensemble holdout PnL={p2_pnl:.4f}  trades={p2_trades}", flush=True)
    else:
        p2_pnl = None
        p2_trades = None
        p2_ensemble_pred = p1_ensemble_pred

    # ============================================================
    # Save NPZ for submission (Phase2 models, or Phase1 if skip)
    # ============================================================
    print("\n=== Saving NPZ models ===", flush=True)
    model_to_save = p2_models if (not args.skip_phase2 and p2_models) else p1_models

    for seed, model in model_to_save.items():
        npz_path = os.path.join(HERE, f"nn_hybrid_h60_seed{seed}.npz")
        state = model.state_dict()
        save_dict = {
            "arch": np.array(["HybridNet_v1"], dtype=object),
            "n_static": np.array([n_static]),
            "n_raw": np.array([n_raw]),
            "n_ticks": np.array([WINDOW]),
            "d_static": np.array([args.d_static]),
            "d_seq": np.array([args.d_seq]),
            "dropout": np.array([args.dropout]),
            "target_scale": np.array([target_scale]),
            "feat_mean": feat_mean,
            "feat_std": feat_std,
            "keep_idx": keep_idx,
        }
        # Flatten state dict into npz
        for k, v in state.items():
            save_dict[f"sd_{k.replace('.', '_')}"] = v.cpu().numpy()
        np.savez(npz_path, **save_dict)
        print(f"  Saved {npz_path}", flush=True)

    # ============================================================
    # Load v2 LGB ensemble preds (from T170 territory) for comparison
    # ============================================================
    T170_BASELINE_PNL = 149.95  # T170 v2 ensemble holdout
    T87_STANDALONE_PNL = 22.0   # T87 NN standalone holdout ~

    # ============================================================
    # Results summary
    # ============================================================
    results = {
        "task": "T185 Hybrid CNN (schemeP MLP + LOB Transformer) with val+early stop",
        "arch": {
            "n_static": n_static, "n_raw": n_raw, "n_ticks": WINDOW,
            "d_static": args.d_static, "d_seq": args.d_seq,
            "dropout": args.dropout,
        },
        "phase1_best_epochs_per_seed": {str(k): v for k, v in p1_best_epochs.items()},
        "phase1_val_mses_per_seed": {str(k): v for k, v in p1_val_mses.items()},
        "phase1_standalone_pnls_per_seed": {str(k): float(v) for k, v in p1_standalone_pnls.items()},
        "phase1_ensemble_holdout_pnl": float(p1_pnl),
        "phase1_trades": p1_trades,
        "phase2_epochs": p2_epochs,
        "phase2_holdout_pnl": float(p2_pnl) if p2_pnl is not None else None,
        "phase2_trades": p2_trades,
        "T87_baseline_standalone": T87_STANDALONE_PNL,
        "T170_baseline_ensemble_holdout": T170_BASELINE_PNL,
        "hybrid_standalone_holdout": float(p1_pnl),
        "delta_vs_T87_standalone": float(p1_pnl) - T87_STANDALONE_PNL,
        "seeds": args.seeds,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    rp = os.path.join(HERE, "results.json")
    with open(rp, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n=== RESULTS ===", flush=True)
    print(json.dumps(results, indent=2), flush=True)

    if use_wandb:
        import wandb as wm
        wm.log(results)
        wm.finish()

    progress("done", pnl=float(p1_pnl), best_epochs=p1_best_epochs)
    print(f"\nRESULT: task=T185_HybridNet "
          f"metrics={{phase1_ensemble_pnl={p1_pnl:.4f}, "
          f"phase2_pnl={p2_pnl if p2_pnl else 'N/A'}}} "
          f"notes=best_epochs={p1_best_epochs}", flush=True)


if __name__ == "__main__":
    main()
