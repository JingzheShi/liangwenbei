"""T188: Combined T81-style pretrain + T170 SPO+ fine-tune for one NN seed.

Phase 1: T81-style L2 NN pretrain with diverse HPs (varied by seed index)
Phase 2: T170-style SPO+ DFL fine-tune on M7 data (all dates 0-119)

HP diversity formula (seed 1-50):
  lr      = [1e-4, 3e-4, 1e-3][(seed-1) % 3]
  dropout = [0.05, 0.10, 0.15, 0.20][(seed-1) % 4]
  batch   = [2048, 4096, 8192][(seed-1) % 3]
  hidden  = (256, 128, 64) fixed (required for SPO+ phase compatibility)

Self-contained: no T26 dependency (class_balanced_weight inlined).
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

T68_DIR = os.path.join(ROOT, "experiments", "T68_stage5_features")
CACHE_DIR = os.path.join(T68_DIR, "cache")

NUM_CLASS = 3
FEE = 0.0001
HIDDEN = (256, 128, 64)
CLIP = 10.0

LR_LIST = [1e-4, 3e-4, 1e-3]
DROPOUT_LIST = [0.05, 0.10, 0.15, 0.20]
BATCH_LIST = [2048, 4096, 8192]

# Epochs and other Phase-1 fixed config
PHASE1_EPOCHS = 30
PHASE1_PATIENCE = 5
PHASE1_WD = 1e-4
AUG_LO, AUG_HI = 0.80, 1.20

# Phase-2 SPO+ DFL config (same as T170)
PHASE2_EPOCHS = 11
PHASE2_LR = 3e-5
PHASE2_LAMBDA_SPO = 30.0
PHASE2_WD = 1e-4
PHASE2_BATCH = 4096

T59_FAIL_NAMES = [
    "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
    "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
    "qrank_W100_cumspread",
    "kyle_lam_W50", "kyle_lam_W100",
    "roll_eff_spr_ratio_W100",
]
STAGE5_FAIL_NAMES = ["liq_asym_top5_W5"]
DROP_NAMES = T59_FAIL_NAMES + STAGE5_FAIL_NAMES


def class_balanced_weight(y: np.ndarray, num_class: int = 3) -> np.ndarray:
    counts = np.bincount(y.astype(np.int64), minlength=num_class).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    cw = len(y) / (num_class * counts)
    return cw[y.astype(np.int64)].astype(np.float32)


def progress(step, **extra):
    p = os.path.join(HERE, "worker-progress.json")
    payload = {
        "status": "running",
        "step": step,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **extra,
    }
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


def load_npz(split):
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
    def __init__(self, in_dim, hidden=HIDDEN, dropout=0.10, use_layernorm=True):
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
        self._init_xavier()

    def _init_xavier(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def predict_chunked(model, X_std_tensor, batch=16384, device="cuda"):
    model.eval()
    out = np.zeros(len(X_std_tensor), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(X_std_tensor), batch):
            xb = X_std_tensor[s:s + batch].to(device, non_blocking=True)
            out[s:s + batch] = model(xb).detach().cpu().numpy()
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


def extract_npz(ckpt, npz_path):
    """Extract weights from checkpoint dict to .npz for torch-free inference."""
    sd = ckpt["state_dict"]
    hidden = ckpt["hidden"]
    use_layernorm = ckpt.get("use_layernorm", True)
    target_scale = float(ckpt["target_scale"])
    in_dim = int(ckpt["in_dim"])

    out = {
        "in_dim": np.array([in_dim], dtype=np.int64),
        "hidden": np.array(list(hidden), dtype=np.int64),
        "use_layernorm": np.array([1 if use_layernorm else 0], dtype=np.int8),
        "target_scale": np.array([target_scale], dtype=np.float32),
        "feat_mean": np.asarray(ckpt["feat_mean"], dtype=np.float32),
        "feat_std": np.asarray(ckpt["feat_std"], dtype=np.float32),
        "keep_idx": np.asarray(ckpt["keep_idx"], dtype=np.int64),
        "clip": np.array([float(ckpt.get("clip", CLIP))], dtype=np.float32),
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
    print(f"  npz saved: {npz_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, help="Seed index 1-50")
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--cuda", default="0")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--skip-phase1", action="store_true",
                    help="Skip Phase 1 if anchor checkpoint already exists")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    S = args.seed
    H = args.horizon

    # HP diversity from seed index
    lr_p1 = LR_LIST[(S - 1) % len(LR_LIST)]
    dropout = DROPOUT_LIST[(S - 1) % len(DROPOUT_LIST)]
    batch_p1 = BATCH_LIST[(S - 1) % len(BATCH_LIST)]

    print(f"\n{'='*60}", flush=True)
    print(f"T188 NN seed={S} h={H} lr_p1={lr_p1} dropout={dropout} batch_p1={batch_p1}", flush=True)
    print(f"device={device}", flush=True)

    torch.manual_seed(S)
    torch.cuda.manual_seed_all(S)
    np.random.seed(S)

    # === Load feature names and compute keep_idx ===
    feat_names_path = os.path.join(CACHE_DIR, "schemeP_feat_names.txt")
    with open(feat_names_path) as f:
        all_feat_names = [line.strip() for line in f]
    total_dim = len(all_feat_names)
    name_to_idx = {n: i for i, n in enumerate(all_feat_names)}
    drop_idx = set(name_to_idx[n] for n in DROP_NAMES if n in name_to_idx)
    keep_idx = np.array([i for i in range(total_dim) if i not in drop_idx], dtype=np.int64)
    feat_dim = len(keep_idx)
    feat_names = [all_feat_names[i] for i in keep_idx]
    forbidden = {"date", "sym", "time"}
    assert not (forbidden & set(feat_names)), f"LEAK: {forbidden & set(feat_names)}"
    print(f"  feat_dim={feat_dim} (dropped {len(drop_idx)})", flush=True)

    # Paths
    anchor_path = os.path.join(HERE, "T81_pretrained", f"anchor_seed{S}.pt")
    pt_path = os.path.join(HERE, f"nn_h60_seed{S}.pt")
    npz_path = os.path.join(HERE, f"nn_h60_seed{S}.npz")
    os.makedirs(os.path.join(HERE, "T81_pretrained"), exist_ok=True)

    # =========================================================
    # PHASE 1: T81-style L2 NN pretrain (anchor)
    # =========================================================
    if args.skip_phase1 and os.path.isfile(anchor_path):
        print(f"\n--- Phase 1 SKIPPED (anchor exists: {anchor_path}) ---", flush=True)
        ckpt_p1 = torch.load(anchor_path, map_location="cpu", weights_only=False)
        feat_mean = np.asarray(ckpt_p1["feat_mean"], dtype=np.float32)
        feat_std = np.asarray(ckpt_p1["feat_std"], dtype=np.float32)
        target_scale = float(ckpt_p1["target_scale"])
    else:
        print(f"\n--- Phase 1: T81-style pretrain (seed={S}) ---", flush=True)
        progress("phase1_loading", seed=S)

        train_d = load_npz("train")

        # V4 split: train on dates 0-75, val on 76-79
        date_tr = train_d["date"]
        m_va = date_tr >= 76
        m_t = ~m_va

        X_tr = train_d["X"][m_t][:, keep_idx]
        y_cls_tr = train_d[f"y{H}"][m_t].astype(np.int64)
        y_regr_tr = regr_target(train_d["mp_t"][m_t], train_d[f"mp_t{H}"][m_t])

        X_va = train_d["X"][m_va][:, keep_idx]
        y_regr_va = regr_target(train_d["mp_t"][m_va], train_d[f"mp_t{H}"][m_va])
        del train_d

        print(f"  n_train={len(X_tr):,}  n_val={len(X_va):,}", flush=True)

        # Standardization (global, no per-sym)
        feat_mean = np.nanmean(X_tr, axis=0).astype(np.float32)
        feat_std = np.nanstd(X_tr, axis=0).astype(np.float32)
        feat_std = np.maximum(feat_std, 1e-6)

        # NaN imputation
        X_tr_imp = X_tr.copy()
        nan_mask = np.isnan(X_tr_imp)
        if nan_mask.any():
            for d_idx in np.where(nan_mask.any(axis=0))[0]:
                col_nan = np.isnan(X_tr_imp[:, d_idx])
                X_tr_imp[col_nan, d_idx] = feat_mean[d_idx]
        del X_tr

        # Target scaling
        target_scale = 1.0 / float(max(y_regr_tr.std(), 1e-8))
        y_regr_tr_s = (y_regr_tr * target_scale).astype(np.float32)
        print(f"  target_scale={target_scale:.4f}", flush=True)

        # Aug-concat (same as T81)
        rng_aug = np.random.default_rng(S * 7919 + 1)
        scales = rng_aug.uniform(AUG_LO, AUG_HI, size=X_tr_imp.shape).astype(np.float32)
        X_aug = X_tr_imp * scales
        X_full = np.concatenate([X_tr_imp, X_aug], axis=0)
        y_full_s = np.concatenate([y_regr_tr_s, y_regr_tr_s])
        y_cls_full = np.concatenate([y_cls_tr, y_cls_tr])
        del X_aug, X_tr_imp

        sw_full = class_balanced_weight(y_cls_full, num_class=NUM_CLASS)

        # Standardize val
        def standardize(X):
            Xs = (X - feat_mean) / feat_std
            Xs = np.where(np.isnan(Xs), 0.0, Xs)
            return np.clip(Xs, -CLIP, CLIP).astype(np.float32)

        X_va_std = torch.from_numpy(standardize(X_va))
        del X_va

        X_tr_raw = torch.from_numpy(X_full.astype(np.float32))
        y_tr_t = torch.from_numpy(y_full_s)
        sw_tr_t = torch.from_numpy(sw_full)

        # Tensors for batched standardization on GPU
        fm_t = torch.from_numpy(feat_mean).to(device)
        fs_t = torch.from_numpy(feat_std).to(device)

        model = MLPRegr(feat_dim, hidden=HIDDEN, dropout=dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr_p1, weight_decay=PHASE1_WD)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE1_EPOCHS)

        best_val_mse = float("inf")
        best_epoch = -1
        best_state = None
        bad_epochs = 0
        n_train = len(X_tr_raw)

        t0_p1 = time.time()
        for epoch in range(PHASE1_EPOCHS):
            model.train()
            perm = torch.randperm(n_train)
            r_loss = 0.0
            r_n = 0
            for s in range(0, n_train, batch_p1):
                idx = perm[s:s + batch_p1]
                xb = X_tr_raw[idx].to(device, non_blocking=True)
                yb = y_tr_t[idx].to(device, non_blocking=True)
                wb = sw_tr_t[idx].to(device, non_blocking=True)
                xb = torch.clamp((xb - fm_t) / fs_t, -CLIP, CLIP)
                pred = model(xb)
                mse = ((pred - yb) ** 2 * wb).sum() / wb.sum().clamp_min(1.0)
                optimizer.zero_grad()
                mse.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                r_loss += float(mse.item()) * xb.size(0)
                r_n += xb.size(0)
            scheduler.step()

            va_pred_s = predict_chunked(model, X_va_std, device=device)
            va_pred = va_pred_s / target_scale
            val_mse = float(((va_pred - y_regr_va) ** 2).mean())
            va_corr = float(np.corrcoef(va_pred, y_regr_va)[0, 1]) if va_pred.std() > 0 else 0.0
            print(f"  p1 ep {epoch:2d}  train={r_loss/r_n:.4e}  val_mse={val_mse:.4e}  "
                  f"corr={va_corr:.4f}", flush=True)

            if val_mse < best_val_mse - 1e-10:
                best_val_mse = val_mse
                best_epoch = epoch
                best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= PHASE1_PATIENCE:
                    print(f"  early stop at ep {epoch} (patience={PHASE1_PATIENCE})", flush=True)
                    break

        t_p1 = time.time() - t0_p1
        print(f"  Phase 1 done in {t_p1:.1f}s  best_ep={best_epoch}  "
              f"best_val_mse={best_val_mse:.4e}", flush=True)
        assert best_state is not None

        # Save anchor
        ckpt_p1 = {
            "state_dict": best_state,
            "feat_mean": feat_mean,
            "feat_std": feat_std,
            "keep_idx": keep_idx,
            "feat_names": feat_names,
            "hidden": HIDDEN,
            "dropout": dropout,
            "use_layernorm": True,
            "in_dim": feat_dim,
            "target_scale": float(target_scale),
            "clip": float(CLIP),
            "best_epoch": best_epoch,
            "best_val_mse": best_val_mse,
            "phase1_lr": lr_p1,
            "phase1_batch": batch_p1,
            "seed": S,
        }
        torch.save(ckpt_p1, anchor_path)
        print(f"  anchor saved: {anchor_path}", flush=True)

    # =========================================================
    # PHASE 2: SPO+ DFL fine-tune on M7 data (all dates 0-119)
    # =========================================================
    print(f"\n--- Phase 2: SPO+ fine-tune M7 (seed={S}) ---", flush=True)
    progress("phase2_loading_data", seed=S)

    # Reload meta from anchor ckpt
    feat_mean = np.asarray(ckpt_p1["feat_mean"], dtype=np.float32)
    feat_std = np.asarray(ckpt_p1["feat_std"], dtype=np.float32)
    keep_idx_p2 = np.asarray(ckpt_p1["keep_idx"], dtype=np.int64)
    target_scale = float(ckpt_p1["target_scale"])
    hidden = tuple(int(x) for x in ckpt_p1["hidden"])
    dropout = float(ckpt_p1["dropout"])
    use_layernorm = bool(ckpt_p1.get("use_layernorm", True))
    feat_dim_p2 = int(ckpt_p1["in_dim"])
    clip_val = float(ckpt_p1.get("clip", CLIP))

    # Load all 3 splits (M7 = dates 0-119)
    t0_load = time.time()
    train_d = load_npz("train")
    val_d = load_npz("val")
    test_d = load_npz("test")
    print(f"  all splits loaded in {time.time()-t0_load:.1f}s", flush=True)

    n_total = len(train_d["date"]) + len(val_d["date"]) + len(test_d["date"])
    print(f"  total rows={n_total:,}", flush=True)

    # Concatenate
    X_all = np.concatenate([train_d["X"], val_d["X"], test_d["X"]], axis=0)[:, keep_idx_p2]
    mp_t_all = np.concatenate([train_d["mp_t"], val_d["mp_t"], test_d["mp_t"]])
    mp_th_all = np.concatenate([train_d[f"mp_t{H}"], val_d[f"mp_t{H}"], test_d[f"mp_t{H}"]])
    y_cls_all = np.concatenate(
        [train_d[f"y{H}"], val_d[f"y{H}"], test_d[f"y{H}"]]).astype(np.int64)
    del train_d, val_d, test_d

    y_regr_all = regr_target(mp_t_all, mp_th_all)
    fee_all = fee_eff(mp_t_all, mp_th_all)

    # NaN imputation using Phase 1 stats
    nan_mask = np.isnan(X_all)
    if nan_mask.any():
        for d_idx in np.where(nan_mask.any(axis=0))[0]:
            col_nan = np.isnan(X_all[:, d_idx])
            X_all[col_nan, d_idx] = feat_mean[d_idx]

    # Aug-concat
    rng_aug2 = np.random.default_rng(S * 7919 + 137)
    n_orig = len(X_all)
    X_full = np.empty((2 * n_orig, X_all.shape[1]), dtype=np.float32)
    X_full[:n_orig] = X_all
    CHUNK = 200_000
    for i in range(0, n_orig, CHUNK):
        j = min(i + CHUNK, n_orig)
        scales = rng_aug2.uniform(AUG_LO, AUG_HI, size=(j - i, X_all.shape[1])).astype(np.float32)
        X_full[n_orig + i:n_orig + j] = X_all[i:j] * scales
    del X_all

    y_regr_full = np.concatenate([y_regr_all, y_regr_all])
    y_cls_full = np.concatenate([y_cls_all, y_cls_all])
    fee_full = np.concatenate([fee_all, fee_all])

    y_scaled_full = (y_regr_full * target_scale).astype(np.float32)
    fee_scaled_full = (fee_full * target_scale).astype(np.float32)
    sw_full = class_balanced_weight(y_cls_full, num_class=NUM_CLASS)

    X_tr_raw = torch.from_numpy(X_full.astype(np.float32))
    y_scaled_t = torch.from_numpy(y_scaled_full)
    fee_scaled_t = torch.from_numpy(fee_scaled_full)
    sw_t = torch.from_numpy(sw_full)
    del X_full

    # Build model from anchor weights
    model = MLPRegr(feat_dim_p2, hidden=hidden, dropout=dropout,
                    use_layernorm=use_layernorm).to(device)
    model.load_state_dict(ckpt_p1["state_dict"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=PHASE2_LR, weight_decay=PHASE2_WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE2_EPOCHS)

    fm_t = torch.from_numpy(feat_mean).to(device)
    fs_t = torch.from_numpy(np.maximum(feat_std, 1e-6)).to(device)

    n_train2 = len(X_tr_raw)
    progress("phase2_training", seed=S)
    t0_p2 = time.time()
    for epoch in range(PHASE2_EPOCHS):
        t_ep = time.time()
        model.train()
        perm = torch.randperm(n_train2)
        r_l2, r_spo, r_n = 0.0, 0.0, 0

        for s in range(0, n_train2, PHASE2_BATCH):
            idx = perm[s:s + PHASE2_BATCH]
            xb = X_tr_raw[idx].to(device, non_blocking=True)
            yb = y_scaled_t[idx].to(device, non_blocking=True)
            fb = fee_scaled_t[idx].to(device, non_blocking=True)
            wb = sw_t[idx].to(device, non_blocking=True)

            xb = torch.clamp((xb - fm_t) / fs_t, -clip_val, clip_val)
            pred = model(xb)
            l2_per = (pred - yb) ** 2
            spo_per = spo_plus_loss(pred, yb, fb)
            l2_loss = (l2_per * wb).sum() / wb.sum().clamp_min(1.0)
            spo_loss = (spo_per * wb).sum() / wb.sum().clamp_min(1.0)
            loss = l2_loss + PHASE2_LAMBDA_SPO * spo_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            bs = xb.size(0)
            r_l2 += float(l2_loss.item()) * bs
            r_spo += float(spo_loss.item()) * bs
            r_n += bs

        scheduler.step()
        ep_t = time.time() - t_ep
        print(f"  p2 ep {epoch:2d}  l2={r_l2/r_n:.4e}  spo={r_spo/r_n:.4e}  ({ep_t:.1f}s)",
              flush=True)
        progress("phase2_training", seed=S, epoch=epoch, l2=float(r_l2 / r_n))

    t_p2 = time.time() - t0_p2
    print(f"  Phase 2 done in {t_p2:.1f}s", flush=True)

    # Save final checkpoint
    final_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
    final_ckpt = {
        "state_dict": final_state,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "keep_idx": keep_idx_p2,
        "feat_names": feat_names,
        "hidden": hidden,
        "dropout": dropout,
        "use_layernorm": use_layernorm,
        "in_dim": feat_dim_p2,
        "target_scale": float(target_scale),
        "clip": float(clip_val),
        "epochs_fixed": PHASE2_EPOCHS,
        "lambda_spo": PHASE2_LAMBDA_SPO,
        "warm_start_from": f"T188_anchor_seed{S}",
        "M7_full_retrain": True,
        "seed": S,
    }
    torch.save(final_ckpt, pt_path)
    print(f"  pt saved: {pt_path}", flush=True)

    # Extract .npz for Predictor
    extract_npz(final_ckpt, npz_path)

    # Summary
    total_t = time.time()
    summary = {
        "seed": S,
        "horizon": H,
        "phase1_lr": lr_p1,
        "phase1_batch": batch_p1,
        "dropout": dropout,
        "hidden": list(HIDDEN),
        "phase1_best_epoch": ckpt_p1.get("best_epoch", -1),
        "phase1_best_val_mse": ckpt_p1.get("best_val_mse", None),
        "phase2_epochs": PHASE2_EPOCHS,
        "phase2_lambda_spo": PHASE2_LAMBDA_SPO,
        "npz_path": npz_path,
    }
    sum_path = os.path.join(HERE, f"summary_nn_seed{S}.json")
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  summary -> {sum_path}", flush=True)

    progress("done", seed=S, phase1_best_val_mse=ckpt_p1.get("best_val_mse"))
    print(f"\nDONE seed={S}  Phase1_time={t_p1:.0f}s  Phase2_time={t_p2:.0f}s", flush=True)


if __name__ == "__main__":
    main()
