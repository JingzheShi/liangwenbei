#!/usr/bin/env python3
"""T149: T87 SPO+ NN OOD Regularization Sweep — TSH-FT Evaluator.

Data splits (verified from actual file date ranges):
  schemeP_train.npz : dates  0-79  (1,473,600 rows)
  schemeP_val.npz   : dates 80-95  (  294,720 rows)
  schemeP_test.npz  : dates 96-119 (  442,080 rows)

TSH-FT protocol:
  INNER_DATES   = 0-95   (NN trains on 0-91, val on 92-95)
  HOLDOUT_DATES = 96-119 (OOD; NN never sees this during training)
  TSH-FT = 0.3*inner_pnl + 0.7*holdout_pnl - 0.3*max(0, inner_pnl - holdout_pnl)

Ensemble: pred = (1.0*nn + 1.5*lgb) / 2.5  + conformal wrapper.
Sweep order: Baseline → A (dropout) → B (wd) → E (epochs) → [F,C,D if time]
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
T26_DIR = os.path.join(ROOT, "experiments", "T26_domain_randomization")
sys.path.insert(0, T26_DIR)
from build_aug import class_balanced_weight  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "experiments", "T68_stage5_features", "cache")
T81_DIR = os.path.join(ROOT, "experiments", "T81_nn_regression_pnl")
R_FULL_DIR = os.path.join(ROOT, "experiments", "R_full_retrain")

FEE = 0.0001
NUM_CLASS = 3
THR_UP = 4.21e-4
THR_DN = 1.86e-4
PER_SYM_BETA = {0: 0.10, 1: 0.40, 2: 0.30, 3: 0.00, 4: 0.00}
PER_SYM_SIGMA = {0: 0.0002403901, 1: 0.0004712397, 2: 0.0004522216,
                 3: 0.0004235249, 4: 0.0004279811}
DEFAULT_BETA = 0.16
DEFAULT_SIGMA = 0.0003998317
W_NN, W_LGB = 1.0, 1.5


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Losses and helpers
# ---------------------------------------------------------------------------

def spo_plus_loss(pred_scaled, y_scaled, fee_scaled):
    z_star = torch.where(y_scaled > fee_scaled, torch.ones_like(y_scaled),
                         torch.where(y_scaled < -fee_scaled, -torch.ones_like(y_scaled),
                                     torch.zeros_like(y_scaled)))
    spread = 2.0 * pred_scaled - y_scaled
    relu_max = F.relu(torch.abs(spread) - fee_scaled)
    return relu_max - z_star * spread + fee_scaled * torch.abs(z_star)


def regr_target(mp_t, mp_th):
    return ((mp_th.astype(np.float64) - mp_t.astype(np.float64))
            / (mp_t.astype(np.float64) + 1.0)).astype(np.float32)


def fee_eff_fn(mp_t, mp_th):
    return (FEE * ((mp_th.astype(np.float64) + 1.0) + (mp_t.astype(np.float64) + 1.0))
            / (mp_t.astype(np.float64) + 1.0)).astype(np.float32)


def evaluate_pnl_simple(pred_unscaled, mp_t, mp_th, thr=2 * FEE):
    """Simple EV-gate PnL for model selection (matches T87 usage)."""
    action = np.full(len(pred_unscaled), 1, dtype=np.int8)
    action[pred_unscaled > thr] = 2
    action[pred_unscaled < -thr] = 0
    side = action.astype(np.float64) - 1.0
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th.astype(np.float64) + 1.0) +
                                           (mp_t.astype(np.float64) + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return float(((side * diff - fee_pnl) / denom).sum())


def predict_chunked(model, X_std_t, batch=16384, device="cuda"):
    model.eval()
    out = np.zeros(len(X_std_t), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(X_std_t), batch):
            xb = X_std_t[s:s + batch].to(device, non_blocking=True)
            out[s:s + batch] = model(xb).detach().cpu().numpy()
    return out


def compute_pnl_conformal(pred_ens, mp_t, mp_th, sym_arr):
    """PnL with conformal wrapper (vectorized)."""
    beta_arr = np.vectorize(lambda s: PER_SYM_BETA.get(int(s), DEFAULT_BETA))(sym_arr)
    sigma_arr = np.vectorize(lambda s: PER_SYM_SIGMA.get(int(s), DEFAULT_SIGMA))(sym_arr)
    band = beta_arr * sigma_arr
    eff_up = THR_UP + band
    eff_dn = THR_DN + band

    action = np.ones(len(pred_ens), dtype=np.int8)
    action[pred_ens > eff_up] = 2
    action[pred_ens < -eff_dn] = 0

    side = action.astype(np.float64) - 1.0
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs(
        (mp_th.astype(np.float64) + 1.0) + (mp_t.astype(np.float64) + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return float(((side * diff - fee_pnl) / denom).sum())


def tshft_score(inner_pnl, holdout_pnl):
    return 0.3 * inner_pnl + 0.7 * holdout_pnl - 0.3 * max(0.0, inner_pnl - holdout_pnl)


def standardize(X, feat_mean, feat_std, clip_val):
    Xs = (X.astype(np.float32) - feat_mean) / np.maximum(feat_std, 1e-6)
    Xs = np.where(np.isnan(Xs), 0.0, Xs)
    return np.clip(Xs, -clip_val, clip_val).astype(np.float32)


def progress(step, **extra):
    p = os.path.join(HERE, "worker-progress.json")
    payload = {"status": "running", "step": step,
               "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), **extra}
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_data():
    """Load and split data once. Returns dict of arrays for the full sweep."""
    print("Loading schemeP_train.npz ...", flush=True)
    tr = np.load(os.path.join(CACHE_DIR, "schemeP_train.npz"))
    print("Loading schemeP_val.npz ...", flush=True)
    va = np.load(os.path.join(CACHE_DIR, "schemeP_val.npz"))
    print("Loading schemeP_test.npz ...", flush=True)
    te = np.load(os.path.join(CACHE_DIR, "schemeP_test.npz"))

    # keep_idx: NN and LGB use the same 359 features (verified)
    ckpt42 = torch.load(os.path.join(T81_DIR, "model_T81_seed42.pt"),
                        map_location="cpu", weights_only=False)
    keep_idx = np.asarray(ckpt42["keep_idx"], dtype=np.int64)

    X_tr = tr["X"][:, keep_idx].astype(np.float32)   # (1473600, 359)
    X_va = va["X"][:, keep_idx].astype(np.float32)   # ( 294720, 359)
    X_te = te["X"][:, keep_idx].astype(np.float32)   # ( 442080, 359)

    date_va = np.asarray(va["date"])     # values 80-95
    m_ext = date_va < 92   # dates 80-91 → extend NN train
    m_val = date_va >= 92  # dates 92-95 → NN val

    # NN training (dates 0-91)
    X_nn_tr = np.concatenate([X_tr, X_va[m_ext]], axis=0)
    mp_t_nn_tr = np.concatenate([tr["mp_t"], va["mp_t"][m_ext]])
    mp_t60_nn_tr = np.concatenate([tr["mp_t60"], va["mp_t60"][m_ext]])
    y60_nn_tr = np.concatenate([tr["y60"], va["y60"][m_ext]])

    # NN val (dates 92-95)
    X_nn_va = X_va[m_val]
    mp_t_nn_va = va["mp_t"][m_val]
    mp_t60_nn_va = va["mp_t60"][m_val]

    # Phase 5: full training (dates 0-119 = nn_tr + nn_va + holdout)
    X_full_tr = np.concatenate([X_nn_tr, X_nn_va, X_te], axis=0)
    mp_t_full_tr = np.concatenate([mp_t_nn_tr, mp_t_nn_va, te["mp_t"]])
    mp_t60_full_tr = np.concatenate([mp_t60_nn_tr, mp_t60_nn_va, te["mp_t60"]])
    y60_full_tr = np.concatenate([y60_nn_tr, va["y60"][m_val], te["y60"]])

    # INNER eval (dates 0-95)
    X_inner = np.concatenate([X_tr, X_va], axis=0)
    mp_t_inner = np.concatenate([tr["mp_t"], va["mp_t"]])
    mp_t60_inner = np.concatenate([tr["mp_t60"], va["mp_t60"]])
    sym_inner = np.concatenate([tr["sym"], va["sym"]])

    # HOLDOUT eval (dates 96-119)
    X_holdout = X_te
    mp_t_holdout = te["mp_t"]
    mp_t60_holdout = te["mp_t60"]
    sym_holdout = te["sym"]

    print(f"  X_nn_tr={X_nn_tr.shape}  X_nn_va={X_nn_va.shape}", flush=True)
    print(f"  X_inner={X_inner.shape}  X_holdout={X_holdout.shape}", flush=True)

    return {
        "keep_idx": keep_idx,
        # NN train/val
        "X_nn_tr": X_nn_tr, "mp_t_nn_tr": mp_t_nn_tr,
        "mp_t60_nn_tr": mp_t60_nn_tr, "y60_nn_tr": y60_nn_tr,
        "X_nn_va": X_nn_va, "mp_t_nn_va": mp_t_nn_va, "mp_t60_nn_va": mp_t60_nn_va,
        # Full training (Phase 5)
        "X_full_tr": X_full_tr, "mp_t_full_tr": mp_t_full_tr,
        "mp_t60_full_tr": mp_t60_full_tr, "y60_full_tr": y60_full_tr,
        # INNER eval
        "X_inner": X_inner, "mp_t_inner": mp_t_inner,
        "mp_t60_inner": mp_t60_inner, "sym_inner": sym_inner,
        # HOLDOUT eval
        "X_holdout": X_holdout, "mp_t_holdout": mp_t_holdout,
        "mp_t60_holdout": mp_t60_holdout, "sym_holdout": sym_holdout,
    }


def load_t81(seed):
    pt = os.path.join(T81_DIR, f"model_T81_seed{seed}.pt")
    ckpt = torch.load(pt, map_location="cpu", weights_only=False)
    return {
        "feat_mean": np.asarray(ckpt["feat_mean"], dtype=np.float32),
        "feat_std": np.asarray(ckpt["feat_std"], dtype=np.float32),
        "keep_idx": np.asarray(ckpt["keep_idx"], dtype=np.int64),
        "target_scale": float(ckpt["target_scale"]),
        "hidden": tuple(int(x) for x in ckpt["hidden"]),
        "dropout": float(ckpt["dropout"]),
        "use_layernorm": bool(ckpt.get("use_layernorm", True)),
        "in_dim": int(ckpt["in_dim"]),
        "clip": float(ckpt.get("clip", 10.0)),
        "state_dict": ckpt["state_dict"],
    }


def precompute_lgb_preds(data):
    """Compute 5-model averaged LGB predictions on inner + holdout."""
    print("Precomputing LGB predictions ...", flush=True)
    seeds_lgb = [1, 7, 13, 42, 100]
    boosters = [lgb.Booster(model_file=os.path.join(R_FULL_DIR, f"model_h60_seed{s}.txt"))
                for s in seeds_lgb]
    inner_preds = np.mean([b.predict(data["X_inner"]) for b in boosters], axis=0).astype(np.float32)
    holdout_preds = np.mean([b.predict(data["X_holdout"]) for b in boosters], axis=0).astype(np.float32)
    del boosters
    print(f"  LGB inner mean={inner_preds.mean():.6f}  holdout mean={holdout_preds.mean():.6f}", flush=True)
    return inner_preds, holdout_preds


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _build_aug_train(X_tr_raw, y_regr_tr, y_cls_tr, fee_eff_tr, seed):
    """Double training data via concat domain-randomization augmentation."""
    aug_rng = np.random.default_rng(seed * 7919 + 137)
    n = len(X_tr_raw)
    X_aug = np.empty((2 * n, X_tr_raw.shape[1]), dtype=np.float32)
    X_aug[:n] = X_tr_raw
    for i in range(0, n, 100_000):
        j = min(i + 100_000, n)
        scales = aug_rng.uniform(0.80, 1.20, size=(j - i, X_tr_raw.shape[1])).astype(np.float32)
        X_aug[n + i:n + j] = X_tr_raw[i:j] * scales
    return (X_aug,
            np.concatenate([y_regr_tr, y_regr_tr]),
            np.concatenate([y_cls_tr, y_cls_tr]),
            np.concatenate([fee_eff_tr, fee_eff_tr]))


def train_nn(cfg: dict, seed: int, data: dict, t81: dict, device: torch.device,
             use_full_data: bool = False) -> Tuple[Optional[dict], float]:
    """Train NN with given config. Returns (best_state_dict, best_val_pnl)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    feat_mean = t81["feat_mean"]
    feat_std = t81["feat_std"]
    target_scale = t81["target_scale"]
    clip_val = t81["clip"]
    use_layernorm = t81["use_layernorm"]
    in_dim = t81["in_dim"]

    hidden = cfg.get("hidden", t81["hidden"])
    dropout = cfg.get("dropout", 0.10)
    lr = cfg.get("lr", 3e-5)
    wd = cfg.get("wd", 1e-4)
    epochs = cfg.get("epochs", 10)
    patience = cfg.get("patience", 4)
    lambda_spo = cfg.get("lambda_spo", 30.0)
    sigma_smooth = cfg.get("sigma", 0.0)
    batch_size = cfg.get("batch_size", 4096)

    # Select training data
    if use_full_data:
        X_tr_raw = data["X_full_tr"]
        mp_t_tr = data["mp_t_full_tr"]
        mp_t60_tr = data["mp_t60_full_tr"]
        y60_tr = data["y60_full_tr"]
    else:
        X_tr_raw = data["X_nn_tr"]
        mp_t_tr = data["mp_t_nn_tr"]
        mp_t60_tr = data["mp_t60_nn_tr"]
        y60_tr = data["y60_nn_tr"]

    y_regr_tr = regr_target(mp_t_tr, mp_t60_tr)
    fee_eff_tr = fee_eff_fn(mp_t_tr, mp_t60_tr)
    y_cls_tr = y60_tr.astype(np.int64)

    if sigma_smooth > 0.0:
        y_std = float(y_regr_tr.std())
        noise = np.random.default_rng(seed + 999).normal(
            0, sigma_smooth * y_std, size=len(y_regr_tr)).astype(np.float32)
        y_regr_tr = y_regr_tr + noise

    X_aug, y_regr_full, y_cls_full, fee_full = _build_aug_train(
        X_tr_raw, y_regr_tr, y_cls_tr, fee_eff_tr, seed)

    y_scaled_tr = (y_regr_full * target_scale).astype(np.float32)
    fee_scaled_tr = (fee_full * target_scale).astype(np.float32)
    sw_tr = class_balanced_weight(y_cls_full, num_class=NUM_CLASS).astype(np.float32)

    X_tr_t = torch.from_numpy(X_aug)
    y_scaled_t = torch.from_numpy(y_scaled_tr)
    fee_scaled_t = torch.from_numpy(fee_scaled_tr)
    sw_t = torch.from_numpy(sw_tr)
    fm_t = torch.from_numpy(feat_mean).to(device)
    fs_t = torch.from_numpy(np.maximum(feat_std, 1e-6)).to(device)

    X_va_std = torch.from_numpy(standardize(data["X_nn_va"], feat_mean, feat_std, clip_val))
    mp_t_va = data["mp_t_nn_va"]
    mp_t60_va = data["mp_t60_nn_va"]

    # Build model
    model = MLPRegr(in_dim, hidden=hidden, dropout=dropout, use_layernorm=use_layernorm).to(device)
    if hidden == t81["hidden"]:
        model.load_state_dict(t81["state_dict"])  # warm-start from T81

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Init eval (T81 baseline on val)
    va_pred_init = predict_chunked(model, X_va_std, device=device) / target_scale
    val_pnl_init = evaluate_pnl_simple(va_pred_init, mp_t_va, mp_t60_va)

    best_val_pnl = val_pnl_init
    best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
    bad_epochs = 0
    n_train = len(X_tr_t)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        rl2 = rspo = rn = 0.0

        for s in range(0, n_train, batch_size):
            idx = perm[s:s + batch_size]
            xb_raw = X_tr_t[idx].to(device, non_blocking=True)
            yb = y_scaled_t[idx].to(device, non_blocking=True)
            fb = fee_scaled_t[idx].to(device, non_blocking=True)
            wb = sw_t[idx].to(device, non_blocking=True)

            xb = torch.clamp((xb_raw - fm_t) / fs_t, -clip_val, clip_val)
            pred = model(xb)
            l2_per = (pred - yb) ** 2
            spo_per = spo_plus_loss(pred, yb, fb)
            l2_loss = (l2_per * wb).sum() / wb.sum().clamp_min(1.0)
            spo_loss = (spo_per * wb).sum() / wb.sum().clamp_min(1.0)

            optimizer.zero_grad()
            (l2_loss + lambda_spo * spo_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            bs = xb.size(0)
            rl2 += float(l2_loss.item()) * bs
            rspo += float(spo_loss.item()) * bs
            rn += bs

        scheduler.step()
        va_pred = predict_chunked(model, X_va_std, device=device) / target_scale
        val_pnl = evaluate_pnl_simple(va_pred, mp_t_va, mp_t60_va)

        if val_pnl > best_val_pnl:
            best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
            best_val_pnl = val_pnl
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    del X_aug, X_tr_t, y_scaled_t, fee_scaled_t, sw_t, X_va_std, model
    gc.collect()
    return best_state, best_val_pnl


def eval_tshft(best_state: dict, t81: dict, hidden: tuple, data: dict,
               lgb_inner: np.ndarray, lgb_holdout: np.ndarray,
               device: torch.device) -> Tuple[float, float, float]:
    """Returns (tshft_score, inner_pnl, holdout_pnl)."""
    feat_mean, feat_std = t81["feat_mean"], t81["feat_std"]
    target_scale = t81["target_scale"]
    clip_val = t81["clip"]
    use_layernorm = t81["use_layernorm"]
    in_dim = t81["in_dim"]

    model = MLPRegr(in_dim, hidden=hidden, dropout=0.0, use_layernorm=use_layernorm).to(device)
    model.load_state_dict(best_state)
    model.eval()

    X_inner_std = torch.from_numpy(standardize(data["X_inner"], feat_mean, feat_std, clip_val))
    X_holdout_std = torch.from_numpy(standardize(data["X_holdout"], feat_mean, feat_std, clip_val))

    nn_inner = predict_chunked(model, X_inner_std, device=device) / target_scale
    nn_holdout = predict_chunked(model, X_holdout_std, device=device) / target_scale

    del X_inner_std, X_holdout_std, model
    gc.collect()

    ens_inner = (W_NN * nn_inner + W_LGB * lgb_inner) / (W_NN + W_LGB)
    ens_holdout = (W_NN * nn_holdout + W_LGB * lgb_holdout) / (W_NN + W_LGB)

    inner_pnl = compute_pnl_conformal(ens_inner, data["mp_t_inner"],
                                       data["mp_t60_inner"], data["sym_inner"])
    holdout_pnl = compute_pnl_conformal(ens_holdout, data["mp_t_holdout"],
                                         data["mp_t60_holdout"], data["sym_holdout"])
    return tshft_score(inner_pnl, holdout_pnl), inner_pnl, holdout_pnl


# ---------------------------------------------------------------------------
# NPZ export for package
# ---------------------------------------------------------------------------

def export_nn_npz(state_dict: dict, t81: dict, hidden: tuple, npz_path: str):
    """Export to NPZ format expected by _MLPNumpy in Predictor.py."""
    use_layernorm = t81["use_layernorm"]
    in_dim = t81["in_dim"]
    sd = state_dict

    arrays: dict = {
        "in_dim": np.array([in_dim], dtype=np.int64),
        "hidden": np.array(list(hidden), dtype=np.int64),
        "use_layernorm": np.array([int(use_layernorm)], dtype=np.int8),
        "target_scale": np.array([t81["target_scale"]], dtype=np.float32),
        "feat_mean": t81["feat_mean"].astype(np.float32),
        "feat_std": t81["feat_std"].astype(np.float32),
        "keep_idx": t81["keep_idx"].astype(np.int64),
        "clip": np.array([t81["clip"]], dtype=np.float32),
    }

    # net layout: [Linear, (LayerNorm), GELU, Dropout] * len(hidden), Linear_final
    net_offset = 0
    for i in range(len(hidden)):
        arrays[f"L{i}_W"] = sd[f"net.{net_offset}.weight"].numpy().astype(np.float32)
        arrays[f"L{i}_b"] = sd[f"net.{net_offset}.bias"].numpy().astype(np.float32)
        net_offset += 1
        if use_layernorm:
            arrays[f"LN{i}_W"] = sd[f"net.{net_offset}.weight"].numpy().astype(np.float32)
            arrays[f"LN{i}_b"] = sd[f"net.{net_offset}.bias"].numpy().astype(np.float32)
            net_offset += 1
        net_offset += 2  # GELU + Dropout
    arrays["LF_W"] = sd[f"net.{net_offset}.weight"].numpy().astype(np.float32)
    arrays["LF_b"] = sd[f"net.{net_offset}.bias"].numpy().astype(np.float32)
    np.savez(npz_path, **arrays)


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def run_config(name: str, cfg: dict, seeds: List[int], data: dict,
               lgb_inner: np.ndarray, lgb_holdout: np.ndarray,
               device: torch.device, wandb_proj: str,
               use_full_data: bool = False) -> dict:
    """Run config on given seeds, return aggregated results."""
    scores, inners, holdouts, val_pnls = [], [], [], []

    for seed in seeds:
        t81 = load_t81(seed)
        run = None
        try:
            import wandb
            run = wandb.init(
                project=wandb_proj, entity="cjxh21-Tsinghua University",
                name=f"T149-{name}-seed{seed}",
                config={"name": name, "seed": seed, **cfg},
                tags=["T149"], reinit=True,
            )
        except Exception:
            pass

        t0 = time.time()
        best_state, best_val_pnl = train_nn(cfg, seed, data, t81, device, use_full_data)
        hidden = cfg.get("hidden", t81["hidden"])
        score, inner_pnl, holdout_pnl = eval_tshft(best_state, t81, hidden, data,
                                                     lgb_inner, lgb_holdout, device)
        elapsed = time.time() - t0
        scores.append(score)
        inners.append(inner_pnl)
        holdouts.append(holdout_pnl)
        val_pnls.append(best_val_pnl)

        print(f"  [{name} seed={seed}] TSH-FT={score:+.4f} "
              f"inner={inner_pnl:+.4f} holdout={holdout_pnl:+.4f} "
              f"val_pnl={best_val_pnl:+.4f}  ({elapsed:.0f}s)", flush=True)

        if run:
            try:
                import wandb
                wandb.log({"tshft_score": score, "inner_pnl": inner_pnl,
                           "holdout_pnl": holdout_pnl, "val_pnl": best_val_pnl})
                run.finish()
            except Exception:
                pass

    return {
        "name": name, "config": cfg, "seeds": seeds,
        "tshft_mean": float(np.mean(scores)),
        "tshft_scores": [float(s) for s in scores],
        "inner_pnl_mean": float(np.mean(inners)),
        "holdout_pnl_mean": float(np.mean(holdouts)),
        "val_pnl_mean": float(np.mean(val_pnls)),
    }


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------

def save_results(all_results, extra=None):
    out = {"task": "T149 NN T87 OOD sweep on TSH-FT",
           "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "all_results": all_results}
    if extra:
        out.update(extra)
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(out, f, indent=2)


def write_report(all_results, baseline_tshft, best_name, best_tshft, delta):
    lines = [
        "# T149 NN OOD Sweep — TSH-FT Report\n\n",
        f"**Baseline TSH-FT**: {baseline_tshft:+.4f}\n\n",
        f"**Best config**: `{best_name}` TSH-FT={best_tshft:+.4f} Δ={delta:+.4f}\n\n",
        "## All Configs\n\n",
        "| Config | Seeds | TSH-FT mean | Inner PnL | Holdout PnL | Val PnL |\n",
        "|--------|-------|-------------|-----------|-------------|--------|\n",
    ]
    for n, r in sorted(all_results.items(), key=lambda x: -x[1].get("tshft_mean", -999)):
        s = r.get("tshft_mean", float("nan"))
        ip = r.get("inner_pnl_mean", float("nan"))
        hp = r.get("holdout_pnl_mean", float("nan"))
        vp = r.get("val_pnl_mean", float("nan"))
        ns = len(r.get("seeds", []))
        marker = " ★" if n == best_name else ""
        lines.append(f"| {n}{marker} | {ns} | {s:+.4f} | {ip:+.4f} | {hp:+.4f} | {vp:+.4f} |\n")
    with open(os.path.join(HERE, "REPORT.md"), "w") as f:
        f.writelines(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", default="0")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== T149 OOD Sweep | device={device} ===", flush=True)

    WANDB_PROJ = "liangwenbei-T149-nn-ood-sweep" if not args.no_wandb else ""
    if WANDB_PROJ:
        try:
            import wandb
            wandb.login(key="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",
                        relogin=True)
        except Exception as e:
            print(f"  WandB login failed ({e}); disabling", flush=True)
            WANDB_PROJ = ""

    progress("loading_data")
    t0 = time.time()
    data = load_all_data()
    lgb_inner, lgb_holdout = precompute_lgb_preds(data)
    print(f"Data + LGB ready in {time.time() - t0:.1f}s", flush=True)

    BASE = {"dropout": 0.10, "wd": 1e-4, "epochs": 10, "lr": 3e-5,
            "sigma": 0.0, "lambda_spo": 30.0, "batch_size": 4096, "patience": 4}

    all_results: Dict[str, dict] = {}
    start_time = time.time()
    TIME_BUDGET = 90 * 60  # 90 minutes

    def remaining():
        return TIME_BUDGET - (time.time() - start_time)

    def do_config(name, cfg, seeds):
        progress(f"training_{name}")
        r = run_config(name, cfg, seeds, data, lgb_inner, lgb_holdout, device, WANDB_PROJ)
        all_results[name] = r
        save_results(all_results)
        print(f"  → {name}: TSH-FT_mean={r['tshft_mean']:+.4f}", flush=True)
        return r

    # -----------------------------------------------------------------------
    # Phase 1: single-seed sweep (seed=42)
    # -----------------------------------------------------------------------
    print("\n=== Phase 1: Single-seed sweep (seed=42) ===", flush=True)

    # Baseline
    base_res = do_config("baseline", dict(BASE), [42])
    baseline_tshft = base_res["tshft_mean"]

    # Sweep A: dropout
    for do in [0.05, 0.20, 0.30, 0.40]:
        cfg = dict(BASE); cfg["dropout"] = do
        do_config(f"A_do{do}", cfg, [42])

    # Sweep B: weight decay
    for wd in [1e-5, 5e-4, 1e-3, 5e-3]:
        cfg = dict(BASE); cfg["wd"] = wd
        do_config(f"B_wd{wd:.0e}", cfg, [42])

    # Sweep E: epochs
    for ep in [3, 5, 8, 12, 20]:
        cfg = dict(BASE); cfg["epochs"] = ep
        do_config(f"E_ep{ep}", cfg, [42])

    print(f"\nPhase 1 done. Baseline TSH-FT={baseline_tshft:+.4f} | {remaining()/60:.1f}min remain",
          flush=True)

    # -----------------------------------------------------------------------
    # Phase 2: top-2 from A, B, E with seeds {1, 7, 42}
    # -----------------------------------------------------------------------
    print(f"\n=== Phase 2: top-2 verification (seeds 1,7,42) ===", flush=True)
    progress("phase2_start")

    def get_top2_configs(prefix):
        items = [(n, r) for n, r in all_results.items()
                 if n.startswith(prefix) and n != "baseline"]
        items.sort(key=lambda x: x[1]["tshft_mean"], reverse=True)
        return items[:2]

    phase2_res = {}
    for prefix in ["A_do", "B_wd", "E_ep"]:
        for name, r in get_top2_configs(prefix):
            name2 = name + "_3s"
            print(f"\n  Verifying {name} (1-seed TSH-FT={r['tshft_mean']:+.4f}) ...", flush=True)
            r2 = do_config(name2, r["config"], [1, 7, 42])
            phase2_res[name2] = r2

    # -----------------------------------------------------------------------
    # Phase 3: best combo
    # -----------------------------------------------------------------------
    print(f"\n=== Phase 3: best combo | {remaining()/60:.1f}min remain ===", flush=True)
    progress("phase3_start")

    def best_from(prefix):
        """Best verified config for sweep prefix (3-seed if exists, else 1-seed)."""
        verified = {n: r for n, r in phase2_res.items() if f"_{prefix[0]}_" in n or n.startswith(prefix)}
        # Try 3-seed verified first
        p2 = [(n, r) for n, r in phase2_res.items() if prefix in n]
        if p2:
            p2.sort(key=lambda x: x[1]["tshft_mean"], reverse=True)
            return p2[0][1]["config"]
        # Fall back to 1-seed
        p1 = [(n, r) for n, r in all_results.items()
              if n.startswith(prefix) and n != "baseline" and "_3s" not in n]
        if p1:
            p1.sort(key=lambda x: x[1]["tshft_mean"], reverse=True)
            return p1[0][1]["config"]
        return {}

    best_a_cfg = best_from("A_do")
    best_b_cfg = best_from("B_wd")
    best_e_cfg = best_from("E_ep")

    combo_cfg = dict(BASE)
    if best_a_cfg:
        combo_cfg["dropout"] = best_a_cfg.get("dropout", BASE["dropout"])
    if best_b_cfg:
        combo_cfg["wd"] = best_b_cfg.get("wd", BASE["wd"])
    if best_e_cfg:
        combo_cfg["epochs"] = best_e_cfg.get("epochs", BASE["epochs"])

    print(f"  Combo: {combo_cfg}", flush=True)
    combo_res = do_config("combo_ABE", combo_cfg, [1, 7, 42])

    # -----------------------------------------------------------------------
    # Phase 4: F, C, D (if >30 min remain)
    # -----------------------------------------------------------------------
    best_multi_seed = {n: r for n, r in all_results.items()
                       if len(r.get("seeds", [])) >= 3}
    best_so_far = combo_cfg
    best_tshft_so_far = combo_res["tshft_mean"]

    if remaining() > 30 * 60:
        print(f"\n=== Phase 4: F, C, D | {remaining()/60:.1f}min remain ===", flush=True)
        progress("phase4_start")

        # Sweep F: lr (1-seed ranking)
        f_scores = {}
        for lr_val in [1e-5, 1e-4, 3e-4]:
            cfg = dict(best_so_far); cfg["lr"] = lr_val
            r = do_config(f"F_lr{lr_val:.0e}", cfg, [42])
            f_scores[lr_val] = r["tshft_mean"]

        # Sweep C: sigma (1-seed ranking)
        c_scores = {}
        for sig in [0.05, 0.10, 0.20]:
            cfg = dict(best_so_far); cfg["sigma"] = sig
            r = do_config(f"C_sig{sig}", cfg, [42])
            c_scores[sig] = r["tshft_mean"]

        # Update combo if F or C helps
        best_f_lr = max(f_scores, key=f_scores.get)
        best_c_sig = max(c_scores, key=c_scores.get)

        if f_scores[best_f_lr] > best_tshft_so_far:
            best_so_far["lr"] = best_f_lr
        if c_scores[best_c_sig] > best_tshft_so_far:
            best_so_far["sigma"] = best_c_sig

        # Verify updated combo if changed
        if best_so_far != combo_cfg and remaining() > 15 * 60:
            r_fc = do_config("combo_ABEF_C", best_so_far, [1, 7, 42])
            if r_fc["tshft_mean"] > best_tshft_so_far:
                best_tshft_so_far = r_fc["tshft_mean"]

        # Architecture sweep if time permits
        if remaining() > 20 * 60:
            for arch in [(128, 64), (256, 128), (128, 64, 32), (64, 32)]:
                if remaining() < 10 * 60:
                    break
                cfg = dict(best_so_far); cfg["hidden"] = arch
                arch_str = "_".join(str(x) for x in arch)
                do_config(f"D_arch_{arch_str}", cfg, [42])

    # -----------------------------------------------------------------------
    # Determine best overall config
    # -----------------------------------------------------------------------
    all_multi = {n: r for n, r in all_results.items() if len(r.get("seeds", [])) >= 3}
    if all_multi:
        best_name = max(all_multi, key=lambda n: all_multi[n]["tshft_mean"])
        best_tshft = all_multi[best_name]["tshft_mean"]
        best_cfg_final = all_multi[best_name]["config"]
    else:
        best_name = "baseline"
        best_tshft = baseline_tshft
        best_cfg_final = BASE

    delta = best_tshft - baseline_tshft
    print(f"\n=== Best: {best_name} TSH-FT={best_tshft:+.4f} Δ={delta:+.4f} ===", flush=True)

    # -----------------------------------------------------------------------
    # Phase 5: 5-seed final on dates 0-119 (if delta > +0.5)
    # -----------------------------------------------------------------------
    pkg_dir = None
    if delta > 0.5:
        print(f"\n=== Phase 5: 5-seed final (delta={delta:+.4f} > 0.5) ===", flush=True)
        progress("phase5_5seed")

        final_states = {}
        final_t81s = {}
        for seed in [1, 7, 13, 42, 100]:
            t81 = load_t81(seed)
            best_state, _ = train_nn(best_cfg_final, seed, data, t81, device,
                                     use_full_data=True)
            final_states[seed] = best_state
            final_t81s[seed] = t81
            print(f"  Trained final seed={seed}", flush=True)

        # Copy package and replace NN NPZ files
        pkg_src = os.path.join(R_FULL_DIR, "pkg_iter019_v2")
        pkg_dst = os.path.join(HERE, "pkg_iter019_v6_nn_tuned")
        import shutil
        if os.path.exists(pkg_dst):
            shutil.rmtree(pkg_dst)
        shutil.copytree(pkg_src, pkg_dst)

        for seed in [1, 7, 13, 42, 100]:
            t81 = final_t81s[seed]
            hidden = best_cfg_final.get("hidden", t81["hidden"])
            npz_path = os.path.join(pkg_dst, f"nn_h60_seed{seed}.npz")
            export_nn_npz(final_states[seed], t81, hidden, npz_path)
            print(f"  Exported {npz_path}", flush=True)

        # Build zip
        import zipfile
        zip_name = "submission_050818_iter019_v6_nn_tuned.zip"
        zip_path = os.path.join(HERE, zip_name)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root_d, _, files in os.walk(pkg_dst):
                for file in files:
                    fp = os.path.join(root_d, file)
                    arcname = os.path.relpath(fp, pkg_dst)
                    zf.write(fp, arcname)

        out_dir = "/tmp/metabot-outputs/worker-cdd018d5"
        os.makedirs(out_dir, exist_ok=True)
        shutil.copy2(zip_path, out_dir)
        print(f"  Built and copied {zip_name}", flush=True)
        pkg_dir = pkg_dst
    else:
        print(f"  Δ={delta:+.4f} ≤ 0.5 → skipping Phase 5", flush=True)

    # -----------------------------------------------------------------------
    # Write final results and report
    # -----------------------------------------------------------------------
    progress("writing_results")
    final_summary = {
        "baseline_t87_tshft": baseline_tshft,
        "best_config_name": best_name,
        "best_combo": best_cfg_final,
        "best_tshft_score": best_tshft,
        "vs_baseline_delta": delta,
        "packaged": pkg_dir is not None,
        "verdict": (f"Best config '{best_name}' achieves TSH-FT={best_tshft:+.4f} "
                    f"(delta={delta:+.4f} vs baseline)"),
    }
    save_results(all_results, final_summary)
    write_report(all_results, baseline_tshft, best_name, best_tshft, delta)

    total_min = (time.time() - start_time) / 60
    print(f"\n=== T149 Sweep Done in {total_min:.1f}min ===", flush=True)
    print(f"RESULT: task=T149_NN_OOD_sweep "
          f"metrics={{baseline_tshft={baseline_tshft:.4f}, best_tshft={best_tshft:.4f}, "
          f"delta={delta:.4f}, best_config={best_name}}} "
          f"notes=[{'PACKAGED' if pkg_dir else 'no_package'}]", flush=True)

    progress("done", best_tshft=best_tshft, delta=delta, best_name=best_name,
             total_min=total_min)


if __name__ == "__main__":
    main()
