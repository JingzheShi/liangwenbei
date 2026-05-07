"""T60: NN trained with direct expected-PnL loss (RL-style, but fully
differentiable — no policy-gradient).

Inputs: schemeN cache from T53_r34_stage3 (drop-fail-extras + n_drop_tail=3
=> 340-d).

Loss (per sample i):
    prob   = softmax(logits)            shape (B, 3)
    e_side = prob[:, 2] - prob[:, 0]    in [-1, +1]
    e_abs  = prob[:, 2] + prob[:, 0]    in [ 0, +1]
    diff   = mp_th - mp_t
    fee    = fee_rate * e_abs * |(mp_th+1) + (mp_t+1)|
    pnl_i  = (e_side * diff - fee) / (mp_t + 1)
    L      = - mean(pnl_i)
             - ent_beta * mean(entropy(prob))            (anti-collapse)

LOSO 5-fold per held sym × 3 seeds.

Notes:
  - sym is NEVER fed to the model (sym-agnostic, per CRITICAL_CONSTRAINTS §1)
  - date is NEVER used (per §1)
  - per-feature global mean/std on training set (NOT per-sym)
  - aug_a applied AFTER z-score standardization on training batches
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from src.eval.pnl import _per_horizon_metrics, sanitize_for_json  # noqa: E402

T53_DIR = os.path.join(ROOT, "experiments", "T53_r34_stage3")
CACHE_DIR = os.path.join(T53_DIR, "cache")
NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)


def progress(step: str, **extra) -> None:
    p = os.path.join(HERE, "worker-progress.json")
    payload = {
        "status": "running",
        "step": step,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **extra,
    }
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


def load_split(split: str) -> dict:
    p = os.path.join(CACHE_DIR, f"schemeN_{split}.npz")
    print(f"  loading {p} ...", flush=True)
    d = np.load(p)
    return {k: d[k] for k in d.files}


def make_slicer(total_dim: int, base_dim: int = 226, n_drop_tail: int = 3,
                drop_fail: bool = True):
    extras_names_path = os.path.join(CACHE_DIR, "schemeN_extra_feat_names.txt")
    feat_names_path = os.path.join(CACHE_DIR, "schemeN_feat_names.txt")
    with open(extras_names_path) as f:
        extra_names = [line.strip() for line in f]
    with open(feat_names_path) as f:
        all_feat_names = [line.strip() for line in f]
    n_extras = total_dim - base_dim
    assert len(extra_names) == n_extras, f"{len(extra_names)} vs {n_extras}"

    drop_set: set = set()
    if drop_fail:
        report_path = os.path.join(T53_DIR, "sym_invariance_report.json")
        with open(report_path) as f:
            report = json.load(f)
        fail_names = report["summary"]["fail_names"]
        fail_local = [extra_names.index(n) for n in fail_names]
        drop_set.update(base_dim + i for i in fail_local)
    for i in range(n_drop_tail):
        drop_set.add(base_dim - 1 - i)
    keep_idx = np.array(
        [i for i in range(total_dim) if i not in drop_set], dtype=np.int64
    )
    feat_names = [all_feat_names[i] for i in keep_idx]
    forbidden = {"date", "sym", "time"}
    leak = forbidden & set(feat_names)
    assert not leak, f"FORBIDDEN feature leak: {leak}"
    return keep_idx, feat_names


# ======================================================================
# Model
# ======================================================================
class PnLNet(nn.Module):
    def __init__(self, n_feat: int, hidden: int = 256, depth: int = 4,
                 dropout: float = 0.4):
        super().__init__()
        layers = []
        dims = [n_feat] + [hidden] * depth
        for i in range(depth):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, NUM_CLASS)

    def forward(self, x):
        return self.head(self.encoder(x))


# ======================================================================
# Loss
# ======================================================================
def differentiable_pnl_components(logits: torch.Tensor,
                                  mp_t: torch.Tensor,
                                  mp_th: torch.Tensor,
                                  fee: float = 0.0001):
    """Return (pnl_per_sample, e_side, e_abs, prob)."""
    prob = F.softmax(logits, dim=-1)
    e_side = prob[:, 2] - prob[:, 0]
    e_abs = prob[:, 2] + prob[:, 0]
    diff = mp_th - mp_t
    fee_term = fee * e_abs * (mp_th + mp_t + 2.0).abs()
    pnl = (e_side * diff - fee_term) / (mp_t + 1.0 + 1e-9)
    return pnl, e_side, e_abs, prob


def pnl_loss(logits, mp_t, mp_th, y, fee=0.0001, ent_beta=0.01,
             activity_target=None, lam_act=0.0, ce_weight=0.0,
             pnl_scale=1.0):
    """Composite loss:
       - pnl_scale * mean(expected_pnl)
       - ent_beta  * mean(entropy(prob))
       + lam_act   * (mean(e_abs) - activity_target)^2
       + ce_weight * CE(logits, y)         -- auxiliary direction supervision
    """
    pnl, e_side, e_abs, prob = differentiable_pnl_components(logits, mp_t, mp_th, fee)
    main_loss = -pnl.mean() * pnl_scale
    ent = -(prob * prob.clamp_min(1e-9).log()).sum(dim=-1).mean()
    total = main_loss - ent_beta * ent
    if activity_target is not None and lam_act > 0:
        act_loss = lam_act * (e_abs.mean() - activity_target).pow(2)
        total = total + act_loss
    if ce_weight > 0:
        ce = F.cross_entropy(logits, y)
        total = total + ce_weight * ce
    else:
        ce = torch.tensor(0.0, device=logits.device)
    info = {
        "main": float(main_loss.item()),
        "ent": float(ent.item()),
        "ce": float(ce.item()),
        "e_abs_mean": float(e_abs.mean().item()),
        "e_side_mean": float(e_side.mean().item()),
        "prob1_mean": float(prob[:, 1].mean().item()),
    }
    return total, info


# ======================================================================
# Dataset
# ======================================================================
class FeatDataset(Dataset):
    def __init__(self, X: np.ndarray, mp_t: np.ndarray, mp_th: np.ndarray,
                 y: np.ndarray):
        self.X = X
        self.mp_t = mp_t
        self.mp_th = mp_th
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.mp_t[i], self.mp_th[i], self.y[i]


def standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray,
                clip: float = 10.0) -> np.ndarray:
    """Z-score, replace NaN with 0, clip to [-clip, clip]."""
    Z = (X - mean) / std
    Z = np.where(np.isfinite(Z), Z, 0.0)  # NaN/Inf → 0
    if clip is not None:
        np.clip(Z, -clip, clip, out=Z)
    return Z.astype(np.float32, copy=False)


# ======================================================================
# Evaluation
# ======================================================================
@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, device, batch=8192) -> np.ndarray:
    model.eval()
    out = np.empty((X.shape[0], NUM_CLASS), dtype=np.float32)
    for s in range(0, X.shape[0], batch):
        xb = torch.from_numpy(X[s:s + batch]).to(device)
        logits = model(xb)
        out[s:s + batch] = F.softmax(logits, dim=-1).cpu().numpy()
    return out


@torch.no_grad()
def expected_pnl_eval(model, X, mp_t, mp_th, device, batch=8192, fee=0.0001):
    """Expected PnL using probabilistic side (used for early stopping val)."""
    model.eval()
    total = 0.0
    n = 0
    e_abs_sum = 0.0
    for s in range(0, X.shape[0], batch):
        xb = torch.from_numpy(X[s:s + batch]).to(device)
        mt = torch.from_numpy(mp_t[s:s + batch]).to(device)
        mh = torch.from_numpy(mp_th[s:s + batch]).to(device)
        logits = model(xb)
        pnl, _, e_abs, _ = differentiable_pnl_components(logits, mt, mh, fee)
        total += float(pnl.sum().item())
        e_abs_sum += float(e_abs.sum().item())
        n += xb.shape[0]
    return total, total / max(n, 1), e_abs_sum / max(n, 1)


def evaluate_with_argmax(probs, y, mp_t, mp_th, fee=0.0001):
    pred = probs.argmax(axis=1).astype(np.int64)
    m = _per_horizon_metrics(pred, y, mp_t, mp_th, fee_rate=fee)
    m["pred_distribution"] = dict(m["pred_distribution"])
    m["label_distribution"] = dict(m["label_distribution"])
    return m


# ======================================================================
# Training a single fold
# ======================================================================
def train_one_fold(
    X_tr, mp_t_tr, mp_th_tr, y_tr,
    X_va, mp_t_va, mp_th_va, y_va,
    X_te, mp_t_te, mp_th_te, y_te,
    feat_mean, feat_std,
    seed: int,
    epochs: int = 30,
    batch_size: int = 4096,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    aug_lo: float = 0.80,
    aug_hi: float = 1.20,
    ent_beta: float = 0.01,
    activity_target: float | None = 0.30,
    lam_act: float = 0.10,
    ce_weight_start: float = 1.0,
    ce_weight_end: float = 0.0,
    ce_warmup_frac: float = 0.5,
    pnl_scale: float = 100.0,
    hidden: int = 256,
    depth: int = 4,
    dropout: float = 0.4,
    fee: float = 0.0001,
    device: str = "cuda",
    log_every: int = 100,
    patience: int = 5,
):
    n_feat = X_tr.shape[1]
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Standardize once (numpy → reused)
    Xtr_z = standardize(X_tr, feat_mean, feat_std)
    Xva_z = standardize(X_va, feat_mean, feat_std)
    Xte_z = standardize(X_te, feat_mean, feat_std)

    model = PnLNet(n_feat=n_feat, hidden=hidden, depth=depth, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_train = Xtr_z.shape[0]
    steps_per_epoch = (n_train + batch_size - 1) // batch_size
    total_steps = steps_per_epoch * epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=lr * 0.01)

    rng = np.random.default_rng(seed)
    best_val_pnl = -1e18
    best_state = None
    best_epoch = -1
    no_improve = 0

    # Pre-extract numpy arrays into pinned memory tensors? skip, just use np
    history = []

    warmup_steps = int(round(total_steps * ce_warmup_frac))
    global_step = 0
    for ep in range(epochs):
        model.train()
        idx_perm = rng.permutation(n_train)
        ep_main = 0.0
        ep_ent = 0.0
        ep_ce = 0.0
        ep_eabs = 0.0
        ep_n = 0

        for s in range(0, n_train, batch_size):
            sel = idx_perm[s:s + batch_size]
            xb_np = Xtr_z[sel]
            # Augment: per-(sample, feature) random scale on z-scored features
            scales = rng.uniform(aug_lo, aug_hi, size=xb_np.shape).astype(np.float32)
            xb_np = xb_np * scales
            mtb = mp_t_tr[sel]
            mhb = mp_th_tr[sel]
            yb = y_tr[sel]

            xb = torch.from_numpy(xb_np).to(device, non_blocking=True)
            mt = torch.from_numpy(mtb).to(device, non_blocking=True)
            mh = torch.from_numpy(mhb).to(device, non_blocking=True)
            yt = torch.from_numpy(yb).to(device, non_blocking=True).long()

            # CE weight schedule: linear from start->end over warmup_steps
            if warmup_steps > 0 and global_step < warmup_steps:
                frac = global_step / max(warmup_steps, 1)
                cw = ce_weight_start * (1 - frac) + ce_weight_end * frac
            else:
                cw = ce_weight_end

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss, info = pnl_loss(
                logits, mt, mh, yt, fee=fee, ent_beta=ent_beta,
                activity_target=activity_target, lam_act=lam_act,
                ce_weight=cw, pnl_scale=pnl_scale,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            sched.step()
            global_step += 1

            bs = xb.shape[0]
            ep_main += info["main"] * bs
            ep_ent += info["ent"] * bs
            ep_ce += info["ce"] * bs
            ep_eabs += info["e_abs_mean"] * bs
            ep_n += bs

        train_main = ep_main / max(ep_n, 1)
        train_ent = ep_ent / max(ep_n, 1)
        train_ce = ep_ce / max(ep_n, 1)
        train_eabs = ep_eabs / max(ep_n, 1)

        # Val eval
        val_pnl_total, val_pnl_per_sample, val_eabs = expected_pnl_eval(
            model, Xva_z, mp_t_va, mp_th_va, device, fee=fee
        )

        history.append({
            "epoch": ep,
            "train_main_loss": train_main,
            "train_entropy": train_ent,
            "train_ce": train_ce,
            "train_e_abs_mean": train_eabs,
            "val_expected_pnl_sum": val_pnl_total,
            "val_e_abs_mean": val_eabs,
        })

        improved = val_pnl_total > best_val_pnl + 1e-6
        if improved:
            best_val_pnl = val_pnl_total
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = ep
            no_improve = 0
        else:
            no_improve += 1

        print(f"    ep={ep:02d} main={train_main:+.4f} ent={train_ent:.3f} "
              f"ce={train_ce:.3f} eabs_tr={train_eabs:.3f} | "
              f"val_pnl={val_pnl_total:+.4f} eabs_va={val_eabs:.3f} "
              f"cw={cw:.3f} {'*BEST*' if improved else ''}", flush=True)

        if no_improve >= patience:
            print(f"    early stop @ ep={ep} (best={best_epoch})", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test eval (argmax)
    probs_te = predict_proba(model, Xte_z, device)
    probs_va = predict_proba(model, Xva_z, device)
    test_metrics = evaluate_with_argmax(probs_te, y_te, mp_t_te, mp_th_te, fee=fee)
    val_metrics = evaluate_with_argmax(probs_va, y_va, mp_t_va, mp_th_va, fee=fee)

    return {
        "model": model,
        "best_state": best_state,
        "best_epoch": best_epoch,
        "best_val_expected_pnl": best_val_pnl,
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
        "probs_te": probs_te,
        "probs_va": probs_va,
        "history": history,
    }


# ======================================================================
# Main: LOSO run
# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--syms", default="0,1,2,3,4")
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--ent-beta", type=float, default=0.01)
    ap.add_argument("--lam-act", type=float, default=0.0)
    ap.add_argument("--activity-target", type=float, default=0.30)
    ap.add_argument("--ce-weight-start", type=float, default=1.0)
    ap.add_argument("--ce-weight-end", type=float, default=0.0)
    ap.add_argument("--ce-warmup-frac", type=float, default=0.5)
    ap.add_argument("--pnl-scale", type=float, default=100.0)
    ap.add_argument("--aug-lo", type=float, default=0.80)
    ap.add_argument("--aug-hi", type=float, default=1.20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--out-tag", default="t60_pnl_nn")
    ap.add_argument("--save-preds", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-drop-tail", type=int, default=3)
    args = ap.parse_args()

    H = args.horizon
    seeds = [int(x) for x in args.seeds.split(",")]
    syms = [int(x) for x in args.syms.split(",")]
    print(f"=== T60 LOSO h={H} seeds={seeds} syms={syms} device={args.device} ===",
          flush=True)
    device = torch.device(args.device)

    progress("loading_caches", seeds=seeds, syms=syms)
    t0 = time.time()
    train_full = load_split("train")
    val_full = load_split("val")
    test_full = load_split("test")
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

    total_dim = train_full["X"].shape[1]
    keep_idx, feat_names = make_slicer(total_dim=total_dim, base_dim=226,
                                       n_drop_tail=args.n_drop_tail, drop_fail=True)
    feat_dim = len(keep_idx)
    print(f"  feat_dim={feat_dim} (base 226 - {args.n_drop_tail} tail - 10 FAIL + extras)",
          flush=True)

    X_tr_full = train_full["X"][:, keep_idx].astype(np.float32, copy=False)
    X_va_full = val_full["X"][:, keep_idx].astype(np.float32, copy=False)
    X_te_full = test_full["X"][:, keep_idx].astype(np.float32, copy=False)
    y_tr_full = train_full[f"y{H}"].astype(np.int64)
    y_va_full = val_full[f"y{H}"].astype(np.int64)
    y_te_full = test_full[f"y{H}"].astype(np.int64)
    sym_tr = train_full["sym"]
    sym_va = val_full["sym"]
    sym_te = test_full["sym"]
    mp_t_va_full = val_full["mp_t"].astype(np.float32)
    mp_th_va_full = val_full[f"mp_t{H}"].astype(np.float32)
    mp_t_te_full = test_full["mp_t"].astype(np.float32)
    mp_th_te_full = test_full[f"mp_t{H}"].astype(np.float32)
    mp_t_tr_full = train_full["mp_t"].astype(np.float32)
    mp_th_tr_full = train_full[f"mp_t{H}"].astype(np.float32)

    # WandB
    use_wandb = not args.no_wandb
    wandb_run = None
    if use_wandb:
        try:
            import wandb
            run_name = f"T60-LOSO-h{H}-{args.out_tag}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            wandb_run = wandb.init(
                project="liangwenbei",
                entity="cjxh21-Tsinghua University",
                name=run_name,
                config={
                    "model": "PnLNet (4-layer MLP)",
                    "scheme": "N (R34 stage 1+2+3, 340-d)",
                    "n_features": feat_dim,
                    "horizon": H,
                    "loss": "differentiable expected PnL",
                    "seeds": seeds,
                    "syms": syms,
                    **{k: v for k, v in vars(args).items()},
                },
                tags=["T60", "RL-NN", "PnL-loss", f"h{H}"],
            )
        except Exception as e:
            print(f"  WandB init failed ({e!r}); skipping", flush=True)
            use_wandb = False

    all_results = {}
    fold_count = 0
    n_total_folds = len(seeds) * len(syms)

    for seed in seeds:
        all_results[seed] = {}
        print(f"\n{'='*78}\n=== seed={seed} ===\n{'='*78}", flush=True)
        for held in syms:
            fold_count += 1
            tag = f"h{H}_seed{seed}_held{held}"
            print(f"\n  --- fold {tag} ({fold_count}/{n_total_folds}) ---", flush=True)
            progress("training_fold", seed=seed, held=held, fold=fold_count, total=n_total_folds)

            m_tr = sym_tr != held
            # IMPORTANT: val from held-out sym (different dates than test
            # but same sym distribution) — much better proxy than val from
            # train syms whose distribution differs from test.
            m_va = sym_va == held
            m_te = sym_te == held

            X_tr = X_tr_full[m_tr]
            y_tr = y_tr_full[m_tr]
            mp_t_tr = mp_t_tr_full[m_tr]
            mp_th_tr = mp_th_tr_full[m_tr]
            X_va = X_va_full[m_va]
            y_va = y_va_full[m_va]
            mp_t_va = mp_t_va_full[m_va]
            mp_th_va = mp_th_va_full[m_va]
            X_te = X_te_full[m_te]
            y_te = y_te_full[m_te]
            mp_t_te = mp_t_te_full[m_te]
            mp_th_te = mp_th_te_full[m_te]

            print(f"    n_train={len(X_tr):,} n_val={len(X_va):,} n_test={len(X_te):,}",
                  flush=True)

            # Compute per-feature stats on TRAIN only (no leakage; sym-pooled but
            # held-out sym is excluded from train so its stats don't appear)
            t_stats = time.time()
            # Use nan-safe stats: 107/340 cols contain NaN (LightGBM handled
            # natively; for NN we impute → mean (z=0) AFTER scaling).
            feat_mean = np.nanmean(X_tr, axis=0).astype(np.float32)
            feat_std = np.nanstd(X_tr, axis=0).astype(np.float32)
            feat_std = np.maximum(feat_std, 1e-6)
            # If a column is fully NaN (shouldn't happen here) replace mean with 0
            feat_mean = np.where(np.isfinite(feat_mean), feat_mean, 0.0).astype(np.float32)
            print(f"    feat_stats computed in {time.time() - t_stats:.1f}s "
                  f"mean[range]=({feat_mean.min():.3g}, {feat_mean.max():.3g}) "
                  f"std[range]=({feat_std.min():.3g}, {feat_std.max():.3g})",
                  flush=True)

            t_fit = time.time()
            res = train_one_fold(
                X_tr, mp_t_tr, mp_th_tr, y_tr,
                X_va, mp_t_va, mp_th_va, y_va,
                X_te, mp_t_te, mp_th_te, y_te,
                feat_mean, feat_std,
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                aug_lo=args.aug_lo,
                aug_hi=args.aug_hi,
                ent_beta=args.ent_beta,
                activity_target=args.activity_target,
                lam_act=args.lam_act,
                ce_weight_start=args.ce_weight_start,
                ce_weight_end=args.ce_weight_end,
                ce_warmup_frac=args.ce_warmup_frac,
                pnl_scale=args.pnl_scale,
                hidden=args.hidden,
                depth=args.depth,
                dropout=args.dropout,
                device=device,
                patience=args.patience,
            )
            t_fit = time.time() - t_fit

            tm = res["test_metrics"]
            print(f"  TEST: cum_pnl={tm['cum_pnl']:+.4f} acc={tm['accuracy']:.3f} "
                  f"f0.5={tm['f0_5_macro']:.3f} n_active={tm['n_predictions_active']:,} "
                  f"pred_dist={tm['pred_distribution']}", flush=True)
            print(f"  fit_time={t_fit:.1f}s", flush=True)

            fold_record = {
                "horizon": H,
                "held": held,
                "seed": seed,
                "n_train": int(len(X_tr)),
                "n_val": int(len(X_va)),
                "n_test": int(len(X_te)),
                "best_epoch": res["best_epoch"],
                "best_val_expected_pnl": res["best_val_expected_pnl"],
                "fit_time_sec": t_fit,
                "test": tm,
                "val": res["val_metrics"],
                "history": res["history"],
            }
            all_results[seed][held] = fold_record

            # Save preds + state
            if args.save_preds:
                np.savez_compressed(
                    os.path.join(HERE, f"loso_pred_h{H}_seed{seed}_held{held}.npz"),
                    probs_te=res["probs_te"].astype(np.float32),
                    probs_va=res["probs_va"].astype(np.float32),
                    y_te=y_te.astype(np.int8),
                    y_va=y_va.astype(np.int8),
                    mp_t_te=mp_t_te.astype(np.float32),
                    mp_th_te=mp_th_te.astype(np.float32),
                    mp_t_va=mp_t_va.astype(np.float32),
                    mp_th_va=mp_th_va.astype(np.float32),
                )
                # save model
                torch.save(res["best_state"],
                           os.path.join(HERE, f"loso_model_h{H}_seed{seed}_held{held}.pt"))

            if use_wandb:
                import wandb
                wandb.log({
                    f"fold_{held}/seed_{seed}/test_cum_pnl": tm["cum_pnl"],
                    f"fold_{held}/seed_{seed}/test_acc": tm["accuracy"],
                    f"fold_{held}/seed_{seed}/test_n_active": tm["n_predictions_active"],
                    f"fold_{held}/seed_{seed}/best_val_pnl": res["best_val_expected_pnl"],
                    f"fold_{held}/seed_{seed}/best_epoch": res["best_epoch"],
                    f"fold_{held}/seed_{seed}/fit_time": t_fit,
                })

    # Aggregate
    aggregate = {}
    for seed in seeds:
        cum_pnls = [all_results[seed][h]["test"]["cum_pnl"] for h in syms]
        accs = [all_results[seed][h]["test"]["accuracy"] for h in syms]
        n_act = [all_results[seed][h]["test"]["n_predictions_active"] for h in syms]
        aggregate[seed] = {
            "horizon": H,
            "seed": seed,
            "n_folds": len(syms),
            "cum_pnl_per_fold": cum_pnls,
            "acc_per_fold": accs,
            "n_active_per_fold": n_act,
            "cum_pnl_sum": float(sum(cum_pnls)),
            "cum_pnl_mean": float(np.mean(cum_pnls)),
            "n_pos_folds": int(sum(1 for x in cum_pnls if x > 0)),
        }

    summary = {
        "task": "T60 NN with direct expected-PnL loss (LOSO)",
        "params": vars(args),
        "horizon": H,
        "n_features": feat_dim,
        "fold_results": all_results,
        "aggregate": aggregate,
    }
    out_path = os.path.join(HERE, f"loso_summary_h{H}_{args.out_tag}.json")
    with open(out_path, "w") as f:
        json.dump(sanitize_for_json(summary), f, indent=2, default=str)
    print(f"\nSAVED summary -> {out_path}", flush=True)

    # Print final summary
    print(f"\n{'='*78}\n=== FINAL aggregate ===\n{'='*78}", flush=True)
    for seed in seeds:
        a = aggregate[seed]
        print(f"  seed={seed} sum={a['cum_pnl_sum']:+.4f} mean={a['cum_pnl_mean']:+.4f} "
              f"per_fold={[f'{x:+.2f}' for x in a['cum_pnl_per_fold']]} "
              f"n_pos={a['n_pos_folds']}/5", flush=True)

    if use_wandb:
        import wandb
        for seed in seeds:
            a = aggregate[seed]
            wandb.log({
                f"agg/seed_{seed}/cum_pnl_sum": a["cum_pnl_sum"],
                f"agg/seed_{seed}/cum_pnl_mean": a["cum_pnl_mean"],
                f"agg/seed_{seed}/n_pos_folds": a["n_pos_folds"],
            })
        wandb.finish()

    progress("done", aggregate=aggregate)


if __name__ == "__main__":
    main()
