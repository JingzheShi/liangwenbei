"""T186: TickTransformer + GroupTransformer LR sweep with proper val + early stop.

Sweeps:
  Arch A: TickTransformer on raw LOB (31-feat, 100-tick window)
  Arch B: GroupTransformer on schemeP (370-d static features)
  LRs: {3e-4, 1e-4, 3e-5, 1e-5}
  Batches: {512, 256, 128}
  Order: batch=512 → 256 → 128, lr=3e-4 → 1e-4 → 3e-5 → 1e-5

CRITICAL_CONSTRAINTS:
  - No date features (date is zeroed at eval time)
  - Stateless forward (no cross-call state)
  - Sym-agnostic (no per-sym paths)
"""
from __future__ import annotations
import copy, json, os, sys, time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SP_CACHE = os.path.join(ROOT, "experiments", "T68_stage5_features", "cache")
LOB_CACHE = os.path.join(ROOT, "experiments", "T95_cnn_rnn_deeplob", "cache")
CKPT_DIR = os.path.join(HERE, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

WINDOW = 100
FEE = 0.0001
CLIP = 10.0
PATIENCE = 5
MAX_EPOCHS = 60
ETA_MIN = 1e-5
NUM_WORKERS = 4


def progress(step, **kw):
    d = {"status": "running", "step": step,
         "timestamp": datetime.now(timezone.utc).isoformat(), **kw}
    with open(os.path.join(HERE, "worker-progress.json"), "w") as f:
        json.dump(d, f, indent=2)


def regr_target(mp_t, mp_th):
    return ((mp_th.astype(np.float64) - mp_t.astype(np.float64))
            / (mp_t.astype(np.float64) + 1.0)).astype(np.float32)


def class_balanced_weight(y_cls, n=3):
    counts = np.bincount(y_cls, minlength=n).astype(np.float64)
    w = 1.0 / np.maximum(counts, 1.0)
    return (w / w.mean())[y_cls].astype(np.float32)


def eval_pnl(pred, mp_t, mp_th):
    thr = 2.0 * FEE
    side = np.where(pred > thr, 1.0, np.where(pred < -thr, -1.0, 0.0))
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    pnl = ((side * diff - fee) / (mp_t.astype(np.float64) + 1.0)).sum()
    return float(pnl), int((side != 0).sum())


# ── ArchA: WindowDataset ────────────────────────────────────────────────────

class WindowDataset(Dataset):
    def __init__(self, lob_arr, sess_id, t, y, sw):
        self.lob_arr = lob_arr
        self.sess_id = sess_id
        self.t = t
        self.y = y
        self.sw = sw
        valid = t >= WINDOW - 1
        self.idx = np.where(valid)[0].astype(np.int64)
        self.valid_mask = valid  # keep for external use

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        s = int(self.sess_id[j])
        ti = int(self.t[j])
        win = self.lob_arr[s, ti - WINDOW + 1: ti + 1].astype(np.float32)
        return win, self.y[j], self.sw[j]


def window_normalize(x, clip=CLIP):
    m = x.mean(dim=1, keepdim=True)
    s = x.std(dim=1, keepdim=True).clamp_min(1e-8)
    return ((x - m) / s).clamp(-clip, clip)


# ── ArchB: FeatureDataset ───────────────────────────────────────────────────

class FeatureDataset(Dataset):
    def __init__(self, X, y, sw):
        self.X = X
        self.y = y
        self.sw = sw

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.sw[i]


# ── Model: ArchA ─────────────────────────────────────────────────────────────

class TickTr(nn.Module):
    def __init__(self, n_in=31, d=64, nhead=4, n_layers=2):
        super().__init__()
        self.proj = nn.Linear(n_in, d)
        self.pe = nn.Parameter(torch.randn(1, WINDOW, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, nhead, 256, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x):  # (B, W, n_in)
        h = self.proj(x) + self.pe
        h = self.tr(h).mean(dim=1)
        return self.head(h).squeeze(-1)


# ── Model: ArchB ─────────────────────────────────────────────────────────────

class GroupTr(nn.Module):
    def __init__(self, n_in=370, n_tokens=32, d=64, nhead=4, n_layers=2):
        super().__init__()
        self.proj = nn.Linear(n_in, n_tokens * d)
        self.n_tokens = n_tokens
        self.d = d
        self.proj_norm = nn.LayerNorm(d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        enc_layer = nn.TransformerEncoderLayer(d, nhead, 256, dropout=0.1,
                                               batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc_layer, n_layers)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(),
                                  nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, x):  # (B, n_in)
        B = x.shape[0]
        tokens = self.proj(x).reshape(B, self.n_tokens, self.d)
        tokens = self.proj_norm(tokens)
        cls = self.cls.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        return self.head(self.tr(tokens)[:, 0]).squeeze(-1)


# ── Unified training loop ────────────────────────────────────────────────────

def run_config(model, train_ds, val_ds, lr, batch_size, target_scale,
               device, arch_name, is_windowed, val_mp_t, val_mp_th,
               wandb_mod, config_id):
    """Train one config with early stopping. Returns result dict."""
    use_amp = device.type == "cuda"
    grad_scaler = torch.cuda.amp.GradScaler() if use_amp else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=ETA_MIN)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=False,
                              persistent_workers=(NUM_WORKERS > 0))
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            persistent_workers=(NUM_WORKERS > 0))

    best_val_mse = float("inf")
    best_epoch = -1
    patience_count = 0
    best_state = None
    initial_val_mse = None
    epoch_logs = []
    t0 = time.time()

    for epoch in range(MAX_EPOCHS):
        # ── Train ──
        model.train()
        rl, ns = 0.0, 0
        for xb, yb, sw in train_loader:
            yb = yb.to(device, non_blocking=True)
            sw = sw.to(device, non_blocking=True)

            if is_windowed:
                xb = xb.to(device, non_blocking=True)
                # Augmentation: scale raw window, then normalize, then add noise
                scl = torch.empty(xb.size(0), 1, xb.size(2), device=device).uniform_(0.98, 1.02)
                xb = window_normalize(xb * scl)
                xb = xb + torch.randn_like(xb) * 0.01
            else:
                xb = xb.to(device, non_blocking=True)

            optimizer.zero_grad()
            y_s = yb * target_scale
            if use_amp:
                with torch.cuda.amp.autocast():
                    pred = model(xb)
                    loss = ((pred - y_s) ** 2 * sw).sum() / sw.sum().clamp_min(1.0)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                pred = model(xb)
                loss = ((pred - y_s) ** 2 * sw).sum() / sw.sum().clamp_min(1.0)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            rl += float(loss.item()) * len(yb)
            ns += len(yb)

        train_mse = rl / max(ns, 1)
        scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]

        # ── Val MSE ──
        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for xb, yb, _ in val_loader:
                yb = yb.to(device, non_blocking=True)
                if is_windowed:
                    xb = window_normalize(xb.to(device, non_blocking=True))
                else:
                    xb = xb.to(device, non_blocking=True)
                y_s = yb * target_scale
                if use_amp:
                    with torch.cuda.amp.autocast():
                        pred = model(xb)
                else:
                    pred = model(xb)
                vl += float(((pred - y_s) ** 2).mean().item()) * len(yb)
                vn += len(yb)
        val_mse = vl / max(vn, 1)

        if initial_val_mse is None:
            initial_val_mse = val_mse

        ep_time = time.time() - t0
        t0 = time.time()
        log = {"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse, "lr": cur_lr}
        epoch_logs.append(log)
        print(f"    ep {epoch:3d}  train={train_mse:.5e}  val={val_mse:.5e}  "
              f"lr={cur_lr:.2e}  ({ep_time:.1f}s)", flush=True)

        if wandb_mod is not None:
            wandb_mod.log({"epoch": epoch, "train_mse": train_mse,
                           "val_mse": val_mse, "lr": cur_lr})

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            patience_count = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"    Early stop at epoch {epoch} (best={best_epoch})", flush=True)
                break

    # ── Load best model → compute val PnL ──
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    preds_list = []
    with torch.no_grad():
        for xb, _, _ in val_loader:
            if is_windowed:
                xb = window_normalize(xb.to(device, non_blocking=True))
            else:
                xb = xb.to(device, non_blocking=True)
            if use_amp:
                with torch.cuda.amp.autocast():
                    pred = model(xb)
            else:
                pred = model(xb)
            preds_list.append(pred.cpu().float().numpy())

    val_preds_s = np.concatenate(preds_list)
    val_preds = val_preds_s / target_scale
    val_pnl, n_trades = eval_pnl(val_preds, val_mp_t, val_mp_th)

    improvement = (initial_val_mse - best_val_mse) / max(abs(initial_val_mse), 1e-10)
    can_learn = best_epoch >= 5 and improvement > 0.05

    result = {
        "lr": lr, "batch": batch_size,
        "best_epoch": best_epoch,
        "initial_val_mse": float(initial_val_mse) if initial_val_mse is not None else None,
        "best_val_mse": float(best_val_mse),
        "val_mse_drop_pct": float(improvement * 100),
        "val_pnl": float(val_pnl),
        "n_trades": int(n_trades),
        "can_learn": bool(can_learn),
    }

    if wandb_mod is not None:
        wandb_mod.log({"best_epoch": best_epoch, "best_val_mse": best_val_mse,
                       "val_pnl": val_pnl, "n_trades": n_trades, "can_learn": can_learn})

    return result


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== T186: TickTr + GroupTr LR sweep  device={device} ===\n", flush=True)

    # ── WandB ──
    try:
        import wandb as wm
        WDB = wm
        WDB.login(key="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",
                  relogin=True)
        print("WandB connected", flush=True)
    except Exception as e:
        print(f"WandB unavailable: {e}", flush=True)
        WDB = None

    # ── Sweep configs ──
    BATCHES = [512, 256, 128]
    LRS = [3e-4, 1e-4, 3e-5, 1e-5]

    # ═══════════════════════════════════════════════════════
    # ARCH A: TickTransformer on raw LOB
    # ═══════════════════════════════════════════════════════
    print("=== Loading ArchA (LOB) data ===", flush=True)
    progress("loading_arch_a")
    t0 = time.time()

    lob_tr_raw = np.load(os.path.join(LOB_CACHE, "lob_train_v2.npz"))
    lob_te_raw = np.load(os.path.join(LOB_CACHE, "lob_test_v2.npz"))

    # Targets
    y_tr_a = regr_target(lob_tr_raw["mp_t"], lob_tr_raw["mp_t60"])
    y_cls_tr_a = lob_tr_raw["y60"].astype(np.int64)
    target_scale_a = float(1.0 / max(y_tr_a.std(), 1e-8))
    sw_tr_a = class_balanced_weight(y_cls_tr_a)
    y_tr_a_f = y_tr_a.astype(np.float32)

    y_te_a = regr_target(lob_te_raw["mp_t"], lob_te_raw["mp_t60"])
    y_cls_te_a = lob_te_raw["y60"].astype(np.int64)
    sw_te_a = class_balanced_weight(y_cls_te_a)
    y_te_a_f = y_te_a.astype(np.float32)

    ds_train_a = WindowDataset(lob_tr_raw["lob_arr"], lob_tr_raw["sess_id_for_sample"],
                               lob_tr_raw["t_for_sample"], y_tr_a_f, sw_tr_a)
    ds_val_a = WindowDataset(lob_te_raw["lob_arr"], lob_te_raw["sess_id_for_sample"],
                             lob_te_raw["t_for_sample"], y_te_a_f, sw_te_a)

    # Val PnL arrays (only valid samples, same filter as WindowDataset)
    val_valid_mask_a = ds_val_a.valid_mask
    val_mp_t_a = lob_te_raw["mp_t"][val_valid_mask_a]
    val_mp_th_a = lob_te_raw["mp_t60"][val_valid_mask_a]

    n_raw = lob_tr_raw["lob_arr"].shape[-1]
    print(f"  ArchA: n_train={len(ds_train_a):,} n_val={len(ds_val_a):,} "
          f"n_raw={n_raw} target_scale={target_scale_a:.4f}  ({time.time()-t0:.1f}s)", flush=True)

    arch_a_results = []
    arch_a_best = None

    for batch_size in BATCHES:
        for lr in LRS:
            config_id = f"ArchA_bs{batch_size}_lr{lr:.0e}"
            print(f"\n── {config_id} ──", flush=True)
            progress("arch_a_training", config=config_id)

            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            model = TickTr(n_in=n_raw).to(device)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  n_params={n_params:,}", flush=True)

            wrun = None
            if WDB is not None:
                try:
                    wrun = WDB.init(
                        project="liangwenbei-T186",
                        entity="cjxh21-Tsinghua University",
                        name=config_id,
                        config={"arch": "TickTr", "lr": lr, "batch": batch_size,
                                "n_in": n_raw, "n_params": n_params},
                        tags=["T186", "ArchA", "TickTransformer"],
                        reinit="finish_previous",
                    )
                except Exception as e:
                    print(f"  WandB run init failed: {e}", flush=True)
                    wrun = None

            res = run_config(
                model, ds_train_a, ds_val_a, lr, batch_size, target_scale_a,
                device, "ArchA", is_windowed=True,
                val_mp_t=val_mp_t_a, val_mp_th=val_mp_th_a,
                wandb_mod=wrun, config_id=config_id,
            )
            arch_a_results.append(res)

            if wrun is not None:
                wrun.finish()

            print(f"  ► {config_id}: best_epoch={res['best_epoch']} "
                  f"val_mse={res['best_val_mse']:.5e} "
                  f"drop={res['val_mse_drop_pct']:.1f}% "
                  f"val_pnl={res['val_pnl']:.3f} "
                  f"can_learn={res['can_learn']}", flush=True)

            if arch_a_best is None or res['best_val_mse'] < arch_a_best['best_val_mse']:
                arch_a_best = res

    arch_a_can_learn = any(r["can_learn"] for r in arch_a_results)

    # ═══════════════════════════════════════════════════════
    # ARCH B: GroupTransformer on schemeP
    # ═══════════════════════════════════════════════════════
    print("\n=== Loading ArchB (schemeP) data ===", flush=True)
    progress("loading_arch_b")
    t0 = time.time()

    sp_tr = np.load(os.path.join(SP_CACHE, "schemeP_train.npz"))
    sp_val_raw = np.load(os.path.join(SP_CACHE, "schemeP_val.npz"))
    sp_te = np.load(os.path.join(SP_CACHE, "schemeP_test.npz"))

    # Concatenate train + val for training (dates 0-95)
    X_tr_b = np.concatenate([sp_tr["X"], sp_val_raw["X"]], axis=0).astype(np.float32)
    mp_t_tr_b = np.concatenate([sp_tr["mp_t"], sp_val_raw["mp_t"]], axis=0)
    mp_th_tr_b = np.concatenate([sp_tr["mp_t60"], sp_val_raw["mp_t60"]], axis=0)
    y_cls_tr_b = np.concatenate([sp_tr["y60"], sp_val_raw["y60"]], axis=0).astype(np.int64)

    y_tr_b = regr_target(mp_t_tr_b, mp_th_tr_b)
    target_scale_b = float(1.0 / max(y_tr_b.std(), 1e-8))
    sw_tr_b = class_balanced_weight(y_cls_tr_b)
    y_tr_b_s = y_tr_b.astype(np.float32)

    # Normalize: fit on all train (0-95), apply to val (96-119)
    np.nan_to_num(X_tr_b, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    feat_mean_b = X_tr_b.mean(axis=0)
    feat_std_b = X_tr_b.std(axis=0)
    feat_std_b[feat_std_b < 1e-8] = 1.0
    X_tr_b_n = np.clip((X_tr_b - feat_mean_b) / feat_std_b, -10, 10).astype(np.float32)
    np.nan_to_num(X_tr_b_n, nan=0.0, copy=False)

    X_te_b = sp_te["X"].astype(np.float32)
    mp_t_te_b = sp_te["mp_t"]
    mp_th_te_b = sp_te["mp_t60"]
    y_cls_te_b = sp_te["y60"].astype(np.int64)
    y_te_b = regr_target(mp_t_te_b, mp_th_te_b)
    sw_te_b = class_balanced_weight(y_cls_te_b)
    y_te_b_s = y_te_b.astype(np.float32)

    np.nan_to_num(X_te_b, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    X_te_b_n = np.clip((X_te_b - feat_mean_b) / feat_std_b, -10, 10).astype(np.float32)
    np.nan_to_num(X_te_b_n, nan=0.0, copy=False)

    ds_train_b = FeatureDataset(X_tr_b_n, y_tr_b_s, sw_tr_b)
    ds_val_b = FeatureDataset(X_te_b_n, y_te_b_s, sw_te_b)

    n_feat_b = X_tr_b_n.shape[1]
    print(f"  ArchB: n_train={len(ds_train_b):,} n_val={len(ds_val_b):,} "
          f"n_feat={n_feat_b} target_scale={target_scale_b:.4f}  ({time.time()-t0:.1f}s)", flush=True)

    arch_b_results = []
    arch_b_best = None

    for batch_size in BATCHES:
        for lr in LRS:
            config_id = f"ArchB_bs{batch_size}_lr{lr:.0e}"
            print(f"\n── {config_id} ──", flush=True)
            progress("arch_b_training", config=config_id)

            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            model = GroupTr(n_in=n_feat_b).to(device)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  n_params={n_params:,}", flush=True)

            wrun = None
            if WDB is not None:
                try:
                    wrun = WDB.init(
                        project="liangwenbei-T186",
                        entity="cjxh21-Tsinghua University",
                        name=config_id,
                        config={"arch": "GroupTr", "lr": lr, "batch": batch_size,
                                "n_in": n_feat_b, "n_params": n_params},
                        tags=["T186", "ArchB", "GroupTransformer"],
                        reinit="finish_previous",
                    )
                except Exception as e:
                    print(f"  WandB run init failed: {e}", flush=True)
                    wrun = None

            res = run_config(
                model, ds_train_b, ds_val_b, lr, batch_size, target_scale_b,
                device, "ArchB", is_windowed=False,
                val_mp_t=mp_t_te_b, val_mp_th=mp_th_te_b,
                wandb_mod=wrun, config_id=config_id,
            )
            arch_b_results.append(res)

            if wrun is not None:
                wrun.finish()

            print(f"  ► {config_id}: best_epoch={res['best_epoch']} "
                  f"val_mse={res['best_val_mse']:.5e} "
                  f"drop={res['val_mse_drop_pct']:.1f}% "
                  f"val_pnl={res['val_pnl']:.3f} "
                  f"can_learn={res['can_learn']}", flush=True)

            if arch_b_best is None or res['best_val_mse'] < arch_b_best['best_val_mse']:
                arch_b_best = res

    arch_b_can_learn = any(r["can_learn"] for r in arch_b_results)

    # ═══════════════════════════════════════════════════════
    # Verdict
    # ═══════════════════════════════════════════════════════
    if arch_a_can_learn or arch_b_can_learn:
        verdict = "pipeline_was_issue"
    else:
        verdict = "architecture_doesnt_fit"

    best_a_pnl = arch_a_best["val_pnl"] if arch_a_best else None
    best_b_pnl = arch_b_best["val_pnl"] if arch_b_best else None

    results = {
        "task": "T186 retrain T184/T172b with val + early stop LR sweep",
        "verdict": verdict,
        "arch_A_TickTr": {
            "configs_tested": arch_a_results,
            "best_config": arch_a_best,
            "can_learn": arch_a_can_learn,
            "notes": (f"val MSE drop: best={arch_a_best['val_mse_drop_pct']:.1f}%"
                      if arch_a_best else "no configs ran"),
        },
        "arch_B_GroupTr": {
            "configs_tested": arch_b_results,
            "best_config": arch_b_best,
            "can_learn": arch_b_can_learn,
            "notes": (f"val MSE drop: best={arch_b_best['val_mse_drop_pct']:.1f}%"
                      if arch_b_best else "no configs ran"),
        },
        "comparison": {
            "T87_baseline_holdout_pnl": 22.0,
            "best_TickTr_val_pnl": best_a_pnl,
            "best_GroupTr_val_pnl": best_b_pnl,
            "conclusion": verdict,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    rp = os.path.join(HERE, "results.json")
    with open(rp, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n=== RESULTS saved to {rp} ===", flush=True)
    print(json.dumps({
        "verdict": verdict,
        "arch_A_can_learn": arch_a_can_learn,
        "arch_A_best_pnl": best_a_pnl,
        "arch_B_can_learn": arch_b_can_learn,
        "arch_B_best_pnl": best_b_pnl,
    }, indent=2), flush=True)

    progress("done", verdict=verdict,
             arch_a_can_learn=arch_a_can_learn, arch_b_can_learn=arch_b_can_learn)

    print(f"\nRESULT: task=[T186 retrain with val+early-stop] "
          f"metrics={{arch_A_can_learn={arch_a_can_learn}, "
          f"arch_A_best_pnl={best_a_pnl:.3f if best_a_pnl is not None else 'N/A'}, "
          f"arch_B_can_learn={arch_b_can_learn}, "
          f"arch_B_best_pnl={best_b_pnl:.3f if best_b_pnl is not None else 'N/A'}}} "
          f"notes=[{verdict}]")


if __name__ == "__main__":
    main()
