"""T157: Temporal Transformer NN — 100-tick sequence input, encoder-only.

Architecture:
  Input: (B, T=100, F=359)
  Linear projection → d_model=128
  TransformerEncoder: 3 layers, 4 heads, d_ff=256, dropout=0.1
  Global Average Pooling across time
  Head: Linear(128, 1)

Training:
  V4 split: train=date 0-75, val=date 76-79
  20 epochs, lr=1e-4, batch=512, AdamW wd=1e-4
  EarlyStop on val loss (patience=5)
  Class-balanced sample weights
  Target: (mp_t60 - mp_t)/(mp_t+1)

CRITICAL_CONSTRAINTS compliance:
  - date/sym/time never fed to model
  - Global normalization stats (no per-sym)
  - Predictor is stateless (sequences built from per-call window)
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

T26_DIR = os.path.join(ROOT, "experiments", "T26_domain_randomization")
sys.path.insert(0, T26_DIR)
from build_aug import class_balanced_weight  # noqa: E402

T68_DIR = os.path.join(ROOT, "experiments", "T68_stage5_features")
CACHE_DIR = os.path.join(T68_DIR, "cache")

SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001
SEQ_LEN = 100

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


def build_seq_indices(sym_arr, date_arr, sess_arr, t_arr, seq_len=SEQ_LEN):
    """Build (n, seq_len) int32 array of row indices for each sample's sequence.

    For each sample i, seq_indices[i] = [i-99, i-98, ..., i] clipped to group start.
    Groups are defined by (sym, date, sess_idx). Within-group positions are consecutive.
    Samples before group_start+seq_len-1 are padded by repeating group_start.
    """
    n = len(sym_arr)
    # Sort order for grouping
    sort_order = np.lexsort((t_arr, sess_arr, date_arr, sym_arr))
    # Inverse sort order for later reindexing
    inv_order = np.argsort(sort_order).astype(np.int32)

    sym_s = sym_arr[sort_order]
    date_s = date_arr[sort_order]
    sess_s = sess_arr[sort_order]

    # Detect group boundaries
    group_changes = np.ones(n, dtype=bool)
    group_changes[1:] = (
        (sym_s[1:] != sym_s[:-1]) |
        (date_s[1:] != date_s[:-1]) |
        (sess_s[1:] != sess_s[:-1])
    )
    group_starts_sorted = np.where(group_changes)[0].astype(np.int32)  # in sorted space

    # For each position in sorted order, find its group start
    group_start_for_pos = np.empty(n, dtype=np.int32)
    for k in range(len(group_starts_sorted)):
        gs = group_starts_sorted[k]
        ge = group_starts_sorted[k + 1] if k + 1 < len(group_starts_sorted) else n
        group_start_for_pos[gs:ge] = gs

    # Build seq_indices in sorted space (vectorized over seq positions)
    seq_indices_sorted = np.empty((n, seq_len), dtype=np.int32)
    for offset in range(seq_len):
        look_back = seq_len - 1 - offset  # 99, 98, ..., 0
        positions = np.arange(n, dtype=np.int32) - look_back
        np.maximum(positions, group_start_for_pos, out=positions)
        seq_indices_sorted[:, offset] = positions

    # Map back to original (unsorted) space
    # seq_indices[i] should use original row indices
    # The sorted row at position p corresponds to original row sort_order[p]
    # So seq_indices_orig[inv_order[p], :] = sort_order[seq_indices_sorted[p, :]]
    seq_indices_orig = np.empty((n, seq_len), dtype=np.int32)
    # For each sorted position p, its original index is sort_order[p]
    # Its sequence references sorted positions seq_indices_sorted[p]
    # Which correspond to original indices sort_order[seq_indices_sorted[p]]
    # Build: seq_indices_orig[sort_order[p]] = sort_order[seq_indices_sorted[p]]
    # Vectorized:
    seq_indices_orig[sort_order] = sort_order[seq_indices_sorted]
    return seq_indices_orig


class TemporalTransformer(nn.Module):
    """Encoder-only Transformer for temporal sequence of features."""
    def __init__(self, in_dim=359, d_model=128, n_heads=4, n_layers=3, d_ff=256, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        # x: (B, T, F)
        h = self.proj(x)          # (B, T, d_model)
        h = self.encoder(h)       # (B, T, d_model)
        h = self.norm(h.mean(dim=1))  # GAP + norm: (B, d_model)
        return self.head(h).squeeze(-1)


def extract_npz_weights(model):
    """Extract model weights as numpy dict for npz save."""
    return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}


def predict_chunked(model, X_gpu, seq_idx_gpu, idxs, batch=512, device="cuda"):
    """Run inference over a subset of rows. idxs: array of row indices."""
    model.eval()
    out = np.zeros(len(idxs), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(idxs), batch):
            batch_idxs = idxs[s:s + batch]
            seq_rows = seq_idx_gpu[batch_idxs]   # (b, 100) int32
            xb = X_gpu[seq_rows]                  # (b, 100, F) fancy index on GPU
            yb = model(xb).float().cpu().numpy()
            out[s:s + len(batch_idxs)] = yb
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--d-ff", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--cuda", default="0")
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    H = args.horizon
    seed = args.seed
    print(f"=== T157 TemporalTransformer seed={seed} lr={args.lr} ===", flush=True)
    print(f"  device={device}", flush=True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    progress("loading_caches", seed=seed)
    t0 = time.time()
    train_full = load_split("train")
    test_full = load_split("test")
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    # Feature selection
    feat_names_path = os.path.join(CACHE_DIR, "schemeP_feat_names.txt")
    with open(feat_names_path) as f:
        all_feat_names = [line.strip() for line in f]
    total_dim = train_full["X"].shape[1]
    assert len(all_feat_names) == total_dim

    name_to_idx = {n: i for i, n in enumerate(all_feat_names)}
    drop_idx = {name_to_idx[n] for n in DROP_NAMES if n in name_to_idx}
    keep_idx = np.array([i for i in range(total_dim) if i not in drop_idx], dtype=np.int64)
    feat_dim = len(keep_idx)
    print(f"  feat_dim={feat_dim}", flush=True)

    # V4 split
    date_tr = train_full["date"]
    m_va = date_tr >= 76
    m_t = ~m_va

    X_tr_raw = train_full["X"][m_t][:, keep_idx]
    sym_tr = train_full["sym"][m_t]
    date_tr2 = train_full["date"][m_t]
    sess_tr = train_full["sess_idx"][m_t]
    t_tr = train_full["t"][m_t]
    y_cls_tr = train_full["y60"][m_t].astype(np.int64)
    y_regr_tr = regr_target(train_full["mp_t"][m_t], train_full["mp_t60"][m_t])

    X_va_raw = train_full["X"][m_va][:, keep_idx]
    sym_va = train_full["sym"][m_va]
    date_va = train_full["date"][m_va]
    sess_va = train_full["sess_idx"][m_va]
    t_va = train_full["t"][m_va]
    y_regr_va = regr_target(train_full["mp_t"][m_va], train_full["mp_t60"][m_va])

    X_te_raw = test_full["X"][:, keep_idx]
    sym_te = test_full["sym"]
    date_te = test_full["date"]
    sess_te = test_full["sess_idx"]
    t_te = test_full["t"]
    y_cls_te = test_full[f"y{H}"].astype(np.int64)
    y_regr_te = regr_target(test_full["mp_t"], test_full[f"mp_t{H}"])
    mp_t_te = test_full["mp_t"]
    mp_th_te = test_full[f"mp_t{H}"]

    print(f"  V4 split: n_train={len(X_tr_raw):,} n_val={len(X_va_raw):,} n_test={len(X_te_raw):,}", flush=True)

    # Normalize using train split stats
    progress("normalizing", seed=seed)
    mu = np.nanmean(X_tr_raw, axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(np.nanstd(X_tr_raw, axis=0, dtype=np.float64).astype(np.float32), 1e-6)

    def normalize(X):
        Xs = (X - mu) / std
        Xs = np.where(np.isnan(Xs), 0.0, Xs)
        return np.clip(Xs, -10.0, 10.0).astype(np.float32)

    X_tr_s = normalize(X_tr_raw)
    X_va_s = normalize(X_va_raw)
    X_te_s = normalize(X_te_raw)
    del X_tr_raw, X_va_raw, X_te_raw  # free memory

    # Build sequence indices
    progress("building_seq_indices", seed=seed)
    t_seq = time.time()
    print("  building train seq indices ...", flush=True)
    seq_tr = build_seq_indices(sym_tr, date_tr2, sess_tr, t_tr)
    print(f"  train seq built in {time.time()-t_seq:.1f}s", flush=True)

    t_seq = time.time()
    print("  building val seq indices ...", flush=True)
    seq_va = build_seq_indices(sym_va, date_va, sess_va, t_va)
    print(f"  val seq built in {time.time()-t_seq:.1f}s", flush=True)

    t_seq = time.time()
    print("  building test seq indices ...", flush=True)
    seq_te = build_seq_indices(sym_te, date_te, sess_te, t_te)
    print(f"  test seq built in {time.time()-t_seq:.1f}s", flush=True)

    # Target scale
    target_scale = 1.0 / float(max(y_regr_tr.std(), 1e-8))
    print(f"  target_scale={target_scale:.4f}", flush=True)
    y_tr_s = (y_regr_tr * target_scale).astype(np.float32)

    # Sample weights
    sw = class_balanced_weight(y_cls_tr, num_class=3).astype(np.float32)

    # Move data to GPU
    progress("moving_to_gpu", seed=seed)
    print("  moving X tensors to GPU ...", flush=True)
    t_gpu = time.time()
    X_tr_gpu = torch.from_numpy(X_tr_s).to(device)   # (n_tr, F)
    X_va_gpu = torch.from_numpy(X_va_s).to(device)   # (n_va, F)
    X_te_gpu = torch.from_numpy(X_te_s).to(device)   # (n_te, F)

    seq_tr_gpu = torch.from_numpy(seq_tr).to(device)  # (n_tr, 100) int32
    seq_va_gpu = torch.from_numpy(seq_va).to(device)  # (n_va, 100) int32
    seq_te_gpu = torch.from_numpy(seq_te).to(device)  # (n_te, 100) int32

    y_tr_gpu = torch.from_numpy(y_tr_s).to(device)
    sw_gpu = torch.from_numpy(sw).to(device)
    print(f"  GPU transfer in {time.time()-t_gpu:.1f}s", flush=True)

    # WandB
    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = f"T157-transformer-seed{seed}-{datetime.now().strftime('%H%M%S')}"
            wandb.init(
                project="liangwenbei",
                entity="cjxh21-Tsinghua University",
                name=run_name,
                config={"task": "T157_NN_transformer",
                        **{k: v for k, v in vars(args).items()}},
                tags=["T157", "transformer", f"seed{seed}"],
                reinit=True,
            )
        except Exception as e:
            print(f"  wandb init failed ({e!r}); skip", flush=True)
            use_wandb = False

    # Build model
    model = TemporalTransformer(
        in_dim=feat_dim, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, d_ff=args.d_ff, dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params={n_params:,}", flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    use_amp = args.amp and torch.cuda.is_available()
    amp_dtype = torch.bfloat16

    n_train = len(X_tr_gpu)
    best_val_mse = float("inf")
    best_epoch = -1
    best_state = None
    bad = 0
    epoch_logs = []

    progress("training", seed=seed, n_train=n_train)
    t_train = time.time()

    n_va = len(X_va_gpu)
    va_all_idxs = torch.arange(n_va, device=device)

    for epoch in range(args.epochs):
        t_ep = time.time()
        model.train()
        perm = torch.randperm(n_train, device=device)
        running_loss = 0.0
        running_n = 0

        for s in range(0, n_train, args.batch_size):
            idx = perm[s:s + args.batch_size]
            # GPU fancy index: seq_tr_gpu[idx] -> (b, 100) int32
            # X_tr_gpu[seq_rows] -> (b, 100, F)
            seq_rows = seq_tr_gpu[idx]           # (b, 100) int32
            xb = X_tr_gpu[seq_rows]              # (b, 100, F)
            yb = y_tr_gpu[idx]
            wb = sw_gpu[idx]

            optim.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    pred = model(xb)
                    loss = ((pred.float() - yb) ** 2 * wb).sum() / wb.sum().clamp_min(1.0)
            else:
                pred = model(xb)
                loss = ((pred - yb) ** 2 * wb).sum() / wb.sum().clamp_min(1.0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()
            running_loss += float(loss.item()) * len(idx)
            running_n += len(idx)

        sched.step()
        train_loss_avg = running_loss / running_n

        # Val eval
        model.eval()
        va_preds = []
        with torch.no_grad():
            for s in range(0, n_va, 2048):
                batch_idxs = va_all_idxs[s:s + 2048]
                seq_rows = seq_va_gpu[batch_idxs]
                xb = X_va_gpu[seq_rows]
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        yb = model(xb).float().cpu().numpy()
                else:
                    yb = model(xb).cpu().numpy()
                va_preds.append(yb)
        va_pred_s = np.concatenate(va_preds)
        va_pred = va_pred_s / target_scale
        val_mse = float(((va_pred - y_regr_va) ** 2).mean())
        val_corr = float(np.corrcoef(va_pred, y_regr_va)[0, 1]) if va_pred.std() > 0 else 0.0
        ep_t = time.time() - t_ep
        cur_lr = optim.param_groups[0]["lr"]

        epoch_logs.append({"epoch": epoch, "train_loss": train_loss_avg, "val_mse": val_mse,
                           "val_corr": val_corr, "lr": cur_lr, "time": ep_t})
        print(f"  ep {epoch:3d}  train_loss={train_loss_avg:.5e}  "
              f"val_mse={val_mse:.5e}  val_corr={val_corr:.4f}  lr={cur_lr:.2e}  ({ep_t:.1f}s)",
              flush=True)

        if use_wandb:
            try:
                wandb.log({"epoch": epoch, "train_loss": train_loss_avg, "val_mse": val_mse,
                           "val_corr": val_corr, "lr": cur_lr})
            except Exception:
                pass

        if val_mse < best_val_mse - 1e-10:
            best_val_mse = val_mse
            best_epoch = epoch
            best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  early stop at ep {epoch}", flush=True)
                break

    train_time = time.time() - t_train
    print(f"  train done in {train_time:.1f}s; best_epoch={best_epoch} best_val_mse={best_val_mse:.5e}",
          flush=True)

    assert best_state is not None
    model.load_state_dict(best_state)

    # Test predictions
    model.eval()
    n_te = len(X_te_gpu)
    te_all_idxs = torch.arange(n_te, device=device)
    te_preds = []
    with torch.no_grad():
        for s in range(0, n_te, 2048):
            batch_idxs = te_all_idxs[s:s + 2048]
            seq_rows = seq_te_gpu[batch_idxs]
            xb = X_te_gpu[seq_rows]
            if use_amp:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    yb = model(xb).float().cpu().numpy()
            else:
                yb = model(xb).cpu().numpy()
            te_preds.append(yb)
    te_pred_s = np.concatenate(te_preds) / target_scale

    te_mse = float(((te_pred_s - y_regr_te) ** 2).mean())
    te_corr = float(np.corrcoef(te_pred_s, y_regr_te)[0, 1]) if te_pred_s.std() > 0 else 0.0

    fee_thr = 2.0 * FEE
    pred_action = np.full(len(te_pred_s), 1, dtype=np.int8)
    pred_action[te_pred_s > fee_thr] = 2
    pred_action[te_pred_s < -fee_thr] = 0
    side = pred_action.astype(np.float64) - 1.0
    diff = mp_th_te.astype(np.float64) - mp_t_te.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th_te + 1.0) + (mp_t_te + 1.0))
    denom = mp_t_te.astype(np.float64) + 1.0
    pnl = (side * diff - fee_pnl) / denom
    cum_pnl = float(pnl.sum())
    n_active = int((pred_action != 1).sum())
    print(f"  TEST k=1: cum_pnl={cum_pnl:+.4f} n_active={n_active:,} corr={te_corr:.4f}", flush=True)

    # Save model
    out_pt = os.path.join(HERE, f"model_T157_seed{seed}.pt")
    torch.save(best_state, out_pt)
    print(f"  saved {out_pt}", flush=True)

    out_npz = os.path.join(HERE, f"model_T157_seed{seed}.npz")
    np.savez(out_npz, **{k: v.numpy() for k, v in best_state.items()})
    print(f"  saved {out_npz}", flush=True)

    # Save norm stats + config
    norm_path = os.path.join(HERE, f"norm_stats_seed{seed}.npz")
    np.savez(norm_path, mean=mu, std=std, keep_idx=keep_idx,
             target_scale=np.float32(target_scale))
    print(f"  saved {norm_path}", flush=True)

    # Save pred parquet (same format as T87/T93)
    sess_map = {0: "am", 1: "pm"}
    te_df = pd.DataFrame({
        "sym": test_full["sym"].astype(np.int8),
        "date": test_full["date"].astype(np.int16),
        "session": np.array([sess_map[int(s)] for s in test_full["sess_idx"]], dtype=object),
        "t": test_full["t"].astype(np.int16),
        "true_label": y_cls_te.astype(np.int8),
        "true_dmid_norm": y_regr_te.astype(np.float32),
        "pred_dmid_norm": te_pred_s.astype(np.float32),
        "midprice_t": mp_t_te.astype(np.float32),
        "midprice_th": mp_th_te.astype(np.float32),
    })
    pred_path = os.path.join(HERE, f"pred_T157_seed{seed}.parquet")
    te_df.to_parquet(pred_path, index=False)
    print(f"  saved {pred_path}", flush=True)

    summary = {
        "seed": seed, "horizon": H, "feat_dim": feat_dim,
        "n_params": int(n_params),
        "best_epoch": best_epoch, "best_val_mse": float(best_val_mse),
        "test": {"mse": float(te_mse), "corr": float(te_corr),
                 "ev_gate_k1_cum_pnl": float(cum_pnl),
                 "ev_gate_k1_n_active": int(n_active)},
        "train_time_sec": float(train_time),
        "epoch_logs": epoch_logs,
        "params": vars(args),
        "target_scale": float(target_scale),
    }
    out_json = os.path.join(HERE, f"summary_T157_seed{seed}.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"  summary -> {out_json}", flush=True)

    if use_wandb:
        try:
            wandb.log({"best_epoch": best_epoch, "best_val_mse": best_val_mse,
                       "test_mse": te_mse, "test_corr": te_corr,
                       "test_ev_gate_k1_cum_pnl": cum_pnl, "train_time_sec": float(train_time)})
            wandb.finish()
        except Exception:
            pass

    progress("done", seed=seed, best_epoch=best_epoch, val_mse=float(best_val_mse),
             test_corr=float(te_corr), ev_gate_k1=float(cum_pnl))

    print(f"\nRESULT: task=T157-transformer-seed{seed} "
          f"metrics={{val_mse={best_val_mse:.5e}, test_corr={te_corr:.4f}, cum_pnl={cum_pnl:+.4f}}} "
          f"notes=best_epoch_{best_epoch}", flush=True)


if __name__ == "__main__":
    main()
