"""Cross-sym NN training script.

CLI: python train.py --arch {cross_mlp,sym_attn,hybrid} --seed N [--epochs N] [--patience N]
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent

sys.path.insert(0, str(HERE))
from data_loader import load_grouped_data
from models import build_model, count_params

FEE  = 1e-4
CLIP = 10.0


# ---------------------------------------------------------------------------
# Eval helpers (per-sym threshold search, matches baseline compute_pnl protocol)
# ---------------------------------------------------------------------------

def compute_pnl(y_pred, mp_t, mp_th, thr_up, thr_dn, fee=FEE):
    a = np.zeros_like(y_pred)
    a[y_pred >  thr_up] = 1.0
    a[y_pred < -thr_dn] = -1.0
    raw  = a * (mp_th - mp_t)
    cost = fee * np.abs(a) * ((mp_th + 1) + (mp_t + 1))
    pnl  = (raw - cost) / (mp_t + 1.0)
    return float(pnl.sum()), int((a != 0).sum())


def search_thr(y_pred, mp_t, mp_th, fee=FEE):
    mean_abs = max(float(np.mean(np.abs(y_pred))), 1e-8)
    best = (-1e18, None, None)
    for k in np.linspace(0.5, 3.0, 26):
        thr = k * mean_abs
        s, _ = compute_pnl(y_pred, mp_t, mp_th, thr, thr, fee)
        if s > best[0]:
            best = (s, float(thr), float(k))
    return best  # (pnl, thr, k)


def pearson_ic(y_pred, y_true):
    if len(y_pred) < 2:
        return 0.0
    p = np.corrcoef(y_pred.astype(np.float64), y_true.astype(np.float64))[0, 1]
    return float(p) if np.isfinite(p) else 0.0


def eval_predictions(preds_grp, split_data, target_scale):
    """Evaluate grouped predictions (N_groups, 5).

    Returns dict with per-sym + total PnL/IC, and per-sym thresholds.
    Preds are in normalized units (/ target_scale); we de-normalize before eval.
    """
    n_sym = preds_grp.shape[1]
    mp_t  = split_data['mp_t']   # (N_groups, 5)
    mp_th = split_data['mp_th']  # (N_groups, 5)
    y_reg = split_data['y_reg']  # (N_groups, 5)

    preds_raw = preds_grp * target_scale  # de-normalize

    total_pnl  = 0.0
    total_ic   = 0.0
    sym_results = {}
    thresholds  = {}

    for s in range(n_sym):
        yp = preds_raw[:, s]
        yt = y_reg[:, s]
        mpt  = mp_t[:, s]
        mpth = mp_th[:, s]

        pnl, thr, k = search_thr(yp, mpt, mpth)
        ic = pearson_ic(yp, yt)
        total_pnl += pnl
        total_ic  += ic
        sym_results[f'sym{s}_pnl'] = pnl
        sym_results[f'sym{s}_ic']  = ic
        thresholds[s] = thr

    sym_results['total_pnl'] = total_pnl
    sym_results['mean_ic']   = total_ic / n_sym
    return sym_results, thresholds


def eval_test(preds_grp, test_data, thresholds, target_scale):
    """Apply fixed thresholds from val to test."""
    n_sym  = preds_grp.shape[1]
    mp_t   = test_data['mp_t']
    mp_th  = test_data['mp_th']
    y_reg  = test_data['y_reg']
    preds_raw = preds_grp * target_scale

    total_pnl = 0.0
    total_ic  = 0.0
    total_nact = 0
    sym_results = {}

    for s in range(n_sym):
        yp  = preds_raw[:, s]
        yt  = y_reg[:, s]
        mpt  = mp_t[:, s]
        mpth = mp_th[:, s]
        thr  = thresholds[s]
        pnl, nact = compute_pnl(yp, mpt, mpth, thr, thr)
        ic = pearson_ic(yp, yt)
        total_pnl += pnl
        total_ic  += ic
        total_nact += nact
        sym_results[f'sym{s}_pnl'] = pnl
        sym_results[f'sym{s}_ic']  = ic
        sym_results[f'sym{s}_nact'] = nact

    sym_results['total_pnl'] = total_pnl
    sym_results['mean_ic']   = total_ic / n_sym
    sym_results['total_nact'] = total_nact
    return sym_results


# ---------------------------------------------------------------------------
# Predict grouped
# ---------------------------------------------------------------------------

def predict_grouped(model, X_grp, device, batch_size=512):
    """Run model on grouped data. Returns (N_groups, 5) float32."""
    model.eval()
    N = X_grp.shape[0]
    preds_list = []
    with torch.no_grad():
        for s in range(0, N, batch_size):
            xb = torch.from_numpy(X_grp[s:s+batch_size]).to(device, non_blocking=True)
            out = model(xb).detach().cpu().numpy()  # (bsz, 5)
            preds_list.append(out)
    return np.concatenate(preds_list, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one(arch, seed, device, batch_size=256, max_ep=50, patience=10,
              use_wandb=True, wandb_project="liangwenbei-cross-sym"):
    """Train one arch × seed. Returns results dict."""
    t_start = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n=== arch={arch} seed={seed} ===", flush=True)

    # Load data
    print("  loading grouped data ...", flush=True)
    train_data, val_data, test_data, stats = load_grouped_data(level="L8", verbose=True)
    target_scale = stats['target_scale']
    n_feat = stats['keep_idx'].shape[0]

    X_tr  = train_data['X_grp']   # (2*N_tr, 5, 359)
    y_tr  = train_data['y_reg']   # (2*N_tr, 5)
    sw_tr = train_data['sw']      # (2*N_tr, 5)

    # WandB init
    run_name = f"{arch}_s{seed}"
    if use_wandb:
        try:
            import wandb
            os.environ.setdefault(
                "WANDB_API_KEY",
                "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",
            )
            wandb.init(
                project=wandb_project,
                entity="jingzheshi",
                name=run_name,
                config=dict(arch=arch, seed=seed, n_feat=n_feat,
                            target_scale=target_scale,
                            n_train_groups=int(X_tr.shape[0]),
                            n_val_groups=int(val_data['X_grp'].shape[0]),
                            batch_size=batch_size, max_ep=max_ep, patience=patience),
                reinit=True, settings=wandb.Settings(start_method="thread"),
            )
        except Exception as e:
            print(f"  wandb init failed: {e}", flush=True)
            use_wandb = False

    # Build model
    model = build_model(arch, n_sym=5, n_feat=n_feat).to(device)
    n_params = count_params(model)
    print(f"  n_params={n_params:,}", flush=True)
    if use_wandb:
        import wandb
        wandb.config.update({"n_params": int(n_params)}, allow_val_change=True)

    opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep, eta_min=1e-5)

    # Pre-convert to tensors (keep on CPU, transfer batch by batch)
    Xtr_t  = torch.from_numpy(X_tr)
    ytr_t  = torch.from_numpy((y_tr / target_scale).astype(np.float32))
    swtr_t = torch.from_numpy(sw_tr)
    N = Xtr_t.shape[0]

    best_val_pnl = -1e18
    best_state   = None
    best_ep      = -1
    pat_left     = patience
    t0 = time.time()

    for ep in range(max_ep):
        model.train()
        perm   = torch.randperm(N)
        losses = []
        for s in range(0, N, batch_size):
            idx  = perm[s:s + batch_size]
            xb   = Xtr_t[idx].to(device, non_blocking=True)   # (bsz, 5, F)
            yb   = ytr_t[idx].to(device, non_blocking=True)   # (bsz, 5)
            wb   = swtr_t[idx].to(device, non_blocking=True)  # (bsz, 5)

            scale = float(np.random.uniform(0.80, 1.20))
            preds = model(xb)                                  # (bsz, 5)
            sq    = (preds - yb * scale) ** 2                  # (bsz, 5)
            loss  = (sq * wb).sum() / wb.sum().clamp_min(1.0)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.item()))

        sched.step()

        # Val eval
        preds_val = predict_grouped(model, val_data['X_grp'], device, batch_size=512)
        val_results, val_thrs = eval_predictions(preds_val, val_data, target_scale)
        val_pnl  = val_results['total_pnl']
        val_ic   = val_results['mean_ic']
        avg_loss = float(np.mean(losses))

        print(f"  ep{ep:02d} loss={avg_loss:.5f} val_pnl={val_pnl:+.4f} val_ic={val_ic:.5f}", flush=True)

        if not np.isfinite(avg_loss):
            print(f"  NaN/Inf loss at ep{ep}, abort", flush=True)
            break

        if use_wandb:
            import wandb
            log_dict = {"epoch": ep, "train_loss": avg_loss,
                        "val_pnl": val_pnl, "val_ic": val_ic}
            for k, v in val_results.items():
                if k not in ("total_pnl", "mean_ic"):
                    log_dict[f"val/{k}"] = v
            wandb.log(log_dict)

        if val_pnl > best_val_pnl:
            best_val_pnl = val_pnl
            best_state   = {k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()}
            best_ep   = ep
            pat_left  = patience
        else:
            pat_left -= 1
            if pat_left <= 0:
                print(f"  early stop at ep {ep} (best_ep={best_ep})", flush=True)
                break

    t_train = time.time() - t0

    if best_state is None:
        print("  TRAINING FAILED", flush=True)
        if use_wandb:
            import wandb; wandb.finish()
        return {"arch": arch, "seed": seed, "n_params": int(n_params),
                "status": "failed", "t_train_sec": float(t_train)}

    # Load best model, final eval
    model.load_state_dict(best_state)
    preds_val  = predict_grouped(model, val_data['X_grp'],  device, batch_size=512)
    preds_test = predict_grouped(model, test_data['X_grp'], device, batch_size=512)

    val_results, val_thrs  = eval_predictions(preds_val,  val_data,  target_scale)
    test_results           = eval_test(preds_test, test_data, val_thrs, target_scale)

    print(f"  best_ep={best_ep} val_pnl={val_results['total_pnl']:+.4f} "
          f"test_pnl={test_results['total_pnl']:+.4f}  t={t_train:.1f}s", flush=True)

    if use_wandb:
        import wandb
        final_log = {"final/val_pnl": val_results['total_pnl'],
                     "final/val_ic":  val_results['mean_ic'],
                     "final/test_pnl": test_results['total_pnl'],
                     "final/test_ic":  test_results['mean_ic'],
                     "final/best_ep":  best_ep}
        for k, v in test_results.items():
            final_log[f"test/{k}"] = v
        wandb.log(final_log)
        wandb.finish()

    result = {
        "arch": arch, "seed": seed, "n_params": int(n_params), "n_feat": int(n_feat),
        "best_ep": int(best_ep), "status": "ok",
        "val_pnl":  val_results['total_pnl'],
        "test_pnl": test_results['total_pnl'],
        "val_ic":   val_results['mean_ic'],
        "test_ic":  test_results['mean_ic'],
        "val_sym_pnl":  {f"sym{s}": val_results[f"sym{s}_pnl"]  for s in range(5)},
        "test_sym_pnl": {f"sym{s}": test_results[f"sym{s}_pnl"] for s in range(5)},
        "test_sym_ic":  {f"sym{s}": test_results[f"sym{s}_ic"]  for s in range(5)},
        "val_thresholds": {str(s): v for s, v in val_thrs.items()},
        "t_train_sec":  float(t_train),
        "total_time_sec": float(time.time() - t_start),
    }

    out_dir = HERE / "runs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)

    np.savez(out_dir / "preds.npz",
             preds_val=preds_val.astype(np.float32),
             preds_test=preds_test.astype(np.float32))

    # Save model
    torch.save(best_state, out_dir / "model.pt")
    print(f"  saved → {out_dir}", flush=True)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch",     type=str, default="cross_mlp",
                    choices=["cross_mlp", "sym_attn", "hybrid"])
    ap.add_argument("--seed",     type=int, default=0)
    ap.add_argument("--cuda",     type=str, default="0")
    ap.add_argument("--epochs",   type=int, default=50)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-project", type=str, default="liangwenbei-cross-sym")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    result = train_one(
        arch=args.arch, seed=args.seed, device=device,
        batch_size=args.batch_size, max_ep=args.epochs, patience=args.patience,
        use_wandb=not args.no_wandb, wandb_project=args.wandb_project,
    )

    pnl  = result.get("test_pnl")
    pnls = f"{pnl:+.4f}" if pnl is not None else "FAILED"
    print(f"\n=== DONE: arch={args.arch} seed={args.seed} "
          f"test_pnl={pnls} ===", flush=True)


if __name__ == "__main__":
    main()
