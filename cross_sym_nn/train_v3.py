"""Cross-sym NN training v3: HP sweep across 4 configs × 2 arch × 3 seed.

Configs:
  A: v1-style HP + step early stop (light reg, fast eval)
  B: middle ground (medium reg, medium patience)
  C: light reg + step early stop (similar to A but slightly more reg)
  D: v2 strong reg but with larger model

Adds trajectory.json output per run + arch size overrides via CLI.
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
from models import build_model, count_params, SupervisedAutoEncoderMLP

FEE  = 1e-4
CLIP = 10.0


# ---------------------------------------------------------------------------
# Eval helpers (identical to train_v2.py)
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
    return best


def pearson_ic(y_pred, y_true):
    if len(y_pred) < 2:
        return 0.0
    p = np.corrcoef(y_pred.astype(np.float64), y_true.astype(np.float64))[0, 1]
    return float(p) if np.isfinite(p) else 0.0


def eval_predictions(preds_grp, split_data, target_scale):
    n_sym = preds_grp.shape[1]
    mp_t  = split_data['mp_t']
    mp_th = split_data['mp_th']
    y_reg = split_data['y_reg']
    preds_raw = preds_grp * target_scale

    total_pnl = 0.0; total_ic = 0.0
    sym_results = {}; thresholds = {}
    for s in range(n_sym):
        yp = preds_raw[:, s]; yt = y_reg[:, s]
        mpt = mp_t[:, s]; mpth = mp_th[:, s]
        pnl, thr, k = search_thr(yp, mpt, mpth)
        ic = pearson_ic(yp, yt)
        total_pnl += pnl; total_ic += ic
        sym_results[f'sym{s}_pnl'] = pnl
        sym_results[f'sym{s}_ic']  = ic
        thresholds[s] = thr
    sym_results['total_pnl'] = total_pnl
    sym_results['mean_ic']   = total_ic / n_sym
    return sym_results, thresholds


def eval_test(preds_grp, test_data, thresholds, target_scale):
    n_sym = preds_grp.shape[1]
    mp_t  = test_data['mp_t']
    mp_th = test_data['mp_th']
    y_reg = test_data['y_reg']
    preds_raw = preds_grp * target_scale

    total_pnl = 0.0; total_ic = 0.0; total_nact = 0
    sym_results = {}
    for s in range(n_sym):
        yp = preds_raw[:, s]; yt = y_reg[:, s]
        mpt = mp_t[:, s]; mpth = mp_th[:, s]; thr = thresholds[s]
        pnl, nact = compute_pnl(yp, mpt, mpth, thr, thr)
        ic = pearson_ic(yp, yt)
        total_pnl += pnl; total_ic += ic; total_nact += nact
        sym_results[f'sym{s}_pnl']  = pnl
        sym_results[f'sym{s}_ic']   = ic
        sym_results[f'sym{s}_nact'] = nact
    sym_results['total_pnl']  = total_pnl
    sym_results['mean_ic']    = total_ic / n_sym
    sym_results['total_nact'] = total_nact
    return sym_results


def predict_grouped(model, X_grp, device, batch_size=1024):
    model.eval()
    N = X_grp.shape[0]
    preds_list = []
    with torch.no_grad():
        for s in range(0, N, batch_size):
            xb = torch.from_numpy(X_grp[s:s+batch_size]).to(device, non_blocking=True)
            out = model(xb).detach().cpu().numpy()
            preds_list.append(out)
    return np.concatenate(preds_list, axis=0).astype(np.float32)


def make_lr_lambda(warmup_steps, total_steps, eta_ratio=0.05):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, progress)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_ratio + (1.0 - eta_ratio) * cos
    return lr_lambda


# ---------------------------------------------------------------------------
# Config presets — v3 sweet-spot HP sweep
# ---------------------------------------------------------------------------

CONFIGS = {
    "A": dict(d_model=128, d_hidden=256, dropout=0.10, weight_decay=1e-4,
              lr=3e-4, warmup_steps=100, eval_every=50, patience_steps=800,
              label="v1+step_early_stop"),
    "B": dict(d_model=96, d_hidden=192, dropout=0.20, weight_decay=5e-4,
              lr=2e-4, warmup_steps=200, eval_every=100, patience_steps=1500,
              label="middle_ground"),
    "C": dict(d_model=96, d_hidden=192, dropout=0.15, weight_decay=3e-4,
              lr=3e-4, warmup_steps=100, eval_every=50, patience_steps=800,
              label="weak_reg_step_early"),
    "D": dict(d_model=128, d_hidden=256, dropout=0.30, weight_decay=1e-3,
              lr=1e-4, warmup_steps=200, eval_every=200, patience_steps=1500,
              label="v2_reg_large_model"),
}


def arch_overrides(arch, d_model, d_hidden, dropout):
    """Apply HP table d_model/d_hidden columns to arch-specific kwargs."""
    if arch == "isab":
        return dict(d_model=d_model, n_heads=4, n_inducing=4, depth=2, dropout=dropout)
    elif arch == "sae_mlp":
        # scale d_latent with d_hidden roughly (keep ratio ~1:4)
        d_latent = max(32, d_hidden // 4)
        return dict(d_latent=d_latent, d_hidden=d_hidden, dropout=dropout,
                    recon_alpha=0.3, noise_sigma=0.035)
    elif arch == "deepsets":
        d_phi = max(32, d_hidden // 4)
        return dict(d_phi=d_phi, d_hidden=d_hidden, d_rho=d_phi, dropout=dropout)
    elif arch == "sym_attn":
        return dict(d_token=d_model, n_heads=4, depth=2, dropout=dropout)
    elif arch == "hybrid":
        return dict(d_enc=d_model, n_heads=4, dropout=dropout)
    else:
        return dict(dropout=dropout)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one(arch, seed, config_key, device,
              batch_size=1024,
              max_epochs=20,
              grad_clip=1.0,
              use_wandb=True,
              wandb_project="liangwenbei-cross-sym"):
    t_start = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = CONFIGS[config_key]
    print(f"\n=== arch={arch} seed={seed} cfg={config_key} ({cfg['label']}) ===", flush=True)
    print(f"  HP: d_model={cfg['d_model']} d_hidden={cfg['d_hidden']} "
          f"dropout={cfg['dropout']} wd={cfg['weight_decay']:g} lr={cfg['lr']:g} "
          f"warmup={cfg['warmup_steps']} eval_every={cfg['eval_every']} "
          f"patience={cfg['patience_steps']}", flush=True)

    train_data, val_data, test_data, stats = load_grouped_data(level="L8", verbose=False)
    target_scale = stats['target_scale']
    n_feat = stats['keep_idx'].shape[0]

    X_tr  = train_data['X_grp']
    y_tr  = train_data['y_reg']
    sw_tr = train_data['sw']

    overrides = arch_overrides(arch, cfg['d_model'], cfg['d_hidden'], cfg['dropout'])
    model = build_model(arch, n_sym=5, n_feat=n_feat, **overrides).to(device)
    is_sae = isinstance(model, SupervisedAutoEncoderMLP)
    n_params = count_params(model)
    print(f"  n_params={n_params:,}  is_sae={is_sae}", flush=True)

    run_name = f"{arch}_{config_key}_s{seed}"
    if use_wandb:
        try:
            import wandb
            os.environ.setdefault(
                "WANDB_API_KEY",
                "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",
            )
            wandb.init(
                project=wandb_project, entity="jingzheshi",
                name=run_name,
                config=dict(arch=arch, seed=seed, config_key=config_key,
                            label=cfg['label'], n_feat=n_feat,
                            target_scale=target_scale,
                            n_train_groups=int(X_tr.shape[0]),
                            n_val_groups=int(val_data['X_grp'].shape[0]),
                            batch_size=batch_size, max_epochs=max_epochs,
                            n_params=int(n_params), is_sae=is_sae,
                            train_loop="step_level_v3",
                            **{k: v for k, v in cfg.items() if k != "label"}),
                reinit=True, settings=wandb.Settings(start_method="thread"),
            )
        except Exception as e:
            print(f"  wandb init failed: {e}", flush=True)
            use_wandb = False

    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'],
                            weight_decay=cfg['weight_decay'])
    steps_per_epoch = math.ceil(X_tr.shape[0] / batch_size)
    total_steps = steps_per_epoch * max_epochs
    lr_lambda = make_lr_lambda(cfg['warmup_steps'], total_steps, eta_ratio=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    print(f"  steps_per_epoch={steps_per_epoch}  total_steps={total_steps}  "
          f"eval_every={cfg['eval_every']}", flush=True)

    Xtr_t  = torch.from_numpy(X_tr)
    ytr_t  = torch.from_numpy((y_tr / target_scale).astype(np.float32))
    swtr_t = torch.from_numpy(sw_tr)
    N = Xtr_t.shape[0]

    best_val_pnl = -1e18
    best_state   = None
    best_step    = -1
    best_step_val_results = None
    val_history  = []
    no_improve_count = 0
    step = 0
    break_all = False

    loss_accum = []

    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(N)
        for s_idx in range(0, N, batch_size):
            idx = perm[s_idx:s_idx + batch_size]
            xb = Xtr_t[idx].to(device, non_blocking=True)
            yb = ytr_t[idx].to(device, non_blocking=True)
            wb = swtr_t[idx].to(device, non_blocking=True)

            scale = float(np.random.uniform(0.80, 1.20))
            preds = model(xb)
            sq = (preds - yb * scale) ** 2
            pred_loss = (sq * wb).sum() / wb.sum().clamp_min(1.0)

            if is_sae and model._last_recon_loss is not None:
                alpha = model.recon_alpha
                loss = (1.0 - alpha) * pred_loss + alpha * model._last_recon_loss
            else:
                loss = pred_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            sched.step()
            step += 1
            loss_accum.append(float(loss.item()))

            if step % cfg['eval_every'] == 0:
                avg_loss = float(np.mean(loss_accum[-cfg['eval_every']:]))
                preds_val = predict_grouped(model, val_data['X_grp'], device,
                                            batch_size=1024)
                val_results, _ = eval_predictions(preds_val, val_data, target_scale)
                val_pnl = val_results['total_pnl']
                val_ic  = val_results['mean_ic']
                cur_lr  = opt.param_groups[0]['lr']

                val_history.append({
                    "step": step, "epoch": ep, "train_loss": avg_loss,
                    "val_pnl": val_pnl, "val_ic": val_ic, "lr": cur_lr,
                })
                improved = val_pnl > best_val_pnl
                marker = "*BEST*" if improved else ""
                print(f"  step={step:5d} ep={ep:02d} loss={avg_loss:.5f} "
                      f"val_pnl={val_pnl:+.4f} val_ic={val_ic:.5f} lr={cur_lr:.2e} {marker}",
                      flush=True)

                if not np.isfinite(avg_loss):
                    print("  NaN/Inf loss, aborting", flush=True)
                    break_all = True
                    break

                if use_wandb:
                    import wandb
                    log_dict = {"step": step, "epoch": ep,
                                "train_loss": avg_loss,
                                "val_pnl": val_pnl, "val_ic": val_ic,
                                "lr": cur_lr,
                                "no_improve_count": no_improve_count}
                    for k, v in val_results.items():
                        if k not in ("total_pnl", "mean_ic"):
                            log_dict[f"val/{k}"] = v
                    wandb.log(log_dict, step=step)

                if improved:
                    best_val_pnl = val_pnl
                    best_state   = {k: v.detach().cpu().clone()
                                    for k, v in model.state_dict().items()}
                    best_step    = step
                    best_step_val_results = val_results
                    no_improve_count = 0
                else:
                    no_improve_count += cfg['eval_every']
                    if no_improve_count >= cfg['patience_steps']:
                        print(f"  early stop @ step={step} "
                              f"(best_step={best_step}, no_improve={no_improve_count})",
                              flush=True)
                        break_all = True
                        break

                model.train()
        if break_all:
            break

    t_train = time.time() - t_start

    if best_state is None:
        print("  TRAINING FAILED — no best state", flush=True)
        if use_wandb:
            import wandb; wandb.finish()
        return {"arch": arch, "seed": seed, "config_key": config_key,
                "n_params": int(n_params),
                "status": "failed", "t_train_sec": float(t_train)}

    model.load_state_dict(best_state)
    preds_val  = predict_grouped(model, val_data['X_grp'],  device, batch_size=1024)
    preds_test = predict_grouped(model, test_data['X_grp'], device, batch_size=1024)

    val_results, val_thrs = eval_predictions(preds_val, val_data, target_scale)
    test_results = eval_test(preds_test, test_data, val_thrs, target_scale)

    final_val_pnl = val_history[-1]['val_pnl'] if val_history else None

    print(f"  best_step={best_step} val_pnl@best={val_results['total_pnl']:+.4f} "
          f"final_val_pnl={final_val_pnl} test_pnl={test_results['total_pnl']:+.4f} "
          f"t={t_train:.1f}s", flush=True)

    if use_wandb:
        import wandb
        final_log = {"final/val_pnl": val_results['total_pnl'],
                     "final/val_ic":  val_results['mean_ic'],
                     "final/test_pnl": test_results['total_pnl'],
                     "final/test_ic":  test_results['mean_ic'],
                     "final/best_step": best_step,
                     "final/final_step_val_pnl": final_val_pnl,
                     "final/total_steps": step}
        for k, v in test_results.items():
            final_log[f"test/{k}"] = v
        wandb.log(final_log)
        wandb.finish()

    result = {
        "arch": arch, "seed": seed, "config_key": config_key,
        "config_label": cfg['label'],
        "n_params": int(n_params), "n_feat": int(n_feat),
        "best_step": int(best_step), "total_steps": int(step),
        "final_step_val_pnl": final_val_pnl,
        "status": "ok",
        "val_pnl":  val_results['total_pnl'],
        "test_pnl": test_results['total_pnl'],
        "val_ic":   val_results['mean_ic'],
        "test_ic":  test_results['mean_ic'],
        "val_sym_pnl":  {f"sym{s}": val_results[f"sym{s}_pnl"]  for s in range(5)},
        "test_sym_pnl": {f"sym{s}": test_results[f"sym{s}_pnl"] for s in range(5)},
        "test_sym_ic":  {f"sym{s}": test_results[f"sym{s}_ic"]  for s in range(5)},
        "val_thresholds": {str(s): v for s, v in val_thrs.items()},
        "val_history":  val_history,
        "t_train_sec": float(t_train),
        "total_time_sec": float(time.time() - t_start),
        "train_cfg": dict(batch_size=batch_size, max_epochs=max_epochs,
                          grad_clip=grad_clip, **{k: v for k, v in cfg.items()}),
        "arch_overrides": overrides,
    }

    out_dir = HERE / "runs_v3" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(out_dir / "trajectory.json", "w") as f:
        json.dump({"val_history": val_history,
                   "best_step": best_step,
                   "config_key": config_key,
                   "arch": arch, "seed": seed}, f, indent=2)
    np.savez(out_dir / "preds.npz",
             preds_val=preds_val.astype(np.float32),
             preds_test=preds_test.astype(np.float32))
    torch.save(best_state, out_dir / "model.pt")
    print(f"  saved → {out_dir}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", type=str, default="isab",
                    choices=["isab", "deepsets", "sae_mlp",
                             "cross_mlp", "sym_attn", "hybrid"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", type=str, default="A", choices=list(CONFIGS.keys()))
    ap.add_argument("--cuda", type=str, default="0")
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-project", type=str, default="liangwenbei-cross-sym")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    result = train_one(
        arch=args.arch, seed=args.seed, config_key=args.config, device=device,
        batch_size=args.batch_size, max_epochs=args.max_epochs,
        grad_clip=args.grad_clip,
        use_wandb=not args.no_wandb, wandb_project=args.wandb_project,
    )

    pnl = result.get("test_pnl")
    pnls = f"{pnl:+.4f}" if pnl is not None else "FAILED"
    print(f"\n=== DONE: arch={args.arch} cfg={args.config} seed={args.seed} "
          f"test_pnl={pnls} best_step={result.get('best_step')} ===", flush=True)


if __name__ == "__main__":
    main()
