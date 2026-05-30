"""Cross-sym NN training v2 with LGB-pruned features (drop bottom 100, keep 259).

Wraps train_v2.train_one but slices the loaded X_grp by --keep-idx before training.
Saves runs to runs_v2_pruned/<arch>_pruned<dim>_s<seed>/.

CLI: python train_v2_pruned.py --arch sae_mlp --seed 0 --keep-idx keep_idx_259d.npy
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Reuse helpers from train_v2 to avoid duplication
from train_v2 import (
    eval_predictions, eval_test, predict_grouped, make_lr_lambda,
    FEE, CLIP,
)
from data_loader import load_grouped_data
from models import build_model, count_params, SupervisedAutoEncoderMLP


def train_one_pruned(arch, seed, device, keep_idx_npy,
                     batch_size=1024, max_epochs=20, eval_every=200,
                     patience_steps=1500, lr=1e-4, weight_decay=1e-3,
                     warmup_steps=200, grad_clip=1.0,
                     use_wandb=True, wandb_project="liangwenbei-cross-sym",
                     out_root="runs_v2_pruned"):
    t_start = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)

    keep_idx = np.load(keep_idx_npy).astype(np.int64)
    n_keep = int(len(keep_idx))
    print(f"\n=== arch={arch} seed={seed}  PRUNED keep={n_keep} ===", flush=True)
    print(f"  keep_idx_npy={keep_idx_npy}", flush=True)

    print("  loading grouped data ...", flush=True)
    train_data, val_data, test_data, stats = load_grouped_data(level="L8", verbose=True)
    target_scale = stats['target_scale']
    full_feat_names = stats['feat_names']

    # Slice X_grp along feature axis
    train_data['X_grp'] = train_data['X_grp'][:, :, keep_idx].astype(np.float32, copy=False)
    val_data['X_grp']   = val_data['X_grp'][:, :, keep_idx].astype(np.float32, copy=False)
    test_data['X_grp']  = test_data['X_grp'][:, :, keep_idx].astype(np.float32, copy=False)
    kept_names = [full_feat_names[i] for i in keep_idx]
    print(f"  after prune: train X_grp={train_data['X_grp'].shape}  val={val_data['X_grp'].shape}  test={test_data['X_grp'].shape}", flush=True)

    n_feat = n_keep
    X_tr  = train_data['X_grp']
    y_tr  = train_data['y_reg']
    sw_tr = train_data['sw']

    run_name = f"{arch}_pruned{n_keep}_s{seed}"
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
                config=dict(arch=arch, seed=seed, n_feat=n_feat,
                            n_keep=n_keep, keep_idx_file=str(keep_idx_npy),
                            target_scale=target_scale,
                            n_train_groups=int(X_tr.shape[0]),
                            n_val_groups=int(val_data['X_grp'].shape[0]),
                            batch_size=batch_size, max_epochs=max_epochs,
                            eval_every=eval_every, patience_steps=patience_steps,
                            lr=lr, weight_decay=weight_decay,
                            warmup_steps=warmup_steps, grad_clip=grad_clip,
                            train_loop="step_level_v2_pruned"),
                reinit=True, settings=wandb.Settings(start_method="thread"),
            )
        except Exception as e:
            print(f"  wandb init failed: {e}", flush=True)
            use_wandb = False

    model = build_model(arch, n_sym=5, n_feat=n_feat).to(device)
    is_sae = isinstance(model, SupervisedAutoEncoderMLP)
    n_params = count_params(model)
    print(f"  n_params={n_params:,}  is_sae={is_sae}", flush=True)
    if use_wandb:
        import wandb
        wandb.config.update({"n_params": int(n_params), "is_sae": is_sae},
                            allow_val_change=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = math.ceil(X_tr.shape[0] / batch_size)
    total_steps = steps_per_epoch * max_epochs
    lr_lambda = make_lr_lambda(warmup_steps, total_steps, eta_ratio=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    print(f"  steps_per_epoch={steps_per_epoch}  total_steps={total_steps}  eval_every={eval_every}", flush=True)

    Xtr_t  = torch.from_numpy(X_tr)
    ytr_t  = torch.from_numpy((y_tr / target_scale).astype(np.float32))
    swtr_t = torch.from_numpy(sw_tr)
    N = Xtr_t.shape[0]

    best_val_pnl = -1e18
    best_state   = None
    best_step    = -1
    best_val_results = None
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

            if step % eval_every == 0:
                avg_loss = float(np.mean(loss_accum[-eval_every:]))
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
                    best_val_results = val_results
                    no_improve_count = 0
                else:
                    no_improve_count += eval_every
                    if no_improve_count >= patience_steps:
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
        return {"arch": arch, "seed": seed, "n_params": int(n_params),
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
        "arch": arch, "seed": seed, "n_params": int(n_params),
        "n_feat": int(n_feat), "n_keep": int(n_keep),
        "keep_idx_file": str(keep_idx_npy),
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
                          eval_every=eval_every, patience_steps=patience_steps,
                          lr=lr, weight_decay=weight_decay,
                          warmup_steps=warmup_steps, grad_clip=grad_clip),
        "kept_names": kept_names,
    }

    out_dir = HERE / out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)
    np.savez(out_dir / "preds.npz",
             preds_val=preds_val.astype(np.float32),
             preds_test=preds_test.astype(np.float32))
    torch.save(best_state, out_dir / "model.pt")
    print(f"  saved → {out_dir}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", type=str, default="sae_mlp",
                    choices=["isab", "deepsets", "sae_mlp",
                             "cross_mlp", "sym_attn", "hybrid"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cuda", type=str, default="0")
    ap.add_argument("--keep-idx", type=str, required=True,
                    help="Path to keep_idx_<dim>d.npy")
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--patience-steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-project", type=str, default="liangwenbei-cross-sym")
    ap.add_argument("--out-root", type=str, default="runs_v2_pruned")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    result = train_one_pruned(
        arch=args.arch, seed=args.seed, device=device,
        keep_idx_npy=args.keep_idx,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        eval_every=args.eval_every,
        patience_steps=args.patience_steps,
        lr=args.lr, weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps, grad_clip=args.grad_clip,
        use_wandb=not args.no_wandb, wandb_project=args.wandb_project,
        out_root=args.out_root,
    )

    pnl = result.get("test_pnl")
    pnls = f"{pnl:+.4f}" if pnl is not None else "FAILED"
    print(f"\n=== DONE: arch={args.arch} seed={args.seed} n_keep={result.get('n_keep')} "
          f"test_pnl={pnls} best_step={result.get('best_step')} ===", flush=True)


if __name__ == "__main__":
    main()
