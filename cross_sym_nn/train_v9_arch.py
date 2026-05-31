"""V9: 60 features (40 v7-selected + 20 NF v8 new) with D no_aug pipeline.

Identical to train_v7_no_aug.py except uses compute_v9_features() and
selected_features_v9_60.npy + inter_zstats_v9.npz.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from train_v2 import (
    eval_predictions, eval_test, predict_grouped, make_lr_lambda,
)
from data_loader import load_grouped_data
from models_v9 import build_model_v9, count_params
from interaction_features_v8 import compute_v9_features


def _build_inter_features(X_359, sel_idx, mu, sd, batch=200_000):
    out_chunks = []
    n = X_359.shape[0]
    for s in range(0, n, batch):
        Xs = X_359[s:s + batch]
        Xi, _, _ = compute_v9_features(Xs)
        Xi = (Xi - mu) / sd
        Xi = np.clip(Xi, -10.0, 10.0)
        Xi = Xi[:, :, sel_idx]
        out_chunks.append(Xi.astype(np.float32))
    return np.concatenate(out_chunks, axis=0)


def train_v9(seed, device, keep_idx_npy, sel_idx_npy, zstats_npz,
             arch='v6_pairwise', out_tag='v9_features60',
             batch_size=1024, max_epochs=20, eval_every=200,
             patience_steps=1500, lr=1e-4, weight_decay=1e-3,
             warmup_steps=200, grad_clip=1.0, dropout_override=None,
             use_wandb=True, wandb_project='liangwenbei-cross-sym',
             out_root='runs_v9'):
    t_start = time.time()
    torch.manual_seed(seed); np.random.seed(seed)

    keep_idx = np.load(keep_idx_npy).astype(np.int64)
    sel_idx  = np.load(sel_idx_npy).astype(np.int64)
    zst = np.load(zstats_npz, allow_pickle=True)
    mu_inter = zst['mu'].astype(np.float32)
    sd_inter = zst['sd'].astype(np.float32)
    print(f'\n=== [v9_features] arch={arch} seed={seed} keep_259={len(keep_idx)} sel_inter={len(sel_idx)} ===',
          flush=True)
    n_keep = len(keep_idx)
    n_sel  = len(sel_idx)

    print('  loading data ...', flush=True)
    train_data, val_data, test_data, stats = load_grouped_data(level='L8', verbose=True)
    target_scale = stats['target_scale']

    print('  computing v9 interaction features (train aug, val, test) ...', flush=True)
    t1 = time.time()
    X_tr_inter = _build_inter_features(train_data['X_grp'], sel_idx, mu_inter, sd_inter)
    X_va_inter = _build_inter_features(val_data['X_grp'],   sel_idx, mu_inter, sd_inter)
    X_te_inter = _build_inter_features(test_data['X_grp'],  sel_idx, mu_inter, sd_inter)
    print(f'    inter shapes: train={X_tr_inter.shape} val={X_va_inter.shape} test={X_te_inter.shape}  dt={time.time()-t1:.1f}s',
          flush=True)

    def cat_259_inter(X_grp, X_inter):
        return np.concatenate([
            X_grp[:, :, keep_idx].astype(np.float32, copy=False),
            X_inter.astype(np.float32, copy=False),
        ], axis=2)

    X_tr = cat_259_inter(train_data['X_grp'], X_tr_inter)
    X_va = cat_259_inter(val_data['X_grp'],   X_va_inter)
    X_te = cat_259_inter(test_data['X_grp'],  X_te_inter)
    del train_data['X_grp']; del val_data['X_grp']; del test_data['X_grp']
    del X_tr_inter, X_va_inter, X_te_inter

    # D no_aug: discard mirror half + no scale jitter
    N_aug = X_tr.shape[0]
    N_orig = N_aug // 2
    X_tr = X_tr[:N_orig]
    train_data['y_reg'] = train_data['y_reg'][:N_orig]
    train_data['sw']    = train_data['sw'][:N_orig]
    print(f'  [D no_aug] truncated train: {N_aug} → {N_orig} groups', flush=True)

    train_data['X_grp'] = X_tr
    val_data['X_grp']   = X_va
    test_data['X_grp']  = X_te
    n_feat = X_tr.shape[-1]
    print(f'  after concat: n_feat={n_feat} (259+{n_sel})  train={X_tr.shape}', flush=True)

    sw_tr = train_data['sw']
    y_tr  = train_data['y_reg']
    ytr_t = torch.from_numpy((y_tr / target_scale).astype(np.float32))

    run_name = f'{arch}_{out_tag}_s{seed}'
    if use_wandb:
        try:
            import wandb
            os.environ.setdefault('WANDB_API_KEY',
                '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')
            wandb.init(project=wandb_project, entity='jingzheshi', name=run_name,
                       config=dict(arch=arch, seed=seed, n_feat=n_feat,
                                   n_keep_259=n_keep, n_sel_inter=n_sel,
                                   ablation='v9_features60', mirror_aug=False, scale_jitter=False,
                                   n_train_groups=int(N_orig),
                                   target_scale=float(target_scale),
                                   batch_size=batch_size, max_epochs=max_epochs,
                                   lr=lr, weight_decay=weight_decay,
                                   warmup_steps=warmup_steps, grad_clip=grad_clip,
                                   dropout_override=dropout_override,
                                   train_loop='step_level_v9'),
                       reinit=True, settings=wandb.Settings(start_method='thread'))
        except Exception as e:
            print(f'  wandb init failed: {e}', flush=True)
            use_wandb = False

    overrides = {}
    if dropout_override is not None:
        overrides['dropout'] = float(dropout_override)
    model = build_model_v9(arch, n_sym=5, n_feat=n_feat, **overrides).to(device)
    n_params = count_params(model)
    print(f'  n_params={n_params:,}', flush=True)
    if use_wandb:
        import wandb
        wandb.config.update({'n_params': int(n_params)}, allow_val_change=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = math.ceil(X_tr.shape[0] / batch_size)
    total_steps = steps_per_epoch * max_epochs
    lr_lambda = make_lr_lambda(warmup_steps, total_steps, eta_ratio=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    print(f'  steps_per_epoch={steps_per_epoch} total_steps={total_steps} eval_every={eval_every}', flush=True)

    Xtr_t  = torch.from_numpy(X_tr)
    swtr_t = torch.from_numpy(sw_tr)
    N = Xtr_t.shape[0]

    best_val_pnl = -1e18
    best_state = None
    best_step = -1
    val_history = []
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
            wb = swtr_t[idx].to(device, non_blocking=True)
            yb = ytr_t[idx].to(device, non_blocking=True)

            scale = float(1.0)  # D no_aug — no jitter
            preds = model(xb)
            sq = (preds - yb * scale) ** 2
            pred_loss = (sq * wb).sum() / wb.sum().clamp_min(1.0)

            if getattr(model, '_last_recon_loss', None) is not None:
                alpha = model.recon_alpha
                loss = (1.0 - alpha) * pred_loss + alpha * model._last_recon_loss
            else:
                loss = pred_loss

            if getattr(model, '_last_aux_loss', None) is not None:
                aux_scale = getattr(model, 'aux_alpha', 1.0)
                loss = loss + aux_scale * model._last_aux_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step(); sched.step()
            step += 1
            loss_accum.append(float(loss.item()))

            if step % eval_every == 0:
                avg_loss = float(np.mean(loss_accum[-eval_every:]))
                preds_val = predict_grouped(model, val_data['X_grp'], device, batch_size=1024)
                val_results, _ = eval_predictions(preds_val, val_data, target_scale)
                val_pnl = val_results['total_pnl']
                val_ic = val_results['mean_ic']
                cur_lr = opt.param_groups[0]['lr']

                val_history.append({'step': step, 'epoch': ep, 'train_loss': avg_loss,
                                    'val_pnl': val_pnl, 'val_ic': val_ic, 'lr': cur_lr})
                improved = val_pnl > best_val_pnl
                marker = '*BEST*' if improved else ''
                print(f'  step={step:5d} ep={ep:02d} loss={avg_loss:.5f} '
                      f'val_pnl={val_pnl:+.4f} val_ic={val_ic:.5f} lr={cur_lr:.2e} {marker}',
                      flush=True)

                if not np.isfinite(avg_loss):
                    print('  NaN/Inf loss, aborting', flush=True)
                    break_all = True
                    break

                if use_wandb:
                    import wandb
                    log_dict = {'step': step, 'epoch': ep, 'train_loss': avg_loss,
                                'val_pnl': val_pnl, 'val_ic': val_ic, 'lr': cur_lr,
                                'no_improve_count': no_improve_count}
                    for k, v in val_results.items():
                        if k not in ('total_pnl', 'mean_ic'):
                            log_dict[f'val/{k}'] = v
                    wandb.log(log_dict, step=step)

                if improved:
                    best_val_pnl = val_pnl
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    best_step = step
                    no_improve_count = 0
                else:
                    no_improve_count += eval_every
                    if no_improve_count >= patience_steps:
                        print(f'  early stop @ step={step} (best_step={best_step})', flush=True)
                        break_all = True
                        break
                model.train()
        if break_all:
            break

    t_train = time.time() - t_start
    if best_state is None:
        print('  TRAINING FAILED — no best state', flush=True)
        if use_wandb:
            import wandb; wandb.finish()
        return {'arch': arch, 'seed': seed, 'n_params': int(n_params),
                'status': 'failed', 't_train_sec': float(t_train)}

    model.load_state_dict(best_state)
    preds_val  = predict_grouped(model, val_data['X_grp'],  device, batch_size=1024)
    preds_test = predict_grouped(model, test_data['X_grp'], device, batch_size=1024)
    val_results, val_thrs = eval_predictions(preds_val, val_data, target_scale)
    test_results = eval_test(preds_test, test_data, val_thrs, target_scale)

    print(f'  best_step={best_step} val_pnl={val_results["total_pnl"]:+.4f} '
          f'test_pnl={test_results["total_pnl"]:+.4f} t={t_train:.1f}s', flush=True)

    if use_wandb:
        import wandb
        final_log = {'final/val_pnl': val_results['total_pnl'],
                     'final/val_ic': val_results['mean_ic'],
                     'final/test_pnl': test_results['total_pnl'],
                     'final/test_ic': test_results['mean_ic'],
                     'final/best_step': best_step,
                     'final/total_steps': step}
        for k, v in test_results.items():
            final_log[f'test/{k}'] = v
        wandb.log(final_log); wandb.finish()

    result = {
        'arch': arch, 'seed': seed, 'n_params': int(n_params),
        'n_feat': int(n_feat), 'n_keep_259': int(n_keep), 'n_sel_inter': int(n_sel),
        'ablation': 'v9_features60', 'mirror_aug': False, 'scale_jitter': False,
        'n_train_groups': int(N_orig),
        'keep_idx_file': str(keep_idx_npy), 'sel_idx_file': str(sel_idx_npy),
        'best_step': int(best_step), 'total_steps': int(step),
        'status': 'ok',
        'val_pnl': val_results['total_pnl'],
        'test_pnl': test_results['total_pnl'],
        'val_ic': val_results['mean_ic'],
        'test_ic': test_results['mean_ic'],
        'val_sym_pnl':  {f'sym{s}': val_results[f'sym{s}_pnl'] for s in range(5)},
        'test_sym_pnl': {f'sym{s}': test_results[f'sym{s}_pnl'] for s in range(5)},
        'test_sym_ic':  {f'sym{s}': test_results[f'sym{s}_ic'] for s in range(5)},
        'val_thresholds': {str(s): v for s, v in val_thrs.items()},
        'val_history': val_history,
        't_train_sec': float(t_train),
        'total_time_sec': float(time.time() - t_start),
    }
    out_dir = HERE / out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'results.json').write_text(json.dumps(result, indent=2))
    print(f'  saved → {out_dir}', flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', type=str, default='v8_pairformer')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cuda', type=str, default='0')
    ap.add_argument('--keep-idx', type=str, default='keep_idx_259d.npy')
    ap.add_argument('--sel-idx',  type=str, default='selected_features_v9_60.npy')
    ap.add_argument('--zstats',   type=str, default='inter_zstats_v9.npz')
    ap.add_argument('--max-epochs', type=int, default=20)
    ap.add_argument('--eval-every', type=int, default=200)
    ap.add_argument('--patience-steps', type=int, default=1500)
    ap.add_argument('--batch-size', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-3)
    ap.add_argument('--warmup-steps', type=int, default=200)
    ap.add_argument('--grad-clip', type=float, default=1.0)
    ap.add_argument('--dropout', type=float, default=None)
    ap.add_argument('--no-wandb', action='store_true')
    ap.add_argument('--wandb-project', type=str, default='liangwenbei-cross-sym')
    ap.add_argument('--out-root', type=str, default='runs_v9')
    ap.add_argument('--out-tag', type=str, default='v9_arch')
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}', flush=True)

    result = train_v9(
        seed=args.seed, device=device,
        keep_idx_npy=args.keep_idx, sel_idx_npy=args.sel_idx, zstats_npz=args.zstats,
        arch=args.arch, out_tag=args.out_tag,
        batch_size=args.batch_size, max_epochs=args.max_epochs,
        eval_every=args.eval_every, patience_steps=args.patience_steps,
        lr=args.lr, weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps, grad_clip=args.grad_clip,
        dropout_override=args.dropout,
        use_wandb=not args.no_wandb, wandb_project=args.wandb_project,
        out_root=args.out_root,
    )
    pnl = result.get('test_pnl')
    pnls = f'{pnl:+.4f}' if pnl is not None else 'FAILED'
    print(f'\n=== DONE: arch={args.arch} seed={args.seed} test_pnl={pnls} '
          f'best_step={result.get("best_step")} ===', flush=True)


if __name__ == '__main__':
    main()
