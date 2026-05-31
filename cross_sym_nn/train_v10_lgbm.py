"""
train_v10_lgbm.py — Strict train/val/test LGBM with optional interaction features.

Configs:
  A: 259d pruned baseline (LGB-pruned from 359d L8)
  B: 259d + 40 v7 interaction features (selected_features.npy)
  C: 259d + 60 v9 interaction features (selected_features_v9_60.npy)  ← same as NN SOTA
  D: 359d full + 60 v9 interaction features

Strict split:
  train: date 0-79   (schemeP_train.npz)
  val  : date 80-95  (schemeP_val.npz)   — early-stop + threshold search
  test : date 96-119 (schemeP_test.npz)  — frozen, zero-tune

Usage:
  python3 train_v10_lgbm.py --config A --seed 0 --out-root runs_v10_lgbm --out-tag lgbm_A
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CACHE = Path('/root/projects/liangwenbei_workdir/ablation_runs/cache_log1p')
PARTITION = Path('/root/projects/liangwenbei_workdir/ablation_runs/feat_family_progressive/partition.json')
FEAT_NAMES_FILE = CACHE / 'schemeP_feat_names.txt'

FEE = 1e-4
H = 60
CLIP = 10.0

LOG1P_PREFIXES = (
    "bid", "ask", "bsize", "asize", "amount_delta", "volume_delta",
    "avgbid", "avgask", "totalbsize", "totalasize", "midprice", "spread", "cumspread",
)

HP_CONFIG = dict(
    feature_fraction=0.8, bagging_fraction=0.8,
    num_leaves=127, lambda_l2=1.0,
)


# ---------------------------------------------------------------------------
# PnL evaluation (identical protocol to train_v2.py)
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


def eval_on_val(preds_grp, mp_t_grp, mp_th_grp, y_reg_grp):
    """Search threshold per sym on val; return total_pnl, thresholds dict."""
    n_sym = preds_grp.shape[1]
    total_pnl = 0.0
    total_ic  = 0.0
    thresholds = {}
    sym_results = {}
    for s in range(n_sym):
        yp   = preds_grp[:, s]
        mpt  = mp_t_grp[:, s]
        mpth = mp_th_grp[:, s]
        yt   = y_reg_grp[:, s]
        pnl, thr, k = search_thr(yp, mpt, mpth)
        ic = pearson_ic(yp, yt)
        total_pnl += pnl
        total_ic  += ic
        thresholds[s]  = thr
        sym_results[f'sym{s}_pnl'] = pnl
        sym_results[f'sym{s}_ic']  = ic
    sym_results['total_pnl'] = total_pnl
    sym_results['mean_ic']   = total_ic / n_sym
    return sym_results, thresholds


def eval_on_test(preds_grp, mp_t_grp, mp_th_grp, y_reg_grp, thresholds):
    """Apply fixed thresholds from val to test; no search."""
    n_sym = preds_grp.shape[1]
    total_pnl = 0.0
    total_ic  = 0.0
    total_nact = 0
    sym_results = {}
    for s in range(n_sym):
        yp   = preds_grp[:, s]
        mpt  = mp_t_grp[:, s]
        mpth = mp_th_grp[:, s]
        yt   = y_reg_grp[:, s]
        thr  = thresholds[s]
        pnl, nact = compute_pnl(yp, mpt, mpth, thr, thr)
        ic = pearson_ic(yp, yt)
        total_pnl  += pnl
        total_ic   += ic
        total_nact += nact
        sym_results[f'sym{s}_pnl']  = pnl
        sym_results[f'sym{s}_ic']   = ic
        sym_results[f'sym{s}_nact'] = nact
    sym_results['total_pnl']  = total_pnl
    sym_results['mean_ic']    = total_ic / n_sym
    sym_results['total_nact'] = total_nact
    return sym_results


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def get_L8_keep_idx():
    p = json.loads(PARTITION.read_text())
    return np.array(p['levels']['L8'], dtype=np.int64)  # (359,)


def get_feat_names(keep_idx):
    all_names = FEAT_NAMES_FILE.read_text().splitlines()
    return [all_names[i] for i in keep_idx]


def apply_log1p(X_flat: np.ndarray, feat_names: list[str]) -> np.ndarray:
    magn_idx = np.array(
        [i for i, n in enumerate(feat_names)
         if any(n.startswith(p) for p in LOG1P_PREFIXES)],
        dtype=np.int64,
    )
    if len(magn_idx) > 0:
        v = X_flat[:, magn_idx]
        X_flat[:, magn_idx] = np.sign(v) * np.log1p(np.abs(v))
    return X_flat


def load_split_grouped(split: str, keep_idx_L8: np.ndarray, feat_names_L8: list[str]):
    """Load one split, sort, group → (N_grp, 5, 359)."""
    d = np.load(CACHE / f'schemeP_{split}.npz')
    X_raw  = d['X'][:, keep_idx_L8].astype(np.float32, copy=False)
    mp_t   = d['mp_t'].astype(np.float64)
    mp_th  = d[f'mp_t{H}'].astype(np.float64)
    y_cls  = d[f'y{H}'].astype(np.int64)
    date   = d['date'].astype(np.int32)
    sess   = d['sess_idx'].astype(np.int32)
    t_     = d['t'].astype(np.int32)
    sym    = d['sym'].astype(np.int32)
    d.close()

    order  = np.lexsort((sym, t_, sess, date))
    X_raw  = X_raw[order]
    mp_t   = mp_t[order]
    mp_th  = mp_th[order]
    y_cls  = y_cls[order]

    # apply log1p (in-place on flat)
    apply_log1p(X_raw, feat_names_L8)

    n   = len(mp_t)
    assert n % 5 == 0
    N   = n // 5
    n_f = X_raw.shape[1]

    X_grp    = X_raw.reshape(N, 5, n_f)
    mp_t_grp = mp_t.reshape(N, 5)
    mp_th_grp = mp_th.reshape(N, 5)
    y_cls_grp = y_cls.reshape(N, 5)
    y_reg_grp = ((mp_th_grp - mp_t_grp) / (mp_t_grp + 1.0)).astype(np.float32)

    return X_grp, y_reg_grp, y_cls_grp, mp_t_grp, mp_th_grp


def zscore_groups(X_grp: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    N, S, F = X_grp.shape
    flat = X_grp.reshape(-1, F).astype(np.float32, copy=True)
    flat = (flat - mu) / sd
    np.nan_to_num(flat, copy=False, nan=0.0, posinf=CLIP, neginf=-CLIP)
    np.clip(flat, -CLIP, CLIP, out=flat)
    return flat.reshape(N, S, F)


def class_balanced_weight(y_flat: np.ndarray, num_class=3) -> np.ndarray:
    counts = np.bincount(y_flat, minlength=num_class).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    cw = len(y_flat) / (num_class * counts)
    return cw.astype(np.float32)


# ---------------------------------------------------------------------------
# Interaction feature helpers
# ---------------------------------------------------------------------------

def _build_inter_v7(X_grp_359z: np.ndarray, sel_idx: np.ndarray,
                    mu_inter: np.ndarray, sd_inter: np.ndarray,
                    batch: int = 100_000) -> np.ndarray:
    from interaction_features import compute_top50_features
    chunks = []
    N = X_grp_359z.shape[0]
    for s in range(0, N, batch):
        Xs = X_grp_359z[s:s + batch]
        Xi, _ = compute_top50_features(Xs)                   # (B, 5, 116)
        Xi = ((Xi - mu_inter) / np.maximum(sd_inter, 1e-6)).clip(-10, 10)
        Xi = Xi[:, :, sel_idx].astype(np.float32)            # (B, 5, 40)
        chunks.append(Xi)
    return np.concatenate(chunks, axis=0)


def _build_inter_v9(X_grp_359z: np.ndarray, sel_idx: np.ndarray,
                    mu_inter: np.ndarray, sd_inter: np.ndarray,
                    batch: int = 100_000) -> np.ndarray:
    from interaction_features_v8 import compute_v9_features
    chunks = []
    N = X_grp_359z.shape[0]
    for s in range(0, N, batch):
        Xs = X_grp_359z[s:s + batch]
        Xi, _, _ = compute_v9_features(Xs)                   # (B, 5, 136)
        Xi = ((Xi - mu_inter) / np.maximum(sd_inter, 1e-6)).clip(-10, 10)
        Xi = Xi[:, :, sel_idx].astype(np.float32)            # (B, 5, 60)
        chunks.append(Xi)
    return np.concatenate(chunks, axis=0)


def build_lgbm_features(config: str, X_grp_359z: np.ndarray,
                        keep_idx_259d: np.ndarray,
                        sel_v7, mu_v7, sd_v7,
                        sel_v9, mu_v9, sd_v9) -> np.ndarray:
    """Return (N*5, F) flat feature matrix for LGBM."""
    N, S, _ = X_grp_359z.shape

    if config == 'A':
        # 259d pruned only
        base = X_grp_359z[:, :, keep_idx_259d]          # (N, 5, 259)
        flat = base.reshape(N * S, -1)

    elif config == 'B':
        # 259d + 40 v7 interaction
        base  = X_grp_359z[:, :, keep_idx_259d]         # (N, 5, 259)
        inter = _build_inter_v7(X_grp_359z, sel_v7, mu_v7, sd_v7)  # (N, 5, 40)
        flat  = np.concatenate([base, inter], axis=2).reshape(N * S, -1)

    elif config == 'C':
        # 259d + 60 v9 interaction
        base  = X_grp_359z[:, :, keep_idx_259d]         # (N, 5, 259)
        inter = _build_inter_v9(X_grp_359z, sel_v9, mu_v9, sd_v9)  # (N, 5, 60)
        flat  = np.concatenate([base, inter], axis=2).reshape(N * S, -1)

    elif config == 'D':
        # 359d full + 60 v9 interaction
        base  = X_grp_359z                               # (N, 5, 359)
        inter = _build_inter_v9(X_grp_359z, sel_v9, mu_v9, sd_v9)  # (N, 5, 60)
        flat  = np.concatenate([base, inter], axis=2).reshape(N * S, -1)

    else:
        raise ValueError(f'Unknown config: {config}')

    return flat.astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, choices=['A', 'B', 'C', 'D'])
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--out-root', default='runs_v10_lgbm')
    ap.add_argument('--out-tag', default=None)
    ap.add_argument('--num-boost-round', type=int, default=500)
    ap.add_argument('--no-wandb', action='store_true')
    args = ap.parse_args()

    cfg = args.config
    seed = args.seed
    tag = args.out_tag or f'lgbm_{cfg}'
    run_name = f'{tag}_s{seed}'
    out_dir = Path(args.out_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n{"="*60}', flush=True)
    print(f'v10 LGBM  config={cfg}  seed={seed}  run={run_name}', flush=True)

    # ------------------------------------------------------------------
    # Load reference files
    # ------------------------------------------------------------------
    keep_idx_L8   = get_L8_keep_idx()                      # (359,)
    feat_names_L8 = get_feat_names(keep_idx_L8)            # list[359]

    keep_idx_259d = np.load(HERE / 'keep_idx_259d.npy').astype(np.int64)  # (259,)

    sel_v7 = np.load(HERE / 'selected_features.npy').astype(np.int64)         # (40,)
    sel_v9 = np.load(HERE / 'selected_features_v9_60.npy').astype(np.int64)   # (60,)

    zst_v7 = np.load(HERE / 'inter_zstats.npz', allow_pickle=True)
    mu_v7  = zst_v7['mu'].astype(np.float32)    # (116,)
    sd_v7  = zst_v7['sd'].astype(np.float32)

    zst_v9 = np.load(HERE / 'inter_zstats_v9.npz', allow_pickle=True)
    mu_v9  = zst_v9['mu'].astype(np.float32)    # (136,)
    sd_v9  = zst_v9['sd'].astype(np.float32)

    print(f'  keep_idx_259d={len(keep_idx_259d)}  sel_v7={len(sel_v7)}  sel_v9={len(sel_v9)}', flush=True)

    # ------------------------------------------------------------------
    # Load and preprocess splits
    # ------------------------------------------------------------------
    t0 = time.time()
    print('  loading train...', flush=True)
    X_tr_grp, y_reg_tr, y_cls_tr, mp_t_tr, mp_th_tr = load_split_grouped(
        'train', keep_idx_L8, feat_names_L8)
    print('  loading val...', flush=True)
    X_va_grp, y_reg_va, y_cls_va, mp_t_va, mp_th_va = load_split_grouped(
        'val', keep_idx_L8, feat_names_L8)
    print('  loading test...', flush=True)
    X_te_grp, y_reg_te, y_cls_te, mp_t_te, mp_th_te = load_split_grouped(
        'test', keep_idx_L8, feat_names_L8)
    print(f'  loaded in {time.time()-t0:.1f}s  '
          f'train={X_tr_grp.shape}  val={X_va_grp.shape}  test={X_te_grp.shape}', flush=True)

    # Window-z: fit on train (no mirror aug for LGBM)
    flat_tr = X_tr_grp.reshape(-1, len(keep_idx_L8))
    mu_base = np.nanmean(flat_tr, axis=0).astype(np.float32)
    sd_base = np.maximum(np.nanstd(flat_tr, axis=0), 1e-6).astype(np.float32)
    del flat_tr

    print('  z-scoring...', flush=True)
    X_tr_z = zscore_groups(X_tr_grp, mu_base, sd_base)
    X_va_z = zscore_groups(X_va_grp, mu_base, sd_base)
    X_te_z = zscore_groups(X_te_grp, mu_base, sd_base)
    del X_tr_grp, X_va_grp, X_te_grp

    # ------------------------------------------------------------------
    # Build LGBM feature matrices
    # ------------------------------------------------------------------
    print(f'  building features for config {cfg}...', flush=True)
    t1 = time.time()
    X_tr_flat = build_lgbm_features(cfg, X_tr_z, keep_idx_259d,
                                    sel_v7, mu_v7, sd_v7,
                                    sel_v9, mu_v9, sd_v9)
    X_va_flat = build_lgbm_features(cfg, X_va_z, keep_idx_259d,
                                    sel_v7, mu_v7, sd_v7,
                                    sel_v9, mu_v9, sd_v9)
    X_te_flat = build_lgbm_features(cfg, X_te_z, keep_idx_259d,
                                    sel_v7, mu_v7, sd_v7,
                                    sel_v9, mu_v9, sd_v9)
    print(f'  features built in {time.time()-t1:.1f}s  '
          f'dim={X_tr_flat.shape[1]}', flush=True)
    del X_tr_z, X_va_z, X_te_z

    # Targets + sample weights (flat)
    y_tr_flat  = y_reg_tr.reshape(-1).astype(np.float32)
    y_va_flat  = y_reg_va.reshape(-1).astype(np.float32)
    y_cls_flat = y_cls_tr.reshape(-1)

    cw = class_balanced_weight(y_cls_flat)
    sw_tr = cw[y_cls_flat]
    print(f'  y_tr stats: mean={y_tr_flat.mean():.4e}  std={y_tr_flat.std():.4e}', flush=True)
    print(f'  class_weights: {cw.tolist()}', flush=True)

    # ------------------------------------------------------------------
    # WandB
    # ------------------------------------------------------------------
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            os.environ.setdefault('WANDB_API_KEY',
                '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')
            wandb.init(
                project='liangwenbei-cross-sym',
                entity='jingzheshi',
                name=run_name,
                config=dict(
                    config=cfg, seed=seed, feat_dim=int(X_tr_flat.shape[1]),
                    n_train=int(len(y_tr_flat)), n_val=int(len(y_va_flat)),
                    num_boost_round=args.num_boost_round,
                    **HP_CONFIG,
                ),
                reinit=True,
                settings=wandb.Settings(start_method='thread'),
            )
        except Exception as e:
            print(f'  wandb init failed: {e}', flush=True)
            use_wandb = False

    # ------------------------------------------------------------------
    # LGBM training with early stopping on val
    # ------------------------------------------------------------------
    params = {
        'objective': 'regression_l2',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': int(HP_CONFIG['num_leaves']),
        'feature_fraction': float(HP_CONFIG['feature_fraction']),
        'bagging_fraction': float(HP_CONFIG['bagging_fraction']),
        'bagging_freq': 5,
        'lambda_l2': float(HP_CONFIG['lambda_l2']),
        'min_data_in_leaf': 100,
        'device': 'gpu',
        'gpu_use_dp': False,
        'seed': seed,
        'verbose': -1,
        'num_threads': 8,
    }

    dtrain = lgb.Dataset(X_tr_flat, label=y_tr_flat, weight=sw_tr, free_raw_data=False)
    dval   = lgb.Dataset(X_va_flat, label=y_va_flat, free_raw_data=False)

    t_tr = time.time()
    print('  training LGBM...', flush=True)
    booster = lgb.train(
        params, dtrain,
        num_boost_round=args.num_boost_round,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )
    train_time = time.time() - t_tr
    best_iter = booster.best_iteration
    print(f'  training done in {train_time:.1f}s  best_iter={best_iter}', flush=True)

    # ------------------------------------------------------------------
    # Threshold search on VAL (never touch test here)
    # ------------------------------------------------------------------
    print('  threshold search on val...', flush=True)
    preds_va_flat = booster.predict(X_va_flat, num_iteration=best_iter)
    N_va = y_reg_va.shape[0]
    preds_va_grp  = preds_va_flat.reshape(N_va, 5)
    val_results, thresholds = eval_on_val(preds_va_grp, mp_t_va, mp_th_va, y_reg_va)
    print(f'  val total_pnl={val_results["total_pnl"]:+.4f}  '
          f'mean_ic={val_results["mean_ic"]:.5f}', flush=True)
    print(f'  val thresholds: {thresholds}', flush=True)

    # ------------------------------------------------------------------
    # Test evaluation (frozen thresholds, no tuning)
    # ------------------------------------------------------------------
    print('  evaluating on test (frozen)...', flush=True)
    preds_te_flat = booster.predict(X_te_flat, num_iteration=best_iter)
    N_te = y_reg_te.shape[0]
    preds_te_grp  = preds_te_flat.reshape(N_te, 5)
    test_results = eval_on_test(preds_te_grp, mp_t_te, mp_th_te, y_reg_te, thresholds)
    print(f'  test total_pnl={test_results["total_pnl"]:+.4f}  '
          f'mean_ic={test_results["mean_ic"]:.5f}  '
          f'n_actions={test_results["total_nact"]}', flush=True)

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------
    fi_gain  = booster.feature_importance(importance_type='gain').tolist()
    fi_split = booster.feature_importance(importance_type='split').tolist()
    top20_gain  = sorted(range(len(fi_gain)),  key=lambda i: fi_gain[i],  reverse=True)[:20]
    top20_split = sorted(range(len(fi_split)), key=lambda i: fi_split[i], reverse=True)[:20]

    # ------------------------------------------------------------------
    # WandB final log
    # ------------------------------------------------------------------
    if use_wandb:
        try:
            import wandb
            wandb.log({
                'val_pnl':    val_results['total_pnl'],
                'val_ic':     val_results['mean_ic'],
                'test_pnl':   test_results['total_pnl'],
                'test_ic':    test_results['mean_ic'],
                'best_iter':  best_iter,
                'train_time': train_time,
                **{f'val_{k}': v for k, v in val_results.items()},
                **{f'test_{k}': v for k, v in test_results.items()},
            })
            wandb.finish()
        except Exception as e:
            print(f'  wandb log failed: {e}', flush=True)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    result = {
        'task': f'v10_lgbm_config{cfg}_s{seed}',
        'config': cfg,
        'seed': seed,
        'feat_dim': int(X_tr_flat.shape[1]),
        'best_iter': int(best_iter),
        'train_time_sec': float(train_time),
        'metrics': {
            'val_pnl':  float(val_results['total_pnl']),
            'val_ic':   float(val_results['mean_ic']),
            'test_pnl': float(test_results['total_pnl']),
            'test_ic':  float(test_results['mean_ic']),
            'test_nact': int(test_results['total_nact']),
        },
        'val_results':  val_results,
        'test_results': test_results,
        'thresholds':   {str(k): float(v) for k, v in thresholds.items()},
        'top20_gain_idx':  top20_gain,
        'top20_split_idx': top20_split,
        'notes': (
            f'config={cfg}, seed={seed}, strict train/val/test split, '
            f'early_stopping on val, threshold search on val (frozen on test), '
            f'gpu LGBM, HP_CONFIGS[0]'
        ),
    }
    result_path = out_dir / 'results.json'
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'  saved -> {result_path}', flush=True)

    print(
        f'\nRESULT: task=v10_lgbm_config{cfg}_s{seed} '
        f'metrics={{val_pnl={val_results["total_pnl"]:+.4f}, '
        f'test_pnl={test_results["total_pnl"]:+.4f}, '
        f'test_ic={test_results["mean_ic"]:.5f}, '
        f'best_iter={best_iter}}} '
        f'notes=config_{cfg}_seed_{seed}',
        flush=True,
    )


if __name__ == '__main__':
    main()
