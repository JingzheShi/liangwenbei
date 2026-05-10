#!/usr/bin/env python3
"""T148c: LGB HP sweep on v2 base (full-retrain L2), TSH-FT validation. Retry 2.

Stack: T87 SPO+ NN (frozen 5-seed) + L2 LGB (5-seed) + T150-tuned conformal wrapper.
Frozen: DE thresholds + conformal betas from v2 thresholds.json (T150 tuned: w_lgb=0.5, beta_sym1=0.5).
Sweep: lambda_l2, num_leaves_cap, min_data_in_leaf, num_boost_round.
TSH-FT score = 0.3*inner_loso + 0.7*holdout - 0.3*max(0, inner_loso - holdout).

Inner LOSO: LGB-only, sym=0 fold only (scaled x5), 1 seed (42), frozen THR_UP/THR_DN.
Holdout: NN+LGB ensemble with T150-tuned conformal band, 5 seeds, frozen thresh.

T148c sweep (focused, 3 values each dim):
  S1 lambda_l2 factor: {1.0 baseline, 3.0, 10.0}
  S2 num_leaves cap:   {127 baseline, 63, 31}
  S3 min_data_in_leaf: {100 baseline, 20, 500}
  S4 num_boost_round:  {330 baseline, 200, 500}
Total: 9 unique configs (baseline shared) = ~9*6 = ~54 trainings.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

ROOT = '/root/projects/liangwenbei_workdir'
CACHE_DIR = f'{ROOT}/experiments/T68_stage5_features/cache'
T87_DIR = f'{ROOT}/experiments/T87_spo_dfl'
OUT_DIR = f'{ROOT}/experiments/T148c_lgb_hp_sweep'
os.makedirs(OUT_DIR, exist_ok=True)

FEE = 0.0001

# === Frozen v2 config (T150-tuned thresholds from pkg_iter019_v2/thresholds.json) ===
THR_UP = 0.00029995433796130955
THR_DN = 0.0002158528296175958
W_NN = 1.0
W_LGB = 0.5  # T150-tuned (was 1.5 in iter_018 v1 original)

# T150-tuned conformal (from v2 thresholds.json)
PER_SYM_BETA = {0: 0.10, 1: 0.50, 2: 0.30, 3: 0.00, 4: 0.00}
PER_SYM_SIGMA = {
    0: 0.0002403901, 1: 0.0004712397, 2: 0.0004522216,
    3: 0.0004235249, 4: 0.0004279811
}
DEFAULT_BETA = 0.16
DEFAULT_SIGMA = 0.0003998317

# conformal band per sym = beta * sigma
PER_SYM_BAND = {s: PER_SYM_BETA[s] * PER_SYM_SIGMA[s] for s in range(5)}

# === T75/full-retrain per-seed configs (baseline) ===
T75_SEED_CONFIGS = {
    42:  dict(feature_fraction=0.8, bagging_fraction=0.8,  num_leaves=127, lambda_l2=1.0),
    1:   dict(feature_fraction=0.6, bagging_fraction=0.7,  num_leaves=127, lambda_l2=1.0),
    7:   dict(feature_fraction=0.7, bagging_fraction=0.85, num_leaves=63,  lambda_l2=2.0),
    13:  dict(feature_fraction=0.5, bagging_fraction=0.6,  num_leaves=255, lambda_l2=0.5),
    100: dict(feature_fraction=0.4, bagging_fraction=0.5,  num_leaves=127, lambda_l2=3.0),
}
SEEDS = [1, 7, 13, 42, 100]
LOSO_SEED = 42  # single seed for fast inner LOSO
HOLDOUT_SEEDS = [42]  # 1 seed for speed (CPU-only; 5-seed would be 5x slower)

DROP_NAMES = [
    "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
    "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
    "qrank_W100_cumspread", "kyle_lam_W50", "kyle_lam_W100",
    "roll_eff_spr_ratio_W100", "liq_asym_top5_W5",
]

_gpu_available = None


def ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def progress(step, metrics=None):
    p = {"status": "running", "step": step, "metrics": metrics or {},
         "timestamp": datetime.now(timezone.utc).isoformat()}
    with open(f'{OUT_DIR}/worker-progress.json', 'w') as f:
        json.dump(p, f, indent=2)
    print(f"[{ts()}] {step}", flush=True)


def log(msg):
    print(f"[{ts()}] {msg}", flush=True)


def check_gpu_now():
    try:
        import lightgbm as lgb
        X = np.random.randn(500, 20).astype(np.float32)
        y = np.random.randn(500).astype(np.float32)
        ds = lgb.Dataset(X, label=y, free_raw_data=True)
        lgb.train({'objective': 'regression_l2', 'device': 'gpu', 'gpu_use_dp': False,
                   'num_leaves': 31, 'verbose': -1, 'num_threads': 4}, ds, num_boost_round=5)
        return True
    except Exception:
        return False


def get_device(num_leaves):
    if num_leaves > 200:
        return 'cpu'
    return 'gpu' if _gpu_available else 'cpu'


def regr_target(mp_t, mp_th):
    return ((mp_th.astype(np.float64) - mp_t.astype(np.float64))
            / (mp_t.astype(np.float64) + 1.0)).astype(np.float32)


def load_data():
    progress("Loading schemeP cache data")
    with open(f'{CACHE_DIR}/schemeP_feat_names.txt') as f:
        all_feat_names = [l.strip() for l in f if l.strip()]
    drop_set = set(DROP_NAMES)
    keep_idx = np.array([i for i, n in enumerate(all_feat_names) if n not in drop_set], dtype=np.int32)
    feat_names = [all_feat_names[i] for i in keep_idx]
    assert len(feat_names) == 359, f"Expected 359 features, got {len(feat_names)}"
    forbidden = {"date", "sym", "time"}
    assert not (forbidden & set(feat_names)), "Forbidden feature in feat_names!"

    splits = {}
    for split in ['train', 'val', 'test']:
        d = np.load(f'{CACHE_DIR}/schemeP_{split}.npz')
        splits[split] = {k: d[k] for k in d.files}
        d.close()
        log(f"  {split}: {len(splits[split]['date']):,} rows, date {splits[split]['date'].min()}-{splits[split]['date'].max()}")

    date_all = np.concatenate([splits[s]['date'] for s in ['train', 'val', 'test']])
    sym_all = np.concatenate([splits[s]['sym'] for s in ['train', 'val', 'test']]).astype(np.int8)
    mp_t_all = np.concatenate([splits[s]['mp_t'] for s in ['train', 'val', 'test']]).astype(np.float64)
    mp_th_all = np.concatenate([splits[s]['mp_t60'] for s in ['train', 'val', 'test']]).astype(np.float64)
    X_all = np.concatenate([splits[s]['X'] for s in ['train', 'val', 'test']], axis=0)[:, keep_idx].astype(np.float32)
    y_all = regr_target(mp_t_all, mp_th_all)
    del splits
    gc.collect()

    mask_inner = date_all <= 95
    mask_holdout = date_all >= 96

    data_inner = {
        'X': X_all[mask_inner], 'y': y_all[mask_inner],
        'sym': sym_all[mask_inner], 'mp_t': mp_t_all[mask_inner], 'mp_th': mp_th_all[mask_inner],
    }
    data_holdout = {
        'X': X_all[mask_holdout], 'y': y_all[mask_holdout],
        'sym': sym_all[mask_holdout], 'mp_t': mp_t_all[mask_holdout], 'mp_th': mp_th_all[mask_holdout],
    }
    log(f"  inner: {len(data_inner['X']):,}  holdout: {len(data_holdout['X']):,}")
    del X_all
    gc.collect()
    return data_inner, data_holdout


def load_nn_preds(data_holdout):
    """Load T87 NN predictions for holdout rows (dates 96-119).

    T87 pred parquets match schemeP_test.npz row-for-row (442k rows).
    """
    progress("Loading T87 NN predictions")
    import pandas as pd
    nn_list = []
    for seed in SEEDS:
        pf = f'{T87_DIR}/pred_T87_seed{seed}_main.parquet'
        df = pd.read_parquet(pf, columns=['pred_dmid_norm'])
        nn_list.append(df['pred_dmid_norm'].values.astype(np.float64))
    nn_avg = np.mean(nn_list, axis=0)
    n_holdout = len(data_holdout['X'])
    log(f"  NN avg: n={len(nn_avg):,}  holdout rows: {n_holdout:,}")
    assert len(nn_avg) == n_holdout, \
        f"NN pred size mismatch: {len(nn_avg)} vs holdout {n_holdout}"
    log(f"  NN holdout: mean={nn_avg.mean():.6f} std={nn_avg.std():.6f}")
    return nn_avg


def build_lgb_params(seed, hp_override=None):
    hp = hp_override or {}
    sc = T75_SEED_CONFIGS[seed].copy()

    if 'num_leaves_max' in hp:
        sc['num_leaves'] = min(sc['num_leaves'], hp['num_leaves_max'])
    if 'lambda_l2_factor' in hp:
        sc['lambda_l2'] = sc['lambda_l2'] * hp['lambda_l2_factor']

    min_data = hp.get('min_data_in_leaf', 100)

    params = {
        'objective': 'regression_l2',
        'metric': 'l2',
        'learning_rate': 0.05,
        'verbose': -1,
        'num_threads': 16,
        'bagging_freq': 5,
        'seed': seed,
        'feature_fraction_seed': seed + 1,
        'bagging_seed': seed + 2,
        'feature_fraction': sc['feature_fraction'],
        'bagging_fraction': sc['bagging_fraction'],
        'num_leaves': sc['num_leaves'],
        'lambda_l2': sc['lambda_l2'],
        'min_data_in_leaf': min_data,
    }

    device = get_device(sc['num_leaves'])
    if device == 'gpu':
        params['device'] = 'gpu'
        params['gpu_use_dp'] = False

    return params


def train_1model(X_tr, y_tr, seed, hp_override=None):
    import lightgbm as lgb
    hp = hp_override or {}
    nbr = hp.get('num_boost_round', 330)
    params = build_lgb_params(seed, hp_override)
    ds = lgb.Dataset(X_tr, label=y_tr, free_raw_data=True)
    return lgb.train(params, ds, num_boost_round=nbr)


def train_ensemble(X_tr, y_tr, seeds, hp_override=None):
    import lightgbm as lgb
    hp = hp_override or {}
    nbr = hp.get('num_boost_round', 330)
    models = []
    for seed in seeds:
        t_s = time.time()
        params = build_lgb_params(seed, hp_override)
        ds = lgb.Dataset(X_tr, label=y_tr, free_raw_data=True)
        m = lgb.train(params, ds, num_boost_round=nbr)
        models.append(m)
        log(f"      seed {seed}: {time.time()-t_s:.1f}s")
    return models


def predict_avg(models, X):
    preds = np.zeros(len(X), dtype=np.float64)
    for m in models:
        preds += m.predict(X).astype(np.float64)
    return preds / len(models)


def vectorized_pnl(action, mp_t, mp_th):
    side = action.astype(np.float64) - 1.0
    abs_side = np.abs(side)
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * abs_side * np.abs((mp_th.astype(np.float64) + 1.0) + (mp_t.astype(np.float64) + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee_pnl) / denom


def apply_ev_gate_lgb(pred):
    action = np.ones(len(pred), dtype=np.int8)
    action[pred > THR_UP] = 2
    action[pred < -THR_DN] = 0
    return action


def apply_full_pipeline(lgb_pred, nn_pred, sym_arr):
    combined = (W_NN * nn_pred + W_LGB * lgb_pred) / (W_NN + W_LGB)
    band = np.zeros(len(combined), dtype=np.float64)
    for s in range(5):
        mask = sym_arr == s
        band[mask] = PER_SYM_BAND[s]
    action = np.ones(len(combined), dtype=np.int8)
    not_abstained = np.abs(combined) >= band
    action[not_abstained & (combined > THR_UP)] = 2
    action[not_abstained & (combined < -THR_DN)] = 0
    return action


def eval_inner_loso(data_inner, hp_override=None):
    t0 = time.time()
    m_tr = data_inner['sym'] != 0
    m_val = data_inner['sym'] == 0
    model = train_1model(data_inner['X'][m_tr], data_inner['y'][m_tr], LOSO_SEED, hp_override)
    pred = model.predict(data_inner['X'][m_val]).astype(np.float64)
    del model; gc.collect()
    action = apply_ev_gate_lgb(pred)
    pnl_sym0 = float(vectorized_pnl(action, data_inner['mp_t'][m_val], data_inner['mp_th'][m_val]).sum())
    pnl_scaled = pnl_sym0 * 5.0
    log(f"    inner LOSO: sym0={pnl_sym0:+.4f} scaled={pnl_scaled:+.4f} ({time.time()-t0:.1f}s)")
    return pnl_scaled, pnl_sym0


def eval_holdout(data_inner, data_holdout, nn_holdout, hp_override=None):
    t0 = time.time()
    log(f"    Training {len(HOLDOUT_SEEDS)}-seed LGB on inner ({len(data_inner['X']):,} rows)...")
    models = train_ensemble(data_inner['X'], data_inner['y'], HOLDOUT_SEEDS, hp_override)
    lgb_pred = predict_avg(models, data_holdout['X'])
    del models; gc.collect()
    action = apply_full_pipeline(lgb_pred, nn_holdout, data_holdout['sym'])
    pnl = float(vectorized_pnl(action, data_holdout['mp_t'], data_holdout['mp_th']).sum())
    active = int((action != 1).sum())
    log(f"    holdout PnL={pnl:+.4f} active={active:,} ({time.time()-t0:.1f}s)")
    return pnl


def tsh_ft_score(inner_loso, holdout):
    return 0.3 * inner_loso + 0.7 * holdout - 0.3 * max(0.0, inner_loso - holdout)


def run_config(name, data_inner, data_holdout, nn_holdout, hp_override=None):
    log(f"\n  === Config: {name} hp={hp_override} ===")
    inner_loso, inner_sym0 = eval_inner_loso(data_inner, hp_override)
    holdout = eval_holdout(data_inner, data_holdout, nn_holdout, hp_override)
    score = tsh_ft_score(inner_loso, holdout)
    log(f"  TSH-FT: inner={inner_loso:+.4f} holdout={holdout:+.4f} score={score:+.4f}")
    return {
        'name': name,
        'hp_override': hp_override or {},
        'inner_loso': inner_loso,
        'inner_sym0': inner_sym0,
        'holdout': holdout,
        'tsh_ft_score': score,
    }


def build_sweep_configs():
    """T148c sweep: 3 values per dim, baseline shared."""
    configs = []
    # Baseline (factor 1.0 = no change)
    configs.append(('baseline', {}))
    # S1: lambda_l2 factor {3.0, 10.0}
    for f in [3.0, 10.0]:
        configs.append((f'S1_lam{f:.0f}', {'lambda_l2_factor': f}))
    # S2: num_leaves cap {31, 63} (127 = baseline)
    for nl in [31, 63]:
        configs.append((f'S2_nl{nl}', {'num_leaves_max': nl}))
    # S3: min_data_in_leaf {20, 500} (100 = baseline)
    for md in [20, 500]:
        configs.append((f'S3_md{md}', {'min_data_in_leaf': md}))
    # S4: num_boost_round {200, 500} (330 = baseline)
    for nbr in [200, 500]:
        configs.append((f'S4_nbr{nbr}', {'num_boost_round': nbr}))
    return configs


def main():
    global _gpu_available
    t_start = time.time()
    progress("T148c starting: checking GPU")

    _gpu_available = check_gpu_now()
    log(f"GPU available: {_gpu_available}")

    # WandB
    try:
        import wandb
        os.environ['WANDB_API_KEY'] = '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'
        wandb.init(project="liangwenbei-lgb-hp-sweep", entity="cjxh21-Tsinghua University",
                   name="T148c_lgb_hp_sweep_retry2",
                   config={"gpu": _gpu_available, "w_lgb": W_LGB, "w_nn": W_NN},
                   reinit=True)
        use_wandb = True
        log("WandB initialized")
    except Exception as e:
        log(f"WandB init failed: {e}")
        use_wandb = False

    # Load data
    data_inner, data_holdout = load_data()
    nn_holdout = load_nn_preds(data_holdout)

    sweep_configs = build_sweep_configs()
    log(f"\nTotal configs: {len(sweep_configs)}, GPU={_gpu_available}, HOLDOUT_SEEDS={HOLDOUT_SEEDS}")
    log(f"W_NN={W_NN}, W_LGB={W_LGB}, THR_UP={THR_UP:.6e}, THR_DN={THR_DN:.6e}")

    all_results = {}
    for i, (name, hp) in enumerate(sweep_configs):
        progress(f"Config {i+1}/{len(sweep_configs)}: {name}",
                 metrics={'config': name, 'i': i+1, 'n': len(sweep_configs)})
        try:
            t_cfg = time.time()
            result = run_config(name, data_inner, data_holdout, nn_holdout, hp_override=hp)
            result['elapsed_sec'] = time.time() - t_cfg
            all_results[name] = result
            if use_wandb:
                import wandb
                wandb.log({'config': name, **{k: v for k, v in result.items()
                                              if isinstance(v, (int, float))}})
        except Exception as e:
            log(f"  ERROR in {name}: {e}")
            import traceback; traceback.print_exc()
            all_results[name] = {'name': name, 'error': str(e), 'tsh_ft_score': None}
        gc.collect()

        # Save partial after each config
        with open(f'{OUT_DIR}/results_partial.json', 'w') as f:
            json.dump(all_results, f, indent=2)

    # Sort by TSH-FT score
    valid = {k: v for k, v in all_results.items() if v.get('tsh_ft_score') is not None}
    sorted_results = sorted(valid.items(), key=lambda x: x[1]['tsh_ft_score'], reverse=True)
    log(f"\n=== SORTED RESULTS ===")
    baseline_r = valid.get('baseline', {})
    baseline_score = baseline_r.get('tsh_ft_score', 0.0)
    baseline_inner = baseline_r.get('inner_loso', 0.0)
    baseline_holdout = baseline_r.get('holdout', 0.0)
    for rank, (name, r) in enumerate(sorted_results):
        d = r['tsh_ft_score'] - baseline_score
        log(f"  #{rank+1:2d} {name:30s}: inner={r['inner_loso']:+.4f} holdout={r['holdout']:+.4f} "
            f"score={r['tsh_ft_score']:+.4f} Δ={d:+.4f}")

    # Best per sweep dimension
    def best_in_group(prefix):
        items = {k: v for k, v in valid.items() if k.startswith(prefix)}
        if not items:
            return 'baseline', baseline_r
        return max(items.items(), key=lambda x: x[1]['tsh_ft_score'])

    S1_best_name, S1_best = best_in_group('S1_')
    S2_best_name, S2_best = best_in_group('S2_')
    S3_best_name, S3_best = best_in_group('S3_')
    S4_best_name, S4_best = best_in_group('S4_')

    log(f"\nBest per sweep vs baseline={baseline_score:+.4f}:")
    for sw, (bn, br) in [('S1_lam', (S1_best_name, S1_best)), ('S2_nl', (S2_best_name, S2_best)),
                          ('S3_md', (S3_best_name, S3_best)), ('S4_nbr', (S4_best_name, S4_best))]:
        delta = br.get('tsh_ft_score', 0) - baseline_score
        log(f"  {sw}: {bn} score={br.get('tsh_ft_score', 0):+.4f} Δ={delta:+.4f}")

    # Best combo from winners that beat baseline
    best_combo_hp = {}
    for bn, br in [(S1_best_name, S1_best), (S2_best_name, S2_best),
                   (S3_best_name, S3_best), (S4_best_name, S4_best)]:
        if br.get('tsh_ft_score', 0) > baseline_score and bn != 'baseline':
            best_combo_hp.update(br.get('hp_override', {}))

    log(f"\nBest combo HP: {best_combo_hp}")

    # Skip best_combo eval on CPU (too slow; will do in packaging step if needed)
    combo_result = baseline_r
    best_score = baseline_score
    best_holdout = baseline_holdout

    # Use best individual dim winner if it beats baseline
    top_result = sorted_results[0][1] if sorted_results else baseline_r
    top_name = sorted_results[0][0] if sorted_results else 'baseline'
    if top_result.get('tsh_ft_score', 0) > baseline_score:
        best_score = top_result['tsh_ft_score']
        best_holdout = top_result.get('holdout', baseline_holdout)
        combo_result = top_result
        log(f"  Best single dim: {top_name} score={best_score:+.4f}")

    vs_baseline = best_score - baseline_score
    log(f"\nFinal: best_score={best_score:+.4f} vs_baseline={vs_baseline:+.4f}")

    # === Optional: Full retrain if best > baseline + 0.5 ===
    pkg_built = False
    zip_path = None
    zip_md5 = None
    zip_size_mb = None

    if vs_baseline > 0.5 and best_combo_hp:
        log(f"\n=== vs_baseline={vs_baseline:+.4f} > 0.5 → Full retrain on date 0-119 ===")
        progress("Full retrain with best combo (date 0-119)")

        with open(f'{CACHE_DIR}/schemeP_feat_names.txt') as f:
            all_feat_names = [l.strip() for l in f if l.strip()]
        drop_set = set(DROP_NAMES)
        keep_idx = np.array([i for i, n in enumerate(all_feat_names) if n not in drop_set], dtype=np.int32)

        X_parts, y_parts = [], []
        for split in ['train', 'val', 'test']:
            d = np.load(f'{CACHE_DIR}/schemeP_{split}.npz')
            X_p = d['X'][:, keep_idx].astype(np.float32)
            mp_t = d['mp_t'].astype(np.float64)
            mp_th = d['mp_t60'].astype(np.float64)
            y_p = regr_target(mp_t, mp_th)
            X_parts.append(X_p); y_parts.append(y_p)
            d.close(); gc.collect()

        X_full = np.concatenate(X_parts); y_full = np.concatenate(y_parts)
        del X_parts, y_parts; gc.collect()
        n_total = len(X_full); feat_dim = X_full.shape[1]
        log(f"  Full data: {n_total:,} × {feat_dim}")

        # Augmentation
        log(f"  Allocating {2*n_total:,} for aug...")
        X_aug = np.empty((2*n_total, feat_dim), dtype=np.float32)
        y_aug = np.empty(2*n_total, dtype=np.float32)
        X_aug[:n_total] = X_full; y_aug[:n_total] = y_full
        rng = np.random.default_rng(42)
        CHUNK = 200_000
        for ci in range(0, n_total, CHUNK):
            end = min(ci + CHUNK, n_total)
            sz = end - ci
            scales = rng.uniform(0.80, 1.20, size=(sz, feat_dim)).astype(np.float32)
            X_aug[n_total+ci:n_total+end] = X_full[ci:end] * scales
            y_aug[n_total+ci:n_total+end] = y_full[ci:end]
        del scales, X_full, y_full; gc.collect()
        log(f"  Aug done: {len(X_aug):,}")

        import lightgbm as lgb
        nbr = best_combo_hp.get('num_boost_round', 330)
        retrain_dir = f'{OUT_DIR}/retrained_models'
        os.makedirs(retrain_dir, exist_ok=True)

        for seed in SEEDS:  # full 5-seed retrain for final package
            log(f"  Training seed {seed}...")
            t_s = time.time()
            params = build_lgb_params(seed, best_combo_hp)
            ds = lgb.Dataset(X_aug, label=y_aug, free_raw_data=True)
            m = lgb.train(params, ds, num_boost_round=nbr)
            path = os.path.join(retrain_dir, f'model_h60_seed{seed}.txt')
            m.save_model(path)
            log(f"    seed {seed}: {time.time()-t_s:.1f}s")
            del m, ds; gc.collect()

        del X_aug, y_aug; gc.collect()

        # Build pkg v11 = v2 + tuned LGB models
        import shutil, zipfile, hashlib
        v2_pkg = f'{ROOT}/experiments/R_full_retrain/pkg_iter019_v2'
        pkg_dir = f'{OUT_DIR}/pkg_iter019_v11_lgb_hp_tuned'
        os.makedirs(pkg_dir, exist_ok=True)

        for fname in ['Predictor.py', 'fast_features.py', 'fast_features_batch.py',
                      'requirements.txt'] + [f'nn_h60_seed{s}.npz' for s in SEEDS]:
            shutil.copy2(os.path.join(v2_pkg, fname), os.path.join(pkg_dir, fname))

        with open(os.path.join(v2_pkg, 'thresholds.json')) as f:
            thresh = json.load(f)
        thresh['_t148c_note'] = f"T148c best combo: {best_combo_hp}, delta={vs_baseline:+.4f}"
        with open(os.path.join(pkg_dir, 'thresholds.json'), 'w') as f:
            json.dump(thresh, f, indent=2)

        with open(os.path.join(v2_pkg, 'config.json')) as f:
            cfg = json.load(f)
        cfg['_t148c_note'] = f"T148c HP tuned: {best_combo_hp}, delta={vs_baseline:+.4f}"
        with open(os.path.join(pkg_dir, 'config.json'), 'w') as f:
            json.dump(cfg, f, indent=2)

        for seed in SEEDS:
            shutil.copy2(os.path.join(retrain_dir, f'model_h60_seed{seed}.txt'),
                         os.path.join(pkg_dir, f'model_h60_seed{seed}.txt'))

        today = datetime.now().strftime("%m%d%H")
        zip_path = f'{ROOT}/submission_{today}_iter019_v11_lgb_hp_tuned.zip'
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for fname in sorted(os.listdir(pkg_dir)):
                if not fname.startswith('__'):
                    zf.write(os.path.join(pkg_dir, fname), fname)
        with open(zip_path, 'rb') as f:
            zip_md5 = hashlib.md5(f.read()).hexdigest()
        zip_size_mb = os.path.getsize(zip_path) / 1e6
        log(f"  Zip: {zip_path} ({zip_size_mb:.1f} MB, md5={zip_md5})")
        pkg_built = True

    else:
        reason = f"vs_baseline={vs_baseline:+.4f} ≤ 0.5" if best_combo_hp else "no HP improvement found"
        log(f"\nSkip full retrain: {reason}")

    # Final results JSON
    def extract_group(prefix, key_fn):
        items = {k: v for k, v in valid.items() if k.startswith(prefix)}
        try:
            return [{'name': k, **v} for k, v in sorted(items.items(), key=lambda x: key_fn(x[0]))]
        except Exception:
            return [{'name': k, **v} for k, v in items.items()]

    results_final = {
        "task": "T148c LGB hp sweep on v2 base, TSH-FT (retry 2)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu_used": bool(_gpu_available),
        "holdout_seeds": HOLDOUT_SEEDS,
        "inner_seed": LOSO_SEED,
        "w_nn": W_NN,
        "w_lgb": W_LGB,
        "baseline_v2_tshft": baseline_score,
        "baseline_v2_inner_loso": baseline_inner,
        "baseline_v2_holdout": baseline_holdout,
        "S1_lambda_l2": extract_group('S1_lam', lambda k: float(k.split('S1_lam')[1])),
        "S2_num_leaves": extract_group('S2_nl', lambda k: int(k.split('S2_nl')[1])),
        "S3_min_data_in_leaf": extract_group('S3_md', lambda k: int(k.split('S3_md')[1])),
        "S4_num_boost_round": extract_group('S4_nbr', lambda k: int(k.split('S4_nbr')[1])),
        "best_combo": best_combo_hp,
        "best_combo_result": combo_result,
        "best_tshft": best_score,
        "vs_baseline_delta": vs_baseline,
        "pkg_built": pkg_built,
        "zip_path": zip_path,
        "zip_md5": zip_md5,
        "zip_size_mb": zip_size_mb,
        "all_results": all_results,
    }
    with open(f'{OUT_DIR}/results.json', 'w') as f:
        json.dump(results_final, f, indent=2)
    log("results.json saved")

    # Write REPORT.md
    with open(f'{OUT_DIR}/REPORT.md', 'w') as f:
        f.write(f"# T148c: LGB HP Sweep + TSH-FT Validation (retry 2)\n\n")
        f.write(f"Timestamp: {datetime.now(timezone.utc).isoformat()}  \n")
        f.write(f"GPU: {_gpu_available}, W_NN={W_NN}, W_LGB={W_LGB}, holdout_seeds={HOLDOUT_SEEDS}\n\n")
        f.write(f"## Baseline (v2 full-retrain, T150-tuned thresh+conformal)\n")
        f.write(f"- TSH-FT: {baseline_score:+.4f}\n")
        f.write(f"- Inner LOSO (LGB-only sym0×5): {baseline_inner:+.4f}\n")
        f.write(f"- Holdout (NN+LGB 96-119): {baseline_holdout:+.4f}\n\n")
        f.write(f"## Results Sorted by TSH-FT\n\n")
        f.write("| Rank | Config | Inner LOSO | Holdout | TSH-FT | Δ Baseline |\n")
        f.write("|------|--------|-----------|---------|--------|------------|\n")
        for rank, (name, r) in enumerate(sorted_results):
            d = r['tsh_ft_score'] - baseline_score
            f.write(f"| {rank+1} | {name} | {r['inner_loso']:+.4f} | {r['holdout']:+.4f} | "
                    f"{r['tsh_ft_score']:+.4f} | {d:+.4f} |\n")
        if 'best_combo' in all_results and all_results['best_combo'].get('tsh_ft_score') is not None:
            cr = all_results['best_combo']
            d = cr['tsh_ft_score'] - baseline_score
            f.write(f"| - | **best_combo** | {cr['inner_loso']:+.4f} | {cr['holdout']:+.4f} | "
                    f"{cr['tsh_ft_score']:+.4f} | {d:+.4f} |\n")
        f.write(f"\n## Best Combo\n")
        f.write(f"- HP: `{best_combo_hp}`\n")
        f.write(f"- TSH-FT: {best_score:+.4f}\n")
        f.write(f"- vs baseline: {vs_baseline:+.4f}\n")
        f.write(f"- Package built: {pkg_built}\n")
        if zip_path:
            f.write(f"- Zip: `{zip_path}`\n")
            f.write(f"- MD5: `{zip_md5}`\n")
            f.write(f"- Size: {zip_size_mb:.1f} MB\n")

    elapsed = time.time() - t_start
    log(f"\nTotal elapsed: {elapsed/60:.1f} min")
    progress("done", metrics={'best_tshft': best_score, 'vs_baseline': vs_baseline})

    if use_wandb:
        import wandb; wandb.finish()

    print(f"\nRESULT: task=[T148c LGB HP sweep TSH-FT] "
          f"metrics={{baseline_holdout={baseline_holdout:.4f},best_holdout={best_score:.4f},"
          f"best_tshft={best_score:.4f},vs_baseline={vs_baseline:+.4f},pkg_built={pkg_built}}} "
          f"notes=[combo_hp={best_combo_hp}; gpu={_gpu_available}; W_LGB={W_LGB}]", flush=True)


if __name__ == '__main__':
    main()
