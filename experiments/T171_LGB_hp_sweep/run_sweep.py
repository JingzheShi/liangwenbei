#!/usr/bin/env python3
"""T171: LGB HP sweep on REAL holdout (replaces dead T148c).

Protocol:
- Train 5-seed LGB on date 0-95 (train+val splits) with variant HP
- Predict date 96-119 (test split) with FROZEN v2 thresholds + conformal
- Compute holdout PnL as proxy
- Sweep: lambda_l2 factor, num_leaves cap, min_data_in_leaf, num_boost_round

Conformal logic (from v2 Predictor):
  effective_thr_up = THR_UP + band
  effective_thr_dn = THR_DN + band
  pred > effective_thr_up  -> long
  pred < -effective_thr_dn -> short

Combined: (W_NN * nn_pred + W_LGB * lgb_pred) / (W_NN + W_LGB)

If best > v2 baseline + 0.5: full retrain on date 0-119 (M7 trick) + build zip.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile
from datetime import datetime, timezone

import numpy as np

ROOT = '/root/projects/liangwenbei_workdir'
CACHE_DIR = f'{ROOT}/experiments/T68_stage5_features/cache'
T87_DIR = f'{ROOT}/experiments/T87_spo_dfl'
V2_PKG_DIR = f'{ROOT}/experiments/R_full_retrain/pkg_iter019_v2'
OUT_DIR = f'{ROOT}/experiments/T171_LGB_hp_sweep'
os.makedirs(OUT_DIR, exist_ok=True)

FEE = 0.0001

# Frozen v2 config (T150-tuned thresholds from pkg_iter019_v2/thresholds.json)
THR_UP = 0.00029995433796130955
THR_DN = 0.0002158528296175958
W_NN = 1.0
W_LGB = 0.5
W_TOTAL = W_NN + W_LGB  # = 1.5

PER_SYM_BETA = {0: 0.10, 1: 0.50, 2: 0.30, 3: 0.0, 4: 0.0}
PER_SYM_SIGMA = {
    0: 0.0002403901, 1: 0.0004712397, 2: 0.0004522216,
    3: 0.0004235249, 4: 0.0004279811,
}
# band per sym = beta * sigma, added to threshold
PER_SYM_BAND = {s: PER_SYM_BETA[s] * PER_SYM_SIGMA[s] for s in range(5)}

# Baseline per-seed configs (v2/T140c full-retrain)
T75_SEED_CONFIGS = {
    42:  dict(feature_fraction=0.8, bagging_fraction=0.8,  num_leaves=127, lambda_l2=1.0),
    1:   dict(feature_fraction=0.6, bagging_fraction=0.7,  num_leaves=127, lambda_l2=1.0),
    7:   dict(feature_fraction=0.7, bagging_fraction=0.85, num_leaves=63,  lambda_l2=2.0),
    13:  dict(feature_fraction=0.5, bagging_fraction=0.6,  num_leaves=255, lambda_l2=0.5),
    100: dict(feature_fraction=0.4, bagging_fraction=0.5,  num_leaves=127, lambda_l2=3.0),
}
SEEDS = [1, 7, 13, 42, 100]

DROP_NAMES = [
    "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
    "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
    "qrank_W100_cumspread", "kyle_lam_W50", "kyle_lam_W100",
    "roll_eff_spr_ratio_W100", "liq_asym_top5_W5",
]

_gpu_available = None
t_start = time.time()


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


def check_gpu():
    global _gpu_available
    try:
        import lightgbm as lgb
        X = np.random.randn(500, 20).astype(np.float32)
        y = np.random.randn(500).astype(np.float32)
        ds = lgb.Dataset(X, label=y, free_raw_data=True)
        lgb.train({'objective': 'regression_l2', 'device': 'gpu', 'gpu_use_dp': False,
                   'num_leaves': 31, 'verbose': -1, 'num_threads': 4}, ds, num_boost_round=5)
        _gpu_available = True
        log("GPU available for LightGBM")
    except Exception as e:
        _gpu_available = False
        log(f"GPU not available ({e}), using CPU")


def regr_target(mp_t, mp_th):
    return ((mp_th.astype(np.float64) - mp_t.astype(np.float64))
            / (mp_t.astype(np.float64) + 1.0)).astype(np.float32)


def load_data():
    """Load schemeP cache: returns train data (0-95) and holdout (96-119)."""
    progress("Loading schemeP cache data")
    with open(f'{CACHE_DIR}/schemeP_feat_names.txt') as f:
        all_feat_names = [l.strip() for l in f if l.strip()]
    drop_set = set(DROP_NAMES)
    keep_idx = np.array([i for i, n in enumerate(all_feat_names) if n not in drop_set], dtype=np.int32)
    feat_names = [all_feat_names[i] for i in keep_idx]
    assert len(feat_names) == 359, f"Expected 359 features, got {len(feat_names)}"
    forbidden = {"date", "sym", "time"}
    assert not (forbidden & set(feat_names)), "Forbidden feature in feat_names!"

    parts = []
    for split in ['train', 'val', 'test']:
        d = np.load(f'{CACHE_DIR}/schemeP_{split}.npz')
        parts.append({k: d[k] for k in d.files})
        d.close()
        log(f"  {split}: {len(parts[-1]['date']):,} rows")

    date_all = np.concatenate([p['date'] for p in parts])
    sym_all = np.concatenate([p['sym'] for p in parts]).astype(np.int8)
    mp_t_all = np.concatenate([p['mp_t'] for p in parts]).astype(np.float64)
    mp_th_all = np.concatenate([p['mp_t60'] for p in parts]).astype(np.float64)
    X_all = np.concatenate([p['X'] for p in parts], axis=0)[:, keep_idx].astype(np.float32)
    y_all = regr_target(mp_t_all, mp_th_all)
    del parts
    gc.collect()

    mask_tr = date_all <= 95
    mask_ho = date_all >= 96

    log(f"  Train (0-95): {mask_tr.sum():,} rows")
    log(f"  Holdout (96-119): {mask_ho.sum():,} rows")

    return {
        'keep_idx': keep_idx,
        'X_all': X_all, 'y_all': y_all, 'date_all': date_all, 'sym_all': sym_all,
        'mp_t_all': mp_t_all, 'mp_th_all': mp_th_all,
        'X_tr': X_all[mask_tr], 'y_tr': y_all[mask_tr],
        'X_ho': X_all[mask_ho], 'sym_ho': sym_all[mask_ho],
        'mp_t_ho': mp_t_all[mask_ho], 'mp_th_ho': mp_th_all[mask_ho],
    }


def load_nn_preds(n_holdout):
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
    log(f"  NN avg: n={len(nn_avg):,}  holdout rows: {n_holdout:,}")
    assert len(nn_avg) == n_holdout, \
        f"NN pred size mismatch: {len(nn_avg)} vs holdout {n_holdout}"
    log(f"  NN holdout: mean={nn_avg.mean():.6f} std={nn_avg.std():.6f}")
    return nn_avg


def build_lgb_params(seed, hp_override):
    """Build LGB params for seed with hp overrides."""
    hp = hp_override or {}
    sc = T75_SEED_CONFIGS[seed].copy()

    nl_cap = hp.get('num_leaves_cap', 999)
    sc['num_leaves'] = min(sc['num_leaves'], nl_cap)

    lam_factor = hp.get('lambda_l2_factor', 1.0)
    sc['lambda_l2'] = sc['lambda_l2'] * lam_factor

    min_data = hp.get('min_data_in_leaf', 100)

    # GPU supported for num_leaves <= 200
    device = ('gpu' if (_gpu_available and sc['num_leaves'] <= 200) else 'cpu')

    params = {
        'objective': 'regression_l2',
        'learning_rate': 0.05,
        'num_leaves': sc['num_leaves'],
        'lambda_l2': sc['lambda_l2'],
        'feature_fraction': sc['feature_fraction'],
        'bagging_fraction': sc['bagging_fraction'],
        'bagging_freq': 5,
        'min_data_in_leaf': min_data,
        'num_threads': 18,
        'verbose': -1,
        'seed': seed,
        'feature_fraction_seed': seed + 1,
        'bagging_seed': seed + 2,
        'device': device,
    }
    if device == 'gpu':
        params['gpu_use_dp'] = False
    return params


def vectorized_pnl(action, mp_t, mp_th):
    """Vectorized PnL matching v2 evaluation (same formula as T148c)."""
    side = action.astype(np.float64) - 1.0  # +1=long, -1=short, 0=hold
    abs_side = np.abs(side)
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * abs_side * np.abs((mp_th.astype(np.float64) + 1.0) + (mp_t.astype(np.float64) + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee_pnl) / denom


def apply_pipeline(lgb_pred, nn_pred, sym_ho):
    """Apply v2 ensemble + conformal. Returns action array (0/1/2)."""
    # Combined prediction: weighted avg
    combined = (W_NN * nn_pred + W_LGB * lgb_pred) / W_TOTAL

    # Per-sym conformal band (added to threshold)
    band = np.zeros(len(sym_ho), dtype=np.float64)
    for s in range(5):
        mask_s = sym_ho == s
        band[mask_s] = PER_SYM_BAND[s]

    # Gate with band
    action = np.ones(len(combined), dtype=np.int8)
    action[combined > (THR_UP + band)] = 2   # long
    action[combined < -(THR_DN + band)] = 0  # short
    return action


def train_5seed_lgb(X_tr, y_tr, nbr, hp_override, label=''):
    """Train 5-seed LGB ensemble, return averaged holdout predictions."""
    import lightgbm as lgb
    models = []
    for seed in SEEDS:
        t0 = time.time()
        params = build_lgb_params(seed, hp_override)
        ds = lgb.Dataset(X_tr, label=y_tr, free_raw_data=True)
        m = lgb.train(params, ds, num_boost_round=nbr)
        models.append(m)
        gc.collect()
        log(f"    {label} seed{seed}: {time.time()-t0:.1f}s device={params['device']}")
    return models


def eval_config(X_tr, y_tr, X_ho, sym_ho, mp_t_ho, mp_th_ho, nn_ho,
                nbr, hp_override, label):
    """Train on 0-95, eval on 96-119, return holdout PnL."""
    progress(f"Eval: {label}", metrics={'label': label})
    t0 = time.time()

    models = train_5seed_lgb(X_tr, y_tr, nbr, hp_override, label=label)
    lgb_pred = np.zeros(len(X_ho), dtype=np.float64)
    for m in models:
        lgb_pred += m.predict(X_ho).astype(np.float64)
    lgb_pred /= len(models)
    del models
    gc.collect()

    action = apply_pipeline(lgb_pred, nn_ho, sym_ho)
    pnl = float(vectorized_pnl(action, mp_t_ho, mp_th_ho).sum())
    active = int((action != 1).sum())
    elapsed = time.time() - t0
    log(f"  {label}: holdout_pnl={pnl:+.4f}  active={active:,}  ({elapsed:.1f}s)")
    return {'holdout_pnl': pnl, 'elapsed': elapsed, 'hp': hp_override or {}, 'nbr': nbr}


def main():
    import wandb
    run = wandb.init(
        project='liangwenbei-T171-lgb-hp-sweep',
        entity='jingzheshi',
        config={'task': 'T171_LGB_hp_sweep', 'date': '2026-05-09'},
        name='T171_lgb_hp_sweep_real_holdout',
    )

    check_gpu()
    data = load_data()
    nn_ho = load_nn_preds(len(data['X_ho']))

    X_tr = data['X_tr']
    y_tr = data['y_tr']
    X_ho = data['X_ho']
    sym_ho = data['sym_ho']
    mp_t_ho = data['mp_t_ho']
    mp_th_ho = data['mp_th_ho']

    def run_eval(label, hp_override, nbr=330):
        return eval_config(X_tr, y_tr, X_ho, sym_ho, mp_t_ho, mp_th_ho,
                           nn_ho, nbr, hp_override, label)

    all_results = {}

    # === Baseline ===
    log("\n=== Baseline ===")
    baseline_r = run_eval('baseline', {}, nbr=330)
    v2_base = baseline_r['holdout_pnl']
    all_results['baseline'] = baseline_r
    wandb.log({'baseline_holdout': v2_base})
    log(f"v2 baseline holdout: {v2_base:+.4f}")

    # === S1: lambda_l2 factor {1.0(base), 2.0, 5.0, 10.0} ===
    log("\n=== S1: lambda_l2 factor sweep ===")
    s1_results = [{'factor': 1.0, 'holdout_pnl': v2_base, 'delta': 0.0}]
    for f in [2.0, 5.0, 10.0]:
        label = f'S1_lam{f:.0f}'
        r = run_eval(label, {'lambda_l2_factor': f})
        all_results[label] = r
        d = r['holdout_pnl'] - v2_base
        s1_results.append({'factor': f, 'holdout_pnl': r['holdout_pnl'], 'delta': d})
        wandb.log({f's1_lam{f:.0f}': r['holdout_pnl'], f's1_lam{f:.0f}_delta': d})

    # === S2: num_leaves cap {127(base), 63, 31} ===
    log("\n=== S2: num_leaves cap sweep ===")
    s2_results = [{'cap': 127, 'holdout_pnl': v2_base, 'delta': 0.0}]
    for cap in [63, 31]:
        label = f'S2_nl{cap}'
        r = run_eval(label, {'num_leaves_cap': cap})
        all_results[label] = r
        d = r['holdout_pnl'] - v2_base
        s2_results.append({'cap': cap, 'holdout_pnl': r['holdout_pnl'], 'delta': d})
        wandb.log({f's2_nl{cap}': r['holdout_pnl'], f's2_nl{cap}_delta': d})

    # === S3: min_data_in_leaf {100(base), 500, 2000} ===
    log("\n=== S3: min_data_in_leaf sweep ===")
    s3_results = [{'min_data': 100, 'holdout_pnl': v2_base, 'delta': 0.0}]
    for md in [500, 2000]:
        label = f'S3_md{md}'
        r = run_eval(label, {'min_data_in_leaf': md})
        all_results[label] = r
        d = r['holdout_pnl'] - v2_base
        s3_results.append({'min_data': md, 'holdout_pnl': r['holdout_pnl'], 'delta': d})
        wandb.log({f's3_md{md}': r['holdout_pnl'], f's3_md{md}_delta': d})

    # === S4: num_boost_round {200, 330(base), 500} ===
    log("\n=== S4: num_boost_round sweep ===")
    s4_results = [{'nbr': 330, 'holdout_pnl': v2_base, 'delta': 0.0}]
    for nbr in [200, 500]:
        label = f'S4_nbr{nbr}'
        r = run_eval(label, {}, nbr=nbr)
        all_results[label] = r
        d = r['holdout_pnl'] - v2_base
        s4_results.append({'nbr': nbr, 'holdout_pnl': r['holdout_pnl'], 'delta': d})
        wandb.log({f's4_nbr{nbr}': r['holdout_pnl'], f's4_nbr{nbr}_delta': d})

    # === Find best ===
    non_base = {k: v for k, v in all_results.items() if k != 'baseline'}
    best_name = max(non_base, key=lambda k: non_base[k]['holdout_pnl'])
    best_r = all_results[best_name]
    best_holdout = best_r['holdout_pnl']
    best_hp = best_r['hp']
    best_nbr = best_r['nbr']
    delta_vs_v2 = best_holdout - v2_base

    log(f"\n=== SUMMARY ===")
    log(f"v2 baseline holdout: {v2_base:+.4f}")
    for k, v in sorted(all_results.items(), key=lambda x: -x[1]['holdout_pnl']):
        d = v['holdout_pnl'] - v2_base
        log(f"  {k}: {v['holdout_pnl']:+.4f}  ({d:+.4f})")
    log(f"\nBest: {best_name} = {best_holdout:+.4f} (delta={delta_vs_v2:+.4f})")
    wandb.log({'best_holdout': best_holdout, 'best_delta': delta_vs_v2, 'best_name': best_name})

    # === Optional: Full retrain on 0-119 if delta > +0.5 ===
    pkg_built = False
    zip_path = None
    zip_md5 = None
    zip_size_mb = None

    if delta_vs_v2 > 0.5:
        log(f"\n=== delta={delta_vs_v2:+.4f} > 0.5 → Full retrain on date 0-119 (M7) ===")
        progress("Full retrain with best HP on date 0-119 (M7 augmentation)")

        X_full = data['X_all']
        y_full = data['y_all']
        n_total = len(X_full)
        feat_dim = X_full.shape[1]
        log(f"  Full data: {n_total:,} × {feat_dim}")

        # Augment x2 (M7 trick)
        log(f"  Allocating {2*n_total:,} rows for aug...")
        X_aug = np.empty((2 * n_total, feat_dim), dtype=np.float32)
        y_aug = np.empty(2 * n_total, dtype=np.float32)
        X_aug[:n_total] = X_full
        y_aug[:n_total] = y_full
        rng = np.random.default_rng(42)
        CHUNK = 200_000
        for ci in range(0, n_total, CHUNK):
            end = min(ci + CHUNK, n_total)
            sz = end - ci
            scales = rng.uniform(0.80, 1.20, size=(sz, feat_dim)).astype(np.float32)
            X_aug[n_total + ci:n_total + end] = X_full[ci:end] * scales
            y_aug[n_total + ci:n_total + end] = y_full[ci:end]
        del scales
        gc.collect()
        log(f"  Aug done: {len(X_aug):,} rows")

        import lightgbm as lgb
        retrain_dir = f'{OUT_DIR}/retrained_models'
        os.makedirs(retrain_dir, exist_ok=True)

        for seed in SEEDS:
            log(f"  Training seed {seed} (full 0-119)...")
            t0 = time.time()
            params = build_lgb_params(seed, best_hp)
            ds = lgb.Dataset(X_aug, label=y_aug, free_raw_data=True)
            m = lgb.train(params, ds, num_boost_round=best_nbr)
            path = f'{retrain_dir}/model_h60_seed{seed}.txt'
            m.save_model(path)
            log(f"    seed {seed}: {time.time()-t0:.1f}s")
            del m, ds
            gc.collect()

        del X_aug, y_aug
        gc.collect()

        # Build package: copy v2, replace LGB models
        pkg_dir = f'{OUT_DIR}/pkg_iter019_v2L_hp_tuned'
        os.makedirs(pkg_dir, exist_ok=True)
        for fname in ['Predictor.py', 'fast_features.py', 'fast_features_batch.py',
                      'requirements.txt', 'config.json']:
            src = f'{V2_PKG_DIR}/{fname}'
            if os.path.exists(src):
                shutil.copy2(src, f'{pkg_dir}/{fname}')

        for seed in SEEDS:
            shutil.copy2(f'{V2_PKG_DIR}/nn_h60_seed{seed}.npz', f'{pkg_dir}/nn_h60_seed{seed}.npz')

        with open(f'{V2_PKG_DIR}/thresholds.json') as f:
            thresh = json.load(f)
        thresh['_t171_note'] = f"T171 best HP: {best_hp}, nbr={best_nbr}, delta={delta_vs_v2:+.4f}"
        with open(f'{pkg_dir}/thresholds.json', 'w') as f:
            json.dump(thresh, f, indent=2)

        for seed in SEEDS:
            shutil.copy2(f'{retrain_dir}/model_h60_seed{seed}.txt',
                         f'{pkg_dir}/model_h60_seed{seed}.txt')

        today = datetime.now().strftime("%m%d%H")
        zip_path = f'{ROOT}/submission_{today}_iter019_v2L_hp_tuned.zip'
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for fname in sorted(os.listdir(pkg_dir)):
                if not fname.startswith('__') and not fname.endswith('.pyc'):
                    zf.write(f'{pkg_dir}/{fname}', fname)
        with open(zip_path, 'rb') as f:
            zip_md5 = hashlib.md5(f.read()).hexdigest()
        zip_size_mb = os.path.getsize(zip_path) / 1e6
        log(f"  Zip built: {zip_path} ({zip_size_mb:.1f} MB, md5={zip_md5})")
        pkg_built = True
        wandb.log({'pkg_built': True, 'zip_size_mb': zip_size_mb})
    else:
        log(f"\nSkip full retrain: delta={delta_vs_v2:+.4f} ≤ 0.5")

    # === Write results ===
    results = {
        "task": "T171 LGB hp sweep on TSH-FT (real holdout)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu_used": bool(_gpu_available),
        "v2_baseline_holdout": v2_base,
        "S1_lambda_l2": s1_results,
        "S2_leaves_cap": s2_results,
        "S3_min_data": s3_results,
        "S4_rounds": s4_results,
        "best_name": best_name,
        "best_combo": best_hp,
        "best_nbr": best_nbr,
        "best_holdout": best_holdout,
        "delta_vs_v2": delta_vs_v2,
        "pkg_built": pkg_built,
        "v2L_zip": zip_path,
        "zip_md5": zip_md5,
        "zip_size_mb": zip_size_mb,
        "all_results": {k: {'holdout_pnl': v['holdout_pnl'], 'elapsed': v['elapsed'],
                            'hp': v['hp'], 'nbr': v['nbr']}
                        for k, v in all_results.items()},
    }
    with open(f'{OUT_DIR}/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    log("results.json saved")

    # REPORT.md
    with open(f'{OUT_DIR}/REPORT.md', 'w') as f:
        f.write("# T171: LGB HP Sweep on Real Holdout (96-119)\n\n")
        f.write(f"Timestamp: {datetime.now(timezone.utc).isoformat()}\n\n")
        f.write(f"GPU: {_gpu_available}\n\n")
        f.write(f"## Baseline (v2 full-retrain, T150 conf)\n")
        f.write(f"- Holdout PnL: {v2_base:+.4f}\n\n")
        f.write(f"## All Results (sorted by holdout PnL)\n\n")
        f.write("| Config | Holdout PnL | Δ Baseline |\n")
        f.write("|--------|-------------|------------|\n")
        for k, v in sorted(all_results.items(), key=lambda x: -x[1]['holdout_pnl']):
            d = v['holdout_pnl'] - v2_base
            f.write(f"| {k} | {v['holdout_pnl']:+.4f} | {d:+.4f} |\n")
        f.write(f"\n## Best: {best_name}\n")
        f.write(f"- HP: `{best_hp}`, nbr={best_nbr}\n")
        f.write(f"- Holdout PnL: {best_holdout:+.4f}\n")
        f.write(f"- Delta vs v2: {delta_vs_v2:+.4f}\n")
        f.write(f"- Package built: {pkg_built}\n")
        if zip_path:
            f.write(f"- Zip: `{zip_path}`\n")

    elapsed_total = time.time() - t_start
    log(f"\nTotal elapsed: {elapsed_total/60:.1f} min")
    progress("done", metrics={'best_holdout': best_holdout, 'delta_vs_v2': delta_vs_v2})
    wandb.finish()

    print(f"\nRESULT: task=[T171 LGB HP sweep real holdout] "
          f"metrics={{v2_baseline={v2_base:.4f},"
          f"best_holdout={best_holdout:.4f},delta={delta_vs_v2:+.4f},"
          f"best_name={best_name},pkg_built={pkg_built}}} "
          f"notes=[gpu={_gpu_available}; best_hp={best_hp}; nbr={best_nbr}]", flush=True)


if __name__ == '__main__':
    main()
