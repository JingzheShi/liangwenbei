#!/usr/bin/env python3
"""T147: Feature Pruning + TSH-FT Evaluation

Protocol:
- INNER: date 0-95 (train + val)
- HOLDOUT_PROXY: date 96-119 (test)
- TSH-FT = 0.3*inner_loso + 0.7*holdout - 0.3*max(0, inner_loso-holdout)
"""
from __future__ import annotations
import os
import sys
import json
import time
import numpy as np
from datetime import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

ROOT = '/root/projects/liangwenbei_workdir'
CACHE_DIR = f'{ROOT}/experiments/T68_stage5_features/cache'
OUT_DIR = f'{ROOT}/experiments/T147_feature_pruning'
os.makedirs(OUT_DIR, exist_ok=True)

FEE = 0.0001
SEEDS = [1, 7, 13, 42, 100]
INNER_SEEDS = [42]  # 1 seed for inner LOSO (speed: 19s/fold × 4 folds = 76s vs 150s for 2-seed)

LGB_PARAMS = {
    'objective': 'regression_l2',
    'metric': 'l2',
    'learning_rate': 0.05,
    'verbose': -1,
    'num_threads': 8,
    'device': 'gpu',
    'gpu_use_dp': False,
    'num_leaves': 63,
    'min_child_samples': 20,
}
NUM_BOOST_ROUND = 330

DROP_NAMES = [
    "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
    "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
    "qrank_W100_cumspread",
    "kyle_lam_W50", "kyle_lam_W100",
    "roll_eff_spr_ratio_W100",
    "liq_asym_top5_W5",
]


def ts():
    return datetime.now().strftime('%H:%M:%S')


def progress_update(step, metrics=None):
    p = {
        "status": "running",
        "step": step,
        "metrics": metrics or {},
        "timestamp": datetime.now().isoformat(),
    }
    with open(f'{OUT_DIR}/worker-progress.json', 'w') as f:
        json.dump(p, f, indent=2)
    print(f"[{ts()}] {step}", flush=True)


def vectorized_pnl(action, mp_t, mp_th):
    side = action.astype(np.float64) - 1.0
    abs_side = np.abs(side)
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * abs_side * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee_pnl) / denom


def regr_target(mp_t, mp_th):
    return ((mp_th.astype(np.float64) - mp_t.astype(np.float64))
            / (mp_t.astype(np.float64) + 1.0)).astype(np.float32)


def load_data():
    progress_update("Loading data")
    train = np.load(f'{CACHE_DIR}/schemeP_train.npz')
    val = np.load(f'{CACHE_DIR}/schemeP_val.npz')
    test = np.load(f'{CACHE_DIR}/schemeP_test.npz')

    with open(f'{CACHE_DIR}/schemeP_feat_names.txt') as f:
        all_feat_names = [line.strip() for line in f if line.strip()]
    assert len(all_feat_names) == 370, f"Expected 370, got {len(all_feat_names)}"

    drop_set = set(DROP_NAMES)
    keep_idx = np.array([i for i, n in enumerate(all_feat_names) if n not in drop_set], dtype=np.int32)
    feat_names = [all_feat_names[i] for i in keep_idx]
    assert len(feat_names) == 359, f"Expected 359, got {len(feat_names)}"
    print(f"  Base feature set: {len(feat_names)} features", flush=True)

    X_all = np.concatenate([train['X'], val['X'], test['X']], axis=0)[:, keep_idx].astype(np.float32)
    date_all = np.concatenate([train['date'], val['date'], test['date']])
    sym_all = np.concatenate([train['sym'], val['sym'], test['sym']]).astype(np.int8)
    mp_t_all = np.concatenate([train['mp_t'], val['mp_t'], test['mp_t']]).astype(np.float64)
    mp_th_all = np.concatenate([train['mp_t60'], val['mp_t60'], test['mp_t60']]).astype(np.float64)
    y_all = regr_target(mp_t_all, mp_th_all)

    mask_inner = date_all <= 95
    mask_holdout = (date_all >= 96) & (date_all <= 119)
    print(f"  Inner: {mask_inner.sum():,}  Holdout: {mask_holdout.sum():,}", flush=True)

    data_inner = {k: v[mask_inner] for k, v in [
        ('X', X_all), ('y', y_all), ('sym', sym_all), ('mp_t', mp_t_all), ('mp_th', mp_th_all)]}
    data_holdout = {k: v[mask_holdout] for k, v in [
        ('X', X_all), ('y', y_all), ('sym', sym_all), ('mp_t', mp_t_all), ('mp_th', mp_th_all)]}

    return data_inner, data_holdout, feat_names


def train_lgb(X_tr, y_tr, X_val, seeds, num_boost_round=NUM_BOOST_ROUND):
    """Train LGB with given seeds, return averaged predictions on X_val and averaged importances."""
    import lightgbm as lgb
    preds = np.zeros(len(X_val), dtype=np.float64)
    importances = np.zeros(X_tr.shape[1], dtype=np.float64)
    for seed in seeds:
        p = {**LGB_PARAMS, 'seed': seed, 'feature_fraction_seed': seed + 1,
             'bagging_seed': seed + 2}
        ds = lgb.Dataset(X_tr, label=y_tr, free_raw_data=True)
        model = lgb.train(p, ds, num_boost_round=num_boost_round)
        preds += model.predict(X_val).astype(np.float64)
        importances += model.feature_importance(importance_type='gain').astype(np.float64)
    preds /= len(seeds)
    importances /= len(seeds)
    return preds, importances


def loso_oof(X, y, sym, inner_seeds=None):
    """4-fold sym-out LOSO: leave out syms 0,1,2,3; sym4 always in train."""
    inner_seeds = inner_seeds or INNER_SEEDS
    oof = np.zeros(len(X), dtype=np.float64)
    oof_mask = np.zeros(len(X), dtype=bool)
    fold_importances = np.zeros(X.shape[1], dtype=np.float64)

    for val_sym in [0, 1, 2, 3]:
        val_m = sym == val_sym
        tr_m = ~val_m
        t0 = time.time()
        fold_pred, fold_imp = train_lgb(X[tr_m], y[tr_m], X[val_m], inner_seeds)
        oof[val_m] = fold_pred
        oof_mask[val_m] = True
        fold_importances += fold_imp
        print(f"    LOSO fold sym={val_sym}: {val_m.sum():,} rows, {time.time()-t0:.1f}s", flush=True)

    fold_importances /= 4.0
    return oof, oof_mask, fold_importances


def run_de(oof_v, mp_t_v, mp_th_v, fast=True):
    """DE to find optimal buy/sell thresholds maximizing LOSO PnL."""
    from scipy.optimize import differential_evolution

    def neg_pnl(thresh):
        buy_t, sell_t = thresh
        action = np.ones(len(oof_v), dtype=np.int8)
        action[oof_v > buy_t] = 2
        action[oof_v < -sell_t] = 0
        return -float(vectorized_pnl(action, mp_t_v, mp_th_v).sum())

    maxiter = 40 if fast else 80
    popsize = 12 if fast else 24
    result = differential_evolution(
        neg_pnl, bounds=[(0, 0.004), (0, 0.004)],
        seed=42, maxiter=maxiter, popsize=popsize,
        workers=1, polish=True, tol=1e-7,
    )
    return float(result.x[0]), float(result.x[1]), float(-result.fun)


def eval_subset(data_inner, data_holdout, feat_idx, fast_de=True,
                inner_seeds=None, hold_seeds=None, tag=""):
    """Full TSH-FT evaluation for a feature subset."""
    inner_seeds = inner_seeds or INNER_SEEDS
    hold_seeds = hold_seeds or SEEDS
    feat_idx = np.asarray(feat_idx, dtype=np.int32)
    t0 = time.time()

    Xi = data_inner['X'][:, feat_idx]
    Xh = data_holdout['X'][:, feat_idx]
    yi = data_inner['y']

    # LOSO inner
    oof, oof_mask, loso_imp = loso_oof(Xi, yi, data_inner['sym'], inner_seeds)

    # DE threshold on OOF preds (syms 0-3 only)
    oof_v = oof[oof_mask]
    buy_t, sell_t, inner_pnl = run_de(oof_v, data_inner['mp_t'][oof_mask],
                                       data_inner['mp_th'][oof_mask], fast=fast_de)

    # Train full model on inner, predict holdout
    ho_pred, hold_imp = train_lgb(Xi, yi, Xh, hold_seeds)

    # Apply FROZEN threshold to holdout
    ho_action = np.ones(len(ho_pred), dtype=np.int8)
    ho_action[ho_pred > buy_t] = 2
    ho_action[ho_pred < -sell_t] = 0
    holdout_pnl = float(vectorized_pnl(ho_action, data_holdout['mp_t'], data_holdout['mp_th']).sum())

    tsh_ft = 0.3 * inner_pnl + 0.7 * holdout_pnl - 0.3 * max(0.0, inner_pnl - holdout_pnl)
    elapsed = time.time() - t0

    print(f"  [{tag}] n={len(feat_idx)}: inner={inner_pnl:+.4f} holdout={holdout_pnl:+.4f} "
          f"tsh_ft={tsh_ft:+.4f} buy={buy_t:.6f} sell={sell_t:.6f} ({elapsed:.0f}s)", flush=True)

    return {
        'inner_loso': inner_pnl,
        'holdout': holdout_pnl,
        'tsh_ft': tsh_ft,
        'buy_thresh': buy_t,
        'sell_thresh': sell_t,
        'n_features': len(feat_idx),
        'loso_importances': loso_imp.tolist(),
        'hold_importances': hold_imp.tolist(),
    }


def main():
    import wandb
    try:
        run = wandb.init(
            project="T147-feature-pruning",
            entity="cjxh21-Tsinghua University",
            config={"inner_seeds": INNER_SEEDS, "hold_seeds": SEEDS, "num_boost_round": NUM_BOOST_ROUND},
        )
        USE_WANDB = True
    except Exception as e:
        print(f"  WandB init failed: {e}. Continuing without WandB.", flush=True)
        USE_WANDB = False
        class _NullWandB:
            def log(self, *a, **kw): pass
            def finish(self): pass
        wandb = _NullWandB()

    data_inner, data_holdout, feat_names = load_data()
    n_base = len(feat_names)  # 359
    all_results = {}

    # ==========================================
    # BASELINE: all 359 features
    # ==========================================
    progress_update("Baseline: all 359 features")
    baseline_res = eval_subset(data_inner, data_holdout, np.arange(n_base),
                                fast_de=True, tag="baseline")
    all_results['baseline_359'] = {
        'inner_loso': baseline_res['inner_loso'],
        'holdout': baseline_res['holdout'],
        'tsh_ft': baseline_res['tsh_ft'],
        'n_features': n_base,
    }
    wandb.log({'baseline_inner_loso': baseline_res['inner_loso'],
               'baseline_holdout': baseline_res['holdout'],
               'baseline_tsh_ft': baseline_res['tsh_ft']})

    # Gain importances from holdout training (5 seeds) on all features
    base_gain = np.array(baseline_res['hold_importances'], dtype=np.float64)
    base_rank = np.argsort(base_gain)[::-1]  # highest importance first
    print(f"  Top-10 features by gain: {[feat_names[i] for i in base_rank[:10]]}", flush=True)

    # Save progress
    with open(f'{OUT_DIR}/results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # ==========================================
    # METHOD A: LGB Gain Importance Pruning
    # ==========================================
    progress_update("Method A: LGB importance pruning")
    method_a = {}
    best_a = None

    for k in [50, 100, 150, 200, 250]:
        feat_idx_k = base_rank[:k]
        tag = f"A_k{k}"
        progress_update(f"Method A: k={k}")
        res = eval_subset(data_inner, data_holdout, feat_idx_k, fast_de=True, tag=tag)
        method_a[f'k{k}'] = {
            'inner_loso': res['inner_loso'],
            'holdout': res['holdout'],
            'tsh_ft': res['tsh_ft'],
            'n_features': k,
        }
        wandb.log({'method': 'A', 'k': k, 'inner_loso': res['inner_loso'],
                   'holdout': res['holdout'], 'tsh_ft': res['tsh_ft']})
        if best_a is None or res['tsh_ft'] > best_a['tsh_ft']:
            best_a = {'k': k, 'tsh_ft': res['tsh_ft'], 'feat_idx': feat_idx_k.tolist(),
                      'inner_loso': res['inner_loso'], 'holdout': res['holdout']}

    method_a['best'] = best_a
    all_results['method_A'] = method_a
    with open(f'{OUT_DIR}/results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  Method A best: k={best_a['k']} tsh_ft={best_a['tsh_ft']:+.4f}", flush=True)

    # ==========================================
    # METHOD B: Correlation Cluster Pruning
    # ==========================================
    progress_update("Method B: Correlation cluster pruning")

    # Sample 50k rows for correlation matrix
    rng = np.random.default_rng(42)
    n_inner = len(data_inner['X'])
    sample_idx = rng.choice(n_inner, size=min(50000, n_inner), replace=False)
    X_sample = data_inner['X'][sample_idx]

    print("  Computing correlation matrix...", flush=True)
    t_c = time.time()
    # Compute correlation in chunks to save memory
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    # Standardize
    mu = X_sample.mean(axis=0, keepdims=True)
    sigma = X_sample.std(axis=0, keepdims=True) + 1e-8
    X_std = (X_sample - mu) / sigma

    # Correlation matrix
    corr = (X_std.T @ X_std) / len(X_sample)  # (359, 359)
    corr = np.clip(corr, -1, 1)
    print(f"  Corr matrix done in {time.time()-t_c:.1f}s", flush=True)

    # Distance = 1 - |corr|
    dist_mat = 1.0 - np.abs(corr)
    np.fill_diagonal(dist_mat, 0)
    dist_mat = (dist_mat + dist_mat.T) / 2  # ensure symmetry
    condensed = squareform(dist_mat, checks=False)

    Z = linkage(condensed, method='average')
    cluster_labels = fcluster(Z, t=0.1, criterion='distance')  # 0.1 = |corr| >= 0.9 threshold
    n_clusters = cluster_labels.max()
    print(f"  Hierarchical clustering: {n_clusters} clusters from {n_base} features (threshold |corr|>=0.9)", flush=True)

    # Within each cluster, keep the feature with highest gain importance
    cluster_reps = []
    for c in range(1, n_clusters + 1):
        cluster_members = np.where(cluster_labels == c)[0]
        if len(cluster_members) == 1:
            cluster_reps.append(int(cluster_members[0]))
        else:
            best_in_cluster = cluster_members[np.argmax(base_gain[cluster_members])]
            cluster_reps.append(int(best_in_cluster))

    b_feat_idx = np.array(cluster_reps, dtype=np.int32)
    print(f"  Method B: {len(b_feat_idx)} features after cluster dedup", flush=True)
    progress_update(f"Method B: evaluating {len(b_feat_idx)} features")
    b_res = eval_subset(data_inner, data_holdout, b_feat_idx, fast_de=True, tag="B")
    all_results['method_B'] = {
        'inner_loso': b_res['inner_loso'],
        'holdout': b_res['holdout'],
        'tsh_ft': b_res['tsh_ft'],
        'n_features': len(b_feat_idx),
        'n_clusters': int(n_clusters),
        'feat_idx': b_feat_idx.tolist(),
    }
    wandb.log({'method': 'B', 'k': len(b_feat_idx), 'inner_loso': b_res['inner_loso'],
               'holdout': b_res['holdout'], 'tsh_ft': b_res['tsh_ft']})
    with open(f'{OUT_DIR}/results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # ==========================================
    # METHOD C: Adversarial Feature Filter
    # ==========================================
    progress_update("Method C: Adversarial feature filter")
    import lightgbm as lgb

    # Label: 1 if date >= 96 (holdout), 0 if date < 96 (inner)
    # Combine inner + holdout for adversarial training
    X_adv = np.concatenate([data_inner['X'], data_holdout['X']], axis=0)
    y_adv = np.concatenate([
        np.zeros(len(data_inner['X']), dtype=np.float32),
        np.ones(len(data_holdout['X']), dtype=np.float32),
    ])

    adv_params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'min_child_samples': 20,
        'verbose': -1,
        'num_threads': 8,
        'device': 'gpu',
        'gpu_use_dp': False,
        'seed': 42,
    }
    print("  Training adversarial classifier...", flush=True)
    t_adv = time.time()
    adv_ds = lgb.Dataset(X_adv, label=y_adv, free_raw_data=True)
    adv_model = lgb.train(adv_params, adv_ds, num_boost_round=200)
    adv_imp = adv_model.feature_importance(importance_type='gain').astype(np.float64)
    print(f"  Adversarial training done in {time.time()-t_adv:.1f}s", flush=True)

    # Sort by drift score (high = more drift-susceptible = drop first)
    drift_rank = np.argsort(adv_imp)[::-1]  # highest drift first

    method_c = {}
    best_c = None
    for drop_k in [50, 100, 150]:
        drop_set_idx = set(drift_rank[:drop_k].tolist())
        keep_idx_c = np.array([i for i in range(n_base) if i not in drop_set_idx], dtype=np.int32)
        tag = f"C_drop{drop_k}"
        progress_update(f"Method C: drop top-{drop_k} drift features ({n_base - drop_k} remain)")
        res = eval_subset(data_inner, data_holdout, keep_idx_c, fast_de=True, tag=tag)
        method_c[f'drop{drop_k}'] = {
            'inner_loso': res['inner_loso'],
            'holdout': res['holdout'],
            'tsh_ft': res['tsh_ft'],
            'n_features': len(keep_idx_c),
        }
        wandb.log({'method': 'C', 'drop_k': drop_k, 'inner_loso': res['inner_loso'],
                   'holdout': res['holdout'], 'tsh_ft': res['tsh_ft']})
        if best_c is None or res['tsh_ft'] > best_c['tsh_ft']:
            best_c = {'drop_k': drop_k, 'tsh_ft': res['tsh_ft'], 'feat_idx': keep_idx_c.tolist(),
                      'inner_loso': res['inner_loso'], 'holdout': res['holdout']}

    method_c['best'] = best_c
    all_results['method_C'] = method_c
    # Save drift rank for method E
    all_results['_drift_rank'] = drift_rank.tolist()
    with open(f'{OUT_DIR}/results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  Method C best: drop_k={best_c['drop_k']} tsh_ft={best_c['tsh_ft']:+.4f}", flush=True)

    # ==========================================
    # METHOD D: Mutual Information Pruning
    # ==========================================
    progress_update("Method D: Mutual information pruning")
    from sklearn.feature_selection import mutual_info_regression

    # Sample 30k rows for speed
    mi_idx = rng.choice(n_inner, size=min(30000, n_inner), replace=False)
    X_mi = data_inner['X'][mi_idx].astype(np.float32)
    y_mi = data_inner['y'][mi_idx].astype(np.float32)

    print("  Computing MI scores...", flush=True)
    t_mi = time.time()
    mi_scores = mutual_info_regression(X_mi, y_mi, random_state=42, n_neighbors=5)
    print(f"  MI computation done in {time.time()-t_mi:.1f}s", flush=True)

    # Sort ascending (lowest MI = drop first)
    mi_rank_asc = np.argsort(mi_scores)  # lowest MI first

    method_d = {}
    best_d = None
    for drop_k in [50, 100, 150, 200]:
        drop_mi_idx = set(mi_rank_asc[:drop_k].tolist())
        keep_idx_d = np.array([i for i in range(n_base) if i not in drop_mi_idx], dtype=np.int32)
        tag = f"D_drop{drop_k}"
        progress_update(f"Method D: drop bottom-{drop_k} MI features ({n_base - drop_k} remain)")
        res = eval_subset(data_inner, data_holdout, keep_idx_d, fast_de=True, tag=tag)
        method_d[f'drop{drop_k}'] = {
            'inner_loso': res['inner_loso'],
            'holdout': res['holdout'],
            'tsh_ft': res['tsh_ft'],
            'n_features': len(keep_idx_d),
        }
        wandb.log({'method': 'D', 'drop_k': drop_k, 'inner_loso': res['inner_loso'],
                   'holdout': res['holdout'], 'tsh_ft': res['tsh_ft']})
        if best_d is None or res['tsh_ft'] > best_d['tsh_ft']:
            best_d = {'drop_k': drop_k, 'tsh_ft': res['tsh_ft'], 'feat_idx': keep_idx_d.tolist(),
                      'inner_loso': res['inner_loso'], 'holdout': res['holdout']}

    method_d['best'] = best_d
    all_results['method_D'] = method_d
    all_results['_mi_scores'] = mi_scores.tolist()
    with open(f'{OUT_DIR}/results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  Method D best: drop_k={best_d['drop_k']} tsh_ft={best_d['tsh_ft']:+.4f}", flush=True)

    # ==========================================
    # METHOD E: Combined (Corr-cluster + Adversarial)
    # ==========================================
    progress_update("Method E: Combined corr-cluster + adversarial")
    # Start from Method B's representative set, then drop top-100 drift features
    drift_top100 = set(drift_rank[:100].tolist())
    e_feat_idx = np.array([i for i in b_feat_idx if i not in drift_top100], dtype=np.int32)
    print(f"  Method E: {len(e_feat_idx)} features (B={len(b_feat_idx)}, after dropping drift-top100)", flush=True)
    e_res = eval_subset(data_inner, data_holdout, e_feat_idx, fast_de=True, tag="E")
    all_results['method_E'] = {
        'inner_loso': e_res['inner_loso'],
        'holdout': e_res['holdout'],
        'tsh_ft': e_res['tsh_ft'],
        'n_features': len(e_feat_idx),
        'feat_idx': e_feat_idx.tolist(),
    }
    wandb.log({'method': 'E', 'k': len(e_feat_idx), 'inner_loso': e_res['inner_loso'],
               'holdout': e_res['holdout'], 'tsh_ft': e_res['tsh_ft']})
    with open(f'{OUT_DIR}/results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # ==========================================
    # Find overall best
    # ==========================================
    progress_update("Comparing all methods")
    candidates = [
        ('baseline', 'all359', all_results['baseline_359']['tsh_ft'],
         list(range(n_base))),
        ('A', best_a['k'], best_a['tsh_ft'], best_a['feat_idx']),
        ('B', len(b_feat_idx), b_res['tsh_ft'], b_feat_idx.tolist()),
        ('C', n_base - best_c['drop_k'], best_c['tsh_ft'], best_c['feat_idx']),
        ('D', n_base - best_d['drop_k'], best_d['tsh_ft'], best_d['feat_idx']),
        ('E', len(e_feat_idx), e_res['tsh_ft'], e_feat_idx.tolist()),
    ]
    candidates.sort(key=lambda x: x[2], reverse=True)
    best_method, best_k, best_score, best_feat_idx_list = candidates[0]
    print(f"\n=== RESULTS SUMMARY ===", flush=True)
    for m, k, score, _ in candidates:
        print(f"  {m:12s} n={str(k):5s} tsh_ft={score:+.4f}", flush=True)
    print(f"\n  BEST: method={best_method} n_features={best_k} tsh_ft={best_score:+.4f}", flush=True)

    best_feat_names = [feat_names[i] for i in best_feat_idx_list]
    all_results['overall_best'] = {
        'method': best_method,
        'n_features': len(best_feat_idx_list),
        'tsh_ft': best_score,
        'features': best_feat_names,
    }

    # Re-evaluate best with full DE for final score
    if best_method != 'baseline':
        progress_update(f"Re-evaluating best subset with full DE: {best_method}")
        final_res = eval_subset(data_inner, data_holdout, np.array(best_feat_idx_list, dtype=np.int32),
                                fast_de=False, tag=f"final_{best_method}")
        all_results['overall_best'].update({
            'inner_loso': final_res['inner_loso'],
            'holdout': final_res['holdout'],
            'tsh_ft': final_res['tsh_ft'],
            'buy_thresh': final_res['buy_thresh'],
            'sell_thresh': final_res['sell_thresh'],
        })
        print(f"  Final (full DE): inner={final_res['inner_loso']:+.4f} holdout={final_res['holdout']:+.4f} tsh_ft={final_res['tsh_ft']:+.4f}", flush=True)
    else:
        # Baseline already evaluated
        all_results['overall_best'].update({
            'inner_loso': all_results['baseline_359']['inner_loso'],
            'holdout': all_results['baseline_359']['holdout'],
        })

    # Save final results
    with open(f'{OUT_DIR}/results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else str(x))

    # Save best features list
    with open(f'{OUT_DIR}/best_features.txt', 'w') as f:
        f.write('\n'.join(best_feat_names) + '\n')
    print(f"  Saved best_features.txt ({len(best_feat_names)} features)", flush=True)

    # Write report
    write_report(all_results, feat_names, base_gain, base_rank, candidates)

    # Final wandb log
    best = all_results['overall_best']
    wandb.log({
        'best_method': best['method'],
        'best_n_features': best['n_features'],
        'best_tsh_ft': best['tsh_ft'],
        'best_inner_loso': best.get('inner_loso', 0),
        'best_holdout': best.get('holdout', 0),
    })
    wandb.finish()

    progress_update("DONE", {"best_tsh_ft": best['tsh_ft'], "best_method": best['method']})

    inner_l = best.get('inner_loso', all_results['baseline_359']['inner_loso'])
    holdout_l = best.get('holdout', all_results['baseline_359']['holdout'])
    print(f"\nRESULT: task=[T147-feature-pruning] "
          f"metrics={{best_tsh_ft={best['tsh_ft']:.4f}, best_method={best['method']}, "
          f"best_n_features={best['n_features']}, "
          f"inner_loso={inner_l:.4f}, holdout={holdout_l:.4f}}} "
          f"notes=[Evaluated {len(candidates)} methods; baseline vs pruned feature sets]",
          flush=True)


def write_report(all_results, feat_names, base_gain, base_rank, candidates):
    b = all_results['baseline_359']
    ob = all_results['overall_best']
    lines = [
        "# T147: Feature Pruning + TSH-FT Evaluation",
        "",
        "## Summary",
        "",
        f"Evaluated {len(feat_names)} base features across 5 pruning methods.",
        f"TSH-FT = 0.3×inner_LOSO + 0.7×holdout − 0.3×max(0, inner−holdout)",
        "",
        "## Results",
        "",
        "| Method | N features | Inner LOSO | Holdout | TSH-FT |",
        "|--------|-----------|-----------|---------|--------|",
        f"| Baseline (all 359) | 359 | {b['inner_loso']:+.4f} | {b['holdout']:+.4f} | {b['tsh_ft']:+.4f} |",
    ]
    ma = all_results.get('method_A', {})
    for k in [50, 100, 150, 200, 250]:
        v = ma.get(f'k{k}', {})
        if v:
            lines.append(f"| A (top-{k} gain) | {k} | {v['inner_loso']:+.4f} | {v['holdout']:+.4f} | {v['tsh_ft']:+.4f} |")

    mb = all_results.get('method_B', {})
    if mb and 'n_clusters' in mb:
        lines.append(f"| B (corr-cluster) | {mb['n_features']} | {mb['inner_loso']:+.4f} | {mb['holdout']:+.4f} | {mb['tsh_ft']:+.4f} |")

    mc = all_results.get('method_C', {})
    for drop_k in [50, 100, 150]:
        v = mc.get(f'drop{drop_k}', {})
        if v:
            lines.append(f"| C (drop {drop_k} drift) | {v['n_features']} | {v['inner_loso']:+.4f} | {v['holdout']:+.4f} | {v['tsh_ft']:+.4f} |")

    md = all_results.get('method_D', {})
    for drop_k in [50, 100, 150, 200]:
        v = md.get(f'drop{drop_k}', {})
        if v:
            lines.append(f"| D (drop {drop_k} low-MI) | {v['n_features']} | {v['inner_loso']:+.4f} | {v['holdout']:+.4f} | {v['tsh_ft']:+.4f} |")

    me = all_results.get('method_E', {})
    if me:
        lines.append(f"| E (B+C combined) | {me['n_features']} | {me['inner_loso']:+.4f} | {me['holdout']:+.4f} | {me['tsh_ft']:+.4f} |")

    lines += [
        "",
        "## Best Result",
        "",
        f"**Method {ob['method']}**: {ob['n_features']} features, TSH-FT = {ob['tsh_ft']:+.4f}",
        f"- Inner LOSO: {ob.get('inner_loso', 'N/A'):+.4f}",
        f"- Holdout: {ob.get('holdout', 'N/A'):+.4f}",
        "",
        "## Top-20 Features by Gain Importance",
        "",
    ]
    for rank, i in enumerate(base_rank[:20]):
        lines.append(f"{rank+1}. {feat_names[i]} (gain={base_gain[i]:.1f})")

    lines += [
        "",
        "## Recommendation",
        "",
        f"Use best_features.txt ({ob['n_features']} features) for next iteration packaging.",
        f"TSH-FT improvement over baseline: {ob['tsh_ft'] - b['tsh_ft']:+.4f}",
    ]

    with open(f'{OUT_DIR}/REPORT.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Saved REPORT.md", flush=True)


if __name__ == '__main__':
    main()
