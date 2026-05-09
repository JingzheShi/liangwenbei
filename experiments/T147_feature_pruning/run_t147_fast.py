#!/usr/bin/env python3
"""T147 Fast: 2-Phase Feature Pruning + TSH-FT Evaluation

Phase 1: Screen all subsets with 1-seed holdout + frozen baseline thresholds (fast)
Phase 2: Full TSH-FT (LOSO + DE + 2-seed holdout) for top-5 candidates

Baseline thresholds from completed baseline run:
  buy=0.002717  sell=0.000288
  inner=+14.1006  holdout=+16.8739  tsh_ft=+16.0419
"""
from __future__ import annotations
import os
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
SCREEN_SEEDS = [42]       # 1 seed for phase 1 screening
FULL_SEEDS = [1, 42]      # 2 seeds for phase 2 holdout
LOSO_SEEDS = [42]         # 1 seed for LOSO

# Baseline thresholds (from completed baseline run)
BASELINE_BUY = 0.002717
BASELINE_SELL = 0.000288
BASELINE_INNER = 14.1006
BASELINE_HOLDOUT = 16.8739
BASELINE_TSH_FT = 16.0419

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
FAST_ROUNDS = 180  # for LOSO inner (faster, less precise)

DROP_NAMES = [
    "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
    "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
    "qrank_W100_cumspread", "kyle_lam_W50", "kyle_lam_W100",
    "roll_eff_spr_ratio_W100", "liq_asym_top5_W5",
]


def ts():
    return datetime.now().strftime('%H:%M:%S')


def progress_update(step, metrics=None):
    p = {"status": "running", "step": step, "metrics": metrics or {},
         "timestamp": datetime.now().isoformat()}
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
    drop_set = set(DROP_NAMES)
    keep_idx = np.array([i for i, n in enumerate(all_feat_names) if n not in drop_set], dtype=np.int32)
    feat_names = [all_feat_names[i] for i in keep_idx]
    assert len(feat_names) == 359

    X_all = np.concatenate([train['X'], val['X'], test['X']], axis=0)[:, keep_idx].astype(np.float32)
    # LightGBM handles NaN natively; but correlation/MI need finite values
    nan_count = np.isnan(X_all).sum()
    if nan_count > 0:
        print(f"  Warning: {nan_count:,} NaN values in X, will handle per-analysis", flush=True)
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


def train_lgb(X_tr, y_tr, X_val, seeds, num_boost_round=NUM_BOOST_ROUND, get_importance=False):
    import lightgbm as lgb
    preds = np.zeros(len(X_val), dtype=np.float64)
    importances = np.zeros(X_tr.shape[1], dtype=np.float64) if get_importance else None
    for seed in seeds:
        p = {**LGB_PARAMS, 'seed': seed, 'feature_fraction_seed': seed + 1,
             'bagging_seed': seed + 2}
        ds = lgb.Dataset(X_tr, label=y_tr, free_raw_data=True)
        model = lgb.train(p, ds, num_boost_round=num_boost_round)
        preds += model.predict(X_val).astype(np.float64)
        if get_importance:
            importances += model.feature_importance(importance_type='gain').astype(np.float64)
    preds /= len(seeds)
    if get_importance:
        importances /= len(seeds)
    return preds, importances


def screen_subset(Xi, Xh, yi, mp_t_h, mp_th_h, buy_t=BASELINE_BUY, sell_t=BASELINE_SELL, tag=""):
    """Phase 1 fast screening: 1 seed holdout + frozen thresholds."""
    t0 = time.time()
    ho_pred, _ = train_lgb(Xi, yi, Xh, SCREEN_SEEDS, num_boost_round=NUM_BOOST_ROUND)
    ho_action = np.ones(len(ho_pred), dtype=np.int8)
    ho_action[ho_pred > buy_t] = 2
    ho_action[ho_pred < -sell_t] = 0
    holdout_pnl = float(vectorized_pnl(ho_action, mp_t_h, mp_th_h).sum())
    print(f"  [screen {tag}] n={Xi.shape[1]}: holdout_pnl={holdout_pnl:+.4f} ({time.time()-t0:.1f}s)", flush=True)
    return holdout_pnl


def loso_oof(Xi, yi, sym, seeds=None, num_boost_round=FAST_ROUNDS):
    """4-fold sym-out LOSO inner OOF."""
    seeds = seeds or LOSO_SEEDS
    oof = np.zeros(len(Xi), dtype=np.float64)
    oof_mask = np.zeros(len(Xi), dtype=bool)
    for val_sym in [0, 1, 2, 3]:
        val_m = sym == val_sym
        tr_m = ~val_m
        fold_pred, _ = train_lgb(Xi[tr_m], yi[tr_m], Xi[val_m], seeds, num_boost_round=num_boost_round)
        oof[val_m] = fold_pred
        oof_mask[val_m] = True
        print(f"    fold sym={val_sym}: {val_m.sum():,} rows done", flush=True)
    return oof, oof_mask


def run_de(oof_v, mp_t_v, mp_th_v, fast=True):
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
        seed=42, maxiter=maxiter, popsize=popsize, workers=1, polish=True, tol=1e-7)
    return float(result.x[0]), float(result.x[1]), float(-result.fun)


def full_tsh_ft(data_inner, data_holdout, feat_idx, fast_de=True, tag=""):
    """Full TSH-FT: LOSO inner + DE + 2-seed holdout."""
    feat_idx = np.asarray(feat_idx, dtype=np.int32)
    t0 = time.time()
    Xi = data_inner['X'][:, feat_idx]
    Xh = data_holdout['X'][:, feat_idx]
    yi = data_inner['y']
    # LOSO inner
    oof, oof_mask = loso_oof(Xi, yi, data_inner['sym'])
    # DE on OOF
    oof_v = oof[oof_mask]
    buy_t, sell_t, inner_pnl = run_de(oof_v, data_inner['mp_t'][oof_mask],
                                       data_inner['mp_th'][oof_mask], fast=fast_de)
    # Holdout prediction (2 seeds, full rounds)
    ho_pred, _ = train_lgb(Xi, yi, Xh, FULL_SEEDS, num_boost_round=NUM_BOOST_ROUND)
    ho_action = np.ones(len(ho_pred), dtype=np.int8)
    ho_action[ho_pred > buy_t] = 2
    ho_action[ho_pred < -sell_t] = 0
    holdout_pnl = float(vectorized_pnl(ho_action, data_holdout['mp_t'], data_holdout['mp_th']).sum())
    tsh_ft = 0.3 * inner_pnl + 0.7 * holdout_pnl - 0.3 * max(0.0, inner_pnl - holdout_pnl)
    elapsed = time.time() - t0
    print(f"  [{tag}] n={len(feat_idx)}: inner={inner_pnl:+.4f} holdout={holdout_pnl:+.4f} "
          f"tsh_ft={tsh_ft:+.4f} ({elapsed:.0f}s)", flush=True)
    return {'inner_loso': inner_pnl, 'holdout': holdout_pnl, 'tsh_ft': tsh_ft,
            'buy_thresh': buy_t, 'sell_thresh': sell_t, 'n_features': len(feat_idx)}


def main():
    t_start = time.time()
    data_inner, data_holdout, feat_names = load_data()
    n_base = len(feat_names)
    rng = np.random.default_rng(42)

    # ===== GET GAIN IMPORTANCES (1-seed fast training on full inner) =====
    progress_update("Computing gain importances (1 seed, full inner)")
    t0 = time.time()
    _, base_gain = train_lgb(data_inner['X'], data_inner['y'],
                              data_holdout['X'], [42], get_importance=True)
    print(f"  Importance training done in {time.time()-t0:.1f}s", flush=True)
    base_rank = np.argsort(base_gain)[::-1]  # highest importance first
    print(f"  Top-10: {[feat_names[i] for i in base_rank[:10]]}", flush=True)

    # ===== COMPUTE FEATURE SETS FOR ALL METHODS =====
    # Method A: gain importance top-k
    method_a_sets = {
        'k50': base_rank[:50].tolist(),
        'k100': base_rank[:100].tolist(),
        'k150': base_rank[:150].tolist(),
        'k200': base_rank[:200].tolist(),
        'k250': base_rank[:250].tolist(),
    }

    # Method B: correlation cluster dedup
    progress_update("Computing correlation matrix for Method B")
    sample_idx = rng.choice(len(data_inner['X']), size=50000, replace=False)
    X_sample = data_inner['X'][sample_idx].astype(np.float64)
    X_sample = np.where(np.isfinite(X_sample), X_sample, 0.0)  # replace NaN/inf with 0
    mu = X_sample.mean(axis=0)
    sigma = X_sample.std(axis=0) + 1e-8
    X_std = (X_sample - mu) / sigma
    corr = (X_std.T @ X_std) / len(X_sample)
    corr = np.clip(corr, -1, 1)
    np.fill_diagonal(corr, 1.0)  # ensure diagonal is 1
    corr = np.nan_to_num(corr, nan=0.0)  # handle constant features
    dist_mat = 1.0 - np.abs(corr)
    np.fill_diagonal(dist_mat, 0)
    dist_mat = np.clip((dist_mat + dist_mat.T) / 2, 0, 1)  # ensure [0,1] range
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    condensed = squareform(dist_mat, checks=False)
    condensed = np.clip(condensed, 0, None)  # ensure non-negative
    Z = linkage(condensed, method='average')
    cluster_labels = fcluster(Z, t=0.1, criterion='distance')
    n_clusters = cluster_labels.max()
    cluster_reps = []
    for c in range(1, n_clusters + 1):
        members = np.where(cluster_labels == c)[0]
        cluster_reps.append(int(members[np.argmax(base_gain[members])]))
    b_feat_idx = sorted(set(cluster_reps))
    print(f"  Method B: {len(b_feat_idx)} features after corr dedup (threshold |corr|>=0.9)", flush=True)
    method_b_sets = {'corr_dedup': b_feat_idx}

    # Method C: adversarial
    progress_update("Adversarial feature filter (Method C)")
    import lightgbm as lgb
    X_adv = np.concatenate([data_inner['X'], data_holdout['X']], axis=0)
    y_adv = np.concatenate([np.zeros(len(data_inner['X'])), np.ones(len(data_holdout['X']))]).astype(np.float32)
    adv_params = {**LGB_PARAMS, 'objective': 'binary', 'metric': 'auc', 'seed': 42}
    adv_ds = lgb.Dataset(X_adv, label=y_adv, free_raw_data=True)
    adv_model = lgb.train(adv_params, adv_ds, num_boost_round=150)
    adv_imp = adv_model.feature_importance(importance_type='gain').astype(np.float64)
    drift_rank = np.argsort(adv_imp)[::-1]  # highest drift first
    print(f"  Adversarial done. Top drift features: {[feat_names[i] for i in drift_rank[:5]]}", flush=True)
    method_c_sets = {}
    for drop_k in [50, 100, 150]:
        drop_set = set(drift_rank[:drop_k].tolist())
        method_c_sets[f'drop{drop_k}'] = [i for i in range(n_base) if i not in drop_set]

    # Method D: MI pruning
    progress_update("Mutual information pruning (Method D)")
    from sklearn.feature_selection import mutual_info_regression
    mi_idx = rng.choice(len(data_inner['X']), size=30000, replace=False)
    X_mi = data_inner['X'][mi_idx].astype(np.float32)
    X_mi = np.where(np.isfinite(X_mi), X_mi, 0.0)  # handle NaN for sklearn
    y_mi = data_inner['y'][mi_idx].astype(np.float32)
    t_mi = time.time()
    mi_scores = mutual_info_regression(X_mi, y_mi, random_state=42, n_neighbors=5)
    print(f"  MI done in {time.time()-t_mi:.1f}s", flush=True)
    mi_rank_asc = np.argsort(mi_scores)  # lowest MI first
    method_d_sets = {}
    for drop_k in [50, 100, 150, 200]:
        drop_set = set(mi_rank_asc[:drop_k].tolist())
        method_d_sets[f'drop{drop_k}'] = [i for i in range(n_base) if i not in drop_set]

    # Method E: combined
    drift_top100 = set(drift_rank[:100].tolist())
    e_feat_idx = [i for i in b_feat_idx if i not in drift_top100]
    method_e_sets = {'combined': e_feat_idx}

    # All candidates for screening
    all_candidates = {}
    for k, idx in method_a_sets.items():
        all_candidates[f'A_{k}'] = idx
    for k, idx in method_b_sets.items():
        all_candidates[f'B_{k}'] = idx
    for k, idx in method_c_sets.items():
        all_candidates[f'C_{k}'] = idx
    for k, idx in method_d_sets.items():
        all_candidates[f'D_{k}'] = idx
    for k, idx in method_e_sets.items():
        all_candidates[f'E_{k}'] = idx

    print(f"\nTotal candidates to screen: {len(all_candidates)}", flush=True)

    # ===== PHASE 1: FAST SCREENING =====
    progress_update(f"Phase 1: screening {len(all_candidates)} subsets with frozen thresholds")
    screen_results = {}
    for tag, feat_idx in all_candidates.items():
        feat_idx_arr = np.array(feat_idx, dtype=np.int32)
        Xi = data_inner['X'][:, feat_idx_arr]
        Xh = data_holdout['X'][:, feat_idx_arr]
        holdout_pnl = screen_subset(Xi, Xh, data_inner['y'],
                                     data_holdout['mp_t'], data_holdout['mp_th'], tag=tag)
        screen_results[tag] = {'holdout_pnl': holdout_pnl, 'n_features': len(feat_idx),
                                'feat_idx': feat_idx}

    # Sort by holdout PnL descending
    ranked = sorted(screen_results.items(), key=lambda x: x[1]['holdout_pnl'], reverse=True)
    print(f"\n=== PHASE 1 SCREENING RESULTS ===", flush=True)
    for name, res in ranked:
        print(f"  {name:15s} n={res['n_features']:4d} holdout_pnl={res['holdout_pnl']:+.4f}", flush=True)

    # Baseline holdout with frozen threshold for comparison
    print(f"  {'BASELINE':15s} n={n_base:4d} holdout_pnl={BASELINE_HOLDOUT:+.4f} (reference)", flush=True)

    # ===== PHASE 2: FULL TSH-FT FOR TOP-5 CANDIDATES =====
    top5 = [name for name, _ in ranked[:5]]
    progress_update(f"Phase 2: full TSH-FT for top-5: {top5}")

    all_results = {
        'baseline_359': {
            'inner_loso': BASELINE_INNER,
            'holdout': BASELINE_HOLDOUT,
            'tsh_ft': BASELINE_TSH_FT,
            'n_features': n_base,
            'buy_thresh': BASELINE_BUY,
            'sell_thresh': BASELINE_SELL,
        },
        'screen_results': {k: {'holdout_pnl': v['holdout_pnl'], 'n_features': v['n_features']}
                           for k, v in screen_results.items()},
    }

    phase2_results = {}
    for name in top5:
        feat_idx = screen_results[name]['feat_idx']
        progress_update(f"Phase 2: full TSH-FT for {name} ({len(feat_idx)} features)")
        res = full_tsh_ft(data_inner, data_holdout, np.array(feat_idx, dtype=np.int32),
                          fast_de=False, tag=name)
        phase2_results[name] = res
        # Save intermediate
        all_results['phase2'] = phase2_results
        with open(f'{OUT_DIR}/results.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

    # Find overall best
    best_name = max(phase2_results.items(), key=lambda x: x[1]['tsh_ft'])[0]
    best_res = phase2_results[best_name]
    best_feat_idx = screen_results[best_name]['feat_idx']
    best_feat_names = [feat_names[i] for i in best_feat_idx]

    print(f"\n=== PHASE 2 TSH-FT RESULTS ===", flush=True)
    for name, res in sorted(phase2_results.items(), key=lambda x: x[1]['tsh_ft'], reverse=True):
        print(f"  {name:15s} n={res['n_features']:4d} inner={res['inner_loso']:+.4f} "
              f"holdout={res['holdout']:+.4f} tsh_ft={res['tsh_ft']:+.4f}", flush=True)
    print(f"  {'BASELINE':15s} n={n_base:4d} inner={BASELINE_INNER:+.4f} "
          f"holdout={BASELINE_HOLDOUT:+.4f} tsh_ft={BASELINE_TSH_FT:+.4f}", flush=True)
    print(f"\n  BEST: {best_name}, n={best_res['n_features']}, TSH-FT={best_res['tsh_ft']:+.4f}", flush=True)

    # Save complete results
    method_summary = {}
    for k, idx in method_a_sets.items():
        sr = screen_results.get(f'A_{k}', {})
        p2 = phase2_results.get(f'A_{k}', {})
        method_summary.setdefault('method_A', {})[k] = {
            **sr.get('', {}),
            'n_features': len(idx),
            'screen_holdout': sr.get('holdout_pnl', None),
            **{kk: vv for kk, vv in p2.items() if kk != 'n_features'},
        }
    # Similar for B,C,D,E...

    all_results.update({
        'method_A': {k: {
            'n_features': len(idx),
            'screen_holdout': screen_results.get(f'A_{k}', {}).get('holdout_pnl', None),
            **phase2_results.get(f'A_{k}', {}),
        } for k, idx in method_a_sets.items()},
        'method_B': {k: {
            'n_features': len(idx),
            'screen_holdout': screen_results.get(f'B_{k}', {}).get('holdout_pnl', None),
            **phase2_results.get(f'B_{k}', {}),
        } for k, idx in method_b_sets.items()},
        'method_C': {k: {
            'n_features': len(idx),
            'screen_holdout': screen_results.get(f'C_{k}', {}).get('holdout_pnl', None),
            **phase2_results.get(f'C_{k}', {}),
        } for k, idx in method_c_sets.items()},
        'method_D': {k: {
            'n_features': len(idx),
            'screen_holdout': screen_results.get(f'D_{k}', {}).get('holdout_pnl', None),
            **phase2_results.get(f'D_{k}', {}),
        } for k, idx in method_d_sets.items()},
        'method_E': {k: {
            'n_features': len(idx),
            'screen_holdout': screen_results.get(f'E_{k}', {}).get('holdout_pnl', None),
            **phase2_results.get(f'E_{k}', {}),
        } for k, idx in method_e_sets.items()},
        'overall_best': {
            'method': best_name,
            'n_features': best_res['n_features'],
            'tsh_ft': best_res['tsh_ft'],
            'inner_loso': best_res['inner_loso'],
            'holdout': best_res['holdout'],
            'buy_thresh': best_res['buy_thresh'],
            'sell_thresh': best_res['sell_thresh'],
            'features': best_feat_names,
        },
    })

    with open(f'{OUT_DIR}/results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    with open(f'{OUT_DIR}/best_features.txt', 'w') as f:
        f.write('\n'.join(best_feat_names) + '\n')
    print(f"  Saved best_features.txt ({len(best_feat_names)} features)", flush=True)

    # Write report
    write_report(all_results, feat_names, base_gain, base_rank,
                 screen_results, phase2_results, best_name)

    progress_update("DONE", {"best_tsh_ft": best_res['tsh_ft'], "best_method": best_name})
    total_elapsed = time.time() - t_start
    print(f"\nTotal elapsed: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)", flush=True)
    print(f"\nRESULT: task=[T147-feature-pruning] "
          f"metrics={{best_tsh_ft={best_res['tsh_ft']:.4f}, best_method={best_name}, "
          f"best_n_features={best_res['n_features']}, "
          f"inner_loso={best_res['inner_loso']:.4f}, holdout={best_res['holdout']:.4f}}} "
          f"notes=[Phase1 screened {len(all_candidates)} subsets; Phase2 full TSH-FT on top-5; "
          f"baseline TSH-FT={BASELINE_TSH_FT:.4f}]",
          flush=True)


def write_report(all_results, feat_names, base_gain, base_rank,
                 screen_results, phase2_results, best_name):
    b = all_results['baseline_359']
    ob = all_results.get('overall_best', {})
    lines = [
        "# T147: Feature Pruning + TSH-FT Evaluation",
        "",
        "## Protocol",
        "- INNER: date 0-95 (train+val), HOLDOUT: date 96-119 (test)",
        "- Phase 1: 1-seed holdout with frozen baseline thresholds for screening",
        "- Phase 2: full TSH-FT (LOSO+DE+2-seed holdout) for top-5 candidates",
        f"- TSH-FT = 0.3×inner_LOSO + 0.7×holdout − 0.3×max(0, inner−holdout)",
        "",
        "## Baseline (359 features)",
        f"- Inner LOSO: {b['inner_loso']:+.4f}",
        f"- Holdout: {b['holdout']:+.4f}",
        f"- TSH-FT: {b['tsh_ft']:+.4f}",
        f"- Thresholds: buy={b['buy_thresh']:.6f} sell={b['sell_thresh']:.6f}",
        "",
        "## Phase 1 Screening Results (sorted by holdout PnL)",
        "",
        "| Candidate | N features | Holdout PnL (1-seed, frozen thresh) |",
        "|-----------|-----------|-------------------------------------|",
    ]
    ranked = sorted(screen_results.items(), key=lambda x: x[1]['holdout_pnl'], reverse=True)
    for name, res in ranked:
        marker = " **best**" if name == best_name else ""
        lines.append(f"| {name} | {res['n_features']} | {res['holdout_pnl']:+.4f}{marker} |")
    lines.append(f"| BASELINE | 359 | {b['holdout']:+.4f} (reference) |")

    lines += [
        "",
        "## Phase 2 Full TSH-FT Results",
        "",
        "| Candidate | N | Inner LOSO | Holdout | TSH-FT |",
        "|-----------|---|-----------|---------|--------|",
        f"| BASELINE | 359 | {b['inner_loso']:+.4f} | {b['holdout']:+.4f} | {b['tsh_ft']:+.4f} |",
    ]
    for name, res in sorted(phase2_results.items(), key=lambda x: x[1]['tsh_ft'], reverse=True):
        marker = " **BEST**" if name == best_name else ""
        lines.append(f"| {name} | {res['n_features']} | {res['inner_loso']:+.4f} | "
                     f"{res['holdout']:+.4f} | {res['tsh_ft']:+.4f}{marker} |")

    lines += [
        "",
        "## Best Result",
        f"**{best_name}**: {ob.get('n_features', '?')} features, TSH-FT = {ob.get('tsh_ft', 0):+.4f}",
        f"- vs Baseline TSH-FT = {b['tsh_ft']:+.4f}",
        f"- Delta TSH-FT: {ob.get('tsh_ft', 0) - b['tsh_ft']:+.4f}",
        "",
        "## Top-20 Features by Gain Importance",
        "",
    ]
    for rank, i in enumerate(base_rank[:20]):
        lines.append(f"{rank+1}. {feat_names[i]} (gain={base_gain[i]:.1f})")

    lines += [
        "",
        "## Recommendation",
        f"Use best_features.txt ({ob.get('n_features', '?')} features) for next iteration packaging.",
    ]
    with open(f'{OUT_DIR}/REPORT.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Saved REPORT.md", flush=True)


if __name__ == '__main__':
    main()
