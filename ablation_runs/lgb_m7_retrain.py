"""
LGB M7 full-data retrain (train+val = 0-95) as reference row.
For each seed: read best_iter from V4 trained model, retrain on train+val with
num_boost_round = best_iter * 1.1, use threshold from V4 step, evaluate on test.
"""
import os, sys, json, time, gc
import numpy as np
import lightgbm as lgb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train_50_lgb_5hp import (
    bidask_mirror_indices, apply_mirror, make_weights,
    compute_pnl, search_threshold_sym,
)

CACHE_DIR = Path('/root/projects/liangwenbei_workdir/ablation_runs/cache_log1p')
V4_DIR = Path('/root/projects/liangwenbei_workdir/ablation_runs/big50_lgb')
OUT = Path('/root/projects/liangwenbei_workdir/ablation_runs/big50_lgb_m7')
OUT.mkdir(parents=True, exist_ok=True)

# Load splits (matching train_50_lgb_5hp.py protocol)
H = 60
def load_split(name):
    d = np.load(CACHE_DIR / f'schemeP_{name}.npz')
    return {
        "X": d["X"],
        "mp_t": d["mp_t"].astype(np.float64),
        "mp_th": d[f"mp_t{H}"].astype(np.float64),
        "y_cls": d[f"y{H}"],
    }

train = load_split('train')
val = load_split('val')
test = load_split('test')
feat_names = open(CACHE_DIR / 'schemeP_feat_names.txt').read().strip().splitlines()

# y_reg = (mp_th - mp_t) / (mp_t + 1)
y_reg_tr  = ((train['mp_th'] - train['mp_t']) / (train['mp_t'] + 1.0)).astype(np.float32)
y_reg_val = ((val['mp_th']   - val['mp_t'])   / (val['mp_t']   + 1.0)).astype(np.float32)
y_cls_tr  = train['y_cls'].astype(np.int64)
y_cls_val = val['y_cls'].astype(np.int64)

# Apply mirror aug to train
Xm = apply_mirror(train['X'], feat_names)
ymir = -y_reg_tr
ycls_mir = (2 - y_cls_tr).clip(0, 2).astype(np.int64)
Xtr_aug = np.concatenate([train['X'], Xm], axis=0)
ytr_aug = np.concatenate([y_reg_tr, ymir])
ycls_tr_aug = np.concatenate([y_cls_tr, ycls_mir])

# Apply mirror to val too (M7 uses train+val combined)
Xm_v = apply_mirror(val['X'], feat_names)
ymir_v = -y_reg_val
ycls_v_mir = (2 - y_cls_val).clip(0, 2).astype(np.int64)
Xval_aug = np.concatenate([val['X'], Xm_v], axis=0)
yval_aug = np.concatenate([y_reg_val, ymir_v])
ycls_val_aug = np.concatenate([y_cls_val, ycls_v_mir])

# Combine train+val
X_full = np.concatenate([Xtr_aug, Xval_aug], axis=0)
y_full = np.concatenate([ytr_aug, yval_aug])
ycls_full = np.concatenate([ycls_tr_aug, ycls_val_aug])
sw_full = make_weights(ycls_full)
print(f'M7 train+val combined: {X_full.shape}', flush=True)

# Run all 50 seeds with project HP cycling
HP_CONFIGS = [
    dict(feature_fraction=0.8, bagging_fraction=0.8, num_leaves=127, lambda_l2=1.0),
    dict(feature_fraction=0.6, bagging_fraction=0.7, num_leaves=127, lambda_l2=1.0),
    dict(feature_fraction=0.7, bagging_fraction=0.85, num_leaves=63, lambda_l2=2.0),
    dict(feature_fraction=0.5, bagging_fraction=0.6, num_leaves=255, lambda_l2=0.5),
    dict(feature_fraction=0.4, bagging_fraction=0.5, num_leaves=127, lambda_l2=3.0),
]

results = []
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--start-seed', type=int, default=1)
ap.add_argument('--n-seeds', type=int, default=50)
args = ap.parse_args()
SEED_LIST = list(range(args.start_seed, args.start_seed + args.n_seeds))

for S in SEED_LIST:
    # Read V4 best_iter and thr from existing seed
    v4_res = json.loads((V4_DIR / f'seed{S}' / 'results.json').read_text())
    hp_group = v4_res['hp_group']
    HP = HP_CONFIGS[hp_group]
    best_iter_v4 = v4_res['best_iter']
    n_rounds = max(1, int(round(best_iter_v4 * 1.1)))
    thr_v4 = v4_res['thr_sym']
    print(f'\n=== M7 LGB seed={S} hp{hp_group} (best_iter={best_iter_v4}, n_rounds={n_rounds}) ===', flush=True)

    params = {
        'objective': 'regression', 'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': HP['num_leaves'],
        'feature_fraction': HP['feature_fraction'],
        'bagging_fraction': HP['bagging_fraction'],
        'bagging_freq': 5,
        'lambda_l2': HP['lambda_l2'],
        'min_data_in_leaf': 100,
        'num_threads': 18,
        'seed': S, 'verbose': -1,
        'device': 'gpu', 'gpu_use_dp': False,
    }

    dtrain = lgb.Dataset(X_full, label=y_full, weight=sw_full, free_raw_data=False)
    t0 = time.time()
    booster = lgb.train(params, dtrain, num_boost_round=n_rounds, valid_sets=[dtrain])
    train_time = time.time() - t0

    yp_test = booster.predict(test['X']).astype(np.float32)
    # Use V4's thr for fairness (M7 has no val for threshold search)
    test_pnl, n_act = compute_pnl(yp_test, test['mp_t'], test['mp_th'], thr_v4, thr_v4)
    print(f'  M7 seed{S}: time={train_time:.1f}s n_rounds={n_rounds} test_pnl={test_pnl:.4f} (thr from V4)', flush=True)

    results.append({'seed': S, 'hp_group': hp_group, 'best_iter_v4': best_iter_v4, 'n_rounds_m7': n_rounds,
                    'thr_from_v4': thr_v4, 'test_pnl_m7': float(test_pnl),
                    'test_pnl_v4': float(v4_res['test_pnl_h60'])})

    seed_dir = OUT / f'seed{S}'
    seed_dir.mkdir(exist_ok=True)
    np.savez(seed_dir / 'preds.npz', yp_test=yp_test, mp_t_test=test['mp_t'], mp_th_test=test['mp_th'])
    booster.save_model(str(seed_dir / 'model.txt'))
    (seed_dir / 'results.json').write_text(json.dumps(results[-1], indent=2))

# Summary
(OUT / 'summary.json').write_text(json.dumps(results, indent=2))
import statistics
m7_tests = [r['test_pnl_m7'] for r in results]
v4_tests = [r['test_pnl_v4'] for r in results]
print(f'\n=== LGB M7 retrain summary ({len(results)} seeds, HP_CONFIGS[0]) ===')
print(f'  V4 test PnL (orig):  mean={statistics.mean(v4_tests):.4f}  std={statistics.stdev(v4_tests):.4f}')
print(f'  M7 test PnL (new):   mean={statistics.mean(m7_tests):.4f}  std={statistics.stdev(m7_tests):.4f}')
print(f'  Δ = {statistics.mean(m7_tests) - statistics.mean(v4_tests):+.4f}')
