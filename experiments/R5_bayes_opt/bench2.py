"""Bench LGB with no aug + smaller leaves to find sweet spot."""
import sys, os, time
import numpy as np
ROOT = '/root/projects/liangwenbei_workdir'
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'T26_domain_randomization'))
from build_aug import class_balanced_weight, aug_a_scale
import lightgbm as lgb

CACHE_DIR = os.path.join(ROOT, 'experiments', 'T68_stage5_features', 'cache')
print('loading...', flush=True)
d = np.load(os.path.join(CACHE_DIR, 'schemeP_train.npz'))
date = d['date']; m_va = date >= 76; m_t = ~m_va
X_tr = d['X'][m_t].astype(np.float32, copy=False)
y_regr_tr = ((d['mp_t60'][m_t].astype(np.float64) - d['mp_t'][m_t].astype(np.float64))/(d['mp_t'][m_t].astype(np.float64)+1)).astype(np.float32)
y_cls_tr = d['y60'][m_t].astype(np.int64)

# Scenario 1: no aug
print('NO AUG:', flush=True)
sw = class_balanced_weight(y_cls_tr, num_class=3)
dt = lgb.Dataset(X_tr, label=y_regr_tr, weight=sw, free_raw_data=False)
for dev, threads, leaves in [('cpu', 16, 63), ('cpu', 16, 31), ('gpu', 8, 63)]:
    p = {'objective':'huber','alpha':0.001,'metric':'l1',
         'learning_rate':0.05,'num_leaves':leaves,'min_data_in_leaf':200,
         'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,
         'lambda_l2':1.0,'num_threads':threads,'verbose':-1}
    if dev=='gpu': p['device']='gpu'; p['gpu_use_dp']=False
    t0=time.time()
    b = lgb.train(p, dt, num_boost_round=100, callbacks=[lgb.log_evaluation(period=0)])
    elapsed = time.time()-t0
    print(f'  {dev:3s} thr={threads} leaves={leaves}: {elapsed:.1f}s ({elapsed/100*1000:.0f}ms/round)', flush=True)

# With aug
print('\nWITH AUG (2x):', flush=True)
rng = np.random.default_rng(42)
X_aug = aug_a_scale(X_tr, rng, lo=0.80, hi=1.20)
X_full = np.concatenate([X_tr, X_aug], axis=0)
y_full = np.concatenate([y_regr_tr, y_regr_tr], axis=0)
y_cls_full = np.concatenate([y_cls_tr, y_cls_tr], axis=0)
sw_full = class_balanced_weight(y_cls_full, num_class=3)
dt = lgb.Dataset(X_full, label=y_full, weight=sw_full, free_raw_data=False)
for dev, threads, leaves in [('cpu', 16, 63), ('cpu', 16, 31)]:
    p = {'objective':'huber','alpha':0.001,'metric':'l1',
         'learning_rate':0.05,'num_leaves':leaves,'min_data_in_leaf':200,
         'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,
         'lambda_l2':1.0,'num_threads':threads,'verbose':-1}
    if dev=='gpu': p['device']='gpu'; p['gpu_use_dp']=False
    t0=time.time()
    b = lgb.train(p, dt, num_boost_round=100, callbacks=[lgb.log_evaluation(period=0)])
    elapsed = time.time()-t0
    print(f'  {dev:3s} thr={threads} leaves={leaves}: {elapsed:.1f}s ({elapsed/100*1000:.0f}ms/round)', flush=True)
