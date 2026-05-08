"""Quick LGB GPU vs CPU benchmark on actual data."""
import sys, os, time
import numpy as np
ROOT = '/root/projects/liangwenbei_workdir'
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'T26_domain_randomization'))
from build_aug import class_balanced_weight
import lightgbm as lgb

CACHE_DIR = os.path.join(ROOT, 'experiments', 'T68_stage5_features', 'cache')
print('loading...', flush=True)
d = np.load(os.path.join(CACHE_DIR, 'schemeP_train.npz'))
date = d['date']; m_va = date >= 76; m_t = ~m_va
X_tr = d['X'][m_t].astype(np.float32, copy=False)
y_regr_tr = ((d['mp_t60'][m_t].astype(np.float64) - d['mp_t'][m_t].astype(np.float64))/(d['mp_t'][m_t].astype(np.float64)+1)).astype(np.float32)
y_cls_tr = d['y60'][m_t].astype(np.int64)
sw = class_balanced_weight(y_cls_tr, num_class=3)
print(f'data: {X_tr.shape}', flush=True)

dt = lgb.Dataset(X_tr, label=y_regr_tr, weight=sw, free_raw_data=False)

for dev, threads, leaves in [('cpu', 16, 127), ('cpu', 16, 63), ('cpu', 8, 127), ('gpu', 8, 127)]:
    p = {'objective':'huber','alpha':0.001,'metric':'l1',
         'learning_rate':0.05,'num_leaves':leaves,'min_data_in_leaf':100,
         'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,
         'lambda_l2':1.0,'num_threads':threads,'verbose':-1}
    if dev=='gpu': p['device']='gpu'; p['gpu_use_dp']=False
    t0=time.time()
    b = lgb.train(p, dt, num_boost_round=100, callbacks=[lgb.log_evaluation(period=0)])
    elapsed = time.time()-t0
    print(f'  {dev:3s} threads={threads:2d} leaves={leaves:3d}: 100 rounds = {elapsed:.2f}s ({elapsed*2:.1f}s for 200, {elapsed/100*1000:.1f}ms/round)', flush=True)
