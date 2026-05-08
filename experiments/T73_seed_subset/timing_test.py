"""Quick timing test: load all seeds, run coarse DE on 1 subset and fine DE on 1 subset."""
import os, sys, time
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments", "T53_r34_stage3"))
from de_thresh import PROB_COLS, gate_asymmetric, vectorized_pnl

T70_DIR = os.path.join(ROOT, "experiments", "T70_v4_stage5")
SEEDS = (1, 7, 13, 42, 100)
BOUNDS = [(0.34, 0.75), (0.34, 0.75), (0.0, 0.30), (0.0, 0.30)]

t0 = time.time()
base = pd.read_parquet(os.path.join(T70_DIR, f"pred_T70_seed{SEEDS[0]}.parquet"))
sym = base["sym"].to_numpy(np.int64)
label = base["true_label"].to_numpy(np.int64)
mp_t = base["midprice_t"].to_numpy(np.float64)
mp_th = base["midprice_th"].to_numpy(np.float64)
probs = {SEEDS[0]: base[PROB_COLS].to_numpy(np.float64)}
for s in SEEDS[1:]:
    probs[s] = pd.read_parquet(os.path.join(T70_DIR, f"pred_T70_seed{s}.parquet"))[PROB_COLS].to_numpy(np.float64)
print(f"load all 5 seeds: {time.time()-t0:.1f}s, n={len(base)}")

# avg over all 5
sub = SEEDS
p_sum = probs[sub[0]].copy()
for s in sub[1:]:
    p_sum += probs[s]
p_avg = (p_sum / float(len(sub))).astype(np.float32)

folds = []
for k in range(5):
    m = sym == k
    folds.append({"probs": p_avg[m], "label": label[m], "mp_t": mp_t[m], "mp_th": mp_th[m]})

def obj(x):
    Tu, Td, du, dd = x
    s = 0.0
    for fk in folds:
        pred = gate_asymmetric(fk["probs"], Tu, Td, du, dd)
        s += vectorized_pnl(pred, fk["label"], fk["mp_t"], fk["mp_th"]).sum()
    return -float(s)

# coarse
t0 = time.time()
res = differential_evolution(obj, bounds=BOUNDS, seed=42, maxiter=40, popsize=15, polish=True, tol=1e-7, mutation=(0.5,1.0), recombination=0.7, updating="deferred", workers=1, init="sobol")
print(f"COARSE DE: {time.time()-t0:.1f}s, obj={-res.fun:+.4f}, nfev={res.nfev}")

# fine
t0 = time.time()
res = differential_evolution(obj, bounds=BOUNDS, seed=0, maxiter=80, popsize=24, polish=True, tol=1e-7, mutation=(0.5,1.0), recombination=0.7, updating="deferred", workers=1, init="sobol")
print(f"FINE DE seed=0: {time.time()-t0:.1f}s, obj={-res.fun:+.4f}, nfev={res.nfev}")
