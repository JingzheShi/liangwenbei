"""
T62 Unbiased thresh evaluation.

Loads 5-seed-avg OOF predictions from T59. For each split (A-E), runs DE
thresh optimization on the TUNE half, then evaluates the resulting thresh on
the EVAL half. The gap (tune - eval) quantifies test-set leakage of the
+24.52 number from iter_009.
"""
import json
import time
import os
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

FEE = 0.0001
HERE = '/root/projects/liangwenbei_workdir/experiments/T59_fullsym_train'
OUT = '/root/projects/liangwenbei_workdir/experiments/T62_unbiased_thresh'
SEEDS = [42, 1, 7, 13, 100]
DE_SEEDS = [0, 1, 7, 42]

LOG = open(os.path.join(OUT, 'run.log'), 'w', buffering=1)


def log(m):
    LOG.write(m + '\n')
    LOG.flush()
    print(m, flush=True)


def gate(probs, T_up, T_dn, d_up, d_dn):
    p0 = probs[:, 0]
    p1 = probs[:, 1]
    p2 = probs[:, 2]
    pred = np.full(len(probs), 1, dtype=np.int8)
    pred[(p2 > T_up) & (p2 - p1 > d_up)] = 2
    pred[(p0 > T_dn) & (p0 - p1 > d_dn)] = 0
    return pred


def vec_pnl(pred, label, mp_t, mp_th):
    side = pred.astype(np.float64) - 1.0
    diff = mp_th - mp_t
    fee = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    return (side * diff - fee) / (mp_t + 1.0)


def de_search(probs, label, mp_t, mp_th, tag=''):
    """Run DE 4D search; return (best_sum, best_thresh)."""
    def obj(x):
        pred = gate(probs, *x)
        return -float(vec_pnl(pred, label, mp_t, mp_th).sum())
    bounds = [(0.34, 0.65), (0.34, 0.65), (0.0, 0.30), (0.0, 0.30)]
    best = -1e9
    bx = None
    for sd in DE_SEEDS:
        r = differential_evolution(
            obj, bounds, seed=sd, maxiter=80, popsize=25,
            tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            updating='deferred', workers=1, init='sobol'
        )
        v = -r.fun
        if v > best:
            best = v
            bx = r.x
    log(f"  [{tag}] tune_sum={best:+.4f}  T_up={bx[0]:.4f} T_dn={bx[1]:.4f} "
        f"d_up={bx[2]:.4f} d_dn={bx[3]:.4f}")
    return best, list(map(float, bx))


def eval_thresh(thresh, probs, label, mp_t, mp_th):
    pred = gate(probs, *thresh)
    s = float(vec_pnl(pred, label, mp_t, mp_th).sum())
    n_act = int((pred != 1).sum())
    return s, n_act


# -------- Load 5-seed avg --------
log(f"loading {len(SEEDS)} seeds...")
ps = None
base = None
for s in SEEDS:
    df = pd.read_parquet(f'{HERE}/full_pred_h60_seed{s}.parquet')
    p = df[['prob_0', 'prob_1', 'prob_2']].to_numpy()
    ps = p.copy() if ps is None else ps + p
    if base is None:
        base = df.copy()
probs_avg = ps / len(SEEDS)
N = len(probs_avg)
label = base['true_label'].to_numpy(int)
mp_t = base['midprice_t'].to_numpy(float)
mp_th = base['midprice_th'].to_numpy(float)
sym = base['sym'].to_numpy()
date = base['date'].to_numpy()
session = base['session'].to_numpy()
log(f"loaded N={N}")

# Sanity: replicate iter_009 +24.52 on full data
full_sum, full_thr = de_search(probs_avg, label, mp_t, mp_th, tag='full_replicate')
log(f"replication: full DE sum = {full_sum:+.4f}  (expected ~+24.52)")

results = {
    'meta': {
        'N': int(N),
        'iter_009_full_sum': full_sum,
        'iter_009_full_thresh': full_thr,
    },
    'splits': {},
}


def run_split(name, mask_tune, mask_eval, probs=probs_avg):
    """Run one split: DE on tune, evaluate on eval."""
    p_t = probs[mask_tune]
    p_e = probs[mask_eval]
    n_t = mask_tune.sum()
    n_e = mask_eval.sum()
    log(f"\n--- {name}: tune N={n_t}, eval N={n_e} ---")
    tune_sum, thr = de_search(p_t, label[mask_tune], mp_t[mask_tune], mp_th[mask_tune], tag=name + '/tune')
    eval_sum, n_act = eval_thresh(thr, p_e, label[mask_eval], mp_t[mask_eval], mp_th[mask_eval])
    gap = tune_sum - eval_sum
    log(f"  [{name}] eval_sum={eval_sum:+.4f}  n_active={n_act}/{n_e}  gap={gap:+.4f}")
    return {
        'tune_n': int(n_t), 'eval_n': int(n_e),
        'tune_sum': float(tune_sum), 'eval_sum': float(eval_sum),
        'gap': float(gap), 'thresh': thr, 'eval_n_active': n_act,
    }


# ============== Split A: random 50/50 (seed=42) ==============
log("\n========== SPLIT A: random 50/50 ==========")
rng = np.random.RandomState(42)
perm = rng.permutation(N)
half = N // 2
mA1_tune = np.zeros(N, bool); mA1_tune[perm[:half]] = True
mA1_eval = ~mA1_tune
A1 = run_split('A1_rand_first_tune', mA1_tune, mA1_eval)
mA2_tune = ~mA1_tune
mA2_eval = mA1_tune
A2 = run_split('A2_rand_second_tune', mA2_tune, mA2_eval)
A_avg_eval = (A1['eval_sum'] + A2['eval_sum']) / 2.0
A_avg_gap = (A1['gap'] + A2['gap']) / 2.0
# Scale to full-data equivalent (each eval covers ~half data, so average eval_sum on
# half ≈ half of full-data sum at the unbiased thresh; *2 to compare with +24.52 scale)
A_unbiased_full = A_avg_eval * 2.0
log(f"\n[A] avg_eval_sum (half-data scale)= {A_avg_eval:+.4f}")
log(f"[A] full-scale unbiased estimate = {A_unbiased_full:+.4f}  (vs full DE +{full_sum:.2f}, gap {full_sum - A_unbiased_full:+.4f})")
results['splits']['A_random_5050'] = {
    'A1': A1, 'A2': A2, 'avg_eval_half': A_avg_eval,
    'unbiased_full_scale': A_unbiased_full, 'avg_gap': A_avg_gap,
}

# ============== Split B: date 50/50 ==============
log("\n========== SPLIT B: date 50/50 ==========")
mB1_tune = date <= 107  # first half dates
mB1_eval = ~mB1_tune
B1 = run_split('B1_early_tune', mB1_tune, mB1_eval)
B2 = run_split('B2_late_tune', ~mB1_tune, mB1_tune)
B_avg_eval = (B1['eval_sum'] + B2['eval_sum']) / 2.0
B_unbiased_full = B_avg_eval * 2.0
log(f"\n[B] avg_eval_sum = {B_avg_eval:+.4f}  full-scale = {B_unbiased_full:+.4f}")
results['splits']['B_date_5050'] = {
    'B1': B1, 'B2': B2, 'avg_eval_half': B_avg_eval, 'unbiased_full_scale': B_unbiased_full,
}

# ============== Split C: sym 50/50 (3 vs 2) ==============
log("\n========== SPLIT C: sym {0,1,2} vs {3,4} ==========")
mC1_tune = (sym == 0) | (sym == 1) | (sym == 2)
mC1_eval = ~mC1_tune
C1 = run_split('C1_sym012_tune', mC1_tune, mC1_eval)
C2 = run_split('C2_sym34_tune', mC1_eval, mC1_tune)
# tune set sizes differ: 3 sym vs 2 sym, so weighted scale-up to 5-sym
C_eval_per_row = (C1['eval_sum'] / C1['eval_n'] + C2['eval_sum'] / C2['eval_n']) / 2.0
C_unbiased_full = C_eval_per_row * N
log(f"\n[C] avg_eval_per_row = {C_eval_per_row:+.6e}  full-scale = {C_unbiased_full:+.4f}")
results['splits']['C_sym_5050'] = {
    'C1': C1, 'C2': C2, 'avg_eval_per_row': C_eval_per_row, 'unbiased_full_scale': C_unbiased_full,
}

# ============== Split D: session am vs pm ==============
log("\n========== SPLIT D: am vs pm ==========")
mD1_tune = session == 'am'
mD1_eval = ~mD1_tune
D1 = run_split('D1_am_tune', mD1_tune, mD1_eval)
D2 = run_split('D2_pm_tune', ~mD1_tune, mD1_tune)
D_avg_eval = (D1['eval_sum'] + D2['eval_sum']) / 2.0
D_unbiased_full = D_avg_eval * 2.0
log(f"\n[D] avg_eval_sum = {D_avg_eval:+.4f}  full-scale = {D_unbiased_full:+.4f}")
results['splits']['D_session_5050'] = {
    'D1': D1, 'D2': D2, 'avg_eval_half': D_avg_eval, 'unbiased_full_scale': D_unbiased_full,
}

# ============== Split E: 5-fold time CV ==============
log("\n========== SPLIT E: 5-fold time CV (by date) ==========")
# 24 dates -> 5 contiguous folds ~5 dates each
all_dates = sorted(np.unique(date))
folds = [all_dates[i::5] for i in range(5)]  # interleaved by date
# Better: 5 contiguous time blocks
fold_size = len(all_dates) // 5  # 4
fold_extra = len(all_dates) - fold_size * 5  # 4 extra dates
folds = []
i = 0
for k in range(5):
    n = fold_size + (1 if k < fold_extra else 0)
    folds.append(all_dates[i:i + n])
    i += n
log(f"  fold dates: " + " | ".join([f"{f[0]}-{f[-1]}" for f in folds]))

E_sum_eval = 0.0
E_sum_tune = 0.0
E_total_eval_n = 0
E_per_fold = []
E_thresh_list = []
for k in range(5):
    test_dates = set(folds[k])
    m_eval = np.isin(date, list(test_dates))
    m_tune = ~m_eval
    fk = run_split(f'E_fold{k}', m_tune, m_eval)
    E_sum_eval += fk['eval_sum']
    E_sum_tune += fk['tune_sum']
    E_total_eval_n += fk['eval_n']
    E_per_fold.append(fk)
    E_thresh_list.append(fk['thresh'])

# E_sum_eval is sum over all 5 eval folds = full data covered exactly once
log(f"\n[E] 5-fold total eval_sum (full-data unbiased) = {E_sum_eval:+.4f}")
log(f"[E] gap to full DE = {full_sum - E_sum_eval:+.4f}")
results['splits']['E_5fold_time_cv'] = {
    'fold_dates': [list(map(int, f)) for f in folds],
    'per_fold': E_per_fold,
    'total_eval_sum': float(E_sum_eval),
    'total_tune_sum': float(E_sum_tune),
    'unbiased_full_scale': float(E_sum_eval),  # already full-scale
    'gap_to_full_de': float(full_sum - E_sum_eval),
}

# ============== Recommended unbiased thresh ==============
# Average of A1, A2, E folds threshes (most stable picks)
all_thr = np.array([A1['thresh'], A2['thresh']] + E_thresh_list)
mean_thr = all_thr.mean(axis=0)
median_thr = np.median(all_thr, axis=0)
# Evaluate mean_thr on full data (this is itself biased upward, but smaller than DE-on-full)
mean_full_sum, mean_full_nact = eval_thresh(mean_thr, probs_avg, label, mp_t, mp_th)
median_full_sum, median_full_nact = eval_thresh(median_thr, probs_avg, label, mp_t, mp_th)
log(f"\n--- recommended unbiased thresh (mean of 7 folds) ---")
log(f"  mean = T_up={mean_thr[0]:.4f} T_dn={mean_thr[1]:.4f} "
    f"d_up={mean_thr[2]:.4f} d_dn={mean_thr[3]:.4f}")
log(f"  applied to full data: sum={mean_full_sum:+.4f} n_active={mean_full_nact}")
log(f"  median = T_up={median_thr[0]:.4f} T_dn={median_thr[1]:.4f} "
    f"d_up={median_thr[2]:.4f} d_dn={median_thr[3]:.4f}")
log(f"  applied to full data: sum={median_full_sum:+.4f} n_active={median_full_nact}")

results['recommended'] = {
    'mean_thresh_AE': list(map(float, mean_thr)),
    'mean_thresh_full_sum': float(mean_full_sum),
    'mean_thresh_full_nact': int(mean_full_nact),
    'median_thresh_AE': list(map(float, median_thr)),
    'median_thresh_full_sum': float(median_full_sum),
    'median_thresh_full_nact': int(median_full_nact),
}

# ============== Summary ==============
log(f"\n========== SUMMARY ==========")
log(f"original iter_009 (DE on full 442k labels): {full_sum:+.4f}")
log(f"split A (random 50/50, full-scale)         : {A_unbiased_full:+.4f}  (gap {full_sum - A_unbiased_full:+.4f})")
log(f"split B (date 50/50, full-scale)           : {B_unbiased_full:+.4f}  (gap {full_sum - B_unbiased_full:+.4f})")
log(f"split C (sym 50/50, full-scale)            : {C_unbiased_full:+.4f}  (gap {full_sum - C_unbiased_full:+.4f})")
log(f"split D (session 50/50, full-scale)        : {D_unbiased_full:+.4f}  (gap {full_sum - D_unbiased_full:+.4f})")
log(f"split E (5-fold time CV, full-scale)       : {E_sum_eval:+.4f}  (gap {full_sum - E_sum_eval:+.4f})")
log(f"recommended mean-thresh on full            : {mean_full_sum:+.4f}")
log(f"recommended median-thresh on full          : {median_full_sum:+.4f}")

results['summary'] = {
    'iter_009_full_DE': full_sum,
    'split_A_unbiased_full': A_unbiased_full,
    'split_B_unbiased_full': B_unbiased_full,
    'split_C_unbiased_full': C_unbiased_full,
    'split_D_unbiased_full': D_unbiased_full,
    'split_E_unbiased_full': float(E_sum_eval),
    'leakage_gap_A': full_sum - A_unbiased_full,
    'leakage_gap_E': full_sum - E_sum_eval,
}

with open(os.path.join(OUT, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2, default=float)
log(f"\nsaved {OUT}/results.json")
LOG.close()
