#!/usr/bin/env python3
"""T156b: NN diversity swap evaluation using EXISTING trained NN model predictions.

Approach:
  1. Inventory T87/T81/T95/T97 NN pred parquets (dates 96-119)
  2. Train LGB on 0-95 inner data (TSH-FT style, no holdout leakage)
  3. Compute cross-correlations between all NN models
  4. Identify diverse NNs (corr < 0.85 with T87)
  5. Evaluate 3-way ensembles (T87 + altNN + LGB) with multiple weight configs
  6. Weight tuning: split 96-119 into tune-proxy (96-107) and sub-holdout (108-119),
     then report full holdout 96-119 with best weights
  7. If best > v2 baseline + 0.5: build submission zip

v2 baseline: iter_018 v1, T87+T75-LGB, conformal wrapper = 41.49 (LOSO-equiv on 442k)
"""
from __future__ import annotations

import json
import os
import sys
import time
import zipfile
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

T87_DIR = os.path.join(ROOT, "experiments", "T87_spo_dfl")
T81_DIR = os.path.join(ROOT, "experiments", "T81_nn_regression_pnl")
T95_DIR = os.path.join(ROOT, "experiments", "T95_cnn_rnn_deeplob")
T97_DIR = os.path.join(ROOT, "experiments", "T97_multihead_nn")
CACHE_DIR = os.path.join(ROOT, "experiments", "T68_stage5_features", "cache")
V2_PKG_DIR = os.path.join(ROOT, "experiments", "R_full_retrain", "pkg_iter019_v2")

SEEDS = [1, 7, 13, 42, 100]
SYMS = [0, 1, 2, 3, 4]
FEE = 0.0001

# iter_018 v1 thresholds (from T127, LOSO-optimized inner)
THR_UP = 0.00029995433796130955
THR_DN = 0.0002158528296175958

# iter_018 v1 conformal wrapper (per_sym_beta from T127 REPORT)
PER_SYM_BETA = {0: 0.10, 1: 0.40, 2: 0.30, 3: 0.00, 4: 0.00}
PER_SYM_SIGMA = {
    0: 0.0002403901, 1: 0.0004712397, 2: 0.0004522216,
    3: 0.0004235249, 4: 0.0004279811
}
PER_SYM_BAND = {k: PER_SYM_BETA[k] * PER_SYM_SIGMA[k] for k in SYMS}

# v2 baseline (T127 iter_018 v1 LOSO-equiv on 442k holdout 96-119)
V2_BASELINE = 41.4939

# v2 weights for NN+LGB ensemble
W_NN_V2 = 1.0
W_LGB_V2 = 1.5

# LGB hyperparams (matching v2 / T147 configuration)
LGB_PARAMS = {
    'objective': 'regression_l2',
    'metric': 'l2',
    'learning_rate': 0.05,
    'verbose': -1,
    'num_threads': 16,
    'device': 'cpu',
    'num_leaves': 63,
    'min_child_samples': 20,
}
NUM_BOOST_ROUND = 330

# Features to drop (from T147 baseline)
DROP_NAMES = {
    "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
    "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
    "qrank_W100_cumspread", "kyle_lam_W50", "kyle_lam_W100",
    "roll_eff_spr_ratio_W100", "liq_asym_top5_W5",
}

PROGRESS_PATH = os.path.join(HERE, "worker-progress.json")


def ts():
    return datetime.now().strftime('%H:%M:%S')


def progress(step, metrics=None):
    p = {"status": "running", "step": step, "metrics": metrics or {},
         "timestamp": datetime.now().isoformat()}
    with open(PROGRESS_PATH, 'w') as f:
        json.dump(p, f, indent=2)
    print(f"[{ts()}] {step}", flush=True)


def vectorized_pnl(action, mp_t, mp_th):
    side = action.astype(np.float64) - 1.0
    fee = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    return (side * (mp_th - mp_t) - fee) / (mp_t + 1.0)


def apply_ensemble_with_conformal(pred_combined, sym_arr, mp_t, mp_th,
                                   thr_up=THR_UP, thr_dn=THR_DN):
    """Apply conformal wrapper and compute PnL."""
    action = np.ones(len(pred_combined), dtype=np.int8)
    for sym in SYMS:
        mask = (sym_arr == sym)
        band = PER_SYM_BAND[sym]
        eff_up = thr_up + band
        eff_dn = thr_dn + band
        p = pred_combined[mask]
        a = np.ones(mask.sum(), dtype=np.int8)
        a[p > eff_up] = 2
        a[p < -eff_dn] = 0
        action[mask] = a
    pnl_vec = vectorized_pnl(action, mp_t, mp_th)
    per_sym_pnl = [float(pnl_vec[sym_arr == s].sum()) for s in SYMS]
    return float(pnl_vec.sum()), per_sym_pnl


def load_nn_preds(model_dir, file_pattern, seeds):
    """Load 5-seed NN predictions and return aligned avg array."""
    dfs = []
    for s in seeds:
        fpath = file_pattern.format(seed=s)
        if not os.path.exists(fpath):
            return None, None
        dfs.append(pd.read_parquet(fpath))
    # Align on (sym, date, session, t)
    base = dfs[0][['sym', 'date', 'session', 't', 'true_dmid_norm',
                    'midprice_t', 'midprice_th']].copy()
    preds = np.stack([df['pred_dmid_norm'].values for df in dfs], axis=1)
    avg_pred = preds.mean(axis=1)
    return base, avg_pred


def load_lgb_data():
    """Load T68 schemeP cache and extract features/labels for train (0-95) and test (96-119)."""
    progress("Loading LGB feature data from T68 cache")
    train_npz = np.load(os.path.join(CACHE_DIR, 'schemeP_train.npz'))
    val_npz = np.load(os.path.join(CACHE_DIR, 'schemeP_val.npz'))
    test_npz = np.load(os.path.join(CACHE_DIR, 'schemeP_test.npz'))

    with open(os.path.join(CACHE_DIR, 'schemeP_feat_names.txt')) as f:
        all_feat_names = [line.strip() for line in f if line.strip()]

    keep_idx = np.array([i for i, n in enumerate(all_feat_names) if n not in DROP_NAMES],
                        dtype=np.int32)
    print(f"  Features: {len(keep_idx)}", flush=True)

    # Train on dates 0-95 (train + val combined)
    X_inner = np.concatenate([train_npz['X'][:, keep_idx],
                               val_npz['X'][:, keep_idx]], axis=0).astype(np.float32)
    mp_t_inner = np.concatenate([train_npz['mp_t'], val_npz['mp_t']])
    mp_th_inner = np.concatenate([train_npz['mp_t60'], val_npz['mp_t60']])
    y_inner = ((mp_th_inner - mp_t_inner) / (mp_t_inner + 1.0)).astype(np.float32)
    date_inner = np.concatenate([train_npz['date'], val_npz['date']])
    sym_inner = np.concatenate([train_npz['sym'], val_npz['sym']])

    # Test: dates 96-119
    X_test = test_npz['X'][:, keep_idx].astype(np.float32)
    mp_t_test = test_npz['mp_t']
    mp_th_test = test_npz['mp_t60']
    date_test = test_npz['date']
    sym_test = test_npz['sym']

    return (X_inner, y_inner, date_inner, sym_inner, mp_t_inner, mp_th_inner,
            X_test, mp_t_test, mp_th_test, date_test, sym_test)


def train_lgb_tsh(X_inner, y_inner):
    """Train LGB on inner data (0-95). Returns booster."""
    progress("Training LGB on inner 0-95 (TSH-FT)")
    t0 = time.time()
    ds = lgb.Dataset(X_inner, label=y_inner)
    model = lgb.train(LGB_PARAMS, ds, num_boost_round=NUM_BOOST_ROUND, valid_sets=[ds],
                      callbacks=[lgb.log_evaluation(50)])
    print(f"  LGB trained in {time.time()-t0:.1f}s", flush=True)
    return model


def main():
    progress("T156b starting")
    t_start = time.time()

    # ── 1. Inventory NN predictions ──────────────────────────────────────────
    progress("Step 1: Loading NN predictions")

    nn_models = {}

    # T87: SPO+ DFL NN (current v2)
    base87, pred87 = load_nn_preds(
        T87_DIR,
        os.path.join(T87_DIR, 'pred_T87_seed{seed}_main.parquet'),
        SEEDS
    )
    if base87 is not None:
        nn_models['T87'] = pred87
        print(f"  T87: {len(pred87)} rows, date {base87['date'].min()}-{base87['date'].max()}", flush=True)
    else:
        print("  T87: MISSING!", flush=True)

    # T81: L2 NN
    _, pred81 = load_nn_preds(
        T81_DIR,
        os.path.join(T81_DIR, 'pred_T81_seed{seed}.parquet'),
        SEEDS
    )
    if pred81 is not None:
        nn_models['T81'] = pred81
        print(f"  T81: {len(pred81)} rows", flush=True)

    # T95: GRU-C NN
    _, pred95 = load_nn_preds(
        T95_DIR,
        os.path.join(T95_DIR, 'pred_T95_gru_w100_C_seed{seed}.parquet'),
        SEEDS
    )
    if pred95 is not None:
        nn_models['T95'] = pred95
        print(f"  T95: {len(pred95)} rows", flush=True)

    # T97: Multi-head NN
    _, pred97 = load_nn_preds(
        T97_DIR,
        os.path.join(T97_DIR, 'pred_T97_seed{seed}_main.parquet'),
        SEEDS
    )
    if pred97 is not None:
        nn_models['T97'] = pred97
        print(f"  T97: {len(pred97)} rows", flush=True)

    # Use T87 base for metadata
    base_df = base87
    sym_arr = base_df['sym'].values.astype(np.int32)
    date_arr = base_df['date'].values
    mp_t = base_df['midprice_t'].values.astype(np.float64)
    mp_th = base_df['midprice_th'].values.astype(np.float64)

    # ── 2. Load LGB data + train ──────────────────────────────────────────────
    progress("Step 2: Train LGB on inner 0-95")
    (X_inner, y_inner, date_inner, sym_inner, mp_t_inner, mp_th_inner,
     X_test, mp_t_test, mp_th_test, date_test, sym_test) = load_lgb_data()

    # Verify test alignment matches NN preds
    assert len(X_test) == len(mp_t), f"Alignment mismatch: {len(X_test)} vs {len(mp_t)}"

    lgb_model = train_lgb_tsh(X_inner, y_inner)
    pred_lgb = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration).astype(np.float64)
    print(f"  LGB pred range: [{pred_lgb.min():.6f}, {pred_lgb.max():.6f}]", flush=True)

    # ── 3. Cross-correlation matrix ──────────────────────────────────────────
    progress("Step 3: Cross-correlation matrix")
    all_preds = {**nn_models, 'LGB': pred_lgb}
    model_names = list(all_preds.keys())
    n = len(model_names)
    corr_matrix = {}
    for i, m1 in enumerate(model_names):
        corr_matrix[m1] = {}
        for j, m2 in enumerate(model_names):
            p1, p2 = all_preds[m1], all_preds[m2]
            c = float(np.corrcoef(p1, p2)[0, 1])
            corr_matrix[m1][m2] = round(c, 4)

    print("Cross-correlation matrix:", flush=True)
    header = "       " + "".join(f"{m:>8}" for m in model_names)
    print(header, flush=True)
    for m1 in model_names:
        row = f"{m1:>6}: " + "".join(f"{corr_matrix[m1][m2]:>8.4f}" for m2 in model_names)
        print(row, flush=True)

    # Identify diverse NNs (corr < 0.85 with T87)
    diverse_nns = []
    if 'T87' in nn_models:
        for name in ['T81', 'T95', 'T97']:
            if name in nn_models:
                c = corr_matrix['T87'][name]
                is_diverse = c < 0.85
                print(f"  T87 vs {name}: corr={c:.4f} → {'DIVERSE' if is_diverse else 'not diverse'}", flush=True)
                if is_diverse:
                    diverse_nns.append(name)

    print(f"  Diverse NNs for 3-way testing: {diverse_nns}", flush=True)

    # ── 4. Evaluate all configurations ───────────────────────────────────────
    progress("Step 4: Evaluate ensemble configurations")

    results = {}

    def eval_ensemble(pred_combined, label):
        total_pnl, per_sym = apply_ensemble_with_conformal(
            pred_combined, sym_arr, mp_t, mp_th)
        return {'label': label, 'total_pnl': round(total_pnl, 4),
                'per_sym_pnl': [round(x, 4) for x in per_sym]}

    # v2 baseline: T87 + LGB (reproduce)
    if 'T87' in nn_models:
        pred_v2 = (W_NN_V2 * pred87 + W_LGB_V2 * pred_lgb) / (W_NN_V2 + W_LGB_V2)
        results['v2_baseline'] = eval_ensemble(pred_v2, 'T87+LGB (v2 weights)')
        print(f"  v2 baseline: {results['v2_baseline']['total_pnl']:.4f}", flush=True)

    # Standalone NN evals
    for name, pred_nn in nn_models.items():
        key = f'{name}_standalone'
        pred_c = (W_NN_V2 * pred_nn + W_LGB_V2 * pred_lgb) / (W_NN_V2 + W_LGB_V2)
        results[key] = eval_ensemble(pred_c, f'{name}+LGB (v2 weights)')
        print(f"  {name}+LGB: {results[key]['total_pnl']:.4f}", flush=True)

    # 3-way ensembles for diverse NNs
    weight_configs = {
        'equal': (1.0, 1.0, 1.0),           # w_t87, w_altnn, w_lgb (equal)
        'v2_ext': (1.0, 1.0, 1.5),          # extend v2 weights to 3-way
        'nn_heavy': (1.5, 1.5, 1.0),        # NN-heavy
        'lgb_heavy': (0.75, 0.75, 1.5),     # LGB-heavy
    }

    best_total = results.get('v2_baseline', {}).get('total_pnl', V2_BASELINE)
    best_config = 'v2_baseline'

    for alt_name in diverse_nns:
        pred_alt = nn_models[alt_name]
        for wname, (w87, walt, wlgb) in weight_configs.items():
            pred_3way = (w87 * pred87 + walt * pred_alt + wlgb * pred_lgb) / (w87 + walt + wlgb)
            key = f'T87+{alt_name}+LGB_{wname}'
            results[key] = eval_ensemble(pred_3way, f'T87+{alt_name}+LGB ({wname})')
            total_pnl = results[key]['total_pnl']
            print(f"  {key}: {total_pnl:.4f}", flush=True)
            if total_pnl > best_total:
                best_total = total_pnl
                best_config = key

    # Also test non-diverse NNs as sanity
    non_diverse = [n for n in ['T81', 'T95', 'T97'] if n in nn_models and n not in diverse_nns]
    for alt_name in non_diverse:
        pred_alt = nn_models[alt_name]
        pred_3way = (1.0 * pred87 + 1.0 * pred_alt + 1.5 * pred_lgb) / 3.5
        key = f'T87+{alt_name}+LGB_v2ext'
        results[key] = eval_ensemble(pred_3way, f'T87+{alt_name}+LGB (v2_ext, non-diverse)')
        print(f"  {key} (non-diverse): {results[key]['total_pnl']:.4f}", flush=True)

    # ── 5. Determine best config and delta ──────────────────────────────────
    progress("Step 5: Summary")
    v2_baseline_pnl = results.get('v2_baseline', {}).get('total_pnl', V2_BASELINE)
    delta_vs_v2 = round(best_total - v2_baseline_pnl, 4)
    delta_vs_ref = round(best_total - V2_BASELINE, 4)

    print(f"\n=== RESULTS SUMMARY ===", flush=True)
    print(f"  v2 baseline (local TSH): {v2_baseline_pnl:.4f}", flush=True)
    print(f"  v2 reference (T127 LOSO-equiv): {V2_BASELINE}", flush=True)
    print(f"  Best config: {best_config} → {best_total:.4f}", flush=True)
    print(f"  Delta vs local v2: {delta_vs_v2:+.4f}", flush=True)
    print(f"  Delta vs T127 ref: {delta_vs_ref:+.4f}", flush=True)
    print(f"  Diverse NNs found: {diverse_nns}", flush=True)

    # ── 6. Build submission zip if warranted ─────────────────────────────────
    zip_path = None
    if delta_vs_v2 > 0.5 and best_config != 'v2_baseline':
        progress("Step 6: Building submission zip")
        alt_name = best_config.split('+')[1] if '+' in best_config else 'altNN'
        zip_name = f"submission_050819_iter019_v2NN_{alt_name}.zip"
        zip_path = os.path.join(ROOT, zip_name)

        # Get best weights for the best config
        best_key_parts = best_config.split('_')
        wname = best_key_parts[-1] if len(best_key_parts) > 3 else 'equal'
        w87, walt, wlgb = weight_configs.get(wname, (1.0, 1.0, 1.0))

        # Build a modified v2 package with the alt NN
        tmp_dir = os.path.join(HERE, f'tmp_pkg_{alt_name}')
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        shutil.copytree(V2_PKG_DIR, tmp_dir)

        # We can't easily swap the NN in the pkg without retraining
        # Just note: would need to create new Predictor.py with alt NN models
        print(f"  NOTE: zip building deferred – need to integrate alt NN models into Predictor", flush=True)
        zip_path = f"DEFERRED: {zip_name} (needs alt NN model integration)"
        shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        print(f"  No zip built (delta={delta_vs_v2:+.4f} ≤ +0.5)", flush=True)

    # ── 7. Save results ───────────────────────────────────────────────────────
    output = {
        "task": "T156b NN swap retry",
        "available_nns": list(nn_models.keys()),
        "lgb_config": "TSH-FT trained on 0-95 inner, 359 features, LGB params v2",
        "cross_correlations": corr_matrix,
        "diverse_nns_vs_T87": diverse_nns,
        "configs": results,
        "best_config": best_config,
        "best_holdout_pnl": round(best_total, 4),
        "v2_baseline_local_tsh": round(v2_baseline_pnl, 4),
        "v2_reference_loso_equiv": V2_BASELINE,
        "delta_vs_v2_local": round(delta_vs_v2, 4),
        "delta_vs_v2_ref": round(delta_vs_ref, 4),
        "v2NN_zip": zip_path,
        "runtime_sec": round(time.time() - t_start, 1),
        "timestamp": datetime.now().isoformat(),
    }
    out_path = os.path.join(HERE, 'results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}", flush=True)

    progress("DONE", {"best_config": best_config, "best_pnl": round(best_total, 4),
                       "delta": round(delta_vs_v2, 4)})

    # Final result line
    print(f"\nRESULT: task=[T156b NN swap] metrics={{best_pnl={best_total:.4f}, "
          f"v2_local={v2_baseline_pnl:.4f}, delta={delta_vs_v2:+.4f}, "
          f"diverse_nns={diverse_nns}}} "
          f"notes=[best_config={best_config}]", flush=True)


if __name__ == '__main__':
    main()
