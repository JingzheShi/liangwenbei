"""Sanity: run Predictor (with H4 per-model weights) on holdout, verify pred matches lsq_linear pred.

Expected: test_pnl ≈ +193.94 (from H4 in run_t192_bounded_v2.py).
"""
import json, os, sys, time
import numpy as np

# Insert pkg dir on path so Predictor.py loads (it expects relative paths)
PKG = "/root/projects/liangwenbei_workdir/experiments/T190_150plus150_pkg/final_pkg"
sys.path.insert(0, PKG)
os.chdir(PKG)

import Predictor as P

p = P.Predictor()
print(f"Device: {p._device}")
print(f"per_model_weights loaded for h60: {60 in p._per_model_weights}")
if 60 in p._per_model_weights:
    nn_w = p._per_model_weights[60]["nn"]
    lgb_w = p._per_model_weights[60]["lgb"]
    print(f"  NN weights: 150, sample (1, 75, 150) = {nn_w[1]:.6f}, {nn_w[75]:.6f}, {nn_w[150]:.6f}")
    print(f"  LGB weights: 150, sample (1, 75, 150) = {lgb_w[1]:.6f}, {lgb_w[75]:.6f}, {lgb_w[150]:.6f}")

# Load test data
import pandas as pd
CACHE = "/root/projects/liangwenbei_workdir/experiments/T68_stage5_features/cache"
test_d = np.load(os.path.join(CACHE, "schemeP_test.npz"))
print(f"Test rows: {len(test_d['X']):,}")

# Compare against pre-computed preds: feed each row's raw cols into Predictor
# But Predictor expects 100-tick window per row. We don't have that easily.
# Instead, compute pred_dmid directly using p._lgb_lists and p._nn_batched on cached features.

# Get features (already 359-d schemeP, same as what Predictor uses)
X = test_d["X"]
sym = test_d["sym"]
mp_t = test_d["mp_t"]
mp_t60 = test_d["mp_t60"]

# Need to compute on schemeP-style feature matrix
# Predictor.predict() expects 100-tick batches; we bypass by directly calling internal methods.

import lightgbm as lgb_

# Get lgb_keep_idx (LGB uses subset of schemeP features)
with open(os.path.join(CACHE, "schemeP_feat_names.txt")) as f:
    all_feat_names = [l.strip() for l in f]
m0 = p._lgb_lists[60][0]
lgb_keep = np.array([all_feat_names.index(fn) for fn in m0.feature_name()], dtype=np.int64)
X_lgb = X[:, lgb_keep]

# LGB weighted sum
w_lgb_arr = p._lgb_weights[60]
pred_lgb = np.zeros(X.shape[0], dtype=np.float32)
for booster, w in zip(p._lgb_lists[60], w_lgb_arr):
    if w > 0:
        pred_lgb += float(w) * booster.predict(X_lgb).astype(np.float32)

# NN weighted sum
nn_ens = p._nn_batched[60]
B = 4096
pred_nn = np.zeros(X.shape[0], dtype=np.float32)
for i in range(0, X.shape[0], B):
    batch = X[i:i+B]
    pred_nn[i:i+B] = nn_ens.predict_weighted_sum(batch.astype(np.float32))

pred_dmid = pred_nn + pred_lgb

# Apply gating + PnL (same as eval_holdout_150plus150.py)
FEE = 0.0001
with open(os.path.join(PKG, "thresholds.json")) as f:
    tcfg = json.load(f)
h60 = next(h for h in tcfg["horizons"] if h["h"] == 60)
thr_up, thr_dn = h60["thr_up"], h60["thr_dn"]
conformal = tcfg.get("conformal_wrapper", {"enabled": False})

def ev_gate_pnl(pred, sym_arr, mp_t, mp_th, thr_up, thr_dn, conformal_cfg):
    eff_up = np.full(len(pred), thr_up, dtype=np.float64)
    eff_dn = np.full(len(pred), thr_dn, dtype=np.float64)
    if conformal_cfg and conformal_cfg.get("enabled"):
        betas = conformal_cfg["per_sym_beta"]
        sigmas = conformal_cfg["per_sym_sigma"]
        def_beta = conformal_cfg.get("default_beta_for_ood", 0.16)
        def_sigma = conformal_cfg.get("default_sigma_for_ood", 0.0004)
        for sid in np.unique(sym_arr):
            mask = (sym_arr == sid)
            b = betas.get(str(int(sid)), def_beta)
            s = sigmas.get(str(int(sid)), def_sigma)
            eff_up[mask] += b * s
            eff_dn[mask] += b * s
    action = np.full(len(pred), 1, dtype=np.int8)
    action[pred > eff_up] = 2
    action[pred < -eff_dn] = 0
    side = action.astype(np.float64) - 1.0
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    pnl = (side * diff - fee_pnl) / denom
    return float(pnl.sum()), int((action != 1).sum())

p_h4, n_h4 = ev_gate_pnl(pred_dmid, sym, mp_t, mp_t60, thr_up, thr_dn, conformal)
print(f"\n=== Predictor H4 weights holdout ===")
print(f"  test_pnl = {p_h4:+.4f}  n_act = {n_h4:,}")
print(f"  expected = +193.94 (from lsq_linear in run_t192_bounded_v2.py)")
print(f"  diff = {p_h4 - 193.9364:+.4f}")
print(f"  match: {'YES' if abs(p_h4 - 193.9364) < 0.5 else 'NO — Predictor implementation BUG'}")
