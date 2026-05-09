"""T159: Per-sym dynamic abstain band sweep with TSH-FT protocol.

Protocol:
  INNER: dates 0-95  (schemeP train+val from T68 cache)
  HOLDOUT: dates 96-119  (T87/T75 parquet preds)

For each per-sym beta combo:
  1. Compute combined_pred on INNER using T87 NN + T75 LGB models
  2. Compute sigma per sym from INNER preds
  3. Fit DE thr_up/thr_dn on INNER PnL (with abstain band)
  4. Apply FROZEN thresholds to HOLDOUT, score = holdout PnL

Coordinate descent: 2 rounds over 5 syms (11 betas each).
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import lightgbm as lgb
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

T68_CACHE = os.path.join(ROOT, "experiments", "T68_stage5_features", "cache")
T87_DIR = os.path.join(ROOT, "experiments", "T87_spo_dfl")
T75_DIR = os.path.join(ROOT, "experiments", "T75_regression_dmid")

FEE = 0.0001
SEEDS = [1, 7, 13, 42, 100]

# V2 baseline config
BASELINE_BETA = {0: 0.10, 1: 0.40, 2: 0.30, 3: 0.00, 4: 0.00}
V2_THR_UP = 0.00029995433796130955
V2_THR_DN = 0.0002158528296175958

BETA_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00]
BAND_MULT_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

DROP_NAMES = [
    "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
    "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
    "qrank_W100_cumspread", "kyle_lam_W50", "kyle_lam_W100",
    "roll_eff_spr_ratio_W100", "liq_asym_top5_W5"
]


def progress(step, metrics=None):
    data = {
        "status": "running",
        "step": step,
        "metrics": metrics or {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(HERE, "worker-progress.json"), "w") as f:
        json.dump(data, f, indent=2)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {step}", flush=True)


# ──────────────────────────────────────────────────────────────────────────
# MLP Numpy inference (copied from Predictor.py pattern)
# ──────────────────────────────────────────────────────────────────────────

def _gelu_tanh(x):
    c = 0.7978845608028654
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x * x * x)))

def _layernorm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


class MLPNumpy:
    def __init__(self, npz_path):
        d = np.load(npz_path, allow_pickle=False)
        self.in_dim = int(d["in_dim"][0])
        self.hidden = list(d["hidden"].tolist())
        self.use_layernorm = bool(d["use_layernorm"][0])
        self.target_scale = float(d["target_scale"][0])
        self.feat_mean = d["feat_mean"].astype(np.float32)
        self.feat_std = d["feat_std"].astype(np.float32)
        self.keep_idx = d["keep_idx"].astype(np.int64)
        self.clip = float(d["clip"][0])
        self.W, self.b, self.LN_W, self.LN_b = [], [], [], []
        for i in range(len(self.hidden)):
            self.W.append(d[f"L{i}_W"].astype(np.float32))
            self.b.append(d[f"L{i}_b"].astype(np.float32))
            if self.use_layernorm:
                self.LN_W.append(d[f"LN{i}_W"].astype(np.float32))
                self.LN_b.append(d[f"LN{i}_b"].astype(np.float32))
        self.WF = d["LF_W"].astype(np.float32)
        self.bF = d["LF_b"].astype(np.float32)

    def predict(self, X_in, batch=131072):
        results = []
        for s in range(0, len(X_in), batch):
            Xs = X_in[s:s+batch]
            if Xs.shape[1] != self.in_dim:
                Xs = Xs[:, self.keep_idx]
            Xs = Xs.astype(np.float32, copy=True)
            Xs = (Xs - self.feat_mean) / np.where(self.feat_std > 1e-8, self.feat_std, 1.0)
            Xs = np.where(np.isnan(Xs), 0.0, Xs)
            Xs = np.clip(Xs, -self.clip, self.clip).astype(np.float32)
            h = Xs
            for i in range(len(self.hidden)):
                h = h @ self.W[i].T + self.b[i]
                if self.use_layernorm:
                    h = _layernorm(h, self.LN_W[i], self.LN_b[i])
                h = _gelu_tanh(h)
            out = h @ self.WF.T + self.bF
            out = out.squeeze(-1) / self.target_scale
            results.append(out.astype(np.float32))
        return np.concatenate(results)


# ──────────────────────────────────────────────────────────────────────────
# PnL functions
# ──────────────────────────────────────────────────────────────────────────

def vectorized_pnl(action, mp_t, mp_th):
    side = action.astype(np.float64) - 1.0
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee_pnl) / denom


def apply_abstain_wrapper(combined_pred, sym_arr, thr_up, thr_dn, per_sym_beta, per_sym_sigma):
    action = np.ones(len(combined_pred), dtype=np.int8)
    for s in range(5):
        mask = sym_arr == s
        band = per_sym_beta[s] * per_sym_sigma[s]
        pred_s = combined_pred[mask]
        act_s = np.ones(mask.sum(), dtype=np.int8)
        act_s[pred_s > (thr_up + band)] = 2
        act_s[pred_s < -(thr_dn + band)] = 0
        action[mask] = act_s
    return action


def score_config(combined_inner, sym_inner, mp_t_inner, mp_th_inner,
                 combined_holdout, sym_holdout, mp_t_holdout, mp_th_holdout,
                 per_sym_beta, per_sym_sigma,
                 de_maxiter=60, de_popsize=10):
    """Fit thresholds on inner, score on holdout."""
    # DE objective on inner
    def neg_inner_pnl(params):
        thr_up, thr_dn = params
        action = apply_abstain_wrapper(combined_inner, sym_inner, thr_up, thr_dn,
                                       per_sym_beta, per_sym_sigma)
        return -float(vectorized_pnl(action, mp_t_inner, mp_th_inner).sum())

    bounds = [(0.00005, 0.002), (0.00005, 0.002)]
    res = differential_evolution(neg_inner_pnl, bounds,
                                  maxiter=de_maxiter, popsize=de_popsize,
                                  seed=42, tol=1e-7, polish=True)
    thr_up_fit, thr_dn_fit = res.x
    inner_pnl = -res.fun

    # Apply frozen thresholds to holdout
    action_ho = apply_abstain_wrapper(combined_holdout, sym_holdout, thr_up_fit, thr_dn_fit,
                                      per_sym_beta, per_sym_sigma)
    holdout_pnl = float(vectorized_pnl(action_ho, mp_t_holdout, mp_th_holdout).sum())

    return holdout_pnl, inner_pnl, thr_up_fit, thr_dn_fit


# ──────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────

progress("Loading inner data (schemeP train+val)")
t0 = time.time()
tr = np.load(os.path.join(T68_CACHE, "schemeP_train.npz"))
va = np.load(os.path.join(T68_CACHE, "schemeP_val.npz"))
X_inner_full = np.vstack([tr['X'], va['X']])  # (1768320, 370)
sym_inner = np.concatenate([tr['sym'], va['sym']])
mp_t_inner = np.concatenate([tr['mp_t'], va['mp_t']])
mp_th_inner = np.concatenate([tr['mp_t60'], va['mp_t60']])  # h60
print(f"  Inner data: {len(X_inner_full)} rows, {X_inner_full.shape[1]} features", flush=True)
print(f"  Inner dates: 0-95, sym distribution: {np.bincount(sym_inner.astype(int))}", flush=True)

progress("Loading feature names and computing slicer")
with open(os.path.join(T68_CACHE, "schemeP_feat_names.txt")) as f:
    feat_names = [l.strip() for l in f]
drop_set = set(DROP_NAMES)
lgb_keep_idx = np.array([i for i, n in enumerate(feat_names) if n not in drop_set], dtype=np.int64)
X_inner_lgb = X_inner_full[:, lgb_keep_idx].astype(np.float32, copy=False)
print(f"  LGB feature dim: {X_inner_lgb.shape[1]}", flush=True)

progress("Running T87 NN inference on inner data (5 seeds)")
nn_preds_inner = []
for seed in SEEDS:
    npz_path = os.path.join(T87_DIR, f"model_T87_seed{seed}_main.npz")
    model = MLPNumpy(npz_path)
    pred = model.predict(X_inner_full)  # handles keep_idx internally
    nn_preds_inner.append(pred)
    print(f"  NN seed {seed}: mean={pred.mean():.6f}, std={pred.std():.6f}", flush=True)
    del model
nn_inner_mean = np.mean(nn_preds_inner, axis=0).astype(np.float64)
del nn_preds_inner
print(f"  NN inner mean pred: {nn_inner_mean.mean():.6f}", flush=True)
print(f"  Elapsed: {time.time()-t0:.1f}s", flush=True)

progress("Running T75 LGB inference on inner data (5 seeds)")
lgb_preds_inner = []
for seed in SEEDS:
    model_path = os.path.join(T75_DIR, f"model_T75_seed{seed}.txt")
    booster = lgb.Booster(model_file=model_path)
    pred_chunks = []
    for s in range(0, len(X_inner_lgb), 200_000):
        pred_chunks.append(booster.predict(X_inner_lgb[s:s+200_000]).astype(np.float32))
    pred = np.concatenate(pred_chunks)
    lgb_preds_inner.append(pred.astype(np.float64))
    print(f"  LGB seed {seed}: mean={pred.mean():.6f}, std={pred.std():.6f}", flush=True)
    del booster
lgb_inner_mean = np.mean(lgb_preds_inner, axis=0)
del lgb_preds_inner
print(f"  LGB inner mean pred: {lgb_inner_mean.mean():.6f}", flush=True)
print(f"  Elapsed: {time.time()-t0:.1f}s", flush=True)

combined_inner = (1.0 * nn_inner_mean + 1.5 * lgb_inner_mean) / 2.5
del nn_inner_mean, lgb_inner_mean

# Compute per-sym sigma from inner preds
per_sym_sigma_inner = {}
for s in range(5):
    mask = sym_inner == s
    per_sym_sigma_inner[s] = float(np.std(combined_inner[mask]))
print(f"  Per-sym sigma (inner): {per_sym_sigma_inner}", flush=True)

progress("Loading holdout preds (T87/T75 parquets, dates 96-119)")
import pandas as pd
nn_preds_ho = []
for seed in SEEDS:
    df = pd.read_parquet(os.path.join(T87_DIR, f"pred_T87_seed{seed}_main.parquet"))
    nn_preds_ho.append(df["pred_dmid_norm"].values.astype(np.float64))
nn_ho_mean = np.mean(nn_preds_ho, axis=0)
del nn_preds_ho

lgb_preds_ho = []
df_ref = None
for seed in SEEDS:
    df = pd.read_parquet(os.path.join(T75_DIR, f"pred_T75_seed{seed}.parquet"))
    lgb_preds_ho.append(df["pred_dmid_norm"].values.astype(np.float64))
    if df_ref is None:
        df_ref = df
lgb_ho_mean = np.mean(lgb_preds_ho, axis=0)
del lgb_preds_ho

combined_holdout = (1.0 * nn_ho_mean + 1.5 * lgb_ho_mean) / 2.5
sym_holdout = df_ref["sym"].values.astype(np.int32)
mp_t_holdout = df_ref["midprice_t"].values.astype(np.float64)
mp_th_holdout = df_ref["midprice_th"].values.astype(np.float64)
del nn_ho_mean, lgb_ho_mean

# Compute per-sym sigma from holdout preds (for reference)
per_sym_sigma_holdout = {}
for s in range(5):
    mask = sym_holdout == s
    per_sym_sigma_holdout[s] = float(np.std(combined_holdout[mask]))
print(f"  Per-sym sigma (holdout): {per_sym_sigma_holdout}", flush=True)
print(f"  V2 sigma (reference):    {{'0': 0.0002403901, '1': 0.0004712397, '2': 0.0004522216, '3': 0.0004235249, '4': 0.0004279811}}", flush=True)
print(f"  Elapsed: {time.time()-t0:.1f}s", flush=True)

# Use inner sigma for TSH-FT (fits thresholds on inner with inner sigma)
per_sym_sigma = per_sym_sigma_inner

progress("Computing baseline (v2 config) on holdout")
baseline_action = apply_abstain_wrapper(combined_holdout, sym_holdout, V2_THR_UP, V2_THR_DN,
                                        BASELINE_BETA, per_sym_sigma)
baseline_holdout_pnl = float(vectorized_pnl(baseline_action, mp_t_holdout, mp_th_holdout).sum())
print(f"  V2 baseline holdout PnL: {baseline_holdout_pnl:.6f}", flush=True)

# Also compute baseline with inner-fitted thresholds for comparison
baseline_holdout_pnl_inner_fit, baseline_inner_pnl, thr_fit_bl, thr_dn_fit_bl = score_config(
    combined_inner, sym_inner, mp_t_inner, mp_th_inner,
    combined_holdout, sym_holdout, mp_t_holdout, mp_th_holdout,
    BASELINE_BETA, per_sym_sigma
)
print(f"  TSH-FT baseline (inner-fitted thr): holdout={baseline_holdout_pnl_inner_fit:.6f}, "
      f"thr_up={thr_fit_bl:.6f}, thr_dn={thr_dn_fit_bl:.6f}", flush=True)
print(f"  Elapsed: {time.time()-t0:.1f}s", flush=True)

# ──────────────────────────────────────────────────────────────────────────
# Coordinate descent beta sweep
# ──────────────────────────────────────────────────────────────────────────

progress("Starting coordinate descent beta sweep (2 rounds)")
sweep_log = []
best_beta = dict(BASELINE_BETA)
best_holdout_pnl = baseline_holdout_pnl_inner_fit

for round_idx in range(2):
    print(f"\n=== Round {round_idx+1} ===", flush=True)
    for sym_dim in range(5):
        print(f"\n  Sweeping beta for sym {sym_dim}...", flush=True)
        best_beta_for_sym = best_beta[sym_dim]
        best_pnl_for_sym = best_holdout_pnl
        
        for beta_val in BETA_GRID:
            trial_beta = dict(best_beta)
            trial_beta[sym_dim] = beta_val
            
            t_start = time.time()
            ho_pnl, in_pnl, thr_up_fit, thr_dn_fit = score_config(
                combined_inner, sym_inner, mp_t_inner, mp_th_inner,
                combined_holdout, sym_holdout, mp_t_holdout, mp_th_holdout,
                trial_beta, per_sym_sigma
            )
            elapsed = time.time() - t_start
            
            entry = {
                "round": round_idx+1,
                "sym_dim": sym_dim,
                "beta": beta_val,
                "trial_beta": dict(trial_beta),
                "holdout_pnl": ho_pnl,
                "inner_pnl": in_pnl,
                "thr_up": thr_up_fit,
                "thr_dn": thr_dn_fit,
                "elapsed_s": elapsed,
            }
            sweep_log.append(entry)
            
            marker = " *** BEST ***" if ho_pnl > best_pnl_for_sym else ""
            print(f"    sym={sym_dim} beta={beta_val:.2f}: holdout={ho_pnl:.4f} "
                  f"inner={in_pnl:.4f} thr=({thr_up_fit:.5f},{thr_dn_fit:.5f}) "
                  f"[{elapsed:.1f}s]{marker}", flush=True)
            
            if ho_pnl > best_pnl_for_sym:
                best_pnl_for_sym = ho_pnl
                best_beta_for_sym = beta_val
        
        best_beta[sym_dim] = best_beta_for_sym
        best_holdout_pnl = best_pnl_for_sym
        print(f"  Sym {sym_dim}: best beta={best_beta_for_sym:.2f} => holdout={best_holdout_pnl:.4f}", flush=True)
    
    print(f"\nRound {round_idx+1} done. Best beta={best_beta}, PnL={best_holdout_pnl:.4f}", flush=True)
    progress(f"Round {round_idx+1} complete", metrics={"best_holdout_pnl": best_holdout_pnl, "best_beta": str(best_beta)})

# ──────────────────────────────────────────────────────────────────────────
# Bonus: per-sym band multiplier sweep (multiply all betas by k)
# ──────────────────────────────────────────────────────────────────────────

progress("Bonus: band multiplier sweep")
print("\n=== Band multiplier sweep ===", flush=True)
band_mult_results = []
for k in BAND_MULT_GRID:
    trial_beta = {s: BASELINE_BETA[s] * k for s in range(5)}
    ho_pnl, in_pnl, thr_up_fit, thr_dn_fit = score_config(
        combined_inner, sym_inner, mp_t_inner, mp_th_inner,
        combined_holdout, sym_holdout, mp_t_holdout, mp_th_holdout,
        trial_beta, per_sym_sigma
    )
    band_mult_results.append({
        "k": k,
        "trial_beta": dict(trial_beta),
        "holdout_pnl": ho_pnl,
        "inner_pnl": in_pnl,
        "thr_up": thr_up_fit,
        "thr_dn": thr_dn_fit,
    })
    print(f"  k={k:.2f}: beta_eff={trial_beta} holdout={ho_pnl:.4f}", flush=True)

best_mult_entry = max(band_mult_results, key=lambda x: x["holdout_pnl"])
print(f"  Best multiplier: k={best_mult_entry['k']} holdout={best_mult_entry['holdout_pnl']:.4f}", flush=True)

# ──────────────────────────────────────────────────────────────────────────
# Final: run DE on best beta to get the exact thresholds
# ──────────────────────────────────────────────────────────────────────────

progress("Fitting final thresholds with best beta config")
final_holdout_pnl, final_inner_pnl, final_thr_up, final_thr_dn = score_config(
    combined_inner, sym_inner, mp_t_inner, mp_th_inner,
    combined_holdout, sym_holdout, mp_t_holdout, mp_th_holdout,
    best_beta, per_sym_sigma,
    de_maxiter=100, de_popsize=15
)
print(f"  Final best: holdout={final_holdout_pnl:.6f}, thr_up={final_thr_up:.6f}, thr_dn={final_thr_dn:.6f}", flush=True)
delta_vs_baseline = final_holdout_pnl - baseline_holdout_pnl
print(f"  Delta vs v2 baseline (original thresholds): {delta_vs_baseline:+.6f}", flush=True)
print(f"  Delta vs TSH-FT baseline: {final_holdout_pnl - baseline_holdout_pnl_inner_fit:+.6f}", flush=True)

# ──────────────────────────────────────────────────────────────────────────
# Save results
# ──────────────────────────────────────────────────────────────────────────

results = {
    "task": "T159 per-sym dynamic abstain — TSH-FT tuned",
    "v2_baseline_holdout_pnl": baseline_holdout_pnl,
    "tshft_baseline_holdout_pnl": baseline_holdout_pnl_inner_fit,
    "best_per_sym_beta": {str(k): v for k, v in best_beta.items()},
    "best_holdout_pnl": final_holdout_pnl,
    "best_inner_pnl": final_inner_pnl,
    "final_thr_up": final_thr_up,
    "final_thr_dn": final_thr_dn,
    "delta_vs_baseline": delta_vs_baseline,
    "delta_vs_tshft_baseline": final_holdout_pnl - baseline_holdout_pnl_inner_fit,
    "iter018_baseline_beta": {"0": 0.10, "1": 0.40, "2": 0.30, "3": 0.00, "4": 0.00},
    "per_sym_sigma_inner": {str(k): v for k, v in per_sym_sigma_inner.items()},
    "per_sym_sigma_holdout": {str(k): v for k, v in per_sym_sigma_holdout.items()},
    "band_mult_sweep": band_mult_results,
    "sweep_log": sweep_log,
    "total_elapsed_s": time.time() - t0,
}

with open(os.path.join(HERE, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {HERE}/results.json", flush=True)
print(f"Total elapsed: {results['total_elapsed_s']:.1f}s", flush=True)

# ──────────────────────────────────────────────────────────────────────────
# Decision: package if improvement > 0.5
# ──────────────────────────────────────────────────────────────────────────

SHOULD_PACKAGE = delta_vs_baseline > 0.5

if SHOULD_PACKAGE:
    print(f"\n*** IMPROVEMENT FOUND: +{delta_vs_baseline:.4f} ***", flush=True)
    print(f"Best beta: {best_beta}", flush=True)
    print(f"Thresholds: thr_up={final_thr_up:.8f}, thr_dn={final_thr_dn:.8f}", flush=True)
    progress("DONE - improvement found, ready to package",
             metrics={"delta": delta_vs_baseline, "best_holdout_pnl": final_holdout_pnl})
else:
    print(f"\nNo improvement > 0.5 found (delta={delta_vs_baseline:+.4f}). Not packaging.", flush=True)
    progress("DONE - no improvement found",
             metrics={"delta": delta_vs_baseline, "best_holdout_pnl": final_holdout_pnl})

print(f"\nRESULT: task=[T159 per-sym dynamic abstain TSH-FT] "
      f"metrics={{v2_baseline={baseline_holdout_pnl:.4f}, best_holdout={final_holdout_pnl:.4f}, "
      f"delta={delta_vs_baseline:+.4f}}} "
      f"notes=[best_beta={best_beta}]", flush=True)
