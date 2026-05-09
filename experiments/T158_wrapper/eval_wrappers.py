"""T158: Wrapper improvements — agreement filter + magnitude shrinkage.

Evaluates on holdout dates 96-119 using precomputed T87 NN + T75 LGB preds.
All eval against v2 baseline (T150 tuned: w_lgb=0.5, beta={0:0.1,1:0.5,2:0.3,3:0,4:0}).
"""
import json
import os
import time
from datetime import datetime, timezone
from itertools import product

import numpy as np
import pandas as pd

OUTDIR = os.path.dirname(os.path.abspath(__file__))

NN_SEEDS = [1, 7, 13, 42, 100]
LGB_SEEDS = [1, 7, 13, 42, 100]

NN_PRED_TPL = (
    "/root/projects/liangwenbei_workdir/experiments/T87_spo_dfl/"
    "pred_T87_seed{seed}_main.parquet"
)
LGB_PRED_TPL = (
    "/root/projects/liangwenbei_workdir/experiments/T75_regression_dmid/"
    "pred_T75_seed{seed}.parquet"
)

# iter_018 thresholds (unchanged)
THR_UP = 0.00029995433796130955
THR_DN = 0.0002158528296175958

PER_SYM_SIGMA = {
    0: 0.0002403901,
    1: 0.0004712397,
    2: 0.0004522216,
    3: 0.0004235249,
    4: 0.0004279811,
}
DEFAULT_SIGMA_OOD = 0.0003998317

# V2 best config (T150 sweep C)
V2_BETA = {0: 0.1, 1: 0.5, 2: 0.3, 3: 0.0, 4: 0.0}
V2_W_NN = 1.0
V2_W_LGB = 0.5
DEFAULT_BETA_OOD = 0.16

FEE = 0.0001


def update_progress(step, metrics=None):
    data = {
        "status": "running",
        "step": step,
        "metrics": metrics or {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(OUTDIR, "worker-progress.json"), "w") as f:
        json.dump(data, f, indent=2)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {step}", flush=True)


def vectorized_pnl(action, mp_t, mp_th):
    side = action.astype(np.float64) - 1.0
    abs_side = np.abs(side)
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * abs_side * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee_pnl) / denom


def apply_gate_conformal(combined_pred, sym_arr, per_sym_beta):
    """Standard conformal gate: thr + beta*sigma band."""
    action = np.ones(len(combined_pred), dtype=np.int8)
    for s in range(5):
        mask = sym_arr == s
        beta = per_sym_beta.get(s, DEFAULT_BETA_OOD)
        sigma = PER_SYM_SIGMA.get(s, DEFAULT_SIGMA_OOD)
        band = beta * sigma
        pred_s = combined_pred[mask]
        act_s = np.ones(mask.sum(), dtype=np.int8)
        act_s[pred_s > (THR_UP + band)] = 2
        act_s[pred_s < -(THR_DN + band)] = 0
        action[mask] = act_s
    ood_mask = ~np.isin(sym_arr, list(range(5)))
    if ood_mask.any():
        ood_band = DEFAULT_BETA_OOD * DEFAULT_SIGMA_OOD
        pred_ood = combined_pred[ood_mask]
        act_ood = np.ones(ood_mask.sum(), dtype=np.int8)
        act_ood[pred_ood > (THR_UP + ood_band)] = 2
        act_ood[pred_ood < -(THR_DN + ood_band)] = 0
        action[ood_mask] = act_ood
    return action


def compute_base_pnl(combined_pred, sym_arr, mp_t, mp_th, per_sym_beta=V2_BETA):
    action = apply_gate_conformal(combined_pred, sym_arr, per_sym_beta)
    pnl = vectorized_pnl(action, mp_t, mp_th)
    n_trades = int((action != 1).sum())
    return float(pnl.sum()), n_trades


def make_combined(nn_mean, lgb_mean, w_nn=V2_W_NN, w_lgb=V2_W_LGB):
    return (w_nn * nn_mean + w_lgb * lgb_mean) / (w_nn + w_lgb)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ── Load data ─────────────────────────────────────────────────────────────────

update_progress("Loading NN predictions (5 seeds)")
nn_preds_list = []
for seed in NN_SEEDS:
    df = pd.read_parquet(NN_PRED_TPL.format(seed=seed))
    nn_preds_list.append(df["pred_dmid_norm"].values.astype(np.float64))

df_ref = pd.read_parquet(NN_PRED_TPL.format(seed=1))
nn_pred_mean = np.mean(nn_preds_list, axis=0)
print(f"  NN rows: {len(nn_pred_mean)}, dates: {df_ref['date'].min()}-{df_ref['date'].max()}", flush=True)

update_progress("Loading LGB predictions (5 seeds)")
lgb_preds_list = []
for seed in LGB_SEEDS:
    df = pd.read_parquet(LGB_PRED_TPL.format(seed=seed))
    lgb_preds_list.append(df["pred_dmid_norm"].values.astype(np.float64))

df_lgb_ref = pd.read_parquet(LGB_PRED_TPL.format(seed=1))
lgb_pred_mean = np.mean(lgb_preds_list, axis=0)

# Verify alignment
assert len(nn_pred_mean) == len(lgb_pred_mean), "Row count mismatch!"
print(f"  LGB rows: {len(lgb_pred_mean)}", flush=True)

sym_arr = df_ref["sym"].values.astype(np.int32)
mp_t = df_ref["midprice_t"].values.astype(np.float64)
mp_th = df_ref["midprice_th"].values.astype(np.float64)

# ── V2 Baseline ───────────────────────────────────────────────────────────────

update_progress("Computing v2 baseline PnL")
combined_v2 = make_combined(nn_pred_mean, lgb_pred_mean)
v2_pnl, v2_trades = compute_base_pnl(combined_v2, sym_arr, mp_t, mp_th, V2_BETA)
print(f"  V2 baseline holdout PnL: {v2_pnl:.6f}, n_trades={v2_trades}", flush=True)

# ── Wrapper 1: NN/LGB Agreement Filter ───────────────────────────────────────

update_progress("Wrapper 1: Agreement filter")

# Combined pred uses w_nn=1.0, w_lgb=0.5
combined_w1 = make_combined(nn_pred_mean, lgb_pred_mean)
# Apply agreement mask
agree_mask = np.sign(nn_pred_mean) == np.sign(lgb_pred_mean)
# Compute standard gate actions
actions_w1 = apply_gate_conformal(combined_w1, sym_arr, V2_BETA)
# Override to hold where disagreement
actions_w1[~agree_mask] = 1
pnl_w1_arr = vectorized_pnl(actions_w1, mp_t, mp_th)
pnl_w1 = float(pnl_w1_arr.sum())
n_trades_w1 = int((actions_w1 != 1).sum())
delta_w1 = pnl_w1 - v2_pnl
print(f"  W1 agreement PnL: {pnl_w1:.6f}, n_trades={n_trades_w1}, delta={delta_w1:+.6f}", flush=True)

# ── Wrapper 2: Magnitude Shrinkage ───────────────────────────────────────────

update_progress("Wrapper 2: Magnitude shrinkage sweep")

# Per-sym sigma for shrinkage (global mean ≈ 4e-4)
sigma_by_sym = np.vectorize(lambda s: PER_SYM_SIGMA.get(s, DEFAULT_SIGMA_OOD))(sym_arr)

alpha_grid = [0.5, 1.0, 1.5, 2.0]
beta_shrink_grid = [1.0, 2.0, 5.0, 10.0]

sweep_w2_results = []
best_w2_pnl = -np.inf
best_w2_cfg = None

for alpha, beta_shrink in product(alpha_grid, beta_shrink_grid):
    abs_pred = np.abs(combined_v2)
    ratio = abs_pred / sigma_by_sym
    shrink_factor = sigmoid((ratio - alpha) * beta_shrink)
    pred_shrunk = combined_v2 * shrink_factor
    actions = apply_gate_conformal(pred_shrunk, sym_arr, V2_BETA)
    pnl_val = float(vectorized_pnl(actions, mp_t, mp_th).sum())
    n_trades = int((actions != 1).sum())
    sweep_w2_results.append({
        "alpha": alpha,
        "beta_shrink": beta_shrink,
        "pnl": pnl_val,
        "n_trades": n_trades,
        "delta": pnl_val - v2_pnl,
    })
    if pnl_val > best_w2_pnl:
        best_w2_pnl = pnl_val
        best_w2_cfg = {"alpha": alpha, "beta_shrink": beta_shrink, "pnl": pnl_val, "n_trades": n_trades}
    print(f"  alpha={alpha}, beta_shrink={beta_shrink}: PnL={pnl_val:.6f} delta={pnl_val-v2_pnl:+.6f}", flush=True)

print(f"  Best W2: alpha={best_w2_cfg['alpha']}, beta_shrink={best_w2_cfg['beta_shrink']}, PnL={best_w2_pnl:.6f}, delta={best_w2_pnl-v2_pnl:+.6f}", flush=True)

# ── Wrapper 1+2 Combined ──────────────────────────────────────────────────────

update_progress("Wrapper 1+2 Combined")

alpha_best = best_w2_cfg["alpha"]
beta_shrink_best = best_w2_cfg["beta_shrink"]

abs_pred_c = np.abs(combined_v2)
ratio_c = abs_pred_c / sigma_by_sym
shrink_factor_c = sigmoid((ratio_c - alpha_best) * beta_shrink_best)
pred_shrunk_c = combined_v2 * shrink_factor_c

actions_comb = apply_gate_conformal(pred_shrunk_c, sym_arr, V2_BETA)
actions_comb[~agree_mask] = 1  # agreement filter on top
pnl_comb = float(vectorized_pnl(actions_comb, mp_t, mp_th).sum())
n_trades_comb = int((actions_comb != 1).sum())
delta_comb = pnl_comb - v2_pnl
print(f"  Combined PnL: {pnl_comb:.6f}, n_trades={n_trades_comb}, delta={delta_comb:+.6f}", flush=True)

# ── Summary ───────────────────────────────────────────────────────────────────

results = {
    "v2_baseline": {
        "holdout_pnl": v2_pnl,
        "n_trades": v2_trades,
        "w_lgb": V2_W_LGB,
        "beta": V2_BETA,
    },
    "wrapper_1_agreement": {
        "holdout_pnl": pnl_w1,
        "n_trades": n_trades_w1,
        "delta_vs_v2": delta_w1,
        "note": "abstain if sign(nn) != sign(lgb), then standard gate+conformal",
    },
    "wrapper_2_shrinkage": {
        "best": {
            "alpha": best_w2_cfg["alpha"],
            "beta_shrink": best_w2_cfg["beta_shrink"],
            "holdout_pnl": best_w2_pnl,
            "n_trades": best_w2_cfg["n_trades"],
            "delta_vs_v2": best_w2_pnl - v2_pnl,
        },
        "sweep": sweep_w2_results,
    },
    "wrapper_combined": {
        "alpha": alpha_best,
        "beta_shrink": beta_shrink_best,
        "holdout_pnl": pnl_comb,
        "n_trades": n_trades_comb,
        "delta_vs_v2": delta_comb,
        "note": "agreement filter + shrinkage together",
    },
}

# Determine best overall
best_pnl = v2_pnl
best_name = "v2_baseline (no improvement)"
for name, data in [("wrapper_1", pnl_w1), ("wrapper_2", best_w2_pnl), ("wrapper_combined", pnl_comb)]:
    if data > best_pnl:
        best_pnl = data
        best_name = name

results["best_overall"] = best_name
results["best_overall_pnl"] = best_pnl
results["best_delta"] = best_pnl - v2_pnl

out_path = os.path.join(OUTDIR, "results_wrappers.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved wrapper eval results to {out_path}", flush=True)
print(f"\n=== SUMMARY ===")
print(f"V2 baseline:       {v2_pnl:.6f} ({v2_trades} trades)")
print(f"W1 agreement:      {pnl_w1:.6f} ({n_trades_w1} trades) delta={delta_w1:+.6f}")
print(f"W2 best shrink:    {best_w2_pnl:.6f} alpha={alpha_best} beta_s={beta_shrink_best} delta={best_w2_pnl-v2_pnl:+.6f}")
print(f"Combined W1+W2:    {pnl_comb:.6f} ({n_trades_comb} trades) delta={delta_comb:+.6f}")
print(f"Best overall:      {best_name} ({best_pnl:.6f}, delta={best_pnl-v2_pnl:+.6f})")
