"""T158 ext: Extended agreement filter variations.

Tries more specific agreement filter formulations to push past +0.5 delta.
"""
import json
import os
import numpy as np
import pandas as pd
from itertools import product
from datetime import datetime, timezone

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

THR_UP = 0.00029995433796130955
THR_DN = 0.0002158528296175958
PER_SYM_SIGMA = {0: 0.0002403901, 1: 0.0004712397, 2: 0.0004522216, 3: 0.0004235249, 4: 0.0004279811}
DEFAULT_SIGMA_OOD = 0.0003998317
V2_BETA = {0: 0.1, 1: 0.5, 2: 0.3, 3: 0.0, 4: 0.0}
V2_W_NN = 1.0
V2_W_LGB = 0.5
DEFAULT_BETA_OOD = 0.16
FEE = 0.0001


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def vectorized_pnl(action, mp_t, mp_th):
    side = action.astype(np.float64) - 1.0
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    return (side * diff - fee_pnl) / (mp_t.astype(np.float64) + 1.0)


def apply_gate_conformal(combined_pred, sym_arr, per_sym_beta=V2_BETA):
    action = np.ones(len(combined_pred), dtype=np.int8)
    for s in range(5):
        mask = sym_arr == s
        band = per_sym_beta.get(s, DEFAULT_BETA_OOD) * PER_SYM_SIGMA.get(s, DEFAULT_SIGMA_OOD)
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


# Load data
print("Loading preds...", flush=True)
nn_preds_list, lgb_preds_list = [], []
for seed in NN_SEEDS:
    df = pd.read_parquet(NN_PRED_TPL.format(seed=seed))
    nn_preds_list.append(df["pred_dmid_norm"].values.astype(np.float64))
for seed in LGB_SEEDS:
    df = pd.read_parquet(LGB_PRED_TPL.format(seed=seed))
    lgb_preds_list.append(df["pred_dmid_norm"].values.astype(np.float64))

df_ref = pd.read_parquet(NN_PRED_TPL.format(seed=1))
nn_pred_mean = np.mean(nn_preds_list, axis=0)
lgb_pred_mean = np.mean(lgb_preds_list, axis=0)
sym_arr = df_ref["sym"].values.astype(np.int32)
mp_t = df_ref["midprice_t"].values.astype(np.float64)
mp_th = df_ref["midprice_th"].values.astype(np.float64)

combined_v2 = (V2_W_NN * nn_pred_mean + V2_W_LGB * lgb_pred_mean) / (V2_W_NN + V2_W_LGB)
sigma_by_sym = np.vectorize(lambda s: PER_SYM_SIGMA.get(s, DEFAULT_SIGMA_OOD))(sym_arr)

# V2 baseline
base_actions = apply_gate_conformal(combined_v2, sym_arr)
v2_pnl = float(vectorized_pnl(base_actions, mp_t, mp_th).sum())
v2_trades = int((base_actions != 1).sum())
print(f"V2 baseline: {v2_pnl:.6f} ({v2_trades} trades)", flush=True)

# Precompute: agreement mask
agree_mask = np.sign(nn_pred_mean) == np.sign(lgb_pred_mean)

results_ext = {}

# ── Variation: different min-magnitude thresholds for agreement ──────────────
# Only trade if BOTH models exceed frac*sigma in the same direction
print("\nAgreement + min-magnitude sweep:", flush=True)
frac_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
mag_sweep = []
for frac in frac_grid:
    nn_min_mask = np.abs(nn_pred_mean) >= frac * sigma_by_sym
    lgb_min_mask = np.abs(lgb_pred_mean) >= frac * sigma_by_sym
    filter_mask = agree_mask & nn_min_mask & lgb_min_mask
    actions = apply_gate_conformal(combined_v2, sym_arr)
    actions[~filter_mask] = 1
    pnl_val = float(vectorized_pnl(actions, mp_t, mp_th).sum())
    n_trades = int((actions != 1).sum())
    delta = pnl_val - v2_pnl
    mag_sweep.append({"frac": frac, "pnl": pnl_val, "n_trades": n_trades, "delta": delta})
    print(f"  frac={frac:.1f}: PnL={pnl_val:.6f} ({n_trades} trades) delta={delta:+.6f}", flush=True)

results_ext["agreement_minmag_sweep"] = mag_sweep

# ── Variation: per-sym agreement (only for specific syms) ───────────────────
# Which syms benefit from agreement filter?
print("\nPer-sym agreement analysis:", flush=True)
per_sym_data = {}
for s in range(5):
    mask_s = sym_arr == s
    base_actions_s = base_actions[mask_s]
    pnl_base_s = float(vectorized_pnl(base_actions_s, mp_t[mask_s], mp_th[mask_s]).sum())
    # Apply agreement to this sym only
    actions_w1_s = base_actions.copy()
    disagree_s = mask_s & ~agree_mask
    actions_w1_s[disagree_s] = 1
    pnl_w1_s = float(vectorized_pnl(actions_w1_s[mask_s], mp_t[mask_s], mp_th[mask_s]).sum())
    n_disagree_s = int(disagree_s.sum())
    # Per sym delta when applying the filter
    delta_s = pnl_w1_s - pnl_base_s
    per_sym_data[str(s)] = {
        "pnl_base": pnl_base_s, "pnl_w1": pnl_w1_s, "delta": delta_s,
        "n_disagree_filtered": n_disagree_s
    }
    print(f"  sym {s}: base={pnl_base_s:.4f}, w1={pnl_w1_s:.4f}, delta={delta_s:+.4f}, filtered={n_disagree_s}", flush=True)

results_ext["per_sym_agreement"] = per_sym_data

# ── Variation: selective agreement by sym ────────────────────────────────────
# Apply agreement filter only to syms where it helps
from itertools import combinations

print("\nSelective sym agreement sweep:", flush=True)
# Find which combination of syms to apply agreement filter
good_syms = [s for s, v in per_sym_data.items() if v["delta"] > 0]
print(f"  Beneficial syms: {good_syms}", flush=True)
if good_syms:
    good_sym_ints = [int(s) for s in good_syms]
    filter_good = np.zeros(len(combined_v2), dtype=bool)
    for s in good_sym_ints:
        filter_good |= (sym_arr == s)
    # Apply agreement only to beneficial syms
    actions_sel = base_actions.copy()
    sel_disagree = filter_good & ~agree_mask
    actions_sel[sel_disagree] = 1
    pnl_sel = float(vectorized_pnl(actions_sel, mp_t, mp_th).sum())
    n_sel = int((actions_sel != 1).sum())
    delta_sel = pnl_sel - v2_pnl
    print(f"  Selective (syms {good_sym_ints}): PnL={pnl_sel:.6f} ({n_sel} trades) delta={delta_sel:+.6f}", flush=True)
    results_ext["selective_agreement"] = {
        "syms": good_sym_ints, "pnl": pnl_sel, "n_trades": n_sel, "delta": delta_sel
    }

# ── Variation: W1 + per-sym beta re-tune ─────────────────────────────────────
# After applying agreement filter, maybe per-sym beta needs retuning?
# Quick scan with beta=0 everywhere (no conformal band) under agreement filter
print("\nW1 + beta=0 (no conformal band):", flush=True)
beta_zero = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
actions_nb = apply_gate_conformal(combined_v2, sym_arr, beta_zero)
actions_nb[~agree_mask] = 1
pnl_nb = float(vectorized_pnl(actions_nb, mp_t, mp_th).sum())
n_nb = int((actions_nb != 1).sum())
print(f"  W1 + no-band: PnL={pnl_nb:.6f} ({n_nb} trades) delta={pnl_nb-v2_pnl:+.6f}", flush=True)
results_ext["w1_no_band"] = {"pnl": pnl_nb, "n_trades": n_nb, "delta": pnl_nb - v2_pnl}

# ── Variation: NN-only vs LGB-only when disagreement ─────────────────────────
# Instead of abstaining, use only one model when they disagree
print("\nDisagreement fallback to stronger model:", flush=True)
# When disagree, use just NN prediction
combined_nn_only = nn_pred_mean  # pure NN pred
combined_lgb_only = lgb_pred_mean  # pure LGB pred

for name, pred_disagree in [("nn_only", combined_nn_only), ("lgb_only", combined_lgb_only)]:
    actions_fb = base_actions.copy()
    # Replace disagreement actions with single-model decisions
    disagree_mask = ~agree_mask
    if disagree_mask.any():
        actions_fb_disagree = apply_gate_conformal(pred_disagree, sym_arr)
        actions_fb[disagree_mask] = actions_fb_disagree[disagree_mask]
    pnl_fb = float(vectorized_pnl(actions_fb, mp_t, mp_th).sum())
    n_fb = int((actions_fb != 1).sum())
    print(f"  fallback_{name}: PnL={pnl_fb:.6f} ({n_fb} trades) delta={pnl_fb-v2_pnl:+.6f}", flush=True)
    results_ext[f"fallback_{name}"] = {"pnl": pnl_fb, "n_trades": n_fb, "delta": pnl_fb - v2_pnl}

# ── Tight W2: try alpha<0.5, very gentle shrinkage ────────────────────────────
print("\nExtra-gentle shrinkage (very low alpha):", flush=True)
gentle_results = []
for alpha in [0.01, 0.05, 0.1, 0.2, 0.3]:
    for beta_s in [0.1, 0.5, 1.0]:
        abs_pred = np.abs(combined_v2)
        shrink = sigmoid((abs_pred / sigma_by_sym - alpha) * beta_s)
        pred_s = combined_v2 * shrink
        acts = apply_gate_conformal(pred_s, sym_arr)
        pnl_g = float(vectorized_pnl(acts, mp_t, mp_th).sum())
        n_g = int((acts != 1).sum())
        delta_g = pnl_g - v2_pnl
        gentle_results.append({"alpha": alpha, "beta_s": beta_s, "pnl": pnl_g, "n_trades": n_g, "delta": delta_g})
        print(f"  alpha={alpha}, beta_s={beta_s}: PnL={pnl_g:.6f} delta={delta_g:+.6f}", flush=True)
results_ext["gentle_shrinkage"] = gentle_results

# Find overall best
all_configs = [
    ("v2_baseline", v2_pnl, v2_trades),
    ("w1_agreement", float(mag_sweep[0]["pnl"]), int(mag_sweep[0]["n_trades"])),
]
if good_syms:
    all_configs.append(("selective_agreement", float(results_ext["selective_agreement"]["pnl"]), int(results_ext["selective_agreement"]["n_trades"])))

best_ext_name = "v2_baseline"
best_ext_pnl = v2_pnl
for name, pnl_val, _ in all_configs:
    if pnl_val > best_ext_pnl:
        best_ext_pnl = pnl_val
        best_ext_name = name

# Also check mag sweep
for row in mag_sweep:
    if row["pnl"] > best_ext_pnl:
        best_ext_pnl = row["pnl"]
        best_ext_name = f"agreement_minmag_frac{row['frac']}"

results_ext["best_ext_overall"] = {"name": best_ext_name, "pnl": best_ext_pnl, "delta": best_ext_pnl - v2_pnl}

with open(os.path.join(OUTDIR, "results_ext.json"), "w") as f:
    json.dump(results_ext, f, indent=2)
print(f"\nBest ext: {best_ext_name} = {best_ext_pnl:.6f} (delta={best_ext_pnl-v2_pnl:+.6f})", flush=True)
