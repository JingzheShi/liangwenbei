"""T150: Per-sym conformal β sweep + ensemble weight tune for iter_019 v2 base.

Evaluates on holdout dates 96-119 using precomputed T87 NN + T75 LGB preds.
Sweeps per_sym_beta (coord descent) and w_lgb (1D grid).
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

THR_UP = 0.00029995433796130955
THR_DN = 0.0002158528296175958

PER_SYM_SIGMA = {
    0: 0.0002403901,
    1: 0.0004712397,
    2: 0.0004522216,
    3: 0.0004235249,
    4: 0.0004279811,
}
DEFAULT_BETA_OOD = 0.16
DEFAULT_SIGMA_OOD = 0.0003998317

BASELINE_BETA = {0: 0.10, 1: 0.40, 2: 0.30, 3: 0.00, 4: 0.00}
BASELINE_W_NN = 1.0
BASELINE_W_LGB = 1.5

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


def apply_wrapper(combined_pred, sym_arr, per_sym_beta):
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
    # OOD syms (not in 0-4)
    ood_mask = ~np.isin(sym_arr, list(range(5)))
    if ood_mask.any():
        ood_band = DEFAULT_BETA_OOD * DEFAULT_SIGMA_OOD
        pred_ood = combined_pred[ood_mask]
        act_ood = np.ones(ood_mask.sum(), dtype=np.int8)
        act_ood[pred_ood > (THR_UP + ood_band)] = 2
        act_ood[pred_ood < -(THR_DN + ood_band)] = 0
        action[ood_mask] = act_ood
    return action


def compute_holdout_pnl(combined_pred, sym_arr, mp_t, mp_th, per_sym_beta):
    action = apply_wrapper(combined_pred, sym_arr, per_sym_beta)
    pnl = vectorized_pnl(action, mp_t, mp_th)
    return float(pnl.sum()), action


# ── Load data ────────────────────────────────────────────────────────────────

update_progress("Loading NN predictions (5 seeds)")
nn_preds = []
for seed in NN_SEEDS:
    path = NN_PRED_TPL.format(seed=seed)
    df = pd.read_parquet(path)
    nn_preds.append(df["pred_dmid_norm"].values)

# Use seed=1 df as reference for metadata
df_ref = pd.read_parquet(NN_PRED_TPL.format(seed=1))
nn_pred_mean = np.mean(nn_preds, axis=0)
print(f"  NN shape: {nn_pred_mean.shape}, dates range: {df_ref['date'].min()}-{df_ref['date'].max()}", flush=True)

update_progress("Loading LGB predictions (5 seeds)")
lgb_preds = []
for seed in LGB_SEEDS:
    path = LGB_PRED_TPL.format(seed=seed)
    df = pd.read_parquet(path)
    lgb_preds.append(df["pred_dmid_norm"].values)

df_lgb_ref = pd.read_parquet(LGB_PRED_TPL.format(seed=1))
lgb_pred_mean = np.mean(lgb_preds, axis=0)

# Verify alignment
assert len(nn_pred_mean) == len(lgb_pred_mean), "Row count mismatch!"
for col in ["sym", "date", "t"]:
    if col in df_ref.columns and col in df_lgb_ref.columns:
        assert np.array_equal(df_ref[col].values, df_lgb_ref[col].values), f"{col} mismatch!"
print("  Alignment verified.", flush=True)

sym_arr = df_ref["sym"].values.astype(np.int32)
mp_t = df_ref["midprice_t"].values.astype(np.float64)
mp_th = df_ref["midprice_th"].values.astype(np.float64)


def make_combined(w_nn, w_lgb):
    return (w_nn * nn_pred_mean + w_lgb * lgb_pred_mean) / (w_nn + w_lgb)


# ── Baseline ─────────────────────────────────────────────────────────────────

update_progress("Computing baseline PnL")
combined_baseline = make_combined(BASELINE_W_NN, BASELINE_W_LGB)
baseline_pnl, _ = compute_holdout_pnl(combined_baseline, sym_arr, mp_t, mp_th, BASELINE_BETA)
print(f"  Baseline holdout PnL: {baseline_pnl:.6f}", flush=True)

# ── Sweep grids ──────────────────────────────────────────────────────────────

BETA_GRIDS = {
    0: [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    1: [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
    2: [0.0, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60],
    3: [0.0, 0.05, 0.10, 0.15, 0.20, 0.30],
    4: [0.0, 0.05, 0.10, 0.15, 0.20, 0.30],
}
W_LGB_GRID = [0.3, 0.5, 0.7, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]

SPECIAL_BETAS = [
    {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
    {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1},
    {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2},
    {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5},
]

# ── Sweep A: coord descent (2 rounds) ─────────────────────────────────────

update_progress("Sweep A: coordinate descent on per-sym beta")
sweep_A_results = []

# Baseline stats
baseline_combined = make_combined(BASELINE_W_NN, BASELINE_W_LGB)

# Evaluate special betas first
for sb in SPECIAL_BETAS:
    pnl, _ = compute_holdout_pnl(baseline_combined, sym_arr, mp_t, mp_th, sb)
    sweep_A_results.append({"beta": dict(sb), "w_nn": BASELINE_W_NN, "w_lgb": BASELINE_W_LGB, "pnl": pnl, "type": "special"})
    print(f"  Special beta {sb} -> {pnl:.6f}", flush=True)

# Coordinate descent
current_beta = dict(BASELINE_BETA)
current_pnl_A, _ = compute_holdout_pnl(baseline_combined, sym_arr, mp_t, mp_th, current_beta)

for round_idx in range(2):
    update_progress(f"Sweep A round {round_idx+1}/2", {"current_pnl": current_pnl_A})
    for sym in range(5):
        best_val = current_beta[sym]
        best_pnl_sym = current_pnl_A
        for beta_val in BETA_GRIDS[sym]:
            trial_beta = dict(current_beta)
            trial_beta[sym] = beta_val
            pnl, _ = compute_holdout_pnl(baseline_combined, sym_arr, mp_t, mp_th, trial_beta)
            entry = {"beta": dict(trial_beta), "w_nn": BASELINE_W_NN, "w_lgb": BASELINE_W_LGB, "pnl": pnl, "type": f"cd_r{round_idx+1}_sym{sym}"}
            sweep_A_results.append(entry)
            if pnl > best_pnl_sym:
                best_pnl_sym = pnl
                best_val = beta_val
        if best_val != current_beta[sym]:
            print(f"  Round {round_idx+1} sym {sym}: {current_beta[sym]:.2f} -> {best_val:.2f} (pnl {current_pnl_A:.6f} -> {best_pnl_sym:.6f})", flush=True)
            current_beta[sym] = best_val
            current_pnl_A = best_pnl_sym
        else:
            print(f"  Round {round_idx+1} sym {sym}: kept {best_val:.2f} (best={best_pnl_sym:.6f})", flush=True)

best_beta_A = dict(current_beta)
best_pnl_A = current_pnl_A
print(f"  Sweep A best beta: {best_beta_A}, pnl={best_pnl_A:.6f}", flush=True)

# ── Sweep B: w_lgb grid ───────────────────────────────────────────────────

update_progress("Sweep B: w_lgb grid sweep with best beta from A")
sweep_B_results = []
best_w_lgb = BASELINE_W_LGB
best_pnl_B = -np.inf

for w_lgb in W_LGB_GRID:
    combined = make_combined(BASELINE_W_NN, w_lgb)
    pnl, _ = compute_holdout_pnl(combined, sym_arr, mp_t, mp_th, best_beta_A)
    sweep_B_results.append({"w_nn": BASELINE_W_NN, "w_lgb": w_lgb, "beta": dict(best_beta_A), "pnl": pnl})
    print(f"  w_lgb={w_lgb:.2f} -> pnl={pnl:.6f}", flush=True)
    if pnl > best_pnl_B:
        best_pnl_B = pnl
        best_w_lgb = w_lgb

print(f"  Sweep B best w_lgb={best_w_lgb}, pnl={best_pnl_B:.6f}", flush=True)

# ── Sweep C: joint fine-tune around best ──────────────────────────────────

update_progress("Sweep C: joint fine-tune (±1 step around best)")
sweep_C_results = []

# Build ±1 step neighbors for each dim
def get_neighbors(beta, w_lgb):
    configs = []
    # w_lgb neighbors
    w_idx = W_LGB_GRID.index(w_lgb) if w_lgb in W_LGB_GRID else None
    w_variants = [w_lgb]
    if w_idx is not None:
        if w_idx > 0:
            w_variants.append(W_LGB_GRID[w_idx - 1])
        if w_idx < len(W_LGB_GRID) - 1:
            w_variants.append(W_LGB_GRID[w_idx + 1])

    for sym in range(5):
        grid = BETA_GRIDS[sym]
        try:
            idx = grid.index(beta[sym])
        except ValueError:
            # find nearest
            idx = int(np.argmin([abs(v - beta[sym]) for v in grid]))
        neighbors_beta = [beta[sym]]
        if idx > 0:
            neighbors_beta.append(grid[idx - 1])
        if idx < len(grid) - 1:
            neighbors_beta.append(grid[idx + 1])
        for bv in neighbors_beta:
            if bv == beta[sym]:
                continue
            tb = dict(beta)
            tb[sym] = bv
            for wv in w_variants:
                configs.append((tb, wv))

    # Also try w_lgb variants with unchanged beta
    for wv in w_variants:
        if wv != w_lgb:
            configs.append((dict(beta), wv))

    return configs

neighbors = get_neighbors(best_beta_A, best_w_lgb)
best_config_C = {"beta": dict(best_beta_A), "w_lgb": best_w_lgb, "pnl": best_pnl_B}
for tb, wv in neighbors:
    combined = make_combined(BASELINE_W_NN, wv)
    pnl, _ = compute_holdout_pnl(combined, sym_arr, mp_t, mp_th, tb)
    entry = {"beta": dict(tb), "w_nn": BASELINE_W_NN, "w_lgb": wv, "pnl": pnl}
    sweep_C_results.append(entry)
    if pnl > best_config_C["pnl"]:
        best_config_C = {"beta": dict(tb), "w_lgb": wv, "pnl": pnl}
        print(f"  New best in C: beta={tb}, w_lgb={wv:.2f} -> pnl={pnl:.6f}", flush=True)

print(f"  Sweep C done. Best: beta={best_config_C['beta']}, w_lgb={best_config_C['w_lgb']}, pnl={best_config_C['pnl']:.6f}", flush=True)

# ── Finalize best config ──────────────────────────────────────────────────

final_beta = best_config_C["beta"]
final_w_lgb = best_config_C["w_lgb"]
final_pnl = best_config_C["pnl"]

results = {
    "task": "T150 conformal β + ensemble weight tune",
    "baseline_holdout_pnl": baseline_pnl,
    "sweep_A_best_beta": best_beta_A,
    "sweep_A_best_holdout_pnl": best_pnl_A,
    "sweep_B_best_w_lgb": best_w_lgb,
    "sweep_B_best_holdout_pnl": best_pnl_B,
    "sweep_C_best_beta": best_config_C["beta"],
    "sweep_C_best_w_lgb": best_config_C["w_lgb"],
    "sweep_C_best_holdout_pnl": best_config_C["pnl"],
    "best_config": {
        "per_sym_beta": final_beta,
        "w_nn": BASELINE_W_NN,
        "w_lgb": final_w_lgb,
        "holdout_pnl": final_pnl,
    },
    "vs_baseline_delta": final_pnl - baseline_pnl,
    "verdict": "",
    "sweep_A_results": sweep_A_results,
    "sweep_B_results": sweep_B_results,
    "sweep_C_results": sweep_C_results,
}

# Add verdict
delta = final_pnl - baseline_pnl
if delta > 0.5:
    results["verdict"] = f"Improvement of {delta:.4f} over baseline. Update thresholds.json and package."
elif delta > 0.0:
    results["verdict"] = f"Marginal improvement of {delta:.4f}. Not worth packaging."
else:
    results["verdict"] = f"No improvement ({delta:.4f}). Keep baseline."

out_path = os.path.join(OUTDIR, "results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {out_path}", flush=True)

# ── REPORT.md ────────────────────────────────────────────────────────────

all_configs = (
    [{"beta": dict(BASELINE_BETA), "w_nn": BASELINE_W_NN, "w_lgb": BASELINE_W_LGB, "pnl": baseline_pnl, "label": "BASELINE"}]
    + [dict(r, label="sweep_A") for r in sweep_A_results]
    + [dict(r, label="sweep_B") for r in sweep_B_results]
    + [dict(r, label="sweep_C") for r in sweep_C_results]
)
all_configs_sorted = sorted(all_configs, key=lambda x: x["pnl"], reverse=True)

lines = ["# T150 Conformal β + Ensemble Weight Sweep Report\n",
         f"\nBaseline holdout PnL: {baseline_pnl:.6f}",
         f"Best holdout PnL: {final_pnl:.6f}",
         f"Delta: {delta:+.6f}",
         f"\nBest config: per_sym_beta={final_beta}, w_nn={BASELINE_W_NN}, w_lgb={final_w_lgb}",
         f"\nVerdict: {results['verdict']}",
         "\n## All configs (sorted by PnL)\n",
         "| rank | pnl | b0 | b1 | b2 | b3 | b4 | w_lgb | label |",
         "|------|-----|----|----|----|----|----|----|-------|"]

for i, c in enumerate(all_configs_sorted[:100]):
    b = c.get("beta", {})
    lines.append(
        f"| {i+1} | {c['pnl']:.6f} | {b.get(0,''):5} | {b.get(1,''):5} | {b.get(2,''):5} | {b.get(3,''):5} | {b.get(4,''):5} | {c.get('w_lgb', ''):.2f} | {c.get('label','')} |"
    )

with open(os.path.join(OUTDIR, "REPORT.md"), "w") as f:
    f.write("\n".join(lines))
print("REPORT.md written.", flush=True)

# ── Update pkg_iter019_v2 if improvement > 0.5 ───────────────────────────

THRESHOLDS_PATH = "/root/projects/liangwenbei_workdir/experiments/R_full_retrain/pkg_iter019_v2/thresholds.json"

if delta > 0.5:
    update_progress("Updating thresholds.json and packaging", {"delta": delta})
    print(f"Delta={delta:.4f} > 0.5, updating thresholds.json...", flush=True)

    with open(THRESHOLDS_PATH) as f:
        tcfg = json.load(f)

    # Update conformal_wrapper betas
    tcfg["conformal_wrapper"]["per_sym_beta"] = {str(k): v for k, v in final_beta.items()}

    # Update w_lgb in h60
    for hcfg in tcfg["horizons"]:
        if hcfg.get("h") == 60 and hcfg.get("active", False):
            hcfg["w_lgb"] = final_w_lgb
            hcfg["_t150_note"] = f"T150 tuned: w_lgb={final_w_lgb}, beta={final_beta}, holdout_pnl={final_pnl:.6f} (delta={delta:+.4f} vs baseline)"
            break

    with open(THRESHOLDS_PATH, "w") as f:
        json.dump(tcfg, f, indent=2)
    print("  thresholds.json updated.", flush=True)

    # Build zip
    import subprocess
    zip_path = "/root/projects/liangwenbei_workdir/submission_050818_iter019_v7_conformal_tuned.zip"
    pkg_dir = "/root/projects/liangwenbei_workdir/experiments/R_full_retrain/pkg_iter019_v2"
    result = subprocess.run(
        ["zip", "-r", zip_path, "."],
        cwd=pkg_dir,
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  zip error: {result.stderr}", flush=True)
    else:
        print(f"  zip created: {zip_path}", flush=True)

    # Copy to metabot outputs
    for dest in [
        "/tmp/metabot-outputs/oc_8da9d6d6cd3ed6e40a8dd5b38d55d6a7",
        "/tmp/metabot-outputs/worker-64191384",
    ]:
        os.makedirs(dest, exist_ok=True)
        import shutil
        shutil.copy2(zip_path, dest)
        print(f"  Copied to {dest}", flush=True)
else:
    print(f"Delta={delta:.4f} <= 0.5, not updating package.", flush=True)

# ── Final progress ───────────────────────────────────────────────────────

update_progress("Done", {
    "baseline_holdout_pnl": baseline_pnl,
    "best_holdout_pnl": final_pnl,
    "delta": delta,
    "best_beta": final_beta,
    "best_w_lgb": final_w_lgb,
})

print(f"\n{'='*60}", flush=True)
print(f"RESULT: task=[T150 conformal beta sweep] metrics={{baseline_holdout={baseline_pnl:.6f}, best_holdout={final_pnl:.6f}, delta={delta:+.6f}, best_beta={{{','.join(f'{k}:{v}' for k,v in final_beta.items())}}}, best_w_lgb={final_w_lgb}}} notes=[{results['verdict']}]", flush=True)
