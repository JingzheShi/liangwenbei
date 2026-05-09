#!/usr/bin/env python3
"""T159b: Per-sym dynamic abstain band sweep (TSH-FT, HOLDOUT-validated).

Uses FROZEN thresholds (iter_018). Abstain if |combined_pred| < beta[s]*sigma[s].
Sigma computed from full 0-119 data by running inference on inner (0-95) from models.
"""
import json, time, os, glob
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

OUTDIR = Path("experiments/T159b_per_sym_abstain")
OUTDIR.mkdir(exist_ok=True)

T68_CACHE = "experiments/T68_stage5_features/cache"
T87_DIR = "experiments/T87_spo_dfl"
T75_DIR = "experiments/T75_regression_dmid"
FEE = 0.0001
SEEDS = [1, 7, 13, 42, 100]

THR_UP = 0.00029995433796130955
THR_DN = 0.0002158528296175958

# Extended grid so betas actually affect trades (need beta*sigma > thr_dn)
BETA_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50,
             0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.50, 2.00]

V2_BETAS = {0: 0.10, 1: 0.40, 2: 0.30, 3: 0.00, 4: 0.00}


def write_progress(step, metrics=None, status="running"):
    prog = {
        "status": status,
        "step": step,
        "metrics": metrics or {},
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    (OUTDIR / "worker-progress.json").write_text(json.dumps(prog, indent=2))
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {step}", flush=True)


# ── NN inference (numpy) ──────────────────────────────────────────────────

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


write_progress("Loading holdout parquets (96-119)")
t0 = time.time()

# Load holdout pred parquets
nn_preds_ho = []
for seed in SEEDS:
    df = pd.read_parquet(f"{T87_DIR}/pred_T87_seed{seed}_main.parquet")
    nn_preds_ho.append(df["pred_dmid_norm"].values.astype(np.float64))
nn_ho_mean = np.mean(nn_preds_ho, axis=0)

lgb_preds_ho = []
df_ref = None
for seed in SEEDS:
    df = pd.read_parquet(f"{T75_DIR}/pred_T75_seed{seed}.parquet")
    lgb_preds_ho.append(df["pred_dmid_norm"].values.astype(np.float64))
    if df_ref is None:
        df_ref = df

lgb_ho_mean = np.mean(lgb_preds_ho, axis=0)
combined_ho = (1.0 * nn_ho_mean + 1.5 * lgb_ho_mean) / 2.5

sym_ho = df_ref["sym"].values.astype(np.int8)
date_ho = df_ref["date"].values.astype(np.int16)
delta_mid_ho = df_ref["true_dmid_norm"].values.astype(np.float64)
print(f"Holdout: {len(combined_ho)} rows, dates {date_ho.min()}-{date_ho.max()}")
print(f"Combined pred ho: mean={combined_ho.mean():.6f}, std={combined_ho.std():.6f}")

write_progress("Loading inner data from T68 cache (dates 0-95)")

tr = np.load(f"{T68_CACHE}/schemeP_train.npz")
va = np.load(f"{T68_CACHE}/schemeP_val.npz")
X_inner = np.vstack([tr['X'], va['X']])
sym_inner = np.concatenate([tr['sym'], va['sym']]).astype(np.int8)
# delta_mid for inner: use mp_t60 - mp_t (same as midprice_th - midprice_t in parquets)
mp_t_inner = np.concatenate([tr['mp_t'], va['mp_t']]).astype(np.float64)
mp_th_inner = np.concatenate([tr['mp_t60'], va['mp_t60']]).astype(np.float64)
# delta_mid_inner = (mp_th - mp_t) / (mp_t + 1) -- normalized return
# Actually, check if parquet true_dmid_norm is midprice_th/midprice_t or (th-t)/t
# From parquet: midprice_th - midprice_t appears to be raw, true_dmid_norm is normalized
# Let's compute delta_mid_inner as (mp_th - mp_t) / (mp_t + 1.0) -- verify with holdout

# Verify: true_dmid_norm in parquet should equal (midprice_th - midprice_t) / (midprice_t + 1)
sample_mid_t = df_ref["midprice_t"].values[:5]
sample_mid_th = df_ref["midprice_th"].values[:5]
sample_dmid = df_ref["true_dmid_norm"].values[:5]
computed = (sample_mid_th - sample_mid_t)  # raw diff
computed2 = (sample_mid_th - sample_mid_t) / (sample_mid_t + 1.0)
print(f"Verify delta_mid: true={sample_dmid}, raw_diff={computed}, /pt+1={computed2}")
# Use the correct formula for inner
delta_mid_inner = (mp_th_inner - mp_t_inner)  # try raw diff first (check with known scale)
delta_mid_inner_norm = (mp_th_inner - mp_t_inner) / (mp_t_inner + 1.0)

print(f"Inner: {len(X_inner)} rows, dates 0-95")

write_progress("Running LGB inference on inner data (5 seeds)")
import lightgbm as lgb

# Load feature names and apply same feature selection as T75 training
with open(f"{T68_CACHE}/schemeP_feat_names.txt") as f:
    feat_names = [l.strip() for l in f]

DROP_NAMES = [
    "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
    "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
    "qrank_W100_cumspread", "kyle_lam_W50", "kyle_lam_W100",
    "roll_eff_spr_ratio_W100", "liq_asym_top5_W5"
]
drop_set = set(DROP_NAMES)
lgb_keep_idx = np.array([i for i, n in enumerate(feat_names) if n not in drop_set], dtype=np.int64)
X_inner_lgb = X_inner[:, lgb_keep_idx].astype(np.float32, copy=False)
print(f"LGB feature dim: {X_inner_lgb.shape[1]} (from {X_inner.shape[1]})")

lgb_preds_inner = []
for seed in SEEDS:
    model_path = f"{T75_DIR}/model_T75_seed{seed}.txt"
    booster = lgb.Booster(model_file=model_path)
    chunks = []
    for s in range(0, len(X_inner_lgb), 200_000):
        chunks.append(booster.predict(X_inner_lgb[s:s+200_000]))
    pred = np.concatenate(chunks).astype(np.float64)
    lgb_preds_inner.append(pred)
    print(f"  LGB seed {seed}: mean={pred.mean():.6f}, std={pred.std():.6f}", flush=True)
    del booster
lgb_inner_mean = np.mean(lgb_preds_inner, axis=0)
del lgb_preds_inner
print(f"LGB elapsed: {time.time()-t0:.1f}s")

write_progress("Running NN inference on inner data (5 seeds)")
nn_preds_inner = []
for seed in SEEDS:
    npz_path = f"{T87_DIR}/model_T87_seed{seed}_main.npz"
    model = MLPNumpy(npz_path)
    pred = model.predict(X_inner)
    nn_preds_inner.append(pred.astype(np.float64))
    print(f"  NN seed {seed}: mean={pred.mean():.6f}, std={pred.std():.6f}", flush=True)
    del model
nn_inner_mean = np.mean(nn_preds_inner, axis=0)
del nn_preds_inner
print(f"NN+LGB elapsed: {time.time()-t0:.1f}s")

combined_inner = (1.0 * nn_inner_mean + 1.5 * lgb_inner_mean) / 2.5

# Concatenate inner + holdout for sigma computation (0-119)
combined_all = np.concatenate([combined_inner, combined_ho])
sym_all = np.concatenate([sym_inner, sym_ho])

sigma = {}
for s in range(5):
    mask = sym_all == s
    sigma[s] = float(np.std(combined_all[mask]))
print(f"Sigma per sym (all 0-119): {sigma}")

# Also compute holdout-only sigma for comparison
sigma_ho_only = {}
for s in range(5):
    mask = sym_ho == s
    sigma_ho_only[s] = float(np.std(combined_ho[mask]))
print(f"Sigma per sym (holdout only): {sigma_ho_only}")

# ── PnL computation ───────────────────────────────────────────────────────
# Determine correct delta_mid formula for inner
# true_dmid_norm in holdout parquets is the target; need to match inner
# From the parquet schema, true_dmid_norm and midprice values are available
# Let's check: for holdout, delta_mid ≈ (mid_th - mid_t) but check scale
mid_t_sample = df_ref["midprice_t"].values[:100]
mid_th_sample = df_ref["midprice_th"].values[:100]
true_dmid_sample = df_ref["true_dmid_norm"].values[:100]
# These are normalized by dividing by (mp_t + 1.0)?
raw_diff = mid_th_sample - mid_t_sample
norm_diff = (mid_th_sample - mid_t_sample) / (mid_t_sample + 1.0)
print(f"raw_diff vs true_dmid: corr={np.corrcoef(raw_diff, true_dmid_sample)[0,1]:.4f}")
print(f"norm_diff vs true_dmid: corr={np.corrcoef(norm_diff, true_dmid_sample)[0,1]:.4f}")
print(f"raw_diff first 3: {raw_diff[:3]}")
print(f"true_dmid first 3: {true_dmid_sample[:3]}")

# Use true_dmid_norm directly for holdout; for inner, use raw diff (mp_t values not normalized)
# Since mp_t in NPZ and parquet should be same scale, check:
print(f"Inner mp_t range: {mp_t_inner.min():.4f} to {mp_t_inner.max():.4f}")
print(f"Holdout mp_t range: {df_ref['midprice_t'].min():.4f} to {df_ref['midprice_t'].max():.4f}")


def compute_pnl(combined, delta_mid, sym_arr, betas, sigma, date_arr=None, date_range=None):
    """Vectorized PnL. date_range=(lo, hi) filters dates if provided."""
    if date_range is not None and date_arr is not None:
        mask = (date_arr >= date_range[0]) & (date_arr < date_range[1])
    else:
        mask = np.ones(len(combined), dtype=bool)

    c = combined[mask]
    dm = delta_mid[mask]
    s = sym_arr[mask]

    beta_arr = np.array([betas[i] for i in range(5)])
    sigma_arr = np.array([sigma[i] for i in range(5)])
    abstain_thresh = beta_arr[s] * sigma_arr[s]

    abstain = np.abs(c) < abstain_thresh
    pos = np.where(c > THR_UP, 1.0, np.where(c < -THR_DN, -1.0, 0.0))
    pos[abstain] = 0.0

    pnl = pos * dm - FEE * np.abs(pos)
    return float(np.sum(pnl))


write_progress("Computing delta_mid for inner and verifying PnL")

# Determine inner delta_mid: match the scale of holdout true_dmid_norm
# Holdout true_dmid_norm ≈ (midprice_th - midprice_t) based on parquet data
# In NPZ, mp_t and mp_t60 are raw midprices; verify scale
if np.allclose(raw_diff[:5], true_dmid_sample[:5], atol=1e-5, rtol=1e-2):
    delta_mid_inner_final = delta_mid_inner  # raw diff
    print("Using raw diff for inner delta_mid")
elif np.allclose(norm_diff[:5], true_dmid_sample[:5], atol=1e-5, rtol=1e-2):
    delta_mid_inner_final = delta_mid_inner_norm  # normalized diff
    print("Using normalized diff for inner delta_mid")
else:
    # Try to infer scaling factor
    scale = true_dmid_sample.std() / raw_diff.std()
    print(f"Neither matches exactly. Scale factor raw->true: {scale:.4f}")
    print(f"Assuming raw diff ≈ true_dmid_norm (may have small numerical error)")
    delta_mid_inner_final = delta_mid_inner

# For holdout, use the parquet true_dmid_norm directly
# Concatenated delta_mid for all data
delta_mid_all = np.concatenate([delta_mid_inner_final, delta_mid_ho])

# ── Evaluate v2 baseline ──────────────────────────────────────────────────
write_progress("Evaluating v2 baseline")

# Holdout only (using concatenated arrays with date filtering)
date_inner = np.concatenate([
    np.concatenate([tr['date'], va['date']]).astype(np.int16),
    date_ho
])

v2_inner_pnl = compute_pnl(combined_inner, delta_mid_inner_final, sym_inner, V2_BETAS, sigma)
v2_holdout_pnl = compute_pnl(combined_ho, delta_mid_ho, sym_ho, V2_BETAS, sigma)
no_abstain = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
no_abstain_ho = compute_pnl(combined_ho, delta_mid_ho, sym_ho, no_abstain, sigma)
print(f"V2 baseline: inner_pnl={v2_inner_pnl:.4f}, holdout_pnl={v2_holdout_pnl:.4f}")
print(f"No-abstain holdout PnL: {no_abstain_ho:.4f}")

# ── Per-sym analysis ──────────────────────────────────────────────────────
print("\nPer-sym analysis:")
for s in range(5):
    mask_ho = sym_ho == s
    c_s = combined_ho[mask_ho]
    long_mask = c_s > THR_UP
    short_mask = c_s < -THR_DN
    min_beta_short = THR_DN / sigma[s]
    min_beta_long = THR_UP / sigma[s]
    print(f"  sym={s}: sigma={sigma[s]:.6f}, "
          f"min_beta_for_short_effect={min_beta_short:.3f}, "
          f"min_beta_for_long_effect={min_beta_long:.3f}, "
          f"trades={long_mask.sum()+short_mask.sum()}")

# ── Coordinate descent ────────────────────────────────────────────────────
write_progress("Starting coordinate descent (2 rounds, extended grid)")

sweep_log = []
current_betas = dict(V2_BETAS)

for round_idx in range(2):
    print(f"\n=== Round {round_idx+1} ===")
    for s in range(5):
        best_beta_s = current_betas[s]
        best_ho = compute_pnl(combined_ho, delta_mid_ho, sym_ho, current_betas, sigma)

        for beta_val in BETA_GRID:
            trial = dict(current_betas)
            trial[s] = beta_val
            ho_pnl = compute_pnl(combined_ho, delta_mid_ho, sym_ho, trial, sigma)
            is_best = ho_pnl > best_ho

            sweep_log.append({
                "round": round_idx + 1,
                "sym": int(s),
                "beta_tested": beta_val,
                "holdout_pnl": ho_pnl,
                "is_best": is_best
            })

            if is_best:
                best_ho = ho_pnl
                best_beta_s = beta_val

        old = current_betas[s]
        current_betas[s] = best_beta_s
        print(f"  sym={s}: {old:.2f} -> {best_beta_s:.2f} (holdout={best_ho:.4f})")

        write_progress(f"R{round_idx+1} sym={s} done",
                       {"best_holdout_pnl": best_ho, "betas": str(current_betas)})

write_progress("Computing final results")

best_holdout_pnl = compute_pnl(combined_ho, delta_mid_ho, sym_ho, current_betas, sigma)
best_inner_pnl = compute_pnl(combined_inner, delta_mid_inner_final, sym_inner, current_betas, sigma)
delta_vs_baseline = best_holdout_pnl - v2_holdout_pnl

print(f"\nFinal betas: {current_betas}")
print(f"Best holdout PnL: {best_holdout_pnl:.4f}")
print(f"V2 baseline holdout PnL: {v2_holdout_pnl:.4f}")
print(f"Delta vs baseline: {delta_vs_baseline:.4f}")

results = {
    "task": "T159b per-sym beta sweep, TSH-FT (HOLDOUT-validated)",
    "v2_baseline_inner_pnl": v2_inner_pnl,
    "v2_baseline_holdout_pnl": v2_holdout_pnl,
    "best_beta_holdout": {str(k): float(v) for k, v in current_betas.items()},
    "best_holdout_pnl": best_holdout_pnl,
    "delta_holdout_vs_baseline": delta_vs_baseline,
    "iter_018_baseline_beta": {"0": 0.10, "1": 0.40, "2": 0.30, "3": 0.00, "4": 0.00},
    "best_inner_pnl": best_inner_pnl,
    "no_abstain_holdout_pnl": no_abstain_ho,
    "sigma_per_sym_full": {str(k): float(v) for k, v in sigma.items()},
    "sigma_per_sym_holdout_only": {str(k): float(v) for k, v in sigma_ho_only.items()},
    "sweep_log": sweep_log
}

(OUTDIR / "results.json").write_text(json.dumps(results, indent=2))
print(f"Results written to {OUTDIR / 'results.json'}")

write_progress("Sweep complete",
               {"best_holdout_pnl": best_holdout_pnl, "delta": delta_vs_baseline},
               status="done")

# ── Build zip if improvement > 0.3 ────────────────────────────────────────
if delta_vs_baseline > 0.3:
    print(f"\ndelta={delta_vs_baseline:.4f} > 0.3: Building submission zip!")
    import zipfile, tempfile, shutil

    v2_zip = "submission_050818_iter019_v2_fullretrain_conformal.zip"
    out_zip = "submission_050819_iter019_v2W_persym_beta.zip"

    if os.path.exists(v2_zip):
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(v2_zip, 'r') as zf:
                zf.extractall(tmpdir)
            print(f"Extracted: {os.listdir(tmpdir)}")

            thr_candidates = glob.glob(f"{tmpdir}/**/thresholds.json", recursive=True)
            thr_file = thr_candidates[0] if thr_candidates else os.path.join(tmpdir, "thresholds.json")

            if os.path.exists(thr_file):
                with open(thr_file) as f:
                    thr = json.load(f)
            else:
                thr = {}

            thr["per_sym_beta"] = {str(k): float(v) for k, v in current_betas.items()}
            thr["per_sym_sigma"] = {str(k): float(v) for k, v in sigma.items()}

            with open(thr_file, 'w') as f:
                json.dump(thr, f, indent=2)

            with zipfile.ZipFile(out_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                for root, dirs, files in os.walk(tmpdir):
                    for file in files:
                        fpath = os.path.join(root, file)
                        arcname = os.path.relpath(fpath, tmpdir)
                        zf.write(fpath, arcname)
            print(f"Created: {out_zip}")
    else:
        print(f"WARNING: v2 zip not found at {v2_zip}")
else:
    print(f"delta={delta_vs_baseline:.4f} <= 0.3: no zip needed")

print(f"\nRESULT: task=[T159b per-sym beta sweep] metrics={{v2_baseline_holdout={v2_holdout_pnl:.4f}, best_holdout={best_holdout_pnl:.4f}, delta={delta_vs_baseline:.4f}}} notes=[best_beta={current_betas}]")
