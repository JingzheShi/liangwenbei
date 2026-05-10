"""T180: OOD density-based abstain wrapper experiment.

Design:
1. Fit PCA(20) + GMM(k=5) on schemeP train+val features (dates 0-95)
2. Compute log p(x) threshold at percentiles 10, 20, 30 of training log-likelihoods
3. On test (dates 96-119): compute T170 predictions, then apply OOD abstain
4. Compare PnL: T170 baseline vs V1/V2/V3 abstain variants
5. Build zip for variants that improve over T170 baseline

Constraints:
- GMM fitted on dates 0-95 only (no leakage from test)
- Predictor stateless (GMM loaded at __init__, not mutated across calls)
- date feature never used; sym not fed to model forward
"""
from __future__ import annotations

import os
import sys
import json
import pickle
import shutil
import zipfile
import numpy as np
import lightgbm as lgb

# Add workdir src to path for evaluator
WORKDIR = "/root/projects/liangwenbei_workdir"
sys.path.insert(0, os.path.join(WORKDIR, "src"))
from eval.pnl import compute_pnl, sanitize_for_json

CACHE_DIR = os.path.join(WORKDIR, "experiments/T68_stage5_features/cache")
T170_PKG = os.path.join(WORKDIR, "experiments/T170_T87M7_clean_pkg")
T180_DIR = os.path.join(WORKDIR, "experiments/T180_OOD_abstain")
PROGRESS_FILE = os.path.join(T180_DIR, "worker-progress.json")

import datetime

def update_progress(status, step, metrics=None):
    prog = {
        "status": status,
        "step": step,
        "metrics": metrics or {},
        "timestamp": datetime.datetime.utcnow().isoformat()
    }
    with open(PROGRESS_FILE, "w") as f:
        json.dump(prog, f, indent=2)
    print(f"[T180] {step}", flush=True)


def load_schemeP_data():
    update_progress("running", "Loading schemeP cache data")
    train = np.load(os.path.join(CACHE_DIR, "schemeP_train.npz"))
    val   = np.load(os.path.join(CACHE_DIR, "schemeP_val.npz"))
    test  = np.load(os.path.join(CACHE_DIR, "schemeP_test.npz"))
    print(f"  Train: {train['X'].shape}, dates {train['date'].min()}-{train['date'].max()}")
    print(f"  Val:   {val['X'].shape},   dates {val['date'].min()}-{val['date'].max()}")
    print(f"  Test:  {test['X'].shape},  dates {test['date'].min()}-{test['date'].max()}")
    return train, val, test


def get_fail_indices():
    FAIL_NAMES = {
        "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
        "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
        "qrank_W100_cumspread",
        "kyle_lam_W50", "kyle_lam_W100",
        "roll_eff_spr_ratio_W100",
        "liq_asym_top5_W5",
    }
    with open(os.path.join(CACHE_DIR, "schemeP_feat_names.txt")) as f:
        feat_names = [l.strip() for l in f]
    keep_idx = np.array([i for i, n in enumerate(feat_names) if n not in FAIL_NAMES], dtype=np.int64)
    assert len(keep_idx) == 359, f"Expected 359 keep features, got {len(keep_idx)}"
    return keep_idx


def compute_t170_predictions(X_359, lgb_boosters, nn_models, w_nn=1.0, w_lgb=1.5):
    """Compute T170 ensemble predictions on N×359 feature matrix."""
    # LGB ensemble
    lgb_preds = np.zeros(X_359.shape[0], dtype=np.float32)
    for b in lgb_boosters:
        lgb_preds += b.predict(X_359).astype(np.float32)
    lgb_preds /= len(lgb_boosters)

    # NN ensemble — NN's keep_idx selects from 370-feature space
    # but feats is already 359 so shape[1]==in_dim, skip keep_idx
    nn_preds = np.zeros(X_359.shape[0], dtype=np.float32)
    for nn in nn_models:
        nn_preds += nn.predict(X_359)
    nn_preds /= len(nn_models)

    pred = (w_nn * nn_preds + w_lgb * lgb_preds) / (w_nn + w_lgb)
    return pred.astype(np.float32)


def apply_threshold(pred_dmid, thr_up, thr_dn, band=None):
    """Apply asymmetric EV gate. Returns array of 0/1/2."""
    N = len(pred_dmid)
    out = np.ones(N, dtype=np.int64)
    if band is not None:
        out[pred_dmid > (thr_up + band)] = 2
        out[pred_dmid < -(thr_dn + band)] = 0
    else:
        out[pred_dmid > thr_up] = 2
        out[pred_dmid < -thr_dn] = 0
    return out


# NN numpy inference (copied from T170 Predictor)
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

    def predict(self, X_in):
        Xs = X_in
        if Xs.shape[1] != self.in_dim:
            Xs = Xs[:, self.keep_idx]
        Xs = Xs.astype(np.float32, copy=True)
        Xs = (Xs - self.feat_mean) / self.feat_std
        Xs = np.where(np.isnan(Xs), 0.0, Xs)
        Xs = np.clip(Xs, -self.clip, self.clip).astype(np.float32)
        h = Xs
        for i in range(len(self.hidden)):
            h = h @ self.W[i].T + self.b[i]
            if self.use_layernorm:
                h = _layernorm(h, self.LN_W[i], self.LN_b[i])
            h = _gelu_tanh(h)
        out = h @ self.WF.T + self.bF
        return (out.squeeze(-1) / self.target_scale).astype(np.float32)


def fit_ood_detector(X_train_val_359, percentiles=(10, 20, 30), n_components=20, n_gmm=5, subsample=300000, seed=42):
    """Fit PCA + GMM on training features. Returns detector dict."""
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture

    update_progress("running", f"Fitting OOD detector: PCA({n_components}) + GMM(k={n_gmm})")
    N = X_train_val_359.shape[0]
    print(f"  Input: {X_train_val_359.shape}, subsample to {min(N, subsample)} for PCA fit")

    # Impute NaNs with 0 (same as LGB/NN)
    X_imp = np.where(np.isnan(X_train_val_359), 0.0, X_train_val_359).astype(np.float32)

    rng = np.random.default_rng(seed)
    sub_idx = rng.choice(N, size=min(N, subsample), replace=False)
    X_sub = X_imp[sub_idx]

    print(f"  Fitting PCA on {X_sub.shape}...")
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(X_sub)
    explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA explains {explained:.3f} variance")

    # Transform FULL training set for GMM fitting (use subsample if too large)
    sub_gmm_idx = rng.choice(N, size=min(N, subsample), replace=False)
    X_gmm = pca.transform(X_imp[sub_gmm_idx].astype(np.float64))
    print(f"  Fitting GMM(k={n_gmm}) on {X_gmm.shape}...")
    gmm = GaussianMixture(
        n_components=n_gmm,
        covariance_type="full",
        random_state=seed,
        max_iter=200,
        n_init=3,
        verbose=0,
    )
    gmm.fit(X_gmm)
    print(f"  GMM converged: {gmm.converged_}")

    # Compute log-likelihoods on FULL train+val (batched to save memory)
    update_progress("running", "Computing training log-likelihoods for threshold")
    batch_size = 50000
    log_probs_train = []
    for start in range(0, N, batch_size):
        X_b = X_imp[start:start+batch_size].astype(np.float64)
        X_b_pca = pca.transform(X_b)
        lp = gmm.score_samples(X_b_pca)
        log_probs_train.append(lp)
    log_probs_train = np.concatenate(log_probs_train)
    print(f"  Train log-prob: mean={log_probs_train.mean():.3f}, std={log_probs_train.std():.3f}")
    print(f"  Train log-prob: min={log_probs_train.min():.3f}, max={log_probs_train.max():.3f}")

    thresholds = {}
    for p in percentiles:
        thr = float(np.percentile(log_probs_train, p))
        thresholds[f"perc{p}"] = thr
        n_ood = int((log_probs_train < thr).sum())
        print(f"  Threshold p{p}: {thr:.4f} → {n_ood}/{N} ({100*n_ood/N:.1f}%) train rows are OOD")

    return {
        "pca": pca,
        "gmm": gmm,
        "thresholds": thresholds,
        "train_log_prob_stats": {
            "mean": float(log_probs_train.mean()),
            "std": float(log_probs_train.std()),
            "min": float(log_probs_train.min()),
            "max": float(log_probs_train.max()),
        }
    }, log_probs_train


def build_submission_zip(variant_name, threshold_value, detector_dict, output_dir):
    """Build a submission zip with OOD abstain wrapper."""
    pkg_dir = os.path.join(T180_DIR, f"pkg_{variant_name}")
    if os.path.exists(pkg_dir):
        shutil.rmtree(pkg_dir)
    shutil.copytree(T170_PKG, pkg_dir)

    # Save GMM detector for this variant
    detector_v = {
        "pca": detector_dict["pca"],
        "gmm": detector_dict["gmm"],
        "threshold": threshold_value,
    }
    pkl_path = os.path.join(pkg_dir, "gmm_ood_detector.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(detector_v, f, protocol=4)

    # Patch Predictor.py with OOD abstain
    predictor_path = os.path.join(pkg_dir, "Predictor.py")
    with open(predictor_path) as f:
        code = f.read()

    # Insert import at top (after existing imports)
    insert_after = "import lightgbm as lgb"
    ood_import = "\nimport pickle"
    code = code.replace(insert_after, insert_after + ood_import, 1)

    # Insert GMM load in __init__ after loading thresholds
    init_insert_target = "self._col_idx: Dict[str, int] = dict(self._raw_col_to_idx)"
    gmm_init_code = """
        # OOD density-based abstain
        gmm_path = os.path.join(here, "gmm_ood_detector.pkl")
        if os.path.isfile(gmm_path):
            with open(gmm_path, "rb") as _f:
                _ood = pickle.load(_f)
            self._ood_pca = _ood["pca"]
            self._ood_gmm = _ood["gmm"]
            self._ood_threshold = float(_ood["threshold"])
            self._ood_enabled = True
        else:
            self._ood_enabled = False

        """
    code = code.replace(init_insert_target, gmm_init_code + init_insert_target, 1)

    # Insert OOD abstain in predict() after computing feats, before EV gate
    # Find the location: after "feats = self._compute_batch_features(batches)"
    predict_insert_target = "        if self._cw_enabled:"
    ood_predict_code = """        # OOD density-based abstain: pre-compute mask
        _ood_mask = np.zeros(B, dtype=bool)
        if self._ood_enabled:
            _X_imp = np.where(np.isnan(feats), 0.0, feats).astype(np.float64)
            _X_pca = self._ood_pca.transform(_X_imp)
            _log_probs = self._ood_gmm.score_samples(_X_pca)
            _ood_mask = _log_probs < self._ood_threshold

        """
    code = code.replace(predict_insert_target, ood_predict_code + predict_insert_target, 1)

    # After computing actions for each horizon, apply OOD abstain
    actions_insert_target = "            out[:, HORIZON_TO_IDX[H]] = actions"
    ood_actions_code = """            # Apply OOD abstain: force flat (1) for OOD rows
            actions[_ood_mask] = 1
            out[:, HORIZON_TO_IDX[H]] = actions"""
    code = code.replace(actions_insert_target, ood_actions_code, 1)

    with open(predictor_path, "w") as f:
        f.write(code)

    # Build zip
    zip_name = f"submission_050910_iter019_v2N_T87M7_OOD_{variant_name}.zip"
    zip_path = os.path.join(output_dir, zip_name)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(pkg_dir):
            fpath = os.path.join(pkg_dir, fname)
            if os.path.isfile(fpath) and not fname.startswith("__"):
                zf.write(fpath, fname)
    print(f"  Built zip: {zip_name} ({os.path.getsize(zip_path)//1024}KB)")
    return zip_path, pkg_dir


def copy_to_outputs(zip_path):
    """Copy zip to /tmp/metabot-outputs."""
    out_dir = "/tmp/metabot-outputs/worker-d98c6853"
    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, os.path.basename(zip_path))
    shutil.copy2(zip_path, dest)
    print(f"  Copied to outputs: {os.path.basename(dest)}")


def main():
    os.makedirs(T180_DIR, exist_ok=True)
    update_progress("running", "Starting T180 OOD abstain experiment")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    train, val, test = load_schemeP_data()
    keep_idx = get_fail_indices()

    X_train_359 = train["X"][:, keep_idx]
    X_val_359   = val["X"][:, keep_idx]
    X_test_359  = test["X"][:, keep_idx]
    X_trval_359 = np.concatenate([X_train_359, X_val_359], axis=0)

    # For OOD, also use FULL 370-feature X for the PCA (captures more signal)
    X_trval_full = np.concatenate([train["X"], val["X"]], axis=0)
    X_test_full  = test["X"]

    # Test labels and midprices for PnL evaluation
    mp_t  = test["mp_t"].astype(np.float64)
    mp_t5  = test["mp_t5"].astype(np.float64)
    mp_t10 = test["mp_t10"].astype(np.float64)
    mp_t20 = test["mp_t20"].astype(np.float64)
    mp_t40 = test["mp_t40"].astype(np.float64)
    mp_t60 = test["mp_t60"].astype(np.float64)
    y60   = test["y60"].astype(np.int64)

    N_test = X_test_359.shape[0]
    print(f"Test set size: {N_test}")

    # ── 2. Load T170 models ───────────────────────────────────────────────────
    update_progress("running", "Loading T170 LGB and NN models")
    seeds = [1, 7, 13, 42, 100]
    lgb_boosters = [lgb.Booster(model_file=os.path.join(T170_PKG, f"model_h60_seed{s}.txt")) for s in seeds]
    nn_models    = [MLPNumpy(os.path.join(T170_PKG, f"nn_h60_seed{s}.npz")) for s in seeds]
    print(f"  Loaded {len(lgb_boosters)} LGB + {len(nn_models)} NN models")

    # ── 3. Compute T170 baseline predictions on test ──────────────────────────
    update_progress("running", "Computing T170 baseline predictions on test (dates 96-119)")
    pred_dmid = compute_t170_predictions(X_test_359, lgb_boosters, nn_models)

    # T170 uses conformal wrapper with per-sym beta, but for local eval simplify:
    # Use the exact thresholds from T170 config
    import json as json_mod
    with open(os.path.join(T170_PKG, "thresholds.json")) as f:
        tcfg = json_mod.load(f)
    h60_cfg = next(h for h in tcfg["horizons"] if h["h"] == 60 and h.get("active", False))
    thr_up = float(h60_cfg["thr_up"])
    thr_dn = float(h60_cfg["thr_dn"])

    # Per-sym conformal band (from T170)
    cw = tcfg.get("conformal_wrapper", {})
    per_sym_beta  = {int(k): float(v) for k, v in cw.get("per_sym_beta", {}).items()}
    per_sym_sigma = {int(k): float(v) for k, v in cw.get("per_sym_sigma", {}).items()}
    default_band  = float(cw.get("default_beta_for_ood", 0.16)) * float(cw.get("default_sigma_for_ood", 4e-4))
    sym_arr = test["sym"].astype(np.int64)
    band_per_row = np.full(N_test, default_band, dtype=np.float64)
    for s in range(5):
        if s in per_sym_beta and s in per_sym_sigma:
            mask_s = sym_arr == s
            band_per_row[mask_s] = per_sym_beta[s] * per_sym_sigma[s]

    actions_h60_baseline = apply_threshold(pred_dmid, thr_up, thr_dn, band=band_per_row)

    # Build (N, 5) prediction matrix (only h60 is active, rest=1)
    pred_matrix_baseline = np.ones((N_test, 5), dtype=np.int64)
    pred_matrix_baseline[:, 4] = actions_h60_baseline  # h60 is index 4

    # Ground truth labels
    labels_matrix = np.stack([
        test["y5"], test["y10"], test["y20"], test["y40"], y60
    ], axis=1).astype(np.int64)
    mp_matrix = np.stack([mp_t5, mp_t10, mp_t20, mp_t40, mp_t60], axis=1)

    pnl_baseline = compute_pnl(pred_matrix_baseline, labels_matrix, mp_t, mp_matrix)
    t170_holdout_pnl = pnl_baseline["label_60"]["cum_pnl"]
    t170_n_active    = pnl_baseline["label_60"]["n_predictions_active"]
    print(f"  T170 baseline holdout PnL (h60): {t170_holdout_pnl:.6f}")
    print(f"  T170 baseline active trades: {t170_n_active}/{N_test} ({100*t170_n_active/N_test:.1f}%)")

    # ── 4. Fit OOD detector ───────────────────────────────────────────────────
    # Use 370-feature space for density estimation (includes all features)
    detector_dict, train_log_probs = fit_ood_detector(
        X_trval_full,  # 370-dim features on dates 0-95
        percentiles=[10, 20, 30],
        n_components=20,
        n_gmm=5,
        subsample=300000,
        seed=42,
    )

    # ── 5. Compute test log-likelihoods ──────────────────────────────────────
    update_progress("running", "Computing test log-likelihoods")
    pca = detector_dict["pca"]
    gmm = detector_dict["gmm"]

    X_test_imp = np.where(np.isnan(X_test_full), 0.0, X_test_full).astype(np.float64)
    batch_size = 50000
    test_log_probs = []
    for start in range(0, N_test, batch_size):
        X_b = X_test_imp[start:start+batch_size]
        X_b_pca = pca.transform(X_b)
        lp = gmm.score_samples(X_b_pca)
        test_log_probs.append(lp)
    test_log_probs = np.concatenate(test_log_probs)
    print(f"  Test log-prob: mean={test_log_probs.mean():.3f}, std={test_log_probs.std():.3f}")
    print(f"  Test log-prob: min={test_log_probs.min():.3f}, max={test_log_probs.max():.3f}")

    # ── 6. Evaluate OOD abstain variants ──────────────────────────────────────
    update_progress("running", "Evaluating OOD abstain variants V1/V2/V3")
    variants = {}
    output_dir = "/tmp/metabot-outputs/worker-d98c6853"
    os.makedirs(output_dir, exist_ok=True)

    for variant_key, perc_name in [("V1", "perc10"), ("V2", "perc20"), ("V3", "perc30")]:
        thr = detector_dict["thresholds"][perc_name]
        ood_mask = test_log_probs < thr
        n_abstained = int(ood_mask.sum())
        abstain_rate = n_abstained / N_test

        # Build variant actions: OOD rows get action=1 (flat)
        actions_h60_v = actions_h60_baseline.copy()
        actions_h60_v[ood_mask] = 1  # abstain OOD

        pred_matrix_v = np.ones((N_test, 5), dtype=np.int64)
        pred_matrix_v[:, 4] = actions_h60_v

        pnl_v = compute_pnl(pred_matrix_v, labels_matrix, mp_t, mp_matrix)
        holdout_pnl_v = pnl_v["label_60"]["cum_pnl"]
        n_active_v = pnl_v["label_60"]["n_predictions_active"]
        delta = holdout_pnl_v - t170_holdout_pnl

        print(f"  {variant_key} ({perc_name}, thr={thr:.4f}): "
              f"PnL={holdout_pnl_v:.6f}, delta={delta:+.6f}, "
              f"abstained={n_abstained}/{N_test} ({100*abstain_rate:.1f}%), "
              f"active={n_active_v}")

        # Check if variant gains on holdout
        build_zip_flag = delta > 0
        zip_path = None
        if build_zip_flag:
            print(f"  → Building zip for {variant_key} (holdout gain: {delta:+.6f})")
            zip_path, _ = build_submission_zip(
                variant_key, thr, detector_dict, output_dir
            )
            copy_to_outputs(zip_path)
        else:
            print(f"  → Skipping zip (holdout delta={delta:+.6f} <= 0)")

        variants[f"{variant_key}_{perc_name}"] = {
            "threshold": float(thr),
            "holdout_pnl": float(holdout_pnl_v),
            "delta": float(delta),
            "n_abstained": n_abstained,
            "abstain_rate": float(abstain_rate),
            "n_active": n_active_v,
            "zip": os.path.basename(zip_path) if zip_path else None,
        }

    # ── 7. Analysis: OOD log-prob distribution shift ──────────────────────────
    print("\n  === OOD distribution analysis ===")
    print(f"  Train log-prob mean: {train_log_probs.mean():.4f}")
    print(f"  Test  log-prob mean: {test_log_probs.mean():.4f}")
    print(f"  Shift (test-train):  {test_log_probs.mean() - train_log_probs.mean():+.4f}")

    # Per-date analysis on test
    test_dates = test["date"].astype(np.int64)
    date_stats = {}
    for d in sorted(np.unique(test_dates)):
        mask = test_dates == d
        lp_d = test_log_probs[mask]
        date_stats[int(d)] = float(lp_d.mean())

    # Show dates where mean log_p is lowest (most OOD)
    sorted_dates = sorted(date_stats.items(), key=lambda x: x[1])
    print(f"  Most OOD test dates (lowest mean log_p):")
    for d, lp in sorted_dates[:5]:
        print(f"    date={d}: mean_log_p={lp:.4f}")

    # ── 8. Save results ────────────────────────────────────────────────────────
    best_variant = max(variants.items(), key=lambda x: x[1]["holdout_pnl"])
    best_key = best_variant[0]

    results = {
        "task": "T180 OOD density-based abstain wrapper",
        "T170_baseline_holdout": float(t170_holdout_pnl),
        "T170_n_active": t170_n_active,
        "ood_detector": {
            "pca_components": 20,
            "gmm_components": 5,
            "train_data": "dates 0-95, 370 features",
            "train_log_prob_stats": detector_dict["train_log_prob_stats"],
            "test_log_prob_mean": float(test_log_probs.mean()),
            "test_log_prob_std": float(test_log_probs.std()),
            "distribution_shift": float(test_log_probs.mean() - train_log_probs.mean()),
        },
        "variants": variants,
        "best": best_key,
        "best_holdout_pnl": best_variant[1]["holdout_pnl"],
        "best_delta": best_variant[1]["delta"],
        "warning": "Holdout != platform; many abstain wrappers hurt platform despite holdout gains",
        "date_stats_most_ood": sorted_dates[:10],
    }

    results_path = os.path.join(T180_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(sanitize_for_json(results), f, indent=2)
    print(f"\n  Results saved to {results_path}")

    update_progress("done", "T180 complete", {
        "T170_baseline": t170_holdout_pnl,
        "best_variant": best_key,
        "best_delta": best_variant[1]["delta"],
    })

    print("\n=== T180 SUMMARY ===")
    print(f"  T170 baseline holdout: {t170_holdout_pnl:.6f}")
    for k, v in variants.items():
        print(f"  {k}: PnL={v['holdout_pnl']:.6f}, delta={v['delta']:+.6f}, "
              f"abstained={v['n_abstained']} ({100*v['abstain_rate']:.1f}%)")
    print(f"  Best: {best_key} (delta={best_variant[1]['delta']:+.6f})")
    return results


if __name__ == "__main__":
    results = main()
    # Final RESULT line
    bv = results["best"]
    bd = results["best_delta"]
    bp = results["best_holdout_pnl"]
    bl = results["T170_baseline_holdout"]
    print(f"\nRESULT: task=[T180 OOD density-based abstain] metrics={{T170_baseline={bl:.6f}, best_variant={bv}, best_pnl={bp:.6f}, best_delta={bd:+.6f}}} notes=[OOD GMM abstain on schemeP features; distribution shift test-train logged]")
