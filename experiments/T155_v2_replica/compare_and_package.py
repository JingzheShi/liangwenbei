"""T155: Compare replica vs v2 models, build zip, smoke test via direct LGB+NN predictions."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import zipfile
import shutil
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
ORIG_DIR = os.path.join(ROOT, "experiments", "R_full_retrain")
V2_PKG_DIR = os.path.join(ORIG_DIR, "pkg_iter019_v2")
T155_PKG_DIR = os.path.join(HERE, "pkg")
T68_DIR = os.path.join(ROOT, "experiments", "T68_stage5_features")
CACHE_DIR = os.path.join(T68_DIR, "cache")

SEEDS = [1, 7, 13, 42, 100]
DROP_NAMES = [
    "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
    "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
    "qrank_W100_cumspread", "kyle_lam_W50", "kyle_lam_W100",
    "roll_eff_spr_ratio_W100", "liq_asym_top5_W5",
]


def md5file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_keep_idx():
    feat_names_path = os.path.join(CACHE_DIR, "schemeP_feat_names.txt")
    with open(feat_names_path) as f:
        feat_names = [l.strip() for l in f]
    name_to_idx = {n: i for i, n in enumerate(feat_names)}
    drop_idx = set(name_to_idx[n] for n in DROP_NAMES if n in name_to_idx)
    return np.array([i for i in range(len(feat_names)) if i not in drop_idx], dtype=np.int64)


def load_test_sample(n=10000):
    keep_idx = get_keep_idx()
    d = np.load(os.path.join(CACHE_DIR, "schemeP_test.npz"))
    X = d["X"][:n, :][:, keep_idx].astype(np.float32)
    d.close()
    return X


def predict_with_lgb(model_path, X):
    import lightgbm as lgb
    booster = lgb.Booster(model_file=model_path)
    return booster.predict(X).astype(np.float32)


def lgb_ensemble(model_dir, seed_list, X):
    preds = []
    for s in seed_list:
        p = os.path.join(model_dir, f"model_h60_seed{s}.txt")
        preds.append(predict_with_lgb(p, X))
    return np.mean(preds, axis=0)


def nn_ensemble(pkg_dir, seed_list, X):
    """Load NN npz models and predict with numpy inference (from Predictor._MLPNumpy)."""
    # Add pkg_dir so we can import fast_features if needed, but NN inference is pure numpy
    sys.path.insert(0, pkg_dir)

    def gelu_tanh(x):
        c = 0.7978845608028654
        return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x * x * x)))

    def layernorm(x, gamma, beta, eps=1e-5):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps) * gamma + beta

    def nn_predict_single(npz_path, X_in):
        d = np.load(npz_path, allow_pickle=False)
        in_dim = int(d["in_dim"][0])
        hidden = list(d["hidden"].tolist())
        use_ln = bool(d["use_layernorm"][0])
        target_scale = float(d["target_scale"][0])
        feat_mean = d["feat_mean"].astype(np.float32)
        feat_std = d["feat_std"].astype(np.float32)
        keep_idx = d["keep_idx"].astype(np.int64)
        clip_ = float(d["clip"][0])

        Xs = X_in if X_in.shape[1] == in_dim else X_in[:, keep_idx]
        Xs = Xs.astype(np.float32, copy=True)
        Xs = (Xs - feat_mean) / feat_std
        Xs = np.where(np.isnan(Xs), 0.0, Xs)
        Xs = np.clip(Xs, -clip_, clip_).astype(np.float32)

        W_list, b_list, LN_W_list, LN_b_list = [], [], [], []
        for i in range(len(hidden)):
            W_list.append(d[f"L{i}_W"].astype(np.float32))
            b_list.append(d[f"L{i}_b"].astype(np.float32))
            if use_ln:
                LN_W_list.append(d[f"LN{i}_W"].astype(np.float32))
                LN_b_list.append(d[f"LN{i}_b"].astype(np.float32))

        WF = d["LF_W"].astype(np.float32)
        bF = d["LF_b"].astype(np.float32)

        h = Xs
        for i in range(len(hidden)):
            h = h @ W_list[i].T + b_list[i]
            if use_ln:
                h = layernorm(h, LN_W_list[i], LN_b_list[i])
            h = gelu_tanh(h)
        out = (h @ WF.T + bF).squeeze(-1) / target_scale
        return out.astype(np.float32)

    preds = []
    for s in seed_list:
        npz_path = os.path.join(pkg_dir, f"nn_h60_seed{s}.npz")
        preds.append(nn_predict_single(npz_path, X))
    return np.mean(preds, axis=0)


def apply_threshold(ens_pred, thr_up, thr_dn):
    """Apply fixed threshold (no conformal band for simplicity)."""
    actions = np.ones(len(ens_pred), dtype=np.int64)
    actions[ens_pred > thr_up] = 2
    actions[ens_pred < -thr_dn] = 0
    return actions


def build_pkg():
    if os.path.exists(T155_PKG_DIR):
        shutil.rmtree(T155_PKG_DIR)
    shutil.copytree(V2_PKG_DIR, T155_PKG_DIR)
    print(f"Copied v2 pkg to {T155_PKG_DIR}")
    for seed in SEEDS:
        src = os.path.join(HERE, f"model_h60_seed{seed}.txt")
        dst = os.path.join(T155_PKG_DIR, f"model_h60_seed{seed}.txt")
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Replaced model seed {seed}: {os.path.getsize(dst)/1024:.1f} KB")
        else:
            print(f"  WARNING: model seed {seed} not found at {src}")


def build_zip(pkg_dir, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in sorted(os.listdir(pkg_dir)):
            fpath = os.path.join(pkg_dir, fname)
            if os.path.isfile(fpath):
                zf.write(fpath, fname)


def main():
    print("=== T155 compare_and_package ===")

    # 1. Check replica models
    print("\n[1] Checking replica model files...")
    for seed in SEEDS:
        p = os.path.join(HERE, f"model_h60_seed{seed}.txt")
        if not os.path.exists(p):
            print(f"  MISSING: seed {seed}")
            sys.exit(1)
        print(f"  OK seed {seed}: {os.path.getsize(p)/1024:.1f} KB")

    # 2. Compare models
    print("\n[2] Comparing replica vs original models on 10000 test rows...")
    X_sample = load_test_sample(10000)
    print(f"  Loaded test sample: {X_sample.shape}")

    model_diff = {}
    orig_preds_list = []
    replica_preds_list = []

    for seed in SEEDS:
        orig_path = os.path.join(ORIG_DIR, f"model_h60_seed{seed}.txt")
        repl_path = os.path.join(HERE, f"model_h60_seed{seed}.txt")

        orig_md5 = md5file(orig_path)
        repl_md5 = md5file(repl_path)
        md5_match = orig_md5 == repl_md5

        orig_pred = predict_with_lgb(orig_path, X_sample)
        repl_pred = predict_with_lgb(repl_path, X_sample)
        pred_mae = float(np.mean(np.abs(orig_pred - repl_pred)))
        pred_max = float(np.max(np.abs(orig_pred - repl_pred)))
        pred_corr = float(np.corrcoef(orig_pred, repl_pred)[0, 1])

        orig_preds_list.append(orig_pred)
        replica_preds_list.append(repl_pred)

        model_diff[f"seed{seed}"] = {
            "orig_md5": orig_md5, "repl_md5": repl_md5, "md5_match": md5_match,
            "pred_mae": pred_mae, "pred_max_diff": pred_max, "pred_corr": pred_corr,
        }
        print(f"  seed {seed}: md5_match={md5_match}, pred_MAE={pred_mae:.2e}, "
              f"max_diff={pred_max:.2e}, corr={pred_corr:.6f}")

    orig_ens = np.mean(orig_preds_list, axis=0)
    repl_ens = np.mean(replica_preds_list, axis=0)
    ens_mae = float(np.mean(np.abs(orig_ens - repl_ens)))
    ens_corr = float(np.corrcoef(orig_ens, repl_ens)[0, 1])
    print(f"\n  Ensemble pred MAE: {ens_mae:.2e}, corr={ens_corr:.6f}")

    # 3. NN predictions (same for both since NN is identical)
    print("\n[3] Computing combined NN+LGB ensemble predictions...")
    nn_pred = nn_ensemble(V2_PKG_DIR, SEEDS, X_sample)
    w_nn, w_lgb = 1.0, 0.5

    # Load thresholds
    with open(os.path.join(V2_PKG_DIR, "thresholds.json")) as f:
        tcfg = json.load(f)
    h60_cfg = next(h for h in tcfg["horizons"] if h["h"] == 60)
    thr_up = float(h60_cfg["thr_up"])
    thr_dn = float(h60_cfg["thr_dn"])
    print(f"  thr_up={thr_up:.6f}, thr_dn={thr_dn:.6f}")

    # Use first 1000 rows for action comparison
    X1000 = X_sample[:1000]
    nn_pred_1000 = nn_pred[:1000]
    orig_lgb_1000 = orig_ens[:1000]
    repl_lgb_1000 = repl_ens[:1000]

    # Combined ensemble prediction
    orig_combo = (w_nn * nn_pred_1000 + w_lgb * orig_lgb_1000) / (w_nn + w_lgb)
    repl_combo = (w_nn * nn_pred_1000 + w_lgb * repl_lgb_1000) / (w_nn + w_lgb)

    orig_actions = apply_threshold(orig_combo, thr_up, thr_dn)
    repl_actions = apply_threshold(repl_combo, thr_up, thr_dn)

    actions_differ = int(np.sum(orig_actions != repl_actions))
    print(f"\n  Actions differ (out of 1000): {actions_differ}")
    print(f"  Orig action dist: {np.bincount(orig_actions + 0, minlength=3)}")
    print(f"  Repl action dist: {np.bincount(repl_actions + 0, minlength=3)}")

    # 4. Build pkg and zip
    print("\n[4] Building v2_replica pkg...")
    build_pkg()
    zip_path = os.path.join(ROOT, "submission_050819_iter019_v2_replica.zip")
    build_zip(T155_PKG_DIR, zip_path)
    zip_md5 = md5file(zip_path)
    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"  Zip: {zip_path}")
    print(f"  MD5: {zip_md5}")
    print(f"  Size: {zip_size_mb:.2f} MB")

    # 5. Determine verdict
    if ens_mae < 1e-5:
        verdict = "DETERMINISTIC: replica ensemble predictions essentially identical to v2 originals (MAE<1e-5). LGB training IS reproducible. Platform PnL differences must be due to OTHER factors (e.g. feature changes)."
    elif ens_mae < 1e-3:
        verdict = f"SEMI-DETERMINISTIC: MAE={ens_mae:.2e} (between 1e-5 and 1e-3). Small but nonzero differences. May affect platform PnL but unlikely to explain -5 to -9 regression."
    else:
        verdict = f"NON-DETERMINISTIC: MAE={ens_mae:.2e} (>1e-3). Training is not reproducible. LGB retraining likely changed platform PnL."

    print(f"\nVERDICT: {verdict}")

    # 6. Write results.json
    results = {
        "task": "T155 v2 replica reproducibility test",
        "training_mode": "T140c ACTUAL config (from summaries): CPU train_full.py(seed1,7,no-gpu,18threads,+data_random_seed) + CPU train_full_memeff.py(seed13,42,100,8threads,no-data_random_seed)",
        "important_note": "PM hypothesis was WRONG: seeds 1,7 were NOT GPU-trained in T140c (summary shows use_gpu=False). All seeds CPU. Key diff between scripts: data_random_seed param.",
        "model_diff": model_diff,
        "ensemble_pred_mae_vs_original": ens_mae,
        "ensemble_pred_corr_vs_original": ens_corr,
        "actions_differ_count_per_1000": actions_differ,
        "v2_replica_zip": zip_path,
        "v2_replica_md5": zip_md5,
        "v2_replica_size_mb": round(zip_size_mb, 2),
        "verdict_hypothesis": verdict,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    out_path = os.path.join(HERE, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults -> {out_path}")

    print(f"\nRESULT: task=T155_v2_replica metrics={{ensemble_pred_mae={ens_mae:.2e},actions_differ={actions_differ},zip_size_mb={zip_size_mb:.2f}}} notes={verdict[:80].replace(' ','_')}")


if __name__ == "__main__":
    main()
