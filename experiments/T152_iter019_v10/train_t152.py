#!/usr/bin/env python3
"""T152: Build iter_019 v10 = v2 stack + 259 adversarial-filtered features.

Steps:
1. Compute pruned feature indices (259 features from 359)
2. TSH-FT validation: seed=42, date 0-95 train → 96-119 holdout
3. Train 5-seed LGB on full data (all dates)
4. Save models + pruned_features.json
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

# GPU occupied by another process (RL training) - use CPU mode
os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = "/root/projects/liangwenbei_workdir"
HERE = os.path.join(ROOT, "experiments", "T152_iter019_v10")
T68_DIR = os.path.join(ROOT, "experiments", "T68_stage5_features")
CACHE_DIR = os.path.join(T68_DIR, "cache")
T147_DIR = os.path.join(ROOT, "experiments", "T147_feature_pruning")

sys.path.insert(0, os.path.join(ROOT, "experiments", "T26_domain_randomization"))
from build_aug import class_balanced_weight  # noqa: E402

import lightgbm as lgb  # noqa: E402

try:
    import wandb
    WANDB_OK = True
except ImportError:
    WANDB_OK = False

FEE = 0.0001
H = 60
CHUNK = 200_000

DROP_NAMES = [
    "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
    "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
    "qrank_W100_cumspread", "kyle_lam_W50", "kyle_lam_W100",
    "roll_eff_spr_ratio_W100", "liq_asym_top5_W5",
]

# Per-seed configs matching v2 (T140c)
SEED_CONFIGS = {
    42:  dict(seed=42,  feature_fraction=0.8, bagging_fraction=0.8,  num_leaves=127, lambda_l2=1.0),
    1:   dict(seed=1,   feature_fraction=0.6, bagging_fraction=0.7,  num_leaves=127, lambda_l2=1.0),
    7:   dict(seed=7,   feature_fraction=0.7, bagging_fraction=0.85, num_leaves=63,  lambda_l2=2.0),
    13:  dict(seed=13,  feature_fraction=0.5, bagging_fraction=0.6,  num_leaves=255, lambda_l2=0.5),
    100: dict(seed=100, feature_fraction=0.4, bagging_fraction=0.5,  num_leaves=127, lambda_l2=3.0),
}

NUM_BOOST_ROUND = 330

# TSH-FT validation thresholds (from task instruction: iter_018 v1)
TSHFT_THR_UP = 4.21e-4
TSHFT_THR_DN = 1.86e-4


def ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def progress(step, **extra):
    p = os.path.join(HERE, "worker-progress.json")
    payload = {"status": "running", "step": step, "metrics": extra, "timestamp": ts()}
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {step}", flush=True)


def vectorized_pnl(pred, mp_t, mp_th, thr_up, thr_dn):
    action = np.ones(len(pred), dtype=np.int64)
    action[pred > thr_up] = 2
    action[pred < -thr_dn] = 0
    side = action.astype(np.float64) - 1.0
    abs_side = np.abs(side)
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * abs_side * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee_pnl) / denom


def aug_a_scale_chunked(rng, X_src, X_dst, lo=0.80, hi=1.20):
    n = len(X_src)
    for i in range(0, n, CHUNK):
        end = min(i + CHUNK, n)
        sz = end - i
        scales = rng.uniform(lo, hi, size=(sz, X_src.shape[1])).astype(np.float32)
        X_dst[i:end] = X_src[i:end] * scales


def compute_feature_indices():
    """Compute indices for the 259-feature pruned set."""
    # Load all 370 feature names from cache
    feat_names_path = os.path.join(CACHE_DIR, "schemeP_feat_names.txt")
    with open(feat_names_path) as f:
        all_feat_names = [line.strip() for line in f if line.strip()]
    print(f"  Total feat names: {len(all_feat_names)}", flush=True)

    # Step 1: Drop FAIL_NAMES → 359 features
    drop_set = set(DROP_NAMES)
    keep_idx_in_370 = np.array(
        [i for i, n in enumerate(all_feat_names) if n not in drop_set], dtype=np.int64
    )
    feat_names_359 = [all_feat_names[i] for i in keep_idx_in_370]
    assert len(feat_names_359) == 359, f"Expected 359, got {len(feat_names_359)}"
    print(f"  After DROP_NAMES: {len(feat_names_359)} features", flush=True)

    # Verify no forbidden features
    forbidden = {"date", "sym", "time"}
    assert not (forbidden & set(feat_names_359)), "Forbidden features in feat_names_359!"

    # Step 2: Load 259 kept features from T147
    best_features_path = os.path.join(T147_DIR, "best_features.txt")
    with open(best_features_path) as f:
        kept_features_259 = [line.strip() for line in f if line.strip()]
    print(f"  T147 kept features: {len(kept_features_259)}", flush=True)
    assert len(kept_features_259) == 259, f"Expected 259, got {len(kept_features_259)}"

    # Step 3: Find indices within 359-feature vector for the 259 kept features
    kept_set = set(kept_features_259)
    pruned_idx_in_359 = np.array(
        [i for i, n in enumerate(feat_names_359) if n in kept_set], dtype=np.int64
    )
    pruned_feat_names = [feat_names_359[i] for i in pruned_idx_in_359]
    assert len(pruned_idx_in_359) == 259, f"Expected 259, got {len(pruned_idx_in_359)}"
    print(f"  Pruned idx in 359: {len(pruned_idx_in_359)} features", flush=True)

    # Final indices into 370-dim cache array
    keep_idx_in_370_final = keep_idx_in_370[pruned_idx_in_359]
    assert len(keep_idx_in_370_final) == 259

    # Verify no forbidden features in pruned set
    assert not (forbidden & set(pruned_feat_names)), "Forbidden features in pruned set!"

    return {
        "keep_idx_in_370": keep_idx_in_370,       # 359 indices into 370
        "pruned_idx_in_359": pruned_idx_in_359,   # 259 indices into 359
        "keep_idx_in_370_final": keep_idx_in_370_final,  # 259 indices into 370
        "feat_names_359": feat_names_359,
        "pruned_feat_names": pruned_feat_names,
        "kept_features_259": kept_features_259,
    }


def load_all_data(feat_info, date_mask_fn=None):
    """Load all splits, filter by date_mask_fn if provided, return X, y, mp_t, mp_th, date."""
    keep_idx_in_370_final = feat_info["keep_idx_in_370_final"]
    X_parts, y_parts, mp_t_parts, mp_th_parts, date_parts = [], [], [], [], []

    for split in ["train", "val", "test"]:
        print(f"  loading {split}...", flush=True)
        d = np.load(os.path.join(CACHE_DIR, f"schemeP_{split}.npz"))
        date_arr = d["date"]

        if date_mask_fn is not None:
            mask = date_mask_fn(date_arr)
            if mask.sum() == 0:
                d.close()
                continue
            X_parts.append(d["X"][mask][:, keep_idx_in_370_final].astype(np.float32))
            y_parts.append(((d[f"mp_t{H}"][mask].astype(np.float64) - d["mp_t"][mask].astype(np.float64))
                            / (d["mp_t"][mask].astype(np.float64) + 1.0)).astype(np.float32))
            mp_t_parts.append(d["mp_t"][mask].astype(np.float64))
            mp_th_parts.append(d[f"mp_t{H}"][mask].astype(np.float64))
            date_parts.append(date_arr[mask])
        else:
            X_parts.append(d["X"][:, keep_idx_in_370_final].astype(np.float32))
            y_parts.append(((d[f"mp_t{H}"].astype(np.float64) - d["mp_t"].astype(np.float64))
                            / (d["mp_t"].astype(np.float64) + 1.0)).astype(np.float32))
            mp_t_parts.append(d["mp_t"].astype(np.float64))
            mp_th_parts.append(d[f"mp_t{H}"].astype(np.float64))
            date_parts.append(date_arr)
        d.close()
        gc.collect()

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    mp_t = np.concatenate(mp_t_parts, axis=0)
    mp_th = np.concatenate(mp_th_parts, axis=0)
    date = np.concatenate(date_parts, axis=0)
    return X, y, mp_t, mp_th, date


def run_tshft_validation(feat_info):
    """Quick TSH-FT validation: seed=42, train date 0-95, holdout date 96-119.
    Memory-efficient: load all splits once, split by date, free intermediate arrays.
    """
    progress("tshft_validation")
    print("\n=== TSH-FT Validation (seed=42, date 0-95 → 96-119) ===", flush=True)

    keep_idx_in_370_final = feat_info["keep_idx_in_370_final"]
    feat_dim = len(keep_idx_in_370_final)

    # Collect arrays split-by-split, filter by date inline to avoid double-loading
    X_tr_parts, y_tr_parts = [], []
    X_ho_parts, mp_t_ho_parts, mp_th_ho_parts = [], [], []

    for split in ["train", "val", "test"]:
        d = np.load(os.path.join(CACHE_DIR, f"schemeP_{split}.npz"))
        date_arr = d["date"]
        X_raw = d["X"][:, keep_idx_in_370_final].astype(np.float32)
        mp_t = d["mp_t"].astype(np.float64)
        mp_th = d[f"mp_t{H}"].astype(np.float64)
        y_regr = ((mp_th - mp_t) / (mp_t + 1.0)).astype(np.float32)
        d.close()

        mask_tr = date_arr <= 95
        mask_ho = (date_arr >= 96) & (date_arr <= 119)

        if mask_tr.any():
            X_tr_parts.append(X_raw[mask_tr])
            y_tr_parts.append(y_regr[mask_tr])
        if mask_ho.any():
            X_ho_parts.append(X_raw[mask_ho])
            mp_t_ho_parts.append(mp_t[mask_ho])
            mp_th_ho_parts.append(mp_th[mask_ho])

        del X_raw, mp_t, mp_th, y_regr
        gc.collect()

    X_tr = np.concatenate(X_tr_parts, axis=0); del X_tr_parts
    y_tr = np.concatenate(y_tr_parts, axis=0); del y_tr_parts
    X_ho = np.concatenate(X_ho_parts, axis=0); del X_ho_parts
    mp_t_ho = np.concatenate(mp_t_ho_parts, axis=0); del mp_t_ho_parts
    mp_th_ho = np.concatenate(mp_th_ho_parts, axis=0); del mp_th_ho_parts
    gc.collect()
    print(f"  Train: {X_tr.shape}  Holdout: {X_ho.shape}", flush=True)

    cfg = SEED_CONFIGS[42]
    params = {
        "objective": "regression_l2",
        "metric": "l2",
        "learning_rate": 0.05,
        "num_leaves": int(cfg["num_leaves"]),
        "min_data_in_leaf": 100,
        "feature_fraction": float(cfg["feature_fraction"]),
        "bagging_fraction": float(cfg["bagging_fraction"]),
        "bagging_freq": 5,
        "lambda_l2": float(cfg["lambda_l2"]),
        "num_threads": 8,
        "seed": 42,
        "verbose": -1,
        # CPU mode: GPU occupied by another process

    }

    t0 = time.time()
    # Use free_raw_data=True so LGB frees X_tr after bin construction
    ds = lgb.Dataset(X_tr, label=y_tr, free_raw_data=True)
    ds.construct()
    del X_tr, y_tr
    gc.collect()

    model = lgb.train(params, ds, num_boost_round=NUM_BOOST_ROUND,
                      callbacks=[lgb.log_evaluation(period=100)])
    print(f"  Trained in {time.time()-t0:.1f}s", flush=True)
    del ds
    gc.collect()

    # Predict on holdout
    pred_ho = model.predict(X_ho)
    pnl = vectorized_pnl(pred_ho, mp_t_ho, mp_th_ho, TSHFT_THR_UP, TSHFT_THR_DN)
    total_pnl = float(pnl.sum())
    n_trades = int((pred_ho > TSHFT_THR_UP).sum() + (pred_ho < -TSHFT_THR_DN).sum())
    print(f"  Holdout PnL: {total_pnl:+.4f}  n_trades: {n_trades}", flush=True)

    del X_ho, mp_t_ho, mp_th_ho, model
    gc.collect()

    return total_pnl, n_trades


def train_full_seed(feat_info, seed, wandb_run=None):
    """Train one seed on full data (all dates) with augmentation."""
    progress(f"training_seed_{seed}", seed=seed)
    print(f"\n=== Training seed={seed} on full data ===", flush=True)

    # Load all data
    print("  Loading all splits...", flush=True)
    splits = ["train", "val", "test"]
    keep_idx_in_370_final = feat_info["keep_idx_in_370_final"]
    feat_dim = len(keep_idx_in_370_final)

    # Count rows first
    split_sizes = {}
    for split in splits:
        d = np.load(os.path.join(CACHE_DIR, f"schemeP_{split}.npz"))
        split_sizes[split] = len(d["date"])
        d.close()
    n_total = sum(split_sizes.values())
    n_aug = n_total
    n_train_used = n_total + n_aug

    print(f"  n_total={n_total:,}  feat_dim={feat_dim}  n_train_used={n_train_used:,}", flush=True)
    print(f"  Pre-allocating {n_train_used*feat_dim*4/1e9:.2f} GB...", flush=True)

    X_tr_full = np.empty((n_train_used, feat_dim), dtype=np.float32)
    y_regr_tr = np.empty(n_train_used, dtype=np.float32)
    y_cls_tr = np.empty(n_train_used, dtype=np.int64)

    offset = 0
    for split in splits:
        d = np.load(os.path.join(CACHE_DIR, f"schemeP_{split}.npz"))
        n = split_sizes[split]
        X_tr_full[offset:offset+n] = d["X"][:, keep_idx_in_370_final].astype(np.float32)
        mp_t = d["mp_t"].astype(np.float64)
        mp_th = d[f"mp_t{H}"].astype(np.float64)
        y_regr_tr[offset:offset+n] = ((mp_th - mp_t) / (mp_t + 1.0)).astype(np.float32)
        y_cls_tr[offset:offset+n] = d[f"y{H}"]
        offset += n
        d.close()
        gc.collect()
        print(f"  loaded {split} ({split_sizes[split]:,} rows)", flush=True)

    # Augmentation
    seed_rng = np.random.default_rng(seed * 7919 + 1)
    aug_a_scale_chunked(seed_rng, X_tr_full[:n_total], X_tr_full[n_total:])
    y_regr_tr[n_total:] = y_regr_tr[:n_total]
    y_cls_tr[n_total:] = y_cls_tr[:n_total]
    gc.collect()
    print(f"  augmentation done", flush=True)

    sw_tr = class_balanced_weight(y_cls_tr, num_class=3)

    cfg = SEED_CONFIGS[seed]
    params = {
        "objective": "regression_l2",
        "metric": "l2",
        "learning_rate": 0.05,
        "num_leaves": int(cfg["num_leaves"]),
        "min_data_in_leaf": 100,
        "feature_fraction": float(cfg["feature_fraction"]),
        "bagging_fraction": float(cfg["bagging_fraction"]),
        "bagging_freq": 5,
        "lambda_l2": float(cfg["lambda_l2"]),
        "num_threads": 8,
        "seed": seed,
        "feature_fraction_seed": seed + 1,
        "bagging_seed": seed + 2,
        "verbose": -1,
        # CPU mode: GPU occupied by another process

    }

    print(f"  cfg: {cfg}", flush=True)
    dtrain = lgb.Dataset(X_tr_full, label=y_regr_tr, weight=sw_tr,
                         feature_name=feat_info["pruned_feat_names"],
                         free_raw_data=True)
    dtrain.construct()
    del X_tr_full, y_regr_tr, sw_tr
    gc.collect()
    print(f"  dataset constructed", flush=True)

    t_start = time.time()
    booster = lgb.train(params, dtrain, num_boost_round=NUM_BOOST_ROUND,
                        callbacks=[lgb.log_evaluation(period=50)])
    train_time = time.time() - t_start
    print(f"  trained in {train_time:.1f}s", flush=True)

    model_path = os.path.join(HERE, f"model_h{H}_seed{seed}.txt")
    booster.save_model(model_path)
    print(f"  saved {model_path}", flush=True)

    if wandb_run is not None:
        wandb_run.log({
            f"seed_{seed}/train_time": train_time,
            f"seed_{seed}/n_total": n_total,
        })

    del dtrain, booster
    gc.collect()
    return train_time


def main():
    print("=== T152: iter_019 v10 = v2 + 259 adversarial-filtered features ===", flush=True)
    print(f"GPU: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    import subprocess
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.free",
                                       "--format=csv,noheader"], text=True).strip()
        print(f"GPU: {out}", flush=True)
    except Exception:
        print("nvidia-smi not available", flush=True)

    os.makedirs(HERE, exist_ok=True)

    # Step 1: Compute feature indices
    progress("computing_feature_indices")
    print("\n--- Step 1: Feature indices ---", flush=True)
    feat_info = compute_feature_indices()

    # Save pruned_features.json
    pruned_info = {
        "kept_features": feat_info["kept_features_259"],
        "pruned_idx_in_359": feat_info["pruned_idx_in_359"].tolist(),
        "n_kept": 259,
        "n_dropped": 100,
    }
    pruned_json_path = os.path.join(HERE, "pruned_features.json")
    with open(pruned_json_path, "w") as f:
        json.dump(pruned_info, f, indent=2)
    print(f"  Saved {pruned_json_path}", flush=True)

    # Step 2: TSH-FT validation
    print("\n--- Step 2: TSH-FT Validation ---", flush=True)
    tshft_pnl, n_trades = run_tshft_validation(feat_info)
    baseline_v2_pnl = 16.8739  # from T147 results
    tshft_delta = tshft_pnl - baseline_v2_pnl
    print(f"\nTSH-FT result: {tshft_pnl:+.4f} vs baseline {baseline_v2_pnl:+.4f} → delta={tshft_delta:+.4f}", flush=True)

    if tshft_delta < -3.0:
        print(f"ABORT: TSH-FT regression {tshft_delta:.2f} > 3.0 threshold!", flush=True)
        results = {
            "task": "T152 iter_019 v10 aborted: TSH-FT regression",
            "tshft_v10_259": tshft_pnl,
            "tshft_baseline_v2_359": baseline_v2_pnl,
            "tshft_delta": tshft_delta,
            "abort_reason": f"TSH-FT regression {tshft_delta:.2f} exceeds -3.0 threshold",
        }
        with open(os.path.join(HERE, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        progress("aborted", tshft_delta=tshft_delta)
        sys.exit(1)

    print(f"TSH-FT OK (delta={tshft_delta:+.4f}), proceeding with full train.", flush=True)

    # Step 3: Init WandB
    wandb_run = None
    if WANDB_OK:
        try:
            import wandb as wb
            wb.login(key="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",
                     relogin=True)
            wandb_run = wb.init(
                project="liangwenbei",
                entity="cjxh21-Tsinghua University",
                name="T152-v10-pruned259-fulldata",
                config={
                    "n_features": 259,
                    "n_dropped": 100,
                    "method": "adversarial_filter_C_drop100",
                    "base_version": "iter_019_v2",
                    "seeds": [1, 7, 13, 42, 100],
                    "num_boost_round": NUM_BOOST_ROUND,
                    "tshft_pnl": tshft_pnl,
                    "tshft_delta": tshft_delta,
                }
            )
            print("WandB initialized", flush=True)
        except Exception as e:
            print(f"WandB init failed: {e}", flush=True)
            wandb_run = None

    # Step 4: Train 5 seeds on full data
    print("\n--- Step 4: Training 5 seeds on full data ---", flush=True)
    seeds = [1, 7, 13, 42, 100]
    train_times = {}
    for seed in seeds:
        t = train_full_seed(feat_info, seed, wandb_run)
        train_times[seed] = t
        progress(f"training_done_seed_{seed}", seed=seed, train_time=t)

    if wandb_run is not None:
        wandb_run.log({"total_train_time": sum(train_times.values())})
        wandb_run.finish()

    # Step 5: Save results
    print("\n--- Step 5: Saving results ---", flush=True)
    model_files = [f for f in os.listdir(HERE) if f.startswith("model_h60_seed")]
    results = {
        "task": "T152 iter_019 v10 = v2 + adversarial-filtered 259 features",
        "n_features_kept": 259,
        "n_features_dropped": 100,
        "tshft_baseline_v2_359": baseline_v2_pnl,
        "tshft_v10_259": tshft_pnl,
        "tshft_delta": tshft_delta,
        "tshft_n_trades": n_trades,
        "train_times_per_seed": train_times,
        "model_files": sorted(model_files),
        "status": "models_trained",
    }
    results_path = os.path.join(HERE, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {results_path}", flush=True)

    progress("done",
             tshft_pnl=tshft_pnl,
             tshft_delta=tshft_delta,
             n_features=259,
             models_trained=len(model_files))

    print(f"\n=== Training complete ===", flush=True)
    print(f"TSH-FT: {tshft_pnl:+.4f} (delta={tshft_delta:+.4f})", flush=True)
    print(f"Models: {sorted(model_files)}", flush=True)


if __name__ == "__main__":
    main()
