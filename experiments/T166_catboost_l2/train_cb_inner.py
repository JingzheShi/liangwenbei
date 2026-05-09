"""T166: CatBoost L2 (RMSE) 5-seed train on INNER (0-95), predict on test (96-119).

Protocol:
- Train on inner: train split (0-79) + val split (80-95) concatenated
- No early stopping (M7: fixed 800 iters)
- aug_a ratio=1.0, lo=0.8, hi=1.2, sym-coupled scaling
- class-balanced sample weights
- task_type='GPU' if available
- Per-seed configs: depth/l2_leaf_reg/learning_rate diversity

Predictions on test (96-119) are proper holdout (no leakage from test dates).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from catboost import CatBoostRegressor, Pool

T26_DIR = os.path.join(ROOT, "experiments", "T26_domain_randomization")
sys.path.insert(0, T26_DIR)
from build_aug import class_balanced_weight

T68_DIR = os.path.join(ROOT, "experiments", "T68_stage5_features")
CACHE_DIR = os.path.join(T68_DIR, "cache")

NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001

# Per-seed configs from task spec
SEED_CONFIGS = {
    1:   dict(seed=1,   depth=6, l2_leaf_reg=3.0, learning_rate=0.05),
    7:   dict(seed=7,   depth=8, l2_leaf_reg=5.0, learning_rate=0.05),
    13:  dict(seed=13,  depth=6, l2_leaf_reg=2.0, learning_rate=0.05),
    42:  dict(seed=42,  depth=7, l2_leaf_reg=3.0, learning_rate=0.05),
    100: dict(seed=100, depth=8, l2_leaf_reg=5.0, learning_rate=0.05),
}

T59_FAIL_NAMES = [
    "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
    "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
    "qrank_W100_cumspread", "kyle_lam_W50", "kyle_lam_W100",
    "roll_eff_spr_ratio_W100",
]
STAGE5_FAIL_NAMES = ["liq_asym_top5_W5"]
DROP_NAMES = T59_FAIL_NAMES + STAGE5_FAIL_NAMES

CHUNK = 200_000


def aug_a_scale_chunked(rng, X_src, X_dst, lo=0.80, hi=1.20):
    n = len(X_src)
    for i in range(0, n, CHUNK):
        end = min(i + CHUNK, n)
        sz = end - i
        scales = rng.uniform(lo, hi, size=(sz, X_src.shape[1])).astype(np.float32)
        X_dst[i:end] = X_src[i:end] * scales
    del scales


def progress(step, **extra):
    p = os.path.join(HERE, "worker-progress.json")
    payload = {"status": "running", "step": step,
               "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               **extra}
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--iterations", type=int, default=800)
    ap.add_argument("--aug-lo", type=float, default=0.80)
    ap.add_argument("--aug-hi", type=float, default=1.20)
    ap.add_argument("--use-gpu", action="store_true", default=True)
    ap.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    H = args.horizon
    seed = args.seed
    cfg = SEED_CONFIGS[seed]
    print(f"=== T166 CB L2 inner-train seed={seed} h={H} iters={args.iterations} cfg={cfg} ===",
          flush=True)

    # Load feature names
    with open(os.path.join(CACHE_DIR, "schemeP_feat_names.txt")) as f:
        all_feat_names = [line.strip() for line in f]
    total_dim = len(all_feat_names)
    name_to_idx = {n: i for i, n in enumerate(all_feat_names)}
    drop_idx = {name_to_idx[n] for n in DROP_NAMES if n in name_to_idx}
    keep_idx = np.array([i for i in range(total_dim) if i not in drop_idx], dtype=np.int64)
    feat_dim = len(keep_idx)
    feat_names = [all_feat_names[i] for i in keep_idx]
    forbidden = {"date", "sym", "time"}
    assert not (forbidden & set(feat_names)), f"leakage: {forbidden & set(feat_names)}"
    print(f"  feat_dim={feat_dim}", flush=True)

    progress("loading_inner", seed=seed)
    t0 = time.time()

    # Load inner data: train (0-79) + val (80-95)
    parts = []
    mp_t_parts, mp_th_parts, y_cls_parts = [], [], []
    for split in ["train", "val"]:
        d = np.load(os.path.join(CACHE_DIR, f"schemeP_{split}.npz"))
        parts.append(d["X"][:, keep_idx].astype(np.float32))
        mp_t_parts.append(d["mp_t"].astype(np.float64))
        mp_th_parts.append(d[f"mp_t{H}"].astype(np.float64))
        y_cls_parts.append(d[f"y{H}"].astype(np.int64))
        d.close()
        gc.collect()

    X_inner = np.concatenate(parts, axis=0)
    mp_t_inner = np.concatenate(mp_t_parts)
    mp_th_inner = np.concatenate(mp_th_parts)
    y_cls_inner = np.concatenate(y_cls_parts)
    del parts, mp_t_parts, mp_th_parts, y_cls_parts
    gc.collect()

    y_regr_inner = ((mp_th_inner - mp_t_inner) / (mp_t_inner + 1.0)).astype(np.float32)
    n_inner = len(X_inner)
    print(f"  n_inner={n_inner:,} feat_dim={feat_dim}", flush=True)
    print(f"  y_regr: mean={y_regr_inner.mean():.6f} std={y_regr_inner.std():.6f}", flush=True)

    # Augmentation (sym-coupled scale aug_a)
    progress("augmentation", seed=seed)
    seed_rng = np.random.default_rng(seed * 7919 + 1)
    X_aug = np.empty_like(X_inner)
    aug_a_scale_chunked(seed_rng, X_inner, X_aug, lo=args.aug_lo, hi=args.aug_hi)
    y_regr_aug = y_regr_inner.copy()
    y_cls_aug = y_cls_inner.copy()

    X_tr = np.concatenate([X_inner, X_aug], axis=0)
    y_regr_tr = np.concatenate([y_regr_inner, y_regr_aug])
    y_cls_tr = np.concatenate([y_cls_inner, y_cls_aug])
    del X_aug, y_regr_aug, y_cls_aug, X_inner
    gc.collect()
    print(f"  n_train_used={len(X_tr):,} (inner + aug_a)", flush=True)

    sw_tr = class_balanced_weight(y_cls_tr, num_class=NUM_CLASS)
    print(f"  loaded+aug in {time.time()-t0:.1f}s", flush=True)

    # WandB init
    use_wandb = not args.no_wandb
    run = None
    if use_wandb:
        try:
            import wandb
            run_name = f"T166-CB-L2-inner-seed{seed}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            run = wandb.init(
                project="liangwenbei",
                entity="cjxh21-Tsinghua University",
                name=run_name,
                config={"seed": seed, "horizon": H, "iterations": args.iterations,
                        "n_inner": n_inner, "n_train_used": len(X_tr),
                        "feat_dim": feat_dim, **cfg},
                tags=["T166", "catboost", "L2", "inner", f"h{H}"],
            )
        except Exception as e:
            print(f"  WandB init failed ({e!r}); skipping", flush=True)
            use_wandb = False

    # Build CatBoost pool
    progress("building_pool", seed=seed)
    train_pool = Pool(X_tr, label=y_regr_tr, weight=sw_tr, feature_names=feat_names)
    del X_tr, y_regr_tr, sw_tr, y_cls_tr
    gc.collect()

    # CatBoost params
    cb_params = dict(
        iterations=args.iterations,
        learning_rate=cfg["learning_rate"],
        depth=cfg["depth"],
        l2_leaf_reg=cfg["l2_leaf_reg"],
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=seed,
        bootstrap_type="Bernoulli",
        subsample=0.8,
        verbose=100,
        early_stopping_rounds=None,  # M7: no early stop
        allow_writing_files=False,
    )
    if args.use_gpu:
        cb_params["task_type"] = "GPU"
        cb_params["devices"] = "0"
        print("  Using GPU", flush=True)
    else:
        cb_params["thread_count"] = 18
        print("  Using CPU", flush=True)

    # Train
    progress("training", seed=seed, n_train=n_inner)
    t_start = time.time()
    model = CatBoostRegressor(**cb_params)
    model.fit(train_pool)
    train_time = time.time() - t_start
    print(f"  trained in {train_time:.1f}s, tree_count={model.tree_count_}", flush=True)

    model_path = os.path.join(HERE, f"model_CB_inner_seed{seed}.cbm")
    model.save_model(model_path)
    print(f"  saved {model_path}", flush=True)

    # Predict on test (96-119) — proper holdout
    progress("predicting_test", seed=seed)
    d_test = np.load(os.path.join(CACHE_DIR, "schemeP_test.npz"))
    X_te = d_test["X"][:, keep_idx].astype(np.float32)
    mp_t_te = d_test["mp_t"].astype(np.float64)
    mp_th_te = d_test[f"mp_t{H}"].astype(np.float64)
    y_cls_te = d_test[f"y{H}"].astype(np.int64)
    y_regr_te = ((mp_th_te - mp_t_te) / (mp_t_te + 1.0)).astype(np.float32)
    sym_te = d_test["sym"].astype(np.int8)
    date_te = d_test["date"].astype(np.int16)
    sess_map = {0: "am", 1: "pm"}
    sess_te = np.array([sess_map[int(s)] for s in d_test["sess_idx"]], dtype=object)
    t_te = d_test["t"].astype(np.int16)
    d_test.close()

    te_pred = model.predict(X_te).astype(np.float32)
    te_corr = float(np.corrcoef(te_pred, y_regr_te)[0, 1])
    te_mse = float(((te_pred - y_regr_te) ** 2).mean())
    print(f"  TEST corr={te_corr:.4f} mse={te_mse:.8f}", flush=True)

    # Save predictions
    pred_df = pd.DataFrame({
        "sym": sym_te,
        "date": date_te,
        "session": sess_te,
        "t": t_te,
        "true_label": y_cls_te.astype(np.int8),
        "true_dmid_norm": y_regr_te.astype(np.float32),
        "pred_dmid_norm": te_pred,
        "midprice_t": mp_t_te.astype(np.float32),
        "midprice_th": mp_th_te.astype(np.float32),
    })
    pred_path = os.path.join(HERE, f"pred_CB_inner_seed{seed}.parquet")
    pred_df.to_parquet(pred_path, index=False)
    print(f"  saved {pred_path} ({len(pred_df):,} rows)", flush=True)

    summary = {
        "task": f"T166 CB L2 inner seed={seed}",
        "seed": seed, "horizon": H,
        "n_inner": n_inner, "n_train_used": n_inner * 2,
        "iterations": args.iterations, "tree_count": int(model.tree_count_),
        "train_time_sec": float(train_time),
        "test_corr": te_corr, "test_mse": te_mse,
        "cb_params": {k: v for k, v in cb_params.items() if k not in ("verbose",)},
    }
    with open(os.path.join(HERE, f"summary_CB_inner_seed{seed}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if use_wandb and run is not None:
        run.log({"train_time_sec": train_time, "tree_count": int(model.tree_count_),
                 "test_corr": te_corr, "test_mse": te_mse})
        run.finish()

    progress("done", seed=seed, train_time_sec=float(train_time), test_corr=te_corr)
    print(f"\n=== seed={seed} DONE in {train_time:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
