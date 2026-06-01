"""Train one LGB L2 model on 154-d raw-last features and evaluate h=60 PnL.

Used for P1 amount_delta ablation:
  - raw154_rawamt (amount_delta unchanged)
  - raw154_log1p  (sign(v)*log1p(|v|) applied)

Training: dates 0-79
Evaluation: dates 80-95 (V4 walk-forward holdout)
Reports h=60 cumulative PnL after EV gate decision.
"""
from __future__ import annotations
import argparse, os, json, time
import numpy as np
import lightgbm as lgb


def make_weights(y_cls, num_class=3):
    counts = np.bincount(y_cls.astype(np.int64), minlength=num_class).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    cw = len(y_cls) / (num_class * counts)
    return cw[y_cls.astype(np.int64)].astype(np.float32)


def compute_pnl(y_pred, mp_t, mp_th, thr_up, thr_dn, fee=1e-4):
    a = np.zeros_like(y_pred)
    a[y_pred > thr_up] = 1.0
    a[y_pred < -thr_dn] = -1.0
    raw = a * (mp_th - mp_t)
    cost = fee * np.abs(a) * ((mp_th + 1) + (mp_t + 1))
    pnl = (raw - cost) / (mp_t + 1.0)
    return pnl.sum(), int((a != 0).sum())


def search_threshold_2d(y_pred, mp_t, mp_th, fee=1e-4, lo=0.5e-4, hi=5e-4, n=21):
    grid = np.linspace(lo, hi, n)
    best = (-1e18, None, None)
    for up in grid:
        for dn in grid:
            s, _ = compute_pnl(y_pred, mp_t, mp_th, up, dn, fee)
            if s > best[0]:
                best = (s, up, dn)
    return best


def search_threshold_sym(y_pred, mp_t, mp_th, fee=1e-4):
    mean_abs = float(np.mean(np.abs(y_pred)))
    if mean_abs <= 0:
        mean_abs = 1e-5
    best = (-1e18, None, None)
    for k in np.linspace(0.2, 3.0, 29):
        thr = k * mean_abs
        s, _ = compute_pnl(y_pred, mp_t, mp_th, thr, thr, fee)
        if s > best[0]:
            best = (s, thr, k)
    return best


HP = dict(feature_fraction=0.8, bagging_fraction=0.8, num_leaves=127, lambda_l2=1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--num-rounds", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--objective", default="regression_l2",
                    choices=["regression_l2", "multiclass"])
    ap.add_argument("--wandb-run-name", default=None)
    ap.add_argument("--wandb-project", default="liangwenbei-ablation-rerun")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--train-extra-test", action="store_true",
                    help="Train on 0-79 + 96-119 (skip val 80-95 to use as holdout)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    H = args.horizon
    rng = np.random.default_rng(args.seed * 7919 + 42)

    # Detect prefix
    if os.path.exists(os.path.join(args.cache_dir, "raw154_train.npz")):
        prefix = "raw154"
    else:
        prefix = "schemeP"

    print(f"=== load {prefix} from {args.cache_dir} ===", flush=True)
    train_d = np.load(os.path.join(args.cache_dir, f"{prefix}_train.npz"))
    val_d = np.load(os.path.join(args.cache_dir, f"{prefix}_val.npz"))
    test_d = np.load(os.path.join(args.cache_dir, f"{prefix}_test.npz"))

    X_tr = train_d["X"].astype(np.float32)
    mp_t_tr = train_d["mp_t"].astype(np.float64)
    mp_th_tr = train_d[f"mp_t{H}"].astype(np.float64)
    y_cls_tr = train_d[f"y{H}"]
    y_reg_tr = ((mp_th_tr - mp_t_tr) / (mp_t_tr + 1.0)).astype(np.float32)
    print(f"  train X={X_tr.shape}", flush=True)

    X_val = val_d["X"].astype(np.float32)
    mp_t_val = val_d["mp_t"].astype(np.float64)
    mp_th_val = val_d[f"mp_t{H}"].astype(np.float64)
    y_cls_val = val_d[f"y{H}"]
    y_reg_val = ((mp_th_val - mp_t_val) / (mp_t_val + 1.0)).astype(np.float32)
    print(f"  val X={X_val.shape}", flush=True)

    X_test = test_d["X"].astype(np.float32)
    mp_t_test = test_d["mp_t"].astype(np.float64)
    mp_th_test = test_d[f"mp_t{H}"].astype(np.float64)
    y_cls_test = test_d[f"y{H}"]
    y_reg_test = ((mp_th_test - mp_t_test) / (mp_t_test + 1.0)).astype(np.float32)
    print(f"  test X={X_test.shape}", flush=True)

    if args.train_extra_test:
        X_full = np.concatenate([X_tr, X_test], axis=0)
        y_reg_full = np.concatenate([y_reg_tr, y_reg_test], axis=0)
        y_cls_full = np.concatenate([y_cls_tr, y_cls_test], axis=0)
        print(f"  TRAIN-EXTRA-TEST: combined X={X_full.shape}", flush=True)
        X_tr, y_reg_tr, y_cls_tr = X_full, y_reg_full, y_cls_full

    sw = make_weights(y_cls_tr, num_class=3)
    if args.objective == "multiclass":
        params = {
            "objective": "multiclass", "num_class": 3,
            "metric": "multi_logloss",
        }
        ytr = y_cls_tr.astype(np.int64)
    else:
        params = {"objective": "regression_l2", "metric": "rmse"}
        ytr = y_reg_tr

    params.update({
        "learning_rate": args.lr,
        "num_leaves": int(HP["num_leaves"]),
        "feature_fraction": float(HP["feature_fraction"]),
        "bagging_fraction": float(HP["bagging_fraction"]),
        "bagging_freq": 5,
        "lambda_l2": float(HP["lambda_l2"]),
        "min_data_in_leaf": 100,
        "num_threads": 18,
        "seed": args.seed,
        "verbose": -1,
    })
    if args.gpu:
        params["device"] = "gpu"
        params["gpu_use_dp"] = False
    print(f"  params={ {k:params[k] for k in ['objective','learning_rate','num_leaves','feature_fraction','bagging_fraction','lambda_l2']} }", flush=True)

    dtrain = lgb.Dataset(X_tr, label=ytr, weight=sw, free_raw_data=False)

    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            os.environ.setdefault("WANDB_API_KEY", "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")
            wandb.init(
                project=args.wandb_project,
                entity="cjxh21-Tsinghua University",
                name=args.wandb_run_name or os.path.basename(args.out),
                config={
                    "feat_dim": X_tr.shape[1], "horizon": H, "seed": args.seed,
                    "objective": args.objective, "lr": args.lr, "rounds": args.num_rounds,
                    "cache": args.cache_dir, "train_extra_test": args.train_extra_test,
                    "n_train": X_tr.shape[0], "n_val": X_val.shape[0],
                    "hp": HP,
                },
                reinit=True, settings=wandb.Settings(start_method="thread"),
            )
        except Exception as e:
            print(f"  wandb init failed: {e}", flush=True)
            use_wandb = False

    t0 = time.time()
    booster = lgb.train(
        params, dtrain,
        num_boost_round=args.num_rounds,
        valid_sets=[dtrain],
        callbacks=[lgb.log_evaluation(period=50)],
    )
    train_time = time.time() - t0
    print(f"  train_time={train_time:.1f}s", flush=True)

    def predict_yreg(X):
        if args.objective == "multiclass":
            p = booster.predict(X)  # (N, 3) for cls 0,1,2 = down,flat,up
            yr = p[:, 2] - p[:, 0]  # EV-like sign: P(up) - P(down)
            return yr.astype(np.float32)
        return booster.predict(X).astype(np.float32)

    yp_val = predict_yreg(X_val)
    yp_test = predict_yreg(X_test)

    # search both symmetric and asymmetric thresholds on val for h=60 PnL
    sym_score_val, sym_thr_val, k_val = search_threshold_sym(yp_val, mp_t_val, mp_th_val)
    asym_score_val, up_val, dn_val = search_threshold_2d(yp_val, mp_t_val, mp_th_val)

    # apply val-tuned asym to test
    asym_score_test, _ = compute_pnl(yp_test, mp_t_test, mp_th_test, up_val, dn_val)
    sym_score_test, _ = compute_pnl(yp_test, mp_t_test, mp_th_test, sym_thr_val, sym_thr_val)

    # all-horizon PnL on val
    pnl_all_h = {}
    for h2 in (5, 10, 20, 40, 60):
        mp_th_h = val_d[f"mp_t{h2}"].astype(np.float64)
        scale = (h2 / 60.0) ** 0.5
        s, _ = compute_pnl(yp_val, mp_t_val, mp_th_h, up_val * scale, dn_val * scale)
        pnl_all_h[h2] = float(s)

    results = {
        "cache_dir": args.cache_dir,
        "objective": args.objective,
        "horizon": H,
        "seed": args.seed,
        "n_train_used": int(X_tr.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_test": int(X_test.shape[0]),
        "rounds": args.num_rounds,
        "train_time_sec": float(train_time),
        "val_h60_pnl_sym": float(sym_score_val),
        "val_h60_thr_sym": float(sym_thr_val),
        "val_h60_k_sym": float(k_val),
        "val_h60_pnl_asym": float(asym_score_val),
        "val_h60_thr_up": float(up_val),
        "val_h60_thr_dn": float(dn_val),
        "test_h60_pnl_asym_at_val_thr": float(asym_score_test),
        "test_h60_pnl_sym_at_val_thr": float(sym_score_test),
        "val_pnl_by_horizon": pnl_all_h,
    }
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2), flush=True)
    booster.save_model(os.path.join(args.out, "model.txt"))
    np.savez(os.path.join(args.out, "preds.npz"),
             yp_val=yp_val, yp_test=yp_test,
             mp_t_val=mp_t_val, mp_th_val=mp_th_val,
             mp_t_test=mp_t_test, mp_th_test=mp_th_test,
             y_cls_val=y_cls_val, y_cls_test=y_cls_test)

    if use_wandb:
        import wandb
        wandb.log({
            "val/h60_pnl_sym": sym_score_val,
            "val/h60_thr_sym": sym_thr_val,
            "val/h60_k_sym": k_val,
            "val/h60_pnl_asym": asym_score_val,
            "val/h60_thr_up": up_val,
            "val/h60_thr_dn": dn_val,
            "test/h60_pnl_asym": asym_score_test,
            "test/h60_pnl_sym": sym_score_test,
            "train_time_sec": train_time,
            **{f"val/h{h2}_pnl": v for h2, v in pnl_all_h.items()},
        })
        wandb.finish()

    print(f"RESULT_LINE: val_h60_asym={asym_score_val:.4f} val_h60_sym={sym_score_val:.4f} "
          f"test_h60_asym={asym_score_test:.4f}", flush=True)


if __name__ == "__main__":
    main()
