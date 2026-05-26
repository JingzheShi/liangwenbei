"""Unified LGB training for the 13-row big ablation.

Supports:
  - Cache choice: raw154 (cache_raw154_rawamt) or schemeP (cache_log1p, 370-d).
  - Objective: regression_l2 or multiclass.
  - Feature subset modes: full | schemeC72 | drop11.
  - Augmentations applied at training time:
      --window-z      : per-feature standardization (mean/std from train set).
      --mirror-flip   : duplicate train set with bid/ask mirrored + label negated.
      --log1p-rawlast : sign(v)*log1p(|v|) on the price/size raw_last cols.
  - --fulltrain      : train on train + val (dates 0-95), iters x 1.1 from val-best.
  - --rounds         : max rounds.
  - --early-stop     : early stopping rounds on val (when not fulltrain).

V4 protocol: train 0-79, val 80-95 (ES, threshold search), test 96-119 (final eval).
"""
from __future__ import annotations
import argparse, os, json, time
import numpy as np
import lightgbm as lgb


SCHEMEP_RAW_NAMES_ORDER = None  # populated when needed
DROP_NAMES = [
    "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
    "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
    "qrank_W100_cumspread",
    "kyle_lam_W50", "kyle_lam_W100",
    "roll_eff_spr_ratio_W100",
    "liq_asym_top5_W5",
]


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


def compute_pnl_ce(p3, mp_t, mp_th, T, delta, fee=1e-4):
    """Threshold gate for 3-class CE: trade if max(p0, p2) >= T and > p1 + delta."""
    p_dn = p3[:, 0]
    p_fl = p3[:, 1]
    p_up = p3[:, 2]
    a = np.zeros(len(p3), dtype=np.float32)
    max_extreme = np.maximum(p_up, p_dn)
    conf = (max_extreme >= T) & (max_extreme > p_fl + delta)
    up_choice = conf & (p_up >= p_dn)
    dn_choice = conf & (p_dn > p_up)
    a[up_choice] = 1.0
    a[dn_choice] = -1.0
    raw = a * (mp_th - mp_t)
    cost = fee * np.abs(a) * ((mp_th + 1) + (mp_t + 1))
    pnl = (raw - cost) / (mp_t + 1.0)
    return pnl.sum(), int((a != 0).sum())


def search_threshold_sym(y_pred, mp_t, mp_th, fee=1e-4):
    mean_abs = float(np.mean(np.abs(y_pred)))
    if mean_abs <= 0:
        mean_abs = 1e-5
    best = (-1e18, None, None)
    for k in np.linspace(0.5, 3.0, 26):
        thr = k * mean_abs
        s, _ = compute_pnl(y_pred, mp_t, mp_th, thr, thr, fee)
        if s > best[0]:
            best = (s, thr, k)
    return best


def search_threshold_ce(p3, mp_t, mp_th, fee=1e-4):
    """Grid search T and delta for high-confidence gate."""
    best = (-1e18, None, None)
    for T in np.linspace(0.40, 0.85, 19):
        for delta in np.linspace(-0.10, 0.30, 9):
            s, _ = compute_pnl_ce(p3, mp_t, mp_th, T, delta, fee)
            if s > best[0]:
                best = (s, float(T), float(delta))
    return best


def search_threshold_asym_2d_de(y_pred, mp_t, mp_th, fee=1e-4, mean_abs=None):
    """Use differential evolution to search 2D asym thresholds (theta_up, theta_dn)."""
    from scipy.optimize import differential_evolution
    if mean_abs is None:
        mean_abs = max(float(np.mean(np.abs(y_pred))), 1e-6)
    lo = 0.2 * mean_abs
    hi = 3.5 * mean_abs

    def neg_pnl(x):
        s, _ = compute_pnl(y_pred, mp_t, mp_th, x[0], x[1], fee)
        return -s

    res = differential_evolution(
        neg_pnl, bounds=[(lo, hi), (lo, hi)],
        maxiter=40, popsize=15, tol=1e-7, seed=1, polish=True,
    )
    s, _ = compute_pnl(y_pred, mp_t, mp_th, res.x[0], res.x[1], fee)
    return float(s), float(res.x[0]), float(res.x[1])


HP_DEFAULT = dict(feature_fraction=0.8, bagging_fraction=0.8,
                  num_leaves=127, lambda_l2=1.0,
                  min_data_in_leaf=100, bagging_freq=5)


def load_cache(cache_dir, prefix, split, H, want_X=True):
    p = os.path.join(cache_dir, f"{prefix}_{split}.npz")
    d = np.load(p)
    out = {
        "mp_t": d["mp_t"].astype(np.float64),
        "mp_th": d[f"mp_t{H}"].astype(np.float64),
        "y_cls": d[f"y{H}"],
    }
    if want_X:
        out["X"] = d["X"].astype(np.float32)
    return out


def get_feat_names(cache_dir, prefix):
    if prefix == "raw154":
        p = os.path.join(cache_dir, "feat_names.txt")
    else:
        p = os.path.join(cache_dir, "schemeP_feat_names.txt")
    with open(p) as f:
        return [line.strip() for line in f if line.strip()]


def get_raw_price_size_idx(names):
    """Return indices into raw_last that are bid/ask/bsize/asize/midprice cols, for log1p magnitude transform."""
    out = []
    for i, n in enumerate(names):
        if n.startswith(("bid", "ask", "bsize", "asize", "midprice", "spread",
                          "bid_diff", "ask_diff", "bsize_rate", "asize_rate",
                          "bid_rate", "ask_rate", "bid_mean", "ask_mean",
                          "bsize_mean", "asize_mean", "totalbsize", "totalasize",
                          "avgbid", "avgask", "cumspread", "amount_delta",
                          "volume_delta")):
            out.append(i)
    return out


def bidask_mirror_indices(names):
    """Map index i -> index j such that mirroring swaps i<->j (bid_k<->ask_k, bsize_k<->asize_k, etc.)."""
    name_to_idx = {n: i for i, n in enumerate(names)}
    swap = {}
    sign_flip = set()  # indices where mirror = negate

    def add_pair(a, b):
        if a in name_to_idx and b in name_to_idx:
            swap[name_to_idx[a]] = name_to_idx[b]
            swap[name_to_idx[b]] = name_to_idx[a]

    for k in range(1, 11):
        add_pair(f"bid{k}", f"ask{k}")
        add_pair(f"bsize{k}", f"asize{k}")
        add_pair(f"bid_diff{k}", f"ask_diff{k}")
        add_pair(f"bid_rate{k}", f"ask_rate{k}")
        add_pair(f"bsize_rate{k}", f"asize_rate{k}")
    add_pair("avgbid", "avgask")
    add_pair("totalbsize", "totalasize")
    add_pair("bid_mean", "ask_mean")
    add_pair("bsize_mean", "asize_mean")
    add_pair("lb_intst", "la_intst")
    add_pair("mb_intst", "ma_intst")
    add_pair("cb_intst", "ca_intst")
    add_pair("lb_ind", "la_ind")
    add_pair("mb_ind", "ma_ind")
    add_pair("cb_ind", "ca_ind")
    add_pair("lb_acc", "la_acc")
    add_pair("mb_acc", "ma_acc")
    add_pair("cb_acc", "ca_acc")
    # imbalance is sign-flipped under mirror
    if "imbalance" in name_to_idx:
        sign_flip.add(name_to_idx["imbalance"])
    return swap, sign_flip


def apply_mirror(X, names):
    """Build mirrored copy of X (deep). Used for augmentation."""
    swap, sign_flip = bidask_mirror_indices(names)
    Xm = X.copy()
    # swap pairs
    seen = set()
    for a, b in swap.items():
        key = tuple(sorted([a, b]))
        if key in seen:
            continue
        seen.add(key)
        Xm[:, a], Xm[:, b] = X[:, b].copy(), X[:, a].copy()
    # sign flip
    for i in sign_flip:
        Xm[:, i] = -X[:, i]
    return Xm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--early-stop", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--objective", default="regression_l2",
                    choices=["regression_l2", "multiclass"])
    ap.add_argument("--feature-mode", default="full",
                    choices=["full", "schemeC72", "drop11"])
    ap.add_argument("--window-z", action="store_true")
    ap.add_argument("--mirror-flip", action="store_true")
    ap.add_argument("--log1p-rawlast", action="store_true")
    ap.add_argument("--fulltrain", action="store_true",
                    help="Train on train+val 0-95; uses rounds_best * 1.1")
    ap.add_argument("--fulltrain-rounds", type=int, default=None,
                    help="Override fulltrain rounds (M7-style fixed; default = best_iter*1.1)")
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-project", default="liangwenbei-ablation-rerun")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--save-preds", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    H = args.horizon

    # detect prefix
    if os.path.exists(os.path.join(args.cache_dir, "raw154_train.npz")):
        prefix = "raw154"
        is_raw154 = True
    else:
        prefix = "schemeP"
        is_raw154 = False
    feat_names = get_feat_names(args.cache_dir, prefix)

    print(f"=== cache={args.cache_dir} prefix={prefix} feat_dim_orig={len(feat_names)} ===", flush=True)

    # subset / drop
    if args.feature_mode == "full":
        keep_idx = np.arange(len(feat_names), dtype=np.int64)
    elif args.feature_mode == "schemeC72":
        # First 154 raw + first 72 extras = 226-d SchemeC
        assert not is_raw154, "schemeC72 only valid for schemeP cache"
        keep_idx = np.arange(154 + 72, dtype=np.int64)
    elif args.feature_mode == "drop11":
        assert not is_raw154, "drop11 only valid for schemeP cache"
        drop_set = set(i for i, n in enumerate(feat_names) if n in DROP_NAMES)
        keep_idx = np.array([i for i in range(len(feat_names)) if i not in drop_set], dtype=np.int64)
    feat_names = [feat_names[i] for i in keep_idx]
    print(f"  feat_dim after subset/drop: {len(feat_names)}", flush=True)

    # load
    print("  loading train/val/test ...", flush=True)
    train = load_cache(args.cache_dir, prefix, "train", H)
    val = load_cache(args.cache_dir, prefix, "val", H)
    test = load_cache(args.cache_dir, prefix, "test", H)
    train["X"] = train["X"][:, keep_idx]
    val["X"] = val["X"][:, keep_idx]
    test["X"] = test["X"][:, keep_idx]
    print(f"  shapes: tr={train['X'].shape} va={val['X'].shape} te={test['X'].shape}", flush=True)

    # log1p on rawlast (price+size cols within raw_last block)
    if args.log1p_rawlast:
        # determine which kept cols are raw_last (first 154 originals if from schemeP)
        if is_raw154:
            raw_last_local_idx = np.arange(len(feat_names))
            raw_last_names = feat_names
        else:
            # raw_last cols are those whose original idx < 154
            orig_keep = keep_idx
            raw_last_local_idx = np.array(
                [li for li, oi in enumerate(orig_keep) if oi < 154], dtype=np.int64
            )
            raw_last_names = [feat_names[i] for i in raw_last_local_idx]
        magn_local = [raw_last_local_idx[i] for i, n in enumerate(raw_last_names)
                      if any(n.startswith(p) for p in ("bid", "ask", "bsize", "asize",
                                                       "amount_delta", "volume_delta",
                                                       "avgbid", "avgask", "totalbsize",
                                                       "totalasize", "midprice", "spread",
                                                       "cumspread"))]
        magn_local = np.array(magn_local, dtype=np.int64)
        for arr in (train["X"], val["X"], test["X"]):
            v = arr[:, magn_local]
            arr[:, magn_local] = np.sign(v) * np.log1p(np.abs(v))
        print(f"  applied log1p to {len(magn_local)} rawlast cols", flush=True)

    # window-z (global per-feature standardization computed on train)
    if args.window_z:
        feat_mean = np.nanmean(train["X"], axis=0).astype(np.float32)
        feat_std = np.nanstd(train["X"], axis=0).astype(np.float32)
        feat_std = np.maximum(feat_std, 1e-6)
        for arr in (train["X"], val["X"], test["X"]):
            arr -= feat_mean
            arr /= feat_std
            np.nan_to_num(arr, copy=False, nan=0.0)
            np.clip(arr, -10.0, 10.0, out=arr)
        print("  applied window-z (global standardization)", flush=True)

    # mirror flip aug (train only)
    if args.mirror_flip:
        # full raw_last + (try to also flip schemeP extras with bid/ask names)
        mirror_names = feat_names
        Xm = apply_mirror(train["X"], mirror_names)
        y_reg_tr_orig = ((train["mp_th"] - train["mp_t"]) / (train["mp_t"] + 1.0)).astype(np.float32)
        y_reg_tr_mir = -y_reg_tr_orig
        y_cls_tr_orig = train["y_cls"].astype(np.int64)
        y_cls_tr_mir = (2 - y_cls_tr_orig).clip(0, 2).astype(np.int64)
        mp_t_mir = train["mp_t"]
        mp_th_mir = train["mp_th"]
        # concat
        train["X"] = np.concatenate([train["X"], Xm], axis=0)
        del Xm
        train["mp_t"] = np.concatenate([train["mp_t"], mp_t_mir], axis=0)
        train["mp_th"] = np.concatenate([train["mp_th"], mp_th_mir], axis=0)
        train["y_cls"] = np.concatenate([train["y_cls"], y_cls_tr_mir.astype(np.int8)], axis=0)
        print(f"  mirror-flip augmented: train rows now {train['X'].shape[0]:,}", flush=True)

    # labels
    y_reg_tr = ((train["mp_th"] - train["mp_t"]) / (train["mp_t"] + 1.0)).astype(np.float32)
    y_reg_val = ((val["mp_th"] - val["mp_t"]) / (val["mp_t"] + 1.0)).astype(np.float32)
    y_reg_test = ((test["mp_th"] - test["mp_t"]) / (test["mp_t"] + 1.0)).astype(np.float32)
    y_cls_tr = train["y_cls"]
    y_cls_val = val["y_cls"]

    sw_tr = make_weights(y_cls_tr, num_class=3)
    sw_val = make_weights(y_cls_val, num_class=3)

    if args.objective == "multiclass":
        params = {"objective": "multiclass", "num_class": 3, "metric": "multi_logloss"}
        ytr = y_cls_tr.astype(np.int64)
        yva = y_cls_val.astype(np.int64)
    else:
        params = {"objective": "regression_l2", "metric": "rmse"}
        ytr = y_reg_tr
        yva = y_reg_val

    params.update({
        "learning_rate": args.lr,
        "num_leaves": int(HP_DEFAULT["num_leaves"]),
        "feature_fraction": float(HP_DEFAULT["feature_fraction"]),
        "bagging_fraction": float(HP_DEFAULT["bagging_fraction"]),
        "bagging_freq": int(HP_DEFAULT["bagging_freq"]),
        "lambda_l2": float(HP_DEFAULT["lambda_l2"]),
        "min_data_in_leaf": int(HP_DEFAULT["min_data_in_leaf"]),
        "num_threads": 18,
        "seed": args.seed,
        "verbose": -1,
    })
    if args.gpu:
        params["device"] = "gpu"
        params["gpu_use_dp"] = False

    # wandb
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            os.environ.setdefault(
                "WANDB_API_KEY",
                "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",
            )
            wandb.init(
                project=args.wandb_project,
                entity="cjxh21-Tsinghua University",
                name=args.wandb_name or os.path.basename(args.out),
                config={**vars(args), "feat_dim": len(feat_names),
                        "n_train": int(train["X"].shape[0]),
                        "n_val": int(val["X"].shape[0])},
                reinit=True,
                settings=wandb.Settings(start_method="thread"),
            )
        except Exception as e:
            print(f"  wandb init failed: {e}", flush=True)
            use_wandb = False

    # ===== Phase A: train on 0-79 with ES on 80-95 =====
    dtrain = lgb.Dataset(train["X"], label=ytr, weight=sw_tr, free_raw_data=False)
    dval = lgb.Dataset(val["X"], label=yva, weight=sw_val, reference=dtrain, free_raw_data=False)
    t0 = time.time()
    booster = lgb.train(
        params, dtrain,
        num_boost_round=args.rounds,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(args.early_stop, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )
    train_time = time.time() - t0
    best_iter = booster.best_iteration or args.rounds
    print(f"  Phase A done: train_time={train_time:.1f}s best_iter={best_iter}", flush=True)

    # ===== Phase B (fulltrain): retrain on 0-95 with iters * 1.1 =====
    fulltrain_time = 0.0
    if args.fulltrain:
        print("  --- Fulltrain (train+val 0-95) ---", flush=True)
        X_full = np.concatenate([train["X"], val["X"]], axis=0)
        if args.objective == "multiclass":
            y_full = np.concatenate([y_cls_tr.astype(np.int64), y_cls_val.astype(np.int64)], axis=0)
            sw_full = make_weights(y_full.astype(np.int64), num_class=3)
        else:
            y_full = np.concatenate([y_reg_tr, y_reg_val], axis=0)
            y_cls_full = np.concatenate([y_cls_tr, y_cls_val], axis=0)
            sw_full = make_weights(y_cls_full, num_class=3)
        dtrain_full = lgb.Dataset(X_full, label=y_full, weight=sw_full, free_raw_data=False)
        if args.fulltrain_rounds is not None:
            rounds_full = int(args.fulltrain_rounds)
        else:
            rounds_full = int(round(best_iter * 1.1))
        t0 = time.time()
        booster_full = lgb.train(
            params, dtrain_full,
            num_boost_round=rounds_full,
            valid_sets=[dtrain_full],
            callbacks=[lgb.log_evaluation(period=50)],
        )
        fulltrain_time = time.time() - t0
        print(f"  Fulltrain done: rounds_full={rounds_full} time={fulltrain_time:.1f}s", flush=True)
        # use fulltrain booster for final test eval
        scoring_booster = booster_full
    else:
        scoring_booster = booster

    # ===== predict =====
    def predict(model, X):
        if args.objective == "multiclass":
            return model.predict(X)
        return model.predict(X).astype(np.float32)

    yp_val = predict(booster, val["X"])  # val uses Phase A booster for threshold search
    yp_test = predict(scoring_booster, test["X"])

    if args.objective == "multiclass":
        # threshold gate (T, delta)
        val_score, T_best, delta_best = search_threshold_ce(
            yp_val, val["mp_t"], val["mp_th"]
        )
        test_score, n_act_test = compute_pnl_ce(
            yp_test, test["mp_t"], test["mp_th"], T_best, delta_best
        )
        # also save the EV-style p_up - p_dn for reuse in ensembles
        yp_val_signed = (yp_val[:, 2] - yp_val[:, 0]).astype(np.float32)
        yp_test_signed = (yp_test[:, 2] - yp_test[:, 0]).astype(np.float32)
        decision_meta = {"T": float(T_best), "delta": float(delta_best),
                          "val_pnl_h60": float(val_score),
                          "test_pnl_h60": float(test_score)}
    else:
        # symmetric EV gate
        val_score, thr_best, k_best = search_threshold_sym(
            yp_val, val["mp_t"], val["mp_th"]
        )
        test_score, n_act_test = compute_pnl(
            yp_test, test["mp_t"], test["mp_th"], thr_best, thr_best
        )
        yp_val_signed = yp_val
        yp_test_signed = yp_test
        decision_meta = {"thr_sym": float(thr_best), "k_sym": float(k_best),
                          "val_pnl_h60": float(val_score),
                          "test_pnl_h60": float(test_score)}

    results = {
        "task_args": vars(args),
        "feat_dim": int(len(feat_names)),
        "best_iter": int(best_iter),
        "train_time_sec": float(train_time),
        "fulltrain_time_sec": float(fulltrain_time),
        "n_train": int(train["X"].shape[0]),
        "n_val": int(val["X"].shape[0]),
        "n_test": int(test["X"].shape[0]),
        "decision": decision_meta,
    }
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2), flush=True)

    booster.save_model(os.path.join(args.out, "model_phaseA.txt"))
    if args.fulltrain:
        scoring_booster.save_model(os.path.join(args.out, "model_fulltrain.txt"))

    if args.save_preds:
        np.savez(os.path.join(args.out, "preds.npz"),
                 yp_val=yp_val_signed, yp_test=yp_test_signed,
                 mp_t_val=val["mp_t"], mp_th_val=val["mp_th"],
                 mp_t_test=test["mp_t"], mp_th_test=test["mp_th"])

    if use_wandb:
        import wandb
        wandb.log({
            "val/h60_pnl": float(decision_meta["val_pnl_h60"]),
            "test/h60_pnl": float(decision_meta["test_pnl_h60"]),
            "best_iter": int(best_iter),
            "train_time_sec": float(train_time),
            "fulltrain_time_sec": float(fulltrain_time),
        })
        wandb.finish()

    print(f"RESULT_LINE: feat_dim={len(feat_names)} val={decision_meta['val_pnl_h60']:.4f} "
          f"test={decision_meta['test_pnl_h60']:.4f} best_iter={best_iter}",
          flush=True)


if __name__ == "__main__":
    main()
