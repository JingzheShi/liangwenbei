"""50 LGB driver: 5 HP_CONFIGS x 10 seeds, V4 walk-forward.

- Cache: cache_log1p (schemeP 370-d, drop11 -> 359-d).
- Features: window-z + mirror-flip + log1p-rawlast (matches row 19 5+5 setting).
- Objective: regression_l2 with class-balanced weights.
- Train 0-79 (rows after split = train cache), val 80-95 ES (max 330 rounds), test 96-119 zero-tune.
- HP cycling: HP_CONFIGS[(seed-1)%5] from project final-submission script.
- GPU on, num_threads=18.

Loads data ONCE; trains 50 LGB sequentially. Saves preds.npz per seed to out_dir/seed{S}/.
"""
from __future__ import annotations
import argparse, os, json, time, gc
import numpy as np
import lightgbm as lgb

FEE = 1e-4
NUM_CLASS = 3
CLIP = 10.0

HP_CONFIGS = [
    dict(feature_fraction=0.8, bagging_fraction=0.8,  num_leaves=127, lambda_l2=1.0),
    dict(feature_fraction=0.6, bagging_fraction=0.7,  num_leaves=127, lambda_l2=1.0),
    dict(feature_fraction=0.7, bagging_fraction=0.85, num_leaves=63,  lambda_l2=2.0),
    dict(feature_fraction=0.5, bagging_fraction=0.6,  num_leaves=255, lambda_l2=0.5),
    dict(feature_fraction=0.4, bagging_fraction=0.5,  num_leaves=127, lambda_l2=3.0),
]

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


def compute_pnl(y_pred, mp_t, mp_th, thr_up, thr_dn, fee=FEE):
    a = np.zeros_like(y_pred)
    a[y_pred > thr_up] = 1.0
    a[y_pred < -thr_dn] = -1.0
    raw = a * (mp_th - mp_t)
    cost = fee * np.abs(a) * ((mp_th + 1) + (mp_t + 1))
    pnl = (raw - cost) / (mp_t + 1.0)
    return pnl.sum(), int((a != 0).sum())


def search_threshold_sym(y_pred, mp_t, mp_th, fee=FEE):
    mean_abs = max(float(np.mean(np.abs(y_pred))), 1e-8)
    best = (-1e18, None, None)
    for k in np.linspace(0.5, 3.0, 26):
        thr = k * mean_abs
        s, _ = compute_pnl(y_pred, mp_t, mp_th, thr, thr, fee)
        if s > best[0]:
            best = (s, thr, k)
    return best


def bidask_mirror_indices(names):
    name_to_idx = {n: i for i, n in enumerate(names)}
    swap = {}
    sign_flip = set()

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
    if "imbalance" in name_to_idx:
        sign_flip.add(name_to_idx["imbalance"])
    return swap, sign_flip


def apply_mirror(X, names):
    swap, sign_flip = bidask_mirror_indices(names)
    Xm = X.copy()
    seen = set()
    for a, b in swap.items():
        key = tuple(sorted([a, b]))
        if key in seen:
            continue
        seen.add(key)
        Xm[:, a], Xm[:, b] = X[:, b].copy(), X[:, a].copy()
    for i in sign_flip:
        Xm[:, i] = -X[:, i]
    return Xm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="ablation_runs/cache_log1p")
    ap.add_argument("--out", default="ablation_runs/big50_lgb")
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--rounds", type=int, default=330)
    ap.add_argument("--early-stop", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--gpu", action="store_true", default=True)
    ap.add_argument("--wandb-project", default="liangwenbei-50plus50-final")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--n-seeds", type=int, default=50)
    ap.add_argument("--start-seed", type=int, default=1)
    args = ap.parse_args()

    H = args.horizon
    os.makedirs(args.out, exist_ok=True)

    # Load features
    feat_path = os.path.join(args.cache_dir, "schemeP_feat_names.txt")
    with open(feat_path) as f:
        feat_names_all = [line.strip() for line in f if line.strip()]
    drop_set = set(i for i, n in enumerate(feat_names_all) if n in DROP_NAMES)
    keep_idx = np.array([i for i in range(len(feat_names_all)) if i not in drop_set], dtype=np.int64)
    feat_names = [feat_names_all[i] for i in keep_idx]
    feat_dim = len(feat_names)
    print(f"feat_dim={feat_dim} (dropped {len(drop_set)})", flush=True)

    # Load splits
    def load_split(split):
        d = np.load(os.path.join(args.cache_dir, f"schemeP_{split}.npz"))
        return {
            "X": d["X"][:, keep_idx].astype(np.float32),
            "mp_t": d["mp_t"].astype(np.float64),
            "mp_th": d[f"mp_t{H}"].astype(np.float64),
            "y_cls": d[f"y{H}"],
        }

    print("loading train/val/test...", flush=True)
    t0 = time.time()
    train = load_split("train")
    val = load_split("val")
    test = load_split("test")
    print(f"  shapes tr={train['X'].shape} va={val['X'].shape} te={test['X'].shape} "
          f"({time.time()-t0:.1f}s)", flush=True)

    # log1p rawlast (first 154 originals -> intersect keep_idx)
    rl_local = np.array([li for li, oi in enumerate(keep_idx) if oi < 154], dtype=np.int64)
    rl_names = [feat_names_all[oi] for oi in keep_idx if oi < 154]
    magn_prefixes = ("bid", "ask", "bsize", "asize", "amount_delta", "volume_delta",
                     "avgbid", "avgask", "totalbsize", "totalasize", "midprice",
                     "spread", "cumspread")
    magn_local = np.array(
        [rl_local[i] for i, n in enumerate(rl_names)
         if any(n.startswith(p) for p in magn_prefixes)], dtype=np.int64)
    for arr in (train["X"], val["X"], test["X"]):
        v = arr[:, magn_local]
        arr[:, magn_local] = np.sign(v) * np.log1p(np.abs(v))
    print(f"  log1p applied to {len(magn_local)} cols", flush=True)

    # window-z
    feat_mean = np.nanmean(train["X"], axis=0).astype(np.float32)
    feat_std = np.maximum(np.nanstd(train["X"], axis=0).astype(np.float32), 1e-6)
    for arr in (train["X"], val["X"], test["X"]):
        arr -= feat_mean
        arr /= feat_std
        np.nan_to_num(arr, copy=False, nan=0.0)
        np.clip(arr, -CLIP, CLIP, out=arr)
    print("  window-z applied", flush=True)

    # Mirror-flip aug on train
    Xm = apply_mirror(train["X"], feat_names)
    y_reg_tr_orig = ((train["mp_th"] - train["mp_t"]) / (train["mp_t"] + 1.0)).astype(np.float32)
    y_cls_tr_orig = train["y_cls"].astype(np.int64)
    y_cls_tr_mir = (2 - y_cls_tr_orig).clip(0, 2).astype(np.int64)

    Xtr_full = np.concatenate([train["X"], Xm], axis=0)
    del Xm
    y_reg_tr_full = np.concatenate([y_reg_tr_orig, -y_reg_tr_orig], axis=0)
    y_cls_tr_full = np.concatenate([y_cls_tr_orig.astype(np.int8), y_cls_tr_mir.astype(np.int8)], axis=0)
    print(f"  mirror-flip aug: train rows {Xtr_full.shape[0]:,}", flush=True)

    y_reg_val = ((val["mp_th"] - val["mp_t"]) / (val["mp_t"] + 1.0)).astype(np.float32)
    sw_tr = make_weights(y_cls_tr_full, num_class=3)
    sw_val = make_weights(val["y_cls"], num_class=3)

    # WandB top-level
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            os.environ.setdefault(
                "WANDB_API_KEY",
                "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",
            )
        except Exception as e:
            print(f"wandb import fail: {e}")
            use_wandb = False

    # Loop seeds
    summary = []
    for S in range(args.start_seed, args.start_seed + args.n_seeds):
        hp = HP_CONFIGS[(S - 1) % 5]
        seed_dir = os.path.join(args.out, f"seed{S}")
        os.makedirs(seed_dir, exist_ok=True)
        preds_path = os.path.join(seed_dir, "preds.npz")
        if os.path.isfile(preds_path):
            print(f"[seed {S}] SKIP (preds.npz exists)", flush=True)
            continue
        print(f"\n=== LGB seed={S} hp_group={(S-1)%5} {hp} ===", flush=True)

        params = {
            "objective": "regression_l2", "metric": "rmse",
            "learning_rate": args.lr,
            "num_leaves": int(hp["num_leaves"]),
            "feature_fraction": float(hp["feature_fraction"]),
            "bagging_fraction": float(hp["bagging_fraction"]),
            "bagging_freq": 5,
            "lambda_l2": float(hp["lambda_l2"]),
            "min_data_in_leaf": 100,
            "num_threads": 18,
            "seed": S,
            "verbose": -1,
        }
        if args.gpu:
            params["device"] = "gpu"
            params["gpu_use_dp"] = False

        if use_wandb:
            try:
                import wandb
                wandb.init(project=args.wandb_project, entity="cjxh21-Tsinghua University",
                           name=f"lgb_seed{S}",
                           group="50_lgb",
                           config={**params, "hp_group": (S-1)%5, "seed": S,
                                   "feat_dim": feat_dim, "n_train": int(Xtr_full.shape[0]),
                                   "n_val": int(val["X"].shape[0])},
                           reinit=True, settings=wandb.Settings(start_method="thread"))
            except Exception as e:
                print(f"  wandb init fail: {e}")

        dtrain = lgb.Dataset(Xtr_full, label=y_reg_tr_full, weight=sw_tr, free_raw_data=False)
        dval = lgb.Dataset(val["X"], label=y_reg_val, weight=sw_val, reference=dtrain, free_raw_data=False)
        t0 = time.time()
        booster = lgb.train(
            params, dtrain,
            num_boost_round=args.rounds,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            callbacks=[lgb.early_stopping(args.early_stop, verbose=False),
                        lgb.log_evaluation(period=100)],
        )
        train_time = time.time() - t0
        best_iter = booster.best_iteration or args.rounds

        yp_val = booster.predict(val["X"]).astype(np.float32)
        yp_test = booster.predict(test["X"]).astype(np.float32)

        val_pnl, thr, k = search_threshold_sym(yp_val, val["mp_t"], val["mp_th"])
        test_pnl, n_act = compute_pnl(yp_test, test["mp_t"], test["mp_th"], thr, thr)
        print(f"  seed{S}: time={train_time:.1f}s best_iter={best_iter} val={val_pnl:.4f} test={test_pnl:.4f}",
              flush=True)

        np.savez(preds_path,
                 yp_val=yp_val, yp_test=yp_test,
                 mp_t_val=val["mp_t"], mp_th_val=val["mp_th"],
                 mp_t_test=test["mp_t"], mp_th_test=test["mp_th"])
        booster.save_model(os.path.join(seed_dir, "model.txt"))
        result = {"seed": S, "hp_group": (S - 1) % 5, "hp": hp,
                  "best_iter": int(best_iter), "train_time_sec": float(train_time),
                  "val_pnl_h60": float(val_pnl), "test_pnl_h60": float(test_pnl),
                  "thr_sym": float(thr), "k_sym": float(k), "feat_dim": feat_dim}
        with open(os.path.join(seed_dir, "results.json"), "w") as f:
            json.dump(result, f, indent=2)
        summary.append(result)

        if use_wandb:
            try:
                import wandb
                wandb.log({"val/h60_pnl": float(val_pnl), "test/h60_pnl": float(test_pnl),
                           "best_iter": int(best_iter), "train_time_sec": float(train_time)})
                wandb.finish()
            except Exception:
                pass

        del booster, dtrain, dval, yp_val, yp_test
        gc.collect()

    sum_path = os.path.join(args.out, "summary_50lgb.json")
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nALL DONE  -> {sum_path}", flush=True)
    print(f"avg val_pnl = {np.mean([s['val_pnl_h60'] for s in summary]):.4f}",
          flush=True)
    print(f"avg test_pnl = {np.mean([s['test_pnl_h60'] for s in summary]):.4f}",
          flush=True)


if __name__ == "__main__":
    main()
