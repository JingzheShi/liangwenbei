"""H-Horizon Study: Train LGB L2 regression for h in {5,10,20,40,60} x 5 seeds.

For each (h, seed):
  - train on date 0-79 (schemeP, 359-d after KS-drop 11)
  - predict on train (subsample), val (80-95), test (96-119)
  - save predictions

Predictions saved to ./preds/pred_h{H}_seed{S}.npz with keys:
  yhat_train, y_train, sym_train, idx_train (subsample)
  yhat_val,   y_val,   sym_val,   mp_t_val,   mp_th_val
  yhat_test,  y_test,  sym_test,  mp_t_test,  mp_th_test

GPU LightGBM used by default.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb

HERE = Path(__file__).parent
CACHE_DIR = Path("/root/projects/liangwenbei_workdir/ablation_runs/cache_log1p")
PRED_DIR = HERE / "preds"
MODEL_DIR = HERE / "models"
PRED_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR.mkdir(exist_ok=True, parents=True)

HORIZONS = [5, 10, 20, 40, 60]
SEEDS = [1, 2, 3, 4, 5]

# Same KS-drop 11 as main pipeline; ensures features identical to final submission code
DROP_NAMES = [
    "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
    "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
    "qrank_W100_cumspread",
    "kyle_lam_W50", "kyle_lam_W100",
    "roll_eff_spr_ratio_W100",
    "liq_asym_top5_W5",
]

# M7 HP config rotation (matches final_submission_code/02_train_lgb)
HP_CONFIGS = [
    dict(feature_fraction=0.8, bagging_fraction=0.8,  num_leaves=127, lambda_l2=1.0),
    dict(feature_fraction=0.6, bagging_fraction=0.7,  num_leaves=127, lambda_l2=1.0),
    dict(feature_fraction=0.7, bagging_fraction=0.85, num_leaves=63,  lambda_l2=2.0),
    dict(feature_fraction=0.5, bagging_fraction=0.6,  num_leaves=255, lambda_l2=0.5),
    dict(feature_fraction=0.4, bagging_fraction=0.5,  num_leaves=127, lambda_l2=3.0),
]

NUM_BOOST = 330
LR = 0.05
TRAIN_IC_SUBSAMPLE = 300_000  # how many train rows to keep for IC eval


def load_keep_idx():
    feat_names_path = CACHE_DIR / "schemeP_feat_names.txt"
    all_feat = [l.strip() for l in feat_names_path.read_text().splitlines() if l.strip()]
    name_to_idx = {n: i for i, n in enumerate(all_feat)}
    drop_idx = set(name_to_idx[n] for n in DROP_NAMES if n in name_to_idx)
    keep_idx = np.array([i for i in range(len(all_feat)) if i not in drop_idx], dtype=np.int64)
    feat_names = [all_feat[i] for i in keep_idx]
    return keep_idx, feat_names


def load_split(split: str, keep_idx: np.ndarray):
    d = np.load(CACHE_DIR / f"schemeP_{split}.npz")
    X = d["X"][:, keep_idx].astype(np.float32)
    mp_t = d["mp_t"].astype(np.float64)
    sym = d["sym"].astype(np.int8)
    date = d["date"].astype(np.int16)
    t = d["t"].astype(np.int16)
    sess_idx = d["sess_idx"].astype(np.int8)
    mp_h = {h: d[f"mp_t{h}"].astype(np.float64) for h in HORIZONS}
    d.close()
    return dict(X=X, mp_t=mp_t, sym=sym, date=date, t=t, sess_idx=sess_idx, mp_h=mp_h)


def y_regression(mp_t, mp_th):
    # Match training convention used in final_submission_code/02_train_lgb
    # mp_t was log1p'd, so the +1 denominator approximates the actual price level.
    return ((mp_th - mp_t) / (mp_t + 1.0)).astype(np.float32)


def train_one(X_tr, y_tr, X_val, y_val, sw_tr, hp: dict, seed: int, use_gpu: bool):
    params = {
        "objective": "regression_l2",
        "metric": "rmse",
        "learning_rate": LR,
        "num_leaves": int(hp["num_leaves"]),
        "feature_fraction": float(hp["feature_fraction"]),
        "bagging_fraction": float(hp["bagging_fraction"]),
        "bagging_freq": 5,
        "lambda_l2": float(hp["lambda_l2"]),
        "min_data_in_leaf": 100,
        "num_threads": 18,
        "seed": seed,
        "verbose": -1,
    }
    if use_gpu:
        params["device"] = "gpu"
        params["gpu_use_dp"] = False

    # Fixed-budget training — no early stopping — to keep total capacity identical
    # across horizons (else short-h models would always stop early on noisier targets).
    dtr = lgb.Dataset(X_tr, label=y_tr, weight=sw_tr, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtr, free_raw_data=False)
    booster = lgb.train(
        params, dtr,
        num_boost_round=NUM_BOOST,
        valid_sets=[dval],
        callbacks=[lgb.log_evaluation(period=0)],
    )
    return booster, NUM_BOOST


def class_balanced_weight_from_y_regr(y: np.ndarray, n_bins: int = 3) -> np.ndarray:
    # Approximate the original class-balanced weighting: 3-bin discretization
    q = np.quantile(np.abs(y), 0.66)
    cls = np.where(y > q, 2, np.where(y < -q, 0, 1)).astype(np.int64)
    counts = np.bincount(cls, minlength=n_bins).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    cw = len(y) / (n_bins * counts)
    return cw[cls].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true", default=True)
    ap.add_argument("--horizons", type=int, nargs="+", default=HORIZONS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    args = ap.parse_args()

    try:
        import wandb
        wandb_avail = True
    except Exception:
        wandb_avail = False

    t0 = time.time()
    print("[load] features ...", flush=True)
    keep_idx, feat_names = load_keep_idx()
    print(f"  feat_dim={len(keep_idx)}", flush=True)

    print("[load] train split ...", flush=True)
    Tr = load_split("train", keep_idx)
    print(f"  X_train: {Tr['X'].shape}", flush=True)
    print("[load] val split ...", flush=True)
    Vl = load_split("val", keep_idx)
    print(f"  X_val:   {Vl['X'].shape}", flush=True)
    print("[load] test split ...", flush=True)
    Te = load_split("test", keep_idx)
    print(f"  X_test:  {Te['X'].shape}", flush=True)
    print(f"  load done in {time.time()-t0:.1f}s", flush=True)

    # Subsample train rows for IC eval (full train * 5 seeds is wasteful)
    rng_sub = np.random.default_rng(2026)
    ntr = len(Tr["mp_t"])
    sub_size = min(TRAIN_IC_SUBSAMPLE, ntr)
    train_idx = rng_sub.choice(ntr, size=sub_size, replace=False)
    train_idx.sort()
    X_train_sub = Tr["X"][train_idx]
    sym_train_sub = Tr["sym"][train_idx]
    date_train_sub = Tr["date"][train_idx]
    print(f"[load] train-IC subsample: {sub_size}/{ntr}", flush=True)

    # WandB single run that logs everything
    run = None
    if wandb_avail:
        for entity in ["cjxh21-Tsinghua University", None]:
            try:
                kwargs = dict(
                    project="liangwenbei-horizon-study",
                    name="h_horizon_deep_study",
                    config={
                        "horizons": list(args.horizons),
                        "seeds": list(args.seeds),
                        "feat_dim": len(keep_idx),
                        "num_boost_round": NUM_BOOST,
                        "learning_rate": LR,
                    },
                )
                if entity:
                    kwargs["entity"] = entity
                run = wandb.init(**kwargs)
                print(f"  wandb run initialized (entity={entity})", flush=True)
                break
            except Exception as e:
                print(f"  wandb init failed (entity={entity}): {e}", flush=True)
                run = None

    summary = []
    for H in args.horizons:
        y_tr_full = y_regression(Tr["mp_t"], Tr["mp_h"][H])
        y_val = y_regression(Vl["mp_t"], Vl["mp_h"][H])
        y_test = y_regression(Te["mp_t"], Te["mp_h"][H])
        sw_tr_full = class_balanced_weight_from_y_regr(y_tr_full)
        y_train_sub = y_tr_full[train_idx]
        print(f"\n=== H={H} ===  std(y_tr)={y_tr_full.std():.4e}  std(y_val)={y_val.std():.4e}  std(y_test)={y_test.std():.4e}", flush=True)

        for S in args.seeds:
            hp = HP_CONFIGS[(S - 1) % len(HP_CONFIGS)]
            pred_path = PRED_DIR / f"pred_h{H}_seed{S}.npz"
            if args.skip_existing and pred_path.exists():
                print(f"  [skip] H={H} S={S} already done", flush=True)
                continue
            t_s = time.time()
            booster, best_iter = train_one(
                Tr["X"], y_tr_full, Vl["X"], y_val, sw_tr_full,
                hp=hp, seed=S, use_gpu=args.gpu,
            )
            tr_time = time.time() - t_s
            print(f"  [H={H} S={S}] best_iter={best_iter}  train_time={tr_time:.1f}s", flush=True)

            yhat_train_sub = booster.predict(X_train_sub, num_iteration=best_iter)
            yhat_val = booster.predict(Vl["X"], num_iteration=best_iter)
            yhat_test = booster.predict(Te["X"], num_iteration=best_iter)

            np.savez_compressed(
                pred_path,
                yhat_train=yhat_train_sub.astype(np.float32),
                y_train=y_train_sub.astype(np.float32),
                sym_train=sym_train_sub,
                date_train=date_train_sub,
                yhat_val=yhat_val.astype(np.float32),
                y_val=y_val.astype(np.float32),
                sym_val=Vl["sym"],
                date_val=Vl["date"],
                mp_t_val=Vl["mp_t"].astype(np.float64),
                mp_th_val=Vl["mp_h"][H].astype(np.float64),
                yhat_test=yhat_test.astype(np.float32),
                y_test=y_test.astype(np.float32),
                sym_test=Te["sym"],
                date_test=Te["date"],
                mp_t_test=Te["mp_t"].astype(np.float64),
                mp_th_test=Te["mp_h"][H].astype(np.float64),
                horizon=np.int32(H),
                seed=np.int32(S),
                best_iter=np.int32(best_iter),
                train_time=np.float32(tr_time),
            )

            # Skip saving models — predictions are enough for analysis (saves ~10MB/model)

            from scipy.stats import spearmanr, pearsonr
            ic_train = spearmanr(yhat_train_sub, y_train_sub)[0]
            ic_val = spearmanr(yhat_val, y_val)[0]
            ic_test = spearmanr(yhat_test, y_test)[0]
            r_test = pearsonr(yhat_test, y_test)[0]
            print(f"     IC(tr)={ic_train:+.4f}  IC(val)={ic_val:+.4f}  IC(test)={ic_test:+.4f}  r(test)={r_test:+.4f}", flush=True)

            summary.append({
                "horizon": H, "seed": S, "best_iter": int(best_iter),
                "train_time": tr_time,
                "ic_train": float(ic_train),
                "ic_val": float(ic_val),
                "ic_test": float(ic_test),
                "r_test": float(r_test),
            })
            if run is not None:
                wandb.log({
                    f"H{H}/seed{S}/ic_train": ic_train,
                    f"H{H}/seed{S}/ic_val": ic_val,
                    f"H{H}/seed{S}/ic_test": ic_test,
                    f"H{H}/seed{S}/r_test": r_test,
                    f"H{H}/seed{S}/best_iter": best_iter,
                    f"H{H}/seed{S}/train_time": tr_time,
                })
            del booster
            gc.collect()

        gc.collect()

    json.dump(summary, open(HERE / "train_summary.json", "w"), indent=2)
    print(f"\n[done] total wallclock {time.time()-t0:.1f}s. summary -> train_summary.json", flush=True)
    if run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
