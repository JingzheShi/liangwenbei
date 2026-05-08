"""T142: LGB regression_l2 with Hawkes features — 5-seed LOSO CV + full-data retrain.

Mode A (--mode cv):
  5-fold LOSO: for each sym s, train on sym != s (all dates), predict on sym == s.
  No early stopping. Fixed num_boost_round=330 (M7 paradigm).
  Saves loso_pred_seed{S}.parquet (OOF preds on training data).

Mode B (--mode full_retrain):
  Train on ALL data (all dates 0-79, all syms), 330 rounds.
  Saves model_h60_seed{S}_hawkes_full.txt.

CRITICAL CONSTRAINTS (hard):
  - date never used as feature
  - sym never used as feature
  - Predictor stateless
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CACHE = os.path.join(ROOT, "experiments", "T68_stage5_features", "cache")
HAWKES_DIR = os.path.join(ROOT, "experiments", "R_Hawkes_OFI")

FEE = 0.0001
SYMS = (0, 1, 2, 3, 4)
NUM_BOOST_ROUND = 330

SEED_CONFIGS = {
    1:   dict(ff=0.6, bf=0.70, leaves=127, l2=1.0),
    7:   dict(ff=0.7, bf=0.85, leaves=63,  l2=2.0),
    13:  dict(ff=0.5, bf=0.60, leaves=255, l2=0.5),
    42:  dict(ff=0.8, bf=0.80, leaves=127, l2=1.0),
    100: dict(ff=0.4, bf=0.50, leaves=127, l2=3.0),
}

T59_FAIL = [
    "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
    "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
    "qrank_W100_cumspread", "kyle_lam_W50", "kyle_lam_W100",
    "roll_eff_spr_ratio_W100",
]
STAGE5_FAIL = ["liq_asym_top5_W5"]
DROP_NAMES = set(T59_FAIL + STAGE5_FAIL)


def write_progress(step, **extra):
    p = os.path.join(HERE, "worker-progress.json")
    payload = {
        "status": "running",
        "step": step,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **extra,
    }
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[progress] {step}", flush=True)


def regr_target(mp_t, mp_th):
    return ((mp_th.astype(np.float64) - mp_t.astype(np.float64))
            / (mp_t.astype(np.float64) + 1.0)).astype(np.float32)


def predict_chunked(booster, X, batch=200_000):
    chunks = []
    for s in range(0, len(X), batch):
        chunks.append(booster.predict(X[s:s + batch]).astype(np.float32))
    return np.concatenate(chunks)


def class_balanced_weight(y, num_class=3):
    counts = np.bincount(y.astype(np.int64), minlength=num_class).astype(np.float64)
    inv = 1.0 / np.maximum(counts, 1.0)
    inv = inv / inv.sum() * num_class
    return inv[y.astype(np.int64)].astype(np.float32)


def aug_a_scale(X, rng, lo=0.8, hi=1.2):
    scale = rng.uniform(lo, hi, size=(1, X.shape[1])).astype(np.float32)
    return X * scale


def vectorized_pnl(action, mp_t, mp_th):
    side = action.astype(np.float64) - 1.0
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee_pnl) / denom


def gate_2d(p, tu, td):
    a = np.full(len(p), 1, dtype=np.int8)
    a[p > tu] = 2
    a[p < -td] = 0
    return a


def compute_loso_pnl_fast(pred, sym, mp_t, mp_th):
    """Efficient LOSO PnL optimization via separable 1D sorted search.

    Since PnL(tu, td) = PnL_long(tu) + PnL_short(td) and these are separable
    (buy threshold doesn't affect sell PnL), we can independently find optimal
    tu and td via sorted cumulative-sum in O(n log n).

    Returns (best_tu, best_td), total_pnl, per_sym_dict.
    """
    diff = (mp_th - mp_t).astype(np.float64)
    denom = (mp_t.astype(np.float64) + 1.0)
    spread_cost = FEE * np.abs((mp_th.astype(np.float64) + 1.0) + (mp_t.astype(np.float64) + 1.0))

    # Individual long/short PnL per sample (before threshold)
    pnl_long = (diff - spread_cost) / denom     # PnL if bought
    pnl_short = (-diff - spread_cost) / denom   # PnL if sold

    # For long side: consider only samples with pred > 0 (possible buys)
    # Sorted by pred descending — as tu decreases, we include more buys
    buy_mask = pred > 0
    sell_mask = pred < 0

    # Long: best tu from sorted cumsum of pnl_long for positive-pred samples
    if buy_mask.any():
        bpred = pred[buy_mask]
        bpnl = pnl_long[buy_mask]
        # Sort by pred descending
        order = np.argsort(-bpred)
        bpred_sorted = bpred[order]
        bpnl_sorted = bpnl[order]
        cumsum_long = np.cumsum(bpnl_sorted)
        best_long_idx = int(np.argmax(cumsum_long))
        best_long_pnl = float(cumsum_long[best_long_idx])
        best_tu = float(bpred_sorted[best_long_idx])
        if best_long_pnl < 0:
            best_long_pnl = 0.0
            best_tu = float("inf")  # no buys
    else:
        best_long_pnl = 0.0
        best_tu = float("inf")

    # Short: best td from sorted cumsum of pnl_short for negative-pred samples
    if sell_mask.any():
        spred = -pred[sell_mask]  # positive values
        spnl = pnl_short[sell_mask]
        order = np.argsort(-spred)
        spred_sorted = spred[order]
        spnl_sorted = spnl[order]
        cumsum_short = np.cumsum(spnl_sorted)
        best_short_idx = int(np.argmax(cumsum_short))
        best_short_pnl = float(cumsum_short[best_short_idx])
        best_td = float(spred_sorted[best_short_idx])
        if best_short_pnl < 0:
            best_short_pnl = 0.0
            best_td = float("inf")
    else:
        best_short_pnl = 0.0
        best_td = float("inf")

    total_pnl = best_long_pnl + best_short_pnl

    # Cap extreme thresholds
    if not np.isfinite(best_tu):
        best_tu = 0.008
    if not np.isfinite(best_td):
        best_td = 0.008

    per_sym = {}
    for k in SYMS:
        m = sym == k
        a = gate_2d(pred[m], best_tu, best_td)
        per_sym[f"sym{k}"] = float(vectorized_pnl(a, mp_t[m], mp_th[m]).sum())
        per_sym[f"sym{k}_n_active"] = int((a != 1).sum())

    return (best_tu, best_td), total_pnl, per_sym


def build_params(seed, cfg):
    return {
        "objective": "regression_l2",
        "metric": "l2",
        "learning_rate": 0.05,
        "num_leaves": int(cfg["leaves"]),
        "min_data_in_leaf": 100,
        "feature_fraction": float(cfg["ff"]),
        "bagging_fraction": float(cfg["bf"]),
        "bagging_freq": 5,
        "lambda_l2": float(cfg["l2"]),
        "num_threads": 16,
        "seed": seed,
        "feature_fraction_seed": seed + 1,
        "bagging_seed": seed + 2,
        "data_random_seed": seed + 3,
        "device": "gpu",
        "gpu_use_dp": False,
        "verbose": -1,
    }


def load_data():
    """Load schemeP train + hawkes features. Returns dict."""
    write_progress("loading_data")
    t0 = time.time()
    train_npz = np.load(os.path.join(CACHE, "schemeP_train.npz"))
    with open(os.path.join(CACHE, "schemeP_feat_names.txt")) as f:
        all_names = [l.strip() for l in f if l.strip()]

    total_dim = train_npz["X"].shape[1]
    assert len(all_names) == total_dim, f"feat name mismatch {len(all_names)} != {total_dim}"
    name_to_idx = {n: i for i, n in enumerate(all_names)}
    drop_idx = {name_to_idx[n] for n in DROP_NAMES if n in name_to_idx}
    keep_idx = np.array([i for i in range(total_dim) if i not in drop_idx], dtype=np.int64)
    feat_names_base = [all_names[i] for i in keep_idx]

    hawkes = np.load(os.path.join(HAWKES_DIR, "hawkes_train.npy"))
    with open(os.path.join(HAWKES_DIR, "hawkes_train_names.txt")) as f:
        hawkes_names = [l.strip() for l in f if l.strip()]

    feat_names = feat_names_base + hawkes_names

    # Sanity: no forbidden features
    forbidden = {"date", "sym"}
    leaked = forbidden & set(feat_names)
    assert not leaked, f"Feature leak: {leaked}"

    X_base = train_npz["X"][:, keep_idx].astype(np.float32)
    X_all = np.concatenate([X_base, hawkes], axis=1)
    del X_base, hawkes

    data = {
        "X": X_all,
        "y60": train_npz["y60"].astype(np.int64),
        "y_regr": regr_target(train_npz["mp_t"], train_npz["mp_t60"]),
        "mp_t": train_npz["mp_t"],
        "mp_t60": train_npz["mp_t60"],
        "sym": train_npz["sym"].astype(np.int8),
        "date": train_npz["date"].astype(np.int16),
        "feat_names": feat_names,
    }
    print(f"  loaded in {time.time()-t0:.1f}s  X={X_all.shape}  feat_dim={X_all.shape[1]}",
          flush=True)
    return data, keep_idx


def run_cv(data, seeds):
    """5-fold LOSO CV: for each sym, train on others, predict on held-out sym."""
    X = data["X"]
    y_regr = data["y_regr"]
    y_cls = data["y60"]
    sym = data["sym"]
    mp_t = data["mp_t"]
    mp_t60 = data["mp_t60"]
    feat_names = data["feat_names"]
    n_total = len(X)

    seed_results = {}
    all_seed_preds = {}

    for seed in seeds:
        cfg = SEED_CONFIGS[seed]
        params = build_params(seed, cfg)
        write_progress(f"cv_seed{seed}", seed=seed)
        print(f"\n=== LOSO CV  seed={seed} ===", flush=True)

        oof_pred = np.zeros(n_total, dtype=np.float32)
        rng = np.random.default_rng(seed * 7919 + 1)

        for held_sym in SYMS:
            t_fold = time.time()
            mask_train = (sym != held_sym)
            mask_held = (sym == held_sym)

            X_tr = X[mask_train]
            y_cls_tr = y_cls[mask_train]
            y_regr_tr = y_regr[mask_train]

            X_aug = aug_a_scale(X_tr, rng, lo=0.8, hi=1.2)
            X_tr_full = np.concatenate([X_tr, X_aug], axis=0)
            y_cls_tr_full = np.concatenate([y_cls_tr, y_cls_tr], axis=0)
            y_regr_tr_full = np.concatenate([y_regr_tr, y_regr_tr], axis=0)

            sw_tr = class_balanced_weight(y_cls_tr_full.astype(np.int64))

            dtrain = lgb.Dataset(
                X_tr_full, label=y_regr_tr_full, weight=sw_tr,
                feature_name=feat_names, free_raw_data=True,
            )

            booster = lgb.train(
                params=params,
                train_set=dtrain,
                num_boost_round=NUM_BOOST_ROUND,
                callbacks=[lgb.log_evaluation(period=100)],
            )

            X_held = X[mask_held]
            held_pred = predict_chunked(booster, X_held)
            oof_pred[mask_held] = held_pred

            fold_time = time.time() - t_fold
            print(f"  sym{held_sym} held={mask_held.sum():,} train={mask_train.sum():,} "
                  f"time={fold_time:.1f}s", flush=True)
            del booster, dtrain, X_tr_full, X_tr, X_aug

        # Save OOF preds
        oof_df = pd.DataFrame({
            "sym": sym.astype(np.int8),
            "date": data["date"].astype(np.int16),
            "pred_dmid_norm": oof_pred.astype(np.float32),
            "mp_t": mp_t.astype(np.float32),
            "mp_t60": mp_t60.astype(np.float32),
        })
        pred_path = os.path.join(HERE, f"loso_pred_seed{seed}.parquet")
        oof_df.to_parquet(pred_path, index=False)
        print(f"  OOF saved: {pred_path}", flush=True)

        # Compute LOSO PnL with DE
        write_progress(f"cv_de_seed{seed}", seed=seed)
        thresh, pnl, per_sym = compute_loso_pnl_fast(
            oof_pred.astype(np.float64),
            sym.astype(np.int8),
            mp_t.astype(np.float64),
            mp_t60.astype(np.float64),
        )
        print(f"  seed={seed} LOSO PnL={pnl:+.4f} thresh=({thresh[0]:.5f},{thresh[1]:.5f})",
              flush=True)

        seed_results[seed] = {
            "loso_pnl": pnl,
            "thresh_up": thresh[0],
            "thresh_dn": thresh[1],
            "per_sym": per_sym,
        }
        all_seed_preds[seed] = oof_pred

    # Ensemble: average all seed preds
    ens_pred = np.mean(
        np.stack([all_seed_preds[s] for s in seeds], axis=0), axis=0
    ).astype(np.float64)
    thresh_ens, pnl_ens, per_sym_ens = compute_loso_pnl_fast(
        ens_pred, sym.astype(np.int8),
        mp_t.astype(np.float64), mp_t60.astype(np.float64),
    )
    print(f"\n=== ENSEMBLE ({len(seeds)}-seed) LOSO PnL={pnl_ens:+.4f} ===", flush=True)

    # Save ensemble preds
    ens_df = pd.DataFrame({
        "sym": sym.astype(np.int8),
        "date": data["date"].astype(np.int16),
        "pred_dmid_norm": ens_pred.astype(np.float32),
        "mp_t": mp_t.astype(np.float32),
        "mp_t60": mp_t60.astype(np.float32),
    })
    ens_df.to_parquet(os.path.join(HERE, "loso_pred_ensemble.parquet"), index=False)

    return seed_results, {"loso_pnl": pnl_ens, "thresh_up": thresh_ens[0],
                           "thresh_dn": thresh_ens[1], "per_sym": per_sym_ens}


def run_full_retrain(data, seeds):
    """Full retrain on ALL data. Fixed 330 rounds, no validation."""
    X = data["X"]
    y_regr = data["y_regr"]
    y_cls = data["y60"]
    feat_names = data["feat_names"]

    model_paths = {}
    for seed in seeds:
        cfg = SEED_CONFIGS[seed]
        params = build_params(seed, cfg)
        write_progress(f"full_retrain_seed{seed}", seed=seed)
        print(f"\n=== Full Retrain seed={seed} ===", flush=True)

        rng = np.random.default_rng(seed * 7919 + 1)
        X_aug = aug_a_scale(X, rng, lo=0.8, hi=1.2)
        X_tr_full = np.concatenate([X, X_aug], axis=0)
        y_cls_tr_full = np.concatenate([y_cls, y_cls], axis=0)
        y_regr_tr_full = np.concatenate([y_regr, y_regr], axis=0)

        sw_tr = class_balanced_weight(y_cls_tr_full.astype(np.int64))

        dtrain = lgb.Dataset(
            X_tr_full, label=y_regr_tr_full, weight=sw_tr,
            feature_name=feat_names, free_raw_data=True,
        )

        t0 = time.time()
        booster = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            callbacks=[lgb.log_evaluation(period=50)],
        )
        train_time = time.time() - t0

        model_path = os.path.join(HERE, f"model_h60_seed{seed}_hawkes_full.txt")
        booster.save_model(model_path)
        print(f"  seed={seed} trained in {train_time:.1f}s -> {model_path}", flush=True)
        model_paths[seed] = model_path
        del booster, dtrain, X_tr_full, X_aug

    return model_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("cv", "full_retrain", "both"), default="both")
    ap.add_argument("--seeds", type=str, default="1,7,13,42,100")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"T142 hawkes 5-seed | mode={args.mode} seeds={seeds}", flush=True)

    # WandB init
    try:
        import wandb
        wandb.login(key="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")
        run = wandb.init(
            project="liangwenbei",
            entity="cjxh21-Tsinghua University",
            name=f"T142-hawkes-5seed-{args.mode}",
            config={"seeds": seeds, "mode": args.mode, "num_boost_round": NUM_BOOST_ROUND},
        )
        use_wandb = True
    except Exception as e:
        print(f"WandB init failed ({e!r}); continuing without WandB", flush=True)
        use_wandb = False

    data, keep_idx = load_data()
    print(f"  feat_dim={data['X'].shape[1]}  n_total={len(data['X']):,}", flush=True)

    results = {
        "task": "T142 hawkes 5-seed LOSO CV + full-data retrain",
        "inference_feasible": True,
        "inference_notes": (
            "EWMA(alpha=0.2) decays to ~2e-10 after 100 ticks, "
            "fully reproducible from 100-tick window; stateless per-session"
        ),
        "num_boost_round": NUM_BOOST_ROUND,
        "seeds": seeds,
    }

    seed_cv_results = {}
    ens_cv_result = {}
    model_paths = {}

    if args.mode in ("cv", "both"):
        print("\n" + "="*60, flush=True)
        print("MODE A: LOSO CV", flush=True)
        print("="*60, flush=True)
        seed_cv_results, ens_cv_result = run_cv(data, seeds)

        metrics = {}
        for s in seeds:
            metrics[f"loso_pnl_seed{s}"] = seed_cv_results[s]["loso_pnl"]
        metrics["loso_pnl_5seed_ensemble"] = ens_cv_result["loso_pnl"]

        print("\n=== CV Summary ===", flush=True)
        for s in seeds:
            print(f"  seed={s}  LOSO PnL={metrics[f'loso_pnl_seed{s}']:+.4f}", flush=True)
        print(f"  ensemble LOSO PnL={metrics['loso_pnl_5seed_ensemble']:+.4f}", flush=True)
        print(f"  vs T75 baseline (32.78): {metrics['loso_pnl_5seed_ensemble'] - 32.78:+.4f}",
              flush=True)
        print(f"  vs 3-seed hawkes (33.98): "
              f"{metrics['loso_pnl_5seed_ensemble'] - 33.98:+.4f}", flush=True)

        results["metrics"] = metrics
        results["seed_cv_details"] = seed_cv_results
        results["ensemble_cv"] = ens_cv_result

        if use_wandb:
            wandb.log({f"loso_pnl_seed{s}": seed_cv_results[s]["loso_pnl"] for s in seeds})
            wandb.log({
                "loso_pnl_5seed_ensemble": ens_cv_result["loso_pnl"],
                "delta_vs_t75": ens_cv_result["loso_pnl"] - 32.78,
                "delta_vs_3seed_hawkes": ens_cv_result["loso_pnl"] - 33.98,
            })

    if args.mode in ("full_retrain", "both"):
        print("\n" + "="*60, flush=True)
        print("MODE B: FULL RETRAIN", flush=True)
        print("="*60, flush=True)
        model_paths = run_full_retrain(data, seeds)
        results["full_retrain_models"] = [model_paths[s] for s in seeds]

    # Write results
    results_path = os.path.join(HERE, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nresults -> {results_path}", flush=True)

    write_progress("done",
                   loso_ensemble_pnl=ens_cv_result.get("loso_pnl", None),
                   n_full_models=len(model_paths))

    if use_wandb:
        wandb.finish()

    # Final RESULT line
    ens_pnl = ens_cv_result.get("loso_pnl", 0.0)
    vs_t75 = ens_pnl - 32.78
    n_models = len(model_paths)
    print(
        f"\nRESULT: task=[T142 hawkes 5seed LOSO+full-retrain] "
        f"metrics={{loso_5seed_ensemble={ens_pnl:.4f}, vs_t75={vs_t75:+.4f}, "
        f"n_full_models={n_models}}} "
        f"notes=[5-fold LOSO OOF on train data; inference_feasible=True]"
    )


if __name__ == "__main__":
    main()
