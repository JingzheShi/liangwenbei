"""T45 Group DRO LightGBM (R32 idea #2).

Treat sym in {0..4} as 5 environments. Every K boosting rounds, compute
per-sym training cross-entropy on the *augmented* training set, then
softmax(loss / tau) -> per-sym weights, scaled so mean weight per sample
is 1.0 (multiply by 5 since there are 5 syms).

Compliance with CRITICAL_CONSTRAINTS:
  * sym is used ONLY at training time as group label (never a feature, never at inference)
  * date never enters
  * Predictor is sym-agnostic (LightGBM model trained on raw 226-d feature columns)

Usage:
  # tau ablation, single fold (fold=2 by default), seed=42
  python train_loso.py --tau 0.5 --syms 2 --seeds 42 --tag tau_ablation

  # full LOSO with chosen tau
  python train_loso.py --tau 0.5 --syms 0,1,2,3,4 --seeds 42 --tag main
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

import lightgbm as lgb  # noqa: E402
from src.eval.pnl import _per_horizon_metrics, sanitize_for_json  # noqa: E402

T26_DIR = os.path.join(ROOT, "experiments", "T26_domain_randomization")
sys.path.insert(0, T26_DIR)
from build_aug import class_balanced_weight  # noqa: E402

T5B_CACHE = os.path.join(ROOT, "experiments", "T5b_features_multihorizon", "cache")
NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)
H = 60

# Match T37 SEED_CONFIGS for reproducibility / future ensembling
SEED_CONFIGS = {
    42:  dict(feature_fraction=0.8, bagging_fraction=0.8,  num_leaves=127, lambda_l2=1.0),
    1:   dict(feature_fraction=0.6, bagging_fraction=0.7,  num_leaves=127, lambda_l2=1.0),
    7:   dict(feature_fraction=0.7, bagging_fraction=0.85, num_leaves=63,  lambda_l2=2.0),
    13:  dict(feature_fraction=0.5, bagging_fraction=0.6,  num_leaves=255, lambda_l2=0.5),
    100: dict(feature_fraction=0.4, bagging_fraction=0.5,  num_leaves=127, lambda_l2=3.0),
}


def progress(step: str, **extra) -> None:
    p = os.path.join(HERE, "worker-progress.json")
    payload = {
        "status": "running",
        "step": step,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **extra,
    }
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


def load_split(split: str) -> dict:
    p = os.path.join(T5B_CACHE, f"schemeC_{split}.npz")
    print(f"  loading {p} ...", flush=True)
    d = np.load(p)
    return {k: d[k] for k in d.files}


def predict_proba(booster: lgb.Booster, X: np.ndarray, batch: int = 200_000):
    chunks = []
    for s in range(0, len(X), batch):
        chunks.append(booster.predict(X[s:s + batch]).astype(np.float32))
    return np.concatenate(chunks, axis=0)


def evaluate_metrics(name, preds, y, mp_t, mp_th, fee_rate=0.0001):
    m = _per_horizon_metrics(preds, y, mp_t, mp_th, fee_rate=fee_rate)
    m["pred_distribution"] = dict(m["pred_distribution"])
    m["label_distribution"] = dict(m["label_distribution"])
    print(f"  --- {name} ---", flush=True)
    for k in ("accuracy", "cum_pnl", "single_pnl", "n_predictions_active", "f0_5_macro"):
        v = m.get(k)
        if isinstance(v, float):
            print(f"    {k:25s}: {v:+.6g}", flush=True)
        else:
            print(f"    {k:25s}: {v}", flush=True)
    return m


def aug_uniform_concat(X_orig, y_orig, sym_orig, lo, hi, rng):
    """Per-(sample, feature) uniform[lo, hi]; concat orig+aug, also doubled sym."""
    scales = rng.uniform(lo, hi, size=X_orig.shape).astype(X_orig.dtype)
    X_aug = X_orig * scales
    X_merged = np.concatenate([X_orig, X_aug], axis=0)
    y_merged = np.concatenate([y_orig, y_orig], axis=0).astype(np.int64)
    sym_merged = np.concatenate([sym_orig, sym_orig], axis=0).astype(np.int64)
    return X_merged, y_merged, sym_merged


class GroupDROCallback:
    """Every K rounds, compute per-sym CE loss on training data and reweight.

    Weight per sym is softmax(loss / tau); per-sample weight = 5 * w[sym(i)]
    so the sample-mean weight stays 1 (with 5 syms). With tau=0.5 typical,
    worst sym roughly gets 2-3x weight of best sym.

    Fire schedule: fires once at iter == first_update_at, then every K
    iterations thereafter. Models early-stop around iter ~120 with
    early_stopping=40, so first_update_at=20 + K=50 yields >=2 reweights
    (iter 20, 70, 120, ...) before stop.
    """
    order = 100  # run after early_stopping callback (matters less here)

    def __init__(self, X_tr, y_tr, sym_tr, dtrain, K=50, tau=0.5,
                 first_update_at=20, num_class=3, log_lines=None):
        self.X_tr = X_tr
        self.y_tr = np.asarray(y_tr, dtype=np.int64)
        self.sym_tr = np.asarray(sym_tr, dtype=np.int64)
        self.dtrain = dtrain
        self.K = K
        self.tau = tau
        self.first_update_at = int(first_update_at)
        self.num_class = num_class
        self.history = []
        self.log_lines = log_lines if log_lines is not None else []

    def _per_sym_ce(self, raw):
        # raw shape: (n, num_class) for multiclass when raw_score=True
        if raw.ndim == 1:
            raw = raw.reshape(-1, self.num_class)
        x_max = raw.max(axis=1, keepdims=True)
        log_p = raw - x_max - np.log(np.exp(raw - x_max).sum(axis=1, keepdims=True))
        losses = -log_p[np.arange(len(self.y_tr)), self.y_tr]
        sym_losses = np.zeros(5, dtype=np.float64)
        for s in range(5):
            mask = self.sym_tr == s
            sym_losses[s] = float(losses[mask].mean()) if mask.sum() > 0 else 0.0
        return sym_losses

    def __call__(self, env):
        # callback fires AFTER iteration; env.iteration is 0-based.
        # Fire when iteration == first_update_at, or thereafter every K rounds.
        it = int(env.iteration)
        if it < self.first_update_at:
            return
        if it != self.first_update_at and (it - self.first_update_at) % self.K != 0:
            return
        t0 = time.time()
        raw = env.model.predict(self.X_tr, raw_score=True)
        sym_losses = self._per_sym_ce(raw)
        # softmax with temperature (numerically stable)
        m = sym_losses.max()
        scaled = (sym_losses - m) / max(self.tau, 1e-6)
        w = np.exp(scaled)
        w = w / w.sum()
        sample_weights = (w[self.sym_tr] * 5.0).astype(np.float32)
        self.dtrain.set_weight(sample_weights)
        rec = {
            "iter": int(env.iteration),
            "elapsed_sec": float(time.time() - t0),
            "sym_losses": [float(x) for x in sym_losses],
            "sym_weights": [float(x) for x in w],
        }
        self.history.append(rec)
        msg = (
            f"  [DRO@iter{env.iteration}] sym_loss={[f'{x:.4f}' for x in sym_losses]} "
            f"w={[f'{x:.3f}' for x in w]} (took {rec['elapsed_sec']:.1f}s)"
        )
        print(msg, flush=True)
        self.log_lines.append(msg)


def train_one_fold(seed, held, X_tr_full, y_tr_full, sym_tr_full,
                   X_va_full, y_va_full, sym_va_full,
                   X_te_full, y_te_full, sym_te_full,
                   mp_t_te_full, mp_th_te_full,
                   sess_te_full, date_te_full, t_te_full,
                   feat_names, args, cfg, tag):
    m_tr = sym_tr_full != held
    m_va = sym_va_full != held
    m_te = sym_te_full == held

    X_tr_o = X_tr_full[m_tr]
    y_tr_o = y_tr_full[m_tr].astype(np.int64)
    sym_tr_o = sym_tr_full[m_tr].astype(np.int64)
    X_va = X_va_full[m_va]
    y_va = y_va_full[m_va].astype(np.int64)
    X_te = X_te_full[m_te]
    y_te = y_te_full[m_te].astype(np.int64)
    mp_t_te = mp_t_te_full[m_te]
    mp_th_te = mp_th_te_full[m_te]

    fold_rng = np.random.default_rng(seed * 7919 + held * 17 + 1)
    X_tr, y_tr, sym_tr_dbl = aug_uniform_concat(
        X_tr_o, y_tr_o, sym_tr_o, args.aug_lo, args.aug_hi, fold_rng,
    )
    # Initial sample weights = 1.0 (uniform). DRO callback will overwrite at iter K.
    sw_tr = np.ones(len(X_tr), dtype=np.float32)
    sw_va = class_balanced_weight(y_va, NUM_CLASS)

    print(
        f"    n_train_orig={len(X_tr_o):,} n_train_used={len(X_tr):,} "
        f"n_val={len(X_va):,} n_test={len(X_te):,} feat_dim={X_tr.shape[1]}",
        flush=True,
    )
    sym_counts_tr = {int(s): int((sym_tr_dbl == s).sum()) for s in range(5)}
    print(f"    sym counts (training, post-aug): {sym_counts_tr}", flush=True)

    dtrain = lgb.Dataset(
        X_tr, label=y_tr, weight=sw_tr,
        feature_name=feat_names, free_raw_data=False,
    )
    dval = lgb.Dataset(
        X_va, label=y_va, weight=sw_va,
        feature_name=feat_names, reference=dtrain, free_raw_data=False,
    )

    params = {
        "objective": "multiclass", "num_class": NUM_CLASS,
        "metric": "multi_logloss",
        "learning_rate": args.learning_rate,
        "num_leaves": int(cfg["num_leaves"]),
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": float(cfg["feature_fraction"]),
        "bagging_fraction": float(cfg["bagging_fraction"]),
        "bagging_freq": args.bagging_freq,
        "lambda_l2": float(cfg["lambda_l2"]),
        "num_threads": args.num_threads,
        "seed": seed,
        "feature_fraction_seed": seed + 1,
        "bagging_seed": seed + 2,
        "data_random_seed": seed + 3,
        "verbose": -1,
    }
    if args.use_gpu:
        params.update({"device": "gpu", "gpu_use_dp": False})

    log_lines = []
    dro_cb = GroupDROCallback(
        X_tr=X_tr, y_tr=y_tr, sym_tr=sym_tr_dbl,
        dtrain=dtrain, K=args.K, tau=args.tau,
        first_update_at=args.first_update_at,
        num_class=NUM_CLASS, log_lines=log_lines,
    )

    t_start = time.time()
    booster = lgb.train(
        params, dtrain,
        num_boost_round=args.num_boost_round,
        valid_sets=[dval], valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=args.early_stopping, verbose=False),
            dro_cb,
            lgb.log_evaluation(period=100),
        ],
    )
    train_time = time.time() - t_start
    print(
        f"    trained in {train_time:.1f}s, best_iter={booster.best_iteration}",
        flush=True,
    )

    te_prob = predict_proba(booster, X_te)
    te_pred = te_prob.argmax(axis=1).astype(np.int8)
    te_metrics = evaluate_metrics(
        f"HELD-OUT TEST tag={tag} seed={seed} held={held}",
        te_pred, y_te, mp_t_te, mp_th_te,
    )

    sess_map = {0: "am", 1: "pm"}
    te_df = pd.DataFrame({
        "sym": sym_te_full[m_te].astype(np.int8),
        "date": date_te_full[m_te].astype(np.int16),
        "session": np.array([sess_map[s] for s in sess_te_full[m_te]], dtype=object),
        "t": t_te_full[m_te].astype(np.int16),
        "true_label": y_te.astype(np.int8),
        "pred_label": te_pred.astype(np.int8),
        "prob_0": te_prob[:, 0].astype(np.float32),
        "prob_1": te_prob[:, 1].astype(np.float32),
        "prob_2": te_prob[:, 2].astype(np.float32),
        "midprice_t": mp_t_te.astype(np.float32),
        "midprice_th": mp_th_te.astype(np.float32),
    })
    pred_path = os.path.join(
        HERE, f"{tag}_pred_seed{seed}_held{held}.parquet"
    )
    te_df.to_parquet(pred_path, index=False)

    model_path = os.path.join(
        HERE, f"{tag}_model_seed{seed}_held{held}.txt"
    )
    booster.save_model(model_path, num_iteration=booster.best_iteration)

    # Save DRO history for inspection
    hist_path = os.path.join(HERE, f"{tag}_drohist_seed{seed}_held{held}.json")
    with open(hist_path, "w") as f:
        json.dump({
            "tau": args.tau,
            "K": args.K,
            "first_update_at": args.first_update_at,
            "history": dro_cb.history,
            "log_lines": log_lines,
        }, f, indent=2)

    return {
        "tag": tag, "tau": args.tau, "K": args.K,
        "first_update_at": args.first_update_at,
        "seed": seed, "held_out_sym": held,
        "n_train_orig": int(len(X_tr_o)),
        "n_train_used": int(len(X_tr)),
        "n_val": int(len(X_va)),
        "n_test": int(len(X_te)),
        "best_iter": int(booster.best_iteration),
        "train_time_sec": float(train_time),
        "held_cum_pnl": float(te_metrics["cum_pnl"]),
        "held_single_pnl": float(te_metrics["single_pnl"]),
        "held_acc": float(te_metrics["accuracy"]),
        "held_n_active": int(te_metrics["n_predictions_active"]),
        "n_dro_updates": len(dro_cb.history),
    }


def run(seeds, target_syms, args, X_tr, X_va, X_te, feat_names,
        train_full, val_full, test_full):
    print(f"\n{'#'*78}\n### T45 Group DRO tau={args.tau} K={args.K} first_update_at={args.first_update_at} feat_dim={X_tr.shape[1]}\n### tag={args.tag}\n{'#'*78}", flush=True)
    sym_tr = train_full["sym"]
    sym_va = val_full["sym"]
    sym_te = test_full["sym"]
    y_tr = train_full[f"y{H}"]
    y_va = val_full[f"y{H}"]
    y_te = test_full[f"y{H}"]
    mp_t_te = test_full["mp_t"]
    mp_th_te = test_full[f"mp_t{H}"]
    sess_te = test_full["sess_idx"]
    date_te = test_full["date"]
    t_te = test_full["t"]

    all_results = []
    n_total = len(seeds) * len(target_syms)
    done = 0
    for s in seeds:
        cfg = SEED_CONFIGS[s]
        print(f"\n{'='*78}\n=== SEED {s} cfg={cfg} ===\n{'='*78}", flush=True)
        for held in target_syms:
            done += 1
            print(f"\n  [{done}/{n_total}] tag={args.tag} seed={s} held={held}", flush=True)
            progress("training", tag=args.tag, tau=args.tau, K=args.K,
                     seed=s, held=held, done=done, total=n_total)
            r = train_one_fold(
                s, held,
                X_tr, y_tr, sym_tr,
                X_va, y_va, sym_va,
                X_te, y_te, sym_te,
                mp_t_te, mp_th_te,
                sess_te, date_te, t_te,
                feat_names, args, cfg, args.tag,
            )
            all_results.append(r)

    df = pd.DataFrame(all_results)
    print(f"\n{'='*78}\n=== AGGREGATE tag={args.tag} h={H} ===\n{'='*78}", flush=True)
    agg = []
    for s in seeds:
        sub = df[df.seed == s]
        cums = sub["held_cum_pnl"].values
        accs = sub["held_acc"].values
        rec = {
            "seed": s, "n_folds": int(len(cums)),
            "cum_pnl_per_fold": cums.tolist(),
            "cum_pnl_sum": float(cums.sum()),
            "cum_pnl_mean": float(cums.mean()) if len(cums) else 0.0,
            "n_pos_folds": int((cums > 0).sum()),
            "acc_per_fold": accs.tolist(),
            "acc_mean": float(accs.mean()) if len(accs) else 0.0,
        }
        agg.append(rec)
        print(
            f"  seed{s}: sum={rec['cum_pnl_sum']:+.4f} mean_acc={rec['acc_mean']:.4f} "
            f"pos={rec['n_pos_folds']}/{rec['n_folds']} "
            f"per_fold_pnl={[f'{x:+.3f}' for x in cums]}",
            flush=True,
        )
    return all_results, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--syms", default="0,1,2,3,4")
    ap.add_argument("--tag", default="main",
                    help="output filename prefix (e.g., 'tau05', 'main')")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--K", type=int, default=50, dest="K")
    ap.add_argument("--first-update-at", type=int, default=20,
                    dest="first_update_at",
                    help="iteration at which to fire the first DRO reweight")
    ap.add_argument("--aug-lo", type=float, default=0.80)
    ap.add_argument("--aug-hi", type=float, default=1.20)
    ap.add_argument("--num-boost-round", type=int, default=600)
    ap.add_argument("--early-stopping", type=int, default=40)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--min-data-in-leaf", type=int, default=100)
    ap.add_argument("--bagging-freq", type=int, default=5)
    ap.add_argument("--num-threads", type=int, default=18)
    ap.add_argument("--use-gpu", action="store_true", default=True)
    ap.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    ap.add_argument("--drop-tail-cols", type=int, default=0,
                    help="drop last N feature cols from cache (use 3 for 223-d "
                         "iter_006-compatible: drops time_minutes_since_session_start, "
                         "time_session_progress, time_is_pm)")
    args = ap.parse_args()

    target_syms = [int(x) for x in args.syms.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    for s in seeds:
        if s not in SEED_CONFIGS:
            sys.exit(f"unknown seed {s}, valid={list(SEED_CONFIGS)}")

    print(f"=== T45 Group DRO LightGBM h={H} tag={args.tag} tau={args.tau} K={args.K} "
          f"seeds={seeds} syms={target_syms} aug=[{args.aug_lo},{args.aug_hi}] ===",
          flush=True)

    progress("loading_caches", tag=args.tag, tau=args.tau, seeds=seeds)
    t0 = time.time()
    train_full = load_split("train")
    val_full = load_split("val")
    test_full = load_split("test")
    print(
        f"loaded in {time.time() - t0:.1f}s; "
        f"train={train_full['X'].shape} val={val_full['X'].shape} test={test_full['X'].shape}",
        flush=True,
    )

    X_tr = train_full["X"].astype(np.float32, copy=False)
    X_va = val_full["X"].astype(np.float32, copy=False)
    X_te = test_full["X"].astype(np.float32, copy=False)

    feat_names_path = os.path.join(T5B_CACHE, "schemeC_feat_names.txt")
    with open(feat_names_path) as f:
        feat_names = [ln.strip() for ln in f if ln.strip()]
    assert len(feat_names) == X_tr.shape[1], (len(feat_names), X_tr.shape[1])

    # Optionally drop trailing cols (time_* features) to match iter_006 (223-d)
    if args.drop_tail_cols > 0:
        n_drop = int(args.drop_tail_cols)
        dropped = feat_names[-n_drop:]
        X_tr = X_tr[:, :-n_drop]
        X_va = X_va[:, :-n_drop]
        X_te = X_te[:, :-n_drop]
        feat_names = feat_names[:-n_drop]
        print(f"  dropped last {n_drop} cols: {dropped} -> new feat_dim={X_tr.shape[1]}",
              flush=True)

    forbidden = {"date", "sym"}
    leak = forbidden & set(feat_names)
    assert not leak, f"FORBIDDEN features: {leak}"

    print(f"  feat_dim={X_tr.shape[1]}, last 5 feats: {feat_names[-5:]}", flush=True)

    out = {
        "task": "T45 Group DRO LightGBM",
        "horizon": H,
        "tag": args.tag,
        "tau": args.tau,
        "K": args.K,
        "first_update_at": args.first_update_at,
        "seeds": seeds,
        "syms": target_syms,
        "aug_lo": args.aug_lo, "aug_hi": args.aug_hi,
        "params": vars(args),
    }

    results, agg = run(
        seeds, target_syms, args, X_tr, X_va, X_te, feat_names,
        train_full, val_full, test_full,
    )
    out["results"] = results
    out["agg"] = agg
    out["elapsed_sec"] = time.time() - t0

    # Save with tag-specific filename to avoid overwriting on multiple runs
    out_path = os.path.join(HERE, f"results_{args.tag}.json")
    with open(out_path, "w") as f:
        json.dump(sanitize_for_json(out), f, indent=2)
    print(f"\nALL DONE in {out['elapsed_sec']:.1f}s. summary -> {out_path}", flush=True)
    progress("done", tag=args.tag, elapsed_sec=out["elapsed_sec"])


if __name__ == "__main__":
    main()
