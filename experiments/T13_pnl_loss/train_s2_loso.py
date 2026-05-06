"""T13 S2 — Custom LightGBM multiclass objective for expected PnL: 5-fold LOSO.

Loss per sample t:
    Define class rewards (per the leaderboard PnL formula, normalised to be
    per-unit-of-(1+mp_t)):
        r_0(t) = (-Δp_t - 2·fee_eff(t)) / (1 + mp_t)     # short
        r_1(t) = 0                                       # flat
        r_2(t) = (+Δp_t - 2·fee_eff(t)) / (1 + mp_t)     # long
    where fee_eff(t) ≈ fee_rate * 0.5 * ((mp_th+1) + (mp_t+1)).

    Let p = softmax(z) over K=3 classes. Per-sample loss (with optional entropy reg):
        L = - Σ_k p_k · r_k   - β · H(p)

    where H(p) = - Σ_k p_k · log p_k is entropy (encourages exploration to avoid
    "always class 1" collapse).

    Closed-form gradient w.r.t. logit z_k:
        ∂L_pnl/∂z_k     = - p_k · ( r_k - <r> )                where <r> = Σ_j p_j r_j
        ∂L_entropy/∂z_k = - β · p_k · ( H_k - <H_k> )           where H_k = -log p_k
                        wait, derive: H = -Σ_k p_k log p_k.
                        dH/dz_j = -p_j (log p_j + 1) + p_j Σ_k p_k (log p_k + 1)
                                = -p_j (log p_j - <log p>)
                        so d(-βH)/dz_j = β p_j (log p_j - <log p>)

    Use a positive bounded Newton-style hessian per class:
        hess_k = max( p_k * (1-p_k) * |r_k - <r>| , h_min )

Custom feval = realised cum_pnl on val (for early-stopping with higher_is_better=True).

Approach:
  - Warm-start with N_warm rounds of CE multiclass (init_score ↦ raw logits)
  - Switch to custom fobj for the rest of rounds
  - Use lower learning_rate for the PnL phase (gradient is rougher)
  - Pass Δp / mp_t / fee_eff via closure (avoids LightGBM hidden-state hacks)

Usage:
    python3 train_s2_loso.py --horizon 10 --beta-entropy 0.005
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

import lightgbm as lgb  # noqa: E402
from src.eval.pnl import _per_horizon_metrics, sanitize_for_json  # noqa: E402

CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "T5b_features_multihorizon", "cache"))
NUM_CLASS = 3


def load_split(split: str) -> dict:
    p = os.path.join(CACHE_DIR, f"schemeC_{split}.npz")
    print(f"  loading {p} ...", flush=True)
    d = np.load(p)
    return {k: d[k] for k in d.files}


def class_balanced_weight(y: np.ndarray, num_class: int = 3) -> np.ndarray:
    counts = np.bincount(y.astype(np.int64), minlength=num_class).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    cw = len(y) / (num_class * counts)
    return cw[y.astype(np.int64)].astype(np.float32)


def predict_proba(booster: lgb.Booster, X: np.ndarray, batch: int = 200_000):
    chunks = []
    for s in range(0, len(X), batch):
        chunks.append(booster.predict(X[s : s + batch]).astype(np.float32))
    return np.concatenate(chunks, axis=0)


def evaluate(name, preds, y, mp_t, mp_th, fee_rate=0.0001):
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


def compute_rewards(mp_t: np.ndarray, mp_th: np.ndarray, fee_rate: float = 1e-4) -> np.ndarray:
    """Return (N, 3) reward matrix r_k(t) aligned with leaderboard PnL formula."""
    mp_t = mp_t.astype(np.float64)
    mp_th = mp_th.astype(np.float64)
    diff = mp_th - mp_t
    denom = mp_t + 1.0
    denom = np.where(denom <= 0, 1.0, denom)  # safety
    fee_amt = fee_rate * np.abs((mp_th + 1.0) + (mp_t + 1.0))  # 2-side fee absolute
    r = np.zeros((len(mp_t), 3), dtype=np.float64)
    r[:, 0] = (-diff - fee_amt) / denom    # short
    r[:, 1] = 0.0                          # flat
    r[:, 2] = (+diff - fee_amt) / denom    # long
    return r.astype(np.float32)


def _coerce_2d(z: np.ndarray, K: int) -> np.ndarray:
    """Coerce LightGBM y_pred to (N, K). In lgb 4.x with multiclass + custom
    callable objective, y_pred arrives 2D (N, K). Older versions sent (N*K,)
    Fortran-order. Both cases handled."""
    if z.ndim == 2:
        return z
    return z.reshape(-1, K, order="F")


def make_pnl_objective(rewards: np.ndarray, num_class: int = 3,
                       beta_entropy: float = 0.0,
                       hess_min: float = 1e-3,
                       grad_scale: float = 1.0):
    """Return fobj closure for LightGBM multiclass with expected-PnL loss."""
    R = rewards.astype(np.float64)
    K = num_class

    def fobj(y_pred, train_data):
        z = _coerce_2d(y_pred, K).astype(np.float64)
        # Numerically stable softmax
        z = z - z.max(axis=1, keepdims=True)
        exp_z = np.exp(z)
        p = exp_z / exp_z.sum(axis=1, keepdims=True)  # (N, K)

        # Mean reward per sample
        Er = (p * R).sum(axis=1, keepdims=True)  # (N, 1)
        # Gradient of -E[r] w.r.t z_k:  -p_k · (r_k - E[r])
        grad_pnl = -p * (R - Er)  # (N, K)

        if beta_entropy > 0.0:
            # Gradient of -β·H(p) w.r.t z_j:  β · p_j · (log p_j - <log p>)
            log_p = np.log(np.maximum(p, 1e-12))
            mean_logp = (p * log_p).sum(axis=1, keepdims=True)
            grad_ent = beta_entropy * p * (log_p - mean_logp)
            grad = grad_pnl + grad_ent
        else:
            grad = grad_pnl

        grad *= grad_scale

        # Positive Newton-style hessian: p_k(1-p_k)·|r_k - E[r]| + h_min
        hess = (p * (1.0 - p)) * np.abs(R - Er) + hess_min
        hess *= grad_scale

        return grad.astype(np.float64), hess.astype(np.float64)

    return fobj


def make_pnl_feval(rewards_va: np.ndarray, num_class: int = 3,
                   include_init_score: bool = True):
    """feval that computes argmax-PnL on val using the *current combined logits*.

    NOTE: when init_score is set on dval, lgb passes y_pred = init_score + inc,
    so argmax over y_pred gives the correct prediction. Always pass include_init_score=True.
    """
    R = rewards_va.astype(np.float64)
    K = num_class

    def feval(y_pred, train_data):
        z = _coerce_2d(y_pred, K)
        pred = z.argmax(axis=1)
        rows = np.arange(len(R))
        cum = float(R[rows, pred].sum())
        return "cum_pnl_argmax", cum, True

    return feval


def fold_train(X_tr, y_tr, mp_t_tr, mp_th_tr,
               X_va, y_va, mp_t_va, mp_th_va,
               feat_names, args, log_prefix=""):
    """Two-stage training for one fold: CE warmstart → custom PnL objective."""
    # Stage 1: warmstart with CE
    sw_tr_cls = class_balanced_weight(y_tr)
    sw_va_cls = class_balanced_weight(y_va)
    dtrain_ce = lgb.Dataset(X_tr, label=y_tr, weight=sw_tr_cls,
                            feature_name=feat_names, free_raw_data=False)
    dval_ce = lgb.Dataset(X_va, label=y_va, weight=sw_va_cls,
                          feature_name=feat_names, reference=dtrain_ce, free_raw_data=False)
    params_ce = {
        "objective": "multiclass",
        "num_class": NUM_CLASS,
        "metric": "multi_logloss",
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "lambda_l2": args.lambda_l2,
        "num_threads": args.num_threads,
        "seed": args.seed,
        "verbose": -1,
    }
    print(f"{log_prefix}  Stage 1 CE warmstart: {args.warmstart_rounds} rounds", flush=True)
    t0 = time.time()
    booster_ce = lgb.train(
        params_ce, dtrain_ce,
        num_boost_round=args.warmstart_rounds,
        valid_sets=[dtrain_ce, dval_ce],
        valid_names=["train", "val"],
        callbacks=[lgb.log_evaluation(period=max(50, args.warmstart_rounds // 4))],
    )
    print(f"{log_prefix}  CE done in {time.time()-t0:.1f}s, "
          f"#trees={booster_ce.num_trees()}", flush=True)

    # Stage 2: custom PnL objective starting from CE init_score
    init_score_tr = booster_ce.predict(X_tr, raw_score=True).astype(np.float64)  # (N, K)
    init_score_va = booster_ce.predict(X_va, raw_score=True).astype(np.float64)

    rewards_tr = compute_rewards(mp_t_tr, mp_th_tr, fee_rate=1e-4)
    rewards_va = compute_rewards(mp_t_va, mp_th_va, fee_rate=1e-4)

    # lgb 4.x accepts 2D init_score (N, K) for multiclass.
    dtrain_pnl = lgb.Dataset(X_tr, label=y_tr, init_score=init_score_tr,
                             feature_name=feat_names, free_raw_data=False)
    dval_pnl = lgb.Dataset(X_va, label=y_va, init_score=init_score_va,
                           feature_name=feat_names, reference=dtrain_pnl, free_raw_data=False)

    fobj = make_pnl_objective(rewards_tr, num_class=NUM_CLASS,
                              beta_entropy=args.beta_entropy,
                              hess_min=args.hess_min,
                              grad_scale=args.grad_scale)
    feval = make_pnl_feval(rewards_va, num_class=NUM_CLASS)

    params_pnl = {
        "objective": fobj,           # lgb 4.x: pass callable via params
        "num_class": NUM_CLASS,
        "metric": "None",            # disable default metric; use feval only
        "learning_rate": args.pnl_lr,
        "num_leaves": args.pnl_num_leaves,
        "min_data_in_leaf": args.pnl_min_data_in_leaf,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "lambda_l2": args.lambda_l2,
        "num_threads": args.num_threads,
        "seed": args.seed,
        "verbose": -1,
    }

    print(f"{log_prefix}  Stage 2 custom PnL: lr={args.pnl_lr} rounds<={args.pnl_rounds} "
          f"β_ent={args.beta_entropy} hess_min={args.hess_min} grad_scale={args.grad_scale}",
          flush=True)
    t0 = time.time()
    booster_pnl = lgb.train(
        params_pnl, dtrain_pnl,
        num_boost_round=args.pnl_rounds,
        valid_sets=[dtrain_pnl, dval_pnl],
        valid_names=["train", "val"],
        feval=feval,
        callbacks=[
            lgb.early_stopping(stopping_rounds=args.pnl_early_stopping, verbose=False,
                               first_metric_only=True),
            lgb.log_evaluation(period=max(20, args.pnl_rounds // 10)),
        ],
    )
    pnl_time = time.time() - t0
    print(f"{log_prefix}  PnL done in {pnl_time:.1f}s, best_iter={booster_pnl.best_iteration}",
          flush=True)
    return booster_ce, booster_pnl, init_score_tr, init_score_va


def predict_with_warmstart(booster_ce: lgb.Booster, booster_pnl: lgb.Booster,
                           X: np.ndarray) -> np.ndarray:
    """Predict probabilities by combining CE warmstart logits + PnL incremental raw output."""
    z_ce = booster_ce.predict(X, raw_score=True).astype(np.float64)  # (N, K)
    # PnL booster starts from init_score; its raw_score is the *increment* over init_score.
    z_inc = booster_pnl.predict(X, raw_score=True,
                                num_iteration=booster_pnl.best_iteration).astype(np.float64)
    # z_inc shape: (N, K) (LightGBM returns (N, K) for multiclass raw_score)
    if z_inc.ndim == 1:
        z_inc = z_inc.reshape(-1, NUM_CLASS, order='F')
    z = z_ce + z_inc
    z = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    p = exp_z / exp_z.sum(axis=1, keepdims=True)
    return p.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--syms", default="0,1,2,3,4")
    # CE warmstart
    ap.add_argument("--warmstart-rounds", type=int, default=80)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--num-leaves", type=int, default=127)
    ap.add_argument("--min-data-in-leaf", type=int, default=100)
    ap.add_argument("--feature-fraction", type=float, default=0.8)
    ap.add_argument("--bagging-fraction", type=float, default=0.8)
    ap.add_argument("--bagging-freq", type=int, default=5)
    ap.add_argument("--lambda-l2", type=float, default=1.0)
    # PnL stage
    ap.add_argument("--pnl-rounds", type=int, default=400)
    ap.add_argument("--pnl-early-stopping", type=int, default=40)
    ap.add_argument("--pnl-lr", type=float, default=0.02)
    ap.add_argument("--pnl-num-leaves", type=int, default=63)
    ap.add_argument("--pnl-min-data-in-leaf", type=int, default=500)
    ap.add_argument("--beta-entropy", type=float, default=0.005)
    ap.add_argument("--hess-min", type=float, default=1e-3)
    ap.add_argument("--grad-scale", type=float, default=10000.0,
                    help="multiply grad/hess to put on similar scale as CE; tuned for fee=1e-4")
    # Misc
    ap.add_argument("--num-threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out-tag", default="s2")
    args = ap.parse_args()

    H = args.horizon
    target_syms = [int(x) for x in args.syms.split(",")]
    print(f"=== T13 S2 PnL-objective LOSO  H={H}  syms={target_syms} ===", flush=True)

    t0 = time.time()
    train_full = load_split("train")
    val_full = load_split("val")
    test_full = load_split("test")
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)

    feat_names_path = os.path.join(CACHE_DIR, "schemeC_feat_names.txt")
    with open(feat_names_path) as f:
        feat_names = [line.strip() for line in f]
    forbidden = {"date", "sym", "time"}
    leak = forbidden & set(feat_names)
    assert not leak, f"FORBIDDEN features in feat_names: {leak}"
    print(f"feat_dim={len(feat_names)}", flush=True)

    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = f"T13-S2-h{H}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            init_kwargs = dict(
                project="liangwenbei",
                name=run_name,
                config={
                    "model": "lightgbm",
                    "scheme": "C1",
                    "experiment": "T13-S2-pnl-custom-objective",
                    "horizon": H,
                    "n_features": len(feat_names),
                    **{k: v for k, v in vars(args).items() if k != "no_wandb"},
                },
                tags=["T13", "S2", "pnl-loss", f"h{H}"],
            )
            try:
                wandb.init(entity="cjxh21-Tsinghua University", **init_kwargs)
            except Exception as e_team:
                print(f"  WandB team rejected ({e_team!r}); fallback personal", flush=True)
                wandb.init(**init_kwargs)
        except Exception as e:
            print(f"  WandB init failed ({e!r}); skipping", flush=True)
            use_wandb = False

    y_key = f"y{H}"
    mp_th_key = f"mp_t{H}"
    fold_results = {}

    for held in target_syms:
        tag = f"h{H}_held{held}"
        print(f"\n  --- fold {tag} ---", flush=True)
        m_tr = train_full["sym"] != held
        m_va = val_full["sym"] != held
        m_te = test_full["sym"] == held

        X_tr = train_full["X"][m_tr]
        y_tr = train_full[y_key][m_tr].astype(np.int64)
        mp_t_tr = train_full["mp_t"][m_tr]
        mp_th_tr = train_full[mp_th_key][m_tr]

        X_va = val_full["X"][m_va]
        y_va = val_full[y_key][m_va].astype(np.int64)
        mp_t_va = val_full["mp_t"][m_va]
        mp_th_va = val_full[mp_th_key][m_va]

        X_te = test_full["X"][m_te]
        y_te = test_full[y_key][m_te].astype(np.int64)
        mp_t_te = test_full["mp_t"][m_te]
        mp_th_te = test_full[mp_th_key][m_te]

        print(f"    n_train={len(X_tr):,} n_val={len(X_va):,} n_test={len(X_te):,}",
              flush=True)

        booster_ce, booster_pnl, _, _ = fold_train(
            X_tr, y_tr, mp_t_tr, mp_th_tr,
            X_va, y_va, mp_t_va, mp_th_va,
            feat_names, args, log_prefix=f"  [{tag}]",
        )

        # Predict test
        te_prob = predict_with_warmstart(booster_ce, booster_pnl, X_te)
        te_pred = te_prob.argmax(axis=1).astype(np.int8)
        te_metrics = evaluate(f"HELD-OUT TEST h={H} held={held} (warmstart+pnl)",
                              te_pred, y_te, mp_t_te, mp_th_te)

        # Predict val
        va_prob = predict_with_warmstart(booster_ce, booster_pnl, X_va)
        va_pred = va_prob.argmax(axis=1).astype(np.int8)
        va_metrics = evaluate(f"VAL h={H} held={held}",
                              va_pred, y_va, mp_t_va, mp_th_va)

        # Also evaluate CE-only (sanity reference)
        te_prob_ce = booster_ce.predict(X_te).astype(np.float32)
        te_pred_ce = te_prob_ce.argmax(axis=1).astype(np.int8)
        te_metrics_ce = evaluate(f"  (CE-only ref) h={H} held={held}",
                                 te_pred_ce, y_te, mp_t_te, mp_th_te)

        sess_map = {0: "am", 1: "pm"}
        te_df = pd.DataFrame({
            "sym": test_full["sym"][m_te].astype(np.int8),
            "date": test_full["date"][m_te].astype(np.int16),
            "session": np.array([sess_map[s] for s in test_full["sess_idx"][m_te]], dtype=object),
            "t": test_full["t"][m_te].astype(np.int16),
            "true_label": y_te.astype(np.int8),
            "pred_label": te_pred.astype(np.int8),
            "prob_0": te_prob[:, 0].astype(np.float32),
            "prob_1": te_prob[:, 1].astype(np.float32),
            "prob_2": te_prob[:, 2].astype(np.float32),
            "midprice_t": mp_t_te.astype(np.float32),
            "midprice_th": mp_th_te.astype(np.float32),
        })
        pred_path = os.path.join(HERE, f"s2_pred_{tag}.parquet")
        te_df.to_parquet(pred_path, index=False)

        # Save CE booster + PnL booster (need both for inference)
        booster_ce.save_model(os.path.join(HERE, f"s2_ce_{tag}.txt"),
                              num_iteration=booster_ce.num_trees() // NUM_CLASS)
        booster_pnl.save_model(os.path.join(HERE, f"s2_pnl_{tag}.txt"),
                               num_iteration=booster_pnl.best_iteration)

        fold_results[held] = {
            "horizon": H,
            "held_out_sym": held,
            "n_train": int(len(X_tr)),
            "n_val": int(len(X_va)),
            "n_test": int(len(X_te)),
            "ce_warmstart_rounds": int(args.warmstart_rounds),
            "pnl_best_iter": int(booster_pnl.best_iteration),
            "val": va_metrics,
            "held_out": te_metrics,
            "ce_only_held_out": te_metrics_ce,
        }
        if use_wandb:
            wandb.log({
                f"h{H}/fold{held}/pnl_best_iter": int(booster_pnl.best_iteration),
                f"h{H}/fold{held}/val_cum_pnl": va_metrics["cum_pnl"],
                f"h{H}/fold{held}/val_acc": va_metrics["accuracy"],
                f"h{H}/fold{held}/held_cum_pnl": te_metrics["cum_pnl"],
                f"h{H}/fold{held}/held_single_pnl": te_metrics["single_pnl"],
                f"h{H}/fold{held}/held_acc": te_metrics["accuracy"],
                f"h{H}/fold{held}/held_f0_5_macro": te_metrics["f0_5_macro"],
                f"h{H}/fold{held}/ce_only_cum_pnl": te_metrics_ce["cum_pnl"],
            })

    print(f"\n{'='*78}\n=== Aggregate (raw argmax) ===\n{'='*78}", flush=True)
    cums = [fold_results[k]["held_out"]["cum_pnl"] for k in sorted(fold_results)]
    cums_a = np.array(cums)
    cums_ce = [fold_results[k]["ce_only_held_out"]["cum_pnl"] for k in sorted(fold_results)]
    print(f"  S2 (CE→PnL) sum={cums_a.sum():+.4f}  per-fold={[f'{x:+.3f}' for x in cums_a]}",
          flush=True)
    print(f"  CE-only ref sum={sum(cums_ce):+.4f}  per-fold={[f'{x:+.3f}' for x in cums_ce]}",
          flush=True)

    agg = {
        "horizon": H,
        "n_folds": len(cums_a),
        "cum_pnl_per_fold": cums_a.tolist(),
        "cum_pnl_sum": float(cums_a.sum()),
        "cum_pnl_mean": float(cums_a.mean()) if len(cums_a) > 0 else 0.0,
        "cum_pnl_std": float(cums_a.std(ddof=0)) if len(cums_a) > 0 else 0.0,
        "n_pos_folds": int((cums_a > 0).sum()) if len(cums_a) > 0 else 0,
        "ce_only_per_fold": cums_ce,
        "ce_only_sum": float(sum(cums_ce)),
    }

    summary = {
        "scheme": "T13-S2",
        "params": vars(args),
        "horizon": H,
        "fold_results": fold_results,
        "aggregate": agg,
    }
    out_path = os.path.join(HERE, f"loso_results_s2_h{H}.json")
    with open(out_path, "w") as f:
        json.dump(sanitize_for_json(summary), f, indent=2)
    print(f"\nsummary -> {out_path}", flush=True)

    if use_wandb:
        wandb.summary.update({
            f"agg_h{H}/cum_pnl_sum": agg["cum_pnl_sum"],
            f"agg_h{H}/cum_pnl_mean": agg["cum_pnl_mean"],
            f"agg_h{H}/cum_pnl_std": agg["cum_pnl_std"],
            f"agg_h{H}/n_pos_folds": agg["n_pos_folds"],
            f"agg_h{H}/ce_only_sum": agg["ce_only_sum"],
        })
        wandb.finish()


if __name__ == "__main__":
    main()
