"""T54: CatBoost LOSO Stage 2 327-d, h_60 only, 5 seeds × 5 folds, aug_a [0.80, 1.20].

Goal: pair iter_007 features (R34 Stage 2, 327 dim after drop_fail+drop_tail) with
CatBoost Plain GPU instead of LightGBM, and check whether the standalone CatBoost
LOSO sum can match/beat:
  * T46 CatBoost Plain Scheme C (226-d) = +12.64
  * iter_007 LightGBM 5-seed Stage 2 = +15.06

Reuses T51 npz cache (schemeM_*.npz, 339-d, drop_fail_extras + drop_tail 3 -> 327-d)
and the T46 train loop (only difference: cache path, slicing, output tag).

For each (seed, held_out_sym) tuple:
    train: (sym != held, dates 0..79) + per-(sample,feat) U[0.80,1.20] scale
           (orig + aug concatenated 1:1)
    val:   (sym != held, dates 80..95)  -- early stopping (NO augment)
    test:  (sym == held, dates 96..119) -- held-out predictions (NO augment)

Outputs (per (seed, fold)):
    cb_stage2_pred_seed{S}_held{K}.parquet   -- full schema for DE thresh
    cb_stage2_model_seed{S}_held{K}.cbm      -- CatBoost native model file

Compliance with CRITICAL_CONSTRAINTS:
  * sym/date NEVER in feature vector
  * Per-sample stateless augment, only at training time
  * sym-agnostic model
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

from catboost import CatBoostClassifier, Pool  # noqa: E402
from src.eval.pnl import _per_horizon_metrics, sanitize_for_json  # noqa: E402

# Reuse T26 build_aug helpers
T26_DIR = os.path.join(ROOT, "experiments", "T26_domain_randomization")
sys.path.insert(0, T26_DIR)
from build_aug import class_balanced_weight  # noqa: E402

T51_DIR = os.path.join(ROOT, "experiments", "T51_r34_stage2")
T51_CACHE = os.path.join(T51_DIR, "cache")
T51_REPORT = os.path.join(T51_DIR, "sym_invariance_report.json")

NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)
BASE_DIM = 226
N_DROP_TAIL = 3  # drop trailing time-encoding cols
H = 60

DEFAULT_SEEDS = [42, 1, 7, 13, 100]


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
    p = os.path.join(T51_CACHE, f"schemeM_{split}.npz")
    print(f"  loading {p} ...", flush=True)
    d = np.load(p)
    return {k: d[k] for k in d.files}


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


def aug_uniform_concat(X_orig, y_orig, lo, hi, rng, num_class=3):
    """Per-(sample, feature) uniform[lo, hi] scale; concatenate orig + aug 1:1."""
    scales = rng.uniform(lo, hi, size=X_orig.shape).astype(X_orig.dtype)
    X_aug = X_orig * scales
    X_merged = np.concatenate([X_orig, X_aug], axis=0)
    y_merged = np.concatenate([y_orig, y_orig], axis=0).astype(np.int32)
    w_merged = class_balanced_weight(y_merged, num_class)
    return X_merged, y_merged, w_merged


def train_one_fold(seed, held, X_tr_full, y_tr_full, sym_tr,
                   X_va_full, y_va_full, sym_va,
                   X_te_full, y_te_full, sym_te,
                   mp_t_te_full, mp_th_te_full,
                   sess_te_full, date_te_full, t_te_full,
                   feat_names, args, boosting_type, tag):
    m_tr = sym_tr != held
    m_va = sym_va != held
    m_te = sym_te == held

    X_tr_o = X_tr_full[m_tr]
    y_tr_o = y_tr_full[m_tr].astype(np.int32)
    X_va = X_va_full[m_va]
    y_va = y_va_full[m_va].astype(np.int32)
    X_te = X_te_full[m_te]
    y_te = y_te_full[m_te].astype(np.int64)
    mp_t_te = mp_t_te_full[m_te]
    mp_th_te = mp_th_te_full[m_te]

    fold_rng = np.random.default_rng(seed * 7919 + held * 17 + 1)
    X_tr, y_tr, sw_tr = aug_uniform_concat(
        X_tr_o, y_tr_o, args.aug_lo, args.aug_hi, fold_rng, NUM_CLASS,
    )
    sw_va = class_balanced_weight(y_va, NUM_CLASS)
    print(
        f"    n_train_orig={len(X_tr_o):,} n_train_used={len(X_tr):,} "
        f"n_val={len(X_va):,} n_test={len(X_te):,} feat_dim={X_tr.shape[1]}",
        flush=True,
    )

    train_pool = Pool(X_tr, y_tr, weight=sw_tr, feature_names=feat_names)
    val_pool = Pool(X_va, y_va, weight=sw_va, feature_names=feat_names)

    model_kwargs = dict(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
        bagging_temperature=args.bagging_temperature,
        random_strength=args.random_strength,
        border_count=args.border_count,
        boosting_type=boosting_type,
        early_stopping_rounds=args.early_stopping,
        random_seed=seed,
        verbose=200,
        allow_writing_files=False,
    )
    if args.use_gpu:
        model_kwargs["task_type"] = "GPU"
        model_kwargs["devices"] = args.devices
    else:
        model_kwargs["task_type"] = "CPU"
        model_kwargs["thread_count"] = args.num_threads

    model = CatBoostClassifier(**model_kwargs)

    t_start = time.time()
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    train_time = time.time() - t_start
    best_iter = int(model.get_best_iteration() or model.tree_count_)
    print(
        f"    trained in {train_time:.1f}s, best_iter={best_iter}, "
        f"tree_count={model.tree_count_}",
        flush=True,
    )

    te_prob = model.predict_proba(X_te).astype(np.float32)
    te_pred = te_prob.argmax(axis=1).astype(np.int8)
    te_metrics = evaluate_metrics(
        f"HELD-OUT TEST {tag} seed={seed} held={held}",
        te_pred, y_te, mp_t_te, mp_th_te,
    )

    sess_map = {0: "am", 1: "pm"}
    te_df = pd.DataFrame({
        "sym": sym_te[m_te].astype(np.int8),
        "date": date_te_full[m_te].astype(np.int16),
        "session": np.array(
            [sess_map[s] for s in sess_te_full[m_te]], dtype=object
        ),
        "t": t_te_full[m_te].astype(np.int16),
        "true_label": y_te.astype(np.int8),
        "pred_label": te_pred.astype(np.int8),
        "prob_0": te_prob[:, 0].astype(np.float32),
        "prob_1": te_prob[:, 1].astype(np.float32),
        "prob_2": te_prob[:, 2].astype(np.float32),
        "midprice_t": mp_t_te.astype(np.float32),
        "midprice_th": mp_th_te.astype(np.float32),
    })
    pred_path = os.path.join(HERE, f"{tag}_pred_seed{seed}_held{held}.parquet")
    te_df.to_parquet(pred_path, index=False)

    model_path = os.path.join(HERE, f"{tag}_model_seed{seed}_held{held}.cbm")
    model.save_model(model_path)

    return {
        "tag": tag, "seed": seed, "held_out_sym": held,
        "boosting_type": boosting_type,
        "n_train_orig": int(len(X_tr_o)),
        "n_train_used": int(len(X_tr)),
        "n_val": int(len(X_va)),
        "n_test": int(len(X_te)),
        "best_iter": best_iter,
        "tree_count": int(model.tree_count_),
        "train_time_sec": float(train_time),
        "held_cum_pnl": float(te_metrics["cum_pnl"]),
        "held_single_pnl": float(te_metrics["single_pnl"]),
        "held_acc": float(te_metrics["accuracy"]),
        "held_n_active": int(te_metrics["n_predictions_active"]),
    }


def build_keep_idx(extra_names):
    """327 = 339 - 9 (FAIL extras) - 3 (trailing time enc).

    Mirrors T51 train_loso_h60.py with --drop-fail-extras --n-drop-tail 3.
    """
    with open(T51_REPORT) as f:
        report = json.load(f)
    fail_names = report["summary"]["fail_names"]
    fail_local = [extra_names.index(n) for n in fail_names]
    fail_global = [BASE_DIM + i for i in fail_local]
    drop_set = set(fail_global)
    for i in range(N_DROP_TAIL):
        drop_set.add(BASE_DIM - 1 - i)
    total_dim = BASE_DIM + len(extra_names)
    keep_idx = np.array(
        [i for i in range(total_dim) if i not in drop_set],
        dtype=np.int64,
    )
    return keep_idx, fail_names, sorted(drop_set)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--syms", default="0,1,2,3,4")
    ap.add_argument("--aug-lo", type=float, default=0.80)
    ap.add_argument("--aug-hi", type=float, default=1.20)
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--early-stopping", type=int, default=100)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--l2-leaf-reg", type=float, default=3.0)
    ap.add_argument("--bagging-temperature", type=float, default=0.0)
    ap.add_argument("--random-strength", type=float, default=1.0)
    ap.add_argument("--border-count", type=int, default=128)
    ap.add_argument("--boosting-type", default="Plain",
                    choices=("Ordered", "Plain"),
                    help="GPU CatBoost requires Plain.")
    ap.add_argument("--tag", default="cb_stage2",
                    help="Output filename tag.")
    ap.add_argument("--num-threads", type=int, default=18)
    ap.add_argument("--devices", default="0")
    ap.add_argument("--use-gpu", action="store_true", default=True)
    ap.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    target_syms = [int(x) for x in args.syms.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    print(
        f"=== T54 CatBoost h_{H} LOSO Stage2: seeds={seeds} syms={target_syms} "
        f"aug=[{args.aug_lo},{args.aug_hi}] boosting={args.boosting_type} tag={args.tag} ===",
        flush=True,
    )

    progress("loading_caches", seeds=seeds, syms=target_syms)
    t0 = time.time()
    train_full = load_split("train")
    val_full = load_split("val")
    test_full = load_split("test")
    print(
        f"loaded in {time.time() - t0:.1f}s; "
        f"train={train_full['X'].shape} val={val_full['X'].shape} test={test_full['X'].shape}",
        flush=True,
    )

    # Slice 339 -> 327 (drop FAIL extras + tail time-encoding)
    extras_path = os.path.join(T51_CACHE, "schemeM_extra_feat_names.txt")
    with open(extras_path) as f:
        extra_names = [ln.strip() for ln in f if ln.strip()]
    keep_idx, fail_names, drop_global = build_keep_idx(extra_names)
    feat_dim = len(keep_idx)
    print(f"  fail_extras={fail_names}", flush=True)
    print(f"  drop_global={drop_global}  feat_dim={feat_dim} (expect 327)", flush=True)
    assert feat_dim == 327, f"expected 327-d; got {feat_dim}"

    X_tr_327 = train_full["X"][:, keep_idx].astype(np.float32, copy=False)
    X_va_327 = val_full["X"][:, keep_idx].astype(np.float32, copy=False)
    X_te_327 = test_full["X"][:, keep_idx].astype(np.float32, copy=False)

    feat_names_path = os.path.join(T51_CACHE, "schemeM_feat_names.txt")
    with open(feat_names_path) as f:
        all_feat_names = [ln.strip() for ln in f if ln.strip()]
    feat_names = [all_feat_names[i] for i in keep_idx]
    forbidden = {"date", "sym", "time"}
    leak = forbidden & set(feat_names)
    assert not leak, f"FORBIDDEN features in feat_names: {leak}"
    assert len(feat_names) == feat_dim
    print(f"  feat_dim={len(feat_names)}, no date/sym/time. OK.", flush=True)

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

    # WandB
    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = (
                f"T54-cb-stage2-h{H}-{args.tag}-"
                f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
            init_kwargs = dict(
                project="liangwenbei",
                name=run_name,
                config={
                    "model": "catboost",
                    "scheme": "M-327d (R34 Stage 2)",
                    "horizon": H,
                    "seeds": seeds,
                    "aug_lo": args.aug_lo, "aug_hi": args.aug_hi,
                    **{k: v for k, v in vars(args).items() if k != "no_wandb"},
                },
                tags=["T54", "catboost", "schemeM", "327d", f"h{H}", args.tag],
            )
            try:
                wandb.init(entity="cjxh21-Tsinghua University", **init_kwargs)
            except Exception as e_team:
                print(f"  WandB team rejected ({e_team!r}); fallback personal", flush=True)
                wandb.init(**init_kwargs)
        except Exception as e:
            print(f"  WandB init failed ({e!r}); skipping", flush=True)
            use_wandb = False

    all_results = []
    n_total = len(seeds) * len(target_syms)
    done = 0
    for s in seeds:
        print(f"\n{'='*78}\n=== SEED {s} ({args.boosting_type}) ===\n{'='*78}", flush=True)
        for held in target_syms:
            done += 1
            print(f"\n  [{done}/{n_total}] tag={args.tag} seed={s} held={held}", flush=True)
            progress("training", tag=args.tag, seed=s, held=held,
                     done=done, total=n_total,
                     elapsed_min=round((time.time() - t0) / 60.0, 2))
            r = train_one_fold(
                s, held,
                X_tr_327, y_tr, sym_tr,
                X_va_327, y_va, sym_va,
                X_te_327, y_te, sym_te,
                mp_t_te, mp_th_te,
                sess_te, date_te, t_te,
                feat_names, args, args.boosting_type, args.tag,
            )
            all_results.append(r)
            if use_wandb:
                wandb.log({
                    f"seed{s}/fold{held}/cum_pnl": r["held_cum_pnl"],
                    f"seed{s}/fold{held}/acc": r["held_acc"],
                    f"seed{s}/fold{held}/best_iter": r["best_iter"],
                    f"seed{s}/fold{held}/train_time_sec": r["train_time_sec"],
                    f"seed{s}/fold{held}/n_active": r["held_n_active"],
                })
            interim = {
                "task": f"T54 CatBoost Stage2 h_{H} LOSO ({args.boosting_type}, tag={args.tag})",
                "horizon": H,
                "feat_dim": feat_dim,
                "seeds": seeds,
                "syms": target_syms,
                "params": {k: v for k, v in vars(args).items() if k != "no_wandb"},
                "results": all_results,
                "elapsed_sec": time.time() - t0,
            }
            with open(os.path.join(HERE, f"results_{args.tag}.json"), "w") as f:
                json.dump(sanitize_for_json(interim), f, indent=2)

    df = pd.DataFrame(all_results)
    print(f"\n{'='*78}\n=== AGGREGATE h={H} (raw argmax per fold) ===\n{'='*78}", flush=True)
    agg = []
    for s in seeds:
        sub = df[df.seed == s]
        cums = sub["held_cum_pnl"].values
        rec = {
            "seed": s, "n_folds": int(len(cums)),
            "cum_pnl_per_fold": cums.tolist(),
            "cum_pnl_sum": float(cums.sum()),
            "cum_pnl_mean": float(cums.mean()) if len(cums) else 0.0,
            "n_pos_folds": int((cums > 0).sum()),
        }
        agg.append(rec)
        print(
            f"  seed{s}: sum={rec['cum_pnl_sum']:+.4f} "
            f"mean={rec['cum_pnl_mean']:+.4f} pos={rec['n_pos_folds']}/{rec['n_folds']} "
            f"per_fold={[f'{x:+.3f}' for x in cums]}",
            flush=True,
        )

    out = {
        "task": f"T54 CatBoost Stage2 h_{H} LOSO ({args.boosting_type}, tag={args.tag})",
        "horizon": H,
        "feat_dim": feat_dim,
        "seeds": seeds,
        "syms": target_syms,
        "boosting_type": args.boosting_type,
        "params": {k: v for k, v in vars(args).items() if k != "no_wandb"},
        "results": all_results,
        "agg_by_seed": agg,
        "elapsed_sec": time.time() - t0,
    }
    with open(os.path.join(HERE, f"results_{args.tag}.json"), "w") as f:
        json.dump(sanitize_for_json(out), f, indent=2)
    print(f"\nALL DONE in {out['elapsed_sec']:.1f}s. -> results_{args.tag}.json", flush=True)
    progress("loso_done", tag=args.tag, elapsed_sec=out["elapsed_sec"])

    if use_wandb:
        for s in seeds:
            sub = df[df.seed == s]
            wandb.summary[f"seed{s}_argmax_sum_cum_pnl"] = float(sub["held_cum_pnl"].sum())
        wandb.summary["overall_argmax_sum_cum_pnl"] = float(df["held_cum_pnl"].sum())
        wandb.finish()


if __name__ == "__main__":
    main()
