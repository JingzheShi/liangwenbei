"""T17: CatBoost LOSO trainer for Scheme C 223-d, horizon h=10.

Mirrors T11's LOSO LightGBM split exactly so OOF predictions are alignable
for ensembling with T11 LightGBM OOF parquets.

For each (seed, fold=held_out_sym) tuple:
    train: (sym != held, dates 0..79)
    val:   (sym != held, dates 80..95)  -- early stopping
    test:  (sym == held, dates 96..119) -- held-out predictions

Outputs (per (seed, fold)):
    loso_pred_cat_h10_seed{S}_held{K}.parquet  -- prob_0/1/2 + meta
    loso_model_cat_h10_seed{S}_held{K}.cbm     -- model on disk

Sym-agnostic: only feeds the 223-d Scheme C feature slice (drops last 3 cols
which are time-encoding tail). No sym/date features.
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

T5B_CACHE = os.path.join(ROOT, "experiments", "T5b_features_multihorizon", "cache")
NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)
N_DROP_TAIL = 3  # match T11/iter_002/iter_004 feature-space (223-d)

# Single seed default + diversity-forcing config for multi-seed mode.
# CatBoost-natural varying axes: depth, l2_leaf_reg, bagging_temperature,
# random_strength, border_count.
SEED_CONFIGS = {
    42:  dict(seed=42,  depth=6, l2_leaf_reg=3.0, bagging_temperature=0.0, random_strength=1.0, border_count=128),
    1:   dict(seed=1,   depth=7, l2_leaf_reg=5.0, bagging_temperature=0.5, random_strength=1.5, border_count=128),
    7:   dict(seed=7,   depth=5, l2_leaf_reg=3.0, bagging_temperature=1.0, random_strength=1.0, border_count=64),
    13:  dict(seed=13,  depth=6, l2_leaf_reg=7.0, bagging_temperature=0.0, random_strength=2.0, border_count=128),
    100: dict(seed=100, depth=8, l2_leaf_reg=3.0, bagging_temperature=0.5, random_strength=1.0, border_count=254),
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


def train_one(H, seed_cfg, held, train_full, val_full, test_full, feat_names, args):
    seed = int(seed_cfg["seed"])
    m_tr = train_full["sym"] != held
    m_va = val_full["sym"] != held
    m_te = test_full["sym"] == held

    X_tr = train_full["X"][m_tr][:, :-N_DROP_TAIL]
    y_tr = train_full[f"y{H}"][m_tr].astype(np.int32)
    X_va = val_full["X"][m_va][:, :-N_DROP_TAIL]
    y_va = val_full[f"y{H}"][m_va].astype(np.int32)
    mp_t_va = val_full["mp_t"][m_va]
    mp_th_va = val_full[f"mp_t{H}"][m_va]
    X_te = test_full["X"][m_te][:, :-N_DROP_TAIL]
    y_te = test_full[f"y{H}"][m_te].astype(np.int64)
    mp_t_te = test_full["mp_t"][m_te]
    mp_th_te = test_full[f"mp_t{H}"][m_te]

    print(f"    n_train={len(X_tr):,} n_val={len(X_va):,} n_test={len(X_te):,} feat_dim={X_tr.shape[1]}",
          flush=True)

    train_pool = Pool(X_tr, y_tr, feature_names=feat_names)
    val_pool = Pool(X_va, y_va, feature_names=feat_names)

    model = CatBoostClassifier(
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=int(seed_cfg["depth"]),
        l2_leaf_reg=float(seed_cfg["l2_leaf_reg"]),
        random_strength=float(seed_cfg["random_strength"]),
        bagging_temperature=float(seed_cfg["bagging_temperature"]),
        border_count=int(seed_cfg["border_count"]),
        auto_class_weights="Balanced",
        task_type="CPU",
        thread_count=args.num_threads,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        early_stopping_rounds=args.early_stopping,
        verbose=200,
        random_seed=seed,
        allow_writing_files=False,
    )

    t_start = time.time()
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    train_time = time.time() - t_start
    best_iter = int(model.get_best_iteration() or model.tree_count_)
    print(f"    trained in {train_time:.1f}s, best_iter={best_iter}", flush=True)

    te_prob = model.predict_proba(X_te).astype(np.float32)
    te_pred = te_prob.argmax(axis=1).astype(np.int8)
    te_metrics = evaluate(f"HELD-OUT TEST h={H} seed={seed} held={held}",
                          te_pred, y_te, mp_t_te, mp_th_te)

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
    pred_path = os.path.join(HERE, f"loso_pred_cat_h{H}_seed{seed}_held{held}.parquet")
    te_df.to_parquet(pred_path, index=False)
    model_path = os.path.join(HERE, f"loso_model_cat_h{H}_seed{seed}_held{held}.cbm")
    model.save_model(model_path)

    return {
        "horizon": H,
        "seed": seed,
        "held_out_sym": held,
        "best_iter": best_iter,
        "train_time_sec": float(train_time),
        "held_out_cum_pnl": float(te_metrics["cum_pnl"]),
        "held_out_acc": float(te_metrics["accuracy"]),
        "held_out_n_active": int(te_metrics["n_predictions_active"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="10")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--syms", default="0,1,2,3,4")
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--early-stopping", type=int, default=50)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--num-threads", type=int, default=16)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out-tag", default="T17")
    args = ap.parse_args()

    horizons = [int(x) for x in args.horizons.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    target_syms = [int(x) for x in args.syms.split(",")]
    for s in seeds:
        if s not in SEED_CONFIGS:
            sys.exit(f"seed {s} not in SEED_CONFIGS")

    print(f"=== T17 CatBoost LOSO Scheme C 223-d, horizons={horizons}, seeds={seeds}, syms={target_syms} ===",
          flush=True)

    progress("loading_caches", horizons=horizons, seeds=seeds)
    t0 = time.time()
    train_full = load_split("train")
    val_full = load_split("val")
    test_full = load_split("test")
    print(f"loaded in {time.time() - t0:.1f}s; "
          f"train={train_full['X'].shape} val={val_full['X'].shape} test={test_full['X'].shape}",
          flush=True)

    feat_names_path = os.path.join(T5B_CACHE, "schemeC_223d_feat_names.txt")
    with open(feat_names_path) as f:
        feat_names = [line.strip() for line in f]
    print(f"feat dim = {len(feat_names)}", flush=True)
    assert len(feat_names) == train_full["X"].shape[1] - N_DROP_TAIL
    forbidden = {"date", "sym", "time"}
    leak = forbidden & set(feat_names)
    assert not leak, f"FORBIDDEN features: {leak}"
    print("  OK: no date/sym/time in features", flush=True)

    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = f"T17-catboost-h10-seeds{','.join(str(s) for s in seeds)}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            init_kwargs = dict(
                project="liangwenbei",
                name=run_name,
                config={
                    "model": "catboost",
                    "scheme": "C-223d",
                    "n_features": len(feat_names),
                    "horizons": horizons,
                    "seeds": seeds,
                    "seed_configs": {str(s): SEED_CONFIGS[s] for s in seeds},
                    **{k: v for k, v in vars(args).items() if k != "no_wandb"},
                },
                tags=["T17", "catboost", "schemeC", "223d", "h10"],
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
    n_total = len(horizons) * len(seeds) * len(target_syms)
    done = 0
    for H in horizons:
        print(f"\n{'='*78}\n=== HORIZON h={H} ===\n{'='*78}", flush=True)
        for s in seeds:
            seed_cfg = SEED_CONFIGS[s]
            print(f"\n  --- seed {s} cfg={seed_cfg} ---", flush=True)
            for held in target_syms:
                done += 1
                tag = f"h{H}_seed{s}_held{held}"
                print(f"\n  [{done}/{n_total}] fold {tag}", flush=True)
                progress("training", horizon=H, seed=s, held=held,
                         done=done, total=n_total)
                r = train_one(H, seed_cfg, held, train_full, val_full, test_full,
                              feat_names, args)
                all_results.append(r)
                if use_wandb:
                    wandb.log({
                        f"h{H}/seed{s}/fold{held}/cum_pnl": r["held_out_cum_pnl"],
                        f"h{H}/seed{s}/fold{held}/acc": r["held_out_acc"],
                        f"h{H}/seed{s}/fold{held}/best_iter": r["best_iter"],
                        f"h{H}/seed{s}/fold{held}/train_time_sec": r["train_time_sec"],
                    })

    summary_path = os.path.join(HERE, f"loso_train_{args.out_tag}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(sanitize_for_json({
            "task": "T17 CatBoost LOSO Scheme C 223-d",
            "horizons": horizons, "seeds": seeds, "target_syms": target_syms,
            "seed_configs": {str(s): SEED_CONFIGS[s] for s in seeds},
            "results": all_results,
            "elapsed_sec": time.time() - t0,
        }), f, indent=2)
    print(f"\nsummary -> {summary_path}", flush=True)
    progress("loso_done", n_runs=len(all_results))

    if use_wandb:
        df = pd.DataFrame(all_results)
        for H in horizons:
            for s in seeds:
                sub = df[(df.horizon == H) & (df.seed == s)]
                wandb.summary[f"h{H}_seed{s}_argmax_sum_cum_pnl"] = float(sub["held_out_cum_pnl"].sum())
        wandb.finish()


if __name__ == "__main__":
    main()
