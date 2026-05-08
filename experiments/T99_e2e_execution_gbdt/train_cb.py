"""T99: CatBoost regression on Δmid_norm with various built-in loss functions.

Supported loss_function values:
  - RMSE          (baseline = T89)
  - MAE
  - Quantile:alpha=0.5
  - Huber:delta=...

Plus optional custom eval_metric for early-stop selection (PnL-based).
Pipeline matches T89: schemeP cache 359-dim, V4 split, aug_a, class_balanced_weight.

The PnL early-stop variant uses CatBoost's user_defined_metric capability
(get_metric() approach via 'PythonClass' style). For simplicity and CB-GPU
compatibility, we instead post-hoc track best_iter by re-evaluating PnL on val
across train iterations using staged predictions, choosing the iter with max
val PnL — equivalent to "select model on PnL, not RMSE".
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

from catboost import CatBoostRegressor, Pool  # noqa: E402

T26_DIR = os.path.join(ROOT, "experiments", "T26_domain_randomization")
sys.path.insert(0, T26_DIR)
from build_aug import aug_a_scale, class_balanced_weight  # noqa: E402

T68_DIR = os.path.join(ROOT, "experiments", "T68_stage5_features")
CACHE_DIR = os.path.join(T68_DIR, "cache")

NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001

# Same SEED_CONFIGS as T89
SEED_CONFIGS = {
    42:  dict(seed=42,  depth=6, l2_leaf_reg=3.0,  rsm=0.8,  subsample=0.8),
    1:   dict(seed=1,   depth=6, l2_leaf_reg=1.0,  rsm=0.6,  subsample=0.7),
    7:   dict(seed=7,   depth=7, l2_leaf_reg=2.0,  rsm=0.7,  subsample=0.85),
    13:  dict(seed=13,  depth=8, l2_leaf_reg=0.5,  rsm=0.5,  subsample=0.6),
    100: dict(seed=100, depth=5, l2_leaf_reg=5.0,  rsm=0.4,  subsample=0.5),
}

T59_FAIL_NAMES = [
    "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
    "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
    "qrank_W100_cumspread",
    "kyle_lam_W50", "kyle_lam_W100",
    "roll_eff_spr_ratio_W100",
]
STAGE5_FAIL_NAMES = ["liq_asym_top5_W5"]
DROP_NAMES = T59_FAIL_NAMES + STAGE5_FAIL_NAMES


def progress(step, **extra):
    p = os.path.join(HERE, "worker-progress.json")
    payload = {"status": "running", "step": step,
               "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               **extra}
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


def load_split(split):
    p = os.path.join(CACHE_DIR, f"schemeP_{split}.npz")
    print(f"  loading {p} ...", flush=True)
    d = np.load(p)
    return {k: d[k] for k in d.files}


def regr_target(mp_t, mp_th):
    return ((mp_th.astype(np.float64) - mp_t.astype(np.float64))
            / (mp_t.astype(np.float64) + 1.0)).astype(np.float32)


def vectorized_pnl(pred_action, mp_t, mp_th):
    side = pred_action.astype(np.float64) - 1.0
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee_pnl) / denom


def build_v4_split(train_full, slicer):
    date_tr = train_full["date"]
    m_va = date_tr >= 76
    m_t = ~m_va
    X_tr = train_full["X"][m_t][:, slicer].astype(np.float32, copy=False)
    y_cls_tr = train_full["y60"][m_t].astype(np.int64)
    y_regr_tr = regr_target(train_full["mp_t"][m_t], train_full["mp_t60"][m_t])

    X_va = train_full["X"][m_va][:, slicer].astype(np.float32, copy=False)
    y_cls_va = train_full["y60"][m_va].astype(np.int64)
    y_regr_va = regr_target(train_full["mp_t"][m_va], train_full["mp_t60"][m_va])
    mp_t_va = train_full["mp_t"][m_va]
    mp_th_va = train_full["mp_t60"][m_va]

    info = {
        "strategy": "V4",
        "train_dates": "0-75",
        "val_dates": "76-79 (walk-forward last 5%)",
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_va)),
    }
    return (X_tr, y_cls_tr, y_regr_tr,
            X_va, y_cls_va, y_regr_va, mp_t_va, mp_th_va, info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--loss-function", default="RMSE",
                    choices=["RMSE", "MAE", "Quantile", "Huber"])
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="For Quantile: q. For Huber: delta.")
    ap.add_argument("--pnl-early-stop", action="store_true",
                    help="Track val PnL across iterations and pick best on PnL.")
    ap.add_argument("--aug-ratio", type=float, default=1.0)
    ap.add_argument("--aug-lo", type=float, default=0.80)
    ap.add_argument("--aug-hi", type=float, default=1.20)
    ap.add_argument("--iterations", type=int, default=1500)
    ap.add_argument("--early-stopping", type=int, default=80)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--use-gpu", action="store_true", default=True)
    ap.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    H = args.horizon
    seed = args.seed
    lf = args.loss_function

    if args.tag:
        tag = args.tag
    elif lf in ("Quantile", "Huber"):
        tag = f"cb_{lf.lower()}_a{args.alpha:g}"
    else:
        tag = f"cb_{lf.lower()}"
    if args.pnl_early_stop:
        tag = tag + "_pnles"

    print(f"=== T99 CB seed={seed} h={H} loss={lf} tag={tag} ===", flush=True)

    progress("loading_caches", seed=seed, tag=tag)
    t0 = time.time()
    train_full = load_split("train")
    test_full = load_split("test")
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    feat_names_path = os.path.join(CACHE_DIR, "schemeP_feat_names.txt")
    with open(feat_names_path) as f:
        all_feat_names = [line.strip() for line in f]
    name_to_idx = {n: i for i, n in enumerate(all_feat_names)}
    drop_idx = set()
    for n in DROP_NAMES:
        if n in name_to_idx:
            drop_idx.add(name_to_idx[n])
    keep_idx = np.array(
        [i for i in range(train_full["X"].shape[1]) if i not in drop_idx],
        dtype=np.int64,
    )
    feat_dim = len(keep_idx)
    feat_names = [all_feat_names[i] for i in keep_idx]
    forbidden = {"date", "sym", "time"}
    assert not (forbidden & set(feat_names))

    (X_tr, y_cls_tr, y_regr_tr,
     X_va, y_cls_va, y_regr_va, mp_t_va, mp_th_va, info) = build_v4_split(train_full, keep_idx)
    print(f"  feat_dim={feat_dim}, {info}", flush=True)

    X_te = test_full["X"][:, keep_idx].astype(np.float32, copy=False)
    y_cls_te = test_full[f"y{H}"].astype(np.int64)
    y_regr_te = regr_target(test_full["mp_t"], test_full[f"mp_t{H}"])
    mp_t_te = test_full["mp_t"]
    mp_th_te = test_full[f"mp_t{H}"]

    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = (f"T99-cb-{tag}-seed{seed}-"
                        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}")
            wandb.init(
                project="liangwenbei",
                entity="cjxh21-Tsinghua University",
                name=run_name,
                config={
                    "model": "catboost",
                    "task": "T99_e2e_execution_gbdt",
                    "loss_function": lf,
                    "alpha": args.alpha,
                    "split_info": info,
                    "n_features": feat_dim,
                    "horizon": H,
                    "seed": seed,
                    "pnl_early_stop": args.pnl_early_stop,
                    **{k: v for k, v in vars(args).items()},
                },
                tags=["T99", "cb", lf, f"h{H}"],
            )
        except Exception as e:
            print(f"  WandB init failed ({e!r}); skipping", flush=True)
            use_wandb = False

    cfg = SEED_CONFIGS[seed]
    print(f"  cfg={cfg}", flush=True)
    progress("training", seed=seed, tag=tag)

    seed_rng = np.random.default_rng(seed * 7919 + 1)
    n_aug = int(round(args.aug_ratio * len(X_tr)))
    if n_aug > 0:
        if n_aug == len(X_tr):
            aug_idx = np.arange(len(X_tr))
        else:
            aug_idx = seed_rng.integers(0, len(X_tr), size=n_aug)
        X_aug = aug_a_scale(X_tr[aug_idx], seed_rng,
                            lo=args.aug_lo, hi=args.aug_hi)
        y_cls_aug = y_cls_tr[aug_idx]
        y_regr_aug = y_regr_tr[aug_idx]
        X_tr_full = np.concatenate([X_tr, X_aug], axis=0)
        y_cls_tr_full = np.concatenate([y_cls_tr, y_cls_aug], axis=0)
        y_regr_tr_full = np.concatenate([y_regr_tr, y_regr_aug], axis=0)
    else:
        X_tr_full = X_tr
        y_cls_tr_full = y_cls_tr
        y_regr_tr_full = y_regr_tr

    sw_tr = class_balanced_weight(y_cls_tr_full, num_class=NUM_CLASS)
    sw_va = class_balanced_weight(y_cls_va, num_class=NUM_CLASS)
    print(f"  n_train_used={len(X_tr_full):,}  n_val={len(X_va):,}", flush=True)

    train_pool = Pool(X_tr_full, label=y_regr_tr_full, weight=sw_tr,
                      feature_names=feat_names)
    val_pool = Pool(X_va, label=y_regr_va, weight=sw_va,
                    feature_names=feat_names)

    if lf == "Quantile":
        loss_str = f"Quantile:alpha={args.alpha:g}"
    elif lf == "Huber":
        loss_str = f"Huber:delta={args.alpha:g}"
    else:
        loss_str = lf

    # CB GPU + MAE/Quantile: GPU implementation early-stops at iter 0
    # under weighted MAE/Quantile (probably due to unweighted GPU eval kernel).
    # Huber works on GPU. Use CPU only for MAE/Quantile.
    use_gpu = args.use_gpu and (lf in ("RMSE", "Huber"))
    eval_str = "RMSE"  # Always early-stop on RMSE — comparable across losses

    cb_params = dict(
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=cfg["depth"],
        l2_leaf_reg=cfg["l2_leaf_reg"],
        loss_function=loss_str,
        eval_metric=eval_str,
        random_seed=seed,
        rsm=cfg["rsm"] if not use_gpu else 1.0,
        bootstrap_type="Bernoulli",
        subsample=cfg["subsample"],
        verbose=200,
        early_stopping_rounds=args.early_stopping,
        allow_writing_files=False,
    )
    if use_gpu:
        cb_params["task_type"] = "GPU"
        cb_params["devices"] = "0"
    else:
        cb_params["thread_count"] = 18
        print(f"  CB CPU mode (loss_function={loss_str})", flush=True)

    model = CatBoostRegressor(**cb_params)
    t_start = time.time()
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    train_time = time.time() - t_start
    best_iter_cb = int(model.tree_count_)
    print(f"  trained in {train_time:.1f}s, best_iter (CB)={best_iter_cb}", flush=True)

    # Optional: re-pick best iteration based on PnL on val (compute staged preds)
    pnl_best_iter = best_iter_cb
    pnl_search_log = None
    if args.pnl_early_stop:
        # Stride to avoid memory blow up
        stride = max(1, best_iter_cb // 100)
        idx_list = list(range(stride, best_iter_cb + 1, stride))
        if best_iter_cb not in idx_list:
            idx_list.append(best_iter_cb)
        print(f"  PnL early-stop: scanning {len(idx_list)} iters (stride={stride})", flush=True)
        fee_thr = 2.0 * FEE
        pnls = []
        for it in idx_list:
            pv = model.predict(X_va, ntree_end=it).astype(np.float32)
            a = np.full(len(pv), 1, dtype=np.int8)
            a[pv > fee_thr] = 2
            a[pv < -fee_thr] = 0
            cum = float(vectorized_pnl(a, mp_t_va, mp_th_va).sum())
            pnls.append(cum)
        best_idx = int(np.argmax(pnls))
        pnl_best_iter = idx_list[best_idx]
        pnl_search_log = list(zip(idx_list, pnls))
        print(f"  PnL-best iter = {pnl_best_iter}  val_pnl={pnls[best_idx]:+.4f}",
              flush=True)

    eff_best_iter = pnl_best_iter

    model_path = os.path.join(HERE, f"model_T99_{tag}_seed{seed}.cbm")
    model.save_model(model_path)
    # Note: catboost .cbm preserves all iterations; we just use ntree_end during predict.

    # Eval at chosen iter
    va_pred = model.predict(X_va, ntree_end=eff_best_iter).astype(np.float32)
    va_mse = float(((va_pred - y_regr_va) ** 2).mean())
    va_mae = float(np.abs(va_pred - y_regr_va).mean())
    va_corr = float(np.corrcoef(va_pred, y_regr_va)[0, 1]) if va_pred.std() > 0 else 0.0
    print(f"  VAL @iter={eff_best_iter}: mse={va_mse:.7f} mae={va_mae:.7f} corr={va_corr:.4f}",
          flush=True)

    te_pred = model.predict(X_te, ntree_end=eff_best_iter).astype(np.float32)
    te_mse = float(((te_pred - y_regr_te) ** 2).mean())
    te_mae = float(np.abs(te_pred - y_regr_te).mean())
    te_corr = float(np.corrcoef(te_pred, y_regr_te)[0, 1]) if te_pred.std() > 0 else 0.0
    print(f"  TEST: mse={te_mse:.7f} mae={te_mae:.7f} corr={te_corr:.4f}", flush=True)

    sess_map = {0: "am", 1: "pm"}
    te_df = pd.DataFrame({
        "sym": test_full["sym"].astype(np.int8),
        "date": test_full["date"].astype(np.int16),
        "session": np.array([sess_map[int(s)] for s in test_full["sess_idx"]], dtype=object),
        "t": test_full["t"].astype(np.int16),
        "true_label": y_cls_te.astype(np.int8),
        "true_dmid_norm": y_regr_te.astype(np.float32),
        "pred_dmid_norm": te_pred.astype(np.float32),
        "midprice_t": mp_t_te.astype(np.float32),
        "midprice_th": mp_th_te.astype(np.float32),
    })
    pred_path = os.path.join(HERE, f"pred_T99_{tag}_seed{seed}.parquet")
    te_df.to_parquet(pred_path, index=False)
    print(f"  saved {pred_path}", flush=True)

    fee_thr = 2.0 * FEE
    pa = np.full(len(te_pred), 1, dtype=np.int8)
    pa[te_pred > fee_thr] = 2
    pa[te_pred < -fee_thr] = 0
    pnl = vectorized_pnl(pa, mp_t_te, mp_th_te)
    cum_pnl = float(pnl.sum())
    n_active = int((pa != 1).sum())
    print(f"  TEST EV-gate k=1: cum_pnl={cum_pnl:+.4f} n_active={n_active:,}", flush=True)

    summary = {
        "task": f"T99 CB {lf} seed={seed}",
        "seed": seed,
        "horizon": H,
        "loss_function": lf,
        "alpha": args.alpha,
        "tag": tag,
        "n_features": feat_dim,
        "best_iter_cb_default": best_iter_cb,
        "best_iter_pnl": pnl_best_iter,
        "pnl_early_stop": args.pnl_early_stop,
        "pnl_search_log": pnl_search_log,
        "train_time_sec": float(train_time),
        "val": {"mse": va_mse, "mae": va_mae, "corr": va_corr},
        "test": {"mse": te_mse, "mae": te_mae, "corr": te_corr,
                 "ev_gate_k1_cum_pnl": cum_pnl, "ev_gate_k1_n_active": n_active},
        "params": vars(args),
        "cb_params": {k: v for k, v in cb_params.items() if k != "verbose"},
    }
    out_path = os.path.join(HERE, f"summary_T99_{tag}_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsummary -> {out_path}", flush=True)

    if use_wandb:
        wandb.log({
            "best_iter_cb": best_iter_cb,
            "best_iter_pnl": pnl_best_iter,
            "train_time_sec": float(train_time),
            "val_mse": va_mse, "val_mae": va_mae, "val_corr": va_corr,
            "test_mse": te_mse, "test_mae": te_mae, "test_corr": te_corr,
            "test_ev_gate_k1_cum_pnl": cum_pnl,
            "test_ev_gate_k1_n_active": n_active,
        })
        wandb.finish()
    progress("done", seed=seed, tag=tag,
             best_iter=best_iter_cb, val_mse=va_mse,
             test_corr=te_corr, test_ev_gate_k1_cum_pnl=cum_pnl)


if __name__ == "__main__":
    main()
