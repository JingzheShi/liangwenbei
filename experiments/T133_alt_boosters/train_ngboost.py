"""T133: NGBoost (Normal dist) on T75 LGB L2 baseline schema.

NGBoost is CPU-only and ~10-50x slower than LightGBM. To fit in budget:
  - subsample training to ~200k rows (stratified per-sym × class)
  - n_estimators=200, learning_rate=0.05
  - minibatch_frac=0.5 (additional in-tree subsampling)

Predicts mean of Normal distribution. Saves predictions in same format as LGB
variants for downstream DE LOSO eval.
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

from ngboost import NGBRegressor  # noqa: E402
from ngboost.distns import Normal  # noqa: E402
from sklearn.tree import DecisionTreeRegressor  # noqa: E402

T26_DIR = os.path.join(ROOT, "experiments", "T26_domain_randomization")
sys.path.insert(0, T26_DIR)
from build_aug import class_balanced_weight  # noqa: E402

T68_DIR = os.path.join(ROOT, "experiments", "T68_stage5_features")
SCHEME_CACHE = os.path.join(T68_DIR, "cache")

NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001
H = 60

T59_FAIL = ["dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
            "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
            "qrank_W100_cumspread", "kyle_lam_W50", "kyle_lam_W100",
            "roll_eff_spr_ratio_W100"]
STAGE5_FAIL = ["liq_asym_top5_W5"]
DROP_NAMES = T59_FAIL + STAGE5_FAIL


def progress(step, **extra):
    p = os.path.join(HERE, "worker-progress.json")
    payload = {"status": "running", "step": step,
               "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               **extra}
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


def load_scheme(split):
    p = os.path.join(SCHEME_CACHE, f"schemeP_{split}.npz")
    d = np.load(p)
    return {k: d[k] for k in d.files}


def regr_target(mp_t, mp_th):
    return ((mp_th.astype(np.float64) - mp_t.astype(np.float64))
            / (mp_t.astype(np.float64) + 1.0)).astype(np.float32)


def predict_chunked(model, X, batch=50_000):
    chunks = []
    for s in range(0, len(X), batch):
        chunks.append(model.predict(X[s:s+batch]).astype(np.float32))
    return np.concatenate(chunks, axis=0)


def vectorized_pnl(action, mp_t, mp_th):
    side = action.astype(np.float64) - 1.0
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee_pnl) / denom


def stratified_subsample(rng, n_total, sym_arr, y_cls, n_samples):
    """Per-sym × class stratified subsample to size ~n_samples."""
    indices = []
    n_per_strat = n_samples // (len(SYMS) * NUM_CLASS)
    for s in SYMS:
        for c in range(NUM_CLASS):
            mask = (sym_arr == s) & (y_cls == c)
            idx = np.flatnonzero(mask)
            if len(idx) <= n_per_strat:
                indices.append(idx)
            else:
                pick = rng.choice(idx, size=n_per_strat, replace=False)
                indices.append(pick)
    return np.sort(np.concatenate(indices))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--minibatch-frac", type=float, default=0.5)
    ap.add_argument("--n-subsample", type=int, default=200_000)
    ap.add_argument("--tree-max-depth", type=int, default=5)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    seed = args.seed
    variant = "ngboost"
    tag = f"{variant}_seed{seed}"
    print(f"=== T133 NGBoost seed={seed} ===", flush=True)
    progress("loading", variant=variant, seed=seed)

    t0 = time.time()
    train_full = load_scheme("train")
    test_full = load_scheme("test")
    print(f"  loaded schemeP in {time.time()-t0:.1f}s", flush=True)

    feat_names_path = os.path.join(SCHEME_CACHE, "schemeP_feat_names.txt")
    with open(feat_names_path) as f:
        all_names = [l.strip() for l in f if l.strip()]
    total_dim = train_full["X"].shape[1]
    name_to_idx = {n: i for i, n in enumerate(all_names)}
    drop_idx = {name_to_idx[n] for n in DROP_NAMES if n in name_to_idx}
    keep_idx = np.array([i for i in range(total_dim) if i not in drop_idx], dtype=np.int64)
    feat_names = [all_names[i] for i in keep_idx]
    forbidden = {"date", "sym", "time"}
    leak = forbidden & set(feat_names)
    assert not leak, f"leak: {leak}"

    # build train/val split (V4: train 0-75, val 76-79)
    date_tr = train_full["date"]
    m_va = date_tr >= 76
    m_t = ~m_va

    sym_train = train_full["sym"][m_t]
    y_cls_tr_full = train_full["y60"][m_t].astype(np.int64)

    rng = np.random.default_rng(seed)
    sub_idx = stratified_subsample(rng, m_t.sum(), sym_train, y_cls_tr_full, args.n_subsample)
    print(f"  stratified subsample: {len(sub_idx):,}/{m_t.sum():,} rows", flush=True)

    X_tr_all = train_full["X"][m_t][:, keep_idx].astype(np.float32, copy=False)
    X_tr = X_tr_all[sub_idx]
    del X_tr_all
    X_va = train_full["X"][m_va][:, keep_idx].astype(np.float32, copy=False)

    y_cls_tr = y_cls_tr_full[sub_idx]
    y_regr_tr = regr_target(train_full["mp_t"][m_t], train_full["mp_t60"][m_t])[sub_idx]
    y_cls_va = train_full["y60"][m_va].astype(np.int64)
    y_regr_va = regr_target(train_full["mp_t"][m_va], train_full["mp_t60"][m_va])
    print(f"  X_tr{X_tr.shape}  X_va{X_va.shape}", flush=True)

    X_te = test_full["X"][:, keep_idx].astype(np.float32, copy=False)
    y_regr_te = regr_target(test_full["mp_t"], test_full["mp_t60"])
    sym_te = test_full["sym"]
    mp_t_te = test_full["mp_t"]
    mp_th_te = test_full["mp_t60"]

    sw_tr = class_balanced_weight(y_cls_tr, num_class=NUM_CLASS)

    # WandB
    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = f"T133-{tag}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            wandb.init(project="liangwenbei", entity="cjxh21-Tsinghua University",
                       name=run_name,
                       config={"variant": variant, "seed": seed,
                               "n_estimators": args.n_estimators,
                               "learning_rate": args.learning_rate,
                               "minibatch_frac": args.minibatch_frac,
                               "n_subsample": args.n_subsample,
                               "tree_max_depth": args.tree_max_depth,
                               "horizon": H},
                       tags=["T133", "alt_boosters", variant, f"seed{seed}"])
        except Exception as e:
            print(f"  WandB init failed ({e!r}); skipping", flush=True)
            use_wandb = False

    progress("training", variant=variant, seed=seed)

    base_tree = DecisionTreeRegressor(
        criterion="friedman_mse",
        max_depth=args.tree_max_depth,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=seed,
    )

    model = NGBRegressor(
        Dist=Normal,
        Base=base_tree,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        minibatch_frac=args.minibatch_frac,
        col_sample=0.8,
        natural_gradient=True,
        random_state=seed,
        tol=1e-5,
        verbose=False,
        verbose_eval=20,
    )

    t1 = time.time()
    model.fit(X_tr, y_regr_tr.astype(np.float64),
              X_val=X_va, Y_val=y_regr_va.astype(np.float64),
              sample_weight=sw_tr.astype(np.float64),
              early_stopping_rounds=50)
    train_time = time.time() - t1
    best_iter = int(model.best_val_loss_itr) if hasattr(model, "best_val_loss_itr") and model.best_val_loss_itr else args.n_estimators
    print(f"  trained in {train_time:.1f}s, best_iter={best_iter}", flush=True)

    # Save model
    import pickle
    model_path = os.path.join(HERE, f"model_T133_{tag}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    va_pred = predict_chunked(model, X_va)
    va_mae = float(np.abs(va_pred - y_regr_va).mean())
    va_mse = float(((va_pred - y_regr_va) ** 2).mean())
    va_corr = float(np.corrcoef(va_pred, y_regr_va)[0, 1])
    print(f"  VAL mse={va_mse:.7f} mae={va_mae:.7f} corr={va_corr:.4f}", flush=True)

    te_pred = predict_chunked(model, X_te)
    te_mae = float(np.abs(te_pred - y_regr_te).mean())
    te_corr = float(np.corrcoef(te_pred, y_regr_te)[0, 1])
    print(f"  TEST mae={te_mae:.7f} corr={te_corr:.4f}", flush=True)

    sess_map = {0: "am", 1: "pm"}
    df = pd.DataFrame({
        "sym": sym_te.astype(np.int8),
        "date": test_full["date"].astype(np.int16),
        "session": np.array([sess_map[int(s)] for s in test_full["sess_idx"]],
                            dtype=object),
        "t": test_full["t"].astype(np.int16),
        "true_dmid_norm": y_regr_te.astype(np.float32),
        "pred_dmid_norm": te_pred.astype(np.float32),
        "midprice_t": mp_t_te.astype(np.float32),
        "midprice_th": mp_th_te.astype(np.float32),
    })
    pred_path = os.path.join(HERE, f"pred_T133_{tag}.parquet")
    df.to_parquet(pred_path, index=False)
    print(f"  saved {pred_path} ({len(df):,} rows)", flush=True)

    fee_thr = 2.0 * FEE
    pa = np.full(len(te_pred), 1, dtype=np.int8)
    pa[te_pred > fee_thr] = 2
    pa[te_pred < -fee_thr] = 0
    pnl = vectorized_pnl(pa, mp_t_te, mp_th_te)
    cum_pnl = float(pnl.sum())
    n_active = int((pa != 1).sum())
    per_sym = []
    for s in SYMS:
        m = (sym_te == s)
        per_sym.append(float(pnl[m].sum()))
    print(f"  EV-gate k=1 cum_pnl={cum_pnl:+.4f} n_active={n_active:,} "
          f"per_sym={[round(x,2) for x in per_sym]}", flush=True)

    summary = {
        "task": f"T133 NGBoost seed={seed}",
        "variant": variant, "seed": seed, "horizon": H, "tag": tag,
        "n_features": X_te.shape[1],
        "best_iter": best_iter, "train_time_sec": float(train_time),
        "val": {"mse": va_mse, "mae": va_mae, "corr": va_corr},
        "test": {"mae": te_mae, "corr": te_corr,
                 "ev_gate_k1_cum_pnl": cum_pnl,
                 "ev_gate_k1_n_active": n_active,
                 "per_sym_ev_gate": per_sym},
    }
    out_path = os.path.join(HERE, f"summary_T133_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"summary -> {out_path}", flush=True)

    if use_wandb and wandb is not None:
        wandb.log({"best_iter": best_iter, "train_time_sec": train_time,
                   "val_mse": va_mse, "val_mae": va_mae, "val_corr": va_corr,
                   "test_mae": te_mae, "test_corr": te_corr,
                   "test_ev_gate_k1_cum_pnl": cum_pnl,
                   "test_ev_gate_k1_n_active": n_active})
        wandb.finish()
    progress("done", variant=variant, seed=seed,
             best_iter=best_iter, test_ev_gate_k1_cum_pnl=cum_pnl)


if __name__ == "__main__":
    main()
