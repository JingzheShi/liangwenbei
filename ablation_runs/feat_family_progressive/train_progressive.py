"""Train one LGB or NN model on level L1..L8 feature subset and report Test h60 PnL.

Walk-forward V4: train (dates 0-79), val (80-95) for ES + threshold search,
test (96-119) held out for final number.

Usage:
  python3 train_progressive.py --model lgb --level L1 --seed 1 --out <dir>
  python3 train_progressive.py --model nn  --level L7 --seed 1 --out <dir>
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE.parent / "cache_log1p"  # schemeP 370-d
FEE = 1e-4
NUM_CLASS = 3
CLIP = 10.0
HIDDEN = (256, 128, 64)
HP_LGB = dict(feature_fraction=0.8, bagging_fraction=0.8, num_leaves=127, lambda_l2=1.0)


def load_partition():
    with open(HERE / "partition.json") as f:
        return json.load(f)


def compute_pnl(y_pred, mp_t, mp_th, thr_up, thr_dn, fee=FEE):
    a = np.zeros_like(y_pred)
    a[y_pred > thr_up] = 1.0
    a[y_pred < -thr_dn] = -1.0
    raw = a * (mp_th - mp_t)
    cost = fee * np.abs(a) * ((mp_th + 1) + (mp_t + 1))
    pnl = (raw - cost) / (mp_t + 1.0)
    return float(pnl.sum()), int((a != 0).sum())


def search_thr_sym(y_pred, mp_t, mp_th, fee=FEE):
    mean_abs = max(float(np.mean(np.abs(y_pred))), 1e-9)
    best = (-1e18, None, None)
    for k in np.linspace(0.3, 3.0, 28):
        thr = k * mean_abs
        s, _ = compute_pnl(y_pred, mp_t, mp_th, thr, thr, fee)
        if s > best[0]:
            best = (s, float(thr), float(k))
    return best  # (pnl, thr, k)


def make_class_weight(y_cls, num_class=3):
    counts = np.bincount(y_cls.astype(np.int64), minlength=num_class).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    cw = len(y_cls) / (num_class * counts)
    return cw[y_cls.astype(np.int64)].astype(np.float32)


def load_split(split, keep_idx, horizon=60):
    p = CACHE_DIR / f"schemeP_{split}.npz"
    d = np.load(p)
    X = d["X"][:, keep_idx].astype(np.float32, copy=False)
    mp_t = d["mp_t"].astype(np.float64)
    mp_th = d[f"mp_t{horizon}"].astype(np.float64)
    y_cls = d[f"y{horizon}"]
    y_reg = ((mp_th - mp_t) / (mp_t + 1.0)).astype(np.float32)
    d.close()
    return {"X": X, "mp_t": mp_t, "mp_th": mp_th, "y_cls": y_cls, "y_reg": y_reg}


def train_lgb(level: str, keep_idx: np.ndarray, seed: int, out_dir: Path,
              horizon: int, use_gpu: bool, use_wandb: bool):
    import lightgbm as lgb
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== LGB {level} seed={seed} feat_dim={len(keep_idx)} ===", flush=True)
    t0 = time.time()
    train = load_split("train", keep_idx, horizon)
    val = load_split("val", keep_idx, horizon)
    test = load_split("test", keep_idx, horizon)
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    sw = make_class_weight(train["y_cls"], num_class=3)
    params = dict(
        objective="regression_l2", metric="rmse",
        learning_rate=0.05,
        num_leaves=int(HP_LGB["num_leaves"]),
        feature_fraction=float(HP_LGB["feature_fraction"]),
        bagging_fraction=float(HP_LGB["bagging_fraction"]),
        bagging_freq=5,
        lambda_l2=float(HP_LGB["lambda_l2"]),
        min_data_in_leaf=100,
        num_threads=18,
        seed=seed,
        verbose=-1,
    )
    if use_gpu:
        params["device"] = "gpu"
        params["gpu_use_dp"] = False

    dtrain = lgb.Dataset(train["X"], label=train["y_reg"], weight=sw, free_raw_data=False)
    dval = lgb.Dataset(val["X"], label=val["y_reg"], reference=dtrain, free_raw_data=False)

    t_tr = time.time()
    booster = lgb.train(
        params, dtrain,
        num_boost_round=330,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False),
                   lgb.log_evaluation(period=100)],
    )
    train_time = time.time() - t_tr
    best_iter = int(booster.best_iteration or booster.current_iteration())

    yp_val = booster.predict(val["X"], num_iteration=best_iter).astype(np.float32)
    yp_test = booster.predict(test["X"], num_iteration=best_iter).astype(np.float32)

    val_pnl, thr, k = search_thr_sym(yp_val, val["mp_t"], val["mp_th"])
    test_pnl, n_act = compute_pnl(yp_test, test["mp_t"], test["mp_th"], thr, thr)

    booster.save_model(str(out_dir / "model.txt"), num_iteration=best_iter)
    np.savez(out_dir / "preds.npz", yp_val=yp_val, yp_test=yp_test,
             mp_t_val=val["mp_t"], mp_th_val=val["mp_th"],
             mp_t_test=test["mp_t"], mp_th_test=test["mp_th"])
    results = {
        "model": "lgb", "level": level, "seed": seed, "feat_dim": int(len(keep_idx)),
        "horizon": horizon, "best_iter": best_iter,
        "train_time_sec": float(train_time),
        "val_pnl_h60_sym": float(val_pnl),
        "val_thr_sym": float(thr), "val_k_sym": float(k),
        "test_pnl_h60_sym": float(test_pnl),
        "n_act_test": int(n_act),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  best_iter={best_iter} val_pnl={val_pnl:.4f} test_pnl={test_pnl:.4f} "
          f"train_time={train_time:.1f}s", flush=True)

    if use_wandb:
        import wandb
        wandb.log({"val_pnl_h60": val_pnl, "test_pnl_h60": test_pnl,
                   "best_iter": best_iter, "train_time_sec": train_time,
                   "feat_dim": int(len(keep_idx))})
    return results


def train_nn(level: str, keep_idx: np.ndarray, seed: int, out_dir: Path,
             horizon: int, use_wandb: bool):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== NN {level} seed={seed} feat_dim={len(keep_idx)} ===", flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    train = load_split("train", keep_idx, horizon)
    val = load_split("val", keep_idx, horizon)
    test = load_split("test", keep_idx, horizon)
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    # window-z standardization (mandatory)
    mu = np.nanmean(train["X"], axis=0).astype(np.float32)
    sd = np.maximum(np.nanstd(train["X"], axis=0), 1e-6).astype(np.float32)

    def std_apply(X):
        Y = (X - mu) / sd
        np.nan_to_num(Y, copy=False, nan=0.0)
        np.clip(Y, -CLIP, CLIP, out=Y)
        return Y.astype(np.float32)

    train["X"] = std_apply(train["X"])
    val["X"] = std_apply(val["X"])
    test["X"] = std_apply(test["X"])

    y_reg_tr = train["y_reg"]
    target_scale = float(np.std(y_reg_tr))
    if target_scale < 1e-7:
        target_scale = 1.0

    class MLP(nn.Module):
        def __init__(self, in_dim, hidden=HIDDEN, dropout=0.10):
            super().__init__()
            layers = []
            d = in_dim
            for h in hidden:
                layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
                d = h
            layers.append(nn.Linear(d, 1))
            self.net = nn.Sequential(*layers)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = MLP(in_dim=train["X"].shape[1], dropout=0.10).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    max_epochs = 25
    patience = 6
    bs = 4096
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)

    Xtr = torch.from_numpy(train["X"])
    ytr = torch.from_numpy((y_reg_tr / target_scale).astype(np.float32))
    Xva = torch.from_numpy(val["X"])
    Xte = torch.from_numpy(test["X"])

    def predict_np(Xt):
        model.eval()
        preds = []
        with torch.no_grad():
            for s in range(0, len(Xt), 16384):
                xb = Xt[s:s+16384].to(device, non_blocking=True)
                preds.append(model(xb).detach().cpu().numpy())
        return (np.concatenate(preds) * target_scale).astype(np.float32)

    n = Xtr.shape[0]
    best_val_pnl = -1e18
    best_state = None
    best_epoch = -1
    pat_left = patience
    t_tr0 = time.time()
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(n)
        losses = []
        for s in range(0, n, bs):
            idx = perm[s:s+bs]
            xb = Xtr[idx].to(device, non_blocking=True)
            yb = ytr[idx].to(device, non_blocking=True)
            scale = float(np.random.uniform(0.80, 1.20))
            pred = model(xb)
            loss = F.mse_loss(pred, yb * scale)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.item()))
        sched.step()
        yp_val = predict_np(Xva)
        val_pnl, thr, k = search_thr_sym(yp_val, val["mp_t"], val["mp_th"])
        avg_loss = float(np.mean(losses))
        print(f"  ep{ep:02d} loss={avg_loss:.4f} val_pnl={val_pnl:.4f} thr={thr:.6f} k={k:.2f}",
              flush=True)
        if use_wandb:
            import wandb
            wandb.log({"phase1/epoch": ep, "phase1/loss": avg_loss,
                       "phase1/val_pnl_h60": val_pnl})
        if val_pnl > best_val_pnl:
            best_val_pnl = val_pnl
            best_state = {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()}
            best_epoch = ep
            pat_left = patience
        else:
            pat_left -= 1
            if pat_left <= 0:
                print(f"  early stop at epoch {ep} (best={best_epoch})", flush=True)
                break
    train_time = time.time() - t_tr0
    if best_state is not None:
        model.load_state_dict(best_state)

    yp_val = predict_np(Xva)
    yp_test = predict_np(Xte)
    val_pnl, thr, k = search_thr_sym(yp_val, val["mp_t"], val["mp_th"])
    test_pnl, n_act = compute_pnl(yp_test, test["mp_t"], test["mp_th"], thr, thr)

    np.savez(out_dir / "preds.npz", yp_val=yp_val, yp_test=yp_test,
             mp_t_val=val["mp_t"], mp_th_val=val["mp_th"],
             mp_t_test=test["mp_t"], mp_th_test=test["mp_th"])
    results = {
        "model": "nn", "level": level, "seed": seed, "feat_dim": int(len(keep_idx)),
        "horizon": horizon, "best_epoch": best_epoch,
        "train_time_sec": float(train_time),
        "target_scale": float(target_scale),
        "val_pnl_h60_sym": float(val_pnl),
        "val_thr_sym": float(thr), "val_k_sym": float(k),
        "test_pnl_h60_sym": float(test_pnl),
        "n_act_test": int(n_act),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  best_epoch={best_epoch} val_pnl={val_pnl:.4f} test_pnl={test_pnl:.4f} "
          f"train_time={train_time:.1f}s", flush=True)

    if use_wandb:
        import wandb
        wandb.log({"val_pnl_h60": val_pnl, "test_pnl_h60": test_pnl,
                   "best_epoch": best_epoch, "train_time_sec": train_time,
                   "feat_dim": int(len(keep_idx))})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lgb", "nn"], required=True)
    ap.add_argument("--level", choices=[f"L{i}" for i in range(1, 9)], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    part = load_partition()
    keep_idx = np.array(part["levels"][args.level], dtype=np.int64)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            os.environ.setdefault(
                "WANDB_API_KEY",
                "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",
            )
            wandb.init(
                project="liangwenbei-feat-family-progressive",
                entity="cjxh21-Tsinghua University",
                name=f"{args.level}_{args.model}_s{args.seed}",
                config={
                    "level": args.level, "model": args.model, "seed": args.seed,
                    "feat_dim": int(len(keep_idx)), "horizon": args.horizon,
                    "gpu": args.gpu,
                },
                reinit=True,
                settings=wandb.Settings(start_method="thread"),
            )
        except Exception as e:
            print(f"wandb init failed: {e}", flush=True)
            use_wandb = False

    if args.model == "lgb":
        res = train_lgb(args.level, keep_idx, args.seed, out_dir,
                        args.horizon, args.gpu, use_wandb)
    else:
        res = train_nn(args.level, keep_idx, args.seed, out_dir,
                       args.horizon, use_wandb)

    if use_wandb:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass
    print(f"RESULT_LINE: model={args.model} level={args.level} seed={args.seed} "
          f"val={res['val_pnl_h60_sym']:.4f} test={res['test_pnl_h60_sym']:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
