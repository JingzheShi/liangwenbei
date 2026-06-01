"""Unified NN MLP training for big ablation rows 9, 10.

Phase 1: L2 regression pretrain on dates 0-79, ES on 80-95.
Phase 2 (optional): SPO+ DFL fine-tune on dates 0-79+80-95 (or 0-79), short epochs.

Features: schemeP 370-d -> drop 11 (359-d), apply window-z (global standardization),
optional mirror-flip aug on train, optional log1p on rawlast magnitudes.
Saves preds (val + test) to preds.npz for downstream ensemble.
"""
from __future__ import annotations
import argparse, os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FEE = 1e-4
NUM_CLASS = 3
CLIP = 10.0
HIDDEN = (256, 128, 64)

DROP_NAMES = [
    "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
    "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
    "qrank_W100_cumspread",
    "kyle_lam_W50", "kyle_lam_W100",
    "roll_eff_spr_ratio_W100",
    "liq_asym_top5_W5",
]


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


def make_weights(y_cls, num_class=3):
    counts = np.bincount(y_cls.astype(np.int64), minlength=num_class).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    cw = len(y_cls) / (num_class * counts)
    return cw[y_cls.astype(np.int64)].astype(np.float32)


def spo_plus_loss(pred_s, y_s, fee_s):
    z = torch.where(
        y_s > fee_s, torch.ones_like(y_s),
        torch.where(y_s < -fee_s, -torch.ones_like(y_s), torch.zeros_like(y_s))
    )
    spread = 2.0 * pred_s - y_s
    ell = F.relu(torch.abs(spread) - fee_s) - z * spread + fee_s * torch.abs(z)
    return ell


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
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def bidask_mirror_apply(X, names):
    """Return mirrored copy of X (NumPy)."""
    swap = {}
    sign_flip = set()
    name_to_idx = {n: i for i, n in enumerate(names)}

    def add_pair(a, b):
        if a in name_to_idx and b in name_to_idx:
            i, j = name_to_idx[a], name_to_idx[b]
            swap[i] = j
            swap[j] = i

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
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cuda", default="0")
    ap.add_argument("--phase1-epochs", type=int, default=25)
    ap.add_argument("--phase1-patience", type=int, default=6)
    ap.add_argument("--phase1-lr", type=float, default=3e-4)
    ap.add_argument("--phase1-batch", type=int, default=4096)
    ap.add_argument("--phase1-dropout", type=float, default=0.10)
    ap.add_argument("--phase2", action="store_true",
                    help="Enable SPO+ DFL fine-tune (Phase 2 M7).")
    ap.add_argument("--phase2-epochs", type=int, default=8)
    ap.add_argument("--phase2-lr", type=float, default=3e-5)
    ap.add_argument("--phase2-lambda-spo", type=float, default=30.0)
    ap.add_argument("--phase2-batch", type=int, default=4096)
    ap.add_argument("--mirror-flip", action="store_true")
    ap.add_argument("--log1p-rawlast", action="store_true")
    ap.add_argument("--no-drop", action="store_true",
                    help="Keep all 370 features (skip drop11).")
    ap.add_argument("--no-window-z", action="store_true",
                    help="Skip global standardization (window-z).")
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-project", default="liangwenbei-ablation-rerun")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    os.makedirs(args.out, exist_ok=True)
    H = args.horizon
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"device={device}", flush=True)

    # load schemeP cache
    with open(os.path.join(args.cache_dir, "schemeP_feat_names.txt")) as f:
        feat_names_all = [line.strip() for line in f if line.strip()]
    if args.no_drop:
        keep_idx = np.arange(len(feat_names_all), dtype=np.int64)
    else:
        drop_set = set(i for i, n in enumerate(feat_names_all) if n in DROP_NAMES)
        keep_idx = np.array([i for i in range(len(feat_names_all)) if i not in drop_set], dtype=np.int64)
    feat_names = [feat_names_all[i] for i in keep_idx]
    print(f"feat_dim={len(keep_idx)} no_drop={args.no_drop}", flush=True)

    def load(split):
        p = os.path.join(args.cache_dir, f"schemeP_{split}.npz")
        d = np.load(p)
        return {
            "X": d["X"][:, keep_idx].astype(np.float32, copy=False),
            "mp_t": d["mp_t"].astype(np.float64),
            "mp_th": d[f"mp_t{H}"].astype(np.float64),
            "y_cls": d[f"y{H}"],
        }

    train = load("train"); val = load("val"); test = load("test")
    print(f"shapes: tr={train['X'].shape} va={val['X'].shape} te={test['X'].shape}", flush=True)

    # log1p on raw_last magnitudes
    if args.log1p_rawlast:
        # raw_last cols correspond to feat_names_all idx < 154; after keep_idx subset they're local indices
        rl_local = np.array([li for li, oi in enumerate(keep_idx) if oi < 154], dtype=np.int64)
        rl_names = [feat_names_all[oi] for oi in keep_idx if oi < 154]
        magn_local = [rl_local[i] for i, n in enumerate(rl_names)
                      if any(n.startswith(p) for p in
                             ("bid", "ask", "bsize", "asize", "amount_delta",
                              "volume_delta", "avgbid", "avgask", "totalbsize",
                              "totalasize", "midprice", "spread", "cumspread"))]
        magn_local = np.array(magn_local, dtype=np.int64)
        for arr in (train["X"], val["X"], test["X"]):
            v = arr[:, magn_local]
            arr[:, magn_local] = np.sign(v) * np.log1p(np.abs(v))
        print(f"applied log1p to {len(magn_local)} cols", flush=True)

    # mirror-flip aug on train
    if args.mirror_flip:
        Xm = bidask_mirror_apply(train["X"], feat_names)
        y_reg_orig = ((train["mp_th"] - train["mp_t"]) / (train["mp_t"] + 1.0)).astype(np.float32)
        train["X"] = np.concatenate([train["X"], Xm], axis=0)
        del Xm
        train["mp_t"] = np.concatenate([train["mp_t"], train["mp_t"]], axis=0)
        train["mp_th"] = np.concatenate([train["mp_th"], train["mp_th"]], axis=0)  # placeholder
        # For mirrored half, y_reg = -y_reg_orig.  We accomplish this by storing the regression label directly.
        train["y_reg_concat"] = np.concatenate([y_reg_orig, -y_reg_orig], axis=0)
        train["y_cls"] = np.concatenate([train["y_cls"], (2 - train["y_cls"]).clip(0, 2)], axis=0)
        print(f"mirror-flip aug: train rows now {train['X'].shape[0]:,}", flush=True)
    else:
        train["y_reg_concat"] = ((train["mp_th"] - train["mp_t"]) / (train["mp_t"] + 1.0)).astype(np.float32)

    # standardize using train statistics (optional)
    if args.no_window_z:
        feat_mean = np.zeros(train["X"].shape[1], dtype=np.float32)
        feat_std = np.ones(train["X"].shape[1], dtype=np.float32)
        # still clean NaNs for safety (but no clipping to ±10)
        for arr in (train["X"], val["X"], test["X"]):
            np.nan_to_num(arr, copy=False, nan=0.0)
        print("skipped window-z (no standardization)", flush=True)
    else:
        feat_mean = np.nanmean(train["X"], axis=0).astype(np.float32)
        feat_std = np.nanstd(train["X"], axis=0).astype(np.float32)
        feat_std = np.maximum(feat_std, 1e-6)

        def std_apply(X):
            out = (X - feat_mean) / feat_std
            np.nan_to_num(out, copy=False, nan=0.0)
            np.clip(out, -CLIP, CLIP, out=out)
            return out.astype(np.float32)

        train["X"] = std_apply(train["X"])
        val["X"] = std_apply(val["X"])
        test["X"] = std_apply(test["X"])

    # y scaling
    y_reg_tr = train["y_reg_concat"]
    target_scale = float(np.std(y_reg_tr))
    if target_scale < 1e-7:
        target_scale = 1.0
    print(f"target_scale={target_scale:.6e}", flush=True)

    # wandb
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            os.environ.setdefault(
                "WANDB_API_KEY",
                "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",
            )
            wandb.init(project=args.wandb_project, entity="cjxh21-Tsinghua University",
                       name=args.wandb_name or os.path.basename(args.out),
                       config={**vars(args), "feat_dim": len(feat_names),
                                "target_scale": target_scale,
                                "n_train": int(train["X"].shape[0]),
                                "n_val": int(val["X"].shape[0])},
                       reinit=True, settings=wandb.Settings(start_method="thread"))
        except Exception as e:
            print(f"wandb init failed: {e}", flush=True)
            use_wandb = False

    # tensors
    Xtr = torch.from_numpy(train["X"])
    ytr = torch.from_numpy((y_reg_tr / target_scale).astype(np.float32))
    Xva = torch.from_numpy(val["X"])
    Xte = torch.from_numpy(test["X"])
    yva_reg = ((val["mp_th"] - val["mp_t"]) / (val["mp_t"] + 1.0)).astype(np.float32)
    yte_reg = ((test["mp_th"] - test["mp_t"]) / (test["mp_t"] + 1.0)).astype(np.float32)
    fee_eff_tr = (FEE * ((train["mp_th"] + 1.0) + (train["mp_t"] + 1.0))
                  / (train["mp_t"] + 1.0)).astype(np.float32)
    fee_eff_tr_s = torch.from_numpy(fee_eff_tr / target_scale)

    # model
    model = MLP(in_dim=Xtr.shape[1], hidden=HIDDEN, dropout=args.phase1_dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.phase1_lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.phase1_epochs))

    def eval_val():
        model.eval()
        preds = []
        with torch.no_grad():
            for s in range(0, len(Xva), 16384):
                xb = Xva[s:s + 16384].to(device, non_blocking=True)
                preds.append(model(xb).detach().cpu().numpy())
        return np.concatenate(preds) * target_scale

    # Phase 1
    print("=== Phase 1: L2 pretrain ===", flush=True)
    n = Xtr.shape[0]
    bs = args.phase1_batch
    best_val_pnl = -1e18
    best_state = None
    best_epoch = -1
    patience_left = args.phase1_patience
    t0 = time.time()
    for ep in range(args.phase1_epochs):
        model.train()
        perm = torch.randperm(n)
        losses = []
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            xb = Xtr[idx].to(device, non_blocking=True)
            yb = ytr[idx].to(device, non_blocking=True)
            pred = model(xb)
            # tiny multiplicative noise as in T188v2
            scale = np.random.uniform(0.80, 1.20)
            loss = F.mse_loss(pred, yb * scale)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.item()))
        sched.step()
        yp_val_pred = eval_val()
        val_pnl, thr, k = search_threshold_sym(yp_val_pred, val["mp_t"], val["mp_th"])
        print(f"  ep{ep:2d} loss={np.mean(losses):.4f} val_pnl={val_pnl:.4f} thr={thr:.6f} k={k:.2f}",
              flush=True)
        if use_wandb:
            import wandb
            wandb.log({"phase1/epoch": ep, "phase1/loss": float(np.mean(losses)),
                        "phase1/val_pnl_h60": float(val_pnl)})
        if val_pnl > best_val_pnl:
            best_val_pnl = val_pnl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = ep
            patience_left = args.phase1_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  early stop at epoch {ep} (best={best_epoch} val_pnl={best_val_pnl:.4f})",
                      flush=True)
                break
    t_p1 = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Phase 1 done: best_epoch={best_epoch} best_val_pnl={best_val_pnl:.4f} time={t_p1:.1f}s",
          flush=True)

    # Phase 2: SPO+
    t_p2 = 0.0
    if args.phase2:
        print("=== Phase 2: SPO+ DFL fine-tune ===", flush=True)
        # M7: train on train + val (0-95) combined
        Xva_np = val["X"]
        y_reg_va = yva_reg
        fee_eff_va = (FEE * ((val["mp_th"] + 1.0) + (val["mp_t"] + 1.0))
                      / (val["mp_t"] + 1.0)).astype(np.float32)
        Xall = np.concatenate([train["X"], Xva_np], axis=0)
        yall = np.concatenate([y_reg_tr, y_reg_va], axis=0)
        feeall = np.concatenate([fee_eff_tr, fee_eff_va], axis=0)
        Xall_t = torch.from_numpy(Xall)
        yall_t = torch.from_numpy((yall / target_scale).astype(np.float32))
        feeall_t = torch.from_numpy((feeall / target_scale).astype(np.float32))
        n2 = Xall.shape[0]
        bs2 = args.phase2_batch
        opt2 = torch.optim.AdamW(model.parameters(), lr=args.phase2_lr, weight_decay=1e-4)
        t0 = time.time()
        for ep in range(args.phase2_epochs):
            model.train()
            perm = torch.randperm(n2)
            losses = []
            for s in range(0, n2, bs2):
                idx = perm[s:s + bs2]
                xb = Xall_t[idx].to(device, non_blocking=True)
                yb = yall_t[idx].to(device, non_blocking=True)
                feeb = feeall_t[idx].to(device, non_blocking=True)
                pred = model(xb)
                spo = spo_plus_loss(pred, yb, feeb)
                l2 = F.mse_loss(pred, yb)
                loss = l2 + args.phase2_lambda_spo * spo.mean()
                opt2.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt2.step()
                losses.append(float(loss.item()))
            yp_val_pred = eval_val()
            val_pnl, thr, k = search_threshold_sym(yp_val_pred, val["mp_t"], val["mp_th"])
            print(f"  p2 ep{ep} loss={np.mean(losses):.4f} val_pnl={val_pnl:.4f}", flush=True)
            if use_wandb:
                import wandb
                wandb.log({"phase2/epoch": ep, "phase2/loss": float(np.mean(losses)),
                           "phase2/val_pnl_h60": float(val_pnl)})
        t_p2 = time.time() - t0

    # Final predictions
    model.eval()
    def predict_all(X):
        preds = []
        with torch.no_grad():
            for s in range(0, len(X), 16384):
                xb = torch.from_numpy(X[s:s + 16384]).to(device, non_blocking=True)
                preds.append(model(xb).detach().cpu().numpy())
        return np.concatenate(preds) * target_scale

    yp_val = predict_all(val["X"])
    yp_test = predict_all(test["X"])
    val_pnl, thr, k = search_threshold_sym(yp_val, val["mp_t"], val["mp_th"])
    test_pnl, n_act = compute_pnl(yp_test, test["mp_t"], test["mp_th"], thr, thr)

    results = {
        "task_args": vars(args),
        "feat_dim": int(Xtr.shape[1]),
        "phase1_time_sec": float(t_p1),
        "phase2_time_sec": float(t_p2),
        "best_epoch": int(best_epoch),
        "best_phase1_val_pnl": float(best_val_pnl),
        "target_scale": float(target_scale),
        "decision": {
            "thr_sym": float(thr), "k_sym": float(k),
            "val_pnl_h60": float(val_pnl),
            "test_pnl_h60": float(test_pnl),
        },
    }
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2), flush=True)

    np.savez(os.path.join(args.out, "preds.npz"),
             yp_val=yp_val.astype(np.float32), yp_test=yp_test.astype(np.float32),
             mp_t_val=val["mp_t"], mp_th_val=val["mp_th"],
             mp_t_test=test["mp_t"], mp_th_test=test["mp_th"])

    torch.save({"state_dict": model.state_dict(),
                 "feat_mean": feat_mean, "feat_std": feat_std,
                 "target_scale": target_scale, "hidden": HIDDEN,
                 "in_dim": int(Xtr.shape[1])},
                os.path.join(args.out, "model.pt"))

    if use_wandb:
        import wandb
        wandb.log({"val/h60_pnl": float(val_pnl), "test/h60_pnl": float(test_pnl),
                   "best_epoch": int(best_epoch), "thr_sym": float(thr), "k_sym": float(k)})
        wandb.finish()

    print(f"RESULT_LINE: val={val_pnl:.4f} test={test_pnl:.4f} best_ep={best_epoch}",
          flush=True)


if __name__ == "__main__":
    main()
