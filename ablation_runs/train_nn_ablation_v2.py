"""Extended NN MLP training: supports raw154 cache + 3-class CE head.

V2 additions over train_nn_ablation.py:
  - Auto-detect cache prefix (raw154 vs schemeP).
  - --objective {l2, ce}: CE means 3-way softmax + CrossEntropyLoss + threshold gate.
  - For raw154 cache, no schemeP-style DROP_NAMES filtering.
  - For CE head, signed pred (p_up - p_dn) is also saved for ensemble compatibility.

V4 walk-forward protocol preserved: train 0-79, val 80-95 (ES + threshold search),
test 96-119 (held out).
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


def compute_pnl_ce(p3, mp_t, mp_th, T, delta, fee=FEE):
    """Threshold gate for 3-class CE: trade if max(p_dn, p_up) >= T and > p_flat + delta."""
    p_dn = p3[:, 0]
    p_fl = p3[:, 1]
    p_up = p3[:, 2]
    a = np.zeros(len(p3), dtype=np.float32)
    max_extreme = np.maximum(p_up, p_dn)
    conf = (max_extreme >= T) & (max_extreme > p_fl + delta)
    up_choice = conf & (p_up >= p_dn)
    dn_choice = conf & (p_dn > p_up)
    a[up_choice] = 1.0
    a[dn_choice] = -1.0
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


def search_threshold_ce(p3, mp_t, mp_th, fee=FEE):
    best = (-1e18, None, None)
    for T in np.linspace(0.40, 0.85, 19):
        for delta in np.linspace(-0.10, 0.30, 9):
            s, _ = compute_pnl_ce(p3, mp_t, mp_th, T, delta, fee)
            if s > best[0]:
                best = (s, float(T), float(delta))
    return best


def make_class_weights(y_cls, num_class=3):
    counts = np.bincount(y_cls.astype(np.int64), minlength=num_class).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    cw = len(y_cls) / (num_class * counts)
    return cw.astype(np.float32)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=HIDDEN, dropout=0.10, out_dim=1):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
        self.out_dim = out_dim
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        z = self.net(x)
        if self.out_dim == 1:
            return z.squeeze(-1)
        return z


def bidask_mirror_apply(X, names):
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


def detect_prefix(cache_dir):
    if os.path.exists(os.path.join(cache_dir, "raw154_train.npz")):
        return "raw154"
    return "schemeP"


def get_feat_names(cache_dir, prefix):
    if prefix == "raw154":
        p = os.path.join(cache_dir, "feat_names.txt")
    else:
        p = os.path.join(cache_dir, "schemeP_feat_names.txt")
    with open(p) as f:
        return [line.strip() for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cuda", default="0")
    ap.add_argument("--objective", choices=["l2", "ce"], default="l2")
    ap.add_argument("--phase1-epochs", type=int, default=25)
    ap.add_argument("--phase1-patience", type=int, default=6)
    ap.add_argument("--phase1-lr", type=float, default=3e-4)
    ap.add_argument("--phase1-batch", type=int, default=4096)
    ap.add_argument("--phase1-dropout", type=float, default=0.10)
    ap.add_argument("--mirror-flip", action="store_true")
    ap.add_argument("--log1p-rawlast", action="store_true")
    ap.add_argument("--no-drop", action="store_true",
                    help="schemeP only: keep all 370 features (skip drop11).")
    ap.add_argument("--schemeC72", action="store_true",
                    help="schemeP only: keep first 154 raw + first 72 extras = 226-d SchemeC.")
    ap.add_argument("--no-window-z", action="store_true",
                    help="Skip global standardization.")
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-project", default="liangwenbei-ablation-bigger")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    os.makedirs(args.out, exist_ok=True)
    H = args.horizon
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"device={device}", flush=True)

    prefix = detect_prefix(args.cache_dir)
    print(f"cache prefix: {prefix}", flush=True)
    feat_names_all = get_feat_names(args.cache_dir, prefix)
    if prefix == "schemeP" and args.schemeC72:
        assert not args.no_drop, "Cannot combine --schemeC72 and --no-drop"
        keep_idx = np.arange(154 + 72, dtype=np.int64)
    elif prefix == "schemeP" and not args.no_drop:
        drop_set = set(i for i, n in enumerate(feat_names_all) if n in DROP_NAMES)
        keep_idx = np.array([i for i in range(len(feat_names_all)) if i not in drop_set],
                            dtype=np.int64)
    else:
        keep_idx = np.arange(len(feat_names_all), dtype=np.int64)
    feat_names = [feat_names_all[i] for i in keep_idx]
    print(f"feat_dim={len(keep_idx)}", flush=True)

    def load(split):
        p = os.path.join(args.cache_dir, f"{prefix}_{split}.npz")
        d = np.load(p)
        return {
            "X": d["X"][:, keep_idx].astype(np.float32, copy=False),
            "mp_t": d["mp_t"].astype(np.float64),
            "mp_th": d[f"mp_t{H}"].astype(np.float64),
            "y_cls": d[f"y{H}"],
        }

    train = load("train"); val = load("val"); test = load("test")
    print(f"shapes: tr={train['X'].shape} va={val['X'].shape} te={test['X'].shape}", flush=True)

    if args.log1p_rawlast:
        if prefix == "raw154":
            magn_local = np.arange(len(feat_names_all), dtype=np.int64)  # all 154 cols are raw
            magn_names = feat_names_all
            magn_local = np.array(
                [i for i, n in enumerate(magn_names)
                 if any(n.startswith(p) for p in
                        ("bid", "ask", "bsize", "asize", "amount_delta",
                         "volume_delta", "avgbid", "avgask", "totalbsize",
                         "totalasize", "midprice", "spread", "cumspread"))],
                dtype=np.int64)
        else:
            rl_local = np.array([li for li, oi in enumerate(keep_idx) if oi < 154], dtype=np.int64)
            rl_names = [feat_names_all[oi] for oi in keep_idx if oi < 154]
            magn_local = np.array(
                [rl_local[i] for i, n in enumerate(rl_names)
                 if any(n.startswith(p) for p in
                        ("bid", "ask", "bsize", "asize", "amount_delta",
                         "volume_delta", "avgbid", "avgask", "totalbsize",
                         "totalasize", "midprice", "spread", "cumspread"))],
                dtype=np.int64)
        for arr in (train["X"], val["X"], test["X"]):
            v = arr[:, magn_local]
            arr[:, magn_local] = np.sign(v) * np.log1p(np.abs(v))
        print(f"applied log1p to {len(magn_local)} cols", flush=True)

    # mirror-flip aug (train only)
    if args.mirror_flip:
        Xm = bidask_mirror_apply(train["X"], feat_names)
        y_reg_orig = ((train["mp_th"] - train["mp_t"]) / (train["mp_t"] + 1.0)).astype(np.float32)
        train["X"] = np.concatenate([train["X"], Xm], axis=0)
        del Xm
        train["mp_t"] = np.concatenate([train["mp_t"], train["mp_t"]], axis=0)
        train["mp_th"] = np.concatenate([train["mp_th"], train["mp_th"]], axis=0)
        train["y_reg_concat"] = np.concatenate([y_reg_orig, -y_reg_orig], axis=0)
        train["y_cls"] = np.concatenate(
            [train["y_cls"], (2 - train["y_cls"]).clip(0, 2)], axis=0
        )
        print(f"mirror-flip aug: train rows now {train['X'].shape[0]:,}", flush=True)
    else:
        train["y_reg_concat"] = ((train["mp_th"] - train["mp_t"]) / (train["mp_t"] + 1.0)).astype(np.float32)

    # window-z standardization
    if args.no_window_z:
        feat_mean = np.zeros(train["X"].shape[1], dtype=np.float32)
        feat_std = np.ones(train["X"].shape[1], dtype=np.float32)
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

    # labels
    y_reg_tr = train["y_reg_concat"]
    y_cls_tr = train["y_cls"].astype(np.int64)

    use_ce = (args.objective == "ce")
    if use_ce:
        cw = make_class_weights(y_cls_tr, num_class=3)
        print(f"class weights: {cw.tolist()}", flush=True)
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
    Xva = torch.from_numpy(val["X"])
    Xte = torch.from_numpy(test["X"])
    if use_ce:
        ytr = torch.from_numpy(y_cls_tr.astype(np.int64))
    else:
        ytr = torch.from_numpy((y_reg_tr / target_scale).astype(np.float32))

    # model
    out_dim = 3 if use_ce else 1
    model = MLP(in_dim=Xtr.shape[1], hidden=HIDDEN,
                dropout=args.phase1_dropout, out_dim=out_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.phase1_lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.phase1_epochs))
    cw_t = torch.from_numpy(cw).to(device) if use_ce else None

    def eval_val_reg():
        model.eval()
        preds = []
        with torch.no_grad():
            for s in range(0, len(Xva), 16384):
                xb = Xva[s:s + 16384].to(device, non_blocking=True)
                preds.append(model(xb).detach().cpu().numpy())
        return np.concatenate(preds) * target_scale

    def eval_val_ce():
        model.eval()
        probs = []
        with torch.no_grad():
            for s in range(0, len(Xva), 16384):
                xb = Xva[s:s + 16384].to(device, non_blocking=True)
                logits = model(xb)
                probs.append(F.softmax(logits, dim=-1).detach().cpu().numpy())
        return np.concatenate(probs, axis=0)

    print(f"=== Phase 1: {'CE' if use_ce else 'L2'} pretrain ===", flush=True)
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
            if use_ce:
                logits = model(xb)
                loss = F.cross_entropy(logits, yb, weight=cw_t)
            else:
                pred = model(xb)
                scale = np.random.uniform(0.80, 1.20)
                loss = F.mse_loss(pred, yb * scale)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.item()))
        sched.step()
        if use_ce:
            p3 = eval_val_ce()
            val_pnl, T_best, d_best = search_threshold_ce(p3, val["mp_t"], val["mp_th"])
            print(f"  ep{ep:2d} loss={np.mean(losses):.4f} val_pnl={val_pnl:.4f} "
                  f"T={T_best:.3f} delta={d_best:.3f}", flush=True)
            if use_wandb:
                import wandb
                wandb.log({"phase1/epoch": ep, "phase1/loss": float(np.mean(losses)),
                            "phase1/val_pnl_h60": float(val_pnl)})
        else:
            yp_val_pred = eval_val_reg()
            val_pnl, thr, k = search_threshold_sym(yp_val_pred, val["mp_t"], val["mp_th"])
            print(f"  ep{ep:2d} loss={np.mean(losses):.4f} val_pnl={val_pnl:.4f} "
                  f"thr={thr:.6f} k={k:.2f}", flush=True)
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

    # final preds
    model.eval()
    def predict_all_reg(X):
        preds = []
        with torch.no_grad():
            for s in range(0, len(X), 16384):
                xb = torch.from_numpy(X[s:s + 16384]).to(device, non_blocking=True)
                preds.append(model(xb).detach().cpu().numpy())
        return np.concatenate(preds) * target_scale

    def predict_all_ce(X):
        probs = []
        with torch.no_grad():
            for s in range(0, len(X), 16384):
                xb = torch.from_numpy(X[s:s + 16384]).to(device, non_blocking=True)
                logits = model(xb)
                probs.append(F.softmax(logits, dim=-1).detach().cpu().numpy())
        return np.concatenate(probs, axis=0)

    if use_ce:
        p_val = predict_all_ce(val["X"])
        p_test = predict_all_ce(test["X"])
        val_pnl, T_best, d_best = search_threshold_ce(p_val, val["mp_t"], val["mp_th"])
        test_pnl, n_act = compute_pnl_ce(p_test, test["mp_t"], test["mp_th"], T_best, d_best)
        decision = {"objective": "ce", "T": float(T_best), "delta": float(d_best),
                    "val_pnl_h60": float(val_pnl), "test_pnl_h60": float(test_pnl)}
        # signed pred for ensemble compatibility (p_up - p_dn)
        yp_val_signed = (p_val[:, 2] - p_val[:, 0]).astype(np.float32)
        yp_test_signed = (p_test[:, 2] - p_test[:, 0]).astype(np.float32)
    else:
        yp_val = predict_all_reg(val["X"])
        yp_test = predict_all_reg(test["X"])
        val_pnl, thr, k = search_threshold_sym(yp_val, val["mp_t"], val["mp_th"])
        test_pnl, n_act = compute_pnl(yp_test, test["mp_t"], test["mp_th"], thr, thr)
        decision = {"objective": "l2", "thr_sym": float(thr), "k_sym": float(k),
                    "val_pnl_h60": float(val_pnl), "test_pnl_h60": float(test_pnl)}
        yp_val_signed = yp_val.astype(np.float32)
        yp_test_signed = yp_test.astype(np.float32)

    results = {
        "task_args": vars(args),
        "feat_dim": int(Xtr.shape[1]),
        "phase1_time_sec": float(t_p1),
        "best_epoch": int(best_epoch),
        "best_phase1_val_pnl": float(best_val_pnl),
        "target_scale": float(target_scale),
        "decision": decision,
    }
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2), flush=True)

    np.savez(os.path.join(args.out, "preds.npz"),
             yp_val=yp_val_signed, yp_test=yp_test_signed,
             mp_t_val=val["mp_t"], mp_th_val=val["mp_th"],
             mp_t_test=test["mp_t"], mp_th_test=test["mp_th"])

    if use_wandb:
        import wandb
        wandb.log({"val/h60_pnl": float(val_pnl), "test/h60_pnl": float(test_pnl),
                   "best_epoch": int(best_epoch)})
        wandb.finish()

    print(f"RESULT_LINE: val={val_pnl:.4f} test={test_pnl:.4f} best_ep={best_epoch}",
          flush=True)


if __name__ == "__main__":
    main()
