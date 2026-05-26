"""Unified NN training script for the 13-row ablation rerun.

All 13 rows × 5 seeds use ONE shared implementation. Only feature subset,
augmentations, objective, and (for row 13) M7 retrain phase differ.

Fixed HP for all rows (matches row 12 history):
  lr=3e-4, dropout=0.10, batch=4096, max_epochs=25, patience=6,
  AdamW(weight_decay=1e-4), CosineAnnealingLR(eta_min=1e-5),
  multiplicative-noise scale U(0.80,1.20) on regression target,
  grad clip 5.0, class-balanced sample weights, target_scale=std(y_reg_tr).

Row table:
  | row | cache_src   | keep_idx              | obj | wz    | mirror | log1p | M7  |
  |-----|-------------|-----------------------|-----|-------|--------|-------|-----|
  | 1   | rawamt 154  | range(154)            | ce  | False | False  | False | No  |
  | 2   | rawamt 154  | range(154)            | l2  | False | False  | False | No  |
  | 3   | rawamt 154  | range(154)            | l2  | True  | False  | False | No  |
  | 4   | log1p 370   | partition.L2 (165)    | l2  | True  | False  | False | No  |
  | 5   | log1p 370   | partition.L3 (213)    | l2  | True  | False  | False | No  |
  | 6   | log1p 370   | partition.L4 (240)    | l2  | True  | False  | False | No  |
  | 7   | log1p 370   | partition.L5 (272)    | l2  | True  | False  | False | No  |
  | 8   | log1p 370   | partition.L6 (335)    | l2  | True  | False  | False | No  |
  | 9   | log1p 370   | partition.L7 (370)    | l2  | True  | False  | False | No  |
  | 10  | log1p 370   | partition.L8 (359)    | l2  | True  | False  | False | No  |
  | 11  | log1p 370   | partition.L8 (359)    | l2  | True  | True   | False | No  |
  | 12  | log1p 370   | partition.L8 (359)    | l2  | True  | True   | True  | No  |
  | 13  | log1p 370   | partition.L8 (359)    | l2  | True  | True   | True  | Yes |
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
CACHE_LOG1P = HERE / "cache_log1p"
CACHE_RAWAMT = HERE / "cache_raw154_rawamt"
PARTITION = HERE / "feat_family_progressive" / "partition.json"
OUT = HERE / "rerun_13rows"
OUT.mkdir(parents=True, exist_ok=True)

H = 60
FEE = 1e-4
NUM_CLASS = 3
CLIP = 10.0
HIDDEN = (256, 128, 64)
SEEDS = [1, 2, 3, 4, 5]

ROW_CFG = {
    1:  dict(cache="rawamt", level=None, obj="ce", wz=False, mirror=False, log1p=False, m7=False),
    2:  dict(cache="rawamt", level=None, obj="l2", wz=False, mirror=False, log1p=False, m7=False),
    3:  dict(cache="rawamt", level=None, obj="l2", wz=True,  mirror=False, log1p=False, m7=False),
    4:  dict(cache="log1p",  level="L2", obj="l2", wz=True,  mirror=False, log1p=False, m7=False),
    5:  dict(cache="log1p",  level="L3", obj="l2", wz=True,  mirror=False, log1p=False, m7=False),
    6:  dict(cache="log1p",  level="L4", obj="l2", wz=True,  mirror=False, log1p=False, m7=False),
    7:  dict(cache="log1p",  level="L5", obj="l2", wz=True,  mirror=False, log1p=False, m7=False),
    8:  dict(cache="log1p",  level="L6", obj="l2", wz=True,  mirror=False, log1p=False, m7=False),
    9:  dict(cache="log1p",  level="L7", obj="l2", wz=True,  mirror=False, log1p=False, m7=False),
    10: dict(cache="log1p",  level="L8", obj="l2", wz=True,  mirror=False, log1p=False, m7=False),
    11: dict(cache="log1p",  level="L8", obj="l2", wz=True,  mirror=True,  log1p=False, m7=False),
    12: dict(cache="log1p",  level="L8", obj="l2", wz=True,  mirror=True,  log1p=True,  m7=False),
    13: dict(cache="log1p",  level="L8", obj="l2", wz=True,  mirror=True,  log1p=True,  m7=True),
}

LOG1P_PREFIXES = (
    "bid", "ask", "bsize", "asize", "amount_delta", "volume_delta",
    "avgbid", "avgask", "totalbsize", "totalasize", "midprice", "spread", "cumspread",
)


def compute_pnl(y_pred, mp_t, mp_th, thr_up, thr_dn, fee=FEE):
    a = np.zeros_like(y_pred)
    a[y_pred > thr_up] = 1.0
    a[y_pred < -thr_dn] = -1.0
    raw = a * (mp_th - mp_t)
    cost = fee * np.abs(a) * ((mp_th + 1) + (mp_t + 1))
    pnl = (raw - cost) / (mp_t + 1.0)
    return float(pnl.sum()), int((a != 0).sum())


def compute_pnl_ce(p3, mp_t, mp_th, T, delta, fee=FEE):
    p_dn, p_fl, p_up = p3[:, 0], p3[:, 1], p3[:, 2]
    a = np.zeros(len(p3), dtype=np.float32)
    max_extreme = np.maximum(p_up, p_dn)
    conf = (max_extreme >= T) & (max_extreme > p_fl + delta)
    a[conf & (p_up >= p_dn)] = 1.0
    a[conf & (p_dn > p_up)] = -1.0
    raw = a * (mp_th - mp_t)
    cost = fee * np.abs(a) * ((mp_th + 1) + (mp_t + 1))
    pnl = (raw - cost) / (mp_t + 1.0)
    return float(pnl.sum()), int((a != 0).sum())


def search_thr_sym(y_pred, mp_t, mp_th, fee=FEE):
    mean_abs = max(float(np.mean(np.abs(y_pred))), 1e-8)
    best = (-1e18, None, None)
    for k in np.linspace(0.5, 3.0, 26):
        thr = k * mean_abs
        s, _ = compute_pnl(y_pred, mp_t, mp_th, thr, thr, fee)
        if s > best[0]:
            best = (s, float(thr), float(k))
    return best


def search_thr_ce(p3, mp_t, mp_th, fee=FEE):
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
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        z = self.net(x)
        return z.squeeze(-1) if self.out_dim == 1 else z


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


def load_partition():
    with open(PARTITION) as f:
        return json.load(f)


def load_split(cache_kind, split, keep_idx):
    if cache_kind == "rawamt":
        d = np.load(CACHE_RAWAMT / f"raw154_{split}.npz")
    else:
        d = np.load(CACHE_LOG1P / f"schemeP_{split}.npz")
    X = d["X"][:, keep_idx].astype(np.float32, copy=False)
    mp_t = d["mp_t"].astype(np.float64)
    mp_th = d[f"mp_t{H}"].astype(np.float64)
    y_cls = d[f"y{H}"]
    y_reg = ((mp_th - mp_t) / (mp_t + 1.0)).astype(np.float32)
    d.close()
    return dict(X=X, mp_t=mp_t, mp_th=mp_th, y_cls=y_cls, y_reg=y_reg)


def get_feat_names(cache_kind, keep_idx):
    if cache_kind == "rawamt":
        names_all = open(CACHE_RAWAMT / "feat_names.txt").read().splitlines()
    else:
        names_all = open(CACHE_LOG1P / "schemeP_feat_names.txt").read().splitlines()
    return [names_all[i] for i in keep_idx]


def apply_log1p_rawlast(X, feat_names):
    """Sign-log1p on magnitude cols (in-place)."""
    magn_idx = [i for i, n in enumerate(feat_names)
                if any(n.startswith(p) for p in LOG1P_PREFIXES)]
    if not magn_idx:
        return X
    magn_idx = np.array(magn_idx, dtype=np.int64)
    v = X[:, magn_idx]
    X[:, magn_idx] = np.sign(v) * np.log1p(np.abs(v))
    return X


def train_one(row, seed, partition, wandb_project, use_wandb, device):
    cfg = ROW_CFG[row]
    print(f"\n=== row {row} seed {seed} cfg={cfg} ===", flush=True)
    t_start = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Pick keep_idx
    if cfg["cache"] == "rawamt":
        keep_idx = np.arange(154, dtype=np.int64)
    else:
        keep_idx = np.array(partition["levels"][cfg["level"]], dtype=np.int64)
    feat_names = get_feat_names(cfg["cache"], keep_idx)
    print(f"  feat_dim={len(keep_idx)}", flush=True)

    # 2. Load
    tr = load_split(cfg["cache"], "train", keep_idx)
    va = load_split(cfg["cache"], "val", keep_idx)
    te = load_split(cfg["cache"], "test", keep_idx)
    print(f"  shapes: tr={tr['X'].shape} va={va['X'].shape} te={te['X'].shape}", flush=True)

    # 3. log1p rawlast (before any other transform)
    if cfg["log1p"]:
        apply_log1p_rawlast(tr["X"], feat_names)
        apply_log1p_rawlast(va["X"], feat_names)
        apply_log1p_rawlast(te["X"], feat_names)
        print("  applied sign-log1p on magnitude cols", flush=True)

    # 4. Mirror-flip augmentation (train only)
    if cfg["mirror"]:
        Xm = bidask_mirror_apply(tr["X"], feat_names)
        tr["X"] = np.concatenate([tr["X"], Xm], axis=0)
        tr["mp_t"] = np.concatenate([tr["mp_t"], tr["mp_t"]], axis=0)
        tr["mp_th"] = np.concatenate([tr["mp_th"], tr["mp_th"]], axis=0)
        tr["y_reg"] = np.concatenate([tr["y_reg"], -tr["y_reg"]], axis=0)
        tr["y_cls"] = np.concatenate([tr["y_cls"], (2 - tr["y_cls"]).clip(0, 2)], axis=0)
        del Xm
        print(f"  mirror aug: train rows now {tr['X'].shape[0]:,}", flush=True)

    # 5. window-z (global mu/sd on train)
    if cfg["wz"]:
        mu = np.nanmean(tr["X"], axis=0).astype(np.float32)
        sd = np.maximum(np.nanstd(tr["X"], axis=0), 1e-6).astype(np.float32)
        def zsc(X):
            out = (X - mu) / sd
            np.nan_to_num(out, copy=False, nan=0.0, posinf=10.0, neginf=-10.0)
            np.clip(out, -CLIP, CLIP, out=out)
            return out.astype(np.float32)
        tr["X"] = zsc(tr["X"])
        va["X"] = zsc(va["X"])
        te["X"] = zsc(te["X"])
    else:
        for arr in (tr["X"], va["X"], te["X"]):
            np.nan_to_num(arr, copy=False, nan=0.0)
        mu = np.zeros(tr["X"].shape[1], dtype=np.float32)
        sd = np.ones(tr["X"].shape[1], dtype=np.float32)
        print("  no window-z (raw inputs)", flush=True)

    # 6. labels + target scale + class-balanced sample weights
    use_ce = (cfg["obj"] == "ce")
    y_cls_tr = tr["y_cls"].astype(np.int64)
    cw = make_class_weights(y_cls_tr, num_class=3)  # used for both CE and L2 weighting
    # Per-sample weights for L2 path (class-balanced by h=60 三分类 inverse-frequency)
    sw_tr = cw[y_cls_tr].astype(np.float32)
    target_scale = float(np.std(tr["y_reg"]))
    if target_scale < 1e-7:
        target_scale = 1.0
    print(f"  target_scale={target_scale:.6e}  class_weights={cw.tolist()}", flush=True)

    # 7. WandB init (fresh per-seed run)
    run_name = f"row{row:02d}_s{seed}"
    if use_wandb:
        try:
            import wandb
            os.environ.setdefault(
                "WANDB_API_KEY",
                "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",
            )
            wandb.init(
                project=wandb_project,
                name=run_name,
                config={"row": row, "seed": seed, **cfg,
                        "feat_dim": len(keep_idx),
                        "target_scale": target_scale,
                        "n_train": int(tr["X"].shape[0]),
                        "n_val": int(va["X"].shape[0])},
                reinit=True, settings=wandb.Settings(start_method="thread"),
            )
        except Exception as e:
            print(f"  wandb init failed: {e}", flush=True)
            use_wandb = False

    # 8. Model + optim
    out_dim = 3 if use_ce else 1
    model = MLP(in_dim=tr["X"].shape[1], hidden=HIDDEN, dropout=0.10, out_dim=out_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=25, eta_min=1e-5)
    cw_t = torch.from_numpy(cw).to(device) if use_ce else None

    Xtr = torch.from_numpy(tr["X"])
    Xva = torch.from_numpy(va["X"])
    Xte = torch.from_numpy(te["X"])
    if use_ce:
        ytr = torch.from_numpy(y_cls_tr.astype(np.int64))
    else:
        ytr = torch.from_numpy((tr["y_reg"] / target_scale).astype(np.float32))
    swtr = torch.from_numpy(sw_tr)

    def predict_reg(Xt):
        model.eval()
        preds = []
        with torch.no_grad():
            for s in range(0, len(Xt), 16384):
                xb = Xt[s:s+16384].to(device, non_blocking=True)
                preds.append(model(xb).detach().cpu().numpy())
        return (np.concatenate(preds) * target_scale).astype(np.float32)

    def predict_ce(Xt):
        model.eval()
        probs = []
        with torch.no_grad():
            for s in range(0, len(Xt), 16384):
                xb = Xt[s:s+16384].to(device, non_blocking=True)
                probs.append(F.softmax(model(xb), dim=-1).detach().cpu().numpy())
        return np.concatenate(probs, axis=0)

    # 9. V4 training (train 0-79, ES on val 80-95)
    bs = 4096
    max_ep = 25
    patience = 6
    n = Xtr.shape[0]
    best_val_pnl = -1e18
    best_state = None
    best_ep = -1
    pat_left = patience
    t0 = time.time()
    for ep in range(max_ep):
        model.train()
        perm = torch.randperm(n)
        losses = []
        for s in range(0, n, bs):
            idx = perm[s:s+bs]
            xb = Xtr[idx].to(device, non_blocking=True)
            yb = ytr[idx].to(device, non_blocking=True)
            if use_ce:
                logits = model(xb)
                loss = F.cross_entropy(logits, yb, weight=cw_t)
            else:
                scale = float(np.random.uniform(0.80, 1.20))
                pred = model(xb)
                wb = swtr[idx].to(device, non_blocking=True)
                sq = (pred - yb * scale) ** 2
                loss = (sq * wb).sum() / wb.sum().clamp_min(1.0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.item()))
        sched.step()
        if use_ce:
            p3 = predict_ce(Xva)
            val_pnl, T_b, d_b = search_thr_ce(p3, va["mp_t"], va["mp_th"])
        else:
            yp = predict_reg(Xva)
            val_pnl, thr_b, k_b = search_thr_sym(yp, va["mp_t"], va["mp_th"])
        avg_loss = float(np.mean(losses))
        print(f"  ep{ep:02d} loss={avg_loss:.4f} val_pnl={val_pnl:.4f}", flush=True)
        if use_wandb:
            import wandb
            wandb.log({"phase1/epoch": ep, "phase1/loss": avg_loss,
                       "phase1/val_pnl_h60": val_pnl})
        if val_pnl > best_val_pnl:
            best_val_pnl = val_pnl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_ep = ep
            pat_left = patience
        else:
            pat_left -= 1
            if pat_left <= 0:
                print(f"  early stop at ep {ep} (best={best_ep})", flush=True)
                break
    t_v4 = time.time() - t0
    model.load_state_dict(best_state)

    # 10. V4 final test
    if use_ce:
        p_val = predict_ce(Xva)
        p_test = predict_ce(Xte)
        val_pnl, T_b, d_b = search_thr_ce(p_val, va["mp_t"], va["mp_th"])
        test_pnl, n_act = compute_pnl_ce(p_test, te["mp_t"], te["mp_th"], T_b, d_b)
        decision = {"objective": "ce", "T": float(T_b), "delta": float(d_b)}
        yp_val_signed = (p_val[:, 2] - p_val[:, 0]).astype(np.float32)
        yp_test_signed = (p_test[:, 2] - p_test[:, 0]).astype(np.float32)
    else:
        yp_val = predict_reg(Xva)
        yp_test = predict_reg(Xte)
        val_pnl, thr_b, k_b = search_thr_sym(yp_val, va["mp_t"], va["mp_th"])
        test_pnl, n_act = compute_pnl(yp_test, te["mp_t"], te["mp_th"], thr_b, thr_b)
        decision = {"objective": "l2", "thr_sym": float(thr_b), "k_sym": float(k_b)}
        yp_val_signed = yp_val.astype(np.float32)
        yp_test_signed = yp_test.astype(np.float32)

    print(f"  V4 best_ep={best_ep} val_pnl={best_val_pnl:.4f} test_pnl={test_pnl:.4f} time={t_v4:.1f}s",
          flush=True)
    if use_wandb:
        import wandb
        wandb.log({"v4/val_pnl_h60": float(val_pnl),
                   "v4/test_pnl_h60": float(test_pnl),
                   "v4/best_epoch": int(best_ep)})

    # 11. M7 retrain (only row 13)
    test_pnl_m7 = None
    yp_test_m7_signed = None
    n_ep_m7 = None
    if cfg["m7"]:
        assert not use_ce, "M7 only for L2 objective"
        n_ep_m7 = max(1, int(round(best_ep * 1.1))) if best_ep >= 0 else 1
        # combine train + val (X already standardized identically since mu/sd shared)
        Xm7 = np.concatenate([tr["X"], va["X"]], axis=0).astype(np.float32)
        ym7_reg = np.concatenate([tr["y_reg"], va["y_reg"]], axis=0).astype(np.float32)
        y_cls_m7 = np.concatenate([y_cls_tr, va["y_cls"].astype(np.int64)], axis=0)
        sw_m7 = cw[y_cls_m7].astype(np.float32)
        # Re-seed so M7 init weights match V4 init weights (project convention)
        torch.manual_seed(seed)
        np.random.seed(seed)
        model2 = MLP(in_dim=tr["X"].shape[1], hidden=HIDDEN, dropout=0.10, out_dim=1).to(device)
        opt2 = torch.optim.AdamW(model2.parameters(), lr=3e-4, weight_decay=1e-4)
        sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=n_ep_m7, eta_min=1e-5)
        Xm7_t = torch.from_numpy(Xm7)
        ym7_t = torch.from_numpy((ym7_reg / target_scale).astype(np.float32))
        swm7_t = torch.from_numpy(sw_m7)
        nm7 = Xm7_t.shape[0]
        t0 = time.time()
        for ep in range(n_ep_m7):
            model2.train()
            perm = torch.randperm(nm7)
            for s in range(0, nm7, bs):
                idx = perm[s:s+bs]
                xb = Xm7_t[idx].to(device, non_blocking=True)
                yb = ym7_t[idx].to(device, non_blocking=True)
                wb = swm7_t[idx].to(device, non_blocking=True)
                scale = float(np.random.uniform(0.80, 1.20))
                pred = model2(xb)
                sq = (pred - yb * scale) ** 2
                loss = (sq * wb).sum() / wb.sum().clamp_min(1.0)
                opt2.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model2.parameters(), 5.0)
                opt2.step()
            sched2.step()
        t_m7 = time.time() - t0
        model2.eval()
        preds = []
        with torch.no_grad():
            for s in range(0, len(Xte), 16384):
                xb = Xte[s:s+16384].to(device, non_blocking=True)
                preds.append(model2(xb).detach().cpu().numpy())
        yp_test_m7 = (np.concatenate(preds) * target_scale).astype(np.float32)
        # Apply V4 thr
        thr_v4 = decision["thr_sym"]
        test_pnl_m7, _ = compute_pnl(yp_test_m7, te["mp_t"], te["mp_th"], thr_v4, thr_v4)
        yp_test_m7_signed = yp_test_m7
        print(f"  M7 n_ep={n_ep_m7} test_pnl={test_pnl_m7:.4f} (vs V4 {test_pnl:+.4f}) time={t_m7:.1f}s",
              flush=True)
        if use_wandb:
            import wandb
            wandb.log({"m7/test_pnl_h60": float(test_pnl_m7), "m7/n_ep": int(n_ep_m7)})

    if use_wandb:
        import wandb
        wandb.finish()

    # 12. Save
    out_dir = OUT / f"row{row:02d}_s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    yp_save = yp_test_m7_signed if cfg["m7"] else yp_test_signed
    np.savez(
        out_dir / "preds.npz",
        yp_val=yp_val_signed, yp_test=yp_save,
        mp_t_val=va["mp_t"], mp_th_val=va["mp_th"],
        mp_t_test=te["mp_t"], mp_th_test=te["mp_th"],
    )
    res = {
        "row": row, "seed": seed, "cfg": cfg,
        "feat_dim": int(tr["X"].shape[1]),
        "n_train_used": int(Xtr.shape[0]),
        "n_val": int(Xva.shape[0]),
        "n_test": int(Xte.shape[0]),
        "target_scale": float(target_scale),
        "best_ep_v4": int(best_ep),
        "val_pnl_v4": float(val_pnl),
        "test_pnl_v4": float(test_pnl),
        "n_act_v4": int(n_act),
        "decision_v4": decision,
        "t_v4_sec": float(t_v4),
    }
    if cfg["m7"]:
        res["n_ep_m7"] = int(n_ep_m7)
        res["test_pnl_m7"] = float(test_pnl_m7)
        res["test_pnl_reported"] = float(test_pnl_m7)
    else:
        res["test_pnl_reported"] = float(test_pnl)
    res["total_time_sec"] = float(time.time() - t_start)
    with open(out_dir / "results.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"  saved → {out_dir}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=str, default="1-13", help="e.g. '1-13' or '1,3,5'")
    ap.add_argument("--seeds", type=str, default="1,2,3,4,5")
    ap.add_argument("--cuda", default="0")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-project", default="liangwenbei-rerun-13rows")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    # Parse rows
    rows = []
    for part in args.rows.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            rows.extend(range(int(a), int(b) + 1))
        else:
            rows.append(int(part))
    seeds = [int(s) for s in args.seeds.split(",")]

    partition = load_partition()
    use_wandb = not args.no_wandb

    all_results = []
    t_all = time.time()
    for row in rows:
        for seed in seeds:
            out_dir = OUT / f"row{row:02d}_s{seed}"
            if (out_dir / "results.json").exists():
                with open(out_dir / "results.json") as f:
                    res = json.load(f)
                print(f"[skip] row{row} seed{seed} → already done: test_pnl={res['test_pnl_reported']:.4f}",
                      flush=True)
                all_results.append(res)
                continue
            try:
                res = train_one(row, seed, partition, args.wandb_project, use_wandb, device)
                all_results.append(res)
            except Exception as e:
                import traceback
                print(f"[FAIL] row{row} seed{seed}: {e}", flush=True)
                traceback.print_exc()
    print(f"\n=== ALL DONE in {(time.time()-t_all)/60:.1f} min, {len(all_results)} runs ===")

    # Per-row aggregation
    print("\n=== Per-row 5-seed summary ===")
    by_row = {}
    for r in all_results:
        by_row.setdefault(r["row"], []).append(r["test_pnl_reported"])
    for row in sorted(by_row.keys()):
        vals = sorted(by_row[row])
        m = sum(vals) / len(vals)
        sd = (sum((v - m) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5
        print(f"  row{row:2d} n={len(vals)} mean={m:+.4f} std={sd:.4f} range=[{vals[0]:+.4f},{vals[-1]:+.4f}]",
              flush=True)


if __name__ == "__main__":
    main()
