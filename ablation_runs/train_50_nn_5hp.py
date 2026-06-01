"""50 NN driver: HP cycling x 50 seeds, V4 walk-forward.

- Cache: cache_log1p (schemeP 370-d, drop11 -> 359-d).
- Features: log1p_rawlast + window-z + mirror-flip aug (matches row 10 / row 19 NN(SPO)).
- Phase 1: L2 pretrain on train (0-79), val (80-95) ES (max 25 ep, patience 6).
- Phase 2: SPO+ DFL fine-tune on train+val (0-95) for 11 epochs, lr=3e-5, lambda_spo=30.
- HP cycling per seed: lr in [1e-4, 3e-4, 1e-3], dropout in [0.05,0.10,0.15,0.20], batch in [2048,4096,8192].
- Saves preds.npz per seed (post-Phase-2 predictions).

Loads data ONCE, trains 50 NN sequentially on cuda:0.
"""
from __future__ import annotations
import argparse, os, json, time, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FEE = 1e-4
NUM_CLASS = 3
CLIP = 10.0
HIDDEN = (256, 128, 64)

LR_LIST = [1e-4, 3e-4, 1e-3]
DROPOUT_LIST = [0.05, 0.10, 0.15, 0.20]
BATCH_LIST = [2048, 4096, 8192]

PHASE1_EPOCHS = 25
PHASE1_PATIENCE = 6
PHASE1_WD = 1e-4

PHASE2_EPOCHS = 11
PHASE2_LR = 3e-5
PHASE2_LAMBDA_SPO = 30.0
PHASE2_BATCH = 4096
PHASE2_WD = 1e-4

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


def class_balanced_weight(y_cls, num_class=3):
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
    ap.add_argument("--cache-dir", default="ablation_runs/cache_log1p")
    ap.add_argument("--out", default="ablation_runs/big50_nn")
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--cuda", default="0")
    ap.add_argument("--wandb-project", default="liangwenbei-50plus50-final")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--n-seeds", type=int, default=50)
    ap.add_argument("--start-seed", type=int, default=1)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    H = args.horizon
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    # Load features
    feat_path = os.path.join(args.cache_dir, "schemeP_feat_names.txt")
    with open(feat_path) as f:
        feat_names_all = [line.strip() for line in f if line.strip()]
    drop_set = set(i for i, n in enumerate(feat_names_all) if n in DROP_NAMES)
    keep_idx = np.array([i for i in range(len(feat_names_all)) if i not in drop_set], dtype=np.int64)
    feat_names = [feat_names_all[i] for i in keep_idx]
    feat_dim = len(feat_names)
    print(f"feat_dim={feat_dim}", flush=True)

    # Load splits
    def load_split(split):
        d = np.load(os.path.join(args.cache_dir, f"schemeP_{split}.npz"))
        return {
            "X": d["X"][:, keep_idx].astype(np.float32),
            "mp_t": d["mp_t"].astype(np.float64),
            "mp_th": d[f"mp_t{H}"].astype(np.float64),
            "y_cls": d[f"y{H}"],
        }

    print("loading train/val/test...", flush=True)
    t0 = time.time()
    train = load_split("train")
    val = load_split("val")
    test = load_split("test")
    print(f"  shapes tr={train['X'].shape} va={val['X'].shape} te={test['X'].shape}  "
          f"({time.time()-t0:.1f}s)", flush=True)

    # log1p rawlast (orig idx < 154)
    rl_local = np.array([li for li, oi in enumerate(keep_idx) if oi < 154], dtype=np.int64)
    rl_names = [feat_names_all[oi] for oi in keep_idx if oi < 154]
    magn_prefixes = ("bid", "ask", "bsize", "asize", "amount_delta", "volume_delta",
                     "avgbid", "avgask", "totalbsize", "totalasize", "midprice",
                     "spread", "cumspread")
    magn_local = np.array(
        [rl_local[i] for i, n in enumerate(rl_names)
         if any(n.startswith(p) for p in magn_prefixes)], dtype=np.int64)
    for arr in (train["X"], val["X"], test["X"]):
        v = arr[:, magn_local]
        arr[:, magn_local] = np.sign(v) * np.log1p(np.abs(v))
    print(f"  log1p applied to {len(magn_local)} cols", flush=True)

    # Mirror-flip aug on train (do BEFORE window-z so mirroring acts on raw scale)
    Xm = bidask_mirror_apply(train["X"], feat_names)
    y_reg_tr_orig = ((train["mp_th"] - train["mp_t"]) / (train["mp_t"] + 1.0)).astype(np.float32)
    y_cls_tr_orig = train["y_cls"].astype(np.int64)
    y_cls_tr_mir = (2 - y_cls_tr_orig).clip(0, 2).astype(np.int64)
    fee_eff_tr_orig = (FEE * ((train["mp_th"] + 1.0) + (train["mp_t"] + 1.0))
                       / (train["mp_t"] + 1.0)).astype(np.float32)

    Xtr_full = np.concatenate([train["X"], Xm], axis=0)
    del Xm
    y_reg_tr_full = np.concatenate([y_reg_tr_orig, -y_reg_tr_orig], axis=0)
    y_cls_tr_full = np.concatenate([y_cls_tr_orig, y_cls_tr_mir], axis=0).astype(np.int8)
    fee_eff_tr_full = np.concatenate([fee_eff_tr_orig, fee_eff_tr_orig], axis=0)
    print(f"  mirror-flip aug: train rows {Xtr_full.shape[0]:,}", flush=True)

    # window-z (using AUGMENTED train statistics for matching row 19 pipeline)
    feat_mean = np.nanmean(Xtr_full, axis=0).astype(np.float32)
    feat_std = np.maximum(np.nanstd(Xtr_full, axis=0).astype(np.float32), 1e-6)
    def std_apply(X):
        out = (X - feat_mean) / feat_std
        np.nan_to_num(out, copy=False, nan=0.0)
        np.clip(out, -CLIP, CLIP, out=out)
        return out.astype(np.float32)
    Xtr_std = std_apply(Xtr_full)
    del Xtr_full
    Xva_std = std_apply(val["X"])
    Xte_std = std_apply(test["X"])
    print("  window-z applied", flush=True)

    y_reg_va = ((val["mp_th"] - val["mp_t"]) / (val["mp_t"] + 1.0)).astype(np.float32)
    y_reg_te = ((test["mp_th"] - test["mp_t"]) / (test["mp_t"] + 1.0)).astype(np.float32)
    fee_eff_va = (FEE * ((val["mp_th"] + 1.0) + (val["mp_t"] + 1.0))
                  / (val["mp_t"] + 1.0)).astype(np.float32)

    target_scale = float(np.std(y_reg_tr_full))
    if target_scale < 1e-7:
        target_scale = 1.0
    print(f"target_scale={target_scale:.4e}", flush=True)

    # Tensors (kept on CPU; moved to GPU per batch)
    Xtr_t = torch.from_numpy(Xtr_std)
    ytr_t = torch.from_numpy((y_reg_tr_full / target_scale).astype(np.float32))
    fee_eff_tr_t = torch.from_numpy((fee_eff_tr_full / target_scale).astype(np.float32))
    sw_tr_t = torch.from_numpy(class_balanced_weight(y_cls_tr_full, num_class=3))
    Xva_t = torch.from_numpy(Xva_std)
    Xte_t = torch.from_numpy(Xte_std)

    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            os.environ.setdefault(
                "WANDB_API_KEY",
                "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",
            )
        except Exception:
            use_wandb = False

    summary = []
    for S in range(args.start_seed, args.start_seed + args.n_seeds):
        seed_dir = os.path.join(args.out, f"seed{S}")
        os.makedirs(seed_dir, exist_ok=True)
        preds_path = os.path.join(seed_dir, "preds.npz")
        if os.path.isfile(preds_path):
            print(f"[seed {S}] SKIP (preds.npz exists)", flush=True)
            continue

        lr_p1 = LR_LIST[(S - 1) % len(LR_LIST)]
        dropout = DROPOUT_LIST[(S - 1) % len(DROPOUT_LIST)]
        batch_p1 = BATCH_LIST[(S - 1) % len(BATCH_LIST)]

        print(f"\n=== NN seed={S} lr_p1={lr_p1:.4f} dropout={dropout} batch={batch_p1} ===",
              flush=True)

        torch.manual_seed(S); torch.cuda.manual_seed_all(S); np.random.seed(S)

        wandb_run = None
        if use_wandb:
            try:
                import wandb
                wandb_run = wandb.init(
                    project=args.wandb_project, entity="cjxh21-Tsinghua University",
                    name=f"nn_seed{S}", group="50_nn",
                    config={"seed": S, "lr_p1": lr_p1, "dropout": dropout,
                            "batch_p1": batch_p1, "feat_dim": feat_dim,
                            "n_train": int(Xtr_t.shape[0]),
                            "phase1_epochs": PHASE1_EPOCHS, "phase1_patience": PHASE1_PATIENCE,
                            "phase2_epochs": PHASE2_EPOCHS, "phase2_lr": PHASE2_LR,
                            "phase2_lambda_spo": PHASE2_LAMBDA_SPO,
                            "target_scale": target_scale},
                    reinit=True, settings=wandb.Settings(start_method="thread"),
                )
            except Exception as e:
                print(f"  wandb init fail: {e}")

        model = MLP(in_dim=feat_dim, hidden=HIDDEN, dropout=dropout).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr_p1, weight_decay=PHASE1_WD)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=PHASE1_EPOCHS, eta_min=1e-5)

        def eval_set(Xtensor):
            model.eval()
            preds = []
            with torch.no_grad():
                for s in range(0, len(Xtensor), 16384):
                    xb = Xtensor[s:s + 16384].to(device, non_blocking=True)
                    preds.append(model(xb).detach().cpu().numpy())
            return np.concatenate(preds) * target_scale

        # ===== Phase 1: L2 pretrain on train (0-79), ES on val =====
        n = Xtr_t.shape[0]
        bs = batch_p1
        best_val_pnl = -1e18; best_state = None; best_epoch = -1
        patience = PHASE1_PATIENCE
        t_p1_0 = time.time()
        for ep in range(PHASE1_EPOCHS):
            model.train()
            perm = torch.randperm(n)
            losses = []
            for s in range(0, n, bs):
                idx = perm[s:s + bs]
                xb = Xtr_t[idx].to(device, non_blocking=True)
                yb = ytr_t[idx].to(device, non_blocking=True)
                wb = sw_tr_t[idx].to(device, non_blocking=True)
                pred = model(xb)
                scale = np.random.uniform(0.80, 1.20)
                mse = (((pred - yb * scale) ** 2) * wb).sum() / wb.sum().clamp_min(1.0)
                opt.zero_grad()
                mse.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                losses.append(float(mse.item()))
            sched.step()
            yp_val = eval_set(Xva_t)
            val_pnl, thr, k = search_threshold_sym(yp_val, val["mp_t"], val["mp_th"])
            print(f"  p1 ep{ep:2d} loss={np.mean(losses):.4f} val_pnl={val_pnl:.4f}",
                  flush=True)
            if wandb_run is not None:
                wandb_run.log({"phase1/epoch": ep, "phase1/loss": float(np.mean(losses)),
                               "phase1/val_pnl": float(val_pnl)})
            if val_pnl > best_val_pnl:
                best_val_pnl = val_pnl
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = ep
                patience = PHASE1_PATIENCE
            else:
                patience -= 1
                if patience <= 0:
                    print(f"  early stop at ep {ep} (best={best_epoch} val_pnl={best_val_pnl:.4f})",
                          flush=True)
                    break
        t_p1 = time.time() - t_p1_0
        model.load_state_dict(best_state)
        print(f"  Phase 1 done: time={t_p1:.1f}s best_ep={best_epoch} best_val_pnl={best_val_pnl:.4f}",
              flush=True)

        # ===== Phase 2: SPO+ fine-tune on train+val (0-95) =====
        # Build phase-2 data on the fly
        Xall_t = torch.cat([Xtr_t, torch.from_numpy(Xva_std)], dim=0)
        y_reg_va_full = y_reg_va.astype(np.float32)
        fee_eff_va_arr = fee_eff_va.astype(np.float32)
        yall = torch.cat([ytr_t, torch.from_numpy(y_reg_va_full / target_scale)], dim=0)
        feeall = torch.cat([fee_eff_tr_t,
                            torch.from_numpy(fee_eff_va_arr / target_scale)], dim=0)
        sw_va_arr = class_balanced_weight(val["y_cls"], num_class=3)
        sw_all = torch.cat([sw_tr_t, torch.from_numpy(sw_va_arr)], dim=0)
        n2 = Xall_t.shape[0]
        opt2 = torch.optim.AdamW(model.parameters(), lr=PHASE2_LR, weight_decay=PHASE2_WD)
        sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=PHASE2_EPOCHS)
        t_p2_0 = time.time()
        for ep in range(PHASE2_EPOCHS):
            model.train()
            perm = torch.randperm(n2)
            l2_run = spo_run = nn_run = 0
            for s in range(0, n2, PHASE2_BATCH):
                idx = perm[s:s + PHASE2_BATCH]
                xb = Xall_t[idx].to(device, non_blocking=True)
                yb = yall[idx].to(device, non_blocking=True)
                fb = feeall[idx].to(device, non_blocking=True)
                wb = sw_all[idx].to(device, non_blocking=True)
                pred = model(xb)
                l2_per = (pred - yb) ** 2
                spo_per = spo_plus_loss(pred, yb, fb)
                l2_loss = (l2_per * wb).sum() / wb.sum().clamp_min(1.0)
                spo_loss = (spo_per * wb).sum() / wb.sum().clamp_min(1.0)
                loss = l2_loss + PHASE2_LAMBDA_SPO * spo_loss
                opt2.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt2.step()
                bsz = xb.size(0)
                l2_run += float(l2_loss.item()) * bsz
                spo_run += float(spo_loss.item()) * bsz
                nn_run += bsz
            sched2.step()
            yp_val = eval_set(Xva_t)
            val_pnl, _, _ = search_threshold_sym(yp_val, val["mp_t"], val["mp_th"])
            print(f"  p2 ep{ep:2d} l2={l2_run/nn_run:.4e} spo={spo_run/nn_run:.4e} val_pnl={val_pnl:.4f}",
                  flush=True)
            if wandb_run is not None:
                wandb_run.log({"phase2/epoch": ep, "phase2/l2": float(l2_run/nn_run),
                               "phase2/spo": float(spo_run/nn_run),
                               "phase2/val_pnl": float(val_pnl)})
        t_p2 = time.time() - t_p2_0

        # Final preds (post-Phase-2)
        yp_val = eval_set(Xva_t)
        yp_test = eval_set(Xte_t)
        val_pnl, thr, k = search_threshold_sym(yp_val, val["mp_t"], val["mp_th"])
        test_pnl, n_act = compute_pnl(yp_test, test["mp_t"], test["mp_th"], thr, thr)
        print(f"  seed{S}: P1={t_p1:.1f}s P2={t_p2:.1f}s val={val_pnl:.4f} test={test_pnl:.4f}",
              flush=True)

        np.savez(preds_path,
                 yp_val=yp_val.astype(np.float32), yp_test=yp_test.astype(np.float32),
                 mp_t_val=val["mp_t"], mp_th_val=val["mp_th"],
                 mp_t_test=test["mp_t"], mp_th_test=test["mp_th"])
        result = {"seed": S, "lr_p1": lr_p1, "dropout": dropout, "batch_p1": batch_p1,
                  "phase1_time_sec": float(t_p1), "phase2_time_sec": float(t_p2),
                  "best_epoch_p1": int(best_epoch),
                  "best_phase1_val_pnl": float(best_val_pnl),
                  "val_pnl_h60": float(val_pnl), "test_pnl_h60": float(test_pnl),
                  "thr_sym": float(thr), "k_sym": float(k),
                  "target_scale": target_scale, "feat_dim": feat_dim}
        with open(os.path.join(seed_dir, "results.json"), "w") as f:
            json.dump(result, f, indent=2)
        summary.append(result)

        if wandb_run is not None:
            wandb_run.log({"val/h60_pnl": float(val_pnl), "test/h60_pnl": float(test_pnl)})
            wandb_run.finish()

        del model, opt, opt2, sched, sched2, Xall_t, yall, feeall, sw_all
        if best_state is not None:
            del best_state
        torch.cuda.empty_cache()
        gc.collect()

    sum_path = os.path.join(args.out, "summary_50nn.json")
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nALL NN DONE -> {sum_path}", flush=True)
    if summary:
        print(f"avg val_pnl  = {np.mean([s['val_pnl_h60'] for s in summary]):.4f}", flush=True)
        print(f"avg test_pnl = {np.mean([s['test_pnl_h60'] for s in summary]):.4f}", flush=True)


if __name__ == "__main__":
    main()
