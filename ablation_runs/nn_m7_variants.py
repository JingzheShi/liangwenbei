"""Try multiple NN+M7 retrain strategies to fix the -4.17 hurt."""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

CACHE_DIR = Path('/root/projects/liangwenbei_workdir/ablation_runs/cache_log1p')
OUT = Path('/root/projects/liangwenbei_workdir/ablation_runs/nn_m7_variants')
OUT.mkdir(parents=True, exist_ok=True)
H = 60
FEE = 1e-4
HIDDEN = (256, 128, 64)
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def load_split(name):
    d = np.load(CACHE_DIR / f'schemeP_{name}.npz')
    return {'X': d['X'].astype(np.float32), 'mp_t': d['mp_t'].astype(np.float64),
            'mp_th': d[f'mp_t{H}'].astype(np.float64), 'y_cls': d[f'y{H}'].astype(np.int64)}

def compute_pnl(y_pred, mp_t, mp_th, thr_up, thr_dn, fee=FEE):
    a = np.where(y_pred > thr_up, 1.0, np.where(y_pred < -thr_dn, -1.0, 0.0))
    raw = a * (mp_th - mp_t)
    cost = fee * np.abs(a) * ((mp_th+1) + (mp_t+1))
    return float(((raw - cost) / (mp_t+1)).sum())

def search_sym(yp, mp_t, mp_th):
    ya = np.abs(yp).mean()
    best = (-1e18, 0, 0)
    for k in np.arange(0.3, 2.5, 0.02):
        thr = k * ya
        s = compute_pnl(yp, mp_t, mp_th, thr, thr)
        if s > best[0]: best = (s, thr, k)
    return best

def class_balanced_weight(y_cls, num_class=3):
    cnt = np.bincount(y_cls, minlength=num_class).astype(np.float64)
    cnt = np.where(cnt == 0, 1.0, cnt)
    cw = len(y_cls) / (num_class * cnt)
    return cw[y_cls].astype(np.float32)

def bidask_mirror(X, names):
    swap = {}
    nm = {n: i for i, n in enumerate(names)}
    def pair(a, b):
        if a in nm and b in nm:
            swap[nm[a]] = nm[b]; swap[nm[b]] = nm[a]
    for k in range(1, 11):
        pair(f"bid{k}", f"ask{k}"); pair(f"bsize{k}", f"asize{k}")
        pair(f"bid_diff{k}", f"ask_diff{k}")
        pair(f"bid_rate{k}", f"ask_rate{k}"); pair(f"bsize_rate{k}", f"asize_rate{k}")
    pair("avgbid", "avgask"); pair("totalbsize", "totalasize")
    pair("bid_mean", "ask_mean"); pair("bsize_mean", "asize_mean")
    for kp in [("lb","la"),("mb","ma"),("cb","ca")]:
        for suf in ["_intst","_ind","_acc"]:
            pair(kp[0]+suf, kp[1]+suf)
    Xm = X.copy(); seen = set()
    for a, b in swap.items():
        key = tuple(sorted([a, b]))
        if key in seen: continue
        seen.add(key)
        Xm[:, a], Xm[:, b] = X[:, b].copy(), X[:, a].copy()
    if 'imbalance' in nm:
        Xm[:, nm['imbalance']] = -X[:, nm['imbalance']]
    return Xm

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=HIDDEN, dropout=0.10):
        super().__init__()
        dims = [in_dim] + list(hidden) + [1]
        self.blocks = nn.ModuleList()
        for i in range(len(dims) - 1):
            if i < len(dims) - 2:
                self.blocks.append(nn.Sequential(
                    nn.Linear(dims[i], dims[i+1]),
                    nn.LayerNorm(dims[i+1]),
                    nn.GELU(),
                    nn.Dropout(dropout)))
            else:
                self.blocks.append(nn.Linear(dims[i], dims[i+1]))
    def forward(self, x):
        for b in self.blocks: x = b(x)
        return x.squeeze(-1)


def train_one(X, y, sw, n_ep, lr, batch, target_scale, model=None, dropout=0.10):
    if model is None:
        torch.manual_seed(1); np.random.seed(1)
        model = MLP(X.shape[1], dropout=dropout).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, n_ep), eta_min=1e-5)
    Xt = torch.from_numpy(X).to(DEVICE)
    ys = torch.from_numpy((y / target_scale).astype(np.float32)).to(DEVICE)
    wt = torch.from_numpy(sw).to(DEVICE)
    n = len(Xt)
    for ep in range(n_ep):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            pred = model(Xt[idx])
            sc = np.random.uniform(0.80, 1.20)
            loss = (((pred - ys[idx]*sc)**2) * wt[idx]).sum() / wt[idx].sum().clamp_min(1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
    return model

def predict(model, X, target_scale):
    model.eval()
    Xt = torch.from_numpy(X).to(DEVICE)
    out = []
    with torch.no_grad():
        for i in range(0, len(Xt), 65536):
            out.append(model(Xt[i:i+65536]).cpu().numpy())
    return np.concatenate(out) * target_scale


def main():
    print('Loading...', flush=True)
    tr = load_split('train'); val = load_split('val'); te = load_split('test')
    feat_names = open(CACHE_DIR / 'schemeP_feat_names.txt').read().strip().splitlines()

    # All augs (window-z + mirror + log1p), same as row 12
    Xm_tr = bidask_mirror(tr['X'], feat_names)
    Xtr = np.concatenate([tr['X'], Xm_tr], axis=0)
    y_reg_tr = ((tr['mp_th'] - tr['mp_t']) / (tr['mp_t']+1)).astype(np.float32)
    y_full_tr = np.concatenate([y_reg_tr, -y_reg_tr])
    yc_tr = np.concatenate([tr['y_cls'], (2-tr['y_cls']).clip(0,2)])

    Xm_v = bidask_mirror(val['X'], feat_names)
    Xv = np.concatenate([val['X'], Xm_v], axis=0)
    y_reg_v = ((val['mp_th'] - val['mp_t']) / (val['mp_t']+1)).astype(np.float32)
    y_full_v = np.concatenate([y_reg_v, -y_reg_v])
    yc_v = np.concatenate([val['y_cls'], (2-val['y_cls']).clip(0,2)])

    # Standardize using train+mirror stats (same as project)
    mu = np.nanmean(Xtr, axis=0).astype(np.float32)
    sd = np.maximum(np.nanstd(Xtr, axis=0).astype(np.float32), 1e-6)
    def zsc(X):
        out = (X - mu) / sd
        np.nan_to_num(out, copy=False, nan=0.0, posinf=10.0, neginf=-10.0)
        np.clip(out, -10, 10, out=out)
        return out.astype(np.float32)
    Xtr_z = zsc(Xtr); Xv_z = zsc(Xv); Xte_z = zsc(te['X'])
    sw_tr = class_balanced_weight(yc_tr)

    # Combine train+val for M7
    Xm7 = np.concatenate([Xtr_z, Xv_z], axis=0)
    ym7 = np.concatenate([y_full_tr, y_full_v])
    swm7 = class_balanced_weight(np.concatenate([yc_tr, yc_v]))

    target_scale = float(np.std(y_full_tr) + 1e-9)
    print(f'target_scale={target_scale:.6e}', flush=True)

    BATCH = 4096; LR_P1 = 3e-4

    results = []
    # Multiple seeds for stability
    for seed in [1, 2, 3, 4, 5]:
        torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
        print(f'\n=== Seed {seed} ===', flush=True)

        # === Phase 1: V4 train, find best_ep, save best_state ===
        model = MLP(Xtr_z.shape[1], dropout=0.10).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR_P1, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=25, eta_min=1e-5)
        Xt = torch.from_numpy(Xtr_z).to(DEVICE)
        ys = torch.from_numpy((y_full_tr / target_scale).astype(np.float32)).to(DEVICE)
        wt = torch.from_numpy(sw_tr).to(DEVICE)
        best_val_pnl = -1e18; best_ep = 0; best_state = None; thr_v4 = None
        no_improve = 0
        for ep in range(25):
            model.train()
            perm = torch.randperm(len(Xt), device=DEVICE)
            for i in range(0, len(Xt), BATCH):
                idx = perm[i:i+BATCH]
                pred = model(Xt[idx])
                sc = np.random.uniform(0.80, 1.20)
                loss = (((pred - ys[idx]*sc)**2) * wt[idx]).sum() / wt[idx].sum().clamp_min(1.0)
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            sched.step()
            yp_val = predict(model, Xv_z[:len(val['X'])], target_scale)
            v_pnl, thr, k = search_sym(yp_val, val['mp_t'], val['mp_th'])
            if v_pnl > best_val_pnl:
                best_val_pnl = v_pnl; best_ep = ep+1; thr_v4 = thr
                best_state = {n: v.cpu().clone() for n, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 6:
                    break

        # V4 final test
        model.load_state_dict(best_state)
        yp_test = predict(model, Xte_z, target_scale)
        v4_test = compute_pnl(yp_test, te['mp_t'], te['mp_th'], thr_v4, thr_v4)
        print(f'  V4: best_ep={best_ep} val={best_val_pnl:.2f} test={v4_test:.2f} thr_v4={thr_v4:.6e}', flush=True)

        # === Variants for M7 retrain ===
        variants = {}

        # E1: current (n_ep = best_ep × 1.1)
        n_ep_e1 = max(1, int(round(best_ep * 1.1)))
        m_e1 = train_one(Xm7, ym7, swm7, n_ep_e1, LR_P1, BATCH, target_scale)
        yp = predict(m_e1, Xte_z, target_scale)
        t = compute_pnl(yp, te['mp_t'], te['mp_th'], thr_v4, thr_v4)
        variants['E1: from-scratch, n_ep=best_ep×1.1'] = (n_ep_e1, t)

        # E2: from-scratch, n_ep=3
        m = train_one(Xm7, ym7, swm7, 3, LR_P1, BATCH, target_scale)
        yp = predict(m, Xte_z, target_scale)
        t = compute_pnl(yp, te['mp_t'], te['mp_th'], thr_v4, thr_v4)
        variants['E2: from-scratch, n_ep=3'] = (3, t)

        # E3: from-scratch, n_ep=5
        m = train_one(Xm7, ym7, swm7, 5, LR_P1, BATCH, target_scale)
        yp = predict(m, Xte_z, target_scale)
        t = compute_pnl(yp, te['mp_t'], te['mp_th'], thr_v4, thr_v4)
        variants['E3: from-scratch, n_ep=5'] = (5, t)

        # E4: warm-start V4 best, fine-tune at lr/10 for 2 ep
        m = MLP(Xm7.shape[1], dropout=0.10).to(DEVICE)
        m.load_state_dict(best_state)
        opt = torch.optim.AdamW(m.parameters(), lr=LR_P1/10, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2, eta_min=1e-5)
        Xt = torch.from_numpy(Xm7).to(DEVICE)
        ys2 = torch.from_numpy((ym7 / target_scale).astype(np.float32)).to(DEVICE)
        wt2 = torch.from_numpy(swm7).to(DEVICE)
        for ep in range(2):
            m.train()
            perm = torch.randperm(len(Xt), device=DEVICE)
            for i in range(0, len(Xt), BATCH):
                idx = perm[i:i+BATCH]
                pred = m(Xt[idx])
                sc = np.random.uniform(0.80, 1.20)
                loss = (((pred - ys2[idx]*sc)**2) * wt2[idx]).sum() / wt2[idx].sum().clamp_min(1.0)
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
                opt.step()
            sched.step()
        yp = predict(m, Xte_z, target_scale)
        t = compute_pnl(yp, te['mp_t'], te['mp_th'], thr_v4, thr_v4)
        variants['E4: warm-start V4, lr/10, 2 ep'] = (2, t)

        # E5: warm-start V4 best, fine-tune at lr/30 for 3 ep
        m = MLP(Xm7.shape[1], dropout=0.10).to(DEVICE)
        m.load_state_dict(best_state)
        opt = torch.optim.AdamW(m.parameters(), lr=LR_P1/30, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=3, eta_min=1e-6)
        for ep in range(3):
            m.train()
            perm = torch.randperm(len(Xt), device=DEVICE)
            for i in range(0, len(Xt), BATCH):
                idx = perm[i:i+BATCH]
                pred = m(Xt[idx])
                sc = np.random.uniform(0.80, 1.20)
                loss = (((pred - ys2[idx]*sc)**2) * wt2[idx]).sum() / wt2[idx].sum().clamp_min(1.0)
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
                opt.step()
            sched.step()
        yp = predict(m, Xte_z, target_scale)
        t = compute_pnl(yp, te['mp_t'], te['mp_th'], thr_v4, thr_v4)
        variants['E5: warm-start V4, lr/30, 3 ep'] = (3, t)

        for name, (n_ep, t_pnl) in variants.items():
            print(f'  {name}: n_ep={n_ep} test={t_pnl:.2f} (Δ vs V4={t_pnl-v4_test:+.2f})', flush=True)

        results.append({'seed': seed, 'v4_test': v4_test, 'variants': variants})

    # Aggregate
    print('\n=== Summary (5 seed mean) ===')
    import statistics
    v4s = [r['v4_test'] for r in results]
    print(f'  V4 baseline:                    mean={statistics.mean(v4s):.3f} std={statistics.stdev(v4s):.3f}')
    for vname in results[0]['variants'].keys():
        vals = [r['variants'][vname][1] for r in results]
        print(f'  {vname}:  mean={statistics.mean(vals):.3f} std={statistics.stdev(vals):.3f} Δ={statistics.mean(vals)-statistics.mean(v4s):+.3f}')


if __name__ == '__main__':
    main()
