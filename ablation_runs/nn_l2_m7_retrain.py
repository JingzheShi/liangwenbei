"""NN L2 + M7 retrain — apples-to-apples with row 16 (LGB L2 + M7).
For each of 5 seeds: train NN L2 on V4 (0-79) → find best epoch + threshold on val 80-95;
then retrain on train+val (0-95) for best_ep × 1.1 → apply V4 threshold → eval test.
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

CACHE_DIR = Path('/root/projects/liangwenbei_workdir/ablation_runs/cache_log1p')
OUT = Path('/root/projects/liangwenbei_workdir/ablation_runs/big50_nn_l2_m7')
OUT.mkdir(parents=True, exist_ok=True)
H = 60
FEE = 1e-4
HIDDEN = (256, 128, 64)
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def load_split(name):
    d = np.load(CACHE_DIR / f'schemeP_{name}.npz')
    return {
        'X': d['X'].astype(np.float32),
        'mp_t': d['mp_t'].astype(np.float64),
        'mp_th': d[f'mp_t{H}'].astype(np.float64),
        'y_cls': d[f'y{H}'].astype(np.int64),
    }


def compute_pnl(y_pred, mp_t, mp_th, thr_up, thr_dn, fee=FEE):
    a = np.where(y_pred > thr_up, 1.0, np.where(y_pred < -thr_dn, -1.0, 0.0))
    raw = a * (mp_th - mp_t)
    cost = fee * np.abs(a) * ((mp_th + 1) + (mp_t + 1))
    return float(((raw - cost) / (mp_t + 1.0)).sum())


def search_thr_sym(yp, mp_t, mp_th):
    ya = np.abs(yp).mean()
    best = (-1e18, None, None)
    for k in np.arange(0.3, 2.5, 0.05):
        thr = k * ya
        s = compute_pnl(yp, mp_t, mp_th, thr, thr)
        if s > best[0]:
            best = (s, float(thr), float(k))
    return best


def class_balanced_weight(y_cls, num_class=3):
    cnt = np.bincount(y_cls, minlength=num_class).astype(np.float64)
    cnt = np.where(cnt == 0, 1.0, cnt)
    cw = len(y_cls) / (num_class * cnt)
    return cw[y_cls].astype(np.float32)


def bidask_mirror_apply(X, names):
    swap = []
    def add_pair(a, b):
        if a in names and b in names:
            swap.append((names.index(a), names.index(b)))
    for i in range(1, 11):
        add_pair(f'bid{i}', f'ask{i}')
        add_pair(f'bsize{i}', f'asize{i}')
        add_pair(f'bid_diff{i}', f'ask_diff{i}')
    map_intst = {'lb':'la', 'la':'lb', 'mb':'ma', 'ma':'mb', 'cb':'ca', 'ca':'cb'}
    for k, kp in map_intst.items():
        for suf in ['_intst', '_ind', '_acc']:
            add_pair(k + suf, kp + suf)
    seen = set()
    swap_uniq = []
    for a, b in swap:
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        swap_uniq.append((a, b))
    Xm = X.copy()
    for i, j in swap_uniq:
        Xm[:, i], Xm[:, j] = X[:, j].copy(), X[:, i].copy()
    if 'imbalance' in names:
        Xm[:, names.index('imbalance')] = -X[:, names.index('imbalance')]
    return Xm


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=HIDDEN, dropout=0.10):
        super().__init__()
        dims = [in_dim] + list(hidden) + [1]
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                self.layers.append(nn.LayerNorm(dims[i+1]))
                self.layers.append(nn.GELU())
                self.layers.append(nn.Dropout(dropout))
    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x.squeeze(-1)


def train_nn(Xtr, ytr, swtr, Xval, yval, mp_t_val, mp_th_val, hp, n_ep, search_thr=True, target_scale=None, patience=6):
    if target_scale is None:
        target_scale = float(np.std(ytr) + 1e-9)
    ytr_s = (ytr / target_scale).astype(np.float32)
    torch.manual_seed(hp.get('seed', 1))
    np.random.seed(hp.get('seed', 1))
    model = MLP(Xtr.shape[1], hidden=HIDDEN, dropout=hp['dropout']).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=hp['lr'], weight_decay=1e-4)
    # Use CosineAnnealingLR with eta_min=1e-5 matching project
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep, eta_min=1e-5)
    Xt = torch.from_numpy(Xtr).to(DEVICE)
    yt = torch.from_numpy(ytr_s).to(DEVICE)
    wt = torch.from_numpy(swtr).to(DEVICE)
    n = len(Xt)
    best_val_pnl, best_ep, best_thr = -1e18, 0, None
    best_state = None
    no_improve = 0
    for ep in range(n_ep):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, hp['batch']):
            idx = perm[i:i+hp['batch']]
            xb, yb, wb = Xt[idx], yt[idx], wt[idx]
            pred = model(xb)
            scale = np.random.uniform(0.80, 1.20)
            loss = (((pred - yb * scale)**2) * wb).sum() / wb.sum().clamp_min(1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()  # once per epoch (T_max = n_ep)
        # eval on val
        if search_thr and Xval is not None:
            model.eval()
            with torch.no_grad():
                Xv = torch.from_numpy(Xval).to(DEVICE)
                yp_val_s = []
                B = 65536
                for j in range(0, len(Xv), B):
                    yp_val_s.append(model(Xv[j:j+B]).cpu().numpy())
                yp_val_s = np.concatenate(yp_val_s)
            yp_val = yp_val_s * target_scale
            v_pnl, thr, k = search_thr_sym(yp_val, mp_t_val, mp_th_val)
            if v_pnl > best_val_pnl:
                best_val_pnl, best_ep, best_thr = v_pnl, ep + 1, thr
                best_state = {nn: v.cpu().clone() for nn, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f'    early stop at ep {ep+1} (best={best_ep})', flush=True)
                    break
    return model, target_scale, best_val_pnl, best_ep, best_thr, best_state


def predict(model, X, target_scale):
    model.eval()
    Xt = torch.from_numpy(X).to(DEVICE)
    out = []
    B = 65536
    with torch.no_grad():
        for i in range(0, len(Xt), B):
            out.append(model(Xt[i:i+B]).cpu().numpy())
    yp_s = np.concatenate(out)
    return yp_s * target_scale


def main():
    print('Loading...', flush=True)
    tr = load_split('train')
    val = load_split('val')
    te = load_split('test')
    feat_names = open(CACHE_DIR / 'schemeP_feat_names.txt').read().strip().splitlines()

    # Apply mirror aug to train
    Xm_tr = bidask_mirror_apply(tr['X'], feat_names)
    Xtr = np.concatenate([tr['X'], Xm_tr], axis=0)
    y_reg_tr = ((tr['mp_th'] - tr['mp_t']) / (tr['mp_t'] + 1.0)).astype(np.float32)
    y_reg_tr_mir = -y_reg_tr
    y_reg_tr_full = np.concatenate([y_reg_tr, y_reg_tr_mir])
    y_cls_tr_full = np.concatenate([tr['y_cls'], (2 - tr['y_cls']).clip(0, 2)])

    Xm_v = bidask_mirror_apply(val['X'], feat_names)
    Xv = np.concatenate([val['X'], Xm_v], axis=0)
    y_reg_v = ((val['mp_th'] - val['mp_t']) / (val['mp_t'] + 1.0)).astype(np.float32)
    y_reg_v_mir = -y_reg_v
    y_reg_v_full = np.concatenate([y_reg_v, y_reg_v_mir])
    y_cls_v_full = np.concatenate([val['y_cls'], (2 - val['y_cls']).clip(0, 2)])

    # Standardize using train+mirror stats (matching project)
    mu = np.nanmean(Xtr, axis=0).astype(np.float32)
    sd = np.maximum(np.nanstd(Xtr, axis=0).astype(np.float32), 1e-6)
    def zsc(X):
        out = (X - mu) / sd
        np.nan_to_num(out, copy=False, nan=0.0, posinf=10.0, neginf=-10.0)
        np.clip(out, -10, 10, out=out)
        return out.astype(np.float32)
    Xtr_z = zsc(Xtr)
    Xv_z = zsc(Xv)
    Xte_z = zsc(te['X'])
    sw_tr = class_balanced_weight(y_cls_tr_full)
    sw_v  = class_balanced_weight(y_cls_v_full)

    # M7: train+val combined
    Xm7 = np.concatenate([Xtr_z, Xv_z], axis=0)
    ym7 = np.concatenate([y_reg_tr_full, y_reg_v_full])
    swm7 = class_balanced_weight(np.concatenate([y_cls_tr_full, y_cls_v_full]))

    # Fixed HP matching row 12 (sweep_5seed/nn/r12): lr=3e-4, dropout=0.10, batch=4096
    HP_LIST = [
        dict(lr=3e-4, dropout=0.10, batch=4096),  # seed 1
        dict(lr=3e-4, dropout=0.10, batch=4096),  # seed 2
        dict(lr=3e-4, dropout=0.10, batch=4096),  # seed 3
        dict(lr=3e-4, dropout=0.10, batch=4096),  # seed 4
        dict(lr=3e-4, dropout=0.10, batch=4096),  # seed 5
    ]

    results = []
    for S in [1, 2, 3, 4, 5]:
        hp = dict(HP_LIST[S - 1], seed=S)
        # First: V4 train (0-79) only, search thr on val 80-95
        print(f'\n=== Seed {S} HP=lr{hp["lr"]:.0e}/dr{hp["dropout"]}/bs{hp["batch"]} ===', flush=True)
        print('  -- V4 phase (train 0-79, val 80-95 ES, max 15 ep) --', flush=True)
        t0 = time.time()
        # mp_t/mp_th for val (with mirror, the mirror half is artificial — only use original val for thr search)
        mp_t_v_orig = val['mp_t']
        mp_th_v_orig = val['mp_th']
        Xv_z_orig = Xv_z[:len(val['X'])]
        model_v4, ts_v4, val_pnl_v4, best_ep_v4, thr_v4, best_state = train_nn(
            Xtr_z, y_reg_tr_full, sw_tr, Xv_z_orig, y_reg_v[:len(val['X'])],
            mp_t_v_orig, mp_th_v_orig, hp, n_ep=25, search_thr=True, patience=6)
        # Load best state and predict test
        model_v4.load_state_dict(best_state)
        yp_test_v4 = predict(model_v4, Xte_z, ts_v4)
        test_pnl_v4 = compute_pnl(yp_test_v4, te['mp_t'], te['mp_th'], thr_v4, thr_v4)
        print(f'  V4: best_ep={best_ep_v4} val_pnl={val_pnl_v4:.4f} thr={thr_v4:.6f} test_pnl={test_pnl_v4:.4f}', flush=True)

        # Then: M7 train (0-95) for best_ep × 1.1 epochs, no val ES, apply V4 thr to test
        n_ep_m7 = max(1, int(round(best_ep_v4 * 1.1)))
        print(f'  -- M7 phase (train+val, n_ep={n_ep_m7}) --', flush=True)
        model_m7, ts_m7, _, _, _, _ = train_nn(
            Xm7, ym7, swm7, None, None, None, None, hp, n_ep=n_ep_m7, search_thr=False, target_scale=ts_v4)
        yp_test_m7 = predict(model_m7, Xte_z, ts_m7)
        # Use V4 thr (already in mp-scale units since predict returns yp * target_scale)
        test_pnl_m7 = compute_pnl(yp_test_m7, te['mp_t'], te['mp_th'], thr_v4, thr_v4)
        t_tot = time.time() - t0
        print(f'  M7: thr (from V4) = {thr_v4:.6f}  test_pnl={test_pnl_m7:.4f}  (Δ vs V4 = {test_pnl_m7 - test_pnl_v4:+.4f})  time={t_tot:.1f}s', flush=True)

        seed_dir = OUT / f'seed{S}'
        seed_dir.mkdir(exist_ok=True)
        np.savez(seed_dir / 'preds.npz', yp_test_v4=yp_test_v4.astype(np.float32),
                 yp_test_m7=yp_test_m7.astype(np.float32),
                 mp_t_test=te['mp_t'], mp_th_test=te['mp_th'])
        res = {'seed': S, 'hp': {'lr': hp['lr'], 'dropout': hp['dropout'], 'batch': hp['batch']},
               'best_ep_v4': best_ep_v4, 'n_ep_m7': n_ep_m7,
               'thr_v4': float(thr_v4), 'target_scale': float(ts_v4),
               'val_pnl_v4': float(val_pnl_v4),
               'test_pnl_v4': float(test_pnl_v4),
               'test_pnl_m7': float(test_pnl_m7)}
        (seed_dir / 'results.json').write_text(json.dumps(res, indent=2))
        results.append(res)

    (OUT / 'summary.json').write_text(json.dumps(results, indent=2))
    import statistics
    v4s = [r['test_pnl_v4'] for r in results]
    m7s = [r['test_pnl_m7'] for r in results]
    print(f'\n=== NN L2 (no SPO+) summary (5 seeds) ===')
    print(f'  V4 (train 0-79):       mean={statistics.mean(v4s):.4f}  std={statistics.stdev(v4s):.4f}')
    print(f'  M7 (train+val 0-95):   mean={statistics.mean(m7s):.4f}  std={statistics.stdev(m7s):.4f}')
    print(f'  Δ = {statistics.mean(m7s) - statistics.mean(v4s):+.4f}')
    print(f'  M7 values: {m7s}')


if __name__ == '__main__':
    main()
