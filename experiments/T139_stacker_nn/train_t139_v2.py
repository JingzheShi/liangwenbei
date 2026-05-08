"""T139 v2: Stacker NN with Huber loss + residual skip connection.

The v1 (MSE on y_reg directly) failed: stacker output scale (std ~y_reg std = 1.9e-3)
is 4x base preds (std ~5e-4), so iter_015_v1 thresholds are mis-calibrated, and at DE
optimum the stacker is no better than zero predictor (cum_pnl ~ 0).

V2 design:
  - Target = y_reg - ens_base_pred  (residual; mean ~0, std ~y_reg std)
  - Stacker output = residual prediction
  - FINAL prediction = base_ens_pred + alpha * stacker_residual
  - Loss: Huber (delta=5e-4) — limits gradient from heavy tails, forces calibrated output
  - Alpha sweep (post-hoc): 0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0

This is much closer to "augment the base model" than "replace it".
"""
import os, sys, json, time, datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import differential_evolution

ROOT = "/root/projects/liangwenbei_workdir"
HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS_BASE = (1, 7, 13, 42, 100)
SEEDS_NN = (1, 7, 42)
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001

ITER_015_V1_THR_UP = 4.21e-4
ITER_015_V1_THR_DN = 1.86e-4

W_NN = 1.0
W_LGB = 1.5

DEVICE = "cuda"
LR = 1e-4
WD = 1e-4
BS = 4096
EPOCHS = 20
PATIENCE = 4
HUBER_DELTA = 5e-4

ALPHA_GRID = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00]


def log(msg):
    print(f"[{datetime.datetime.utcnow().isoformat()}] {msg}", flush=True)


def write_progress(step, metrics=None):
    p = {"status": "running", "step": step, "metrics": metrics or {},
         "timestamp": datetime.datetime.utcnow().isoformat()}
    with open(os.path.join(HERE, "worker-progress.json"), "w") as f:
        json.dump(p, f, indent=2)


def avg_preds(prefix, dir_, suffix=""):
    p = None
    for s in SEEDS_BASE:
        df = pd.read_parquet(os.path.join(dir_, f"{prefix}_seed{s}{suffix}.parquet"))
        if p is None: p = np.zeros(len(df), dtype=np.float32)
        p += df["pred_dmid_norm"].to_numpy(np.float32)
    return p / len(SEEDS_BASE), df


def load_data():
    log("Loading data")
    d = np.load(f"{ROOT}/experiments/T68_stage5_features/cache/schemeP_test.npz", allow_pickle=True)
    X = d["X"].astype(np.float32)
    sym = d["sym"].astype(np.int8)
    date = d["date"].astype(np.int16)
    mp_t = d["mp_t"].astype(np.float32)
    mp_t60 = d["mp_t60"].astype(np.float32)
    y_reg = ((mp_t60 - mp_t) / (mp_t + 1.0)).astype(np.float32)

    t75_pred, t75_base = avg_preds("pred_T75", f"{ROOT}/experiments/T75_regression_dmid")
    t87_pred, t87_base = avg_preds("pred_T87", f"{ROOT}/experiments/T87_spo_dfl", suffix="_main")

    assert np.array_equal(t75_base["sym"].to_numpy().astype(np.int8), sym)

    mp_t_eval = t75_base["midprice_t"].to_numpy(np.float64)
    mp_th_eval = t75_base["midprice_th"].to_numpy(np.float64)

    return X, sym, date, t75_pred, t87_pred, y_reg, mp_t_eval, mp_th_eval


def global_standardize(X):
    log("  computing global percentile clip (1%, 99%)")
    lo = np.percentile(X, 1, axis=0).astype(np.float32)
    hi = np.percentile(X, 99, axis=0).astype(np.float32)
    Xc = np.clip(X, lo, hi)
    mu = Xc.mean(0).astype(np.float32)
    sd = (Xc.std(0) + 1e-6).astype(np.float32)
    return (Xc - mu) / sd, mu, sd


class MLP(nn.Module):
    def __init__(self, in_dim, h=(128, 64, 32), p_drop=0.10):
        super().__init__()
        layers = []
        d = in_dim
        for w in h:
            layers += [nn.Linear(d, w), nn.ReLU(), nn.Dropout(p_drop)]
            d = w
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
        # init final layer small so initial output ~0 (residual ~0)
        nn.init.normal_(self.net[-1].weight, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_one(Xtr, ytr_residual, Xva, yva_residual, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP(in_dim=Xtr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.HuberLoss(delta=HUBER_DELTA)

    Xtr_t = torch.from_numpy(Xtr).to(DEVICE)
    ytr_t = torch.from_numpy(ytr_residual).to(DEVICE)
    Xva_t = torch.from_numpy(Xva).to(DEVICE)
    yva_t = torch.from_numpy(yva_residual).to(DEVICE)

    n = len(Xtr_t)
    best_va = float("inf")
    best_state = None
    bad = 0

    for ep in range(EPOCHS):
        model.train()
        idx = torch.randperm(n, device=DEVICE)
        tot, cnt = 0.0, 0
        for i in range(0, n, BS):
            b = idx[i:i + BS]
            xb = Xtr_t[b]; yb = ytr_t[b]
            opt.zero_grad()
            p = model(xb)
            l = loss_fn(p, yb)
            l.backward()
            opt.step()
            tot += l.item() * len(b); cnt += len(b)
        tr_loss = tot / cnt

        model.eval()
        with torch.no_grad():
            pvas = []
            for i in range(0, len(Xva_t), 16384):
                pvas.append(model(Xva_t[i:i + 16384]))
            pva = torch.cat(pvas).cpu().numpy()
        va_loss = float(np.mean(np.where(np.abs(pva - yva_residual) < HUBER_DELTA,
                                         0.5 * (pva - yva_residual) ** 2,
                                         HUBER_DELTA * (np.abs(pva - yva_residual) - 0.5 * HUBER_DELTA))))
        log(f"      seed{seed} ep{ep + 1:02d}: tr_huber={tr_loss:.6e} va_huber={va_loss:.6e} "
            f"pred_std={pva.std():.4e}")
        if va_loss < best_va - 1e-9:
            best_va = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                log(f"      early stop at ep{ep + 1}")
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pvas = []
        for i in range(0, len(Xva_t), 16384):
            pvas.append(model(Xva_t[i:i + 16384]))
        pva_best = torch.cat(pvas).cpu().numpy().astype(np.float32)
    return pva_best, best_va


def vectorized_pnl(action, mp_t, mp_th):
    side = action.astype(np.float64) - 1.0
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee_pnl) / denom


def gate_asym(p, tu, td):
    a = np.full(len(p), 1, dtype=np.int8)
    a[p > tu] = 2
    a[p < -td] = 0
    return a


def split_sym(sym, pred, mp_t, mp_th):
    folds = []
    for k in SYMS:
        m = (sym == k)
        folds.append({"sym": k, "pred": pred[m].astype(np.float64),
                      "mp_t": mp_t[m], "mp_th": mp_th[m]})
    return folds


def make_obj(folds):
    def f(x):
        s = 0.0
        for fold in folds:
            a = gate_asym(fold["pred"], x[0], x[1])
            s += vectorized_pnl(a, fold["mp_t"], fold["mp_th"]).sum()
        return -float(s)
    return f


def de_loso(folds, bounds=[(0.0, 0.005), (0.0, 0.005)], seeds=(0, 1, 2, 7, 42)):
    obj = make_obj(folds)
    runs = []
    for sd in seeds:
        r = differential_evolution(obj, bounds=bounds, seed=sd, maxiter=80, popsize=24,
                                   polish=True, tol=1e-7, init="sobol")
        runs.append((float(-r.fun), float(r.x[0]), float(r.x[1])))
    runs.sort(key=lambda x: x[0], reverse=True)
    return runs[0]


def eval_at_thr(folds, tu, td):
    s = 0.0
    per = []
    for fold in folds:
        a = gate_asym(fold["pred"], tu, td)
        v = float(vectorized_pnl(a, fold["mp_t"], fold["mp_th"]).sum())
        per.append(v); s += v
    return s, per


def main():
    t0 = time.time()
    write_progress("v2 loading")
    X, sym, date, t75_pred, t87_pred, y_reg, mp_t_eval, mp_th_eval = load_data()

    # base ensemble pred
    ens_base = (W_NN * t87_pred + W_LGB * t75_pred) / (W_NN + W_LGB)
    log(f"  ens_base: mean={ens_base.mean():+.6e} std={ens_base.std():.6e}")

    # residual target
    y_residual = y_reg - ens_base
    log(f"  y_residual: mean={y_residual.mean():+.6e} std={y_residual.std():.6e}")

    write_progress("v2 standardizing")
    log("Standardizing features")
    X_norm, _, _ = global_standardize(X)

    # input: 370 raw + 2 base preds = 372
    Xfull = np.concatenate([X_norm, t75_pred[:, None], t87_pred[:, None]], axis=1).astype(np.float32)
    log(f"  input shape: {Xfull.shape}, residual delta={HUBER_DELTA}")

    DATES_ALL = np.sort(np.unique(date))
    fold_dates = [DATES_ALL[i * 6:(i + 1) * 6] for i in range(4)]

    # OOF residual preds
    oof_resid = {sd: np.zeros(len(date), dtype=np.float32) for sd in SEEDS_NN}

    for fi, va_dates in enumerate(fold_dates):
        write_progress(f"v2 fold {fi+1}/4")
        log(f"\n=== fold {fi+1}/4: val dates {list(va_dates)} ===")
        tr_mask = ~np.isin(date, va_dates)
        va_mask = np.isin(date, va_dates)
        Xtr = Xfull[tr_mask]; Xva = Xfull[va_mask]
        ytr = y_residual[tr_mask]; yva = y_residual[va_mask]

        for sd in SEEDS_NN:
            log(f"    -- seed {sd} --")
            t1 = time.time()
            pva, va_loss = train_one(Xtr, ytr, Xva, yva, sd)
            oof_resid[sd][va_mask] = pva
            log(f"    seed{sd} done in {time.time()-t1:.1f}s")

    # ===== Eval =====
    write_progress("v2 evaluating")
    log("\n=== Eval (residual stacker, alpha sweep) ===")

    # 3-seed avg residual
    oof_resid_avg = np.mean(np.stack([oof_resid[sd] for sd in SEEDS_NN]), axis=0)
    log(f"  oof_resid_avg: mean={oof_resid_avg.mean():+.6e} std={oof_resid_avg.std():.6e}")

    # baseline ens at iter015v1 thr
    folds_base = split_sym(sym, ens_base, mp_t_eval, mp_th_eval)
    s_base_v1, per_base_v1 = eval_at_thr(folds_base, ITER_015_V1_THR_UP, ITER_015_V1_THR_DN)
    s_base_de, tu_base, td_base = de_loso(folds_base)
    log(f"  baseline T87+T75 ens: iter015v1={s_base_v1:+.4f} per={[round(x,2) for x in per_base_v1]} | DE={s_base_de:+.4f}")

    # alpha sweep
    alpha_results = {}
    for alpha in ALPHA_GRID:
        pred_final = ens_base + alpha * oof_resid_avg
        folds_a = split_sym(sym, pred_final, mp_t_eval, mp_th_eval)
        s_v1, per_v1 = eval_at_thr(folds_a, ITER_015_V1_THR_UP, ITER_015_V1_THR_DN)
        s_de, tu_de, td_de = de_loso(folds_a)
        alpha_results[alpha] = {
            "iter015v1": {"cum_pnl": s_v1, "per_sym": per_v1},
            "de_opt": {"cum_pnl": s_de, "thr_up": tu_de, "thr_dn": td_de},
        }
        delta_v1 = s_v1 - s_base_v1; delta_de = s_de - s_base_de
        log(f"  alpha={alpha:.2f}: iter015v1={s_v1:+.4f} (Δ={delta_v1:+.3f})  | DE={s_de:+.4f} (Δ={delta_de:+.3f})  thr=({tu_de:.3e},{td_de:.3e})")

    # save OOF
    oof_df = pd.DataFrame({
        "sym": sym, "date": date,
        "midprice_t": mp_t_eval, "midprice_th": mp_th_eval,
        "ens_base": ens_base, "y_reg": y_reg,
        "stacker_resid_seed1": oof_resid[1], "stacker_resid_seed7": oof_resid[7],
        "stacker_resid_seed42": oof_resid[42],
        "stacker_resid_avg": oof_resid_avg,
    })
    oof_df.to_parquet(os.path.join(HERE, "oof_resid_v2.parquet"))

    # find best alpha
    best_alpha = max(alpha_results.keys(), key=lambda a: alpha_results[a]["de_opt"]["cum_pnl"])
    best_de = alpha_results[best_alpha]["de_opt"]["cum_pnl"]
    delta_best = best_de - s_base_de

    out = {
        "task": "T139 v2 stacker (residual, Huber)",
        "huber_delta": HUBER_DELTA,
        "epochs": EPOCHS,
        "alpha_grid": ALPHA_GRID,
        "baseline_T87+T75_ens": {
            "iter015v1": {"cum_pnl": s_base_v1, "per_sym": per_base_v1},
            "de_opt": {"cum_pnl": s_base_de, "thr_up": tu_base, "thr_dn": td_base},
        },
        "alpha_sweep": alpha_results,
        "best_alpha": best_alpha,
        "best_de_pnl": best_de,
        "delta_vs_baseline_de": delta_best,
        "elapsed_min": (time.time() - t0) / 60.0,
    }
    out_path = os.path.join(HERE, "results_v2.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    log(f"\n  saved {out_path}")
    log(f"\n  BEST alpha={best_alpha} de_pnl={best_de:+.4f} delta_vs_baseline={delta_best:+.4f}")
    write_progress("v2 done", {"best_alpha": best_alpha, "best_de_pnl": best_de})

    print(f"\nRESULT: task=T139v2_stacker_residual_huber metrics={{best_alpha={best_alpha}, "
          f"best_de_pnl={best_de:.4f}, base_de_pnl={s_base_de:.4f}, "
          f"delta={delta_best:+.4f}}} notes=[residual+Huber stacker {('beat' if delta_best > 0 else 'lost vs')} pure ensemble]")


if __name__ == "__main__":
    main()
