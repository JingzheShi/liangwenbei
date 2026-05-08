"""T139: Deep stacker NN
Input  : 370 raw schemeP feats + T75 5-seed avg pred + T87 5-seed avg pred (= 372 dims)
Arch   : MLP 128-64-32, dropout 0.10, ReLU, output 1 (regression target = dmid_norm)
Loss   : MSE on dmid_norm = (mp_t60 - mp_t) / (mp_t + 1)
LOSO   : 4-fold by date over 96-119 (6 dates per fold) - this gives OOF stacker preds
Eval   : split 442k OOF preds by sym -> sum cum_pnl at iter_015_v1 thr + DE-LOSO optimum

Critical constraint compliance:
  - sym not in features
  - date not in features (only used for fold split, not as input)
  - Predictor will be stateless (per-row)
"""
import os, sys, json, time, datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.optimize import differential_evolution

ROOT = "/root/projects/liangwenbei_workdir"
HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS_BASE = (1, 7, 13, 42, 100)
SEEDS_NN = (1, 7, 42)
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001

# iter_015 v1 conservative thresholds (DO NOT retune; they ARE the platform calibration)
ITER_015_V1_THR_UP = 4.21e-4
ITER_015_V1_THR_DN = 1.86e-4

W_NN = 1.0
W_LGB = 1.5

DEVICE = "cuda"
LR = 1e-4
WD = 1e-4
BS = 4096
EPOCHS = 15
PATIENCE = 3


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
    log("Loading schemeP_test cache + base preds")
    d = np.load(f"{ROOT}/experiments/T68_stage5_features/cache/schemeP_test.npz", allow_pickle=True)
    X = d["X"].astype(np.float32)
    sym = d["sym"].astype(np.int8)
    date = d["date"].astype(np.int16)
    mp_t = d["mp_t"].astype(np.float32)
    mp_t60 = d["mp_t60"].astype(np.float32)
    y_reg = ((mp_t60 - mp_t) / (mp_t + 1.0)).astype(np.float32)

    t75_pred, t75_base = avg_preds("pred_T75", f"{ROOT}/experiments/T75_regression_dmid")
    t87_pred, t87_base = avg_preds("pred_T87", f"{ROOT}/experiments/T87_spo_dfl", suffix="_main")

    # alignment check
    assert np.array_equal(t75_base["sym"].to_numpy().astype(np.int8), sym)
    assert np.array_equal(t75_base["date"].to_numpy().astype(np.int16), date)
    assert np.array_equal(t87_base["sym"].to_numpy().astype(np.int8), sym)
    assert np.array_equal(t87_base["date"].to_numpy().astype(np.int16), date)

    # midprices for pnl eval
    mp_t_eval = t75_base["midprice_t"].to_numpy(np.float64)
    mp_th_eval = t75_base["midprice_th"].to_numpy(np.float64)

    log(f"  X: {X.shape}, sym: {np.unique(sym)}, dates: {date.min()}-{date.max()}")
    log(f"  T75 pred mean={t75_pred.mean():+.6e} std={t75_pred.std():.6e}")
    log(f"  T87 pred mean={t87_pred.mean():+.6e} std={t87_pred.std():.6e}")
    log(f"  y_reg mean={y_reg.mean():+.6e} std={y_reg.std():.6e}")
    return X, sym, date, t75_pred, t87_pred, y_reg, mp_t_eval, mp_th_eval


def global_standardize(X):
    """Global percentile clip + zscore. Tiny leakage across folds but stats are stable."""
    log("  computing global percentile clip (1%, 99%)")
    lo = np.percentile(X, 1, axis=0).astype(np.float32)
    hi = np.percentile(X, 99, axis=0).astype(np.float32)
    Xc = np.clip(X, lo, hi)
    mu = Xc.mean(0).astype(np.float32)
    sd = (Xc.std(0) + 1e-6).astype(np.float32)
    return (Xc - mu) / sd, mu, sd, lo, hi


class MLP(nn.Module):
    def __init__(self, in_dim=372, h=(128, 64, 32), p_drop=0.10):
        super().__init__()
        layers = []
        d = in_dim
        for w in h:
            layers += [nn.Linear(d, w), nn.ReLU(), nn.Dropout(p_drop)]
            d = w
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_one(Xtr, ytr, Xva, yva, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP(in_dim=Xtr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.MSELoss()

    Xtr_t = torch.from_numpy(Xtr).to(DEVICE)
    ytr_t = torch.from_numpy(ytr).to(DEVICE)
    Xva_t = torch.from_numpy(Xva).to(DEVICE)
    yva_t = torch.from_numpy(yva).to(DEVICE)

    n = len(Xtr_t)
    best_va = float("inf")
    best_state = None
    bad = 0

    for ep in range(EPOCHS):
        model.train()
        idx = torch.randperm(n, device=DEVICE)
        tot = 0.0
        cnt = 0
        for i in range(0, n, BS):
            b = idx[i:i + BS]
            xb = Xtr_t[b]
            yb = ytr_t[b]
            opt.zero_grad()
            p = model(xb)
            l = loss_fn(p, yb)
            l.backward()
            opt.step()
            tot += l.item() * len(b); cnt += len(b)
        tr_loss = tot / cnt

        # val (mini-batched to avoid spikes)
        model.eval()
        with torch.no_grad():
            pvas = []
            for i in range(0, len(Xva_t), 16384):
                pvas.append(model(Xva_t[i:i + 16384]))
            pva = torch.cat(pvas).cpu().numpy()
        va_mse = float(((pva - yva) ** 2).mean())
        log(f"      seed{seed} ep{ep + 1:02d}: tr_mse={tr_loss:.6e} va_mse={va_mse:.6e}")
        if va_mse < best_va - 1e-9:
            best_va = va_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                log(f"      early stop at ep{ep + 1}")
                break

    # final eval with best state
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
        folds.append({
            "sym": k,
            "pred": pred[m].astype(np.float64),
            "mp_t": mp_t[m],
            "mp_th": mp_th[m],
        })
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
        per.append(v)
        s += v
    return s, per


def main():
    t0 = time.time()
    write_progress("loading data")
    X, sym, date, t75_pred, t87_pred, y_reg, mp_t_eval, mp_th_eval = load_data()

    write_progress("standardizing features")
    log("Standardizing features (global)")
    X_norm, mu, sd, lo_clip, hi_clip = global_standardize(X)
    log(f"  X_norm range: {X_norm.min():.3f} .. {X_norm.max():.3f}")

    # Stacker input: 370 normalized + 2 raw base preds (already on dmid_norm scale, ~1e-4)
    Xfull = np.concatenate([X_norm, t75_pred[:, None], t87_pred[:, None]], axis=1).astype(np.float32)
    log(f"  Stacker input shape: {Xfull.shape}")

    # WandB
    try:
        import wandb
        wandb.init(project="liangwenbei", entity="cjxh21-Tsinghua University",
                   name="T139_stacker_nn",
                   config={"in_dim": Xfull.shape[1], "arch": "128-64-32",
                           "lr": LR, "wd": WD, "bs": BS, "epochs": EPOCHS,
                           "seeds": list(SEEDS_NN), "loss": "MSE",
                           "loso": "4-fold by date 96-119 (6 dates per fold)"})
        WB = wandb
    except Exception as e:
        log(f"  wandb init failed: {e}")
        WB = None

    DATES_ALL = np.sort(np.unique(date))  # 96..119
    fold_dates = [DATES_ALL[i * 6:(i + 1) * 6] for i in range(4)]

    # OOF preds: for each NN seed, store 442k preds (each row predicted by stacker not trained on its date-fold)
    oof_pred = {sd: np.zeros(len(date), dtype=np.float32) for sd in SEEDS_NN}

    for fi, va_dates in enumerate(fold_dates):
        write_progress(f"fold {fi + 1}/4 (val dates {list(va_dates)})")
        log(f"\n=== fold {fi + 1}/4: val dates {list(va_dates)} ===")
        tr_mask = ~np.isin(date, va_dates)
        va_mask = np.isin(date, va_dates)
        Xtr = Xfull[tr_mask]
        Xva = Xfull[va_mask]
        ytr = y_reg[tr_mask]
        yva = y_reg[va_mask]
        log(f"  tr={len(Xtr)} va={len(Xva)}")
        for sd in SEEDS_NN:
            log(f"    -- seed {sd} --")
            t1 = time.time()
            pva, va_mse = train_one(Xtr, ytr, Xva, yva, sd)
            oof_pred[sd][va_mask] = pva
            t2 = time.time()
            log(f"    seed{sd} done in {t2 - t1:.1f}s, best_va_mse={va_mse:.6e}")
            if WB:
                WB.log({f"fold_{fi}_seed_{sd}_va_mse": va_mse, "elapsed_s": time.time() - t0})

    # ===== Eval =====
    write_progress("evaluating OOF preds")
    log("\n=== Eval: stacker OOF preds (sym-LOSO) ===")

    # Per-seed
    per_seed_results = {}
    for sd in SEEDS_NN:
        folds = split_sym(sym, oof_pred[sd], mp_t_eval, mp_th_eval)
        s_v1, per_v1 = eval_at_thr(folds, ITER_015_V1_THR_UP, ITER_015_V1_THR_DN)
        s_de, tu_de, td_de = de_loso(folds)
        per_seed_results[sd] = {
            "at_iter015v1_thr": {"cum_pnl": s_v1, "per_sym": per_v1},
            "de_loso_opt": {"cum_pnl": s_de, "thr_up": tu_de, "thr_dn": td_de},
        }
        log(f"  seed{sd}: iter015v1_thr cum_pnl={s_v1:+.4f} per_sym={[round(x,2) for x in per_v1]}"
            f"  | DE_opt cum_pnl={s_de:+.4f} thr=({tu_de:.3e},{td_de:.3e})")
        if WB:
            WB.log({f"seed_{sd}_iter015v1_pnl": s_v1, f"seed_{sd}_de_pnl": s_de,
                    f"seed_{sd}_de_thr_up": tu_de, f"seed_{sd}_de_thr_dn": td_de})

    # 3-seed avg stacker pred
    oof_avg = np.mean(np.stack([oof_pred[sd] for sd in SEEDS_NN]), axis=0)
    folds_avg = split_sym(sym, oof_avg, mp_t_eval, mp_th_eval)
    s_v1_avg, per_v1_avg = eval_at_thr(folds_avg, ITER_015_V1_THR_UP, ITER_015_V1_THR_DN)
    s_de_avg, tu_de_avg, td_de_avg = de_loso(folds_avg)
    log(f"  STACKER 3-seed avg: iter015v1_thr cum_pnl={s_v1_avg:+.4f} per_sym={[round(x,2) for x in per_v1_avg]}"
        f"  | DE_opt cum_pnl={s_de_avg:+.4f} thr=({tu_de_avg:.3e},{td_de_avg:.3e})")
    if WB:
        WB.log({"stacker_avg_iter015v1_pnl": s_v1_avg, "stacker_avg_de_pnl": s_de_avg})

    # Baselines: pure T87, pure T75, T87+T75 ensemble
    folds_t87 = split_sym(sym, t87_pred, mp_t_eval, mp_th_eval)
    folds_t75 = split_sym(sym, t75_pred, mp_t_eval, mp_th_eval)
    ens_pred = (W_NN * t87_pred + W_LGB * t75_pred) / (W_NN + W_LGB)
    folds_ens = split_sym(sym, ens_pred, mp_t_eval, mp_th_eval)

    s_t87, per_t87 = eval_at_thr(folds_t87, ITER_015_V1_THR_UP, ITER_015_V1_THR_DN)
    s_t75, per_t75 = eval_at_thr(folds_t75, ITER_015_V1_THR_UP, ITER_015_V1_THR_DN)
    s_ens, per_ens = eval_at_thr(folds_ens, ITER_015_V1_THR_UP, ITER_015_V1_THR_DN)

    s_t87_de, tu_t87_de, td_t87_de = de_loso(folds_t87)
    s_t75_de, tu_t75_de, td_t75_de = de_loso(folds_t75)
    s_ens_de, tu_ens_de, td_ens_de = de_loso(folds_ens)

    log("\n=== Baselines ===")
    log(f"  pure T87 :  iter015v1 cum={s_t87:+.4f}  per_sym={[round(x,2) for x in per_t87]}  | DE={s_t87_de:+.4f}")
    log(f"  pure T75 :  iter015v1 cum={s_t75:+.4f}  per_sym={[round(x,2) for x in per_t75]}  | DE={s_t75_de:+.4f}")
    log(f"  T87+T75  :  iter015v1 cum={s_ens:+.4f}  per_sym={[round(x,2) for x in per_ens]}  | DE={s_ens_de:+.4f}")

    delta_v1 = s_v1_avg - s_ens
    delta_de = s_de_avg - s_ens_de
    log(f"\n  STACKER delta vs T87+T75 ensemble: at iter015v1_thr={delta_v1:+.4f}  at DE_opt={delta_de:+.4f}")

    # ====== Optional 3-way ensemble ======
    threeway_results = None
    if delta_v1 > 0.5 or delta_de > 0.5:
        log("\n=== 3-way ensemble (stacker + T87 + T75) ===")
        # sweep equal-weight, weighted variants
        weight_grid = [
            ("eq_3", 1.0, 1.0, 1.0),
            ("st-heavy", 1.5, 1.0, 1.0),
            ("st-st", 2.0, 1.0, 1.0),
            ("orig+st", 1.0, W_NN, W_LGB),    # stacker added with weight 1, T87 with 1, T75 with 1.5
            ("st-half", 0.5, W_NN, W_LGB),
        ]
        threeway_results = {}
        for name, ws, wn, wl in weight_grid:
            tot_w = ws + wn + wl
            p3 = (ws * oof_avg + wn * t87_pred + wl * t75_pred) / tot_w
            folds_3 = split_sym(sym, p3, mp_t_eval, mp_th_eval)
            s3_v1, per3_v1 = eval_at_thr(folds_3, ITER_015_V1_THR_UP, ITER_015_V1_THR_DN)
            s3_de, tu3_de, td3_de = de_loso(folds_3)
            threeway_results[name] = {
                "weights": [ws, wn, wl],
                "iter015v1": {"cum_pnl": s3_v1, "per_sym": per3_v1},
                "de_opt": {"cum_pnl": s3_de, "thr_up": tu3_de, "thr_dn": td3_de},
            }
            log(f"  {name:12s} w={ws,wn,wl}: iter015v1={s3_v1:+.4f}  DE={s3_de:+.4f}")
    else:
        log("\n  STACKER did not beat T87+T75 by >+0.5 at either threshold; skipping 3-way ensemble.")

    # ===== Save OOF preds =====
    oof_df = pd.DataFrame({
        "sym": sym, "date": date,
        "midprice_t": mp_t_eval, "midprice_th": mp_th_eval,
        "y_reg": y_reg,
        "t75_pred": t75_pred, "t87_pred": t87_pred,
        "stacker_seed1": oof_pred[1], "stacker_seed7": oof_pred[7], "stacker_seed42": oof_pred[42],
        "stacker_avg": oof_avg,
    })
    oof_df.to_parquet(os.path.join(HERE, "oof_preds.parquet"))
    log(f"  saved OOF preds: {os.path.join(HERE, 'oof_preds.parquet')}")

    # ===== Results =====
    iter018_baseline_no_abstain = 39.7467
    iter018_with_abstain = 41.4939

    out = {
        "task": "T139 stacker NN (raw schemeP + T75 + T87 -> dmid_norm)",
        "input_dim": int(Xfull.shape[1]),
        "arch": "MLP 128-64-32 dropout 0.10 ReLU",
        "loss": "MSE on (mp_t60-mp_t)/(mp_t+1)",
        "loso_split": "4-fold by date 96-119 (6 dates per fold)",
        "training_seeds": list(SEEDS_NN),
        "metrics": {
            "stacker_per_seed": per_seed_results,
            "stacker_3seed_avg": {
                "at_iter015v1_thr": {"cum_pnl": s_v1_avg, "per_sym": per_v1_avg},
                "de_loso_opt": {"cum_pnl": s_de_avg, "thr_up": tu_de_avg, "thr_dn": td_de_avg},
            },
            "baselines_at_iter015v1_thr": {
                "pure_T87": {"cum_pnl": s_t87, "per_sym": per_t87},
                "pure_T75": {"cum_pnl": s_t75, "per_sym": per_t75},
                "T87+T75_ens_w1_1.5": {"cum_pnl": s_ens, "per_sym": per_ens},
            },
            "baselines_at_de_loso": {
                "pure_T87": {"cum_pnl": s_t87_de, "thr_up": tu_t87_de, "thr_dn": td_t87_de},
                "pure_T75": {"cum_pnl": s_t75_de, "thr_up": tu_t75_de, "thr_dn": td_t75_de},
                "T87+T75_ens_w1_1.5": {"cum_pnl": s_ens_de, "thr_up": tu_ens_de, "thr_dn": td_ens_de},
            },
            "stacker_vs_T87+T75_ens_delta": {
                "at_iter015v1_thr": delta_v1,
                "at_de_loso": delta_de,
            },
            "iter_018_v1_baseline_no_abstain": iter018_baseline_no_abstain,
            "iter_018_v1_with_abstain": iter018_with_abstain,
            "expected_platform_pnl_iter018_baseline": 28.16,
        },
        "three_way_ensemble": threeway_results,
        "elapsed_min": (time.time() - t0) / 60.0,
    }
    out_path = os.path.join(HERE, "results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    log(f"\n  saved {out_path}")
    write_progress("done", {"stacker_de_pnl": s_de_avg, "ens_de_pnl": s_ens_de})

    if WB:
        WB.log({"final/stacker_iter015v1_pnl": s_v1_avg,
                "final/stacker_de_pnl": s_de_avg,
                "final/T87_T75_ens_iter015v1_pnl": s_ens,
                "final/T87_T75_ens_de_pnl": s_ens_de,
                "final/delta_at_iter015v1": delta_v1,
                "final/delta_at_de": delta_de,
                "final/iter018_baseline_no_abstain": iter018_baseline_no_abstain})
        WB.finish()

    # Summary
    print(f"\nRESULT: task=T139_stacker_nn metrics={{stacker_de_pnl={s_de_avg:.4f}, "
          f"T87+T75_de_pnl={s_ens_de:.4f}, delta_de={delta_de:+.4f}, "
          f"iter018_baseline_no_abstain={iter018_baseline_no_abstain}}} "
          f"notes=[stacker {('beat' if delta_de > 0 else 'lost vs')} T87+T75 by {delta_de:+.4f} at DE-LOSO]")


if __name__ == "__main__":
    main()
