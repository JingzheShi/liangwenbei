"""T178: Evaluate 10-seed LGB bag vs T170 5-seed baseline on holdout 96-119."""
from __future__ import annotations
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

# Paths
T170_PKG = os.path.join(ROOT, "experiments", "T170_T87M7_clean_pkg")
T170_M7_DIR = os.path.join(ROOT, "experiments", "T170_T87_M7")
R_FULL = os.path.join(ROOT, "experiments", "R_full_retrain")
T178_DIR = HERE
CACHE_DIR = os.path.join(ROOT, "experiments", "T68_stage5_features", "cache")

FEE = 0.0001
OLD_SEEDS = (1, 7, 13, 42, 100)
NEW_SEEDS = (200, 211, 222, 233, 244)


def _gelu_tanh(x):
    c = 0.7978845608028654
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))

def _layernorm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta

class MLPNumpy:
    def __init__(self, npz_path):
        d = np.load(npz_path, allow_pickle=False)
        self.in_dim = int(d["in_dim"][0])
        self.hidden = list(d["hidden"].tolist())
        self.use_layernorm = bool(d["use_layernorm"][0])
        self.target_scale = float(d["target_scale"][0])
        self.feat_mean = d["feat_mean"].astype(np.float32)
        self.feat_std = np.maximum(d["feat_std"].astype(np.float32), 1e-6)
        self.keep_idx = d["keep_idx"].astype(np.int64)
        self.clip = float(d["clip"][0])
        self.W, self.b, self.LN_W, self.LN_b = [], [], [], []
        for i in range(len(self.hidden)):
            self.W.append(d[f"L{i}_W"].astype(np.float32))
            self.b.append(d[f"L{i}_b"].astype(np.float32))
            if self.use_layernorm:
                self.LN_W.append(d[f"LN{i}_W"].astype(np.float32))
                self.LN_b.append(d[f"LN{i}_b"].astype(np.float32))
        self.WF = d["LF_W"].astype(np.float32)
        self.bF = d["LF_b"].astype(np.float32)
    def predict(self, X_all):
        Xs = X_all[:, self.keep_idx].astype(np.float32, copy=True)
        Xs = (Xs - self.feat_mean) / self.feat_std
        Xs = np.where(np.isnan(Xs), 0.0, Xs)
        Xs = np.clip(Xs, -self.clip, self.clip).astype(np.float32)
        h = Xs
        for i in range(len(self.hidden)):
            h = h @ self.W[i].T + self.b[i]
            if self.use_layernorm:
                h = _layernorm(h, self.LN_W[i], self.LN_b[i])
            h = _gelu_tanh(h)
        out = h @ self.WF.T + self.bF
        return (out.squeeze(-1) / self.target_scale).astype(np.float32)


def ev_gate_pnl(pred, sym_arr, mp_t, mp_th, thr_up, thr_dn, conformal_cfg=None):
    eff_up = np.full(len(pred), thr_up, dtype=np.float64)
    eff_dn = np.full(len(pred), thr_dn, dtype=np.float64)
    if conformal_cfg and conformal_cfg.get("enabled"):
        betas = conformal_cfg["per_sym_beta"]
        sigmas = conformal_cfg["per_sym_sigma"]
        def_beta = conformal_cfg.get("default_beta_for_ood", 0.16)
        def_sigma = conformal_cfg.get("default_sigma_for_ood", 0.0004)
        for sym_id in np.unique(sym_arr):
            mask = (sym_arr == sym_id)
            b = betas.get(str(sym_id), def_beta)
            s = sigmas.get(str(sym_id), def_sigma)
            eff_up[mask] += b * s
            eff_dn[mask] += b * s
    action = np.full(len(pred), 1, dtype=np.int8)
    action[pred > eff_up] = 2
    action[pred < -eff_dn] = 0
    side = action.astype(np.float64) - 1.0
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    pnl = (side * diff - fee_pnl) / denom
    return float(pnl.sum()), int((action != 1).sum())


def main():
    import lightgbm as lgb

    # Load thresholds from T170 clean pkg
    with open(os.path.join(T170_PKG, "thresholds.json")) as f:
        tcfg = json.load(f)
    h60_cfg = next(h for h in tcfg["horizons"] if h["h"] == 60)
    thr_up = h60_cfg["thr_up"]
    thr_dn = h60_cfg["thr_dn"]
    w_nn = h60_cfg.get("w_nn", 1.0)
    w_lgb = h60_cfg.get("w_lgb", 1.5)
    conformal_cfg = tcfg.get("conformal_wrapper", {"enabled": False})
    print(f"Config: thr_up={thr_up:.6f} thr_dn={thr_dn:.6f} w_nn={w_nn} w_lgb={w_lgb}")

    # Load test data (dates 96-119)
    test_d = np.load(os.path.join(CACHE_DIR, "schemeP_test.npz"))
    X_te = test_d["X"]
    mp_t = test_d["mp_t"]
    mp_th = test_d["mp_t60"]
    sym_arr = test_d["sym"]
    print(f"Test: {len(X_te):,} rows, date {test_d['date'].min()}-{test_d['date'].max()}")

    # Build LGB feature index
    with open(os.path.join(CACHE_DIR, "schemeP_feat_names.txt")) as f:
        all_feat_names = [l.strip() for l in f]
    m0 = lgb.Booster(model_file=os.path.join(T170_PKG, f"model_h60_seed1.txt"))
    lgb_keep_idx = np.array([all_feat_names.index(fn) for fn in m0.feature_name()], dtype=np.int64)
    X_te_lgb = X_te[:, lgb_keep_idx]
    del m0

    # Load 5 old LGB seeds (from T170 clean pkg = R_full_retrain M7)
    print("\nLoading 5 old LGB seeds (T170 pkg)...")
    old_lgb_preds = []
    for s in OLD_SEEDS:
        m = lgb.Booster(model_file=os.path.join(T170_PKG, f"model_h60_seed{s}.txt"))
        p = m.predict(X_te_lgb)
        old_lgb_preds.append(p.astype(np.float32))
        print(f"  LGB seed{s}: mean={p.mean():.6f} std={p.std():.6f}")
    pred_lgb_5 = np.mean(old_lgb_preds, axis=0)

    # Load 5 new LGB seeds (from T178 dir)
    print("\nLoading 5 new LGB seeds (T178)...")
    new_lgb_preds = []
    train_times = {}
    for s in NEW_SEEDS:
        path = os.path.join(T178_DIR, f"model_h60_seed{s}.txt")
        m = lgb.Booster(model_file=path)
        p = m.predict(X_te_lgb)
        new_lgb_preds.append(p.astype(np.float32))
        print(f"  LGB seed{s}: mean={p.mean():.6f} std={p.std():.6f}")
        # Get training time from summary if available
        sum_path = os.path.join(T178_DIR, f"summary_h60_seed{s}.json")
        if os.path.exists(sum_path):
            with open(sum_path) as f:
                sdata = json.load(f)
            train_times[str(s)] = sdata.get("train_time_sec", None)
    pred_lgb_10 = np.mean(old_lgb_preds + new_lgb_preds, axis=0)

    # Load T170 M7 NN (5 seeds)
    print("\nLoading T170 M7 NN (5 seeds)...")
    nn_preds = []
    for s in OLD_SEEDS:
        m = MLPNumpy(os.path.join(T170_M7_DIR, f"nn_h60_seed{s}.npz"))
        p = m.predict(X_te)
        nn_preds.append(p)
        print(f"  NN seed{s}: mean={p.mean():.6f} std={p.std():.6f}")
    pred_nn = np.mean(nn_preds, axis=0)

    # Ensemble predictions
    pred_T170_5seed = (w_nn * pred_nn + w_lgb * pred_lgb_5) / (w_nn + w_lgb)
    pred_T178_10seed = (w_nn * pred_nn + w_lgb * pred_lgb_10) / (w_nn + w_lgb)

    # Evaluate
    pnl_T170, nact_T170 = ev_gate_pnl(pred_T170_5seed, sym_arr, mp_t, mp_th, thr_up, thr_dn, conformal_cfg)
    pnl_T178, nact_T178 = ev_gate_pnl(pred_T178_10seed, sym_arr, mp_t, mp_th, thr_up, thr_dn, conformal_cfg)

    print(f"\n=== HOLDOUT RESULTS (date 96-119, IN-SAMPLE: LGB+NN trained on 0-119) ===")
    print(f"  T170 5-seed LGB (baseline): pnl={pnl_T170:+.4f}  n_act={nact_T170:,}")
    print(f"  T178 10-seed LGB:           pnl={pnl_T178:+.4f}  n_act={nact_T178:,}")
    print(f"  delta: {pnl_T178 - pnl_T170:+.4f}")

    # Per-sym
    print("\n  Per-sym breakdown:")
    for sym_id in range(5):
        mask = (sym_arr == sym_id)
        if mask.sum() == 0:
            continue
        p5, _ = ev_gate_pnl(pred_T170_5seed[mask], sym_arr[mask], mp_t[mask], mp_th[mask], thr_up, thr_dn, None)
        p10, _ = ev_gate_pnl(pred_T178_10seed[mask], sym_arr[mask], mp_t[mask], mp_th[mask], thr_up, thr_dn, None)
        print(f"    sym{sym_id}: T170={p5:+.3f}  T178={p10:+.3f}  delta={p10-p5:+.3f}")

    delta = pnl_T178 - pnl_T170
    results = {
        "task": "T178 LGB 10-seed bag (M7 retrain 5 new seeds)",
        "T170_5seed_holdout": round(pnl_T170, 6),
        "T178_10seed_holdout": round(pnl_T178, 6),
        "delta": round(delta, 6),
        "new_seeds": list(NEW_SEEDS),
        "training_times_per_seed": train_times,
        "config": {"thr_up": thr_up, "thr_dn": thr_dn, "w_nn": w_nn, "w_lgb": w_lgb},
        "note": "IN-SAMPLE: both LGB and NN trained on 0-119 (includes test 96-119). Delta measures variance reduction from 10-seed bag."
    }
    out_path = os.path.join(HERE, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults -> {out_path}")
    return delta

if __name__ == "__main__":
    main()
