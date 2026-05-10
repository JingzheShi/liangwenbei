"""T183 holdout eval: compare T183 random-init NN vs T170 warm-start NN.

Both use T170_T87M7_clean_pkg LGB + thresholds. Only swap the NN weights.
Holdout = test 96-119 (in-sample for both M7 variants).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

T170_PKG = os.path.join(ROOT, "experiments", "T170_T87M7_clean_pkg")
T170_DIR = os.path.join(ROOT, "experiments", "T170_T87_M7")
T183_DIR = HERE
CACHE_DIR = os.path.join(ROOT, "experiments", "T68_stage5_features", "cache")

FEE = 0.0001
SEEDS = (1, 7, 13, 42, 100)


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
    print(f"Config (T170 pkg): thr_up={thr_up:.6f} thr_dn={thr_dn:.6f} w_nn={w_nn} w_lgb={w_lgb}")
    print(f"Conformal: {conformal_cfg.get('enabled', False)}")

    # Load test data
    test_d = np.load(os.path.join(CACHE_DIR, "schemeP_test.npz"))
    X_te = test_d["X"]
    mp_t = test_d["mp_t"]
    mp_th = test_d["mp_t60"]
    sym_arr = test_d["sym"]
    print(f"Test: {len(X_te):,} rows, date {test_d['date'].min()}-{test_d['date'].max()}")

    with open(os.path.join(CACHE_DIR, "schemeP_feat_names.txt")) as f:
        all_feat_names = [l.strip() for l in f]

    # Load LGB models from T170 clean pkg (unchanged)
    print("\nLoading LGB models (T170 pkg)...")
    m0 = lgb.Booster(model_file=os.path.join(T170_PKG, f"model_h60_seed1.txt"))
    lgb_keep_idx = np.array([all_feat_names.index(fn) for fn in m0.feature_name()],
                             dtype=np.int64)
    X_te_lgb = X_te[:, lgb_keep_idx]
    del m0
    lgb_preds = []
    for s in SEEDS:
        m = lgb.Booster(model_file=os.path.join(T170_PKG, f"model_h60_seed{s}.txt"))
        p = m.predict(X_te_lgb)
        lgb_preds.append(p.astype(np.float32))
        print(f"  LGB seed{s}: mean={p.mean():.6f} std={p.std():.6f}")
    pred_lgb = np.mean(lgb_preds, axis=0)

    # Load T170 M7 NN models (warm-start baseline)
    print("\nLoading T170 warm-start NN models...")
    t170_nn_preds = []
    for s in SEEDS:
        m = MLPNumpy(os.path.join(T170_DIR, f"nn_h60_seed{s}.npz"))
        p = m.predict(X_te)
        t170_nn_preds.append(p)
        print(f"  T170 NN seed{s}: mean={p.mean():.6f} std={p.std():.6f}")
    pred_nn_t170 = np.mean(t170_nn_preds, axis=0)

    # Load T183 random-init NN models
    print("\nLoading T183 random-init NN models...")
    t183_nn_preds = []
    for s in SEEDS:
        m = MLPNumpy(os.path.join(T183_DIR, f"nn_h60_seed{s}.npz"))
        p = m.predict(X_te)
        t183_nn_preds.append(p)
        print(f"  T183 NN seed{s}: mean={p.mean():.6f} std={p.std():.6f}")
    pred_nn_t183 = np.mean(t183_nn_preds, axis=0)

    # Ensemble
    pred_T170 = (w_nn * pred_nn_t170 + w_lgb * pred_lgb) / (w_nn + w_lgb)
    pred_T183 = (w_nn * pred_nn_t183 + w_lgb * pred_lgb) / (w_nn + w_lgb)

    # Evaluate with conformal wrapper
    pnl_T170, nact_T170 = ev_gate_pnl(pred_T170, sym_arr, mp_t, mp_th,
                                       thr_up, thr_dn, conformal_cfg)
    pnl_T183, nact_T183 = ev_gate_pnl(pred_T183, sym_arr, mp_t, mp_th,
                                       thr_up, thr_dn, conformal_cfg)

    print(f"\n=== HOLDOUT RESULTS (date 96-119, in-sample for both M7 variants) ===")
    print(f"  T170 M7 warm-start NN + T170 LGB: pnl={pnl_T170:+.4f}  n_act={nact_T170:,}")
    print(f"  T183 M7 random-init NN + T170 LGB: pnl={pnl_T183:+.4f}  n_act={nact_T183:,}")
    print(f"  delta T183 - T170: {pnl_T183 - pnl_T170:+.4f}")

    # Per-sym breakdown
    print("\n  Per-sym breakdown:")
    for sym_id in range(5):
        mask = (sym_arr == sym_id)
        if mask.sum() == 0:
            continue
        p170, _ = ev_gate_pnl(pred_T170[mask], sym_arr[mask], mp_t[mask], mp_th[mask],
                               thr_up, thr_dn, None)
        p183, _ = ev_gate_pnl(pred_T183[mask], sym_arr[mask], mp_t[mask], mp_th[mask],
                               thr_up, thr_dn, None)
        print(f"    sym{sym_id}: T170={p170:+.3f}  T183={p183:+.3f}  delta={p183-p170:+.3f}")

    results = {
        "task": "T183 T87 NN with Xavier random init (no T81 warm-start), M7 retrain",
        "T170_baseline_holdout_insample": round(pnl_T170, 6),
        "T183_holdout_insample": round(pnl_T183, 6),
        "delta_T183_minus_T170": round(pnl_T183 - pnl_T170, 6),
        "n_act_T170": nact_T170,
        "n_act_T183": nact_T183,
        "warning": "Both variants trained on 0-119 (in-sample for test 96-119). Holdout comparison is RELATIVE, not absolute.",
        "config_used": {"thr_up": thr_up, "thr_dn": thr_dn, "w_nn": w_nn, "w_lgb": w_lgb,
                        "conformal": conformal_cfg.get("enabled", False)},
    }
    out_path = os.path.join(HERE, "holdout_eval.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults -> {out_path}")
    return results


if __name__ == "__main__":
    main()
