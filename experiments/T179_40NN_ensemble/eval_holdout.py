"""T179: Holdout eval (dates 96-119) for 40-NN ensemble vs T170 5-NN baseline.

All models trained on full data 0-119 -> this is IN-SAMPLE for the models.
Used only as a sanity check (mean/variance comparison), NOT a decision criterion.
Real signal = platform result.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CACHE_DIR = os.path.join(ROOT, "experiments", "T68_stage5_features", "cache")
V2_PKG = os.path.join(ROOT, "experiments", "R_full_retrain", "pkg_iter019_v2")

FEE = 0.0001
SEEDS = (1, 7, 13, 42, 100)
NN_VARIANTS = ["v1","v2","v3","v4","v5","v6","v7","v8"]

# Thresholds (from T179 thresholds.json = T170/iter_018)
THR_UP = 0.00029995433796130955
THR_DN = 0.0002158528296175958
W_NN = 1.0
W_LGB = 1.5
CONFORMAL = {
    "enabled": True,
    "per_sym_beta": {"0": 0.10, "1": 0.40, "2": 0.30, "3": 0.00, "4": 0.00},
    "default_beta_for_ood": 0.16,
    "per_sym_sigma": {
        "0": 0.0002403901, "1": 0.0004712397, "2": 0.0004522216,
        "3": 0.0004235249, "4": 0.0004279811,
    },
    "default_sigma_for_ood": 0.0003998317,
}


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
        return (h @ self.WF.T + self.bF).squeeze(-1) / self.target_scale


def ev_gate_pnl(pred, sym_arr, mp_t, mp_th):
    """EV-gate with per-sym conformal band."""
    psb = CONFORMAL["per_sym_beta"]
    pss = CONFORMAL["per_sym_sigma"]
    def_beta = CONFORMAL["default_beta_for_ood"]
    def_sigma = CONFORMAL["default_sigma_for_ood"]
    eff_up = np.full(len(pred), THR_UP, dtype=np.float64)
    eff_dn = np.full(len(pred), THR_DN, dtype=np.float64)
    for sym_id in np.unique(sym_arr):
        mask = (sym_arr == sym_id)
        b = float(psb.get(str(sym_id), def_beta))
        s = float(pss.get(str(sym_id), def_sigma))
        eff_up[mask] += b * s
        eff_dn[mask] += b * s

    action = np.full(len(pred), 1, dtype=np.int8)
    action[pred > eff_up] = 2
    action[pred < -eff_dn] = 0
    side = action.astype(np.float64) - 1.0
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    pnl_arr = (side * diff - fee_pnl) / denom
    return float(pnl_arr.sum()), int((action != 1).sum())


def main():
    print("=== T179 40-NN Holdout Eval (INFLATED in-sample, sanity only) ===", flush=True)

    # Load test data
    test_d = np.load(os.path.join(CACHE_DIR, "schemeP_test.npz"))
    X_te = test_d["X"]
    mp_t = test_d["mp_t"]
    mp_th = test_d["mp_t60"]
    sym_arr = test_d["sym"].astype(np.int64)
    print(f"Test: {len(X_te):,} rows, date {test_d['date'].min()}-{test_d['date'].max()}", flush=True)

    # LGB feature mapping
    with open(os.path.join(CACHE_DIR, "schemeP_feat_names.txt")) as f:
        all_feat_names = [l.strip() for l in f]
    m0 = lgb.Booster(model_file=os.path.join(HERE, "model_h60_seed1.txt"))
    lgb_keep_idx = np.array([all_feat_names.index(fn) for fn in m0.feature_name()], dtype=np.int64)
    X_te_lgb = X_te[:, lgb_keep_idx]
    del m0

    # LGB predictions (5 seeds, v2 LGB = T170 LGB)
    print("Loading LGB models (5 seeds)...", flush=True)
    lgb_preds = []
    for s in SEEDS:
        m = lgb.Booster(model_file=os.path.join(HERE, f"model_h60_seed{s}.txt"))
        p = m.predict(X_te_lgb).astype(np.float32)
        lgb_preds.append(p)
    pred_lgb = np.mean(lgb_preds, axis=0)
    print(f"  LGB mean={pred_lgb.mean():.6f} std={pred_lgb.std():.6f}", flush=True)

    # T170 5-NN baseline prediction
    print("Loading T170 5-NN baseline (v1 only)...", flush=True)
    t170_preds = []
    for s in SEEDS:
        m = MLPNumpy(os.path.join(HERE, f"nn_h60_v1_seed{s}.npz"))
        p = m.predict(X_te)
        t170_preds.append(p)
    pred_nn_5 = np.mean(t170_preds, axis=0)

    # T179 40-NN prediction (all 8 variants x 5 seeds)
    print("Loading T179 40-NN ensemble (8 variants x 5 seeds)...", flush=True)
    all_nn_preds = []
    for variant in NN_VARIANTS:
        for s in SEEDS:
            m = MLPNumpy(os.path.join(HERE, f"nn_h60_{variant}_seed{s}.npz"))
            p = m.predict(X_te)
            all_nn_preds.append(p)
    pred_nn_40 = np.mean(all_nn_preds, axis=0)
    print(f"  40-NN mean={pred_nn_40.mean():.6f} std={pred_nn_40.std():.6f}", flush=True)
    print(f"  5-NN  mean={pred_nn_5.mean():.6f} std={pred_nn_5.std():.6f}", flush=True)

    # Combined predictions
    pred_T170 = (W_NN * pred_nn_5 + W_LGB * pred_lgb) / (W_NN + W_LGB)
    pred_T179 = (W_NN * pred_nn_40 + W_LGB * pred_lgb) / (W_NN + W_LGB)

    # Evaluate
    pnl_T170, nact_T170 = ev_gate_pnl(pred_T170, sym_arr, mp_t, mp_th)
    pnl_T179, nact_T179 = ev_gate_pnl(pred_T179, sym_arr, mp_t, mp_th)

    print(f"\n=== HOLDOUT RESULTS (date 96-119, IN-SAMPLE) ===")
    print(f"  T170 5-NN:  pnl={pnl_T170:+.4f}  n_act={nact_T170:,}")
    print(f"  T179 40-NN: pnl={pnl_T179:+.4f}  n_act={nact_T179:,}")
    print(f"  delta: {pnl_T179 - pnl_T170:+.4f}")

    # Per-sym
    print("\n  Per-sym breakdown:")
    for sym_id in range(5):
        mask = (sym_arr == sym_id)
        if mask.sum() == 0:
            continue
        p170 = pred_T170[mask]; p179 = pred_T179[mask]
        pv, _ = ev_gate_pnl(p170, sym_arr[mask], mp_t[mask], mp_th[mask])
        pt, _ = ev_gate_pnl(p179, sym_arr[mask], mp_t[mask], mp_th[mask])
        print(f"    sym{sym_id}: T170={pv:+.3f}  T179={pt:+.3f}  delta={pt-pv:+.3f}")

    results = {
        "task": "T179 40-NN ensemble (8 M7 variants x 5 seeds)",
        "n_nns_total": 40,
        "n_lgbs": 5,
        "T170_5NN_holdout_insample": round(pnl_T170, 6),
        "T179_40NN_holdout_insample": round(pnl_T179, 6),
        "delta_holdout": round(pnl_T179 - pnl_T170, 6),
        "n_active_T170": nact_T170,
        "n_active_T179": nact_T179,
        "warning": "IN-SAMPLE: all models trained on 0-119 including test. Mean comparison valid, absolute PnL inflated.",
    }
    out = os.path.join(HERE, "holdout_eval.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults -> {out}")
    return results


if __name__ == "__main__":
    main()
