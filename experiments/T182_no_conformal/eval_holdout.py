"""T182: Compare T170 (with conformal) vs T182 (no conformal) on holdout dates 96-119.

Models are IDENTICAL — only the decision threshold differs:
  T170: thr_up = 0.000300 + beta*sigma per sym  (conformal band adds abstain)
  T182: thr_up = 0.000300                         (pure gate_asym, no band)

Both models trained on full data 0-119 -> holdout is IN-SAMPLE. Used as sanity/delta check.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CACHE_DIR = os.path.join(ROOT, "experiments", "T68_stage5_features", "cache")

FEE = 0.0001
SEEDS = (1, 7, 13, 42, 100)

THR_UP = 0.00029995433796130955
THR_DN = 0.0002158528296175958
W_NN = 1.0
W_LGB = 1.5

# T170 conformal config
CONFORMAL = {
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


def ev_gate_pnl_raw(pred, mp_t, mp_th):
    """Pure gate_asym (T182 no conformal)."""
    action = np.full(len(pred), 1, dtype=np.int8)
    action[pred > THR_UP] = 2
    action[pred < -THR_DN] = 0
    side = action.astype(np.float64) - 1.0
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    pnl_arr = (side * diff - fee_pnl) / denom
    return float(pnl_arr.sum()), int((action != 1).sum())


def ev_gate_pnl_conformal(pred, sym_arr, mp_t, mp_th):
    """EV-gate with per-sym conformal band (T170)."""
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
    print("=== T182 no-conformal vs T170 with-conformal holdout eval ===", flush=True)

    test_d = np.load(os.path.join(CACHE_DIR, "schemeP_test.npz"))
    X_te = test_d["X"]
    mp_t = test_d["mp_t"]
    mp_th = test_d["mp_t60"]
    sym_arr = test_d["sym"].astype(np.int64)
    print(f"Test: {len(X_te):,} rows, date {test_d['date'].min()}-{test_d['date'].max()}", flush=True)

    with open(os.path.join(CACHE_DIR, "schemeP_feat_names.txt")) as f:
        all_feat_names = [l.strip() for l in f]

    print("Loading LGB models (5 seeds)...", flush=True)
    m0 = lgb.Booster(model_file=os.path.join(HERE, f"model_h60_seed1.txt"))
    lgb_keep_idx = np.array([all_feat_names.index(fn) for fn in m0.feature_name()], dtype=np.int64)
    X_te_lgb = X_te[:, lgb_keep_idx]
    lgb_preds = [m0.predict(X_te_lgb).astype(np.float32)]
    for s in (7, 13, 42, 100):
        m = lgb.Booster(model_file=os.path.join(HERE, f"model_h60_seed{s}.txt"))
        lgb_preds.append(m.predict(X_te_lgb).astype(np.float32))
    pred_lgb = np.mean(lgb_preds, axis=0)
    print(f"  LGB: mean={pred_lgb.mean():.6f} std={pred_lgb.std():.6f}", flush=True)

    print("Loading NN models (5 seeds)...", flush=True)
    nn_preds = []
    for s in SEEDS:
        m = MLPNumpy(os.path.join(HERE, f"nn_h60_seed{s}.npz"))
        nn_preds.append(m.predict(X_te))
    pred_nn = np.mean(nn_preds, axis=0)
    print(f"  NN:  mean={pred_nn.mean():.6f} std={pred_nn.std():.6f}", flush=True)

    pred_combined = (W_NN * pred_nn + W_LGB * pred_lgb) / (W_NN + W_LGB)

    pnl_T170, nact_T170 = ev_gate_pnl_conformal(pred_combined, sym_arr, mp_t, mp_th)
    pnl_T182, nact_T182 = ev_gate_pnl_raw(pred_combined, mp_t, mp_th)
    delta = pnl_T182 - pnl_T170

    print(f"\n=== HOLDOUT RESULTS (dates 96-119, IN-SAMPLE) ===")
    print(f"  T170 (with conformal): pnl={pnl_T170:+.4f}  n_act={nact_T170:,}")
    print(f"  T182 (no conformal):   pnl={pnl_T182:+.4f}  n_act={nact_T182:,}")
    print(f"  delta (T182-T170):     {delta:+.4f}")

    print("\n  Per-sym breakdown:")
    per_sym = {}
    for sym_id in range(5):
        mask = (sym_arr == sym_id)
        if mask.sum() == 0:
            continue
        p = pred_combined[mask]
        pv_t170, _ = ev_gate_pnl_conformal(p, sym_arr[mask], mp_t[mask], mp_th[mask])
        pv_t182, _ = ev_gate_pnl_raw(p, mp_t[mask], mp_th[mask])
        print(f"    sym{sym_id}: T170={pv_t170:+.4f}  T182={pv_t182:+.4f}  delta={pv_t182-pv_t170:+.4f}  (beta={CONFORMAL['per_sym_beta'][str(sym_id)]})")
        per_sym[str(sym_id)] = {
            "T170_pnl": round(pv_t170, 6),
            "T182_pnl": round(pv_t182, 6),
            "delta": round(pv_t182 - pv_t170, 6),
            "beta": CONFORMAL["per_sym_beta"][str(sym_id)],
        }

    results = {
        "task": "T182 no-conformal baseline vs T170 with-conformal",
        "T170_with_conformal_holdout": round(pnl_T170, 6),
        "T182_no_conformal_holdout": round(pnl_T182, 6),
        "delta": round(delta, 6),
        "n_active_T170": nact_T170,
        "n_active_T182": nact_T182,
        "per_sym": per_sym,
        "interpretation": "delta>0 means conformal HURTS at this stack level; delta<0 means conformal HELPS",
        "warning": "IN-SAMPLE: models trained on 0-119 incl. test. Delta comparison valid, absolute PnL inflated.",
    }
    out = os.path.join(HERE, "holdout_eval.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults -> {out}")
    return results


if __name__ == "__main__":
    main()
