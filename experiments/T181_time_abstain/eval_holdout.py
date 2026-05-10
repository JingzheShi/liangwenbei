"""T181: Holdout eval for time-window abstain variants V1/V2/V3.

Uses NPZ cache (schemeP_test.npz) which has sess_idx and t fields:
  sess_idx : 0 = AM session, 1 = PM session
  t        : last-tick index within session (0-indexed), range [99, 1940]
             session_minute = t * 3.0 / 60.0

Time abstain logic mirrors the Predictor.py wrapper exactly.
All model predictions come from T170 pkg (unchanged). Only the final
action for abstained rows is overridden to 1 (hold).

Caveat: models trained on 0-119 -> in-sample for test set (inflated).
Delta vs T170 baseline is the meaningful signal, not absolute PnL.
"""
from __future__ import annotations
import json, os
import numpy as np
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
T170_PKG = os.path.join(ROOT, "experiments", "T170_T87M7_clean_pkg")
CACHE_DIR = os.path.join(ROOT, "experiments", "T68_stage5_features", "cache")

FEE = 0.0001
SEEDS = (1, 7, 13, 42, 100)
W_NN = 1.0
W_LGB = 1.5

# T170 thresholds/conformal config (from thresholds.json)
with open(os.path.join(T170_PKG, "thresholds.json")) as _f:
    _tcfg = json.load(_f)
_h60 = next(h for h in _tcfg["horizons"] if h["h"] == 60)
THR_UP = float(_h60["thr_up"])
THR_DN = float(_h60["thr_dn"])
_cw = _tcfg.get("conformal_wrapper", {"enabled": False})
CONFORMAL = _cw

# Time abstain windows for each variant
# Each window: (sess_idx, lo_session_minute, hi_session_minute)  [inclusive]
# session_minute = t * 3.0 / 60.0 where t is the last-tick index in the NPZ
VARIANTS = {
    "V1_conservative": [
        (0, 0.0,  10.0),   # AM first 10 min  (9:40-9:50)
        (0, 90.0, 100.0),  # AM last 10 min   (11:10-11:20)
        (1, 0.0,  10.0),   # PM first 10 min  (13:10-13:20)
        (1, 90.0, 100.0),  # PM last 10 min   (14:40-14:50)
    ],
    "V2_only_open": [
        (0, 0.0, 5.0),   # AM first 5 min  (9:40-9:45)
        (1, 0.0, 5.0),   # PM first 5 min  (13:10-13:15)
    ],
    "V3_only_close": [
        (0, 95.0, 100.0),  # AM last 5 min  (11:15-11:20)
        (1, 95.0, 100.0),  # PM last 5 min  (14:45-14:50)
    ],
}

ABSTAIN_WINDOWS_INFO = {
    "V1_conservative": {
        "description": "abstain first 10 min + last 10 min of each session",
        "am_windows": ["9:40-9:50", "11:10-11:20"],
        "pm_windows": ["13:10-13:20", "14:40-14:50"],
    },
    "V2_only_open": {
        "description": "abstain first 5 min of each session only",
        "am_windows": ["9:40-9:45"],
        "pm_windows": ["13:10-13:15"],
    },
    "V3_only_close": {
        "description": "abstain last 5 min of each session only",
        "am_windows": ["11:15-11:20"],
        "pm_windows": ["14:45-14:50"],
    },
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


def build_abstain_mask(sess_arr, t_arr, windows):
    """Build abstain mask from sess_idx and t arrays using window definitions."""
    mask = np.zeros(len(sess_arr), dtype=bool)
    session_minute = t_arr * 3.0 / 60.0
    for (w_sess, lo, hi) in windows:
        m = (sess_arr == w_sess) & (session_minute >= lo) & (session_minute <= hi)
        mask |= m
    return mask


def ev_gate_pnl_with_abstain(pred, sym_arr, mp_t, mp_th, abstain_mask=None):
    """EV-gate + per-sym conformal band, with optional abstain override."""
    eff_up = np.full(len(pred), THR_UP, dtype=np.float64)
    eff_dn = np.full(len(pred), THR_DN, dtype=np.float64)
    if CONFORMAL.get("enabled"):
        betas = CONFORMAL["per_sym_beta"]
        sigmas = CONFORMAL["per_sym_sigma"]
        def_beta = CONFORMAL.get("default_beta_for_ood", 0.16)
        def_sigma = CONFORMAL.get("default_sigma_for_ood", 4e-4)
        for sym_id in np.unique(sym_arr):
            mask = (sym_arr == sym_id)
            b = float(betas.get(str(sym_id), def_beta))
            s = float(sigmas.get(str(sym_id), def_sigma))
            eff_up[mask] += b * s
            eff_dn[mask] += b * s

    action = np.full(len(pred), 1, dtype=np.int8)
    action[pred > eff_up] = 2
    action[pred < -eff_dn] = 0

    # Apply time abstain: force action=1 for abstained rows
    if abstain_mask is not None:
        action[abstain_mask] = 1

    n_abstained = int(abstain_mask.sum()) if abstain_mask is not None else 0
    side = action.astype(np.float64) - 1.0
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    pnl_arr = (side * diff - fee_pnl) / denom
    return float(pnl_arr.sum()), int((action != 1).sum()), n_abstained


def main():
    print("=== T181 Time-Window Abstain Holdout Eval ===", flush=True)

    # Load test data
    test_d = np.load(os.path.join(CACHE_DIR, "schemeP_test.npz"))
    X_te = test_d["X"]
    mp_t = test_d["mp_t"]
    mp_th = test_d["mp_t60"]
    sym_arr = test_d["sym"].astype(np.int64)
    sess_arr = test_d["sess_idx"].astype(np.int64)
    t_arr = test_d["t"].astype(np.int64)
    print(f"Test: {len(X_te):,} rows, date {test_d['date'].min()}-{test_d['date'].max()}", flush=True)
    print(f"t range: [{t_arr.min()}, {t_arr.max()}], session_minute range: [{t_arr.min()*3/60:.2f}, {t_arr.max()*3/60:.2f}]", flush=True)

    # LGB feature mapping
    with open(os.path.join(CACHE_DIR, "schemeP_feat_names.txt")) as f:
        all_feat_names = [l.strip() for l in f]
    m0 = lgb.Booster(model_file=os.path.join(T170_PKG, "model_h60_seed1.txt"))
    lgb_keep_idx = np.array([all_feat_names.index(fn) for fn in m0.feature_name()], dtype=np.int64)
    del m0
    X_te_lgb = X_te[:, lgb_keep_idx]

    # LGB predictions (5 seeds, T170 LGB unchanged)
    print("Loading LGB models (5 seeds)...", flush=True)
    lgb_preds = []
    for s in SEEDS:
        m = lgb.Booster(model_file=os.path.join(T170_PKG, f"model_h60_seed{s}.txt"))
        lgb_preds.append(m.predict(X_te_lgb).astype(np.float32))
    pred_lgb = np.mean(lgb_preds, axis=0)

    # T170 NN predictions (5 seeds)
    print("Loading T170 NN models (5 seeds)...", flush=True)
    nn_preds = []
    for s in SEEDS:
        m = MLPNumpy(os.path.join(T170_PKG, f"nn_h60_seed{s}.npz"))
        nn_preds.append(m.predict(X_te))
    pred_nn = np.mean(nn_preds, axis=0)

    # Combined T170 prediction
    pred_T170 = (W_NN * pred_nn + W_LGB * pred_lgb) / (W_NN + W_LGB)

    # T170 baseline (no abstain)
    pnl_T170, nact_T170, _ = ev_gate_pnl_with_abstain(pred_T170, sym_arr, mp_t, mp_th)
    print(f"\nT170 baseline: pnl={pnl_T170:+.4f}  n_act={nact_T170:,}", flush=True)

    # Evaluate each variant
    variant_results = {}
    for vname, windows in VARIANTS.items():
        abstain_mask = build_abstain_mask(sess_arr, t_arr, windows)
        n_abs = int(abstain_mask.sum())
        pct_abs = 100.0 * n_abs / len(abstain_mask)
        pnl_v, nact_v, n_abs_check = ev_gate_pnl_with_abstain(
            pred_T170, sym_arr, mp_t, mp_th, abstain_mask
        )
        delta = pnl_v - pnl_T170
        print(f"\n{vname}: pnl={pnl_v:+.4f}  delta={delta:+.4f}  n_abstained={n_abs:,} ({pct_abs:.2f}%)  n_act={nact_v:,}", flush=True)

        # Per-sym breakdown
        print(f"  Per-sym:", flush=True)
        per_sym = {}
        for sym_id in range(5):
            mask = (sym_arr == sym_id)
            if mask.sum() == 0:
                continue
            abs_m = abstain_mask[mask]
            pv, _, _ = ev_gate_pnl_with_abstain(pred_T170[mask], sym_arr[mask], mp_t[mask], mp_th[mask])
            pt, _, _ = ev_gate_pnl_with_abstain(pred_T170[mask], sym_arr[mask], mp_t[mask], mp_th[mask], abs_m)
            per_sym[str(sym_id)] = round(pt - pv, 6)
            print(f"    sym{sym_id}: T170={pv:+.3f}  {vname}={pt:+.3f}  delta={pt-pv:+.3f}", flush=True)

        variant_results[vname] = {
            "windows": [{"sess": w[0], "lo_min": w[1], "hi_min": w[2]} for w in windows],
            "n_abstained": n_abs,
            "pct_abstained": round(pct_abs, 3),
            "holdout_pnl": round(pnl_v, 6),
            "delta_vs_T170": round(delta, 6),
            "n_active": nact_v,
            "per_sym_delta": per_sym,
            **ABSTAIN_WINDOWS_INFO[vname],
        }

    results = {
        "task": "T181 time-window hard abstain (holdout eval)",
        "T170_baseline_holdout": round(pnl_T170, 6),
        "T170_n_active": nact_T170,
        "variants": variant_results,
        "warning": "IN-SAMPLE eval (models trained on 0-119 including test). Delta meaningful, absolute PnL inflated.",
    }

    out = os.path.join(HERE, "holdout_eval.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults -> {out}", flush=True)
    return results


if __name__ == "__main__":
    main()
