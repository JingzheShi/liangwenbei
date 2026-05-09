"""T161: Test-Time Augmentation (TTA) evaluation on holdout dates 96-119.

Uses precomputed schemeP_test.npz features (442080 rows, dates 96-119).
Evaluates horizon h=60 only (only active horizon in T160 v2W config).

Method A: Gaussian noise on all 359 features (sigma=0.01 * feature_std)
Method C: Per-row multiplicative price-scale on price-level raw features
          (approximation: computed features are mostly scale-invariant, so
           only the 154 last-tick raw features' price-subset is scaled)

Both methods use K=5 augmentations (k=0 = original, k=1..4 = augmented).
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import lightgbm as lgb

OUTDIR = os.path.dirname(os.path.abspath(__file__))
WORKDIR = "/root/projects/liangwenbei_workdir"
PKG_DIR = os.path.join(WORKDIR, "experiments/T160_v2W_clean_pkg")
CACHE_DIR = os.path.join(WORKDIR, "experiments/T68_stage5_features/cache")

SEEDS = [1, 7, 13, 42, 100]
H = 60  # only active horizon
K = 5   # augmentations (incl. original)
SIGMA_A = 0.01  # Method A noise sigma multiplier
SCALE_RANGE = (0.97, 1.03)  # Method C price scale range

FEE = 0.0001

# === v2W Config (from thresholds.json) ===
THR_UP = 0.00029995433796130955
THR_DN = 0.0002158528296175958
W_NN = 1.0
W_LGB = 1.5
PER_SYM_BETA = {0: 0.10, 1: 0.40, 2: 0.30, 3: 0.00, 4: 0.00}
PER_SYM_SIGMA = {
    0: 0.0002403901, 1: 0.0004712397, 2: 0.0004522216,
    3: 0.0004235249, 4: 0.0004279811,
}
DEFAULT_BETA_OOD = 0.16
DEFAULT_SIGMA_OOD = 0.0003998317
AGREE_FILTER_SYMS = frozenset({0, 1, 2})

FAIL_NAMES = frozenset({
    "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
    "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
    "qrank_W100_cumspread",
    "kyle_lam_W50", "kyle_lam_W100",
    "roll_eff_spr_ratio_W100",
    "liq_asym_top5_W5",
})

# Price-level raw feature names that scale with price (for Method C)
_PRICE_SCALE_COLS = (
    "open", "high", "low", "close",
    *[f"bid{k}" for k in range(1, 11)],
    *[f"ask{k}" for k in range(1, 11)],
    "avgbid", "avgask",
    *[f"midprice{k}" for k in range(1, 11)],
    *[f"spread{k}" for k in range(1, 11)],
    *[f"bid_diff{k}" for k in range(1, 11)],
    *[f"ask_diff{k}" for k in range(1, 11)],
    "bid_mean", "ask_mean", "cumspread",
)


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _update_progress(step: str, metrics: Optional[Dict] = None) -> None:
    data = {
        "status": "running",
        "step": step,
        "metrics": metrics or {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(OUTDIR, "worker-progress.json"), "w") as f:
        json.dump(data, f, indent=2)
    _log(step)


# ── MLP inference ───────────────────────────────────────────────────────────

def _gelu_tanh(x: np.ndarray) -> np.ndarray:
    c = 0.7978845608028654
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x * x * x)))


def _layernorm(x: np.ndarray, g: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    m = x.mean(axis=-1, keepdims=True)
    v = x.var(axis=-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b


class _MLP:
    def __init__(self, path: str):
        d = np.load(path, allow_pickle=False)
        self.in_dim = int(d["in_dim"][0])
        self.hidden = list(d["hidden"].tolist())
        self.use_ln = bool(d["use_layernorm"][0])
        self.target_scale = float(d["target_scale"][0])
        self.feat_mean = d["feat_mean"].astype(np.float32)
        self.feat_std = d["feat_std"].astype(np.float32)
        self.keep_idx = d["keep_idx"].astype(np.int64)
        self.clip = float(d["clip"][0])
        self.W, self.b, self.LN_W, self.LN_b = [], [], [], []
        for i in range(len(self.hidden)):
            self.W.append(d[f"L{i}_W"].astype(np.float32))
            self.b.append(d[f"L{i}_b"].astype(np.float32))
            if self.use_ln:
                self.LN_W.append(d[f"LN{i}_W"].astype(np.float32))
                self.LN_b.append(d[f"LN{i}_b"].astype(np.float32))
        self.WF = d["LF_W"].astype(np.float32)
        self.bF = d["LF_b"].astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = X if X.shape[1] == self.in_dim else X[:, self.keep_idx]
        Xs = Xs.astype(np.float32, copy=True)
        Xs = (Xs - self.feat_mean) / self.feat_std
        Xs = np.where(np.isnan(Xs), 0.0, Xs)
        Xs = np.clip(Xs, -self.clip, self.clip).astype(np.float32)
        h = Xs
        for i in range(len(self.hidden)):
            h = h @ self.W[i].T + self.b[i]
            if self.use_ln:
                h = _layernorm(h, self.LN_W[i], self.LN_b[i])
            h = _gelu_tanh(h)
        out = h @ self.WF.T + self.bF
        return (out.squeeze(-1) / self.target_scale).astype(np.float32)


# ── Ensemble predict ─────────────────────────────────────────────────────────

def _pred_lgb(boosters: list, X: np.ndarray) -> np.ndarray:
    acc = None
    for b in boosters:
        p = b.predict(X).astype(np.float32)
        acc = p if acc is None else acc + p
    return acc / len(boosters)


def _pred_nn(nns: list, X: np.ndarray) -> np.ndarray:
    acc = None
    for n in nns:
        p = n.predict(X)
        acc = p if acc is None else acc + p
    return acc / len(nns)


# ── Gating logic ─────────────────────────────────────────────────────────────

def _get_band(sym_arr: np.ndarray) -> np.ndarray:
    band = np.full(len(sym_arr), DEFAULT_BETA_OOD * DEFAULT_SIGMA_OOD, dtype=np.float64)
    for s, b in PER_SYM_BETA.items():
        band[sym_arr == s] = b * PER_SYM_SIGMA[s]
    return band


def _apply_gate_agreement(
    lgb_pred: np.ndarray, nn_pred: np.ndarray, sym_arr: np.ndarray
) -> np.ndarray:
    combined = (W_NN * nn_pred + W_LGB * lgb_pred) / (W_NN + W_LGB)
    band = _get_band(sym_arr)
    actions = np.ones(len(combined), dtype=np.int8)
    actions[combined > (THR_UP + band)] = 2
    actions[combined < -(THR_DN + band)] = 0
    # Agreement filter on syms 0,1,2
    disagree = np.sign(nn_pred) != np.sign(lgb_pred)
    filter_mask = np.isin(sym_arr, list(AGREE_FILTER_SYMS))
    actions[filter_mask & disagree] = 1
    return actions


def _compute_pnl(actions: np.ndarray, mp_t: np.ndarray, mp_th: np.ndarray) -> float:
    side = actions.astype(np.float64) - 1.0
    abs_side = np.abs(side)
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * abs_side * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return float(((side * diff - fee_pnl) / denom).sum())


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    _update_progress("Loading schemeP_test.npz")
    d = np.load(os.path.join(CACHE_DIR, "schemeP_test.npz"))
    X_370 = d["X"].astype(np.float32)  # (442080, 370)
    sym_arr = d["sym"].astype(np.int32)
    mp_t = d["mp_t"].astype(np.float64)
    mp_t60 = d["mp_t60"].astype(np.float64)
    N = len(X_370)
    _log(f"  Loaded {N} rows, dates {d['date'].min()}-{d['date'].max()}")

    # Feature name mapping: 370 → 359 (remove FAIL_NAMES)
    with open(os.path.join(CACHE_DIR, "schemeP_feat_names.txt")) as f:
        feat_names_370 = [l.strip() for l in f]
    keep_idx_370 = np.array(
        [i for i, n in enumerate(feat_names_370) if n not in FAIL_NAMES], dtype=np.int64
    )
    assert len(keep_idx_370) == 359, f"Expected 359, got {len(keep_idx_370)}"
    X_359 = X_370[:, keep_idx_370]  # (N, 359) float32

    # Price-scale col indices in X_370 (for Method C)
    price_col_idx_370 = np.array(
        [i for i, n in enumerate(feat_names_370) if n in _PRICE_SCALE_COLS], dtype=np.int64
    )
    _log(f"  Price-scale cols in X_370: {len(price_col_idx_370)}")

    _update_progress("Loading LGB and NN models (h=60, 5 seeds each)")
    lgb_models = [
        lgb.Booster(model_file=os.path.join(PKG_DIR, f"model_h60_seed{s}.txt"))
        for s in SEEDS
    ]
    nn_models = [
        _MLP(os.path.join(PKG_DIR, f"nn_h60_seed{s}.npz"))
        for s in SEEDS
    ]
    _log("  Models loaded")

    # ── V2W Baseline ──────────────────────────────────────────────────────────
    _update_progress("Computing v2W baseline (original features)")
    t0 = time.time()
    lgb_base = _pred_lgb(lgb_models, X_359)
    nn_base = _pred_nn(nn_models, X_359)
    actions_base = _apply_gate_agreement(lgb_base, nn_base, sym_arr)
    pnl_base = _compute_pnl(actions_base, mp_t, mp_t60)
    n_trades_base = int((actions_base != 1).sum())
    _log(f"  v2W baseline: PnL={pnl_base:.6f}, n_trades={n_trades_base}, time={time.time()-t0:.1f}s")

    # ── Method A: Feature-noise TTA ───────────────────────────────────────────
    _update_progress(f"Method A: feature-noise TTA (K={K}, sigma={SIGMA_A})")
    feat_std = X_359.std(axis=0)  # (359,)
    rng = np.random.default_rng(42)
    lgb_acc_A = np.zeros(N, dtype=np.float32)
    nn_acc_A = np.zeros(N, dtype=np.float32)

    for k in range(K):
        t_k = time.time()
        if k == 0:
            X_aug = X_359
        else:
            noise = (rng.standard_normal((N, 359)) * (SIGMA_A * feat_std)).astype(np.float32)
            X_aug = X_359 + noise
        lgb_acc_A += _pred_lgb(lgb_models, X_aug) / K
        nn_acc_A += _pred_nn(nn_models, X_aug) / K
        _log(f"  Method A k={k}: {time.time()-t_k:.1f}s")

    actions_A = _apply_gate_agreement(lgb_acc_A, nn_acc_A, sym_arr)
    pnl_A = _compute_pnl(actions_A, mp_t, mp_t60)
    n_trades_A = int((actions_A != 1).sum())
    delta_A = pnl_A - pnl_base
    _log(f"  Method A: PnL={pnl_A:.6f}, delta={delta_A:+.6f}, n_trades={n_trades_A}")

    # ── Method C: Price-scale TTA (feature-level approx) ─────────────────────
    _update_progress(f"Method C: price-scale TTA (K={K}, scale={SCALE_RANGE})")
    rng_C = np.random.default_rng(123)
    lgb_acc_C = np.zeros(N, dtype=np.float32)
    nn_acc_C = np.zeros(N, dtype=np.float32)

    # Precompute: which cols in X_359 correspond to price-scale cols?
    price_col_set_370 = set(price_col_idx_370.tolist())
    price_in_359 = np.array(
        [j for j, idx in enumerate(keep_idx_370) if int(idx) in price_col_set_370],
        dtype=np.int64,
    )
    _log(f"  Price-scale cols mapped to X_359: {len(price_in_359)}")

    for k in range(K):
        t_k = time.time()
        if k == 0:
            X_aug = X_359
        else:
            # Per-row random scale factors
            scales = rng_C.uniform(SCALE_RANGE[0], SCALE_RANGE[1], size=(N,)).astype(np.float32)
            X_aug = X_359.copy()
            X_aug[:, price_in_359] *= scales[:, None]
        lgb_acc_C += _pred_lgb(lgb_models, X_aug) / K
        nn_acc_C += _pred_nn(nn_models, X_aug) / K
        _log(f"  Method C k={k}: {time.time()-t_k:.1f}s")

    actions_C = _apply_gate_agreement(lgb_acc_C, nn_acc_C, sym_arr)
    pnl_C = _compute_pnl(actions_C, mp_t, mp_t60)
    n_trades_C = int((actions_C != 1).sum())
    delta_C = pnl_C - pnl_base
    _log(f"  Method C: PnL={pnl_C:.6f}, delta={delta_C:+.6f}, n_trades={n_trades_C}")

    # ── Determine best ─────────────────────────────────────────────────────────
    best_pnl = max(pnl_A, pnl_C)
    best_method = "method_A" if pnl_A >= pnl_C else "method_C"
    best_delta = best_pnl - pnl_base

    results = {
        "task": "T161 TTA per-row K=5 augmentation",
        "v2W_baseline_holdout": pnl_base,
        "v2W_n_trades": n_trades_base,
        "method_A_feature_noise": {
            "K": K,
            "sigma": SIGMA_A,
            "holdout_pnl": pnl_A,
            "n_trades": n_trades_A,
            "delta": delta_A,
        },
        "method_C_price_scale": {
            "K": K,
            "scale_range": list(SCALE_RANGE),
            "note": "feature-level approx: scale price-level raw features only",
            "n_price_cols": len(price_col_idx_370),
            "holdout_pnl": pnl_C,
            "n_trades": n_trades_C,
            "delta": delta_C,
        },
        "best_method": best_method,
        "best_delta": best_delta,
        "v2W_TTA_zip": None,
        "expected_platform": None,
    }

    out_path = os.path.join(OUTDIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    _log(f"\nSaved results to {out_path}")
    _log(f"\n=== SUMMARY ===")
    _log(f"v2W baseline:  {pnl_base:.6f} ({n_trades_base} trades)")
    _log(f"Method A:      {pnl_A:.6f} delta={delta_A:+.6f}")
    _log(f"Method C:      {pnl_C:.6f} delta={delta_C:+.6f}")
    _log(f"Best method:   {best_method} (delta={best_delta:+.6f})")

    _update_progress("done", {"v2W_pnl": pnl_base, "method_A_delta": delta_A, "method_C_delta": delta_C})

    print(f"\nRESULT: task=[T161 TTA K=5] metrics={{v2W_baseline={pnl_base:.4f}, method_A_delta={delta_A:+.4f}, method_C_delta={delta_C:+.4f}}} notes=[best={best_method} delta={best_delta:+.4f}]")
    return results


if __name__ == "__main__":
    main()
