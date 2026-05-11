"""T191 holdout eval — compute predictions of T191 (5 NN + 5 LGB on 375-d) and
T188v2 (50 NN + 50 LGB on 370-d) on date 96-119, then compute residual IC.

Outputs:
  - per-row predictions for both
  - corr(pred_T191, pred_T188v2)
  - β = OLS(T191 ~ T188v2), residual = pred_T191 - β * pred_T188v2
  - IC_residual (Spearman) overall + per-sym
  - Standalone PnL with same threshold config as T170

Verdict logic in main.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import lightgbm as lgb
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

T188_PKG = os.path.join(ROOT, "experiments", "T188v3_optimized_pkg")
CACHE_SP_DIR = os.path.join(ROOT, "experiments", "T68_stage5_features", "cache")
CACHE_TR_DIR = os.path.join(HERE, "cache")
T191_MODELS = HERE

FEE = 0.0001
SEEDS_T191 = (1, 7, 13, 42, 100)
SEEDS_T188 = list(range(1, 51))

TREND_FEAT_NAMES = [
    "mean_logret_W10", "mean_logret_W20", "mean_logret_W50",
    "mean_logret_W100", "mid_pct_change_W100",
]


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
    with open(os.path.join(T188_PKG, "thresholds.json")) as f:
        tcfg = json.load(f)
    h60_cfg = next(h for h in tcfg["horizons"] if h["h"] == 60)
    thr_up = h60_cfg["thr_up"]; thr_dn = h60_cfg["thr_dn"]
    w_nn = h60_cfg.get("w_nn", 1.0); w_lgb = h60_cfg.get("w_lgb", 1.5)
    conformal_cfg = tcfg.get("conformal_wrapper", {"enabled": False})
    print(f"Config thr_up={thr_up} thr_dn={thr_dn} w_nn={w_nn} w_lgb={w_lgb} conformal={conformal_cfg.get('enabled', False)}")

    # --- Load holdout test set ---
    test_d = np.load(os.path.join(CACHE_SP_DIR, "schemeP_test.npz"))
    X_te_sp = test_d["X"].astype(np.float32)
    mp_t = test_d["mp_t"]
    mp_th = test_d["mp_t60"]
    sym_arr = test_d["sym"]
    n_te = len(X_te_sp)
    print(f"Test rows: {n_te:,} date {test_d['date'].min()}-{test_d['date'].max()}")

    tr_d = np.load(os.path.join(CACHE_TR_DIR, "trend_feat_test.npz"))
    T_te = tr_d["T"].astype(np.float32)
    assert len(T_te) == n_te
    # X_375 := [schemeP(370), trend(5)]  ; this is what T191 models expect
    X_te_375 = np.concatenate([X_te_sp, T_te], axis=1)

    with open(os.path.join(CACHE_SP_DIR, "schemeP_feat_names.txt")) as f:
        sp_names = [l.strip() for l in f]
    all_names_375 = sp_names + TREND_FEAT_NAMES

    # --- T188v2 predictions (50+50 ensemble) ---
    print("\n--- T188v2 (50 NN + 50 LGB) ---")
    t0 = time.time()
    m0 = lgb.Booster(model_file=os.path.join(T188_PKG, "model_h60_seed1.txt"))
    lgb_keep_188 = np.array([sp_names.index(fn) for fn in m0.feature_name()], dtype=np.int64)
    X_te_lgb_188 = X_te_sp[:, lgb_keep_188]
    del m0
    lgb_preds_188 = []
    for s in SEEDS_T188:
        m = lgb.Booster(model_file=os.path.join(T188_PKG, f"model_h60_seed{s}.txt"))
        lgb_preds_188.append(m.predict(X_te_lgb_188).astype(np.float32))
    pred_lgb_188 = np.mean(lgb_preds_188, axis=0)
    nn_preds_188 = []
    for s in SEEDS_T188:
        m = MLPNumpy(os.path.join(T188_PKG, f"nn_h60_seed{s}.npz"))
        nn_preds_188.append(m.predict(X_te_sp))
    pred_nn_188 = np.mean(nn_preds_188, axis=0)
    pred_188 = (w_nn * pred_nn_188 + w_lgb * pred_lgb_188) / (w_nn + w_lgb)
    pnl_188, nact_188 = ev_gate_pnl(pred_188, sym_arr, mp_t, mp_th, thr_up, thr_dn, conformal_cfg)
    print(f"T188v2 holdout: pnl={pnl_188:+.4f}  n_act={nact_188:,}  ({time.time()-t0:.1f}s)")

    # --- T191 predictions (5+5 ensemble on 375-d) ---
    print("\n--- T191 (5 NN + 5 LGB on 375-d incl. 5 trend) ---")
    t0 = time.time()
    m0 = lgb.Booster(model_file=os.path.join(T191_MODELS, f"model_h60_seed1_T191.txt"))
    lgb_keep_191 = np.array([all_names_375.index(fn) for fn in m0.feature_name()], dtype=np.int64)
    X_te_lgb_191 = X_te_375[:, lgb_keep_191]
    del m0
    lgb_preds_191 = []
    for s in SEEDS_T191:
        m = lgb.Booster(model_file=os.path.join(T191_MODELS, f"model_h60_seed{s}_T191.txt"))
        lgb_preds_191.append(m.predict(X_te_lgb_191).astype(np.float32))
    pred_lgb_191 = np.mean(lgb_preds_191, axis=0)
    nn_preds_191 = []
    for s in SEEDS_T191:
        m = MLPNumpy(os.path.join(T191_MODELS, f"nn_h60_seed{s}_T191.npz"))
        nn_preds_191.append(m.predict(X_te_375))
    pred_nn_191 = np.mean(nn_preds_191, axis=0)
    pred_191 = (w_nn * pred_nn_191 + w_lgb * pred_lgb_191) / (w_nn + w_lgb)
    pnl_191, nact_191 = ev_gate_pnl(pred_191, sym_arr, mp_t, mp_th, thr_up, thr_dn, conformal_cfg)
    print(f"T191 holdout: pnl={pnl_191:+.4f}  n_act={nact_191:,}  ({time.time()-t0:.1f}s)")

    # --- Residual IC analysis ---
    print("\n--- Residual IC analysis ---")
    # Ground truth dmid = (mp_th - mp_t) / (mp_t + 1)
    true_dmid = ((mp_th.astype(np.float64) - mp_t.astype(np.float64))
                 / (mp_t.astype(np.float64) + 1.0)).astype(np.float64)

    # Raw correlation
    corr_raw = float(np.corrcoef(pred_191, pred_188)[0, 1])
    print(f"corr(T191, T188v2) raw   = {corr_raw:.4f}")

    # OLS pooled: β = cov / var of T188v2
    b_pool = float(np.cov(pred_191, pred_188)[0, 1] / max(np.var(pred_188), 1e-12))
    resid_pool = pred_191 - b_pool * pred_188
    ic_pool, _ = spearmanr(resid_pool, true_dmid)
    print(f"IC_residual pooled (Spearman) = {ic_pool:.4f}  (β_pool={b_pool:.3f})")

    # IC of standalone T191 and T188v2
    ic_t191, _ = spearmanr(pred_191, true_dmid)
    ic_t188, _ = spearmanr(pred_188, true_dmid)
    print(f"IC_T191 standalone (Spearman) = {ic_t191:.4f}")
    print(f"IC_T188v2 standalone (Spearman) = {ic_t188:.4f}")

    # Per-sym residual IC
    per_sym_ic = {}
    per_sym_corr = {}
    per_sym_beta = {}
    for sym_id in np.unique(sym_arr).tolist():
        mask = (sym_arr == sym_id)
        if mask.sum() < 100:
            continue
        p1 = pred_191[mask]; p8 = pred_188[mask]; td = true_dmid[mask]
        if p8.var() < 1e-20 or p1.var() < 1e-20:
            continue
        c = float(np.corrcoef(p1, p8)[0, 1])
        b = float(np.cov(p1, p8)[0, 1] / max(np.var(p8), 1e-12))
        resid = p1 - b * p8
        ic_s, _ = spearmanr(resid, td)
        per_sym_ic[int(sym_id)] = float(ic_s)
        per_sym_corr[int(sym_id)] = c
        per_sym_beta[int(sym_id)] = b
        print(f"  sym{sym_id}: corr={c:.4f}  β={b:.3f}  ic_residual={ic_s:.4f}  n={mask.sum():,}")

    # --- Verdict ---
    abs_ic_pool = abs(ic_pool)
    if abs_ic_pool > 0.02 and corr_raw < 0.95 and pnl_191 > 30.0:
        verdict = "STRONG"
        rec = "RUN large-scale 150+150 retrain with trend features"
    elif abs_ic_pool > 0.005 and corr_raw < 0.97:
        verdict = "WEAK"
        rec = "BLEND with small weight (~0.1); not worth full retrain"
    else:
        verdict = "NONE"
        rec = "KILL — trend signal already captured by existing 359-d features"
    reasoning = (
        f"|IC_residual_pooled|={abs_ic_pool:.4f} vs thresholds (STRONG>0.02 / WEAK>0.005), "
        f"corr(T191,T188v2)={corr_raw:.4f}, T191 standalone PnL={pnl_191:.2f}"
    )
    print(f"\nVERDICT = {verdict}")
    print(f"  reasoning: {reasoning}")
    print(f"  recommendation: {rec}")

    # Save predictions and results
    np.savez(
        os.path.join(HERE, "preds_holdout.npz"),
        pred_t191=pred_191.astype(np.float32),
        pred_t188v2=pred_188.astype(np.float32),
        pred_nn_191=pred_nn_191.astype(np.float32),
        pred_lgb_191=pred_lgb_191.astype(np.float32),
        true_dmid=true_dmid.astype(np.float32),
        sym=sym_arr,
        mp_t=mp_t.astype(np.float32),
        mp_th=mp_th.astype(np.float32),
    )

    results = {
        "task": "T191: Trend features pilot — 5-NN + 5-LGB on 375-d (370 schemeP + 5 trend, after drop 11 -> feat_dim=364)",
        "n_features": 364,
        "n_new_features": 5,
        "trend_features": TREND_FEAT_NAMES,
        "n_test_rows": int(n_te),
        "T191_standalone_holdout_pnl": round(pnl_191, 4),
        "T188v2_holdout_pnl": round(pnl_188, 4),
        "T191_n_act": int(nact_191),
        "T188v2_n_act": int(nact_188),
        "corr_T191_T188v2_pred": round(corr_raw, 4),
        "beta_OLS_pooled": round(b_pool, 4),
        "ic_T191_standalone": round(float(ic_t191), 4),
        "ic_T188v2_standalone": round(float(ic_t188), 4),
        "ic_residual_T191_pooled_spearman": round(float(ic_pool), 4),
        "ic_residual_per_sym": per_sym_ic,
        "corr_per_sym": per_sym_corr,
        "beta_per_sym": per_sym_beta,
        "verdict": verdict,
        "verdict_reasoning": reasoning,
        "recommendation_for_user": rec,
        "thr_up": thr_up, "thr_dn": thr_dn, "w_nn": w_nn, "w_lgb": w_lgb,
        "conformal_enabled": conformal_cfg.get("enabled", False),
        "note": "Both T191 and T188v2 trained on dates 0-119 -> in-sample for holdout; comparison is RELATIVE",
    }
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {os.path.join(HERE, 'results.json')}")

    # Last-line summary
    print(f"\nRESULT: task=T191_trend_pilot metrics={{ic_residual={ic_pool:.4f}, corr={corr_raw:.4f}, "
          f"standalone_pnl={pnl_191:.2f}}} verdict={verdict}")


if __name__ == "__main__":
    main()
