"""Holdout eval (date 96-119) for T188v2 50 NN + 50 LGB ensemble vs T170 5 NN + 5 LGB.

Both ensembles trained on 0-119 -> in-sample for holdout. Comparison is RELATIVE
between the two — same data, same metric, only model count and seed diversity differ.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
T170_DIR = os.path.join(ROOT, "experiments", "T170_T87_M7")
V2_PKG = os.path.join(ROOT, "experiments", "R_full_retrain", "pkg_iter019_v2")
T188_PKG = os.path.join(ROOT, "experiments", "T188v3_optimized_pkg")
CACHE_DIR = os.path.join(ROOT, "experiments", "T68_stage5_features", "cache")

FEE = 0.0001
T170_SEEDS = (1, 7, 13, 42, 100)
T188_SEEDS = list(range(1, 51))


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
    # Load thresholds (T188v2 == iter_018 v2 == T170: same config)
    with open(os.path.join(T188_PKG, "thresholds.json")) as f:
        tcfg = json.load(f)
    h60_cfg = next(h for h in tcfg["horizons"] if h["h"] == 60)
    thr_up = h60_cfg["thr_up"]
    thr_dn = h60_cfg["thr_dn"]
    w_nn = h60_cfg.get("w_nn", 1.0)
    w_lgb = h60_cfg.get("w_lgb", 1.5)
    conformal_cfg = tcfg.get("conformal_wrapper", {"enabled": False})
    print(f"Config: thr_up={thr_up:.6f} thr_dn={thr_dn:.6f} w_nn={w_nn} w_lgb={w_lgb}")
    print(f"Conformal enabled: {conformal_cfg.get('enabled', False)}")

    # Load test (date 96-119)
    test_d = np.load(os.path.join(CACHE_DIR, "schemeP_test.npz"))
    X_te = test_d["X"]
    mp_t = test_d["mp_t"]
    mp_th = test_d["mp_t60"]
    sym_arr = test_d["sym"]
    n_te = len(X_te)
    print(f"Test: {n_te:,} rows, date {test_d['date'].min()}-{test_d['date'].max()}")

    # Build LGB feature index
    with open(os.path.join(CACHE_DIR, "schemeP_feat_names.txt")) as f:
        all_feat_names = [l.strip() for l in f]

    # === T170 ensemble (5 NN + 5 LGB from V2 pkg) ===
    print("\n--- T170 (5 NN T87M7 + 5 LGB v2) ---")
    t0 = time.time()
    m0 = lgb.Booster(model_file=os.path.join(V2_PKG, f"model_h60_seed1.txt"))
    lgb_keep = np.array([all_feat_names.index(fn) for fn in m0.feature_name()], dtype=np.int64)
    X_te_lgb_t170 = X_te[:, lgb_keep]
    del m0
    lgb_preds = []
    for s in T170_SEEDS:
        m = lgb.Booster(model_file=os.path.join(V2_PKG, f"model_h60_seed{s}.txt"))
        lgb_preds.append(m.predict(X_te_lgb_t170).astype(np.float32))
    pred_lgb_t170 = np.mean(lgb_preds, axis=0)
    nn_preds = []
    for s in T170_SEEDS:
        m = MLPNumpy(os.path.join(T170_DIR, f"nn_h60_seed{s}.npz"))
        nn_preds.append(m.predict(X_te))
    pred_nn_t170 = np.mean(nn_preds, axis=0)
    pred_t170 = (w_nn * pred_nn_t170 + w_lgb * pred_lgb_t170) / (w_nn + w_lgb)
    pnl_t170, nact_t170 = ev_gate_pnl(pred_t170, sym_arr, mp_t, mp_th, thr_up, thr_dn, conformal_cfg)
    print(f"T170 holdout: pnl={pnl_t170:+.4f}  n_act={nact_t170:,} ({nact_t170/n_te*100:.1f}%)  ({time.time()-t0:.1f}s)")

    # === T188v2 ensemble (50 NN + 50 LGB from T188v3 pkg) ===
    print("\n--- T188v2 (50 NN proper-pretrain + 50 LGB diverse) ---")
    t0 = time.time()
    m0 = lgb.Booster(model_file=os.path.join(T188_PKG, f"model_h60_seed1.txt"))
    lgb_keep_188 = np.array([all_feat_names.index(fn) for fn in m0.feature_name()], dtype=np.int64)
    X_te_lgb_188 = X_te[:, lgb_keep_188]
    del m0
    lgb_preds = []
    for s in T188_SEEDS:
        m = lgb.Booster(model_file=os.path.join(T188_PKG, f"model_h60_seed{s}.txt"))
        lgb_preds.append(m.predict(X_te_lgb_188).astype(np.float32))
    pred_lgb_188 = np.mean(lgb_preds, axis=0)
    nn_preds = []
    for s in T188_SEEDS:
        m = MLPNumpy(os.path.join(T188_PKG, f"nn_h60_seed{s}.npz"))
        nn_preds.append(m.predict(X_te))
    pred_nn_188 = np.mean(nn_preds, axis=0)
    pred_188 = (w_nn * pred_nn_188 + w_lgb * pred_lgb_188) / (w_nn + w_lgb)
    pnl_188, nact_188 = ev_gate_pnl(pred_188, sym_arr, mp_t, mp_th, thr_up, thr_dn, conformal_cfg)
    print(f"T188v2 holdout: pnl={pnl_188:+.4f}  n_act={nact_188:,} ({nact_188/n_te*100:.1f}%)  ({time.time()-t0:.1f}s)")

    print(f"\n=== DELTA ===")
    print(f"T170     : {pnl_t170:+.4f}  ({nact_t170:,} acts)")
    print(f"T188v2   : {pnl_188:+.4f}  ({nact_188:,} acts)")
    print(f"delta    : {pnl_188 - pnl_t170:+.4f}")
    print(f"\nNote: BOTH in-sample (NN/LGB trained on 0-119); relative comparison only.")

    # Per-sym breakdown
    print("\nPer-sym breakdown:")
    for sym_id in range(5):
        mask = (sym_arr == sym_id)
        if mask.sum() == 0:
            continue
        pT, _ = ev_gate_pnl(pred_t170[mask], sym_arr[mask], mp_t[mask], mp_th[mask], thr_up, thr_dn, None)
        p8, _ = ev_gate_pnl(pred_188[mask], sym_arr[mask], mp_t[mask], mp_th[mask], thr_up, thr_dn, None)
        print(f"  sym{sym_id}: T170={pT:+.3f}  T188v2={p8:+.3f}  delta={p8-pT:+.3f}")

    # NN-only / LGB-only ablation
    print("\nNN-only / LGB-only (no LGB blend, no conformal):")
    for tag, p in [("T170-NN-only", pred_nn_t170), ("T188v2-NN-only", pred_nn_188),
                    ("T170-LGB-only", pred_lgb_t170), ("T188v2-LGB-only", pred_lgb_188)]:
        pnl, nact = ev_gate_pnl(p, sym_arr, mp_t, mp_th, thr_up, thr_dn, None)
        print(f"  {tag:20s} pnl={pnl:+.4f}  acts={nact:,}")

    # Save
    out = {
        "T170_5plus5_holdout_pnl": round(pnl_t170, 4),
        "T188v2_50plus50_holdout_pnl": round(pnl_188, 4),
        "delta": round(pnl_188 - pnl_t170, 4),
        "T170_n_act": nact_t170,
        "T188v2_n_act": nact_188,
        "thr_up": thr_up,
        "thr_dn": thr_dn,
        "w_nn": w_nn,
        "w_lgb": w_lgb,
        "conformal_enabled": conformal_cfg.get("enabled", False),
        "n_test_rows": n_te,
        "warning": "BOTH ensembles in-sample (NN/LGB trained on 0-119); only relative comparison meaningful",
    }
    with open(os.path.join(HERE, "holdout_50plus50_vs_t170.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {os.path.join(HERE, 'holdout_50plus50_vs_t170.json')}")


if __name__ == "__main__":
    main()
