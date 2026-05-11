"""Run T188v2 (50 NN + 50 LGB) baseline predictions locally on holdout.
Saves pred_t188v2.npz containing pred (442080,) for downstream IC analysis.
Can run in parallel with remote training to save wall-clock time.
"""
from __future__ import annotations
import json, os, time
import numpy as np
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
T188_PKG = os.path.join(ROOT, "experiments", "T188v3_optimized_pkg")
CACHE_SP_DIR = os.path.join(ROOT, "experiments", "T68_stage5_features", "cache")

FEE = 0.0001
SEEDS_T188 = list(range(1, 51))


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

    test_d = np.load(os.path.join(CACHE_SP_DIR, "schemeP_test.npz"))
    X_te = test_d["X"].astype(np.float32)
    mp_t = test_d["mp_t"]; mp_th = test_d["mp_t60"]; sym_arr = test_d["sym"]
    n_te = len(X_te)
    print(f"Test rows: {n_te:,}")

    with open(os.path.join(CACHE_SP_DIR, "schemeP_feat_names.txt")) as f:
        all_names = [l.strip() for l in f]

    print("\n--- T188v2 (50 NN + 50 LGB) ---")
    t0 = time.time()
    m0 = lgb.Booster(model_file=os.path.join(T188_PKG, "model_h60_seed1.txt"))
    lgb_keep = np.array([all_names.index(fn) for fn in m0.feature_name()], dtype=np.int64)
    X_te_lgb = X_te[:, lgb_keep]
    del m0
    lgb_preds = []
    for s in SEEDS_T188:
        m = lgb.Booster(model_file=os.path.join(T188_PKG, f"model_h60_seed{s}.txt"))
        lgb_preds.append(m.predict(X_te_lgb).astype(np.float32))
        if s % 10 == 0:
            print(f"  LGB seed {s} done ({time.time()-t0:.1f}s)", flush=True)
    pred_lgb = np.mean(lgb_preds, axis=0)
    print(f"  all 50 LGB done in {time.time()-t0:.1f}s", flush=True)

    t1 = time.time()
    nn_preds = []
    for s in SEEDS_T188:
        m = MLPNumpy(os.path.join(T188_PKG, f"nn_h60_seed{s}.npz"))
        nn_preds.append(m.predict(X_te))
        if s % 10 == 0:
            print(f"  NN seed {s} done ({time.time()-t1:.1f}s)", flush=True)
    pred_nn = np.mean(nn_preds, axis=0)
    print(f"  all 50 NN done in {time.time()-t1:.1f}s", flush=True)

    pred_t188v2 = (w_nn * pred_nn + w_lgb * pred_lgb) / (w_nn + w_lgb)
    pnl, nact = ev_gate_pnl(pred_t188v2, sym_arr, mp_t, mp_th, thr_up, thr_dn, conformal_cfg)
    print(f"T188v2 holdout: pnl={pnl:+.4f}  n_act={nact:,}")

    np.savez(
        os.path.join(HERE, "pred_t188v2.npz"),
        pred=pred_t188v2.astype(np.float32),
        pred_nn=pred_nn.astype(np.float32),
        pred_lgb=pred_lgb.astype(np.float32),
        pnl=np.array([pnl], dtype=np.float64),
        n_act=np.array([nact], dtype=np.int64),
    )
    print(f"Saved -> {os.path.join(HERE, 'pred_t188v2.npz')}")


if __name__ == "__main__":
    main()
