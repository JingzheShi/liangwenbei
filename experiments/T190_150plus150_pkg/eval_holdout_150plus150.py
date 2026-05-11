"""Holdout eval (test set, dates 96-119) for T190: 150 NN + 150 LGB vs T170 / T188v2.

Both T190 and T188v2 ensembles trained on dates 0-119 → in-sample for holdout.
Comparison is RELATIVE — same data, same metric, only model count/diversity differs.
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
T190_PKG = HERE
CACHE_DIR = os.path.join(ROOT, "experiments", "T68_stage5_features", "cache")

FEE = 0.0001
T170_SEEDS = (1, 7, 13, 42, 100)
T188_SEEDS = list(range(1, 51))
T190_SEEDS = list(range(1, 151))


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


def load_ensemble_preds(pkg_dir, seeds, X_te, all_feat_names, label="pkg"):
    lgb_preds = []
    nn_preds = []
    missing_lgb = []
    missing_nn = []
    lgb_keep = None

    for s in seeds:
        lp = os.path.join(pkg_dir, f"model_h60_seed{s}.txt")
        np_ = os.path.join(pkg_dir, f"nn_h60_seed{s}.npz")
        if os.path.isfile(lp):
            m = lgb.Booster(model_file=lp)
            if lgb_keep is None:
                lgb_keep = np.array(
                    [all_feat_names.index(fn) for fn in m.feature_name()], dtype=np.int64)
            lgb_preds.append(m.predict(X_te[:, lgb_keep]).astype(np.float32))
        else:
            missing_lgb.append(s)
        if os.path.isfile(np_):
            m = MLPNumpy(np_)
            nn_preds.append(m.predict(X_te))
        else:
            missing_nn.append(s)

    if missing_lgb:
        print(f"  WARNING: missing {len(missing_lgb)} LGB: seeds {missing_lgb[:5]}{'...' if len(missing_lgb) > 5 else ''}")
    if missing_nn:
        print(f"  WARNING: missing {len(missing_nn)} NN: seeds {missing_nn[:5]}{'...' if len(missing_nn) > 5 else ''}")

    pred_lgb = np.mean(lgb_preds, axis=0) if lgb_preds else np.zeros(len(X_te), dtype=np.float32)
    pred_nn = np.mean(nn_preds, axis=0) if nn_preds else np.zeros(len(X_te), dtype=np.float32)
    print(f"  {label}: {len(lgb_preds)} LGB, {len(nn_preds)} NN loaded")
    return pred_lgb, pred_nn, len(lgb_preds), len(nn_preds)


def main():
    with open(os.path.join(T190_PKG, "thresholds.json")) as f:
        tcfg = json.load(f)
    h60_cfg = next(h for h in tcfg["horizons"] if h["h"] == 60)
    thr_up = h60_cfg["thr_up"]
    thr_dn = h60_cfg["thr_dn"]
    w_nn = h60_cfg.get("w_nn", 1.0)
    w_lgb = h60_cfg.get("w_lgb", 1.5)
    conformal_cfg = tcfg.get("conformal_wrapper", {"enabled": False})
    print(f"Config: thr_up={thr_up:.6f} thr_dn={thr_dn:.6f} w_nn={w_nn} w_lgb={w_lgb}")
    print(f"Conformal: {conformal_cfg.get('enabled', False)}")

    test_d = np.load(os.path.join(CACHE_DIR, "schemeP_test.npz"))
    X_te = test_d["X"]
    mp_t = test_d["mp_t"]
    mp_th = test_d["mp_t60"]
    sym_arr = test_d["sym"]
    n_te = len(X_te)
    print(f"Test: {n_te:,} rows, date {test_d['date'].min()}-{test_d['date'].max()}")

    with open(os.path.join(CACHE_DIR, "schemeP_feat_names.txt")) as f:
        all_feat_names = [l.strip() for l in f]

    results = {}

    # === T170 (5+5) ===
    print("\n--- T170 (5 NN + 5 LGB) ---")
    t0 = time.time()
    pred_lgb_t170, pred_nn_t170, n_lgb_t170, n_nn_t170 = load_ensemble_preds(
        V2_PKG, T170_SEEDS, X_te, all_feat_names, "T170")
    pred_t170 = (w_nn * pred_nn_t170 + w_lgb * pred_lgb_t170) / (w_nn + w_lgb)
    pnl_t170, nact_t170 = ev_gate_pnl(pred_t170, sym_arr, mp_t, mp_th, thr_up, thr_dn, conformal_cfg)
    print(f"T170 holdout: pnl={pnl_t170:+.4f}  n_act={nact_t170:,} ({nact_t170/n_te*100:.1f}%)  ({time.time()-t0:.1f}s)")
    results["T170_5plus5"] = {"pnl": round(pnl_t170, 4), "n_act": nact_t170, "n_lgb": n_lgb_t170, "n_nn": n_nn_t170}

    # === T188v2 (50+50) ===
    print("\n--- T188v2 (50 NN + 50 LGB) ---")
    t0 = time.time()
    pred_lgb_188, pred_nn_188, n_lgb_188, n_nn_188 = load_ensemble_preds(
        T188_PKG, T188_SEEDS, X_te, all_feat_names, "T188v2")
    pred_188 = (w_nn * pred_nn_188 + w_lgb * pred_lgb_188) / (w_nn + w_lgb)
    pnl_188, nact_188 = ev_gate_pnl(pred_188, sym_arr, mp_t, mp_th, thr_up, thr_dn, conformal_cfg)
    print(f"T188v2 holdout: pnl={pnl_188:+.4f}  n_act={nact_188:,} ({nact_188/n_te*100:.1f}%)  ({time.time()-t0:.1f}s)")
    results["T188v2_50plus50"] = {"pnl": round(pnl_188, 4), "n_act": nact_188, "n_lgb": n_lgb_188, "n_nn": n_nn_188}

    # === T190 (150+150) ===
    print("\n--- T190 (150 NN + 150 LGB) ---")
    t0 = time.time()
    pred_lgb_190, pred_nn_190, n_lgb_190, n_nn_190 = load_ensemble_preds(
        T190_PKG, T190_SEEDS, X_te, all_feat_names, "T190")
    pred_190 = (w_nn * pred_nn_190 + w_lgb * pred_lgb_190) / (w_nn + w_lgb)
    pnl_190, nact_190 = ev_gate_pnl(pred_190, sym_arr, mp_t, mp_th, thr_up, thr_dn, conformal_cfg)
    print(f"T190 holdout: pnl={pnl_190:+.4f}  n_act={nact_190:,} ({nact_190/n_te*100:.1f}%)  ({time.time()-t0:.1f}s)")
    results["T190_150plus150"] = {"pnl": round(pnl_190, 4), "n_act": nact_190, "n_lgb": n_lgb_190, "n_nn": n_nn_190}

    print(f"\n{'='*50}")
    print(f"T170     : {pnl_t170:+.4f}  ({nact_t170:,} acts)")
    print(f"T188v2   : {pnl_188:+.4f}  ({nact_188:,} acts)  delta vs T170: {pnl_188-pnl_t170:+.4f}")
    print(f"T190     : {pnl_190:+.4f}  ({nact_190:,} acts)  delta vs T170: {pnl_190-pnl_t170:+.4f}  delta vs T188v2: {pnl_190-pnl_188:+.4f}")
    print(f"\nNote: all in-sample (trained 0-119, test 96-119); relative comparison only.")

    print("\nPer-sym breakdown (T190 vs T188v2):")
    for sym_id in range(5):
        mask = (sym_arr == sym_id)
        if mask.sum() == 0:
            continue
        p8, _ = ev_gate_pnl(pred_188[mask], sym_arr[mask], mp_t[mask], mp_th[mask], thr_up, thr_dn, None)
        p9, _ = ev_gate_pnl(pred_190[mask], sym_arr[mask], mp_t[mask], mp_th[mask], thr_up, thr_dn, None)
        print(f"  sym{sym_id}: T188v2={p8:+.3f}  T190={p9:+.3f}  delta={p9-p8:+.3f}  (n={mask.sum():,})")

    out = {
        "results": results,
        "deltas": {
            "T188v2_vs_T170": round(pnl_188 - pnl_t170, 4),
            "T190_vs_T170": round(pnl_190 - pnl_t170, 4),
            "T190_vs_T188v2": round(pnl_190 - pnl_188, 4),
        },
        "config": {"thr_up": thr_up, "thr_dn": thr_dn, "w_nn": w_nn, "w_lgb": w_lgb,
                   "conformal": conformal_cfg.get("enabled", False)},
        "n_test_rows": n_te,
        "warning": "All ensembles in-sample (trained 0-119, tested 96-119); relative comparison only",
    }
    out_path = os.path.join(HERE, "results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
