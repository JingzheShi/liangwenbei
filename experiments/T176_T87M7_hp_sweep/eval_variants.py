"""T176: Evaluate all trained variants on holdout (date 96-119).

All T176 NNs are in-sample on the holdout (trained on 0-119).
Comparison is relative: rank variants by holdout PnL delta vs v2 baseline.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

V2_PKG = os.path.join(ROOT, "experiments", "R_full_retrain", "pkg_iter019_v2")
T170_DIR = os.path.join(ROOT, "experiments", "T170_T87_M7")
CACHE_DIR = os.path.join(ROOT, "experiments", "T68_stage5_features", "cache")
VARIANTS_DIR = os.path.join(HERE, "variants")

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

    with open(os.path.join(V2_PKG, "thresholds.json")) as f:
        tcfg = json.load(f)
    h60_cfg = next(h for h in tcfg["horizons"] if h["h"] == 60)
    thr_up = h60_cfg["thr_up"]
    thr_dn = h60_cfg["thr_dn"]
    w_nn = h60_cfg.get("w_nn", 1.0)
    w_lgb = h60_cfg.get("w_lgb", 0.5)
    conformal_cfg = tcfg.get("conformal_wrapper", {"enabled": False})
    print(f"Config: thr_up={thr_up:.6f} thr_dn={thr_dn:.6f} w_nn={w_nn} w_lgb={w_lgb}")

    test_d = np.load(os.path.join(CACHE_DIR, "schemeP_test.npz"))
    X_te = test_d["X"]
    mp_t = test_d["mp_t"]
    mp_th = test_d["mp_t60"]
    sym_arr = test_d["sym"]
    print(f"Test: {len(X_te):,} rows, date {test_d['date'].min()}-{test_d['date'].max()}")

    with open(os.path.join(CACHE_DIR, "schemeP_feat_names.txt")) as f:
        all_feat_names = [l.strip() for l in f]
    m0 = lgb.Booster(model_file=os.path.join(V2_PKG, "model_h60_seed1.txt"))
    lgb_keep_idx = np.array([all_feat_names.index(fn) for fn in m0.feature_name()],
                             dtype=np.int64)
    X_te_lgb = X_te[:, lgb_keep_idx]
    del m0

    print("\nLoading LGB models (v2)...")
    lgb_preds = []
    for s in SEEDS:
        m = lgb.Booster(model_file=os.path.join(V2_PKG, f"model_h60_seed{s}.txt"))
        p = m.predict(X_te_lgb)
        lgb_preds.append(p.astype(np.float32))
    pred_lgb = np.mean(lgb_preds, axis=0)
    print(f"  LGB ensemble: mean={pred_lgb.mean():.6f}")

    print("\nLoading v2 old T87 NN models (baseline)...")
    old_nn_preds = []
    for s in SEEDS:
        m = MLPNumpy(os.path.join(V2_PKG, f"nn_h60_seed{s}.npz"))
        p = m.predict(X_te)
        old_nn_preds.append(p)
    pred_nn_old = np.mean(old_nn_preds, axis=0)
    pred_v2_baseline = (w_nn * pred_nn_old + w_lgb * pred_lgb) / (w_nn + w_lgb)
    pnl_v2, nact_v2 = ev_gate_pnl(pred_v2_baseline, sym_arr, mp_t, mp_th,
                                   thr_up, thr_dn, conformal_cfg)
    print(f"  v2 baseline holdout pnl = {pnl_v2:.4f}  n_act={nact_v2:,}")

    # T170 baseline (S1_ep11 = S2_lr3e-5 = S3_lam30)
    print("\nLoading T170 M7 NN models (ep11, lr3e-5, lam30)...")
    t170_nn_preds = []
    for s in SEEDS:
        npz = os.path.join(T170_DIR, f"nn_h60_seed{s}.npz")
        if not os.path.exists(npz):
            print(f"  WARNING: T170 npz not found for seed{s}, checking variants/S1_ep11 ...")
            npz = os.path.join(VARIANTS_DIR, "S1_ep11", f"nn_h60_seed{s}.npz")
        m = MLPNumpy(npz)
        p = m.predict(X_te)
        t170_nn_preds.append(p)
    pred_nn_t170 = np.mean(t170_nn_preds, axis=0)
    pred_t170 = (w_nn * pred_nn_t170 + w_lgb * pred_lgb) / (w_nn + w_lgb)
    pnl_t170, nact_t170 = ev_gate_pnl(pred_t170, sym_arr, mp_t, mp_th,
                                       thr_up, thr_dn, conformal_cfg)
    print(f"  T170 holdout pnl = {pnl_t170:.4f}  delta={pnl_t170-pnl_v2:+.4f}  n_act={nact_t170:,}")

    results = {
        "task": "T176 T87 NN M7 HP variant sweep",
        "config": {"thr_up": thr_up, "thr_dn": thr_dn, "w_nn": w_nn, "w_lgb": w_lgb},
        "v2_baseline_holdout_pnl": round(pnl_v2, 6),
        "T170_baseline_holdout_insample": round(pnl_t170, 6),
        "T170_delta": round(pnl_t170 - pnl_v2, 6),
        "variants": {
            "S1_ep11": {
                "holdout_pnl": round(pnl_t170, 6),
                "delta_vs_v2": round(pnl_t170 - pnl_v2, 6),
                "delta_vs_T170": 0.0,
                "config": {"epochs": 11, "lr": 3e-5, "lambda_spo": 30},
                "note": "T170 baseline (from T170 dir)",
            }
        },
        "warning": "Holdout in-sample for all T176 NNs (trained 0-119). Ranking only — ship to verify.",
    }

    # Evaluate each variant
    variant_dirs = sorted(os.listdir(VARIANTS_DIR)) if os.path.exists(VARIANTS_DIR) else []
    for variant_name in variant_dirs:
        vdir = os.path.join(VARIANTS_DIR, variant_name)
        if not os.path.isdir(vdir):
            continue
        npz_files = [f for f in os.listdir(vdir) if f.startswith("nn_h60_seed") and f.endswith(".npz")]
        if len(npz_files) < 5:
            print(f"\n  SKIP {variant_name}: only {len(npz_files)}/5 seeds done")
            continue

        print(f"\nLoading variant {variant_name} ...")
        nn_preds = []
        for s in SEEDS:
            npz = os.path.join(vdir, f"nn_h60_seed{s}.npz")
            m = MLPNumpy(npz)
            p = m.predict(X_te)
            nn_preds.append(p)
        pred_nn_var = np.mean(nn_preds, axis=0)
        pred_var = (w_nn * pred_nn_var + w_lgb * pred_lgb) / (w_nn + w_lgb)
        pnl_var, nact_var = ev_gate_pnl(pred_var, sym_arr, mp_t, mp_th,
                                         thr_up, thr_dn, conformal_cfg)
        delta_v2 = pnl_var - pnl_v2
        delta_t170 = pnl_var - pnl_t170

        print(f"  {variant_name}: pnl={pnl_var:.4f}  delta_v2={delta_v2:+.4f}  "
              f"delta_T170={delta_t170:+.4f}  n_act={nact_var:,}")

        # Load config from first summary
        cfg_ep, cfg_lr, cfg_lam = 11, 3e-5, 30
        for s in SEEDS:
            sj = os.path.join(vdir, f"summary_seed{s}.json")
            if os.path.exists(sj):
                with open(sj) as f:
                    sd = json.load(f)
                cfg_ep = sd.get("epochs_fixed", 11)
                cfg_lr = sd.get("lr", 3e-5)
                cfg_lam = sd.get("lambda_spo", 30)
                break

        results["variants"][variant_name] = {
            "holdout_pnl": round(pnl_var, 6),
            "delta_vs_v2": round(delta_v2, 6),
            "delta_vs_T170": round(delta_t170, 6),
            "n_act": nact_var,
            "config": {"epochs": cfg_ep, "lr": cfg_lr, "lambda_spo": cfg_lam},
        }

    # Rank by holdout_pnl
    ranked = sorted(results["variants"].items(),
                    key=lambda kv: kv[1]["holdout_pnl"], reverse=True)
    print("\n=== RANKING ===")
    for i, (name, info) in enumerate(ranked):
        print(f"  #{i+1} {name}: pnl={info['holdout_pnl']:.4f}  delta_T170={info.get('delta_vs_T170', 0):+.4f}")

    top3 = [name for name, _ in ranked[:3]]
    results["top_3_by_holdout_pnl"] = top3

    out_path = os.path.join(HERE, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults -> {out_path}")
    return results


if __name__ == "__main__":
    main()
