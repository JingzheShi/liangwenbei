"""T192 bounded v2: tighter bounds [uniform/10, uniform*2] + collinearity check.

Also uses the EXACT production ev_gate_pnl formula so test_pnl is comparable to:
  T170 simple   = +146.24
  T188v2 simple = +150.43
  T190 simple   = +140.62
"""
from __future__ import annotations
import json, os, time
import numpy as np
from scipy.optimize import lsq_linear

WORKDIR = "/root/projects/liangwenbei_workdir/experiments/T190_150plus150_pkg/T192_NNLS"
T188_PKG = "/root/projects/liangwenbei_workdir/experiments/T188v3_optimized_pkg"
CACHE = "/root/projects/liangwenbei_workdir/experiments/T68_stage5_features/cache"

FEE = 0.0001


def ev_gate_pnl_production(pred, sym_arr, mp_t, mp_th, thr_up, thr_dn, conformal_cfg):
    """EXACT formula from eval_holdout_150plus150.py."""
    eff_up = np.full(len(pred), thr_up, dtype=np.float64)
    eff_dn = np.full(len(pred), thr_dn, dtype=np.float64)
    if conformal_cfg and conformal_cfg.get("enabled"):
        betas = conformal_cfg["per_sym_beta"]
        sigmas = conformal_cfg["per_sym_sigma"]
        def_beta = conformal_cfg.get("default_beta_for_ood", 0.16)
        def_sigma = conformal_cfg.get("default_sigma_for_ood", 0.0004)
        for sid in np.unique(sym_arr):
            mask = (sym_arr == sid)
            b = betas.get(str(int(sid)), def_beta)
            s = sigmas.get(str(int(sid)), def_sigma)
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


def fit_eval(X_val, y_val, X_test, sym_test, mp_t_test, mp_th_test, sym_val, mp_t_val, mp_th_val,
              tag, w_min, w_max, thr_up, thr_dn, conformal):
    N = X_val.shape[1]
    print(f"\n--- {tag}: w in [{w_min:.6f}, {w_max:.5f}] (ratio {w_max/w_min:.0f}) ---")
    t0 = time.time()
    res = lsq_linear(X_val, y_val,
                     bounds=(np.full(N, w_min), np.full(N, w_max)),
                     method='trf', tol=1e-8, max_iter=200, verbose=0)
    w = res.x
    elapsed = time.time() - t0
    n_at_min_NN = int(np.sum(np.abs(w[:150]-w_min)<1e-7))
    n_at_max_NN = int(np.sum(np.abs(w[:150]-w_max)<1e-7))
    n_int_NN = 150 - n_at_min_NN - n_at_max_NN
    n_at_min_LGB = int(np.sum(np.abs(w[150:]-w_min)<1e-7))
    n_at_max_LGB = int(np.sum(np.abs(w[150:]-w_max)<1e-7))
    n_int_LGB = 150 - n_at_min_LGB - n_at_max_LGB
    print(f"  fit {elapsed:.1f}s  sum(w)={w.sum():.4f}")
    print(f"  NN:  min={n_at_min_NN}  max={n_at_max_NN}  interior={n_int_NN}")
    print(f"  LGB: min={n_at_min_LGB}  max={n_at_max_LGB}  interior={n_int_LGB}")

    pred_val = X_val @ w
    pred_test = X_test @ w
    p_v, n_v = ev_gate_pnl_production(pred_val, sym_val, mp_t_val, mp_th_val, thr_up, thr_dn, conformal)
    p_t, n_t = ev_gate_pnl_production(pred_test, sym_test, mp_t_test, mp_th_test, thr_up, thr_dn, conformal)
    ratio = p_t / max(p_v, 1e-9)
    print(f"  val_pnl={p_v:+.4f}  n_act_val={n_v:,}")
    print(f"  test_pnl={p_t:+.4f}  n_act_test={n_t:,}")
    print(f"  Val→Test ratio={ratio:.3f}  (1.0=honest, >1.05=in-sample artifact)")
    return {
        "tag": tag, "w_min": w_min, "w_max": w_max, "sum_w": float(w.sum()),
        "val_pnl": p_v, "test_pnl": p_t, "n_act_test": n_t,
        "val_test_ratio": ratio,
        "n_at_min_NN": n_at_min_NN, "n_at_max_NN": n_at_max_NN, "n_int_NN": n_int_NN,
        "n_at_min_LGB": n_at_min_LGB, "n_at_max_LGB": n_at_max_LGB, "n_int_LGB": n_int_LGB,
    }, w


def main():
    print("Loading cached preds + raw mp data...")
    val = np.load(os.path.join(WORKDIR, "preds_val.npz"))
    test = np.load(os.path.join(WORKDIR, "preds_test.npz"))
    val_raw = np.load(os.path.join(CACHE, "schemeP_val.npz"))
    test_raw = np.load(os.path.join(CACHE, "schemeP_test.npz"))

    X_val = np.concatenate([val["arr_nn"], val["arr_lgb"]], axis=1).astype(np.float64)
    y_val = val["y_target"].astype(np.float64)
    X_test = np.concatenate([test["arr_nn"], test["arr_lgb"]], axis=1).astype(np.float64)
    sym_test = test["sym"]
    sym_val = val["sym"]

    mp_t_val = val_raw["mp_t"]
    mp_th_val = val_raw["mp_t60"]
    mp_t_test = test_raw["mp_t"]
    mp_th_test = test_raw["mp_t60"]
    print(f"  shapes match: val={len(X_val)}={len(mp_t_val)}, test={len(X_test)}={len(mp_t_test)}")

    with open(os.path.join(T188_PKG, "thresholds.json")) as f:
        tcfg = json.load(f)
    h60 = next(h for h in tcfg["horizons"] if h["h"] == 60)
    thr_up, thr_dn = h60["thr_up"], h60["thr_dn"]
    conformal = tcfg.get("conformal_wrapper", {"enabled": False})
    print(f"  thresh: up={thr_up} dn={thr_dn} conformal={conformal.get('enabled')}")

    # ===== Sanity: simple mean (production formula) =====
    print("\n--- Sanity: simple mean baseline (production formula) ---")
    w_nn_g, w_lgb_g = 1.0, 1.5
    pred_test_sm = (w_nn_g * test["arr_nn"].mean(axis=1) + w_lgb_g * test["arr_lgb"].mean(axis=1)) / (w_nn_g + w_lgb_g)
    p_sm, n_sm = ev_gate_pnl_production(pred_test_sm, sym_test, mp_t_test, mp_th_test, thr_up, thr_dn, conformal)
    print(f"  T190 simple mean: pnl={p_sm:+.4f}  n_act={n_sm:,}")
    print(f"  (should match the +140.62 from eval_holdout_150plus150.py)")

    # ===== Collinearity check (subsample for speed) =====
    print("\n--- Collinearity check (10k random rows) ---")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X_val), size=10000, replace=False)
    Xs = X_val[idx]
    # Center each column
    Xs_c = Xs - Xs.mean(axis=0, keepdims=True)
    # Pairwise correlation among NN models, among LGB, cross
    nn_block = Xs_c[:, :150]
    lgb_block = Xs_c[:, 150:]
    # Normalize columns
    def normcols(X):
        s = X.std(axis=0, keepdims=True) + 1e-12
        return X / s
    nnN = normcols(nn_block)
    lgbN = normcols(lgb_block)
    corr_nn = (nnN.T @ nnN) / nnN.shape[0]
    corr_lgb = (lgbN.T @ lgbN) / lgbN.shape[0]
    corr_cross = (nnN.T @ lgbN) / nnN.shape[0]
    # Off-diagonal stats
    nn_offdiag = corr_nn[np.triu_indices(150, k=1)]
    lgb_offdiag = corr_lgb[np.triu_indices(150, k=1)]
    print(f"  NN-NN pairwise corr:  mean={nn_offdiag.mean():.4f}  median={np.median(nn_offdiag):.4f}  min={nn_offdiag.min():.3f}  max={nn_offdiag.max():.3f}")
    print(f"  LGB-LGB pairwise corr: mean={lgb_offdiag.mean():.4f}  median={np.median(lgb_offdiag):.4f}  min={lgb_offdiag.min():.3f}  max={lgb_offdiag.max():.3f}")
    print(f"  NN-LGB cross corr:    mean={corr_cross.mean():.4f}  median={np.median(corr_cross):.4f}  min={corr_cross.min():.3f}  max={corr_cross.max():.3f}")

    # Eigenvalue spread of full 300x300 corr matrix → condition number
    XsN = normcols(Xs_c)
    full_corr = (XsN.T @ XsN) / XsN.shape[0]
    eigvals = np.linalg.eigvalsh(full_corr)
    eigvals_pos = eigvals[eigvals > 1e-12]
    print(f"  Full corr matrix eigvals: min={eigvals.min():.4e}  max={eigvals.max():.2f}  cond={eigvals.max()/max(eigvals_pos.min(), 1e-12):.1e}")
    print(f"  Top-5 eigvals (out of 300): {sorted(eigvals)[-5:][::-1]}")
    print(f"  → A condition number > 1e6 means severe multicollinearity (OLS unstable)")

    # ===== Bounded NNLS variants =====
    results = {"baseline_simple_mean": {"test_pnl": p_sm, "n_act": n_sm}}

    # H3: all-300 with bounds [1/3000, 2/300] (per user request)
    N = 300
    res_h3, w_h3 = fit_eval(X_val, y_val, X_test, sym_test, mp_t_test, mp_th_test, sym_val, mp_t_val, mp_th_val,
                              "H3) all-300 [1/3000, 2/300]", 1.0/(10*N), 2.0/N, thr_up, thr_dn, conformal)
    results["H3_all300_uniform_div10_x2"] = res_h3

    # H4: per-group [1/1500, 2/150]
    Ng = 150
    res_h4, w_h4 = fit_eval(X_val, y_val, X_test, sym_test, mp_t_test, mp_th_test, sym_val, mp_t_val, mp_th_val,
                              "H4) per-group [1/1500, 2/150]", 1.0/(10*Ng), 2.0/Ng, thr_up, thr_dn, conformal)
    results["H4_pergroup_uniform_div10_x2"] = res_h4

    # H5: tighter still [1/3000, 1.5/300]
    res_h5, w_h5 = fit_eval(X_val, y_val, X_test, sym_test, mp_t_test, mp_th_test, sym_val, mp_t_val, mp_th_val,
                              "H5) all-300 [1/3000, 1.5/300]", 1.0/(10*N), 1.5/N, thr_up, thr_dn, conformal)
    results["H5_all300_uniform_div10_x1p5"] = res_h5

    # H6: tightest [1/600, 1.5/300] = ratio 3
    res_h6, w_h6 = fit_eval(X_val, y_val, X_test, sym_test, mp_t_test, mp_th_test, sym_val, mp_t_val, mp_th_val,
                              "H6) all-300 [1/600, 1.5/300]", 1.0/(2*N), 1.5/N, thr_up, thr_dn, conformal)
    results["H6_all300_uniform_div2_x1p5"] = res_h6

    # Save weights
    np.savez(os.path.join(WORKDIR, "weights_bounded_v2.npz"),
             w_h3=w_h3.astype(np.float32),
             w_h4=w_h4.astype(np.float32),
             w_h5=w_h5.astype(np.float32),
             w_h6=w_h6.astype(np.float32))

    with open(os.path.join(WORKDIR, "results_bounded_v2.json"), "w") as f:
        json.dump({
            "task": "T192 bounded v2 — production PnL formula + collinearity check",
            "T190_simple_mean_test_pnl": p_sm,
            "collinearity": {
                "nn_nn_mean_corr": float(nn_offdiag.mean()),
                "lgb_lgb_mean_corr": float(lgb_offdiag.mean()),
                "nn_lgb_cross_mean_corr": float(corr_cross.mean()),
                "condition_number": float(eigvals.max()/max(eigvals_pos.min(), 1e-12)),
            },
            "variants": results,
        }, f, indent=2)
    print(f"\nSaved -> {os.path.join(WORKDIR, 'results_bounded_v2.json')}")


if __name__ == "__main__":
    main()
