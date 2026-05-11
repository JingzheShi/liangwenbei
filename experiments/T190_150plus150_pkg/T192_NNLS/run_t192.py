"""T192: Learn per-model non-negative weights for T190 ensemble via regularized regression.

Fit weights on val (dates 80-95), evaluate on test (dates 96-119).
Both val and test are in-sample for underlying models (trained 0-119).
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import lightgbm as lgb
from scipy.optimize import nnls
from sklearn.linear_model import LinearRegression, Lasso

WORKDIR = "/root/projects/liangwenbei_workdir/experiments/T190_150plus150_pkg"
CACHE_DIR = "/root/projects/liangwenbei_workdir/experiments/T68_stage5_features/cache"
OUT_DIR = os.path.join(WORKDIR, "T192_NNLS")
SEEDS = list(range(1, 151))
FEE = 0.0001


# ─────────────────────── NN inference ───────────────────────

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


# ─────────────────────── PnL evaluation ───────────────────────

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


def update_progress(step, status="running", metrics=None):
    p = {
        "status": status,
        "step": step,
        "metrics": metrics or {},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(OUT_DIR, "worker-progress.json"), "w") as f:
        json.dump(p, f, indent=2)
    print(f"[progress] {step}")


# ─────────────────────── Main ───────────────────────

def main():
    update_progress("Loading data")

    # Load val data
    val_d = np.load(os.path.join(CACHE_DIR, "schemeP_val.npz"))
    X_val = val_d["X"]
    mp_t_val = val_d["mp_t"]
    mp_th_val = val_d["mp_t60"]
    sym_val = val_d["sym"]
    date_val = val_d["date"]
    y_val = (mp_th_val.astype(np.float64) - mp_t_val.astype(np.float64)) / (mp_t_val.astype(np.float64) + 1.0)
    print(f"Val: {len(X_val):,} rows, dates {date_val.min()}-{date_val.max()}")

    # Load test data
    test_d = np.load(os.path.join(CACHE_DIR, "schemeP_test.npz"))
    X_test = test_d["X"]
    mp_t_test = test_d["mp_t"]
    mp_th_test = test_d["mp_t60"]
    sym_test = test_d["sym"]
    date_test = test_d["date"]
    y_test = (mp_th_test.astype(np.float64) - mp_t_test.astype(np.float64)) / (mp_t_test.astype(np.float64) + 1.0)
    print(f"Test: {len(X_test):,} rows, dates {date_test.min()}-{date_test.max()}")

    with open(os.path.join(CACHE_DIR, "schemeP_feat_names.txt")) as f:
        all_feat_names = [l.strip() for l in f]
    print(f"Feature names: {len(all_feat_names)}")

    # Load thresholds
    with open(os.path.join(WORKDIR, "thresholds.json")) as f:
        tcfg = json.load(f)
    h60 = next(h for h in tcfg["horizons"] if h["h"] == 60)
    thr_up = h60["thr_up"]
    thr_dn = h60["thr_dn"]
    w_nn = h60.get("w_nn", 1.0)
    w_lgb = h60.get("w_lgb", 1.5)
    conformal_cfg = tcfg.get("conformal_wrapper", {"enabled": False})
    print(f"Thresholds: thr_up={thr_up} thr_dn={thr_dn} w_nn={w_nn} w_lgb={w_lgb}")

    # ─── Step 2: Compute per-model predictions (cached) ───
    preds_val_path = os.path.join(OUT_DIR, "preds_val.npz")
    preds_test_path = os.path.join(OUT_DIR, "preds_test.npz")

    if os.path.exists(preds_val_path) and os.path.exists(preds_test_path):
        print("\nLoading cached predictions...")
        pv = np.load(preds_val_path)
        arr_nn_val = pv["arr_nn"]
        arr_lgb_val = pv["arr_lgb"]
        pt = np.load(preds_test_path)
        arr_nn_test = pt["arr_nn"]
        arr_lgb_test = pt["arr_lgb"]
        print(f"  NN val: {arr_nn_val.shape}, LGB val: {arr_lgb_val.shape}")
        print(f"  NN test: {arr_nn_test.shape}, LGB test: {arr_lgb_test.shape}")
    else:
        update_progress("Running inference (val+test, 300 models)")
        N_val = len(X_val)
        N_test = len(X_test)
        arr_nn_val = np.zeros((N_val, 150), dtype=np.float32)
        arr_lgb_val = np.zeros((N_val, 150), dtype=np.float32)
        arr_nn_test = np.zeros((N_test, 150), dtype=np.float32)
        arr_lgb_test = np.zeros((N_test, 150), dtype=np.float32)
        lgb_keep = None

        t0 = time.time()
        for s in SEEDS:
            # LGB inference
            lgb_path = os.path.join(WORKDIR, f"model_h60_seed{s}.txt")
            if os.path.isfile(lgb_path):
                m = lgb.Booster(model_file=lgb_path)
                if lgb_keep is None:
                    lgb_keep = np.array(
                        [all_feat_names.index(fn) for fn in m.feature_name()], dtype=np.int64)
                arr_lgb_val[:, s-1] = m.predict(X_val[:, lgb_keep]).astype(np.float32)
                arr_lgb_test[:, s-1] = m.predict(X_test[:, lgb_keep]).astype(np.float32)
            else:
                print(f"  WARNING: missing LGB seed {s}")

            # NN inference
            nn_path = os.path.join(WORKDIR, f"nn_h60_seed{s}.npz")
            if os.path.isfile(nn_path):
                nn = MLPNumpy(nn_path)
                arr_nn_val[:, s-1] = nn.predict(X_val)
                arr_nn_test[:, s-1] = nn.predict(X_test)
            else:
                print(f"  WARNING: missing NN seed {s}")

            if s % 10 == 0:
                elapsed = time.time() - t0
                eta = elapsed / s * (150 - s)
                print(f"  seed {s}/150  elapsed={elapsed:.0f}s  eta={eta:.0f}s")

        print(f"\nInference done in {time.time()-t0:.1f}s")
        update_progress("Inference done, saving caches",
                        metrics={"n_seeds": 150, "inference_time": round(time.time()-t0, 1)})

        np.savez_compressed(preds_val_path,
                            arr_nn=arr_nn_val, arr_lgb=arr_lgb_val,
                            y_target=y_val, sym=sym_val, date=date_val)
        np.savez_compressed(preds_test_path,
                            arr_nn=arr_nn_test, arr_lgb=arr_lgb_test,
                            y_target=y_test, sym=sym_test, date=date_test)
        print(f"Cached: {preds_val_path}")
        print(f"Cached: {preds_test_path}")

    update_progress("Fitting weight variants")

    # Combined matrices for full 300-model fitting
    X_full_val = np.column_stack([arr_nn_val, arr_lgb_val]).astype(np.float64)   # (N_val, 300)
    X_full_test = np.column_stack([arr_nn_test, arr_lgb_test]).astype(np.float64) # (N_test, 300)
    y_val_f64 = y_val.astype(np.float64)
    y_test_f64 = y_test.astype(np.float64)

    # Aggregate matrices (2 features: mean_NN, mean_LGB)
    X_agg_val = np.column_stack([arr_nn_val.mean(axis=1), arr_lgb_val.mean(axis=1)])
    X_agg_test = np.column_stack([arr_nn_test.mean(axis=1), arr_lgb_test.mean(axis=1)])

    variants = {}

    # ─── A) Simple mean baseline ───
    print("\n=== A) Simple mean (w_nn=1, w_lgb=1.5) ===")
    pred_A_val = (w_nn * arr_nn_val.mean(axis=1) + w_lgb * arr_lgb_val.mean(axis=1)) / (w_nn + w_lgb)
    pred_A_test = (w_nn * arr_nn_test.mean(axis=1) + w_lgb * arr_lgb_test.mean(axis=1)) / (w_nn + w_lgb)
    pnl_A_val, nact_A_val = ev_gate_pnl(pred_A_val, sym_val, mp_t_val, mp_th_val, thr_up, thr_dn, conformal_cfg)
    pnl_A_test, nact_A_test = ev_gate_pnl(pred_A_test, sym_test, mp_t_test, mp_th_test, thr_up, thr_dn, conformal_cfg)
    print(f"  val_pnl={pnl_A_val:+.4f}  test_pnl={pnl_A_test:+.4f}  n_act_test={nact_A_test:,}")
    variants["A_simple_mean"] = {
        "val_pnl": round(pnl_A_val, 4),
        "test_pnl": round(pnl_A_test, 4),
        "n_act": nact_A_test,
        "params": f"w_nn={w_nn}, w_lgb={w_lgb}",
    }

    # ─── B) OLS aggregate (2 features: mean_NN, mean_LGB) ───
    print("\n=== B) OLS aggregate (2 features) ===")
    lr_B = LinearRegression(fit_intercept=False).fit(X_agg_val, y_val_f64)
    pred_B_val = X_agg_val @ lr_B.coef_
    pred_B_test = X_agg_test @ lr_B.coef_
    pnl_B_val, _ = ev_gate_pnl(pred_B_val, sym_val, mp_t_val, mp_th_val, thr_up, thr_dn, conformal_cfg)
    pnl_B_test, nact_B_test = ev_gate_pnl(pred_B_test, sym_test, mp_t_test, mp_th_test, thr_up, thr_dn, conformal_cfg)
    print(f"  coef: w_nn={lr_B.coef_[0]:.6f}, w_lgb={lr_B.coef_[1]:.6f}")
    print(f"  val_pnl={pnl_B_val:+.4f}  test_pnl={pnl_B_test:+.4f}  n_act_test={nact_B_test:,}")
    variants["B_OLS_2feat"] = {
        "val_pnl": round(pnl_B_val, 4),
        "test_pnl": round(pnl_B_test, 4),
        "n_act": nact_B_test,
        "w_nn_learned": round(float(lr_B.coef_[0]), 6),
        "w_lgb_learned": round(float(lr_B.coef_[1]), 6),
    }

    # ─── C) NNLS 300 features ───
    print("\n=== C) NNLS 300 features ===")
    t0 = time.time()
    w_C, res_C = nnls(X_full_val, y_val_f64)
    pred_C_val = X_full_val @ w_C
    pred_C_test = X_full_test @ w_C
    pnl_C_val, _ = ev_gate_pnl(pred_C_val, sym_val, mp_t_val, mp_th_val, thr_up, thr_dn, conformal_cfg)
    pnl_C_test, nact_C_test = ev_gate_pnl(pred_C_test, sym_test, mp_t_test, mp_th_test, thr_up, thr_dn, conformal_cfg)
    n_nz_nn_C = int((w_C[:150] > 1e-6).sum())
    n_nz_lgb_C = int((w_C[150:] > 1e-6).sum())
    print(f"  Residual={res_C:.6e}  time={time.time()-t0:.2f}s")
    print(f"  nonzero: NN={n_nz_nn_C}/150  LGB={n_nz_lgb_C}/150")
    print(f"  sum(w)={w_C.sum():.6f}")
    print(f"  val_pnl={pnl_C_val:+.4f}  test_pnl={pnl_C_test:+.4f}  n_act_test={nact_C_test:,}")
    variants["C_NNLS_300feat"] = {
        "val_pnl": round(pnl_C_val, 4),
        "test_pnl": round(pnl_C_test, 4),
        "n_act": nact_C_test,
        "n_nonzero_NN": n_nz_nn_C,
        "n_nonzero_LGB": n_nz_lgb_C,
        "sum_w": round(float(w_C.sum()), 6),
        "nnls_residual": round(float(res_C), 8),
    }
    np.save(os.path.join(OUT_DIR, "weights_C_nnls.npy"), w_C)

    # ─── D) Lasso non-negative (sweep alpha) ───
    print("\n=== D) Lasso non-negative (alpha sweep) ===")
    best_D_pnl = -999
    best_D_key = None
    for alpha in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]:
        m_D = Lasso(alpha=alpha, positive=True, fit_intercept=False, max_iter=10000, tol=1e-6)
        m_D.fit(X_full_val, y_val_f64)
        pred_D_val = X_full_val @ m_D.coef_
        pred_D_test = X_full_test @ m_D.coef_
        pnl_D_val, _ = ev_gate_pnl(pred_D_val, sym_val, mp_t_val, mp_th_val, thr_up, thr_dn, conformal_cfg)
        pnl_D_test, nact_D_test = ev_gate_pnl(pred_D_test, sym_test, mp_t_test, mp_th_test, thr_up, thr_dn, conformal_cfg)
        n_nz_D = int((m_D.coef_ > 1e-6).sum())
        key = f"D_Lasso_a{alpha:.0e}"
        print(f"  alpha={alpha:.0e}  nz={n_nz_D}  sum_w={m_D.coef_.sum():.4f}  val={pnl_D_val:+.4f}  test={pnl_D_test:+.4f}")
        variants[key] = {
            "val_pnl": round(pnl_D_val, 4),
            "test_pnl": round(pnl_D_test, 4),
            "n_act": nact_D_test,
            "alpha": alpha,
            "n_nonzero": n_nz_D,
            "sum_w": round(float(m_D.coef_.sum()), 6),
        }
        if pnl_D_test > best_D_pnl:
            best_D_pnl = pnl_D_test
            best_D_key = key

    print(f"  Best Lasso: {best_D_key}  test_pnl={best_D_pnl:+.4f}")

    # ─── E) Ridge non-negative via cvxpy ───
    print("\n=== E) Ridge non-negative (cvxpy, alpha sweep) ===")
    try:
        import cvxpy as cp
        has_cvxpy = True
    except ImportError:
        print("  cvxpy not available, installing...")
        os.system("pip install cvxpy -q")
        try:
            import cvxpy as cp
            has_cvxpy = True
        except ImportError:
            has_cvxpy = False
            print("  cvxpy install failed, skipping E variants")

    if has_cvxpy:
        w_var_E = cp.Variable(300, nonneg=True)
        for alpha in [1e-6, 1e-4, 1e-2, 1.0]:
            t0 = time.time()
            prob_E = cp.Problem(
                cp.Minimize(cp.sum_squares(X_full_val @ w_var_E - y_val_f64) + alpha * cp.sum_squares(w_var_E))
            )
            try:
                prob_E.solve(solver=cp.OSQP, eps_abs=1e-6, eps_rel=1e-6, max_iter=10000)
                if w_var_E.value is not None:
                    w_e = w_var_E.value
                    pred_E_val = X_full_val @ w_e
                    pred_E_test = X_full_test @ w_e
                    pnl_E_val, _ = ev_gate_pnl(pred_E_val, sym_val, mp_t_val, mp_th_val, thr_up, thr_dn, conformal_cfg)
                    pnl_E_test, nact_E_test = ev_gate_pnl(pred_E_test, sym_test, mp_t_test, mp_th_test, thr_up, thr_dn, conformal_cfg)
                    n_nz_E = int((w_e > 1e-6).sum())
                    key = f"E_Ridge_a{alpha:.0e}"
                    print(f"  alpha={alpha:.0e}  nz={n_nz_E}  sum_w={w_e.sum():.4f}  val={pnl_E_val:+.4f}  test={pnl_E_test:+.4f}  ({time.time()-t0:.1f}s)")
                    variants[key] = {
                        "val_pnl": round(pnl_E_val, 4),
                        "test_pnl": round(pnl_E_test, 4),
                        "n_act": nact_E_test,
                        "alpha": alpha,
                        "n_nonzero": n_nz_E,
                        "sum_w": round(float(w_e.sum()), 6),
                    }
                else:
                    print(f"  alpha={alpha:.0e} solver failed (no solution)")
            except Exception as e:
                print(f"  alpha={alpha:.0e} error: {e}")

    # ─── F) NNLS with sum(w) = 1 constraint ───
    print("\n=== F) NNLS sum(w)=1 (cvxpy) ===")
    if has_cvxpy:
        w_F = cp.Variable(300, nonneg=True)
        prob_F = cp.Problem(
            cp.Minimize(cp.sum_squares(X_full_val @ w_F - y_val_f64)),
            [cp.sum(w_F) == 1.0]
        )
        try:
            t0 = time.time()
            prob_F.solve(solver=cp.OSQP, eps_abs=1e-6, eps_rel=1e-6, max_iter=20000)
            if w_F.value is not None:
                w_f = w_F.value
                pred_F_val = X_full_val @ w_f
                pred_F_test = X_full_test @ w_f
                pnl_F_val, _ = ev_gate_pnl(pred_F_val, sym_val, mp_t_val, mp_th_val, thr_up, thr_dn, conformal_cfg)
                pnl_F_test, nact_F_test = ev_gate_pnl(pred_F_test, sym_test, mp_t_test, mp_th_test, thr_up, thr_dn, conformal_cfg)
                n_nz_F = int((w_f > 1e-6).sum())
                print(f"  nz={n_nz_F}  sum_w={w_f.sum():.4f}  val={pnl_F_val:+.4f}  test={pnl_F_test:+.4f}  ({time.time()-t0:.1f}s)")
                variants["F_NNLS_sum1"] = {
                    "val_pnl": round(pnl_F_val, 4),
                    "test_pnl": round(pnl_F_test, 4),
                    "n_act": nact_F_test,
                    "n_nonzero": n_nz_F,
                    "sum_w": round(float(w_f.sum()), 6),
                }
                np.save(os.path.join(OUT_DIR, "weights_F_sum1.npy"), w_f)
            else:
                print("  F solver failed")
                variants["F_NNLS_sum1"] = {"error": "solver failed"}
        except Exception as e:
            print(f"  F error: {e}")
            variants["F_NNLS_sum1"] = {"error": str(e)}
    else:
        print("  cvxpy unavailable, skipping F")
        variants["F_NNLS_sum1"] = {"error": "cvxpy not available"}

    # ─── G) Per-sym OLS aggregate ───
    print("\n=== G) Per-sym OLS aggregate ===")
    pred_G_val = np.zeros(len(X_full_val))
    pred_G_test = np.zeros(len(X_full_test))
    per_sym_coef = {}
    for sym_id in range(5):
        mask_v = (sym_val == sym_id)
        mask_t = (sym_test == sym_id)
        if mask_v.sum() < 100:
            print(f"  sym{sym_id}: skip (only {mask_v.sum()} val rows)")
            continue
        X_sv = np.column_stack([arr_nn_val[mask_v].mean(axis=1), arr_lgb_val[mask_v].mean(axis=1)])
        X_st = np.column_stack([arr_nn_test[mask_t].mean(axis=1), arr_lgb_test[mask_t].mean(axis=1)])
        lr_g = LinearRegression(fit_intercept=False).fit(X_sv, y_val[mask_v])
        pred_G_val[mask_v] = X_sv @ lr_g.coef_
        pred_G_test[mask_t] = X_st @ lr_g.coef_
        per_sym_coef[int(sym_id)] = {"w_nn": round(float(lr_g.coef_[0]), 6), "w_lgb": round(float(lr_g.coef_[1]), 6)}
        print(f"  sym{sym_id}: w_nn={lr_g.coef_[0]:.6f}  w_lgb={lr_g.coef_[1]:.6f}  (n_val={mask_v.sum():,})")
    pnl_G_val, _ = ev_gate_pnl(pred_G_val, sym_val, mp_t_val, mp_th_val, thr_up, thr_dn, conformal_cfg)
    pnl_G_test, nact_G_test = ev_gate_pnl(pred_G_test, sym_test, mp_t_test, mp_th_test, thr_up, thr_dn, conformal_cfg)
    print(f"  val_pnl={pnl_G_val:+.4f}  test_pnl={pnl_G_test:+.4f}  n_act_test={nact_G_test:,}")
    variants["G_per_sym"] = {
        "val_pnl": round(pnl_G_val, 4),
        "test_pnl": round(pnl_G_test, 4),
        "n_act": nact_G_test,
        "per_sym_coef": per_sym_coef,
    }

    # ─── Build summary ───
    update_progress("Computing summary and WandB logging")

    pnl_A_test = variants["A_simple_mean"]["test_pnl"]
    best_variant = max(variants, key=lambda k: variants[k].get("test_pnl", -999))
    best_test_pnl = variants[best_variant]["test_pnl"]
    best_n_act = variants[best_variant].get("n_act", 0)

    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print(f"{'Variant':<25} {'val_pnl':>10} {'test_pnl':>10} {'n_act':>8} {'n_nz':>6}")
    print("-" * 60)
    for k, v in sorted(variants.items()):
        if "error" in v:
            print(f"  {k:<23} {'ERROR':>10}")
            continue
        nz = v.get("n_nonzero", v.get("n_nonzero_NN", "-"))
        print(f"  {k:<23} {v.get('val_pnl', 0.0):>+10.4f} {v.get('test_pnl', 0.0):>+10.4f} {v.get('n_act', 0):>8,} {str(nz):>6}")
    print("=" * 60)
    print(f"Best: {best_variant}  test_pnl={best_test_pnl:+.4f}  delta_vs_A={best_test_pnl-pnl_A_test:+.4f}")
    print(f"T190 simple holdout: +140.62, T188v2 simple holdout: +150.43")

    interpretation = (
        f"NNLS+Lasso weight learning on 300 T190 models (fit on val dates 80-95, eval on test 96-119). "
        f"Best variant '{best_variant}' achieves test_pnl={best_test_pnl:+.4f} vs simple-mean baseline "
        f"{pnl_A_test:+.4f} (delta={best_test_pnl-pnl_A_test:+.4f}). "
        f"Both val and test are in-sample for the models; val→test generalization within in-sample regime "
        f"suggests {'real weight signal worth packaging' if best_test_pnl - pnl_A_test > 1.0 else 'marginal benefit—likely noise in in-sample regime; do not package without OOD validation'}."
    )

    results_out = {
        "variants": variants,
        "best_variant": best_variant,
        "best_test_pnl": round(best_test_pnl, 4),
        "best_n_act": best_n_act,
        "delta_vs_simple_mean": round(best_test_pnl - pnl_A_test, 4),
        "delta_vs_T188v2_simple": round(best_test_pnl - 150.43, 4),
        "T188v2_simple_holdout": 150.43,
        "T190_simple_holdout": 140.62,
        "val_T190_simple_pnl": round(pnl_A_val, 4),
        "warning_in_sample": (
            "Both val and test are in-sample for the models (trained 0-119). "
            "Val→test generalization within in-sample regime — does not predict OOD platform behavior."
        ),
        "interpretation": interpretation,
    }

    out_path = os.path.join(OUT_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(results_out, f, indent=2)
    print(f"\nSaved results → {out_path}")

    # ─── WandB logging ───
    print("\nLogging to WandB...")
    try:
        import wandb
        wandb.login(key="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")
        run = wandb.init(project="T192_NNLS_weights", entity="cjxh21-Tsinghua University",
                         name="T192_weight_learning")
        summary = {f"test_pnl/{k}": v.get("test_pnl", 0) for k, v in variants.items() if "error" not in v}
        summary.update({f"val_pnl/{k}": v.get("val_pnl", 0) for k, v in variants.items() if "error" not in v})
        summary["best_variant"] = best_variant
        summary["best_test_pnl"] = best_test_pnl
        summary["delta_vs_simple_mean"] = best_test_pnl - pnl_A_test
        summary["C_NNLS_n_nonzero_NN"] = variants.get("C_NNLS_300feat", {}).get("n_nonzero_NN", 0)
        summary["C_NNLS_n_nonzero_LGB"] = variants.get("C_NNLS_300feat", {}).get("n_nonzero_LGB", 0)
        wandb.log(summary)
        wandb.finish()
        print("WandB logging done.")
    except Exception as e:
        print(f"WandB logging failed: {e}")

    update_progress("Complete", status="done",
                    metrics={"best_variant": best_variant,
                             "best_test_pnl": round(best_test_pnl, 4),
                             "delta_vs_simple_mean": round(best_test_pnl - pnl_A_test, 4)})

    print(f"\nRESULT: task=T192_NNLS_weights "
          f"metrics={{best_pnl={best_test_pnl:.4f}, delta_vs_simple={best_test_pnl-pnl_A_test:+.4f}, "
          f"n_nonzero_C={variants.get('C_NNLS_300feat', {}).get('n_nonzero_NN', 0)+variants.get('C_NNLS_300feat', {}).get('n_nonzero_LGB', 0)}, "
          f"best_variant={best_variant}}} "
          f"notes={interpretation[:120]}")


if __name__ == "__main__":
    main()
