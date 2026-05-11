"""T192 bounded: per-model weights with box constraint.

Each weight w_i ∈ [uniform/10, uniform*3]
- Forces every model to contribute (no zero weights)
- Caps max contribution (no single model dominates)

Two interpretations of "uniform":
  H1) all 300 as one ensemble: uniform = 1/300 → bounds [1/3000, 1/100]
  H2) per-group: NN uniform = 1/150, LGB uniform = 1/150 → bounds per group [1/1500, 1/50]
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
from scipy.optimize import lsq_linear
from sklearn.linear_model import LinearRegression

WORKDIR = "/root/projects/liangwenbei_workdir/experiments/T190_150plus150_pkg/T192_NNLS"
T188_PKG = "/root/projects/liangwenbei_workdir/experiments/T188v3_optimized_pkg"

FEE = 0.0001


def ev_gate_pnl(pred, sym_arr, mp_t_norm, mp_th_norm, thr_up, thr_dn, conformal_cfg=None):
    """Same as eval_holdout — pred and target both in normalized delta_mid units.

    Returns sum PnL and n_actions.
    """
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
    # mp_th and mp_t already in normalized scale; reconstruct PnL
    fee_pnl = FEE * np.abs(side)
    pnl = side * (mp_th_norm - mp_t_norm) - fee_pnl
    return float(pnl.sum()), int((action != 1).sum())


def load_data():
    v = np.load(os.path.join(WORKDIR, "preds_val.npz"))
    t = np.load(os.path.join(WORKDIR, "preds_test.npz"))
    return v, t


def get_pnl_inputs(d, mp_t_orig, mp_th_orig):
    """Reconstruct mp_t and mp_t60 in original scale from y_target.

    The y_target stored is normalized: y = (mp_t60 - mp_t) / mp_t.
    For PnL we need: side * (mp_th - mp_t) / (mp_t + 1) - fee_pnl.
    But we already have y = delta_mid_normalized (matches model output).
    For thresh comparison and PnL formula, just use y as the ground-truth normalized return.
    """
    pass


def main():
    print("Loading cached preds...")
    val = np.load(os.path.join(WORKDIR, "preds_val.npz"))
    test = np.load(os.path.join(WORKDIR, "preds_test.npz"))
    print(f"  val: {val['arr_nn'].shape}, test: {test['arr_nn'].shape}")

    # Build features (300-d) and target
    X_val = np.concatenate([val["arr_nn"], val["arr_lgb"]], axis=1).astype(np.float64)
    y_val = val["y_target"].astype(np.float64)
    X_test = np.concatenate([test["arr_nn"], test["arr_lgb"]], axis=1).astype(np.float64)
    y_test = test["y_target"].astype(np.float64)
    sym_test = test["sym"]

    # Sanity: simple-mean PnL
    print("\n--- Sanity check: simple mean baseline ---")
    # Recreate simple mean weights: NN gets 1/150 * w_nn / (w_nn+w_lgb), same for LGB
    w_nn_global, w_lgb_global = 1.0, 1.5
    norm = w_nn_global + w_lgb_global
    pred_test_sm = (w_nn_global * test["arr_nn"].mean(axis=1) + w_lgb_global * test["arr_lgb"].mean(axis=1)) / norm

    # PnL eval — load thresholds
    with open(os.path.join(T188_PKG, "thresholds.json")) as f:
        tcfg = json.load(f)
    h60 = next(h for h in tcfg["horizons"] if h["h"] == 60)
    thr_up, thr_dn = h60["thr_up"], h60["thr_dn"]
    conformal = tcfg.get("conformal_wrapper", {"enabled": False})
    print(f"  thresh: up={thr_up} dn={thr_dn} conformal={conformal.get('enabled')}")

    # Need raw mp_t and mp_th to compute proper PnL with fee — but cached y_target is already normalized
    # The original eval used: pnl = (side*diff - fee_pnl)/denom  where denom = mp_t + 1
    # But here we have only y_target = (mp_t60 - mp_t)/mp_t
    # Since denom ≈ 1 (mp_t is small), let's use the simplified PnL: side*y_target - fee*|side|
    # This matches the units the threshold operates on

    def pnl(pred):
        return ev_gate_pnl(pred, sym_test, np.zeros_like(y_test), y_test, thr_up, thr_dn, conformal)

    p_sm, n_sm = pnl(pred_test_sm)
    print(f"  simple mean: pnl={p_sm:+.4f}  n_act={n_sm:,}")

    # Also reproduce variant C (NNLS unconstrained) for sanity
    print("\n--- Sanity: NNLS unconstrained (should match results.json C) ---")
    from scipy.optimize import nnls
    w_nnls, _ = nnls(X_val, y_val)
    p_c, n_c = pnl(X_test @ w_nnls)
    print(f"  C: pnl={p_c:+.4f}  n_act={n_c:,}  nz_NN={np.sum(w_nnls[:150]>1e-9)}/150  nz_LGB={np.sum(w_nnls[150:]>1e-9)}/150")

    # =====================================================
    # H1: All 300 as one ensemble, bounds [1/3000, 1/100]
    # =====================================================
    print("\n--- H1) all-300 bounded: w in [1/3000, 1/100] ---")
    N = 300
    w_min = 1.0 / (10 * N)  # 1/3000 ≈ 3.33e-4
    w_max = 3.0 / N         # 3/300 = 0.01
    print(f"  w_min={w_min:.6f}  w_max={w_max:.4f}  ratio={w_max/w_min:.0f}")

    t0 = time.time()
    res = lsq_linear(X_val, y_val,
                     bounds=(np.full(N, w_min), np.full(N, w_max)),
                     method='trf', tol=1e-8, max_iter=200, verbose=0)
    elapsed = time.time() - t0
    w_h1 = res.x
    print(f"  fit done in {elapsed:.1f}s, status={res.status}")
    print(f"  sum(w)={w_h1.sum():.4f}  min(w)={w_h1.min():.6f}  max(w)={w_h1.max():.6f}")
    print(f"  NN: {np.sum(np.abs(w_h1[:150]-w_min)<1e-7)} at min  {np.sum(np.abs(w_h1[:150]-w_max)<1e-7)} at max  {np.sum((w_h1[:150]>w_min+1e-7)&(w_h1[:150]<w_max-1e-7))} interior")
    print(f"  LGB: {np.sum(np.abs(w_h1[150:]-w_min)<1e-7)} at min  {np.sum(np.abs(w_h1[150:]-w_max)<1e-7)} at max  {np.sum((w_h1[150:]>w_min+1e-7)&(w_h1[150:]<w_max-1e-7))} interior")

    pred_val_h1 = X_val @ w_h1
    pred_test_h1 = X_test @ w_h1

    # PnL on val + test
    sym_val = val["sym"]
    p_v, n_v = ev_gate_pnl(pred_val_h1, sym_val, np.zeros_like(y_val), y_val, thr_up, thr_dn, conformal)
    p_h1, n_h1 = pnl(pred_test_h1)
    print(f"  val_pnl={p_v:+.4f}  n_act_val={n_v:,}")
    print(f"  test_pnl={p_h1:+.4f}  n_act_test={n_h1:,}")
    print(f"  delta vs simple-mean: {p_h1 - p_sm:+.4f}")
    print(f"  Val→Test ratio: {p_h1 / max(p_v, 1e-9):.3f}  (should be ~1.0 for honest fit)")

    # =====================================================
    # H2: Per-group bounded: NN ∈ [1/1500, 1/50], LGB same
    # =====================================================
    print("\n--- H2) per-group bounded: w_i in [1/1500, 1/50] each group ---")
    Ng = 150
    w_min_g = 1.0 / (10 * Ng)  # 1/1500
    w_max_g = 3.0 / Ng         # 3/150 = 0.02
    print(f"  w_min={w_min_g:.6f}  w_max={w_max_g:.4f}")

    t0 = time.time()
    res2 = lsq_linear(X_val, y_val,
                      bounds=(np.full(N, w_min_g), np.full(N, w_max_g)),
                      method='trf', tol=1e-8, max_iter=200, verbose=0)
    w_h2 = res2.x
    print(f"  fit done in {time.time()-t0:.1f}s")
    print(f"  sum(w)={w_h2.sum():.4f}  min={w_h2.min():.6f}  max={w_h2.max():.6f}")
    print(f"  NN: {np.sum(np.abs(w_h2[:150]-w_min_g)<1e-7)} at min  {np.sum(np.abs(w_h2[:150]-w_max_g)<1e-7)} at max")
    print(f"  LGB: {np.sum(np.abs(w_h2[150:]-w_min_g)<1e-7)} at min  {np.sum(np.abs(w_h2[150:]-w_max_g)<1e-7)} at max")

    pred_val_h2 = X_val @ w_h2
    pred_test_h2 = X_test @ w_h2
    p_v2, n_v2 = ev_gate_pnl(pred_val_h2, sym_val, np.zeros_like(y_val), y_val, thr_up, thr_dn, conformal)
    p_h2, n_h2 = pnl(pred_test_h2)
    print(f"  val_pnl={p_v2:+.4f}  n_act_val={n_v2:,}")
    print(f"  test_pnl={p_h2:+.4f}  n_act_test={n_h2:,}")
    print(f"  delta vs simple-mean: {p_h2 - p_sm:+.4f}")
    print(f"  Val→Test ratio: {p_h2 / max(p_v2, 1e-9):.3f}")

    # Save
    out = {
        "task": "T192 bounded NNLS — w in [uniform/10, uniform*3]",
        "baseline_simple_mean": {"test_pnl": p_sm, "n_act": n_sm},
        "H1_all300_bounded": {
            "w_min": w_min, "w_max": w_max,
            "val_pnl": p_v, "test_pnl": p_h1, "n_act_test": n_h1,
            "delta_vs_simple": p_h1 - p_sm,
            "val_test_ratio": p_h1 / max(p_v, 1e-9),
            "n_at_min_NN": int(np.sum(np.abs(w_h1[:150]-w_min)<1e-7)),
            "n_at_max_NN": int(np.sum(np.abs(w_h1[:150]-w_max)<1e-7)),
            "n_interior_NN": int(np.sum((w_h1[:150]>w_min+1e-7)&(w_h1[:150]<w_max-1e-7))),
            "n_at_min_LGB": int(np.sum(np.abs(w_h1[150:]-w_min)<1e-7)),
            "n_at_max_LGB": int(np.sum(np.abs(w_h1[150:]-w_max)<1e-7)),
            "n_interior_LGB": int(np.sum((w_h1[150:]>w_min+1e-7)&(w_h1[150:]<w_max-1e-7))),
            "sum_w": float(w_h1.sum()),
        },
        "H2_pergroup_bounded": {
            "w_min": w_min_g, "w_max": w_max_g,
            "val_pnl": p_v2, "test_pnl": p_h2, "n_act_test": n_h2,
            "delta_vs_simple": p_h2 - p_sm,
            "val_test_ratio": p_h2 / max(p_v2, 1e-9),
            "sum_w": float(w_h2.sum()),
        },
    }
    np.savez(os.path.join(WORKDIR, "weights_bounded.npz"),
             w_h1=w_h1.astype(np.float32),
             w_h2=w_h2.astype(np.float32))
    with open(os.path.join(WORKDIR, "results_bounded.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {os.path.join(WORKDIR, 'results_bounded.json')}")
    print("\nRESULT:")
    print(f"  simple_mean: {p_sm:+.4f}")
    print(f"  H1 all-300:  {p_h1:+.4f}  (delta {p_h1-p_sm:+.4f})")
    print(f"  H2 per-grp:  {p_h2:+.4f}  (delta {p_h2-p_sm:+.4f})")


if __name__ == "__main__":
    main()
