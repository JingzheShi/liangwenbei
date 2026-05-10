"""T177: Wrapper-only tricks evaluation.

Tricks A/B/C/D on top of T170 SOTA (v2 LGB + T170 M7 NN + iter018 conformal).
Tune on dates 0-95 (inner), eval frozen on 96-119 (holdout).
"""
import json, os, sys, itertools
import numpy as np
import lightgbm as lgb

WORKDIR = "/root/projects/liangwenbei_workdir"
CACHE = f"{WORKDIR}/experiments/T68_stage5_features/cache"
V2_PKG = f"{WORKDIR}/experiments/R_full_retrain/pkg_iter019_v2"
T170_DIR = f"{WORKDIR}/experiments/T170_T87_M7"
HERE = f"{WORKDIR}/experiments/T177_wrapper_tricks"
FEE = 0.0001
SEEDS = (1, 7, 13, 42, 100)

# T170 baseline config
THR_UP = 0.00029995433796130955
THR_DN = 0.0002158528296175958
W_NN = 1.0
W_LGB = 1.5
CONF_BETA = {0: 0.10, 1: 0.40, 2: 0.30, 3: 0.00, 4: 0.00}
CONF_SIGMA = {0: 0.0002403901, 1: 0.0004712397, 2: 0.0004522216, 3: 0.0004235249, 4: 0.0004279811}
# Precompute bands
CONF_BAND = {k: CONF_BETA[k] * CONF_SIGMA[k] for k in range(5)}
CONF_BAND_DEFAULT = 0.16 * 0.0003998317  # OOD fallback

def gelu_tanh(x):
    c = 0.7978845608028654
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x**3)))

def layernorm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta

class MLPNumpy:
    def __init__(self, path):
        d = np.load(path, allow_pickle=False)
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
                h = layernorm(h, self.LN_W[i], self.LN_b[i])
            h = gelu_tanh(h)
        out = h @ self.WF.T + self.bF
        return (out.squeeze(-1) / self.target_scale).astype(np.float32)

def compute_pnl(pred, sym_arr, mp_t, mp_th, thr_up, thr_dn,
                temps=None, thr_factors=None):
    """Compute PnL with optional per-sym temp and thr_factors."""
    # Effective thresholds
    eff_up = np.empty(len(pred), dtype=np.float64)
    eff_dn = np.empty(len(pred), dtype=np.float64)
    p = pred.astype(np.float64).copy()

    for sym_id in range(5):
        mask = (sym_arr == sym_id)
        if mask.sum() == 0:
            continue
        # Apply temperature scaling to predictions
        t_factor = temps[sym_id] if temps is not None else 1.0
        p[mask] *= t_factor
        # Apply threshold factor
        tf = thr_factors[sym_id] if thr_factors is not None else 1.0
        band = CONF_BAND.get(sym_id, CONF_BAND_DEFAULT)
        eff_up[mask] = thr_up * tf + band
        eff_dn[mask] = thr_dn * tf + band
    # Handle OOD syms
    ood_mask = ~np.isin(sym_arr, list(range(5)))
    if ood_mask.any():
        eff_up[ood_mask] = thr_up + CONF_BAND_DEFAULT
        eff_dn[ood_mask] = thr_dn + CONF_BAND_DEFAULT

    action = np.ones(len(p), dtype=np.int8)
    action[p > eff_up] = 2
    action[p < -eff_dn] = 0
    side = action.astype(np.float64) - 1.0
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    pnl = (side * diff - fee_pnl) / denom
    return float(pnl.sum()), int((action != 1).sum())

def main():
    print("=== T177: Wrapper-only tricks ===")

    # --- Load data ---
    print("\nLoading data...")
    tr = np.load(f"{CACHE}/schemeP_train.npz")
    va = np.load(f"{CACHE}/schemeP_val.npz")
    te = np.load(f"{CACHE}/schemeP_test.npz")

    # Combine train+val for inner tuning (0-95)
    X_inner = np.concatenate([tr["X"], va["X"]], axis=0)
    sym_inner = np.concatenate([tr["sym"], va["sym"]])
    mp_t_inner = np.concatenate([tr["mp_t"], va["mp_t"]])
    mp_th_inner = np.concatenate([tr["mp_t60"], va["mp_t60"]])
    print(f"  Inner (0-95): {len(X_inner):,} rows")

    X_te = te["X"]
    sym_te = te["sym"]
    mp_t_te = te["mp_t"]
    mp_th_te = te["mp_t60"]
    print(f"  Test (96-119): {len(X_te):,} rows")

    # --- Load feature names for LGB ---
    with open(f"{CACHE}/schemeP_feat_names.txt") as f:
        all_feat_names = [l.strip() for l in f]

    # Get LGB feature indices from first model
    m0 = lgb.Booster(model_file=f"{V2_PKG}/model_h60_seed1.txt")
    lgb_feat_names = m0.feature_name()
    lgb_keep_idx = np.array([all_feat_names.index(fn) for fn in lgb_feat_names], dtype=np.int64)
    del m0

    X_inner_lgb = X_inner[:, lgb_keep_idx]
    X_te_lgb = X_te[:, lgb_keep_idx]

    # --- Load LGB models and get raw per-seed preds ---
    print("\nLoading LGB models (v2, 5 seeds)...")
    lgb_preds_inner = []  # list of (N_inner,) arrays
    lgb_preds_te = []
    for s in SEEDS:
        m = lgb.Booster(model_file=f"{V2_PKG}/model_h60_seed{s}.txt")
        p_inner = m.predict(X_inner_lgb).astype(np.float32)
        p_te = m.predict(X_te_lgb).astype(np.float32)
        lgb_preds_inner.append(p_inner)
        lgb_preds_te.append(p_te)
        print(f"  LGB seed{s}: inner_mean={p_inner.mean():.6f} te_mean={p_te.mean():.6f}")

    lgb_stack_inner = np.stack(lgb_preds_inner, axis=0)  # (5, N_inner)
    lgb_stack_te = np.stack(lgb_preds_te, axis=0)        # (5, N_te)

    # --- Load T170 M7 NN models ---
    print("\nLoading T170 M7 NN models (5 seeds)...")
    nn_preds_inner = []
    nn_preds_te = []
    for s in SEEDS:
        m = MLPNumpy(f"{T170_DIR}/nn_h60_seed{s}.npz")
        p_inner = m.predict(X_inner)
        p_te = m.predict(X_te)
        nn_preds_inner.append(p_inner)
        nn_preds_te.append(p_te)
        print(f"  NN-M7 seed{s}: inner_mean={p_inner.mean():.6f} te_mean={p_te.mean():.6f}")

    nn_stack_inner = np.stack(nn_preds_inner, axis=0)  # (5, N_inner)
    nn_stack_te = np.stack(nn_preds_te, axis=0)        # (5, N_te)

    # --- BASELINE: mean ensemble (T170 current) ---
    pred_lgb_mean_inner = np.mean(lgb_stack_inner, axis=0)
    pred_lgb_mean_te = np.mean(lgb_stack_te, axis=0)
    pred_nn_mean_inner = np.mean(nn_stack_inner, axis=0)
    pred_nn_mean_te = np.mean(nn_stack_te, axis=0)

    pred_base_inner = (W_NN * pred_nn_mean_inner + W_LGB * pred_lgb_mean_inner) / (W_NN + W_LGB)
    pred_base_te = (W_NN * pred_nn_mean_te + W_LGB * pred_lgb_mean_te) / (W_NN + W_LGB)

    pnl_base_inner, nact_base_inner = compute_pnl(pred_base_inner, sym_inner, mp_t_inner, mp_th_inner, THR_UP, THR_DN)
    pnl_base_te, nact_base_te = compute_pnl(pred_base_te, sym_te, mp_t_te, mp_th_te, THR_UP, THR_DN)
    print(f"\n=== T170 BASELINE ===")
    print(f"  Inner (0-95): pnl={pnl_base_inner:+.4f}  n_act={nact_base_inner:,}")
    print(f"  Holdout (96-119): pnl={pnl_base_te:+.4f}  n_act={nact_base_te:,}")

    results = {
        "task": "T177 wrapper-only tricks",
        "T170_baseline": {
            "inner_pnl": round(pnl_base_inner, 6),
            "holdout_pnl": round(pnl_base_te, 6),
            "n_act_holdout": nact_base_te,
        },
        "tricks": {}
    }

    # =========================================================
    # TRICK A: Median instead of mean across seeds
    # =========================================================
    print("\n=== TRICK A: Median across seeds ===")
    pred_lgb_med_inner = np.median(lgb_stack_inner, axis=0)
    pred_lgb_med_te = np.median(lgb_stack_te, axis=0)
    pred_nn_med_inner = np.median(nn_stack_inner, axis=0)
    pred_nn_med_te = np.median(nn_stack_te, axis=0)

    pred_A_inner = (W_NN * pred_nn_med_inner + W_LGB * pred_lgb_med_inner) / (W_NN + W_LGB)
    pred_A_te = (W_NN * pred_nn_med_te + W_LGB * pred_lgb_med_te) / (W_NN + W_LGB)

    pnl_A_inner, nact_A_inner = compute_pnl(pred_A_inner, sym_inner, mp_t_inner, mp_th_inner, THR_UP, THR_DN)
    pnl_A_te, nact_A_te = compute_pnl(pred_A_te, sym_te, mp_t_te, mp_th_te, THR_UP, THR_DN)
    delta_A = pnl_A_te - pnl_base_te
    print(f"  Inner: pnl={pnl_A_inner:+.4f}  n_act={nact_A_inner:,}")
    print(f"  Holdout: pnl={pnl_A_te:+.4f}  delta={delta_A:+.4f}  n_act={nact_A_te:,}")

    results["tricks"]["A_median"] = {
        "inner_pnl": round(pnl_A_inner, 6),
        "holdout_pnl": round(pnl_A_te, 6),
        "delta": round(delta_A, 6),
        "n_act_holdout": nact_A_te,
    }

    # =========================================================
    # TRICK B: Per-sym temperature scaling
    # Tune on 0-95 (inner), eval on 96-119 (holdout)
    # =========================================================
    print("\n=== TRICK B: Per-sym temperature scaling ===")
    TEMP_GRID = [0.8, 0.9, 1.0, 1.1, 1.2]

    # Coordinate descent over per-sym temps
    best_temps = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    best_pnl_B = pnl_base_inner  # start from baseline

    # Use MEAN predictions for Trick B (isolate the temperature trick)
    for round_cd in range(3):  # 3 coordinate descent rounds
        improved = False
        for sym_id in range(5):
            best_t_for_sym = best_temps[sym_id]
            for t in TEMP_GRID:
                if t == best_temps[sym_id]:
                    continue
                trial_temps = dict(best_temps)
                trial_temps[sym_id] = t
                pnl_trial, _ = compute_pnl(pred_base_inner, sym_inner, mp_t_inner, mp_th_inner,
                                            THR_UP, THR_DN, temps=trial_temps)
                if pnl_trial > best_pnl_B:
                    best_pnl_B = pnl_trial
                    best_temps[sym_id] = t
                    best_t_for_sym = t
                    improved = True
        if not improved:
            print(f"  CD round {round_cd+1}: converged")
            break
        print(f"  CD round {round_cd+1}: inner_pnl={best_pnl_B:+.4f} temps={best_temps}")

    print(f"  Best temps: {best_temps}  inner_pnl={best_pnl_B:+.4f}")

    # Frozen eval on holdout
    pnl_B_te, nact_B_te = compute_pnl(pred_base_te, sym_te, mp_t_te, mp_th_te,
                                        THR_UP, THR_DN, temps=best_temps)
    delta_B = pnl_B_te - pnl_base_te
    print(f"  Holdout: pnl={pnl_B_te:+.4f}  delta={delta_B:+.4f}  n_act={nact_B_te:,}")

    results["tricks"]["B_per_sym_temp"] = {
        "best_temps": best_temps,
        "inner_pnl": round(best_pnl_B, 6),
        "holdout_pnl": round(pnl_B_te, 6),
        "delta": round(delta_B, 6),
        "n_act_holdout": nact_B_te,
    }

    # =========================================================
    # TRICK C: Per-sym threshold scaling
    # Tune on 0-95 (inner), eval on 96-119 (holdout)
    # =========================================================
    print("\n=== TRICK C: Per-sym threshold scaling ===")
    THR_FACTOR_GRID = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

    best_thr_factors = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    best_pnl_C = pnl_base_inner

    for round_cd in range(3):
        improved = False
        for sym_id in range(5):
            best_tf_for_sym = best_thr_factors[sym_id]
            for tf in THR_FACTOR_GRID:
                if tf == best_thr_factors[sym_id]:
                    continue
                trial_factors = dict(best_thr_factors)
                trial_factors[sym_id] = tf
                pnl_trial, _ = compute_pnl(pred_base_inner, sym_inner, mp_t_inner, mp_th_inner,
                                            THR_UP, THR_DN, thr_factors=trial_factors)
                if pnl_trial > best_pnl_C:
                    best_pnl_C = pnl_trial
                    best_thr_factors[sym_id] = tf
                    best_tf_for_sym = tf
                    improved = True
        if not improved:
            print(f"  CD round {round_cd+1}: converged")
            break
        print(f"  CD round {round_cd+1}: inner_pnl={best_pnl_C:+.4f} factors={best_thr_factors}")

    print(f"  Best thr_factors: {best_thr_factors}  inner_pnl={best_pnl_C:+.4f}")

    pnl_C_te, nact_C_te = compute_pnl(pred_base_te, sym_te, mp_t_te, mp_th_te,
                                        THR_UP, THR_DN, thr_factors=best_thr_factors)
    delta_C = pnl_C_te - pnl_base_te
    print(f"  Holdout: pnl={pnl_C_te:+.4f}  delta={delta_C:+.4f}  n_act={nact_C_te:,}")

    results["tricks"]["C_per_sym_thresh"] = {
        "best_factors": best_thr_factors,
        "inner_pnl": round(best_pnl_C, 6),
        "holdout_pnl": round(pnl_C_te, 6),
        "delta": round(delta_C, 6),
        "n_act_holdout": nact_C_te,
    }

    # =========================================================
    # TRICK D: Combined best (A + best of B + best of C)
    # Only if any individual trick shows positive delta
    # =========================================================
    print("\n=== TRICK D: Combined best tricks ===")

    # Use median preds (from A) + best temps from B + best thr_factors from C
    pnl_D_te, nact_D_te = compute_pnl(pred_A_te, sym_te, mp_t_te, mp_th_te,
                                        THR_UP, THR_DN,
                                        temps=best_temps,
                                        thr_factors=best_thr_factors)
    delta_D = pnl_D_te - pnl_base_te
    print(f"  Holdout (A+B+C): pnl={pnl_D_te:+.4f}  delta={delta_D:+.4f}  n_act={nact_D_te:,}")

    # Also try: A+C only (no temp)
    pnl_AC_te, nact_AC_te = compute_pnl(pred_A_te, sym_te, mp_t_te, mp_th_te,
                                          THR_UP, THR_DN, thr_factors=best_thr_factors)
    # And A+B only (no thr scale)
    pnl_AB_te, nact_AB_te = compute_pnl(pred_A_te, sym_te, mp_t_te, mp_th_te,
                                          THR_UP, THR_DN, temps=best_temps)
    print(f"  A+B (no thr_factor): pnl={pnl_AB_te:+.4f}  delta={pnl_AB_te-pnl_base_te:+.4f}")
    print(f"  A+C (no temp):       pnl={pnl_AC_te:+.4f}  delta={pnl_AC_te-pnl_base_te:+.4f}")

    # Also tune D's inner PnL properly: apply both B and C on inner with median preds
    pnl_D_inner, _ = compute_pnl(pred_A_inner, sym_inner, mp_t_inner, mp_th_inner,
                                   THR_UP, THR_DN, temps=best_temps, thr_factors=best_thr_factors)

    results["tricks"]["D_combined"] = {
        "combo": "median (A) + best_temps (B) + best_thr_factors (C)",
        "inner_pnl": round(pnl_D_inner, 6),
        "holdout_pnl": round(pnl_D_te, 6),
        "delta": round(delta_D, 6),
        "n_act_holdout": nact_D_te,
        "AB_holdout_pnl": round(pnl_AB_te, 6),
        "AC_holdout_pnl": round(pnl_AC_te, 6),
    }

    # Summary
    best_trick = max(
        [("A_median", delta_A), ("B_per_sym_temp", delta_B),
         ("C_per_sym_thresh", delta_C), ("D_combined", delta_D)],
        key=lambda x: x[1]
    )
    print(f"\n=== SUMMARY ===")
    print(f"  Baseline holdout: {pnl_base_te:+.4f}")
    print(f"  A (median):       {pnl_A_te:+.4f}  delta={delta_A:+.4f}")
    print(f"  B (temp):         {pnl_B_te:+.4f}  delta={delta_B:+.4f}")
    print(f"  C (thr_scale):    {pnl_C_te:+.4f}  delta={delta_C:+.4f}")
    print(f"  D (combined):     {pnl_D_te:+.4f}  delta={delta_D:+.4f}")
    print(f"  Best trick: {best_trick[0]}  delta={best_trick[1]:+.4f}")

    results["T170_baseline_holdout"] = round(pnl_base_te, 6)
    results["best_trick"] = best_trick[0]
    results["warning"] = "Holdout != platform; T170 M7 NN trained on 0-119 (includes test). Ship to verify."

    # Save interim results
    with open(f"{HERE}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults -> {HERE}/results.json")

    return results, {
        "A": (pred_A_te, None, None),
        "B": (pred_base_te, best_temps, None),
        "C": (pred_base_te, None, best_thr_factors),
        "D": (pred_A_te, best_temps, best_thr_factors),
    }, pred_base_te

if __name__ == "__main__":
    main()
