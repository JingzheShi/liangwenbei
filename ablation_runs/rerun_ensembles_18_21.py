"""Rebuild ensemble rows 18-21 with the corrected structure.

Row 18 (1+1, sym):     NN row13 seed1 + LGB row17 seed1
Row 19 (5+5, sym):     NN row13 seeds 1-5 (same HP) + LGB M7 HP_CONFIGS[0] seeds {1,6,11,16,21}
Row 20 (5+5, asym):    same data as row 19, asym EV via 2D DE on val
Row 21 (50+50, sym/asym): big50_nn (M7+SPO, HP cycling) + big50_lgb_m7 (M7, HP cycling)

For all rows:
  - LGB M7 preds (big50_lgb_m7) have only yp_test.
    Val preds for threshold tuning sourced from big50_lgb (V4) — task spec.
  - NN row 13 preds: yp_val from V4 (best ES state), yp_test from M7 model.
  - NN big50_nn preds: yp_val from M7 (overfit), yp_test from M7.
  - Cross-family weighting w_NN:w_LGB = 1.0:1.5.

Output: ablation_runs/rerun_ensembles/row{N}/{sym|asym}/results.json
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RERUN_NN = HERE / "rerun_13rows"
BIG_NN = HERE / "big50_nn"
BIG_LGB_V4 = HERE / "big50_lgb"
BIG_LGB_M7 = HERE / "big50_lgb_m7"
OUT_ROOT = HERE / "rerun_ensembles"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

FEE = 1e-4
W_NN = 1.0
W_LGB = 1.5
W_SUM = W_NN + W_LGB


def compute_pnl(y_pred, mp_t, mp_th, thr_up, thr_dn, fee=FEE):
    a = np.zeros_like(y_pred)
    a[y_pred > thr_up] = 1.0
    a[y_pred < -thr_dn] = -1.0
    raw = a * (mp_th - mp_t)
    cost = fee * np.abs(a) * ((mp_th + 1) + (mp_t + 1))
    return float(((raw - cost) / (mp_t + 1.0)).sum()), int((a != 0).sum())


def search_thr_sym(y_pred, mp_t, mp_th, fee=FEE):
    mean_abs = max(float(np.mean(np.abs(y_pred))), 1e-8)
    best = (-1e18, None, None)
    for k in np.linspace(0.5, 3.0, 26):
        thr = k * mean_abs
        s, _ = compute_pnl(y_pred, mp_t, mp_th, thr, thr, fee)
        if s > best[0]:
            best = (s, float(thr), float(k))
    return best


def search_thr_asym_de(y_pred, mp_t, mp_th, fee=FEE):
    from scipy.optimize import differential_evolution
    _, sym_thr, _ = search_thr_sym(y_pred, mp_t, mp_th, fee)
    lo = max(0.7 * sym_thr, 1e-9)
    hi = 1.5 * sym_thr

    def neg(x):
        s, _ = compute_pnl(y_pred, mp_t, mp_th, x[0], x[1], fee)
        return -s

    res = differential_evolution(
        neg, bounds=[(lo, hi), (lo, hi)],
        maxiter=80, popsize=25, tol=1e-9, seed=1, polish=True,
        mutation=(0.5, 1.0),
    )
    s, _ = compute_pnl(y_pred, mp_t, mp_th, res.x[0], res.x[1], fee)
    return float(s), float(res.x[0]), float(res.x[1]), float(sym_thr)


def load_nn_seed(row_dir):
    """Return (yp_val, yp_test, mp_t_val, mp_th_val, mp_t_test, mp_th_test)."""
    z = np.load(row_dir / "preds.npz")
    return (z["yp_val"], z["yp_test"],
            z["mp_t_val"], z["mp_th_val"],
            z["mp_t_test"], z["mp_th_test"])


def load_lgb_v4(seed):
    z = np.load(BIG_LGB_V4 / f"seed{seed}" / "preds.npz")
    return (z["yp_val"], z["yp_test"],
            z["mp_t_val"], z["mp_th_val"],
            z["mp_t_test"], z["mp_th_test"])


def load_lgb_m7(seed):
    """M7 has only yp_test. Pair with V4's yp_val for threshold tuning."""
    z_m7 = np.load(BIG_LGB_M7 / f"seed{seed}" / "preds.npz")
    z_v4 = np.load(BIG_LGB_V4 / f"seed{seed}" / "preds.npz")
    return (z_v4["yp_val"],     # val from V4 (non-overfit)
            z_m7["yp_test"],    # test from M7
            z_v4["mp_t_val"], z_v4["mp_th_val"],
            z_m7["mp_t_test"], z_m7["mp_th_test"])


def avg_family(seed_preds):
    yv = np.mean([p[0].astype(np.float64) for p in seed_preds], axis=0)
    yt = np.mean([p[1].astype(np.float64) for p in seed_preds], axis=0)
    return yv, yt


def run_ensemble(nn_preds, lgb_preds, gate, out_dir, label=""):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Verify ref mp arrays match across families
    yv_nn, yt_nn = avg_family(nn_preds)
    yv_lgb, yt_lgb = avg_family(lgb_preds)
    # Reference mp from first nn
    _, _, mp_t_v, mp_th_v, mp_t_t, mp_th_t = nn_preds[0]
    # Assert lgb matches
    _, _, mp_t_v_l, mp_th_v_l, mp_t_t_l, mp_th_t_l = lgb_preds[0]
    assert np.allclose(mp_t_v, mp_t_v_l), "val mp_t mismatch between NN and LGB"
    assert np.allclose(mp_t_t, mp_t_t_l), "test mp_t mismatch between NN and LGB"

    yv_ens = (W_NN * yv_nn + W_LGB * yv_lgb) / W_SUM
    yt_ens = (W_NN * yt_nn + W_LGB * yt_lgb) / W_SUM

    if gate == "sym":
        val_pnl, thr, k = search_thr_sym(yv_ens, mp_t_v, mp_th_v)
        test_pnl, n_act = compute_pnl(yt_ens, mp_t_t, mp_th_t, thr, thr)
        decision = {"gate": "sym", "thr_sym": thr, "k_sym": k,
                    "val_pnl_h60": val_pnl, "test_pnl_h60": test_pnl,
                    "n_act_test": n_act}
    elif gate == "asym":
        val_pnl, up, dn, sym_thr = search_thr_asym_de(yv_ens, mp_t_v, mp_th_v)
        test_pnl, n_act = compute_pnl(yt_ens, mp_t_t, mp_th_t, up, dn)
        decision = {"gate": "asym_de_bounded", "thr_up": up, "thr_dn": dn,
                    "sym_thr_ref": sym_thr, "val_pnl_h60": val_pnl,
                    "test_pnl_h60": test_pnl, "n_act_test": n_act}
    else:
        raise ValueError(f"unknown gate: {gate}")

    val_pnl_nn_only, *_ = search_thr_sym(yv_nn, mp_t_v, mp_th_v)
    val_pnl_lgb_only, *_ = search_thr_sym(yv_lgb, mp_t_v, mp_th_v)
    test_pnl_nn_only, *_ = search_thr_sym(yt_nn, mp_t_t, mp_th_t)
    test_pnl_lgb_only, *_ = search_thr_sym(yt_lgb, mp_t_t, mp_th_t)

    results = {
        "label": label,
        "n_nn": len(nn_preds), "n_lgb": len(lgb_preds),
        "w_nn": W_NN, "w_lgb": W_LGB,
        "decision": decision,
        "diagnostic": {
            "val_pnl_nn_only_avg_sym": val_pnl_nn_only,
            "val_pnl_lgb_only_avg_sym": val_pnl_lgb_only,
            "test_pnl_nn_only_avg_sym": test_pnl_nn_only,
            "test_pnl_lgb_only_avg_sym": test_pnl_lgb_only,
        },
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [{label}] gate={gate}  val={val_pnl:.4f}  test={test_pnl:.4f}", flush=True)
    return results


def main():
    print("=== Rebuilding ensembles rows 18-21 ===", flush=True)
    t0 = time.time()

    # --- Row 18: 1+1, sym ---
    nn18 = [load_nn_seed(RERUN_NN / "row13_s1")]
    lgb18 = [load_lgb_m7(1)]
    row18 = run_ensemble(nn18, lgb18, "sym", OUT_ROOT / "row18_sym",
                          label="(1+1) NN row13 s1 + LGB row17 s1 (sym)")

    # --- Row 19: 5+5, sym ---
    nn19 = [load_nn_seed(RERUN_NN / f"row13_s{s}") for s in [1, 2, 3, 4, 5]]
    lgb_hp0_seeds = [1, 6, 11, 16, 21]
    lgb19 = [load_lgb_m7(s) for s in lgb_hp0_seeds]
    row19 = run_ensemble(nn19, lgb19, "sym", OUT_ROOT / "row19_sym",
                          label="(5+5) NN row13 + LGB M7 HP[0] (sym)")

    # --- Row 20: 5+5, asym ---
    row20 = run_ensemble(nn19, lgb19, "asym", OUT_ROOT / "row20_asym",
                          label="(5+5) NN row13 + LGB M7 HP[0] (asym)")

    # --- Row 21: 50+50, both gates ---
    big_nn = [
        (lambda s: tuple(np.load(BIG_NN / f"seed{s}" / "preds.npz")[k]
                          for k in ("yp_val", "yp_test", "mp_t_val", "mp_th_val", "mp_t_test", "mp_th_test")))(s)
        for s in range(1, 51)
    ]
    big_lgb_m7 = [load_lgb_m7(s) for s in range(1, 51)]
    row21_sym = run_ensemble(big_nn, big_lgb_m7, "sym", OUT_ROOT / "row21_sym",
                              label="(50+50) big50_nn + big50_lgb_m7 (sym)")
    row21_asym = run_ensemble(big_nn, big_lgb_m7, "asym", OUT_ROOT / "row21_asym",
                               label="(50+50) big50_nn + big50_lgb_m7 (asym)")

    summary = {
        "row18_sym": row18["decision"],
        "row19_sym": row19["decision"],
        "row20_asym": row20["decision"],
        "row21_sym": row21_sym["decision"],
        "row21_asym": row21_asym["decision"],
        "t_sec": time.time() - t0,
    }
    with open(OUT_ROOT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== Done in {summary['t_sec']:.1f}s. Summary:")
    for k, v in summary.items():
        if k != "t_sec":
            print(f"  {k}: val={v['val_pnl_h60']:+.4f}  test={v['test_pnl_h60']:+.4f}")


if __name__ == "__main__":
    main()
