"""T166: Ensemble eval — CB L2 5-seed + T87 + T97 + LGB on holdout 96-119.

Eval configs:
  A: v2 baseline (T87 + LGB, w=1.0:1.5)
  B: T87 + LGB + CB (3-way w=1.0:1.5:0.5)
  C: T87 + LGB + CB (3-way w=1.0:1.5:1.0 — LGB:CB equal)
  D: T87 + T97 + LGB + CB (4-way w=1.5:1.5:1.0:0.7)
  E: T87 + T97 + LGB + CB (4-way w=1.5:1.5:1.0:0.7 + T163 agreement filter syms 0,1,2)
  F: T163 replica (T87+T97+LGB w=1.5:1.5:1.0 + agreement filter) for cross-check

Reference:
  v2_baseline_holdout  = 41.4939 (T127 LOSO-equiv, T156b local TSH-FT)
  T163_combined_holdout = 42.9021 (+1.41 vs v2)
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

T87_DIR = os.path.join(ROOT, "experiments", "T87_spo_dfl")
T97_DIR = os.path.join(ROOT, "experiments", "T97_multihead_nn")
T75_DIR = os.path.join(ROOT, "experiments", "T75_regression_dmid")
CACHE_DIR = os.path.join(ROOT, "experiments", "T68_stage5_features", "cache")

SEEDS = [1, 7, 13, 42, 100]
SYMS = [0, 1, 2, 3, 4]
FEE = 0.0001
H = 60

# v2 / T163 conformal params
THR_UP = 0.00029995433796130955
THR_DN = 0.0002158528296175958
PER_SYM_BETA = {0: 0.10, 1: 0.40, 2: 0.30, 3: 0.00, 4: 0.00}
PER_SYM_SIGMA = {
    0: 0.0002403901, 1: 0.0004712397, 2: 0.0004522216,
    3: 0.0004235249, 4: 0.0004279811
}
PER_SYM_BAND = {k: PER_SYM_BETA[k] * PER_SYM_SIGMA[k] for k in SYMS}

V2_BASELINE = 41.4939
T163_COMBINED = 42.9021
AGREEMENT_FILTER_SYMS = frozenset({0, 1, 2})

DROP_NAMES = {
    "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
    "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
    "qrank_W100_cumspread", "kyle_lam_W50", "kyle_lam_W100",
    "roll_eff_spr_ratio_W100", "liq_asym_top5_W5",
}


PROGRESS_PATH = os.path.join(HERE, "worker-progress.json")


def ts():
    return datetime.now().strftime("%H:%M:%S")


def progress(step, metrics=None):
    p = {"status": "running", "step": step, "metrics": metrics or {},
         "timestamp": datetime.now().isoformat()}
    with open(PROGRESS_PATH, "w") as f:
        json.dump(p, f, indent=2)
    print(f"[{ts()}] {step}", flush=True)


def vectorized_pnl(action, mp_t, mp_th):
    side = action.astype(np.float64) - 1.0
    fee = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    return (side * (mp_th - mp_t) - fee) / (mp_t + 1.0)


def apply_conformal(pred, sym_arr, mp_t, mp_th, thr_up=THR_UP, thr_dn=THR_DN):
    action = np.ones(len(pred), dtype=np.int8)
    for sym in SYMS:
        m = sym_arr == sym
        band = PER_SYM_BAND[sym]
        a = np.ones(m.sum(), dtype=np.int8)
        a[pred[m] > thr_up + band] = 2
        a[pred[m] < -(thr_dn + band)] = 0
        action[m] = a
    pnl = vectorized_pnl(action, mp_t, mp_th)
    per_sym = [float(pnl[sym_arr == s].sum()) for s in SYMS]
    return float(pnl.sum()), per_sym


def apply_conformal_with_agreement(pred, sym_arr, mp_t, mp_th,
                                    nn_side, lgb_pred,
                                    thr_up=THR_UP, thr_dn=THR_DN):
    """Conformal gate + T163 selective agreement filter on syms 0,1,2."""
    action = np.ones(len(pred), dtype=np.int8)
    for sym in SYMS:
        m = sym_arr == sym
        band = PER_SYM_BAND[sym]
        a = np.ones(m.sum(), dtype=np.int8)
        a[pred[m] > thr_up + band] = 2
        a[pred[m] < -(thr_dn + band)] = 0
        action[m] = a

    if nn_side is not None and lgb_pred is not None:
        for sym in AGREEMENT_FILTER_SYMS:
            m = sym_arr == sym
            disagree = np.sign(nn_side[m]) != np.sign(lgb_pred[m])
            idxs = np.where(m)[0][disagree]
            action[idxs] = 1

    pnl = vectorized_pnl(action, mp_t, mp_th)
    per_sym = [float(pnl[sym_arr == s].sum()) for s in SYMS]
    return float(pnl.sum()), per_sym


def load_nn_preds(dirpath, pattern):
    dfs = []
    for s in SEEDS:
        fpath = pattern.format(seed=s)
        if not os.path.exists(fpath):
            print(f"  MISSING: {fpath}", flush=True)
            return None, None
        dfs.append(pd.read_parquet(fpath))
    base = dfs[0][["sym", "date", "session", "t",
                    "true_dmid_norm", "midprice_t", "midprice_th"]].copy()
    preds = np.stack([df["pred_dmid_norm"].values for df in dfs], axis=1)
    return base, preds.mean(axis=1).astype(np.float64)


def main():
    progress("T166 eval starting")
    t_start = time.time()

    # ── 1. Load NN preds ──────────────────────────────────────────────────────
    progress("Step 1: Loading NN predictions")
    base87, pred87 = load_nn_preds(
        T87_DIR, os.path.join(T87_DIR, "pred_T87_seed{seed}_main.parquet"))
    if base87 is None:
        raise RuntimeError("T87 preds missing!")
    print(f"  T87: {len(pred87)} rows, date {base87['date'].min()}-{base87['date'].max()}", flush=True)

    _, pred97 = load_nn_preds(
        T97_DIR, os.path.join(T97_DIR, "pred_T97_seed{seed}_main.parquet"))
    if pred97 is None:
        raise RuntimeError("T97 preds missing!")
    print(f"  T97: {len(pred97)} rows", flush=True)

    # CB preds
    _, pred_cb = load_nn_preds(
        HERE, os.path.join(HERE, "pred_CB_inner_seed{seed}.parquet"))
    if pred_cb is None:
        raise RuntimeError("CB preds missing!")
    print(f"  CB: {len(pred_cb)} rows", flush=True)

    base_df = base87
    sym_arr = base_df["sym"].values.astype(np.int32)
    mp_t = base_df["midprice_t"].values.astype(np.float64)
    mp_th = base_df["midprice_th"].values.astype(np.float64)

    # ── 2. Load T75 LGB preds (from T75_regression_dmid parquets) ─────────────
    # These give exactly v2_baseline = 41.4939 when combined with T87 + conformal.
    progress("Step 2: Load T75 LGB parquets (5-seed avg)")
    _, pred_lgb = load_nn_preds(
        T75_DIR, os.path.join(T75_DIR, "pred_T75_seed{seed}.parquet"))
    if pred_lgb is None:
        raise RuntimeError("T75 LGB preds missing!")
    print(f"  T75 LGB: {len(pred_lgb)} rows, mean={pred_lgb.mean():+.6f} std={pred_lgb.std():.6f}",
          flush=True)

    # ── 3. Evaluate ensemble configs ─────────────────────────────────────────
    progress("Step 3: Evaluate ensemble configs")
    results = {}

    def ev(pred_combined, label, nn_side=None, lgb_pred=None, use_agree=False):
        if use_agree:
            total, per_sym = apply_conformal_with_agreement(
                pred_combined, sym_arr, mp_t, mp_th, nn_side, lgb_pred)
        else:
            total, per_sym = apply_conformal(pred_combined, sym_arr, mp_t, mp_th)
        return {"label": label, "pnl": round(total, 4), "per_sym": [round(x, 4) for x in per_sym]}

    # A: v2 baseline
    pred_v2 = (1.0 * pred87 + 1.5 * pred_lgb) / 2.5
    results["A_v2"] = ev(pred_v2, "T87+LGB (v2 weights)")
    print(f"  A v2 baseline: {results['A_v2']['pnl']:.4f}", flush=True)

    # B: T87 + LGB + CB (3-way w=1.0:1.5:0.5)
    w_b = (1.0, 1.5, 0.5); s_b = sum(w_b)
    pred_B = (w_b[0]*pred87 + w_b[1]*pred_lgb + w_b[2]*pred_cb) / s_b
    results["B_3way_w105_05"] = ev(pred_B, "T87+LGB+CB (1.0:1.5:0.5)")
    print(f"  B 3-way 1.0:1.5:0.5: {results['B_3way_w105_05']['pnl']:.4f}", flush=True)

    # C: T87 + LGB + CB (3-way w=1.0:1.5:1.0)
    w_c = (1.0, 1.5, 1.0); s_c = sum(w_c)
    pred_C = (w_c[0]*pred87 + w_c[1]*pred_lgb + w_c[2]*pred_cb) / s_c
    results["C_3way_w105_10"] = ev(pred_C, "T87+LGB+CB (1.0:1.5:1.0)")
    print(f"  C 3-way 1.0:1.5:1.0: {results['C_3way_w105_10']['pnl']:.4f}", flush=True)

    # D: T87 + T97 + LGB + CB (4-way w=1.5:1.5:1.0:0.7)
    w_d = (1.5, 1.5, 1.0, 0.7); s_d = sum(w_d)
    pred_D = (w_d[0]*pred87 + w_d[1]*pred97 + w_d[2]*pred_lgb + w_d[3]*pred_cb) / s_d
    results["D_4way_1515107"] = ev(pred_D, "T87+T97+LGB+CB (1.5:1.5:1.0:0.7)")
    print(f"  D 4-way 1.5:1.5:1.0:0.7: {results['D_4way_1515107']['pnl']:.4f}", flush=True)

    # D2: T87 + T97 + LGB + CB (4-way equal-NN w=1.5:1.5:1.0:1.0)
    w_d2 = (1.5, 1.5, 1.0, 1.0); s_d2 = sum(w_d2)
    pred_D2 = (w_d2[0]*pred87 + w_d2[1]*pred97 + w_d2[2]*pred_lgb + w_d2[3]*pred_cb) / s_d2
    results["D2_4way_15151010"] = ev(pred_D2, "T87+T97+LGB+CB (1.5:1.5:1.0:1.0)")
    print(f"  D2 4-way 1.5:1.5:1.0:1.0: {results['D2_4way_15151010']['pnl']:.4f}", flush=True)

    # E: T87 + T97 + LGB + CB (4-way w=1.5:1.5:1.0:0.7 + agreement filter)
    # NN-side = T87+T97 avg weighted, agreement filter vs LGB
    nn_side_E = (1.5*pred87 + 1.5*pred97) / 3.0
    pred_E = pred_D  # same combined pred, but with agreement filter
    results["E_4way_agree"] = ev(pred_E, "T87+T97+LGB+CB (agree filter w=1.5:1.5:1.0:0.7)",
                                  nn_side=nn_side_E, lgb_pred=pred_lgb, use_agree=True)
    print(f"  E 4-way + agree: {results['E_4way_agree']['pnl']:.4f}", flush=True)

    # E2: same with equal CB weight
    pred_E2 = pred_D2
    results["E2_4way_agree_equal"] = ev(pred_E2, "T87+T97+LGB+CB (agree filter w=1.5:1.5:1.0:1.0)",
                                         nn_side=nn_side_E, lgb_pred=pred_lgb, use_agree=True)
    print(f"  E2 4-way + agree equal: {results['E2_4way_agree_equal']['pnl']:.4f}", flush=True)

    # F: T163 replica (T87+T97+LGB w=1.5:1.5:1.0 + agreement filter) — cross-check
    nn_side_F = (1.5*pred87 + 1.5*pred97) / 3.0
    pred_F = (1.5*pred87 + 1.5*pred97 + 1.0*pred_lgb) / 4.0
    results["F_T163_replica"] = ev(pred_F, "T163 replica (T87+T97+LGB agree filter)",
                                    nn_side=nn_side_F, lgb_pred=pred_lgb, use_agree=True)
    print(f"  F T163 replica: {results['F_T163_replica']['pnl']:.4f}", flush=True)

    # CB standalone w/ LGB
    pred_cblgb = (1.0*pred_cb + 1.5*pred_lgb) / 2.5
    results["G_CB_LGB"] = ev(pred_cblgb, "CB+LGB (v2 weights)")
    print(f"  G CB+LGB: {results['G_CB_LGB']['pnl']:.4f}", flush=True)

    # ── 4. Find best, compute delta ───────────────────────────────────────────
    all_pnls = {k: v["pnl"] for k, v in results.items()}
    best_key = max(all_pnls, key=all_pnls.get)
    best_pnl = all_pnls[best_key]
    v2_local = results["A_v2"]["pnl"]
    t163_replica_pnl = results["F_T163_replica"]["pnl"]
    delta_vs_v2 = best_pnl - v2_local
    delta_vs_t163 = best_pnl - T163_COMBINED
    delta_vs_t163_local = best_pnl - t163_replica_pnl

    print(f"\n=== RESULTS SUMMARY ===", flush=True)
    print(f"  v2 local: {v2_local:.4f} (ref: {V2_BASELINE})", flush=True)
    print(f"  T163 replica: {t163_replica_pnl:.4f} (ref: {T163_COMBINED})", flush=True)
    print(f"  Best: {best_key} → {best_pnl:.4f}", flush=True)
    print(f"  Delta vs v2 local: {delta_vs_v2:+.4f}", flush=True)
    print(f"  Delta vs T163 ref: {delta_vs_t163:+.4f}", flush=True)
    print(f"  Delta vs T163 local: {delta_vs_t163_local:+.4f}", flush=True)

    output = {
        "task": "T166 CatBoost L2 5-seed inner-train ensemble eval",
        "v2_baseline_holdout": V2_BASELINE,
        "T163_combined_holdout": T163_COMBINED,
        "v2_local": round(v2_local, 4),
        "T163_replica_local": round(t163_replica_pnl, 4),
        "configs": results,
        "best_config": best_key,
        "best_holdout": round(best_pnl, 4),
        "delta_vs_v2": round(delta_vs_v2, 4),
        "delta_vs_T163_ref": round(delta_vs_t163, 4),
        "delta_vs_T163_local": round(delta_vs_t163_local, 4),
        "runtime_sec": round(time.time() - t_start, 1),
        "timestamp": datetime.now().isoformat(),
    }
    out_path = os.path.join(HERE, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}", flush=True)

    progress("DONE", metrics={"best_pnl": best_pnl, "delta_vs_T163": round(delta_vs_t163, 4)})
    print(f"\nRESULT: task=[T166 CB L2 eval] "
          f"metrics={{best_pnl={best_pnl:.4f}, v2_local={v2_local:.4f}, "
          f"delta_vs_T163={delta_vs_t163:+.4f}, t163_replica={t163_replica_pnl:.4f}}} "
          f"notes=[best_config={best_key}]", flush=True)
    return output


if __name__ == "__main__":
    main()
