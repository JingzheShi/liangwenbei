"""T126: 4 decision-rule variants on iter_015 v1 stack.

Stack: (1.0*T87_NN_5seed + 1.5*T75_LGB_5seed) / 2.5

Variants:
  V1  Baseline (iter_015 v1):
        thr_up=4.21e-4, thr_dn=1.86e-4 — global asym, eval on test 442k.
  V2  Per-sym thresh from DE on local test:
        10D DE search (5 sym × (thr_up, thr_dn)) on test set.
        ⚠️ Even more in-sample than V1 — informational only.
  V3  Pure OOF thresh from val date 76-79:
        2D DE on val 73,680 rows (no peek at test).
        Apply thresh to test for LOSO/active-rate.
  V4  5-fold time-block CV thresh from val date 76-79:
        Split val by (date×session): 4*2=8 blocks → 5 contiguous folds.
        For each fold k: DE thresh on the OTHER 4 folds, eval on fold k → "OOF est"
        Final thresh = MEDIAN of 5 fold-thresholds (and we also report MEAN).
        Apply to test.

Output: REPORT.md + results.json
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))

FEE = 0.0001
SYMS = (0, 1, 2, 3, 4)

# iter_015 v1 known thresholds
V1_THR_UP = 4.21e-4
V1_THR_DN = 1.86e-4

# Reproducibility
DE_SEEDS = (0, 1, 2, 7, 42)


def vectorized_pnl(action, mp_t, mp_th):
    side = action.astype(np.float64) - 1.0
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee_pnl) / denom


def gate_asym(p, tu, td):
    a = np.full(len(p), 1, dtype=np.int8)
    a[p > tu] = 2
    a[p < -td] = 0
    return a


def gate_per_sym(p, sym, thr_per_sym):
    """thr_per_sym: dict sym -> (tu, td)."""
    a = np.full(len(p), 1, dtype=np.int8)
    for k in SYMS:
        m = (sym == k)
        tu, td = thr_per_sym[k]
        a[m & (p > tu)] = 2
        a[m & (p < -td)] = 0
    return a


def cum_pnl_at_thr(pred, sym, mp_t, mp_th, tu, td):
    a = gate_asym(pred, tu, td)
    pnl = vectorized_pnl(a, mp_t, mp_th)
    total = float(pnl.sum())
    per = []
    actv = []
    for k in SYMS:
        m = (sym == k)
        per.append(float(pnl[m].sum()))
        actv.append(int((a[m] != 1).sum()))
    actv_total = int((a != 1).sum())
    return total, per, actv, actv_total


def cum_pnl_per_sym_thr(pred, sym, mp_t, mp_th, thr_per_sym):
    a = gate_per_sym(pred, sym, thr_per_sym)
    pnl = vectorized_pnl(a, mp_t, mp_th)
    total = float(pnl.sum())
    per = []
    actv = []
    for k in SYMS:
        m = (sym == k)
        per.append(float(pnl[m].sum()))
        actv.append(int((a[m] != 1).sum()))
    actv_total = int((a != 1).sum())
    return total, per, actv, actv_total


def de_2d(pred, mp_t, mp_th, bounds=((0.0, 0.005), (0.0, 0.005)),
          seeds=DE_SEEDS, maxiter=80, popsize=24):
    """Optimize global (thr_up, thr_dn) to maximize cum_pnl on the given arrays."""
    def obj(x):
        a = gate_asym(pred, x[0], x[1])
        return -float(vectorized_pnl(a, mp_t, mp_th).sum())
    runs = []
    for sd in seeds:
        r = differential_evolution(obj, bounds=list(bounds), seed=sd,
                                   maxiter=maxiter, popsize=popsize,
                                   polish=True, tol=1e-7, init="sobol")
        runs.append((float(-r.fun), float(r.x[0]), float(r.x[1])))
    runs.sort(key=lambda x: x[0], reverse=True)
    return runs[0]  # (best_pnl, tu, td)


def de_per_sym(pred, sym, mp_t, mp_th, seeds=DE_SEEDS, maxiter=120, popsize=30):
    """Per-sym thresh: 10D DE — (tu_0, td_0, tu_1, td_1, ..., tu_4, td_4).

    Optimize sum cum_pnl across syms (equivalent to global pnl since they don't
    overlap in samples).
    """
    sym_masks = {k: (sym == k) for k in SYMS}
    sliced = {k: (pred[m], mp_t[m], mp_th[m]) for k, m in sym_masks.items()}

    def obj(x):
        s = 0.0
        for i, k in enumerate(SYMS):
            tu = x[2*i]
            td = x[2*i + 1]
            p, mt, mh = sliced[k]
            a = gate_asym(p, tu, td)
            s += float(vectorized_pnl(a, mt, mh).sum())
        return -s

    bounds = [(0.0, 0.005)] * (2 * len(SYMS))
    runs = []
    for sd in seeds:
        r = differential_evolution(obj, bounds=bounds, seed=sd,
                                   maxiter=maxiter, popsize=popsize,
                                   polish=True, tol=1e-7, init="sobol")
        thr = {SYMS[i]: (float(r.x[2*i]), float(r.x[2*i + 1])) for i in range(len(SYMS))}
        runs.append((float(-r.fun), thr))
    runs.sort(key=lambda x: x[0], reverse=True)
    return runs[0]  # (best_pnl, thr_per_sym dict)


def make_kfold_blocks(date, sess_idx, k=5):
    """Build k contiguous folds from sorted (date, sess_idx) blocks.

    With val=date 76..79, sess in {0,1} → 8 (date, sess) blocks.
    Fold split [0:1, 2:3, 4:5, 6:7, 7:8] is uneven w/ 5 folds. Use np.array_split
    on block indices for balanced folds.
    """
    keys = list(zip(date.tolist(), sess_idx.tolist()))
    block_id = np.zeros(len(date), dtype=np.int64)
    seen = {}
    next_id = 0
    for i, k_ in enumerate(keys):
        if k_ not in seen:
            seen[k_] = next_id
            next_id += 1
        block_id[i] = seen[k_]
    n_blocks = next_id
    block_to_fold = {}
    splits = np.array_split(np.arange(n_blocks), k)
    for fid, blocks in enumerate(splits):
        for b in blocks.tolist():
            block_to_fold[b] = fid
    fold = np.array([block_to_fold[b] for b in block_id], dtype=np.int64)
    return fold, n_blocks


def main():
    t_start = time.time()

    # Load val + test preds
    df_va = pd.read_parquet(os.path.join(HERE, "val_preds_iter015v1.parquet"))
    df_te = pd.read_parquet(os.path.join(HERE, "test_preds_iter015v1.parquet"))
    print(f"val rows={len(df_va)}, test rows={len(df_te)}", flush=True)

    pred_va = df_va["pred_dmid_norm"].to_numpy(np.float64)
    sym_va = df_va["sym"].to_numpy(np.int64)
    date_va = df_va["date"].to_numpy(np.int64)
    sess_va = df_va["sess_idx"].to_numpy(np.int64)
    mp_t_va = df_va["midprice_t"].to_numpy(np.float64)
    mp_th_va = df_va["midprice_th"].to_numpy(np.float64)

    pred_te = df_te["pred_dmid_norm"].to_numpy(np.float64)
    sym_te = df_te["sym"].to_numpy(np.int64)
    mp_t_te = df_te["midprice_t"].to_numpy(np.float64)
    mp_th_te = df_te["midprice_th"].to_numpy(np.float64)

    n_test = len(df_te)
    n_val = len(df_va)

    results = {
        "task": "T126 per-sym vs OOF thresh on iter_015 v1 stack",
        "stack": "5xT87 NN + 5xT75 LGB at 1.0:1.5",
        "n_val_rows": n_val,
        "n_test_rows": n_test,
        "FEE": FEE,
        "DE_seeds": list(DE_SEEDS),
        "variants": {},
    }

    # ====== V1 Baseline ======
    print("\n=== V1: Baseline iter_015 v1 thresholds ===", flush=True)
    s, per, actv, actv_t = cum_pnl_at_thr(pred_te, sym_te, mp_t_te, mp_th_te,
                                          V1_THR_UP, V1_THR_DN)
    print(f"  test cum_pnl={s:+.4f}  per_sym={[round(x,2) for x in per]}", flush=True)
    print(f"  active rate test: {actv_t/n_test:.4f}  per_sym={[round(a/(n_test/5), 4) for a in actv]}",
          flush=True)
    results["variants"]["V1_baseline"] = {
        "thr_up": V1_THR_UP, "thr_dn": V1_THR_DN,
        "test_cum_pnl": s, "test_per_sym": per,
        "test_active_count": actv, "test_active_total": actv_t,
        "test_active_rate": actv_t / n_test,
    }
    # Also record val PnL at V1 thresh
    sv, perv, actvv, actvv_t = cum_pnl_at_thr(pred_va, sym_va, mp_t_va, mp_th_va,
                                              V1_THR_UP, V1_THR_DN)
    print(f"  val cum_pnl={sv:+.4f}  per_sym={[round(x,2) for x in perv]}  active={actvv_t/n_val:.4f}",
          flush=True)
    results["variants"]["V1_baseline"]["val_cum_pnl"] = sv
    results["variants"]["V1_baseline"]["val_per_sym"] = perv
    results["variants"]["V1_baseline"]["val_active_rate"] = actvv_t / n_val

    # ====== V2 Per-sym thresh from DE on test (in-sample) ======
    print("\n=== V2: Per-sym DE thresh on TEST (in-sample, informational) ===", flush=True)
    t0 = time.time()
    s_de, thr_ps = de_per_sym(pred_te, sym_te, mp_t_te, mp_th_te)
    print(f"  DE-best in-sample test cum_pnl={s_de:+.4f}  ({time.time()-t0:.1f}s)", flush=True)
    s_v2, per_v2, actv_v2, actv_v2_t = cum_pnl_per_sym_thr(pred_te, sym_te, mp_t_te, mp_th_te, thr_ps)
    for k in SYMS:
        tu, td = thr_ps[k]
        print(f"    sym={k}: thr_up={tu:.3e} thr_dn={td:.3e}", flush=True)
    print(f"  test cum_pnl={s_v2:+.4f} per_sym={[round(x,2) for x in per_v2]} active={actv_v2_t/n_test:.4f}",
          flush=True)
    # Also sanity: apply same per-sym thr to val (out-of-sample for val)
    s_v2_val, per_v2_val, _, actvv = cum_pnl_per_sym_thr(pred_va, sym_va, mp_t_va, mp_th_va, thr_ps)
    print(f"  val cum_pnl={s_v2_val:+.4f}  active={actvv/n_val:.4f}", flush=True)
    results["variants"]["V2_per_sym_test"] = {
        "thr_per_sym": {str(k): list(thr_ps[k]) for k in SYMS},
        "test_cum_pnl": s_v2, "test_per_sym": per_v2,
        "test_active_count": actv_v2, "test_active_total": actv_v2_t,
        "test_active_rate": actv_v2_t / n_test,
        "val_cum_pnl_oos": s_v2_val,
        "val_per_sym_oos": per_v2_val,
        "val_active_rate_oos": actvv / n_val,
    }

    # ====== V3 Pure OOF thresh from val date 76-79 ======
    print("\n=== V3: Pure OOF thresh (DE on val 76-79) ===", flush=True)
    t0 = time.time()
    s_de_v, tu_v3, td_v3 = de_2d(pred_va, mp_t_va, mp_th_va)
    print(f"  DE-best val cum_pnl={s_de_v:+.4f} thr_up={tu_v3:.3e} thr_dn={td_v3:.3e}  ({time.time()-t0:.1f}s)",
          flush=True)
    s_v3, per_v3, actv_v3, actv_v3_t = cum_pnl_at_thr(pred_te, sym_te, mp_t_te, mp_th_te, tu_v3, td_v3)
    print(f"  test cum_pnl={s_v3:+.4f}  per_sym={[round(x,2) for x in per_v3]}  active={actv_v3_t/n_test:.4f}",
          flush=True)
    results["variants"]["V3_OOF_val"] = {
        "thr_up": tu_v3, "thr_dn": td_v3,
        "val_cum_pnl_in_sample": s_de_v,
        "test_cum_pnl": s_v3, "test_per_sym": per_v3,
        "test_active_count": actv_v3, "test_active_total": actv_v3_t,
        "test_active_rate": actv_v3_t / n_test,
    }

    # ====== V4 K-fold time-block CV on val ======
    print("\n=== V4: 5-fold time-block CV on val 76-79 ===", flush=True)
    fold, n_blocks = make_kfold_blocks(date_va, sess_va, k=5)
    print(f"  n_blocks={n_blocks}, fold counts:",
          [(int(k), int((fold == k).sum())) for k in range(5)], flush=True)

    fold_thr = []
    fold_pnls = []  # held-out fold pnl when DE'd on others
    for fid in range(5):
        m_train = (fold != fid)
        m_held = (fold == fid)
        # DE on the 4 training folds
        s_tr, tu_f, td_f = de_2d(pred_va[m_train], mp_t_va[m_train], mp_th_va[m_train])
        # Eval on held-out
        s_held, _, _, _ = cum_pnl_at_thr(pred_va[m_held], sym_va[m_held],
                                         mp_t_va[m_held], mp_th_va[m_held], tu_f, td_f)
        fold_thr.append((tu_f, td_f))
        fold_pnls.append(s_held)
        print(f"  fold {fid}: thr_up={tu_f:.3e} thr_dn={td_f:.3e}  train_pnl={s_tr:+.4f}  held_pnl={s_held:+.4f}",
              flush=True)

    fold_thr_arr = np.array(fold_thr, dtype=np.float64)
    median_tu = float(np.median(fold_thr_arr[:, 0]))
    median_td = float(np.median(fold_thr_arr[:, 1]))
    mean_tu = float(np.mean(fold_thr_arr[:, 0]))
    mean_td = float(np.mean(fold_thr_arr[:, 1]))
    sum_held = float(np.sum(fold_pnls))
    print(f"  CV held-out total={sum_held:+.4f}  per_fold={[round(x,3) for x in fold_pnls]}",
          flush=True)
    print(f"  median thresh: tu={median_tu:.3e} td={median_td:.3e}", flush=True)
    print(f"  mean   thresh: tu={mean_tu:.3e} td={mean_td:.3e}", flush=True)

    s_v4_med, per_v4_med, actv_v4_med, actv_v4_med_t = cum_pnl_at_thr(
        pred_te, sym_te, mp_t_te, mp_th_te, median_tu, median_td)
    s_v4_mean, per_v4_mean, actv_v4_mean, actv_v4_mean_t = cum_pnl_at_thr(
        pred_te, sym_te, mp_t_te, mp_th_te, mean_tu, mean_td)
    print(f"  V4 (median) test cum_pnl={s_v4_med:+.4f}  per_sym={[round(x,2) for x in per_v4_med]}  "
          f"active={actv_v4_med_t/n_test:.4f}", flush=True)
    print(f"  V4 (mean)   test cum_pnl={s_v4_mean:+.4f}  per_sym={[round(x,2) for x in per_v4_mean]}  "
          f"active={actv_v4_mean_t/n_test:.4f}", flush=True)

    results["variants"]["V4_kfold_val"] = {
        "k": 5,
        "fold_thresholds": [{"fold": i, "thr_up": fold_thr[i][0], "thr_dn": fold_thr[i][1],
                             "held_out_pnl": fold_pnls[i]} for i in range(5)],
        "median_thr_up": median_tu, "median_thr_dn": median_td,
        "mean_thr_up": mean_tu, "mean_thr_dn": mean_td,
        "cv_held_out_total_pnl": sum_held,
        "test_at_median": {"cum_pnl": s_v4_med, "per_sym": per_v4_med,
                           "active_count": actv_v4_med, "active_total": actv_v4_med_t,
                           "active_rate": actv_v4_med_t / n_test},
        "test_at_mean": {"cum_pnl": s_v4_mean, "per_sym": per_v4_mean,
                         "active_count": actv_v4_mean, "active_total": actv_v4_mean_t,
                         "active_rate": actv_v4_mean_t / n_test},
    }

    # ====== Summary ======
    summary = {
        "V1_baseline": results["variants"]["V1_baseline"]["test_cum_pnl"],
        "V2_per_sym_test_in_sample": results["variants"]["V2_per_sym_test"]["test_cum_pnl"],
        "V3_OOF_val": results["variants"]["V3_OOF_val"]["test_cum_pnl"],
        "V4_kfold_median": results["variants"]["V4_kfold_val"]["test_at_median"]["cum_pnl"],
        "V4_kfold_mean": results["variants"]["V4_kfold_val"]["test_at_mean"]["cum_pnl"],
    }
    results["summary_test_cum_pnl"] = summary

    # Best non-overfit candidate = max(V3, V4)
    candidates = {
        "V3_OOF_val": summary["V3_OOF_val"],
        "V4_kfold_median": summary["V4_kfold_median"],
        "V4_kfold_mean": summary["V4_kfold_mean"],
    }
    best_name = max(candidates, key=candidates.get)
    results["best_oof_variant"] = best_name
    results["best_oof_test_cum_pnl"] = candidates[best_name]
    results["delta_vs_V1"] = candidates[best_name] - summary["V1_baseline"]

    out_path = os.path.join(HERE, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {out_path}", flush=True)

    print(f"\n=== SUMMARY (test cum_pnl) ===", flush=True)
    for k, v in summary.items():
        print(f"  {k}: {v:+.4f}", flush=True)
    print(f"\nbest OOF variant: {best_name} = {candidates[best_name]:+.4f}", flush=True)
    print(f"delta vs V1 = {candidates[best_name] - summary['V1_baseline']:+.4f}", flush=True)
    print(f"\nelapsed: {time.time()-t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
