"""Verify iter_015 ensemble cum_pnl on the full 442k test using cached features.

Loads:
  - schemeP_test.npz cached features (442k × 370)
  - 5 LGB models (T75, copied to iter015_pkg)
  - 5 NN .npz models (T87 SPO+ DFL)

Runs LGB inference + NN numpy inference, weighted-averages with w_nn=1.0, w_lgb=1.5,
applies the EV gate from thresholds.json (thr_up=0.000358, thr_dn=0.000216),
and computes per-sym cum_pnl.

Expected: matches +40.13 LOSO-equiv from ev_gate_ensemble_results.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

T68_CACHE = os.path.join(ROOT, "experiments", "T68_stage5_features", "cache")
T75_DIR = os.path.join(ROOT, "experiments", "T75_regression_dmid")

sys.path.insert(0, os.path.join(ROOT, "experiments", "T81_nn_regression_pnl"))
from numpy_nn_inference import MLPNumpy  # noqa

FEE = 0.0001
SEEDS = (1, 7, 13, 42, 100)


def main():
    print("=== iter_015 full-test verification (T87 SPO+ DFL + T75 LGB) ===", flush=True)

    t0 = time.time()
    d = np.load(os.path.join(T68_CACHE, "schemeP_test.npz"))
    X = d["X"]
    sym = d["sym"]
    mp_t = d["mp_t"].astype(np.float64)
    mp_t60 = d["mp_t60"].astype(np.float64)
    print(f"  loaded {X.shape} test cache in {time.time()-t0:.1f}s", flush=True)

    feat_names_path = os.path.join(T68_CACHE, "schemeP_feat_names.txt")
    with open(feat_names_path) as f:
        all_names = [l.strip() for l in f]
    DROP = {"dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
            "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
            "qrank_W100_cumspread", "kyle_lam_W50", "kyle_lam_W100",
            "roll_eff_spr_ratio_W100", "liq_asym_top5_W5"}
    keep_idx = np.array([i for i, n in enumerate(all_names) if n not in DROP], dtype=np.int64)
    X_kept = X[:, keep_idx]
    print(f"  X_kept: {X_kept.shape}", flush=True)

    t0 = time.time()
    lgb_models = []
    for s in SEEDS:
        p = os.path.join(T75_DIR, f"model_T75_seed{s}.txt")
        lgb_models.append(lgb.Booster(model_file=p))
    print(f"  loaded {len(lgb_models)} LGB in {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    nn_models = []
    for s in SEEDS:
        p = os.path.join(HERE, f"model_T87_seed{s}_main.npz")
        nn_models.append(MLPNumpy(p))
    print(f"  loaded {len(nn_models)} T87 NN in {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    p_lgb_acc = np.zeros(len(X), dtype=np.float64)
    BATCH = 100_000
    for s_idx, m in enumerate(lgb_models):
        for s in range(0, len(X), BATCH):
            p_lgb_acc[s:s+BATCH] += m.predict(X_kept[s:s+BATCH])
    p_lgb = (p_lgb_acc / len(lgb_models)).astype(np.float32)
    print(f"  LGB ensemble inference in {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    p_nn_acc = np.zeros(len(X), dtype=np.float64)
    for nn in nn_models:
        for s in range(0, len(X), BATCH):
            X_s = nn.standardize(X_kept[s:s+BATCH])
            p_nn_acc[s:s+BATCH] += nn.forward(X_s).astype(np.float64)
    p_nn = (p_nn_acc / len(nn_models)).astype(np.float32)
    print(f"  T87 NN ensemble inference in {time.time()-t0:.1f}s", flush=True)

    w_nn, w_lgb = 1.0, 1.5
    p_comb = (w_nn * p_nn + w_lgb * p_lgb) / (w_nn + w_lgb)

    thr_up, thr_dn = 0.000358, 0.000216
    actions = np.full(len(p_comb), 1, dtype=np.int8)
    actions[p_comb > thr_up] = 2
    actions[p_comb < -thr_dn] = 0

    side = actions.astype(np.float64) - 1.0
    abs_side = np.abs(side)
    diff = mp_t60 - mp_t
    fee_pnl = FEE * abs_side * np.abs((mp_t60 + 1.0) + (mp_t + 1.0))
    denom = mp_t + 1.0
    pnl = (side * diff - fee_pnl) / denom
    cum_pnl_total = float(pnl.sum())
    n_active = int((actions != 1).sum())

    print(f"\n  TOTAL cum_pnl = {cum_pnl_total:+.4f}  n_active={n_active:,}/{len(actions):,}",
          flush=True)
    print(f"  per-sym:", flush=True)
    per_sym = []
    for s_id in (0, 1, 2, 3, 4):
        m = sym == s_id
        c = float(pnl[m].sum())
        n = int((actions[m] != 1).sum())
        n_total = int(m.sum())
        per_sym.append(c)
        print(f"    sym={s_id}: cum_pnl={c:+.4f}  n_active={n:,}/{n_total:,}", flush=True)
    sum_per_sym = float(sum(per_sym))
    print(f"\n  sum_per_sym (LOSO-equiv) = {sum_per_sym:+.4f}", flush=True)
    print(f"  (offline ev_gate_ensemble target: +40.1290)", flush=True)
    print(f"  diff: {sum_per_sym - 40.1290:+.6f}", flush=True)
    print(f"  vs iter_013 (+36.2281): {sum_per_sym - 36.2281:+.4f}", flush=True)
    print(f"  vs iter_014 (+38.2810): {sum_per_sym - 38.2810:+.4f}", flush=True)

    out = {
        "iter": "iter_015",
        "cum_pnl_total": cum_pnl_total,
        "sum_per_sym": sum_per_sym,
        "per_sym": per_sym,
        "n_active": n_active,
        "thr_up": thr_up,
        "thr_dn": thr_dn,
        "w_nn": w_nn, "w_lgb": w_lgb,
        "p_lgb_stats": {"mean": float(p_lgb.mean()), "std": float(p_lgb.std())},
        "p_nn_stats": {"mean": float(p_nn.mean()), "std": float(p_nn.std())},
        "p_comb_stats": {"mean": float(p_comb.mean()), "std": float(p_comb.std())},
        "vs_iter013": sum_per_sym - 36.2281,
        "vs_iter014": sum_per_sym - 38.2810,
    }
    out_path = os.path.join(HERE, "full_test_verify.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
