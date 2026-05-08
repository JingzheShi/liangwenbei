"""Verify T120 iter_016 ablation 002 full-442k cum_pnl.

Same as T119 full_test_verify, but loads the 5x T120 LGB Huber monotone
models in place of T99 LGB Huber baselines.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

T68_CACHE = os.path.join(ROOT, "experiments", "T68_stage5_features", "cache")
PKG = os.path.join(HERE, "pkg")

sys.path.insert(0, PKG)
from Predictor import _MLPNumpy

FEE = 0.0001
SEEDS = (1, 7, 13, 42, 100)


def main():
    print("=== T120 iter_016 ablation 002 full-test verification ===", flush=True)

    t0 = time.time()
    d = np.load(os.path.join(T68_CACHE, "schemeP_test.npz"))
    X = d["X"]
    sym = d["sym"]
    mp_t = d["mp_t"].astype(np.float64)
    mp_t60 = d["mp_t60"].astype(np.float64)
    print(f"  loaded schemeP_test {X.shape} in {time.time()-t0:.1f}s", flush=True)

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

    BATCH = 100_000

    # ---- T87 SPO+ NN ensemble ----
    t0 = time.time()
    nn_models = [_MLPNumpy(os.path.join(PKG, f"nn_h60_seed{s}.npz")) for s in SEEDS]
    p_nn = np.zeros(len(X), dtype=np.float64)
    for nn in nn_models:
        for s in range(0, len(X), BATCH):
            p_nn[s:s+BATCH] += nn.predict(X_kept[s:s+BATCH]).astype(np.float64)
    p_nn /= len(nn_models)
    t_nn = time.time() - t0
    print(f"  T87 NN inference in {t_nn:.1f}s  mean={p_nn.mean():+.3e} std={p_nn.std():.3e}", flush=True)

    # ---- T99 CB Huber ensemble (UNCHANGED vs T119) ----
    t0 = time.time()
    cb_models = []
    for s in SEEDS:
        cb = CatBoostRegressor()
        cb.load_model(os.path.join(PKG, f"model_cb_huber_seed{s}.cbm"))
        cb_models.append(cb)
    p_cb = np.zeros(len(X), dtype=np.float64)
    for cb in cb_models:
        for s in range(0, len(X), BATCH):
            p_cb[s:s+BATCH] += cb.predict(X_kept[s:s+BATCH])
    p_cb /= len(cb_models)
    t_cb = time.time() - t0
    print(f"  T99 CB Huber inference in {t_cb:.1f}s  mean={p_cb.mean():+.3e} std={p_cb.std():.3e}", flush=True)

    # ---- T120 LGB Huber MONOTONE ensemble (NEW vs T119) ----
    t0 = time.time()
    huber_models = [lgb.Booster(model_file=os.path.join(PKG, f"model_T120_huber_mono_seed{s}.txt")) for s in SEEDS]
    p_huber = np.zeros(len(X), dtype=np.float64)
    for m in huber_models:
        for s in range(0, len(X), BATCH):
            p_huber[s:s+BATCH] += m.predict(X_kept[s:s+BATCH])
    p_huber /= len(huber_models)
    t_huber = time.time() - t0
    print(f"  T120 LGB Huber monotone inference in {t_huber:.1f}s  mean={p_huber.mean():+.3e} std={p_huber.std():.3e}", flush=True)

    # ---- Combine 3-way (T119 weights, unchanged) ----
    w_nn, w_cb, w_huber = 1.0, 0.7, 1.0
    wsum = w_nn + w_cb + w_huber
    p_comb = (w_nn * p_nn + w_cb * p_cb + w_huber * p_huber) / wsum
    p_comb = p_comb.astype(np.float32)

    with open(os.path.join(PKG, "thresholds.json")) as f:
        tcfg = json.load(f)
    h60_cfg = next(h for h in tcfg["horizons"] if int(h["h"]) == 60)
    thr_up = float(h60_cfg["thr_up"])
    thr_dn = float(h60_cfg["thr_dn"])

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

    print(f"\n  TOTAL cum_pnl = {cum_pnl_total:+.4f}  n_active={n_active:,}/{len(actions):,}", flush=True)
    print(f"  thr_up={thr_up:.6e}  thr_dn={thr_dn:.6e}", flush=True)
    print(f"  w=({w_nn},{w_cb},{w_huber})", flush=True)
    print(f"  per-sym:", flush=True)
    per_sym = []
    for s_id in (0, 1, 2, 3, 4):
        m = sym == s_id
        c = float(pnl[m].sum())
        per_sym.append(c)
        print(f"    sym={s_id}: cum_pnl={c:+.4f}  n_active={(actions[m]!=1).sum():,}/{int(m.sum()):,}", flush=True)
    sum_per_sym = float(sum(per_sym))
    print(f"\n  sum_per_sym (LOSO-equiv) = {sum_per_sym:+.4f}", flush=True)

    ref_T119 = 39.9761
    print(f"  vs T119 ablation_001 (+39.98): diff={sum_per_sym-ref_T119:+.4f}", flush=True)

    out = {
        "iter": "T120 iter_016 ablation 002 (T119 base + LGB Huber monotone)",
        "cum_pnl_total": cum_pnl_total,
        "sum_per_sym": sum_per_sym,
        "per_sym": per_sym,
        "n_active": n_active,
        "n_total": int(len(actions)),
        "active_rate": float(n_active) / float(len(actions)),
        "thr_up": thr_up, "thr_dn": thr_dn,
        "weights": {"nn": w_nn, "cb_huber": w_cb, "lgb_huber_monotone": w_huber, "gru": 0.0},
        "timing_seconds": {
            "T87_nn": t_nn,
            "T99_cb_huber": t_cb,
            "T120_lgb_huber_mono": t_huber,
            "total_inference": t_nn + t_cb + t_huber,
        },
        "p_nn_stats": {"mean": float(p_nn.mean()), "std": float(p_nn.std())},
        "p_cb_huber_stats": {"mean": float(p_cb.mean()), "std": float(p_cb.std())},
        "p_lgb_huber_mono_stats": {"mean": float(p_huber.mean()), "std": float(p_huber.std())},
        "p_comb_stats": {"mean": float(p_comb.mean()), "std": float(p_comb.std())},
        "vs_T119_ablation_001": sum_per_sym - ref_T119,
    }
    out_path = os.path.join(HERE, "full_test_verify.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
