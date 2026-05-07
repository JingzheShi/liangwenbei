"""T77: Val sanity check.

Apply best vote rules (V0 K=4, V1 K=3 DE, V2 DE) to V4 walk-forward val (date 76-79)
to check if test-set DE optimization is overfit.

Reload models and compute val probs for h_5/10/20/40 (already trained).
For h_60, we need a single seed=42 V4 walk-forward model from T70.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

import lightgbm as lgb  # noqa: E402

T68_DIR = os.path.join(ROOT, "experiments", "T68_stage5_features")
CACHE_DIR = os.path.join(T68_DIR, "cache")
T70_DIR = os.path.join(ROOT, "experiments", "T70_v4_stage5")

T53_DIR = os.path.join(ROOT, "experiments", "T53_r34_stage3")
sys.path.insert(0, T53_DIR)
from de_thresh import vectorized_pnl  # noqa: E402

SYMS = (0, 1, 2, 3, 4)
HORIZONS = (5, 10, 20, 40, 60)

T59_FAIL_NAMES = [
    "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
    "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
    "qrank_W100_cumspread",
    "kyle_lam_W50", "kyle_lam_W100",
    "roll_eff_spr_ratio_W100",
]
STAGE5_FAIL_NAMES = ["liq_asym_top5_W5"]
DROP_NAMES = T59_FAIL_NAMES + STAGE5_FAIL_NAMES


def get_slicer():
    feat_names_path = os.path.join(CACHE_DIR, "schemeP_feat_names.txt")
    with open(feat_names_path) as f:
        all_feat_names = [line.strip() for line in f]
    name_to_idx = {n: i for i, n in enumerate(all_feat_names)}
    drop_idx = set()
    for n in DROP_NAMES:
        if n in name_to_idx:
            drop_idx.add(name_to_idx[n])
    keep_idx = np.array([i for i in range(len(all_feat_names)) if i not in drop_idx], dtype=np.int64)
    return keep_idx


def predict_proba(booster, X, batch=200_000):
    chunks = []
    for s in range(0, len(X), batch):
        chunks.append(booster.predict(X[s : s + batch]).astype(np.float32))
    return np.concatenate(chunks, axis=0)


def vote_v0(probs_dict, K):
    n = len(next(iter(probs_dict.values())))
    up = np.zeros(n, dtype=np.int8)
    dn = np.zeros(n, dtype=np.int8)
    for p in probs_dict.values():
        a = p.argmax(axis=1)
        up += (a == 2).astype(np.int8)
        dn += (a == 0).astype(np.int8)
    pred = np.full(n, 1, dtype=np.int8)
    pred[up >= K] = 2
    pred[dn >= K] = 0
    pred[(up >= K) & (dn >= K)] = 1
    return pred


def vote_v1(probs_dict, K, T_h):
    n = len(next(iter(probs_dict.values())))
    up = np.zeros(n, dtype=np.int8)
    dn = np.zeros(n, dtype=np.int8)
    for h, p in probs_dict.items():
        a = p.argmax(axis=1)
        max_p = p.max(axis=1)
        valid = max_p > T_h.get(h, 0.0)
        up += ((a == 2) & valid).astype(np.int8)
        dn += ((a == 0) & valid).astype(np.int8)
    pred = np.full(n, 1, dtype=np.int8)
    pred[up >= K] = 2
    pred[dn >= K] = 0
    pred[(up >= K) & (dn >= K)] = 1
    return pred


def vote_v2(probs_dict, w_h, T):
    n = len(next(iter(probs_dict.values())))
    score = np.zeros(n, dtype=np.float64)
    for h, p in probs_dict.items():
        score += w_h.get(h, 0.0) * (p[:, 2].astype(np.float64) - p[:, 0].astype(np.float64))
    pred = np.full(n, 1, dtype=np.int8)
    pred[score > T] = 2
    pred[score < -T] = 0
    return pred


def evaluate_pred(pred, sym, label, mp_t, mp_th):
    per_sym = []
    for k in SYMS:
        m = sym == k
        per_sym.append(float(vectorized_pnl(pred[m], label[m], mp_t[m], mp_th[m]).sum()))
    total = float(vectorized_pnl(pred, label, mp_t, mp_th).sum())
    return {"loso_equiv_sum": float(sum(per_sym)), "total": total,
            "per_sym": per_sym, "n_active": int((pred != 1).sum()),
            "n_total": len(pred)}


def main():
    print("=== T77 Val (date 76-79) sanity check ===\n", flush=True)

    slicer = get_slicer()
    train_full = np.load(os.path.join(CACHE_DIR, "schemeP_train.npz"))
    date_tr = train_full["date"]
    m_va = date_tr >= 76

    X_va = train_full["X"][m_va][:, slicer].astype(np.float32, copy=False)
    sym_va = train_full["sym"][m_va].astype(np.int8)
    mp_t_va = train_full["mp_t"][m_va]

    print(f"Val rows: {len(X_va):,}, syms: {dict(zip(*np.unique(sym_va, return_counts=True)))}", flush=True)

    probs_val = {}

    # h60 from T70 single seed=42 model
    print("Predicting h_60 with T70 seed=42 model (V4 walk-forward holdout)...", flush=True)
    bst60 = lgb.Booster(model_file=os.path.join(T70_DIR, "model_T70_seed42.txt"))
    probs_val[60] = predict_proba(bst60, X_va)

    for H in (5, 10, 20, 40):
        print(f"Predicting h_{H} with T77 seed=42 model...", flush=True)
        bst = lgb.Booster(model_file=os.path.join(HERE, f"model_T77_h{H}_seed42.txt"))
        probs_val[H] = predict_proba(bst, X_va)

    # We need labels & mp_th for each horizon to compute val pnl
    # But val pnl is per-horizon, while our metric is sum-per-sym evaluating against
    # the TRUE horizon = 60 (since we want to predict h60 outcome using vote ensemble).
    # Actually no - the vote ensemble is meant to predict overall direction; the metric
    # is P&L computed assuming we hold the position for h ticks. Iter_012 used h=60.
    # Following iter_012, we evaluate vote pred with h=60 labels and h=60 forward midprice.
    label_va = train_full["y60"][m_va].astype(np.int64)
    mp_th_va = train_full["mp_t60"][m_va].astype(np.float64)

    print("\n[Val baselines]", flush=True)
    pred_h60_argmax = probs_val[60].argmax(axis=1).astype(np.int8)
    r0 = evaluate_pred(pred_h60_argmax, sym_va, label_va, mp_t_va, mp_th_va)
    print(f"  h60 argmax (single seed=42): loso_equiv={r0['loso_equiv_sum']:+.4f} (n_active={r0['n_active']:,})", flush=True)

    # iter_012 thresh
    from de_thresh import gate_asymmetric
    pred_iter012 = gate_asymmetric(probs_val[60], 0.4379910127008917, 0.4448964062818059,
                                    8.099624641794145e-05, 6.68533652402048e-05)
    r_iter = evaluate_pred(pred_iter012, sym_va, label_va, mp_t_va, mp_th_va)
    print(f"  iter_012 thresh: loso_equiv={r_iter['loso_equiv_sum']:+.4f} (n_active={r_iter['n_active']:,})", flush=True)

    print("\n[Val: V0 raw vote]", flush=True)
    for K in (3, 4, 5):
        pred = vote_v0(probs_val, K)
        r = evaluate_pred(pred, sym_va, label_va, mp_t_va, mp_th_va)
        print(f"  K={K}: loso_equiv={r['loso_equiv_sum']:+.4f} (n_active={r['n_active']:,})", flush=True)

    print("\n[Val: V1 K=3 DE T_h from test optimization]", flush=True)
    T_h_k3 = {5: 0.47798251052007046, 10: 0.38537998706797705, 20: 0.5330342383720678,
              40: 0.4242430270100717, 60: 0.43579704436645106}
    pred = vote_v1(probs_val, 3, T_h_k3)
    r = evaluate_pred(pred, sym_va, label_va, mp_t_va, mp_th_va)
    print(f"  V1 K=3 DE: loso_equiv={r['loso_equiv_sum']:+.4f} (n_active={r['n_active']:,})", flush=True)

    print("\n[Val: V1 K=4 DE T_h from test optimization]", flush=True)
    T_h_k4 = {5: 0.4760337700575928, 10: 0.4014123939947481, 20: 0.4473944753341131,
              40: 0.4196904073019845, 60: 0.3688455428201138}
    pred = vote_v1(probs_val, 4, T_h_k4)
    r = evaluate_pred(pred, sym_va, label_va, mp_t_va, mp_th_va)
    print(f"  V1 K=4 DE: loso_equiv={r['loso_equiv_sum']:+.4f} (n_active={r['n_active']:,})", flush=True)

    print("\n[Val: V2 DE w_h, T from test optimization]", flush=True)
    w_v2 = {5: 1.2143627654532745, 10: 0.40267224874124863, 20: 1.3439619603738586,
            40: 1.3601723540099404, 60: 1.7312293269477994}
    T_v2 = 0.8165
    pred = vote_v2(probs_val, w_v2, T_v2)
    r = evaluate_pred(pred, sym_va, label_va, mp_t_va, mp_th_va)
    print(f"  V2 DE: loso_equiv={r['loso_equiv_sum']:+.4f} (n_active={r['n_active']:,})", flush=True)

    out = {
        "task": "T77 Val sanity check",
        "n_val": len(X_va),
        "h60_argmax_seed42": r0,
        "iter012_thresh": r_iter,
    }
    out["v0_K3"] = evaluate_pred(vote_v0(probs_val, 3), sym_va, label_va, mp_t_va, mp_th_va)
    out["v0_K4"] = evaluate_pred(vote_v0(probs_val, 4), sym_va, label_va, mp_t_va, mp_th_va)
    out["v0_K5"] = evaluate_pred(vote_v0(probs_val, 5), sym_va, label_va, mp_t_va, mp_th_va)
    out["v1_K3_DE"] = evaluate_pred(vote_v1(probs_val, 3, T_h_k3), sym_va, label_va, mp_t_va, mp_th_va)
    out["v1_K4_DE"] = evaluate_pred(vote_v1(probs_val, 4, T_h_k4), sym_va, label_va, mp_t_va, mp_th_va)
    out["v2_DE"] = evaluate_pred(vote_v2(probs_val, w_v2, T_v2), sym_va, label_va, mp_t_va, mp_th_va)

    with open(os.path.join(HERE, "val_check_results.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved val_check_results.json", flush=True)


if __name__ == "__main__":
    main()
