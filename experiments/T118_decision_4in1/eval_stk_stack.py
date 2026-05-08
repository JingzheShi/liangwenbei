"""T118 follow-up: refine STK (use PnL-based win-rate) and test rule stacking.

STK fix: hit_rate = fraction of trades that would have positive PnL (sign-match gives
many false flats due to mp_th == mp_t rows). With raw sign match, top bucket only hits
0.544; with PnL-based win we expect better signal.

Stacking: NSF AND TBT, NSF AND BAT (under shuffle), NSF OR TBT.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from eval_4in1 import (
    build_4way_pred, eval_actions, baseline_de, rule_nsf, rule_tbt, rule_bat,
    compute_pnl, THR_UP_BASE, ROOT, FEE,
)

HERE = Path("/root/projects/liangwenbei_workdir/experiments/T118_decision_4in1")


def rule_stk_pnl(p_comb, mp_t, mp_th, cutoff=0.5, n_buckets=10, n_folds=5,
                 rng_seed=0):
    """5-fold OOF: hit_rate = fraction of trades with positive (per-row) PnL
    (using sign(pred) as the side). Bucket by |pred| quantile.
    """
    n = len(p_comb)
    rng = np.random.default_rng(rng_seed)
    folds = rng.integers(0, n_folds, size=n)
    abs_pred = np.abs(p_comb)
    actions = np.full(n, 1, dtype=np.int8)
    fold_info = []
    for f in range(n_folds):
        train_m = folds != f
        test_m = folds == f
        edges = np.quantile(abs_pred[train_m], np.linspace(0, 1, n_buckets + 1))
        edges[0] = -np.inf
        edges[-1] = np.inf
        train_bins = np.clip(np.digitize(abs_pred[train_m], edges, right=False) - 1,
                             0, n_buckets - 1)
        # simulate firing on every train row in direction sign(pred); per-row PnL
        signs = np.sign(p_comb[train_m])
        a_train = (signs.astype(np.int8) + 1)  # 0 short, 2 long; sign=0 → 1 (flat)
        a_train[signs == 0] = 1
        pnl = compute_pnl(a_train, mp_t[train_m], mp_th[train_m])
        win_rate = np.zeros(n_buckets, dtype=np.float64)
        cnt = np.zeros(n_buckets, dtype=np.int64)
        for b in range(n_buckets):
            m = train_bins == b
            if m.sum() > 0:
                cnt[b] = m.sum()
                win_rate[b] = (pnl[m] > 0).sum() / m.sum()
        # apply
        test_bins = np.clip(np.digitize(abs_pred[test_m], edges, right=False) - 1,
                            0, n_buckets - 1)
        fire = win_rate[test_bins] > cutoff
        local = np.full(test_m.sum(), 1, dtype=np.int8)
        p_t = p_comb[test_m]
        local[fire & (p_t > 0)] = 2
        local[fire & (p_t < 0)] = 0
        actions[test_m] = local
        fold_info.append({"fold": f, "win_rates": win_rate.tolist(), "cnt": cnt.tolist(),
                          "edges": edges.tolist()})
    return actions, fold_info


def main():
    p_comb, meta, sym = build_4way_pred()
    n = len(p_comb)

    # spread for NSF
    schemeP = np.load(ROOT / "experiments/T68_stage5_features/cache/schemeP_test.npz")
    feat_names_path = ROOT / "experiments/T68_stage5_features/cache/schemeP_feat_names.txt"
    with open(feat_names_path) as f:
        all_names = [l.strip() for l in f]
    spread_t = (schemeP["X"][:, all_names.index("ask1")]
                - schemeP["X"][:, all_names.index("bid1")]).astype(np.float64)

    mp_t = meta["midprice_t"].to_numpy(dtype=np.float64)
    mp_th = meta["midprice_th"].to_numpy(dtype=np.float64)

    a_base = baseline_de(p_comb)
    rate_base = float((a_base != 1).sum() / len(a_base))

    out = {"baseline_active_rate": rate_base}

    # ============ STK refined (PnL-based win rate) ============
    print("=== STK (PnL-based win rate) ===", flush=True)
    stk_results = []
    for cutoff in [0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52]:
        a, info = rule_stk_pnl(p_comb, mp_t, mp_th, cutoff=cutoff,
                               n_buckets=10, n_folds=5, rng_seed=42)
        r = eval_actions(a, meta, sym, f"stk_pnl_c{cutoff:.2f}")
        r["cutoff"] = cutoff
        r["fold0_win_rates"] = info[0]["win_rates"]
        stk_results.append(r)
        print(f"  cutoff={cutoff}: loso={r['loso']:+.4f}  per_sym_min={r['per_sym_min']:+.4f}  "
              f"rate={r['active_rate']:.4f}", flush=True)
    out["stk_pnl_results"] = stk_results

    # ============ Stacking ============
    # NSF gate (predict requires |pred| > FEE_EFF + γ*spread) AND TBT (top-K)
    print("\n=== Stacking: NSF AND TBT ===", flush=True)
    nsf_actions = rule_nsf(p_comb, spread_t, fee_eff=THR_UP_BASE, gamma=0.0)
    nsf_fire = nsf_actions != 1
    K = int(rate_base * n)
    tbt_order = np.argpartition(-np.abs(p_comb), K - 1)[:K]
    tbt_fire = np.zeros(n, dtype=bool)
    tbt_fire[tbt_order] = True
    fire = nsf_fire & tbt_fire
    a_stack = np.full(n, 1, dtype=np.int8)
    a_stack[fire & (p_comb > 0)] = 2
    a_stack[fire & (p_comb < 0)] = 0
    r = eval_actions(a_stack, meta, sym, "nsf_AND_tbt")
    print(f"  loso={r['loso']:+.4f}  per_sym_min={r['per_sym_min']:+.4f}  rate={r['active_rate']:.4f}", flush=True)
    out["stack_nsf_AND_tbt"] = r

    # NSF OR TBT
    print("\n=== Stacking: NSF OR TBT ===", flush=True)
    fire = nsf_fire | tbt_fire
    a_stack = np.full(n, 1, dtype=np.int8)
    a_stack[fire & (p_comb > 0)] = 2
    a_stack[fire & (p_comb < 0)] = 0
    r = eval_actions(a_stack, meta, sym, "nsf_OR_tbt")
    print(f"  loso={r['loso']:+.4f}  per_sym_min={r['per_sym_min']:+.4f}  rate={r['active_rate']:.4f}", flush=True)
    out["stack_nsf_OR_tbt"] = r

    # NSF gamma=0.25 alone (lower rate, higher pmin?) at fe=2e-4
    print("\n=== NSF (fe=2e-4, g=0.25) — already tested, recapping ===", flush=True)
    a = rule_nsf(p_comb, spread_t, 2e-4, 0.25)
    r = eval_actions(a, meta, sym, "nsf_fe2e-4_g0.25")
    print(f"  loso={r['loso']:+.4f}  per_sym_min={r['per_sym_min']:+.4f}  rate={r['active_rate']:.4f}", flush=True)
    out["nsf_fe2e-4_g0.25"] = r

    # Try sweeping fee_eff at g=0.1, 0.15 (small gamma) and finer fe grid
    print("\n=== NSF fine sweep (small gamma, near baseline thr) ===", flush=True)
    sweep = []
    for fe in [2.5e-4, 2.7e-4, 2.88e-4, 3.0e-4, 3.2e-4]:
        for g in [0.0, 0.05, 0.10, 0.15, 0.20]:
            a = rule_nsf(p_comb, spread_t, fe, g)
            r = eval_actions(a, meta, sym, f"nsf_fe{fe:.2e}_g{g:.2f}")
            r["fee_eff"] = fe
            r["gamma"] = g
            sweep.append(r)
    # show top-10 by loso, top-10 by per_sym_min
    sweep_by_loso = sorted(sweep, key=lambda r: -r["loso"])[:10]
    sweep_by_pmin = sorted(sweep, key=lambda r: -r["per_sym_min"])[:10]
    print("  Top-5 by LOSO:", flush=True)
    for r in sweep_by_loso[:5]:
        print(f"    fe={r['fee_eff']:.2e} g={r['gamma']:.2f}: loso={r['loso']:+.4f}  "
              f"pmin={r['per_sym_min']:+.4f}  rate={r['active_rate']:.4f}", flush=True)
    print("  Top-5 by per_sym_min:", flush=True)
    for r in sweep_by_pmin[:5]:
        print(f"    fe={r['fee_eff']:.2e} g={r['gamma']:.2f}: loso={r['loso']:+.4f}  "
              f"pmin={r['per_sym_min']:+.4f}  rate={r['active_rate']:.4f}", flush=True)
    out["nsf_fine_sweep_top_loso"] = sweep_by_loso
    out["nsf_fine_sweep_top_pmin"] = sweep_by_pmin

    out_path = HERE / "results_stk_stack.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
