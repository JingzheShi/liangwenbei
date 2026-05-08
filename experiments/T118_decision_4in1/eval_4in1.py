"""T118: post-hoc decision rule comparison.

Compares 4 post-hoc decision rules vs baseline DE asym thresholds (iter_016 v3):
  Rule 1 NSF: pred > FEE_EFF + gamma * spread_t  (3 gammas)
  Rule 2 TBT: top-K by |pred|, K = oof_active_rate * N
  Rule 3 STK: 5-fold OOF win-rate bucketing on |pred|
  Rule 4 BAT: within-batch quantile of pred

Predictions are loaded from existing 5-seed pred parquets:
  T87 (NN), T99 CB Huber, T99 LGB Huber, T95 GRU.
Combined with iter_016_v3 weights (1.0, 0.7, 1.0, 1.0).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root/projects/liangwenbei_workdir")
HERE = ROOT / "experiments" / "T118_decision_4in1"

SEEDS = (1, 7, 13, 42, 100)
FEE = 1e-4
BATCH = 1024  # for BAT

# Baseline iter_016 v3 thresholds
THR_UP_BASE = 2.882863e-4
THR_DN_BASE = 2.186241e-4

W = (1.0, 0.7, 1.0, 1.0)  # (NN, CB_Huber, LGB_Huber, GRU)


# ---------- Data loading ----------

def load_pred_ensemble(template: str, seeds=SEEDS) -> tuple[np.ndarray, pd.DataFrame]:
    """Load 5-seed mean of pred_dmid_norm from parquets matching template (with {seed})."""
    preds = []
    ref = None
    for s in seeds:
        path = template.format(seed=s)
        df = pd.read_parquet(path, columns=["pred_dmid_norm"])
        preds.append(df["pred_dmid_norm"].to_numpy(dtype=np.float64))
        if ref is None:
            ref = pd.read_parquet(path,
                                  columns=["sym", "date", "session", "t",
                                           "true_dmid_norm", "midprice_t",
                                           "midprice_th"])
    p_mean = np.mean(np.stack(preds), axis=0)
    return p_mean, ref


def build_4way_pred() -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Returns (p_comb, meta_df, sym)."""
    print("[load] T87 NN (5 seeds)...", flush=True)
    p_nn, meta = load_pred_ensemble(
        str(ROOT / "experiments/T87_spo_dfl/pred_T87_seed{seed}_main.parquet"))
    print(f"  p_nn  mean={p_nn.mean():+.3e} std={p_nn.std():.3e}", flush=True)

    print("[load] T99 CB Huber (5 seeds)...", flush=True)
    p_cb, _ = load_pred_ensemble(
        str(ROOT / "experiments/T99_e2e_execution_gbdt/pred_T99_cb_huber_a0.001_seed{seed}.parquet"))
    print(f"  p_cb  mean={p_cb.mean():+.3e} std={p_cb.std():.3e}", flush=True)

    print("[load] T99 LGB Huber (5 seeds)...", flush=True)
    p_lgb, _ = load_pred_ensemble(
        str(ROOT / "experiments/T99_e2e_execution_gbdt/pred_T99_huber_a0.001_seed{seed}.parquet"))
    print(f"  p_lgb mean={p_lgb.mean():+.3e} std={p_lgb.std():.3e}", flush=True)

    print("[load] T95 GRU (5 seeds)...", flush=True)
    p_gru, _ = load_pred_ensemble(
        str(ROOT / "experiments/T95_cnn_rnn_deeplob/pred_T95_gru_w100_C_seed{seed}.parquet"))
    print(f"  p_gru mean={p_gru.mean():+.3e} std={p_gru.std():.3e}", flush=True)

    w_nn, w_cb, w_lgb, w_gru = W
    wsum = sum(W)
    p_comb = (w_nn * p_nn + w_cb * p_cb + w_lgb * p_lgb + w_gru * p_gru) / wsum
    print(f"  p_comb mean={p_comb.mean():+.3e} std={p_comb.std():.3e}", flush=True)

    sym = meta["sym"].to_numpy(dtype=np.int8)
    return p_comb.astype(np.float64), meta, sym


# ---------- PnL ----------

def compute_pnl(actions: np.ndarray, mp_t: np.ndarray, mp_th: np.ndarray) -> np.ndarray:
    """Per-row pnl. actions in {0=short, 1=flat, 2=long}."""
    side = actions.astype(np.float64) - 1.0
    abs_side = np.abs(side)
    diff = mp_th - mp_t
    fee_pnl = FEE * abs_side * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t + 1.0
    return (side * diff - fee_pnl) / denom


def eval_actions(actions: np.ndarray, meta: pd.DataFrame, sym: np.ndarray, label: str) -> dict:
    mp_t = meta["midprice_t"].to_numpy(dtype=np.float64)
    mp_th = meta["midprice_th"].to_numpy(dtype=np.float64)
    pnl = compute_pnl(actions, mp_t, mp_th)
    cum = float(pnl.sum())
    per_sym = []
    for s in (0, 1, 2, 3, 4):
        m = sym == s
        per_sym.append(float(pnl[m].sum()))
    n_active = int((actions != 1).sum())
    n_total = len(actions)
    rate = n_active / n_total
    per_sym_min = min(per_sym)
    return {
        "label": label,
        "loso": cum,
        "per_sym": per_sym,
        "per_sym_min": per_sym_min,
        "n_active": n_active,
        "n_total": n_total,
        "active_rate": rate,
    }


# ---------- Rule 0: baseline DE asymmetric ----------

def baseline_de(p_comb: np.ndarray) -> np.ndarray:
    actions = np.full(len(p_comb), 1, dtype=np.int8)
    actions[p_comb > THR_UP_BASE] = 2
    actions[p_comb < -THR_DN_BASE] = 0
    return actions


# ---------- Rule 1: NSF (Net Spread-aware EV Floor) ----------

def rule_nsf(p_comb: np.ndarray, spread_t: np.ndarray, fee_eff: float, gamma: float) -> np.ndarray:
    thr = fee_eff + gamma * spread_t
    actions = np.full(len(p_comb), 1, dtype=np.int8)
    actions[p_comb > thr] = 2
    actions[p_comb < -thr] = 0
    return actions


# ---------- Rule 2: TBT (Trade-Budget Top-K) ----------

def rule_tbt(p_comb: np.ndarray, active_rate: float) -> np.ndarray:
    K = int(active_rate * len(p_comb))
    actions = np.full(len(p_comb), 1, dtype=np.int8)
    if K <= 0:
        return actions
    # top-K by |pred|
    order = np.argpartition(-np.abs(p_comb), K - 1)[:K]
    actions[order] = np.where(p_comb[order] > 0, 2, 0).astype(np.int8)
    return actions


# ---------- Rule 3: STK (Stochastic-Tier-K Win-Rate Bucketing) ----------

def rule_stk(p_comb: np.ndarray, mp_t: np.ndarray, mp_th: np.ndarray,
             cutoff: float = 0.53, n_buckets: int = 10, n_folds: int = 5,
             rng_seed: int = 0) -> tuple[np.ndarray, dict]:
    """5-fold OOF: bin |pred| into n_buckets quantile buckets within each fold's train-half.
    Compute hit_rate per bucket on train-half, apply to held-out fold.
    A row "fires" if its bucket has hit_rate > cutoff; otherwise flat.
    Direction = sign(pred).
    """
    n = len(p_comb)
    rng = np.random.default_rng(rng_seed)
    folds = rng.integers(0, n_folds, size=n)
    actions = np.full(n, 1, dtype=np.int8)
    diff_full = mp_th - mp_t  # raw diff; we use sign(pred) and check sign(diff)*sign(pred)>0
    abs_pred = np.abs(p_comb)

    fold_info = []
    for f in range(n_folds):
        train_m = folds != f
        test_m = folds == f
        # bucketize using train-half quantiles
        edges = np.quantile(abs_pred[train_m], np.linspace(0, 1, n_buckets + 1))
        edges[0] = -np.inf
        edges[-1] = np.inf
        train_bins = np.digitize(abs_pred[train_m], edges, right=False) - 1
        train_bins = np.clip(train_bins, 0, n_buckets - 1)
        # Per bucket: a "fired" trade is correct if sign(pred) == sign(diff) (and diff != 0)
        signs = np.sign(p_comb[train_m])
        diff_signs = np.sign(diff_full[train_m])
        correct = (signs == diff_signs) & (signs != 0)
        # Conditional hit rate (denominator: rows with non-zero pred sign)
        hit = np.zeros(n_buckets, dtype=np.float64)
        cnt = np.zeros(n_buckets, dtype=np.int64)
        for b in range(n_buckets):
            m = train_bins == b
            if m.sum() > 0:
                cnt[b] = m.sum()
                hit[b] = correct[m].sum() / m.sum()
        # apply to test
        test_bins = np.digitize(abs_pred[test_m], edges, right=False) - 1
        test_bins = np.clip(test_bins, 0, n_buckets - 1)
        fire_mask = hit[test_bins] > cutoff
        local_actions = np.full(test_m.sum(), 1, dtype=np.int8)
        p_test = p_comb[test_m]
        local_actions[fire_mask & (p_test > 0)] = 2
        local_actions[fire_mask & (p_test < 0)] = 0
        actions[test_m] = local_actions
        fold_info.append({"fold": f, "hit_rates": hit.tolist(), "edges": edges.tolist()})

    info = {"cutoff": cutoff, "n_buckets": n_buckets, "n_folds": n_folds, "folds": fold_info}
    return actions, info


# ---------- Rule 4: BAT (Batch-Adaptive Threshold) ----------

def rule_bat(p_comb: np.ndarray, active_rate: float, batch: int = BATCH) -> np.ndarray:
    """Within each non-overlapping batch of size `batch` (last partial batch is its own block):
    q_top = quantile(pred, 1 - active_rate/2)
    q_bot = quantile(pred, active_rate/2)
    Fire long if pred > q_top, short if pred < q_bot.
    """
    n = len(p_comb)
    actions = np.full(n, 1, dtype=np.int8)
    half = active_rate / 2.0
    for s in range(0, n, batch):
        e = min(s + batch, n)
        block = p_comb[s:e]
        if len(block) < 2:
            continue
        q_top = np.quantile(block, 1.0 - half)
        q_bot = np.quantile(block, half)
        a = np.full(len(block), 1, dtype=np.int8)
        a[block > q_top] = 2
        a[block < q_bot] = 0
        actions[s:e] = a
    return actions


# ---------- Main ----------

def main():
    HERE.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    p_comb, meta, sym = build_4way_pred()
    mp_t = meta["midprice_t"].to_numpy(dtype=np.float64)
    mp_th = meta["midprice_th"].to_numpy(dtype=np.float64)
    print(f"[time] load+combine in {time.time()-t0:.1f}s", flush=True)

    # ---- Baseline DE ----
    print("\n=== Baseline (DE asym thr) ===", flush=True)
    a_base = baseline_de(p_comb)
    r_base = eval_actions(a_base, meta, sym, "baseline_de")
    print(f"  loso={r_base['loso']:+.4f}  per_sym={[f'{x:+.4f}' for x in r_base['per_sym']]}", flush=True)
    print(f"  per_sym_min={r_base['per_sym_min']:+.4f}  active_rate={r_base['active_rate']:.4f}", flush=True)
    oof_active_rate = r_base["active_rate"]
    print(f"  -> using active_rate={oof_active_rate:.4f} as budget for TBT/BAT", flush=True)

    # ---- Rule 1: NSF ----
    print("\n=== Rule 1: NSF (Net Spread-aware EV Floor) ===", flush=True)
    # load spread1 (raw spread = ask1 - bid1 pre-normalize) from cache
    schemeP = np.load(ROOT / "experiments/T68_stage5_features/cache/schemeP_test.npz")
    feat_names_path = ROOT / "experiments/T68_stage5_features/cache/schemeP_feat_names.txt"
    with open(feat_names_path) as f:
        all_names = [l.strip() for l in f]
    i_ask1 = all_names.index("ask1")
    i_bid1 = all_names.index("bid1")
    # ask1 and bid1 are already in mid-relative (price/mid - 1) form; their diff is the
    # last-tick spread in mid-relative units, same scale as pred_dmid_norm.
    spread_t = (schemeP["X"][:, i_ask1] - schemeP["X"][:, i_bid1]).astype(np.float64)
    print(f"  spread_t mean={spread_t.mean():.3e} std={spread_t.std():.3e} "
          f"min={spread_t.min():.3e} max={spread_t.max():.3e}", flush=True)
    # verify row order matches preds (schemeP test)
    cache_keys = list(zip(schemeP["sym"].tolist(), schemeP["date"].tolist(),
                          schemeP["sess_idx"].tolist(), schemeP["t"].tolist()))
    sess2idx = {"am": 0, "pm": 1}
    pred_keys = list(zip(meta["sym"].tolist(), meta["date"].tolist(),
                         [sess2idx[s] for s in meta["session"].tolist()],
                         meta["t"].tolist()))
    assert cache_keys == pred_keys, "row order mismatch between schemeP cache and pred parquets"
    print("  schemeP / pred row order: MATCH", flush=True)

    nsf_results = []
    fee_eff_grid = [1e-4, 2e-4, THR_UP_BASE]  # try a few EV floors
    gamma_grid = [0.0, 0.25, 0.5]
    for fe in fee_eff_grid:
        for g in gamma_grid:
            a = rule_nsf(p_comb, spread_t, fe, g)
            r = eval_actions(a, meta, sym, f"nsf_fe{fe:.2e}_g{g:.2f}")
            r["fee_eff"] = fe
            r["gamma"] = g
            nsf_results.append(r)
            print(f"  fe={fe:.2e} g={g:.2f}: loso={r['loso']:+.4f}  "
                  f"per_sym_min={r['per_sym_min']:+.4f}  rate={r['active_rate']:.3f}",
                  flush=True)

    # ---- Rule 2: TBT ----
    print("\n=== Rule 2: TBT (Top-K by |pred|) ===", flush=True)
    a_tbt = rule_tbt(p_comb, oof_active_rate)
    r_tbt = eval_actions(a_tbt, meta, sym, "tbt")
    print(f"  loso={r_tbt['loso']:+.4f}  per_sym={[f'{x:+.4f}' for x in r_tbt['per_sym']]}", flush=True)
    print(f"  per_sym_min={r_tbt['per_sym_min']:+.4f}  rate={r_tbt['active_rate']:.4f}", flush=True)

    # ---- Rule 3: STK ----
    print("\n=== Rule 3: STK (5-fold win-rate bucketing) ===", flush=True)
    stk_results = []
    for cutoff in [0.51, 0.52, 0.53, 0.54]:
        a, info = rule_stk(p_comb, mp_t, mp_th, cutoff=cutoff, n_buckets=10, n_folds=5,
                           rng_seed=42)
        r = eval_actions(a, meta, sym, f"stk_c{cutoff:.2f}")
        r["cutoff"] = cutoff
        # save fold-0 hit rates as diagnostic
        r["fold0_hit_rates"] = info["folds"][0]["hit_rates"]
        stk_results.append(r)
        print(f"  cutoff={cutoff}: loso={r['loso']:+.4f}  per_sym_min={r['per_sym_min']:+.4f}  "
              f"rate={r['active_rate']:.4f}", flush=True)

    # ---- Rule 4: BAT ----
    print("\n=== Rule 4: BAT (within-batch quantile) ===", flush=True)
    bat_results = []
    for batch in [1024, 4096, 16384]:
        a = rule_bat(p_comb, oof_active_rate, batch=batch)
        r = eval_actions(a, meta, sym, f"bat_b{batch}")
        r["batch"] = batch
        bat_results.append(r)
        print(f"  batch={batch}: loso={r['loso']:+.4f}  per_sym_min={r['per_sym_min']:+.4f}  "
              f"rate={r['active_rate']:.4f}", flush=True)

    # ---- pick best per rule by loso ----
    nsf_best = max(nsf_results, key=lambda r: r["loso"])
    stk_best = max(stk_results, key=lambda r: r["loso"])
    bat_best = max(bat_results, key=lambda r: r["loso"])

    summary = {
        "task": "T118 decision_4in1: post-hoc decision rule comparison",
        "ensemble": "iter_016 v3 (T87 + 0.7*CB_Huber + 1.0*LGB_Huber + 1.0*GRU)",
        "weights_nn_cb_lgb_gru": list(W),
        "n_total": int(len(p_comb)),
        "baseline_de": r_base,
        "nsf_results": nsf_results,
        "tbt_result": r_tbt,
        "stk_results": stk_results,
        "bat_results": bat_results,
        "best": {
            "nsf": nsf_best,
            "tbt": r_tbt,
            "stk": stk_best,
            "bat": bat_best,
        },
    }
    out_path = HERE / "results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote -> {out_path}", flush=True)

    # ---- compact summary ----
    print("\n========== SUMMARY ==========", flush=True)
    rows = [
        ("baseline_de",     r_base),
        ("nsf_best",        nsf_best),
        ("tbt",             r_tbt),
        ("stk_best",        stk_best),
        ("bat_best",        bat_best),
    ]
    print(f"{'rule':16s} {'loso':>9s} {'pmin':>9s} {'rate':>7s}  per_sym", flush=True)
    for name, r in rows:
        ps = "  ".join(f"{x:+.3f}" for x in r["per_sym"])
        print(f"{name:16s} {r['loso']:+9.4f} {r['per_sym_min']:+9.4f} {r['active_rate']:7.3f}  {ps}",
              flush=True)


if __name__ == "__main__":
    main()
