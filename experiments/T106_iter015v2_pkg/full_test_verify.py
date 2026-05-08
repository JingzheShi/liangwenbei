"""Verify iter_015 v2 ensemble cum_pnl on the full 442k test using cached features.

Loads:
  - schemeP_test.npz cached features (442k x 370) for T87 NN, T89 CB, T99 LGB Huber
  - lob_test_top5.npz cached raw LOB windows (442k x 100 x 20) for T95 GRU
  - 5x SPO+ NN .npz   (T87)
  - 5x CatBoost .cbm  (T89)
  - 5x LightGBM .txt  (T99 Huber alpha=0.001)
  - 5x GRU .npz       (T95 numpy weights)

Combines preds with w=1:1.5:0.7:0.7 and applies the EV gate from thresholds.json
(thr_up=0.000287, thr_dn=0.000207). Computes per-sym cum_pnl.

Expected: matches +43.2270 LOSO-equiv from ev_lgb_huber_ensemble.json.
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
T95_CACHE = os.path.join(ROOT, "experiments", "T95_cnn_rnn_deeplob", "cache")

sys.path.insert(0, HERE)
from Predictor import _MLPNumpy, _GRUNumpy

FEE = 0.0001
SEEDS = (1, 7, 13, 42, 100)


def main():
    print("=== iter_015 v2 full-test verification (4-way ensemble) ===", flush=True)

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

    # Load LOB test cache for GRU
    t0 = time.time()
    lob = np.load(os.path.join(T95_CACHE, "lob_test_top5.npz"))
    lob_arr = lob["lob_arr"]
    sess_id_for_sample = lob["sess_id_for_sample"]
    t_for_sample = lob["t_for_sample"]
    sym_lob = lob["sym"]
    mp_t_lob = lob["mp_t"].astype(np.float64)
    mp_t60_lob = lob["mp_t60"].astype(np.float64)
    print(f"  loaded lob_test_top5 lob_arr={lob_arr.shape} samples={len(sym_lob)} in {time.time()-t0:.1f}s", flush=True)

    if not (np.array_equal(sym, sym_lob) and np.allclose(mp_t, mp_t_lob, rtol=1e-5)):
        print("  WARNING: schemeP and LOB caches have different row order; need to reindex.", flush=True)
        # Build mapping by (sym, date, sess, t)
        d_key = list(zip(d["sym"].tolist(), d["date"].tolist(), d["sess_idx"].tolist(), d["t"].tolist()))
        l_key = list(zip(sym_lob.tolist(), lob["date"].tolist(), lob["sess_idx"].tolist(), t_for_sample.tolist()))
        mp = {k: i for i, k in enumerate(l_key)}
        order_lob = np.array([mp[k] for k in d_key], dtype=np.int64)
    else:
        order_lob = None

    # ----- T87 SPO+ NN ensemble -----
    t0 = time.time()
    nn_models = [_MLPNumpy(os.path.join(HERE, f"nn_h60_seed{s}.npz")) for s in SEEDS]
    p_nn = np.zeros(len(X), dtype=np.float64)
    BATCH = 100_000
    for nn in nn_models:
        for s in range(0, len(X), BATCH):
            p_nn[s:s+BATCH] += nn.predict(X_kept[s:s+BATCH]).astype(np.float64)
    p_nn /= len(nn_models)
    print(f"  T87 NN inference in {time.time()-t0:.1f}s  mean={p_nn.mean():+.3e} std={p_nn.std():.3e}", flush=True)

    # ----- T89 CatBoost ensemble -----
    t0 = time.time()
    cb_models = []
    for s in SEEDS:
        cb = CatBoostRegressor()
        cb.load_model(os.path.join(HERE, f"model_T89_seed{s}.cbm"))
        cb_models.append(cb)
    p_cb = np.zeros(len(X), dtype=np.float64)
    for cb in cb_models:
        for s in range(0, len(X), BATCH):
            p_cb[s:s+BATCH] += cb.predict(X_kept[s:s+BATCH])
    p_cb /= len(cb_models)
    print(f"  T89 CB inference in {time.time()-t0:.1f}s  mean={p_cb.mean():+.3e} std={p_cb.std():.3e}", flush=True)

    # ----- T99 LGB Huber ensemble -----
    t0 = time.time()
    huber_models = [lgb.Booster(model_file=os.path.join(HERE, f"model_T99_huber_a0.001_seed{s}.txt")) for s in SEEDS]
    p_huber = np.zeros(len(X), dtype=np.float64)
    for m in huber_models:
        for s in range(0, len(X), BATCH):
            p_huber[s:s+BATCH] += m.predict(X_kept[s:s+BATCH])
    p_huber /= len(huber_models)
    print(f"  T99 LGB Huber inference in {time.time()-t0:.1f}s  mean={p_huber.mean():+.3e} std={p_huber.std():.3e}", flush=True)

    # ----- T95 GRU ensemble -----
    t0 = time.time()
    gru_models = [_GRUNumpy(os.path.join(HERE, f"gru_h60_seed{s}.npz")) for s in SEEDS]
    n = len(sym_lob)
    p_gru = np.zeros(n, dtype=np.float64)
    GRU_BATCH = 8192
    # Build per-sample raw window from lob_arr indexed by (sess_id, t-99..t)
    # Each window is lob_arr[sess_id, t-99:t+1, :]  (W=100, F=20)
    for start in range(0, n, GRU_BATCH):
        end = min(start + GRU_BATCH, n)
        sids = sess_id_for_sample[start:end]
        ts = t_for_sample[start:end].astype(np.int32)
        # Build (B, W=100, F=20) using fancy indexing
        idx_w = ts[:, None] + np.arange(-99, 1, dtype=np.int32)[None, :]   # (B, 100)
        x_win = lob_arr[sids[:, None], idx_w, :]  # (B, 100, 20)
        for g in gru_models:
            p_gru[start:end] += g.predict(x_win).astype(np.float64)
    p_gru /= len(gru_models)
    print(f"  T95 GRU inference in {time.time()-t0:.1f}s  mean={p_gru.mean():+.3e} std={p_gru.std():.3e}", flush=True)

    # Reorder GRU preds to schemeP row order if needed
    if order_lob is not None:
        p_gru = p_gru[order_lob]

    # ----- Combine 4-way -----
    w_nn, w_cb, w_huber, w_gru = 1.0, 1.5, 0.7, 0.7
    wsum = w_nn + w_cb + w_huber + w_gru
    p_comb = (w_nn * p_nn + w_cb * p_cb + w_huber * p_huber + w_gru * p_gru) / wsum
    p_comb = p_comb.astype(np.float32)

    # Load thresholds from JSON
    with open(os.path.join(HERE, "thresholds.json")) as f:
        tcfg = json.load(f)
    thr_up = float(tcfg["horizons"][0]["thr_up"])
    thr_dn = float(tcfg["horizons"][0]["thr_dn"])

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
    print(f"  thr_up={thr_up:.6f}  thr_dn={thr_dn:.6f}", flush=True)
    print(f"  w=({w_nn},{w_cb},{w_huber},{w_gru})", flush=True)
    print(f"  per-sym:", flush=True)
    per_sym = []
    for s_id in (0, 1, 2, 3, 4):
        m = sym == s_id
        c = float(pnl[m].sum())
        per_sym.append(c)
        print(f"    sym={s_id}: cum_pnl={c:+.4f}  n_active={(actions[m]!=1).sum():,}/{int(m.sum()):,}", flush=True)
    sum_per_sym = float(sum(per_sym))
    print(f"\n  sum_per_sym (LOSO-equiv) = {sum_per_sym:+.4f}", flush=True)
    print(f"  vs offline target +43.2270: diff={sum_per_sym-43.2270:+.6f}", flush=True)
    print(f"  vs iter_015 packaged +40.13: diff={sum_per_sym-40.13:+.4f}", flush=True)
    print(f"  vs iter_014 +38.28: diff={sum_per_sym-38.28:+.4f}", flush=True)

    out = {
        "iter": "iter_015_v2",
        "cum_pnl_total": cum_pnl_total,
        "sum_per_sym": sum_per_sym,
        "per_sym": per_sym,
        "n_active": n_active,
        "thr_up": thr_up, "thr_dn": thr_dn,
        "weights": {"nn": w_nn, "cb": w_cb, "huber": w_huber, "gru": w_gru},
        "p_nn_stats": {"mean": float(p_nn.mean()), "std": float(p_nn.std())},
        "p_cb_stats": {"mean": float(p_cb.mean()), "std": float(p_cb.std())},
        "p_huber_stats": {"mean": float(p_huber.mean()), "std": float(p_huber.std())},
        "p_gru_stats": {"mean": float(p_gru.mean()), "std": float(p_gru.std())},
        "p_comb_stats": {"mean": float(p_comb.mean()), "std": float(p_comb.std())},
        "vs_iter015_v1": sum_per_sym - 40.13,
        "vs_offline_target": sum_per_sym - 43.2270,
    }
    with open(os.path.join(HERE, "full_test_verify.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote -> full_test_verify.json", flush=True)


if __name__ == "__main__":
    main()
