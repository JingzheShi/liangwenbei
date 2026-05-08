"""Verify iter_016 v3 (T109) full-442k cum_pnl on the cached test set.

Reuses the cached schemeP_test (442k x 370) + lob_test_top5 (442k x 100 x 20)
caches built in T68 / T95. Loads:

  - 5x T87 SPO+ NN ensemble (nn_h60_seed*.npz)
  - 5x T99 CB Huber alpha=1e-3 ensemble (model_cb_huber_seed*.cbm)  [swapped]
  - 5x T99 LGB Huber alpha=1e-3 ensemble (model_T99_huber_a0.001_seed*.txt)
  - 5x T95 GRU ensemble (gru_h60_seed*.npz)  -- torch GPU when available

Combines with weights w=(1.0, 0.7, 1.0, 1.0) for (NN, CB_Huber, LGB_Huber, GRU)
and applies DE-asym thresholds thr_up=2.883e-4 / thr_dn=2.186e-4. Target
cum_pnl ≈ +44.72.
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
PKG = os.path.join(HERE, "pkg")

sys.path.insert(0, PKG)
from Predictor import _MLPNumpy, _GRUEnsembleTorch, _GRUNumpy

FEE = 0.0001
SEEDS = (1, 7, 13, 42, 100)


def main():
    print("=== iter_016 v3 (T109) full-test verification ===", flush=True)

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

    t0 = time.time()
    lob = np.load(os.path.join(T95_CACHE, "lob_test_top5.npz"))
    lob_arr = lob["lob_arr"]
    sess_id_for_sample = lob["sess_id_for_sample"]
    t_for_sample = lob["t_for_sample"]
    sym_lob = lob["sym"]
    mp_t_lob = lob["mp_t"].astype(np.float64)
    print(f"  loaded lob_test_top5 lob_arr={lob_arr.shape} samples={len(sym_lob)} in {time.time()-t0:.1f}s", flush=True)

    if not (np.array_equal(sym, sym_lob) and np.allclose(mp_t, mp_t_lob, rtol=1e-5)):
        print("  WARNING: schemeP and LOB caches have different row order; reindexing.", flush=True)
        d_key = list(zip(d["sym"].tolist(), d["date"].tolist(), d["sess_idx"].tolist(), d["t"].tolist()))
        l_key = list(zip(sym_lob.tolist(), lob["date"].tolist(), lob["sess_idx"].tolist(), t_for_sample.tolist()))
        mp = {k: i for i, k in enumerate(l_key)}
        order_lob = np.array([mp[k] for k in d_key], dtype=np.int64)
    else:
        order_lob = None

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

    # ---- T99 CB Huber ensemble (replaces T89 CB RMSE) ----
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

    # ---- T99 LGB Huber ensemble ----
    t0 = time.time()
    huber_models = [lgb.Booster(model_file=os.path.join(PKG, f"model_T99_huber_a0.001_seed{s}.txt")) for s in SEEDS]
    p_huber = np.zeros(len(X), dtype=np.float64)
    for m in huber_models:
        for s in range(0, len(X), BATCH):
            p_huber[s:s+BATCH] += m.predict(X_kept[s:s+BATCH])
    p_huber /= len(huber_models)
    t_huber = time.time() - t0
    print(f"  T99 LGB Huber inference in {t_huber:.1f}s  mean={p_huber.mean():+.3e} std={p_huber.std():.3e}", flush=True)

    # ---- T95 GRU ensemble (torch GPU) ----
    t0 = time.time()
    gru_paths = [os.path.join(PKG, f"gru_h60_seed{s}.npz") for s in SEEDS]
    try:
        import torch
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device_str = None

    if device_str is not None:
        gru_ens = _GRUEnsembleTorch(gru_paths, device_str)
        backend = f"torch_{device_str}"
    else:
        gru_ens = None
        backend = "numpy_fallback"

    n = len(sym_lob)
    p_gru = np.zeros(n, dtype=np.float64)
    GRU_BATCH = 16384
    if gru_ens is not None:
        warm = lob_arr[sess_id_for_sample[:128, None],
                       t_for_sample[:128, None].astype(np.int32) + np.arange(-99, 1, dtype=np.int32)[None, :],
                       :]
        _ = gru_ens.predict_mean(warm)
        if device_str == "cuda":
            import torch
            torch.cuda.synchronize()
        t0 = time.time()
        for start in range(0, n, GRU_BATCH):
            end = min(start + GRU_BATCH, n)
            sids = sess_id_for_sample[start:end]
            ts = t_for_sample[start:end].astype(np.int32)
            idx_w = ts[:, None] + np.arange(-99, 1, dtype=np.int32)[None, :]
            x_win = lob_arr[sids[:, None], idx_w, :]
            p_gru[start:end] = gru_ens.predict_mean(x_win).astype(np.float64)
        if device_str == "cuda":
            import torch
            torch.cuda.synchronize()
    else:
        gru_models = [_GRUNumpy(p) for p in gru_paths]
        for start in range(0, n, GRU_BATCH):
            end = min(start + GRU_BATCH, n)
            sids = sess_id_for_sample[start:end]
            ts = t_for_sample[start:end].astype(np.int32)
            idx_w = ts[:, None] + np.arange(-99, 1, dtype=np.int32)[None, :]
            x_win = lob_arr[sids[:, None], idx_w, :]
            for g in gru_models:
                p_gru[start:end] += g.predict(x_win).astype(np.float64)
            p_gru[start:end] /= len(gru_models)
    t_gru = time.time() - t0
    print(f"  T95 GRU inference [{backend}] in {t_gru:.1f}s  mean={p_gru.mean():+.3e} std={p_gru.std():.3e}", flush=True)

    if order_lob is not None:
        p_gru = p_gru[order_lob]

    # ---- Combine 4-way (T109 weights) ----
    w_nn, w_cb, w_huber, w_gru = 1.0, 0.7, 1.0, 1.0
    wsum = w_nn + w_cb + w_huber + w_gru
    p_comb = (w_nn * p_nn + w_cb * p_cb + w_huber * p_huber + w_gru * p_gru) / wsum
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
    print(f"  w=({w_nn},{w_cb},{w_huber},{w_gru})", flush=True)
    print(f"  per-sym:", flush=True)
    per_sym = []
    for s_id in (0, 1, 2, 3, 4):
        m = sym == s_id
        c = float(pnl[m].sum())
        per_sym.append(c)
        print(f"    sym={s_id}: cum_pnl={c:+.4f}  n_active={(actions[m]!=1).sum():,}/{int(m.sum()):,}", flush=True)
    sum_per_sym = float(sum(per_sym))
    target_pnl = 44.7178
    target_per_sym = [4.685227975340709, 4.456103396463256, 4.543665346189459,
                      12.993312658105966, 18.039483023807605]
    print(f"\n  sum_per_sym (LOSO-equiv) = {sum_per_sym:+.4f}", flush=True)
    print(f"  vs T99 winner refine target {target_pnl:+.4f}: diff={sum_per_sym-target_pnl:+.6f}", flush=True)

    out = {
        "iter": "iter_016_v3 (T109)",
        "gru_backend": backend,
        "cum_pnl_total": cum_pnl_total,
        "sum_per_sym": sum_per_sym,
        "per_sym": per_sym,
        "per_sym_target": target_per_sym,
        "n_active": n_active,
        "thr_up": thr_up, "thr_dn": thr_dn,
        "weights": {"nn": w_nn, "cb_huber": w_cb, "lgb_huber": w_huber, "gru": w_gru},
        "timing_seconds": {
            "T87_nn": t_nn,
            "T99_cb_huber": t_cb,
            "T99_lgb_huber": t_huber,
            "T95_gru": t_gru,
            "total_inference": t_nn + t_cb + t_huber + t_gru,
        },
        "p_nn_stats": {"mean": float(p_nn.mean()), "std": float(p_nn.std())},
        "p_cb_huber_stats": {"mean": float(p_cb.mean()), "std": float(p_cb.std())},
        "p_lgb_huber_stats": {"mean": float(p_huber.mean()), "std": float(p_huber.std())},
        "p_gru_stats": {"mean": float(p_gru.mean()), "std": float(p_gru.std())},
        "p_comb_stats": {"mean": float(p_comb.mean()), "std": float(p_comb.std())},
        "vs_T99_winner_refine": sum_per_sym - target_pnl,
        "vs_T108_baseline": sum_per_sym - 43.20016,
    }
    out_path = os.path.join(HERE, "full_test_verify.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
