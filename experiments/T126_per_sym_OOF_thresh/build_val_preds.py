"""T126: Build val OOF preds (date 76-79) for iter_015 v1 stack.

Stack = 5xT87 NN + 5xT75 LGB at 1.0:1.5 weight (matches iter_015 v1).

Outputs:
  val_preds_iter015v1.parquet   — stacked pred on val 73,680 rows
  test_preds_iter015v1.parquet  — stacked pred on test 442,080 rows (re-saved for convenience)

Both contain: sym, date, sess_idx, t, mp_t, mp_th, pred, true_dmid_norm
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

T68_CACHE = os.path.join(ROOT, "experiments", "T68_stage5_features", "cache")
T75_DIR = os.path.join(ROOT, "experiments", "T75_regression_dmid")
T87_DIR = os.path.join(ROOT, "experiments", "T87_spo_dfl")

SEEDS = (1, 7, 13, 42, 100)
W_NN = 1.0
W_LGB = 1.5

T59_FAIL = [
    "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
    "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
    "qrank_W100_cumspread",
    "kyle_lam_W50", "kyle_lam_W100",
    "roll_eff_spr_ratio_W100",
]
STAGE5_FAIL = ["liq_asym_top5_W5"]
DROP = T59_FAIL + STAGE5_FAIL


def gelu_tanh(x):
    c = 0.7978845608028654
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x * x * x)))


def layernorm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


class MLPNumpy:
    def __init__(self, npz_path):
        d = np.load(npz_path, allow_pickle=False)
        self.in_dim = int(d["in_dim"][0])
        self.hidden = list(d["hidden"].tolist())
        self.use_layernorm = bool(d["use_layernorm"][0])
        self.target_scale = float(d["target_scale"][0])
        self.feat_mean = d["feat_mean"].astype(np.float32)
        self.feat_std = d["feat_std"].astype(np.float32)
        self.keep_idx = d["keep_idx"].astype(np.int64)
        self.clip = float(d["clip"][0])
        self.W, self.b, self.LN_W, self.LN_b = [], [], [], []
        for i in range(len(self.hidden)):
            self.W.append(d[f"L{i}_W"].astype(np.float32))
            self.b.append(d[f"L{i}_b"].astype(np.float32))
            if self.use_layernorm:
                self.LN_W.append(d[f"LN{i}_W"].astype(np.float32))
                self.LN_b.append(d[f"LN{i}_b"].astype(np.float32))
        self.WF = d["LF_W"].astype(np.float32)
        self.bF = d["LF_b"].astype(np.float32)

    def predict(self, X_in):
        # X_in is already sliced to in_dim, but we re-apply keep_idx if shape mismatches
        Xs = X_in
        if Xs.shape[1] != self.in_dim:
            Xs = Xs[:, self.keep_idx]
        Xs = Xs.astype(np.float32, copy=True)
        Xs = (Xs - self.feat_mean) / np.maximum(self.feat_std, 1e-6)
        Xs = np.where(np.isnan(Xs), 0.0, Xs)
        Xs = np.clip(Xs, -self.clip, self.clip).astype(np.float32, copy=False)
        h = Xs
        for i in range(len(self.hidden)):
            h = h @ self.W[i].T + self.b[i]
            if self.use_layernorm:
                h = layernorm(h, self.LN_W[i], self.LN_b[i])
            h = gelu_tanh(h)
        out = h @ self.WF.T + self.bF
        out = out.squeeze(-1) / self.target_scale
        return out.astype(np.float32, copy=False)


def predict_chunked_lgb(booster, X, batch=200_000):
    chunks = []
    for s in range(0, len(X), batch):
        chunks.append(booster.predict(X[s:s+batch]).astype(np.float32))
    return np.concatenate(chunks, axis=0)


def main():
    # Load feat names + drop indices
    with open(os.path.join(T68_CACHE, "schemeP_feat_names.txt")) as f:
        all_names = [l.strip() for l in f]
    name_to_i = {n: i for i, n in enumerate(all_names)}
    drop_idx_set = {name_to_i[n] for n in DROP}
    keep_idx = np.array([i for i in range(len(all_names)) if i not in drop_idx_set],
                        dtype=np.int64)
    assert len(keep_idx) == 359, len(keep_idx)
    print(f"keep_idx (T75/T87 share same drop set): n={len(keep_idx)}", flush=True)

    # Load train cache, filter date 76-79
    print("loading schemeP_train.npz...", flush=True)
    t0 = time.time()
    d = np.load(os.path.join(T68_CACHE, "schemeP_train.npz"))
    m_va = (d["date"] >= 76) & (d["date"] <= 79)
    print(f"  loaded in {time.time()-t0:.1f}s, val rows={int(m_va.sum())}", flush=True)
    X_va_full = d["X"][m_va]               # (Nv, 370)
    sym_va = d["sym"][m_va].astype(np.int64)
    date_va = d["date"][m_va].astype(np.int64)
    sess_va = d["sess_idx"][m_va].astype(np.int64)
    t_va = d["t"][m_va].astype(np.int64)
    mp_t_va = d["mp_t"][m_va].astype(np.float64)
    mp_th_va = d["mp_t60"][m_va].astype(np.float64)
    y60_va = d["y60"][m_va].astype(np.int64)
    true_dmid_norm = ((mp_th_va - mp_t_va) / (mp_t_va + 1.0)).astype(np.float32)
    del d

    X_va_kept = X_va_full[:, keep_idx].astype(np.float32, copy=False)
    print(f"  X_va_kept shape={X_va_kept.shape}", flush=True)

    # NN inference (5 seeds, average)
    print("\n=== NN inference (T87, 5 seeds) ===", flush=True)
    nn_pred_va = np.zeros(len(X_va_kept), dtype=np.float64)
    for s in SEEDS:
        npz = os.path.join(T87_DIR, f"model_T87_seed{s}_main.npz")
        m = MLPNumpy(npz)
        # m.keep_idx is the same as our keep_idx; pass full or sliced — it auto-handles
        p = m.predict(X_va_full[:, m.keep_idx])
        nn_pred_va += p.astype(np.float64)
        print(f"  seed={s}: pred mean={p.mean():+.6e} std={p.std():.6e}", flush=True)
    nn_pred_va /= len(SEEDS)
    print(f"NN avg: mean={nn_pred_va.mean():+.6e} std={nn_pred_va.std():.6e}", flush=True)

    # LGB inference (5 seeds, average)
    print("\n=== LGB inference (T75, 5 seeds) ===", flush=True)
    lgb_pred_va = np.zeros(len(X_va_kept), dtype=np.float64)
    for s in SEEDS:
        path = os.path.join(T75_DIR, f"model_T75_seed{s}.txt")
        b = lgb.Booster(model_file=path)
        p = predict_chunked_lgb(b, X_va_kept)
        lgb_pred_va += p.astype(np.float64)
        print(f"  seed={s}: pred mean={p.mean():+.6e} std={p.std():.6e}", flush=True)
    lgb_pred_va /= len(SEEDS)
    print(f"LGB avg: mean={lgb_pred_va.mean():+.6e} std={lgb_pred_va.std():.6e}", flush=True)

    # Stack
    pred_va = (W_NN * nn_pred_va + W_LGB * lgb_pred_va) / (W_NN + W_LGB)
    print(f"\nStack (1.0:1.5) val: mean={pred_va.mean():+.6e} std={pred_va.std():.6e}",
          flush=True)

    df_va = pd.DataFrame({
        "sym": sym_va.astype(np.int8),
        "date": date_va.astype(np.int16),
        "sess_idx": sess_va.astype(np.int8),
        "t": t_va.astype(np.int32),
        "midprice_t": mp_t_va.astype(np.float64),
        "midprice_th": mp_th_va.astype(np.float64),
        "true_dmid_norm": true_dmid_norm,
        "true_label": y60_va.astype(np.int8),
        "pred_dmid_norm": pred_va.astype(np.float32),
        "pred_nn": nn_pred_va.astype(np.float32),
        "pred_lgb": lgb_pred_va.astype(np.float32),
    })
    out_va = os.path.join(HERE, "val_preds_iter015v1.parquet")
    df_va.to_parquet(out_va, index=False)
    print(f"saved {out_va} shape={df_va.shape}", flush=True)

    # Test side: just average existing seed parquets and stack
    print("\n=== Loading existing test preds + stacking ===", flush=True)
    nn_test = None
    base_test = None
    for s in SEEDS:
        df = pd.read_parquet(os.path.join(T87_DIR, f"pred_T87_seed{s}_main.parquet"))
        if nn_test is None:
            nn_test = np.zeros(len(df), dtype=np.float64)
            base_test = df[["sym", "date", "session", "t", "midprice_t", "midprice_th",
                            "true_dmid_norm", "true_label"]].copy()
        nn_test += df["pred_dmid_norm"].to_numpy(np.float64)
    nn_test /= len(SEEDS)
    lgb_test = np.zeros(len(base_test), dtype=np.float64)
    for s in SEEDS:
        df = pd.read_parquet(os.path.join(T75_DIR, f"pred_T75_seed{s}.parquet"))
        lgb_test += df["pred_dmid_norm"].to_numpy(np.float64)
    lgb_test /= len(SEEDS)
    pred_te = (W_NN * nn_test + W_LGB * lgb_test) / (W_NN + W_LGB)
    base_test["pred_dmid_norm"] = pred_te.astype(np.float32)
    base_test["pred_nn"] = nn_test.astype(np.float32)
    base_test["pred_lgb"] = lgb_test.astype(np.float32)
    out_te = os.path.join(HERE, "test_preds_iter015v1.parquet")
    base_test.to_parquet(out_te, index=False)
    print(f"saved {out_te} shape={base_test.shape}", flush=True)

    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
