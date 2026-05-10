"""T172 evaluation: Transformer-only LOSO PnL + ensemble with v2 LGB/MLP on holdout.

Loads:
  - v2 pkg's fast_features_batch.py for feature computation
  - v2's 5 LGB models + 5 T87 MLP npz (pred_lgb, pred_nn)
  - T172's 5 GroupTransformer npz (pred_tr)

Evaluates on schemeP_test (dates 80-119) which is NOT used in training.

Reports:
  - v2-only (nn + lgb): baseline
  - Transformer-only: standalone
  - v2 + Transformer ensemble at various weights
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
V2_PKG = os.path.join(ROOT, "experiments", "R_full_retrain", "pkg_iter019_v2")
CACHE_DIR = os.path.join(ROOT, "experiments", "T68_stage5_features", "cache")
TR_NPZ_DIR = os.environ.get("TR_NPZ_DIR", HERE)  # where nn_tr_h60_seed*.npz live

sys.path.insert(0, V2_PKG)

T59_FAIL = ["dualz_ask_diff1","dualz_bid_diff5","dualz_ask_diff5",
            "qrank_W100_spread1","qrank_W100_spread5","qrank_W100_spread10",
            "qrank_W100_cumspread","kyle_lam_W50","kyle_lam_W100",
            "roll_eff_spr_ratio_W100"]
S5_FAIL = ["liq_asym_top5_W5"]
DROP_NAMES = T59_FAIL + S5_FAIL
FEE = 0.0001
SEEDS = [1, 7, 13, 42, 100]
HORIZON = 60


def gelu(x):
    c = 0.7978845608028654
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x * x * x)))


def layernorm(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


def mha(x, in_w, in_b, out_w, out_b, n_heads):
    B, L, d = x.shape
    dh = d // n_heads
    qkv = (x.reshape(B * L, d) @ in_w.T + in_b).reshape(B, L, 3, d)
    Q, K, V = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]
    Q = Q.reshape(B, L, n_heads, dh).transpose(0,2,1,3)
    K = K.reshape(B, L, n_heads, dh).transpose(0,2,1,3)
    V = V.reshape(B, L, n_heads, dh).transpose(0,2,1,3)
    sc = np.float32(1.0 / np.sqrt(dh))
    sc_ = (Q @ K.transpose(0,1,3,2)) * sc
    sc_ -= sc_.max(-1, keepdims=True)
    a = np.exp(sc_); a /= a.sum(-1, keepdims=True)
    out = (a @ V).transpose(0,2,1,3).reshape(B*L, d)
    return (out @ out_w.T + out_b).reshape(B, L, d)


class GroupTransformerNumpy:
    def __init__(self, npz_path):
        d = np.load(npz_path, allow_pickle=True)
        self.feat_dim = int(d["n_features"][0])
        self.n_tokens = int(d["n_tokens"][0])
        self.d_token = int(d["d_token"][0])
        self.n_heads = int(d["n_heads"][0])
        self.n_layers = int(d["n_layers"][0])
        self.target_scale = float(d["target_scale"][0])
        self.clip = float(d["clip"][0])
        self.feat_mean = d["feat_mean"].astype(np.float32)
        self.feat_std = d["feat_std"].astype(np.float32)
        self.keep_idx = d["keep_idx"].astype(np.int64)
        self.proj_w = d["proj_w"].astype(np.float32)
        self.proj_b = d["proj_b"].astype(np.float32)
        self.proj_norm_w = d["proj_norm_w"].astype(np.float32)
        self.proj_norm_b = d["proj_norm_b"].astype(np.float32)
        self.cls_token = d["cls_token"].astype(np.float32)
        self.head_ln_w = d["head_ln_w"].astype(np.float32)
        self.head_ln_b = d["head_ln_b"].astype(np.float32)
        self.head_fc1_w = d["head_fc1_w"].astype(np.float32)
        self.head_fc1_b = d["head_fc1_b"].astype(np.float32)
        self.head_fc2_w = d["head_fc2_w"].astype(np.float32)
        self.head_fc2_b = d["head_fc2_b"].astype(np.float32)
        self.layers = []
        for i in range(self.n_layers):
            self.layers.append({k: d[f"L{i}_{k}"].astype(np.float32)
                                 for k in ["norm1_w","norm1_b","norm2_w","norm2_b",
                                           "attn_in_proj_w","attn_in_proj_b",
                                           "attn_out_w","attn_out_b",
                                           "ffn1_w","ffn1_b","ffn2_w","ffn2_b"]})

    def predict(self, X_in):
        Xs = X_in.astype(np.float32, copy=False)
        if Xs.shape[1] != self.feat_dim:
            Xs = Xs[:, self.keep_idx]
        Xs = np.clip(np.where(np.isnan((Xs - self.feat_mean) / self.feat_std), 0.0,
                               (Xs - self.feat_mean) / self.feat_std), -self.clip, self.clip).astype(np.float32)
        B = Xs.shape[0]
        tokens = (Xs @ self.proj_w.T + self.proj_b).reshape(B, self.n_tokens, self.d_token)
        tokens = layernorm(tokens, self.proj_norm_w, self.proj_norm_b)
        cls = np.broadcast_to(self.cls_token, (B, 1, self.d_token)).copy().astype(np.float32)
        seq = np.concatenate([cls, tokens], axis=1)
        for L in self.layers:
            n = layernorm(seq, L["norm1_w"], L["norm1_b"])
            seq = seq + mha(n, L["attn_in_proj_w"], L["attn_in_proj_b"],
                            L["attn_out_w"], L["attn_out_b"], self.n_heads)
            n = layernorm(seq, L["norm2_w"], L["norm2_b"])
            f = gelu(n.reshape(-1, self.d_token) @ L["ffn1_w"].T + L["ffn1_b"])
            seq = seq + (f @ L["ffn2_w"].T + L["ffn2_b"]).reshape(B, -1, self.d_token)
        cls_out = layernorm(seq[:, 0], self.head_ln_w, self.head_ln_b)
        cls_out = gelu(cls_out @ self.head_fc1_w.T + self.head_fc1_b)
        return ((cls_out @ self.head_fc2_w.T + self.head_fc2_b).squeeze(-1) / self.target_scale).astype(np.float32)


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
        Xs = X_in
        if Xs.shape[1] != self.in_dim:
            Xs = Xs[:, self.keep_idx]
        Xs = Xs.astype(np.float32, copy=True)
        Xs = np.clip(np.where(np.isnan((Xs - self.feat_mean) / self.feat_std), 0.0,
                               (Xs - self.feat_mean) / self.feat_std), -self.clip, self.clip).astype(np.float32)
        h = Xs
        for i in range(len(self.hidden)):
            h = h @ self.W[i].T + self.b[i]
            if self.use_layernorm:
                h = layernorm(h, self.LN_W[i], self.LN_b[i])
            h = gelu(h)
        return (h @ self.WF.T + self.bF).squeeze(-1) / self.target_scale


def compute_pnl(pred, mp_t, mp_th, thr_up, thr_dn):
    side = np.where(pred > thr_up, 1.0, np.where(pred < -thr_dn, -1.0, 0.0))
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * (mp_th.astype(np.float64) + 1.0 + mp_t.astype(np.float64) + 1.0)
    pnl = (side * diff - fee_pnl) / (mp_t.astype(np.float64) + 1.0)
    return float(pnl.sum()), int((side != 0).sum())


def main():
    import lightgbm as lgb
    print("Loading schemeP test data...", flush=True)
    t0 = time.time()
    d = np.load(os.path.join(CACHE_DIR, "schemeP_test.npz"))
    test = {k: d[k] for k in d.files}

    feat_names_path = os.path.join(CACHE_DIR, "schemeP_feat_names.txt")
    with open(feat_names_path) as f:
        all_feat_names = [l.strip() for l in f]
    total_dim = test["X"].shape[1]
    name_to_idx = {n: i for i, n in enumerate(all_feat_names)}
    drop_idx = set(name_to_idx[n] for n in DROP_NAMES if n in name_to_idx)
    keep_idx = np.array([i for i in range(total_dim) if i not in drop_idx], dtype=np.int64)
    X_test = test["X"][:, keep_idx]
    mp_t = test["mp_t"]
    mp_th = test[f"mp_t{HORIZON}"]
    date = test["date"]
    sym = test["sym"]
    print(f"  N={len(X_test):,} date={date.min()}-{date.max()} in {time.time()-t0:.1f}s", flush=True)

    # v2: LGB predictions (trained on 359 selected features, matching schemeP cache)
    print("Loading LGB models...", flush=True)
    lgb_preds = []
    for s in SEEDS:
        m = lgb.Booster(model_file=os.path.join(V2_PKG, f"model_h{HORIZON}_seed{s}.txt"))
        lgb_preds.append(m.predict(X_test).astype(np.float32))  # X_test = 359 selected features
    pred_lgb = np.stack(lgb_preds).mean(0)

    # v2: MLP predictions (MLPNumpy handles keep_idx internally)
    print("Loading MLP models...", flush=True)
    mlp_preds = []
    for s in SEEDS:
        m = MLPNumpy(os.path.join(V2_PKG, f"nn_h{HORIZON}_seed{s}.npz"))
        mlp_preds.append(m.predict(X_test))  # pass 359 selected features
    pred_nn = np.stack(mlp_preds).mean(0)

    # T172: GroupTransformer predictions
    print("Loading GroupTransformer models...", flush=True)
    tr_preds = []
    missing = []
    for s in SEEDS:
        p = os.path.join(TR_NPZ_DIR, f"nn_tr_h{HORIZON}_seed{s}.npz")
        if not os.path.exists(p):
            print(f"  WARNING: {p} not found", flush=True)
            missing.append(s)
            continue
        m = GroupTransformerNumpy(p)
        tr_preds.append(m.predict(X_test))  # pass 359 selected features
    if not tr_preds:
        print("ERROR: No GroupTransformer models found!", flush=True)
        return
    pred_tr = np.stack(tr_preds).mean(0)
    print(f"  loaded {len(tr_preds)}/{len(SEEDS)} seeds", flush=True)

    # Load v2 thresholds
    with open(os.path.join(V2_PKG, "thresholds.json")) as f:
        tcfg = json.load(f)
    h60cfg = next(h for h in tcfg["horizons"] if h["h"] == HORIZON)
    thr_up = float(h60cfg["thr_up"])
    thr_dn = float(h60cfg["thr_dn"])
    w_nn = float(h60cfg.get("w_nn", 1.0))
    w_lgb = float(h60cfg.get("w_lgb", 1.0))
    print(f"  v2 weights: w_nn={w_nn} w_lgb={w_lgb}", flush=True)
    print(f"  v2 thresholds: thr_up={thr_up:.6f} thr_dn={thr_dn:.6f}", flush=True)

    # v2 baseline
    pred_v2 = (w_nn * pred_nn + w_lgb * pred_lgb) / (w_nn + w_lgb)

    # Evaluate
    pnl_v2, act_v2 = compute_pnl(pred_v2, mp_t, mp_th, thr_up, thr_dn)
    pnl_tr, act_tr = compute_pnl(pred_tr, mp_t, mp_th, thr_up, thr_dn)

    print(f"\n=== Evaluation on test set (dates {date.min()}-{date.max()}) ===")
    print(f"  v2 baseline:       PnL={pnl_v2:+.4f}  active={act_v2:,}")
    print(f"  Transformer-only:  PnL={pnl_tr:+.4f}  active={act_tr:,}")

    # Correlation between predictions
    corr_tr_nn = float(np.corrcoef(pred_tr, pred_nn)[0, 1])
    corr_tr_lgb = float(np.corrcoef(pred_tr, pred_lgb)[0, 1])
    corr_nn_lgb = float(np.corrcoef(pred_nn, pred_lgb)[0, 1])
    print(f"\n  Correlations: Tr-NN={corr_tr_nn:.3f}  Tr-LGB={corr_tr_lgb:.3f}  NN-LGB={corr_nn_lgb:.3f}")

    # Sweep ensemble weights
    print(f"\n  Ensemble sweep (v2 w_nn={w_nn} w_lgb={w_lgb} fixed, varying w_tr):")
    best_pnl, best_wtr = -999, 0.0
    for w_tr in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
        wtot = w_nn + w_lgb + w_tr
        pred_ens = (w_nn * pred_nn + w_lgb * pred_lgb + w_tr * pred_tr) / wtot
        pnl_ens, act_ens = compute_pnl(pred_ens, mp_t, mp_th, thr_up, thr_dn)
        delta = pnl_ens - pnl_v2
        marker = " <--" if delta > 0 else ""
        print(f"    w_tr={w_tr:.2f}: PnL={pnl_ens:+.4f}  delta={delta:+.4f}  active={act_ens:,}{marker}")
        if pnl_ens > best_pnl:
            best_pnl = pnl_ens
            best_wtr = w_tr

    delta_best = best_pnl - pnl_v2
    print(f"\n  Best: w_tr={best_wtr:.2f} PnL={best_pnl:+.4f} delta={delta_best:+.4f}")
    if delta_best > 0.5:
        print(f"  >> PROCEED TO PACKAGING (delta > +0.5 threshold)")
    else:
        print(f"  >> ABORT packaging (delta <= +0.5, transformer not adding value)")

    results = {
        "v2_pnl": pnl_v2, "v2_active": act_v2,
        "tr_only_pnl": pnl_tr, "tr_only_active": act_tr,
        "corr_tr_nn": corr_tr_nn, "corr_tr_lgb": corr_tr_lgb, "corr_nn_lgb": corr_nn_lgb,
        "best_w_tr": best_wtr, "best_ensemble_pnl": best_pnl, "delta_vs_v2": delta_best,
        "n_tr_seeds": len(tr_preds), "missing_seeds": missing,
        "thr_up": thr_up, "thr_dn": thr_dn,
    }
    out = os.path.join(HERE, "eval_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out}", flush=True)
    return results


if __name__ == "__main__":
    main()
