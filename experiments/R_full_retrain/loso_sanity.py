"""LOSO sanity check on test split (date 96-119, 442k, 5 sym).

NOTE: full-retrain LGB models were trained on date 0-119 (incl. test) so
predictions here are IN-SAMPLE for LGB. PnL will be inflated. Used only to
confirm the stack runs without error and produces sane numbers — NOT a
decision criterion. Real eval = platform.

Computes:
  pred_lgb = avg over 5 full-retrain LGB models on test X features
  pred_nn  = avg over 5 T87 NN cached preds (byte-identical iter_015 v1)
  combined = (1.0*nn + 1.5*lgb) / 2.5

For both:
  iter_019 v1: thr_up=0.000358, thr_dn=0.000216 (iter_015 v1 baseline DE)
  iter_019 v2: thr_up=0.0002999, thr_dn=0.0002159 + per-sym beta abstain band

Sums per-sym PnL = LOSO-equiv (sym-independent fold sum).
"""
from __future__ import annotations
import json
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

T87_DIR = os.path.join(ROOT, "experiments", "T87_spo_dfl")
T68_DIR = os.path.join(ROOT, "experiments", "T68_stage5_features")
CACHE_DIR = os.path.join(T68_DIR, "cache")

SEEDS = (1, 7, 13, 42, 100)
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001

# v1 thresholds (iter_015 v1 baseline DE)
V1_THR_UP = 0.000358
V1_THR_DN = 0.000216

# v2 thresholds (iter_018 v1 DE on combined stack)
V2_THR_UP = 0.00029995433796130955
V2_THR_DN = 0.0002158528296175958

# v2 conformal wrapper
V2_PER_SYM_BETA = {0: 0.10, 1: 0.40, 2: 0.30, 3: 0.00, 4: 0.00}
V2_DEFAULT_BETA = 0.16

# Drop-list (matches train_full.py / fast_features extra dropping)
T59_FAIL_NAMES = [
    "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
    "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
    "qrank_W100_cumspread",
    "kyle_lam_W50", "kyle_lam_W100",
    "roll_eff_spr_ratio_W100",
]
STAGE5_FAIL_NAMES = ["liq_asym_top5_W5"]
DROP_NAMES = T59_FAIL_NAMES + STAGE5_FAIL_NAMES


def vectorized_pnl(action, mp_t, mp_th):
    side = action.astype(np.float64) - 1.0
    abs_side = np.abs(side)
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * abs_side * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee_pnl) / denom


def main():
    import lightgbm as lgb
    print("=== LOSO sanity check (INFLATED, NOT decision) ===", flush=True)

    print("Loading test split ...", flush=True)
    d = np.load(os.path.join(CACHE_DIR, "schemeP_test.npz"))
    feat_names_path = os.path.join(CACHE_DIR, "schemeP_feat_names.txt")
    with open(feat_names_path) as f:
        all_feat_names = [line.strip() for line in f]
    name_to_idx = {n: i for i, n in enumerate(all_feat_names)}
    drop_idx = set(name_to_idx[n] for n in DROP_NAMES if n in name_to_idx)
    keep_idx = np.array([i for i in range(len(all_feat_names)) if i not in drop_idx], dtype=np.int64)
    X = d['X'][:, keep_idx].astype(np.float32, copy=False)
    sym = np.asarray(d['sym']).astype(np.int64)
    mp_t = np.asarray(d['mp_t']).astype(np.float64)
    mp_th = np.asarray(d['mp_t60']).astype(np.float64)
    print(f"  N={len(X):,} feat_dim={X.shape[1]}  syms={sorted(np.unique(sym).tolist())}", flush=True)

    print("Predicting LGB (5 full-retrain models) ...", flush=True)
    pred_lgb = None
    for s in SEEDS:
        path = os.path.join(HERE, f"model_h60_seed{s}.txt")
        b = lgb.Booster(model_file=path)
        p = b.predict(X).astype(np.float64)
        pred_lgb = p if pred_lgb is None else (pred_lgb + p)
        print(f"  seed={s} OK  mean={p.mean():+.6f}  std={p.std():.6f}", flush=True)
    pred_lgb /= len(SEEDS)

    print("Loading T87 NN cached predictions (byte-identical iter_015 v1) ...", flush=True)
    # T87 pred files are sorted matching test split
    pred_nn = None
    for s in SEEDS:
        df = pd.read_parquet(os.path.join(T87_DIR, f"pred_T87_seed{s}_main.parquet"))
        p = df["pred_dmid_norm"].to_numpy(np.float64)
        assert len(p) == len(X), (len(p), len(X))
        pred_nn = p if pred_nn is None else (pred_nn + p)
    pred_nn /= len(SEEDS)
    print(f"  pred_nn mean={pred_nn.mean():+.6f}  std={pred_nn.std():.6f}", flush=True)

    # Combined stack
    pred_combined = (1.0 * pred_nn + 1.5 * pred_lgb) / 2.5
    print(f"  pred_combined mean={pred_combined.mean():+.6f}  std={pred_combined.std():.6f}", flush=True)

    # Compute sigma per sym (for v2 conformal band)
    sigma_per_sym = {}
    for s in SYMS:
        mask = sym == s
        sigma_per_sym[s] = float(pred_combined[mask].std()) if mask.sum() else 0.0
    print(f"  sigma_per_sym (combined): {sigma_per_sym}", flush=True)

    results = {}
    for variant, thr_up, thr_dn, use_conformal in [
        ("v1", V1_THR_UP, V1_THR_DN, False),
        ("v2", V2_THR_UP, V2_THR_DN, True),
    ]:
        per_sym_pnl, per_sym_active, per_sym_win = [], [], []
        for s in SYMS:
            mask = sym == s
            pred = pred_combined[mask]
            mp_t_s = mp_t[mask]; mp_th_s = mp_th[mask]
            if use_conformal:
                beta = V2_PER_SYM_BETA.get(s, V2_DEFAULT_BETA)
                band = beta * sigma_per_sym[s]
                thr_up_eff = thr_up + band
                thr_dn_eff = thr_dn + band
            else:
                thr_up_eff = thr_up
                thr_dn_eff = thr_dn
            action = np.full(len(pred), 1, dtype=np.int8)
            action[pred > thr_up_eff] = 2
            action[pred < -thr_dn_eff] = 0
            pnl = vectorized_pnl(action, mp_t_s, mp_th_s)
            per_sym_pnl.append(float(pnl.sum()))
            am = action != 1
            per_sym_active.append(float(am.mean()))
            per_sym_win.append(float((pnl[am] > 0).mean()) if am.sum() else 0.0)
        total = sum(per_sym_pnl)
        print(f"\n=== {variant} ===", flush=True)
        print(f"  total LOSO-equiv (INFLATED, in-sample LGB) = {total:+.4f}", flush=True)
        print(f"  per_sym_pnl: {[f'{v:+.4f}' for v in per_sym_pnl]}", flush=True)
        print(f"  per_sym_active: {[f'{v:.3f}' for v in per_sym_active]}", flush=True)
        print(f"  per_sym_win: {[f'{v:.3f}' for v in per_sym_win]}", flush=True)
        print(f"  min_per_sym: {min(per_sym_pnl):+.4f}", flush=True)
        results[variant] = {
            "total_loso_equiv_INFLATED": total,
            "per_sym_pnl": per_sym_pnl,
            "per_sym_active": per_sym_active,
            "per_sym_win": per_sym_win,
            "min_per_sym": float(min(per_sym_pnl)),
            "thr_up": thr_up, "thr_dn": thr_dn, "conformal": use_conformal,
        }

    out_path = os.path.join(HERE, "loso_sanity.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}", flush=True)
    print("\nNOTE: these numbers are INFLATED — full-retrain LGB has seen test data.", flush=True)
    print("      Used only as a sanity check that the stack runs without error.", flush=True)


if __name__ == "__main__":
    main()
