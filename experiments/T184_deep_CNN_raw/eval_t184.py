"""T184: Evaluate ensemble and build submission zip.

Strategy:
  - Load T184 predictions from pre-saved parquet (output of train_t184.py)
  - Load schemeP test for LGB + T87 NN
  - Join on (sym, date, session, t) to align
  - Sweep ensemble weights
  - Build zip

Compares to T170 holdout in-sample baseline (149.95).
"""
from __future__ import annotations
import hashlib
import json
import os
import shutil
import zipfile
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
V2_PKG = os.path.join(ROOT, "experiments", "R_full_retrain", "pkg_iter019_v2")
T170_PKG = os.path.join(ROOT, "experiments", "T170_T87M7_clean_pkg")
SCHEMEP_CACHE = os.path.join(ROOT, "experiments", "T68_stage5_features", "cache")

FEE = 0.0001
SEEDS = (1, 42, 100)
ALL_SEEDS = (1, 7, 13, 42, 100)

THR_UP = 0.00029995433796130955
THR_DN = 0.0002158528296175958
PER_SYM_BETA = {0: 0.10, 1: 0.50, 2: 0.30, 3: 0.00, 4: 0.00}
PER_SYM_SIGMA = {0: 0.0002403901, 1: 0.0004712397, 2: 0.0004522216,
                 3: 0.0004235249, 4: 0.0004279811}

DROP_NAMES = [
    "dualz_ask_diff1","dualz_bid_diff5","dualz_ask_diff5",
    "qrank_W100_spread1","qrank_W100_spread5","qrank_W100_spread10","qrank_W100_cumspread",
    "kyle_lam_W50","kyle_lam_W100","roll_eff_spr_ratio_W100","liq_asym_top5_W5",
]


def _gelu_tanh(x):
    c = 0.7978845608028654
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x**3)))

def _layernorm_np(x, w, b, eps=1e-5):
    m = x.mean(axis=-1, keepdims=True); v = x.var(axis=-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b

class _MLPNumpy:
    def __init__(self, npz_path):
        d = np.load(npz_path, allow_pickle=False)
        self.in_dim = int(d["in_dim"][0]); self.hidden = list(d["hidden"].tolist())
        self.use_layernorm = bool(d["use_layernorm"][0])
        self.target_scale = float(d["target_scale"][0])
        self.feat_mean = d["feat_mean"].astype(np.float32); self.feat_std = d["feat_std"].astype(np.float32)
        self.keep_idx = d["keep_idx"].astype(np.int64); self.clip = float(d["clip"][0])
        self.W, self.b, self.LN_W, self.LN_b = [], [], [], []
        for i in range(len(self.hidden)):
            self.W.append(d[f"L{i}_W"].astype(np.float32)); self.b.append(d[f"L{i}_b"].astype(np.float32))
            if self.use_layernorm:
                self.LN_W.append(d[f"LN{i}_W"].astype(np.float32)); self.LN_b.append(d[f"LN{i}_b"].astype(np.float32))
        self.WF = d["LF_W"].astype(np.float32); self.bF = d["LF_b"].astype(np.float32)

    def predict(self, X):
        Xs = X if X.shape[1] == self.in_dim else X[:, self.keep_idx]
        Xs = np.clip(np.nan_to_num((Xs.astype(np.float32) - self.feat_mean) / self.feat_std), -self.clip, self.clip)
        h = Xs
        for i in range(len(self.hidden)):
            h = h @ self.W[i].T + self.b[i]
            if self.use_layernorm: h = _layernorm_np(h, self.LN_W[i], self.LN_b[i])
            h = _gelu_tanh(h)
        return (h @ self.WF.T + self.bF).squeeze(-1).astype(np.float32) / self.target_scale


def load_schemeP_test():
    d = np.load(os.path.join(SCHEMEP_CACHE, "schemeP_test.npz"))
    with open(os.path.join(SCHEMEP_CACHE, "schemeP_feat_names.txt")) as f:
        all_names = [l.strip() for l in f]
    name_to_idx = {n: i for i, n in enumerate(all_names)}
    drop_idx = {name_to_idx[n] for n in DROP_NAMES if n in name_to_idx}
    keep_idx = np.array([i for i in range(len(all_names)) if i not in drop_idx], dtype=np.int64)
    X = d['X'][:, keep_idx].astype(np.float32)
    sym = d['sym'].astype(np.int64)
    mp_t = d['mp_t'].astype(np.float64)
    mp_th = d['mp_t60'].astype(np.float64)
    # Try to get date/session/t for joining
    date = d['date'].astype(np.int32) if 'date' in d else None
    sess = d['sess_idx'].astype(np.int32) if 'sess_idx' in d else None
    t_arr = d['t'].astype(np.int32) if 't' in d else None
    return X, sym, mp_t, mp_th, date, sess, t_arr


def load_t184_predictions():
    """Load and average T184 predictions from per-seed parquets.
    Returns DataFrame with columns: sym, date, session, t, pred_t184
    """
    dfs = []
    for s in SEEDS:
        path = os.path.join(HERE, f"pred_t184_seed{s}.parquet")
        if not os.path.isfile(path):
            print(f"  WARNING: missing {path}", flush=True)
            continue
        df = pd.read_parquet(path)[["sym", "date", "session", "t", "pred_dmid_norm",
                                    "midprice_t", "midprice_th"]]
        df = df.rename(columns={"pred_dmid_norm": f"pred_s{s}"})
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No T184 prediction parquets found!")

    base = dfs[0][["sym", "date", "session", "t", "midprice_t", "midprice_th"]].copy()
    for df in dfs:
        seed_col = [c for c in df.columns if c.startswith("pred_s")][0]
        base = base.merge(df[["sym", "date", "session", "t", seed_col]],
                          on=["sym", "date", "session", "t"], how="left")

    pred_cols = [c for c in base.columns if c.startswith("pred_s")]
    base["pred_t184"] = base[pred_cols].mean(axis=1)
    print(f"  T184 preds: N={len(base):,}  NaN_frac={base['pred_t184'].isna().mean():.3f}", flush=True)
    return base


def vectorized_pnl(action, mp_t, mp_th):
    side = action.astype(np.float64) - 1.0
    fee = FEE * np.abs(side) * np.abs(mp_th + 1.0 + mp_t + 1.0)
    return (side * (mp_th - mp_t) - fee) / (mp_t + 1.0)


def eval_pnl_from_df(df: pd.DataFrame, pred_col: str, thr_up: float, thr_dn: float,
                     per_sym_beta=None, per_sym_sigma=None, default_beta=0.16, default_sigma=3.998e-4):
    """Eval PnL using mp_t, mp_th from df directly (no schemeP alignment needed)."""
    SYMS = sorted(df["sym"].unique().tolist())
    pnls = []
    for s in SYMS:
        mask = df["sym"] == s
        sub = df[mask]
        p = sub[pred_col].fillna(0.0).values
        if per_sym_beta is not None:
            beta = per_sym_beta.get(s, default_beta)
            sigma = per_sym_sigma.get(s, default_sigma) if per_sym_sigma else default_sigma
            band = beta * sigma
        else:
            band = 0.0
        action = np.full(len(p), 1, dtype=np.int8)
        action[p > (thr_up + band)] = 2
        action[p < -(thr_dn + band)] = 0
        pnl = vectorized_pnl(action, sub["midprice_t"].values, sub["midprice_th"].values)
        pnls.append(float(pnl.sum()))
    return sum(pnls), pnls


def eval_pnl_arr(pred, sym, mp_t, mp_th, thr_up, thr_dn,
                 per_sym_beta=None, per_sym_sigma=None, default_beta=0.16, default_sigma=3.998e-4):
    SYMS = sorted(np.unique(sym).tolist())
    pnls = []
    for s in SYMS:
        mask = sym == s; p = pred[mask]
        if per_sym_beta is not None:
            beta = per_sym_beta.get(s, default_beta)
            sigma = per_sym_sigma.get(s, default_sigma) if per_sym_sigma else default_sigma
            band = beta * sigma
        else:
            band = 0.0
        action = np.full(mask.sum(), 1, dtype=np.int8)
        action[p > (thr_up + band)] = 2
        action[p < -(thr_dn + band)] = 0
        pnls.append(float(vectorized_pnl(action, mp_t[mask], mp_th[mask]).sum()))
    return sum(pnls), pnls


def main():
    print("=== T184 Eval ===", flush=True)

    # 1. Load T184 predictions from parquets
    print("\nLoading T184 predictions...", flush=True)
    t184_df = load_t184_predictions()

    # 2. Eval T184 standalone
    pnl_t184, per_t184 = eval_pnl_from_df(t184_df, "pred_t184", THR_UP, THR_DN,
                                            PER_SYM_BETA, PER_SYM_SIGMA)
    print(f"\n[T184 standalone] pnl={pnl_t184:+.4f}  per={[f'{v:+.4f}' for v in per_t184]}", flush=True)

    # 3. Load schemeP test for LGB
    print("\nLoading schemeP test data...", flush=True)
    X_sp, sym_sp, mp_t_sp, mp_th_sp, date_sp, sess_sp, t_sp = load_schemeP_test()
    N_sp = len(sym_sp)
    print(f"  N_schemeP={N_sp:,}", flush=True)

    # 4. LGB predictions (5 seeds)
    print("\nLGB predictions (5 seeds)...", flush=True)
    import lightgbm as lgb_mod
    pred_lgb = None
    for s in ALL_SEEDS:
        mp = os.path.join(V2_PKG, f"model_h60_seed{s}.txt")
        if not os.path.isfile(mp):
            mp = os.path.join(T170_PKG, f"model_h60_seed{s}.txt")
        b = lgb_mod.Booster(model_file=mp)
        p = b.predict(X_sp).astype(np.float32)
        pred_lgb = p if pred_lgb is None else pred_lgb + p
        print(f"  seed={s}  mean={p.mean():+.6f}", flush=True)
    pred_lgb /= len(ALL_SEEDS)

    # 5. T87 NN predictions (5 seeds, for v2 baseline)
    print("\nT87 NN predictions (5 seeds)...", flush=True)
    pred_nn = None
    for s in ALL_SEEDS:
        mp = os.path.join(V2_PKG, f"nn_h60_seed{s}.npz")
        if not os.path.isfile(mp):
            mp = os.path.join(T170_PKG, f"nn_h60_seed{s}.npz")
        p = _MLPNumpy(mp).predict(X_sp)
        pred_nn = p if pred_nn is None else pred_nn + p
        print(f"  seed={s}  mean={p.mean():+.6f}", flush=True)
    pred_nn /= len(ALL_SEEDS)

    # v2 baseline
    pred_v2 = (pred_nn + 0.5 * pred_lgb) / 1.5
    pnl_v2, per_v2 = eval_pnl_arr(pred_v2, sym_sp, mp_t_sp, mp_th_sp, THR_UP, THR_DN,
                                    PER_SYM_BETA, PER_SYM_SIGMA)
    print(f"\n[v2 baseline] pnl={pnl_v2:+.4f}  per={[f'{v:+.4f}' for v in per_v2]}", flush=True)

    # 6. Build T184 predictions aligned with schemeP rows
    # Use the pred_t184 column from t184_df (has t>=99 rows from T95 cache)
    # Build a full-length array aligned with schemeP (N_sp rows), fill NaN as 0
    pred_t184_arr = np.zeros(N_sp, dtype=np.float32)
    if date_sp is not None and t_sp is not None and sess_sp is not None:
        # Join by (sym, date, sess, t)
        sp_key = pd.DataFrame({
            "sym": sym_sp.astype(np.int64),
            "date": date_sp.astype(np.int32),
            "sess_idx": sess_sp.astype(np.int32),
            "t": t_sp.astype(np.int32),
            "__sp_row": np.arange(N_sp),
        })
        # T184 df uses "session" as string (am/pm)
        sess_map = {"am": 0, "pm": 1}
        t184_df_j = t184_df[["sym", "date", "session", "t", "pred_t184"]].copy()
        t184_df_j["sess_idx"] = t184_df_j["session"].map(sess_map).astype(np.int32)
        merged = sp_key.merge(
            t184_df_j[["sym", "date", "sess_idx", "t", "pred_t184"]],
            on=["sym", "date", "sess_idx", "t"], how="left"
        )
        merged_pred = merged["pred_t184"].values
        mask_valid = ~np.isnan(merged_pred)
        pred_t184_arr[merged["__sp_row"].values[mask_valid]] = merged_pred[mask_valid].astype(np.float32)
        n_matched = mask_valid.sum()
        print(f"\n  Joined T184 preds to schemeP: {n_matched:,}/{N_sp:,} rows matched", flush=True)
    else:
        # Fallback: assume same ordering, fill from t184 directly
        n_t184 = min(len(t184_df), N_sp)
        preds_t184 = t184_df["pred_t184"].fillna(0.0).values[:n_t184].astype(np.float32)
        pred_t184_arr[:n_t184] = preds_t184
        print(f"\n  WARNING: No join keys, using positional alignment ({n_t184:,} rows)", flush=True)

    # Correlation
    nonzero = pred_t184_arr != 0
    corr_with_nn = float(np.corrcoef(pred_t184_arr[nonzero], pred_nn[nonzero])[0, 1]) if nonzero.sum() > 1 else 0.0
    corr_with_lgb = float(np.corrcoef(pred_t184_arr[nonzero], pred_lgb[nonzero])[0, 1]) if nonzero.sum() > 1 else 0.0
    print(f"\n  Corr T184 vs T87-NN: {corr_with_nn:.4f}", flush=True)
    print(f"  Corr T184 vs LGB:    {corr_with_lgb:.4f}", flush=True)

    # Ensemble sweep (LGB + T184)
    print("\n=== Ensemble sweep (LGB + T184 NN) ===", flush=True)
    configs = {
        "lgb1.5_t184_1.0": (1.5, 1.0),
        "lgb1.0_t184_1.0": (1.0, 1.0),
        "lgb1.0_t184_1.5": (1.0, 1.5),
        "lgb2.0_t184_1.0": (2.0, 1.0),
        "lgb0.5_t184_1.0": (0.5, 1.0),
        "lgb1.5_t184_0.5": (1.5, 0.5),
        "lgb1.0_t184_0.5": (1.0, 0.5),
    }
    best_pnl, best_cfg = pnl_v2, "v2_baseline"
    best_w = (1.5, 1.0)
    ens_results = {}
    for name, (w_lgb, w_t184) in configs.items():
        p = (w_lgb * pred_lgb + w_t184 * pred_t184_arr) / (w_lgb + w_t184)
        pnl, per = eval_pnl_arr(p, sym_sp, mp_t_sp, mp_th_sp, THR_UP, THR_DN,
                                 PER_SYM_BETA, PER_SYM_SIGMA)
        delta = pnl - pnl_v2
        print(f"  {name:<22} {pnl:+.4f}  delta={delta:+.4f}  per={[f'{v:+.4f}' for v in per]}",
              flush=True)
        ens_results[name] = {"pnl": pnl, "delta": delta, "weights": {"w_lgb": w_lgb, "w_t184": w_t184}}
        if pnl > best_pnl:
            best_pnl, best_cfg, best_w = pnl, name, (w_lgb, w_t184)

    print(f"\nBest: {best_cfg}  pnl={best_pnl:+.4f}  delta={best_pnl-pnl_v2:+.4f}", flush=True)

    # Build zip (always build, even if not better, as replacement for T87)
    w_lgb_best, w_t184_best = configs.get(best_cfg, (1.5, 1.0))
    print(f"\nBuilding zip with weights lgb={w_lgb_best}, t184={w_t184_best}...", flush=True)
    zip_path = build_zip(w_lgb_best, w_t184_best)
    md5 = hashlib.md5(open(zip_path, "rb").read()).hexdigest()
    print(f"  zip={zip_path}  md5={md5}", flush=True)
    os.makedirs("/tmp/metabot-outputs/worker-8b482f90", exist_ok=True)
    shutil.copy2(zip_path, "/tmp/metabot-outputs/worker-8b482f90/")
    print(f"  Copied to output dir", flush=True)

    results = {
        "task": "T184 DeepLOB/Transformer on raw OB sequence",
        "architecture": "TickTransformer (d_model=64, n_layers=2, nhead=4, ff_dim=256)",
        "n_features_input": 31,
        "n_seeds": len(SEEDS),
        "scheme": "v2",
        "T170_baseline_holdout_insample": 149.95,
        "v2_pnl_insample": pnl_v2,
        "T184_holdout_insample": pnl_t184,
        "delta_T184_vs_v2": pnl_t184 - pnl_v2,
        "best_ensemble_pnl": best_pnl,
        "best_ensemble_cfg": best_cfg,
        "best_ensemble_delta": best_pnl - pnl_v2,
        "correlations": {
            "T184_vs_T87_NN": corr_with_nn,
            "T184_vs_LGB": corr_with_lgb,
        },
        "ensemble_results": ens_results,
        "zip": zip_path,
        "md5": md5,
        "expected_platform": "Risk: very different from MLP architecture. Correlation to T87-NN/LGB determines diversification value.",
    }
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results.json", flush=True)

    print(f"\nRESULT: task=T184_DeepLOB_raw metrics={{v2_pnl={pnl_v2:.2f},t184_pnl={pnl_t184:.2f},best_ens={best_pnl:.2f},delta={best_pnl-pnl_v2:+.2f}}} notes=[corr_nn={corr_with_nn:.3f},corr_lgb={corr_with_lgb:.3f},best={best_cfg},zip={zip_path}]")


def build_zip(w_lgb: float, w_t184: float) -> str:
    pkg_dir = os.path.join(HERE, "pkg_t184")
    os.makedirs(pkg_dir, exist_ok=True)

    # Copy T170 pkg as base (LGB + fast_features_batch + config + requirements)
    for fn in os.listdir(T170_PKG):
        src = os.path.join(T170_PKG, fn)
        if os.path.isfile(src) and not fn.endswith(".pyc"):
            shutil.copy2(src, pkg_dir)

    # Copy T184 Transformer models
    for s in SEEDS:
        src = os.path.join(HERE, f"model_t184_seed{s}.pt")
        if os.path.isfile(src):
            dst = os.path.join(pkg_dir, f"nn_raw_h60_seed{s}.pt")
            shutil.copy2(src, dst)
            print(f"  Copied {fn} -> {dst}", flush=True)

    # Write thresholds.json
    thresholds = {
        "_doc": f"T184: v2_LGB(w={w_lgb}) + T184_RawTransformer(w={w_t184}). Raw 100-tick LOB → Transformer NN.",
        "horizons": [
            {"h": 5,  "thr_up": 1.0, "thr_dn": 1.0, "active": False},
            {"h": 10, "thr_up": 1.0, "thr_dn": 1.0, "active": False},
            {"h": 20, "thr_up": 1.0, "thr_dn": 1.0, "active": False},
            {"h": 40, "thr_up": 1.0, "thr_dn": 1.0, "active": False},
            {"h": 60, "thr_up": THR_UP, "thr_dn": THR_DN,
             "w_lgb": w_lgb, "w_nn": w_t184,
             "active": True, "ensemble_seeds": list(SEEDS)},
        ],
        "conformal_wrapper": {
            "enabled": True,
            "per_sym_beta": {str(k): v for k, v in PER_SYM_BETA.items()},
            "default_beta_for_ood": 0.16,
            "per_sym_sigma": {str(k): v for k, v in PER_SYM_SIGMA.items()},
            "default_sigma_for_ood": 3.998317e-4,
        },
        "raw_transformer": {
            "enabled": True,
            "seeds": list(SEEDS),
            "window": 100,
            "n_features": 31,
        }
    }
    with open(os.path.join(pkg_dir, "thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)

    # Write Predictor
    shutil.copy2(os.path.join(HERE, "Predictor_t184.py"),
                 os.path.join(pkg_dir, "Predictor.py"))

    # Smoke test
    import subprocess, sys
    r = subprocess.run([sys.executable, os.path.join(pkg_dir, "Predictor.py")],
                       capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        print(f"  Smoke test FAILED:\nSTDOUT: {r.stdout}\nSTDERR: {r.stderr}", flush=True)
        raise RuntimeError("Smoke test failed")
    print(f"  Smoke test OK: {r.stdout.strip()[:300]}", flush=True)

    zip_name = "submission_050910_iter019_v2N_DeepLOB.zip"
    zip_path = os.path.join(HERE, zip_name)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fn in sorted(os.listdir(pkg_dir)):
            fp = os.path.join(pkg_dir, fn)
            if os.path.isfile(fp) and not fn.endswith(".pyc"):
                zf.write(fp, fn)
    print(f"  Built {zip_path}  ({os.path.getsize(zip_path)/1e6:.1f} MB)", flush=True)
    return zip_path


if __name__ == "__main__":
    os.makedirs("/tmp/metabot-outputs/worker-8b482f90", exist_ok=True)
    main()
