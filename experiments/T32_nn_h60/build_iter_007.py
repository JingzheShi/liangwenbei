"""Build iter_007 submission: 5-seed LightGBM aug_a + 1 NN h_60 ensemble.

For each predict() call:
  - LightGBM 5 boosters → 5 prob arrays → average → lgb_avg
  - NN final model (trained on full train+val, no LOSO held-out) → nn_probs
  - Final = α*nn_probs + (1-α)*lgb_avg, with α from combine_results.json
  - Apply (T, delta) threshold

For the NN final model, we use the SWA-averaged ensemble of all 5 LOSO fold
models. (Inference: average their softmax outputs and treat as a single
"NN consensus" model.) Reason: training one final NN on full data would not
leverage the LOSO models we already have, and validation evidence comes from
those exact models.

Layout:
  submission/iter_007_nn_lgbm_ensemble/
    Predictor.py
    compute.py             -- T3 features for LightGBM
    config.json            -- (= iter_002, raw 154 features in 'feature')
    requirements.txt       -- adds torch
    thresholds.json
    model.py               -- DeepLOB_H60 architecture
    model_h60_lgbm_seed{S}.txt   -- 5 LightGBM finals (S in {42,1,7,13,100})
    model_h60_nn_held{K}.pt      -- 5 NN LOSO models (K in 0..4) — averaged at inference
    model_h40.txt          -- (= iter_002 h_40 baseline)
    submission.zip
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SRC_BASE = ROOT / "submission" / "iter_002_lgbm_schemeC"
SRC_005B = ROOT / "submission" / "iter_005b_aug_a_5seed_h60"
T27 = ROOT / "experiments" / "T27_iter005"
T32 = HERE
DST = ROOT / "submission" / "iter_007_nn_lgbm_ensemble"
ZIP_OUT = ROOT / "submission_050628_iter007.zip"

LGB_SEEDS = (42, 1, 7, 13, 100)
NN_FOLDS = (0, 1, 2, 3, 4)


PREDICTOR_SRC = '''"""Predictor for iter_007: 5-seed LightGBM + 5-fold NN ensemble at h_60.

Compliance with platform contract (CRITICAL_CONSTRAINTS.md §1):
  - sym / date never enter the feature vector
  - No cross-call state on `self` (per-call window only)
  - All models sym-agnostic (DeepLOB conv on 100x154; GBDT on 154-raw + T3 derived)

h_60 strategy:
  lgb_avg = mean(softmax over 5 lightgbm boosters)
  nn_avg  = mean(softmax over 5 NN LOSO models)
  prob    = alpha * nn_avg + (1 - alpha) * lgb_avg
  apply (T, delta) threshold rule

Other horizons (h_5/10/20 inactive, h_40 from iter_002 baseline).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import lightgbm as lgb
import torch

WINDOW = 100
HORIZON_LIST = (5, 10, 20, 40, 60)
HORIZON_TO_IDX = {h: i for i, h in enumerate(HORIZON_LIST)}


def _load_t3_module(here: str):
    spec = importlib.util.spec_from_file_location(
        "iter_007_t3_compute", os.path.join(here, "compute.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Predictor:
    def __init__(self) -> None:
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)

        self._t3 = _load_t3_module(here)
        all_t3_cols = list(self._t3.feature_v1_columns())
        self._t3_feat_cols: List[str] = [c for c in all_t3_cols if not c.startswith("time_")]

        cfg_path = os.path.join(here, "config.json")
        with open(cfg_path) as f:
            cfg_main = json.load(f)
        self._raw_feat_cols: List[str] = list(cfg_main["feature"])
        try:
            self._amount_delta_idx = self._raw_feat_cols.index("amount_delta")
        except ValueError:
            self._amount_delta_idx = -1

        thresholds_path = os.path.join(here, "thresholds.json")
        with open(thresholds_path) as f:
            tcfg = json.load(f)
        self._horizons: List[Dict] = tcfg["horizons"]
        # alpha lives at the top level (used only for h_60)
        self._alpha = float(tcfg.get("nn_alpha_h60", 0.5))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- LightGBM h_60 5-seed ensemble ----
        self._lgb_h60: List[lgb.Booster] = []
        for s in (42, 1, 7, 13, 100):
            mp = os.path.join(here, f"model_h60_lgbm_seed{s}.txt")
            if os.path.isfile(mp):
                self._lgb_h60.append(lgb.Booster(model_file=mp))
        # h_40 single
        self._lgb_h40 = None
        mp40 = os.path.join(here, "model_h40.txt")
        if os.path.isfile(mp40):
            self._lgb_h40 = lgb.Booster(model_file=mp40)

        # ---- NN h_60 5-LOSO ensemble ----
        from model import DeepLOB_H60  # local
        self._nn_models: List[torch.nn.Module] = []
        self._nn_norm: Dict[str, np.ndarray] = {}
        for k in (0, 1, 2, 3, 4):
            pt_path = os.path.join(here, f"model_h60_nn_held{k}.pt")
            if not os.path.isfile(pt_path):
                continue
            try:
                ckpt = torch.load(pt_path, map_location=self.device, weights_only=False)
            except TypeError:
                ckpt = torch.load(pt_path, map_location=self.device)
            meta = dict(ckpt.get("meta") or {})
            fc = list(meta.get("feature_columns", self._raw_feat_cols))
            m = DeepLOB_H60(seq_len=int(meta.get("T", 100)), num_features=len(fc))
            with torch.no_grad():
                dummy = torch.zeros(2, 1, 100, len(fc))
                m(dummy)
            m.load_state_dict(ckpt["model_state"], strict=True)
            m.to(self.device).eval()
            self._nn_models.append(m)
            if not self._nn_norm:
                self._nn_norm["mean"] = np.asarray(meta["mean"], dtype=np.float32)
                self._nn_norm["std"] = np.asarray(meta["std"], dtype=np.float32)
                log1p_cols = list(meta.get("log1p_cols", ["amount_delta"]))
                self._nn_norm["log1p_idx"] = np.array(
                    [fc.index(c) for c in log1p_cols if c in fc], dtype=np.int64
                )
                self._nn_feat_cols = fc

    @staticmethod
    def _threshold_predict(probs: np.ndarray, T_thr: float, delta: float) -> np.ndarray:
        p0 = probs[:, 0]; p1 = probs[:, 1]; p2 = probs[:, 2]
        side_max = np.maximum(p0, p2)
        take = (side_max >= T_thr) & (side_max > p1 + delta)
        side_pred = np.where(p2 > p0, 2, 0)
        return np.where(take, side_pred, 1).astype(np.int64)

    def _compute_t3_no_time(self, df: pd.DataFrame) -> pd.DataFrame:
        mlofi = self._t3.compute_mlofi(df)
        wmp = self._t3.compute_wmp(df)
        rv = self._t3.compute_rv(wmp["wmp_lvl1"])
        ewma = self._t3.compute_ewma_intst(df)
        return pd.concat([mlofi, wmp, rv, ewma], axis=1)

    def _compute_window_features_lgb(self, df: pd.DataFrame) -> np.ndarray:
        t3 = self._compute_t3_no_time(df)
        raw_last = df[self._raw_feat_cols].iloc[-1].to_numpy(dtype=np.float32, copy=True)
        if self._amount_delta_idx >= 0:
            v = raw_last[self._amount_delta_idx]
            raw_last[self._amount_delta_idx] = np.sign(v) * np.log1p(abs(v))
        t3_last = t3[self._t3_feat_cols].iloc[-1].to_numpy(dtype=np.float32, copy=False)
        return np.concatenate([raw_last, t3_last]).astype(np.float32, copy=False)

    def _compute_batch_features_lgb(self, batches: List[pd.DataFrame]) -> np.ndarray:
        feats = np.empty(
            (len(batches), len(self._raw_feat_cols) + len(self._t3_feat_cols)),
            dtype=np.float32,
        )
        for i, df in enumerate(batches):
            feats[i] = self._compute_window_features_lgb(df)
        return feats

    def _compute_batch_features_nn(self, batches: List[pd.DataFrame]) -> np.ndarray:
        B = len(batches)
        F = len(self._nn_feat_cols)
        out = np.empty((B, WINDOW, F), dtype=np.float32)
        mean = self._nn_norm["mean"]
        std = self._nn_norm["std"]
        log1p_idx = self._nn_norm["log1p_idx"]
        for i, df in enumerate(batches):
            x = df[self._nn_feat_cols].to_numpy(dtype=np.float32, copy=True)
            if x.shape[0] < WINDOW:
                pad = np.repeat(x[:1], WINDOW - x.shape[0], axis=0)
                x = np.concatenate([pad, x], axis=0)
            elif x.shape[0] > WINDOW:
                x = x[-WINDOW:]
            x[:, log1p_idx] = np.log1p(x[:, log1p_idx])
            x -= mean
            x /= std
            bad = ~np.isfinite(x)
            if bad.any():
                x[bad] = 0.0
            out[i] = x
        return out

    @torch.no_grad()
    def _nn_predict_h60(self, batches: List[pd.DataFrame]) -> np.ndarray:
        if not self._nn_models:
            B = len(batches)
            return np.full((B, 3), 1.0 / 3.0, dtype=np.float32)
        x_np = self._compute_batch_features_nn(batches)
        x = torch.from_numpy(x_np).unsqueeze(1).to(self.device, dtype=torch.float32)
        prob_sum = torch.zeros(x.shape[0], 3, device=self.device, dtype=torch.float32)
        for m in self._nn_models:
            logits = m(x)
            prob_sum += torch.softmax(logits, dim=1)
        prob_avg = (prob_sum / float(len(self._nn_models))).cpu().numpy().astype(np.float32)
        return prob_avg

    def _lgb_predict_h60(self, feats: np.ndarray) -> np.ndarray:
        if not self._lgb_h60:
            return np.full((feats.shape[0], 3), 1.0 / 3.0, dtype=np.float32)
        acc = None
        for b in self._lgb_h60:
            p = b.predict(feats).astype(np.float32)
            acc = p if acc is None else acc + p
        return acc / float(len(self._lgb_h60))

    def predict(self, batches: List[pd.DataFrame]) -> List[List[int]]:
        if not batches:
            return []
        feats_lgb = self._compute_batch_features_lgb(batches)
        B = feats_lgb.shape[0]
        out = np.ones((B, 5), dtype=np.int64)

        for hcfg in self._horizons:
            if not hcfg.get("active", True):
                continue
            H = int(hcfg["h"])
            T_thr = float(hcfg.get("T", 0.50))
            delta = float(hcfg.get("delta", 0.15))
            if H == 60:
                lgb_p = self._lgb_predict_h60(feats_lgb)
                nn_p = self._nn_predict_h60(batches)
                a = self._alpha
                probs = a * nn_p + (1.0 - a) * lgb_p
            elif H == 40 and self._lgb_h40 is not None:
                probs = self._lgb_h40.predict(feats_lgb).astype(np.float32)
            else:
                continue
            preds = self._threshold_predict(probs, T_thr, delta)
            out[:, HORIZON_TO_IDX[H]] = preds
        return out.tolist()


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    cfg = json.load(open(os.path.join(here, "config.json")))
    feats = cfg["feature"]
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        rng.standard_normal((100, len(feats))).astype(np.float32),
        columns=feats,
    )
    p = Predictor()
    out = p.predict([df, df, df])
    print("smoke test predict ->", out[0], "len=", len(out))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=None,
                    help="NN weight in ensemble; if None, read from combine_results.json best")
    ap.add_argument("--T-override", type=float, default=None)
    ap.add_argument("--d-override", type=float, default=None)
    ap.add_argument("--allow-missing-final", action="store_true")
    args = ap.parse_args()

    # Load combine_results.json
    cr_path = T32 / "combine_results.json"
    if not cr_path.exists():
        sys.exit(f"missing {cr_path} — run combine_with_lgbm.py first")
    with open(cr_path) as f:
        cr = json.load(f)
    best = cr["best"]
    label = best.get("label", "")
    # Parse alpha from "ENS_alpha=0.30" pattern or fall back to args
    if args.alpha is not None:
        alpha = float(args.alpha)
    elif label.startswith("ENS_alpha="):
        alpha = float(label.split("=")[1])
    elif label == "NN_only":
        alpha = 1.0
    elif label == "LGB_5seed":
        alpha = 0.0
    else:
        alpha = float(cr.get("alphas", [0.5])[0])
    T_ens = float(args.T_override) if args.T_override is not None else float(best["T"])
    d_ens = float(args.d_override) if args.d_override is not None else float(best["delta"])
    sum_pnl = float(best["sum_cum_pnl"])
    print(f"=== Build iter_007 (5-seed LGB + 5-fold NN, h_60 ensemble) ===")
    print(f"  alpha={alpha:.2f}, T={T_ens:.2f}, d={d_ens:.2f}, OOF sum_cum_pnl={sum_pnl:+.4f}, label={label}")

    # Stage destination
    DST.mkdir(parents=True, exist_ok=True)
    for old in DST.glob("model_h60*"):
        old.unlink()
    for old in DST.glob("*.pt"):
        old.unlink()

    # Copy iter_002 base files (config, compute, h_40)
    for fname in ("compute.py", "config.json", "model_h40.txt"):
        src = SRC_BASE / fname
        if not src.exists():
            sys.exit(f"missing src: {src}")
        shutil.copy2(src, DST / fname)
        print(f"  cp {src.name}")

    # Write Predictor + model.py
    with open(DST / "Predictor.py", "w") as f:
        f.write(PREDICTOR_SRC)
    print(f"  wrote Predictor.py")
    shutil.copy2(T32 / "model.py", DST / "model.py")
    print(f"  cp model.py")

    # Copy 5 LightGBM final models
    lgb_seeds_done = []
    for s in LGB_SEEDS:
        src = T27 / f"final_model_h60_aug_a_seed{s}.txt"
        if not src.exists():
            msg = f"missing LGB final: {src}"
            if args.allow_missing_final:
                print(f"  WARN: {msg} — skipping")
                continue
            sys.exit(msg)
        shutil.copy2(src, DST / f"model_h60_lgbm_seed{s}.txt")
        lgb_seeds_done.append(s)
        print(f"  cp LGB seed{s}")

    # Copy 5 NN LOSO models
    nn_folds_done = []
    for k in NN_FOLDS:
        src = T32 / f"model_h60_held{k}.pt"
        if not src.exists():
            msg = f"missing NN LOSO model: {src}"
            if args.allow_missing_final:
                print(f"  WARN: {msg} — skipping")
                continue
            sys.exit(msg)
        shutil.copy2(src, DST / f"model_h60_nn_held{k}.pt")
        nn_folds_done.append(k)
        sz = (DST / f"model_h60_nn_held{k}.pt").stat().st_size
        print(f"  cp NN held{k} ({sz:,} bytes)")

    # requirements.txt with torch
    REQUIREMENTS = """numpy==2.4.4
pandas==2.3.3
lightgbm==4.6.0
torch==2.10.0
"""
    with open(DST / "requirements.txt", "w") as f:
        f.write(REQUIREMENTS)
    print("  wrote requirements.txt (numpy/pandas/lightgbm/torch)")

    # thresholds.json
    THRESHOLDS = {
        "_doc": (
            f"iter_007: 5-seed LightGBM aug_a + 5-fold NN h_60 ensemble. "
            f"alpha={alpha} (NN weight), T={T_ens}, d={d_ens}. "
            f"From T32 combine_with_lgbm.py best={label}, OOF sum_cum_pnl={sum_pnl:+.4f}. "
            f"h_5/10/20 disabled, h_40 = iter_002 baseline."
        ),
        "_source": "T32 combine NN+LGB OOF threshold sweep",
        "nn_alpha_h60": float(alpha),
        "horizons": [
            {"h": 5,  "T": 0.99, "delta": 0.99, "active": False,
             "_reason": "iter_002 platform = -7.68. Disabled."},
            {"h": 10, "T": 0.99, "delta": 0.99, "active": False,
             "_reason": "iter_002 platform = -8.64. Disabled."},
            {"h": 20, "T": 0.99, "delta": 0.99, "active": False,
             "_reason": "iter_002 platform = -5.09. Disabled."},
            {"h": 40, "T": 0.50, "delta": 0.00, "active": True,
             "_reason": "iter_002 platform = +2.02. Kept unchanged."},
            {"h": 60, "T": float(T_ens), "delta": float(d_ens), "active": True,
             "_reason": (
                 f"T32 NN+LGB ensemble (alpha={alpha}). "
                 f"OOF sum_cum_pnl={sum_pnl:+.4f}. iter_005b LGB-only was +11.46."
             ),
             "ensemble_meta": {
                 "alpha_nn": float(alpha), "T": float(T_ens), "delta": float(d_ens),
                 "sum_cum_pnl_oof": sum_pnl,
                 "lgb_seeds": list(lgb_seeds_done),
                 "nn_folds": list(nn_folds_done),
                 "_source": "T32 combine OOF sweep",
             }},
        ],
    }
    with open(DST / "thresholds.json", "w") as f:
        json.dump(THRESHOLDS, f, indent=2)
    print(f"  wrote thresholds.json")

    # Sanity test (22 batches, shuffled)
    import importlib.util
    import numpy as np
    import pandas as pd

    cfg = json.load(open(DST / "config.json"))
    feats = cfg["feature"]
    spec = importlib.util.spec_from_file_location("iter007_predictor", DST / "Predictor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rng = np.random.default_rng(0)
    batches = []
    for i in range(22):
        df = pd.DataFrame(
            rng.standard_normal((100, len(feats))).astype(np.float32),
            columns=feats,
        )
        batches.append(df)
    perm = rng.permutation(len(batches))
    batches = [batches[i] for i in perm]

    p = mod.Predictor()
    out = p.predict(batches)
    assert isinstance(out, list) and len(out) == 22, f"expected 22, got {len(out)}"
    for row in out:
        assert isinstance(row, list) and len(row) == 5, f"row shape {row}"
        for v in row:
            assert v in (0, 1, 2), f"invalid pred {v}"
    for row in out:
        assert row[0] == 1 and row[1] == 1 and row[2] == 1, \
            f"expected idx 0/1/2 = 1 (inactive), got {row}"
    print(f"  sanity OK: 22/22 batches, shape (22,5), h_5/10/20 = 1 (inactive)")

    # Zip
    files = ["Predictor.py", "compute.py", "config.json", "requirements.txt",
             "thresholds.json", "model.py", "model_h40.txt"]
    files += [f"model_h60_lgbm_seed{s}.txt" for s in lgb_seeds_done]
    files += [f"model_h60_nn_held{k}.pt" for k in nn_folds_done]
    inside = DST / "submission.zip"
    with zipfile.ZipFile(inside, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in files:
            z.write(DST / fn, arcname=fn)
    shutil.copy2(inside, ZIP_OUT)
    print(f"  zip -> {inside}  +  {ZIP_OUT}  ({ZIP_OUT.stat().st_size:,} bytes)")

    print("\n--- iter_007 contents ---")
    subprocess.run(["ls", "-la", str(DST)])
    print(f"\nbundled zip: {ZIP_OUT}")


if __name__ == "__main__":
    main()
