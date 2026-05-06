"""T26: Build submission iter_006 from a chosen DR variant.

Recipe (per task spec):
  * h_5 / h_10 / h_20: active=False (predict label=1 always)
  * h_40: copy from iter_002 (untouched, no DR retrain)
  * h_60: ensemble of 5 LOSO-fold models (augment-trained from chosen variant);
          Predictor averages the 5 boosters' probs at inference.

Inputs:
  --variant <name>     -- one of baseline,aug_a,aug_b,aug_c,aug_abc (best variant)
  --T --delta          -- chosen threshold for h_60 (from threshold_sweep.py)

Outputs:
  submission/iter_006_h60_aug/
    Predictor.py          (modified to ensemble h_60 across 5 models)
    config.json           (copied)
    compute.py            (copied)
    requirements.txt      (copied)
    thresholds.json       (h_5/10/20 active=False, h_40 from iter_002, h_60 new)
    model_h40.txt         (copied from iter_002)
    model_h60_fold0.txt   (T26 augment-trained LOSO model held=0)
    ...
    model_h60_fold4.txt   (T26 augment-trained LOSO model held=4)

Then sanity test 22/22 and zip to submission_050625_iter006.zip.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


PREDICTOR_TEMPLATE = '''"""Predictor for iter_006: LightGBM Scheme C1 (223-d) with h=60 5-fold ensemble.

Compliance with platform contract (CRITICAL_CONSTRAINTS.md §1):
  - sym / date never enter the feature vector
  - No cross-call state held on `self` (per-call window only)
  - Model is sym-agnostic by construction
  - T3 features (MLOFI / WMP / RV / EWMA) computed fresh per 100-row window

Differences vs iter_002:
  - h_5 / h_10 / h_20 deactivated (active=False in thresholds.json -> always label 1)
  - h_40: same model as iter_002
  - h_60: ENSEMBLE of 5 LOSO-fold models (T26 domain-randomization training);
          inference averages probs across 5 boosters before threshold rule.

Bundle:
  thresholds.json         per-horizon (T, delta, active) plus h_60 ensemble flag
  model_h40.txt           single LightGBM booster (h=40)
  model_h60_fold{k}.txt   5 LightGBM boosters (h=60), k=0..4 (LOSO fold-k models)
"""
from __future__ import annotations

import importlib.util
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import lightgbm as lgb

WINDOW = 100
HORIZON_LIST = (5, 10, 20, 40, 60)
HORIZON_TO_IDX = {h: i for i, h in enumerate(HORIZON_LIST)}


def _load_t3_module(here: str):
    spec = importlib.util.spec_from_file_location(
        "iter_006_t3_compute", os.path.join(here, "compute.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Predictor:
    def __init__(self) -> None:
        here = os.path.dirname(os.path.abspath(__file__))
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

        # Single boosters (h=40) and ensemble boosters (h=60, 5 models)
        self._boosters: Dict[int, lgb.Booster] = {}
        self._ensemble_boosters: Dict[int, List[lgb.Booster]] = {}
        for hcfg in self._horizons:
            if not hcfg.get("active", True):
                continue
            H = int(hcfg["h"])
            if hcfg.get("ensemble", False):
                folds = []
                for k in range(5):
                    mp = os.path.join(here, f"model_h{H}_fold{k}.txt")
                    if os.path.isfile(mp):
                        folds.append(lgb.Booster(model_file=mp))
                if folds:
                    self._ensemble_boosters[H] = folds
            else:
                mp = os.path.join(here, f"model_h{H}.txt")
                if os.path.isfile(mp):
                    self._boosters[H] = lgb.Booster(model_file=mp)

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

    def _compute_window_features(self, df: pd.DataFrame) -> np.ndarray:
        t3 = self._compute_t3_no_time(df)
        raw_last = df[self._raw_feat_cols].iloc[-1].to_numpy(dtype=np.float32, copy=True)
        if self._amount_delta_idx >= 0:
            v = raw_last[self._amount_delta_idx]
            raw_last[self._amount_delta_idx] = np.sign(v) * np.log1p(abs(v))
        t3_last = t3[self._t3_feat_cols].iloc[-1].to_numpy(dtype=np.float32, copy=False)
        return np.concatenate([raw_last, t3_last]).astype(np.float32, copy=False)

    def _compute_batch_features(self, batches: List[pd.DataFrame]) -> np.ndarray:
        feats = np.empty(
            (len(batches), len(self._raw_feat_cols) + len(self._t3_feat_cols)),
            dtype=np.float32,
        )
        for i, df in enumerate(batches):
            feats[i] = self._compute_window_features(df)
        return feats

    def predict(self, batches: List[pd.DataFrame]) -> List[List[int]]:
        if not batches:
            return []
        feats = self._compute_batch_features(batches)
        B = feats.shape[0]
        out = np.ones((B, 5), dtype=np.int64)
        for hcfg in self._horizons:
            if not hcfg.get("active", True):
                continue
            H = int(hcfg["h"])
            T_thr = float(hcfg.get("T", 0.50))
            delta = float(hcfg.get("delta", 0.15))
            if hcfg.get("ensemble", False):
                folds = self._ensemble_boosters.get(H)
                if not folds:
                    continue
                probs_sum = None
                for booster in folds:
                    p = booster.predict(feats)
                    probs_sum = p if probs_sum is None else probs_sum + p
                probs = probs_sum / len(folds)
            else:
                booster = self._boosters.get(H)
                if booster is None:
                    continue
                probs = booster.predict(feats)
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
    ap.add_argument("--variant", required=True,
                    help="DR variant to use for h_60 ensemble (e.g. aug_b, aug_abc)")
    ap.add_argument("--T", type=float, required=True, help="h_60 threshold T")
    ap.add_argument("--delta", type=float, required=True, help="h_60 threshold delta")
    ap.add_argument("--loso-sum", type=float, required=True,
                    help="h_60 LOSO sum_cum_pnl achieved for this variant (for thresholds.json)")
    ap.add_argument("--out-dir",
                    default=os.path.join(ROOT, "submission", "iter_006_h60_aug"))
    ap.add_argument("--src-iter002",
                    default=os.path.join(ROOT, "submission", "iter_002_lgbm_schemeC"))
    ap.add_argument("--zip-name", default="submission_050625_iter006.zip")
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--no-sanity", action="store_true")
    args = ap.parse_args()

    out_dir = args.out_dir
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"-> building iter_006 in {out_dir} (variant={args.variant})", flush=True)

    # 1) Copy static files from iter_002
    for fn in ("config.json", "compute.py", "requirements.txt"):
        shutil.copy(os.path.join(args.src_iter002, fn), out_dir)

    # 2) Copy h_40 model untouched
    shutil.copy(
        os.path.join(args.src_iter002, "model_h40.txt"),
        os.path.join(out_dir, "model_h40.txt"),
    )

    # 3) Copy 5 LOSO h_60 augment-trained models
    for k in range(5):
        src = os.path.join(HERE, f"loso_model_h60_{args.variant}_held{k}.txt")
        dst = os.path.join(out_dir, f"model_h60_fold{k}.txt")
        if not os.path.exists(src):
            sys.exit(f"missing model: {src}")
        shutil.copy(src, dst)

    # 4) Write Predictor.py
    with open(os.path.join(out_dir, "Predictor.py"), "w") as f:
        f.write(PREDICTOR_TEMPLATE)

    # 5) Build thresholds.json
    # Pull h_40 settings from iter_002 thresholds.json
    with open(os.path.join(args.src_iter002, "thresholds.json")) as f:
        iter002_t = json.load(f)
    h40_cfg = next(h for h in iter002_t["horizons"] if int(h["h"]) == 40)

    new_thr = {
        "_doc": (
            "iter_006: h_5/10/20 deactivated; h_40 from iter_002; "
            f"h_60 = 5-fold ensemble of T26 {args.variant} augment-trained LOSO models"
        ),
        "horizons": [
            {"h": 5,  "T": 0.0, "delta": 0.0, "active": False, "note": "deactivated -> label=1"},
            {"h": 10, "T": 0.0, "delta": 0.0, "active": False, "note": "deactivated -> label=1"},
            {"h": 20, "T": 0.0, "delta": 0.0, "active": False, "note": "deactivated -> label=1"},
            {"h": 40,
             "T": float(h40_cfg["T"]), "delta": float(h40_cfg["delta"]),
             "active": True,
             "note": "copied from iter_002, no DR retrain",
             "loso_best_sum": float(h40_cfg.get("loso_best_sum", 0.0))},
            {"h": 60,
             "T": float(args.T), "delta": float(args.delta),
             "active": True, "ensemble": True,
             "variant": args.variant,
             "loso_best_sum": float(args.loso_sum),
             "note": f"5-fold ensemble (T26 {args.variant} augment-trained)"},
        ],
    }
    with open(os.path.join(out_dir, "thresholds.json"), "w") as f:
        json.dump(new_thr, f, indent=2)

    # 6) Smoke test: ensure Predictor loads and predicts on a 100-row dummy
    if not args.no_sanity:
        print("-> smoke testing Predictor.py ...", flush=True)
        prev_cwd = os.getcwd()
        os.chdir(out_dir)
        try:
            spec = __import__("importlib.util", fromlist=["util"]).util.spec_from_file_location(
                "iter_006_predictor_smoke",
                os.path.join(out_dir, "Predictor.py"),
            )
            mod = __import__("importlib.util", fromlist=["util"]).util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            p = mod.Predictor()
            cfg = json.load(open(os.path.join(out_dir, "config.json")))
            import numpy as np
            import pandas as pd
            rng = np.random.default_rng(0)
            df = pd.DataFrame(
                rng.standard_normal((100, len(cfg["feature"]))).astype(np.float32),
                columns=cfg["feature"],
            )
            out = p.predict([df] * 22)
            assert len(out) == 22, f"expected 22, got {len(out)}"
            for row in out:
                assert len(row) == 5, f"expected 5 horizons, got {len(row)}"
                for v in row:
                    assert v in (0, 1, 2), f"invalid label {v}"
            print(f"  sanity OK: 22/22 predictions, sample={out[0]}", flush=True)
        finally:
            os.chdir(prev_cwd)

    # 7) Zip
    if not args.no_zip:
        zip_inner = os.path.join(out_dir, "submission.zip")
        if os.path.isfile(zip_inner):
            os.remove(zip_inner)
        with zipfile.ZipFile(zip_inner, "w", zipfile.ZIP_DEFLATED) as zf:
            for fn in os.listdir(out_dir):
                if fn.endswith(".zip"):
                    continue
                zf.write(os.path.join(out_dir, fn), arcname=fn)
        zip_outer = os.path.join(ROOT, args.zip_name)
        shutil.copy(zip_inner, zip_outer)
        print(f"-> zipped {zip_inner}", flush=True)
        print(f"-> copied {zip_outer}", flush=True)

    print("\n=== iter_006 build done ===", flush=True)
    for fn in sorted(os.listdir(out_dir)):
        path = os.path.join(out_dir, fn)
        sz = os.path.getsize(path)
        print(f"  {fn:35s}  {sz/1024:>10.1f} KB")


if __name__ == "__main__":
    main()
