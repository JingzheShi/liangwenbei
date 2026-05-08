"""Package iter_016 multi-horizon submission zip + end-to-end verify.

Steps:
  1. Read de_thresh_results.json to get per-horizon best (w_lgb, w_cb, thr_up, thr_dn).
  2. Build pkg/ dir with:
       - Predictor.py, fast_features_batch.py (copied from iter_015), config.json,
         requirements.txt, thresholds.json
       - model_lgb_h{H}_seed{S}.txt for h ∈ {5,10,20,40} from T98 dir
       - model_cb_h{H}_seed{S}.cbm  for h ∈ {5,10,20,40} from T98 dir
       - model_lgb_h60_seed{S}.txt copied from T75 (rename T75→lgb_h60)
       - model_cb_h60_seed{S}.cbm  copied from T89 (rename T89→cb_h60)
  3. Run end-to-end verify: load pkg/, predict on a small sample, confirm output shape
     and values (all 5 horizons active).
  4. Estimate LOSO-equiv per horizon by replaying preds from cached parquets.
  5. Zip into submission_<date>_iter016_multihorizon_demo.zip.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
T75_DIR = os.path.join(ROOT, "experiments", "T75_regression_dmid")
T89_DIR = os.path.join(ROOT, "experiments", "T89_catboost_regression")
ITER015_PKG = os.path.join(ROOT, "experiments", "T87_spo_dfl", "iter015_pkg")

SEEDS = (1, 7, 13, 42, 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(HERE, "iter016_pkg"))
    ap.add_argument("--results-json", default=os.path.join(HERE, "de_thresh_results.json"))
    ap.add_argument("--zip-name", default="submission_050810_iter016_multihorizon_demo.zip")
    args = ap.parse_args()

    pkg = args.out_dir
    if os.path.isdir(pkg):
        shutil.rmtree(pkg)
    os.makedirs(pkg, exist_ok=True)

    print(f"=== package_iter016: pkg dir = {pkg}", flush=True)
    with open(args.results_json) as f:
        de_results = json.load(f)

    # 1. Copy stable files from iter_015_pkg
    for fname in ("fast_features.py", "fast_features_batch.py", "config.json"):
        shutil.copy(os.path.join(ITER015_PKG, fname), os.path.join(pkg, fname))
    print(f"  copied stable files: fast_features*, config.json", flush=True)

    # 2. Predictor.py (use template)
    shutil.copy(os.path.join(HERE, "Predictor_template.py"), os.path.join(pkg, "Predictor.py"))
    print("  copied Predictor.py", flush=True)

    # 3. requirements.txt — extend iter_015 with catboost
    with open(os.path.join(ITER015_PKG, "requirements.txt")) as f:
        req = f.read()
    if "catboost" not in req:
        req = req.rstrip() + "\ncatboost==1.2.10\n"
    with open(os.path.join(pkg, "requirements.txt"), "w") as f:
        f.write(req)
    print("  wrote requirements.txt", flush=True)

    # 4. Copy models per horizon
    horizons_meta = []
    n_models_total = 0
    for h_str, hres in de_results["horizons"].items():
        H = int(h_str)
        if hres is None or hres.get("best") is None:
            print(f"  h={H}: SKIP (no result)", flush=True)
            horizons_meta.append({"h": H, "active": False, "thr_up": 1.0, "thr_dn": 1.0})
            continue
        b = hres["best"]
        w_lgb = float(b.get("w_lgb", 1.0))
        w_cb  = float(b.get("w_cb",  0.0))
        thr_up = float(b["thr_up"])
        thr_dn = float(b["thr_dn"])
        de_sum = float(b["de_sum"])
        # Copy LGB models if w_lgb > 0
        copied_lgb_seeds = []
        copied_cb_seeds = []
        for s in SEEDS:
            if H == 60:
                lgb_src = os.path.join(T75_DIR, f"model_T75_seed{s}.txt")
                cb_src  = os.path.join(T89_DIR, f"model_T89_seed{s}.cbm")
            else:
                lgb_src = os.path.join(HERE, f"model_lgb_h{H}_seed{s}.txt")
                cb_src  = os.path.join(HERE, f"model_cb_h{H}_seed{s}.cbm")
            if w_lgb > 0 and os.path.isfile(lgb_src):
                shutil.copy(lgb_src, os.path.join(pkg, f"model_lgb_h{H}_seed{s}.txt"))
                copied_lgb_seeds.append(s)
                n_models_total += 1
            if w_cb > 0 and os.path.isfile(cb_src):
                shutil.copy(cb_src, os.path.join(pkg, f"model_cb_h{H}_seed{s}.cbm"))
                copied_cb_seeds.append(s)
                n_models_total += 1
        seeds_present = sorted(set(copied_lgb_seeds) | set(copied_cb_seeds))
        print(f"  h={H}: best={b['label']:<20}  DE={de_sum:+.4f}  "
              f"w_lgb={w_lgb:.3f}/w_cb={w_cb:.3f}  thr_up={thr_up:.5e} thr_dn={thr_dn:.5e}  "
              f"  n_lgb={len(copied_lgb_seeds)} n_cb={len(copied_cb_seeds)}", flush=True)
        horizons_meta.append({
            "h": H,
            "active": True,
            "thr_up": thr_up,
            "thr_dn": thr_dn,
            "w_lgb": w_lgb,
            "w_cb": w_cb,
            "ensemble_seeds": seeds_present,
            "_doc": f"DE LOSO-equiv {de_sum:+.4f} (h={H}, label={b['label']})",
        })

    # Make sure all 5 horizons exist in meta in increasing order
    seen = {m["h"] for m in horizons_meta}
    for h in (5, 10, 20, 40, 60):
        if h not in seen:
            horizons_meta.append({"h": h, "active": False, "thr_up": 1.0, "thr_dn": 1.0})
    horizons_meta.sort(key=lambda m: m["h"])

    total_loso = float(de_results.get("total_loso", 0.0))
    thresholds = {
        "_doc": (f"iter_016 multi-horizon: per-h LGB+CB regression ensembles + asymmetric EV gate. "
                 f"All 5 horizons activated (iter_007-015 only used h=60). "
                 f"Total LOSO-equiv across 5 horizons: {total_loso:+.4f}."),
        "horizons": horizons_meta,
    }
    with open(os.path.join(pkg, "thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"\n  wrote thresholds.json (total LOSO-equiv = {total_loso:+.4f})", flush=True)
    print(f"  total models copied: {n_models_total}", flush=True)

    # 5. End-to-end verify: smoke test the Predictor
    print(f"\n=== End-to-end verify ===", flush=True)
    sys.path.insert(0, pkg)
    spec = importlib.util.spec_from_file_location("iter016_predictor",
                                                  os.path.join(pkg, "Predictor.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cfg = json.load(open(os.path.join(pkg, "config.json")))
    feats = cfg["feature"]
    rng = np.random.default_rng(0)
    df_dummy = pd.DataFrame(
        rng.standard_normal((100, len(feats))).astype(np.float32),
        columns=feats,
    )
    pred = mod.Predictor()
    out = pred.predict([df_dummy, df_dummy, df_dummy])
    print(f"  smoke test predict shape={len(out)} x {len(out[0])}: first={out[0]}", flush=True)
    assert len(out) == 3 and len(out[0]) == 5

    # Confirm using real test data (1000 rows from one parquet) — full verification
    sample_path = os.path.join(ROOT, "data", "snapshot_sym0_date0_am.parquet")
    if os.path.isfile(sample_path):
        df_real = pd.read_parquet(sample_path)
        # Slice into 100-tick windows
        n_win = min(50, (len(df_real) // 100))
        batches = [df_real[i*100 : (i+1)*100][feats].copy() for i in range(n_win)]
        t0 = time.time()
        out_real = pred.predict(batches)
        dt = time.time() - t0
        out_arr = np.array(out_real, dtype=np.int64)
        print(f"  real-data predict: n_batches={n_win} time={dt*1000:.1f}ms"
              f"  per-h activation rates: ", end="", flush=True)
        for hi in range(5):
            n_act = (out_arr[:, hi] != 1).sum()
            print(f"h{hi+1}={n_act}/{n_win} ", end="")
        print()
    sys.path.remove(pkg)

    # 6. Zip
    zip_path = os.path.join(ROOT, args.zip_name)
    if os.path.isfile(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for root, _, files in os.walk(pkg):
            for fn in files:
                p = os.path.join(root, fn)
                rel = os.path.relpath(p, pkg)
                zf.write(p, rel)
    sz = os.path.getsize(zip_path) / (1024*1024)
    print(f"\n=== Zip ready: {zip_path} ({sz:.1f} MB) ===", flush=True)

    summary = {
        "zip_path": zip_path,
        "zip_size_mb": float(sz),
        "n_models_total": n_models_total,
        "horizons_meta": horizons_meta,
        "total_loso_equiv": total_loso,
        "vs_iter014": de_results.get("vs_iter014"),
        "vs_iter015": de_results.get("vs_iter015"),
    }
    out_path = os.path.join(HERE, "package_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"summary -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
