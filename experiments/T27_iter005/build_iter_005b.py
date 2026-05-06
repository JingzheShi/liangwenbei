"""Build iter_005b submission: aug_a 5-seed ensemble for h_60 (h_40 from iter_002).

Layout:
    submission/iter_005b_aug_a_5seed_h60/
        Predictor.py             (= T27 Predictor_iter_005b.py with ensemble)
        compute.py               (= iter_002)
        config.json              (= iter_002)
        requirements.txt         (= iter_002)
        model_h5.txt             (= iter_002, INACTIVE)
        model_h10.txt            (= iter_002, INACTIVE)
        model_h20.txt            (= iter_002, INACTIVE)
        model_h40.txt            (= iter_002, ACTIVE)
        model_h60_seed42.txt     (NEW aug_a final, ACTIVE — averaged)
        model_h60_seed1.txt
        model_h60_seed7.txt
        model_h60_seed13.txt
        model_h60_seed100.txt
        thresholds.json
        submission.zip

Threshold (T, delta) is read from sweep_5seed_ensemble_results.json — pass
--T-override / --d-override to force.
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
SRC = ROOT / "submission" / "iter_002_lgbm_schemeC"
T27 = ROOT / "experiments" / "T27_iter005"
DST = ROOT / "submission" / "iter_005b_aug_a_5seed_h60"
ZIP_OUT = ROOT / "submission_050626_iter005b.zip"

DEFAULT_SEEDS = (42, 1, 7, 13, 100)


def _load_best_threshold(seeds):
    p = T27 / "sweep_5seed_ensemble_results.json"
    if not p.exists():
        sys.exit(f"missing {p} -- run sweep_5seed_ensemble.py first")
    with open(p) as f:
        d = json.load(f)
    b = d["best"]
    return float(b["T"]), float(b["delta"]), float(b["sum_cum_pnl"]), float(b["std_cum_pnl"]), int(b["n_pos_folds"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,1,7,13,100")
    ap.add_argument("--T-override", type=float, default=None)
    ap.add_argument("--d-override", type=float, default=None)
    ap.add_argument("--allow-missing-final", action="store_true",
                    help="if a final_model file is missing, skip it (else error)")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"=== Build iter_005b (aug_a {len(seeds)}-seed ensemble h_60) ===", flush=True)

    # Locate final aug_a model files
    seed_models = []
    for s in seeds:
        p = T27 / f"final_model_h60_aug_a_seed{s}.txt"
        if not p.exists():
            msg = f"missing final aug_a model: {p}"
            if args.allow_missing_final:
                print(f"  WARN: {msg} -- skipping", flush=True)
                continue
            sys.exit(msg)
        seed_models.append((s, p))
    if not seed_models:
        sys.exit("no aug_a final models found")
    print(f"  found {len(seed_models)} final models: seeds={[s for s,_ in seed_models]}",
          flush=True)

    # Resolve threshold
    if args.T_override is not None and args.d_override is not None:
        T_ens, d_ens = float(args.T_override), float(args.d_override)
        sweep_meta = {"T": T_ens, "delta": d_ens, "_source": "manual override"}
    else:
        T_ens, d_ens, sum_pnl, std_pnl, pos_folds = _load_best_threshold(seeds)
        sweep_meta = {"T": T_ens, "delta": d_ens,
                      "sum_cum_pnl": sum_pnl, "std_cum_pnl": std_pnl,
                      "n_pos_folds": pos_folds,
                      "_source": "T27 5-seed ensemble OOF sweep"}

    print(f"  threshold: T={T_ens:.2f} d={d_ens:.2f} ({sweep_meta.get('_source')})",
          flush=True)

    # Stage destination
    DST.mkdir(parents=True, exist_ok=True)
    # Clean any leftover model_h60* files before installing fresh ones
    for old in DST.glob("model_h60*.txt"):
        old.unlink()

    # Copy iter_002 files (except Predictor.py and model_h60.txt)
    for fname in ("compute.py", "config.json", "requirements.txt",
                  "model_h5.txt", "model_h10.txt", "model_h20.txt",
                  "model_h40.txt"):
        src = SRC / fname
        if not src.exists():
            sys.exit(f"missing src: {src}")
        shutil.copy2(src, DST / fname)
        print(f"  cp {src.name}", flush=True)

    # Custom Predictor (handles ensemble of model_h60_seed*.txt)
    shutil.copy2(T27 / "Predictor_iter_005b.py", DST / "Predictor.py")
    print(f"  cp Predictor_iter_005b.py -> Predictor.py (ensemble-aware)",
          flush=True)

    # Install h_60 ensemble models
    for s, p in seed_models:
        out = DST / f"model_h60_seed{s}.txt"
        shutil.copy2(p, out)
        print(f"  cp aug_a h_60 seed{s} -> {out.name}  ({p.stat().st_size:,} bytes)",
              flush=True)

    # Thresholds
    THRESHOLDS = {
        "_doc": (
            f"iter_005b ({len(seed_models)}-seed aug_a ensemble h_60): h_60 = "
            f"mean of {len(seed_models)} aug_a final models. h_5/h_10/h_20 disabled "
            "(matches iter_004a). h_40 from iter_002 baseline."
        ),
        "_source": "T27 5-seed ensemble OOF threshold sweep",
        "horizons": [
            {"h": 5,  "T": 0.99, "delta": 0.99, "active": False,
             "_reason": "iter_002 platform = -7.68. Disabled."},
            {"h": 10, "T": 0.99, "delta": 0.99, "active": False,
             "_reason": "iter_002 platform = -8.64. Disabled."},
            {"h": 20, "T": 0.99, "delta": 0.99, "active": False,
             "_reason": "iter_002 platform = -5.09. Disabled."},
            {"h": 40, "T": 0.50, "delta": 0.00, "active": True,
             "_reason": "iter_002 platform = +2.02. Kept unchanged."},
            {"h": 60, "T": T_ens, "delta": d_ens, "active": True,
             "_reason": (
                 f"T27 aug_a {len(seed_models)}-seed ensemble best from OOF sweep. "
                 "Single-seed aug_a LOSO best (T=0.45,d=0.10) was sum=+9.67."
             ),
             "ensemble_meta": sweep_meta,
             "ensemble_seeds": [s for s, _ in seed_models]},
        ],
    }
    with open(DST / "thresholds.json", "w") as f:
        json.dump(THRESHOLDS, f, indent=2)
    print(f"  wrote thresholds.json", flush=True)

    # Sanity test (22 batches, shuffled, OOR sym, date=0)
    import importlib.util
    import numpy as np
    import pandas as pd

    cfg = json.load(open(DST / "config.json"))
    feats = cfg["feature"]
    spec = importlib.util.spec_from_file_location("iter005b_predictor", DST / "Predictor.py")
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
    print(f"  sanity OK: 22/22 batches, shape (22,5), h_5/10/20 = 1 (inactive)",
          flush=True)

    # Zip
    files = ["Predictor.py", "compute.py", "config.json", "requirements.txt",
             "thresholds.json",
             "model_h5.txt", "model_h10.txt", "model_h20.txt",
             "model_h40.txt"]
    files += [f"model_h60_seed{s}.txt" for s, _ in seed_models]
    inside = DST / "submission.zip"
    with zipfile.ZipFile(inside, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in files:
            z.write(DST / fn, arcname=fn)
    shutil.copy2(inside, ZIP_OUT)
    print(f"  zip -> {inside}  +  {ZIP_OUT}  ({ZIP_OUT.stat().st_size:,} bytes)",
          flush=True)

    print("\n--- iter_005b contents ---", flush=True)
    subprocess.run(["ls", "-la", str(DST)])
    print(f"\nbundled zip: {ZIP_OUT}", flush=True)


if __name__ == "__main__":
    main()
