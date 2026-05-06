"""Build iter_006 submission: Scheme K (ReVol+SG) + aug_a 5-seed ensemble for h_60.

Layout:
    submission/iter_006_scheme_k_aug_a_5seed/
        Predictor.py             (= T35 Predictor_iter_006.py with extras)
        revol.py                 (= T35 revol.py — used by Predictor)
        savgol.py                (= T35 savgol.py — used by Predictor)
        compute.py               (= iter_002, T3 features)
        config.json              (= iter_002, raw feature list)
        requirements.txt         (= iter_002 + scipy)
        model_h5.txt             (= iter_002, INACTIVE)
        model_h10.txt            (= iter_002, INACTIVE)
        model_h20.txt            (= iter_002, INACTIVE)
        model_h40.txt            (= iter_002, ACTIVE — 223-d Scheme C)
        model_h60_seed{42,1,7,13,100}.txt   (NEW Scheme K aug_a final, 255-d)
        thresholds.json
        submission.zip

Threshold for h_60 read from threshold_results_*.json (5-seed ensemble best).

Sanity test: 22 batches (shuffled, OOR sym=99, date=0) -> 22/22 outputs.
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
SRC = ROOT / "submission" / "iter_002_lgbm_schemeC"
DST = ROOT / "submission" / "iter_006_scheme_k_aug_a_5seed"
ZIP_OUT = ROOT / "submission_050630_iter006.zip"

DEFAULT_SEEDS = (42, 1, 7, 13, 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,1,7,13,100")
    ap.add_argument("--T", type=float, required=True,
                    help="h_60 threshold T from sweep")
    ap.add_argument("--delta", type=float, required=True,
                    help="h_60 delta from sweep")
    ap.add_argument("--ens-meta", default=None,
                    help="optional path to threshold_results_*.json for metadata")
    ap.add_argument("--allow-missing-final", action="store_true")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"=== Build iter_006 (Scheme K + aug_a {len(seeds)}-seed ensemble h_60) ===",
          flush=True)

    # Locate final aug_a Scheme K model files (trained on full train+val).
    # We expect them at HERE/final_model_h60_aug_a_seed{S}.txt
    seed_models = []
    for s in seeds:
        p = HERE / f"final_model_h60_aug_a_seed{s}.txt"
        if not p.exists():
            msg = f"missing final aug_a Scheme K model: {p}"
            if args.allow_missing_final:
                print(f"  WARN: {msg} -- skipping", flush=True)
                continue
            sys.exit(msg)
        seed_models.append((s, p))
    if not seed_models:
        sys.exit("no aug_a Scheme K final models found")
    print(f"  found {len(seed_models)} final models: seeds={[s for s, _ in seed_models]}",
          flush=True)

    sweep_meta = {"T": args.T, "delta": args.delta, "_source": "manual override"}
    if args.ens_meta and os.path.exists(args.ens_meta):
        with open(args.ens_meta) as f:
            d = json.load(f)
        if "ensemble" in d:
            sweep_meta = {
                "T": args.T, "delta": args.delta,
                "sum_cum_pnl": d["ensemble"]["best"]["sum_cum_pnl"],
                "n_pos_folds": d["ensemble"]["best"]["n_pos_folds"],
                "_source": args.ens_meta,
            }

    # Stage destination
    DST.mkdir(parents=True, exist_ok=True)
    for old in DST.glob("model_h60*.txt"):
        old.unlink()

    # Copy iter_002 base files (compute.py, config.json, requirements.txt, model_h5/10/20/40)
    for fname in ("compute.py", "config.json", "requirements.txt",
                  "model_h5.txt", "model_h10.txt", "model_h20.txt",
                  "model_h40.txt"):
        src = SRC / fname
        if not src.exists():
            sys.exit(f"missing src: {src}")
        shutil.copy2(src, DST / fname)
        print(f"  cp {src.name}", flush=True)

    # Patch requirements.txt to include scipy if missing
    req_path = DST / "requirements.txt"
    req_text = req_path.read_text()
    if "scipy" not in req_text:
        with open(req_path, "a") as f:
            f.write("\nscipy>=1.10\n")
        print("  added scipy to requirements.txt", flush=True)

    # Custom Predictor + helper modules
    shutil.copy2(HERE / "Predictor_iter_006.py", DST / "Predictor.py")
    shutil.copy2(HERE / "revol.py", DST / "revol.py")
    shutil.copy2(HERE / "savgol.py", DST / "savgol.py")
    print("  cp Predictor_iter_006.py + revol.py + savgol.py", flush=True)

    # Install h_60 ensemble models
    for s, p in seed_models:
        out = DST / f"model_h60_seed{s}.txt"
        shutil.copy2(p, out)
        print(f"  cp Scheme K aug_a h_60 seed{s} -> {out.name}  ({p.stat().st_size:,} bytes)",
              flush=True)

    # Thresholds
    THRESHOLDS = {
        "_doc": (
            f"iter_006: Scheme K (ReVol+SG, 255-d) + aug_a {len(seed_models)}-seed "
            f"ensemble for h_60. h_5/10/20 disabled (matches iter_005b). "
            f"h_40 from iter_002 baseline (223-d Scheme C)."
        ),
        "_source": "T35 5-seed Scheme K ensemble OOF threshold sweep",
        "horizons": [
            {"h": 5,  "T": 0.99, "delta": 0.99, "active": False,
             "_reason": "iter_002 platform = -7.68. Disabled."},
            {"h": 10, "T": 0.99, "delta": 0.99, "active": False,
             "_reason": "iter_002 platform = -8.64. Disabled."},
            {"h": 20, "T": 0.99, "delta": 0.99, "active": False,
             "_reason": "iter_002 platform = -5.09. Disabled."},
            {"h": 40, "T": 0.50, "delta": 0.00, "active": True,
             "_reason": "iter_002 platform = +2.02. Kept unchanged (uses 223-d Scheme C model)."},
            {"h": 60, "T": float(args.T), "delta": float(args.delta), "active": True,
             "_reason": (
                 f"T35 Scheme K aug_a {len(seed_models)}-seed ensemble best from OOF sweep."
             ),
             "ensemble_meta": sweep_meta,
             "ensemble_seeds": [s for s, _ in seed_models]},
        ],
    }
    with open(DST / "thresholds.json", "w") as f:
        json.dump(THRESHOLDS, f, indent=2)
    print(f"  wrote thresholds.json", flush=True)

    # Sanity test (22 batches, shuffled). Defensive: include OOR-like rng input.
    import importlib.util
    import numpy as np
    import pandas as pd

    cfg = json.load(open(DST / "config.json"))
    feats = cfg["feature"]
    spec = importlib.util.spec_from_file_location("iter006_predictor", DST / "Predictor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rng = np.random.default_rng(0)
    batches = []
    for _ in range(22):
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
    files = ["Predictor.py", "revol.py", "savgol.py",
             "compute.py", "config.json", "requirements.txt",
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

    print("\n--- iter_006 contents ---", flush=True)
    subprocess.run(["ls", "-la", str(DST)])
    print(f"\nbundled zip: {ZIP_OUT}", flush=True)


if __name__ == "__main__":
    main()
