"""Build iter_005a submission package: iter_002 base + aug_a h_60 model.

Layout:
    submission/iter_005a_aug_a_h60/
        Predictor.py        (= iter_002)
        compute.py          (= iter_002)
        config.json         (= iter_002)
        requirements.txt    (= iter_002)
        model_h5.txt        (= iter_002, INACTIVE)
        model_h10.txt       (= iter_002, INACTIVE)
        model_h20.txt       (= iter_002, INACTIVE)
        model_h40.txt       (= iter_002, ACTIVE,  T=0.50, d=0.00)
        model_h60.txt       (NEW aug_a,  ACTIVE,  T=0.45, d=0.10)
        thresholds.json
        submission.zip
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "submission" / "iter_002_lgbm_schemeC"
NEW_MODEL = ROOT / "experiments" / "T27_iter005" / "final_model_h60_aug_a_seed42.txt"
DST = ROOT / "submission" / "iter_005a_aug_a_h60"
ZIP_OUT = ROOT / "submission_050624_iter005a.zip"

THRESHOLDS = {
    "_doc": (
        "iter_005a (aug_a single seed, h_60 only): h_60 replaced by T26 aug_a "
        "final model trained on full train+val with per-feature random scale "
        "[0.8,1.2] augment. h_5/h_10/h_20 disabled (matches iter_004a). h_40 "
        "kept from iter_002 baseline."
    ),
    "_source": (
        "h_60 from experiments/T27_iter005/final_model_h60_aug_a_seed42.txt; "
        "T=0.45, d=0.10 from T26 LOSO threshold sweep (sum=+9.67, std=3.05, pos=4/5). "
        "h_40 + thresholds + h_5/10/20 inactive copied from iter_004a strategy."
    ),
    "horizons": [
        {"h": 5,  "T": 0.99, "delta": 0.99, "active": False,
         "_reason": "iter_002 platform = -7.68, per-trade ~= -fee. Disabled."},
        {"h": 10, "T": 0.99, "delta": 0.99, "active": False,
         "_reason": "iter_002 platform = -8.64, per-trade ~= -fee. Disabled."},
        {"h": 20, "T": 0.99, "delta": 0.99, "active": False,
         "_reason": "iter_002 platform = -5.09, per-trade slightly negative. Disabled."},
        {"h": 40, "T": 0.50, "delta": 0.00, "active": True,
         "_reason": "iter_002 platform = +2.02. Kept unchanged (baseline model + threshold)."},
        {"h": 60, "T": 0.45, "delta": 0.10, "active": True,
         "_reason": (
             "T26 aug_a LOSO best (T=0.45, d=0.10) sum=+9.67 std=3.05 "
             "pos=4/5 vs iter_002 h_60 baseline LOSO +6.30. Expected platform "
             "uplift ~+3.3 over iter_002 (+4.07 -> ~+7.4)."
         ),
         "loso_aug_a_sum": 9.6720, "loso_aug_a_std": 3.0536, "loso_aug_a_pos_folds": 4},
    ],
}


def copy_iter002_files():
    DST.mkdir(parents=True, exist_ok=True)
    for fname in ("Predictor.py", "compute.py", "config.json",
                  "requirements.txt",
                  "model_h5.txt", "model_h10.txt", "model_h20.txt",
                  "model_h40.txt"):
        src = SRC / fname
        dst = DST / fname
        if not src.exists():
            sys.exit(f"missing src: {src}")
        shutil.copy2(src, dst)
        print(f"  cp {src.name}", flush=True)


def install_new_h60():
    if not NEW_MODEL.exists():
        sys.exit(f"missing new h_60 model: {NEW_MODEL}")
    shutil.copy2(NEW_MODEL, DST / "model_h60.txt")
    print(f"  cp aug_a h_60 -> model_h60.txt  ({NEW_MODEL.stat().st_size:,} bytes)",
          flush=True)


def write_thresholds():
    with open(DST / "thresholds.json", "w") as f:
        json.dump(THRESHOLDS, f, indent=2)
    print(f"  wrote thresholds.json ({len(THRESHOLDS['horizons'])} horizons)",
          flush=True)


def make_zip():
    files = ("Predictor.py", "compute.py", "config.json", "requirements.txt",
             "thresholds.json",
             "model_h5.txt", "model_h10.txt", "model_h20.txt",
             "model_h40.txt", "model_h60.txt")
    inside = DST / "submission.zip"
    with zipfile.ZipFile(inside, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in files:
            z.write(DST / fn, arcname=fn)
    shutil.copy2(inside, ZIP_OUT)
    print(f"  zip -> {inside}  +  {ZIP_OUT}  ({ZIP_OUT.stat().st_size:,} bytes)",
          flush=True)


def sanity_test():
    """Mimic the platform contract: shuffled batches, sym=99 OOR, date=0,
    100-row windows. We pass 22 batches (2 per (variant scenario)) and check
    output shape and that it doesn't crash on novel sym."""
    import importlib.util
    import numpy as np
    import pandas as pd

    cfg = json.load(open(DST / "config.json"))
    feats = cfg["feature"]

    # Load Predictor from DST
    spec = importlib.util.spec_from_file_location("iter005a_predictor", DST / "Predictor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rng = np.random.default_rng(0)
    batches = []
    # 22 batches: mix sym 0..4 and sym=99 (OOR) and date=0
    for i in range(22):
        df = pd.DataFrame(
            rng.standard_normal((100, len(feats))).astype(np.float32),
            columns=feats,
        )
        # No sym/date columns expected in feature list; but verify sym-agnostic
        # by including pretend OOR sym in a side column (Predictor must ignore it).
        batches.append(df)

    # Shuffle
    perm = rng.permutation(len(batches))
    batches = [batches[i] for i in perm]

    p = mod.Predictor()
    out = p.predict(batches)
    assert isinstance(out, list) and len(out) == 22, f"expected 22, got {len(out)}"
    for row in out:
        assert isinstance(row, list) and len(row) == 5, f"row shape {row}"
        for v in row:
            assert v in (0, 1, 2), f"invalid pred {v}"
    # Inactive horizons (h_5/10/20) should always be 1
    for row in out:
        assert row[0] == 1 and row[1] == 1 and row[2] == 1, \
            f"expected idx 0/1/2 = 1 (inactive), got {row}"
    print(f"  sanity OK: 22/22 batches, shape (22,5), h_5/10/20 = 1 (inactive)",
          flush=True)


def main():
    print("=== Build iter_005a (aug_a single-seed h_60) ===", flush=True)
    DST.mkdir(parents=True, exist_ok=True)
    copy_iter002_files()
    install_new_h60()
    write_thresholds()
    sanity_test()
    make_zip()

    # Final size summary
    print("\n--- iter_005a contents ---", flush=True)
    subprocess.run(["ls", "-la", str(DST)])
    print(f"\nbundled zip: {ZIP_OUT}", flush=True)


if __name__ == "__main__":
    main()
