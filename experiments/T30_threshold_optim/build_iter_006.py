"""Build iter_006 submission: iter_005b (5-seed aug_a ensemble for h_60)
+ T30 asymmetric DE-optimized thresholds for h_60.

Layout (DST = submission/iter_006_aug_a_5seed_h60_asymthresh):
    Predictor.py             (= T30 Predictor_iter_006.py — asym-aware)
    compute.py               (= iter_002)
    config.json              (= iter_002)
    requirements.txt         (= iter_002)
    model_h5.txt   (INACTIVE = iter_002)
    model_h10.txt  (INACTIVE = iter_002)
    model_h20.txt  (INACTIVE = iter_002)
    model_h40.txt  (ACTIVE   = iter_002)
    model_h60_seed{42,1,7,13,100}.txt (ACTIVE = T27 aug_a finals)
    thresholds.json (asym for h_60)
    submission.zip

Threshold for h_60 read from pnl_search_results.json (DE) by default; override
with --T-up/--T-dn/--d-up/--d-dn.
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
SRC_ITER002 = ROOT / "submission" / "iter_002_lgbm_schemeC"
T27 = ROOT / "experiments" / "T27_iter005"
T30 = ROOT / "experiments" / "T30_threshold_optim"
DST = ROOT / "submission" / "iter_006_aug_a_5seed_h60_asymthresh"
ZIP_OUT = ROOT / "submission_050606_iter006.zip"

DEFAULT_SEEDS = (42, 1, 7, 13, 100)


def _load_de_threshold():
    p = T30 / "pnl_search_results.json"
    if not p.exists():
        sys.exit(f"missing {p}")
    with open(p) as f:
        d = json.load(f)
    b = d["best"]
    return (float(b["T_up"]), float(b["T_dn"]),
            float(b["d_up"]), float(b["d_dn"]),
            float(b["sum_cum_pnl"]), float(b["std_cum_pnl"]),
            int(b["n_pos_folds"]), b["per_fold_pnl"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,1,7,13,100")
    ap.add_argument("--T-up", type=float, default=None)
    ap.add_argument("--T-dn", type=float, default=None)
    ap.add_argument("--d-up", type=float, default=None)
    ap.add_argument("--d-dn", type=float, default=None)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"=== Build iter_006 (aug_a 5-seed h_60 + asym thresh) ===", flush=True)

    # Locate final aug_a models
    seed_models = []
    for s in seeds:
        p = T27 / f"final_model_h60_aug_a_seed{s}.txt"
        if not p.exists():
            sys.exit(f"missing {p}")
        seed_models.append((s, p))
    print(f"  found {len(seed_models)} h_60 final models: seeds={seeds}", flush=True)

    # Threshold
    if (args.T_up is not None and args.T_dn is not None and
            args.d_up is not None and args.d_dn is not None):
        Tu, Td, du, dd = args.T_up, args.T_dn, args.d_up, args.d_dn
        meta = {"T_up": Tu, "T_dn": Td, "d_up": du, "d_dn": dd, "_source": "manual override"}
    else:
        Tu, Td, du, dd, sumP, stdP, npf, pf = _load_de_threshold()
        meta = {"T_up": Tu, "T_dn": Td, "d_up": du, "d_dn": dd,
                "OOF_sum_cum_pnl": sumP, "OOF_std_cum_pnl": stdP,
                "OOF_n_pos_folds": npf, "OOF_per_fold_pnl": pf,
                "_source": "T30 differential_evolution best (5 seeds convergent)"}
    print(f"  h_60 asym thresh: T_up={Tu:.4f} T_dn={Td:.4f} d_up={du:.4f} d_dn={dd:.4f}", flush=True)

    # Stage destination
    DST.mkdir(parents=True, exist_ok=True)
    for old in DST.glob("model_h60*.txt"):
        old.unlink()

    # Copy iter_002 base
    for fname in ("compute.py", "config.json", "requirements.txt",
                  "model_h5.txt", "model_h10.txt", "model_h20.txt",
                  "model_h40.txt"):
        src = SRC_ITER002 / fname
        if not src.exists():
            sys.exit(f"missing src: {src}")
        shutil.copy2(src, DST / fname)
        print(f"  cp {src.name}", flush=True)

    # Custom Predictor
    shutil.copy2(T30 / "Predictor_iter_006.py", DST / "Predictor.py")
    print(f"  cp Predictor_iter_006.py -> Predictor.py", flush=True)

    # h_60 ensemble models
    for s, p in seed_models:
        out = DST / f"model_h60_seed{s}.txt"
        shutil.copy2(p, out)
        print(f"  cp aug_a h_60 seed{s} -> {out.name}  ({p.stat().st_size:,} bytes)", flush=True)

    # thresholds.json
    THRESHOLDS = {
        "_doc": (
            f"iter_006: iter_005b base ({len(seed_models)}-seed aug_a ensemble for h_60)"
            " + T30 asymmetric (T_up,T_dn,d_up,d_dn) DE-optimized gating."
            " h_5/h_10/h_20 disabled (matches iter_005b). h_40 from iter_002 baseline."
        ),
        "_source": "T30 differential_evolution PnL maximization on 5-seed aug_a OOF",
        "horizons": [
            {"h": 5,  "T": 0.99, "delta": 0.99, "active": False,
             "_reason": "iter_002 platform = -7.68. Disabled."},
            {"h": 10, "T": 0.99, "delta": 0.99, "active": False,
             "_reason": "iter_002 platform = -8.64. Disabled."},
            {"h": 20, "T": 0.99, "delta": 0.99, "active": False,
             "_reason": "iter_002 platform = -5.09. Disabled."},
            {"h": 40, "T": 0.50, "delta": 0.00, "active": True,
             "_reason": "iter_002 platform = +2.02. Kept unchanged (symmetric)."},
            {"h": 60, "T_up": Tu, "T_dn": Td, "d_up": du, "d_dn": dd,
             "active": True,
             "_reason": (
                 "T30 asymmetric DE optimum on 5-seed aug_a OOF. Vs iter_005b "
                 "(T=0.45,d=0.10) sum=+11.46 -> asym sum=+13.61."
             ),
             "asym_meta": meta,
             "ensemble_seeds": [s for s, _ in seed_models]},
        ],
    }
    with open(DST / "thresholds.json", "w") as f:
        json.dump(THRESHOLDS, f, indent=2)
    print(f"  wrote thresholds.json", flush=True)

    # Sanity (22 batches, shuffled, OOR-safe — no sym/date)
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
    print(f"  sanity OK: 22/22 batches, shape (22,5), h_5/10/20=1 (inactive)", flush=True)

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
    print(f"  zip -> {inside}  +  {ZIP_OUT}  ({ZIP_OUT.stat().st_size:,} bytes)", flush=True)

    print("\n--- iter_006 contents ---", flush=True)
    subprocess.run(["ls", "-la", str(DST)])
    print(f"\nbundled zip: {ZIP_OUT}", flush=True)


if __name__ == "__main__":
    main()
