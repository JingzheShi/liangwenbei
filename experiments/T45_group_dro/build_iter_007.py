"""Build iter_007 submission: T45 Group DRO 5-seed h_60 ensemble + DE thresh.

Same layout/template as iter_006:
    Predictor.py             (= iter_006 Predictor — asym-aware, sym-agnostic)
    compute.py               (= iter_002)
    config.json              (= iter_002, 154 raw feats)
    requirements.txt         (= iter_002)
    model_h5/h10/h20/h40.txt (from iter_002 — h_5/10/20 INACTIVE, h_40 ACTIVE)
    model_h60_seed{42,1,7,13,100}.txt (= T45 dro5seed_model_seed{S}_held*.txt)
    thresholds.json (asym for h_60, from T45 5-seed DE)
    submission.zip

The 5 h_60 models are trained on full-data LOSO folds; for iter_007 we use
the 5 per-seed models trained on **all 5 syms held out together is wrong** —
correction: T45 trained one model PER (seed, held_sym). For inference we don't
have a held-out sym, so we ensemble across all 25 (seed × held_sym) ... but
that's a 25-model ensemble, ~5x slower than iter_006.

Two options:
  A) Use 5 models per seed averaged (= 25 total); slower but uses every fold.
  B) Pick one held_sym per seed (e.g., held=2 — typically the best fold).
We default to **(A) all 25 models averaged** for max diversity → max OOF gain
should also transfer (and inference cost is fine since 25 LightGBM calls ~ms).

Threshold from T45 dro5seed_thresh_results.json best DE run.
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
SRC_ITER006 = ROOT / "submission" / "iter_006_aug_a_5seed_h60_asymthresh"
T45 = ROOT / "experiments" / "T45_group_dro"
DST = ROOT / "submission" / "iter_007_group_dro_5seed"
ZIP_OUT = ROOT / "submission_050707_iter007.zip"

DEFAULT_SEEDS = (42, 1, 7, 13, 100)
DEFAULT_TAG = "dro5seed"


def _load_de_threshold():
    p = T45 / "dro5seed_thresh_results.json"
    if not p.exists():
        sys.exit(f"missing {p}")
    with open(p) as f:
        d = json.load(f)
    b = d["fresh_DE_best"]
    return (float(b["T_up"]), float(b["T_dn"]),
            float(b["d_up"]), float(b["d_dn"]),
            float(b["sum_cum_pnl"]), float(b["std_cum_pnl"]),
            int(b["n_pos_folds"]), b["per_fold_pnl"],
            d.get("raw_argmax_sum"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--tag", default=DEFAULT_TAG)
    ap.add_argument("--mode", choices=("per_seed_per_held", "per_seed_one_held"),
                    default="per_seed_per_held",
                    help="per_seed_per_held: 25 models (5 seeds * 5 held_syms). "
                         "per_seed_one_held: 5 models (1 per seed, fixed held_sym)")
    ap.add_argument("--fixed-held", type=int, default=2,
                    help="held_sym to use when mode=per_seed_one_held")
    ap.add_argument("--T-up", type=float, default=None)
    ap.add_argument("--T-dn", type=float, default=None)
    ap.add_argument("--d-up", type=float, default=None)
    ap.add_argument("--d-dn", type=float, default=None)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"=== Build iter_007 (T45 Group DRO {len(seeds)}-seed h_60 + asym thresh) ===",
          flush=True)

    # Locate model files
    seed_models = []
    if args.mode == "per_seed_one_held":
        for s in seeds:
            p = T45 / f"{args.tag}_model_seed{s}_held{args.fixed_held}.txt"
            if not p.exists():
                sys.exit(f"missing {p}")
            seed_models.append((s, args.fixed_held, p))
    else:  # per_seed_per_held
        for s in seeds:
            for k in range(5):
                p = T45 / f"{args.tag}_model_seed{s}_held{k}.txt"
                if not p.exists():
                    sys.exit(f"missing {p}")
                seed_models.append((s, k, p))
    print(f"  found {len(seed_models)} h_60 models, mode={args.mode}", flush=True)

    # Threshold
    if (args.T_up is not None and args.T_dn is not None and
            args.d_up is not None and args.d_dn is not None):
        Tu, Td, du, dd = args.T_up, args.T_dn, args.d_up, args.d_dn
        meta = {"T_up": Tu, "T_dn": Td, "d_up": du, "d_dn": dd, "_source": "manual override"}
    else:
        Tu, Td, du, dd, sumP, stdP, npf, pf, raw_sum = _load_de_threshold()
        meta = {"T_up": Tu, "T_dn": Td, "d_up": du, "d_dn": dd,
                "OOF_sum_cum_pnl": sumP, "OOF_std_cum_pnl": stdP,
                "OOF_n_pos_folds": npf, "OOF_per_fold_pnl": pf,
                "OOF_raw_argmax_sum": raw_sum,
                "_source": "T45 differential_evolution best on 5-seed DRO ensemble"}
    print(f"  h_60 asym thresh: T_up={Tu:.4f} T_dn={Td:.4f} d_up={du:.4f} d_dn={dd:.4f}",
          flush=True)

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

    # Predictor (= iter_006 — asym-aware, ensembles all model_h60_*.txt)
    shutil.copy2(SRC_ITER006 / "Predictor.py", DST / "Predictor.py")
    print(f"  cp iter_006 Predictor.py", flush=True)

    # h_60 ensemble models — name with seed+held to allow glob
    for s, k, p in seed_models:
        out = DST / f"model_h60_seed{s}_held{k}.txt"
        shutil.copy2(p, out)
    print(f"  cp {len(seed_models)} h_60 models -> model_h60_seed*_held*.txt",
          flush=True)

    # thresholds.json
    THRESHOLDS = {
        "_doc": (
            f"iter_007: T45 Group DRO {len(seed_models)}-model h_60 ensemble"
            f" (seeds={seeds}, mode={args.mode}) + asymmetric DE-optimized gating."
            " h_5/h_10/h_20 disabled (matches iter_006). h_40 from iter_002 baseline."
        ),
        "_source": "T45 Group DRO + 4D differential_evolution PnL maximization",
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
                 "T45 Group DRO 5-seed ensemble + 4D DE asymmetric optimum. "
                 f"OOF sum cum_pnl = {meta.get('OOF_sum_cum_pnl', 'N/A')} "
                 f"(vs iter_006 +13.61)."
             ),
             "asym_meta": meta,
             "ensemble_seeds": seeds,
             "ensemble_mode": args.mode,
             "ensemble_n_models": len(seed_models)},
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
    print(f"  sanity OK: 22/22 batches, shape (22,5), h_5/10/20=1 (inactive)",
          flush=True)

    # Zip
    files = ["Predictor.py", "compute.py", "config.json", "requirements.txt",
             "thresholds.json",
             "model_h5.txt", "model_h10.txt", "model_h20.txt", "model_h40.txt"]
    files += [f"model_h60_seed{s}_held{k}.txt" for s, k, _ in seed_models]
    inside = DST / "submission.zip"
    with zipfile.ZipFile(inside, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in files:
            z.write(DST / fn, arcname=fn)
    shutil.copy2(inside, ZIP_OUT)
    print(f"  zip -> {inside}  +  {ZIP_OUT}  ({ZIP_OUT.stat().st_size:,} bytes)",
          flush=True)

    print("\n--- iter_007 contents ---", flush=True)
    subprocess.run(["ls", "-la", str(DST)])
    print(f"\nbundled zip: {ZIP_OUT}", flush=True)


if __name__ == "__main__":
    main()
