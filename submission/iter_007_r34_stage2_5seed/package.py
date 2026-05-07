"""Build iter_007 submission package.

Steps:
  1. Reads DE result file (de_5seed_results.json) for h_60 thresholds
  2. Writes thresholds.json (h_5/10/20 disabled, h_40 from iter_006, h_60 from DE)
  3. Verifies all 5 model_h60_seed{1,7,13,42,100}.txt files are present
  4. Verifies model_h40.txt is present
  5. Runs the smoke test on the Predictor
  6. Runs the inference vs cache match test (sanity check)
  7. Zips into submission.zip
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
T51_DIR = ROOT / "experiments" / "T51_r34_stage2"


def _read_de(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--de-results", default=str(T51_DIR / "de_5seed_results.json"))
    ap.add_argument("--zip-out", default=str(HERE / "submission.zip"))
    ap.add_argument("--source", choices=["loso", "full"], default="full",
                    help="Which models to package: 'full' = trained on train+val (recommended); "
                         "'loso' = average of 5 LOSO models per seed (less ideal)")
    args = ap.parse_args()

    de_path = Path(args.de_results)
    if not de_path.exists():
        sys.exit(f"!! DE results missing: {de_path}")
    de = _read_de(de_path)
    best = de["de_best"] if "de_best" in de else de.get("best", de)
    if "best_thresh" in de:
        T_up, T_dn, d_up, d_dn = de["best_thresh"]
        loso_sum = float(de["best_sum"])
    else:
        T_up = float(best["T_up"]); T_dn = float(best["T_dn"])
        d_up = float(best["d_up"]); d_dn = float(best["d_dn"])
        loso_sum = float(best["sum_cum_pnl"])
    print(f"  DE thresholds: T_up={T_up:.4f} T_dn={T_dn:.4f} d_up={d_up:.4f} d_dn={d_dn:.4f}")
    print(f"  LOSO sum (with thresh): {loso_sum:+.4f}")

    # Build thresholds.json
    thresh = {
        "_doc": "iter_007: T51 R34 stage 2 (327-d) 5-seed aug_a ensemble for h_60. "
                "DE 4D asymmetric thresh from 5-seed averaged OOF prob. "
                "h_5/h_10/h_20 disabled (matches iter_006). h_40 = iter_002 baseline (kept).",
        "_loso_sum_with_thresh": loso_sum,
        "horizons": [
            {"h": 5,  "T": 0.99, "delta": 0.99, "active": False, "_reason": "iter_002 platform = -7.68"},
            {"h": 10, "T": 0.99, "delta": 0.99, "active": False, "_reason": "iter_002 platform = -8.64"},
            {"h": 20, "T": 0.99, "delta": 0.99, "active": False, "_reason": "iter_002 platform = -5.09"},
            {"h": 40, "T": 0.5,  "delta": 0.0,  "active": True,  "_reason": "iter_002 baseline (kept)"},
            {"h": 60, "T_up": T_up, "T_dn": T_dn, "d_up": d_up, "d_dn": d_dn,
             "active": True, "_reason": f"T51b DE 5-seed asym optimum, LOSO sum={loso_sum:+.4f}"},
        ],
    }
    with open(HERE / "thresholds.json", "w") as f:
        json.dump(thresh, f, indent=2)
    print(f"  wrote thresholds.json")

    # Copy h60 models (5 seeds)
    seeds = (1, 7, 13, 42, 100)
    for seed in seeds:
        if args.source == "full":
            src = T51_DIR / f"full_model_h60_seed{seed}.txt"
        else:
            # LOSO ensemble: not directly usable as a single model file. Skip.
            sys.exit("source=loso not supported in package; use --source full after running train_full_data_h60.py")
        if not src.exists():
            sys.exit(f"!! missing model: {src}")
        dst = HERE / f"model_h60_seed{seed}.txt"
        shutil.copy2(src, dst)
        print(f"  {src.name} -> {dst.name}")

    # Verify h40 (already copied at template setup time)
    h40 = HERE / "model_h40.txt"
    if not h40.exists():
        sys.exit(f"!! model_h40.txt missing")
    print(f"  model_h40.txt ok")

    # Smoke test the Predictor
    print(f"  smoke test ...")
    r = subprocess.run([sys.executable, str(HERE / "Predictor.py")], capture_output=True, text=True, cwd=str(HERE))
    if r.returncode != 0:
        print(r.stdout); print(r.stderr)
        sys.exit("!! Predictor smoke test failed")
    print(f"   {r.stdout.strip()}")

    # Inference vs cache match test
    print(f"  inference-cache match ...")
    r = subprocess.run([sys.executable, str(HERE / "test_inference_match.py")], capture_output=True, text=True, cwd=str(HERE))
    if r.returncode != 0:
        print(r.stdout); print(r.stderr)
        sys.exit("!! inference-cache match test failed")
    print(r.stdout)

    # Zip
    out_zip = Path(args.zip_out)
    if out_zip.exists():
        out_zip.unlink()
    files_to_zip = [
        "Predictor.py", "compute.py", "r34_features.py", "r34_stage2_features.py",
        "config.json", "thresholds.json", "requirements.txt",
        "model_h40.txt",
    ] + [f"model_h60_seed{s}.txt" for s in seeds]
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for fn in files_to_zip:
            p = HERE / fn
            if not p.exists():
                sys.exit(f"!! file missing: {p}")
            zf.write(p, arcname=fn)
    print(f"  zipped -> {out_zip}  ({out_zip.stat().st_size/1e6:.2f}MB)")

    # Sanity: print zip contents
    with zipfile.ZipFile(out_zip) as zf:
        for info in zf.infolist():
            print(f"    {info.filename:40s}  {info.file_size/1e3:8.1f}KB")


if __name__ == "__main__":
    main()
