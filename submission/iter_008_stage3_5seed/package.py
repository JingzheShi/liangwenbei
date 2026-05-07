"""Build iter_008 submission package.

Steps:
  1. Reads DE result file (de_results_5seed.json) for h_60 thresholds
  2. Writes thresholds.json (h_5/10/20/40 disabled, h_60 from DE)
  3. Verifies all 5 model_h60_seed{1,7,13,42,100}.txt files are present
  4. Runs the smoke test on the Predictor
  5. Runs the inference vs cache match test (sanity check)
  6. Zips into submission_*.zip
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
T53_DIR = ROOT / "experiments" / "T53_r34_stage3"


def _read_de(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--de-results", default=str(T53_DIR / "de_results_5seed.json"))
    ap.add_argument("--zip-out", default=str(HERE / "submission_iter008.zip"))
    args = ap.parse_args()

    de_path = Path(args.de_results)
    if not de_path.exists():
        sys.exit(f"!! DE results missing: {de_path}")
    de = _read_de(de_path)
    best = de["de_best"]
    T_up = float(best["T_up"]); T_dn = float(best["T_dn"])
    d_up = float(best["d_up"]); d_dn = float(best["d_dn"])
    loso_sum = float(best["sum_cum_pnl"])
    per_fold = list(best["per_fold_pnl"])
    n_active = list(best.get("n_active_per_fold", []))
    print(f"  DE thresholds: T_up={T_up:.4f} T_dn={T_dn:.4f} d_up={d_up:.4f} d_dn={d_dn:.4f}")
    print(f"  LOSO sum (with thresh): {loso_sum:+.4f}")
    print(f"  per_fold: {[round(x,3) for x in per_fold]}")

    # Build thresholds.json
    thresh = {
        "_doc": "iter_008: T53 R34 Stage 1+2+3 (340-d) 5-seed aug_a ensemble for h_60. "
                "DE 4D asymmetric thresh from 5-seed averaged OOF prob. "
                "h_5/h_10/h_20 disabled (matches iter_006/007). h_40 disabled (no h_40 model bundled).",
        "_loso_sum_with_thresh": loso_sum,
        "horizons": [
            {"h": 5,  "T": 0.99, "delta": 0.99, "active": False, "_reason": "iter_002 platform = -7.68"},
            {"h": 10, "T": 0.99, "delta": 0.99, "active": False, "_reason": "iter_002 platform = -8.64"},
            {"h": 20, "T": 0.99, "delta": 0.99, "active": False, "_reason": "iter_002 platform = -5.09"},
            {"h": 40, "T": 0.5,  "delta": 0.0,  "active": False, "_reason": "no h_40 model in iter_008"},
            {"h": 60, "T_up": T_up, "T_dn": T_dn, "d_up": d_up, "d_dn": d_dn,
             "active": True,
             "_reason": f"T53b DE 5-seed asym optimum on stage3 OOF, LOSO sum={loso_sum:+.4f}",
             "asym_meta": {
                 "T_up": T_up, "T_dn": T_dn, "d_up": d_up, "d_dn": d_dn,
                 "OOF_sum_cum_pnl": loso_sum,
                 "OOF_per_fold_pnl": per_fold,
                 "OOF_n_active_per_fold": n_active,
                 "_source": "T53b differential_evolution best on 5-seed LOSO OOF (stage 1+2+3)"
             },
             "ensemble_seeds": [1, 7, 13, 42, 100]},
        ],
    }
    with open(HERE / "thresholds.json", "w") as f:
        json.dump(thresh, f, indent=2)
    print(f"  wrote thresholds.json")

    # Copy h60 full-data models (5 seeds)
    seeds = (1, 7, 13, 42, 100)
    for seed in seeds:
        src = T53_DIR / f"full_model_h60_seed{seed}.txt"
        if not src.exists():
            sys.exit(f"!! missing model: {src}")
        dst = HERE / f"model_h60_seed{seed}.txt"
        shutil.copy2(src, dst)
        print(f"  {src.name} -> {dst.name} ({src.stat().st_size/1e6:.2f}MB)")

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
        "Predictor.py", "compute.py",
        "r34_features.py", "r34_stage2_features.py", "r34_stage3_features.py",
        "config.json", "thresholds.json", "requirements.txt",
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
