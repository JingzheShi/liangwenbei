"""Finalize iter_019 packages: verify models, copy, smoke test, zip, results.json."""
from __future__ import annotations
import gc
import hashlib
import importlib.util
import json
import os
import sys
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd

RETRAIN_DIR = "/root/projects/liangwenbei_workdir/experiments/R_full_retrain"
WORKDIR = "/root/projects/liangwenbei_workdir"
T140B_DIR = os.path.join(WORKDIR, "experiments", "T140b_iter019_finalize")
SEEDS = [1, 7, 13, 42, 100]

PKG_V1 = os.path.join(RETRAIN_DIR, "pkg_iter019_v1")
PKG_V2 = os.path.join(RETRAIN_DIR, "pkg_iter019_v2")

ZIP_V1 = os.path.join(WORKDIR, "submission_050818_iter019_v1_fullretrain.zip")
ZIP_V2 = os.path.join(WORKDIR, "submission_050818_iter019_v2_fullretrain_conformal.zip")


def verify_models():
    import lightgbm as lgb
    ok_seeds = []
    fail_seeds = []
    for s in SEEDS:
        path = os.path.join(RETRAIN_DIR, f"model_h60_seed{s}.txt")
        if not os.path.isfile(path):
            print(f"  seed={s} MISSING", flush=True)
            fail_seeds.append(s)
            continue
        try:
            m = lgb.Booster(model_file=path)
            n_feat = m.num_feature()
            n_iter = m.current_iteration()
            assert n_feat == 359, f"seed {s} feat mismatch: {n_feat}"
            print(f"  seed={s} OK ({n_iter} iters, {n_feat} feats)", flush=True)
            ok_seeds.append(s)
        except Exception as e:
            print(f"  seed={s} FAIL: {e}", flush=True)
            fail_seeds.append(s)
    return ok_seeds, fail_seeds


def copy_models(ok_seeds):
    import shutil
    for s in ok_seeds:
        src = os.path.join(RETRAIN_DIR, f"model_h60_seed{s}.txt")
        for pkg in [PKG_V1, PKG_V2]:
            dst = os.path.join(pkg, f"model_h60_seed{s}.txt")
            shutil.copy2(src, dst)
            print(f"  copied seed={s} -> {os.path.basename(pkg)}/", flush=True)


def smoke_test(pkg_dir, has_sym):
    print(f"\n  smoke {os.path.basename(pkg_dir)} (has_sym={has_sym})", flush=True)
    cfg = json.load(open(os.path.join(pkg_dir, "config.json")))
    feats = cfg["feature"]
    sys.path.insert(0, pkg_dir)
    try:
        spec = importlib.util.spec_from_file_location(
            f"P_{os.path.basename(pkg_dir)}",
            os.path.join(pkg_dir, "Predictor.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        Predictor = mod.Predictor
        p = Predictor()
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            rng.standard_normal((100, len(feats))).astype(np.float32),
            columns=feats,
        )
        if has_sym:
            df["sym"] = 2
        out = p.predict([df, df, df])
        assert isinstance(out, list) and len(out) == 3
        assert all(len(row) == 5 for row in out)
        flat = [a for row in out for a in row]
        assert all(a in (0, 1, 2) for a in flat), f"invalid actions: {set(flat)}"
        n_active = sum(1 for a in flat if a != 1)
        print(f"    OK: first row={out[0]}, n_active={n_active}/15", flush=True)
        return True
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False
    finally:
        sys.path.remove(pkg_dir)


def make_zip(src_dir, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(src_dir):
            for f in sorted(files):
                p = os.path.join(root, f)
                zf.write(p, os.path.relpath(p, src_dir))
    return zip_path


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_zip_files(path):
    with zipfile.ZipFile(path) as zf:
        return len(zf.namelist()), sorted(zf.namelist())


def main():
    print("=== T140b finalize ===", flush=True)

    print("\n1. Verifying models...", flush=True)
    ok_seeds, fail_seeds = verify_models()
    print(f"   OK: {ok_seeds}  FAIL/MISSING: {fail_seeds}", flush=True)

    if not ok_seeds:
        print("ERROR: No models available!", flush=True)
        sys.exit(1)

    print("\n2. Copying models to pkg dirs...", flush=True)
    copy_models(ok_seeds)

    print("\n3. Smoke testing Predictors...", flush=True)
    ok_v1 = smoke_test(PKG_V1, has_sym=False)
    ok_v2 = smoke_test(PKG_V2, has_sym=True)

    print("\n4. Building submission zips...", flush=True)
    make_zip(PKG_V1, ZIP_V1)
    make_zip(PKG_V2, ZIP_V2)
    print(f"   v1: {ZIP_V1}", flush=True)
    print(f"   v2: {ZIP_V2}", flush=True)

    print("\n5. Computing checksums and file counts...", flush=True)
    v1_md5 = md5_file(ZIP_V1)
    v2_md5 = md5_file(ZIP_V2)
    v1_size = os.path.getsize(ZIP_V1) / 1e6
    v2_size = os.path.getsize(ZIP_V2) / 1e6
    v1_count, v1_files = count_zip_files(ZIP_V1)
    v2_count, v2_files = count_zip_files(ZIP_V2)

    print(f"   v1: {v1_count} files, {v1_size:.1f} MB, md5={v1_md5}", flush=True)
    print(f"   v2: {v2_count} files, {v2_size:.1f} MB, md5={v2_md5}", flush=True)
    print(f"   v1 files: {v1_files}", flush=True)
    print(f"   v2 files: {v2_files}", flush=True)

    notes = []
    if fail_seeds:
        notes.append(f"Missing seeds: {fail_seeds} — ensemble uses {len(ok_seeds)}/5 LGB models")
    if not ok_v1:
        notes.append("WARNING: v1 smoke test FAILED")
    if not ok_v2:
        notes.append("WARNING: v2 smoke test FAILED")

    results = {
        "task": "T140b: re-train seeds 13/42/100 + finalize iter_019 v1/v2 zips",
        "ok_seeds": ok_seeds,
        "fail_seeds": fail_seeds,
        "v1_zip": ZIP_V1,
        "v2_zip": ZIP_V2,
        "v1_md5": v1_md5,
        "v2_md5": v2_md5,
        "v1_size_mb": round(v1_size, 2),
        "v2_size_mb": round(v2_size, 2),
        "v1_files": v1_count,
        "v2_files": v2_count,
        "v1_file_list": v1_files,
        "v2_file_list": v2_files,
        "smoke_test_v1_ok": ok_v1,
        "smoke_test_v2_ok": ok_v2,
        "smoke_test_ok": ok_v1 and ok_v2,
        "expected_platform_v1": "iter_015_v1 +28.16 + (~1~3 M7 boost) = ~29~31",
        "expected_platform_v2": "iter_018_v1 +28.93 + (~1~3 M7 boost) = ~30~32",
        "notes": "; ".join(notes) if notes else "All 5 seeds trained and packaged successfully",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out = os.path.join(T140B_DIR, "results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nresults.json -> {out}", flush=True)
    print(f"\nRESULT: task=T140b-iter019-finalize metrics={{ok_seeds={len(ok_seeds)}, v1_size_mb={v1_size:.1f}, v2_size_mb={v2_size:.1f}, smoke_ok={ok_v1 and ok_v2}}} notes={results['notes'][:80]}", flush=True)


if __name__ == "__main__":
    main()
