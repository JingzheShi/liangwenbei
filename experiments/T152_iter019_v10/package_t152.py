#!/usr/bin/env python3
"""Package iter_019 v10:
1. Copy v2 pkg → T152 pkg/
2. Replace 5 LGB models with 259-feature v10 models
3. Copy pruned_features.json
4. Modify Predictor.py to apply feature slicing (LGB gets 259, NN gets 359)
5. Smoke test
6. Build zip
7. Record MD5/size
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile

ROOT = "/root/projects/liangwenbei_workdir"
HERE = os.path.join(ROOT, "experiments", "T152_iter019_v10")
V2_PKG = os.path.join(ROOT, "experiments", "R_full_retrain", "pkg_iter019_v2")
PKG_DIR = os.path.join(HERE, "pkg")
ZIP_PATH = os.path.join(ROOT, "submission_050818_iter019_v10_pruned.zip")


def copy_v2_pkg():
    print("=== Step 1: Copy v2 pkg ===", flush=True)
    if os.path.exists(PKG_DIR):
        shutil.rmtree(PKG_DIR)
    shutil.copytree(V2_PKG, PKG_DIR)
    print(f"  Copied {V2_PKG} → {PKG_DIR}", flush=True)
    print(f"  Files: {sorted(os.listdir(PKG_DIR))}", flush=True)


def replace_lgb_models():
    print("\n=== Step 2: Replace LGB models ===", flush=True)
    seeds = [1, 7, 13, 42, 100]
    for s in seeds:
        src = os.path.join(HERE, f"model_h60_seed{s}.txt")
        dst = os.path.join(PKG_DIR, f"model_h60_seed{s}.txt")
        if not os.path.exists(src):
            print(f"  MISSING: {src}", flush=True)
            sys.exit(1)
        shutil.copy2(src, dst)
        src_size = os.path.getsize(src)
        dst_size = os.path.getsize(dst)
        print(f"  seed={s}: {src_size//1024}KB → {dst_size//1024}KB", flush=True)


def copy_pruned_features():
    print("\n=== Step 3: Copy pruned_features.json ===", flush=True)
    src = os.path.join(HERE, "pruned_features.json")
    dst = os.path.join(PKG_DIR, "pruned_features.json")
    shutil.copy2(src, dst)
    with open(dst) as f:
        info = json.load(f)
    print(f"  n_kept={info['n_kept']} n_dropped={info['n_dropped']}", flush=True)
    print(f"  pruned_idx_in_359 count: {len(info['pruned_idx_in_359'])}", flush=True)


def patch_predictor():
    print("\n=== Step 4: Patch Predictor.py ===", flush=True)
    pred_path = os.path.join(PKG_DIR, "Predictor.py")
    with open(pred_path) as f:
        src = f.read()

    # Verify the Predictor doesn't already have pruned_idx
    assert "pruned_idx" not in src, "Predictor already patched!"

    # Patch __init__: after loading LGB/NN ensembles, load pruned_features.json
    old_init_tail = '''        self._lgb_lists: Dict[int, List[lgb.Booster]] = {}
        self._nn_lists: Dict[int, List[_MLPNumpy]] = {}
        self._weights: Dict[int, Tuple[float, float]] = {}
        for hcfg in self._horizons:'''

    new_init_tail = '''        # v10: load pruned feature indices for LGB (259 features out of 359)
        pruned_json_path = os.path.join(here, "pruned_features.json")
        with open(pruned_json_path) as f:
            _pruned_info = json.load(f)
        self._pruned_idx_lgb: np.ndarray = np.array(
            _pruned_info["pruned_idx_in_359"], dtype=np.int64
        )

        self._lgb_lists: Dict[int, List[lgb.Booster]] = {}
        self._nn_lists: Dict[int, List[_MLPNumpy]] = {}
        self._weights: Dict[int, Tuple[float, float]] = {}
        for hcfg in self._horizons:'''

    assert old_init_tail in src, "Could not find __init__ insertion point!"
    src = src.replace(old_init_tail, new_init_tail, 1)

    # Patch predict: feats → feats_359, add feats_259 for LGB, feats_359 for NN
    old_predict_head = '''    def predict(self, batches: List[pd.DataFrame]) -> List[List[int]]:
        if not batches:
            return []
        feats = self._compute_batch_features(batches)
        B = feats.shape[0]'''

    new_predict_head = '''    def predict(self, batches: List[pd.DataFrame]) -> List[List[int]]:
        if not batches:
            return []
        feats_359 = self._compute_batch_features(batches)
        feats_259 = feats_359[:, self._pruned_idx_lgb]  # v10: 259 adversarial-filtered features for LGB
        B = feats_359.shape[0]'''

    assert old_predict_head in src, "Could not find predict insertion point!"
    src = src.replace(old_predict_head, new_predict_head, 1)

    # Replace feats usage in predict body
    # band_per_row extraction and conformal wrapper: use feats_359 (not needed, it's sym-based)
    # LGB: use feats_259
    # NN: use feats_359

    old_lgb_call = "                preds.append(self._ensemble_predict_lgb(lgbs, feats))"
    new_lgb_call = "                preds.append(self._ensemble_predict_lgb(lgbs, feats_259))"
    assert old_lgb_call in src, "Could not find LGB call!"
    src = src.replace(old_lgb_call, new_lgb_call, 1)

    old_nn_call = "                preds.append(self._ensemble_predict_nn(nns, feats))"
    new_nn_call = "                preds.append(self._ensemble_predict_nn(nns, feats_359))"
    assert old_nn_call in src, "Could not find NN call!"
    src = src.replace(old_nn_call, new_nn_call, 1)

    with open(pred_path, "w") as f:
        f.write(src)
    print(f"  Patched {pred_path}", flush=True)

    # Verify patch
    assert "pruned_idx_lgb" in open(pred_path).read()
    assert "feats_259" in open(pred_path).read()
    assert "feats_359" in open(pred_path).read()
    print("  Patch verified OK", flush=True)


def smoke_test():
    print("\n=== Step 5: Smoke test ===", flush=True)
    smoke_script = f"""
import sys
sys.path.insert(0, '{PKG_DIR}')
import os
os.chdir('{PKG_DIR}')
import json, numpy as np, pandas as pd

# Load config
cfg = json.load(open('{PKG_DIR}/config.json'))
feats = cfg['feature']

# Create synthetic batch (100 rows × n_feats)
rng = np.random.default_rng(42)
df = pd.DataFrame(
    rng.standard_normal((100, len(feats))).astype(np.float32),
    columns=feats,
)
df['sym'] = 2

# Import Predictor
import importlib.util
spec = importlib.util.spec_from_file_location('Predictor', '{PKG_DIR}/Predictor.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

p = mod.Predictor()
print('Predictor initialized OK', flush=True)
print(f'pruned_idx_lgb shape: {{p._pruned_idx_lgb.shape}}', flush=True)

# Run predict
out = p.predict([df, df, df])
print(f'predict([3 batches]) -> n_outputs={{len(out)}} first={{out[0]}}', flush=True)

# Run OOD sym
df_ood = df.copy(); df_ood['sym'] = 99
out_ood = p.predict([df_ood])
print(f'OOD sym=99 -> {{out_ood[0]}}', flush=True)

# Run without sym
df_no_sym = df.drop(columns=['sym'])
out_no = p.predict([df_no_sym])
print(f'missing sym -> {{out_no[0]}}', flush=True)

# Check outputs
assert len(out) == 3, f'Expected 3 outputs, got {{len(out)}}'
for row in out:
    assert len(row) == 5, f'Expected 5 horizons, got {{len(row)}}'
    for a in row:
        assert a in [0, 1, 2], f'Invalid action {{a}}'

print('SMOKE TEST PASSED', flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-c", smoke_script],
        capture_output=True, text=True, timeout=120
    )
    print(result.stdout, flush=True)
    if result.returncode != 0:
        print("STDERR:", result.stderr, flush=True)
        raise RuntimeError(f"Smoke test failed: {result.returncode}")
    if "SMOKE TEST PASSED" not in result.stdout:
        raise RuntimeError("Smoke test did not print PASSED")
    print("  Smoke test PASSED", flush=True)
    return True


def build_zip():
    print("\n=== Step 6: Build zip ===", flush=True)
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname in sorted(os.listdir(PKG_DIR)):
            fpath = os.path.join(PKG_DIR, fname)
            if os.path.isfile(fpath):
                zf.write(fpath, fname)
                size_kb = os.path.getsize(fpath) // 1024
                print(f"  + {fname} ({size_kb} KB)", flush=True)
    zip_size = os.path.getsize(ZIP_PATH)
    print(f"  ZIP: {ZIP_PATH}  size={zip_size/1e6:.2f} MB", flush=True)
    return zip_size


def compute_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    copy_v2_pkg()
    replace_lgb_models()
    copy_pruned_features()
    patch_predictor()
    smoke_ok = smoke_test()
    zip_size = build_zip()
    md5 = compute_md5(ZIP_PATH)

    print(f"\n=== Summary ===", flush=True)
    print(f"  ZIP: {ZIP_PATH}", flush=True)
    print(f"  MD5: {md5}", flush=True)
    print(f"  Size: {zip_size/1e6:.2f} MB", flush=True)
    print(f"  Smoke test: {'OK' if smoke_ok else 'FAILED'}", flush=True)

    # Load TSH-FT results
    results_path = os.path.join(HERE, "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            res = json.load(f)
    else:
        res = {}

    # Update results.json
    res.update({
        "v10_zip": ZIP_PATH,
        "v10_md5": md5,
        "v10_size_mb": round(zip_size / 1e6, 3),
        "v10_files": sorted(os.listdir(PKG_DIR)),
        "smoke_test_ok": smoke_ok,
        "expected_platform_v10": "v2 +34.44 + (T147 +19.29 tshft * transmission ~0.7) = ~48",
    })
    with open(results_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"  Updated {results_path}", flush=True)

    # Copy zip to output dir
    out_dir = "/tmp/metabot-outputs/worker-0ad44668"
    os.makedirs(out_dir, exist_ok=True)
    out_zip = os.path.join(out_dir, "submission_050818_iter019_v10_pruned.zip")
    shutil.copy2(ZIP_PATH, out_zip)
    print(f"  Copied to output: {out_zip}", flush=True)


if __name__ == "__main__":
    main()
