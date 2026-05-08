"""T143 Part B: Package iter_019 v3 mae zip.

Copies T127 iter_018 v1 pkg, replaces 5 LGB model.txt files with T137 MAE models,
updates thresholds.json with new DE thresholds, smoke-tests, builds zip.
"""
from __future__ import annotations
import hashlib, json, os, shutil, sys, zipfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(HERE)
WORKDIR = os.path.dirname(EXPERIMENTS_DIR)

SRC_PKG = os.path.join(EXPERIMENTS_DIR, "T127_iter018_v1_pkg/pkg")
MAE_MODEL_DIR = os.path.join(EXPERIMENTS_DIR, "T137_alt_robust_losses")
DST_PKG = os.path.join(EXPERIMENTS_DIR, "T143_iter019_v3_mae_pkg")
ZIP_PATH = os.path.join(WORKDIR, "submission_050818_iter019_v3_mae.zip")
SEEDS = [1, 7, 13, 42, 100]


def build_pkg(thr_up: float, thr_dn: float) -> dict:
    print(f"Building iter_019 v3 mae pkg...", flush=True)
    print(f"  thr_up={thr_up:.8f}  thr_dn={thr_dn:.8f}", flush=True)

    # Step 1: copy from T127
    if os.path.exists(DST_PKG):
        shutil.rmtree(DST_PKG)
    shutil.copytree(SRC_PKG, DST_PKG)
    print(f"  Copied T127 pkg -> {DST_PKG}", flush=True)

    # Step 2: replace 5 LGB model.txt files with MAE versions
    for s in SEEDS:
        src = os.path.join(MAE_MODEL_DIR, f"model_T137_mae_seed{s}.txt")
        dst = os.path.join(DST_PKG, f"model_h60_seed{s}.txt")
        if not os.path.exists(src):
            raise FileNotFoundError(f"MAE model missing: {src}")
        shutil.copy2(src, dst)
        src_sz = os.path.getsize(src) / 1024
        print(f"  Replaced model_h60_seed{s}.txt  ({src_sz:.0f} KB)", flush=True)

    # Step 3: update thresholds.json with new DE thresholds
    thr_path = os.path.join(DST_PKG, "thresholds.json")
    with open(thr_path) as f:
        thr_data = json.load(f)

    for hz in thr_data.get("horizons", []):
        if hz.get("h") == 60 and hz.get("active"):
            hz["thr_up"] = thr_up
            hz["thr_dn"] = thr_dn
            hz["_doc"] = (f"T143 MAE ensemble DE LOSO opt thresholds; "
                          f"iter_019 v3 replacing T75 L2 with T137 MAE 5-seed.")
            break

    with open(thr_path, "w") as f:
        json.dump(thr_data, f, indent=2)
    print(f"  Updated thresholds.json  thr_up={thr_up:.8f} thr_dn={thr_dn:.8f}", flush=True)

    # Step 4: verify file count
    files = []
    for root, _, fnames in os.walk(DST_PKG):
        for fn in fnames:
            files.append(os.path.relpath(os.path.join(root, fn), DST_PKG))
    print(f"  Files in pkg ({len(files)}): {sorted(files)}", flush=True)
    assert 12 <= len(files) <= 16, f"Expected 12-16 files, got {len(files)}"

    # Step 5: smoke test
    print("\nRunning smoke test...", flush=True)
    smoke_ok = smoke_test(DST_PKG)

    # Step 6: build zip
    print("\nBuilding zip...", flush=True)
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, fnames in os.walk(DST_PKG):
            for fn in fnames:
                p = os.path.join(root, fn)
                zf.write(p, os.path.relpath(p, DST_PKG))
    zip_sz = os.path.getsize(ZIP_PATH) / 1024 / 1024
    md5 = hashlib.md5(open(ZIP_PATH, "rb").read()).hexdigest()
    print(f"  zip -> {ZIP_PATH}", flush=True)
    print(f"  size={zip_sz:.2f} MB  md5={md5}  files={len(files)}", flush=True)

    # Copy to output
    out_dir = "/tmp/metabot-outputs/worker-bc9d2176"
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy2(ZIP_PATH, os.path.join(out_dir, os.path.basename(ZIP_PATH)))
    print(f"  Copied to output dir.", flush=True)

    return {
        "v3_zip": ZIP_PATH,
        "v3_md5": md5,
        "v3_size_mb": round(zip_sz, 3),
        "v3_files": len(files),
        "smoke_test_ok": smoke_ok,
    }


RAW_COLS = (
    "open", "high", "low", "close", "volume_delta", "amount_delta",
    *(f"bid{k}" for k in range(1, 11)),
    *(f"bsize{k}" for k in range(1, 11)),
    *(f"ask{k}" for k in range(1, 11)),
    *(f"asize{k}" for k in range(1, 11)),
    "avgbid", "avgask", "totalbsize", "totalasize",
    "lb_intst", "la_intst", "mb_intst", "ma_intst", "cb_intst", "ca_intst",
    "lb_ind", "la_ind", "mb_ind", "ma_ind", "cb_ind", "ca_ind",
    "lb_acc", "la_acc", "mb_acc", "ma_acc", "cb_acc", "ca_acc",
    *(f"midprice{k}" for k in range(1, 11)),
    *(f"spread{k}" for k in range(1, 11)),
    *(f"bid_diff{k}" for k in range(1, 11)),
    *(f"ask_diff{k}" for k in range(1, 11)),
    "bid_mean", "ask_mean", "bsize_mean", "asize_mean", "cumspread", "imbalance",
    *(f"bid_rate{k}" for k in range(1, 11)),
    *(f"ask_rate{k}" for k in range(1, 11)),
    *(f"bsize_rate{k}" for k in range(1, 11)),
    *(f"asize_rate{k}" for k in range(1, 11)),
)
WINDOW = 100


def smoke_test(pkg_dir: str) -> bool:
    """Load Predictor from pkg_dir, build windows from session parquets, run predict()."""
    import importlib.util
    import pandas as pd

    sys.path.insert(0, pkg_dir)
    try:
        if "Predictor" in sys.modules:
            del sys.modules["Predictor"]
        spec = importlib.util.spec_from_file_location("Predictor_T143",
                                                       os.path.join(pkg_dir, "Predictor.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        Predictor = mod.Predictor

        p = Predictor()
        print(f"  SMOKE: Predictor instantiated OK", flush=True)

        # Find session parquet files
        data_dir = os.path.join(WORKDIR, "data")
        session_paths = []
        for sym in range(5):
            for date in (0, 100):
                for sess in ("am", "pm"):
                    path = os.path.join(data_dir, f"snapshot_sym{sym}_date{date}_{sess}.parquet")
                    if os.path.exists(path):
                        session_paths.append((sym, path))
                        break
                if session_paths and session_paths[-1][0] == sym:
                    break

        if not session_paths:
            print("  SMOKE: No session parquets found, instantiation-only passed.", flush=True)
            return True

        total_batches = 0
        n_active = 0
        for sym, path in session_paths[:3]:
            df_full = pd.read_parquet(path)
            n = len(df_full)
            if n < WINDOW + 60:
                continue
            # Build 20 windows
            t_choices = np.linspace(WINDOW - 1, n - 61, 20, dtype=int)
            batch_dfs = []
            for t in t_choices:
                win = df_full.iloc[t - WINDOW + 1: t + 1].copy().reset_index(drop=True)
                cols = [c for c in list(RAW_COLS) + ["sym"] if c in win.columns]
                batch_dfs.append(win[cols].astype(np.float32))

            actions = p.predict(batch_dfs)
            actions = np.array(actions, dtype=np.int64)
            assert actions.shape == (len(batch_dfs), 5), f"Bad shape {actions.shape}"
            assert np.isin(actions, [0, 1, 2]).all(), "Invalid action values"
            n_active += int((actions[:, 4] != 1).sum())
            total_batches += len(batch_dfs)
            print(f"  SMOKE sym={sym}: {len(batch_dfs)} windows ok, "
                  f"h60 active={int((actions[:,4]!=1).sum())}/{len(batch_dfs)}", flush=True)

        # OOD sym test (sym=99)
        if session_paths:
            df_ood = pd.read_parquet(session_paths[0][1]).tail(200).reset_index(drop=True)
            if "sym" in df_ood.columns:
                df_ood["sym"] = 99.0
            win = df_ood.iloc[:WINDOW].copy()
            cols = [c for c in list(RAW_COLS) + ["sym"] if c in win.columns]
            actions_ood = p.predict([win[cols].astype(np.float32)])
            assert np.array(actions_ood).shape == (1, 5), "OOD shape fail"
            print(f"  SMOKE OOD sym=99: actions={np.array(actions_ood)[0].tolist()}", flush=True)

        print(f"  SMOKE: {total_batches} windows total, n_active(h60)={n_active} → PASS", flush=True)
        return True

    except Exception as e:
        print(f"  SMOKE ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False
    finally:
        if pkg_dir in sys.path:
            sys.path.remove(pkg_dir)


def main():
    # Read Part A results
    results_path = os.path.join(HERE, "results.json")
    if not os.path.exists(results_path):
        print("ERROR: results.json not found. Run eval_ensemble.py first.", flush=True)
        sys.exit(1)

    with open(results_path) as f:
        results = json.load(f)

    # Conservative approach: always use B (MAE replaces L2) for packaging.
    # E (3-way) is technically +0.13 better but requires Predictor.py changes.
    pkg_config = "B_replace_l2_mae"
    best_overall = results.get("best_config", "E_3way_mae_heavy")
    pkg_cfg_data = results["configs"][pkg_config]
    thr_up = pkg_cfg_data["thr_up"]
    thr_dn = pkg_cfg_data["thr_dn"]

    print(f"Best overall config: {best_overall}  LOSO={results['configs'][best_overall]['conformal']:+.4f}", flush=True)
    print(f"Packaging config (conservative): {pkg_config}", flush=True)
    print(f"  LOSO (no_conf)={pkg_cfg_data['no_conformal']:+.4f}  (conformal)={pkg_cfg_data['conformal']:+.4f}", flush=True)

    pkg_info = build_pkg(thr_up, thr_dn)

    # Update results.json with Part B info
    results.update(pkg_info)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nUpdated results.json", flush=True)

    return pkg_info


if __name__ == "__main__":
    main()
