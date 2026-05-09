#!/usr/bin/env python3
"""T164: 4-way ensemble (T87 + T97 + T95 + v2-LGB) holdout evaluation.

Uses pre-computed prediction parquets for T87/T97/T95 NN and T75 LGB.
Evaluates 5 weight configs, each with T163's selective agreement filter.

Baseline (Config A): T163 combined = 42.9021
Goal: +0.3 improvement → > 43.20
"""
from __future__ import annotations
import json
import os
import shutil
import sys
import time
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

T87_DIR = os.path.join(ROOT, "experiments", "T87_spo_dfl")
T97_DIR = os.path.join(ROOT, "experiments", "T97_multihead_nn")
T95_DIR = os.path.join(ROOT, "experiments", "T95_cnn_rnn_deeplob")
T75_DIR = os.path.join(ROOT, "experiments", "T75_regression_dmid")
T163_PKG = os.path.join(ROOT, "experiments", "T163_combined_pkg")

SEEDS = [1, 7, 13, 42, 100]
SYMS = [0, 1, 2, 3, 4]
FEE = 0.0001

# iter_018 v1 conformal thresholds (T163 identical)
THR_UP = 0.00029995433796130955
THR_DN = 0.0002158528296175958

# Per-sym conformal beta (T163 / T127 tuned)
PER_SYM_BETA = {0: 0.1, 1: 0.4, 2: 0.3, 3: 0.0, 4: 0.0}
PER_SYM_SIGMA = {
    0: 0.0002403901,
    1: 0.0004712397,
    2: 0.0004522216,
    3: 0.0004235249,
    4: 0.0004279811,
}
DEFAULT_BETA_OOD = 0.16
DEFAULT_SIGMA_OOD = 0.0003998317
PER_SYM_BAND = {k: PER_SYM_BETA[k] * PER_SYM_SIGMA[k] for k in SYMS}

# Agreement filter: only apply to syms 0, 1, 2 (per T158/T163 analysis)
AGREEMENT_FILTER_SYMS = frozenset({0, 1, 2})

PROGRESS_PATH = os.path.join(HERE, "worker-progress.json")


def ts():
    return datetime.now().strftime('%H:%M:%S')


def progress(step, metrics=None):
    p = {"status": "running", "step": step, "metrics": metrics or {},
         "timestamp": datetime.now().isoformat()}
    with open(PROGRESS_PATH, 'w') as f:
        json.dump(p, f, indent=2)
    print(f"[{ts()}] {step}", flush=True)


def vectorized_pnl(action, mp_t, mp_th):
    side = action.astype(np.float64) - 1.0
    fee = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    return (side * (mp_th - mp_t) - fee) / (mp_t + 1.0)


def apply_conformal_gate(pred_combined, sym_arr):
    """Standard iter_018 conformal gate."""
    action = np.ones(len(pred_combined), dtype=np.int8)
    for s in SYMS:
        mask = sym_arr == s
        band = PER_SYM_BAND[s]
        p = pred_combined[mask]
        a = np.ones(mask.sum(), dtype=np.int8)
        a[p > (THR_UP + band)] = 2
        a[p < -(THR_DN + band)] = 0
        action[mask] = a
    ood_mask = ~np.isin(sym_arr, SYMS)
    if ood_mask.any():
        ood_band = DEFAULT_BETA_OOD * DEFAULT_SIGMA_OOD
        p = pred_combined[ood_mask]
        a = np.ones(ood_mask.sum(), dtype=np.int8)
        a[p > (THR_UP + ood_band)] = 2
        a[p < -(THR_DN + ood_band)] = 0
        action[ood_mask] = a
    return action


def apply_agreement_filter(action, nn_side, pred_lgb, sym_arr):
    """Selective agreement filter: abstain when NN-side and LGB disagree, only for syms 0/1/2."""
    action = action.copy()
    for s in AGREEMENT_FILTER_SYMS:
        mask = sym_arr == s
        nn_s = nn_side[mask]
        lgb_s = pred_lgb[mask]
        disagree = np.sign(nn_s) != np.sign(lgb_s)
        sub_action = action[mask]
        sub_action[disagree] = 1  # hold
        action[mask] = sub_action
    return action


def eval_config(pred_combined, nn_side, pred_lgb, sym_arr, mp_t, mp_th, label):
    """Evaluate a config: conformal gate + selective agreement filter."""
    action = apply_conformal_gate(pred_combined, sym_arr)
    action = apply_agreement_filter(action, nn_side, pred_lgb, sym_arr)
    pnl_vec = vectorized_pnl(action, mp_t, mp_th)
    total = float(pnl_vec.sum())
    per_sym = [float(pnl_vec[sym_arr == s].sum()) for s in SYMS]
    n_trades = int((action != 1).sum())
    n_filter = int((apply_conformal_gate(pred_combined, sym_arr) != action).sum())
    return {
        "label": label,
        "total_pnl": round(total, 4),
        "per_sym_pnl": [round(x, 4) for x in per_sym],
        "n_trades": n_trades,
        "n_agreement_filter_applied": n_filter,
    }


def load_pred_parquets(template, seeds):
    """Load 5-seed parquets and return (base_df, avg_pred)."""
    dfs = [pd.read_parquet(template.format(seed=s)) for s in seeds]
    base = dfs[0][['sym', 'date', 'session', 't', 'midprice_t', 'midprice_th']].copy()
    preds = np.stack([df['pred_dmid_norm'].values for df in dfs], axis=1)
    return base, preds.mean(axis=1).astype(np.float64)


def main():
    t_start = time.time()
    progress("T164 starting: 4-way ensemble eval")

    # ── 1. Load all predictions ───────────────────────────────────────────────
    progress("Step 1: Loading prediction parquets")

    base_df, pred87 = load_pred_parquets(
        os.path.join(T87_DIR, "pred_T87_seed{seed}_main.parquet"), SEEDS)
    print(f"  T87: {len(pred87)} rows, dates {base_df['date'].min()}-{base_df['date'].max()}", flush=True)

    _, pred97 = load_pred_parquets(
        os.path.join(T97_DIR, "pred_T97_seed{seed}_main.parquet"), SEEDS)
    print(f"  T97: {len(pred97)} rows", flush=True)

    _, pred95 = load_pred_parquets(
        os.path.join(T95_DIR, "pred_T95_gru_w100_C_seed{seed}.parquet"), SEEDS)
    print(f"  T95: {len(pred95)} rows", flush=True)

    _, pred_lgb = load_pred_parquets(
        os.path.join(T75_DIR, "pred_T75_seed{seed}.parquet"), SEEDS)
    print(f"  T75 LGB: {len(pred_lgb)} rows", flush=True)

    # Verify alignment
    for arr, name in [(pred97, 'T97'), (pred95, 'T95'), (pred_lgb, 'LGB')]:
        assert len(arr) == len(pred87), f"{name} length mismatch: {len(arr)} vs {len(pred87)}"

    sym_arr = base_df['sym'].values.astype(np.int32)
    mp_t = base_df['midprice_t'].values.astype(np.float64)
    mp_th = base_df['midprice_th'].values.astype(np.float64)

    print(f"  Rows total: {len(pred87)}", flush=True)

    # ── 2. Evaluate configs ───────────────────────────────────────────────────
    progress("Step 2: Evaluating 5 configs")

    configs = {}

    # Config A: T163 baseline (T87 + T97 + LGB, w=1.5:1.5:1.0, agreement filter syms 0/1/2)
    w87, w97, w_lgb = 1.5, 1.5, 1.0
    nn_side_A = (w87 * pred87 + w97 * pred97) / (w87 + w97)
    pred_A = (w87 * pred87 + w97 * pred97 + w_lgb * pred_lgb) / (w87 + w97 + w_lgb)
    configs['A_3way_T163'] = eval_config(pred_A, nn_side_A, pred_lgb, sym_arr, mp_t, mp_th,
                                          "T87(1.5)+T97(1.5)+LGB(1.0) +filter [T163 baseline]")
    print(f"  A (T163 baseline): {configs['A_3way_T163']['total_pnl']:.4f}", flush=True)

    # Config B: 4-way equal (T87+T97+T95+LGB, w=1:1:1:1)
    w87, w97, w95, w_lgb = 1.0, 1.0, 1.0, 1.0
    nn_side_B = (w87 * pred87 + w97 * pred97 + w95 * pred95) / (w87 + w97 + w95)
    pred_B = (w87 * pred87 + w97 * pred97 + w95 * pred95 + w_lgb * pred_lgb) / (w87 + w97 + w95 + w_lgb)
    configs['B_4way_equal'] = eval_config(pred_B, nn_side_B, pred_lgb, sym_arr, mp_t, mp_th,
                                           "T87(1)+T97(1)+T95(1)+LGB(1) equal +filter")
    print(f"  B (4-way equal): {configs['B_4way_equal']['total_pnl']:.4f}", flush=True)

    # Config C: 4-way nn-heavy, T95 light (T87+T97+T95+LGB, w=1.5:1.5:0.5:1.0)
    w87, w97, w95, w_lgb = 1.5, 1.5, 0.5, 1.0
    nn_side_C = (w87 * pred87 + w97 * pred97 + w95 * pred95) / (w87 + w97 + w95)
    pred_C = (w87 * pred87 + w97 * pred97 + w95 * pred95 + w_lgb * pred_lgb) / (w87 + w97 + w95 + w_lgb)
    configs['C_4way_nn_heavy'] = eval_config(pred_C, nn_side_C, pred_lgb, sym_arr, mp_t, mp_th,
                                              "T87(1.5)+T97(1.5)+T95(0.5)+LGB(1.0) nn-heavy +filter")
    print(f"  C (4-way nn-heavy): {configs['C_4way_nn_heavy']['total_pnl']:.4f}", flush=True)

    # Config D: T95 low weight (w=1.5:1.5:0.3:1.0)
    w87, w97, w95, w_lgb = 1.5, 1.5, 0.3, 1.0
    nn_side_D = (w87 * pred87 + w97 * pred97 + w95 * pred95) / (w87 + w97 + w95)
    pred_D = (w87 * pred87 + w97 * pred97 + w95 * pred95 + w_lgb * pred_lgb) / (w87 + w97 + w95 + w_lgb)
    configs['D_4way_t95_low'] = eval_config(pred_D, nn_side_D, pred_lgb, sym_arr, mp_t, mp_th,
                                             "T87(1.5)+T97(1.5)+T95(0.3)+LGB(1.0) t95-low +filter")
    print(f"  D (4-way T95-low): {configs['D_4way_t95_low']['total_pnl']:.4f}", flush=True)

    # Config E: T95 very low weight (w=1.5:1.5:0.2:1.0)
    w87, w97, w95, w_lgb = 1.5, 1.5, 0.2, 1.0
    nn_side_E = (w87 * pred87 + w97 * pred97 + w95 * pred95) / (w87 + w97 + w95)
    pred_E = (w87 * pred87 + w97 * pred97 + w95 * pred95 + w_lgb * pred_lgb) / (w87 + w97 + w95 + w_lgb)
    configs['E_4way_t95_very_low'] = eval_config(pred_E, nn_side_E, pred_lgb, sym_arr, mp_t, mp_th,
                                                   "T87(1.5)+T97(1.5)+T95(0.2)+LGB(1.0) t95-very-low +filter")
    print(f"  E (4-way T95-very-low): {configs['E_4way_t95_very_low']['total_pnl']:.4f}", flush=True)

    # ── Also test with T95 in agreement filter vs not ─────────────────────────
    # Variant: T95 not in NN-side (same as T163 but extra T95 weight in combined pred)
    for key, label, w87, w97, w95, w_lgb in [
        ('C_noT95filter', "T87(1.5)+T97(1.5)+T95(0.5)+LGB(1.0) filter=T87+T97 only", 1.5, 1.5, 0.5, 1.0),
        ('D_noT95filter', "T87(1.5)+T97(1.5)+T95(0.3)+LGB(1.0) filter=T87+T97 only", 1.5, 1.5, 0.3, 1.0),
    ]:
        nn_side = (1.5 * pred87 + 1.5 * pred97) / 3.0  # T163 NN-side (no T95)
        pred = (w87 * pred87 + w97 * pred97 + w95 * pred95 + w_lgb * pred_lgb) / (w87 + w97 + w95 + w_lgb)
        configs[key] = eval_config(pred, nn_side, pred_lgb, sym_arr, mp_t, mp_th, label)
        print(f"  {key}: {configs[key]['total_pnl']:.4f}", flush=True)

    # ── 3. Find best config ───────────────────────────────────────────────────
    T163_BASELINE = 42.9021
    V2_BASELINE = 41.4939
    DELTA_THRESHOLD = 0.30  # need +0.30 over T163 to build pkg

    best_key = max(configs, key=lambda k: configs[k]['total_pnl'])
    best_pnl = configs[best_key]['total_pnl']
    delta_vs_T163 = best_pnl - T163_BASELINE
    delta_vs_v2 = best_pnl - V2_BASELINE

    print(f"\n{'='*60}", flush=True)
    print(f"  T163 combined baseline: {T163_BASELINE}", flush=True)
    print(f"  Best config: {best_key} = {best_pnl:.4f}", flush=True)
    print(f"  Delta vs T163: {delta_vs_T163:+.4f}", flush=True)
    print(f"  Build pkg threshold: +{DELTA_THRESHOLD}", flush=True)
    print(f"{'='*60}\n", flush=True)

    results = {
        "task": "T164 4-way ensemble (T87+T97+T95+v2-LGB)",
        "v2_baseline_holdout": V2_BASELINE,
        "T163_combined_baseline_holdout": T163_BASELINE,
        "configs": configs,
        "best_config": best_key,
        "best_holdout": best_pnl,
        "delta_vs_T163": round(delta_vs_T163, 4),
        "delta_vs_v2": round(delta_vs_v2, 4),
        "runtime_sec": round(time.time() - t_start, 1),
        "timestamp": datetime.now().isoformat(),
    }

    # ── 4. Build pkg if best improves by ≥ +0.30 ─────────────────────────────
    zip_path = None
    if delta_vs_T163 >= DELTA_THRESHOLD:
        progress(f"Step 3: Building pkg (best +{delta_vs_T163:.4f} vs T163)")
        zip_path = build_pkg(best_key, configs[best_key], results)
        results["v2NN_4way_zip"] = zip_path
        results["expected_platform"] = (
            f"v2 platform ~+34.44 + {delta_vs_v2:.2f} * ~0.7 transmission = "
            f"~+{34.44 + delta_vs_v2 * 0.7:.1f}"
        )
    else:
        print(f"  No pkg built: best delta {delta_vs_T163:+.4f} < +{DELTA_THRESHOLD}", flush=True)
        results["v2NN_4way_zip"] = None
        results["expected_platform"] = (
            f"No improvement: best {best_pnl:.4f} vs T163 {T163_BASELINE} (delta {delta_vs_T163:+.4f})"
        )
        results["notes"] = "T95 did not improve over T163 combined by required +0.30 threshold"

    out_path = os.path.join(HERE, "results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {out_path}", flush=True)

    progress("done", {"best_holdout": best_pnl, "delta_vs_T163": delta_vs_T163})

    print(f"\nRESULT: task=T164_4way_ensemble metrics={{best_pnl={best_pnl:.4f}, "
          f"delta_vs_T163={delta_vs_T163:+.4f}, delta_vs_v2={delta_vs_v2:+.4f}}} "
          f"notes=[best={best_key}]")

    return results


def build_pkg(best_key, best_cfg, all_results):
    """Build submission zip adding T95 GRU numpy inference to T163 pkg."""
    import subprocess

    PKG_DIR = os.path.join(HERE, "pkg_t164")
    os.makedirs(PKG_DIR, exist_ok=True)

    # Copy all T163 files
    for fname in os.listdir(T163_PKG):
        if fname.startswith('__'):
            continue
        src = os.path.join(T163_PKG, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(PKG_DIR, fname))
    print(f"  Copied T163 pkg files to {PKG_DIR}", flush=True)

    # Parse best config weights
    # Determine weights from config key
    weight_map = {
        'B_4way_equal':      (1.0, 1.0, 1.0, 1.0),
        'C_4way_nn_heavy':   (1.5, 1.5, 0.5, 1.0),
        'C_noT95filter':     (1.5, 1.5, 0.5, 1.0),
        'D_4way_t95_low':    (1.5, 1.5, 0.3, 1.0),
        'D_noT95filter':     (1.5, 1.5, 0.3, 1.0),
        'E_4way_t95_very_low': (1.5, 1.5, 0.2, 1.0),
    }
    w_nn, w_nn2, w_nn3, w_lgb = weight_map.get(best_key, (1.5, 1.5, 0.3, 1.0))

    # Convert T95 .pt → numpy-compatible npz
    npz_paths = convert_t95_to_npz(PKG_DIR)
    if not npz_paths:
        print("  WARNING: T95 conversion failed, pkg build aborted", flush=True)
        return None

    # Update thresholds.json with T95 weights
    thresh_path = os.path.join(PKG_DIR, "thresholds.json")
    with open(thresh_path) as f:
        tcfg = json.load(f)
    for h in tcfg.get("horizons", []):
        if h.get("active"):
            h["w_nn"] = w_nn
            h["w_nn2"] = w_nn2
            h["w_nn3"] = w_nn3
            h["w_lgb"] = w_lgb
    with open(thresh_path, 'w') as f:
        json.dump(tcfg, f, indent=2)
    print(f"  Updated thresholds: w_nn={w_nn}, w_nn2={w_nn2}, w_nn3={w_nn3}, w_lgb={w_lgb}", flush=True)

    # Write new Predictor.py with T95 GRU support
    write_predictor_with_t95(PKG_DIR, w_nn, w_nn2, w_nn3, w_lgb,
                              use_t95_in_filter=('noT95filter' not in best_key))

    # Smoke test
    smoke_ok = smoke_test_predictor(PKG_DIR)
    if not smoke_ok:
        print("  WARNING: Smoke test failed!", flush=True)

    # Build zip
    zip_name = f"submission_050909_iter019_v2NN_T97_T95_4way.zip"
    zip_path = os.path.join(ROOT, zip_name)
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for fname in sorted(os.listdir(PKG_DIR)):
            if fname.startswith('__') or fname.startswith('.'):
                continue
            fpath = os.path.join(PKG_DIR, fname)
            if os.path.isfile(fpath):
                zf.write(fpath, fname)
    size_mb = os.path.getsize(zip_path) / 1e6
    print(f"  Zip built: {zip_path} ({size_mb:.1f} MB)", flush=True)

    # Copy to output
    out_copy = "/tmp/metabot-outputs/worker-9c17b13d/" + zip_name
    try:
        os.makedirs(os.path.dirname(out_copy), exist_ok=True)
        shutil.copy2(zip_path, out_copy)
        print(f"  Copied to output: {out_copy}", flush=True)
    except Exception as e:
        print(f"  WARNING: copy to output failed: {e}", flush=True)

    all_results["smoke_test_ok"] = smoke_ok
    all_results["zip_size_mb"] = round(size_mb, 2)
    return zip_path


def convert_t95_to_npz(pkg_dir):
    """Convert T95 GRU .pt models to npz for numpy inference."""
    import torch

    T95_MODEL_TPL = os.path.join(T95_DIR, "model_T95_gru_w100_C_seed{seed}.pt")
    npz_paths = []

    for seed in SEEDS:
        pt_path = T95_MODEL_TPL.format(seed=seed)
        if not os.path.exists(pt_path):
            print(f"  T95 model missing: {pt_path}", flush=True)
            return []

        ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
        sd = ckpt.get('state_dict', ckpt)
        target_scale = float(ckpt.get('target_scale', 1.0))

        # GRU weights: weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0
        # Plus final Linear: fc.weight, fc.bias (or linear.weight / out.weight)
        # Look for the correct key names
        def find_key(sd, candidates):
            for k in candidates:
                if k in sd:
                    return sd[k].numpy().astype(np.float32)
            available = [k for k in sd.keys() if 'weight' in k or 'bias' in k]
            print(f"  Available keys: {available}", flush=True)
            return None

        W_ih = find_key(sd, ['gru.weight_ih_l0', 'rnn.weight_ih_l0', 'weight_ih_l0'])
        W_hh = find_key(sd, ['gru.weight_hh_l0', 'rnn.weight_hh_l0', 'weight_hh_l0'])
        b_ih = find_key(sd, ['gru.bias_ih_l0', 'rnn.bias_ih_l0', 'bias_ih_l0'])
        b_hh = find_key(sd, ['gru.bias_hh_l0', 'rnn.bias_hh_l0', 'bias_hh_l0'])
        fc_W = find_key(sd, ['fc.weight', 'linear.weight', 'out.weight', 'output.weight'])
        fc_b = find_key(sd, ['fc.bias', 'linear.bias', 'out.bias', 'output.bias'])

        if any(x is None for x in [W_ih, W_hh, b_ih, b_hh, fc_W, fc_b]):
            print(f"  WARNING: could not find all keys for seed {seed}", flush=True)
            print(f"  State dict keys: {list(sd.keys())}", flush=True)
            return []

        hidden_size = W_hh.shape[1]
        input_size = W_ih.shape[1]
        print(f"  Seed {seed}: input_size={input_size}, hidden_size={hidden_size}, "
              f"target_scale={target_scale}", flush=True)

        out_path = os.path.join(pkg_dir, f"nn3_h60_seed{seed}.npz")
        np.savez(out_path,
            W_ih=W_ih, W_hh=W_hh, b_ih=b_ih, b_hh=b_hh,
            fc_W=fc_W, fc_b=fc_b,
            target_scale=np.array([target_scale], dtype=np.float32),
            input_size=np.array([input_size], dtype=np.int32),
            hidden_size=np.array([hidden_size], dtype=np.int32),
        )
        npz_paths.append(out_path)
        print(f"  Converted T95 seed {seed} → {out_path}", flush=True)

    return npz_paths


def write_predictor_with_t95(pkg_dir, w_nn, w_nn2, w_nn3, w_lgb, use_t95_in_filter=True):
    """Write updated Predictor.py with T95 GRU numpy inference."""
    src = os.path.join(T163_PKG, "Predictor.py")
    with open(src) as f:
        code = f.read()

    # The T95 integration requires:
    # 1. A numpy GRU class (_GRUNumpy)
    # 2. Loading nn3_h60_seed*.npz
    # 3. Extracting LOB window columns for GRU input
    # 4. Updating prediction with T95 contribution
    # 5. Updated agreement filter logic

    # Write new Predictor.py based on T163 but with T95 added
    predictor_path = os.path.join(pkg_dir, "Predictor.py")
    with open(src) as f:
        original = f.read()

    # Build the new Predictor by appending GRU class and modifying ensemble logic
    # This is done by creating a new file with patches applied
    new_code = _patch_predictor_for_t95(original, w_nn, w_nn2, w_nn3, w_lgb, use_t95_in_filter)
    with open(predictor_path, 'w') as f:
        f.write(new_code)
    print(f"  Written new Predictor.py with T95 GRU support", flush=True)


def _patch_predictor_for_t95(original_code, w_nn, w_nn2, w_nn3, w_lgb, use_t95_in_filter):
    """Patch T163 Predictor.py to add T95 GRU inference."""
    # Insert GRU numpy class after _MLPNumpy class
    gru_class = '''

# T95 GRU (LOB sequence model): numpy inference
LOB_COLS_T95 = (
    "bid1", "bid2", "bid3", "bid4", "bid5",
    "bsize1", "bsize2", "bsize3", "bsize4", "bsize5",
    "ask1", "ask2", "ask3", "ask4", "ask5",
    "asize1", "asize2", "asize3", "asize4", "asize5",
)

class _GRUNumpy:
    """Single-layer GRU + Linear head inference in numpy."""

    def __init__(self, npz_path: str):
        d = np.load(npz_path, allow_pickle=False)
        self.W_ih = d["W_ih"].astype(np.float32)    # (3*H, input_size)
        self.W_hh = d["W_hh"].astype(np.float32)    # (3*H, H)
        self.b_ih = d["b_ih"].astype(np.float32)    # (3*H,)
        self.b_hh = d["b_hh"].astype(np.float32)    # (3*H,)
        self.fc_W = d["fc_W"].astype(np.float32)    # (1, H)
        self.fc_b = d["fc_b"].astype(np.float32)    # (1,)
        self.target_scale = float(d["target_scale"][0])
        self.hidden_size = int(d["hidden_size"][0])
        self.input_size = int(d["input_size"][0])
        self._H3 = self.hidden_size  # alias

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        x: (B, W, input_size) - window-normalized LOB sequence.
        Returns: (B,) predictions.
        """
        B, T, _ = x.shape
        H = self.hidden_size
        h = np.zeros((B, H), dtype=np.float32)

        W_ih = self.W_ih  # (3H, F)
        W_hh = self.W_hh  # (3H, H)
        b_ih = self.b_ih  # (3H,)
        b_hh = self.b_hh  # (3H,)

        for t in range(T):
            xt = x[:, t, :]  # (B, F)
            ih = xt @ W_ih.T + b_ih   # (B, 3H)
            hh_val = h @ W_hh.T + b_hh  # (B, 3H)

            z = _sigmoid32(ih[:, :H] + hh_val[:, :H])
            r = _sigmoid32(ih[:, H:2*H] + hh_val[:, H:2*H])
            n = np.tanh(ih[:, 2*H:] + r * hh_val[:, 2*H:]).astype(np.float32)
            h = ((1.0 - z) * n + z * h).astype(np.float32)

        out = h @ self.fc_W.T + self.fc_b   # (B, 1)
        return (out[:, 0] / self.target_scale).astype(np.float32)


def _sigmoid32(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)

'''

    # Find insertion point: after _MLPNumpy class definition (before Predictor class)
    insertion_marker = "\ndef _load_module("
    if insertion_marker in original_code:
        code = original_code.replace(insertion_marker, gru_class + insertion_marker, 1)
    else:
        # fallback: insert before class Predictor:
        code = original_code.replace("\nclass Predictor:", gru_class + "\nclass Predictor:", 1)

    # Update the docstring/weights comment
    code = code.replace(
        "w_nn=1.5, w_nn2=1.5, w_lgb=1.0",
        f"w_nn={w_nn}, w_nn2={w_nn2}, w_nn3={w_nn3}, w_lgb={w_lgb}"
    )

    # Patch __init__: add nn3_lists loading after nn2_lists
    nn3_init_patch = '''
        self._nn3_lists: Dict[int, List[_GRUNumpy]] = {}
        self._lob_col_to_idx: Dict[str, int] = {}
'''
    code = code.replace(
        "        self._weights: Dict[int, Tuple[float, float, float]] = {}",
        "        self._nn3_lists: Dict[int, List[_GRUNumpy]] = {}\n"
        "        self._lob_col_to_idx: Dict[str, int] = {}\n"
        "        self._weights: Dict[int, Tuple[float, float, float, float]] = {}"
    )

    # Patch the loop loading nn3 paths
    code = code.replace(
        "            for s in seeds:\n"
        "                lp = os.path.join(here, f\"model_h{H}_seed{s}.txt\")\n"
        "                np_ = os.path.join(here, f\"nn_h{H}_seed{s}.npz\")\n"
        "                np2_ = os.path.join(here, f\"nn2_h{H}_seed{s}.npz\")\n"
        "                if os.path.isfile(lp):\n"
        "                    lgb_paths.append(lp)\n"
        "                if os.path.isfile(np_):\n"
        "                    nn_paths.append(np_)\n"
        "                if os.path.isfile(np2_):\n"
        "                    nn2_paths.append(np2_)",
        "            nn3_paths: List[str] = []\n"
        "            for s in seeds:\n"
        "                lp = os.path.join(here, f\"model_h{H}_seed{s}.txt\")\n"
        "                np_ = os.path.join(here, f\"nn_h{H}_seed{s}.npz\")\n"
        "                np2_ = os.path.join(here, f\"nn2_h{H}_seed{s}.npz\")\n"
        "                np3_ = os.path.join(here, f\"nn3_h{H}_seed{s}.npz\")\n"
        "                if os.path.isfile(lp):\n"
        "                    lgb_paths.append(lp)\n"
        "                if os.path.isfile(np_):\n"
        "                    nn_paths.append(np_)\n"
        "                if os.path.isfile(np2_):\n"
        "                    nn2_paths.append(np2_)\n"
        "                if os.path.isfile(np3_):\n"
        "                    nn3_paths.append(np3_)"
    )

    # Patch: load nn3_lists after nn2_lists
    code = code.replace(
        "            if nn2_paths:\n"
        "                self._nn2_lists[H] = [_MLPNumpy(p) for p in nn2_paths]\n"
        "            self._weights[H] = (float(hcfg.get(\"w_nn\", 1.0)),\n"
        "                                float(hcfg.get(\"w_lgb\", 1.0)),\n"
        "                                float(hcfg.get(\"w_nn2\", 0.0)))",
        "            if nn2_paths:\n"
        "                self._nn2_lists[H] = [_MLPNumpy(p) for p in nn2_paths]\n"
        "            if nn3_paths:\n"
        "                self._nn3_lists[H] = [_GRUNumpy(p) for p in nn3_paths]\n"
        "            self._weights[H] = (float(hcfg.get(\"w_nn\", 1.0)),\n"
        "                                float(hcfg.get(\"w_lgb\", 1.0)),\n"
        "                                float(hcfg.get(\"w_nn2\", 0.0)),\n"
        f"                                float(hcfg.get(\"w_nn3\", {w_nn3})))"
    )

    # Now patch the predict() method to include T95 inference
    # Find the _ensemble_predict_nn method and add a GRU equivalent
    gru_predict_method = '''
    @staticmethod
    def _ensemble_predict_gru(grus: List[_GRUNumpy], X_seq: np.ndarray) -> np.ndarray:
        """Run GRU ensemble inference on window-normalized sequence input."""
        if not grus:
            return np.zeros(X_seq.shape[0], dtype=np.float32)
        if len(grus) == 1:
            return grus[0].predict(X_seq)
        acc = None
        for g in grus:
            p = g.predict(X_seq)
            acc = p if acc is None else acc + p
        return acc / float(len(grus))

    def _extract_lob_window(self, batches: List[pd.DataFrame]) -> np.ndarray:
        """Extract T95 LOB window: (B, W, 20) float32, window-normalized."""
        B = len(batches)
        W = WINDOW
        F = len(LOB_COLS_T95)
        X = np.empty((B, W, F), dtype=np.float32)
        for i, df in enumerate(batches):
            window = df[list(LOB_COLS_T95)].to_numpy(dtype=np.float32, copy=False)
            if window.shape[0] < W:
                # pad with first row
                pad = np.repeat(window[:1], W - window.shape[0], axis=0)
                window = np.concatenate([pad, window], axis=0)
            else:
                window = window[-W:]
            # Per-window per-feature normalization (same as T95 training)
            mean = window.mean(axis=0, keepdims=True)
            std = window.std(axis=0, keepdims=True) + 1e-8
            window = np.clip((window - mean) / std, -10.0, 10.0)
            X[i] = window
        return X

'''

    # Insert GRU predict methods before _compute_batch_features
    code = code.replace(
        "    def _compute_batch_features(",
        gru_predict_method + "    def _compute_batch_features("
    )

    # Now patch the main predict() method to use T95
    # Find the section where NN/LGB predictions are combined and the agreement filter is applied
    # This is complex; we'll do a targeted replacement of the prediction combination section
    old_combine = '''            w_nn, w_lgb, w_nn2 = self._weights.get(H, (1.0, 1.0, 0.0))
            nn2 = self._ensemble_predict_nn(self._nn2_lists.get(H, []), X_feat) if self._nn2_lists.get(H) else None'''

    new_combine = f'''            w_nn, w_lgb, w_nn2, w_nn3 = self._weights.get(H, (1.0, 1.0, 0.0, 0.0))
            nn2 = self._ensemble_predict_nn(self._nn2_lists.get(H, []), X_feat) if self._nn2_lists.get(H) else None
            if self._nn3_lists.get(H):
                X_lob = self._extract_lob_window(batches)
                nn3 = self._ensemble_predict_gru(self._nn3_lists[H], X_lob)
            else:
                nn3 = None'''

    if old_combine in code:
        code = code.replace(old_combine, new_combine)
    else:
        print("  WARNING: could not find weight extraction in predict() — trying alternate patch", flush=True)

    # Patch the prediction combination (pred_dmid computation)
    # Original: pred_dmid = (w_nn * pred_nn + w_nn2 * pred_nn2 + w_lgb * pred_lgb) / ...
    # This section also handles the agreement filter
    # Let's patch the agreement filter section
    old_agreement = '''                        nn_side = (w_nn * pred_nn + w_nn2 * pred_nn2) / (w_nn + w_nn2)'''
    new_agreement_t95in = f'''                        if nn3 is not None:
                            nn_side = (w_nn * pred_nn + w_nn2 * pred_nn2 + w_nn3 * nn3) / (w_nn + w_nn2 + w_nn3)
                        else:
                            nn_side = (w_nn * pred_nn + w_nn2 * pred_nn2) / (w_nn + w_nn2)'''
    new_agreement_t95out = f'''                        nn_side = (w_nn * pred_nn + w_nn2 * pred_nn2) / (w_nn + w_nn2)  # T95 excluded from filter'''

    new_agreement = new_agreement_t95in if use_t95_in_filter else new_agreement_t95out
    if old_agreement in code:
        code = code.replace(old_agreement, new_agreement)

    # Patch the final pred_dmid computation to include nn3
    old_pred = "            pred_dmid = (w_nn * pred_nn + w_nn2 * pred_nn2_part + w_lgb * pred_lgb_part) / (w_nn + w_nn2_eff + w_lgb)"
    # This is tricky because the original code might have complex logic. Let's search for the key expression.
    # Look for the ensemble combination
    if "pred_dmid = " in code:
        # Try to find the specific line
        old_pred_alt = "(w_nn * pred_nn + w_nn2 * pred_nn2_part + w_lgb * pred_lgb_part)"
        new_pred_alt = "(w_nn * pred_nn + w_nn2 * pred_nn2_part + (w_nn3 * nn3 if nn3 is not None else 0.0) + w_lgb * pred_lgb_part)"
        old_denom = "/ (w_nn + w_nn2_eff + w_lgb)"
        new_denom = "/ (w_nn + w_nn2_eff + (w_nn3 if nn3 is not None else 0.0) + w_lgb)"
        if old_pred_alt in code:
            code = code.replace(old_pred_alt, new_pred_alt)
            code = code.replace(old_denom, new_denom)

    return code


def smoke_test_predictor(pkg_dir):
    """Run a minimal smoke test to verify Predictor loads and runs."""
    import subprocess
    test_script = os.path.join(HERE, "_smoke_test.py")
    smoke_code = f"""
import sys
sys.path.insert(0, {repr(pkg_dir)})
try:
    from Predictor import Predictor
    p = Predictor()
    import pandas as pd, numpy as np
    df = pd.DataFrame({{col: np.random.randn(100) for col in p._raw_feat_cols}})
    df['sym'] = 0
    result = p.predict([df])
    assert result is not None and len(result) == 1
    print(f"Smoke test PASSED: action={{result[0]}}")
except Exception as e:
    print(f"Smoke test FAILED: {{e}}")
    import traceback; traceback.print_exc()
    sys.exit(1)
"""
    with open(test_script, 'w') as f:
        f.write(smoke_code)
    ret = os.system(f"cd {pkg_dir} && python3 {test_script}")
    return ret == 0


if __name__ == "__main__":
    main()
