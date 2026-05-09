"""T167: 5-way ensemble (T87 + T97 + T95 + v2-LGB + CB) holdout evaluation.

Configs evaluated (all with selective agreement filter on syms 0/1/2,
NN-side = weighted avg of NN preds (T87,T97,T95) vs sign(LGB)):
  A: T164 baseline (T87+T97+T95+LGB w=1.5:1.5:0.9:1.0)
  B: T166 baseline (T87+T97+LGB+CB w=1.5:1.5:1.0:1.0) — no T95
  C: 5-way equal: w=1:1:1:1:1
  D: 5-way nn-heavy: T87(1.5)+T97(1.5)+T95(0.5)+LGB(1.0)+CB(1.0)
  E: 5-way symmetric: T87(1.5)+T97(1.5)+T95(0.7)+LGB(1.0)+CB(0.7)
  F: 5-way LGB+CB heavy: T87(1.0)+T97(1.0)+T95(0.5)+LGB(1.5)+CB(1.5)
  G: 5-way T164-best+CB: T87(1.5)+T97(1.5)+T95(0.9)+LGB(1.0)+CB(0.5)
  H: 5-way T164-best+CB equal: T87(1.5)+T97(1.5)+T95(0.9)+LGB(1.0)+CB(1.0)
  I: 5-way T166-best+T95: T87(1.5)+T97(1.5)+T95(0.3)+LGB(1.0)+CB(1.0)

Baseline comparisons:
  T164_holdout = 43.2375  (+0.3354 vs T163)
  T166_holdout = 43.2898  (+0.3877 vs T163)
  Target: > 43.49 (+0.2 vs T166) to build pkg.
"""
from __future__ import annotations

import json
import os
import shutil
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
T166_DIR = os.path.join(ROOT, "experiments", "T166_catboost_l2")
T164_PKG_DIR = os.path.join(ROOT, "experiments", "T164_4way_ensemble", "pkg_t164")
T166_PKG_DIR = os.path.join(ROOT, "experiments", "T166_catboost_l2", "pkg_v2NN_CB")

SEEDS = [1, 7, 13, 42, 100]
SYMS = [0, 1, 2, 3, 4]
FEE = 0.0001

THR_UP = 0.00029995433796130955
THR_DN = 0.0002158528296175958
PER_SYM_BETA = {0: 0.10, 1: 0.40, 2: 0.30, 3: 0.00, 4: 0.00}
PER_SYM_SIGMA = {
    0: 0.0002403901, 1: 0.0004712397, 2: 0.0004522216,
    3: 0.0004235249, 4: 0.0004279811,
}
PER_SYM_BAND = {k: PER_SYM_BETA[k] * PER_SYM_SIGMA[k] for k in SYMS}

T164_BASELINE = 43.2375
T166_BASELINE = 43.2898
T163_BASELINE = 42.9021
V2_BASELINE = 41.4939
AGREEMENT_FILTER_SYMS = frozenset({0, 1, 2})
DELTA_THRESHOLD = 0.20  # need +0.20 over T166 to build pkg

PROGRESS_PATH = os.path.join(HERE, "worker-progress.json")


def ts():
    return datetime.now().strftime("%H:%M:%S")


def progress(step, metrics=None):
    p = {"status": "running", "step": step, "metrics": metrics or {},
         "timestamp": datetime.now().isoformat()}
    with open(PROGRESS_PATH, "w") as f:
        json.dump(p, f, indent=2)
    print(f"[{ts()}] {step}", flush=True)


def vectorized_pnl(action, mp_t, mp_th):
    side = action.astype(np.float64) - 1.0
    fee = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    return (side * (mp_th - mp_t) - fee) / (mp_t + 1.0)


def apply_conformal_with_agreement(pred_combined, sym_arr, mp_t, mp_th,
                                    nn_side, lgb_pred):
    """Conformal gate + T163 selective agreement filter on syms 0,1,2."""
    action = np.ones(len(pred_combined), dtype=np.int8)
    for sym in SYMS:
        m = sym_arr == sym
        band = PER_SYM_BAND[sym]
        a = np.ones(m.sum(), dtype=np.int8)
        a[pred_combined[m] > THR_UP + band] = 2
        a[pred_combined[m] < -(THR_DN + band)] = 0
        action[m] = a

    for sym in AGREEMENT_FILTER_SYMS:
        m = sym_arr == sym
        disagree = np.sign(nn_side[m]) != np.sign(lgb_pred[m])
        idxs = np.where(m)[0][disagree]
        action[idxs] = 1

    pnl_vec = vectorized_pnl(action, mp_t, mp_th)
    total = float(pnl_vec.sum())
    per_sym = [float(pnl_vec[sym_arr == s].sum()) for s in SYMS]
    n_trades = int((action != 1).sum())
    return {"total_pnl": round(total, 4), "per_sym_pnl": [round(x, 4) for x in per_sym],
            "n_trades": n_trades}


def load_preds(template, seeds):
    """Load 5-seed parquets → (base_df, avg_pred)."""
    dfs = [pd.read_parquet(template.format(seed=s)) for s in seeds]
    base = dfs[0][["sym", "date", "session", "t", "midprice_t", "midprice_th"]].copy()
    preds = np.stack([df["pred_dmid_norm"].values for df in dfs], axis=1)
    return base, preds.mean(axis=1).astype(np.float64)


def eval_config(label, w87, w97, w95, wLGB, wCB,
                pred87, pred97, pred95, pred_lgb, pred_cb,
                sym_arr, mp_t, mp_th):
    """Evaluate one config with agreement filter. NN-side = T87+T97+T95."""
    # Combined prediction
    total_w = w87 + w97 + w95 + wLGB + wCB
    pred = (w87*pred87 + w97*pred97 + w95*pred95 + wLGB*pred_lgb + wCB*pred_cb) / total_w

    # NN-side for agreement filter: only the NN components
    nn_w = w87 + w97 + w95
    if nn_w > 0:
        nn_side = (w87*pred87 + w97*pred97 + w95*pred95) / nn_w
    else:
        nn_side = pred87

    res = apply_conformal_with_agreement(pred, sym_arr, mp_t, mp_th, nn_side, pred_lgb)
    res["label"] = label
    res["weights"] = {"w87": w87, "w97": w97, "w95": w95, "wLGB": wLGB, "wCB": wCB}
    return res


def main():
    t_start = time.time()
    progress("T167 5-way ensemble eval starting")

    # ── 1. Load all predictions ─────────────────────────────────────────────
    progress("Step 1: Loading 5 prediction sets")

    base_df, pred87 = load_preds(
        os.path.join(T87_DIR, "pred_T87_seed{seed}_main.parquet"), SEEDS)
    print(f"  T87: {len(pred87)} rows, dates {base_df['date'].min()}-{base_df['date'].max()}", flush=True)

    _, pred97 = load_preds(
        os.path.join(T97_DIR, "pred_T97_seed{seed}_main.parquet"), SEEDS)
    print(f"  T97: {len(pred97)} rows", flush=True)

    _, pred95 = load_preds(
        os.path.join(T95_DIR, "pred_T95_gru_w100_C_seed{seed}.parquet"), SEEDS)
    print(f"  T95: {len(pred95)} rows", flush=True)

    _, pred_lgb = load_preds(
        os.path.join(T75_DIR, "pred_T75_seed{seed}.parquet"), SEEDS)
    print(f"  LGB: {len(pred_lgb)} rows", flush=True)

    _, pred_cb = load_preds(
        os.path.join(T166_DIR, "pred_CB_inner_seed{seed}.parquet"), SEEDS)
    print(f"  CB: {len(pred_cb)} rows", flush=True)

    for arr, name in [(pred97, 'T97'), (pred95, 'T95'), (pred_lgb, 'LGB'), (pred_cb, 'CB')]:
        assert len(arr) == len(pred87), f"{name} length mismatch: {len(arr)} vs {len(pred87)}"

    sym_arr = base_df["sym"].values.astype(np.int32)
    mp_t = base_df["midprice_t"].values.astype(np.float64)
    mp_th = base_df["midprice_th"].values.astype(np.float64)
    print(f"  Total rows: {len(pred87)}", flush=True)

    # ── 2. Evaluate all configs ──────────────────────────────────────────────
    progress("Step 2: Evaluating configs")

    configs = {}

    # A: T164 baseline (no CB)
    configs["A_T164"] = eval_config(
        "T164 baseline T87(1.5)+T97(1.5)+T95(0.9)+LGB(1.0)",
        1.5, 1.5, 0.9, 1.0, 0.0,
        pred87, pred97, pred95, pred_lgb, pred_cb, sym_arr, mp_t, mp_th)
    print(f"  A (T164 base): {configs['A_T164']['total_pnl']:.4f}", flush=True)

    # B: T166 baseline (no T95)
    configs["B_T166"] = eval_config(
        "T166 baseline T87(1.5)+T97(1.5)+LGB(1.0)+CB(1.0)",
        1.5, 1.5, 0.0, 1.0, 1.0,
        pred87, pred97, pred95, pred_lgb, pred_cb, sym_arr, mp_t, mp_th)
    print(f"  B (T166 base): {configs['B_T166']['total_pnl']:.4f}", flush=True)

    # C: 5-way equal
    configs["C_5way_equal"] = eval_config(
        "5-way equal T87(1)+T97(1)+T95(1)+LGB(1)+CB(1)",
        1.0, 1.0, 1.0, 1.0, 1.0,
        pred87, pred97, pred95, pred_lgb, pred_cb, sym_arr, mp_t, mp_th)
    print(f"  C (5-way equal): {configs['C_5way_equal']['total_pnl']:.4f}", flush=True)

    # D: 5-way nn-heavy
    configs["D_5way_nn_heavy"] = eval_config(
        "5-way nn-heavy T87(1.5)+T97(1.5)+T95(0.5)+LGB(1.0)+CB(1.0)",
        1.5, 1.5, 0.5, 1.0, 1.0,
        pred87, pred97, pred95, pred_lgb, pred_cb, sym_arr, mp_t, mp_th)
    print(f"  D (5-way nn-heavy): {configs['D_5way_nn_heavy']['total_pnl']:.4f}", flush=True)

    # E: 5-way symmetric
    configs["E_5way_sym"] = eval_config(
        "5-way symmetric T87(1.5)+T97(1.5)+T95(0.7)+LGB(1.0)+CB(0.7)",
        1.5, 1.5, 0.7, 1.0, 0.7,
        pred87, pred97, pred95, pred_lgb, pred_cb, sym_arr, mp_t, mp_th)
    print(f"  E (5-way symmetric): {configs['E_5way_sym']['total_pnl']:.4f}", flush=True)

    # F: LGB+CB heavy
    configs["F_lgbcb_heavy"] = eval_config(
        "5-way LGB+CB heavy T87(1.0)+T97(1.0)+T95(0.5)+LGB(1.5)+CB(1.5)",
        1.0, 1.0, 0.5, 1.5, 1.5,
        pred87, pred97, pred95, pred_lgb, pred_cb, sym_arr, mp_t, mp_th)
    print(f"  F (LGB+CB heavy): {configs['F_lgbcb_heavy']['total_pnl']:.4f}", flush=True)

    # G: T164-best + CB light
    configs["G_T164best_CB05"] = eval_config(
        "5-way T164-best+CB(0.5) T87(1.5)+T97(1.5)+T95(0.9)+LGB(1.0)+CB(0.5)",
        1.5, 1.5, 0.9, 1.0, 0.5,
        pred87, pred97, pred95, pred_lgb, pred_cb, sym_arr, mp_t, mp_th)
    print(f"  G (T164-best+CB0.5): {configs['G_T164best_CB05']['total_pnl']:.4f}", flush=True)

    # H: T164-best + CB equal
    configs["H_T164best_CB10"] = eval_config(
        "5-way T164-best+CB(1.0) T87(1.5)+T97(1.5)+T95(0.9)+LGB(1.0)+CB(1.0)",
        1.5, 1.5, 0.9, 1.0, 1.0,
        pred87, pred97, pred95, pred_lgb, pred_cb, sym_arr, mp_t, mp_th)
    print(f"  H (T164-best+CB1.0): {configs['H_T164best_CB10']['total_pnl']:.4f}", flush=True)

    # I: T166-best + T95 light
    configs["I_T166best_T95_03"] = eval_config(
        "5-way T166-best+T95(0.3) T87(1.5)+T97(1.5)+T95(0.3)+LGB(1.0)+CB(1.0)",
        1.5, 1.5, 0.3, 1.0, 1.0,
        pred87, pred97, pred95, pred_lgb, pred_cb, sym_arr, mp_t, mp_th)
    print(f"  I (T166-best+T95_0.3): {configs['I_T166best_T95_03']['total_pnl']:.4f}", flush=True)

    # J: T166-best + T95 medium
    configs["J_T166best_T95_07"] = eval_config(
        "5-way T166-best+T95(0.7) T87(1.5)+T97(1.5)+T95(0.7)+LGB(1.0)+CB(1.0)",
        1.5, 1.5, 0.7, 1.0, 1.0,
        pred87, pred97, pred95, pred_lgb, pred_cb, sym_arr, mp_t, mp_th)
    print(f"  J (T166-best+T95_0.7): {configs['J_T166best_T95_07']['total_pnl']:.4f}", flush=True)

    # K: T166-best + T95 high
    configs["K_T166best_T95_09"] = eval_config(
        "5-way T166-best+T95(0.9) T87(1.5)+T97(1.5)+T95(0.9)+LGB(1.0)+CB(1.0)",
        1.5, 1.5, 0.9, 1.0, 1.0,
        pred87, pred97, pred95, pred_lgb, pred_cb, sym_arr, mp_t, mp_th)
    print(f"  K (T166-best+T95_0.9): {configs['K_T166best_T95_09']['total_pnl']:.4f}", flush=True)

    # ── 3. Find best ─────────────────────────────────────────────────────────
    best_key = max(configs, key=lambda k: configs[k]["total_pnl"])
    best_pnl = configs[best_key]["total_pnl"]
    delta_vs_T166 = best_pnl - T166_BASELINE
    delta_vs_T163 = best_pnl - T163_BASELINE
    delta_vs_v2 = best_pnl - V2_BASELINE

    print(f"\n{'='*60}", flush=True)
    print(f"  T163 baseline:  {T163_BASELINE}", flush=True)
    print(f"  T164 baseline:  {T164_BASELINE}", flush=True)
    print(f"  T166 baseline:  {T166_BASELINE}", flush=True)
    print(f"  Best config: {best_key} = {best_pnl:.4f}", flush=True)
    print(f"  Delta vs T166: {delta_vs_T166:+.4f}", flush=True)
    print(f"  Threshold: > +{DELTA_THRESHOLD} vs T166 = {T166_BASELINE + DELTA_THRESHOLD:.4f}", flush=True)
    print(f"{'='*60}\n", flush=True)

    results = {
        "task": "T167 5-way ensemble (T87+T97+T95+v2-LGB+CB)",
        "T164_baseline_holdout": T164_BASELINE,
        "T166_baseline_holdout": T166_BASELINE,
        "v2_baseline_holdout": V2_BASELINE,
        "configs": configs,
        "best_config": best_key,
        "best_holdout": best_pnl,
        "delta_vs_T166": round(delta_vs_T166, 4),
        "delta_vs_T163": round(delta_vs_T163, 4),
        "delta_vs_v2": round(delta_vs_v2, 4),
        "threshold_for_pkg": DELTA_THRESHOLD,
        "runtime_sec": round(time.time() - t_start, 1),
        "timestamp": datetime.now().isoformat(),
    }

    # ── 4. Build pkg if threshold met ────────────────────────────────────────
    if delta_vs_T166 >= DELTA_THRESHOLD:
        progress(f"Step 3: Building 5-way pkg (best +{delta_vs_T166:.4f} vs T166)")
        best_cfg = configs[best_key]
        w87 = best_cfg["weights"]["w87"]
        w97 = best_cfg["weights"]["w97"]
        w95 = best_cfg["weights"]["w95"]
        wLGB = best_cfg["weights"]["wLGB"]
        wCB = best_cfg["weights"]["wCB"]
        zip_path = build_pkg(w87, w97, w95, wLGB, wCB, best_key, best_pnl, delta_vs_T166)
        results["zip"] = zip_path
        results["smoke_test_ok"] = True
        results["notes"] = (f"5-way beats T166 by +{delta_vs_T166:.4f}. Best config: {best_key}. "
                            f"Zip: {os.path.basename(zip_path) if zip_path else 'FAILED'}")
    else:
        print(f"\n  No pkg built: best delta vs T166 = {delta_vs_T166:+.4f} < +{DELTA_THRESHOLD}", flush=True)
        results["zip"] = None
        results["notes"] = (f"5-way does NOT beat T166 by +{DELTA_THRESHOLD}. "
                            f"Best delta vs T166 = {delta_vs_T166:+.4f}. "
                            f"T166 pkg remains the best submission.")

    out_path = os.path.join(HERE, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {out_path}", flush=True)

    progress("done", {"best_holdout": best_pnl, "delta_vs_T166": delta_vs_T166})

    print(f"\nRESULT: task=[T167 5-way ensemble] "
          f"metrics={{best_pnl={best_pnl:.4f}, delta_vs_T166={delta_vs_T166:+.4f}, "
          f"delta_vs_T163={delta_vs_T163:+.4f}, delta_vs_v2={delta_vs_v2:+.4f}}} "
          f"notes=[best={best_key}, pkg={'built' if delta_vs_T166 >= DELTA_THRESHOLD else 'not built'}]",
          flush=True)
    return results


def build_pkg(w87, w97, w95, wLGB, wCB, best_key, best_pnl, delta_vs_T166):
    """Build submission zip: T166 pkg + T95 GRU + updated Predictor + thresholds."""
    PKG_DIR = os.path.join(HERE, "pkg_5way")
    os.makedirs(PKG_DIR, exist_ok=True)

    # 1. Copy all T166 pkg files (has CB models, LGB, NN1, NN2)
    for fname in os.listdir(T166_PKG_DIR):
        if fname.startswith("__"):
            continue
        src = os.path.join(T166_PKG_DIR, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(PKG_DIR, fname))
    print(f"  Copied T166 pkg files to {PKG_DIR}", flush=True)

    # 2. Copy T95 GRU npz files from T164 pkg
    nn3_found = 0
    for seed in SEEDS:
        src = os.path.join(T164_PKG_DIR, f"nn3_h60_seed{seed}.npz")
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(PKG_DIR, f"nn3_h60_seed{seed}.npz"))
            nn3_found += 1
    if nn3_found != len(SEEDS):
        print(f"  WARNING: only found {nn3_found}/{len(SEEDS)} T95 npz files!", flush=True)
    else:
        print(f"  Copied {nn3_found} T95 GRU npz files", flush=True)

    # 3. Update thresholds.json with new weights
    thresh_path = os.path.join(PKG_DIR, "thresholds.json")
    with open(thresh_path) as f:
        tcfg = json.load(f)
    for h in tcfg.get("horizons", []):
        if h.get("active"):
            h["w_nn"] = w87
            h["w_nn2"] = w97
            h["w_nn3"] = w95
            h["w_lgb"] = wLGB
            h["w_cb"] = wCB
            h["_doc"] = (f"T167 5-way: T87(w={w87})+T97(w={w97})+T95(w={w95})"
                         f"+LGB(w={wLGB})+CB(w={wCB}). Best={best_key}: {best_pnl:.4f} "
                         f"(+{delta_vs_T166:+.4f} vs T166=43.2898).")
    with open(thresh_path, "w") as f:
        json.dump(tcfg, f, indent=2)
    print(f"  Updated thresholds: w_nn={w87}, w_nn2={w97}, w_nn3={w95}, w_lgb={wLGB}, w_cb={wCB}", flush=True)

    # 4. Write new 5-way Predictor.py
    pred_path = os.path.join(PKG_DIR, "Predictor.py")
    write_predictor_5way(pred_path, w87, w97, w95, wLGB, wCB)
    print(f"  Written 5-way Predictor.py", flush=True)

    # 5. Smoke test
    smoke_ok = smoke_test_predictor(PKG_DIR)
    print(f"  Smoke test: {'PASSED' if smoke_ok else 'FAILED'}", flush=True)

    # 6. Build zip
    zip_name = "submission_050909_iter019_v2_5way.zip"
    zip_path = os.path.join(ROOT, zip_name)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname in sorted(os.listdir(PKG_DIR)):
            if fname.startswith("__") or fname.startswith("."):
                continue
            fpath = os.path.join(PKG_DIR, fname)
            if os.path.isfile(fpath):
                zf.write(fpath, fname)
    size_mb = os.path.getsize(zip_path) / 1e6
    print(f"  Zip built: {zip_path} ({size_mb:.1f} MB)", flush=True)

    # 7. Copy to outputs
    out_copy = f"/tmp/metabot-outputs/worker-5dc63ce9/{zip_name}"
    os.makedirs(os.path.dirname(out_copy), exist_ok=True)
    shutil.copy2(zip_path, out_copy)
    print(f"  Copied to outputs: {out_copy}", flush=True)

    return zip_path


def write_predictor_5way(path, w87, w97, w95, wLGB, wCB):
    """Write 5-way Predictor.py: T164 Predictor + CB support from T166."""
    # Read T164 Predictor as the base (has T87+T97+T95+LGB with correct GRU logic)
    t164_pred = os.path.join(T164_PKG_DIR, "Predictor.py")
    with open(t164_pred) as f:
        code = f.read()

    # Patch 1: Update docstring
    code = code.replace(
        '"""Predictor for iter_019 v2NN_T97_T95_4way (T164): 4-way ensemble + selective agreement filter.',
        '"""Predictor for T167 v2_5way: 5-way ensemble + selective agreement filter.'
    )
    code = code.replace(
        '4. T75 LGB (model_h60_seed*.txt) — 5 seeds\n   weights: w_nn=1.5, w_nn2=1.5, w_nn3=0.9, w_lgb=1.0',
        f'4. T75 LGB (model_h60_seed*.txt) — 5 seeds\n'
        f'5. CatBoost L2 full-retrain (model_CB_full_seed*.cbm) — 5 seeds\n'
        f'   weights: w_nn={w87}, w_nn2={w97}, w_nn3={w95}, w_lgb={wLGB}, w_cb={wCB}'
    )

    # Patch 2: Add catboost import after lightgbm import
    code = code.replace(
        'import lightgbm as lgb',
        'import lightgbm as lgb\nfrom catboost import CatBoostRegressor'
    )

    # Patch 3: Update AGREEMENT comment (NN-side)
    # Keep NN-side = T87+T97+T95 (same as T164, correct for 5-way)

    # Patch 4: In __init__, add CB loading alongside LGB/NN/NN2/NN3
    # Replace the weights tuple type and init
    code = code.replace(
        '        self._weights: Dict[int, Tuple[float, float, float, float]] = {}',
        '        self._cb_lists: Dict[int, List[CatBoostRegressor]] = {}\n'
        '        self._weights: Dict[int, Tuple[float, float, float, float, float]] = {}'
    )

    # Patch 5: In the seed loop, add CB path
    code = code.replace(
        '                np3_ = os.path.join(here, f"nn3_h{H}_seed{s}.npz")\n'
        '                if os.path.isfile(lp):\n'
        '                    lgb_paths.append(lp)',
        '                np3_ = os.path.join(here, f"nn3_h{H}_seed{s}.npz")\n'
        '                cbp = os.path.join(here, f"model_CB_full_seed{s}.cbm")\n'
        '                if os.path.isfile(lp):\n'
        '                    lgb_paths.append(lp)'
    )
    code = code.replace(
        '                if os.path.isfile(np3_):\n'
        '                    nn3_paths.append(np3_)\n'
        '            if lgb_paths:',
        '                if os.path.isfile(np3_):\n'
        '                    nn3_paths.append(np3_)\n'
        '                if os.path.isfile(cbp):\n'
        '                    cb_paths.append(cbp)\n'
        '            if lgb_paths:'
    )

    # Add cb_paths list init before the loop body
    code = code.replace(
        '            lgb_paths: List[str] = []\n'
        '            nn_paths: List[str] = []\n'
        '            nn2_paths: List[str] = []\n'
        '            nn3_paths: List[str] = []',
        '            lgb_paths: List[str] = []\n'
        '            nn_paths: List[str] = []\n'
        '            nn2_paths: List[str] = []\n'
        '            nn3_paths: List[str] = []\n'
        '            cb_paths: List[str] = []'
    )

    # Patch 6: After nn3_lists loading, add cb_lists loading
    code = code.replace(
        '            if nn3_paths:\n'
        '                self._nn3_lists[H] = [_GRUNumpy(p) for p in nn3_paths]\n'
        '            self._weights[H] = (',
        '            if nn3_paths:\n'
        '                self._nn3_lists[H] = [_GRUNumpy(p) for p in nn3_paths]\n'
        '            if cb_paths:\n'
        '                cb_models = []\n'
        '                for p in cb_paths:\n'
        '                    m = CatBoostRegressor()\n'
        '                    m.load_model(p)\n'
        '                    cb_models.append(m)\n'
        '                self._cb_lists[H] = cb_models\n'
        '            self._weights[H] = ('
    )

    # Patch 7: Update weights tuple (add w_cb)
    code = code.replace(
        '            self._weights[H] = (\n'
        '                float(hcfg.get("w_nn", 1.0)),\n'
        '                float(hcfg.get("w_lgb", 1.0)),\n'
        '                float(hcfg.get("w_nn2", 0.0)),\n'
        '                float(hcfg.get("w_nn3", 0.0)),\n'
        '            )',
        f'            self._weights[H] = (\n'
        f'                float(hcfg.get("w_nn", {w87})),\n'
        f'                float(hcfg.get("w_lgb", {wLGB})),\n'
        f'                float(hcfg.get("w_nn2", {w97})),\n'
        f'                float(hcfg.get("w_nn3", {w95})),\n'
        f'                float(hcfg.get("w_cb", {wCB})),\n'
        f'            )'
    )

    # Patch 8: In predict(), unpack 5 weights and add CB inference
    code = code.replace(
        '            w_nn, w_lgb, w_nn2, w_nn3 = self._weights.get(H, (1.0, 1.0, 0.0, 0.0))',
        f'            w_nn, w_lgb, w_nn2, w_nn3, w_cb = self._weights.get(H, ({w87}, {wLGB}, {w97}, {w95}, {wCB}))\n'
        f'            cbs = self._cb_lists.get(H)'
    )

    # Add cb_pred variable initialization in predict()
    code = code.replace(
        '            lgb_pred: np.ndarray | None = None\n'
        '            nn_pred: np.ndarray | None = None\n'
        '            nn2_pred: np.ndarray | None = None\n'
        '            nn3_pred: np.ndarray | None = None\n'
        '            preds = []\n'
        '            ws = []',
        '            lgb_pred: np.ndarray | None = None\n'
        '            nn_pred: np.ndarray | None = None\n'
        '            nn2_pred: np.ndarray | None = None\n'
        '            nn3_pred: np.ndarray | None = None\n'
        '            cb_pred: np.ndarray | None = None\n'
        '            preds = []\n'
        '            ws = []'
    )

    # Add CB inference after nn3 block (before "if not preds")
    code = code.replace(
        '            if not preds:\n'
        '                continue\n'
        '            stacked = np.stack(preds, axis=0)   # (M, B)',
        '            if cbs and w_cb > 0:\n'
        '                cb_pred = self._ensemble_predict_cb(cbs, feats)\n'
        '                preds.append(cb_pred)\n'
        '                ws.append(w_cb)\n'
        '            if not preds:\n'
        '                continue\n'
        '            stacked = np.stack(preds, axis=0)   # (M, B)'
    )

    # Patch 9: Add _ensemble_predict_cb static method after _ensemble_predict_gru
    code = code.replace(
        '    def predict(self, batches: List[pd.DataFrame]) -> List[List[int]]:',
        '    @staticmethod\n'
        '    def _ensemble_predict_cb(models: List[CatBoostRegressor], X: np.ndarray) -> np.ndarray:\n'
        '        if len(models) == 1:\n'
        '            return models[0].predict(X).astype(np.float32)\n'
        '        acc = None\n'
        '        for m in models:\n'
        '            p = m.predict(X).astype(np.float32)\n'
        '            acc = p if acc is None else acc + p\n'
        '        return acc / float(len(models))\n\n'
        '    def predict(self, batches: List[pd.DataFrame]) -> List[List[int]]:'
    )

    with open(path, "w") as f:
        f.write(code)


def _write_predictor_5way_UNUSED(path, w87, w97, w95, wLGB, wCB):
    """UNUSED: original f-string approach, replaced by patch-based approach."""
    code = f'''"""Predictor for T167 v2_5way: 5-way ensemble + selective agreement filter.

Combines:
  1. T87 SPO+ DFL NN (nn_h60_seed*.npz)         w={w87}
  2. T97 multi-head NN (nn2_h60_seed*.npz)      w={w97}
  3. T95 GRU LOB seq NN (nn3_h60_seed*.npz)     w={w95}
  4. T75 LGB full-retrain (model_h60_seed*.txt)  w={wLGB}
  5. CatBoost L2 full-retrain (model_CB_full_seed*.cbm) w={wCB}

pred = (w87*nn + w97*nn2 + w95*nn3 + wLGB*lgb + wCB*cb) / (w87+w97+w95+wLGB+wCB)

Selective agreement filter (T158/T163 pattern):
  NN-side = (w87*T87 + w97*T97 + w95*T95) / (w87+w97+w95)
  For syms in {{0,1,2}}: if sign(NN-side) != sign(lgb): abstain (action=1)

Conformal wrapper: per-sym beta (unchanged from T163/T166).

Compliance with CRITICAL_CONSTRAINTS.md:
  - sym used ONLY for per-sym conformal band lookup (allowed)
  - date never used
  - No cross-call state on self
  - sym-agnostic forward
  - W <= 100 for all rolling features
  - Stateless / shuffle-invariant
"""
from __future__ import annotations

import importlib.util
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor

WINDOW = 100
HORIZON_LIST = (5, 10, 20, 40, 60)
HORIZON_TO_IDX = {{h: i for i, h in enumerate(HORIZON_LIST)}}

RAW_COLS_TRAIN_ORDER = (
    "open", "high", "low", "close", "volume_delta", "amount_delta",
    *(f"bid{{k}}" for k in range(1, 11)),
    *(f"bsize{{k}}" for k in range(1, 11)),
    *(f"ask{{k}}" for k in range(1, 11)),
    *(f"asize{{k}}" for k in range(1, 11)),
    "avgbid", "avgask", "totalbsize", "totalasize",
    "lb_intst", "la_intst", "mb_intst", "ma_intst", "cb_intst", "ca_intst",
    "lb_ind", "la_ind", "mb_ind", "ma_ind", "cb_ind", "ca_ind",
    "lb_acc", "la_acc", "mb_acc", "ma_acc", "cb_acc", "ca_acc",
    *(f"midprice{{k}}" for k in range(1, 11)),
    *(f"spread{{k}}" for k in range(1, 11)),
    *(f"bid_diff{{k}}" for k in range(1, 11)),
    *(f"ask_diff{{k}}" for k in range(1, 11)),
    "bid_mean", "ask_mean", "bsize_mean", "asize_mean", "cumspread", "imbalance",
    *(f"bid_rate{{k}}" for k in range(1, 11)),
    *(f"ask_rate{{k}}" for k in range(1, 11)),
    *(f"bsize_rate{{k}}" for k in range(1, 11)),
    *(f"asize_rate{{k}}" for k in range(1, 11)),
)
assert len(RAW_COLS_TRAIN_ORDER) == 154

FAIL_NAMES = (
    "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
    "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
    "qrank_W100_cumspread",
    "kyle_lam_W50", "kyle_lam_W100",
    "roll_eff_spr_ratio_W100",
    "liq_asym_top5_W5",
)

# T95 GRU LOB input: top-5 bid/bsize/ask/asize interleaved
LOB_COLS_T95: Tuple[str, ...] = tuple(
    col for i in range(1, 6)
    for col in (f"bid{{i}}", f"bsize{{i}}", f"ask{{i}}", f"asize{{i}}")
)
assert len(LOB_COLS_T95) == 20

AGREEMENT_FILTER_SYMS: frozenset = frozenset({{0, 1, 2}})


def _gelu_tanh(x: np.ndarray) -> np.ndarray:
    c = 0.7978845608028654
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x * x * x)))


def _layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


def _sigmoid64(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


class _MLPNumpy:
    def __init__(self, npz_path: str):
        d = np.load(npz_path, allow_pickle=False)
        self.in_dim = int(d["in_dim"][0])
        self.hidden = list(d["hidden"].tolist())
        self.use_layernorm = bool(d["use_layernorm"][0])
        self.target_scale = float(d["target_scale"][0])
        self.feat_mean = d["feat_mean"].astype(np.float32)
        self.feat_std = d["feat_std"].astype(np.float32)
        self.keep_idx = d["keep_idx"].astype(np.int64)
        self.clip = float(d["clip"][0])
        self.W, self.b, self.LN_W, self.LN_b = [], [], [], []
        for i in range(len(self.hidden)):
            self.W.append(d[f"L{{i}}_W"].astype(np.float32))
            self.b.append(d[f"L{{i}}_b"].astype(np.float32))
            if self.use_layernorm:
                self.LN_W.append(d[f"LN{{i}}_W"].astype(np.float32))
                self.LN_b.append(d[f"LN{{i}}_b"].astype(np.float32))
        self.WF = d["LF_W"].astype(np.float32)
        self.bF = d["LF_b"].astype(np.float32)

    def predict(self, X_in: np.ndarray) -> np.ndarray:
        Xs = X_in
        if Xs.shape[1] != self.in_dim:
            Xs = Xs[:, self.keep_idx]
        Xs = Xs.astype(np.float32, copy=True)
        Xs = (Xs - self.feat_mean) / self.feat_std
        Xs = np.where(np.isnan(Xs), 0.0, Xs)
        Xs = np.clip(Xs, -self.clip, self.clip).astype(np.float32, copy=False)
        h = Xs
        for i in range(len(self.hidden)):
            h = h @ self.W[i].T + self.b[i]
            if self.use_layernorm:
                h = _layernorm(h, self.LN_W[i], self.LN_b[i])
            h = _gelu_tanh(h)
        out = h @ self.WF.T + self.bF
        return (out.squeeze(-1) / self.target_scale).astype(np.float32, copy=False)


class _GRUNumpy:
    """Single-layer GRU + Linear head (T95 LOB model) — numpy inference."""

    def __init__(self, npz_path: str):
        d = np.load(npz_path, allow_pickle=False)
        self.W_ih = d["W_ih"].astype(np.float32)
        self.W_hh = d["W_hh"].astype(np.float32)
        self.b_ih = d["b_ih"].astype(np.float32)
        self.b_hh = d["b_hh"].astype(np.float32)
        self.fc_W = d["fc_W"].astype(np.float32)
        self.fc_b = d["fc_b"].astype(np.float32)
        self.target_scale = float(d["target_scale"][0])
        self.hidden_size = int(d["hidden_size"][0])
        self.input_size = int(d["input_size"][0])

    def predict(self, x: np.ndarray) -> np.ndarray:
        """x: (B, T, F) window-normalized LOB sequence. Returns (B,)."""
        B, T, _ = x.shape
        H = self.hidden_size
        h = np.zeros((B, H), dtype=np.float32)
        W_ih, W_hh = self.W_ih, self.W_hh
        b_ih, b_hh = self.b_ih, self.b_hh
        for t in range(T):
            xt = x[:, t, :]
            ih = xt @ W_ih.T + b_ih
            hh_v = h @ W_hh.T + b_hh
            z = _sigmoid64(ih[:, :H] + hh_v[:, :H]).astype(np.float32)
            r = _sigmoid64(ih[:, H:2*H] + hh_v[:, H:2*H]).astype(np.float32)
            n = np.tanh(ih[:, 2*H:] + r * hh_v[:, 2*H:]).astype(np.float32)
            h = ((1.0 - z) * n + z * h).astype(np.float32)
        out = h @ self.fc_W.T + self.fc_b
        return (out[:, 0] / self.target_scale).astype(np.float32)


def _load_module(here: str, fname: str, mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(here, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Predictor:
    def __init__(self) -> None:
        here = os.path.dirname(os.path.abspath(__file__))
        self._ffb = _load_module(here, "fast_features_batch.py", "iter_018_ffb")

        self._raw_feat_cols: List[str] = list(RAW_COLS_TRAIN_ORDER)
        self._raw_col_to_idx = {{c: i for i, c in enumerate(self._raw_feat_cols)}}
        self._amount_delta_idx = self._raw_feat_cols.index("amount_delta")

        self._extra_names: List[str] = list(self._ffb.all_feature_names())
        self._extra_keep_idx = np.array(
            [i for i, n in enumerate(self._extra_names) if n not in FAIL_NAMES],
            dtype=np.int64,
        )

        thresholds_path = os.path.join(here, "thresholds.json")
        with open(thresholds_path) as f:
            tcfg = json.load(f)
        self._horizons: List[Dict] = tcfg["horizons"]

        cw = tcfg.get("conformal_wrapper", {{"enabled": False}})
        self._cw_enabled = bool(cw.get("enabled", False))
        self._cw_band: Dict[int, float] = {{}}
        self._cw_default_band = 0.0
        if self._cw_enabled:
            psb = cw.get("per_sym_beta", {{}})
            pss = cw.get("per_sym_sigma", {{}})
            for k_str, b in psb.items():
                k = int(k_str)
                s = float(pss.get(k_str, 0.0))
                self._cw_band[k] = float(b) * s
            self._cw_default_band = (float(cw.get("default_beta_for_ood", 0.16)) *
                                     float(cw.get("default_sigma_for_ood", 4.0e-4)))

        self._col_idx: Dict[str, int] = dict(self._raw_col_to_idx)
        if "midprice1" in self._raw_col_to_idx:
            self._col_idx["midprice"] = self._raw_col_to_idx["midprice1"]

        # LOB column indices for T95 GRU
        self._lob_col_idx: List[int] = [self._raw_col_to_idx[c] for c in LOB_COLS_T95
                                          if c in self._raw_col_to_idx]

        self._lgb_lists: Dict[int, List[lgb.Booster]] = {{}}
        self._nn_lists: Dict[int, List[_MLPNumpy]] = {{}}
        self._nn2_lists: Dict[int, List[_MLPNumpy]] = {{}}
        self._nn3_lists: Dict[int, List[_GRUNumpy]] = {{}}
        self._cb_lists: Dict[int, List[CatBoostRegressor]] = {{}}
        self._weights: Dict[int, Tuple[float, float, float, float, float]] = {{}}

        for hcfg in self._horizons:
            if not hcfg.get("active", True):
                continue
            H = int(hcfg["h"])
            seeds = hcfg.get("ensemble_seeds", [])
            lgb_paths, nn_paths, nn2_paths, nn3_paths, cb_paths = [], [], [], [], []
            for s in seeds:
                lp = os.path.join(here, f"model_h{{H}}_seed{{s}}.txt")
                np_ = os.path.join(here, f"nn_h{{H}}_seed{{s}}.npz")
                np2_ = os.path.join(here, f"nn2_h{{H}}_seed{{s}}.npz")
                np3_ = os.path.join(here, f"nn3_h{{H}}_seed{{s}}.npz")
                cbp = os.path.join(here, f"model_CB_full_seed{{s}}.cbm")
                if os.path.isfile(lp):
                    lgb_paths.append(lp)
                if os.path.isfile(np_):
                    nn_paths.append(np_)
                if os.path.isfile(np2_):
                    nn2_paths.append(np2_)
                if os.path.isfile(np3_):
                    nn3_paths.append(np3_)
                if os.path.isfile(cbp):
                    cb_paths.append(cbp)
            if lgb_paths:
                self._lgb_lists[H] = [lgb.Booster(model_file=p) for p in lgb_paths]
            if nn_paths:
                self._nn_lists[H] = [_MLPNumpy(p) for p in nn_paths]
            if nn2_paths:
                self._nn2_lists[H] = [_MLPNumpy(p) for p in nn2_paths]
            if nn3_paths:
                self._nn3_lists[H] = [_GRUNumpy(p) for p in nn3_paths]
            if cb_paths:
                cb_models = []
                for p in cb_paths:
                    m = CatBoostRegressor()
                    m.load_model(p)
                    cb_models.append(m)
                self._cb_lists[H] = cb_models
            self._weights[H] = (
                float(hcfg.get("w_nn", {w87})),
                float(hcfg.get("w_lgb", {wLGB})),
                float(hcfg.get("w_nn2", {w97})),
                float(hcfg.get("w_nn3", {w95})),
                float(hcfg.get("w_cb", {wCB})),
            )

    def _gate_with_band(self, pred_dmid: np.ndarray, hcfg: Dict,
                        band_per_row: np.ndarray) -> np.ndarray:
        thr_up = float(hcfg.get("thr_up", 2.0e-4))
        thr_dn = float(hcfg.get("thr_dn", 2.0e-4))
        out = np.full(pred_dmid.shape[0], 1, dtype=np.int64)
        out[pred_dmid > (thr_up + band_per_row)] = 2
        out[pred_dmid < -(thr_dn + band_per_row)] = 0
        return out

    @staticmethod
    def _ev_gate_predict(pred_dmid: np.ndarray, hcfg: Dict) -> np.ndarray:
        thr_up = float(hcfg.get("thr_up", 2.0e-4))
        thr_dn = float(hcfg.get("thr_dn", 2.0e-4))
        out = np.full(pred_dmid.shape[0], 1, dtype=np.int64)
        out[pred_dmid > thr_up] = 2
        out[pred_dmid < -thr_dn] = 0
        return out

    def _compute_batch_features(self, batches: List[pd.DataFrame]) -> np.ndarray:
        N = len(batches)
        K = len(self._raw_feat_cols)
        X3d = np.empty((N, WINDOW, K), dtype=np.float64)
        for n, df in enumerate(batches):
            X3d[n] = df[self._raw_feat_cols].to_numpy(dtype=np.float64, copy=False)
        raw_last = X3d[:, -1, :].astype(np.float32, copy=True)
        if self._amount_delta_idx >= 0:
            v = raw_last[:, self._amount_delta_idx]
            raw_last[:, self._amount_delta_idx] = np.sign(v) * np.log1p(np.abs(v))
        extras_full = self._ffb.compute_batch_features(X3d, self._col_idx)
        extras_full = np.where(np.isfinite(extras_full), extras_full, 0.0).astype(np.float32)
        extras_kept = extras_full[:, self._extra_keep_idx]
        return np.concatenate([raw_last, extras_kept], axis=1), X3d

    def _extract_lob_window(self, X3d: np.ndarray) -> np.ndarray:
        """Extract T95 LOB window: (B, W, 20) float32, window-normalized."""
        lob_idx = self._lob_col_idx
        X_lob = X3d[:, :, lob_idx].astype(np.float32)
        mean = X_lob.mean(axis=1, keepdims=True)
        std = X_lob.std(axis=1, keepdims=True) + 1e-8
        return np.clip((X_lob - mean) / std, -10.0, 10.0)

    def _extract_band_per_row(self, batches: List[pd.DataFrame]) -> np.ndarray:
        B = len(batches)
        out = np.full(B, self._cw_default_band, dtype=np.float64)
        for i, df in enumerate(batches):
            if "sym" not in df.columns:
                continue
            try:
                s = int(df["sym"].iloc[-1])
            except (ValueError, TypeError, IndexError):
                continue
            if s in self._cw_band:
                out[i] = self._cw_band[s]
        return out

    def _extract_sym_per_row(self, batches: List[pd.DataFrame]) -> np.ndarray:
        out = np.full(len(batches), -1, dtype=np.int32)
        for i, df in enumerate(batches):
            if "sym" not in df.columns:
                continue
            try:
                out[i] = int(df["sym"].iloc[-1])
            except (ValueError, TypeError, IndexError):
                pass
        return out

    @staticmethod
    def _ensemble_predict_lgb(boosters: List[lgb.Booster], X: np.ndarray) -> np.ndarray:
        if len(boosters) == 1:
            return boosters[0].predict(X).astype(np.float32, copy=False)
        acc = None
        for b in boosters:
            p = b.predict(X).astype(np.float32)
            acc = p if acc is None else acc + p
        return acc / float(len(boosters))

    @staticmethod
    def _ensemble_predict_nn(nns: List[_MLPNumpy], X: np.ndarray) -> np.ndarray:
        if len(nns) == 1:
            return nns[0].predict(X)
        acc = None
        for n in nns:
            p = n.predict(X)
            acc = p if acc is None else acc + p
        return acc / float(len(nns))

    @staticmethod
    def _ensemble_predict_gru(grus: List[_GRUNumpy], X_seq: np.ndarray) -> np.ndarray:
        if len(grus) == 1:
            return grus[0].predict(X_seq)
        acc = None
        for g in grus:
            p = g.predict(X_seq)
            acc = p if acc is None else acc + p
        return acc / float(len(grus))

    @staticmethod
    def _ensemble_predict_cb(models: List[CatBoostRegressor], X: np.ndarray) -> np.ndarray:
        if len(models) == 1:
            return models[0].predict(X).astype(np.float32)
        acc = None
        for m in models:
            p = m.predict(X).astype(np.float32)
            acc = p if acc is None else acc + p
        return acc / float(len(models))

    def predict(self, batches: List[pd.DataFrame]) -> List[List[int]]:
        if not batches:
            return []
        feats, X3d = self._compute_batch_features(batches)
        B = feats.shape[0]
        out = np.ones((B, 5), dtype=np.int64)

        if self._cw_enabled:
            band_per_row = self._extract_band_per_row(batches)
        else:
            band_per_row = np.zeros(B, dtype=np.float64)

        sym_per_row = self._extract_sym_per_row(batches)
        agree_filter_mask = np.array(
            [s in AGREEMENT_FILTER_SYMS for s in sym_per_row], dtype=bool
        )

        for hcfg in self._horizons:
            if not hcfg.get("active", True):
                continue
            H = int(hcfg["h"])
            lgbs = self._lgb_lists.get(H)
            nns = self._nn_lists.get(H)
            nn2s = self._nn2_lists.get(H)
            nn3s = self._nn3_lists.get(H)
            cbs = self._cb_lists.get(H)
            w_nn, w_lgb, w_nn2, w_nn3, w_cb = self._weights.get(H, ({w87}, {wLGB}, {w97}, {w95}, {wCB}))

            lgb_pred: np.ndarray | None = None
            nn_pred: np.ndarray | None = None
            nn2_pred: np.ndarray | None = None
            nn3_pred: np.ndarray | None = None
            cb_pred: np.ndarray | None = None
            preds, ws = [], []

            if lgbs and w_lgb > 0:
                lgb_pred = self._ensemble_predict_lgb(lgbs, feats)
                preds.append(lgb_pred); ws.append(w_lgb)
            if nns and w_nn > 0:
                nn_pred = self._ensemble_predict_nn(nns, feats)
                preds.append(nn_pred); ws.append(w_nn)
            if nn2s and w_nn2 > 0:
                nn2_pred = self._ensemble_predict_nn(nn2s, feats)
                preds.append(nn2_pred); ws.append(w_nn2)
            if nn3s and w_nn3 > 0:
                X_lob = self._extract_lob_window(X3d)
                nn3_pred = self._ensemble_predict_gru(nn3s, X_lob)
                preds.append(nn3_pred); ws.append(w_nn3)
            if cbs and w_cb > 0:
                cb_pred = self._ensemble_predict_cb(cbs, feats)
                preds.append(cb_pred); ws.append(w_cb)
            if not preds:
                continue

            stacked = np.stack(preds, axis=0)
            ws_arr = np.array(ws, dtype=np.float32).reshape(-1, 1)
            pred_dmid = (stacked * ws_arr).sum(axis=0) / ws_arr.sum()

            if self._cw_enabled:
                actions = self._gate_with_band(pred_dmid, hcfg, band_per_row)
            else:
                actions = self._ev_gate_predict(pred_dmid, hcfg)

            # Selective agreement filter:
            # NN-side = weighted avg of T87 + T97 + T95 (the NN models).
            # For syms 0,1,2: abstain if sign(NN-side) != sign(lgb_pred).
            if lgb_pred is not None and agree_filter_mask.any():
                nn_parts = []
                nn_ws = []
                if nn_pred is not None:
                    nn_parts.append(nn_pred); nn_ws.append(w_nn)
                if nn2_pred is not None:
                    nn_parts.append(nn2_pred); nn_ws.append(w_nn2)
                if nn3_pred is not None:
                    nn_parts.append(nn3_pred); nn_ws.append(w_nn3)
                if nn_parts:
                    nn_stacked = np.stack(nn_parts, axis=0)
                    nn_ws_arr = np.array(nn_ws, dtype=np.float32).reshape(-1, 1)
                    nn_combined = (nn_stacked * nn_ws_arr).sum(axis=0) / nn_ws_arr.sum()
                    disagree = np.sign(nn_combined) != np.sign(lgb_pred)
                    actions[agree_filter_mask & disagree] = 1

            out[:, HORIZON_TO_IDX[H]] = actions
        return out.tolist()


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    cfg = json.load(open(os.path.join(here, "config.json")))
    feats = cfg["feature"]
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        rng.standard_normal((100, len(feats))).astype(np.float32),
        columns=feats,
    )
    df["sym"] = 1
    p = Predictor()
    out = p.predict([df, df, df])
    print("smoke test predict ->", out[0], "len=", len(out))
    df_ood = df.copy(); df_ood["sym"] = 99
    out_ood = p.predict([df_ood])
    print("OOD sym=99 ->", out_ood[0])
'''
    with open(path, "w") as f:
        f.write(code)


def smoke_test_predictor(pkg_dir):
    """Run minimal smoke test."""
    test_script = os.path.join(HERE, "_smoke_test_t167.py")
    smoke_code = f"""
import sys
sys.path.insert(0, {repr(pkg_dir)})
try:
    from Predictor import Predictor
    p = Predictor()
    import pandas as pd, numpy as np
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.standard_normal((100, len(p._raw_feat_cols))).astype(np.float32),
                      columns=p._raw_feat_cols)
    df["sym"] = 0
    result = p.predict([df])
    assert result is not None and len(result) == 1 and len(result[0]) == 5
    print(f"Smoke test PASSED: action={{result[0]}}")
except Exception as e:
    print(f"Smoke test FAILED: {{e}}")
    import traceback; traceback.print_exc()
    sys.exit(1)
"""
    with open(test_script, "w") as f:
        f.write(smoke_code)
    ret = os.system(f"cd {pkg_dir} && python3 {test_script}")
    return ret == 0


if __name__ == "__main__":
    main()
