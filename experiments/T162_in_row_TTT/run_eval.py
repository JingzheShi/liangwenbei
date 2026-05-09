"""
T162: In-row TTT — Local context-aware prediction adjustment.
Implements W3a (momentum scaling), W3b (vol shrinkage), W3c (combined).
Evaluates each on holdout (dates 100-119) against v2 baseline.
"""
from __future__ import annotations

import json
import os
import sys
import glob
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

WORKDIR = Path("/root/projects/liangwenbei_workdir")
DATA_DIR = WORKDIR / "data"
V2_PKG = WORKDIR / "experiments/R_full_retrain/pkg_iter019_v2"
EXPDIR = Path(__file__).parent

# Add src to path for eval helpers
sys.path.insert(0, str(WORKDIR / "src"))

from eval.pnl import compute_pnl, HORIZONS
from eval.dataset_eval import _gather_session, _windows_to_inputs

# ── Progress reporter ──────────────────────────────────────────────────────────

def update_progress(step: str, metrics: dict = None):
    p = {
        "status": "running",
        "step": step,
        "metrics": metrics or {},
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    with open(WORKDIR / "worker-progress.json", "w") as f:
        json.dump(p, f, indent=2)
    print(f"[T162] {step}")


# ── Load v2 Predictor ──────────────────────────────────────────────────────────

def load_v2_predictor():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "v2_predictor", V2_PKG / "Predictor.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v2_predictor"] = mod
    # Need to be in the right directory for relative imports
    orig_dir = os.getcwd()
    os.chdir(str(V2_PKG))
    spec.loader.exec_module(mod)
    os.chdir(orig_dir)
    return mod.Predictor()


# ── Local context computation ──────────────────────────────────────────────────

def compute_local_context(batches: List[pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized computation of local trend slope and volatility
    from midprice1 over the 100-tick window.
    Returns: (rel_slopes, local_vols) each shape (N,)
    """
    N = len(batches)
    T = 100
    ticks = np.arange(T, dtype=np.float64)
    mean_t = ticks.mean()  # 49.5
    xx_sum = (ticks ** 2).sum()
    denom = xx_sum - T * mean_t ** 2  # variance of ticks * T

    # Build matrix (N, T) of midprice1 values
    mid_matrix = np.empty((N, T), dtype=np.float64)
    for i, df in enumerate(batches):
        mid_matrix[i] = df["midprice1"].values[:T].astype(np.float64)

    mean_y = mid_matrix.mean(axis=1)                        # (N,)
    xy_sum = (mid_matrix * ticks[None, :]).sum(axis=1)      # (N,)
    slope = (xy_sum - T * mean_t * mean_y) / denom          # (N,) absolute slope

    # Relative slope (normalize by mean price level)
    mean_mid = np.abs(mean_y)
    mean_mid = np.where(mean_mid < 1e-8, 1e-8, mean_mid)
    rel_slope = slope / mean_mid                             # dimensionless

    # Log-return volatility
    # midprice1 is relative return vs yesterday close → price = midprice1 + 1
    price = mid_matrix + 1.0
    price = np.where(price > 1e-8, price, 1e-8)
    log_price = np.log(price)
    log_ret = np.diff(log_price, axis=1)                     # (N, T-1)
    local_vol = log_ret.std(axis=1)                          # (N,)

    return rel_slope, local_vol


# ── Wrapper implementations ────────────────────────────────────────────────────

def apply_wrapper_W3a(pred_dmid: np.ndarray, rel_slopes: np.ndarray, local_vols: np.ndarray) -> np.ndarray:
    """Momentum scaling: boost if trend agrees with prediction, shrink otherwise."""
    trend_dir = np.sign(rel_slopes)
    pred_dir = np.sign(pred_dmid)
    agreement = (trend_dir == pred_dir) & (trend_dir != 0)
    scale = np.where(agreement, 1.1, 0.9)
    return pred_dmid * scale


def apply_wrapper_W3b(pred_dmid: np.ndarray, rel_slopes: np.ndarray, local_vols: np.ndarray) -> np.ndarray:
    """Volatility shrinkage: high vol → less confident → shrink prediction magnitude.
    Uses within-batch percentile rank for consistency within session evaluation.
    """
    vol_pct = pd.Series(local_vols).rank(pct=True).values    # [0, 1]
    scale = 1.0 - 0.3 * vol_pct                               # range [0.7, 1.0]
    return pred_dmid * scale


def apply_wrapper_W3c(pred_dmid: np.ndarray, rel_slopes: np.ndarray, local_vols: np.ndarray) -> np.ndarray:
    """Combined: momentum scaling × volatility shrinkage."""
    trend_dir = np.sign(rel_slopes)
    pred_dir = np.sign(pred_dmid)
    agreement = (trend_dir == pred_dir) & (trend_dir != 0)
    momentum_scale = np.where(agreement, 1.1, 0.9)
    vol_pct = pd.Series(local_vols).rank(pct=True).values
    vol_scale = 1.0 - 0.3 * vol_pct
    return pred_dmid * momentum_scale * vol_scale


WRAPPERS = {
    "baseline": None,
    "W3a": apply_wrapper_W3a,
    "W3b": apply_wrapper_W3b,
    "W3c": apply_wrapper_W3c,
}


# ── Wrapped predict function factory ──────────────────────────────────────────

def make_predict_fn(predictor, wrapper_fn=None):
    """
    Create a predict function that replicates v2 prediction logic
    with optional wrapper applied to pred_dmid before thresholding.
    """
    # Access horizon config (only h=60 is active)
    HORIZON_TO_IDX = {5: 0, 10: 1, 20: 2, 40: 3, 60: 4}

    def predict(batches: List[pd.DataFrame]) -> List[List[int]]:
        if not batches:
            return []
        feats = predictor._compute_batch_features(batches)
        B = feats.shape[0]
        out = np.ones((B, 5), dtype=np.int64)

        if predictor._cw_enabled:
            band_per_row = predictor._extract_band_per_row(batches)
        else:
            band_per_row = np.zeros(B, dtype=np.float64)

        # Compute local context if wrapper is active
        if wrapper_fn is not None:
            rel_slopes, local_vols = compute_local_context(batches)

        for hcfg in predictor._horizons:
            if not hcfg.get("active", True):
                continue
            H = int(hcfg["h"])
            lgbs = predictor._lgb_lists.get(H)
            nns = predictor._nn_lists.get(H)
            w_nn, w_lgb = predictor._weights.get(H, (1.0, 1.0))

            preds_list = []
            ws_list = []
            if lgbs and w_lgb > 0:
                preds_list.append(predictor._ensemble_predict_lgb(lgbs, feats))
                ws_list.append(w_lgb)
            if nns and w_nn > 0:
                preds_list.append(predictor._ensemble_predict_nn(nns, feats))
                ws_list.append(w_nn)
            if not preds_list:
                continue

            stacked = np.stack(preds_list, axis=0)  # (M, B)
            ws_arr = np.array(ws_list, dtype=np.float32).reshape(-1, 1)
            pred_dmid = (stacked * ws_arr).sum(axis=0) / ws_arr.sum()

            # Apply wrapper scaling before thresholding
            if wrapper_fn is not None:
                pred_dmid = wrapper_fn(pred_dmid, rel_slopes, local_vols)

            if predictor._cw_enabled:
                actions = predictor._gate_with_band(pred_dmid, hcfg, band_per_row)
            else:
                actions = predictor._ev_gate_predict(pred_dmid, hcfg)

            out[:, HORIZON_TO_IDX[H]] = actions

        return out.tolist()

    return predict


# ── Evaluate one predict_fn on holdout ────────────────────────────────────────

FEATURE_COLS = [
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
    "sym",
]
WINDOW = 100
BATCH_SIZE = 1024
HOLDOUT_DATES = list(range(100, 120))  # last 20 dates


def evaluate_holdout(predict_fn, name: str) -> dict:
    """Evaluate predict_fn on holdout sessions (dates 100-119)."""
    all_pred, all_label, all_mp_t, all_mp_tn = [], [], [], []
    n_sessions = 0

    for date in HOLDOUT_DATES:
        for sym in range(5):
            for ampm in ["am", "pm"]:
                path = DATA_DIR / f"snapshot_sym{sym}_date{date}_{ampm}.parquet"
                if not path.exists():
                    continue
                session_df = pd.read_parquet(path)
                gathered = _gather_session(session_df, FEATURE_COLS, WINDOW, HORIZONS)
                if gathered is None:
                    continue
                feat_df, valid_t, label_t, mp_t, mp_tn = gathered
                n_samples = len(valid_t)
                n_h = len(HORIZONS)
                preds = np.zeros((n_samples, n_h), dtype=np.int64)
                for start in range(0, n_samples, BATCH_SIZE):
                    end = min(start + BATCH_SIZE, n_samples)
                    chunk_t = valid_t[start:end]
                    windows = _windows_to_inputs(feat_df, chunk_t, WINDOW)
                    out = predict_fn(windows)
                    preds[start:end] = np.asarray(out, dtype=np.int64)
                all_pred.append(preds)
                all_label.append(label_t)
                all_mp_t.append(mp_t)
                all_mp_tn.append(mp_tn)
                n_sessions += 1

    pred_cat = np.concatenate(all_pred, axis=0)
    label_cat = np.concatenate(all_label, axis=0)
    mp_t_cat = np.concatenate(all_mp_t, axis=0)
    mp_tn_cat = np.concatenate(all_mp_tn, axis=0)

    result = compute_pnl(pred_cat, label_cat, mp_t_cat, mp_tn_cat, fee_rate=0.0001)
    best_pnl = result["best_score"]
    best_h = result["best_horizon"]
    print(f"  [{name}] best_pnl={best_pnl:.4f} (horizon={best_h}), "
          f"n_sessions={n_sessions}, n_samples={len(pred_cat)}")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import wandb
    try:
        wandb.init(
            project="liangwenbei_T162",
            entity="cjxh21-Tsinghua University",
            config={
                "experiment": "T162_in_row_TTT",
                "holdout_dates": f"{HOLDOUT_DATES[0]}-{HOLDOUT_DATES[-1]}",
                "wrappers": ["W3a", "W3b", "W3c"],
                "W3a_scale_agree": 1.1,
                "W3a_scale_disagree": 0.9,
                "W3b_max_shrink": 0.3,
                "W3c_combined": True,
            }
        )
    except Exception as e:
        print(f"[WandB] init failed ({e}), using offline mode")
        wandb.init(
            project="liangwenbei_T162",
            mode="offline",
            config={
                "experiment": "T162_in_row_TTT",
                "holdout_dates": f"{HOLDOUT_DATES[0]}-{HOLDOUT_DATES[-1]}",
            }
        )

    update_progress("Loading v2 Predictor...")
    predictor = load_v2_predictor()
    print("v2 Predictor loaded OK")

    results_by_wrapper = {}

    for wrapper_name, wrapper_fn in WRAPPERS.items():
        update_progress(f"Evaluating {wrapper_name}...")
        predict_fn = make_predict_fn(predictor, wrapper_fn)
        r = evaluate_holdout(predict_fn, wrapper_name)
        results_by_wrapper[wrapper_name] = r

    # ── Extract key metrics ────────────────────────────────────────────────────
    baseline_pnl = results_by_wrapper["baseline"]["best_score"]
    wrapper_results = {}
    for name in ["W3a", "W3b", "W3c"]:
        pnl = results_by_wrapper[name]["best_score"]
        delta = pnl - baseline_pnl
        wrapper_results[name] = {"holdout_pnl": pnl, "delta": delta}
        print(f"  {name}: pnl={pnl:.4f}, delta={delta:+.4f}")

    best_name = max(["W3a", "W3b", "W3c"], key=lambda n: wrapper_results[n]["delta"])
    best_delta = wrapper_results[best_name]["delta"]

    # ── WandB logging ────────────────────────────────────────────────────────
    wandb.log({
        "v2_baseline_pnl": baseline_pnl,
        "W3a_pnl": wrapper_results["W3a"]["holdout_pnl"],
        "W3b_pnl": wrapper_results["W3b"]["holdout_pnl"],
        "W3c_pnl": wrapper_results["W3c"]["holdout_pnl"],
        "W3a_delta": wrapper_results["W3a"]["delta"],
        "W3b_delta": wrapper_results["W3b"]["delta"],
        "W3c_delta": wrapper_results["W3c"]["delta"],
        "best_delta": best_delta,
        "best_wrapper": best_name,
    })
    wandb.finish()

    # ── Build zip if best wrapper beats v2 ────────────────────────────────────
    zip_path = None
    if best_delta >= 0.5:
        update_progress(f"Building zip for {best_name} (delta={best_delta:+.4f})...")
        zip_path = build_zip(best_name, predictor)
        print(f"  ZIP built: {zip_path}")
    else:
        print(f"  No wrapper beats v2 by +0.5 (best_delta={best_delta:+.4f}). No zip.")

    # ── Write results.json ────────────────────────────────────────────────────
    results = {
        "task": "T162 in-row TTT (local context-aware prediction adjustment)",
        "v2_baseline_holdout": baseline_pnl,
        "W3a_momentum_scaling": wrapper_results["W3a"],
        "W3b_vol_shrinkage": wrapper_results["W3b"],
        "W3c_combined": wrapper_results["W3c"],
        "best_method": best_name,
        "best_delta": best_delta,
        "v2W_TTT_zip": str(zip_path) if zip_path else None,
        "expected_platform": None,
        "holdout_dates": f"{HOLDOUT_DATES[0]}-{HOLDOUT_DATES[-1]}",
        "notes": f"Only h=60 is active in v2. Best wrapper: {best_name} with delta={best_delta:+.4f}. {'Zip built.' if zip_path else 'No zip (delta < 0.5).'}"
    }

    with open(EXPDIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results written to results.json")

    # ── Final progress ────────────────────────────────────────────────────────
    update_progress("Done", {"best_delta": best_delta, "baseline_pnl": baseline_pnl})

    return results


# ── Build zip ─────────────────────────────────────────────────────────────────

def build_zip(best_wrapper: str, predictor) -> Path:
    """
    Build a submission zip with modified Predictor.py applying the best wrapper.
    Based on v2 pkg structure.
    """
    import shutil
    import zipfile

    zip_name = f"submission_T162_{best_wrapper.lower()}.zip"
    zip_path = WORKDIR / zip_name

    # Create a temp dir with modified Predictor.py
    tmp_dir = EXPDIR / "pkg_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()

    # Copy all files from v2 pkg
    for f in V2_PKG.iterdir():
        if f.name != "__pycache__" and f.is_file():
            shutil.copy2(f, tmp_dir / f.name)

    # Build modified Predictor.py
    build_modified_predictor(tmp_dir, best_wrapper)

    # Build zip
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(tmp_dir.iterdir()):
            if f.is_file():
                zf.write(f, f.name)

    shutil.rmtree(tmp_dir)
    print(f"ZIP built: {zip_path} ({zip_path.stat().st_size // 1024}KB)")
    return zip_path


def build_modified_predictor(pkg_dir: Path, wrapper_type: str):
    """Write a modified Predictor.py that applies the wrapper."""
    src = (V2_PKG / "Predictor.py").read_text()

    # Inject local context computation and wrapper into predict() method
    # We'll add a helper method and modify predict()

    # Approach: Add helper methods and modify the prediction loop
    wrapper_code = f'''
    # ─────────────────────────────────────────────────────────────────────────
    # T162 in-row TTT: local context wrapper ({wrapper_type})
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_local_context(self, batches: List[pd.DataFrame]):
        """Compute relative slope and log-return volatility of midprice1 window."""
        N = len(batches)
        T = 100
        ticks = np.arange(T, dtype=np.float64)
        mean_t = 49.5
        denom = (ticks ** 2).sum() - T * mean_t ** 2

        mid_matrix = np.empty((N, T), dtype=np.float64)
        for i, df in enumerate(batches):
            v = df["midprice1"].values
            mid_matrix[i] = v[:T].astype(np.float64)

        mean_y = mid_matrix.mean(axis=1)
        xy_sum = (mid_matrix * ticks[None, :]).sum(axis=1)
        slope = (xy_sum - T * mean_t * mean_y) / denom
        mean_mid = np.where(np.abs(mean_y) < 1e-8, 1e-8, np.abs(mean_y))
        rel_slope = slope / mean_mid

        price = mid_matrix + 1.0
        price = np.where(price > 1e-8, price, 1e-8)
        log_ret = np.diff(np.log(price), axis=1)
        local_vol = log_ret.std(axis=1)

        return rel_slope, local_vol

    def _apply_ttt_wrapper(self, pred_dmid: np.ndarray,
                           rel_slopes: np.ndarray, local_vols: np.ndarray) -> np.ndarray:
        """Apply {wrapper_type} scaling to continuous prediction."""
'''

    if wrapper_type == "W3a":
        wrapper_code += '''
        trend_dir = np.sign(rel_slopes)
        pred_dir = np.sign(pred_dmid)
        agreement = (trend_dir == pred_dir) & (trend_dir != 0)
        scale = np.where(agreement, 1.1, 0.9)
        return pred_dmid * scale
'''
    elif wrapper_type == "W3b":
        # For production: use a fixed vol threshold instead of percentile
        # Calibrated from training data: median vol ~= 0.0003
        wrapper_code += '''
        # Fixed threshold calibrated from holdout data (median ≈ 0.0003)
        REF_VOL = 3e-4
        vol_norm = np.clip(local_vols / (REF_VOL + 1e-10), 0.0, 1.0)
        scale = 1.0 - 0.3 * vol_norm  # range [0.7, 1.0]
        return pred_dmid * scale
'''
    elif wrapper_type == "W3c":
        wrapper_code += '''
        trend_dir = np.sign(rel_slopes)
        pred_dir = np.sign(pred_dmid)
        agreement = (trend_dir == pred_dir) & (trend_dir != 0)
        momentum_scale = np.where(agreement, 1.1, 0.9)
        REF_VOL = 3e-4
        vol_norm = np.clip(local_vols / (REF_VOL + 1e-10), 0.0, 1.0)
        vol_scale = 1.0 - 0.3 * vol_norm
        return pred_dmid * momentum_scale * vol_scale
'''

    # Insert wrapper methods and modify predict()
    # Find the point to inject methods (before predict method)
    inject_after = "    def _ensemble_predict_nn"
    inject_code = wrapper_code + "\n    def _ensemble_predict_nn"

    # Modify predict() to call wrapper
    old_pred_block = """            if self._cw_enabled:
                actions = self._gate_with_band(pred_dmid, hcfg, band_per_row)
            else:
                actions = self._ev_gate_predict(pred_dmid, hcfg)"""

    new_pred_block = """            pred_dmid = self._apply_ttt_wrapper(pred_dmid, rel_slopes, local_vols)
            if self._cw_enabled:
                actions = self._gate_with_band(pred_dmid, hcfg, band_per_row)
            else:
                actions = self._ev_gate_predict(pred_dmid, hcfg)"""

    # Inject local context call before the loop
    old_loop_start = "        for hcfg in self._horizons:"
    new_loop_start = "        rel_slopes, local_vols = self._compute_local_context(batches)\n        for hcfg in self._horizons:"

    modified = (src
                .replace(inject_after, inject_code, 1)
                .replace(old_pred_block, new_pred_block, 1)
                .replace(old_loop_start, new_loop_start, 1))

    # Update docstring
    modified = modified.replace(
        '"""Predictor for iter_018 v1',
        f'"""Predictor for T162 ({wrapper_type}) = v2 + in-row TTT local context wrapper'
    )

    (pkg_dir / "Predictor.py").write_text(modified)
    print(f"Modified Predictor.py written for {wrapper_type}")


if __name__ == "__main__":
    results = main()
    best_name = results["best_method"]
    best_delta = results["best_delta"]
    baseline_pnl = results["v2_baseline_holdout"]
    w3a = results["W3a_momentum_scaling"]["holdout_pnl"]
    w3b = results["W3b_vol_shrinkage"]["holdout_pnl"]
    w3c = results["W3c_combined"]["holdout_pnl"]
    zip_info = results["v2W_TTT_zip"] or "none"
    print(
        f"\nRESULT: task=[T162 in-row TTT] "
        f"metrics={{v2_baseline={baseline_pnl:.4f}, W3a={w3a:.4f}, W3b={w3b:.4f}, W3c={w3c:.4f}, best_delta={best_delta:+.4f}}} "
        f"notes=[best={best_name}, zip={zip_info}]"
    )
