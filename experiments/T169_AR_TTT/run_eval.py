"""
T169: In-row AR(5) TTT — fit tiny AR model on 100-tick window per-row,
use its prediction quality as a SCALING factor for v2 ensemble pred.

Pure wrapper around v2; no retraining of LGB or NN models.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

WORKDIR = Path("/root/projects/liangwenbei_workdir")
DATA_DIR = WORKDIR / "data"
V2_PKG = WORKDIR / "experiments/T169_AR_TTT/T169_AR_TTT_pkg_base"
EXPDIR = Path(__file__).parent
OUTPUT_DIR = Path("/tmp/metabot-outputs/oc_8da9d6d6cd3ed6e40a8dd5b38d55d6a7")

sys.path.insert(0, str(WORKDIR / "src"))
from eval.pnl import compute_pnl, HORIZONS
from eval.dataset_eval import _gather_session, _windows_to_inputs

HOLDOUT_DATES = list(range(96, 120))   # dates 96-119
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
HORIZON_TO_IDX = {5: 0, 10: 1, 20: 2, 40: 3, 60: 4}


def update_progress(step: str, metrics: dict = None):
    p = {
        "status": "running",
        "step": step,
        "metrics": metrics or {},
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    with open(WORKDIR / "worker-progress.json", "w") as f:
        json.dump(p, f, indent=2)
    print(f"[T169] {step}")


def load_v2_predictor():
    import importlib.util
    spec = importlib.util.spec_from_file_location("v2_pred_t169", V2_PKG / "Predictor.py")
    mod = importlib.util.module_from_spec(spec)
    orig = os.getcwd()
    os.chdir(str(V2_PKG))
    spec.loader.exec_module(mod)
    os.chdir(orig)
    return mod.Predictor()


# ── AR(5) Confidence computation ──────────────────────────────────────────────

def compute_ar5_confidence(batches: List[pd.DataFrame]) -> np.ndarray:
    """
    Vectorized per-row AR(5) confidence (local R²) from the 100-tick mid-price window.

    For each row:
      1. Extract midprice1 series (100 ticks)
      2. Compute log returns r[0..98] = diff(log(price))
      3. Fit AR(5) via OLS on training ticks t=5..94 (90 samples)
      4. Predict ticks t=94..98 (5 test samples) using lagged returns
      5. Compute R² on those 5 test samples

    Returns: local_R2, shape (N,), clipped to [-5, 1].

    Self-contained per row; no cross-row or cross-call state.
    """
    N = len(batches)
    T = 100

    mid_matrix = np.empty((N, T), dtype=np.float64)
    for i, df in enumerate(batches):
        v = df["midprice1"].values
        mid_matrix[i] = v[:T].astype(np.float64)

    # log returns: midprice1 is return vs yesterday close, so price ≈ 1 + midprice1
    price = mid_matrix + 1.0
    price = np.where(price > 1e-8, price, 1e-8)
    log_r = np.diff(np.log(price), axis=1)  # (N, 99), 0-indexed [0..98]

    # Training: targets log_r[:, 5:95] (indices 5..94, 90 samples)
    # Features: lags 1..5 of each target
    y_train = log_r[:, 5:95]  # (N, 90)
    X_train = np.stack([
        log_r[:, 4:94],  # lag 1
        log_r[:, 3:93],  # lag 2
        log_r[:, 2:92],  # lag 3
        log_r[:, 1:91],  # lag 4
        log_r[:, 0:90],  # lag 5
    ], axis=2)  # (N, 90, 5)

    # Vectorized OLS: coef = (X^T X + eps I)^{-1} X^T y
    eps = 1e-12
    XtX = X_train.transpose(0, 2, 1) @ X_train  # (N, 5, 5)
    XtX += eps * np.eye(5)[None, :, :]
    Xty = (X_train.transpose(0, 2, 1) @ y_train[:, :, None]).squeeze(-1)  # (N, 5)
    coef = np.linalg.solve(XtX, Xty[:, :, None]).squeeze(-1)  # (N, 5)

    # Test: targets log_r[:, 94:99] (5 samples, indices 94..98)
    y_test = log_r[:, 94:99]  # (N, 5)
    X_test = np.stack([
        log_r[:, 93:98],  # lag 1
        log_r[:, 92:97],  # lag 2
        log_r[:, 91:96],  # lag 3
        log_r[:, 90:95],  # lag 4
        log_r[:, 89:94],  # lag 5
    ], axis=2)  # (N, 5, 5)

    y_pred = (X_test @ coef[:, :, None]).squeeze(-1)  # (N, 5)

    ss_res = ((y_test - y_pred) ** 2).sum(axis=1)  # (N,)
    y_test_mean = y_test.mean(axis=1, keepdims=True)
    ss_tot = ((y_test - y_test_mean) ** 2).sum(axis=1)  # (N,)
    ss_tot = np.where(ss_tot < 1e-20, 1e-20, ss_tot)

    local_R2 = 1.0 - ss_res / ss_tot
    local_R2 = np.clip(local_R2, -5.0, 1.0)
    return local_R2


# ── Variant scaling functions ─────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def scale_V1_binary(pred_dmid: np.ndarray, local_R2: np.ndarray) -> np.ndarray:
    """Binary: use pred unchanged if R2 > median, else scale by 0.5."""
    median_R2 = np.median(local_R2)
    factor = np.where(local_R2 > median_R2, 1.0, 0.5)
    return pred_dmid * factor


def scale_V2_sigmoid(pred_dmid: np.ndarray, local_R2: np.ndarray) -> np.ndarray:
    """Continuous sigmoid: factor = sigmoid(R2 * 5 - 1)."""
    factor = _sigmoid(local_R2 * 5.0 - 1.0)
    return pred_dmid * factor


def scale_V3_abstain(pred_dmid: np.ndarray, local_R2: np.ndarray) -> np.ndarray:
    """Abstain rows where R2 < 0.05 by zeroing pred_dmid (→ action=1)."""
    pred_out = pred_dmid.copy()
    pred_out[local_R2 < 0.05] = 0.0
    return pred_out


def scale_V4_magnitude(pred_dmid: np.ndarray, local_R2: np.ndarray) -> np.ndarray:
    """Magnitude-aware: trust trade if |pred| > 2*sigma AND R2 > median."""
    median_R2 = np.median(local_R2)
    sigma = np.std(pred_dmid)
    if sigma < 1e-12:
        sigma = 1e-12
    strong_signal = np.abs(pred_dmid) > 2.0 * sigma
    high_conf = local_R2 > median_R2
    # Full trust if both; shrink to 0.5 otherwise
    factor = np.where(strong_signal & high_conf, 1.0, 0.5)
    return pred_dmid * factor


VARIANTS = {
    "V1_binary":     scale_V1_binary,
    "V2_sigmoid":    scale_V2_sigmoid,
    "V3_abstain":    scale_V3_abstain,
    "V4_magnitude":  scale_V4_magnitude,
}


# ── Wrapped predict factory ───────────────────────────────────────────────────

def make_predict_fn(predictor, scale_fn=None):
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

        if scale_fn is not None:
            local_R2 = compute_ar5_confidence(batches)

        for hcfg in predictor._horizons:
            if not hcfg.get("active", True):
                continue
            H = int(hcfg["h"])
            lgbs = predictor._lgb_lists.get(H)
            nns = predictor._nn_lists.get(H)
            w_nn, w_lgb = predictor._weights.get(H, (1.0, 1.0))

            preds_list, ws_list = [], []
            if lgbs and w_lgb > 0:
                preds_list.append(predictor._ensemble_predict_lgb(lgbs, feats))
                ws_list.append(w_lgb)
            if nns and w_nn > 0:
                preds_list.append(predictor._ensemble_predict_nn(nns, feats))
                ws_list.append(w_nn)
            if not preds_list:
                continue

            stacked = np.stack(preds_list, axis=0)
            ws_arr = np.array(ws_list, dtype=np.float32).reshape(-1, 1)
            pred_dmid = (stacked * ws_arr).sum(axis=0) / ws_arr.sum()

            if scale_fn is not None:
                pred_dmid = scale_fn(pred_dmid, local_R2)

            if predictor._cw_enabled:
                actions = predictor._gate_with_band(pred_dmid, hcfg, band_per_row)
            else:
                actions = predictor._ev_gate_predict(pred_dmid, hcfg)

            out[:, HORIZON_TO_IDX[H]] = actions

        return out.tolist()
    return predict


# ── Holdout evaluation ────────────────────────────────────────────────────────

def evaluate_holdout(predict_fn, name: str) -> dict:
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
                preds = np.zeros((n_samples, len(HORIZONS)), dtype=np.int64)
                for start in range(0, n_samples, BATCH_SIZE):
                    end = min(start + BATCH_SIZE, n_samples)
                    windows = _windows_to_inputs(feat_df, valid_t[start:end], WINDOW)
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
    print(f"  [{name}] best_pnl={best_pnl:.4f} (horizon={best_h}), n_sessions={n_sessions}, n={len(pred_cat)}")
    return result


# ── Package builder ───────────────────────────────────────────────────────────

AR5_METHODS_CODE = '''
    # ── T169: AR(5) in-row TTT ────────────────────────────────────────────────

    @staticmethod
    def _compute_ar5_confidence(batches):
        """
        Vectorized per-row AR(5) confidence (local R²) from 100-tick mid-price window.
        Self-contained per row; no cross-row or cross-call state.
        """
        import numpy as _np
        N = len(batches)
        T = 100

        mid_matrix = _np.empty((N, T), dtype=_np.float64)
        for i, df in enumerate(batches):
            v = df["midprice1"].values
            mid_matrix[i] = v[:T].astype(_np.float64)

        price = mid_matrix + 1.0
        price = _np.where(price > 1e-8, price, 1e-8)
        log_r = _np.diff(_np.log(price), axis=1)  # (N, 99)

        y_train = log_r[:, 5:95]  # (N, 90)
        X_train = _np.stack([
            log_r[:, 4:94], log_r[:, 3:93], log_r[:, 2:92],
            log_r[:, 1:91], log_r[:, 0:90],
        ], axis=2)  # (N, 90, 5)

        eps = 1e-12
        XtX = X_train.transpose(0, 2, 1) @ X_train  # (N, 5, 5)
        XtX += eps * _np.eye(5)[None, :, :]
        Xty = (X_train.transpose(0, 2, 1) @ y_train[:, :, None]).squeeze(-1)  # (N, 5)
        coef = _np.linalg.solve(XtX, Xty[:, :, None]).squeeze(-1)  # (N, 5)

        y_test = log_r[:, 94:99]  # (N, 5)
        X_test = _np.stack([
            log_r[:, 93:98], log_r[:, 92:97], log_r[:, 91:96],
            log_r[:, 90:95], log_r[:, 89:94],
        ], axis=2)  # (N, 5, 5)

        y_pred = (X_test @ coef[:, :, None]).squeeze(-1)  # (N, 5)
        ss_res = ((y_test - y_pred) ** 2).sum(axis=1)
        y_test_mean = y_test.mean(axis=1, keepdims=True)
        ss_tot = ((y_test - y_test_mean) ** 2).sum(axis=1)
        ss_tot = _np.where(ss_tot < 1e-20, 1e-20, ss_tot)

        local_R2 = 1.0 - ss_res / ss_tot
        return _np.clip(local_R2, -5.0, 1.0)

'''

SCALE_BODY = {
    "V1_binary": '''\
    @staticmethod
    def _ar5_scale(pred_dmid, local_R2):
        """V1: binary — full pred if R2 > median, else scale by 0.5."""
        import numpy as _np
        factor = _np.where(local_R2 > _np.median(local_R2), 1.0, 0.5)
        return pred_dmid * factor
''',
    "V2_sigmoid": '''\
    @staticmethod
    def _ar5_scale(pred_dmid, local_R2):
        """V2: continuous sigmoid — factor = sigmoid(R2 * 5 - 1)."""
        import numpy as _np
        factor = 1.0 / (1.0 + _np.exp(-(local_R2 * 5.0 - 1.0)))
        return pred_dmid * factor
''',
    "V3_abstain": '''\
    @staticmethod
    def _ar5_scale(pred_dmid, local_R2):
        """V3: abstain (zero pred) when R2 < 0.05."""
        import numpy as _np
        out = pred_dmid.copy()
        out[local_R2 < 0.05] = 0.0
        return out
''',
    "V4_magnitude": '''\
    @staticmethod
    def _ar5_scale(pred_dmid, local_R2):
        """V4: trust if |pred| > 2*sigma AND R2 > median, else shrink to 0.5."""
        import numpy as _np
        sigma = _np.std(pred_dmid)
        if sigma < 1e-12:
            sigma = 1e-12
        strong = _np.abs(pred_dmid) > 2.0 * sigma
        high_conf = local_R2 > _np.median(local_R2)
        factor = _np.where(strong & high_conf, 1.0, 0.5)
        return pred_dmid * factor
''',
}


def build_variant_predictor(base_src: str, variant: str) -> str:
    """Inject AR(5) methods and scaling into Predictor.py."""
    inject_point = "    @staticmethod\n    def _ensemble_predict_lgb"
    injection = AR5_METHODS_CODE + SCALE_BODY[variant] + "\n    @staticmethod\n    def _ensemble_predict_lgb"
    src = base_src.replace(inject_point, injection, 1)

    # Inject AR(5) call + scaling in predict() before conformal gate
    old_gate = (
        "            if self._cw_enabled:\n"
        "                actions = self._gate_with_band(pred_dmid, hcfg, band_per_row)\n"
        "            else:\n"
        "                actions = self._ev_gate_predict(pred_dmid, hcfg)"
    )
    new_gate = (
        "            pred_dmid = self._ar5_scale(pred_dmid, _ar5_local_R2)\n"
        "            if self._cw_enabled:\n"
        "                actions = self._gate_with_band(pred_dmid, hcfg, band_per_row)\n"
        "            else:\n"
        "                actions = self._ev_gate_predict(pred_dmid, hcfg)"
    )
    src = src.replace(old_gate, new_gate, 1)

    # Inject AR(5) computation before the horizons loop in predict()
    # Use context unique to predict() (the else branch setting band_per_row=zeros)
    old_loop = (
        "        else:\n"
        "            band_per_row = np.zeros(B, dtype=np.float64)\n"
        "\n"
        "        for hcfg in self._horizons:"
    )
    new_loop = (
        "        else:\n"
        "            band_per_row = np.zeros(B, dtype=np.float64)\n"
        "\n"
        "        _ar5_local_R2 = self._compute_ar5_confidence(batches)\n"
        "        for hcfg in self._horizons:"
    )
    src = src.replace(old_loop, new_loop, 1)

    # Update docstring
    src = src.replace(
        '"""Predictor for iter_018 v1',
        f'"""Predictor T169 {variant} = v2 + per-row AR(5) TTT confidence scaling'
    )
    return src


def build_pkg_and_zip(variant: str, base_src: str) -> Path:
    pkg_dir = EXPDIR / f"pkg_{variant}"
    pkg_dir.mkdir(exist_ok=True)

    # Copy all base files
    for f in V2_PKG.iterdir():
        if f.is_file() and f.name != "__pycache__":
            shutil.copy2(f, pkg_dir / f.name)

    # Write modified Predictor.py
    modified_src = build_variant_predictor(base_src, variant)
    (pkg_dir / "Predictor.py").write_text(modified_src)

    # Build zip
    zip_name = f"submission_050909_iter019_v2T_AR_{variant}.zip"
    zip_path = WORKDIR / zip_name
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(pkg_dir.iterdir()):
            if f.is_file():
                zf.write(f, f.name)

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    md5 = hashlib.md5(zip_path.read_bytes()).hexdigest()
    print(f"  [{variant}] ZIP: {zip_path.name} ({size_mb:.2f} MB, md5={md5[:8]})")
    return zip_path, md5, size_mb


def smoke_test_pkg(pkg_dir: Path) -> bool:
    """Quick smoke test: load Predictor, run 3 random batches, verify output."""
    import importlib.util
    import random

    rng = np.random.default_rng(42)
    cols = [c for c in FEATURE_COLS if c != "sym"]
    df = pd.DataFrame(rng.standard_normal((WINDOW, len(cols))).astype(np.float32), columns=cols)
    df["sym"] = 2
    df["midprice1"] = rng.standard_normal(WINDOW).astype(np.float32) * 1e-4

    spec = importlib.util.spec_from_file_location("t169_pred_smoke", pkg_dir / "Predictor.py")
    mod = importlib.util.module_from_spec(spec)
    orig = os.getcwd()
    os.chdir(str(pkg_dir))
    try:
        spec.loader.exec_module(mod)
        pred = mod.Predictor()
        out = pred.predict([df, df, df])
    except Exception as e:
        print(f"  SMOKE FAILED: {e}")
        return False
    finally:
        os.chdir(orig)

    assert len(out) == 3, f"Expected 3 results, got {len(out)}"
    for row in out:
        assert len(row) == 5, f"Expected 5 actions per row, got {len(row)}"
        for a in row:
            assert a in (0, 1, 2), f"Unexpected action {a}"
    assert not any(np.isnan(a) for row in out for a in row), "NaN in output"
    print(f"  Smoke test OK: 3 batches → {out[0]}")
    return True


def main():
    import wandb

    update_progress("Initializing T169 AR(5) TTT experiment...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        wandb.init(
            project="liangwenbei_T169",
            entity="cjxh21-Tsinghua University",
            config={
                "experiment": "T169_AR_TTT",
                "holdout_dates": f"{HOLDOUT_DATES[0]}-{HOLDOUT_DATES[-1]}",
                "ar_order": 5,
                "train_ticks": 90,
                "test_ticks": 5,
                "variants": list(VARIANTS.keys()),
            }
        )
    except Exception as e:
        print(f"[WandB] init failed ({e}), offline mode")
        wandb.init(project="liangwenbei_T169", mode="offline", config={"experiment": "T169"})

    # Read base Predictor source once
    base_src = (V2_PKG / "Predictor.py").read_text()

    # ── Build and smoke-test all 4 variant packages ───────────────────────────
    update_progress("Building variant packages...")
    zip_info = {}
    for variant in VARIANTS:
        print(f"\nBuilding {variant}...")
        zip_path, md5, size_mb = build_pkg_and_zip(variant, base_src)
        pkg_dir = EXPDIR / f"pkg_{variant}"
        ok = smoke_test_pkg(pkg_dir)
        if not ok:
            raise RuntimeError(f"Smoke test failed for {variant}")
        zip_info[variant] = {"zip_path": zip_path, "md5": md5, "size_mb": size_mb}

    # ── Load v2 Predictor for holdout eval ───────────────────────────────────
    update_progress("Loading v2 Predictor...")
    predictor = load_v2_predictor()
    print("v2 Predictor loaded OK")

    # ── Baseline eval ────────────────────────────────────────────────────────
    update_progress("Evaluating baseline (v2, no AR scaling)...")
    baseline_result = evaluate_holdout(make_predict_fn(predictor, None), "baseline")
    baseline_pnl = baseline_result["best_score"]

    # ── Variant evals ────────────────────────────────────────────────────────
    variant_results = {}
    for variant, scale_fn in VARIANTS.items():
        update_progress(f"Evaluating {variant}...")
        r = evaluate_holdout(make_predict_fn(predictor, scale_fn), variant)
        pnl = r["best_score"]
        delta = pnl - baseline_pnl
        variant_results[variant] = {
            "holdout_pnl": round(pnl, 4),
            "delta": round(delta, 4),
            "zip_md5": zip_info[variant]["md5"],
            "zip_size_mb": round(zip_info[variant]["size_mb"], 2),
        }
        print(f"  {variant}: pnl={pnl:.4f}, delta={delta:+.4f}")

    # ── WandB logging ────────────────────────────────────────────────────────
    log_dict = {"v2_baseline_pnl": baseline_pnl}
    for v, r in variant_results.items():
        log_dict[f"{v}_pnl"] = r["holdout_pnl"]
        log_dict[f"{v}_delta"] = r["delta"]
    wandb.log(log_dict)
    wandb.finish()

    # ── Copy zips to output dir ───────────────────────────────────────────────
    update_progress("Copying zips to output dir...")
    for variant, info in zip_info.items():
        dst = OUTPUT_DIR / info["zip_path"].name
        shutil.copy2(info["zip_path"], dst)
        print(f"  Copied {info['zip_path'].name} → {dst}")

    # ── Write results.json ────────────────────────────────────────────────────
    results = {
        "task": "T169 in-row AR(5) TTT confidence calibrator",
        "v2_baseline_holdout": round(baseline_pnl, 4),
        "variants": variant_results,
        "warning": "Holdout deltas are unreliable; user should ship and observe.",
        "holdout_dates": f"{HOLDOUT_DATES[0]}-{HOLDOUT_DATES[-1]}",
        "notes": (
            "AR(5) fit on ticks 5..94 (90 samples), evaluated on ticks 94..98 (5 samples). "
            "R² used as confidence. No cross-row state. "
            "Scaling applied before conformal gate."
        ),
    }

    results_path = EXPDIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {results_path}")
    shutil.copy2(results_path, OUTPUT_DIR / "T169_results.json")

    update_progress("Done", {"baseline": baseline_pnl, "variants": list(variant_results.keys())})

    best = max(variant_results, key=lambda v: variant_results[v]["delta"])
    best_delta = variant_results[best]["delta"]

    print(f"\nRESULT: task=[T169 AR(5) TTT confidence scaling] "
          f"metrics={{baseline={baseline_pnl:.4f}, "
          + ", ".join(f"{v}={r['holdout_pnl']:.4f}" for v, r in variant_results.items())
          + f", best_delta={best_delta:+.4f}}} "
          f"notes=[best={best}, warning=holdout_unreliable]")
    return results


if __name__ == "__main__":
    main()
