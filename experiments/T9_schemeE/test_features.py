"""Unit tests for F1-F4 vector implementations."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from build_features import (
    compute_F1, compute_F2, compute_F3, compute_F4,
    build_features_one_session, get_feature_group_indices, feature_names,
    EPS, WINDOW,
)
from src.data.dataset import get_default_feature_cols


def test_F1_zscore_correctness():
    """F1 last-tick z-score should equal hand-computed (X - mean) / (std + eps)."""
    rng = np.random.default_rng(42)
    T = 250
    amount = rng.uniform(low=100.0, high=1e6, size=T).astype(np.float32)
    z_last, z_q95 = compute_F1(amount, W=100)

    # First 99 should be NaN
    assert np.isnan(z_last[:99]).all(), "first W-1 should be NaN"
    assert np.isnan(z_q95[:99]).all()

    # Manual check at t=99 (first valid)
    log_amt = np.log1p(amount[:100])  # all positive
    mean = log_amt.mean()
    std = log_amt.std() + EPS
    expected_z = (log_amt[-1] - mean) / std
    assert abs(z_last[99] - expected_z) < 1e-4, f"got {z_last[99]} vs {expected_z}"

    # At t=200, window is [101..200]
    log_amt = np.log1p(amount[101:201])
    mean = log_amt.mean()
    std = log_amt.std() + EPS
    expected_z = (log_amt[-1] - mean) / std
    assert abs(z_last[200] - expected_z) < 1e-4, f"got {z_last[200]} vs {expected_z}"

    print(f"  F1 OK: z_last[99]={z_last[99]:.4f}  z_q95[99]={z_q95[99]:.4f}")


def test_F1_zero_amounts():
    """Constant amount should give z-score = 0 (or close due to EPS)."""
    amount = np.full(150, 1000.0, dtype=np.float32)
    z_last, _ = compute_F1(amount, W=100)
    assert abs(z_last[100]) < 1e-3, f"constant amount → z should be 0, got {z_last[100]}"
    print(f"  F1 constant input → z_last[100]={z_last[100]:.6f} (≈0) OK")


def test_F2_spread_normalized_move():
    """For a constant 1.0 spread and Δmid = 0.001 at last tick → norm_move = 0.001."""
    T = 150
    midprice = np.zeros(T, dtype=np.float32)
    midprice[100] = 0.001  # jump at t=100
    spread = np.ones(T, dtype=np.float32)  # spread1+1 = 1.0
    last, max_abs = compute_F2(midprice, spread, W=100)

    # At t=100, the window contains [t-99..t] = ticks 1..100. Δmid at t=100 = 0.001.
    assert abs(last[100] - 0.001) < 1e-5, f"expected 0.001, got {last[100]}"
    # max_abs over window should also be 0.001
    assert abs(max_abs[100] - 0.001) < 1e-5
    # Reduce spread to 0.5 → move should normalize to 0.002
    spread2 = np.full(T, 0.5, dtype=np.float32)
    last2, _ = compute_F2(midprice, spread2, W=100)
    assert abs(last2[100] - 0.002) < 1e-5, f"expected 0.002, got {last2[100]}"
    print(f"  F2 OK: spread=1 → 0.001, spread=0.5 → 0.002")


def test_F3_relvol_quantile():
    """If all moves have same magnitude, current_abs_move / std_window → constant."""
    T = 150
    # constant Δmid pattern: alternating ±0.001 → std ≈ 0.001
    delta = np.zeros(T, dtype=np.float32)
    delta[1:] = 0.001 * (-1) ** np.arange(T - 1)
    midprice = np.cumsum(delta).astype(np.float32)
    rel = compute_F3(midprice, W=100)
    # At t=100, current |Δmid| = 0.001, std ≈ 0.001 → ratio ≈ 1
    assert 0.9 < rel[100] < 1.1, f"expected ~1.0, got {rel[100]}"
    print(f"  F3 OK: alternating moves → rel[100]={rel[100]:.4f} (≈1)")


def test_F4_ratios_bounded():
    """Ratios should all be in [0, 1]."""
    rng = np.random.default_rng(0)
    T = 200
    intst = {k: rng.uniform(0, 100, T).astype(np.float32) for k in
             ("lb", "la", "mb", "ma", "cb", "ca")}
    f4 = compute_F4(intst)  # (T, 5)
    assert f4.shape == (T, 5)
    assert (f4 >= 0).all() and (f4 <= 1).all(), f"ratios out of [0,1]"
    # Edge case: all-zero
    zero_intst = {k: np.zeros(T, dtype=np.float32) for k in intst}
    f4_z = compute_F4(zero_intst)
    assert np.isfinite(f4_z).all()
    print(f"  F4 OK: 5 ratios in [0,1]; zero-input handled")


def test_build_one_session_real_data():
    """Run on actual data to verify shape / NaN handling."""
    feat_cols = get_default_feature_cols()
    df = pd.read_parquet(os.path.join(ROOT, "data", "snapshot_sym0_date0_am.parquet"))
    X, y60, mp_t, mp_t60, t = build_features_one_session(df, feat_cols)
    assert X.shape[1] == 154 + 10, f"unexpected D = {X.shape[1]}"
    assert np.isfinite(X).all(), "NaN / inf in X"
    print(f"  build_one_session OK: X={X.shape}, y={y60.shape}")
    # Check feature group indices
    fnames = feature_names(feat_cols)
    groups = get_feature_group_indices(fnames)
    assert len(groups["A"]) == 154
    assert len(groups["F1"]) == 2
    assert len(groups["F2"]) == 2
    assert len(groups["F3"]) == 1
    assert len(groups["F4"]) == 5
    print(f"  feature groups OK: A={len(groups['A'])} F1={len(groups['F1'])} "
          f"F2={len(groups['F2'])} F3={len(groups['F3'])} F4={len(groups['F4'])}")


def test_sym_agnostic_amount_z():
    """F1 z-score on sym=2 (high vol) vs sym=0 (lower vol) should look comparable."""
    feat_cols = get_default_feature_cols()
    rows = []
    for sym in (0, 2):
        df = pd.read_parquet(
            os.path.join(ROOT, "data", f"snapshot_sym{sym}_date0_am.parquet")
        )
        amount = df["amount_delta"].to_numpy(dtype=np.float32)
        z_last, _ = compute_F1(amount, W=100)
        valid = z_last[99:]
        rows.append({"sym": sym, "z_mean": float(valid.mean()),
                     "z_std": float(valid.std()),
                     "raw_amount_mean": float(amount.mean())})
    print(f"  sym-agnostic F1 check (per-window z eliminates raw scale):")
    for r in rows:
        print(f"    sym={r['sym']}: raw_amount_mean={r['raw_amount_mean']:.0f} "
              f"z_mean={r['z_mean']:.3f} z_std={r['z_std']:.3f}")
    # both should have z_mean ≈ 0 (within-window centered) and z_std ~ small (bounded by W)
    for r in rows:
        assert abs(r["z_mean"]) < 0.5, f"sym {r['sym']} z_mean = {r['z_mean']} not centered"


def main():
    print("=== T9 Scheme E feature unit tests ===")
    test_F1_zscore_correctness()
    test_F1_zero_amounts()
    test_F2_spread_normalized_move()
    test_F3_relvol_quantile()
    test_F4_ratios_bounded()
    test_build_one_session_real_data()
    test_sym_agnostic_amount_z()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
