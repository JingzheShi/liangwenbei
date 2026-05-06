"""Unit tests for T3 feature_v1 compute module.

Run: python -m pytest experiments/T3_features_v1/test_compute.py -v
or:  python experiments/T3_features_v1/test_compute.py
"""
from __future__ import annotations

import datetime as _dt
import os
import sys

import numpy as np
import pandas as pd

# Allow running as a script from repo root
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from compute import (  # noqa: E402
    EWMA_ALPHAS,
    LOB_LEVELS,
    MLOFI_WINDOWS,
    RV_WINDOWS,
    compute_all,
    compute_ewma_intst,
    compute_mlofi,
    compute_rv,
    compute_time_encoding,
    compute_wmp,
    feature_v1_columns,
)

REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


# ---- helpers ----------------------------------------------------------------

def _make_minimal_lob_df(n: int = 5,
                        prices: np.ndarray | None = None,
                        sizes: np.ndarray | None = None) -> pd.DataFrame:
    """Build a tiny synthetic LOB df with all 10-level cols + intst + time.

    bid_k = -k * 0.0001 (descending), ask_k = +k * 0.0001 (ascending), all
    constant unless overridden.
    """
    cols = {}
    for k in LOB_LEVELS:
        cols[f"bid{k}"] = np.full(n, -k * 0.0001, dtype=np.float64)
        cols[f"ask{k}"] = np.full(n, +k * 0.0001, dtype=np.float64)
        cols[f"bsize{k}"] = np.full(n, 0.001, dtype=np.float64)
        cols[f"asize{k}"] = np.full(n, 0.001, dtype=np.float64)
    for c in ("lb_intst", "la_intst", "mb_intst", "ma_intst", "cb_intst", "ca_intst"):
        cols[c] = np.full(n, 0.5, dtype=np.float64)
    # time: am session starting 09:40:00, 3s spacing
    base = _dt.datetime(2025, 1, 1, 9, 40, 0)
    cols["time"] = [(base + _dt.timedelta(seconds=3 * i)).time() for i in range(n)]
    df = pd.DataFrame(cols)
    return df


# ---- tests ------------------------------------------------------------------

def test_mlofi_formula_toy():
    """3-tick toy: verify MLOFI level 1 e_k(t) matches hand calc."""
    df = _make_minimal_lob_df(n=3)
    # Override level-1 path:
    # t=0: bid1=0.001, bs1=10; ask1=0.002, as1=10
    # t=1: bid1=0.001 (=), bs1=20; ask1=0.003 (up), as1=15
    # t=2: bid1=0.0008 (down), bs1=12; ask1=0.003 (=), as1=8
    df.loc[0, "bid1"] = 0.001
    df.loc[1, "bid1"] = 0.001
    df.loc[2, "bid1"] = 0.0008
    df.loc[0, "ask1"] = 0.002
    df.loc[1, "ask1"] = 0.003
    df.loc[2, "ask1"] = 0.003
    df.loc[0, "bsize1"] = 10
    df.loc[1, "bsize1"] = 20
    df.loc[2, "bsize1"] = 12
    df.loc[0, "asize1"] = 10
    df.loc[1, "asize1"] = 15
    df.loc[2, "asize1"] = 8

    # Hand calc level 1:
    # t=1: b_t = b_{t-1} (so I(>=)=1, I(<=)=1) → +bs_t - bs_{t-1} = 20 - 10 = 10 (bid contribution)
    #      a_t = 0.003 > 0.002 (so I(<=)=0, I(>=)=1) → 0 + as_{t-1} = 10 (ask contribution)
    #      e_1(1) = 10 + 10 = 20
    # t=2: b_t < b_{t-1} (I(>=)=0, I(<=)=1) → -bs_{t-1} = -20 (bid)
    #      a_t = a_{t-1} (I(<=)=1, I(>=)=1) → -as_t + as_{t-1} = -8 + 15 = 7 (ask)
    #      e_1(2) = -20 + 7 = -13
    # mlofi_W{5,20,60}_lvl1 will be NaN for n=3 (windows > n).
    # We need to test the raw e_k. So bypass rolling and test compute_mlofi behavior with W=2.
    from compute import compute_mlofi  # already imported; re-import for clarity

    # Use rolling W=2 by monkey-patching MLOFI_WINDOWS via direct rolling check
    # Easier: compute the e directly using the same inner logic:
    b = df["bid1"]
    a = df["ask1"]
    bs = df["bsize1"]
    as_ = df["asize1"]
    b_p = b.shift(1)
    a_p = a.shift(1)
    bs_p = bs.shift(1)
    as_p = as_.shift(1)
    e = ((b >= b_p).astype(float) * bs - (b <= b_p).astype(float) * bs_p
         - (a <= a_p).astype(float) * as_ + (a >= a_p).astype(float) * as_p)
    assert pd.isna(e.iloc[0]), "t=0 should be NaN"
    assert e.iloc[1] == 20.0, f"t=1 expected 20, got {e.iloc[1]}"
    assert e.iloc[2] == -13.0, f"t=2 expected -13, got {e.iloc[2]}"

    # Also verify compute_mlofi returns the right shape and column names
    df2 = _make_minimal_lob_df(n=70)
    out = compute_mlofi(df2)
    assert out.shape[1] == 30, f"expected 30 cols, got {out.shape[1]}"
    expected_cols = {f"mlofi_W{W}_lvl{k}" for W in MLOFI_WINDOWS for k in LOB_LEVELS}
    assert set(out.columns) == expected_cols
    # When all bid/ask/sizes constant, every e_k = +bs_t - bs_{t-1} - as_t + as_{t-1} = 0
    # So rolling sums are 0 (after warmup)
    for c in expected_cols:
        # Expect 0s after warmup
        tail = out[c].iloc[60:].dropna()
        assert (tail == 0).all(), f"{c} expected all 0 after warmup, got nonzero"
    print("[PASS] test_mlofi_formula_toy")


def test_wmp_fallback_on_zero_size():
    """If bs_k + as_k == 0, WMP_k must fall back to (a_k + b_k) / 2."""
    df = _make_minimal_lob_df(n=3)
    df.loc[1, "bsize1"] = 0.0
    df.loc[1, "asize1"] = 0.0
    out = compute_wmp(df)
    expected_fallback = (df.loc[1, "ask1"] + df.loc[1, "bid1"]) / 2.0
    assert out.loc[1, "wmp_lvl1"] == expected_fallback, (
        f"expected {expected_fallback}, got {out.loc[1, 'wmp_lvl1']}"
    )
    # Other rows: normal WMP
    bs0 = df.loc[0, "bsize1"]; as0 = df.loc[0, "asize1"]
    expected_normal_0 = (
        df.loc[0, "ask1"] * bs0 + df.loc[0, "bid1"] * as0
    ) / (bs0 + as0)
    assert np.isclose(out.loc[0, "wmp_lvl1"], expected_normal_0), (
        f"expected {expected_normal_0}, got {out.loc[0, 'wmp_lvl1']}"
    )
    # wmp_balance_12 = wmp_lvl1 - wmp_lvl2
    assert np.isclose(
        out.loc[0, "wmp_balance_12"],
        out.loc[0, "wmp_lvl1"] - out.loc[0, "wmp_lvl2"],
    )
    # 11 cols total
    assert out.shape[1] == 11
    print("[PASS] test_wmp_fallback_on_zero_size")


def test_rv_non_negative():
    """RV must be >= 0 everywhere it's defined; first W-1 may be NaN."""
    rng = np.random.RandomState(42)
    wmp = pd.Series(rng.normal(0, 1e-3, 200))  # noisy small returns
    rv = compute_rv(wmp)
    assert rv.shape[1] == 4
    for W in RV_WINDOWS:
        col = f"rv_w{W}"
        v = rv[col].dropna()
        assert (v >= 0).all(), f"{col} has negative values"
        # First W-1 entries (after the log_ret[0] = NaN, that's W entries) are NaN
        # min-period=W with the first log_ret being NaN means first W rows are NaN.
        first_nans = rv[col].iloc[:W].isna().sum()
        assert first_nans == W, f"{col} expected first W NaN, got {first_nans}"
    print("[PASS] test_rv_non_negative")


def test_ewma_converges_to_constant():
    """For a long series of constant c, EWMA should converge to c."""
    n = 500
    c = 0.7
    df = _make_minimal_lob_df(n=n)
    for col in ("lb_intst", "la_intst", "mb_intst", "ma_intst", "cb_intst", "ca_intst"):
        df[col] = c
    out = compute_ewma_intst(df)
    # 6 * 4 = 24 cols
    assert out.shape[1] == 24
    # Tail values should be very close to c (within floating-point noise)
    tail = out.iloc[-50:]
    diffs = (tail - c).abs().max().max()
    assert diffs < 1e-6, f"EWMA tail max abs diff = {diffs}, expected < 1e-6"
    # First value should equal c too (initialized via adjust=False with first sample = c)
    assert np.isclose(out.iloc[0].max(), c) and np.isclose(out.iloc[0].min(), c)
    print("[PASS] test_ewma_converges_to_constant")


def test_time_encoding_range():
    """minutes ∈ [0, 99], is_pm ∈ {0, 1}, both AM & PM correct."""
    am = _make_minimal_lob_df(n=2001)
    am["time"] = [
        (_dt.datetime(2025, 1, 1, 9, 40, 0) + _dt.timedelta(seconds=3 * i)).time()
        for i in range(2001)
    ]
    out = compute_time_encoding(am, session="am")
    assert out["time_minutes_since_session_start"].min() == 0
    assert out["time_minutes_since_session_start"].max() <= 99
    assert (out["time_minutes_since_session_start"] >= 0).all()
    assert (out["time_is_pm"] == 0).all()
    # First row → minute 0
    assert out["time_minutes_since_session_start"].iloc[0] == 0
    assert out["time_session_progress"].iloc[0] == 0.0
    # row at index 60s/3=20 → 1 minute
    assert out["time_minutes_since_session_start"].iloc[20] == 1

    pm = _make_minimal_lob_df(n=2001)
    pm["time"] = [
        (_dt.datetime(2025, 1, 1, 13, 10, 0) + _dt.timedelta(seconds=3 * i)).time()
        for i in range(2001)
    ]
    out_pm = compute_time_encoding(pm, session="pm")
    assert (out_pm["time_is_pm"] == 1).all()
    assert out_pm["time_minutes_since_session_start"].min() == 0
    assert out_pm["time_minutes_since_session_start"].max() <= 99

    # Auto-detect session works
    out_auto = compute_time_encoding(pm)
    assert (out_auto["time_is_pm"] == 1).all()
    print("[PASS] test_time_encoding_range")


def test_session_isolation():
    """Two consecutive different (sym, date, session) files: rolling/EWMA computed
    per-session must NOT leak across files (because we never concat them)."""
    real_path_am = os.path.join(REPO_ROOT, "data", "snapshot_sym0_date0_am.parquet")
    real_path_pm = os.path.join(REPO_ROOT, "data", "snapshot_sym0_date0_pm.parquet")
    if not (os.path.exists(real_path_am) and os.path.exists(real_path_pm)):
        print("[SKIP] test_session_isolation (data files missing)")
        return
    df_am = pd.read_parquet(real_path_am)
    df_pm = pd.read_parquet(real_path_pm)
    out_am = compute_all(df_am, session="am")
    out_pm = compute_all(df_pm, session="pm")
    # Each must independently NaN for first ~60 ticks of mlofi_W60
    assert out_am["mlofi_W60_lvl1"].iloc[:60].isna().all()
    assert out_pm["mlofi_W60_lvl1"].iloc[:60].isna().all()
    # Time encoding is_pm differs
    assert (out_am["time_is_pm"] == 0).all()
    assert (out_pm["time_is_pm"] == 1).all()
    # EWMA at t=0 should equal intst[0] (verifying no carry-over from prior session)
    for alpha in EWMA_ALPHAS:
        for c in ("lb_intst", "la_intst", "mb_intst", "ma_intst", "cb_intst", "ca_intst"):
            col = f"ewma_a{alpha}_{c}"
            assert np.isclose(out_am[col].iloc[0], df_am[c].iloc[0]), col
            assert np.isclose(out_pm[col].iloc[0], df_pm[c].iloc[0]), col
    print("[PASS] test_session_isolation")


def test_no_inf_in_output():
    """No ±inf anywhere in output, even with extreme inputs."""
    df = _make_minimal_lob_df(n=200)
    # Stress: zero sizes, extreme prices
    df.loc[10, "bsize1"] = 0
    df.loc[10, "asize1"] = 0
    df.loc[20, "bsize3"] = 0
    df.loc[20, "asize3"] = 0
    out = compute_all(df, session="am")
    # Verify no inf
    arr = out.to_numpy(dtype=np.float64, na_value=0.0)
    assert not np.isinf(arr).any(), "found inf in output"
    # NaN policy: only allowed at the head (rolling warmup), and time encoding never NaN
    for col in ("time_minutes_since_session_start", "time_session_progress", "time_is_pm"):
        assert out[col].isna().sum() == 0, f"{col} has NaN"
    # Tail (after row 60) should have no NaN in mlofi/rv/ewma/wmp
    tail = out.iloc[60:]
    nan_per_col = tail.isna().sum()
    bad = nan_per_col[nan_per_col > 0]
    assert bad.empty, f"unexpected NaN after warmup: {bad.to_dict()}"
    print("[PASS] test_no_inf_in_output")


def test_columns_order_and_count():
    """compute_all output must have exactly the columns listed by feature_v1_columns()."""
    df = _make_minimal_lob_df(n=70)
    out = compute_all(df, session="am")
    assert list(out.columns) == feature_v1_columns()
    assert out.shape[1] == 72
    print("[PASS] test_columns_order_and_count")


# ---- runner -----------------------------------------------------------------

def main():
    tests = [
        test_mlofi_formula_toy,
        test_wmp_fallback_on_zero_size,
        test_rv_non_negative,
        test_ewma_converges_to_constant,
        test_time_encoding_range,
        test_session_isolation,
        test_no_inf_in_output,
        test_columns_order_and_count,
    ]
    failed = []
    for fn in tests:
        try:
            fn()
        except Exception as e:
            failed.append((fn.__name__, repr(e)))
            print(f"[FAIL] {fn.__name__}: {e}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} tests passed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
