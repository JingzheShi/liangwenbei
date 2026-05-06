"""Toy hand-calculation tests for the R20 feature module."""
from __future__ import annotations

import sys
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from r20_features import (
    compute_stoikov, compute_imb_slope, compute_depth_skew,
    compute_depth_concentration, compute_pairwise_diffs, compute_triplet_imb,
    _signed_carry_fwd, _wmp_lvl1_from_df,
)


def make_toy_df(n=10):
    rng = np.random.default_rng(0)
    cols = {}
    for k in range(1, 11):
        cols[f"bid{k}"] = rng.uniform(-0.01, 0.01, n)
        cols[f"ask{k}"] = cols[f"bid{k}"] + rng.uniform(0.0001, 0.001, n)
        cols[f"bsize{k}"] = rng.uniform(0.001, 0.01, n)
        cols[f"asize{k}"] = rng.uniform(0.001, 0.01, n)
    cols["open"] = (cols["bid1"] + cols["ask1"]) / 2
    cols["close"] = cols["open"] + rng.uniform(-0.001, 0.001, n)
    cols["high"] = np.maximum(cols["open"], cols["close"]) + 0.0005
    cols["low"] = np.minimum(cols["open"], cols["close"]) - 0.0005
    cols["volume_delta"] = rng.uniform(0.0, 0.1, n)
    cols["amount_delta"] = rng.uniform(-0.5, 0.5, n)
    return pd.DataFrame(cols)


def test_stoikov():
    """At I=0.5 (bs=as), micro_price == mid; otherwise micro skews toward bid (I<0.5)
    or ask (I>0.5).
    """
    df = pd.DataFrame({
        "bid1": [-0.001, -0.001, -0.001],
        "ask1": [+0.001, +0.001, +0.001],
        "bsize1": [10.0, 1.0, 100.0],
        "asize1": [10.0, 100.0, 1.0],
    })
    out = compute_stoikov(df)
    mid = (df["bid1"] + df["ask1"]) / 2
    # I = 0.5 → micro == mid
    assert np.isclose(out["stoikov_micro"].iloc[0], mid.iloc[0]), \
        f"row 0: micro {out['stoikov_micro'].iloc[0]} != mid {mid.iloc[0]}"
    # I < 0.5 → micro < mid (bias toward bid)
    assert out["stoikov_micro"].iloc[1] < mid.iloc[1]
    # I > 0.5 → micro > mid
    assert out["stoikov_micro"].iloc[2] > mid.iloc[2]
    print("OK test_stoikov")


def test_imb_slope():
    """imb_lvl_k = (bs - as) / (bs + as + ε), ranges in [-1, 1]."""
    df = pd.DataFrame({
        f"bsize{k}": [10.0, 0.0, 5.0] for k in range(1, 11)
    } | {
        f"asize{k}": [0.0, 10.0, 5.0] for k in range(1, 11)
    } | {
        f"bid{k}": [0.0, 0.0, 0.0] for k in range(1, 11)
    } | {
        f"ask{k}": [0.0, 0.0, 0.0] for k in range(1, 11)
    })
    out = compute_imb_slope(df)
    assert np.isclose(out["imb_lvl1"].iloc[0], 1.0, atol=1e-6)
    assert np.isclose(out["imb_lvl1"].iloc[1], -1.0, atol=1e-6)
    assert np.isclose(out["imb_lvl1"].iloc[2], 0.0, atol=1e-6)
    print("OK test_imb_slope")


def test_depth_skew():
    """depth_skew = (Σ bs - Σ as) / (Σ bs + Σ as)."""
    cols = {f"bsize{k}": [k * 1.0] for k in range(1, 11)}
    cols.update({f"asize{k}": [k * 0.5] for k in range(1, 11)})
    cols.update({f"bid{k}": [0.0] for k in range(1, 11)})
    cols.update({f"ask{k}": [0.0] for k in range(1, 11)})
    df = pd.DataFrame(cols)
    bs_total = sum(range(1, 11))           # 55
    as_total = sum(range(1, 11)) * 0.5     # 27.5
    expected = (bs_total - as_total) / (bs_total + as_total)
    out = compute_depth_skew(df)
    assert np.isclose(out["depth_skew"].iloc[0], expected, atol=1e-6), \
        f"got {out['depth_skew'].iloc[0]}, expected {expected}"
    print("OK test_depth_skew")


def test_concentration():
    """top-3 / total = (1+2+3) / 55 = 6/55."""
    cols = {f"bsize{k}": [k * 1.0] for k in range(1, 11)}
    cols.update({f"asize{k}": [k * 1.0] for k in range(1, 11)})
    cols.update({f"bid{k}": [0.0] for k in range(1, 11)})
    cols.update({f"ask{k}": [0.0] for k in range(1, 11)})
    df = pd.DataFrame(cols)
    out = compute_depth_concentration(df)
    assert np.isclose(out["bid_top3_concentration"].iloc[0], 6 / 55, atol=1e-6)
    assert np.isclose(out["ask_top5_concentration"].iloc[0], 15 / 55, atol=1e-6)
    print("OK test_concentration")


def test_pairwise_first_pair():
    """(b1, a1) -> b1 - a1 (negative, equals -spread)."""
    df = make_toy_df()
    wmp = _wmp_lvl1_from_df(df)
    out = compute_pairwise_diffs(df, wmp)
    expected = (df["bid1"] - df["ask1"]).to_numpy()
    np.testing.assert_allclose(out["pair_b1_a1"].to_numpy(), expected, atol=1e-9)
    print("OK test_pairwise_first_pair")


def test_signed_carry_fwd():
    """0s replaced by previous nonzero sign; leading 0 -> 0."""
    arr = np.array([0, 1, 0, -1, 0, 0, 1])
    expected = np.array([0, 1, 1, -1, -1, -1, 1])
    out = _signed_carry_fwd(arr)
    np.testing.assert_array_equal(out, expected)
    print("OK test_signed_carry_fwd")


def test_triplet_b1_mid_a1():
    """(b1, mid, a1) sorted is (b1, mid, a1) since b1 < mid < a1.
    triplet_imb = (a1 - mid) / (mid - b1 + ε) ~ 1 (approximately).
    """
    df = pd.DataFrame({
        "bid1": [-0.001], "ask1": [0.001], "bid5": [-0.005], "ask5": [0.005],
        "bid10": [-0.010], "ask10": [0.010], "open": [0.0], "close": [0.0],
        "bsize1": [1.0], "asize1": [1.0], "bsize5": [1.0], "asize5": [1.0],
        "bsize10": [1.0], "asize10": [1.0],
    })
    for k in range(2, 10):
        if k != 5:
            df[f"bid{k}"] = -k * 0.001
            df[f"ask{k}"] = k * 0.001
            df[f"bsize{k}"] = 1.0
            df[f"asize{k}"] = 1.0
    wmp = _wmp_lvl1_from_df(df)
    out = compute_triplet_imb(df, wmp)
    # b1=-0.001, mid=0, a1=0.001
    # sorted: -0.001, 0, 0.001 -> (a1 - mid) / (mid - b1) = 0.001/0.001 = 1
    assert np.isclose(out["trip_b1_mid_a1"].iloc[0], 1.0, atol=1e-3)
    print("OK test_triplet_b1_mid_a1")


if __name__ == "__main__":
    test_stoikov()
    test_imb_slope()
    test_depth_skew()
    test_concentration()
    test_pairwise_first_pair()
    test_signed_carry_fwd()
    test_triplet_b1_mid_a1()
    print("\nAll tests passed.")
