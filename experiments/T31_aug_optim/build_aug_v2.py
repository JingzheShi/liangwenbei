"""T31: aug_a deep-optimization augment library.

Implements:
  * Per-feature uniform random scale [lo, hi]  (T26 aug_a, generalized)
  * Per-feature-GROUP scale (price/volume/amount/intensity/derived/aux)
  * Optional aug_b (Gaussian noise scaled by per-feature std)
  * Optional cutout (random feature mask -> train mean)

All augments are sample-wise stateless; only applied to training data.

Constraint compliance:
  * No use of `sym` / `date` to drive augmentation parameters
  * Augment is stateless across calls
  * Statistics (feat_mean, feat_std) are computed per-fold from THIS fold's
    train data only (no val/test leakage)
"""
from __future__ import annotations

from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Feature group taxonomy (Scheme C 223-d)
# ---------------------------------------------------------------------------
# Indices computed from schemeC_223d_feat_names.txt (0-indexed).
#
# group         | feature names                                     | count
# ------------- | ------------------------------------------------- | -----
# ohlc          | open, high, low, close                            |   4
# volume        | volume_delta                                      |   1
# amount        | amount_delta                                      |   1
# bid_price     | bid1..bid10                                       |  10
# bid_size      | bsize1..bsize10                                   |  10
# ask_price     | ask1..ask10                                       |  10
# ask_size      | asize1..asize10                                   |  10
# avg_price     | avgbid, avgask                                    |   2
# total_size    | totalbsize, totalasize                            |   2
# intst_raw     | lb/la/mb/ma/cb/ca _intst                          |   6
# intst_ind     | lb/la/mb/ma/cb/ca _ind                            |   6
# intst_acc     | lb/la/mb/ma/cb/ca _acc                            |   6
# midprice      | midprice1..midprice10                             |  10
# spread        | spread1..spread10                                 |  10
# bid_diff      | bid_diff1..bid_diff10                             |  10
# ask_diff      | ask_diff1..ask_diff10                             |  10
# means         | bid_mean, ask_mean, bsize_mean, asize_mean        |   4
# cumspread     | cumspread                                         |   1
# imbalance     | imbalance                                         |   1
# rate_price    | bid_rate1..10, ask_rate1..10                      |  20
# rate_size     | bsize_rate1..10, asize_rate1..10                  |  20
# mlofi         | mlofi_W{5,20,60}_lvl{1..10}                       |  30
# wmp           | wmp_lvl1..10, wmp_balance_12                      |  11
# rv            | rv_w5/10/20/50                                    |   4
# ewma_intst    | ewma_a{0.05,0.1,0.3,0.5}_{lb..ca}_intst           |  24
# Total                                                                223


def build_group_indices(feat_names: list[str]) -> dict[str, list[int]]:
    """Return mapping group_name -> list of column indices."""
    g: dict[str, list[int]] = {
        "ohlc": [],
        "volume": [],
        "amount": [],
        "bid_price": [],
        "bid_size": [],
        "ask_price": [],
        "ask_size": [],
        "avg_price": [],
        "total_size": [],
        "intst_raw": [],
        "intst_ind": [],
        "intst_acc": [],
        "midprice": [],
        "spread": [],
        "bid_diff": [],
        "ask_diff": [],
        "means": [],
        "cumspread": [],
        "imbalance": [],
        "rate_price": [],
        "rate_size": [],
        "mlofi": [],
        "wmp": [],
        "rv": [],
        "ewma_intst": [],
    }
    OHLC = {"open", "high", "low", "close"}
    AVG = {"avgbid", "avgask"}
    TOT = {"totalbsize", "totalasize"}
    MEANS = {"bid_mean", "ask_mean", "bsize_mean", "asize_mean"}
    INTST_RAW = {"lb_intst", "la_intst", "mb_intst", "ma_intst", "cb_intst", "ca_intst"}
    INTST_IND = {"lb_ind", "la_ind", "mb_ind", "ma_ind", "cb_ind", "ca_ind"}
    INTST_ACC = {"lb_acc", "la_acc", "mb_acc", "ma_acc", "cb_acc", "ca_acc"}
    for i, n in enumerate(feat_names):
        if n in OHLC:
            g["ohlc"].append(i)
        elif n == "volume_delta":
            g["volume"].append(i)
        elif n == "amount_delta":
            g["amount"].append(i)
        elif n.startswith("bid") and n[3:].isdigit():
            g["bid_price"].append(i)
        elif n.startswith("bsize") and n[5:].isdigit():
            g["bid_size"].append(i)
        elif n.startswith("ask") and n[3:].isdigit():
            g["ask_price"].append(i)
        elif n.startswith("asize") and n[5:].isdigit():
            g["ask_size"].append(i)
        elif n in AVG:
            g["avg_price"].append(i)
        elif n in TOT:
            g["total_size"].append(i)
        elif n in INTST_RAW:
            g["intst_raw"].append(i)
        elif n in INTST_IND:
            g["intst_ind"].append(i)
        elif n in INTST_ACC:
            g["intst_acc"].append(i)
        elif n.startswith("midprice"):
            g["midprice"].append(i)
        elif n.startswith("spread"):
            g["spread"].append(i)
        elif n.startswith("bid_diff"):
            g["bid_diff"].append(i)
        elif n.startswith("ask_diff"):
            g["ask_diff"].append(i)
        elif n in MEANS:
            g["means"].append(i)
        elif n == "cumspread":
            g["cumspread"].append(i)
        elif n == "imbalance":
            g["imbalance"].append(i)
        elif n.startswith("bid_rate") or n.startswith("ask_rate"):
            g["rate_price"].append(i)
        elif n.startswith("bsize_rate") or n.startswith("asize_rate"):
            g["rate_size"].append(i)
        elif n.startswith("mlofi_"):
            g["mlofi"].append(i)
        elif n.startswith("wmp"):
            g["wmp"].append(i)
        elif n.startswith("rv_w"):
            g["rv"].append(i)
        elif n.startswith("ewma_a"):
            g["ewma_intst"].append(i)
        else:
            raise ValueError(f"unmapped feature: {n}")
    total = sum(len(v) for v in g.values())
    assert total == len(feat_names), f"taxonomy missed: {total} vs {len(feat_names)}"
    return g


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def class_balanced_weight(y: np.ndarray, num_class: int = 3) -> np.ndarray:
    counts = np.bincount(y.astype(np.int64), minlength=num_class).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    cw = len(y) / (num_class * counts)
    return cw[y.astype(np.int64)].astype(np.float32)


def compute_feat_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    feat_mean = X.mean(axis=0).astype(np.float32)
    feat_std = X.std(axis=0).astype(np.float32)
    feat_std = np.maximum(feat_std, 1e-6)
    return feat_mean, feat_std


# ---------------------------------------------------------------------------
# Augments
# ---------------------------------------------------------------------------

def aug_scale_uniform(X: np.ndarray, rng: np.random.Generator,
                      lo: float = 0.8, hi: float = 1.2) -> np.ndarray:
    """T26 aug_a: per-(sample, feature) multiplicative scale."""
    scales = rng.uniform(lo, hi, size=X.shape).astype(X.dtype)
    return X * scales


def aug_scale_per_group(X: np.ndarray, rng: np.random.Generator,
                        group_indices: dict[str, list[int]],
                        group_ranges: dict[str, tuple[float, float]]) -> np.ndarray:
    """Per-(sample, feature) multiplicative scale, with each feature's
    [lo, hi] determined by its group. Default [0.8, 1.2] for unspecified groups.
    """
    out = X.copy()
    n = X.shape[0]
    for gname, idxs in group_indices.items():
        if not idxs:
            continue
        lo, hi = group_ranges.get(gname, (0.8, 1.2))
        if lo == 1.0 and hi == 1.0:
            continue  # group is "off" (no aug)
        idxs_arr = np.asarray(idxs, dtype=np.int64)
        scales = rng.uniform(lo, hi, size=(n, idxs_arr.shape[0])).astype(X.dtype)
        out[:, idxs_arr] = X[:, idxs_arr] * scales
    return out


def aug_noise_gauss(X: np.ndarray, rng: np.random.Generator,
                    feat_std: np.ndarray, sigma_frac: float = 0.02) -> np.ndarray:
    """Additive Gaussian noise scaled by per-feature std."""
    noise = rng.standard_normal(size=X.shape).astype(X.dtype)
    scale = (sigma_frac * feat_std).astype(X.dtype)
    return X + noise * scale


def aug_cutout(X: np.ndarray, rng: np.random.Generator,
               feat_mean: np.ndarray, drop_lo: float = 0.05,
               drop_hi: float = 0.10) -> np.ndarray:
    """Per-sample, replace drop_lo..drop_hi fraction of feats by train mean."""
    drop_rates = rng.uniform(drop_lo, drop_hi, size=(X.shape[0], 1)).astype(np.float32)
    keep_prob = 1.0 - drop_rates
    keep_mask = (rng.random(size=X.shape).astype(np.float32) < keep_prob)
    out = np.where(keep_mask, X, feat_mean.astype(X.dtype))
    return out.astype(X.dtype)


# ---------------------------------------------------------------------------
# Variant builder
# ---------------------------------------------------------------------------

def build_train_for_variant(
    variant_cfg: dict,
    X_orig: np.ndarray,
    y_orig: np.ndarray,
    feat_std: np.ndarray,
    feat_mean: np.ndarray,
    group_indices: dict[str, list[int]],
    rng: np.random.Generator,
    num_class: int = 3,
):
    """Apply augment(s) per variant_cfg.

    variant_cfg fields:
      kind: 'baseline' | 'uniform' | 'per_group' | 'uniform+noise'
            | 'uniform+cutout' | 'per_group+noise'
      aug_ratio: float (default 1.0) - ratio of augmented copies to add
      lo, hi: float (for 'uniform' kind)
      group_ranges: dict[str, (lo, hi)] (for 'per_group' kind, group_indices keys)
      sigma_frac: float (for noise variants, default 0.02)
      drop_lo, drop_hi: floats (for cutout variants)

    Returns:
      X_train, y_train, sw_train  (concat of orig + aug copies)
    """
    kind = variant_cfg.get("kind", "baseline")
    if kind == "baseline":
        return X_orig, y_orig.astype(np.int64), class_balanced_weight(y_orig, num_class)

    aug_ratio = float(variant_cfg.get("aug_ratio", 1.0))
    n_aug = int(round(aug_ratio * len(X_orig)))
    if n_aug <= 0:
        return X_orig, y_orig.astype(np.int64), class_balanced_weight(y_orig, num_class)

    if n_aug == len(X_orig):
        idx = np.arange(len(X_orig))
    else:
        idx = rng.integers(0, len(X_orig), size=n_aug)
    X_src = X_orig[idx]
    y_src = y_orig[idx]

    if kind == "uniform":
        lo = float(variant_cfg.get("lo", 0.8))
        hi = float(variant_cfg.get("hi", 1.2))
        X_aug = aug_scale_uniform(X_src, rng, lo=lo, hi=hi)
    elif kind == "per_group":
        gr = variant_cfg["group_ranges"]
        X_aug = aug_scale_per_group(X_src, rng, group_indices, gr)
    elif kind == "uniform+noise":
        lo = float(variant_cfg.get("lo", 0.8))
        hi = float(variant_cfg.get("hi", 1.2))
        sigma = float(variant_cfg.get("sigma_frac", 0.02))
        X_aug = aug_scale_uniform(X_src, rng, lo=lo, hi=hi)
        X_aug = aug_noise_gauss(X_aug, rng, feat_std, sigma_frac=sigma)
    elif kind == "uniform+cutout":
        lo = float(variant_cfg.get("lo", 0.8))
        hi = float(variant_cfg.get("hi", 1.2))
        d_lo = float(variant_cfg.get("drop_lo", 0.05))
        d_hi = float(variant_cfg.get("drop_hi", 0.10))
        X_aug = aug_scale_uniform(X_src, rng, lo=lo, hi=hi)
        X_aug = aug_cutout(X_aug, rng, feat_mean, drop_lo=d_lo, drop_hi=d_hi)
    elif kind == "per_group+noise":
        gr = variant_cfg["group_ranges"]
        sigma = float(variant_cfg.get("sigma_frac", 0.02))
        X_aug = aug_scale_per_group(X_src, rng, group_indices, gr)
        X_aug = aug_noise_gauss(X_aug, rng, feat_std, sigma_frac=sigma)
    else:
        raise ValueError(f"unknown kind {kind}")

    X_merged = np.concatenate([X_orig, X_aug], axis=0)
    y_merged = np.concatenate([y_orig, y_src], axis=0).astype(np.int64)
    w_merged = class_balanced_weight(y_merged, num_class)
    return X_merged, y_merged, w_merged


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
    fnp = os.path.join(ROOT, "experiments", "T5b_features_multihorizon", "cache",
                       "schemeC_223d_feat_names.txt")
    with open(fnp) as f:
        feat_names = [ln.strip() for ln in f if ln.strip()]
    print(f"loaded {len(feat_names)} feature names")
    g = build_group_indices(feat_names)
    print("\nGroup taxonomy:")
    total = 0
    for k, v in g.items():
        print(f"  {k:12s}: n={len(v):3d}  e.g. {[feat_names[i] for i in v[:3]]}")
        total += len(v)
    print(f"  total = {total}")

    # Smoke test variants
    rng = np.random.default_rng(0)
    X = rng.standard_normal((1000, len(feat_names))).astype(np.float32)
    y = rng.integers(0, 3, size=1000)
    fmean, fstd = compute_feat_stats(X)

    for name, cfg in [
        ("baseline", {"kind": "baseline"}),
        ("uniform[0.7,1.3]", {"kind": "uniform", "lo": 0.7, "hi": 1.3}),
        ("per_group", {"kind": "per_group", "group_ranges": {
            "amount": (0.5, 1.5), "volume": (0.7, 1.3), "ohlc": (0.95, 1.05),
            "ewma_intst": (0.8, 1.2),
        }}),
        ("uniform+noise", {"kind": "uniform+noise", "lo": 0.8, "hi": 1.2, "sigma_frac": 0.02}),
        ("uniform+cutout", {"kind": "uniform+cutout", "lo": 0.8, "hi": 1.2, "drop_lo": 0.05, "drop_hi": 0.10}),
    ]:
        Xn, yn, wn = build_train_for_variant(cfg, X, y, fstd, fmean, g, rng)
        print(f"  {name:25s}: X={Xn.shape} y={yn.shape} w_sum={wn.sum():.0f}")
