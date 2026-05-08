"""T115 Phase 2: Build variant feature caches in memory and save to disk.

Variants:
- baseline: schemeP 359-d (T59_FAIL + STAGE5_FAIL dropped)
- variant_A (PM literal additive): baseline + 31 PM-spec ratio features
    (amount_delta + spread1-10 + bid_diff1-10 + ask_diff1-10 + cumspread)
    each divided by (midprice + 1.0 + 1e-8)
- variant_B (PM literal replace): baseline with raw amount_delta + spread1-10 + bid_diff*
    + ask_diff* + cumspread REPLACED by their /(midprice+1) ratios
- variant_C (smart additive): baseline + 31 features using prev_close_proxy
    = bid_mean / (bid1 + 1.0) as divisor for amount_delta, and (raw+1)*prev_close
    for spread/bid_diff/ask_diff/cumspread (which restores 元-tick scale)

Saves npz files in cache/ per split for fast LGB loading.
"""
from __future__ import annotations
import os
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC_CACHE = os.path.join(ROOT, "experiments", "T68_stage5_features", "cache")
DST_CACHE = os.path.join(HERE, "cache")
os.makedirs(DST_CACHE, exist_ok=True)

T59_FAIL_NAMES = [
    "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
    "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
    "qrank_W100_cumspread",
    "kyle_lam_W50", "kyle_lam_W100",
    "roll_eff_spr_ratio_W100",
]
STAGE5_FAIL_NAMES = ["liq_asym_top5_W5"]
DROP_NAMES = T59_FAIL_NAMES + STAGE5_FAIL_NAMES

RAW_NORM_TARGETS = (
    ["amount_delta"]
    + [f"spread{k}" for k in range(1, 11)]
    + [f"bid_diff{k}" for k in range(1, 11)]
    + [f"ask_diff{k}" for k in range(1, 11)]
    + ["cumspread"]
)
assert len(RAW_NORM_TARGETS) == 32  # 1 + 10 + 10 + 10 + 1


def build_split(split):
    print(f"\n=== {split} ===", flush=True)
    t0 = time.time()
    p = os.path.join(SRC_CACHE, f"schemeP_{split}.npz")
    d = np.load(p)
    X = d["X"]
    n, F = X.shape
    feat_names = [l.strip() for l in open(os.path.join(SRC_CACHE, "schemeP_feat_names.txt"))]
    name_to_idx = {n_: i for i, n_ in enumerate(feat_names)}

    # Drop set
    drop_idx = set(name_to_idx[n] for n in DROP_NAMES if n in name_to_idx)
    keep_mask = np.ones(F, dtype=bool)
    for i in drop_idx:
        keep_mask[i] = False
    keep_idx = np.where(keep_mask)[0]

    # Sanitize NaN in X (a few features have NaN in sym=1 - use median imputation)
    nan_mask = ~np.isfinite(X)
    if nan_mask.any():
        n_nan = int(nan_mask.sum())
        print(f"  found {n_nan} NaN values, replacing with 0.0", flush=True)
        X = np.where(np.isfinite(X), X, 0.0)

    # Compute prev_close proxy (in 元): bid_mean / (bid1 + 1), where bid1 is 涨跌幅
    bid1 = X[:, name_to_idx["bid1"]].astype(np.float64)
    bid_mean = X[:, name_to_idx["bid_mean"]].astype(np.float64)
    prev_close = bid_mean / (bid1 + 1.0 + 1e-8)
    prev_close = np.clip(prev_close, 1.0, 1000.0)
    print(f"  prev_close_proxy stats: min={prev_close.min():.2f}, "
          f"max={prev_close.max():.2f}, mean={prev_close.mean():.2f}", flush=True)
    print(f"  prev_close per sym: ", end="")
    sym = d["sym"]
    for s in range(5):
        m = sym == s
        print(f"sym{s}={prev_close[m].mean():.1f} ", end="")
    print(flush=True)

    # midprice + 1.0 (PM divisor) - ~1.0 since midprice is 涨跌幅
    mid = X[:, name_to_idx["midprice1"]].astype(np.float64)
    pm_denom = mid + 1.0 + 1e-8

    # ---- Variant A: PM literal additive ----
    # For amount_delta: cache value is log1p sign-preserving. Dividing by ~1.0
    # is essentially redundant. Apply as PM specified.
    # For spread/diff/cumspread: cache values have -1 offset already (X = raw_pq).
    #   PM's literal proposal: spread_k / (midprice+1). Apply as specified.
    extra_A = np.zeros((n, len(RAW_NORM_TARGETS)), dtype=np.float32)
    for i, name in enumerate(RAW_NORM_TARGETS):
        col = X[:, name_to_idx[name]].astype(np.float64)
        extra_A[:, i] = (col / pm_denom).astype(np.float32)

    # ---- Variant C: smart fix ----
    # amount_delta is log1p in cache; raw amount = exp(log) - 1 ≈ exp(log) for large.
    # We want amount_per_share ~ amount / (price * volume_delta * total_shares),
    # but simplest sym-agnostic: amount_log1p / log(prev_close) gives a stable scale.
    # (Use simple division by prev_close, then re-log1p.)
    # For spread/diff/cumspread: remove -1 offset, multiply by prev_close → 元-scale tick.
    #   (raw + 1.0) * prev_close gives the actual price difference in 元.
    # This makes the tick-floor sym-agnostic (~0.01 元 for all stocks).
    extra_C = np.zeros((n, len(RAW_NORM_TARGETS)), dtype=np.float32)
    for i, name in enumerate(RAW_NORM_TARGETS):
        col = X[:, name_to_idx[name]].astype(np.float64)
        if name == "amount_delta":
            # cache value is log1p(amt). Convert back, divide by prev_close, re-log1p.
            raw_amt = np.expm1(col)
            ratio = raw_amt / prev_close
            extra_C[:, i] = (np.sign(ratio) * np.log1p(np.abs(ratio))).astype(np.float32)
        else:
            # spread/bid_diff/ask_diff/cumspread: undo -1 offset, multiply by prev_close
            extra_C[:, i] = ((col + 1.0) * prev_close).astype(np.float32)

    # Sanity: extra_C variance should be much smaller per sym than baseline
    print("  variant_C abs_mean per sym (first 5 features):", flush=True)
    for s in range(5):
        m = sym == s
        ams = np.abs(extra_C[m, :5]).mean(axis=0)
        print(f"    sym{s}: {[f'{a:.4f}' for a in ams]}", flush=True)

    # Build name lists
    extra_A_names = [f"PMA_{n}" for n in RAW_NORM_TARGETS]
    extra_C_names = [f"SMC_{n}" for n in RAW_NORM_TARGETS]

    # Save: per variant, the keep_idx already excludes T59/Stage5 fails
    base_X = X[:, keep_idx].astype(np.float32, copy=False)
    base_names = [feat_names[i] for i in keep_idx]
    print(f"  baseline kept dims: {base_X.shape[1]}", flush=True)

    # Variant A: baseline + extra_A
    X_A = np.concatenate([base_X, extra_A], axis=1)
    names_A = base_names + extra_A_names

    # Variant B: replace raw with PM ratio - drop 32 raw cols, add 32 ratios
    drop_b_set = set(name_to_idx[n] for n in RAW_NORM_TARGETS)
    keep_b_idx = np.array([i for i in keep_idx if i not in drop_b_set], dtype=np.int64)
    X_B_base = X[:, keep_b_idx].astype(np.float32, copy=False)
    X_B = np.concatenate([X_B_base, extra_A], axis=1)
    names_B = [feat_names[i] for i in keep_b_idx] + extra_A_names

    # Variant C: baseline + extra_C
    X_C = np.concatenate([base_X, extra_C], axis=1)
    names_C = base_names + extra_C_names

    print(f"  shapes: base={base_X.shape}, A={X_A.shape}, B={X_B.shape}, C={X_C.shape}",
          flush=True)

    # Save (only meta differs in feat_names per variant; X is per variant)
    base_meta = {k: d[k] for k in d.files if k != "X"}
    np.savez_compressed(
        os.path.join(DST_CACHE, f"baseline_{split}.npz"),
        X=base_X, **base_meta
    )
    np.savez_compressed(
        os.path.join(DST_CACHE, f"variant_A_{split}.npz"),
        X=X_A, **base_meta
    )
    np.savez_compressed(
        os.path.join(DST_CACHE, f"variant_B_{split}.npz"),
        X=X_B, **base_meta
    )
    np.savez_compressed(
        os.path.join(DST_CACHE, f"variant_C_{split}.npz"),
        X=X_C, **base_meta
    )
    if split == "train":
        with open(os.path.join(DST_CACHE, "baseline_feat_names.txt"), "w") as f:
            f.write("\n".join(base_names) + "\n")
        with open(os.path.join(DST_CACHE, "variant_A_feat_names.txt"), "w") as f:
            f.write("\n".join(names_A) + "\n")
        with open(os.path.join(DST_CACHE, "variant_B_feat_names.txt"), "w") as f:
            f.write("\n".join(names_B) + "\n")
        with open(os.path.join(DST_CACHE, "variant_C_feat_names.txt"), "w") as f:
            f.write("\n".join(names_C) + "\n")
    print(f"  saved in {time.time()-t0:.1f}s", flush=True)


def main():
    for split in ("train", "val", "test"):
        build_split(split)


if __name__ == "__main__":
    main()
