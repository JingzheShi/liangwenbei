"""T85 Feature leakage audit.

GOAL: For every feature in compute_batch_features (T3+S1+S2+S3+S5),
verify that the *last-tick* feature value depends ONLY on the past
[t-W+1 .. t] of the 100-tick window — never on indices > t.

STRATEGY:
  1. Static review (already done): every reduction is over [:, -W:],
     all lag-shifts go past->present, EWMA is causal (lfilter axis=-1).

  2. Empirical "perturb future" test:
       - Build a baseline 100-tick window with non-trivial structure.
       - For each tick index p in [0, 99]: clone the baseline, perturb
         only the row at p, recompute the feature vector for the
         perturbed window, and record which features change.
       - For a *causal* feature with sub-window W:
           * Perturbing p in (99-W, 99] (last W rows including last) → SHOULD change
           * Perturbing p in [0, 99-W] (older than last W rows) → MAY change (history)
           * Perturbing p > 99: not applicable (window ends at 99)
       - Since the window only spans [0, 99], "future leak" within a single
         window is impossible at the WINDOW level. The real concern is:
         could a feature value at last tick p=99 be reading from p=99
         only — or actually from p<99 but in a way that reflects forward
         shift? E.g., feature stored at "row 99" but actually = stat over
         [99..99+W-1] would be impossible to compute since rows > 99 don't
         exist; it would either error or use rows [99-W+1..99] anyway.

  3. Build / Cache leakage test:
       - In build_features.py the i-th row of cache stores features
         for absolute tick t = i + 99 (last tick of window A[i:i+100]).
       - Verify: the cache row labeled with t = T0 has features that
         use ONLY parquet rows [T0-99 .. T0]. Equivalently, a fresh
         compute of compute_batch_features on parquet rows [T0-99..T0]
         should reproduce the cached row exactly.

  4. Per-feature "shift-the-cliff" test:
       - Create a window with a step at tick s: rows < s are 0, rows >= s
         are 1.  Move s.  Track the feature value at last tick.
       - For a causal feature with sub-window W:
           * If s <= 99 - W + 1 → the entire last-W window is 1s.
             (so the feature = whatever 1s give)
           * If s = 99 → only the last tick is 1.
             (so the feature = transition of "all 0 then 1 spike at end")
       - This lets us check if the feature responds to the *recent* part
         of the window, not the early part.

OUTPUT:
  audit_results.json — machine-readable per-feature pass/fail
  REPORT.md          — written separately by the report step.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SUBDIR = os.path.join(ROOT, "submission", "iter_013_time_features")
sys.path.insert(0, SUBDIR)

import fast_features_batch as ffb  # noqa: E402

WINDOW = 100
RAW_COLS: List[str] = (
    ["open", "high", "low", "close", "volume_delta", "amount_delta"]
    + [f"bid{k}" for k in range(1, 11)]
    + [f"bsize{k}" for k in range(1, 11)]
    + [f"ask{k}" for k in range(1, 11)]
    + [f"asize{k}" for k in range(1, 11)]
    + ["avgbid", "avgask", "totalbsize", "totalasize",
       "lb_intst", "la_intst", "mb_intst", "ma_intst", "cb_intst", "ca_intst",
       "lb_ind", "la_ind", "mb_ind", "ma_ind", "cb_ind", "ca_ind",
       "lb_acc", "la_acc", "mb_acc", "ma_acc", "cb_acc", "ca_acc"]
    + [f"midprice{k}" for k in range(1, 11)]
    + [f"spread{k}" for k in range(1, 11)]
    + [f"bid_diff{k}" for k in range(1, 11)]
    + [f"ask_diff{k}" for k in range(1, 11)]
    + ["bid_mean", "ask_mean", "bsize_mean", "asize_mean", "cumspread", "imbalance"]
    + [f"bid_rate{k}" for k in range(1, 11)]
    + [f"ask_rate{k}" for k in range(1, 11)]
    + [f"bsize_rate{k}" for k in range(1, 11)]
    + [f"asize_rate{k}" for k in range(1, 11)]
)
assert len(RAW_COLS) == 154
COL_IDX = {c: i for i, c in enumerate(RAW_COLS)}
COL_IDX["midprice"] = COL_IDX["midprice1"]

FEAT_NAMES = ffb.all_feature_names()
NF = len(FEAT_NAMES)
print(f"[audit] feature count: {NF}", flush=True)


def _load_real_window(parquet_path: str, end_idx: int) -> np.ndarray:
    """Load 100-tick window ending at row end_idx (last row included)."""
    df = pd.read_parquet(parquet_path)
    arr = df[RAW_COLS].to_numpy(dtype=np.float64, copy=True)
    return arr[end_idx - WINDOW + 1: end_idx + 1].copy()  # (100, K)


def _make_baseline_window() -> np.ndarray:
    """Pick a real 100-tick window with non-degenerate structure."""
    p = os.path.join(ROOT, "data", "snapshot_sym0_date0_am.parquet")
    return _load_real_window(p, 500)  # arbitrary mid-session tick


def _compute_one(window_2d: np.ndarray) -> np.ndarray:
    """Run compute_batch_features on a single (100, K) window. Returns (NF,)."""
    X3d = window_2d[None, :, :]  # (1, 100, K)
    out = ffb.compute_batch_features(X3d, COL_IDX)
    return out[0]  # (NF,)


# =============================================================================
# Test 1: Future-vs-past tick perturbation
# For each tick p in [0, 99], perturb that row by adding NOISE_SCALE*noise to
# every column, recompute features, count which features differ.
# Result table: feature_idx -> earliest_changed_p (smallest p at which that
# feature first responds to a perturbation).
# Causal feature with sub-window W (e.g., W=20) should:
#   - Respond to perturbations at p in [99-W+1, 99]
#   - Be invariant to perturbations at p in [0, 99-W]   (idle past)
# Maximum-W feature (e.g., W=100) responds to all p (full window).
# So earliest_responsive_p tells us the *effective sub-window depth*.
# =============================================================================

def test1_perturb_each_tick(window_2d: np.ndarray, noise_scale: float = 0.01,
                              tol: float = 1e-9) -> Dict:
    """Perturb each row p; record per-feature earliest p that changes the feature."""
    base_feats = _compute_one(window_2d)
    K = window_2d.shape[1]
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(K).astype(np.float64) * noise_scale
    earliest = np.full(NF, -1, dtype=np.int64)  # -1 = never responsive
    n_responsive_per_p = np.zeros(WINDOW, dtype=np.int64)
    for p in range(WINDOW):
        w2 = window_2d.copy()
        w2[p, :] = w2[p, :] + noise * np.maximum(np.abs(w2[p, :]), 1.0)
        feats_p = _compute_one(w2)
        diff = np.abs(feats_p - base_feats)
        changed = diff > tol
        n_responsive_per_p[p] = int(changed.sum())
        for f_idx in range(NF):
            if changed[f_idx] and earliest[f_idx] < 0:
                earliest[f_idx] = p
    return {
        "earliest_p_per_feature": earliest.tolist(),
        "n_responsive_per_p": n_responsive_per_p.tolist(),
    }


# =============================================================================
# Test 2: Step-signal "shift the cliff"
# Construct a synthetic 100-tick window where a "step" at tick s means
# rows [0..s-1] = LOW values, rows [s..99] = HIGH values.
# For a causal feature at last tick (p=99) with sub-window W:
#   - When s <= 100 - W (step entirely in the past part): last W ticks all HIGH
#   - When 100 - W < s <= 99: step is INSIDE the last W ticks
#   - When s = 100: no step (all LOW)
# We expect the feature value to plateau when s <= 100 - W.
# =============================================================================

def test2_step_signal(noise_scale: float = 0.0) -> Dict:
    """Sweep step position s and record feature values."""
    K = len(RAW_COLS)
    LOW = np.zeros(K, dtype=np.float64)
    HIGH = np.zeros(K, dtype=np.float64)
    # Set sane prices/sizes so EWMA & rolling sums don't div by zero everywhere.
    for c in RAW_COLS:
        i = COL_IDX[c]
        if c.startswith("bid") and not c.startswith("bid_diff") and not c.startswith("bid_mean") and not c.startswith("bid_rate") and not c.startswith("bsize"):
            try:
                int(c.replace("bid", ""))  # bid1..bid10
                LOW[i] = 1.000
                HIGH[i] = 1.005
            except ValueError:
                pass
        if c.startswith("ask") and not c.startswith("ask_diff") and not c.startswith("ask_mean") and not c.startswith("ask_rate") and not c.startswith("asize"):
            try:
                int(c.replace("ask", ""))
                LOW[i] = 1.001
                HIGH[i] = 1.006
            except ValueError:
                pass
        if c.startswith("bsize") or c.startswith("asize"):
            try:
                int(c[5:])
                LOW[i] = 1000.0
                HIGH[i] = 2000.0
            except ValueError:
                pass
        if c.startswith("midprice"):
            LOW[i] = 1.0005
            HIGH[i] = 1.0055
        if c.startswith("spread"):
            LOW[i] = 0.001
            HIGH[i] = 0.001
    LOW[COL_IDX["amount_delta"]] = 100.0
    HIGH[COL_IDX["amount_delta"]] = 200.0
    LOW[COL_IDX["volume_delta"]] = 100.0
    HIGH[COL_IDX["volume_delta"]] = 200.0
    LOW[COL_IDX["close"]] = 1.0005
    HIGH[COL_IDX["close"]] = 1.0055
    for c in ("lb_intst", "la_intst", "mb_intst", "ma_intst", "cb_intst", "ca_intst"):
        LOW[COL_IDX[c]] = 1.0
        HIGH[COL_IDX[c]] = 5.0

    feat_traces = np.zeros((WINDOW + 1, NF), dtype=np.float64)
    # s in [0, 100]: s=0 means all HIGH; s=100 means all LOW.
    # but to stress-test, also use s in [0..100]
    for s in range(WINDOW + 1):
        w2 = np.empty((WINDOW, K), dtype=np.float64)
        if s > 0:
            w2[:s, :] = LOW[None, :]
        if s < WINDOW:
            w2[s:, :] = HIGH[None, :]
        feat_traces[s] = _compute_one(w2)
    return {"feat_traces_shape": list(feat_traces.shape),
            "feat_traces": feat_traces.tolist()}


# =============================================================================
# Test 3: Reverse window
# Time-reverse a real window: w_rev[i] = w[99-i]. For a causal feature
# at last tick of `w`, perturbing only the row at position 99 should:
#   - In w: change the feature (last tick).
#   - In w_rev: now appears at position 0 of w_rev. If the feature at
#     last tick of w_rev (which is row 99 = original row 0) is causal,
#     it should NOT depend on row 0 (deep history) for short-W features.
# This is a sanity stand-in: causal feature on reversed window = same
# math but with time direction flipped, so a perturbation at the end of
# the original (= start of reversed) should NOT propagate into short-W
# stats of the reversed window's last tick.
# =============================================================================

def test3_reverse_window(window_2d: np.ndarray, noise_scale: float = 0.01,
                          tol: float = 1e-9) -> Dict:
    K = window_2d.shape[1]
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(K).astype(np.float64) * noise_scale
    w = window_2d.copy()
    w_rev = w[::-1, :].copy()

    base_w = _compute_one(w)
    base_rev = _compute_one(w_rev)

    # Perturb row 99 of original = first row of reversed
    w_pert = w.copy()
    w_pert[99, :] = w_pert[99, :] + noise * np.maximum(np.abs(w_pert[99, :]), 1.0)
    w_rev_pert = w_pert[::-1, :].copy()  # perturbation now at row 0
    feats_w_pert = _compute_one(w_pert)
    feats_rev_pert = _compute_one(w_rev_pert)

    diff_orig = np.abs(feats_w_pert - base_w)
    diff_rev = np.abs(feats_rev_pert - base_rev)
    changed_orig = (diff_orig > tol).sum()
    changed_rev = (diff_rev > tol).sum()

    # Causal: on `w`, perturbing row 99 should change MANY features (last-tick
    # is part of every -W: slice). On `w_rev`, perturbing row 0 (deep history
    # for the reversed window) should change FEWER features than full
    # — only the W=100 (full-window) features should change.
    return {
        "n_changed_perturb_last_orig": int(changed_orig),
        "n_changed_perturb_first_rev": int(changed_rev),
        "diff_orig_max": float(diff_orig.max()),
        "diff_rev_max": float(diff_rev.max()),
    }


# =============================================================================
# Test 4: Cache vs fresh-compute
# Pick rows from the existing cache and recompute features from the parquet.
# If cache row at (sym, date, sess, t) was built with future leak, the values
# would diverge from a fresh call to compute_batch_features on parquet rows
# [t-99..t].
# =============================================================================

def test4_cache_vs_fresh(n_check: int = 32) -> Dict:
    """Verify cache row at (sym, date, sess, t) == fresh compute on rows [t-99..t]."""
    cache_path = os.path.join(ROOT, "experiments", "T68_stage5_features", "cache",
                              "schemeP_train.npz")
    if not os.path.isfile(cache_path):
        return {"skipped": True, "reason": f"cache not found at {cache_path}"}
    z = np.load(cache_path)
    X_cache = z["X"]  # (n_total, 370 = 154 raw + 216 extras)
    sym = z["sym"]
    date = z["date"]
    sess = z["sess_idx"]
    t_arr = z["t"]
    n_total = X_cache.shape[0]
    rng = np.random.default_rng(7)
    pick = rng.choice(n_total, size=min(n_check, n_total), replace=False)
    SESS_NAMES = {0: "am", 1: "pm"}

    diffs: List[Dict] = []
    max_abs = 0.0
    for idx in pick:
        s_int = int(sym[idx]); d_int = int(date[idx])
        sn = SESS_NAMES[int(sess[idx])]; t_int = int(t_arr[idx])
        path = os.path.join(ROOT, "data", f"snapshot_sym{s_int}_date{d_int}_{sn}.parquet")
        if not os.path.isfile(path):
            continue
        win = _load_real_window(path, t_int)
        feats = _compute_one(win)
        # cache row layout: 154 raw_last + 216 extras (T3+S1+S2+S3+S5)
        cache_extras = X_cache[idx, 154:154 + NF].astype(np.float64)
        d_vec = np.abs(feats - cache_extras)
        max_abs = max(max_abs, float(d_vec.max()))
        diffs.append({"idx": int(idx), "sym": s_int, "date": d_int, "sess": sn, "t": t_int,
                       "max_abs_diff": float(d_vec.max()),
                       "argmax_feature": FEAT_NAMES[int(np.argmax(d_vec))]})
    return {
        "n_checked": len(diffs),
        "global_max_abs_diff": max_abs,
        "samples": diffs[:8],  # only keep first 8 for brevity
    }


# =============================================================================
# Test 5: "Front shift" — what if I duplicate row 99 onto rows 80..99?
# A causal feature with W=20 should reflect "all-row-99" since last 20 are
# constant.  A feature using rows 0..19 (forward leak) would still see the
# original first 20 rows.
# =============================================================================

def test5_overwrite_back20_with_row99(window_2d: np.ndarray) -> Dict:
    base = _compute_one(window_2d)
    w2 = window_2d.copy()
    w2[80:, :] = w2[99, :][None, :]  # last 20 = constant copies of row 99
    feats = _compute_one(w2)
    diff = np.abs(feats - base)
    return {"n_changed": int((diff > 1e-9).sum()),
            "max_diff": float(diff.max())}


# =============================================================================
# Test 6: Replace last 20 ticks with NaN/garbage.  Causal short-W (<=20)
# features should change drastically; very-long-W features (W=100) change
# moderately.  Crucially: features with NO recent dependency should NOT
# change.  If a feature is *forward-looking* (would actually need rows 100+),
# then since those rows don't exist, it must fallback to using past data
# anyway, making this distinction moot — the only way to fail this audit is
# if the feature value at "time 99 of window" actually = some statistic of
# rows [99..118] (impossible since rows >99 don't exist).
# =============================================================================

def test6_replace_back20_with_garbage(window_2d: np.ndarray) -> Dict:
    base = _compute_one(window_2d)
    w2 = window_2d.copy()
    rng = np.random.default_rng(1234)
    w2[80:, :] = rng.standard_normal((20, w2.shape[1])) * 1e3
    feats = _compute_one(w2)
    diff = np.abs(feats - base)
    return {"n_changed_big": int((diff > 1.0).sum()),
            "n_changed_any": int((diff > 1e-9).sum()),
            "max_diff": float(diff.max())}


# =============================================================================
# Test 7: Replace FIRST tick (row 0) with garbage. ONLY features that depend
# on the deep history (W=100, EWMA initial-condition tail) should change.
# Sub-window-W features (W < 100, W < t-something) should be invariant.
# =============================================================================

def test7_perturb_row0_only(window_2d: np.ndarray, noise_scale: float = 0.01,
                              tol: float = 1e-9) -> Dict:
    base = _compute_one(window_2d)
    K = window_2d.shape[1]
    rng = np.random.default_rng(2025)
    noise = rng.standard_normal(K).astype(np.float64) * noise_scale
    w2 = window_2d.copy()
    w2[0, :] = w2[0, :] + noise * np.maximum(np.abs(w2[0, :]), 1.0)
    feats = _compute_one(w2)
    diff = np.abs(feats - base)
    changed = diff > tol
    n_changed = int(changed.sum())
    by_feature = {FEAT_NAMES[i]: float(diff[i]) for i in np.where(changed)[0]}
    return {
        "n_changed": n_changed,
        "max_diff": float(diff.max()),
        "changed_features": by_feature,  # which features depend on row 0
    }


# =============================================================================
# Driver
# =============================================================================

def main() -> int:
    out: Dict = {"feat_names": FEAT_NAMES}

    print("[1/7] perturb each tick & locate earliest responsive p ...", flush=True)
    t0 = time.time()
    base_w = _make_baseline_window()
    out["test1"] = test1_perturb_each_tick(base_w)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    print("[2/7] step-signal sweep over s in [0,100] ...", flush=True)
    t0 = time.time()
    out["test2"] = test2_step_signal()
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    print("[3/7] reverse-window sanity ...", flush=True)
    t0 = time.time()
    out["test3"] = test3_reverse_window(base_w)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    print("[4/7] cache vs fresh compute on real cached rows ...", flush=True)
    t0 = time.time()
    out["test4"] = test4_cache_vs_fresh(n_check=32)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    print("[5/7] overwrite back 20 with row99 ...", flush=True)
    t0 = time.time()
    out["test5"] = test5_overwrite_back20_with_row99(base_w)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    print("[6/7] replace back 20 with garbage ...", flush=True)
    t0 = time.time()
    out["test6"] = test6_replace_back20_with_garbage(base_w)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    print("[7/7] perturb only row 0 ...", flush=True)
    t0 = time.time()
    out["test7"] = test7_perturb_row0_only(base_w)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    out_path = os.path.join(HERE, "audit_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"wrote {out_path} ({os.path.getsize(out_path)/1024:.1f} KB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
