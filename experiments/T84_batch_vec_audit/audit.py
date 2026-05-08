"""T84 — Audit batch-vectorized inference for cross-window data leakage.

Tests `submission/iter_013_time_features/fast_features_batch.py`.

Five empirical invariance checks plus a static-axis review (manual). Any
deviation from byte-for-byte equality (subject to a tiny tolerance for
numpy reduction-order non-determinism) means the batch dimension leaks
information across windows.

Checks:
  1. **Single-row vs batched-row** — for each of the N windows, run the
     batch function with N=1 (just that window) vs the full batch and
     compare row i.
  2. **Shuffle invariance** — random-permute window order, check that
     each window's features survive the permutation unchanged.
  3. **Replication** — feed N copies of the same window, all rows must
     be byte-identical.
  4. **Reverse ordering (future-then-past)** — reverse the windows in
     time order, results should equal time-ordered (just permuted).
  5. **Mixed-symbol batch** — pull windows from different syms / dates
     into one batch, compare each row to its single-row reference.
"""
from __future__ import annotations

import sys
import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/root/projects/liangwenbei_workdir")
SUBM = REPO / "submission/iter_013_time_features"
sys.path.insert(0, str(SUBM))

from fast_features_batch import compute_batch_features, all_feature_names  # noqa
from fast_features import (
    compute_window_features as compute_window_features_single,  # noqa
    all_feature_names as single_feature_names,  # noqa
)


# Columns the batch feature function reads (collected from constants in
# fast_features_batch.py).
RAW_COLS = []
# T3 LOB / WMP / OFI inputs
for k in range(1, 11):
    RAW_COLS += [f"bid{k}", f"ask{k}", f"bsize{k}", f"asize{k}"]
# T3 EWMA intensities
RAW_COLS += ["lb_intst", "la_intst", "mb_intst", "ma_intst", "cb_intst", "ca_intst"]
# DUAL_Z columns
RAW_COLS += [
    "spread1", "spread5", "spread10", "cumspread",
    "bid_mean", "ask_mean",
    "midprice1", "midprice2", "midprice5", "midprice10",
    "bsize_mean", "totalbsize",
    "asize_mean", "totalasize",
    "volume_delta", "amount_delta",
    "imbalance",
    "lb_acc", "la_acc", "mb_acc", "ma_acc", "cb_acc", "ca_acc",
    "bid_diff1", "ask_diff1", "bid_diff5", "ask_diff5",
    "close",
]
RAW_COLS = list(dict.fromkeys(RAW_COLS))  # dedupe preserving order


def load_window(df: pd.DataFrame, end_idx: int, T: int = 100) -> np.ndarray:
    """Return (T, K) float64 window ending (inclusive) at end_idx."""
    seg = df.iloc[end_idx - T + 1 : end_idx + 1]
    arr = np.zeros((T, len(RAW_COLS)), dtype=np.float64)
    for i, c in enumerate(RAW_COLS):
        arr[:, i] = seg[c].to_numpy(dtype=np.float64)
    return arr


def collect_windows(parquet_paths, end_indices, T=100):
    """Return X3d (N, T, K), col_idx."""
    col_idx = {c: i for i, c in enumerate(RAW_COLS)}
    windows = []
    metadata = []
    for path in parquet_paths:
        df = pd.read_parquet(path)
        for ei in end_indices:
            if ei >= T - 1 and ei < len(df):
                w = load_window(df, ei, T)
                windows.append(w)
                metadata.append((path.name, ei))
    X3d = np.stack(windows, axis=0)
    return X3d, col_idx, metadata


def max_abs_diff(a, b):
    return float(np.max(np.abs(a - b)))


def main():
    out_dir = REPO / "experiments/T84_batch_vec_audit"
    out_dir.mkdir(exist_ok=True, parents=True)

    rng = np.random.default_rng(20260508)

    # === Build a heterogeneous batch of windows ===
    parquet_paths = [
        REPO / "data/snapshot_sym0_date0_am.parquet",
        REPO / "data/snapshot_sym0_date0_pm.parquet",
        REPO / "data/snapshot_sym0_date100_am.parquet",
        REPO / "data/snapshot_sym0_date101_pm.parquet",
    ]
    parquet_paths = [p for p in parquet_paths if p.exists()]
    end_indices = [99, 200, 500, 999, 1500, 1900]
    X3d, col_idx, metadata = collect_windows(parquet_paths, end_indices)
    N = X3d.shape[0]
    print(f"[setup] X3d shape = {X3d.shape}  (N={N} windows from {len(parquet_paths)} parquets)")

    # Replace NaNs in raw features (some columns may have NaNs at session start)
    X3d = np.where(np.isfinite(X3d), X3d, 0.0)

    feat_names = all_feature_names()
    F = len(feat_names)
    print(f"[setup] Feature count F = {F}")

    report = {
        "windows": N,
        "features": F,
        "checks": {},
        "axis_review": {},
    }

    # ---------------------------------------------------------------
    # Check 1: Single-row vs batched
    # ---------------------------------------------------------------
    print("\n[check 1] single-row vs batched ...")
    feats_batch = compute_batch_features(X3d, col_idx)  # (N, F)
    feats_single = np.stack(
        [compute_batch_features(X3d[i:i+1], col_idx)[0] for i in range(N)],
        axis=0,
    )
    diff1 = np.abs(feats_batch - feats_single)
    max1 = float(diff1.max())
    nz_rows = int((diff1.max(axis=1) > 0).sum())
    nz_cols = int((diff1.max(axis=0) > 0).sum())
    nz_idx = np.argwhere(diff1 > 1e-15)
    sample = [
        (int(r), int(c), feat_names[c], float(feats_batch[r, c]), float(feats_single[r, c]))
        for r, c in nz_idx[:10]
    ]
    print(f"  max |Δ| = {max1:.3e}   rows with diff: {nz_rows}/{N}   cols with diff: {nz_cols}/{F}")
    if sample:
        print("  sample diffs:")
        for r, c, name, v_b, v_s in sample[:5]:
            print(f"    row={r}  col={c}({name})   batch={v_b:.6e}  single={v_s:.6e}  Δ={abs(v_b-v_s):.3e}")
    report["checks"]["1_single_vs_batched"] = {
        "max_abs_diff": max1,
        "rows_with_diff": nz_rows,
        "cols_with_diff": nz_cols,
        "leakage": bool(max1 > 1e-9),
    }

    # ---------------------------------------------------------------
    # Check 2: Shuffle invariance
    # ---------------------------------------------------------------
    print("\n[check 2] shuffle invariance ...")
    perm = rng.permutation(N)
    inv_perm = np.argsort(perm)
    X3d_shuf = X3d[perm]
    feats_shuf = compute_batch_features(X3d_shuf, col_idx)
    feats_unshuf = feats_shuf[inv_perm]
    diff2 = np.abs(feats_unshuf - feats_batch)
    max2 = float(diff2.max())
    print(f"  max |Δ| = {max2:.3e}  (perm = {perm.tolist()})")
    report["checks"]["2_shuffle_invariance"] = {
        "max_abs_diff": max2,
        "perm": perm.tolist(),
        "leakage": bool(max2 > 1e-9),
    }

    # ---------------------------------------------------------------
    # Check 3: Replication (N copies of one window)
    # ---------------------------------------------------------------
    print("\n[check 3] replication (N copies of same window) ...")
    Nrep = 8
    one = X3d[0:1]
    rep = np.repeat(one, Nrep, axis=0)
    feats_rep = compute_batch_features(rep, col_idx)
    diff3 = np.abs(feats_rep - feats_rep[0:1])
    max3 = float(diff3.max())
    print(f"  max |Δ| across replicated rows = {max3:.3e}")
    report["checks"]["3_replication"] = {
        "max_abs_diff": max3,
        "n_copies": Nrep,
        "leakage": bool(max3 > 1e-9),
    }

    # ---------------------------------------------------------------
    # Check 4: Reverse ordering (future-then-past)
    # ---------------------------------------------------------------
    print("\n[check 4] reverse ordering ...")
    rev_perm = np.arange(N)[::-1].copy()
    X3d_rev = X3d[rev_perm]
    feats_rev = compute_batch_features(X3d_rev, col_idx)
    feats_unrev = feats_rev[rev_perm]
    diff4 = np.abs(feats_unrev - feats_batch)
    max4 = float(diff4.max())
    print(f"  max |Δ| = {max4:.3e}")
    report["checks"]["4_reverse_ordering"] = {
        "max_abs_diff": max4,
        "leakage": bool(max4 > 1e-9),
    }

    # ---------------------------------------------------------------
    # Check 5: Mixed batch — interleave windows of different sources
    # ---------------------------------------------------------------
    print("\n[check 5] mixed-source interleave ...")
    # batch is already heterogeneous (multiple parquets/end_indices). For
    # this check we compare every row against its individual single-window
    # batch invocation, but in a different ordering (interleave).
    inter_idx = np.argsort(np.array([(i % 2) * N + i for i in range(N)]))
    X3d_inter = X3d[inter_idx]
    feats_inter = compute_batch_features(X3d_inter, col_idx)
    inv_inter = np.argsort(inter_idx)
    feats_inter_unsort = feats_inter[inv_inter]
    diff5 = np.abs(feats_inter_unsort - feats_batch)
    max5 = float(diff5.max())
    print(f"  max |Δ| = {max5:.3e}")
    report["checks"]["5_mixed_interleave"] = {
        "max_abs_diff": max5,
        "leakage": bool(max5 > 1e-9),
    }

    # ---------------------------------------------------------------
    # Check 6: Adversarial — surrounding window with wildly different scale
    # ---------------------------------------------------------------
    print("\n[check 6] adversarial neighbours (huge-magnitude pair next to target) ...")
    target = X3d[0:1].copy()
    # Build a batch of [target, garbage_huge, target, garbage_tiny, target, garbage_neg]
    huge = X3d[0:1].copy() * 1e6
    tiny = X3d[0:1].copy() * 1e-6
    negs = -X3d[0:1].copy() * 100.0
    adv = np.concatenate([target, huge, target, tiny, target, negs], axis=0)
    feats_adv = compute_batch_features(adv, col_idx)
    feats_target_alone = compute_batch_features(target, col_idx)[0]
    # rows 0, 2, 4 should all equal feats_target_alone
    diffs6 = []
    for r in (0, 2, 4):
        diffs6.append(float(np.max(np.abs(feats_adv[r] - feats_target_alone))))
    max6 = float(max(diffs6))
    print(f"  max |Δ| (rows 0/2/4 vs alone) = {max6:.3e}  individual: {diffs6}")
    report["checks"]["6_adversarial_neighbours"] = {
        "max_abs_diff": max6,
        "per_target_row_diff": diffs6,
        "leakage": bool(max6 > 1e-9),
    }

    # ---------------------------------------------------------------
    # Check 7: Per-feature leakage attribution if any check fails
    # ---------------------------------------------------------------
    any_leak = any(c.get("leakage", False) for c in report["checks"].values())
    if any_leak:
        print("\n[check 7] per-feature leakage attribution (using check 1 deltas)")
        col_max = diff1.max(axis=0)
        leak_cols = np.argsort(col_max)[::-1][:20]
        report["leak_top_features"] = [
            {"name": feat_names[c], "max_abs_diff": float(col_max[c])}
            for c in leak_cols
            if col_max[c] > 0
        ]
        for entry in report["leak_top_features"]:
            print(f"    {entry['name']:40s}  max|Δ|={entry['max_abs_diff']:.3e}")
    else:
        report["leak_top_features"] = []

    # ---------------------------------------------------------------
    # Check 8: Truly heterogeneous batch — different syms (sym 0..4)
    # ---------------------------------------------------------------
    print("\n[check 8] truly heterogeneous batch (sym 0..4) ...")
    multi_paths = []
    for s in range(5):
        for d in (0, 50, 100):
            for sess in ("am", "pm"):
                p = REPO / f"data/snapshot_sym{s}_date{d}_{sess}.parquet"
                if p.exists():
                    multi_paths.append(p)
    multi_paths = multi_paths[:12]  # keep batch reasonable
    X3d_h, _, _ = collect_windows(multi_paths, [99, 500, 1500])
    X3d_h = np.where(np.isfinite(X3d_h), X3d_h, 0.0)
    Nh = X3d_h.shape[0]
    feats_h_batch = compute_batch_features(X3d_h, col_idx)
    feats_h_single = np.stack(
        [compute_batch_features(X3d_h[i:i+1], col_idx)[0] for i in range(Nh)],
        axis=0,
    )
    diff8 = np.abs(feats_h_batch - feats_h_single)
    max8 = float(diff8.max())
    print(f"  Nh={Nh}  max |Δ| = {max8:.3e}")
    report["checks"]["8_heterogeneous_syms"] = {
        "max_abs_diff": max8,
        "n_windows": Nh,
        "n_paths": len(multi_paths),
        "leakage": bool(max8 > 1e-9),
    }

    # ---------------------------------------------------------------
    # Check 9: Cross-implementation — batch vs fast_features.py single
    # ---------------------------------------------------------------
    print("\n[check 9] batch[:, :196] vs fast_features.py single-window reference ...")
    # Build df_arr-style dict from each window for the single-window function.
    feats_ref = []
    for i in range(min(N, 12)):
        w = X3d[i]  # (T, K)
        df_arr = {RAW_COLS[c]: w[:, c] for c in range(len(RAW_COLS))}
        # midprice alias used by fast_features.py
        df_arr["midprice"] = w[:, RAW_COLS.index("midprice1")].copy()
        feats_ref.append(compute_window_features_single(df_arr))
    feats_ref = np.stack(feats_ref, axis=0)  # (n, 196)
    feats_ref_names = single_feature_names()
    assert feats_ref_names == feat_names[:196], "name mismatch between single and batch"
    # Compare batch[i, :196] vs feats_ref[i]
    feats_batch_first196 = feats_batch[:feats_ref.shape[0], :196]
    diff9 = np.abs(feats_batch_first196 - feats_ref)
    max9 = float(diff9.max())
    # tolerance 1e-3 per docstring; actual leakage would manifest as much larger
    print(f"  max |Δ| (batch vs single-window ref, first 196 features) = {max9:.3e}")
    report["checks"]["9_batch_vs_single_window_ref"] = {
        "max_abs_diff": max9,
        "n_compared": int(feats_ref.shape[0]),
        "tolerance_doc": 1e-3,
        "leakage": bool(max9 > 1e-3),
    }

    # ---------------------------------------------------------------
    # Static axis review summary
    # ---------------------------------------------------------------
    report["axis_review"] = {
        "scanned_file": str(SUBM / "fast_features_batch.py"),
        "summary": (
            "All numpy reductions in fast_features_batch.py use axis=-1 (time) "
            "or axis=1 within (N, T, K) slices. lfilter calls use axis=-1. "
            "EWMA initial conditions zi are per-window: (1-a)*x[:, :1]. "
            "No axis=0 reductions; no global stats; no cross-row indexing."
        ),
    }

    # final summary
    print("\n=== SUMMARY ===")
    for name, c in report["checks"].items():
        flag = "LEAK" if c["leakage"] else "OK"
        print(f"  {name:35s}  max|Δ|={c['max_abs_diff']:.3e}   {flag}")
    leakage_found = any_leak
    severity = max(c["max_abs_diff"] for c in report["checks"].values())
    fix_needed = leakage_found
    report["verdict"] = {
        "leakage_found": leakage_found,
        "max_abs_diff_overall": severity,
        "fix_needed": fix_needed,
    }
    print(f"\n  leakage_found={leakage_found}   severity={severity:.3e}   fix_needed={fix_needed}")

    out_path = out_dir / "audit_results.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n  written: {out_path}")


if __name__ == "__main__":
    main()
