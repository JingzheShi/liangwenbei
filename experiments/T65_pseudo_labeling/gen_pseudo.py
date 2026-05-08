"""T65 Step 1+2: generate 5-seed avg pseudo labels with confidence gate.

Loads T59 full_pred_h60_seed{S}.parquet for S in [1, 7, 13, 42, 100], averages
probs across seeds, applies the user-specified gate (max(prob_0, prob_2) > 0.55
AND |prob_0 - prob_2| > 0.10) and writes:

  pseudo_idx_h60.npz   -> {idx, label, prob_max, prob_diff, sym, date}

The `idx` is the row index into schemeN_test.npz (in original order).
Pseudo label = argmax(prob_0, prob_2)  (binary 0 or 2 only — directional pseudo).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
T59_DIR = os.path.join(ROOT, "experiments", "T59_fullsym_train")

SEEDS = [1, 7, 13, 42, 100]
PROB_COLS = ["prob_0", "prob_1", "prob_2"]


def main():
    thr_max = float(os.environ.get("T65_THR_MAX", "0.55"))
    thr_diff = float(os.environ.get("T65_THR_DIFF", "0.10"))
    print(f"=== T65 gen_pseudo  thr_max={thr_max} thr_diff={thr_diff} ===", flush=True)

    base = pd.read_parquet(os.path.join(T59_DIR, f"full_pred_h60_seed{SEEDS[0]}.parquet"))
    n = len(base)
    psum = base[PROB_COLS].to_numpy(np.float64).copy()
    for s in SEEDS[1:]:
        df_s = pd.read_parquet(os.path.join(T59_DIR, f"full_pred_h60_seed{s}.parquet"))
        if len(df_s) != n:
            raise RuntimeError(f"row count mismatch seed={s}")
        if not (df_s["sym"].equals(base["sym"]) and df_s["date"].equals(base["date"]) and df_s["t"].equals(base["t"])):
            raise RuntimeError(f"row order mismatch seed={s}")
        psum += df_s[PROB_COLS].to_numpy(np.float64)
    pavg = psum / float(len(SEEDS))
    p0, p1, p2 = pavg[:, 0], pavg[:, 1], pavg[:, 2]

    # Verify ordering matches schemeN_test.npz
    test_npz = np.load(os.path.join(T59_DIR, "cache", "schemeN_test.npz"))
    sym_te = test_npz["sym"]
    date_te = test_npz["date"]
    t_te = test_npz["t"]
    sess_te = test_npz["sess_idx"]
    sess_str = np.where(sess_te == 0, "am", "pm")

    if not np.array_equal(sym_te, base["sym"].to_numpy(np.int8)):
        raise RuntimeError("sym order mismatch between cache and parquet")
    if not np.array_equal(date_te.astype(np.int16), base["date"].to_numpy(np.int16)):
        raise RuntimeError("date order mismatch")
    if not np.array_equal(t_te.astype(np.int16), base["t"].to_numpy(np.int16)):
        raise RuntimeError("t order mismatch")
    print(f"  cache <-> parquet alignment OK (n={n:,})", flush=True)

    pmax_dir = np.maximum(p0, p2)
    pdiff = np.abs(p0 - p2)
    pseudo_mask = (pmax_dir > thr_max) & (pdiff > thr_diff)
    pseudo_label = np.where(p0 > p2, 0, 2).astype(np.int8)
    n_pseudo = int(pseudo_mask.sum())
    print(f"  pseudo: n={n_pseudo:,} ({100*n_pseudo/n:.2f}% of test)", flush=True)
    pl_filt = pseudo_label[pseudo_mask]
    n_up = int((pl_filt == 2).sum())
    n_dn = int((pl_filt == 0).sum())
    print(f"  by direction: up={n_up:,} dn={n_dn:,}", flush=True)

    # Per-sym
    psyms = sym_te[pseudo_mask]
    per_sym = {}
    for k in range(5):
        m = (psyms == k)
        per_sym[int(k)] = {
            "n": int(m.sum()),
            "up": int(((pl_filt == 2) & m).sum()),
            "dn": int(((pl_filt == 0) & m).sum()),
        }

    # Indices into the schemeN_test.npz (which equals the parquet order)
    pseudo_idx = np.where(pseudo_mask)[0].astype(np.int64)

    out_path = os.path.join(HERE, "pseudo_idx_h60.npz")
    np.savez_compressed(
        out_path,
        idx=pseudo_idx,
        label=pl_filt.astype(np.int8),
        prob_max=pmax_dir[pseudo_mask].astype(np.float32),
        prob_diff=pdiff[pseudo_mask].astype(np.float32),
        prob_0=p0[pseudo_mask].astype(np.float32),
        prob_1=p1[pseudo_mask].astype(np.float32),
        prob_2=p2[pseudo_mask].astype(np.float32),
        sym=sym_te[pseudo_mask].astype(np.int8),
        date=date_te[pseudo_mask].astype(np.int16),
        thr_max=np.array([thr_max], dtype=np.float32),
        thr_diff=np.array([thr_diff], dtype=np.float32),
    )
    print(f"  saved -> {out_path}", flush=True)

    summary = {
        "thr_max": thr_max,
        "thr_diff": thr_diff,
        "n_test": n,
        "n_pseudo": n_pseudo,
        "frac_pseudo": float(n_pseudo) / n,
        "n_up": n_up,
        "n_dn": n_dn,
        "per_sym": per_sym,
    }
    sum_path = os.path.join(HERE, "pseudo_summary.json")
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  summary -> {sum_path}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
