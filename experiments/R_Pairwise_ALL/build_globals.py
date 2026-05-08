"""R_Pairwise_ALL — 7 train-set global mid-price statistics, broadcast as features.

These are constants computed on the *full train split* (dates 0-79), then
broadcast to every row as a constant column. LGB can use them as priors but
they carry no per-row information — so the test risk is purely whether they
help split the tree differently in conjunction with other features.

Stats (over training mp_t, sym-agnostic, date-agnostic):
  g_mp_mean, g_mp_std, g_mp_p10, g_mp_p25, g_mp_p50, g_mp_p75, g_mp_p90

CRITICAL_CONSTRAINTS-safe: data-derived once on train, broadcast to all rows
(test inference: same constants used).

Output: cache/globals_consts.npz {names, values} — a single 7-vector reused for
all train/val/test rows.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

T68_DIR = os.path.join(ROOT, "experiments", "T68_stage5_features", "cache")

GLOBAL_NAMES = [
    "g_mp_mean", "g_mp_std",
    "g_mp_p10", "g_mp_p25", "g_mp_p50", "g_mp_p75", "g_mp_p90",
]


def main():
    train = np.load(os.path.join(T68_DIR, "schemeP_train.npz"))
    mp = train["mp_t"].astype(np.float64)
    print(f"  train rows={len(mp):,}  mp_t range=[{mp.min():.4f}, {mp.max():.4f}]")

    g_mean = float(mp.mean())
    g_std = float(mp.std(ddof=0))
    qs = np.quantile(mp, [0.10, 0.25, 0.50, 0.75, 0.90])
    values = np.array([g_mean, g_std, *qs], dtype=np.float32)
    print("  globals:", dict(zip(GLOBAL_NAMES, values.tolist())))

    out_dir = os.path.join(HERE, "cache")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "globals_consts.npz")
    np.savez(out_path, names=np.array(GLOBAL_NAMES), values=values)
    print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
