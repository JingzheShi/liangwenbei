"""T28: Train 4 additional aug_a final models (seeds 1, 7, 13, 100) for iter_005b.

Wraps T27/train_final_aug_a.py — the heavy lifting (data loading, aug, GPU
LightGBM) lives there. We only re-target seeds and pin --out-dir to T27 so the
build script (which scans T27/final_model_h60_aug_a_seed{S}.txt) can find them.

seed=42 was already produced by T27; this run only adds {1, 7, 13, 100}.
"""
from __future__ import annotations

import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
T27_DIR = os.path.join(ROOT, "experiments", "T27_iter005")
T27_SCRIPT = os.path.join(T27_DIR, "train_final_aug_a.py")

# Default new seeds (complement of seed=42 which T27 already trained).
SEEDS = "1,7,13,100"


def main():
    sys.argv = [
        T27_SCRIPT,
        "--horizons", "60",
        "--variant", "aug_a",
        "--seeds", SEEDS,
        "--out-dir", T27_DIR,  # build_iter_005b.py expects models in T27/
    ]
    runpy.run_path(T27_SCRIPT, run_name="__main__")


if __name__ == "__main__":
    main()
