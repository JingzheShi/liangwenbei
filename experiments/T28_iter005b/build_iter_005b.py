"""T28: thin wrapper around T27/build_iter_005b.py for the 5-seed ensemble bundle.

Lives in T28 only for documentation / locality of the task. The heavy lifting
(staging files, threshold lookup from sweep_5seed_ensemble_results.json,
sanity test, zip) is in T27.
"""
from __future__ import annotations

import os
import runpy
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
T27_BUILD = os.path.join(ROOT, "experiments", "T27_iter005", "build_iter_005b.py")


if __name__ == "__main__":
    sys.argv = [T27_BUILD] + sys.argv[1:]
    runpy.run_path(T27_BUILD, run_name="__main__")
