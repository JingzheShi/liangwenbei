"""
Smoke test for evaluate_on_session — uses a trivial random predict_fn to verify
the helper runs end-to-end without errors and yields finite results.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from dataset_eval import evaluate_on_session  # noqa: E402
from pnl import HORIZON_NAMES  # noqa: E402

DATA_PATH = ROOT / "data" / "snapshot_sym0_date0_am.parquet"


def main() -> int:
    df = pd.read_parquet(DATA_PATH)

    # Use a small subset of feature columns just to verify the pipeline works.
    feature_cols = [
        "bid1", "ask1", "bsize1", "asize1", "midprice1", "spread1", "imbalance",
    ]

    rng = np.random.default_rng(0)

    def predict_fn(batch: List[pd.DataFrame]) -> List[List[int]]:
        # Pretend to read the window (we don't actually use the features),
        # output 5 random class labels per sample.
        out = rng.integers(0, 3, size=(len(batch), 5)).tolist()
        # Assert each window has expected shape.
        for w in batch:
            assert w.shape == (100, len(feature_cols)), f"unexpected window shape {w.shape}"
        return out

    res = evaluate_on_session(
        predict_fn,
        session_df=df,
        feature_cols=feature_cols,
        window=100,
        batch_size=512,
    )

    assert res is not None
    print("[smoke] evaluate_on_session ran end-to-end. Per-horizon summary:")
    for h in HORIZON_NAMES:
        m = res[h]
        print(f"  {h}: cum={m['cum_pnl']:+.6f}  acc={m['accuracy']:.4f}  "
              f"active={m['n_predictions_active']}/{m['n_total']}")
    print(f"  best={res['best_score']:+.6f} @ {res['best_horizon']}")

    # No NaN in PnL or accuracy.
    for h in HORIZON_NAMES:
        for k in ("cum_pnl", "single_pnl", "accuracy"):
            v = res[h][k]
            assert np.isfinite(v), f"{h}.{k} not finite: {v}"
    print("[smoke] PASS — all PnL/accuracy values finite")
    return 0


if __name__ == "__main__":
    sys.exit(main())
