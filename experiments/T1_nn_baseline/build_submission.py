"""Build submission/iter_001_deeplob_finetuned/ from this experiment's best_model.pt.

Steps:
  1) Copy model.py from examples/mmpc_demo (architecture identical)
  2) Copy best_model.pt (already contains mean/std/feature_cols in meta)
  3) Write Predictor.py (same input handling as mmpc_demo + apply log1p+z-score from meta)
  4) Write config.json with full 154 feature list
  5) Write requirements.txt
"""
from __future__ import annotations

import json
import os
import shutil
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

ITER_DIR = os.path.join(ROOT, "submission", "iter_001_deeplob-finetuned")
EXAMPLE_DIR = os.path.join(ROOT, "examples", "mmpc_demo")


PREDICTOR_TEMPLATE = '''from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch

try:
    from model import DeepLOB
except ImportError:
    from .model import DeepLOB


class Predictor:
    """DeepLOB 5-head multi-task predictor with z-score normalization.

    The mean/std and log1p column indices are stored in the checkpoint meta dict;
    inference applies the same transform that was used at training time.
    """

    def __init__(self) -> None:
        pkl_path = os.path.join(os.path.dirname(__file__), "best_model.pt")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            ckpt = torch.load(pkl_path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(pkl_path, map_location=self.device)

        self.meta = dict(ckpt.get("meta") or {})
        nums = self.meta["num_classes_per_head"]
        feature_cols = list(self.meta["feature_columns"])
        self.feature_cols = feature_cols
        log1p_cols = list(self.meta.get("log1p_cols", []))
        # Materialize LazyLinear by running a dummy forward at the correct shape.
        n_features = len(feature_cols)
        self.model = DeepLOB(list(nums)).to(self.device)
        with torch.no_grad():
            dummy = torch.zeros(2, 1, 100, n_features, device=self.device)
            self.model(dummy)
        self.model.load_state_dict(ckpt["model_state"], strict=True)
        self.model.eval()

        # Normalization statistics (broadcast over (B, T, F))
        mean = np.asarray(self.meta["mean"], dtype=np.float32)
        std = np.asarray(self.meta["std"], dtype=np.float32)
        self._mean = mean.reshape(1, 1, -1)
        self._std = std.reshape(1, 1, -1)
        # Indices for columns to log1p before z-score
        self._log1p_idx = np.array(
            [feature_cols.index(c) for c in log1p_cols], dtype=np.int64
        )

    def predict(self, batches: list) -> list[list[int]]:
        """
        batches: list of pd.DataFrame, each (window, n_features) in self.feature_cols order.
        returns: list of length len(batches), each a list of K ints (argmax per head).
        """
        arrs = [df.to_numpy(dtype=np.float32, copy=False) for df in batches]
        x_np = np.ascontiguousarray(np.stack(arrs, axis=0))  # (B, T, F)
        # log1p selected columns
        if self._log1p_idx.size > 0:
            x_np[:, :, self._log1p_idx] = np.log1p(x_np[:, :, self._log1p_idx])
        # z-score
        x_np = (x_np - self._mean) / self._std
        # replace NaN/Inf (training-time convention)
        x_np = np.where(np.isfinite(x_np), x_np, 0.0).astype(np.float32, copy=False)

        x = torch.from_numpy(x_np).unsqueeze(1).to(self.device, dtype=torch.float32)  # (B, 1, T, F)
        with torch.no_grad():
            heads = self.model(x)
        preds_per_head = [h.argmax(1).cpu().numpy() for h in heads]
        pred_matrix = np.stack(preds_per_head, axis=1)  # (B, K)
        return pred_matrix.astype(int).tolist()


if __name__ == "__main__":
    import pandas as pd

    p = Predictor()
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        rng.standard_normal((100, len(p.feature_cols))).astype(np.float32),
        columns=p.feature_cols,
    )
    print(p.predict([df, df]))
'''


def main():
    if not os.path.exists(os.path.join(HERE, "best_model.pt")):
        sys.exit("[build] best_model.pt not found; run train.py first")

    if os.path.exists(ITER_DIR):
        shutil.rmtree(ITER_DIR)
    os.makedirs(ITER_DIR, exist_ok=True)

    # 1) model.py
    shutil.copyfile(os.path.join(EXAMPLE_DIR, "model.py"), os.path.join(ITER_DIR, "model.py"))

    # 2) best_model.pt
    shutil.copyfile(os.path.join(HERE, "best_model.pt"), os.path.join(ITER_DIR, "best_model.pt"))

    # 3) Predictor.py
    with open(os.path.join(ITER_DIR, "Predictor.py"), "w") as f:
        f.write(PREDICTOR_TEMPLATE)

    # 4) config.json — feature must match the order in the checkpoint meta
    ckpt = torch.load(os.path.join(HERE, "best_model.pt"), map_location="cpu", weights_only=False)
    feats = list(ckpt["meta"]["feature_columns"])
    config = {
        "python_version": "3.11",
        "batch": 1024,
        "feature": feats,
        "label": ["label_5", "label_10", "label_20", "label_40", "label_60"],
    }
    with open(os.path.join(ITER_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # 5) requirements.txt — keep it minimal (torch / numpy / pandas)
    with open(os.path.join(ITER_DIR, "requirements.txt"), "w") as f:
        # pin to versions that match training environment
        f.write("numpy==2.4.3\n")
        f.write("pandas==3.0.1\n")
        f.write("torch==2.10.0\n")

    print(f"[build] wrote {ITER_DIR}")
    for fn in sorted(os.listdir(ITER_DIR)):
        size = os.path.getsize(os.path.join(ITER_DIR, fn)) / 1024
        print(f"  {fn} ({size:.1f} KB)")


if __name__ == "__main__":
    main()
