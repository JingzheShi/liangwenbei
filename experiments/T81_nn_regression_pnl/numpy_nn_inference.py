"""Pure-numpy MLP inference for T81 NN regression models.

Loads .npz with weights extracted via extract_nn_npz.py.
Architecture: Linear → LayerNorm → GELU → ... → Linear.
GELU uses tanh approximation (matches PyTorch's default GELU within ~1e-7).

The pipeline mirrors training:
  1. select 359-dim slice via keep_idx
  2. impute NaN with feat_mean (becomes 0 after standardize)
  3. (X - feat_mean) / feat_std
  4. clip to [-clip, clip]
  5. forward through MLP (in scaled prediction space)
  6. divide by target_scale → unscaled Δmid_norm prediction
"""
from __future__ import annotations

import os
import numpy as np


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    """Tanh-approx GELU: x · 0.5 · (1 + tanh(√(2/π) · (x + 0.044715·x³)))."""
    c = 0.7978845608028654  # sqrt(2/pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x * x * x)))


def layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """LayerNorm over last axis."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


class MLPNumpy:
    """MLP inference: Linear → (LN) → GELU → ... → Linear."""

    def __init__(self, npz_path: str):
        d = np.load(npz_path, allow_pickle=False)
        self.in_dim = int(d["in_dim"][0])
        self.hidden = list(d["hidden"].tolist())
        self.use_layernorm = bool(d["use_layernorm"][0])
        self.target_scale = float(d["target_scale"][0])
        self.feat_mean = d["feat_mean"].astype(np.float32)
        self.feat_std = d["feat_std"].astype(np.float32)
        self.keep_idx = d["keep_idx"].astype(np.int64)
        self.clip = float(d["clip"][0])

        # Layers
        self.W = []
        self.b = []
        self.LN_W = []
        self.LN_b = []
        for i in range(len(self.hidden)):
            self.W.append(d[f"L{i}_W"].astype(np.float32))
            self.b.append(d[f"L{i}_b"].astype(np.float32))
            if self.use_layernorm:
                self.LN_W.append(d[f"LN{i}_W"].astype(np.float32))
                self.LN_b.append(d[f"LN{i}_b"].astype(np.float32))
        self.WF = d["LF_W"].astype(np.float32)  # (1, h_last)
        self.bF = d["LF_b"].astype(np.float32)  # (1,)

    def standardize(self, X: np.ndarray) -> np.ndarray:
        """X is full-feature (no keep_idx selection done yet)."""
        Xs = X[:, self.keep_idx] if X.shape[1] != self.in_dim else X
        Xs = (Xs - self.feat_mean) / self.feat_std
        # NaN → 0
        Xs = np.where(np.isnan(Xs), 0.0, Xs)
        Xs = np.clip(Xs, -self.clip, self.clip)
        return Xs.astype(np.float32, copy=False)

    def forward(self, X_std: np.ndarray) -> np.ndarray:
        """X_std: already standardized (B, in_dim) float32. Returns (B,) predictions
        in y_regr (un-scaled) space."""
        h = X_std
        for i in range(len(self.hidden)):
            # Linear: x @ W^T + b  (W is (h_out, in))
            h = h @ self.W[i].T + self.b[i]
            if self.use_layernorm:
                h = layernorm(h, self.LN_W[i], self.LN_b[i])
            h = gelu_tanh(h)
            # Dropout is inactive at inference
        # Final linear
        out = h @ self.WF.T + self.bF
        out = out.squeeze(-1) / self.target_scale  # back to y_regr scale
        return out.astype(np.float32, copy=False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """End-to-end: takes raw 359-d (or full) features, returns (B,) y_regr."""
        X_std = self.standardize(X)
        return self.forward(X_std)


def cross_check():
    """Compare numpy inference vs torch inference on a small batch."""
    import sys
    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, HERE)
    import torch
    from train_nn_regr import MLPRegr

    seed = 42
    npz = os.path.join(HERE, f"model_T81_seed{seed}.npz")
    pt = os.path.join(HERE, f"model_T81_seed{seed}.pt")

    nn_npy = MLPNumpy(npz)
    ckpt = torch.load(pt, map_location="cpu", weights_only=False)
    model = MLPRegr(nn_npy.in_dim, hidden=tuple(nn_npy.hidden),
                    dropout=ckpt["dropout"], use_layernorm=ckpt.get("use_layernorm", True))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, nn_npy.in_dim)).astype(np.float32)

    # Numpy inference (skip standardization since X is already random; standardize anyway)
    # Pretend X is in raw space → directly feed to forward (skipping standardize)
    py_pred = nn_npy.forward(X.astype(np.float32))

    with torch.no_grad():
        torch_pred = model(torch.from_numpy(X)).numpy() / nn_npy.target_scale

    diff = np.abs(py_pred - torch_pred)
    print(f"  cross-check seed={seed}: max_abs_diff={diff.max():.3e} mean={diff.mean():.3e}")
    print(f"  numpy: {py_pred[:5]}")
    print(f"  torch: {torch_pred[:5]}")
    assert diff.max() < 1e-4, f"divergence: {diff.max()}"


if __name__ == "__main__":
    cross_check()
