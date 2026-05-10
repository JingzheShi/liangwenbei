"""Numpy-only inference for GroupTransformer (T172).

GroupTransformer:
  1. Linear projection: (B, feat_dim) → (B, K*d_token) → reshape (B, K, d_token)
  2. LayerNorm per token
  3. Prepend CLS token → sequence (B, K+1, d_token)
  4. N pre-norm TransformerEncoder blocks (multi-head attention + FFN)
  5. CLS token → head: LayerNorm + FC + GELU + FC → (B, 1)

Saved in .npz format by train_transformer.py's save_npz().
"""
from __future__ import annotations

import numpy as np


def _gelu(x):
    c = 0.7978845608028654
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x * x * x)))


def _layernorm(x, w, b, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * w + b


def _mha(x, in_proj_w, in_proj_b, out_w, out_b, n_heads):
    """Pre-norm multi-head self-attention (no causal mask).
    x: (B, L, d)  →  (B, L, d)
    """
    B, L, d = x.shape
    d_head = d // n_heads

    qkv = x.reshape(B * L, d) @ in_proj_w.T + in_proj_b  # (B*L, 3d)
    qkv = qkv.reshape(B, L, 3, d)
    Q, K, V = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

    Q = Q.reshape(B, L, n_heads, d_head).transpose(0, 2, 1, 3)  # (B, h, L, dh)
    K = K.reshape(B, L, n_heads, d_head).transpose(0, 2, 1, 3)
    V = V.reshape(B, L, n_heads, d_head).transpose(0, 2, 1, 3)

    scale = np.float32(1.0 / np.sqrt(d_head))
    scores = (Q @ K.transpose(0, 1, 3, 2)) * scale
    scores = scores - scores.max(axis=-1, keepdims=True)
    attn = np.exp(scores)
    attn = attn / attn.sum(axis=-1, keepdims=True)

    out = attn @ V  # (B, h, L, dh)
    out = out.transpose(0, 2, 1, 3).reshape(B * L, d)
    out = out @ out_w.T + out_b
    return out.reshape(B, L, d)


class GroupTransformerNumpy:
    """Numpy-only GroupTransformer inference loaded from .npz."""

    def __init__(self, npz_path: str):
        d = np.load(npz_path, allow_pickle=True)
        self.feat_dim = int(d["n_features"][0])
        self.n_tokens = int(d["n_tokens"][0])
        self.d_token = int(d["d_token"][0])
        self.n_heads = int(d["n_heads"][0])
        self.n_layers = int(d["n_layers"][0])
        self.target_scale = float(d["target_scale"][0])
        self.clip = float(d["clip"][0])

        self.feat_mean = d["feat_mean"].astype(np.float32)
        self.feat_std = d["feat_std"].astype(np.float32)
        self.keep_idx = d["keep_idx"].astype(np.int64)

        self.proj_w = d["proj_w"].astype(np.float32)        # (K*d, feat_dim)
        self.proj_b = d["proj_b"].astype(np.float32)        # (K*d,)
        self.proj_norm_w = d["proj_norm_w"].astype(np.float32)
        self.proj_norm_b = d["proj_norm_b"].astype(np.float32)
        self.cls_token = d["cls_token"].astype(np.float32)  # (1, 1, d)

        self.head_ln_w = d["head_ln_w"].astype(np.float32)
        self.head_ln_b = d["head_ln_b"].astype(np.float32)
        self.head_fc1_w = d["head_fc1_w"].astype(np.float32)
        self.head_fc1_b = d["head_fc1_b"].astype(np.float32)
        self.head_fc2_w = d["head_fc2_w"].astype(np.float32)
        self.head_fc2_b = d["head_fc2_b"].astype(np.float32)

        self.layers = []
        for i in range(self.n_layers):
            self.layers.append({
                "norm1_w": d[f"L{i}_norm1_w"].astype(np.float32),
                "norm1_b": d[f"L{i}_norm1_b"].astype(np.float32),
                "norm2_w": d[f"L{i}_norm2_w"].astype(np.float32),
                "norm2_b": d[f"L{i}_norm2_b"].astype(np.float32),
                "attn_in_proj_w": d[f"L{i}_attn_in_proj_w"].astype(np.float32),
                "attn_in_proj_b": d[f"L{i}_attn_in_proj_b"].astype(np.float32),
                "attn_out_w": d[f"L{i}_attn_out_w"].astype(np.float32),
                "attn_out_b": d[f"L{i}_attn_out_b"].astype(np.float32),
                "ffn1_w": d[f"L{i}_ffn1_w"].astype(np.float32),
                "ffn1_b": d[f"L{i}_ffn1_b"].astype(np.float32),
                "ffn2_w": d[f"L{i}_ffn2_w"].astype(np.float32),
                "ffn2_b": d[f"L{i}_ffn2_b"].astype(np.float32),
            })

    def predict(self, X_in: np.ndarray) -> np.ndarray:
        """X_in: (B, total_feat) or (B, feat_dim). Returns (B,) predictions."""
        Xs = X_in.astype(np.float32, copy=False)
        if Xs.shape[1] != self.feat_dim:
            Xs = Xs[:, self.keep_idx]
        Xs = (Xs - self.feat_mean) / self.feat_std
        Xs = np.where(np.isnan(Xs), 0.0, Xs)
        Xs = np.clip(Xs, -self.clip, self.clip).astype(np.float32)

        B = Xs.shape[0]
        # Project to K super-tokens
        tokens = (Xs @ self.proj_w.T + self.proj_b).reshape(B, self.n_tokens, self.d_token)
        tokens = _layernorm(tokens, self.proj_norm_w, self.proj_norm_b)

        # Prepend CLS
        cls = np.broadcast_to(self.cls_token, (B, 1, self.d_token)).copy().astype(np.float32)
        seq = np.concatenate([cls, tokens], axis=1)  # (B, K+1, d)

        # Transformer encoder (pre-norm)
        for layer in self.layers:
            normed = _layernorm(seq, layer["norm1_w"], layer["norm1_b"])
            attn_out = _mha(normed,
                            layer["attn_in_proj_w"], layer["attn_in_proj_b"],
                            layer["attn_out_w"], layer["attn_out_b"],
                            self.n_heads)
            seq = seq + attn_out
            normed = _layernorm(seq, layer["norm2_w"], layer["norm2_b"])
            ffn = normed.reshape(-1, self.d_token) @ layer["ffn1_w"].T + layer["ffn1_b"]
            ffn = _gelu(ffn)
            ffn = ffn @ layer["ffn2_w"].T + layer["ffn2_b"]
            seq = seq + ffn.reshape(B, -1, self.d_token)

        # CLS token → head
        cls_out = seq[:, 0, :]
        cls_out = _layernorm(cls_out, self.head_ln_w, self.head_ln_b)
        cls_out = cls_out @ self.head_fc1_w.T + self.head_fc1_b
        cls_out = _gelu(cls_out)
        out = cls_out @ self.head_fc2_w.T + self.head_fc2_b
        return (out.squeeze(-1) / self.target_scale).astype(np.float32)
