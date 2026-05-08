"""Extract MLP weights from T87 .pt → .npz for torch-free inference.

Same layout as T81's extract_nn_npz.py — Predictor.py loads npz directly.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def extract_one(pt_path: str, npz_path: str) -> dict:
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    hidden = ckpt["hidden"]
    in_dim = ckpt["in_dim"]
    use_layernorm = ckpt.get("use_layernorm", True)
    target_scale = float(ckpt.get("target_scale", 1.0))
    feat_mean = np.asarray(ckpt["feat_mean"], dtype=np.float32)
    feat_std = np.asarray(ckpt["feat_std"], dtype=np.float32)
    keep_idx = np.asarray(ckpt["keep_idx"], dtype=np.int64)
    clip_val = float(ckpt.get("clip", 10.0))

    n_hidden_layers = len(hidden)

    out = {
        "in_dim": np.array([in_dim], dtype=np.int64),
        "hidden": np.array(list(hidden), dtype=np.int64),
        "use_layernorm": np.array([1 if use_layernorm else 0], dtype=np.int8),
        "target_scale": np.array([target_scale], dtype=np.float32),
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "keep_idx": keep_idx,
        "clip": np.array([clip_val], dtype=np.float32),
    }

    layer_idx = 0
    for i in range(n_hidden_layers):
        wkey = f"net.{layer_idx}.weight"
        bkey = f"net.{layer_idx}.bias"
        out[f"L{i}_W"] = sd[wkey].numpy().astype(np.float32)
        out[f"L{i}_b"] = sd[bkey].numpy().astype(np.float32)
        layer_idx += 1
        if use_layernorm:
            ln_w = f"net.{layer_idx}.weight"
            ln_b = f"net.{layer_idx}.bias"
            out[f"LN{i}_W"] = sd[ln_w].numpy().astype(np.float32)
            out[f"LN{i}_b"] = sd[ln_b].numpy().astype(np.float32)
            layer_idx += 1
        layer_idx += 2  # GELU + Dropout

    out["LF_W"] = sd[f"net.{layer_idx}.weight"].numpy().astype(np.float32)
    out["LF_b"] = sd[f"net.{layer_idx}.bias"].numpy().astype(np.float32)

    np.savez(npz_path, **out)
    return out


def main():
    seeds = (1, 7, 13, 42, 100)
    tag = "main"
    for s in seeds:
        pt = os.path.join(HERE, f"model_T87_seed{s}_{tag}.pt")
        npz = os.path.join(HERE, f"model_T87_seed{s}_{tag}.npz")
        info = extract_one(pt, npz)
        print(f"  seed={s} hidden={info['hidden'].tolist()} "
              f"target_scale={info['target_scale'][0]:.4f} -> {npz}",
              flush=True)


if __name__ == "__main__":
    main()
