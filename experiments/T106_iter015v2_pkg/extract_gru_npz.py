"""Convert 5-seed T95 GRU .pt checkpoints into .npz for numpy inference."""
from __future__ import annotations
import os
import sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
T95 = os.path.join(ROOT, "experiments", "T95_cnn_rnn_deeplob")

SEEDS = (1, 7, 13, 42, 100)


def main():
    for s in SEEDS:
        src = os.path.join(T95, f"model_T95_gru_w100_C_seed{s}.pt")
        ckpt = torch.load(src, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"]
        out = {
            "in_norm_W": sd["in_norm.weight"].cpu().numpy().astype(np.float32),
            "in_norm_b": sd["in_norm.bias"].cpu().numpy().astype(np.float32),
            # PyTorch GRU stores as (3*hidden, in_dim/hidden), order: r, z, n
            "gru_Wih": sd["gru.weight_ih_l0"].cpu().numpy().astype(np.float32),
            "gru_Whh": sd["gru.weight_hh_l0"].cpu().numpy().astype(np.float32),
            "gru_bih": sd["gru.bias_ih_l0"].cpu().numpy().astype(np.float32),
            "gru_bhh": sd["gru.bias_hh_l0"].cpu().numpy().astype(np.float32),
            "fc_W": sd["fc.weight"].cpu().numpy().astype(np.float32),
            "fc_b": sd["fc.bias"].cpu().numpy().astype(np.float32),
            "in_dim": np.array([int(ckpt["in_dim"])], dtype=np.int32),
            "hidden": np.array([int(ckpt["gru_hidden"])], dtype=np.int32),
            "window": np.array([int(ckpt["window"])], dtype=np.int32),
            "target_scale": np.array([float(ckpt["target_scale"])], dtype=np.float32),
            "clip": np.array([float(ckpt["clip"])], dtype=np.float32),
            "norm_mode": np.array([ckpt["args"]["norm_mode"]], dtype="U16"),
        }
        out_path = os.path.join(HERE, f"gru_h60_seed{s}.npz")
        np.savez(out_path, **out)
        print(f"  wrote {out_path}  hidden={out['hidden'][0]}  in_dim={out['in_dim'][0]}  norm_mode={out['norm_mode'][0]}  scale={out['target_scale'][0]:.4f}")


if __name__ == "__main__":
    main()
