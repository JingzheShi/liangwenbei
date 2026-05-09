#!/usr/bin/env python3
"""Convert T97 PyTorch model to T87-compatible npz format for numpy inference.

T97 trunk+h60 head is architecturally identical to T87:
  Linear(359→256) + LayerNorm + GELU
  Linear(256→128) + LayerNorm + GELU
  Linear(128→64) + LayerNorm + GELU
  Linear(64→1) / target_scale
"""
from __future__ import annotations
import os
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
T97_DIR = os.path.join(ROOT, "experiments", "T97_multihead_nn")
OUT_DIR = HERE

SEEDS = [1, 7, 13, 42, 100]

for seed in SEEDS:
    pt_path = os.path.join(T97_DIR, f"model_T97_seed{seed}_main.pt")
    out_path = os.path.join(OUT_DIR, f"nn_T97_seed{seed}.npz")

    d = torch.load(pt_path, map_location='cpu', weights_only=False)
    sd = d['state_dict']
    main_idx = d['main_idx']   # 3 for h60
    target_scales = d['target_scales']
    target_scale = target_scales[main_idx]

    def t(key):
        return sd[key].numpy().astype(np.float32)

    np.savez(out_path,
        L0_W=t('trunk.0.weight'),   L0_b=t('trunk.0.bias'),
        LN0_W=t('trunk.1.weight'),  LN0_b=t('trunk.1.bias'),
        L1_W=t('trunk.4.weight'),   L1_b=t('trunk.4.bias'),
        LN1_W=t('trunk.5.weight'),  LN1_b=t('trunk.5.bias'),
        L2_W=t('trunk.8.weight'),   L2_b=t('trunk.8.bias'),
        LN2_W=t('trunk.9.weight'),  LN2_b=t('trunk.9.bias'),
        LF_W=t(f'heads.{main_idx}.weight'),
        LF_b=t(f'heads.{main_idx}.bias'),
        feat_mean=d['feat_mean'].astype(np.float32),
        feat_std=d['feat_std'].astype(np.float32),
        keep_idx=d['keep_idx'].astype(np.int32),
        hidden=np.array([256, 128, 64], dtype=np.int32),
        in_dim=np.array([int(d['in_dim'])], dtype=np.int32),
        target_scale=np.array([float(target_scale)], dtype=np.float32),
        use_layernorm=np.array([1], dtype=np.int32),
        clip=np.array([float(d['clip'])], dtype=np.float32),
    )
    print(f"  Converted seed {seed}: {out_path}", flush=True)

print("Done converting T97 models to npz format.")
