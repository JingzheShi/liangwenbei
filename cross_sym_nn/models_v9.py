"""V9 architectures: 1 new candidate from v8 research.

v8_pairformer (V8_PairformerCrossSym): extends V6_PairwiseInteract with a
triangular-update-style pair-representation matrix R[i,j] = MLP([z_i, z_j, z_i-z_j])
and gated symmetric aggregation. Inspired by AlphaFold/Pairformer architectures.

Wraps models_v6 builders.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models_v6 import (
    _build_sae_encoder, _build_sae_decoder, _build_pred_head, _kaiming_init,
    ARCH_CLASSES_V6, ARCH_DEFAULTS_V6,
)


class V8_PairformerCrossSym(nn.Module):
    """Pairformer-style cross-sym architecture (Pairformer triangular update).

    1. per-sym encoder → latent (B, 5, d_latent)
    2. Pair representation: R[b, i, j] = MLP_pair([z_i, z_j, z_i - z_j])  ∈ d_pair
       (sym-agnostic: same MLP applied to every ordered pair).
    3. Triangle update (sym-agnostic):
         g[b, i, j] = sigmoid(gate_proj(R[b, i, j]))
         For each sym i: pair_summary_i = sum_j (g[b, i, j] * value_proj(R[b, i, j]))
         (sums over j != i. Self-loop masked.)
       This is the "outgoing edge" axial update from Pairformer.
    4. Concat [latent_i, pair_summary_i, x_noisy_i] → pred head (sym-agnostic).

    All operations are sym-agnostic (no per-sym params).
    """
    def __init__(self, n_sym: int = 5, n_feat: int = 259,
                 d_latent: int = 32, d_hidden: int = 128,
                 d_pair: int = 32, dropout: float = 0.30,
                 recon_alpha: float = 0.3, noise_sigma: float = 0.035):
        super().__init__()
        self.n_sym = n_sym
        self.n_feat = n_feat
        self.recon_alpha = recon_alpha
        self.noise_sigma = noise_sigma
        self.d_pair = d_pair
        self.d_latent = d_latent
        self._last_recon_loss = None
        self._last_aux_loss = None

        self.encoder = _build_sae_encoder(n_feat, d_hidden, d_latent, dropout)
        self.decoder = _build_sae_decoder(n_feat, d_hidden, d_latent)

        # Pair-representation MLP: input [z_i, z_j, z_i-z_j] ∈ 3*d_latent
        self.pair_mlp = nn.Sequential(
            nn.Linear(3 * d_latent, d_pair), nn.LayerNorm(d_pair), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_pair, d_pair),
        )
        # Gate (scalar per pair) and value projection
        self.gate_proj = nn.Linear(d_pair, d_pair)
        self.value_proj = nn.Linear(d_pair, d_pair)
        self.summary_ln = nn.LayerNorm(d_pair)

        self.pred_head = _build_pred_head(d_latent + d_pair + n_feat, d_hidden, dropout)

        _kaiming_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, F_ = x.shape
        x_flat = x.reshape(B * S, F_)
        if self.training and self.noise_sigma > 0:
            x_noisy = x_flat + torch.randn_like(x_flat) * self.noise_sigma
        else:
            x_noisy = x_flat
        latent = self.encoder(x_noisy).view(B, S, -1)         # (B, 5, d_latent)
        recon = self.decoder(latent.view(B * S, -1))

        # Build pair representation R[b, i, j] = MLP([z_i, z_j, z_i - z_j])
        z_i = latent.unsqueeze(2).expand(B, S, S, self.d_latent)  # (B, i, j, d)
        z_j = latent.unsqueeze(1).expand(B, S, S, self.d_latent)
        z_diff = z_i - z_j
        pair_in = torch.cat([z_i, z_j, z_diff], dim=-1)            # (B, S, S, 3d)
        R = self.pair_mlp(pair_in)                                 # (B, S, S, d_pair)

        # Triangle update: pair_summary_i = sum_{j!=i} sigmoid(gate(R[i,j])) * value(R[i,j])
        gate = torch.sigmoid(self.gate_proj(R))                    # (B, S, S, d_pair)
        value = self.value_proj(R)                                 # (B, S, S, d_pair)
        gated = gate * value                                       # (B, S, S, d_pair)
        # Mask self-loops
        eye_mask = torch.eye(S, device=x.device, dtype=torch.bool).view(1, S, S, 1)
        gated = gated.masked_fill(eye_mask, 0.0)
        pair_summary = gated.sum(dim=2)                            # (B, S, d_pair)
        # Normalize by number of partners (= 4) for scale invariance
        pair_summary = pair_summary / float(S - 1)
        pair_summary = self.summary_ln(pair_summary)

        latent_flat = latent.reshape(B * S, -1)
        pair_flat = pair_summary.reshape(B * S, -1)
        cat_in = torch.cat([latent_flat, pair_flat, x_noisy], dim=-1)
        pred = self.pred_head(cat_in).squeeze(-1).reshape(B, S)

        self._last_recon_loss = F.mse_loss(recon, x_flat) if self.training else None
        return pred


ARCH_CLASSES_V9 = dict(ARCH_CLASSES_V6)
ARCH_CLASSES_V9['v8_pairformer'] = V8_PairformerCrossSym

ARCH_DEFAULTS_V9 = dict(ARCH_DEFAULTS_V6)
ARCH_DEFAULTS_V9['v8_pairformer'] = dict(
    d_latent=32, d_hidden=128, d_pair=32, dropout=0.30,
    recon_alpha=0.3, noise_sigma=0.035,
)


def build_model_v9(arch: str, n_sym: int = 5, n_feat: int = 259, **overrides):
    cls = ARCH_CLASSES_V9[arch]
    kwargs = dict(ARCH_DEFAULTS_V9[arch])
    kwargs.update(overrides)
    return cls(n_sym=n_sym, n_feat=n_feat, **kwargs)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == '__main__':
    torch.manual_seed(0)
    B, S, F_ = 4, 5, 259
    x = torch.randn(B, S, F_)
    for arch in ['v6_pairwise', 'v8_pairformer']:
        m = build_model_v9(arch)
        m.eval()
        with torch.no_grad():
            y = m(x)
        m.train()
        y = m(x)
        n = count_params(m)
        print(f'  {arch:18s}  n_params={n:>8d}  out_shape={tuple(y.shape)}  '
              f'recon={float(m._last_recon_loss):.4f}')
