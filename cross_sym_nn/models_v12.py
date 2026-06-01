"""V12 architectures: 5 intra-sym × cross-sym interleave variants.

Goal: beat v11_minillama SOTA (+40.42 ± 1.06) by exploring different
ways of interleaving intra-sym FFN with cross-sym MHSA.

V11_MiniLlama baseline: encoder(intra) → [cross_attn + intra_ffn] × 2 → head

V12.A — V12_DeeperLlama       (same Llama block but depth=4)
V12.B — V12_InvertedLlama     (intra-FFN first, then cross-MHSA in block)
V12.C — V12_ExplicitAlternate (cross-block ↔ intra-block, 4 separate blocks)
V12.D — V12_MultiResRefine    (cross + intra up/down projection refine)
V12.E — V12_ParallelGated     (parallel branches with per-sym gate)
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models_v6 import (
    _build_sae_encoder, _build_sae_decoder, _build_pred_head, _kaiming_init,
)
from models_v11 import RMSNorm, SwiGLU, LlamaBlock, V11_MiniLlama


# ===========================================================================
# V12.A — Deeper Mini-Llama (depth=4)
# ===========================================================================

class V12_DeeperLlama(V11_MiniLlama):
    """Same architecture as V11_MiniLlama but depth=4 (was depth=2).

    Test whether Llama-style pre-RMSNorm gives gradient flow that lets us
    go deeper than depth=2 (vanilla attn failed at depth=4).
    """
    pass  # All differences are in defaults (depth=4)


# ===========================================================================
# V12.B — Inverted Llama Block (intra-FFN → cross-MHSA)
# ===========================================================================

class InvertedLlamaBlock(nn.Module):
    """Pre-RMSNorm + SwiGLU FFN (intra) first, then MHSA (cross-sym).

    V11 Llama block ordering: norm → MHSA(cross) + residual → norm → SwiGLU + residual
    Inverted ordering:        norm → SwiGLU(intra) + residual → norm → MHSA(cross) + residual
    """
    def __init__(self, d_model: int, n_heads: int, d_ffn_mult: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_model * d_ffn_mult, dropout=dropout)
        self.norm2 = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True, bias=False)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.ffn(h)
        h = self.norm2(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.attn_drop(a)
        return x


class V12_InvertedLlama(nn.Module):
    """Same as V11_MiniLlama but with InvertedLlamaBlock (intra → cross)."""
    def __init__(self, n_sym: int = 5, n_feat: int = 319,
                 d_latent: int = 64, d_hidden: int = 128,
                 n_heads: int = 4, depth: int = 2,
                 d_ffn_mult: int = 2, dropout: float = 0.30,
                 recon_alpha: float = 0.3, noise_sigma: float = 0.035):
        super().__init__()
        assert d_latent % n_heads == 0
        self.n_sym = n_sym
        self.n_feat = n_feat
        self.d_latent = d_latent
        self.recon_alpha = recon_alpha
        self.noise_sigma = noise_sigma
        self._last_recon_loss = None
        self._last_aux_loss = None

        self.encoder = _build_sae_encoder(n_feat, d_hidden, d_latent, dropout)
        self.decoder = _build_sae_decoder(n_feat, d_hidden, d_latent)

        self.blocks = nn.ModuleList([
            InvertedLlamaBlock(d_latent, n_heads, d_ffn_mult=d_ffn_mult, dropout=dropout)
            for _ in range(depth)
        ])
        self.post_norm = RMSNorm(d_latent)
        self.pred_head = _build_pred_head(d_latent + n_feat, d_hidden, dropout)

        _kaiming_init(self)
        for m in self.modules():
            if isinstance(m, RMSNorm):
                nn.init.ones_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, F_ = x.shape
        x_flat = x.reshape(B * S, F_)
        if self.training and self.noise_sigma > 0:
            x_noisy = x_flat + torch.randn_like(x_flat) * self.noise_sigma
        else:
            x_noisy = x_flat
        latent = self.encoder(x_noisy)
        recon = self.decoder(latent)

        z = latent.view(B, S, -1)
        for blk in self.blocks:
            z = blk(z)
        z = self.post_norm(z)
        z_flat = z.reshape(B * S, -1)

        cat_in = torch.cat([z_flat, x_noisy], dim=-1)
        pred = self.pred_head(cat_in).squeeze(-1).reshape(B, S)
        self._last_recon_loss = F.mse_loss(recon, x_flat) if self.training else None
        return pred


# ===========================================================================
# V12.C — Explicit Alternating: separate cross-blocks & intra-blocks
# ===========================================================================

class CrossOnlyBlock(nn.Module):
    """Pre-RMSNorm + cross-sym MHSA + residual (no FFN)."""
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        return x + self.drop(a)


class IntraOnlyBlock(nn.Module):
    """Pre-RMSNorm + SwiGLU FFN (per-sym) + residual."""
    def __init__(self, d_model: int, d_ffn_mult: int, dropout: float):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_model * d_ffn_mult, dropout=dropout)

    def forward(self, x):
        h = self.norm(x)
        return x + self.ffn(h)


class V12_ExplicitAlternate(nn.Module):
    """Explicitly alternates cross-only blocks and intra-only blocks.

    Pattern (n_pairs=2 means 4 sub-blocks total):
        [cross, intra, cross, intra]

    Each block is independent (own parameters), unlike Llama block where
    cross+intra share residual stream within one block.
    """
    def __init__(self, n_sym: int = 5, n_feat: int = 319,
                 d_latent: int = 64, d_hidden: int = 128,
                 n_heads: int = 4, n_pairs: int = 2,
                 d_ffn_mult: int = 2, dropout: float = 0.30,
                 recon_alpha: float = 0.3, noise_sigma: float = 0.035):
        super().__init__()
        assert d_latent % n_heads == 0
        self.n_sym = n_sym
        self.n_feat = n_feat
        self.d_latent = d_latent
        self.recon_alpha = recon_alpha
        self.noise_sigma = noise_sigma
        self._last_recon_loss = None
        self._last_aux_loss = None

        self.encoder = _build_sae_encoder(n_feat, d_hidden, d_latent, dropout)
        self.decoder = _build_sae_decoder(n_feat, d_hidden, d_latent)

        blocks = []
        for _ in range(n_pairs):
            blocks.append(CrossOnlyBlock(d_latent, n_heads, dropout))
            blocks.append(IntraOnlyBlock(d_latent, d_ffn_mult, dropout))
        self.blocks = nn.ModuleList(blocks)
        self.post_norm = RMSNorm(d_latent)
        self.pred_head = _build_pred_head(d_latent + n_feat, d_hidden, dropout)

        _kaiming_init(self)
        for m in self.modules():
            if isinstance(m, RMSNorm):
                nn.init.ones_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, F_ = x.shape
        x_flat = x.reshape(B * S, F_)
        if self.training and self.noise_sigma > 0:
            x_noisy = x_flat + torch.randn_like(x_flat) * self.noise_sigma
        else:
            x_noisy = x_flat
        latent = self.encoder(x_noisy)
        recon = self.decoder(latent)

        z = latent.view(B, S, -1)
        for blk in self.blocks:
            z = blk(z)
        z = self.post_norm(z)
        z_flat = z.reshape(B * S, -1)
        cat_in = torch.cat([z_flat, x_noisy], dim=-1)
        pred = self.pred_head(cat_in).squeeze(-1).reshape(B, S)
        self._last_recon_loss = F.mse_loss(recon, x_flat) if self.training else None
        return pred


# ===========================================================================
# V12.D — Multi-Resolution Cross-Sym Refine
# ===========================================================================

class V12_MultiResRefine(nn.Module):
    """Multi-resolution refinement: cross-attn at d_latent, then up/down
    project for intra-sym refinement, smooth update.

    Step (repeated n_steps=3 times):
        z = z + cross_attn_block(z)           # cross-sym mix at d_latent
        z_up = up_proj(z)                     # d_latent -> d_up (intra-sym)
        z_up = GELU + dropout(z_up)
        z_refined = down_proj(z_up)           # d_up -> d_latent
        z = z + smooth_alpha * (z_refined - z)
    """
    def __init__(self, n_sym: int = 5, n_feat: int = 319,
                 d_latent: int = 64, d_hidden: int = 128, d_up: int = 128,
                 n_heads: int = 4, n_steps: int = 3, smooth_alpha: float = 0.5,
                 dropout: float = 0.30,
                 recon_alpha: float = 0.3, noise_sigma: float = 0.035):
        super().__init__()
        assert d_latent % n_heads == 0
        self.n_sym = n_sym
        self.n_feat = n_feat
        self.d_latent = d_latent
        self.n_steps = n_steps
        self.smooth_alpha = smooth_alpha
        self.recon_alpha = recon_alpha
        self.noise_sigma = noise_sigma
        self._last_recon_loss = None
        self._last_aux_loss = None

        self.encoder = _build_sae_encoder(n_feat, d_hidden, d_latent, dropout)
        self.decoder = _build_sae_decoder(n_feat, d_hidden, d_latent)

        # Per-step cross-attn blocks (independent)
        self.cross_blocks = nn.ModuleList([
            CrossOnlyBlock(d_latent, n_heads, dropout) for _ in range(n_steps)
        ])
        # Per-step intra up/down (independent)
        self.up_blocks = nn.ModuleList([
            nn.Sequential(
                RMSNorm(d_latent),
                nn.Linear(d_latent, d_up, bias=False),
                nn.GELU(),
                nn.Dropout(dropout),
            ) for _ in range(n_steps)
        ])
        self.down_blocks = nn.ModuleList([
            nn.Linear(d_up, d_latent, bias=False) for _ in range(n_steps)
        ])
        self.post_norm = RMSNorm(d_latent)
        self.pred_head = _build_pred_head(d_latent + n_feat, d_hidden, dropout)

        _kaiming_init(self)
        for m in self.modules():
            if isinstance(m, RMSNorm):
                nn.init.ones_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, F_ = x.shape
        x_flat = x.reshape(B * S, F_)
        if self.training and self.noise_sigma > 0:
            x_noisy = x_flat + torch.randn_like(x_flat) * self.noise_sigma
        else:
            x_noisy = x_flat
        latent = self.encoder(x_noisy)
        recon = self.decoder(latent)

        z = latent.view(B, S, -1)
        for i in range(self.n_steps):
            z = self.cross_blocks[i](z)
            z_up = self.up_blocks[i](z)
            z_ref = self.down_blocks[i](z_up)
            z = z + self.smooth_alpha * (z_ref - z)
        z = self.post_norm(z)
        z_flat = z.reshape(B * S, -1)
        cat_in = torch.cat([z_flat, x_noisy], dim=-1)
        pred = self.pred_head(cat_in).squeeze(-1).reshape(B, S)
        self._last_recon_loss = F.mse_loss(recon, x_flat) if self.training else None
        return pred


# ===========================================================================
# V12.E — Parallel Branches + Gate
# ===========================================================================

class V12_ParallelGated(nn.Module):
    """Parallel intra and cross branches with per-sym gate.

    Step (repeated n_steps=2):
        intra_out = intra_block(z)            # per-sym SwiGLU
        cross_out = cross_block(z)            # cross-sym MHSA
        gate = sigmoid(linear(z))             # (B, 5, 1) per-sym gate
        z = z + gate * cross_out + (1-gate) * intra_out
    """
    def __init__(self, n_sym: int = 5, n_feat: int = 319,
                 d_latent: int = 64, d_hidden: int = 128,
                 n_heads: int = 4, n_steps: int = 2,
                 d_ffn_mult: int = 2, dropout: float = 0.30,
                 recon_alpha: float = 0.3, noise_sigma: float = 0.035):
        super().__init__()
        assert d_latent % n_heads == 0
        self.n_sym = n_sym
        self.n_feat = n_feat
        self.d_latent = d_latent
        self.n_steps = n_steps
        self.recon_alpha = recon_alpha
        self.noise_sigma = noise_sigma
        self._last_recon_loss = None
        self._last_aux_loss = None

        self.encoder = _build_sae_encoder(n_feat, d_hidden, d_latent, dropout)
        self.decoder = _build_sae_decoder(n_feat, d_hidden, d_latent)

        self.cross_blocks = nn.ModuleList([
            CrossOnlyBlock(d_latent, n_heads, dropout) for _ in range(n_steps)
        ])
        self.intra_blocks = nn.ModuleList([
            IntraOnlyBlock(d_latent, d_ffn_mult, dropout) for _ in range(n_steps)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(d_latent, 1) for _ in range(n_steps)
        ])
        self.post_norm = RMSNorm(d_latent)
        self.pred_head = _build_pred_head(d_latent + n_feat, d_hidden, dropout)

        _kaiming_init(self)
        for m in self.modules():
            if isinstance(m, RMSNorm):
                nn.init.ones_(m.weight)
        # Bias gates toward 0.5 initially
        for g in self.gates:
            nn.init.zeros_(g.weight)
            nn.init.zeros_(g.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, F_ = x.shape
        x_flat = x.reshape(B * S, F_)
        if self.training and self.noise_sigma > 0:
            x_noisy = x_flat + torch.randn_like(x_flat) * self.noise_sigma
        else:
            x_noisy = x_flat
        latent = self.encoder(x_noisy)
        recon = self.decoder(latent)

        z = latent.view(B, S, -1)
        for i in range(self.n_steps):
            # Each branch starts from z (parallel) but returns z + delta
            intra_out = self.intra_blocks[i](z) - z      # extract delta
            cross_out = self.cross_blocks[i](z) - z
            gate = torch.sigmoid(self.gates[i](z))       # (B, S, 1)
            z = z + gate * cross_out + (1 - gate) * intra_out
        z = self.post_norm(z)
        z_flat = z.reshape(B * S, -1)
        cat_in = torch.cat([z_flat, x_noisy], dim=-1)
        pred = self.pred_head(cat_in).squeeze(-1).reshape(B, S)
        self._last_recon_loss = F.mse_loss(recon, x_flat) if self.training else None
        return pred


# ===========================================================================
# Registry
# ===========================================================================

ARCH_CLASSES_V12 = {
    'v12_a_deeper':    V12_DeeperLlama,
    'v12_b_inverted':  V12_InvertedLlama,
    'v12_c_alternate': V12_ExplicitAlternate,
    'v12_d_multires':  V12_MultiResRefine,
    'v12_e_gated':     V12_ParallelGated,
}

ARCH_DEFAULTS_V12 = {
    'v12_a_deeper': dict(
        d_latent=64, d_hidden=128, n_heads=4, depth=4,
        d_ffn_mult=2, dropout=0.30,
        recon_alpha=0.3, noise_sigma=0.035,
    ),
    'v12_b_inverted': dict(
        d_latent=64, d_hidden=128, n_heads=4, depth=2,
        d_ffn_mult=2, dropout=0.30,
        recon_alpha=0.3, noise_sigma=0.035,
    ),
    'v12_c_alternate': dict(
        d_latent=64, d_hidden=128, n_heads=4, n_pairs=2,
        d_ffn_mult=2, dropout=0.30,
        recon_alpha=0.3, noise_sigma=0.035,
    ),
    'v12_d_multires': dict(
        d_latent=64, d_hidden=128, d_up=128, n_heads=4,
        n_steps=3, smooth_alpha=0.5, dropout=0.30,
        recon_alpha=0.3, noise_sigma=0.035,
    ),
    'v12_e_gated': dict(
        d_latent=64, d_hidden=128, n_heads=4, n_steps=2,
        d_ffn_mult=2, dropout=0.30,
        recon_alpha=0.3, noise_sigma=0.035,
    ),
}


def build_model_v12(arch: str, n_sym: int = 5, n_feat: int = 319, **overrides):
    cls = ARCH_CLASSES_V12[arch]
    kwargs = dict(ARCH_DEFAULTS_V12[arch])
    kwargs.update(overrides)
    return cls(n_sym=n_sym, n_feat=n_feat, **kwargs)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == '__main__':
    torch.manual_seed(0)
    B, S, F_ = 4, 5, 319
    x = torch.randn(B, S, F_)
    for arch in ARCH_CLASSES_V12.keys():
        m = build_model_v12(arch)
        m.eval()
        with torch.no_grad():
            y = m(x)
        m.train()
        y = m(x)
        n = count_params(m)
        recon = float(m._last_recon_loss) if m._last_recon_loss is not None else float('nan')
        aux = float(m._last_aux_loss) if m._last_aux_loss is not None else float('nan')
        print(f'  {arch:18s} n_params={n:>8d}  out={tuple(y.shape)}  '
              f'recon={recon:.4f}  aux={aux:.4f}')
