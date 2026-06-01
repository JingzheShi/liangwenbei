"""V5 NN variants: deeper / gated / wider / multi-block cross-sym attention.

Baseline = models_v4.SAEMLPCrossSym (d_latent=32, d_hidden=128, n_heads=4, 1 block).
Goal: surpass the +36.35 ± 0.75 (5-seed) single-model SOTA by deepening or
widening the cross-sym interaction layer.

Variants implemented:

  V1.  SAECrossSymDeeper  – stack N transformer blocks (pre-LN MHSA + FFN).
                            FFN ratio 2x latent. N default 2.
  V2.  SAECrossSymGated   – 1 attention layer with learned per-sym sigmoid
                            gate controlling how much cross-sym info to inject.
  V3.  SAECrossSymWider   – 1 block but d_latent=64 (head_dim 8), n_heads=8.
  V5.  SAECrossSymTransformer – same template as V1 but N=3 by default (also
                                explicit name to test "more is better").

All take input (B, 5, n_feat) and emit (B, 5). Same encoder/decoder/head
structure as the v4 SAEMLPCrossSym baseline. Only the cross-sym block changes.

Each module exposes ._last_recon_loss for the trainer.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class PreLNCrossSymBlock(nn.Module):
    """Pre-LN cross-sym transformer block: LN→MHSA→res, LN→FFN→res.

    Operates on (B, S, d) tokens. Permutation-equivariant.
    """
    def __init__(self, d_latent: int, n_heads: int,
                 ffn_ratio: float = 2.0, dropout: float = 0.3):
        super().__init__()
        assert d_latent % n_heads == 0
        self.ln_attn = nn.LayerNorm(d_latent)
        self.attn = nn.MultiheadAttention(
            d_latent, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.attn_drop = nn.Dropout(dropout)

        d_ffn = int(round(d_latent * ffn_ratio))
        self.ln_ffn = nn.LayerNorm(d_latent)
        self.ffn = nn.Sequential(
            nn.Linear(d_latent, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_latent),
        )
        self.ffn_drop = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.ln_attn(z)
        a, _ = self.attn(h, h, h, need_weights=False)
        z = z + self.attn_drop(a)
        h = self.ln_ffn(z)
        z = z + self.ffn_drop(self.ffn(h))
        return z


# ---------------------------------------------------------------------------
# V1 / V5.  SAECrossSymDeeper — stacked pre-LN transformer blocks
# ---------------------------------------------------------------------------

class SAECrossSymDeeper(nn.Module):
    """Per-sym SAE-MLP encoder + N stacked pre-LN cross-sym blocks + per-sym head.

    With n_attn_layers=1 this is roughly the same as v4 SAEMLPCrossSym but with
    pre-LN ordering (cleaner gradients). With n_attn_layers≥2 we test whether
    deeper cross-sym reasoning helps.
    """
    def __init__(self, n_sym: int = 5, n_feat: int = 259,
                 d_latent: int = 32, d_hidden: int = 128,
                 n_heads: int = 4, n_attn_layers: int = 2,
                 ffn_ratio: float = 2.0,
                 dropout: float = 0.30,
                 recon_alpha: float = 0.3, noise_sigma: float = 0.035):
        super().__init__()
        self.n_sym = n_sym
        self.n_feat = n_feat
        self.recon_alpha = recon_alpha
        self.noise_sigma = noise_sigma

        self.encoder = nn.Sequential(
            nn.Linear(n_feat, d_hidden), nn.LayerNorm(d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_latent), nn.LayerNorm(d_latent), nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, n_feat),
        )
        self.blocks = nn.ModuleList([
            PreLNCrossSymBlock(d_latent, n_heads, ffn_ratio, dropout)
            for _ in range(n_attn_layers)
        ])
        self.final_ln = nn.LayerNorm(d_latent)

        self.pred_head = nn.Sequential(
            nn.Linear(d_latent + n_feat, d_hidden), nn.LayerNorm(d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1),
        )
        self._last_recon_loss = None

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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
        z = self.final_ln(z)
        z_flat = z.reshape(B * S, -1)

        cat_in = torch.cat([z_flat, x_noisy], dim=-1)
        pred = self.pred_head(cat_in).squeeze(-1).reshape(B, S)

        if self.training:
            self._last_recon_loss = F.mse_loss(recon, x_flat)
        else:
            self._last_recon_loss = None
        return pred


# ---------------------------------------------------------------------------
# V2.  SAECrossSymGated — 1 attention layer with learned residual gate
# ---------------------------------------------------------------------------

class SAECrossSymGated(nn.Module):
    """1 cross-sym MHSA + per-sym sigmoid gate on the residual.

    z_out = z + sigmoid(g(z)) * attn(z, z, z)

    The gate is a tiny MLP (d_latent → d_latent//4 → d_latent) per token.
    Lets the model learn (per-sym) how much cross-sym signal to inject.
    """
    def __init__(self, n_sym: int = 5, n_feat: int = 259,
                 d_latent: int = 32, d_hidden: int = 128,
                 n_heads: int = 4, dropout: float = 0.30,
                 recon_alpha: float = 0.3, noise_sigma: float = 0.035,
                 gate_init_bias: float = -1.0):
        super().__init__()
        self.n_sym = n_sym
        self.n_feat = n_feat
        self.recon_alpha = recon_alpha
        self.noise_sigma = noise_sigma

        self.encoder = nn.Sequential(
            nn.Linear(n_feat, d_hidden), nn.LayerNorm(d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_latent), nn.LayerNorm(d_latent), nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, n_feat),
        )

        assert d_latent % n_heads == 0
        self.attn_ln = nn.LayerNorm(d_latent)
        self.attn = nn.MultiheadAttention(d_latent, num_heads=n_heads,
                                          dropout=dropout, batch_first=True)
        self.attn_drop = nn.Dropout(dropout)
        gate_hidden = max(4, d_latent // 4)
        self.gate = nn.Sequential(
            nn.Linear(d_latent, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, d_latent),
        )
        self.post_ln = nn.LayerNorm(d_latent)

        self.pred_head = nn.Sequential(
            nn.Linear(d_latent + n_feat, d_hidden), nn.LayerNorm(d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1),
        )
        self._last_recon_loss = None

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # bias gate towards low (sigmoid(-1) ≈ 0.27) initially → small cross-sym
        # contribution at init, network learns to open it
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, gate_init_bias)

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
        h = self.attn_ln(z)
        a, _ = self.attn(h, h, h, need_weights=False)
        a = self.attn_drop(a)
        gate_v = torch.sigmoid(self.gate(h))
        z = z + gate_v * a
        z = self.post_ln(z)
        z_flat = z.reshape(B * S, -1)

        cat_in = torch.cat([z_flat, x_noisy], dim=-1)
        pred = self.pred_head(cat_in).squeeze(-1).reshape(B, S)

        if self.training:
            self._last_recon_loss = F.mse_loss(recon, x_flat)
        else:
            self._last_recon_loss = None
        return pred


# ---------------------------------------------------------------------------
# V3.  SAECrossSymWider — wider attention (d_latent=64, n_heads=8)
# ---------------------------------------------------------------------------

class SAECrossSymWider(nn.Module):
    """1-block cross-sym attention with wider latent and more heads.

    d_latent default 64 (vs SOTA 32), n_heads 8 (head_dim 8).
    Decoder/head also widened proportionally (d_hidden=256).
    """
    def __init__(self, n_sym: int = 5, n_feat: int = 259,
                 d_latent: int = 64, d_hidden: int = 256,
                 n_heads: int = 8, dropout: float = 0.30,
                 recon_alpha: float = 0.3, noise_sigma: float = 0.035,
                 ffn_ratio: float = 2.0, n_attn_layers: int = 1):
        super().__init__()
        self.n_sym = n_sym
        self.n_feat = n_feat
        self.recon_alpha = recon_alpha
        self.noise_sigma = noise_sigma

        self.encoder = nn.Sequential(
            nn.Linear(n_feat, d_hidden), nn.LayerNorm(d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_latent), nn.LayerNorm(d_latent), nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, n_feat),
        )
        self.blocks = nn.ModuleList([
            PreLNCrossSymBlock(d_latent, n_heads, ffn_ratio, dropout)
            for _ in range(n_attn_layers)
        ])
        self.final_ln = nn.LayerNorm(d_latent)

        self.pred_head = nn.Sequential(
            nn.Linear(d_latent + n_feat, d_hidden), nn.LayerNorm(d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1),
        )
        self._last_recon_loss = None

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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
        z = self.final_ln(z)
        z_flat = z.reshape(B * S, -1)

        cat_in = torch.cat([z_flat, x_noisy], dim=-1)
        pred = self.pred_head(cat_in).squeeze(-1).reshape(B, S)

        if self.training:
            self._last_recon_loss = F.mse_loss(recon, x_flat)
        else:
            self._last_recon_loss = None
        return pred


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

ARCH_CLASSES_V5 = {
    "v1_deeper2":   SAECrossSymDeeper,    # N=2 transformer blocks
    "v1_deeper3":   SAECrossSymDeeper,    # N=3 transformer blocks
    "v2_gated":     SAECrossSymGated,
    "v3_wider":     SAECrossSymWider,
    "v3_wider_d96": SAECrossSymWider,     # even wider sanity
    "v5_xfmr3":     SAECrossSymDeeper,    # alias N=3 (transformer flavor)
}

ARCH_DEFAULTS_V5 = {
    "v1_deeper2":   dict(d_latent=32, d_hidden=128, n_heads=4, n_attn_layers=2,
                         ffn_ratio=2.0, dropout=0.30,
                         recon_alpha=0.3, noise_sigma=0.035),
    "v1_deeper3":   dict(d_latent=32, d_hidden=128, n_heads=4, n_attn_layers=3,
                         ffn_ratio=2.0, dropout=0.30,
                         recon_alpha=0.3, noise_sigma=0.035),
    "v2_gated":     dict(d_latent=32, d_hidden=128, n_heads=4, dropout=0.30,
                         recon_alpha=0.3, noise_sigma=0.035,
                         gate_init_bias=-1.0),
    "v3_wider":     dict(d_latent=64, d_hidden=256, n_heads=8, n_attn_layers=1,
                         ffn_ratio=2.0, dropout=0.30,
                         recon_alpha=0.3, noise_sigma=0.035),
    "v3_wider_d96": dict(d_latent=96, d_hidden=256, n_heads=8, n_attn_layers=1,
                         ffn_ratio=2.0, dropout=0.25,
                         recon_alpha=0.3, noise_sigma=0.035),
    "v5_xfmr3":     dict(d_latent=32, d_hidden=128, n_heads=4, n_attn_layers=3,
                         ffn_ratio=2.0, dropout=0.30,
                         recon_alpha=0.3, noise_sigma=0.035),
}


def build_model_v5(arch: str, n_sym: int = 5, n_feat: int = 259, **overrides):
    cls = ARCH_CLASSES_V5[arch]
    kwargs = dict(ARCH_DEFAULTS_V5[arch])
    kwargs.update(overrides)
    return cls(n_sym=n_sym, n_feat=n_feat, **kwargs)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
