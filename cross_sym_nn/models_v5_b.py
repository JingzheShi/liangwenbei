"""V5 NN architectures: Differential Attention, GLU-V, Mask-SSL, SwiGLU, AdaLN.

All models expose:
  ._last_recon_loss  — MSE reconstruction loss (or None at eval)
  ._last_aux_loss    — auxiliary SSL loss (or None at eval / if unused)

Usage:
  from models_v5_b import build_model_v5b
  model = build_model_v5b("v5_diff_attn")   # top priority
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Shared Utilities
# ===========================================================================

def swiglu(x: torch.Tensor) -> torch.Tensor:
    """SwiGLU: x = [gate | value], return SiLU(gate) * value."""
    gate, val = x.chunk(2, dim=-1)
    return F.silu(gate) * val


class SwiGLUFFN(nn.Module):
    """Drop-in FFN replacement using SwiGLU gating (iso-param with 2× intermediate)."""
    def __init__(self, d_model: int, d_ff: int | None = None, dropout: float = 0.1):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 2          # same param budget as standard 4×FFN split 2:1
        self.fc1 = nn.Linear(d_model, 2 * d_ff)   # projects to gate + value jointly
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(swiglu(self.fc1(x))))


class RMSNorm(nn.Module):
    """RMS Normalization (no mean subtraction)."""
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.w * x / rms


# ===========================================================================
# Architecture 1: DiffAttnCrossSym (TOP PRIORITY)
# ===========================================================================

class DiffAttnCrossSym(nn.Module):
    """Differential Attention (Ye et al., Microsoft 2024) for cross-sym interaction.

    Two softmax attention maps; their difference cancels noise.
      A1 = softmax(Q1 K1^T / sqrt(d/2))
      A2 = softmax(Q2 K2^T / sqrt(d/2))
      DiffAttn = (A1 - λ·A2) @ V  per head, then headwise RMSNorm
      λ = exp(<λ_q1, λ_k1>) - exp(<λ_q2, λ_k2>) + λ_init
    """
    def __init__(self, n_sym: int = 5, n_feat: int = 259,
                 d_latent: int = 32, d_hidden: int = 128,
                 n_heads: int = 4, dropout: float = 0.30,
                 recon_alpha: float = 0.3, noise_sigma: float = 0.035,
                 lambda_init: float = 0.2):
        super().__init__()
        assert d_latent % n_heads == 0, "d_latent must be divisible by n_heads"
        self.n_sym = n_sym
        self.n_feat = n_feat
        self.n_heads = n_heads
        self.d_head = d_latent // n_heads
        self.d_half = self.d_head // 2
        self.recon_alpha = recon_alpha
        self.noise_sigma = noise_sigma
        self.lambda_init = lambda_init
        self._last_recon_loss = None
        self._last_aux_loss = None

        self.encoder = nn.Sequential(
            nn.Linear(n_feat, d_hidden), nn.LayerNorm(d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_latent), nn.LayerNorm(d_latent), nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, n_feat),
        )

        self.q_proj = nn.Linear(d_latent, d_latent, bias=False)
        self.k_proj = nn.Linear(d_latent, d_latent, bias=False)
        self.v_proj = nn.Linear(d_latent, d_latent, bias=False)
        self.out_proj = nn.Linear(d_latent, d_latent)

        self.lambda_q1 = nn.Parameter(torch.randn(n_heads, self.d_half) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(n_heads, self.d_half) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(n_heads, self.d_half) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(n_heads, self.d_half) * 0.1)

        self.head_norm = nn.ModuleList([RMSNorm(self.d_head) for _ in range(n_heads)])
        self.attn_norm = nn.LayerNorm(d_latent)

        self.pred_head = nn.Sequential(
            nn.Linear(d_latent + n_feat, d_hidden), nn.LayerNorm(d_hidden),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1),
        )

        self.drop_p = dropout
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def _diff_attention(self, z: torch.Tensor) -> torch.Tensor:
        B, T, D = z.shape
        H, dh, dh2 = self.n_heads, self.d_head, self.d_half

        q = self.q_proj(z).reshape(B, T, H, dh).permute(0, 2, 1, 3)
        k = self.k_proj(z).reshape(B, T, H, dh).permute(0, 2, 1, 3)
        v = self.v_proj(z).reshape(B, T, H, dh).permute(0, 2, 1, 3)

        q1, q2 = q[..., :dh2], q[..., dh2:]
        k1, k2 = k[..., :dh2], k[..., dh2:]

        scale = math.sqrt(dh2)
        A1 = F.softmax(torch.matmul(q1, k1.transpose(-2, -1)) / scale, dim=-1)
        A2 = F.softmax(torch.matmul(q2, k2.transpose(-2, -1)) / scale, dim=-1)

        lam = (torch.einsum('hd,hd->h', self.lambda_q1, self.lambda_k1).exp() -
               torch.einsum('hd,hd->h', self.lambda_q2, self.lambda_k2).exp() +
               self.lambda_init)

        if self.training:
            A1 = F.dropout(A1, p=self.drop_p)
            A2 = F.dropout(A2, p=self.drop_p)

        lam = lam.view(1, H, 1, 1)
        diff_attn = A1 - lam * A2
        out = torch.matmul(diff_attn, v)

        out_norm = []
        for h in range(H):
            oh = self.head_norm[h](out[:, h, :, :])
            out_norm.append(oh * (1.0 - self.lambda_init))
        out = torch.stack(out_norm, dim=1)

        out = out.permute(0, 2, 1, 3).reshape(B, T, D)
        out = self.out_proj(out)
        return out

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
        z_attn = self._diff_attention(z)
        z_combined = self.attn_norm(z + z_attn)
        z_flat = z_combined.reshape(B * S, -1)

        pred = self.pred_head(torch.cat([z_flat, x_noisy], dim=-1))
        pred = pred.squeeze(-1).reshape(B, S)

        if self.training:
            self._last_recon_loss = F.mse_loss(recon, x_flat)
        else:
            self._last_recon_loss = None
        self._last_aux_loss = None
        return pred


# ===========================================================================
# Architecture 2: GLUVCrossSym (Very Easy, High ROI)
# ===========================================================================

class GLUVCrossSym(nn.Module):
    """SAEMLPCrossSym with GLU-gated values in cross-sym attention.

    V_proj outputs 2×d_latent; split and gate with SiLU.
      V_gated = SiLU(V1) * V2  (V1, V2 each d_latent)
    """
    def __init__(self, n_sym: int = 5, n_feat: int = 259,
                 d_latent: int = 32, d_hidden: int = 128,
                 n_heads: int = 4, dropout: float = 0.30,
                 recon_alpha: float = 0.3, noise_sigma: float = 0.035):
        super().__init__()
        assert d_latent % n_heads == 0
        self.n_sym = n_sym
        self.n_feat = n_feat
        self.n_heads = n_heads
        self.d_head = d_latent // n_heads
        self.recon_alpha = recon_alpha
        self.noise_sigma = noise_sigma
        self._last_recon_loss = None
        self._last_aux_loss = None

        self.encoder = nn.Sequential(
            nn.Linear(n_feat, d_hidden), nn.LayerNorm(d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_latent), nn.LayerNorm(d_latent), nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, n_feat),
        )

        self.q_proj = nn.Linear(d_latent, d_latent, bias=False)
        self.k_proj = nn.Linear(d_latent, d_latent, bias=False)
        self.v_proj = nn.Linear(d_latent, 2 * d_latent, bias=False)
        self.out_proj = nn.Linear(d_latent, d_latent)
        self.attn_norm = nn.LayerNorm(d_latent)

        self.pred_head = nn.Sequential(
            nn.Linear(d_latent + n_feat, d_hidden), nn.LayerNorm(d_hidden),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1),
        )
        self.drop_p = dropout
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _glu_attn(self, z: torch.Tensor) -> torch.Tensor:
        B, T, D = z.shape
        H, dh = self.n_heads, self.d_head
        scale = math.sqrt(dh)

        q = self.q_proj(z).reshape(B, T, H, dh).permute(0, 2, 1, 3)
        k = self.k_proj(z).reshape(B, T, H, dh).permute(0, 2, 1, 3)

        v_raw = self.v_proj(z).reshape(B, T, H, 2 * dh).permute(0, 2, 1, 3)
        v1, v2 = v_raw[..., :dh], v_raw[..., dh:]
        v = F.silu(v1) * v2

        attn = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.drop_p if self.training else 0.0
        )
        out = attn.permute(0, 2, 1, 3).reshape(B, T, D)
        return self.out_proj(out)

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
        z_attn = self._glu_attn(z)
        z_combined = self.attn_norm(z + z_attn)
        z_flat = z_combined.reshape(B * S, -1)

        pred = self.pred_head(torch.cat([z_flat, x_noisy], dim=-1))
        pred = pred.squeeze(-1).reshape(B, S)

        if self.training:
            self._last_recon_loss = F.mse_loss(recon, x_flat)
        else:
            self._last_recon_loss = None
        self._last_aux_loss = None
        return pred


# ===========================================================================
# Architecture 3: MaskSSLCrossSym (VIME-style dual pretext)
# ===========================================================================

class MaskSSLCrossSym(nn.Module):
    """SAEMLPCrossSym + VIME-style dual pretext: reconstruction + mask estimation.

    Random 30% feature corruption → encoder must predict binary mask.
    Loss: L_pred + alpha * L_recon + mask_alpha * L_mask_bce
    """
    def __init__(self, n_sym: int = 5, n_feat: int = 259,
                 d_latent: int = 32, d_hidden: int = 128,
                 n_heads: int = 4, dropout: float = 0.30,
                 recon_alpha: float = 0.3, noise_sigma: float = 0.035,
                 mask_p: float = 0.30, mask_alpha: float = 0.5):
        super().__init__()
        assert d_latent % n_heads == 0
        self.n_sym = n_sym
        self.n_feat = n_feat
        self.recon_alpha = recon_alpha
        self.noise_sigma = noise_sigma
        self.mask_p = mask_p
        self.mask_alpha = mask_alpha
        self._last_recon_loss = None
        self._last_aux_loss = None

        self.encoder = nn.Sequential(
            nn.Linear(n_feat, d_hidden), nn.LayerNorm(d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_latent), nn.LayerNorm(d_latent), nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, n_feat),
        )

        self.mask_head = nn.Linear(d_latent, n_feat)

        self.attn = nn.MultiheadAttention(d_latent, num_heads=n_heads,
                                          dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_latent)

        self.pred_head = nn.Sequential(
            nn.Linear(d_latent + n_feat, d_hidden), nn.LayerNorm(d_hidden),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.mask_head.weight)
        nn.init.constant_(self.mask_head.bias, -2.0)

    def _corrupt(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = torch.bernoulli(torch.full_like(x, self.mask_p))
        noise = torch.randn_like(x) * x.std(dim=0, keepdim=True).clamp(min=1e-6)
        x_corrupted = x * (1 - mask) + noise * mask
        return x_corrupted, mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, F_ = x.shape
        x_flat = x.reshape(B * S, F_)

        if self.training:
            x_noisy = x_flat + torch.randn_like(x_flat) * self.noise_sigma
            x_corrupted, mask = self._corrupt(x_noisy)
        else:
            x_corrupted = x_flat
            mask = None

        latent = self.encoder(x_corrupted)
        recon = self.decoder(latent)

        z = latent.view(B, S, -1)
        z_attn, _ = self.attn(z, z, z)
        z_combined = self.attn_norm(z + z_attn)
        z_flat = z_combined.reshape(B * S, -1)

        pred = self.pred_head(torch.cat([z_flat, x_corrupted], dim=-1))
        pred = pred.squeeze(-1).reshape(B, S)

        if self.training:
            self._last_recon_loss = F.mse_loss(recon, x_flat)
            mask_pred = self.mask_head(latent)
            self._last_aux_loss = F.binary_cross_entropy_with_logits(mask_pred, mask)
        else:
            self._last_recon_loss = None
            self._last_aux_loss = None
        return pred


# ===========================================================================
# Architecture 4: MarketAdaLNCrossSym (MASTER-inspired AdaLN)
# ===========================================================================

class MarketAdaLNCrossSym(nn.Module):
    """Cross-sym attention with MASTER-style market-state AdaLN conditioning.

    Market context = mean(z_attn, dim=sym). AdaLN derives scale/shift per sym.
    """
    def __init__(self, n_sym: int = 5, n_feat: int = 259,
                 d_latent: int = 32, d_hidden: int = 128,
                 n_heads: int = 4, dropout: float = 0.30,
                 recon_alpha: float = 0.3, noise_sigma: float = 0.035):
        super().__init__()
        assert d_latent % n_heads == 0
        self.n_sym = n_sym
        self.n_feat = n_feat
        self.recon_alpha = recon_alpha
        self.noise_sigma = noise_sigma
        self._last_recon_loss = None
        self._last_aux_loss = None

        self.encoder = nn.Sequential(
            nn.Linear(n_feat, d_hidden), nn.LayerNorm(d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_latent), nn.LayerNorm(d_latent), nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, n_feat),
        )

        self.attn = nn.MultiheadAttention(d_latent, num_heads=n_heads,
                                          dropout=dropout, batch_first=True)

        self.adaln_norm = nn.LayerNorm(d_latent, elementwise_affine=False)
        self.adaln_proj = nn.Linear(d_latent, 2 * d_latent)

        self.pred_head = nn.Sequential(
            nn.Linear(d_latent + n_feat, d_hidden), nn.LayerNorm(d_hidden),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.adaln_proj.weight)
        nn.init.zeros_(self.adaln_proj.bias)

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
        z_attn, _ = self.attn(z, z, z)
        z_res = z + z_attn

        ctx = z_res.mean(dim=1)
        adaln_out = self.adaln_proj(ctx)
        scale, shift = adaln_out.chunk(2, dim=-1)
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
        z_normed = self.adaln_norm(z_res)
        z_conditioned = (1 + scale) * z_normed + shift

        z_flat = z_conditioned.reshape(B * S, -1)
        pred = self.pred_head(torch.cat([z_flat, x_noisy], dim=-1))
        pred = pred.squeeze(-1).reshape(B, S)

        if self.training:
            self._last_recon_loss = F.mse_loss(recon, x_flat)
        else:
            self._last_recon_loss = None
        self._last_aux_loss = None
        return pred


# ===========================================================================
# Architecture 5: DiffAttnSwiGLU (Diff-Attn + SwiGLU combo)
# ===========================================================================

class DiffAttnSwiGLU(nn.Module):
    """Combines Differential Attention + SwiGLU FFN in encoder and pred head."""
    def __init__(self, n_sym: int = 5, n_feat: int = 259,
                 d_latent: int = 32, d_hidden: int = 128,
                 n_heads: int = 4, dropout: float = 0.30,
                 recon_alpha: float = 0.3, noise_sigma: float = 0.035,
                 lambda_init: float = 0.2):
        super().__init__()
        assert d_latent % n_heads == 0
        assert (d_latent // n_heads) % 2 == 0, "d_head must be even for Q1/Q2 split"
        self.n_sym = n_sym
        self.n_feat = n_feat
        self.n_heads = n_heads
        self.d_head = d_latent // n_heads
        self.d_half = self.d_head // 2
        self.recon_alpha = recon_alpha
        self.noise_sigma = noise_sigma
        self.lambda_init = lambda_init
        self._last_recon_loss = None
        self._last_aux_loss = None

        self.enc1 = nn.Linear(n_feat, 2 * d_hidden)
        self.enc_norm1 = nn.LayerNorm(d_hidden)
        self.enc_drop1 = nn.Dropout(dropout)
        self.enc2 = nn.Linear(d_hidden, d_latent)
        self.enc_norm2 = nn.LayerNorm(d_latent)

        self.decoder = nn.Sequential(
            nn.Linear(d_latent, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, n_feat),
        )

        self.q_proj = nn.Linear(d_latent, d_latent, bias=False)
        self.k_proj = nn.Linear(d_latent, d_latent, bias=False)
        self.v_proj = nn.Linear(d_latent, d_latent, bias=False)
        self.out_proj = nn.Linear(d_latent, d_latent)
        self.lambda_q1 = nn.Parameter(torch.randn(n_heads, self.d_half) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(n_heads, self.d_half) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(n_heads, self.d_half) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(n_heads, self.d_half) * 0.1)
        self.head_norm = nn.ModuleList([RMSNorm(self.d_head) for _ in range(n_heads)])
        self.attn_norm = nn.LayerNorm(d_latent)

        in_dim = d_latent + n_feat
        self.ph1 = nn.Linear(in_dim, 2 * d_hidden)
        self.ph_norm1 = nn.LayerNorm(d_hidden)
        self.ph_drop1 = nn.Dropout(dropout)
        self.ph2 = nn.Linear(d_hidden, 2 * (d_hidden // 2))
        self.ph_drop2 = nn.Dropout(dropout)
        self.ph3 = nn.Linear(d_hidden // 2, 1)

        self.drop_p = dropout
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.out_proj.weight)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        h = swiglu(self.enc1(x))
        h = self.enc_drop1(self.enc_norm1(h))
        return self.enc_norm2(self.enc2(h))

    def _diff_attention(self, z: torch.Tensor) -> torch.Tensor:
        B, T, D = z.shape
        H, dh, dh2 = self.n_heads, self.d_head, self.d_half
        scale = math.sqrt(dh2)

        q = self.q_proj(z).reshape(B, T, H, dh).permute(0, 2, 1, 3)
        k = self.k_proj(z).reshape(B, T, H, dh).permute(0, 2, 1, 3)
        v = self.v_proj(z).reshape(B, T, H, dh).permute(0, 2, 1, 3)

        q1, q2 = q[..., :dh2], q[..., dh2:]
        k1, k2 = k[..., :dh2], k[..., dh2:]

        A1 = F.softmax(torch.matmul(q1, k1.transpose(-2, -1)) / scale, dim=-1)
        A2 = F.softmax(torch.matmul(q2, k2.transpose(-2, -1)) / scale, dim=-1)

        lam = (torch.einsum('hd,hd->h', self.lambda_q1, self.lambda_k1).exp() -
               torch.einsum('hd,hd->h', self.lambda_q2, self.lambda_k2).exp() +
               self.lambda_init).view(1, H, 1, 1)

        if self.training:
            A1 = F.dropout(A1, p=self.drop_p)
            A2 = F.dropout(A2, p=self.drop_p)

        diff_attn = A1 - lam * A2
        out = torch.matmul(diff_attn, v)

        out_norm = []
        for h in range(H):
            out_norm.append(self.head_norm[h](out[:, h]) * (1.0 - self.lambda_init))
        out = torch.stack(out_norm, dim=1).permute(0, 2, 1, 3).reshape(B, T, D)
        return self.out_proj(out)

    def _pred(self, cat: torch.Tensor) -> torch.Tensor:
        h = self.ph_drop1(self.ph_norm1(swiglu(self.ph1(cat))))
        h = self.ph_drop2(swiglu(self.ph2(h)))
        return self.ph3(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, F_ = x.shape
        x_flat = x.reshape(B * S, F_)
        if self.training and self.noise_sigma > 0:
            x_noisy = x_flat + torch.randn_like(x_flat) * self.noise_sigma
        else:
            x_noisy = x_flat

        latent = self._encode(x_noisy)
        recon = self.decoder(latent)

        z = latent.view(B, S, -1)
        z_attn = self._diff_attention(z)
        z_combined = self.attn_norm(z + z_attn)
        z_flat = z_combined.reshape(B * S, -1)

        pred = self._pred(torch.cat([z_flat, x_noisy], dim=-1))
        pred = pred.squeeze(-1).reshape(B, S)

        if self.training:
            self._last_recon_loss = F.mse_loss(recon, x_flat)
        else:
            self._last_recon_loss = None
        self._last_aux_loss = None
        return pred


# ===========================================================================
# Factory & Defaults
# ===========================================================================

ARCH_CLASSES_V5B = {
    "v5_diff_attn":    DiffAttnCrossSym,
    "v5_glu_v_attn":   GLUVCrossSym,
    "v5_mask_ssl":     MaskSSLCrossSym,
    "v5_market_adaln": MarketAdaLNCrossSym,
    "v5_diff_swiglu":  DiffAttnSwiGLU,
}

ARCH_DEFAULTS_V5B = {
    "v5_diff_attn":    dict(d_latent=32, d_hidden=128, n_heads=4, dropout=0.30,
                            recon_alpha=0.3, noise_sigma=0.035, lambda_init=0.2),
    "v5_glu_v_attn":   dict(d_latent=32, d_hidden=128, n_heads=4, dropout=0.30,
                            recon_alpha=0.3, noise_sigma=0.035),
    "v5_mask_ssl":     dict(d_latent=32, d_hidden=128, n_heads=4, dropout=0.30,
                            recon_alpha=0.3, noise_sigma=0.035, mask_p=0.30, mask_alpha=0.5),
    "v5_market_adaln": dict(d_latent=32, d_hidden=128, n_heads=4, dropout=0.30,
                            recon_alpha=0.3, noise_sigma=0.035),
    "v5_diff_swiglu":  dict(d_latent=32, d_hidden=128, n_heads=4, dropout=0.30,
                            recon_alpha=0.3, noise_sigma=0.035, lambda_init=0.2),
}


def build_model_v5b(arch: str, n_sym: int = 5, n_feat: int = 259, **overrides):
    """Factory for v5b models."""
    cls = ARCH_CLASSES_V5B[arch]
    kwargs = dict(ARCH_DEFAULTS_V5B[arch])
    kwargs.update(overrides)
    return cls(n_sym=n_sym, n_feat=n_feat, **kwargs)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
