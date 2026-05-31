# V5 Implementation Specifications
**For opus orchestrator (Worker B) to implement directly.**

All models: Input `(B, 5, 259)` → Output `(B, 5)`  
Sym-agnostic: no sym embedding, no per-sym normalization, no sym index routing.  
Interface: expose `._last_recon_loss` (and optionally `._last_aux_loss`) for trainer.

---

## File: `models_v5.py`

```python
"""V5 NN architectures: Differential Attention, GLU-V, Mask-SSL, SwiGLU, AdaLN.

All models expose:
  ._last_recon_loss  — MSE reconstruction loss (or None at eval)
  ._last_aux_loss    — auxiliary SSL loss (or None at eval / if unused)

Usage:
  from models_v5 import build_model_v5
  model = build_model_v5("v5_diff_attn")   # top priority
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
# Architecture 1: V5_DiffAttnCrossSym (TOP PRIORITY)
# ===========================================================================

class DiffAttnCrossSym(nn.Module):
    """Differential Attention (Ye et al., Microsoft 2024) for cross-sym interaction.

    Paper: arxiv 2410.05258
    Key idea: Two softmax attention maps; their difference cancels noise.
      A1 = softmax(Q1 K1^T / sqrt(d/2))
      A2 = softmax(Q2 K2^T / sqrt(d/2))
      DiffAttn = (A1 - λ·A2) @ V  per head, then headwise RMSNorm
      λ = exp(<λ_q1, λ_k1>) - exp(<λ_q2, λ_k2>) + λ_init

    For our 5-sym attention: cancels irrelevant sym-sym noise weights,
    sharpens focus on genuinely predictive cross-sym relationships.

    Input:  (B, 5, n_feat)
    Output: (B, 5) predictions
    Side:   ._last_recon_loss (MSE recon)
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
        self.d_head = d_latent // n_heads       # full head dim
        self.d_half = self.d_head // 2          # half head dim for Q1/Q2, K1/K2
        self.recon_alpha = recon_alpha
        self.noise_sigma = noise_sigma
        self.lambda_init = lambda_init
        self._last_recon_loss = None
        self._last_aux_loss = None

        # --- Encoder (per-sym, shared weights) ---
        self.encoder = nn.Sequential(
            nn.Linear(n_feat, d_hidden), nn.LayerNorm(d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_latent), nn.LayerNorm(d_latent), nn.GELU(),
        )

        # --- Decoder (for reconstruction loss) ---
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, n_feat),
        )

        # --- Differential Attention projections ---
        # Q1, Q2 share the Q projection (split in forward)
        # K1, K2 share the K projection (split in forward)
        # V stays full d_latent
        self.q_proj = nn.Linear(d_latent, d_latent, bias=False)    # → split to Q1,Q2
        self.k_proj = nn.Linear(d_latent, d_latent, bias=False)    # → split to K1,K2
        self.v_proj = nn.Linear(d_latent, d_latent, bias=False)
        self.out_proj = nn.Linear(d_latent, d_latent)

        # Lambda parameters (per head): 4 learnable vectors per head
        self.lambda_q1 = nn.Parameter(torch.randn(n_heads, self.d_half) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(n_heads, self.d_half) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(n_heads, self.d_half) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(n_heads, self.d_half) * 0.1)

        # Per-head RMSNorm (applied after differential attention, before concat)
        self.head_norm = nn.ModuleList([RMSNorm(self.d_head) for _ in range(n_heads)])

        # Attention norm (post-norm residual, matching sae_cross_sym convention)
        self.attn_norm = nn.LayerNorm(d_latent)

        # --- Prediction head ---
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
        nn.init.zeros_(self.out_proj.bias) if self.out_proj.bias is not None else None

    def _diff_attention(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, 5, d_latent) → (B, 5, d_latent)"""
        B, T, D = z.shape
        H, dh, dh2 = self.n_heads, self.d_head, self.d_half

        # Project Q,K,V
        q = self.q_proj(z).reshape(B, T, H, dh).permute(0, 2, 1, 3)  # (B, H, T, dh)
        k = self.k_proj(z).reshape(B, T, H, dh).permute(0, 2, 1, 3)
        v = self.v_proj(z).reshape(B, T, H, dh).permute(0, 2, 1, 3)

        # Split Q and K into two halves
        q1, q2 = q[..., :dh2], q[..., dh2:]    # (B, H, T, dh2) each
        k1, k2 = k[..., :dh2], k[..., dh2:]

        # Compute two attention maps
        scale = math.sqrt(dh2)
        A1 = F.softmax(torch.matmul(q1, k1.transpose(-2, -1)) / scale, dim=-1)  # (B,H,T,T)
        A2 = F.softmax(torch.matmul(q2, k2.transpose(-2, -1)) / scale, dim=-1)

        # Compute per-head lambda (scalar)
        # lambda_h = exp(<lq1_h, lk1_h>) - exp(<lq2_h, lk2_h>) + lambda_init
        lam = (torch.einsum('hd,hd->h', self.lambda_q1, self.lambda_k1).exp() -
               torch.einsum('hd,hd->h', self.lambda_q2, self.lambda_k2).exp() +
               self.lambda_init)  # (H,)

        # Differential attention: (A1 - λ·A2) @ V
        if self.training:
            A1 = F.dropout(A1, p=self.drop_p)
            A2 = F.dropout(A2, p=self.drop_p)

        lam = lam.view(1, H, 1, 1)
        diff_attn = (A1 - lam * A2)  # (B, H, T, T)
        out = torch.matmul(diff_attn, v)  # (B, H, T, dh)

        # Per-head RMSNorm then scale by (1 - lambda_init)
        out_norm = []
        for h in range(H):
            oh = self.head_norm[h](out[:, h, :, :])  # (B, T, dh)
            out_norm.append(oh * (1.0 - self.lambda_init))
        out = torch.stack(out_norm, dim=1)  # (B, H, T, dh)

        # Reshape and project out
        out = out.permute(0, 2, 1, 3).reshape(B, T, D)  # (B, T, d_latent)
        out = self.out_proj(out)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, F_ = x.shape
        x_flat = x.reshape(B * S, F_)

        # Noise augmentation
        if self.training and self.noise_sigma > 0:
            x_noisy = x_flat + torch.randn_like(x_flat) * self.noise_sigma
        else:
            x_noisy = x_flat

        # Per-sym encode
        latent = self.encoder(x_noisy)                      # (B*5, d_latent)
        recon = self.decoder(latent)                         # (B*5, n_feat)

        # Cross-sym differential attention
        z = latent.view(B, S, -1)                           # (B, 5, d_latent)
        z_attn = self._diff_attention(z)                    # (B, 5, d_latent)
        z_combined = self.attn_norm(z + z_attn)             # post-norm residual
        z_flat = z_combined.reshape(B * S, -1)

        # Prediction
        pred = self.pred_head(torch.cat([z_flat, x_noisy], dim=-1))
        pred = pred.squeeze(-1).reshape(B, S)

        # Reconstruction loss
        if self.training:
            self._last_recon_loss = F.mse_loss(recon, x_flat)
        else:
            self._last_recon_loss = None
        self._last_aux_loss = None
        return pred


# ===========================================================================
# Architecture 2: V5_GLUVCrossSym (Very Easy, High ROI)
# ===========================================================================

class GLUVCrossSym(nn.Module):
    """SAEMLPCrossSym with GLU-gated values in cross-sym attention.

    Paper: "GLU Attention Improve Transformer" arxiv 2507.00022
    Modification: V_proj outputs 2×d_latent; split and gate with SiLU.
      V_gated = SiLU(V1) * V2  (V1, V2 each d_latent)
    Zero architectural overhead in concept; one extra linear matrix.

    Input:  (B, 5, n_feat)
    Output: (B, 5)
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

        # Encoder + Decoder (same as sae_cross_sym)
        self.encoder = nn.Sequential(
            nn.Linear(n_feat, d_hidden), nn.LayerNorm(d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_latent), nn.LayerNorm(d_latent), nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, n_feat),
        )

        # Cross-sym attention with GLU values
        self.q_proj = nn.Linear(d_latent, d_latent, bias=False)
        self.k_proj = nn.Linear(d_latent, d_latent, bias=False)
        # V projects to 2×d_latent for GLU gating
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
        """z: (B, 5, d_latent) → (B, 5, d_latent)"""
        B, T, D = z.shape
        H, dh = self.n_heads, self.d_head
        scale = math.sqrt(dh)

        q = self.q_proj(z).reshape(B, T, H, dh).permute(0, 2, 1, 3)
        k = self.k_proj(z).reshape(B, T, H, dh).permute(0, 2, 1, 3)

        # V is 2×d_latent, split and GLU-gate
        v_raw = self.v_proj(z).reshape(B, T, H, 2 * dh).permute(0, 2, 1, 3)
        v1, v2 = v_raw[..., :dh], v_raw[..., dh:]
        v = F.silu(v1) * v2  # (B, H, T, dh)

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
# Architecture 3: V5_MaskSSLCrossSym (VIME-style dual pretext)
# ===========================================================================

class MaskSSLCrossSym(nn.Module):
    """SAEMLPCrossSym + VIME-style dual pretext: reconstruction + mask estimation.

    Paper: VIME (Yoon et al., NeurIPS 2020) arxiv 2003.08013
    Extension: Add mask estimation head (d_latent → n_feat binary classifier)
    Loss: L_pred + alpha * L_recon + gamma * L_mask_bce
    
    Training corruption:
    - Random subset of features (p_mask=0.3) replaced with zero or Gaussian noise
    - Encoder sees corrupted input
    - Task 1: reconstruct original (already in sae_cross_sym)
    - Task 2: predict binary mask (which features were corrupted)

    Input:  (B, 5, n_feat)
    Output: (B, 5)
    Side:   ._last_recon_loss, ._last_aux_loss (BCE mask estimation)
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

        # Encoder, decoder (same as sae_cross_sym)
        self.encoder = nn.Sequential(
            nn.Linear(n_feat, d_hidden), nn.LayerNorm(d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_latent), nn.LayerNorm(d_latent), nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, n_feat),
        )

        # Mask estimation head (binary: was feature corrupted or not?)
        self.mask_head = nn.Linear(d_latent, n_feat)

        # Cross-sym attention (same as sae_cross_sym)
        self.attn = nn.MultiheadAttention(d_latent, num_heads=n_heads,
                                          dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_latent)

        # Prediction head
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
        # mask_head: zero init to start predicting "nothing masked"
        nn.init.zeros_(self.mask_head.weight)
        nn.init.constant_(self.mask_head.bias, -2.0)  # sigmoid(-2) ≈ 0.12, slightly biased to 0

    def _corrupt(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Randomly mask p_mask fraction of features per sample.
        Returns (x_corrupted, mask) where mask=1 means the feature was corrupted.
        """
        mask = torch.bernoulli(torch.full_like(x, self.mask_p))  # (B*5, n_feat)
        # Replace masked features with Gaussian noise (marginal distribution approx)
        noise = torch.randn_like(x) * x.std(dim=0, keepdim=True).clamp(min=1e-6)
        x_corrupted = x * (1 - mask) + noise * mask
        return x_corrupted, mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, F_ = x.shape
        x_flat = x.reshape(B * S, F_)

        if self.training:
            # Gaussian noise augmentation (like original sae_cross_sym)
            x_noisy = x_flat + torch.randn_like(x_flat) * self.noise_sigma
            # VIME corruption: mask random features
            x_corrupted, mask = self._corrupt(x_noisy)
        else:
            x_corrupted = x_flat
            mask = None

        latent = self.encoder(x_corrupted)
        recon = self.decoder(latent)

        # Cross-sym attention
        z = latent.view(B, S, -1)
        z_attn, _ = self.attn(z, z, z)
        z_combined = self.attn_norm(z + z_attn)
        z_flat = z_combined.reshape(B * S, -1)

        pred = self.pred_head(torch.cat([z_flat, x_corrupted], dim=-1))
        pred = pred.squeeze(-1).reshape(B, S)

        if self.training:
            # Task 1: reconstruction
            self._last_recon_loss = F.mse_loss(recon, x_flat)
            # Task 2: mask estimation (BCE)
            mask_pred = self.mask_head(latent)    # (B*5, n_feat) logits
            self._last_aux_loss = F.binary_cross_entropy_with_logits(mask_pred, mask)
        else:
            self._last_recon_loss = None
            self._last_aux_loss = None
        return pred


# ===========================================================================
# Architecture 4: V5_SwiGLUSAECrossSym (SwiGLU everywhere)
# ===========================================================================

class SwiGLUSAECrossSym(nn.Module):
    """SAEMLPCrossSym with SwiGLU FFN in encoder, decoder, and pred_head.

    Reference: "GLU Variants Improve Transformer" Shazeer 2020 arxiv 2002.05202
    Standard in LLaMA/PaLM/Mistral — rock-solid empirical improvement.

    SwiGLU encoder: n_feat → [2×d_hidden] → (gate) → d_hidden → d_latent
    SwiGLU decoder: d_latent → [2×d_hidden] → (gate) → d_hidden → n_feat
    SwiGLU pred:    cat → [2×d_hidden] → (gate) → ... → 1

    Input:  (B, 5, n_feat)
    Output: (B, 5)
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

        # SwiGLU Encoder: n_feat → d_hidden (gated) → d_latent
        self.enc1 = nn.Linear(n_feat, 2 * d_hidden)      # gate + value combined
        self.enc_norm1 = nn.LayerNorm(d_hidden)
        self.enc_drop1 = nn.Dropout(dropout)
        self.enc2 = nn.Linear(d_hidden, d_latent)
        self.enc_norm2 = nn.LayerNorm(d_latent)

        # SwiGLU Decoder: d_latent → d_hidden (gated) → n_feat
        self.dec1 = nn.Linear(d_latent, 2 * d_hidden)
        self.dec2 = nn.Linear(d_hidden, n_feat)

        # Cross-sym attention (same as baseline)
        self.attn = nn.MultiheadAttention(d_latent, num_heads=n_heads,
                                          dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_latent)

        # SwiGLU Prediction head: (d_latent + n_feat) → d_hidden (gated) → d_hidden//2 → 1
        in_dim = d_latent + n_feat
        self.ph1 = nn.Linear(in_dim, 2 * d_hidden)
        self.ph_norm1 = nn.LayerNorm(d_hidden)
        self.ph_drop1 = nn.Dropout(dropout)
        self.ph2 = nn.Linear(d_hidden, 2 * (d_hidden // 2))
        self.ph_drop2 = nn.Dropout(dropout)
        self.ph3 = nn.Linear(d_hidden // 2, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        h = swiglu(self.enc1(x))
        h = self.enc_drop1(self.enc_norm1(h))
        return self.enc_norm2(self.enc2(h))

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec2(swiglu(self.dec1(z)))

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
        recon = self._decode(latent)

        z = latent.view(B, S, -1)
        z_attn, _ = self.attn(z, z, z)
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
# Architecture 5: V5_MarketAdaLNCrossSym (MASTER-inspired AdaLN)
# ===========================================================================

class MarketAdaLNCrossSym(nn.Module):
    """Cross-sym attention with MASTER-style market-state AdaLN conditioning.

    Papers:
    - MASTER (Zhang et al., AAAI 2024): market-guided feature selection
    - DiT (Peebles & Xie, ICCV 2023): AdaLN-Zero for stable conditioning

    Flow:
    1. Per-sym encode: z_i = encoder(x_i)     (B*5, d_latent)
    2. Cross-sym attention: z_attn             (B, 5, d_latent)
    3. Market context: ctx = mean(z_attn, dim=1)   (B, d_latent)
    4. AdaLN conditioning: compute scale, shift from ctx
       [scale, shift] = MLP(ctx)               (B, 2*d_latent)
       z_conditioned_i = scale * LN(z_attn_i) + shift
    5. Predict from z_conditioned

    Input:  (B, 5, n_feat)
    Output: (B, 5)
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

        # Cross-sym attention
        self.attn = nn.MultiheadAttention(d_latent, num_heads=n_heads,
                                          dropout=dropout, batch_first=True)

        # AdaLN: market context → scale and shift for each sym's latent
        # AdaLN-Zero: init to output (1, 0) = identity transform
        self.adaln_norm = nn.LayerNorm(d_latent, elementwise_affine=False)
        self.adaln_proj = nn.Linear(d_latent, 2 * d_latent)

        # Prediction head
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
        # AdaLN-Zero: init scale to 0 (so initial output = 0*LN(z)+0 = 0)
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

        z = latent.view(B, S, -1)                          # (B, 5, d_latent)
        z_attn, _ = self.attn(z, z, z)                     # (B, 5, d_latent)
        z_res = z + z_attn                                  # pre-AdaLN residual

        # Market context: mean across syms
        ctx = z_res.mean(dim=1)                             # (B, d_latent)

        # AdaLN conditioning
        adaln_out = self.adaln_proj(ctx)                    # (B, 2*d_latent)
        scale, shift = adaln_out.chunk(2, dim=-1)           # (B, d_latent) each
        # Broadcast over sym dim and apply
        scale = scale.unsqueeze(1)  # (B, 1, d_latent)
        shift = shift.unsqueeze(1)
        z_normed = self.adaln_norm(z_res)                   # (B, 5, d_latent)
        z_conditioned = (1 + scale) * z_normed + shift      # (B, 5, d_latent)

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
# Architecture 6: V5_ContentMoECrossSym (Content-based Sparse MoE)
# ===========================================================================

class ContentMoECrossSym(nn.Module):
    """Cross-sym + Content-based Mixture of Experts in prediction path.

    Sym-agnostic routing: gating based on feature content z, NOT sym index.
    n_experts=3, top_k=2 routing (load-balanced).

    Architecture:
    - Encoder → Cross-sym attn → z_combined (same as baseline)
    - MoE layer: gate(z) → top-2 experts → weighted sum
    - Pred head on MoE output

    Input:  (B, 5, n_feat)
    Output: (B, 5)
    """
    def __init__(self, n_sym: int = 5, n_feat: int = 259,
                 d_latent: int = 32, d_hidden: int = 128,
                 n_heads: int = 4, dropout: float = 0.30,
                 recon_alpha: float = 0.3, noise_sigma: float = 0.035,
                 n_experts: int = 3, top_k: int = 2, d_expert: int = 64,
                 moe_load_alpha: float = 0.01):
        super().__init__()
        assert d_latent % n_heads == 0
        assert top_k <= n_experts
        self.n_sym = n_sym
        self.n_feat = n_feat
        self.recon_alpha = recon_alpha
        self.noise_sigma = noise_sigma
        self.n_experts = n_experts
        self.top_k = top_k
        self.moe_load_alpha = moe_load_alpha
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
        self.attn_norm = nn.LayerNorm(d_latent)

        # MoE gating: z → logits over n_experts
        self.gate = nn.Linear(d_latent, n_experts, bias=False)
        # Expert networks: each small 2-layer MLP
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_latent, d_expert), nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_expert, d_latent),
            ) for _ in range(n_experts)
        ])

        # Prediction head from MoE output
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

    def _moe_forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """z: (N, d_latent) where N = B*5 → (N, d_latent), aux_loss scalar."""
        N, D = z.shape
        K = self.top_k
        E = self.n_experts

        # Gating logits
        logits = self.gate(z)                               # (N, E)
        gate_probs = F.softmax(logits, dim=-1)              # (N, E)

        # Top-K selection
        topk_vals, topk_idx = gate_probs.topk(K, dim=-1)   # (N, K)
        topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)  # renormalize

        # Compute expert outputs for top-k experts
        out = torch.zeros_like(z)
        for k in range(K):
            expert_idx = topk_idx[:, k]                     # (N,)
            weights = topk_vals[:, k].unsqueeze(-1)         # (N, 1)
            for e in range(E):
                mask = (expert_idx == e)                    # (N,)
                if mask.any():
                    out[mask] += weights[mask] * self.experts[e](z[mask])

        # Load-balancing auxiliary loss
        # f_e = fraction of tokens dispatched to expert e
        # P_e = mean gating probability to expert e
        f = torch.zeros(E, device=z.device)
        for k in range(K):
            for e in range(E):
                f[e] += (topk_idx[:, k] == e).float().mean()
        f = f / K  # normalize to fraction
        P = gate_probs.mean(dim=0)  # (E,)
        aux_loss = self.moe_load_alpha * E * (f * P).sum()

        return out, aux_loss

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
        z_combined = self.attn_norm(z + z_attn)
        z_flat = z_combined.reshape(B * S, -1)

        # MoE routing
        z_moe, aux_loss = self._moe_forward(z_flat)

        pred = self.pred_head(torch.cat([z_moe, x_noisy], dim=-1))
        pred = pred.squeeze(-1).reshape(B, S)

        if self.training:
            self._last_recon_loss = F.mse_loss(recon, x_flat)
            self._last_aux_loss = aux_loss
        else:
            self._last_recon_loss = None
            self._last_aux_loss = None
        return pred


# ===========================================================================
# Architecture 7: V5_DiffAttnSwiGLU (Diff-Attn + SwiGLU, kitchen-sink P1 combo)
# ===========================================================================

class DiffAttnSwiGLU(nn.Module):
    """Combines Differential Attention + SwiGLU FFN.

    This is the "go for it" combo that applies both key improvements together:
    1. Diff-Attn in cross-sym attention (noise cancellation)
    2. SwiGLU in encoder and pred head (gated activation)

    Expected: additive improvement over either alone.
    Complexity: still low — both changes are drop-in.

    Input:  (B, 5, n_feat)
    Output: (B, 5)
    """
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

        # SwiGLU Encoder
        self.enc1 = nn.Linear(n_feat, 2 * d_hidden)
        self.enc_norm1 = nn.LayerNorm(d_hidden)
        self.enc_drop1 = nn.Dropout(dropout)
        self.enc2 = nn.Linear(d_hidden, d_latent)
        self.enc_norm2 = nn.LayerNorm(d_latent)

        # Decoder (simple, no SwiGLU needed for recon)
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, n_feat),
        )

        # Diff-Attn projections
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

        # SwiGLU Prediction head
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

ARCH_CLASSES_V5 = {
    "v5_diff_attn":    DiffAttnCrossSym,
    "v5_glu_v_attn":   GLUVCrossSym,
    "v5_mask_ssl":     MaskSSLCrossSym,
    "v5_swiglu_sae":   SwiGLUSAECrossSym,
    "v5_market_adaln": MarketAdaLNCrossSym,
    "v5_content_moe":  ContentMoECrossSym,
    "v5_diff_swiglu":  DiffAttnSwiGLU,
}

ARCH_DEFAULTS_V5 = {
    "v5_diff_attn":    dict(d_latent=32, d_hidden=128, n_heads=4, dropout=0.30,
                            recon_alpha=0.3, noise_sigma=0.035, lambda_init=0.2),
    "v5_glu_v_attn":   dict(d_latent=32, d_hidden=128, n_heads=4, dropout=0.30,
                            recon_alpha=0.3, noise_sigma=0.035),
    "v5_mask_ssl":     dict(d_latent=32, d_hidden=128, n_heads=4, dropout=0.30,
                            recon_alpha=0.3, noise_sigma=0.035, mask_p=0.30, mask_alpha=0.5),
    "v5_swiglu_sae":   dict(d_latent=32, d_hidden=128, n_heads=4, dropout=0.30,
                            recon_alpha=0.3, noise_sigma=0.035),
    "v5_market_adaln": dict(d_latent=32, d_hidden=128, n_heads=4, dropout=0.30,
                            recon_alpha=0.3, noise_sigma=0.035),
    "v5_content_moe":  dict(d_latent=32, d_hidden=128, n_heads=4, dropout=0.30,
                            recon_alpha=0.3, noise_sigma=0.035,
                            n_experts=3, top_k=2, d_expert=64, moe_load_alpha=0.01),
    "v5_diff_swiglu":  dict(d_latent=32, d_hidden=128, n_heads=4, dropout=0.30,
                            recon_alpha=0.3, noise_sigma=0.035, lambda_init=0.2),
}


def build_model_v5(arch: str, n_sym: int = 5, n_feat: int = 259, **overrides):
    """Factory for v5 models.
    
    Example:
        model = build_model_v5("v5_diff_attn")
        model = build_model_v5("v5_diff_attn", d_latent=48, n_heads=4)
    """
    cls = ARCH_CLASSES_V5[arch]
    kwargs = dict(ARCH_DEFAULTS_V5[arch])
    kwargs.update(overrides)
    return cls(n_sym=n_sym, n_feat=n_feat, **kwargs)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
```

---

## Trainer Integration Notes

### Loss computation for each architecture

```python
# In training loop (matches train_v4.py pattern):

pred = model(x_batch)               # (B, 5)
pred_loss = loss_fn(pred, y_batch)   # e.g., MSE or PnL surrogate

total_loss = pred_loss

# Reconstruction loss (all models)
if model._last_recon_loss is not None:
    total_loss = total_loss + model.recon_alpha * model._last_recon_loss

# Auxiliary SSL loss (v5_mask_ssl: BCE, v5_content_moe: load balance)
if model._last_aux_loss is not None:
    total_loss = total_loss + model.mask_alpha * model._last_aux_loss
    # For v5_content_moe: model._last_aux_loss is already scaled by moe_load_alpha

total_loss.backward()
```

### Key hyperparameters to keep the same as baseline (sae_cross_sym)
- `d_latent=32, d_hidden=128, n_heads=4, dropout=0.30`
- `recon_alpha=0.3, noise_sigma=0.035`
- Same optimizer: AdamW, lr=3e-4, weight_decay=1e-4
- Same scheduler: cosine with warmup
- Same batch size and step budget

### New hyperparameters for v5_mask_ssl
- `mask_p=0.30` (fraction of features to corrupt)
- `mask_alpha=0.5` (weight of BCE mask estimation loss)

### New hyperparameters for v5_diff_attn / v5_diff_swiglu
- `lambda_init=0.2` (initial lambda for differential attention, layer 1 default)
- lambda parameters learned automatically via gradient

### WandB logging additions
```python
wandb.log({
    "train/pred_loss": pred_loss.item(),
    "train/recon_loss": model._last_recon_loss.item() if model._last_recon_loss else 0,
    "train/aux_loss": model._last_aux_loss.item() if model._last_aux_loss else 0,
    "train/total_loss": total_loss.item(),
    "val/pnl": val_pnl,
})
```

---

## Experiment Priority Order for Orchestrator

**Phase 1 (start immediately, 5 seeds each):**
1. `v5_diff_attn` — Differential Attention cross-sym
2. `v5_mask_ssl` — Mask estimation SSL

**Phase 2 (after Phase 1 results, pick best):**
3. `v5_glu_v_attn` — GLU-gated values
4. `v5_market_adaln` — Market AdaLN
5. `v5_diff_swiglu` — Combo (if diff_attn shows promise)

**Phase 3 (ablation if needed):**
6. `v5_swiglu_sae` — SwiGLU alone
7. `v5_content_moe` — MoE (higher variance, check if overfits)
