"""V4 NN variants exploring whether sae_mlp 259d (+34.15) can be pushed further.

All models take input (B, 5, n_feat) and produce per-sym predictions.

V1. MultiHorizonSAEMLP   – shared encoder + decoder + 5 horizon heads (multi-task)
V2. SAEMLPCrossSym       – per-sym encoder + 1 cross-sym attention + per-sym head
V3. SAEMLPWiderDeeper    – scale-up sanity check (d_latent=64, d_hidden=256, +1 layer)
V4. VAEMLP               – probabilistic AE (KL regularization in place of dropout)

All expose ._last_recon_loss (and ._last_kl, ._last_aux_loss) for the trainer
to combine with the prediction loss.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# V1. MultiHorizonSAEMLP
# ---------------------------------------------------------------------------

class MultiHorizonSAEMLP(nn.Module):
    """SAE-MLP with 5 horizon heads sharing a single encoder.

    forward(x) returns:
      pred_h60 (B, 5)            — the head used for evaluation (main horizon)
    Multi-horizon supervision is exposed via .forward_multi(x) returning
      preds_list  : list of 5 tensors (B, 5)   for h ∈ horizons
    """
    horizons = (5, 10, 20, 40, 60)

    def __init__(self, n_sym: int = 5, n_feat: int = 259,
                 d_latent: int = 32, d_hidden: int = 128,
                 dropout: float = 0.30, recon_alpha: float = 0.3,
                 noise_sigma: float = 0.035):
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
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_latent + n_feat, d_hidden), nn.LayerNorm(d_hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_hidden, d_hidden // 2), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_hidden // 2, 1),
            ) for _ in self.horizons
        ])
        self._last_recon_loss = None
        self._last_preds_multi = None  # list of (B, 5) tensors, populated in forward
        self.h60_idx = self.horizons.index(60)

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
        recon  = self.decoder(latent)
        cat_in = torch.cat([latent, x_noisy], dim=-1)

        preds_multi = []
        for head in self.heads:
            p = head(cat_in).squeeze(-1).reshape(B, S)
            preds_multi.append(p)
        self._last_preds_multi = preds_multi

        if self.training:
            self._last_recon_loss = F.mse_loss(recon, x_flat)
        else:
            self._last_recon_loss = None
        return preds_multi[self.h60_idx]  # main horizon for eval


# ---------------------------------------------------------------------------
# V2. SAEMLPCrossSym
# ---------------------------------------------------------------------------

class SAEMLPCrossSym(nn.Module):
    """Per-sym SAE-MLP encoder + 1 cross-sym multi-head attention + per-sym head.

    Encoder/decoder are shared across syms (permutation-equivariant).
    The attention layer fuses information across the 5 syms in a group.
    Returns (B, 5) and exposes ._last_recon_loss.
    """
    def __init__(self, n_sym: int = 5, n_feat: int = 259,
                 d_latent: int = 32, d_hidden: int = 128,
                 n_heads: int = 4, dropout: float = 0.30,
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
        # Cross-sym attention block (post-norm pre-residual, simple)
        # adapt d_latent to multiple of n_heads
        assert d_latent % n_heads == 0, f"d_latent {d_latent} not divisible by n_heads {n_heads}"
        self.attn = nn.MultiheadAttention(d_latent, num_heads=n_heads,
                                          dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_latent)

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
        recon  = self.decoder(latent)

        # cross-sym attention: each sym attends to all 5 syms in its group
        z = latent.view(B, S, -1)          # (B, 5, d_latent)
        z_attn, _ = self.attn(z, z, z)     # (B, 5, d_latent)
        z_combined = self.attn_norm(z + z_attn)
        z_flat = z_combined.reshape(B * S, -1)

        cat_in = torch.cat([z_flat, x_noisy], dim=-1)
        pred = self.pred_head(cat_in).squeeze(-1).reshape(B, S)

        if self.training:
            self._last_recon_loss = F.mse_loss(recon, x_flat)
        else:
            self._last_recon_loss = None
        return pred


# ---------------------------------------------------------------------------
# V3. SAEMLPWiderDeeper
# ---------------------------------------------------------------------------

class SAEMLPWiderDeeper(nn.Module):
    """Same architecture template as sae_mlp but wider & deeper.

    d_hidden=256 (vs 128), d_latent=64 (vs 32), +1 extra layer in encoder & head.
    Sanity check: does scaling up help, or does it overfit?
    """
    def __init__(self, n_sym: int = 5, n_feat: int = 259,
                 d_latent: int = 64, d_hidden: int = 256,
                 dropout: float = 0.30, recon_alpha: float = 0.3,
                 noise_sigma: float = 0.035):
        super().__init__()
        self.n_sym = n_sym
        self.n_feat = n_feat
        self.recon_alpha = recon_alpha
        self.noise_sigma = noise_sigma
        d_mid = d_hidden // 2

        # encoder: n_feat → d_hidden → d_mid → d_latent  (3 layers, vs sae_mlp's 2)
        self.encoder = nn.Sequential(
            nn.Linear(n_feat, d_hidden), nn.LayerNorm(d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_mid),  nn.LayerNorm(d_mid),  nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_mid,    d_latent), nn.LayerNorm(d_latent), nn.GELU(),
        )
        # decoder: d_latent → d_mid → d_hidden → n_feat
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, d_mid), nn.GELU(),
            nn.Linear(d_mid, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, n_feat),
        )
        # pred head: 1 more hidden layer than sae_mlp
        self.pred_head = nn.Sequential(
            nn.Linear(d_latent + n_feat, d_hidden), nn.LayerNorm(d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_mid), nn.LayerNorm(d_mid), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_mid, d_mid // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_mid // 2, 1),
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
        recon  = self.decoder(latent)
        pred = self.pred_head(torch.cat([latent, x_noisy], dim=-1)).squeeze(-1).reshape(B, S)
        if self.training:
            self._last_recon_loss = F.mse_loss(recon, x_flat)
        else:
            self._last_recon_loss = None
        return pred


# ---------------------------------------------------------------------------
# V4. VAEMLP
# ---------------------------------------------------------------------------

class VAEMLP(nn.Module):
    """VAE-style supervised AE: encoder outputs (μ, log σ²); reparam → latent.

    KL divergence (to N(0,1)) replaces dropout as the latent regularizer.
    Exposes ._last_recon_loss and ._last_kl for the trainer.
    """
    def __init__(self, n_sym: int = 5, n_feat: int = 259,
                 d_latent: int = 32, d_hidden: int = 128,
                 dropout: float = 0.30, recon_alpha: float = 0.3,
                 noise_sigma: float = 0.035, beta_kl: float = 0.001):
        super().__init__()
        self.n_sym = n_sym
        self.n_feat = n_feat
        self.recon_alpha = recon_alpha
        self.noise_sigma = noise_sigma
        self.beta_kl = beta_kl

        # shared encoder trunk, then split into μ and logvar heads
        self.enc_trunk = nn.Sequential(
            nn.Linear(n_feat, d_hidden), nn.LayerNorm(d_hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.enc_mu     = nn.Linear(d_hidden, d_latent)
        self.enc_logvar = nn.Linear(d_hidden, d_latent)
        self.latent_norm = nn.LayerNorm(d_latent)

        self.decoder = nn.Sequential(
            nn.Linear(d_latent, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, n_feat),
        )
        self.pred_head = nn.Sequential(
            nn.Linear(d_latent + n_feat, d_hidden),
            nn.LayerNorm(d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1),
        )
        self._last_recon_loss = None
        self._last_kl = None

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # initialize logvar head to small values (start near-deterministic)
        nn.init.zeros_(self.enc_logvar.weight)
        nn.init.constant_(self.enc_logvar.bias, -3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, F_ = x.shape
        x_flat = x.reshape(B * S, F_)
        if self.training and self.noise_sigma > 0:
            x_noisy = x_flat + torch.randn_like(x_flat) * self.noise_sigma
        else:
            x_noisy = x_flat
        h = self.enc_trunk(x_noisy)
        mu     = self.enc_mu(h)
        logvar = self.enc_logvar(h).clamp(min=-10.0, max=5.0)  # stability
        if self.training:
            std = (0.5 * logvar).exp()
            z_raw = mu + std * torch.randn_like(std)
        else:
            z_raw = mu
        z = self.latent_norm(z_raw)
        recon = self.decoder(z)
        pred  = self.pred_head(torch.cat([z, x_noisy], dim=-1)).squeeze(-1).reshape(B, S)

        if self.training:
            self._last_recon_loss = F.mse_loss(recon, x_flat)
            # KL(N(μ,σ²) || N(0,1)) per-dim, mean over batch
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
            self._last_kl = kl
        else:
            self._last_recon_loss = None
            self._last_kl = None
        return pred


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

ARCH_CLASSES_V4 = {
    "multi_h_sae":    MultiHorizonSAEMLP,
    "sae_cross_sym":  SAEMLPCrossSym,
    "sae_wider":      SAEMLPWiderDeeper,
    "vae_mlp":        VAEMLP,
}

ARCH_DEFAULTS_V4 = {
    # mirror sae_mlp HPs: d_latent=32, d_hidden=128 (current SOTA)
    "multi_h_sae":   dict(d_latent=32, d_hidden=128, dropout=0.30,
                          recon_alpha=0.3, noise_sigma=0.035),
    "sae_cross_sym": dict(d_latent=32, d_hidden=128, n_heads=4, dropout=0.30,
                          recon_alpha=0.3, noise_sigma=0.035),
    # explicit scale-up
    "sae_wider":     dict(d_latent=64, d_hidden=256, dropout=0.30,
                          recon_alpha=0.3, noise_sigma=0.035),
    # KL replaces dropout-as-noise pressure on latent
    "vae_mlp":       dict(d_latent=32, d_hidden=128, dropout=0.30,
                          recon_alpha=0.3, noise_sigma=0.035, beta_kl=0.001),
}


def build_model_v4(arch: str, n_sym: int = 5, n_feat: int = 259, **overrides):
    cls = ARCH_CLASSES_V4[arch]
    kwargs = dict(ARCH_DEFAULTS_V4[arch])
    kwargs.update(overrides)
    return cls(n_sym=n_sym, n_feat=n_feat, **kwargs)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
