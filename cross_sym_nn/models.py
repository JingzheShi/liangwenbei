"""Cross-sym NN architectures.

Three models, all taking input (B, 5, n_feat) and producing (B, 5) predictions.

A. CrossConcatMLP  – flatten all 5 sym features, deep MLP
B. SymAttentionModel – per-sym token + sym embedding + multi-head self-attention
C. HybridCrossAttn – per-sym MLP encoder + 1 cross-sym attention + mean-others head
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared building block
# ---------------------------------------------------------------------------

class SDPABlock(nn.Module):
    """Pre-norm Transformer encoder block using F.scaled_dot_product_attention."""
    def __init__(self, d_model: int, n_heads: int, ffn_mult: float = 2.0, dropout: float = 0.10):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv   = nn.Linear(d_model, 3 * d_model)
        self.proj  = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        h = int(d_model * ffn_mult)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, h), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h, d_model),
        )
        self.drop_p = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        y = self.norm1(x)
        qkv = self.qkv(y).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.drop_p if self.training else 0.0
        )
        attn = attn.transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(attn)
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# A. CrossConcatMLP
# ---------------------------------------------------------------------------

class CrossConcatMLP(nn.Module):
    """Concatenate all 5-sym features and run a deep MLP.

    Input:  (B, 5, n_feat) -> flatten -> (B, 5*n_feat)
    Output: (B, 5)
    """
    def __init__(self, n_sym: int = 5, n_feat: int = 359,
                 hidden=(512, 256, 128, 64), dropout: float = 0.10):
        super().__init__()
        in_dim = n_sym * n_feat
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, n_sym)]
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 5, n_feat)
        B = x.size(0)
        return self.net(x.reshape(B, -1))  # (B, 5)


# ---------------------------------------------------------------------------
# B. SymAttentionModel
# ---------------------------------------------------------------------------

class SymAttentionModel(nn.Module):
    """Each sym is a token; multi-head self-attention across syms.

    Input:  (B, 5, n_feat)
    Output: (B, 5)
    """
    def __init__(self, n_sym: int = 5, n_feat: int = 359,
                 d_token: int = 64, n_heads: int = 4, depth: int = 2,
                 dropout: float = 0.10):
        super().__init__()
        self.input_proj = nn.Linear(n_feat, d_token)
        self.sym_emb    = nn.Parameter(torch.randn(n_sym, d_token) * 0.02)
        self.blocks     = nn.ModuleList([
            SDPABlock(d_token, n_heads, ffn_mult=2.0, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, 1)

        nn.init.kaiming_normal_(self.input_proj.weight, nonlinearity="relu")
        nn.init.zeros_(self.input_proj.bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 5, n_feat)
        h = self.input_proj(x)        # (B, 5, d_token)
        h = h + self.sym_emb          # broadcast sym embeddings
        for blk in self.blocks:
            h = blk(h)                # (B, 5, d_token)
        h = self.norm(h)
        return self.head(h).squeeze(-1)  # (B, 5)


# ---------------------------------------------------------------------------
# C. HybridCrossAttn
# ---------------------------------------------------------------------------

class HybridCrossAttn(nn.Module):
    """Per-sym MLP encoder + 1 cross-sym self-attention + mean-others head.

    Input:  (B, 5, n_feat)
    Output: (B, 5)
    """
    def __init__(self, n_sym: int = 5, n_feat: int = 359,
                 d_enc: int = 128, n_heads: int = 4, dropout: float = 0.10):
        super().__init__()
        self.n_sym = n_sym
        self.sym_encoder = nn.Sequential(
            nn.Linear(n_feat, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, d_enc),  nn.LayerNorm(d_enc), nn.GELU(),
        )
        self.cross_attn = SDPABlock(d_enc, n_heads, ffn_mult=2.0, dropout=dropout)
        self.norm       = nn.LayerNorm(d_enc)
        self.head       = nn.Sequential(
            nn.Linear(2 * d_enc, d_enc), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_enc, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 5, n_feat)
        B = x.size(0)
        # encode each sym independently
        x_flat = x.reshape(B * self.n_sym, -1)
        h_flat = self.sym_encoder(x_flat)
        h = h_flat.reshape(B, self.n_sym, -1)   # (B, 5, d_enc)

        # cross-sym attention
        h = self.cross_attn(h)                   # (B, 5, d_enc)
        h = self.norm(h)

        # mean-pool others for each sym
        total     = h.sum(dim=1, keepdim=True)              # (B, 1, d_enc)
        mean_oth  = (total - h) / (self.n_sym - 1)          # (B, 5, d_enc)
        combined  = torch.cat([h, mean_oth], dim=-1)         # (B, 5, 2*d_enc)

        return self.head(combined).squeeze(-1)               # (B, 5)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# D. SetTransformerISAB (Top-1 cross-sym, permutation-invariant)
# ---------------------------------------------------------------------------

class MAB(nn.Module):
    """Multi-head Attention Block: MAB(X, Y) = LN(H + rFF(H)) with H=LN(X + Att(X,Y,Y))."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        h = d_model * 2
        self.ff = nn.Sequential(
            nn.Linear(d_model, h), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h, d_model),
        )
        self.drop_p = dropout

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        # X: (B, n, d), Y: (B, m, d) -> (B, n, d)
        B, n, d = X.shape
        q  = self.q_proj(X).reshape(B, n, self.n_heads, self.d_head).transpose(1, 2)
        kv = self.kv_proj(Y).reshape(B, -1, 2, self.n_heads, self.d_head)
        k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)
        h = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.drop_p if self.training else 0.0
        )
        h = h.transpose(1, 2).reshape(B, n, d)
        h = self.out_proj(h)
        X = self.norm1(X + h)
        X = self.norm2(X + self.ff(X))
        return X


class ISAB(nn.Module):
    """Induced Set Attention Block: MAB(X, MAB(I, X))."""
    def __init__(self, d_model: int, n_heads: int, n_inducing: int, dropout: float = 0.1):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, n_inducing, d_model) * 0.02)
        self.mab1 = MAB(d_model, n_heads, dropout)
        self.mab2 = MAB(d_model, n_heads, dropout)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B = X.size(0)
        I = self.I.expand(B, -1, -1)
        H = self.mab1(I, X)
        return self.mab2(X, H)


class SetTransformerISAB(nn.Module):
    """Permutation-equivariant cross-sym architecture via ISAB. No sym embedding."""
    def __init__(self, n_sym: int = 5, n_feat: int = 359,
                 d_model: int = 64, n_heads: int = 4, n_inducing: int = 4,
                 depth: int = 2, dropout: float = 0.30):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_feat, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.isab_blocks = nn.ModuleList([
            ISAB(d_model, n_heads, n_inducing, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 5, n_feat) -> (B, 5)
        h = self.input_proj(x)
        for blk in self.isab_blocks:
            h = blk(h)
        h = self.norm(h)
        return self.head(h).squeeze(-1)


# ---------------------------------------------------------------------------
# E. DeepSetsEnhanced (max + mean + std pooling)
# ---------------------------------------------------------------------------

class DeepSetsEnhanced(nn.Module):
    """DeepSets with max + mean + std pooling cross-sym aggregation."""
    def __init__(self, n_sym: int = 5, n_feat: int = 359,
                 d_phi: int = 32, d_hidden: int = 128,
                 d_rho: int = 32, dropout: float = 0.30):
        super().__init__()
        self.n_sym = n_sym
        self.phi = nn.Sequential(
            nn.Linear(n_feat, d_hidden), nn.LayerNorm(d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_phi),  nn.LayerNorm(d_phi),  nn.GELU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(3 * d_phi, d_rho), nn.LayerNorm(d_rho), nn.GELU(), nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(d_phi + d_rho, d_phi),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_phi, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 5, n_feat)
        B, S, F_ = x.shape
        x_flat = x.reshape(B * S, F_)
        h_flat = self.phi(x_flat)
        h = h_flat.reshape(B, S, -1)            # (B, 5, d_phi)
        h_max  = h.max(dim=1).values             # (B, d_phi)
        h_mean = h.mean(dim=1)                   # (B, d_phi)
        h_std  = h.std(dim=1, unbiased=False)    # (B, d_phi)
        agg    = torch.cat([h_max, h_mean, h_std], dim=-1)  # (B, 3*d_phi)
        ctx = self.rho(agg).unsqueeze(1).expand(-1, S, -1)  # (B, 5, d_rho)
        combined = torch.cat([h, ctx], dim=-1)               # (B, 5, d_phi+d_rho)
        return self.head(combined).squeeze(-1)               # (B, 5)


# ---------------------------------------------------------------------------
# F. SupervisedAutoEncoderMLP (SAE-MLP, Jane Street 2020 1st adapted)
# ---------------------------------------------------------------------------

class SupervisedAutoEncoderMLP(nn.Module):
    """SAE-MLP adapted for (B, 5, n_feat) interface.

    Per-sym independent: flatten to (B*5, n_feat) -> encode/decode/predict,
    reshape pred back to (B, 5). Reconstruction loss is exposed via .recon_loss
    attached as an attribute after each forward.
    """
    def __init__(self, n_sym: int = 5, n_feat: int = 359,
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
        self.pred_head = nn.Sequential(
            nn.Linear(d_latent + n_feat, d_hidden),
            nn.LayerNorm(d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1),
        )
        # placeholder for recon loss read by trainer after forward
        self._last_recon_loss = None
        self._last_x_clean = None
        self._last_recon = None

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 5, n_feat) -> pred (B, 5)
        B, S, F_ = x.shape
        x_flat = x.reshape(B * S, F_)
        if self.training and self.noise_sigma > 0:
            x_noisy = x_flat + torch.randn_like(x_flat) * self.noise_sigma
        else:
            x_noisy = x_flat
        latent = self.encoder(x_noisy)
        recon  = self.decoder(latent)
        pred   = self.pred_head(torch.cat([latent, x_noisy], dim=-1)).squeeze(-1)

        # cache for loss combination (trainer reads these)
        self._last_x_clean = x_flat
        self._last_recon = recon
        if self.training:
            self._last_recon_loss = F.mse_loss(recon, x_flat)
        else:
            self._last_recon_loss = None
        return pred.reshape(B, S)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

ARCH_CLASSES = {
    "cross_mlp":  CrossConcatMLP,
    "sym_attn":   SymAttentionModel,
    "hybrid":     HybridCrossAttn,
    "isab":       SetTransformerISAB,
    "deepsets":   DeepSetsEnhanced,
    "sae_mlp":    SupervisedAutoEncoderMLP,
}

ARCH_DEFAULTS = {
    # original (epoch-level training) — kept for backward compat
    "cross_mlp": dict(hidden=(512, 256, 128, 64), dropout=0.10),
    "sym_attn":  dict(d_token=64, n_heads=4, depth=2, dropout=0.10),
    "hybrid":    dict(d_enc=128, n_heads=4, dropout=0.10),
    # new (small + strong regularization for step-level training)
    "isab":      dict(d_model=64, n_heads=4, n_inducing=4, depth=2, dropout=0.30),
    "deepsets":  dict(d_phi=32, d_hidden=128, d_rho=32, dropout=0.30),
    "sae_mlp":   dict(d_latent=32, d_hidden=128, dropout=0.30,
                      recon_alpha=0.3, noise_sigma=0.035),
}


def build_model(arch: str, n_sym: int = 5, n_feat: int = 359, **overrides) -> nn.Module:
    cls    = ARCH_CLASSES[arch]
    kwargs = dict(ARCH_DEFAULTS[arch])
    kwargs.update(overrides)
    return cls(n_sym=n_sym, n_feat=n_feat, **kwargs)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
