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

ARCH_CLASSES = {
    "cross_mlp":  CrossConcatMLP,
    "sym_attn":   SymAttentionModel,
    "hybrid":     HybridCrossAttn,
}

ARCH_DEFAULTS = {
    "cross_mlp": dict(hidden=(512, 256, 128, 64), dropout=0.10),
    "sym_attn":  dict(d_token=64, n_heads=4, depth=2, dropout=0.10),
    "hybrid":    dict(d_enc=128, n_heads=4, dropout=0.10),
}


def build_model(arch: str, n_sym: int = 5, n_feat: int = 359) -> nn.Module:
    cls     = ARCH_CLASSES[arch]
    kwargs  = ARCH_DEFAULTS[arch]
    return cls(n_sym=n_sym, n_feat=n_feat, **kwargs)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
