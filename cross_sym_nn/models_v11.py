"""V11 architectures: 5 new transformer variants targeting > +40 PnL.

All are sym-agnostic, keep SAE recon loss (alpha=0.3, noise_sigma=0.035),
use LayerNorm/RMSNorm + dropout 0.30, and live in the 100-300k param range.

V11.A — V11_HierarchicalDualAxis  (feature-axis + sym-axis MHSA)
V11.B — V11_IterativeRefinement   (3-step shared-weight refinement)
V11.C — V11_MiniLlama              (Pre-LN + RMSNorm + SwiGLU FFN, depth=2)
V11.D — V11_EncoderDecoder         (encoder self-attn + decoder cross-attn over 5 sym queries)
V11.E — V11_MoTE                   (4 transformer experts + per-sym soft gating)
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models_v6 import (
    _build_sae_encoder, _build_sae_decoder, _build_pred_head, _kaiming_init,
)


# ===========================================================================
# Shared helpers
# ===========================================================================

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm.to(x.dtype)) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_in, d_hidden, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(d_in, d_hidden, bias=False)
        self.w2 = nn.Linear(d_in, d_hidden, bias=False)
        self.w3 = nn.Linear(d_hidden, d_in, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))


# ===========================================================================
# V11.A — Hierarchical Dual-Axis Transformer (TLOB-inspired)
# ===========================================================================

class V11_HierarchicalDualAxis(nn.Module):
    """Two-axis transformer.

    Axis 1 — feature-axis (per-sym): project 319d → n_groups feature tokens,
              run 1 layer MHSA over n_groups tokens.
    Axis 2 — sym-axis: pool feature tokens → 1 sym-level d_latent vector,
              run 1 layer MHSA across 5 sym tokens.

    SAE recon path on per-sym MLP encoder.
    """
    def __init__(self, n_sym: int = 5, n_feat: int = 319,
                 d_latent: int = 32, d_hidden: int = 128,
                 n_groups: int = 16, d_token: int = 16,
                 n_heads_feat: int = 4, n_heads_sym: int = 4,
                 dropout: float = 0.30,
                 recon_alpha: float = 0.3, noise_sigma: float = 0.035):
        super().__init__()
        assert d_token % n_heads_feat == 0
        assert d_latent % n_heads_sym == 0
        self.n_sym = n_sym
        self.n_feat = n_feat
        self.n_groups = n_groups
        self.d_token = d_token
        self.d_latent = d_latent
        self.recon_alpha = recon_alpha
        self.noise_sigma = noise_sigma
        self._last_recon_loss = None
        self._last_aux_loss = None

        # SAE for recon (per-sym path)
        self.encoder = _build_sae_encoder(n_feat, d_hidden, d_latent, dropout)
        self.decoder = _build_sae_decoder(n_feat, d_hidden, d_latent)

        # Feature-axis: project 319d -> n_groups * d_token
        self.feat_to_tokens = nn.Linear(n_feat, n_groups * d_token)
        self.token_ln = nn.LayerNorm(d_token)
        self.pos_emb = nn.Parameter(torch.randn(1, n_groups, d_token) * 0.02)

        self.feat_attn_ln = nn.LayerNorm(d_token)
        self.feat_attn = nn.MultiheadAttention(d_token, n_heads_feat,
                                               dropout=dropout, batch_first=True)
        self.feat_attn_drop = nn.Dropout(dropout)
        self.feat_ffn_ln = nn.LayerNorm(d_token)
        self.feat_ffn = nn.Sequential(
            nn.Linear(d_token, d_token * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_token * 2, d_token),
        )
        self.feat_ffn_drop = nn.Dropout(dropout)

        # Pool features → d_latent (per-sym sym-level token)
        self.feat_pool = nn.Linear(n_groups * d_token, d_latent)
        self.sym_token_ln = nn.LayerNorm(d_latent)

        # Sym-axis: 5 tokens, 1 MHSA layer
        self.sym_attn_ln = nn.LayerNorm(d_latent)
        self.sym_attn = nn.MultiheadAttention(d_latent, n_heads_sym,
                                              dropout=dropout, batch_first=True)
        self.sym_attn_drop = nn.Dropout(dropout)
        self.sym_ffn_ln = nn.LayerNorm(d_latent)
        self.sym_ffn = nn.Sequential(
            nn.Linear(d_latent, d_latent * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_latent * 2, d_latent),
        )
        self.sym_ffn_drop = nn.Dropout(dropout)
        self.sym_post_ln = nn.LayerNorm(d_latent)

        # Pred head: [latent_sae, sym_token_after_attn, raw_x]
        self.pred_head = _build_pred_head(d_latent + d_latent + n_feat, d_hidden, dropout)

        _kaiming_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, F_ = x.shape
        x_flat = x.reshape(B * S, F_)
        if self.training and self.noise_sigma > 0:
            x_noisy = x_flat + torch.randn_like(x_flat) * self.noise_sigma
        else:
            x_noisy = x_flat
        latent_sae = self.encoder(x_noisy)
        recon = self.decoder(latent_sae)

        # Feature-axis attention (per-sym)
        toks = self.feat_to_tokens(x_noisy).view(B * S, self.n_groups, self.d_token)
        toks = self.token_ln(toks) + self.pos_emb
        h = self.feat_attn_ln(toks)
        a, _ = self.feat_attn(h, h, h, need_weights=False)
        toks = toks + self.feat_attn_drop(a)
        h = self.feat_ffn_ln(toks)
        toks = toks + self.feat_ffn_drop(self.feat_ffn(h))     # (B*S, n_groups, d_token)

        pooled = toks.reshape(B * S, self.n_groups * self.d_token)
        sym_tok = self.sym_token_ln(self.feat_pool(pooled))    # (B*S, d_latent)
        sym_tok = sym_tok.view(B, S, self.d_latent)

        # Sym-axis attention (5 sym tokens)
        h = self.sym_attn_ln(sym_tok)
        a, _ = self.sym_attn(h, h, h, need_weights=False)
        sym_tok = sym_tok + self.sym_attn_drop(a)
        h = self.sym_ffn_ln(sym_tok)
        sym_tok = sym_tok + self.sym_ffn_drop(self.sym_ffn(h))
        sym_tok = self.sym_post_ln(sym_tok)
        sym_tok_flat = sym_tok.reshape(B * S, -1)

        cat_in = torch.cat([latent_sae, sym_tok_flat, x_noisy], dim=-1)
        pred = self.pred_head(cat_in).squeeze(-1).reshape(B, S)

        self._last_recon_loss = F.mse_loss(recon, x_flat) if self.training else None
        return pred


# ===========================================================================
# V11.B — Iterative Refinement Transformer
# ===========================================================================

class V11_IterativeRefinement(nn.Module):
    """Lightweight diffusion-style iterative refinement.

    1. Encode features → latent (B, 5, d_latent).
    2. Initial pred = head0(latent + x).
    3. K=3 refinement steps with SHARED weights:
         pred_emb = MLP(pred_k)              # (B, 5, d_emb)
         z' = LN(cat(latent, pred_emb))
         z' = z' + MHSA(z')                  # cross-sym
         z' = z' + FFN(z')
         delta_k = head_delta(z' + x)
         pred_{k+1} = pred_k + delta_k
    4. Final pred = pred_K.
    SAE recon loss on encoder.
    """
    def __init__(self, n_sym: int = 5, n_feat: int = 319,
                 d_latent: int = 32, d_hidden: int = 128,
                 d_pred_emb: int = 16, n_heads: int = 4,
                 n_iter: int = 3, dropout: float = 0.30,
                 recon_alpha: float = 0.3, noise_sigma: float = 0.035):
        super().__init__()
        d_inner = d_latent + d_pred_emb
        assert d_inner % n_heads == 0
        self.n_sym = n_sym
        self.n_feat = n_feat
        self.n_iter = n_iter
        self.d_latent = d_latent
        self.d_pred_emb = d_pred_emb
        self.d_inner = d_inner
        self.recon_alpha = recon_alpha
        self.noise_sigma = noise_sigma
        self._last_recon_loss = None
        self._last_aux_loss = None

        self.encoder = _build_sae_encoder(n_feat, d_hidden, d_latent, dropout)
        self.decoder = _build_sae_decoder(n_feat, d_hidden, d_latent)

        # Initial pred head: latent + x → scalar
        self.init_head = _build_pred_head(d_latent + n_feat, d_hidden, dropout)

        # Shared refinement block
        self.pred_emb = nn.Sequential(
            nn.Linear(1, d_pred_emb), nn.LayerNorm(d_pred_emb), nn.GELU(),
            nn.Linear(d_pred_emb, d_pred_emb),
        )
        self.inner_ln = nn.LayerNorm(d_inner)
        self.attn_ln = nn.LayerNorm(d_inner)
        self.attn = nn.MultiheadAttention(d_inner, n_heads, dropout=dropout, batch_first=True)
        self.attn_drop = nn.Dropout(dropout)
        self.ffn_ln = nn.LayerNorm(d_inner)
        self.ffn = nn.Sequential(
            nn.Linear(d_inner, d_inner * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_inner * 2, d_inner),
        )
        self.ffn_drop = nn.Dropout(dropout)
        # Per-step delta head (shared)
        self.delta_head = _build_pred_head(d_inner + n_feat, d_hidden, dropout)

        _kaiming_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, F_ = x.shape
        x_flat = x.reshape(B * S, F_)
        if self.training and self.noise_sigma > 0:
            x_noisy = x_flat + torch.randn_like(x_flat) * self.noise_sigma
        else:
            x_noisy = x_flat
        latent = self.encoder(x_noisy)         # (B*S, d_latent)
        recon = self.decoder(latent)
        latent_grp = latent.view(B, S, -1)

        # Initial prediction
        cat_in = torch.cat([latent, x_noisy], dim=-1)
        pred = self.init_head(cat_in).squeeze(-1).view(B, S)    # (B, S)

        # K refinement steps with shared weights
        for k in range(self.n_iter):
            p_emb = self.pred_emb(pred.unsqueeze(-1))           # (B, S, d_pred_emb)
            z = torch.cat([latent_grp, p_emb], dim=-1)          # (B, S, d_inner)
            z = self.inner_ln(z)
            h = self.attn_ln(z)
            a, _ = self.attn(h, h, h, need_weights=False)
            z = z + self.attn_drop(a)
            h = self.ffn_ln(z)
            z = z + self.ffn_drop(self.ffn(h))

            z_flat = z.reshape(B * S, -1)
            d_in = torch.cat([z_flat, x_noisy], dim=-1)
            delta = self.delta_head(d_in).squeeze(-1).view(B, S)
            pred = pred + delta

        self._last_recon_loss = F.mse_loss(recon, x_flat) if self.training else None
        return pred


# ===========================================================================
# V11.C — Mini-Llama Transformer
# ===========================================================================

class LlamaBlock(nn.Module):
    """Pre-RMSNorm + MHSA (no RoPE since 5-sym set is order-agnostic) + SwiGLU FFN."""
    def __init__(self, d_model: int, n_heads: int, d_ffn_mult: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_model * d_ffn_mult, dropout=dropout)

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.attn_drop(a)
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x


class V11_MiniLlama(nn.Module):
    """Llama-style transformer over 5 sym tokens.

    - per-sym SAE encoder → (B, 5, d_model)
    - depth=2 Llama blocks (RMSNorm + MHSA + SwiGLU FFN)
    - sym-agnostic pred head

    No RoPE (5 sym is order-invariant); no QK norm.
    """
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
            LlamaBlock(d_latent, n_heads, d_ffn_mult=d_ffn_mult, dropout=dropout)
            for _ in range(depth)
        ])
        self.post_norm = RMSNorm(d_latent)

        self.pred_head = _build_pred_head(d_latent + n_feat, d_hidden, dropout)

        _kaiming_init(self)
        # Re-init RMSNorm weights to 1 (kaiming overrides them)
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
# V11.D — Encoder-Decoder Transformer
# ===========================================================================

class V11_EncoderDecoder(nn.Module):
    """BERT-style encoder + small decoder with cross-attention.

    Encoder: per-sym MLP → 5 latent tokens → 1 self-attn layer (cross-sym).
    Decoder: 5 query tokens (initialized from a parallel projection of features;
             sym-agnostic — same projection applied to every sym) →
             self-attn + cross-attn to encoder latent + FFN.
    Pred head reads decoder output. Recon from encoder latent.
    """
    def __init__(self, n_sym: int = 5, n_feat: int = 319,
                 d_latent: int = 48, d_hidden: int = 128,
                 n_heads: int = 4, dropout: float = 0.30,
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
        self.decoder_recon = _build_sae_decoder(n_feat, d_hidden, d_latent)

        # Encoder cross-sym self-attn (1 layer)
        self.enc_attn_ln = nn.LayerNorm(d_latent)
        self.enc_attn = nn.MultiheadAttention(d_latent, n_heads, dropout=dropout,
                                              batch_first=True)
        self.enc_attn_drop = nn.Dropout(dropout)
        self.enc_ffn_ln = nn.LayerNorm(d_latent)
        self.enc_ffn = nn.Sequential(
            nn.Linear(d_latent, d_latent * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_latent * 2, d_latent),
        )
        self.enc_ffn_drop = nn.Dropout(dropout)
        self.enc_out_ln = nn.LayerNorm(d_latent)

        # Decoder query projector (sym-agnostic — different MLP from encoder)
        self.q_proj = nn.Sequential(
            nn.Linear(n_feat, d_hidden), nn.LayerNorm(d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_latent), nn.LayerNorm(d_latent),
        )

        # Decoder block: self-attn(q) + cross-attn(q, enc) + FFN
        self.dec_self_ln = nn.LayerNorm(d_latent)
        self.dec_self_attn = nn.MultiheadAttention(d_latent, n_heads, dropout=dropout,
                                                   batch_first=True)
        self.dec_self_drop = nn.Dropout(dropout)
        self.dec_cross_ln_q = nn.LayerNorm(d_latent)
        self.dec_cross_ln_k = nn.LayerNorm(d_latent)
        self.dec_cross_attn = nn.MultiheadAttention(d_latent, n_heads, dropout=dropout,
                                                    batch_first=True)
        self.dec_cross_drop = nn.Dropout(dropout)
        self.dec_ffn_ln = nn.LayerNorm(d_latent)
        self.dec_ffn = nn.Sequential(
            nn.Linear(d_latent, d_latent * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_latent * 2, d_latent),
        )
        self.dec_ffn_drop = nn.Dropout(dropout)
        self.dec_out_ln = nn.LayerNorm(d_latent)

        self.pred_head = _build_pred_head(d_latent + n_feat, d_hidden, dropout)

        _kaiming_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, F_ = x.shape
        x_flat = x.reshape(B * S, F_)
        if self.training and self.noise_sigma > 0:
            x_noisy = x_flat + torch.randn_like(x_flat) * self.noise_sigma
        else:
            x_noisy = x_flat
        latent = self.encoder(x_noisy)
        recon = self.decoder_recon(latent)

        enc = latent.view(B, S, -1)
        # Encoder block
        h = self.enc_attn_ln(enc)
        a, _ = self.enc_attn(h, h, h, need_weights=False)
        enc = enc + self.enc_attn_drop(a)
        h = self.enc_ffn_ln(enc)
        enc = enc + self.enc_ffn_drop(self.enc_ffn(h))
        enc = self.enc_out_ln(enc)

        # Decoder queries (sym-agnostic projection of features)
        q = self.q_proj(x_noisy).view(B, S, -1)        # (B, 5, d_latent)

        # Self-attn over queries
        h = self.dec_self_ln(q)
        a, _ = self.dec_self_attn(h, h, h, need_weights=False)
        q = q + self.dec_self_drop(a)
        # Cross-attn: queries attend to encoder
        hq = self.dec_cross_ln_q(q)
        hk = self.dec_cross_ln_k(enc)
        c, _ = self.dec_cross_attn(hq, hk, hk, need_weights=False)
        q = q + self.dec_cross_drop(c)
        # FFN
        h = self.dec_ffn_ln(q)
        q = q + self.dec_ffn_drop(self.dec_ffn(h))
        q = self.dec_out_ln(q)

        q_flat = q.reshape(B * S, -1)
        cat_in = torch.cat([q_flat, x_noisy], dim=-1)
        pred = self.pred_head(cat_in).squeeze(-1).reshape(B, S)

        self._last_recon_loss = F.mse_loss(recon, x_flat) if self.training else None
        return pred


# ===========================================================================
# V11.E — Mixture-of-Transformer-Experts (MoTE)
# ===========================================================================

class _SmallTransformerExpert(nn.Module):
    """1 layer cross-sym MHSA + small FFN (sym-agnostic)."""
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn_ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.attn_drop = nn.Dropout(dropout)
        self.ffn_ln = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.ffn_drop = nn.Dropout(dropout)

    def forward(self, z):       # z: (B, 5, d_model)
        h = self.attn_ln(z)
        a, _ = self.attn(h, h, h, need_weights=False)
        z = z + self.attn_drop(a)
        h = self.ffn_ln(z)
        z = z + self.ffn_drop(self.ffn(h))
        return z


class V11_MoTE(nn.Module):
    """K experts, each a 1-layer cross-sym transformer; per-sym soft gating.

    1. per-sym encoder → latent (B, 5, d_model).
    2. K experts: each produces (B, 5, d_model).
    3. Gate: per-sym (using only its own latent — sym-agnostic) → softmax(K).
    4. Mixed = sum_k gate_k * expert_k(latent) per sym.
    """
    def __init__(self, n_sym: int = 5, n_feat: int = 319,
                 d_latent: int = 32, d_hidden: int = 128,
                 n_experts: int = 4, n_heads: int = 4,
                 dropout: float = 0.30, gate_temperature: float = 1.0,
                 recon_alpha: float = 0.3, noise_sigma: float = 0.035,
                 entropy_alpha: float = 0.01):
        super().__init__()
        assert d_latent % n_heads == 0
        self.n_sym = n_sym
        self.n_feat = n_feat
        self.n_experts = n_experts
        self.d_latent = d_latent
        self.recon_alpha = recon_alpha
        self.noise_sigma = noise_sigma
        self.gate_temperature = gate_temperature
        self.entropy_alpha = entropy_alpha
        self.aux_alpha = 1.0  # train loop reads this for _last_aux_loss
        self._last_recon_loss = None
        self._last_aux_loss = None

        self.encoder = _build_sae_encoder(n_feat, d_hidden, d_latent, dropout)
        self.decoder = _build_sae_decoder(n_feat, d_hidden, d_latent)

        self.experts = nn.ModuleList([
            _SmallTransformerExpert(d_latent, n_heads, dropout)
            for _ in range(n_experts)
        ])
        # Gate: function of per-sym latent only (sym-agnostic)
        self.gate = nn.Sequential(
            nn.Linear(d_latent, d_latent), nn.LayerNorm(d_latent), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_latent, n_experts),
        )
        self.post_ln = nn.LayerNorm(d_latent)

        self.pred_head = _build_pred_head(d_latent + n_feat, d_hidden, dropout)

        _kaiming_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, F_ = x.shape
        x_flat = x.reshape(B * S, F_)
        if self.training and self.noise_sigma > 0:
            x_noisy = x_flat + torch.randn_like(x_flat) * self.noise_sigma
        else:
            x_noisy = x_flat
        latent = self.encoder(x_noisy)
        recon = self.decoder(latent)

        z = latent.view(B, S, -1)                              # (B, 5, d)
        # Each expert: (B, 5, d). Stack → (K, B, 5, d)
        out_k = torch.stack([exp(z) for exp in self.experts], dim=0)
        # Gate per sym: (B, 5, K)
        gate_logits = self.gate(z) / max(self.gate_temperature, 1e-6)
        gate_p = F.softmax(gate_logits, dim=-1)
        # mix: out[b, i] = sum_k gate_p[b, i, k] * out_k[k, b, i]
        gate_p_t = gate_p.permute(2, 0, 1).unsqueeze(-1)        # (K, B, 5, 1)
        mixed = (gate_p_t * out_k).sum(dim=0)                   # (B, 5, d)
        mixed = self.post_ln(mixed)

        # Entropy regularizer (encourage expert specialization but avoid collapse)
        if self.training and self.entropy_alpha > 0:
            ent = -(gate_p * gate_p.clamp_min(1e-9).log()).sum(dim=-1).mean()
            target_ent = math.log(self.n_experts) * 0.6
            self._last_aux_loss = self.entropy_alpha * (target_ent - ent) ** 2
        else:
            self._last_aux_loss = None

        z_flat = mixed.reshape(B * S, -1)
        cat_in = torch.cat([z_flat, x_noisy], dim=-1)
        pred = self.pred_head(cat_in).squeeze(-1).reshape(B, S)

        self._last_recon_loss = F.mse_loss(recon, x_flat) if self.training else None
        return pred


# ===========================================================================
# Registry
# ===========================================================================

ARCH_CLASSES_V11 = {
    'v11_hier':       V11_HierarchicalDualAxis,
    'v11_iter':       V11_IterativeRefinement,
    'v11_minillama':  V11_MiniLlama,
    'v11_encdec':     V11_EncoderDecoder,
    'v11_mote':       V11_MoTE,
}

ARCH_DEFAULTS_V11 = {
    'v11_hier': dict(
        d_latent=32, d_hidden=128, n_groups=16, d_token=16,
        n_heads_feat=4, n_heads_sym=4, dropout=0.30,
        recon_alpha=0.3, noise_sigma=0.035,
    ),
    'v11_iter': dict(
        d_latent=32, d_hidden=128, d_pred_emb=16, n_heads=4,
        n_iter=3, dropout=0.30,
        recon_alpha=0.3, noise_sigma=0.035,
    ),
    'v11_minillama': dict(
        d_latent=64, d_hidden=128, n_heads=4, depth=2,
        d_ffn_mult=2, dropout=0.30,
        recon_alpha=0.3, noise_sigma=0.035,
    ),
    'v11_encdec': dict(
        d_latent=48, d_hidden=128, n_heads=4, dropout=0.30,
        recon_alpha=0.3, noise_sigma=0.035,
    ),
    'v11_mote': dict(
        d_latent=32, d_hidden=128, n_experts=4, n_heads=4,
        dropout=0.30, gate_temperature=1.0,
        recon_alpha=0.3, noise_sigma=0.035,
        entropy_alpha=0.01,
    ),
}


def build_model_v11(arch: str, n_sym: int = 5, n_feat: int = 319, **overrides):
    cls = ARCH_CLASSES_V11[arch]
    kwargs = dict(ARCH_DEFAULTS_V11[arch])
    kwargs.update(overrides)
    return cls(n_sym=n_sym, n_feat=n_feat, **kwargs)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == '__main__':
    torch.manual_seed(0)
    B, S, F_ = 4, 5, 319
    x = torch.randn(B, S, F_)
    for arch in ARCH_CLASSES_V11.keys():
        m = build_model_v11(arch)
        m.eval()
        with torch.no_grad():
            y = m(x)
        m.train()
        y = m(x)
        n = count_params(m)
        recon = float(m._last_recon_loss) if m._last_recon_loss is not None else float('nan')
        aux = float(m._last_aux_loss) if m._last_aux_loss is not None else float('nan')
        print(f'  {arch:16s} n_params={n:>8d}  out={tuple(y.shape)}  '
              f'recon={recon:.4f}  aux={aux:.4f}')
