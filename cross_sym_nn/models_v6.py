"""V6 NN architectures: 5 radical paradigm shifts beyond standard MHSA tweaks.

Baseline to beat: sae_cross_sym (per-sym SAE-MLP + 1 cross-sym MHSA + per-sym head)
  test_pnl = 36.35 ± 0.67 (5-seed)
  n_params ≈ 125,604

V5 has tried (all underperformed):
  v1_deeper2/3 (stacked attn), v2_gated, v5_diff_attn (borderline), v5_glu_v_attn,
  v5_market_adaln, v5_diff_swiglu, v3_wider.

V6 paradigm-shifts:
  V6.A. GATCrossSym       — explicit edge-weight GATv2 (5-node complete graph)
  V6.B. PairwiseInteract  — explicit (sym_i - sym_j) pair embeddings
  V6.C. CrossFeatAttn     — feature-level attention (FT-Transformer style)
  V6.D. Bottleneck        — information bottleneck for shared cross-sym latent
  V6.E. MoE               — sigmoid-gated mixture of experts per-sym

All keep the sae_mlp encoder+decoder template (same recon loss alpha=0.3,
noise_sigma=0.035), 259d input, sym-agnostic (no per-sym params).

Each model exposes:
  ._last_recon_loss  (MSE recon, or None at eval)
  ._last_aux_loss    (None for v6; reserved for compatibility with train_v5 loop)
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Shared SAE encoder/decoder building block (same as sae_cross_sym baseline)
# ===========================================================================

def _build_sae_encoder(n_feat: int, d_hidden: int, d_latent: int, dropout: float):
    return nn.Sequential(
        nn.Linear(n_feat, d_hidden), nn.LayerNorm(d_hidden), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(d_hidden, d_latent), nn.LayerNorm(d_latent), nn.GELU(),
    )

def _build_sae_decoder(n_feat: int, d_hidden: int, d_latent: int):
    return nn.Sequential(
        nn.Linear(d_latent, d_hidden), nn.GELU(),
        nn.Linear(d_hidden, n_feat),
    )

def _build_pred_head(d_in: int, d_hidden: int, dropout: float):
    return nn.Sequential(
        nn.Linear(d_in, d_hidden), nn.LayerNorm(d_hidden), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(d_hidden, d_hidden // 2), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(d_hidden // 2, 1),
    )

def _kaiming_init(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ===========================================================================
# V6.A. GAT cross-sym (GATv2 on 5-node complete graph)
# ===========================================================================

class GATv2Layer(nn.Module):
    """GATv2 (Brody et al., 2022) for a fully-connected graph of S nodes.

    For each ordered pair (i, j):
        e_ij = a^T LeakyReLU(W_l h_i + W_r h_j)
    alpha_ij = softmax_j(e_ij)
    h'_i = sum_j alpha_ij W_v h_j

    Multi-head with concatenation across heads.
    Self-loops are included (j may equal i).
    """
    def __init__(self, d_in: int, d_out: int, n_heads: int = 4,
                 dropout: float = 0.3, negative_slope: float = 0.2):
        super().__init__()
        assert d_out % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_out // n_heads
        self.W_l = nn.Linear(d_in, d_out, bias=True)
        self.W_r = nn.Linear(d_in, d_out, bias=False)
        self.W_v = nn.Linear(d_in, d_out, bias=False)
        # attention vector per head
        self.a = nn.Parameter(torch.empty(1, 1, 1, n_heads, self.d_head))
        nn.init.xavier_uniform_(self.a, gain=1.0)
        self.dropout = nn.Dropout(dropout)
        self.negative_slope = negative_slope

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, S, d_in) → (B, S, d_out)."""
        B, S, _ = h.shape
        H = self.n_heads
        D = self.d_head
        hl = self.W_l(h).view(B, S, 1, H, D)        # left  (B, S=i,  1, H, D)
        hr = self.W_r(h).view(B, 1, S, H, D)        # right (B,  1, S=j, H, D)
        hv = self.W_v(h).view(B, 1, S, H, D)        # values (B, 1, S=j, H, D)

        # GATv2 trick: nonlinearity BEFORE the attention dot product
        e = F.leaky_relu(hl + hr, negative_slope=self.negative_slope)
        e = (e * self.a).sum(dim=-1)                # (B, S=i, S=j, H)
        alpha = F.softmax(e, dim=2)                 # softmax over j (neighbors+self)
        alpha = self.dropout(alpha)

        # weighted sum: out_i = sum_j alpha_ij * hv_j
        # alpha (B, S, S, H) → (B, S, S, H, 1); hv (B, 1, S, H, D)
        out = (alpha.unsqueeze(-1) * hv).sum(dim=2) # (B, S=i, H, D)
        return out.reshape(B, S, H * D)


class V6_GATCrossSym(nn.Module):
    """SAE-MLP per-sym + GATv2 cross-sym + per-sym head.

    Replaces vanilla MHSA with explicit edge-weight GATv2. Includes residual.
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
        self._last_recon_loss = None
        self._last_aux_loss = None

        self.encoder = _build_sae_encoder(n_feat, d_hidden, d_latent, dropout)
        self.decoder = _build_sae_decoder(n_feat, d_hidden, d_latent)
        self.gat_ln = nn.LayerNorm(d_latent)
        self.gat = GATv2Layer(d_latent, d_latent, n_heads=n_heads, dropout=dropout)
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

        z = latent.view(B, S, -1)
        h = self.gat_ln(z)
        z = z + self.gat(h)                  # residual GAT
        z = self.post_ln(z)
        z_flat = z.reshape(B * S, -1)

        cat_in = torch.cat([z_flat, x_noisy], dim=-1)
        pred = self.pred_head(cat_in).squeeze(-1).reshape(B, S)

        self._last_recon_loss = F.mse_loss(recon, x_flat) if self.training else None
        return pred


# ===========================================================================
# V6.B. Pairwise interaction features (explicit cross-sym diff embeddings)
# ===========================================================================

class V6_PairwiseInteract(nn.Module):
    """Explicit (sym_i - sym_j) pair embeddings, attention-pooled over 4 partners.

    1. per-sym encoder → latent (B, 5, d_latent)
    2. For each ordered pair (i, j), j != i:
         diff_ij = MLP_diff(latent_i - latent_j)         ∈ d_pair
       So sym i has 4 partner-difference embeddings.
    3. Aggregate via attention pooling (learned query) → pair_feat (B, 5, d_pair)
    4. Concat [latent_i, pair_feat_i, x_noisy_i] → pred head

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
        self._last_recon_loss = None
        self._last_aux_loss = None

        self.encoder = _build_sae_encoder(n_feat, d_hidden, d_latent, dropout)
        self.decoder = _build_sae_decoder(n_feat, d_hidden, d_latent)

        # Pair-diff embedding MLP (takes a single (d_latent,) diff vector)
        # We concat sign-symmetry breakers: [diff, latent_i, latent_j_mean] is overkill;
        # keep it simple: just feed the diff itself.
        self.diff_mlp = nn.Sequential(
            nn.Linear(d_latent, d_pair), nn.LayerNorm(d_pair), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_pair, d_pair),
        )
        # Attention-pool: 1 learned query attends over 4 partner-diff embeddings
        self.pool_query = nn.Parameter(torch.randn(1, 1, 1, d_pair) * 0.1)
        self.pool_proj_k = nn.Linear(d_pair, d_pair, bias=False)
        self.pool_proj_v = nn.Linear(d_pair, d_pair, bias=False)
        self.pool_ln = nn.LayerNorm(d_pair)

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

        # Pair diffs: diff[b, i, j] = latent[b, i] - latent[b, j]
        diff = latent.unsqueeze(2) - latent.unsqueeze(1)      # (B, S=i, S=j, d_latent)
        # Mask out self-loop (i == j) by zeroing (they would be 0 anyway, but keep clean)
        eye_mask = torch.eye(S, device=x.device, dtype=torch.bool).view(1, S, S, 1)
        diff = diff.masked_fill(eye_mask, 0.0)

        # Embed each pair diff
        pair_emb = self.diff_mlp(diff)                        # (B, S, S, d_pair)

        # Attention-pool over the 4 partners (j != i)
        # Query: learned (broadcast); Keys/Values from pair_emb; mask self.
        q = self.pool_query.expand(B, S, 1, self.d_pair)      # (B, S, 1, d_pair)
        k = self.pool_proj_k(pair_emb)                        # (B, S, S, d_pair)
        v = self.pool_proj_v(pair_emb)                        # (B, S, S, d_pair)
        scale = self.d_pair ** -0.5
        attn_logits = (q * k).sum(dim=-1) * scale             # (B, S, S)
        attn_logits = attn_logits.masked_fill(
            eye_mask.squeeze(-1), float('-inf')
        )
        attn = F.softmax(attn_logits, dim=-1)                 # over j  (B, S, S)
        pair_feat = (attn.unsqueeze(-1) * v).sum(dim=2)       # (B, S, d_pair)
        pair_feat = self.pool_ln(pair_feat)

        latent_flat = latent.reshape(B * S, -1)
        pair_flat = pair_feat.reshape(B * S, -1)
        cat_in = torch.cat([latent_flat, pair_flat, x_noisy], dim=-1)
        pred = self.pred_head(cat_in).squeeze(-1).reshape(B, S)

        self._last_recon_loss = F.mse_loss(recon, x_flat) if self.training else None
        return pred


# ===========================================================================
# V6.C. Cross-feature attention (FT-Transformer style, feature-group tokens)
# ===========================================================================

class V6_CrossFeatAttn(nn.Module):
    """Feature-level self-attention.

    To keep cost manageable on 259d input, we project features into n_groups
    feature-group tokens via a single Linear(259 → n_groups * d_token), then
    run 1 layer of self-attention over n_groups tokens. The output is mean-pooled
    and projected to d_latent. SAE recon path retained.

    Per-sym (5 parallel forwards), sym-agnostic.
    """
    def __init__(self, n_sym: int = 5, n_feat: int = 259,
                 d_latent: int = 32, d_hidden: int = 128,
                 n_groups: int = 32, d_token: int = 16,
                 n_heads: int = 4, dropout: float = 0.30,
                 recon_alpha: float = 0.3, noise_sigma: float = 0.035):
        super().__init__()
        assert d_token % n_heads == 0
        self.n_sym = n_sym
        self.n_feat = n_feat
        self.n_groups = n_groups
        self.d_token = d_token
        self.recon_alpha = recon_alpha
        self.noise_sigma = noise_sigma
        self._last_recon_loss = None
        self._last_aux_loss = None

        # Feature → group-token embedding
        self.feat_to_tokens = nn.Linear(n_feat, n_groups * d_token)
        self.token_ln = nn.LayerNorm(d_token)
        # Learned positional embedding for groups
        self.pos_emb = nn.Parameter(torch.randn(1, n_groups, d_token) * 0.02)

        # 1-layer self-attention over n_groups tokens
        self.attn_ln = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(d_token, num_heads=n_heads,
                                          dropout=dropout, batch_first=True)
        self.attn_drop = nn.Dropout(dropout)
        d_ffn = d_token * 2
        self.ffn_ln = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_ffn), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ffn, d_token),
        )
        self.ffn_drop = nn.Dropout(dropout)

        # Pool tokens → d_latent
        self.pool_proj = nn.Linear(n_groups * d_token, d_latent)
        self.latent_ln = nn.LayerNorm(d_latent)
        self.latent_act = nn.GELU()

        # SAE-style decoder for recon (from pooled latent)
        self.decoder = _build_sae_decoder(n_feat, d_hidden, d_latent)

        # Pred head: latent + raw features
        self.pred_head = _build_pred_head(d_latent + n_feat, d_hidden, dropout)

        _kaiming_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, F_ = x.shape
        x_flat = x.reshape(B * S, F_)
        if self.training and self.noise_sigma > 0:
            x_noisy = x_flat + torch.randn_like(x_flat) * self.noise_sigma
        else:
            x_noisy = x_flat

        # Project features → group tokens
        tokens = self.feat_to_tokens(x_noisy)                  # (B*S, n_groups*d_token)
        tokens = tokens.view(B * S, self.n_groups, self.d_token)
        tokens = self.token_ln(tokens) + self.pos_emb

        # Self-attention over groups (pre-LN)
        h = self.attn_ln(tokens)
        a, _ = self.attn(h, h, h, need_weights=False)
        tokens = tokens + self.attn_drop(a)
        h = self.ffn_ln(tokens)
        tokens = tokens + self.ffn_drop(self.ffn(h))           # (B*S, n_groups, d_token)

        # Pool & project to d_latent (concat pooling, not mean)
        pooled = tokens.reshape(B * S, self.n_groups * self.d_token)
        latent = self.latent_act(self.latent_ln(self.pool_proj(pooled)))

        recon = self.decoder(latent)

        cat_in = torch.cat([latent, x_noisy], dim=-1)
        pred = self.pred_head(cat_in).squeeze(-1).reshape(B, S)

        self._last_recon_loss = F.mse_loss(recon, x_flat) if self.training else None
        return pred


# ===========================================================================
# V6.D. Information bottleneck cross-sym
# ===========================================================================

class V6_Bottleneck(nn.Module):
    """5 sym → shared bottleneck → 5 sym (forces information sharing).

    1. per-sym encoder → latent (B, 5, d_latent)
    2. cross-sym aggregation (mean + max pooling, concat) → (B, 2*d_latent)
       then compress → bottleneck (B, d_bot)
    3. decompress → (B, d_latent), broadcast to (B, 5, d_latent)
    4. concat [latent_i, broadcast_i] → per-sym pred head

    The bottleneck width d_bot is the key knob. d_bot=32 forces real compression
    relative to 5*32=160 per-sym info; d_bot=16 even more aggressive.
    """
    def __init__(self, n_sym: int = 5, n_feat: int = 259,
                 d_latent: int = 32, d_hidden: int = 128,
                 d_bot: int = 32, dropout: float = 0.30,
                 recon_alpha: float = 0.3, noise_sigma: float = 0.035):
        super().__init__()
        self.n_sym = n_sym
        self.n_feat = n_feat
        self.d_bot = d_bot
        self.recon_alpha = recon_alpha
        self.noise_sigma = noise_sigma
        self._last_recon_loss = None
        self._last_aux_loss = None

        self.encoder = _build_sae_encoder(n_feat, d_hidden, d_latent, dropout)
        self.decoder = _build_sae_decoder(n_feat, d_hidden, d_latent)

        # Compress: pool (mean+max) over 5 syms → 2*d_latent → d_bot
        self.compress = nn.Sequential(
            nn.Linear(2 * d_latent, d_hidden), nn.LayerNorm(d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_bot), nn.LayerNorm(d_bot), nn.GELU(),
        )
        # Decompress: d_bot → d_latent
        self.decompress = nn.Sequential(
            nn.Linear(d_bot, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_latent), nn.LayerNorm(d_latent),
        )

        self.pred_head = _build_pred_head(2 * d_latent + n_feat, d_hidden, dropout)

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

        # Pool (mean + max are both permutation-invariant)
        mean_p = latent.mean(dim=1)                           # (B, d_latent)
        max_p, _ = latent.max(dim=1)
        pool = torch.cat([mean_p, max_p], dim=-1)             # (B, 2*d_latent)

        bot = self.compress(pool)                             # (B, d_bot)
        shared = self.decompress(bot)                         # (B, d_latent)
        shared_b = shared.unsqueeze(1).expand(-1, S, -1)      # (B, 5, d_latent)

        latent_aug = torch.cat([latent, shared_b], dim=-1)    # (B, 5, 2*d_latent)
        latent_aug_flat = latent_aug.reshape(B * S, -1)

        cat_in = torch.cat([latent_aug_flat, x_noisy], dim=-1)
        pred = self.pred_head(cat_in).squeeze(-1).reshape(B, S)

        self._last_recon_loss = F.mse_loss(recon, x_flat) if self.training else None
        return pred


# ===========================================================================
# V6.E. Mixture of Experts per-sym (sigmoid-gated, sym-agnostic gate)
# ===========================================================================

class V6_MoE(nn.Module):
    """Sigmoid-gated MoE on top of cross-sym MHSA.

    Encoder + cross-sym MHSA (same as sae_cross_sym baseline) produces per-sym
    representation z (B, 5, d_latent). Then:
      - n_experts expert MLPs each produce a scalar prediction
      - gate: per-sym MLP(z) → n_experts sigmoid weights (sym-agnostic; same MLP)
      - output: sum_e gate_e * expert_e(cat_in)

    Why sigmoid not softmax: avoids expert collapse (any subset can be active).
    Gate is purely data-driven (no sym ID embedding → sym-agnostic).
    """
    def __init__(self, n_sym: int = 5, n_feat: int = 259,
                 d_latent: int = 32, d_hidden: int = 128,
                 n_heads: int = 4, n_experts: int = 5,
                 dropout: float = 0.30,
                 recon_alpha: float = 0.3, noise_sigma: float = 0.035,
                 gate_init_bias: float = 0.0):
        super().__init__()
        self.n_sym = n_sym
        self.n_feat = n_feat
        self.n_experts = n_experts
        self.recon_alpha = recon_alpha
        self.noise_sigma = noise_sigma
        self._last_recon_loss = None
        self._last_aux_loss = None

        self.encoder = _build_sae_encoder(n_feat, d_hidden, d_latent, dropout)
        self.decoder = _build_sae_decoder(n_feat, d_hidden, d_latent)

        assert d_latent % n_heads == 0
        self.attn_ln = nn.LayerNorm(d_latent)
        self.attn = nn.MultiheadAttention(d_latent, num_heads=n_heads,
                                          dropout=dropout, batch_first=True)
        self.post_ln = nn.LayerNorm(d_latent)

        d_in = d_latent + n_feat
        # n_experts smaller MLPs (each ~half-width)
        d_exp_hidden = d_hidden // 2
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_in, d_exp_hidden), nn.LayerNorm(d_exp_hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_exp_hidden, d_exp_hidden // 2), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_exp_hidden // 2, 1),
            ) for _ in range(n_experts)
        ])
        # Gate: MLP(d_latent) → n_experts sigmoid weights (no sym ID!)
        self.gate = nn.Sequential(
            nn.Linear(d_latent, d_latent), nn.GELU(),
            nn.Linear(d_latent, n_experts),
        )
        self.gate_init_bias = gate_init_bias

        _kaiming_init(self)
        # Initialize gate output bias to gate_init_bias (sigmoid → ~0.5 at init)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, gate_init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, F_ = x.shape
        x_flat = x.reshape(B * S, F_)
        if self.training and self.noise_sigma > 0:
            x_noisy = x_flat + torch.randn_like(x_flat) * self.noise_sigma
        else:
            x_noisy = x_flat
        latent = self.encoder(x_noisy)                        # (B*S, d_latent)
        recon = self.decoder(latent)

        z = latent.view(B, S, -1)
        h = self.attn_ln(z)
        a, _ = self.attn(h, h, h, need_weights=False)
        z = self.post_ln(z + a)
        z_flat = z.reshape(B * S, -1)

        cat_in = torch.cat([z_flat, x_noisy], dim=-1)

        # Gate: sigmoid weights per expert
        gate_w = torch.sigmoid(self.gate(z_flat))             # (B*S, n_experts)

        # Expert outputs (each scalar) - aggregate via gate
        # Stack experts: (n_experts, B*S, 1)
        expert_outs = torch.stack([exp(cat_in) for exp in self.experts], dim=0)  # (E, B*S, 1)
        expert_outs = expert_outs.squeeze(-1).transpose(0, 1)  # (B*S, E)

        # Normalize gate weights by sum (so output magnitude doesn't blow up with n_experts)
        # but keep it differentiable and not exactly softmax (avoid collapse)
        gate_norm = gate_w / (gate_w.sum(dim=-1, keepdim=True).clamp_min(1e-6))
        pred = (gate_norm * expert_outs).sum(dim=-1).reshape(B, S)

        self._last_recon_loss = F.mse_loss(recon, x_flat) if self.training else None
        return pred


# ===========================================================================
# Factory
# ===========================================================================

ARCH_CLASSES_V6 = {
    "v6_gat":         V6_GATCrossSym,
    "v6_pairwise":    V6_PairwiseInteract,
    "v6_crossfeat":   V6_CrossFeatAttn,
    "v6_bottleneck":  V6_Bottleneck,
    "v6_moe":         V6_MoE,
}

ARCH_DEFAULTS_V6 = {
    "v6_gat":         dict(d_latent=32, d_hidden=128, n_heads=4, dropout=0.30,
                           recon_alpha=0.3, noise_sigma=0.035),
    "v6_pairwise":    dict(d_latent=32, d_hidden=128, d_pair=32, dropout=0.30,
                           recon_alpha=0.3, noise_sigma=0.035),
    "v6_crossfeat":   dict(d_latent=32, d_hidden=128, n_groups=32, d_token=16,
                           n_heads=4, dropout=0.30,
                           recon_alpha=0.3, noise_sigma=0.035),
    "v6_bottleneck":  dict(d_latent=32, d_hidden=128, d_bot=32, dropout=0.30,
                           recon_alpha=0.3, noise_sigma=0.035),
    "v6_moe":         dict(d_latent=32, d_hidden=128, n_heads=4, n_experts=5,
                           dropout=0.30, recon_alpha=0.3, noise_sigma=0.035,
                           gate_init_bias=0.0),
}


def build_model_v6(arch: str, n_sym: int = 5, n_feat: int = 259, **overrides):
    cls = ARCH_CLASSES_V6[arch]
    kwargs = dict(ARCH_DEFAULTS_V6[arch])
    kwargs.update(overrides)
    return cls(n_sym=n_sym, n_feat=n_feat, **kwargs)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity: build all archs, forward a tiny batch, count params.
    import sys
    torch.manual_seed(0)
    B, S, F_ = 4, 5, 259
    x = torch.randn(B, S, F_)
    for arch in ARCH_CLASSES_V6:
        m = build_model_v6(arch)
        m.eval()
        with torch.no_grad():
            y = m(x)
        m.train()
        y = m(x)  # exercise training path (recon loss)
        n = count_params(m)
        print(f"  {arch:15s}  n_params={n:>8d}  out_shape={tuple(y.shape)}  "
              f"recon={float(m._last_recon_loss):.4f}")
