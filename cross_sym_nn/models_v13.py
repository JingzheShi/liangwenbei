"""V13: 5 directions to push beyond v12_b_inverted SOTA (+40.39 ± 0.69).

All variants build on V12_InvertedLlama (intra-FFN → cross-MHSA block ordering).

V13.A — V13A_SpoInverted          (SPO+ ranking loss replaces MSE)
V13.B — V13B_GatedInverted        (post-attention sigmoid gate per-sym)
V13.C — V13C_Inverted80           (same as v12_b but 80 feature input)
V13.D — V13D_ContrastiveInverted  (cross-sym contrastive aux loss)
V13.E — V13E_MultiHorizonInverted (shared encoder + 5 horizon heads)
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models_v6 import (
    _build_sae_encoder, _build_sae_decoder, _build_pred_head, _kaiming_init,
)
from models_v11 import RMSNorm, SwiGLU
from models_v12 import V12_InvertedLlama, InvertedLlamaBlock


# ===========================================================================
# V13.A — same arch as V12_InvertedLlama, only loss differs (handled in trainer)
# ===========================================================================

class V13A_SpoInverted(V12_InvertedLlama):
    """Same arch as V12_InvertedLlama. Loss switch is done in trainer.

    The trainer reads model.loss_kind == 'spo_plus' and replaces MSE.
    """
    loss_kind = 'spo_plus'


# ===========================================================================
# V13.B — Gated inverted block (post-attention sigmoid gate per-sym, per-dim)
# ===========================================================================

class GatedInvertedLlamaBlock(nn.Module):
    """Pre-RMSNorm + SwiGLU FFN (intra), then MHSA (cross) with sigmoid gate.

    Gate is computed from FFN-refined latent (B, 5, d_latent) → sigmoid,
    applied element-wise to attention delta so each sym can selectively
    accept / reject the cross-sym mix.
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
        self.gate = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.ffn(h)
        h = self.norm2(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        g = torch.sigmoid(self.gate(h))
        x = x + g * self.attn_drop(a)
        return x


class V13B_GatedInverted(nn.Module):
    """V12.B inverted block with post-attention sigmoid gate."""
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
            GatedInvertedLlamaBlock(d_latent, n_heads, d_ffn_mult=d_ffn_mult, dropout=dropout)
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
# V13.C — Same as V12_InvertedLlama; only feature count differs (80 vs 60)
# ===========================================================================

class V13C_Inverted80(V12_InvertedLlama):
    """Same arch as V12.B; n_feat is 339 (259 keep + 80 inter). Trainer wires."""
    pass


# ===========================================================================
# V13.D — Contrastive cross-sym auxiliary loss
# ===========================================================================

def _info_nce_off_diag(z: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    """InfoNCE-style: cosine similarity matrix between 5 sym latents per batch.

    Loss = -log( exp(sim[i,i]/tau) / sum_j exp(sim[i,j]/tau) ) per i, mean.
    Self-sim is the positive; cross-sym is the negative. Encourages syms to
    have distinct latent representations (avoid collapse to one point).
    z: (B, S, d_latent)
    """
    z_n = F.normalize(z, dim=-1)
    sim = torch.matmul(z_n, z_n.transpose(-2, -1)) / tau   # (B, S, S)
    # Targets are diagonal indices
    B, S, _ = sim.shape
    targets = torch.arange(S, device=sim.device).unsqueeze(0).expand(B, S)  # (B, S)
    loss = F.cross_entropy(sim.reshape(B * S, S), targets.reshape(B * S))
    return loss


class V13D_ContrastiveInverted(V12_InvertedLlama):
    """V12.B + contrastive auxiliary loss on post-block latent z (B, 5, d_latent).

    aux_loss = InfoNCE on cross-sym cosine sim (diagonal = positive).
    Trainer reads model._last_aux_loss + model.aux_alpha (default 0.1).
    """
    def __init__(self, *args, aux_alpha: float = 0.1, contrastive_tau: float = 0.5,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_alpha = aux_alpha
        self.contrastive_tau = contrastive_tau

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

        if self.training:
            self._last_aux_loss = _info_nce_off_diag(z, tau=self.contrastive_tau)
        else:
            self._last_aux_loss = None

        z_flat = z.reshape(B * S, -1)
        cat_in = torch.cat([z_flat, x_noisy], dim=-1)
        pred = self.pred_head(cat_in).squeeze(-1).reshape(B, S)
        self._last_recon_loss = F.mse_loss(recon, x_flat) if self.training else None
        return pred


# ===========================================================================
# V13.E — Multi-Horizon Shared Encoder
# ===========================================================================

class V13E_MultiHorizonInverted(nn.Module):
    """Shared encoder + 5 per-horizon prediction heads.

    Horizons: 5, 10, 20, 40, 60. Trainer must supply y_reg per horizon and
    target_scale per horizon. Output is a dict-like tensor: (B, 5, S) where
    dim 0 of last axis = horizon idx.

    During inference we only use head idx 4 (h=60). For training, sum MSE
    across heads (each weighted equally before scale normalization).

    To keep the trainer interface simple, forward() returns pred for h=60.
    Per-horizon predictions are exposed via self._last_preds_dict for the
    trainer to use multi-horizon loss.
    """
    HORIZON_LIST = (5, 10, 20, 40, 60)
    MAIN_H_IDX = 4    # h=60 is index 4 in HORIZON_LIST

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
        self._last_preds_dict = None   # dict[h_idx -> (B, S)]

        self.encoder = _build_sae_encoder(n_feat, d_hidden, d_latent, dropout)
        self.decoder = _build_sae_decoder(n_feat, d_hidden, d_latent)
        self.blocks = nn.ModuleList([
            InvertedLlamaBlock(d_latent, n_heads, d_ffn_mult=d_ffn_mult, dropout=dropout)
            for _ in range(depth)
        ])
        self.post_norm = RMSNorm(d_latent)
        # 5 prediction heads (one per horizon)
        self.heads = nn.ModuleList([
            _build_pred_head(d_latent + n_feat, d_hidden, dropout)
            for _ in self.HORIZON_LIST
        ])

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

        preds_per_h = {}
        for i, h in enumerate(self.HORIZON_LIST):
            p = self.heads[i](cat_in).squeeze(-1).reshape(B, S)
            preds_per_h[h] = p
        self._last_preds_dict = preds_per_h
        self._last_recon_loss = F.mse_loss(recon, x_flat) if self.training else None
        return preds_per_h[self.HORIZON_LIST[self.MAIN_H_IDX]]   # default → h=60


# ===========================================================================
# Registry
# ===========================================================================

ARCH_CLASSES_V13 = {
    'v13_a_spo':           V13A_SpoInverted,
    'v13_b_gated':         V13B_GatedInverted,
    'v13_c_80feat':        V13C_Inverted80,
    'v13_d_contrastive':   V13D_ContrastiveInverted,
    'v13_e_multihorizon':  V13E_MultiHorizonInverted,
}

ARCH_DEFAULTS_V13 = {
    'v13_a_spo': dict(
        d_latent=64, d_hidden=128, n_heads=4, depth=2,
        d_ffn_mult=2, dropout=0.30,
        recon_alpha=0.3, noise_sigma=0.035,
    ),
    'v13_b_gated': dict(
        d_latent=64, d_hidden=128, n_heads=4, depth=2,
        d_ffn_mult=2, dropout=0.30,
        recon_alpha=0.3, noise_sigma=0.035,
    ),
    'v13_c_80feat': dict(
        d_latent=64, d_hidden=128, n_heads=4, depth=2,
        d_ffn_mult=2, dropout=0.30,
        recon_alpha=0.3, noise_sigma=0.035,
    ),
    'v13_d_contrastive': dict(
        d_latent=64, d_hidden=128, n_heads=4, depth=2,
        d_ffn_mult=2, dropout=0.30,
        recon_alpha=0.3, noise_sigma=0.035,
        aux_alpha=0.1, contrastive_tau=0.5,
    ),
    'v13_e_multihorizon': dict(
        d_latent=64, d_hidden=128, n_heads=4, depth=2,
        d_ffn_mult=2, dropout=0.30,
        recon_alpha=0.3, noise_sigma=0.035,
    ),
}


def build_model_v13(arch: str, n_sym: int = 5, n_feat: int = 319, **overrides):
    cls = ARCH_CLASSES_V13[arch]
    kwargs = dict(ARCH_DEFAULTS_V13[arch])
    kwargs.update(overrides)
    return cls(n_sym=n_sym, n_feat=n_feat, **kwargs)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ===========================================================================
# SPO+ surrogate loss
# ===========================================================================

def spo_plus_loss(pred: torch.Tensor, y: torch.Tensor, weight: torch.Tensor,
                  fee: float = 1e-4) -> torch.Tensor:
    """SPO+ (Elmachtoub & Grigas 2022) — convex upper bound on regret of
    a sign-of-prediction decision policy with proportional fee.

    Setup:
      action a(pred) = sign(pred)
      cost  c(a, y)  = -a * y + fee * |a|
      regret = c(a(pred), y) - c(a*(y), y), where a*(y) = sign(y)*[|y|>fee]
    SPO+ surrogate:
      L = max(0, c(a*(y), y) - 2*<a*(y), pred> + max_a <a, pred> + fee*|a|)
        = max_a [<a, pred> + fee*|a|] - 2*<a*(y), pred> + c*(y)
    For scalar action ∈ {-1, 0, +1}, max_a [a*pred + fee*|a|] = max(|pred|, 0) if fee
    paid; effectively the standard SPO+ surrogate reduces to a hinge.

    Simpler closed-form for sign action (Bertsimas style):
      L = 2 * relu(-pred * y) + fee * (relu(pred - |y|) + relu(-pred - |y|))
        ≈ 2 * relu(-pred * y) + fee * |pred|   (smoothed)
    """
    # Convex surrogate: penalize sign disagreement, plus tiny fee regularizer on |pred|.
    hinge = F.relu(-pred * y) * 2.0           # 0 when sign agrees & |y| large
    fee_reg = fee * pred.abs()
    per_elem = hinge + fee_reg
    w = weight.clamp_min(0.0)
    return (per_elem * w).sum() / w.sum().clamp_min(1.0)


# ===========================================================================
# Smoke test
# ===========================================================================

if __name__ == '__main__':
    torch.manual_seed(0)
    B, S = 4, 5
    for arch, F_ in [('v13_a_spo', 319), ('v13_b_gated', 319),
                     ('v13_c_80feat', 339), ('v13_d_contrastive', 319),
                     ('v13_e_multihorizon', 319)]:
        x = torch.randn(B, S, F_)
        m = build_model_v13(arch, n_feat=F_)
        m.eval()
        with torch.no_grad():
            y_eval = m(x)
        m.train()
        y_tr = m(x)
        n = count_params(m)
        recon = float(m._last_recon_loss) if m._last_recon_loss is not None else float('nan')
        aux   = float(m._last_aux_loss)   if m._last_aux_loss   is not None else float('nan')
        print(f'  {arch:22s} n_params={n:>8d}  out={tuple(y_tr.shape)}  '
              f'recon={recon:.4f}  aux={aux:.4f}')
    # SPO+ loss smoke
    pred = torch.randn(B, S, requires_grad=True)
    y    = torch.randn(B, S)
    w    = torch.ones(B, S)
    l = spo_plus_loss(pred, y, w)
    l.backward()
    print(f'  spo_plus_loss = {float(l):.4f}  grad_norm = {float(pred.grad.norm()):.4f}')
