"""V14: 5 new directions to push beyond v12_b_inverted SOTA (+40.39 ± 0.69).

All variants build on V12_InvertedLlama (intra-FFN → cross-MHSA block ordering).

V14.A — V14A_StableInit          (DeepNorm-style zero-init residual paths)
V14.B — V14B_Inverted100         (same arch, 100 feature input — sel idx file decides)
V14.C — V14C_StochDepth          (stochastic depth — drop entire blocks p=0.1)
V14.D — V14D_LabelNoise          (Gaussian label noise σ=0.05·y_std, trainer handles)
V14.E — V14E_DirectPnL           (direct PnL loss with tanh-smoothed sign + temp anneal)

A/B/D/E share the V12_InvertedLlama arch unchanged (loss/noise switches in trainer).
C overrides .blocks with stochastic-depth wrappers.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models_v11 import RMSNorm
from models_v12 import V12_InvertedLlama, InvertedLlamaBlock


# ===========================================================================
# V14.A — Stable Init (DeepNorm-style residual zero init)
# ===========================================================================

class V14A_StableInit(V12_InvertedLlama):
    """Same arch as V12_InvertedLlama, but zero-init residual output proj of
    every InvertedLlamaBlock so training starts as identity through blocks.

    - attn.out_proj.weight → 0      (cross-sym path begins as identity)
    - ffn.w3.weight        → 0      (intra-sym path begins as identity)
    Norms / encoder / pred head left at their existing init.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for blk in self.blocks:
            nn.init.zeros_(blk.attn.out_proj.weight)
            if blk.attn.out_proj.bias is not None:
                nn.init.zeros_(blk.attn.out_proj.bias)
            nn.init.zeros_(blk.ffn.w3.weight)


# ===========================================================================
# V14.B — Same arch, 100-feat input (selection driven by --sel-idx)
# ===========================================================================

class V14B_Inverted100(V12_InvertedLlama):
    """Identical to V12_InvertedLlama, only construction n_feat differs (handled
    by trainer based on chosen --sel-idx file)."""
    pass


# ===========================================================================
# V14.C — Stochastic Depth (entire block drop with prob p)
# ===========================================================================

class StochDepthInvertedBlock(InvertedLlamaBlock):
    """Per-sample-batch stochastic depth wrapper around InvertedLlamaBlock.

    With prob drop_prob during training, the entire block is skipped (identity
    forward). At eval, always runs at full strength (no scaling needed since
    the output is residual-summed).
    """
    def __init__(self, d_model, n_heads, d_ffn_mult=2, dropout=0.0, drop_prob=0.1):
        super().__init__(d_model, n_heads, d_ffn_mult=d_ffn_mult, dropout=dropout)
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.training and self.drop_prob > 0.0:
            if torch.rand((), device=x.device).item() < self.drop_prob:
                return x
        return super().forward(x)


class V14C_StochDepth(V12_InvertedLlama):
    """V12_InvertedLlama but blocks are stochastic-depth wrapped."""
    def __init__(self, n_sym=5, n_feat=319, d_latent=64, d_hidden=128,
                 n_heads=4, depth=2, d_ffn_mult=2, dropout=0.30,
                 recon_alpha=0.3, noise_sigma=0.035, drop_path=0.1):
        super().__init__(n_sym=n_sym, n_feat=n_feat, d_latent=d_latent,
                         d_hidden=d_hidden, n_heads=n_heads, depth=depth,
                         d_ffn_mult=d_ffn_mult, dropout=dropout,
                         recon_alpha=recon_alpha, noise_sigma=noise_sigma)
        # replace blocks with stochastic-depth variants
        self.blocks = nn.ModuleList([
            StochDepthInvertedBlock(d_latent, n_heads, d_ffn_mult=d_ffn_mult,
                                     dropout=dropout, drop_prob=drop_path)
            for _ in range(depth)
        ])
        # re-apply init for the new blocks
        from models_v6 import _kaiming_init
        _kaiming_init(self.blocks)
        for m in self.modules():
            if isinstance(m, RMSNorm):
                nn.init.ones_(m.weight)


# ===========================================================================
# V14.D — Label Noise (trainer handles; arch unchanged)
# ===========================================================================

class V14D_LabelNoise(V12_InvertedLlama):
    """Same arch as V12_InvertedLlama; trainer reads loss_kind == 'label_noise'
    and injects Gaussian noise σ = label_noise_sigma · y.std() into y."""
    loss_kind = 'label_noise'
    label_noise_sigma = 0.05


# ===========================================================================
# V14.E — Direct PnL (trainer handles; arch unchanged)
# ===========================================================================

class V14E_DirectPnL(V12_InvertedLlama):
    """Same arch as V12_InvertedLlama; trainer reads loss_kind == 'direct_pnl'
    and replaces MSE with direct PnL loss with tanh-soft sign + temp anneal."""
    loss_kind = 'direct_pnl'
    direct_pnl_fee = 2e-4
    direct_pnl_temp_init = 1.0
    direct_pnl_temp_final = 0.1


def direct_pnl_loss(pred: torch.Tensor, y: torch.Tensor, weight: torch.Tensor,
                    fee: float = 2e-4, temp: float = 0.5) -> torch.Tensor:
    """Maximize expected PnL with tanh-smoothed sign action.

      action = tanh(pred / temp)
      pnl    = action · y - fee · |action|
      loss   = -mean(pnl · weight) / sum(weight)
    """
    soft_act = torch.tanh(pred / max(temp, 1e-3))
    pnl = soft_act * y - fee * soft_act.abs()
    w = weight.clamp_min(0.0)
    return -(pnl * w).sum() / w.sum().clamp_min(1.0)


# ===========================================================================
# Registry
# ===========================================================================

ARCH_CLASSES_V14 = {
    'v14_a_stable_init': V14A_StableInit,
    'v14_b_100feat':     V14B_Inverted100,
    'v14_c_stochdepth':  V14C_StochDepth,
    'v14_d_labelnoise':  V14D_LabelNoise,
    'v14_e_direct_pnl':  V14E_DirectPnL,
}

ARCH_DEFAULTS_V14 = {
    'v14_a_stable_init': dict(
        d_latent=64, d_hidden=128, n_heads=4, depth=2,
        d_ffn_mult=2, dropout=0.30,
        recon_alpha=0.3, noise_sigma=0.035,
    ),
    'v14_b_100feat': dict(
        d_latent=64, d_hidden=128, n_heads=4, depth=2,
        d_ffn_mult=2, dropout=0.30,
        recon_alpha=0.3, noise_sigma=0.035,
    ),
    'v14_c_stochdepth': dict(
        d_latent=64, d_hidden=128, n_heads=4, depth=2,
        d_ffn_mult=2, dropout=0.30,
        recon_alpha=0.3, noise_sigma=0.035,
        drop_path=0.1,
    ),
    'v14_d_labelnoise': dict(
        d_latent=64, d_hidden=128, n_heads=4, depth=2,
        d_ffn_mult=2, dropout=0.30,
        recon_alpha=0.3, noise_sigma=0.035,
    ),
    'v14_e_direct_pnl': dict(
        d_latent=64, d_hidden=128, n_heads=4, depth=2,
        d_ffn_mult=2, dropout=0.30,
        recon_alpha=0.3, noise_sigma=0.035,
    ),
}


def build_model_v14(arch: str, n_sym: int = 5, n_feat: int = 319, **overrides):
    cls = ARCH_CLASSES_V14[arch]
    kwargs = dict(ARCH_DEFAULTS_V14[arch])
    kwargs.update(overrides)
    return cls(n_sym=n_sym, n_feat=n_feat, **kwargs)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ===========================================================================
# Smoke test
# ===========================================================================

if __name__ == '__main__':
    torch.manual_seed(0)
    B, S = 4, 5
    cases = [('v14_a_stable_init', 319), ('v14_b_100feat', 359),
             ('v14_c_stochdepth', 319), ('v14_d_labelnoise', 319),
             ('v14_e_direct_pnl', 319)]
    for arch, F_ in cases:
        x = torch.randn(B, S, F_)
        m = build_model_v14(arch, n_feat=F_)
        m.eval()
        with torch.no_grad():
            y_eval = m(x)
        m.train()
        y_tr = m(x)
        n = count_params(m)
        recon = float(m._last_recon_loss) if m._last_recon_loss is not None else float('nan')
        kind = getattr(m, 'loss_kind', None)
        finite_tr = bool(torch.isfinite(y_tr).all())
        print(f'  {arch:22s} n_params={n:>8d} out={tuple(y_tr.shape)} '
              f'recon={recon:.4f} kind={kind} finite={finite_tr}')
    # Direct PnL smoke
    pred = torch.randn(B, S, requires_grad=True)
    y    = torch.randn(B, S)
    w    = torch.ones(B, S)
    l = direct_pnl_loss(pred, y, w, temp=0.5)
    l.backward()
    print(f'  direct_pnl_loss = {float(l):+.4f}  grad_norm = {float(pred.grad.norm()):.4f}')
