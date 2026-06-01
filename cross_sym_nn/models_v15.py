"""V15: 5 fundamentally new directions to break v9-v14 +40 plateau.

All variants extend V12_InvertedLlama (the +40.39 SOTA arch).

V15.A — V15A_Temporal           (same arch, n_feat=80 via 60 inter + 20 temporal-agg)
V15.B — V15B_DecoupledLR        (encoder lr=1e-5 slow, head lr=3e-4 fast — trainer handles)
V15.C — V15C_SAM                (Sharpness-Aware Minimization at training step — trainer handles)
V15.D — V15D_PerSymBias         (5 learnable per-sym output biases pred += bias[sym])
V15.E — V15E_TTA                (same arch; TTA averaging at inference — trainer handles)

A/B/C/E share unchanged V12_InvertedLlama arch (flags handled by trainer).
D overrides forward to add per-sym bias to pred.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from models_v11 import RMSNorm
from models_v12 import V12_InvertedLlama


# ===========================================================================
# V15.A — Temporal-Aggregation Features (arch unchanged)
# ===========================================================================

class V15A_Temporal(V12_InvertedLlama):
    """Same arch as V12_InvertedLlama; n_feat differs (caller supplies the
    new --sel-idx file selecting 80 = 60 inter + 20 temporal-agg)."""
    pass


# ===========================================================================
# V15.B — Decoupled Learning Rates (arch unchanged; trainer splits LRs)
# ===========================================================================

class V15B_DecoupledLR(V12_InvertedLlama):
    """Arch unchanged. Trainer reads `lr_groups` and applies different LRs to
    encoder vs head vs others."""
    lr_groups = True


# ===========================================================================
# V15.C — SAM optimizer (arch unchanged; trainer runs 2-pass SAM step)
# ===========================================================================

class V15C_SAM(V12_InvertedLlama):
    """Arch unchanged. Trainer reads `use_sam=True` and runs SAM ascent+descent."""
    use_sam = True
    sam_rho = 0.05


# ===========================================================================
# V15.D — Per-Sym Output Bias (5 learnable scalars)
# ===========================================================================

class V15D_PerSymBias(V12_InvertedLlama):
    """V12_InvertedLlama + 5 learnable per-sym scalar biases added to output.

    Note: This DOES use a per-sym parameter, so technically not sym-equivariant.
    BUT: it's only an OUTPUT bias on the 5 trained sym slots; for sym indices
    outside 0-4 (test-time unseen sym), the indexing would fail. Per task spec
    Predictor evaluates with sym ∈ {0,1,2,3,4}, so this is well-defined for
    the eval protocol. For sym OOD inference, just disable bias (set zeros).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sym_bias = nn.Parameter(torch.zeros(self.n_sym))

    def forward(self, x):
        pred = super().forward(x)            # (B, n_sym)
        return pred + self.sym_bias[None, :]  # broadcast over batch


# ===========================================================================
# V15.E — Test-Time Augmentation (arch unchanged; trainer runs TTA at infer)
# ===========================================================================

class V15E_TTA(V12_InvertedLlama):
    """Arch unchanged. Trainer reads `use_tta=True` and runs N=10 forward
    passes with input noise at final inference, averaging the predictions."""
    use_tta = True
    tta_n = 10


# ===========================================================================
# Registry
# ===========================================================================

ARCH_CLASSES_V15 = {
    'v15_a_temporal':       V15A_Temporal,
    'v15_b_decoupled_lr':   V15B_DecoupledLR,
    'v15_c_sam':            V15C_SAM,
    'v15_d_per_sym_bias':   V15D_PerSymBias,
    'v15_e_tta':            V15E_TTA,
}

ARCH_DEFAULTS_V15 = {
    'v15_a_temporal': dict(
        d_latent=64, d_hidden=128, n_heads=4, depth=2,
        d_ffn_mult=2, dropout=0.30,
        recon_alpha=0.3, noise_sigma=0.035,
    ),
    'v15_b_decoupled_lr': dict(
        d_latent=64, d_hidden=128, n_heads=4, depth=2,
        d_ffn_mult=2, dropout=0.30,
        recon_alpha=0.3, noise_sigma=0.035,
    ),
    'v15_c_sam': dict(
        d_latent=64, d_hidden=128, n_heads=4, depth=2,
        d_ffn_mult=2, dropout=0.30,
        recon_alpha=0.3, noise_sigma=0.035,
    ),
    'v15_d_per_sym_bias': dict(
        d_latent=64, d_hidden=128, n_heads=4, depth=2,
        d_ffn_mult=2, dropout=0.30,
        recon_alpha=0.3, noise_sigma=0.035,
    ),
    'v15_e_tta': dict(
        d_latent=64, d_hidden=128, n_heads=4, depth=2,
        d_ffn_mult=2, dropout=0.30,
        recon_alpha=0.3, noise_sigma=0.035,
    ),
}


def build_model_v15(arch: str, n_sym: int = 5, n_feat: int = 319, **overrides):
    cls = ARCH_CLASSES_V15[arch]
    kwargs = dict(ARCH_DEFAULTS_V15[arch])
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
    cases = [('v15_a_temporal', 339),      # 259+80
             ('v15_b_decoupled_lr', 319),
             ('v15_c_sam', 319),
             ('v15_d_per_sym_bias', 319),
             ('v15_e_tta', 319)]
    for arch, F_ in cases:
        x = torch.randn(B, S, F_)
        m = build_model_v15(arch, n_feat=F_)
        m.eval()
        with torch.no_grad():
            y_eval = m(x)
        m.train()
        y_tr = m(x)
        n = count_params(m)
        finite = bool(torch.isfinite(y_tr).all())
        recon = float(m._last_recon_loss) if m._last_recon_loss is not None else float('nan')
        print(f'  {arch:24s} n_params={n:>8d}  out={tuple(y_tr.shape)}  '
              f'recon={recon:.4f}  finite={finite}')
