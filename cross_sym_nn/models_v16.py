"""V16: combine v15 winners + lr ratio sweep.

V15 SOTAs:
- mean SOTA: v15_b_decoupled_lr +40.98 ± 1.12 (encoder 1e-5 / head 3e-4)
- std  SOTA: v15_d_per_sym_bias +40.35 ± 0.55 (5 learnable per-sym biases)

V16 directions (all extend V12_InvertedLlama backbone):
- V16.A  combine_bd       — decoupled lr + per_sym_bias  (60 feat)
- V16.B  lr_sweep_5e6     — decoupled lr ratio 1:60 (enc 5e-6 / head 3e-4)  (60 feat)
- V16.C  decoupled_80feat — decoupled lr 1e-5/3e-4  (80 feat)
- V16.D  per_sym_80feat   — per_sym_bias only  (80 feat)
- V16.E  triple           — decoupled lr + per_sym_bias  (80 feat)
"""
from __future__ import annotations
import torch
import torch.nn as nn

from models_v12 import V12_InvertedLlama


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _add_per_sym_bias(cls):
    """Class decorator: add learnable per-sym scalar bias added to forward output."""
    base_init = cls.__init__
    base_forward = cls.forward

    def __init__(self, *args, **kwargs):
        base_init(self, *args, **kwargs)
        self.sym_bias = nn.Parameter(torch.zeros(self.n_sym))

    def forward(self, x):
        pred = base_forward(self, x)
        return pred + self.sym_bias[None, :]

    cls.__init__ = __init__
    cls.forward = forward
    return cls


# ---------------------------------------------------------------------------
# V16.A — decoupled lr + per_sym_bias (60 feat)
# ---------------------------------------------------------------------------

@_add_per_sym_bias
class V16A_CombineBD(V12_InvertedLlama):
    lr_groups = True
    lr_encoder = 1e-5
    lr_head = 3e-4
    lr_other = 1e-4


# ---------------------------------------------------------------------------
# V16.B — decoupled lr 5e-6 / 3e-4 (ratio 1:60), 60 feat
# ---------------------------------------------------------------------------

class V16B_LRSweep5e6(V12_InvertedLlama):
    lr_groups = True
    lr_encoder = 5e-6
    lr_head = 3e-4
    lr_other = 1e-4


# ---------------------------------------------------------------------------
# V16.C — decoupled lr (1e-5 / 3e-4), 80 feat
# ---------------------------------------------------------------------------

class V16C_Decoupled80feat(V12_InvertedLlama):
    lr_groups = True
    lr_encoder = 1e-5
    lr_head = 3e-4
    lr_other = 1e-4


# ---------------------------------------------------------------------------
# V16.D — per_sym_bias only, 80 feat
# ---------------------------------------------------------------------------

@_add_per_sym_bias
class V16D_PerSym80feat(V12_InvertedLlama):
    pass


# ---------------------------------------------------------------------------
# V16.E — triple: decoupled lr + per_sym_bias + 80 feat
# ---------------------------------------------------------------------------

@_add_per_sym_bias
class V16E_Triple(V12_InvertedLlama):
    lr_groups = True
    lr_encoder = 1e-5
    lr_head = 3e-4
    lr_other = 1e-4


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ARCH_CLASSES_V16 = {
    'v16_a_combine_bd':       V16A_CombineBD,
    'v16_b_lr_sweep_5e6':     V16B_LRSweep5e6,
    'v16_c_decoupled_80feat': V16C_Decoupled80feat,
    'v16_d_per_sym_80feat':   V16D_PerSym80feat,
    'v16_e_triple':           V16E_Triple,
}

ARCH_DEFAULTS_V16 = {
    arch: dict(
        d_latent=64, d_hidden=128, n_heads=4, depth=2,
        d_ffn_mult=2, dropout=0.30,
        recon_alpha=0.3, noise_sigma=0.035,
    )
    for arch in ARCH_CLASSES_V16
}


def build_model_v16(arch: str, n_sym: int = 5, n_feat: int = 319, **overrides):
    cls = ARCH_CLASSES_V16[arch]
    kwargs = dict(ARCH_DEFAULTS_V16[arch])
    kwargs.update(overrides)
    return cls(n_sym=n_sym, n_feat=n_feat, **kwargs)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    torch.manual_seed(0)
    B, S = 4, 5
    cases = [
        ('v16_a_combine_bd',       319),  # 259+60
        ('v16_b_lr_sweep_5e6',     319),
        ('v16_c_decoupled_80feat', 339),  # 259+80
        ('v16_d_per_sym_80feat',   339),
        ('v16_e_triple',           339),
    ]
    for arch, F_ in cases:
        x = torch.randn(B, S, F_)
        m = build_model_v16(arch, n_feat=F_)
        m.eval()
        with torch.no_grad():
            y_eval = m(x)
        m.train()
        y_tr = m(x)
        n = count_params(m)
        finite = bool(torch.isfinite(y_tr).all())
        lr_groups = getattr(m, 'lr_groups', False)
        lr_enc = getattr(m, 'lr_encoder', None)
        lr_head = getattr(m, 'lr_head', None)
        has_sym_bias = hasattr(m, 'sym_bias')
        recon = float(m._last_recon_loss) if m._last_recon_loss is not None else float('nan')
        print(f'  {arch:26s} n_params={n:>8d}  out={tuple(y_tr.shape)}  '
              f'recon={recon:.4f}  finite={finite}  lr_groups={lr_groups} '
              f'lr_enc={lr_enc} lr_head={lr_head}  sym_bias={has_sym_bias}')
