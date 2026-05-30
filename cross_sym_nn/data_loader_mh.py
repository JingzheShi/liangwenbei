"""Multi-horizon target loader add-on.

Re-uses the (sorted) per-row arrays from `data_loader._load_raw_split`, then
constructs y_reg / mp_t / mp_th tensors for h ∈ {5, 10, 20, 40, 60}, grouped to
(N_groups, 5).

The mirror augmentation negates returns for every horizon (since flipping
bid/ask flips the sign of return regardless of horizon).
"""
from __future__ import annotations
import numpy as np
from pathlib import Path

from data_loader import (
    CACHE, get_keep_idx, get_feat_names,
    apply_log1p_rawlast, bidask_mirror_flat, make_class_weights, CLIP,
)

HORIZONS = (5, 10, 20, 40, 60)


def _load_raw_split_mh(split: str, keep_idx: np.ndarray):
    """Load split + multi-horizon mp_t{h} arrays, sort by (date, sess, t, sym)."""
    d = np.load(CACHE / f'schemeP_{split}.npz')
    X_raw = d['X'][:, keep_idx].astype(np.float32, copy=False)
    mp_t   = d['mp_t'].astype(np.float64)
    mp_th_dict = {h: d[f'mp_t{h}'].astype(np.float64) for h in HORIZONS}
    y_cls_dict = {h: d[f'y{h}'].astype(np.int64) for h in HORIZONS}
    date   = d['date'].astype(np.int32)
    sess   = d['sess_idx'].astype(np.int32)
    t      = d['t'].astype(np.int32)
    sym    = d['sym'].astype(np.int32)
    d.close()
    order = np.lexsort((sym, t, sess, date))
    X_raw = X_raw[order]; mp_t = mp_t[order]
    mp_th_dict = {h: v[order] for h, v in mp_th_dict.items()}
    y_cls_dict = {h: v[order] for h, v in y_cls_dict.items()}
    return X_raw, mp_t, mp_th_dict, y_cls_dict


def load_grouped_data_mh(level: str = "L8", verbose: bool = True):
    """Same as load_grouped_data but with multi-horizon targets.

    Returns (train_data, val_data, test_data, stats) where each *_data dict has
    extra keys:
       y_reg_h{h}  : (N_groups, 5) float32  for each h ∈ HORIZONS
       mp_t_h{h}   : (N_groups, 5) float64
       mp_th_h{h}  : (N_groups, 5) float64
    Plus all the standard keys (X_grp, y_reg=y_reg_h60, y_cls, mp_t, mp_th, sw).

    stats includes target_scale_h{h} for each h, plus target_scale=target_scale_h60.
    """
    keep_idx   = get_keep_idx(level)
    feat_names = get_feat_names(keep_idx)
    n_feat = len(keep_idx)
    if verbose:
        print(f"  [MH] keep_idx level={level}, n_feat={n_feat}, horizons={HORIZONS}")

    tr_X, tr_mp_t, tr_mp_th_d, tr_y_cls_d = _load_raw_split_mh("train", keep_idx)
    va_X, va_mp_t, va_mp_th_d, va_y_cls_d = _load_raw_split_mh("val",   keep_idx)
    te_X, te_mp_t, te_mp_th_d, te_y_cls_d = _load_raw_split_mh("test",  keep_idx)

    N_tr = len(tr_X) // 5
    N_va = len(va_X) // 5
    N_te = len(te_X) // 5
    if verbose:
        print(f"  [MH] groups: train={N_tr:,} val={N_va:,} test={N_te:,}")

    def to_groups(X, mp_t, mp_th_d, y_cls_d, n_groups):
        X_g     = X.reshape(n_groups, 5, n_feat)
        mp_t_g  = mp_t.reshape(n_groups, 5)
        mp_th_g = {h: v.reshape(n_groups, 5) for h, v in mp_th_d.items()}
        y_reg_g = {h: ((mp_th_g[h] - mp_t_g) / (mp_t_g + 1.0)).astype(np.float32)
                   for h in HORIZONS}
        y_cls_g = {h: v.reshape(n_groups, 5) for h, v in y_cls_d.items()}
        return X_g, mp_t_g, mp_th_g, y_reg_g, y_cls_g

    X_tr, mp_t_tr, mp_th_tr_d, y_reg_tr_d, y_cls_tr_d = to_groups(tr_X, tr_mp_t, tr_mp_th_d, tr_y_cls_d, N_tr)
    X_va, mp_t_va, mp_th_va_d, y_reg_va_d, y_cls_va_d = to_groups(va_X, va_mp_t, va_mp_th_d, va_y_cls_d, N_va)
    X_te, mp_t_te, mp_th_te_d, y_reg_te_d, y_cls_te_d = to_groups(te_X, te_mp_t, te_mp_th_d, te_y_cls_d, N_te)

    def log1p_groups(X_g):
        flat = X_g.reshape(-1, n_feat)
        apply_log1p_rawlast(flat, feat_names)
        return flat.reshape(X_g.shape)

    X_tr = log1p_groups(X_tr); X_va = log1p_groups(X_va); X_te = log1p_groups(X_te)

    # mirror aug (synchronized across 5 syms; mirror flips return sign for ALL h)
    X_tr_flat = X_tr.reshape(-1, n_feat)
    X_tr_mirror = bidask_mirror_flat(X_tr_flat, feat_names).reshape(N_tr, 5, n_feat)

    y_reg_tr_aug_d = {h: np.concatenate([y_reg_tr_d[h], -y_reg_tr_d[h]], axis=0)
                      for h in HORIZONS}
    y_cls_tr_aug_d = {h: np.concatenate([y_cls_tr_d[h], (2 - y_cls_tr_d[h]).clip(0, 2)], axis=0)
                      for h in HORIZONS}
    mp_t_tr_aug    = np.concatenate([mp_t_tr, mp_t_tr.copy()], axis=0)
    mp_th_tr_aug_d = {h: np.concatenate([mp_th_tr_d[h], mp_th_tr_d[h].copy()], axis=0)
                      for h in HORIZONS}
    X_tr_aug = np.concatenate([X_tr, X_tr_mirror], axis=0)
    N_tr_aug = 2 * N_tr

    # window-z fit on aug train
    flat_aug = X_tr_aug.reshape(-1, n_feat)
    mu = np.nanmean(flat_aug, axis=0).astype(np.float32)
    sd = np.maximum(np.nanstd(flat_aug, axis=0), 1e-6).astype(np.float32)

    def zsc(X_g):
        flat = X_g.reshape(-1, n_feat)
        out = (flat - mu) / sd
        np.nan_to_num(out, copy=False, nan=0.0, posinf=CLIP, neginf=-CLIP)
        np.clip(out, -CLIP, CLIP, out=out)
        return out.reshape(X_g.shape).astype(np.float32)

    X_tr_aug = zsc(X_tr_aug); X_va = zsc(X_va); X_te = zsc(X_te)

    # per-horizon target_scale (fit on aug train)
    target_scale_d = {h: max(float(np.std(y_reg_tr_aug_d[h])), 1e-7) for h in HORIZONS}
    if verbose:
        print(f"  [MH] target_scale per-h: {[(h, round(target_scale_d[h], 6)) for h in HORIZONS]}")

    # sample weights: use h=60 cls (main horizon) — same as legacy loader
    cw = make_class_weights(y_cls_tr_aug_d[60].flatten(), num_class=3)
    sw_tr = cw[y_cls_tr_aug_d[60]].astype(np.float32)
    sw_va = cw[y_cls_va_d[60]].astype(np.float32)
    sw_te = cw[y_cls_te_d[60]].astype(np.float32)

    stats = dict(mu=mu, sd=sd, target_scale=target_scale_d[60],
                 target_scale_d=target_scale_d,
                 class_weights=cw, keep_idx=keep_idx, feat_names=feat_names,
                 horizons=HORIZONS)

    def pack(X_grp, y_reg_d, y_cls_d, mp_t, mp_th_d, sw):
        out = dict(X_grp=X_grp, y_reg=y_reg_d[60], y_cls=y_cls_d[60],
                   mp_t=mp_t, mp_th=mp_th_d[60], sw=sw)
        for h in HORIZONS:
            out[f'y_reg_h{h}'] = y_reg_d[h]
            out[f'mp_th_h{h}'] = mp_th_d[h]
            out[f'y_cls_h{h}'] = y_cls_d[h]
        return out

    train_data = pack(X_tr_aug, y_reg_tr_aug_d, y_cls_tr_aug_d, mp_t_tr_aug, mp_th_tr_aug_d, sw_tr)
    val_data   = pack(X_va,     y_reg_va_d,    y_cls_va_d,    mp_t_va,    mp_th_va_d,    sw_va)
    test_data  = pack(X_te,     y_reg_te_d,    y_cls_te_d,    mp_t_te,    mp_th_te_d,    sw_te)
    return train_data, val_data, test_data, stats
