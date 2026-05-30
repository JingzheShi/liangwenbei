"""Grouped data loader for cross-sym NN experiments.

Loads schemeP cache, groups by (date, sess_idx, t) into shape (N_groups, 5, 359),
applies log1p rawlast + window-z + mirror augmentation synchronized across 5 syms.
"""
from __future__ import annotations
import json, numpy as np
from pathlib import Path
from typing import Optional

CACHE = Path('/root/projects/liangwenbei_workdir/ablation_runs/cache_log1p')
PARTITION = Path('/root/projects/liangwenbei_workdir/ablation_runs/feat_family_progressive/partition.json')
FEAT_NAMES_FILE = CACHE / 'schemeP_feat_names.txt'

H = 60
CLIP = 10.0
LOG1P_PREFIXES = (
    "bid", "ask", "bsize", "asize", "amount_delta", "volume_delta",
    "avgbid", "avgask", "totalbsize", "totalasize", "midprice", "spread", "cumspread",
)


def load_partition():
    with open(PARTITION) as f:
        return json.load(f)


def get_keep_idx(level="L8"):
    p = load_partition()
    return np.array(p["levels"][level], dtype=np.int64)


def get_feat_names(keep_idx):
    names_all = open(FEAT_NAMES_FILE).read().splitlines()
    return [names_all[i] for i in keep_idx]


def apply_log1p_rawlast(X: np.ndarray, feat_names: list[str]) -> np.ndarray:
    magn_idx = [i for i, n in enumerate(feat_names)
                if any(n.startswith(p) for p in LOG1P_PREFIXES)]
    if not magn_idx:
        return X
    magn_idx = np.array(magn_idx, dtype=np.int64)
    v = X[:, magn_idx]
    X[:, magn_idx] = np.sign(v) * np.log1p(np.abs(v))
    return X


def bidask_mirror_flat(X: np.ndarray, feat_names: list[str]) -> np.ndarray:
    """Apply bid/ask mirror to flat X (N, F). Returns mirrored copy."""
    name_to_idx = {n: i for i, n in enumerate(feat_names)}
    swap = {}
    sign_flip = set()

    def add_pair(a, b):
        if a in name_to_idx and b in name_to_idx:
            i, j = name_to_idx[a], name_to_idx[b]
            swap[i] = j; swap[j] = i

    for k in range(1, 11):
        add_pair(f"bid{k}", f"ask{k}")
        add_pair(f"bsize{k}", f"asize{k}")
        add_pair(f"bid_diff{k}", f"ask_diff{k}")
        add_pair(f"bid_rate{k}", f"ask_rate{k}")
        add_pair(f"bsize_rate{k}", f"asize_rate{k}")
    add_pair("avgbid", "avgask")
    add_pair("totalbsize", "totalasize")
    add_pair("bid_mean", "ask_mean")
    add_pair("bsize_mean", "asize_mean")
    add_pair("lb_intst", "la_intst"); add_pair("mb_intst", "ma_intst"); add_pair("cb_intst", "ca_intst")
    add_pair("lb_ind",   "la_ind");   add_pair("mb_ind",   "ma_ind");   add_pair("cb_ind",   "ca_ind")
    add_pair("lb_acc",   "la_acc");   add_pair("mb_acc",   "ma_acc");   add_pair("cb_acc",   "ca_acc")
    if "imbalance" in name_to_idx:
        sign_flip.add(name_to_idx["imbalance"])

    Xm = X.copy()
    seen = set()
    for a, b in swap.items():
        key = tuple(sorted([a, b]))
        if key in seen: continue
        seen.add(key)
        Xm[:, a], Xm[:, b] = X[:, b].copy(), X[:, a].copy()
    for i in sign_flip:
        Xm[:, i] = -X[:, i]
    return Xm


def make_class_weights(y_cls: np.ndarray, num_class=3) -> np.ndarray:
    counts = np.bincount(y_cls.astype(np.int64), minlength=num_class).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    cw = len(y_cls) / (num_class * counts)
    return cw.astype(np.float32)


def _load_raw_split(split: str, keep_idx: np.ndarray):
    """Load one split; sort rows into (N_groups, 5, n_feat) order; return flat arrays."""
    d = np.load(CACHE / f'schemeP_{split}.npz')
    X_raw = d['X'][:, keep_idx].astype(np.float32, copy=False)
    mp_t   = d['mp_t'].astype(np.float64)
    mp_th  = d[f'mp_t{H}'].astype(np.float64)
    y_cls  = d[f'y{H}'].astype(np.int64)
    date   = d['date'].astype(np.int32)
    sess   = d['sess_idx'].astype(np.int32)
    t      = d['t'].astype(np.int32)
    sym    = d['sym'].astype(np.int32)
    d.close()

    # Sort so groups are contiguous with sym order [0,1,2,3,4]
    order = np.lexsort((sym, t, sess, date))
    return (X_raw[order], mp_t[order], mp_th[order], y_cls[order],
            date[order], sess[order], t[order], sym[order])


def load_grouped_data(
    level: str = "L8",
    verbose: bool = True,
):
    """Return (train_data, val_data, test_data, stats) where each *_data is a dict.

    Dict keys:
      X_grp  : (N_groups, 5, n_feat) float32
      y_reg  : (N_groups, 5) float32
      y_cls  : (N_groups, 5) int64
      mp_t   : (N_groups, 5) float64
      mp_th  : (N_groups, 5) float64
      sw     : (N_groups, 5) float32  sample weights (class-balanced)

    stats keys:
      mu, sd : (n_feat,) float32  window-z params
      target_scale : float
      class_weights : (3,) float32
      keep_idx : (359,) int64
      feat_names : list[str]
    """
    keep_idx   = get_keep_idx(level)
    feat_names = get_feat_names(keep_idx)
    n_feat = len(keep_idx)
    if verbose:
        print(f"  keep_idx level={level}, n_feat={n_feat}")

    # --- load raw splits ---
    tr_raw = _load_raw_split("train", keep_idx)
    va_raw = _load_raw_split("val",   keep_idx)
    te_raw = _load_raw_split("test",  keep_idx)

    N_tr = len(tr_raw[0]) // 5  # number of groups
    N_va = len(va_raw[0]) // 5
    N_te = len(te_raw[0]) // 5
    if verbose:
        print(f"  groups: train={N_tr:,} val={N_va:,} test={N_te:,}")

    def to_groups(raw, n_groups):
        X_r, mp_t_r, mp_th_r, y_cls_r, *_ = raw
        y_reg_r = ((mp_th_r - mp_t_r) / (mp_t_r + 1.0)).astype(np.float32)
        X_g    = X_r.reshape(n_groups, 5, n_feat)
        y_reg_g = y_reg_r.reshape(n_groups, 5)
        y_cls_g = y_cls_r.reshape(n_groups, 5)
        mp_t_g  = mp_t_r.reshape(n_groups, 5)
        mp_th_g = mp_th_r.reshape(n_groups, 5)
        return X_g, y_reg_g, y_cls_g, mp_t_g, mp_th_g

    X_tr, y_reg_tr, y_cls_tr, mp_t_tr, mp_th_tr = to_groups(tr_raw, N_tr)
    X_va, y_reg_va, y_cls_va, mp_t_va, mp_th_va = to_groups(va_raw, N_va)
    X_te, y_reg_te, y_cls_te, mp_t_te, mp_th_te = to_groups(te_raw, N_te)

    # --- log1p rawlast (on flat rows) ---
    def log1p_groups(X_g):
        flat = X_g.reshape(-1, n_feat)
        apply_log1p_rawlast(flat, feat_names)  # in-place
        return flat.reshape(X_g.shape)

    X_tr = log1p_groups(X_tr)
    X_va = log1p_groups(X_va)
    X_te = log1p_groups(X_te)

    # --- mirror augmentation (synchronized across 5 syms) ---
    X_tr_flat = X_tr.reshape(-1, n_feat)
    X_tr_mirror_flat = bidask_mirror_flat(X_tr_flat, feat_names)
    X_tr_mirror = X_tr_mirror_flat.reshape(N_tr, 5, n_feat)

    y_reg_tr_mirror = -y_reg_tr
    y_cls_tr_mirror = (2 - y_cls_tr).clip(0, 2)
    mp_t_tr_mirror  = mp_t_tr.copy()
    mp_th_tr_mirror = mp_th_tr.copy()

    X_tr_aug    = np.concatenate([X_tr,    X_tr_mirror],    axis=0)  # (2*N_tr, 5, F)
    y_reg_tr_aug = np.concatenate([y_reg_tr, y_reg_tr_mirror], axis=0)
    y_cls_tr_aug = np.concatenate([y_cls_tr, y_cls_tr_mirror], axis=0)
    mp_t_tr_aug  = np.concatenate([mp_t_tr,  mp_t_tr_mirror],  axis=0)
    mp_th_tr_aug = np.concatenate([mp_th_tr, mp_th_tr_mirror],  axis=0)

    N_tr_aug = 2 * N_tr
    if verbose:
        print(f"  train after mirror aug: {N_tr_aug:,} groups")

    # --- window-z (fit on augmented train flat) ---
    flat_aug = X_tr_aug.reshape(-1, n_feat)
    mu = np.nanmean(flat_aug, axis=0).astype(np.float32)
    sd = np.maximum(np.nanstd(flat_aug, axis=0), 1e-6).astype(np.float32)

    def zsc_groups(X_g):
        flat = X_g.reshape(-1, n_feat)
        out  = (flat - mu) / sd
        np.nan_to_num(out, copy=False, nan=0.0, posinf=CLIP, neginf=-CLIP)
        np.clip(out, -CLIP, CLIP, out=out)
        return out.reshape(X_g.shape).astype(np.float32)

    X_tr_aug = zsc_groups(X_tr_aug)
    X_va     = zsc_groups(X_va)
    X_te     = zsc_groups(X_te)

    # --- target scale + class weights ---
    target_scale = float(np.std(y_reg_tr_aug))
    if target_scale < 1e-7:
        target_scale = 1.0

    y_cls_flat_aug = y_cls_tr_aug.flatten()
    cw = make_class_weights(y_cls_flat_aug, num_class=3)
    sw_tr_aug = cw[y_cls_tr_aug].astype(np.float32)  # (2*N_tr, 5)
    sw_va     = cw[y_cls_va].astype(np.float32)
    sw_te     = cw[y_cls_te].astype(np.float32)

    if verbose:
        print(f"  target_scale={target_scale:.6e}  class_weights={cw.tolist()}")

    stats = dict(mu=mu, sd=sd, target_scale=target_scale, class_weights=cw,
                 keep_idx=keep_idx, feat_names=feat_names)

    train_data = dict(X_grp=X_tr_aug, y_reg=y_reg_tr_aug, y_cls=y_cls_tr_aug,
                      mp_t=mp_t_tr_aug, mp_th=mp_th_tr_aug, sw=sw_tr_aug)
    val_data   = dict(X_grp=X_va, y_reg=y_reg_va, y_cls=y_cls_va,
                      mp_t=mp_t_va, mp_th=mp_th_va, sw=sw_va)
    test_data  = dict(X_grp=X_te, y_reg=y_reg_te, y_cls=y_cls_te,
                      mp_t=mp_t_te, mp_th=mp_th_te, sw=sw_te)

    return train_data, val_data, test_data, stats
