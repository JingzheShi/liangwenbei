"""Precompute preprocessed training data once, save for reuse across seeds."""
from __future__ import annotations
import json
import os
import sys
import time

import numpy as np

CACHE_DIR = os.environ.get("CACHE_DIR", "/root/lwb_remote_pkg/cache")
OUT_DIR = os.environ.get("OUT_DIR", "/root/T172_out")

T59_FAIL = ["dualz_ask_diff1","dualz_bid_diff5","dualz_ask_diff5",
            "qrank_W100_spread1","qrank_W100_spread5","qrank_W100_spread10",
            "qrank_W100_cumspread","kyle_lam_W50","kyle_lam_W100",
            "roll_eff_spr_ratio_W100"]
S5_FAIL = ["liq_asym_top5_W5"]
DROP_NAMES = T59_FAIL + S5_FAIL
CLIP = 10.0
AUG_LO, AUG_HI = 0.80, 1.20
SEED_AUG = 12345  # fixed aug seed, all training seeds share same aug data
NUM_CLASS = 3
HORIZON = 60


def class_balanced_weight(y_cls, num_class=3):
    w = np.ones(len(y_cls), dtype=np.float32)
    for c in range(num_class):
        mask = y_cls == c
        n_c = mask.sum()
        if n_c > 0:
            w[mask] = len(y_cls) / (num_class * n_c)
    return w.astype(np.float32)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    meta_path = os.path.join(OUT_DIR, "precomputed_meta.json")
    if os.path.exists(meta_path):
        print("Precomputed data already exists, skipping.")
        return

    print("Loading splits...", flush=True)
    t0 = time.time()
    splits = {}
    for split in ["train", "val", "test"]:
        p = os.path.join(CACHE_DIR, f"schemeP_{split}.npz")
        d = np.load(p)
        splits[split] = {k: d[k] for k in d.files}
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    feat_names_path = os.path.join(CACHE_DIR, "schemeP_feat_names.txt")
    with open(feat_names_path) as f:
        all_feat_names = [l.strip() for l in f]
    total_dim = splits["train"]["X"].shape[1]
    name_to_idx = {n: i for i, n in enumerate(all_feat_names)}
    drop_idx = set(name_to_idx[n] for n in DROP_NAMES if n in name_to_idx)
    keep_idx = np.array([i for i in range(total_dim) if i not in drop_idx], dtype=np.int64)
    feat_dim = len(keep_idx)
    print(f"  feat_dim={feat_dim}", flush=True)

    X_all = np.concatenate([splits["train"]["X"][:, keep_idx],
                            splits["val"]["X"][:, keep_idx],
                            splits["test"]["X"][:, keep_idx]], axis=0)
    y_cls = np.concatenate([splits["train"][f"y{HORIZON}"],
                            splits["val"][f"y{HORIZON}"],
                            splits["test"][f"y{HORIZON}"]]).astype(np.int64)
    mp_t = np.concatenate([splits[s]["mp_t"] for s in ["train","val","test"]])
    mp_th = np.concatenate([splits[s][f"mp_t{HORIZON}"] for s in ["train","val","test"]])
    date_all = np.concatenate([splits[s]["date"] for s in ["train","val","test"]])
    print(f"  N={len(X_all):,} date={date_all.min()}-{date_all.max()}", flush=True)

    y_regr = ((mp_th.astype(np.float64) - mp_t.astype(np.float64))
              / (mp_t.astype(np.float64) + 1.0)).astype(np.float32)
    target_scale = float(1.0 / max(y_regr.std(), 1e-8))
    y_s = (y_regr * target_scale).astype(np.float32)
    sw = class_balanced_weight(y_cls)

    print("  standardizing...", flush=True)
    feat_mean = np.nanmean(X_all, axis=0).astype(np.float32)
    feat_std = np.maximum(np.nanstd(X_all, axis=0), 1e-6).astype(np.float32)
    X_s = ((X_all - feat_mean) / feat_std).astype(np.float32)
    X_s = np.where(np.isnan(X_s), 0.0, X_s)
    X_s = np.clip(X_s, -CLIP, CLIP).astype(np.float32)

    print("  augmenting...", flush=True)
    rng = np.random.default_rng(SEED_AUG)
    X_imp = np.where(np.isnan(X_all), feat_mean, X_all)
    del X_all
    scales = rng.uniform(AUG_LO, AUG_HI, size=X_imp.shape).astype(np.float32)
    X_aug = ((X_imp * scales - feat_mean) / feat_std).astype(np.float32)
    del X_imp, scales
    X_aug = np.where(np.isnan(X_aug), 0.0, X_aug)
    X_aug = np.clip(X_aug, -CLIP, CLIP).astype(np.float32)

    X_full = np.concatenate([X_s, X_aug], axis=0)
    del X_s, X_aug
    y_full = np.concatenate([y_s, y_s], axis=0)
    sw_full = np.concatenate([sw, sw], axis=0)
    print(f"  X_full={X_full.shape}", flush=True)

    print("  saving...", flush=True)
    t1 = time.time()
    np.save(os.path.join(OUT_DIR, "X_full.npy"), X_full)
    np.save(os.path.join(OUT_DIR, "y_full.npy"), y_full)
    np.save(os.path.join(OUT_DIR, "sw_full.npy"), sw_full)
    np.save(os.path.join(OUT_DIR, "feat_mean.npy"), feat_mean)
    np.save(os.path.join(OUT_DIR, "feat_std.npy"), feat_std)
    np.save(os.path.join(OUT_DIR, "keep_idx.npy"), keep_idx)
    print(f"  saved in {time.time()-t1:.1f}s", flush=True)

    meta = {"feat_dim": feat_dim, "n_total": len(X_full),
            "target_scale": target_scale, "clip": CLIP,
            "horizon": HORIZON, "date_range": [int(date_all.min()), int(date_all.max())]}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"PRECOMPUTE DONE in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
