"""Build 5 time-of-day features from schemeP cache (t + sess_idx).

Window: each session has 2001 rows × 3s. AM session = 9:40-11:20 (100 min).
PM session = 13:10-14:50 (100 min). First/last 10 min of each session
already excluded by data provider (no auction noise).

We treat the trading day as a continuous 200-min window:
  AM: t=0..2000 → mins 0..100
  PM: t=0..2000 → mins 100..200

Features (all sym-agnostic, stateless, shuffle-safe):
  1. tod_sin       sin(2π * mins_full / 200)
  2. tod_cos       cos(2π * mins_full / 200)
  3. is_first_20   1.0 if mins_into_session <= 20 (post-open)
  4. is_last_20    1.0 if mins_into_session >= 80 (pre-close, includes lunch start in AM)
  5. is_lunch_prox 1.0 if 90 <= mins_full <= 110 (lunch boundary)

Saves /root/lwb_work_tod/tod_{train,test}.npz with key 'tod' (N,5) and feat_names.
"""
from __future__ import annotations
import os
import numpy as np

CACHE_DIR = "/root/lwb_remote_pkg/cache"
OUT_DIR = "/root/lwb_work_tod"
os.makedirs(OUT_DIR, exist_ok=True)

FEAT_NAMES = ["tod_sin", "tod_cos", "is_first_20", "is_last_20", "is_lunch_prox"]


def build_tod(t, sess_idx):
    t = t.astype(np.float32)
    sess_idx = sess_idx.astype(np.float32)
    mins_into_session = t * 3.0 / 60.0  # 0 to 100
    mins_full = mins_into_session + 100.0 * sess_idx  # 0 to 200

    tod_sin = np.sin(2.0 * np.pi * mins_full / 200.0).astype(np.float32)
    tod_cos = np.cos(2.0 * np.pi * mins_full / 200.0).astype(np.float32)
    is_first_20 = (mins_into_session <= 20.0).astype(np.float32)
    is_last_20 = (mins_into_session >= 80.0).astype(np.float32)
    is_lunch_prox = ((mins_full >= 90.0) & (mins_full <= 110.0)).astype(np.float32)
    return np.stack([tod_sin, tod_cos, is_first_20, is_last_20, is_lunch_prox], axis=1)


def process(split):
    src = os.path.join(CACHE_DIR, f"schemeP_{split}.npz")
    print(f"Loading {src}...", flush=True)
    d = np.load(src)
    t = d["t"]
    sess_idx = d["sess_idx"]
    tod = build_tod(t, sess_idx)
    print(f"  {split}: shape={tod.shape}", flush=True)
    print(f"  unique sess_idx={np.unique(sess_idx)}, t range=[{t.min()},{t.max()}]", flush=True)
    print(f"  tod_sin range=[{tod[:,0].min():.4f},{tod[:,0].max():.4f}]", flush=True)
    print(f"  is_first_20 frac={tod[:,2].mean():.4f}", flush=True)
    print(f"  is_last_20 frac={tod[:,3].mean():.4f}", flush=True)
    print(f"  is_lunch_prox frac={tod[:,4].mean():.4f}", flush=True)
    out = os.path.join(OUT_DIR, f"tod_{split}.npz")
    np.savez_compressed(out, tod=tod, feat_names=np.array(FEAT_NAMES))
    print(f"  saved -> {out}", flush=True)


if __name__ == "__main__":
    for sp in ("train", "test"):
        process(sp)
    with open(os.path.join(OUT_DIR, "tod_feat_names.txt"), "w") as f:
        for n in FEAT_NAMES:
            f.write(n + "\n")
    print(f"\nfeat_names -> {os.path.join(OUT_DIR, 'tod_feat_names.txt')}", flush=True)
