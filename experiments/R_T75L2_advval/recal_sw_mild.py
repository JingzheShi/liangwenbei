"""Recompute adversarial-val sample weight with milder cap [0.5, 2.0]
(vs original [0.25, 4.0]).
"""
import numpy as np
import json
import os
HERE = "/root/lwb_work_t75_advval"
oof = np.load(os.path.join(HERE, "oof_proba_past.npy"))
p_clip = np.clip(oof, 0.01, 0.99)
sw_raw = p_clip / (1.0 - p_clip)
sw_clip = np.clip(sw_raw, 0.5, 2.0)
sw_norm = sw_clip / sw_clip.mean()
np.save(os.path.join(HERE, "sw_advval_mild.npy"), sw_norm.astype(np.float32))
print("mild stats: mean={:.4f} std={:.4f} p10={:.4f} p50={:.4f} p90={:.4f} min={:.4f} max={:.4f}".format(
    float(sw_norm.mean()), float(sw_norm.std()),
    float(np.percentile(sw_norm, 10)),
    float(np.percentile(sw_norm, 50)),
    float(np.percentile(sw_norm, 90)),
    float(sw_norm.min()), float(sw_norm.max())))
