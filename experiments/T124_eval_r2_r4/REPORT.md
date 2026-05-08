# T124 Report — eval r2 sym-aug + r4 5-LOSO ensemble

**Status:** EVAL RUNNING (will be filled after eval_t124.py completes)

## Context

Two parallel rescues from r2/r4 worker SDK early-exit:

- **r2 (R-T75L2-symaug)**: T75 LGB L2 + sym-coupled group augmentation (3 variants × 3 seeds)
  - `baseline`: per-element aug_a U[0.80, 1.20] only (matches T75 LGB L2)
  - `group_only`: 4-group coupled scale (price/size/flow/derived); per-element OFF
  - `combined`: per-element × group-coupled (multiplied)
- **r4 (R-T75L2-5loso)**: T75 LGB L2 trained 5×LOSO (held-out sym k, k∈0..4) × 3 seeds (7, 13, 42)

## Reference

T75 baseline DE LOSO (T122 3-seed ens) = **+35.30**
iter_017 v2 candidate threshold: any variant/strategy with **+0.5 LOSO over its baseline**

## Results

(populated by eval_t124.py)

## Decision

(populated after results land)
