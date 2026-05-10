# T188 Mega Ensemble (50+50)

## Status: TRAINING IN PROGRESS

### Timeline
- Started: 2026-05-10 ~22:58 UTC
- Expected finish (NN): ~23:48 UTC
- Expected finish (LGB): ~24:00 UTC

### Plan
- Phase 1+2: Train 50 NN seeds (T81-style anchor + T170 SPO+ fine-tune) on vastai-r2 remote GPU
- Phase 3: Train 50 LGB seeds (M7, 5 base HP configs × 10 seeds each) on remote GPU  
- Phase 4: Build pkg + smoke test + zip locally

### Per-seed timing (measured)
- Phase 1 (T81 anchor): ~19s (early stops at ep 5, best_ep=0)
- Phase 2 (SPO+ DFL): ~40s (11 fixed epochs)
- Total NN: ~60s/seed × 50 = ~50 min
- LGB: ~15s/seed × 50 = ~12 min

### NN HP diversity
seed % 3 → lr: [1e-4, 3e-4, 1e-3]
seed % 4 → dropout: [0.05, 0.10, 0.15, 0.20]  
seed % 3 → batch_p1: [2048, 4096, 8192]
All use Xavier init with torch.manual_seed(S) for diversity

### Expected package
- 50 nn_h60_seed{1..50}.npz (~3MB each = ~150MB)
- 50 model_h60_seed{1..50}.txt (~5MB each = ~250MB)
- Total: ~400MB (< 2GB limit ✓)
- Weights: w_nn=1.0, w_lgb=0.5 (T150-tuned, unchanged)

### Expected outcome
- Similar to T170 (~+34.5) ± small variance noise
- Possibly +0.5 to -0.5 delta vs T170 5+5 (user aware of variance saturation)
