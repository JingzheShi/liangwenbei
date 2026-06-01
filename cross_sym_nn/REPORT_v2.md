# Cross-Sym NN v2: Step-Level Early Stopping + Strong Regularization

**Task**: Test the hypothesis that the epoch-0 peak in cross-sym NN training is **overfitting**, not an
MSE/PnL trade-off. Strategy: small models + dropout 0.30 + weight_decay 1e-3 + LR 1e-4 + step-level eval
+ short patience (1500 steps).

**Date**: 2026-05-30
**Baselines (from REPORT.md)**:
- MLP sym-agnostic +31.44 (best)
- SymAttentionModel (old) +29.80
- FT-Transformer +26.73
- HybridCrossAttn +25.21
- CrossConcatMLP +5.50 (failure mode)

---

## 1. Hypothesis Being Tested

> "MSE-PnL is not a trade-off, it's overfitting. best_ep=0 because the model overfits after epoch 0.
> Try small models + strong regularization."  — user judgment

**Falsifiable predictions**:
1. With small models + strong reg, `best_step > 0` (model improves over steps, doesn't peak at init)
2. `val PnL @ best_step` may not match the old large-model ep-0 peak (~+23), but should be a
   **real generalization optimum** rather than an early-training random-prediction fluke
3. The drop from `val PnL @ best_step` to `val PnL @ final` should be small (the early-stop is
   on a flat plateau, not a sharp peak)

---

## 2. Architecture Implementations

All 3 architectures take input `(B, 5, 359)` and produce `(B, 5)` predictions, in the same data
pipeline as `REPORT.md` (group-by-tick, mirror aug, window-z, target_scale-normalized).

### A. SetTransformerISAB
Permutation-invariant cross-sym via Induced Set Attention Blocks (Lee 2019). No `sym_emb` — fully
equivariant to sym permutation.

```python
d_model=64, n_heads=4, n_inducing=4, depth=2, dropout=0.30
```
Architecture: `(B,5,359) → Linear+LN+GELU → ISAB×2 → LayerNorm → Head → (B,5)`
Params: ~160k

### B. DeepSetsEnhanced
Simplest cross-sym baseline: per-sym MLP encode + (max+mean+std) pooling.

```python
d_phi=32, d_hidden=128, d_rho=32, dropout=0.30
```
Architecture: per-sym `Linear(359→128→32)`, agg `[max, mean, std]` → `Linear(96→32)`,
per-sym head `[token, ctx] → Linear → 1`.
Params: ~56k

### C. SupervisedAutoEncoderMLP (SAE-MLP)
Jane-Street-1st (Yirun 2020) architecture, **per-sym independent**, with reconstruction loss as
intrinsic regularizer.

```python
d_latent=32, d_hidden=128, dropout=0.30, recon_alpha=0.3, noise_sigma=0.035
```
Loss = `0.7·MSE(pred, y) + 0.3·MSE(recon, x)`.
Params: ~160k

---

## 3. Training Loop Changes (vs REPORT.md / `train.py`)

| Knob | v1 (`train.py`) | **v2 (`train_v2.py`)** |
|---|---|---|
| Eval cadence | per epoch | **per 200 steps** |
| Early-stop unit | epoch patience=10 | **step patience=1500** |
| LR | 3e-4 | **1e-4** |
| Weight decay | 1e-4 | **1e-3** |
| Warmup | none | **200 steps linear** |
| LR schedule | CosineAnnealing(T_max=50ep) | **LambdaLR cosine on step grid** |
| Dropout | 0.10 | **0.30** |
| Max epochs | 50 | 20 |
| Grad clip | 5.0 | 1.0 |
| Mirror aug | ✓ | ✓ (unchanged) |
| MSE scale aug | 0.8-1.2× | 0.8-1.2× (unchanged) |
| Batch size | 1024 | 1024 |

Key change: train loop now evaluates and possibly saves checkpoint inside the inner batch loop, not
between epochs. With 589,440 train groups / bs=1024 = 576 steps/epoch, eval-every-200 ≈ 3 checkpoints
per epoch (vs 1 in v1).

---

## 4. Results

<!-- TODO: filled in by analyze_v2.py after all runs complete -->

### 4.1 Summary Table

| arch | n_seeds | mean val_pnl | mean test_pnl | std test_pnl | best test_pnl | mean best_step |
|---|---|---|---|---|---|---|
| _isab_     | -- | -- | -- | -- | -- | -- |
| _deepsets_ | -- | -- | -- | -- | -- | -- |
| _sae_mlp_  | -- | -- | -- | -- | -- | -- |

### 4.2 Per-seed Results

<!-- TODO -->

### 4.3 Per-sym Test PnL (best run per arch)

<!-- TODO -->

---

## 5. Overfit Diagnostic

The central diagnostic: **does `best_step > 0`?** If yes, overfit hypothesis is supported.

| arch | seed | best_step | total_steps | val_pnl@best | val_pnl@final | drop |
|---|---|---|---|---|---|---|

<!-- TODO -->

**Interpretation**: ...

---

## 6. v1 vs v2 Comparison

| Model | v1 (epoch-level) | v2 (step-level, small+reg) | Δ |
|---|---|---|---|
| sym_attn (90k)         | +29.80 | _n/a (arch unchanged)_ | — |
| isab_small (160k)      | n/a    | -- | -- |
| sae_mlp_small (160k)   | n/a    | -- | -- |
| deepsets_small (56k)   | n/a    | -- | -- |
| MLP baseline (sym-agn) | +31.44 | _unchanged baseline_ | — |

---

## 7. Did Small + Strong Reg Solve Overfit?

<!-- TODO: 5-line conclusion -->

---

## 8. Defense Talking Points (Updated)

<!-- TODO -->

---

## 9. Files / WandB

- Code: `models.py` (added `MAB/ISAB/SetTransformerISAB/DeepSetsEnhanced/SupervisedAutoEncoderMLP`)
- Train script: `train_v2.py`
- Run scripts: `run_all_v2.sh`
- Per-run artifacts: `runs_v2/{arch}_small_s{seed}/{results.json, preds.npz, model.pt}`
- Logs: `logs_v2/{arch}_s{seed}.log`
- WandB: https://wandb.ai/jingzheshi/liangwenbei-cross-sym (run names: `{arch}_small_s{seed}`)
