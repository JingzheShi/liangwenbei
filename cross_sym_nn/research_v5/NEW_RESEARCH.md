# V5 Deep Research: HFT / LOB / Cross-Asset NN (2023-2025)
**Date**: 2026-05-31  
**Task**: Identify best architectures to surpass sae_cross_sym SOTA (+36.35 ± 0.67)  
**Context**: Input (B,5,259), output (B,5), sym-agnostic, instantaneous features, no time-series

---

## Current Baseline Architecture Summary

- **sae_cross_sym (SAEMLPCrossSym)** — Current SOTA +36.35 ± 0.67
  - Per-sym encoder: 259 → 128 → 32 (with LayerNorm + GELU + Dropout)
  - Cross-sym: 1 MHSA layer (4 heads, d=32, 5 tokens) + post-norm residual
  - Decoder: 32 → 128 → 259 (reconstruction loss α=0.3)
  - Pred head: [32+259 → 128 → 64 → 1]
  - Training: noise_sigma=0.035, dropout=0.30
  - ~95k parameters, ~70% GPU util

**Key insight already validated**: Cross-sym attention works (+4.91 vs sae_mlp per sym).  
**Next question**: Can we improve the attention mechanism itself?

---

## Direction 1: Cross-Sectional Attention (2023-2025)

### MASTER: Market-Guided Stock Transformer (Zhang et al., AAAI 2024)
- **出处**: https://arxiv.org/abs/2312.15235, AAAI-2024
- **数学原理**:
  - Intra-stock: temporal MHSA (N1=4 heads) per stock across lookback window
  - Inter-stock: spatial MHSA (N2=2 heads) at each time step across all stocks
  - Market-guided feature selection: x̃ = α(m) ⊙ x, where α(m) = F·softmax_β(Wm+b)
  - Market context m is an external market index embedding
  - Prediction loss: L = Σ_u MSE(r_u, r̂_u)
- **为何适合本任务**: The market-guided AdaLN idea (rescaling features via market state) 
  is directly applicable. Our "market state" = mean of cross-sym encoded latents.
  We can use cross-sym attention output as market proxy without external data.
- **预期参数量**: ~100-150k (matching our scale)
- **预期突破潜力**: ⭐⭐⭐ High — feature gating by context is conceptually strong
- **实现难度**: ⭐⭐ Medium
- **优先级**: P1
- **关键 insight**: Market state = mean(cross_sym_attn_output) → gates per-sym features

### MCI-GRU (2024, arxiv 2410.20679)
- **出处**: https://arxiv.org/abs/2410.20679
- **数学原理**:
  - Multi-head cross-attention for latent market state representation
  - Learns latent market state m from interactions of temporal + cross-sectional features
  - GRU for temporal encoding (not applicable to our instant setting)
- **为何适合本任务**: The cross-attention for market state idea is relevant;
  but the GRU temporal component is excluded (instant features only)
- **预期参数量**: ~80k (without GRU)
- **预期突破潜力**: ⭐⭐ Medium
- **实现难度**: ⭐ Easy (just the cross-attention part)
- **优先级**: P3 (subsumed by MASTER-AdaLN idea)

### Attention Factors for Statistical Arbitrage (Epstein et al., 2024/2025)
- **出处**: https://arxiv.org/abs/2510.11616, ACM ICAIF 2024
- **数学原理**:
  - "Attention Factors" = conditional latent factors from firm characteristic embeddings
  - Jointly identifies similar assets (cross-asset attention) + mispricing signal
  - Residual portfolio identification via general sequence model
  - Net Sharpe > 2.3 after transaction costs on 24-year US equity data
- **为何适合本任务**: Cross-asset latent factor learning is our core problem.
  The factor extraction via attention could replace our simple MHSA aggregation.
- **预期参数量**: ~100k
- **预期突破潜力**: ⭐⭐⭐ High (strong empirical results)
- **实现难度**: ⭐⭐ Medium (factor extraction head)
- **优先级**: P2

---

## Direction 2: Self-Supervised / Reconstruction-aware Architectures

### VIME: Value Imputation and Mask Estimation (Yoon et al., NeurIPS 2020)
- **出处**: https://arxiv.org/abs/2003.08013, NeurIPS 2020
- **数学原理**:
  - Corruption: x̃ = m ⊙ x_random + (1-m) ⊙ x, m ~ Bernoulli(p_m=0.3)
  - Pretext Task 1: reconstruction → x̂ = decoder(encoder(x̃))
  - Pretext Task 2: mask estimation → m̂ = mask_head(encoder(x̃))
  - Loss: L_recon + L_mask_bce + L_pred
  - L_mask_bce = BCE(m̂, m) forces encoder to learn feature correlations
- **为何适合本任务**: We already do reconstruction (Task 1). Adding Task 2 (mask
  estimation) costs only 1 linear layer (d_latent → n_feat) and forces the encoder
  to understand which features tend to covary — crucial for financial data.
- **预期参数量**: +n_feat*d_latent ≈ +8288 params (~9%)
- **预期突破潜力**: ⭐⭐⭐ High — masks force feature correlation learning
- **实现难度**: ⭐ Easy
- **优先级**: P1

### SCARF: Self-Supervised Contrastive Learning (Bahri et al., ICLR 2022)
- **出处**: https://arxiv.org/abs/2106.15147, ICLR 2022
- **数学原理**:
  - Creates views by corrupting random feature subset: x̃ = corruptMask(x, p=0.6)
  - Replaces masked features with marginal samples drawn from training distribution
  - InfoNCE loss: pushes x_clean and x_corrupt together, others apart
  - L = L_pred + β * InfoNCE(z_clean, z_corrupt)
  - Proved to improve classification accuracy with label noise
- **为何适合本任务**: Our SAE already has corruption (noise_sigma). Adding contrastive
  objective forces the latent to be invariant to feature noise — stronger regularization
  than pure reconstruction. Positive pair = same sym same time (clean/noisy).
  Negatives = other syms or other time steps in batch.
- **预期参数量**: +projection head (32→64→32) ≈ +6k params
- **预期突破潜力**: ⭐⭐ Medium (feature noise robustness)
- **实现难度**: ⭐⭐ Medium
- **优先级**: P2

### MaskTab / ReMasker: Masked Tabular Autoencoder (2024-2025)
- **出处**: https://arxiv.org/abs/2605.11408 (MaskTab, May 2025)
- **数学原理**:
  - Siamese twin-path: one branch gets full input, other gets masked input
  - Joint optimization: L = α·L_recon_masked + β·L_supervised_clean
  - Dedicated missing-value token for masked positions
  - Scaling laws validated: MaskTab > XGBoost at high feature dim
- **为何适合本任务**: We have 259 features. The dual-path (noisy for SSL, clean for pred)
  is already partially done in our architecture. Formalizing it with separate 
  Siamese encoders or a token-masking scheme could be stronger.
- **预期参数量**: ~2x encoder params (Siamese) or same (shared with mask token)
- **预期突破潜力**: ⭐⭐ Medium
- **实现难度**: ⭐⭐ Medium
- **优先级**: P2

### SAINT: Row Attention + Contrastive Pre-Training (Somepalli et al., 2021)
- **出处**: https://arxiv.org/abs/2106.01342
- **数学原理**:
  - Alternating column-wise (feature) attention + row-wise (inter-sample) attention
  - Row attention: given a batch (B, n_feat), each row attends to all other rows
    via cross-attention: SAB(X_i, X_{j≠i})
  - Pretraining: contrastive (CutMix + Mixup views) + denoising reconstruction
  - Row attention = "inter-sample attention" — each data point learns from neighbors
- **为何适合本任务**: Our 5 syms are the "rows" (inter-sample axis). We already do
  cross-sym attention which is analogous to row attention. The CutMix augmentation
  for contrastive pretraining is a nice addition.
- **预期参数量**: Depends on inter-sample head size; +10-20k
- **预期突破潜力**: ⭐⭐ Medium (already captured by existing cross-sym attn)
- **实现难度**: ⭐⭐ Medium
- **优先级**: P3

---

## Direction 3: Mixture of Experts / Conditional Computation

### Content-Based Sparse MoE for Sym Features (Adaptive MoE 2024)
- **出处**: Adapted from "Adaptive Market Intelligence: A Mixture of Experts Framework
  for Volatility-Sensitive Stock Forecasting" (2025, arxiv 2508.02686) + 
  Switch Transformer (Fedus et al., 2021)
- **数学原理**:
  - K=4 experts, each: Linear(d_latent, d_ff) → SiLU → Linear(d_ff, d_latent)
  - Gating: g = softmax(W_gate @ z)  # (B*5, 4)
  - Top-2 routing: keep only top-2 weights, renormalize
  - Output: Σ_{k∈top2} g_k * Expert_k(z)
  - Load-balancing loss: L_aux = 0.01 * num_experts * Σ_k f_k * P_k
    where f_k = fraction of tokens routed to expert k
  - **SYM-AGNOSTIC**: routing based purely on feature content z, NOT sym index
- **为何适合本任务**: sym3/sym4 dominate PnL → their latent representations likely
  cluster differently. 4 experts let different "market regimes" emerge without
  explicitly conditioning on sym index.
- **预期参数量**: +4 * (32*64 + 64*32) = +16k params
- **预期突破潜力**: ⭐⭐ Medium (theoretical motivation strong, empirical mixed at small scale)
- **实现难度**: ⭐⭐ Medium
- **优先级**: P2

### Per-Expert-Per-Sym INVALID
- ❌ REJECTED: per-sym expert would violate sym-agnostic constraint
  (routing by sym index 0-4 is forbidden)

---

## Direction 4: Lead-lag / Cross-sym Relations

### Feature Difference/Ratio as Cross-Sym Signal
- **出处**: No specific paper; derived from pairs trading / lead-lag NN literature
- **数学原理**:
  - Feature interaction: Δf_{i,j} = f_i - f_j (or f_i/f_j)
  - Reference: LOB mid-price prediction papers often use cross-instrument spread features
  - For instant features: cross_diff[i] = x[i] - mean(x[j for j≠i])
    This is equivalent to z - mean(z) normalization over the sym axis
- **为何适合本任务**: We already do this implicitly in cross-sym attention.
  Adding explicit difference features is redundant if attention is expressive enough.
  Verdict: SKIP — attention handles this automatically.
- **优先级**: P3 (NOT implementing, just documenting)

---

## Direction 5: Attention Improvements (2024-2025)

### Differential Attention (Ye et al., Microsoft/Tsinghua, Oct 2024)
- **出处**: https://arxiv.org/abs/2410.05258, Microsoft Research
- **数学原理**:
  - Split Q → Q1, Q2 ∈ ℝ^(B,h,n,d/2); split K → K1, K2 ∈ ℝ^(B,h,n,d/2)
  - A1 = softmax(Q1 K1^T / √(d/2))  # (B, h, 5, 5)
  - A2 = softmax(Q2 K2^T / √(d/2))  # (B, h, 5, 5)
  - DiffAttn = (A1 - λ·A2) @ V  # noise cancellation
  - λ = exp(⟨λ_q1, λ_k1⟩) - exp(⟨λ_q2, λ_k2⟩) + λ_init
  - λ_init = 0.8 - 0.6·exp(-0.3·(layer-1))  → ~0.2 for layer 1
  - Per-head RMSNorm then scale by (1 - λ_init)
  - Analogy: differential amplifier — common-mode noise cancelled
- **为何适合本任务**: Our 5-sym attention has 5×5=25 attention weights per head.
  Many of these are "noise" (irrelevant sym-sym relationships). Diff-Attn explicitly
  cancels the common-mode attention patterns, sharpening focus on signal.
  For financial cross-asset attention, noise cancellation is paramount.
  Small model (1.4B) shows ~25bp improvement; at our tiny scale might be larger.
- **预期参数量**: Same as current MHSA (Q,K,V same total) + 4 lambda vectors (32 params)
- **预期突破潜力**: ⭐⭐⭐⭐ Very High
- **实现难度**: ⭐ Easy (just change attention computation)
- **优先级**: P1 — TOP PRIORITY

### GLU Attention: Gating Values in MHA (Wang, 2025)
- **出处**: https://arxiv.org/abs/2507.00022
- **数学原理**:
  - Standard MHA: Attn_out = softmax(QK^T/√d) @ V @ W_out
  - GLU-Attn: V_proj ∈ R^{d→2d}, split → V1, V2 ∈ R^{B,h,n,d}
    V_gated = V1 * SiLU(V2)  # (B, h, n, d)
    Attn_out = softmax(QK^T/√d) @ V_gated @ W_out
  - Zero overhead claimed (achievable via weight tying or reduced d)
  - Reported: improvements in performance + convergence speed
  - Compatible with FlashAttention, RoPE, GQA
- **为何适合本任务**: Gating the values forces each attention head to selectively
  "activate" which sym information to pass. In our 5-sym attention, this means
  each sym can learn to gate which aspects of other syms' latents it uses.
  Very cheap modification, high expected ROI.
- **预期参数量**: +d_latent² ≈ +1024 params for doubled V projection
- **预期突破潜力**: ⭐⭐⭐ High (low cost, good theoretical basis)
- **实现难度**: ⭐ Very Easy
- **优先级**: P1

### SwiGLU FFN: Gated FFN Replacement (Noam Shazeer, 2020)
- **出处**: "GLU Variants Improve Transformer" arxiv 2002.05202;  
  Now standard in LLaMA, PaLM, Mistral etc.
- **数学原理**:
  - Standard: FFN(x) = GELU(xW1+b1)(W2+b2), W1: d→4d, W2: 4d→d
  - SwiGLU: FFN(x) = (SiLU(xW1) ⊙ xW2) W3
    W1,W2: d→(8/3)d, W3: (8/3)d→d  (iso-parameter with 4d)
    Or simpler: W1,W2: d→2d, W3: 2d→d (saves params)
  - Controls information flow via learned gating
  - Used in LLaMA1/2/3, Mistral, Gemma — rock-solid improvement
- **为何适合本任务**: Replacing GELU FFN in our encoder + decoder with SwiGLU
  is a well-proven improvement. The gating introduces adaptivity to feature values
  — helpful when different features are relevant at different market states.
- **预期参数量**: ~same (adjust W dimensions to match)
- **预期突破潜力**: ⭐⭐ Medium (but very reliable)
- **实现难度**: ⭐ Very Easy
- **优先级**: P1

### Adaptive LayerNorm (AdaLN) Conditioned on Cross-sym Context
- **出处**: "Scalable Diffusion Models with Transformers" Peebles & Xie, ICCV 2023
  Applied to market context following MASTER (2024)
- **数学原理**:
  - Standard LN: ŷ = γ⊙(y-μ)/σ + β with fixed γ, β
  - AdaLN: γ, β are derived from context c: [γ, β] = MLP(c)
  - Market context: ctx = mean(cross_sym_attn_output, dim=sym)  # (B, d)
  - Per-sym conditioned: z_i = γ(ctx) ⊙ LN(z_attn_i) + β(ctx)
  - AdaLN-Zero: init scale MLP outputs to 0 (stable training)
- **为何适合本任务**: After cross-sym attention aggregates market state, using it
  to adaptively scale/shift each sym's latent is a natural extension.
  The market state (e.g., "high spread regime") should modulate feature importance.
- **预期参数量**: +d*2*d = +2048 params for AdaLN MLP
- **预期突破潜力**: ⭐⭐⭐ High
- **实现难度**: ⭐⭐ Medium
- **优先级**: P1

---

## Architecture Combination Analysis

### Why Differential Attention is Top Priority

For our specific 5-sym attention:
- Each forward pass computes 5×5=25 attention weight pairs per head
- With 4 heads: 100 total attention weights
- Financial cross-sym signals are SPARSE: only a few sym pairs are predictive
- Standard softmax spreads probability mass over all 5 syms even for irrelevant ones
- Diff-Attn's noise cancellation = promotes sparse attention naturally
- Lambda ≈ 0.2 for layer 1: the second softmax map "subtracts" common-mode noise

### Why GLU-V + SwiGLU Together

GLU at value level + SwiGLU at FFN level = gating at two computation stages:
- Stage 1 (GLU-V): which sym features to aggregate
- Stage 2 (SwiGLU): which aggregated features to pass to prediction head
These are additive improvements, easy to combine.

### Why Mask Estimation SSL is Orthogonal

The mask estimation pretext task operates at the ENCODER level, independently of
the cross-sym attention. It forces the encoder to learn feature co-occurrence patterns.
This is purely additive to any attention improvement.

### MoE: Conditional at Scale

At our scale (d_latent=32, ~95k params), MoE might underfit if too many experts.
Recommend 3 experts, top-2 gating, small expert hidden dim=64.

---

## Summary Table: 8 Candidates

| Rank | Name | Paper | Expected Gain | Complexity | Priority |
|------|------|-------|---------------|------------|----------|
| 1 | v5_diff_attn | Diff Transformer (MS 2024) | +0.5~+2.0 | Low | P1 |
| 2 | v5_glu_v_attn | GLU Attention (2025) | +0.3~+0.8 | Very Low | P1 |
| 3 | v5_mask_ssl | VIME (NeurIPS 2020) | +0.3~+1.0 | Low | P1 |
| 4 | v5_swiglu_sae | SwiGLU (2020/standard) | +0.2~+0.6 | Very Low | P1 |
| 5 | v5_market_adaln | MASTER+AdaLN (AAAI 2024) | +0.4~+1.5 | Medium | P1 |
| 6 | v5_content_moe | Sparse MoE (Switch Transformer) | +0.2~+1.0 | Medium | P2 |
| 7 | v5_scarf | SCARF (ICLR 2022) | +0.2~+0.5 | Medium | P2 |
| 8 | v5_diff_deep | Diff-Attn 2L + SwiGLU | +1.0~+2.5 | Medium | P2 |

### Key Hypothesis for v5

> The bottleneck in sae_cross_sym is the **quality of cross-sym attention** 
> (noise in 5×5 attention weights) and **encoder regularization quality** 
> (reconstruction only vs dual pretext). 
>
> v5_diff_attn addresses the former; v5_mask_ssl addresses the latter.
> These two should be the first two experiments.

---

## Literature Survey: What was NOT Found (But Searched)

1. **Hawkes-NN**: REJECTED (stateless violation — requires event history)
2. **SetTransformer ISAB**: Already tested (underfit at v2 config)
3. **FactorVAE**: Too complex for 30-min window; VAE already implemented
4. **SR-GNN**: Graph-based, requires static adjacency — not sym-agnostic
5. **StockMixer**: Time-series based (MLP-Mixer on sequence axis) — excluded
6. **DeepLOB**: LOB-specific CNN, not applicable to our feature space
7. **FlashAttention-3**: Infrastructure optimization only, no accuracy change

---

## References

1. MASTER: Market-Guided Stock Transformer — https://arxiv.org/abs/2312.15235 (AAAI 2024)
2. Differential Transformer — https://arxiv.org/abs/2410.05258 (Microsoft/Tsinghua 2024)
3. GLU Attention Improve Transformer — https://arxiv.org/abs/2507.00022 (2025)
4. VIME: Self-Supervised Tabular — https://arxiv.org/abs/2003.08013 (NeurIPS 2020)
5. SCARF: Contrastive Tabular — https://arxiv.org/abs/2106.15147 (ICLR 2022)
6. SAINT: Row Attention Tabular — https://arxiv.org/abs/2106.01342
7. GLU Variants Improve Transformer — https://arxiv.org/abs/2002.05202 (Shazeer 2020)
8. DiT: Scalable Diffusion Models with Transformers (AdaLN) — ICCV 2023
9. Attention Factors for Statistical Arbitrage — https://arxiv.org/abs/2510.11616 (2024)
10. MCI-GRU Cross-Attention Stock Prediction — https://arxiv.org/abs/2410.20679 (2024)
11. MaskTab Scalable Tabular Pretraining — https://arxiv.org/abs/2605.11408 (2025)
12. Adaptive MoE for Stock Forecasting — https://arxiv.org/abs/2508.02686 (2025)
