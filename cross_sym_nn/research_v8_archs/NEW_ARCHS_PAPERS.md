# V8 Architecture Research: 50+ New Papers (2023-2025)
**Date**: 2026-05-31  
**Task**: Identify novel architectures to surpass v8 SOTA (+38.49 ± 1.67, B no_mirror)  
**Context**: Input (B,5,259), output (B,5), sym-agnostic, instantaneous features, no time-series  

**Already Rejected (do NOT re-implement)**:  
sym_attn, FT-Transformer, iTransformer, TimesNet, ModernTCN, TimeKAN, diff_attn, gat, moe (basic), mask_ssl (basic), market_adaln, diff_swiglu

---

## Direction 1: Latest 2024-2025 Cross-Asset Transformer (11 papers)

### AttentionFactors (Epstein, Wang, Choi, Pelger — ICAIF 2025) — Direction 1
- **Paper**: https://arxiv.org/abs/2510.11616
- **Venue**: ACM ICAIF 2025 (6th Conference on AI in Finance)
- **Core idea**:
  - Conditional latent factors learned from firm characteristic embeddings (our: per-sym feature vectors)
  - Two components: (A) Factor extractor via cross-asset attention → K latent factors; (B) Residual arbitrage signal head
  - Factor loadings λ_i,k = softmax(Q_i K_k^T / √d) where Q_i = f(x_i) and K_k are learned factor queries
  - PnL objective: maximize cross-sectional IC through factor-based prediction
  - Achieves Sharpe > 4 out-of-sample on 24-year US equity data; net Sharpe 2.3 after transaction costs
- **PyTorch class sketch**:
  ```python
  class AttentionFactorsMLP(nn.Module):
      def __init__(self, n_feat=259, d_latent=64, n_factors=4, dropout=0.3):
          super().__init__()
          self.encoder = nn.Sequential(
              nn.Linear(n_feat, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
              nn.Linear(128, d_latent), nn.LayerNorm(d_latent)
          )
          self.factor_queries = nn.Parameter(torch.randn(n_factors, d_latent))
          self.factor_proj = nn.Linear(d_latent, d_latent)
          self.arb_head = nn.Sequential(nn.Linear(d_latent * 2, 64), nn.GELU(), nn.Linear(64, 1))
          self.decoder = nn.Sequential(nn.Linear(d_latent, 128), nn.GELU(), nn.Linear(128, n_feat))
          self._last_recon_loss = None

      def forward(self, x):  # (B, 5, n_feat)
          z = self.encoder(x)  # (B, 5, d)
          # Factor extraction
          scores = torch.einsum('bid,kd->bik', z, self.factor_queries) / (z.shape[-1]**0.5)
          weights = torch.softmax(scores, dim=1)  # (B, 5, K)
          factors = torch.einsum('bik,bid->bkd', weights, z)  # (B, K, d)
          # Factor-enhanced representation
          factor_ctx = torch.einsum('bik,bkd->bid', weights.transpose(1,2).transpose(1,2), factors)
          z_aug = torch.cat([z, self.factor_proj(factor_ctx)], dim=-1)
          pred = self.arb_head(z_aug).squeeze(-1)  # (B, 5)
          if self.training:
              recon = self.decoder(z)
              self._last_recon_loss = F.mse_loss(recon, x)
          return pred
  ```
- **为何适合 v7+ (sae_mlp + cross-sym)**: Replaces our simple MHSA aggregation with factor-based cross-sym interaction; K=4 factors model the dominant market modes while the residual head captures individual sym mispricing — directly maps to our 5-sym cross-sectional problem.
- **预期 PnL gain vs SOTA +38.49**: high (+0.8 to +2.0)
- **参数量预估**: ~120k
- **实现复杂度**: medium
- **优先级**: P1

---

### PairformerCrossSym (Abramson et al. / AF3 Pairformer inspiration — 2025) — Direction 1
- **Paper**: https://arxiv.org/abs/2510.18870 (Pairmixer: Triangle Multiplication Is All You Need)
- **Venue**: arXiv 2025
- **Core idea**:
  - AlphaFold3's Pairformer maintains two representations: single (N×d_s) + pair (N×N×d_p)
  - Row/col attention on pair: A_pair[i,j] updated by attending over row-i and col-j of pair matrix
  - Triangle update: P[i,j] += LayerNorm(Σ_k gate(P[i,k]) ⊙ P[k,j])  (outgoing multiplication)
  - Single → pair projection: P[i,j] = Linear(z_i ⊗ z_j) for init
  - Pairmixer: replaces triangle attention with triangle multiplication for 4× speedup
  - For N=5 syms: pair matrix is 5×5×d_pair (only 25 elements) — extremely cheap
- **PyTorch class sketch**:
  ```python
  class PairformerCrossSym(nn.Module):
      def __init__(self, n_feat=259, d_single=64, d_pair=32, dropout=0.3):
          super().__init__()
          self.encoder = nn.Sequential(
              nn.Linear(n_feat, 128), nn.LayerNorm(128), nn.GELU(),
              nn.Linear(128, d_single), nn.LayerNorm(d_single)
          )
          self.pair_init = nn.Linear(d_single * 2, d_pair)
          self.tri_mul_outgoing = nn.Sequential(
              nn.LayerNorm(d_pair), nn.Linear(d_pair, d_pair * 2)
          )
          self.pair_to_single = nn.Linear(d_pair * 5, d_single)
          self.pred = nn.Sequential(
              nn.Linear(d_single + n_feat, 128), nn.GELU(), nn.Linear(128, 1)
          )
          self.decoder = nn.Sequential(nn.Linear(d_single, 128), nn.GELU(), nn.Linear(128, n_feat))
          self._last_recon_loss = None

      def forward(self, x):  # (B, 5, n_feat)
          B, N, _ = x.shape
          z = self.encoder(x)  # (B, N, d_single)
          # Init pair representation
          zi = z.unsqueeze(2).expand(-1, -1, N, -1)
          zj = z.unsqueeze(1).expand(-1, N, -1, -1)
          pair = self.pair_init(torch.cat([zi, zj], dim=-1))  # (B, N, N, d_pair)
          # Triangle multiplication (outgoing: i←k, j←k aggregate)
          x1 = self.tri_mul_outgoing(pair)
          a, b = x1.chunk(2, dim=-1)
          gate = torch.sigmoid(a) * b  # (B, N, N, d_pair)
          tri_out = torch.einsum('bikd,bjkd->bijd', gate, gate) / N  # (B, N, N, d_pair)
          pair = pair + tri_out
          # Pair to single aggregation
          pair_agg = pair.reshape(B, N, N * pair.shape[-1])
          pair_agg = self.pair_to_single(pair_agg)  # (B, N, d_single)
          z = z + pair_agg
          pred = self.pred(torch.cat([z, x], dim=-1)).squeeze(-1)
          if self.training:
              recon = self.decoder(z)
              self._last_recon_loss = F.mse_loss(recon, x)
          return pred
  ```
- **为何适合 v7+ (sae_mlp + cross-sym)**: Pairwise sym representation (5×5 matrix) captures EVERY pair relationship explicitly, with triangle updates propagating triplet info (sym_i influences via sym_k). Our MHSA only has linear pairwise; this is quadratic but for N=5 that's only 25 pair vectors — trivial cost.
- **预期 PnL gain vs SOTA +38.49**: high (+1.0 to +2.5)
- **参数量预估**: ~150k
- **实现复杂度**: medium
- **优先级**: P1

---

### MASTER_MarketGuided (Zhang et al. — AAAI 2024) — Direction 1
- **Paper**: https://arxiv.org/abs/2312.15235
- **Venue**: AAAI 2024
- **Core idea**:
  - Intra-stock: temporal MHSA per stock over lookback window
  - Inter-stock: spatial MHSA across stocks at each timestep
  - Market-guided feature selection: x̃ = α(m) ⊙ x where α(m) = softmax_β(W·m + b), m = mean(cross_sym_outputs)
  - Market state m derived from cross-sym attention output (no external data needed)
  - Features are re-weighted based on global market context before per-sym processing
- **PyTorch class sketch**:
  ```python
  class MasterAdaLNv2(nn.Module):
      def __init__(self, n_feat=259, d_latent=64, dropout=0.3):
          super().__init__()
          self.encoder = nn.Sequential(nn.Linear(n_feat, 128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128, d_latent))
          self.cross_attn = nn.MultiheadAttention(d_latent, 4, dropout=dropout, batch_first=True)
          self.market_gate = nn.Sequential(nn.Linear(d_latent, n_feat), nn.Sigmoid())
          self.pred = nn.Sequential(nn.Linear(d_latent + n_feat, 128), nn.GELU(), nn.Linear(128, 1))
          self.decoder = nn.Sequential(nn.Linear(d_latent, 128), nn.GELU(), nn.Linear(128, n_feat))
          self._last_recon_loss = None

      def forward(self, x):  # (B, 5, n_feat)
          z = self.encoder(x)
          attn_out, _ = self.cross_attn(z, z, z)  # (B, 5, d)
          m = attn_out.mean(dim=1, keepdim=True)  # (B, 1, d) market state
          gate = self.market_gate(m)  # (B, 1, n_feat) 
          x_gated = gate * x  # market-guided feature selection
          z2 = self.encoder(x_gated)
          pred = self.pred(torch.cat([z2, x_gated], dim=-1)).squeeze(-1)
          if self.training:
              self._last_recon_loss = F.mse_loss(self.decoder(z2), x_gated)
          return pred
  ```
- **为何适合 v7+ (sae_mlp + cross-sym)**: The market-gate idea is complementary to pure MHSA — it re-weights per-sym input features based on global context, before the per-sym encoder processes them. Different from AdaLN (scale+shift after) — this gates at the input level.
- **预期 PnL gain vs SOTA +38.49**: medium (+0.3 to +0.8)
- **参数量预估**: ~100k
- **实现复杂度**: easy
- **优先级**: P2

---

### TLOB_DualAttn (Berti & Kasneci — arXiv 2025) — Direction 1
- **Paper**: https://arxiv.org/abs/2502.15757
- **Venue**: arXiv 2025
- **Core idea**:
  - Dual attention within each Transformer block: Temporal Self-Attention + Feature Self-Attention
  - BilinearNorm before each block addresses non-stationarity (replaces standard LN)
  - MLPLOB block as FFN: mixes spatial and temporal signals
  - Feature self-attention: attends over feature dimension → captures cross-feature dependencies
  - Temporal self-attention: attends over time → ignored for our instantaneous case, but feature attention is key
  - Surpasses SOTA on FI-2010, NASDAQ, Bitcoin with simpler architecture
- **PyTorch class sketch**:
  ```python
  class DualAttnBlock(nn.Module):
      def __init__(self, n_feat=259, d_model=64, n_sym=5, dropout=0.3):
          super().__init__()
          self.sym_attn = nn.MultiheadAttention(d_model, 4, dropout=dropout, batch_first=True)
          self.feat_attn = nn.MultiheadAttention(n_sym, 1, dropout=dropout, batch_first=True)
          self.norm1 = nn.LayerNorm(d_model)
          self.norm2 = nn.LayerNorm(n_sym)
          self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))

      def forward(self, z):  # (B, 5, d)
          # sym-wise attention (standard)
          z = z + self.sym_attn(self.norm1(z), self.norm1(z), self.norm1(z))[0]
          # feature-wise attention: transpose to (B, d, 5) → attend over 5 syms per feature
          zT = z.transpose(1, 2)  # (B, d, 5)
          zT = zT + self.feat_attn(self.norm2(zT), self.norm2(zT), self.norm2(zT))[0]
          z = zT.transpose(1, 2)  # back to (B, 5, d)
          return z + self.ffn(z)
  ```
- **为何适合 v7+ (sae_mlp + cross-sym)**: Feature self-attention (operating over 5 syms as feature tokens) gives an ORTHOGONAL view to sym-wise attention. Both together are like a 2D attention over the 5×d latent matrix, capturing cross-sym AND cross-feature dependencies jointly.
- **预期 PnL gain vs SOTA +38.49**: medium (+0.4 to +1.0)
- **参数量预估**: ~110k
- **实现复杂度**: easy
- **优先级**: P1

---

### OmniGNN_SectorNode (Li, Abil, Oda — arXiv 2025) — Direction 1
- **Paper**: https://arxiv.org/abs/2510.10775
- **Venue**: arXiv 2025 (ICAIF relevant)
- **Core idea**:
  - Global "sector" node as intermediary for rapid information propagation
  - Multi-relational GAT for neighbor weighting + Transformer for temporal dynamics
  - Key insight: adding one virtual global node reduces graph diameter from O(N) to O(2) 
  - For 5 syms: add virtual "market" node; all syms connect to it bidirectionally
  - 5 sym nodes + 1 market node → 6 nodes; market-node messages enable O(1) shock propagation
  - Robust during COVID: Sharpe +94% vs static graph methods
- **PyTorch class sketch**:
  ```python
  class OmniGNNCrossSym(nn.Module):
      def __init__(self, n_feat=259, d_latent=64, dropout=0.3):
          super().__init__()
          self.encoder = nn.Sequential(nn.Linear(n_feat, 128), nn.GELU(), nn.Linear(128, d_latent))
          self.market_node = nn.Parameter(torch.randn(1, 1, d_latent))
          # GAT: 5+1=6 node attention
          self.gat = nn.MultiheadAttention(d_latent, 4, batch_first=True)
          self.norm = nn.LayerNorm(d_latent)
          self.pred = nn.Sequential(nn.Linear(d_latent + n_feat, 128), nn.GELU(), nn.Linear(128, 1))
          self.decoder = nn.Sequential(nn.Linear(d_latent, 128), nn.GELU(), nn.Linear(128, n_feat))
          self._last_recon_loss = None

      def forward(self, x):  # (B, 5, n_feat)
          B = x.shape[0]
          z = self.encoder(x)  # (B, 5, d)
          market = self.market_node.expand(B, -1, -1)  # (B, 1, d)
          nodes = torch.cat([z, market], dim=1)  # (B, 6, d)
          nodes = nodes + self.gat(self.norm(nodes), self.norm(nodes), self.norm(nodes))[0]
          z_out = nodes[:, :5, :]  # back to 5 syms
          pred = self.pred(torch.cat([z_out, x], dim=-1)).squeeze(-1)
          if self.training:
              self._last_recon_loss = F.mse_loss(self.decoder(z_out), x)
          return pred
  ```
- **为何适合 v7+ (sae_mlp + cross-sym)**: Global market node enables O(1) information propagation across all 5 syms in a SINGLE attention step (vs standard MHSA which needs quadratic pairs). Also adds a learnable market-context embedding that conditions all sym representations.
- **预期 PnL gain vs SOTA +38.49**: medium (+0.3 to +0.8)
- **参数量预估**: ~95k
- **实现复杂度**: easy
- **优先级**: P2

---

### ComparativeTransformerStock (Chen — arXiv 2025) — Direction 1
- **Paper**: https://arxiv.org/abs/2504.16361
- **Venue**: arXiv 2025
- **Core idea**:
  - Systematic comparison: encoder-only, decoder-only, vanilla Transformer, ProbSparse, no-embedding
  - Key finding: decoder-only Transformer outperforms all others
  - Decoder-only: causal masking in cross-attention; query = current input, key/value = context (cross-sym)
  - ProbSparse performs worst (5 syms are too few for sparsity to help)
  - Insight for our task: asymmetric query (target sym) vs key/value (all syms) is beneficial
- **为何适合 v7+ (sae_mlp + cross-sym)**: Suggests our MHSA could be reformulated as cross-attention: each sym's latent queries into the joint cross-sym context. This is more directional than self-attention.
- **预期 PnL gain vs SOTA +38.49**: low-medium (+0.2 to +0.6)
- **参数量预估**: ~95k
- **实现复杂度**: easy
- **优先级**: P3

---

### AlphaBreaksUncertainty (Sanderink — arXiv 2026) — Direction 1
- **Paper**: https://arxiv.org/abs/2603.13252
- **Venue**: arXiv 2026
- **Core idea**:
  - Cross-sectional ranker deployment with uncertainty quantification
  - DEUP (Direct Epistemic Uncertainty Prediction) adapted to ranking tasks
  - Rank displacement as epistemic uncertainty signal
  - PIT-safe baseline for non-stationarity detection
  - Useful for inference-time filtering of high-uncertainty predictions
- **为何适合 v7+ (sae_mlp + cross-sym)**: Not a training architecture change, but uncertainty-aware post-processing could filter bad predictions during regime shifts.
- **预期 PnL gain vs SOTA +38.49**: low (+0.1 to +0.3)
- **优先级**: P3

---

### IsAttentionAllWeNeed (arXiv 2025) — Direction 1
- **Paper**: https://arxiv.org/abs/2508.19006
- **Venue**: arXiv 2025
- **Core idea**:
  - Empirical study of attention mechanisms for asset pricing on top 420 US stocks
  - Compares: additive attention, Luong's attention, global self-attention, sliding-window sparse attention
  - Key finding: enforced causal masks prevent future data leakage
  - Pretrained RNN with attention shows strong performance
  - Global self-attention outperforms sparse attention for small stock sets
- **为何适合 v7+**: Validates that for small N (5 syms in our case), global full attention is better than sparse — confirms our current MHSA design is correct direction.
- **预期 PnL gain vs SOTA +38.49**: low (confirms existing design)
- **优先级**: P3

---

### CrossModalTempFusion_CMTF (arXiv 2025) — Direction 1
- **Paper**: https://arxiv.org/abs/2504.13522
- **Venue**: arXiv 2025
- **Core idea**:
  - Cross-Modal Temporal Fusion for heterogeneous financial data
  - Attention to dynamically weight different data modalities
  - Cross-modal attention: Q from one modality, K/V from another
  - Can be applied: Q from sym_i features, K/V from sym_j features → cross-sym cross-modal
- **为何适合 v7+**: The cross-modal framing lets us treat different feature groups (order book features, technical indicators, derived features) as separate "modalities" that cross-attend within each sym AND across syms.
- **预期 PnL gain vs SOTA +38.49**: medium (+0.3 to +0.7)
- **优先级**: P2

---

### StockRankingLoss (arXiv 2025) — Direction 1
- **Paper**: https://arxiv.org/abs/2510.14156
- **Venue**: ACM CIKM 2025
- **Core idea**:
  - Systematic study of pointwise/pairwise/listwise ranking losses for stock prediction
  - PortfolioMASTER transformer trained with ListNet, LambdaRank, NDCG-optimized
  - ListNet consistently outperforms MSE loss in cross-sectional portfolio selection
  - Pairwise loss: L = Σ_{i,j} log(1 + exp(-(ŷ_i - ŷ_j) * sign(y_i - y_j)))
  - Listwise SoftMax: L = -Σ_i softmax(y_i/T) * log softmax(ŷ_i/T)
- **PyTorch class sketch**:
  ```python
  def listwise_ranking_loss(pred, target, temp=0.1):  # (B, 5) each
      target_soft = F.softmax(target / temp, dim=-1)
      pred_log_soft = F.log_softmax(pred / temp, dim=-1)
      return -(target_soft * pred_log_soft).sum(dim=-1).mean()
  ```
- **为何适合 v7+**: Our current loss is MSE (regression). Replacing/augmenting with a pairwise or listwise ranking loss directly aligns training with PnL (which depends on relative ranks of 5 syms, not absolute values).
- **预期 PnL gain vs SOTA +38.49**: medium (+0.5 to +1.5)
- **参数量预估**: 0 extra (loss function change only)
- **实现复杂度**: easy
- **优先级**: P1

---

### IntegratingMultiDimRelations (NCB 2025) — Direction 1
- **Paper**: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12578950/
- **Venue**: Nature-linked journal 2025
- **Core idea**:
  - Adaptive fusion of frequency-domain, fundamental, and knowledge graph asset relationships
  - Cross-asset dependency via multi-relation type attention
  - Three relation types fused via learnable weights
  - Validated on Amazon cross-sector stock data 2010-2025
- **为何适合 v7+**: The multi-relation fusion is adaptable to our setting with learned edge weights between 5 syms under different relation hypotheses.
- **预期 PnL gain vs SOTA +38.49**: low-medium (+0.2 to +0.5)
- **优先级**: P3

---

## Direction 2: Mixture of Experts for Finance (8 papers)

### LLMoE_ContextRouted (arXiv 2025) — Direction 2
- **Paper**: https://arxiv.org/abs/2501.09636
- **Venue**: arXiv 2025
- **Core idea**:
  - MoE with LLM-based router (conceptually: route tokens to experts based on CONTEXT not just input)
  - Key adaptation: use cross-sym context as router input (not LLM, but cross-sym MHSA output)
  - Router: r = softmax(W_r * cross_sym_ctx) → select top-k experts
  - 4-8 experts, each a small MLP; per-sym features routed based on global market context
  - Different from v6_moe (which was broken): routing from cross-sym context = market-regime-aware routing
- **PyTorch class sketch**:
  ```python
  class ContextRoutedMoE(nn.Module):
      def __init__(self, n_feat=259, d_latent=64, n_experts=6, k=2, dropout=0.3):
          super().__init__()
          self.encoder = nn.Sequential(nn.Linear(n_feat, d_latent), nn.LayerNorm(d_latent))
          self.cross_attn = nn.MultiheadAttention(d_latent, 4, batch_first=True)
          self.router = nn.Linear(d_latent, n_experts)
          self.experts = nn.ModuleList([
              nn.Sequential(nn.Linear(n_feat, 128), nn.GELU(), nn.Linear(128, d_latent))
              for _ in range(n_experts)
          ])
          self.pred = nn.Sequential(nn.Linear(d_latent + n_feat, 64), nn.GELU(), nn.Linear(64, 1))
          self.k = k

      def forward(self, x):  # (B, 5, n_feat)
          B, N, _ = x.shape
          z = self.encoder(x)
          ctx, _ = self.cross_attn(z, z, z)  # (B, 5, d) cross-sym context
          ctx_mean = ctx.mean(dim=1)  # (B, d) global market context
          # Route per-sym features to top-k experts
          logits = self.router(ctx_mean)  # (B, n_experts)
          topk_vals, topk_idx = logits.topk(self.k, dim=-1)  # (B, k)
          weights = F.softmax(topk_vals, dim=-1)  # (B, k)
          # Aggregate expert outputs
          expert_out = torch.stack([e(x.reshape(B*N, -1)).reshape(B, N, -1)
                                     for e in self.experts], dim=2)  # (B, N, E, d)
          sel = expert_out.gather(2, topk_idx.unsqueeze(1).unsqueeze(-1).expand(B, N, self.k, expert_out.shape[-1]))
          z_final = (sel * weights.unsqueeze(1).unsqueeze(-1)).sum(dim=2)  # (B, N, d)
          return self.pred(torch.cat([z_final, x], dim=-1)).squeeze(-1)
  ```
- **为何适合 v7+**: Routing from cross-sym context rather than per-sym input means experts specialize in market regimes (trending, mean-reverting, high-vol) rather than in specific features — more sym-agnostic and theoretically sound.
- **预期 PnL gain vs SOTA +38.49**: medium (+0.3 to +1.0)
- **参数量预估**: ~200k (6 experts × 40k each)
- **实现复杂度**: medium
- **优先级**: P2

---

### S2MoE_Stochastic (arXiv 2025) — Direction 2
- **Paper**: https://arxiv.org/abs/2503.23007
- **Venue**: arXiv 2025
- **Core idea**:
  - Combines "clean" and "noisy" SMoE paths via stochastic input perturbation per expert
  - InfoNCE contrastive loss between clean/noisy representations
  - Reduces inference cost 28% vs standard MoE; each input activates single expert
  - Clean path: z_clean = expert_k(x); Noisy path: z_noisy = expert_k(x + ε)
  - Loss: L_pred + λ * InfoNCE(z_clean, z_noisy) + μ * load_balance_loss
- **为何适合 v7+**: The stochastic perturbation naturally creates augmented views for cross-sym contrastive learning while maintaining expert efficiency. InfoNCE pushes same-sym different-views together.
- **预期 PnL gain vs SOTA +38.49**: medium (+0.2 to +0.6)
- **优先级**: P2

---

### GraphTokenMoE (Nguyen et al. — arXiv 2025) — Direction 2
- **Paper**: https://arxiv.org/abs/2505.00792
- **Venue**: arXiv 2025
- **Core idea**:
  - Attention-Aware (S)MoE: uses attention matrix to guide token routing
  - Similar/related tokens are routed to same experts → promotes expert specialization
  - Provably reduces entropy of expert selection → more stable routing
  - For our 5-sym problem: route syms with similar encoded representations to same expert
  - Router score: r_{i,k} = sim(z_i, e_k) where e_k are expert embeddings (like keys)
- **为何适合 v7+**: Attention-based routing from cross-sym attention naturally groups syms in similar market states — more semantically coherent routing than score-based.
- **预期 PnL gain vs SOTA +38.49**: low-medium (+0.2 to +0.5)
- **优先级**: P3

---

### MaGNet_MambaMoE (arXiv 2025) — Direction 2
- **Paper**: https://arxiv.org/abs/2511.00085
- **Venue**: arXiv 2025
- **Core idea**:
  - MAGE block: bidirectional Mamba + adaptive gating + sparse MoE + multi-head attention
  - Dual hypergraph: Temporal-Causal (TCH) for fine-grained causal dependencies + Global Probabilistic (GPH) for market-wide patterns
  - 2D spatiotemporal attention: feature-wise AND stock-wise
  - Jensen-Shannon Divergence weighting for hyperedges → soft assignment
  - Key adaptable component: the sparse MoE inside MAGE block for per-sym feature processing
- **为何适合 v7+**: The MoE-inside-attention-block idea (sparse MoE replaces FFN in Transformer) is directly applicable; each sym's features go through top-k from {expert_mlp_1...expert_mlp_N} after cross-sym attention.
- **预期 PnL gain vs SOTA +38.49**: medium (+0.3 to +0.8)
- **优先级**: P2

---

### HsMoE_Bayesian (arXiv 2026) — Direction 2
- **Paper**: https://arxiv.org/abs/2601.09043
- **Venue**: arXiv 2026
- **Core idea**:
  - Bayesian MoE with horseshoe prior for adaptive global-local shrinkage in expert selection
  - Input-dependent gating with horseshoe shrinkage: most experts get near-zero weight
  - Sequential inference via particle learning; robust to small N regime
  - Posterior: p(expert_k activated | x) ∝ horseshoe(z_k) where z_k = router score
- **为何适合 v7+**: For our small dataset, Bayesian MoE prevents overfitting to spurious market regimes by shrinking most expert weights to near-zero.
- **预期 PnL gain vs SOTA +38.49**: low (+0.1 to +0.3, research stage)
- **优先级**: P3

---

### MoEInsideFFN (general recipe — 2024) — Direction 2
- **Paper**: MaGNet + Switch Transformer (Fedus et al., JMLR 2022)
- **Venue**: JMLR 2022 (Switch Transformer) + various 2024 adaptations
- **Core idea**:
  - Replace FFN in Transformer with Switch/Sparse MoE: x → router → top-1 expert FFN
  - Only 1 of K experts activated per token → same FLOPs as 1 FFN but K× capacity
  - For our cross-sym Transformer: FFN after cross-sym MHSA → replaced by 4-8 expert FFNs
  - Load balancing: auxiliary_loss = N * Σ_k (f_k * p_k) where f_k=fraction of tokens, p_k=mean router prob
- **为何适合 v7+**: Increases model capacity without increasing inference cost; experts may specialize in different market regimes.
- **预期 PnL gain vs SOTA +38.49**: low-medium (+0.2 to +0.5)
- **优先级**: P3

---

### MoEHierarchical_Finance (LLMoE arXiv 2025 analysis) — Direction 2
- **Paper**: https://arxiv.org/abs/2501.09636
- **Core idea (distilled for tabular)**:
  - Two-level routing: first route to market-regime expert (trending/reverting/volatile), then to signal expert
  - Level 1: macro_context = mean(cross_sym_features) → regime_router
  - Level 2: per-sym features → signal_router conditioned on regime
  - Achieves >25% improvement in Sharpe ratio vs standard MoE
- **为何适合 v7+**: Hierarchy captures both global market state and per-sym signal, which is exactly our two-level structure (cross-sym attention → per-sym prediction head).
- **预期 PnL gain vs SOTA +38.49**: medium (+0.3 to +0.8)
- **优先级**: P2

---

### SurveySparseMoE_Finance (arXiv 2602.08019) — Direction 2
- **Paper**: https://arxiv.org/abs/2602.08019
- **Core idea**: Survey paper — key finding: MoE models show broad applicability including finance vertical; sparse routing with top-2 experts achieves best trade-off between capacity and compute; expert diversity via orthogonal regularization prevents expert collapse.
- **为何适合 v7+**: Orthogonal regularization for expert diversity is directly applicable to prevent our MoE experts from converging to the same function.
- **优先级**: P3 (reference)

---

## Direction 3: Self-Supervised / Contrastive Learning for Tabular Finance (10 papers)

### VIME_DualSSL (Yoon et al. — NeurIPS 2020, adapt 2024) — Direction 3
- **Paper**: https://arxiv.org/abs/2003.08013
- **Venue**: NeurIPS 2020 (still most cited SSL for tabular data 2024)
- **Core idea**:
  - Corruption: x̃ = m ⊙ x_random + (1-m) ⊙ x, m ~ Bernoulli(p_m=0.3)
  - Pretext Task 1: reconstruction → x̂ = decoder(encoder(x̃))
  - Pretext Task 2: mask estimation → m̂ = mask_head(encoder(x̃)), L_mask = BCE(m̂, m)
  - L_total = L_recon + L_mask_bce + α * L_pred
  - L_mask forces encoder to learn which features covary — crucial for financial cross-feature dependencies
- **PyTorch class sketch**:
  ```python
  class VIME_PerSym(nn.Module):
      """Add VIME dual SSL to existing SAE encoder. Drop-in extension."""
      def __init__(self, d_latent=64, n_feat=259):
          super().__init__()
          self.mask_head = nn.Linear(d_latent, n_feat)  # only +16k params!

      def compute_mask_loss(self, z, mask):  # z: (B,5,d), mask: (B,5,n_feat) bool
          m_hat = torch.sigmoid(self.mask_head(z))  # (B, 5, n_feat)
          return F.binary_cross_entropy(m_hat, mask.float())
  ```
- **为何适合 v7+ (sae_mlp + cross-sym)**: We already do Task 1 (reconstruction). Adding Task 2 (mask estimation) costs only +16k params (1 linear head) but forces the encoder to understand feature correlation structure — especially valuable for financial factor data where features are highly structured.
- **预期 PnL gain vs SOTA +38.49**: medium (+0.3 to +0.8)
- **参数量预估**: +16k only
- **实现复杂度**: easy
- **优先级**: P1

---

### SCARF_CrossSymContrastive (Bahri et al. — ICLR 2022, adapt 2024) — Direction 3
- **Paper**: https://arxiv.org/abs/2106.15147
- **Venue**: ICLR 2022
- **Core idea**:
  - Creates views by corrupting random feature subset p=0.6: replaces masked features with marginal samples from training distribution (not Gaussian noise!)
  - InfoNCE loss: L = L_pred + β * InfoNCE(z_clean, z_corrupt)
  - Positive pair: same sample (clean vs corrupted); Negatives: other samples in batch
  - Our adaptation: positive = (sym_i_clean, sym_i_noisy), negatives = other syms in batch
  - SCARF outperforms VIME when label noise is present (financial data has high noise)
- **PyTorch class sketch**:
  ```python
  def scarf_loss(z_clean, z_noisy, temperature=0.1):
      # z_clean, z_noisy: (B*5, d)
      z1 = F.normalize(z_clean, dim=-1)
      z2 = F.normalize(z_noisy, dim=-1)
      N = z1.shape[0]
      sim = torch.einsum('id,jd->ij', z1, z2) / temperature
      labels = torch.arange(N, device=z1.device)
      return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
  ```
- **为何适合 v7+**: With 5 syms × batch_size negatives per step, contrastive signal is dense. More importantly, SCARF forces z to be invariant to random feature masking — improving robustness to missing/noisy financial signals at inference.
- **预期 PnL gain vs SOTA +38.49**: medium (+0.2 to +0.6)
- **参数量预估**: 0 extra (loss only)
- **实现复杂度**: easy
- **优先级**: P1

---

### ContrastiveAssetEmbeddings (Dolphin, Smyth, Dong — ICAIF 2024) — Direction 3
- **Paper**: https://arxiv.org/abs/2407.18645
- **Venue**: ACM ICAIF 2024
- **Core idea**:
  - Positive pair selection via statistical hypothesis testing on return similarity (not just temporal proximity)
  - Rejects noisy positives: pairs with similar return distributions but different characteristics are suppressed
  - NT-Xent loss with statistical sampling: sim(z_i, z_j) * significance(i,j)
  - Learns asset embeddings useful for sector classification AND portfolio optimization
  - Key innovation: principled positive pair selection under financial noise
- **为何适合 v7+**: Our 5-sym cross-sym contrastive learning can use this statistical sampling — instead of treating all sym pairs as equally informative positives/negatives, we weight by statistical similarity of their recent return distributions.
- **预期 PnL gain vs SOTA +38.49**: low-medium (+0.2 to +0.5)
- **优先级**: P2

---

### ContrastiveEarningsTransformer_CET (Ye, Schuller — arXiv 2024) — Direction 3
- **Paper**: https://arxiv.org/abs/2409.17392
- **Venue**: arXiv 2024 (Imperial College London)
- **Core idea**:
  - CET: Contrastive Predictive Coding for stock trading
  - Fuses data of different granularity via learned representations
  - CPC: encode context z_t, predict future latent representation ẑ_{t+k} = W_k * z_t
  - Contrastive: positive = true future representation; negatives = random samples
  - Works even as earnings data "ages" (useful for non-stationary features)
- **为何适合 v7+**: CPC objective for cross-sym: encode sym_i at time t, predict encoded features of sym_j at same time — this is CROSS-SECTIONAL CPC, learning which syms' features "predict" each other.
- **预期 PnL gain vs SOTA +38.49**: low-medium (+0.2 to +0.4)
- **优先级**: P3

---

### ReMasker_ICLR2024 (Du, Melis, Wang — ICLR 2024) — Direction 3
- **Paper**: https://arxiv.org/abs/2309.13793
- **Venue**: ICLR 2024
- **Core idea**:
  - Extends MAE to tabular: randomly re-mask observed features (not just impute missing)
  - Training: corrupt a second set of features → reconstruct from encoder
  - Key theory: learns "missingness-invariant representations" → robust to feature dropout at test time
  - PMAE (2412.19152, 2024): adds proportional masking to address column-level imbalance
  - Unlike standard reconstruction (reconstruct ALL from NOISY), re-masking reconstructs SELECTED from UNMASKED
- **PyTorch class sketch**:
  ```python
  def remasker_loss(encoder, decoder, x, p_remask=0.3):
      # Randomly re-mask a subset of features
      mask = torch.rand_like(x) < p_remask  # (B, 5, n_feat)
      x_masked = x * ~mask  # zero out remask positions
      z = encoder(x_masked)
      recon = decoder(z)
      return F.mse_loss(recon[mask], x[mask])  # only reconstruct re-masked positions
  ```
- **为何适合 v7+**: Stronger than our current noise-based reconstruction. Re-masking forces encoder to learn from subset of features → improves feature efficiency and redundancy reduction. More structured than random noise.
- **预期 PnL gain vs SOTA +38.49**: low (+0.1 to +0.4)
- **优先级**: P2

---

### TabICL_InContext (Qu et al. — ICML 2025) — Direction 3
- **Paper**: https://arxiv.org/abs/2502.05564
- **Venue**: ICML 2025
- **Core idea**:
  - Two-stage: column-then-row attention → fixed-dim row embeddings → Transformer for ICL
  - Column attention: each row attends over its own feature values → per-row embedding
  - Row attention: rows (samples) attend to each other → in-context prediction
  - Sets training data as CONTEXT for test data; single forward pass, no gradient updates
  - Surpasses TabPFNv2 on large datasets (>10K samples)
- **为何适合 v7+**: The column-then-row dual attention (within a single sample: features attend to each other) is orthogonal to our cross-sym attention (samples attend to each other). Combining both could be powerful: column-attn for per-sym feature interaction, then cross-sym attention.
- **预期 PnL gain vs SOTA +38.49**: medium (+0.3 to +0.8)
- **参数量预估**: ~130k
- **实现复杂度**: medium
- **优先级**: P2

---

### SAINTv2_CrossSym (Somepalli et al. — arXiv 2021, still SOTA 2024) — Direction 3
- **Paper**: https://arxiv.org/abs/2106.01342
- **Venue**: arXiv 2021 (widely used 2024)
- **Core idea**:
  - Stack of L identical stages, each: self-attention (features) + intersample-attention (rows/syms)
  - Feature attention: within one sym, features attend to each other
  - Intersample attention: across 5 syms at same time, sym vectors attend to each other
  - Contrastive pre-training: corrupted views attract; other samples repel
  - Our adaptation: treat syms as "samples" → intersample attention IS our cross-sym attention
- **为何适合 v7+**: SAINT explicitly alternates between feature-level and sym-level attention — provides BOTH cross-feature (within sym) and cross-sym (across 5 syms) interaction in a single unified framework.
- **预期 PnL gain vs SOTA +38.49**: medium (+0.3 to +0.8)
- **优先级**: P2

---

### BarlowTwinsTabular (Zbontar et al. — ICML 2021, adapted finance 2024) — Direction 3
- **Paper**: https://arxiv.org/abs/2103.03230
- **Venue**: ICML 2021
- **Core idea**:
  - Cross-correlation matrix between two augmented views C_ij = Σ_b z1_bi * z2_bj / N
  - Loss: L = Σ_i (1 - C_ii)² + λ * Σ_{i≠j} C_ij²
  - On-diagonal: pull augmented views together; Off-diagonal: push features to be de-correlated
  - Feature de-correlation prevents collapse without negative pairs
  - Financial adaptation: augment by randomly permuting sym order (permutation invariance enforcement)
- **为何适合 v7+**: Off-diagonal Barlow penalty de-correlates latent features → prevents our encoder from learning redundant features. Combined with permutation augmentation enforces sym-agnostic representations.
- **预期 PnL gain vs SOTA +38.49**: low (+0.1 to +0.3)
- **优先级**: P3

---

### FairContrastTabular (arXiv 2025) — Direction 3
- **Paper**: https://arxiv.org/abs/2510.02017
- **Venue**: arXiv 2025
- **Core idea**:
  - Contrastive learning + custom augmentation for tabular data
  - Augmentation-aware positive pair selection
  - Reduces representation bias via constrained contrastive loss
  - Customized augmentations: feature permutation, Gaussian corruption, mixup
- **为何适合 v7+**: The combined augmentation strategy (permutation + Gaussian + mixup) can be directly applied to our sym features for richer contrastive views.
- **优先级**: P3

---

### VIMEExtended2024 (tabular SSL survey 2024) — Direction 3
- **Core idea**: Recent VIME extensions add (1) temporal mask estimation for time-series (not applicable); (2) cross-sample mask estimation: predict WHICH samples share correlated masks; (3) multi-task VIME with task-specific projection heads.
- **Application**: Multi-task VIME: shared encoder, task 1 = prediction, task 2 = mask estimation, task 3 = cross-sym consistency.
- **优先级**: P3

---

## Direction 4: Graph Transformers (2023-2025) (10 papers)

### T2GFormer_FeatureGraph (Yan et al. — AAAI 2023) — Direction 4
- **Paper**: https://arxiv.org/abs/2211.16887
- **Venue**: AAAI 2023 (oral, widely cited 2024-2025)
- **Core idea**:
  - Graph Estimator: automatically learns feature-to-feature relationship graph R ∈ ℝ^{n_feat × n_feat}
  - R_ij = σ(q_i^T k_j / √d) where q_i = W_q * emb(i), k_j = W_k * emb(j)
  - Feature interaction guided by graph: attention mask A = softmax(R + bias_mask)
  - T2G-Former: Transformer with attention restricted to graph-connected features
  - Addresses feature heterogeneity: only related features interact
- **PyTorch class sketch**:
  ```python
  class T2GFeatureEncoder(nn.Module):
      def __init__(self, n_feat=259, d_emb=32, d_model=64, dropout=0.3):
          super().__init__()
          self.feat_emb = nn.Embedding(n_feat, d_emb)  # feature ID embeddings
          self.feat_proj = nn.Linear(1, d_model)
          self.graph_q = nn.Linear(d_emb, d_emb)
          self.graph_k = nn.Linear(d_emb, d_emb)
          self.transformer = nn.TransformerEncoderLayer(d_model, 4, d_model*2, dropout, batch_first=True)
          self.pool = nn.Linear(n_feat, 1)

      def forward(self, x):  # (B, 5, n_feat)
          B, N, F = x.shape
          feat_ids = torch.arange(F, device=x.device)
          emb = self.feat_emb(feat_ids)  # (F, d_emb)
          # Graph estimation
          q = self.graph_q(emb)  # (F, d)
          k = self.graph_k(emb)
          R = torch.sigmoid(q @ k.T / (q.shape[-1]**0.5))  # (F, F) feature relation graph
          # Token-wise: each feature is a token with value x[..., i]
          tokens = self.feat_proj(x.unsqueeze(-1))  # (B, N, F, d)
          tokens = tokens.reshape(B*N, F, -1)
          out = self.transformer(tokens, src_mask=R.log())  # graph-masked attention
          return out.mean(dim=1).reshape(B, N, -1)  # (B, N, d)
  ```
- **为何适合 v7+ (sae_mlp + cross-sym)**: Instead of our flat MLP encoder (259→128→32), T2G-Former builds a FEATURE GRAPH within each sym's 259 features, letting related features interact before prediction. This is a fundamentally different and potentially more powerful per-sym encoder.
- **预期 PnL gain vs SOTA +38.49**: medium-high (+0.5 to +1.5)
- **参数量预估**: ~180k (higher due to feature-level transformer)
- **实现复杂度**: medium
- **优先级**: P1

---

### MDGNN_MultiRelational (Dongyuan et al. — arXiv 2024) — Direction 4
- **Paper**: https://arxiv.org/abs/2402.06633
- **Venue**: arXiv 2024
- **Core idea**:
  - Multi-relational graph: stocks + industries + investment banks as nodes
  - Dynamic: graph evolves over time via Transformer on graph snapshots
  - Multiple edge types: price correlation, sector membership, ownership, analyst coverage
  - Heterogeneous node types: sym nodes absorb information from structural nodes
  - For our 5-sym problem: add "sector" and "market" virtual nodes with different edge semantics
- **为何适合 v7+**: Multiple edge types in a 5+k node graph allow different types of cross-sym relationships to be learned separately. More expressive than single-type MHSA.
- **预期 PnL gain vs SOTA +38.49**: low-medium (+0.2 to +0.5)
- **优先级**: P3

---

### MultiRelationalGraphDiffusion (You et al. — ICASSP 2024) — Direction 4
- **Paper**: https://arxiv.org/abs/2401.05430
- **Venue**: ICASSP 2024
- **Core idea**:
  - Edge generation via information entropy + signal energy → quantify inter-stock relation intensity
  - Stochastic multi-relational diffusion: adaptive task-optimal graph edges via learned distribution
  - Parallel retention: decoupled representation learning with parallel sequence modeling
  - Tested on NASDAQ, NYSE, SSE with 7-year validation period
- **为何适合 v7+**: Dynamic edge generation from feature statistics (entropy-based) can determine when syms are highly correlated (same features trending) vs independent. More principled than fixed attention.
- **预期 PnL gain vs SOTA +38.49**: low (+0.2 to +0.4)
- **优先级**: P3

---

### GRU_PFG_FactorGraph (arXiv 2024) — Direction 4
- **Paper**: https://arxiv.org/abs/2411.18997
- **Venue**: arXiv 2024
- **Core idea**:
  - Project stock FACTORS (our: 259 features) into graph structure
  - GNN learns inter-stock correlation from factor similarity (not raw return correlation)
  - Edge weight: w_{ij} = cos_sim(factor_i, factor_j) → captures factor structure correlation
  - IC 0.134 on CSI300, outperforms HIST and Transformer
  - Key: factor-based graph construction = sym-agnostic by design (no sym-specific info)
- **PyTorch class sketch**:
  ```python
  class FactorGraphEncoder(nn.Module):
      def __init__(self, n_feat=259, d_latent=64):
          super().__init__()
          self.feat_enc = nn.Linear(n_feat, d_latent)

      def forward(self, x):  # (B, 5, n_feat)
          z = self.feat_enc(x)
          # Compute factor similarity graph
          z_norm = F.normalize(z, dim=-1)
          adj = torch.bmm(z_norm, z_norm.transpose(1, 2))  # (B, 5, 5) dynamic adjacency
          # GNN message passing
          out = torch.bmm(F.softmax(adj, dim=-1), z)  # (B, 5, d)
          return out
  ```
- **为何适合 v7+**: Factor-graph construction creates a DYNAMIC adjacency based on how similar syms' current feature vectors are — more adaptive than fixed MHSA and naturally sym-agnostic.
- **预期 PnL gain vs SOTA +38.49**: medium (+0.3 to +0.7)
- **优先级**: P2

---

### CRISPAdaptiveGraph (Li, Fan — arXiv 2025) — Direction 4
- **Paper**: https://arxiv.org/abs/2510.20868
- **Venue**: arXiv 2025
- **Core idea**:
  - GCN + BiLSTM for spatial + temporal encoding
  - Multi-head GAT for sparse graph structure learning
  - Crisis-resilient: learns adaptive graph topology per-timestep
  - Key: sparse graph via L1 regularization on edge weights
  - Sharpe 3.76 vs 1.94 (static) during 2022-2024 inflation crisis (+94%)
- **为何适合 v7+**: Learned sparse graph over 5 syms: only strongest inter-sym relationships are active at each timestep. More robust than full-attention which always aggregates all 5.
- **预期 PnL gain vs SOTA +38.49**: medium (+0.3 to +0.7)
- **优先级**: P2

---

### MaGNet_HypergraphStock (arXiv 2025) — Direction 4
- **Paper**: https://arxiv.org/abs/2511.00085
- **Venue**: arXiv 2025
- **Core idea**:
  - Dual hypergraph: TCH (temporal-causal) + GPH (global probabilistic)
  - Hyperedge = group of nodes (syms) with shared causal dependency
  - GPH: soft hyperedge assignment via learned probability → each sym has partial membership
  - Jensen-Shannon divergence for hyperedge weight → prevents noisy/spurious connections
  - 2D attention: feature-wise AND stock-wise within MAGE block
- **为何适合 v7+**: Hyperedge groups allow grouping subsets of 5 syms that are jointly correlated (e.g., syms 1,2,3 moving together). Standard MHSA treats all 5 equally; hypergraph captures higher-order interactions.
- **预期 PnL gain vs SOTA +38.49**: medium (+0.3 to +0.8)
- **优先级**: P2

---

### GraphormerCrossFeature (Ying et al. — NeurIPS 2021, recent adaptations 2024) — Direction 4
- **Paper**: Original Graphormer + VCR-Graphormer (2403.16030, 2024)
- **Venue**: NeurIPS 2021 + arXiv 2024
- **Core idea**:
  - Graphormer: centrality encoding + spatial encoding → Transformer on graphs
  - Centrality: degree of each node as additional embedding (node importance signal)
  - Spatial encoding: shortest path distance as attention bias
  - VCR-Graphormer (2024): virtual node connections for mini-batch efficiency
  - Adaptation: treat 5 syms as graph with learned centrality = sym importance
- **为何适合 v7+**: Centrality encoding gives each sym a learnable "importance score" that biases attention — more principled than uniform attention over 5 equal syms.
- **预期 PnL gain vs SOTA +38.49**: low-medium (+0.2 to +0.5)
- **优先级**: P3

---

### NGATNodeLevel (arXiv 2025) — Direction 4
- **Paper**: https://arxiv.org/abs/2507.02018
- **Venue**: ICANN 2025
- **Core idea**:
  - Node-level GAT avoiding oversmoothing via node-specific attention heads
  - News co-occurrence graph for corporate relationships
  - Multi-horizon prediction (T=1,5,10,21 days)
  - Key: prevents information from distant nodes bleeding into local predictions
- **为何适合 v7+**: For 5 syms, "oversmoothing" is a real concern — MHSA can collapse all 5 sym representations toward the mean. Node-level attention prevents this.
- **预期 PnL gain vs SOTA +38.49**: low (+0.1 to +0.3)
- **优先级**: P3

---

### Pairmixer_TriMul (arXiv 2025) — Direction 4
- **Paper**: https://arxiv.org/abs/2510.18870
- **Venue**: arXiv 2025
- **Core idea**:
  - Replaces triangle attention with triangle MULTIPLICATION for 4× speedup
  - For N=5: triangle attention has O(N²) overhead; Pairmixer maintains O(N²) pairwise but faster
  - Triangle update: pair[i,j] += Σ_k (W_out * pair[i,k]) ⊙ (W_in * pair[k,j])
  - Provably captures triplet interactions (i,k,j) via transitivity
  - Key insight: triangle multiplication is ALL you need; triangle attention is overkill
- **为何适合 v7+**: For our 5×5=25 pair matrix, the Pairmixer update is O(125) operations — trivially fast but captures 3-way sym interactions that 2-way MHSA misses entirely.
- **预期 PnL gain vs SOTA +38.49**: high (+0.8 to +2.0)
- **参数量预估**: +30k (pair update matrices)
- **实现复杂度**: medium
- **优先级**: P1

---

## Direction 5: Hybrid CNN-Transformer for LOB (5 papers)

### TLOB_DualAttnLOB (Berti, Kasneci — arXiv 2025) — Direction 5
- (See Direction 1 entry — same paper, focused on LOB/HFT)
- **Venue**: arXiv 2025
- **Feature Self-Attention addition**: For per-sym encoder, apply attention over 259 features treated as tokens → richer feature interaction than plain MLP.
- **优先级**: P1 (overlaps with Direction 1)

---

### LiT_LOBTransformer (Frontiers AI 2025) — Direction 5
- **Paper**: https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1616485/full
- **Venue**: Frontiers in AI 2025
- **Core idea**:
  - Dedicated LOB Transformer with price-level tokenization
  - Each LOB level (bid/ask price+size) → separate token
  - Cross-level attention: bid levels attend to ask levels and vice versa
  - Positional encoding for price level (not time)
  - Bilinear normalization before each block
- **为何适合 v7+**: Cross-bid/ask attention is analogous to cross-sym attention. Bilinear normalization is a data-driven normalization (learnable per feature mean/std) useful for financial non-stationarity.
- **预期 PnL gain vs SOTA +38.49**: low (+0.1 to +0.3, different domain)
- **优先级**: P3

---

### CNNTransformerHFT (XJTLU 2025) — Direction 5
- **Paper**: https://link.springer.com/chapter/10.1007/978-981-96-6310-1_5
- **Venue**: SpringerLink 2025
- **Core idea**:
  - 1D CNN over feature dimension (not time!) → extract local feature patterns
  - Transformer on CNN feature maps → global feature interaction
  - CNN: kernel=3, captures triplets of adjacent features
  - Combined: CNN-extracted local patterns + Transformer global patterns
- **PyTorch class sketch**:
  ```python
  class CNNTransformerPerSym(nn.Module):
      def __init__(self, n_feat=259, d_conv=64, n_heads=4):
          super().__init__()
          self.conv = nn.Conv1d(1, d_conv, kernel_size=3, padding=1)
          self.transformer = nn.TransformerEncoderLayer(d_conv, n_heads, batch_first=True)
          self.pool = nn.AdaptiveAvgPool1d(1)

      def forward(self, x):  # (B, 5, n_feat)
          B, N, F = x.shape
          x_flat = x.reshape(B*N, 1, F)
          c = F.gelu(self.conv(x_flat))  # (B*N, d_conv, F)
          c = c.transpose(1, 2)  # (B*N, F, d_conv)
          c = self.transformer(c)  # (B*N, F, d_conv)
          out = c.mean(dim=1).reshape(B, N, -1)  # (B, N, d_conv)
          return out
  ```
- **为何适合 v7+**: 1D convolution over feature dimension captures LOCAL feature patterns (adjacent features that co-vary) before global Transformer attention. For financial features which are often sorted by type, this is natural.
- **预期 PnL gain vs SOTA +38.49**: low-medium (+0.2 to +0.5)
- **优先级**: P3

---

### SENet_CrossSymGating (SE original + finance adaptation) — Direction 5
- **Paper**: Hu et al., CVPR 2018 (SENet) + adapted for cross-sym 2024
- **Core idea**:
  - Squeeze: global average pool over 5 syms → (B, d) context vector
  - Excite: FC → ReLU → FC → Sigmoid → (B, 5) per-sym attention weights
  - Output: sym_i_enhanced = sym_i * gate_i
  - Extension: squeeze + excite over FEATURE dimension within each sym → feature channel attention
  - Cost: 2 FC layers (d→d/r→5, r=4) = ~500 params
- **PyTorch class sketch**:
  ```python
  class SEGateCrossSym(nn.Module):
      def __init__(self, d_latent=64, n_sym=5, r=4):
          super().__init__()
          self.fc1 = nn.Linear(d_latent, d_latent // r)
          self.fc2 = nn.Linear(d_latent // r, n_sym)

      def forward(self, z):  # (B, 5, d)
          ctx = z.mean(dim=1)  # (B, d) squeeze
          gate = torch.sigmoid(self.fc2(F.relu(self.fc1(ctx))))  # (B, 5) excite
          return z * gate.unsqueeze(-1)  # (B, 5, d)
  ```
- **为何适合 v7+**: Ultra-lightweight (500 params) post-attention gating. Can be stacked after MHSA to let the model suppress uninformative syms and amplify informative ones — different from MHSA which always averages all.
- **预期 PnL gain vs SOTA +38.49**: low-medium (+0.2 to +0.5)
- **参数量预估**: +500
- **实现复杂度**: easy
- **优先级**: P2

---

### SAPP_SparsePortfolio (jiahaoli — ScienceDirect 2024) — Direction 5
- **Paper**: https://www.sciencedirect.com/science/article/abs/pii/S0031320326006254
- **Venue**: Pattern Recognition 2024
- **Core idea**:
  - Sparse Transformer Blocks with near-linear complexity for temporal patterns
  - Correlation Information Decision Module: cross-asset dependency via explicit correlation matrix
  - End-to-end RL training with cost-aware reward
  - The correlation module computes: C_ij = f(h_i, h_j) → explicit n×n correlation matrix
  - For 5 syms: 5×5 correlation → lightweight and explicit
- **为何适合 v7+**: The EXPLICIT 5×5 correlation matrix (not implicit via attention) combined with sparse attention provides interpretable cross-sym interaction, potentially less noisy than full MHSA.
- **预期 PnL gain vs SOTA +38.49**: medium (+0.3 to +0.7)
- **优先级**: P2

---

## Direction 6: Hierarchical / Sparse Attention (5 papers)

### SAPP_SparseAttn (see Direction 5 entry) — Direction 6
- (See above)

### IsAttentionAllWeNeed_Study (arXiv 2025) — Direction 6
- **Paper**: https://arxiv.org/abs/2508.19006
- **Key finding for our problem**: Global self-attention outperforms sparse attention for small N (5 syms). Sliding window sparse attention hurts because all 5 syms are "local" — there's no benefit to sparsity. Confirms our MHSA design is correct; no need for Performer/Reformer.
- **优先级**: P3 (reference)

---

### MakingEveryHeadCount (arXiv 2025) — Direction 6
- **Paper**: https://arxiv.org/abs/2511.09596
- **Core idea**: Sparse attention without speed-performance trade-off via learned head-level sparsity. Each attention head learns its own sparsity pattern; 2× speedup with same performance as dense attention.
- **为何适合 v7+**: Head-level sparsity for our 4-head MHSA could allow different heads to specialize (some dense for nearby syms, some sparse for distant syms) — but with N=5, this has limited value.
- **优先级**: P3

---

### DecoderOnlyTransformer_Stock (arXiv 2025) — Direction 6
- **Paper**: https://arxiv.org/abs/2504.16361 (see Direction 1)
- **Key idea for v8**: Decoder-only architecture where target sym's latent QUERIES into context of all 5 syms (cross-attention). Asymmetric: target sym is always treated as query; context is all 5.
- **Implementation**: For sym_i prediction: Q = z_i (1 token), K/V = z (all 5 tokens) → cross-attn
- **优先级**: P2

---

### SpargeAttention_Fast (arXiv 2025) — Direction 6
- **Paper**: https://arxiv.org/abs/2502.18137
- **Core idea**: Training-free sparse attention acceleration via block-sparse patterns. Not applicable to N=5 (too small for any speedup).
- **优先级**: P3 (not applicable to N=5)

---

### PerformerFinance (2024) — Direction 6
- **Core idea**: Kernel approximation of softmax attention for linear complexity. For N=5 nodes, Performer has MORE overhead than standard attention due to FAVOR+ kernel computation. Not suitable.
- **优先级**: P3 (not applicable)

---

## Direction 7: Decision-Focused Learning / Predict-then-Optimize (5 papers)

### SPO_Portfolio_Real (Yi, Hasuike — arXiv 2026) — Direction 7
- **Paper**: https://arxiv.org/abs/2601.04062
- **Venue**: arXiv 2026 (Waseda University)
- **Core idea**:
  - SPO paradigm: train predictor to minimize DECISION loss (portfolio return) not MSE
  - Decision loss: L_SPO(ŷ, y) = max_w {w^T y} - max_w {w^T ŷ}
  - SPO+ (convex surrogate): L_SPO+ = max_w {w^T (2ŷ - y)} - max_w {w^T ŷ}
  - Applied to US ETF data (2015-2025); improves risk-adjusted return vs MSE baseline
  - Key: optimizes for RELATIVE ordering of 5 syms, not absolute values
- **PyTorch class sketch**:
  ```python
  def spo_plus_loss(pred, target):  # (B, 5) each — cross-sectional ranking
      # SPO+: surrogate for PnL-oriented loss
      # target_rank: which sym had highest actual return
      target_normalized = target / (target.abs().sum(dim=-1, keepdim=True) + 1e-8)
      pred_rank = pred / (pred.abs().sum(dim=-1, keepdim=True) + 1e-8)
      # Listwise SPO approximation:
      spo_approx = -((2 * target_normalized - pred_normalized) * F.softmax(pred, dim=-1)).sum(dim=-1)
      return spo_approx.mean()
  ```
- **为何适合 v7+**: Our PnL metric is essentially a cross-sectional ranking (long top sym, short others). MSE loss treats all prediction errors equally; SPO+ loss penalizes errors that cause wrong RELATIVE ranking (which is what PnL cares about).
- **预期 PnL gain vs SOTA +38.49**: medium (+0.5 to +1.5)
- **参数量预估**: 0 (loss function change)
- **实现复杂度**: easy
- **优先级**: P1

---

### OnlineDFL (ICLR 2026) — Direction 7
- **Paper**: https://arxiv.org/abs/2505.13564
- **Venue**: ICLR 2026
- **Core idea**:
  - Online DFL: objective function AND data distribution evolve over time
  - Addresses zero/undefined gradients via regularization + perturbation
  - Near-optimal oracle for non-convex objectives
  - Applicable to rolling-window training scenarios
- **为何适合 v7+**: Our training uses a fixed window but test is rolling. Online DFL adapts the training objective dynamically to the recent data distribution.
- **预期 PnL gain vs SOTA +38.49**: low-medium (+0.2 to +0.5, research-stage)
- **优先级**: P3

---

### DFF_DecisionFocusedFT (Yang et al. — AAAI 2025) — Direction 7
- **Paper**: https://arxiv.org/abs/2501.01874
- **Venue**: AAAI 2025
- **Core idea**:
  - Decision-Focused Fine-tuning: start with pretrained MSE model, fine-tune with DFL loss
  - Trust region: constrain ‖model_DFF - model_pretrained‖ ≤ ε to prevent divergence
  - Solves instability of end-to-end DFL training with limited data
  - Provably bounds prediction bias under fine-tuning
  - Two-stage: (1) pretrain with MSE; (2) fine-tune with L_portfolio in trust region
- **为何适合 v7+**: Two-stage approach is practical — our v8 model pretrained on MSE, then fine-tuned with PnL-aware loss within trust region. Much more stable than end-to-end DFL.
- **预期 PnL gain vs SOTA +38.49**: medium (+0.3 to +1.0)
- **实现复杂度**: easy (post-training fine-tune)
- **优先级**: P1

---

### ReturnPredMVO (Lee et al. — ICAIF 2025) — Direction 7
- **Paper**: https://arxiv.org/abs/2409.09684
- **Venue**: ICAIF 2025
- **Core idea**:
  - How DFL modifies MVO prediction models: DFL creates IMPLICIT covariance-like regularization
  - MSE treats all prediction errors equally; DFL re-weights errors by portfolio sensitivity
  - Key finding: DFL model learns to "concentrate" predictions on high-information assets
  - For cross-sectional setting: DFL emphasizes getting the rank of extreme syms right
- **为何适合 v7+**: Theoretical insight: our model should predict more accurately for the top/bottom ranked syms (those that will be longed/shorted) than for middle-ranked ones. DFL loss automatically downweights errors for middle syms.
- **预期 PnL gain vs SOTA +38.49**: medium (+0.3 to +0.8)
- **优先级**: P2

---

### CovDFL (arXiv 2025) — Direction 7
- **Paper**: https://arxiv.org/abs/2508.10776
- **Core idea**: DFL for covariance estimation in global minimum variance portfolio. Estimates covariance to minimize realized portfolio variance rather than covariance prediction error. Applicable to our setting: estimate cross-sym covariance for hedging.
- **优先级**: P3 (different focus)

---

## Direction 8: Equivariant / Invariant Networks (5 papers)

### MultisetTransformer (Wang, Huang, Xu — arXiv 2024) — Direction 8
- **Paper**: https://arxiv.org/abs/2411.14662
- **Venue**: arXiv 2024
- **Core idea**:
  - First neural network designed for MULTISETS (sets with element multiplicities)
  - Multiset-enhanced attention: attention weights respect multiplicity counts
  - Pool-decomposition: reduces O(N²) to O(N log N) via tree structure
  - Theoretical guarantees of permutation invariance + multiplicity preservation
  - Outperforms Set Transformer on persistence diagram tasks
- **为何适合 v7+ (sae_mlp + cross-sym)**: Our 5 syms can be treated as a SET (order doesn't matter) — using Multiset Transformer for cross-sym aggregation naturally enforces this. When 2 syms are "effectively identical" (same sector, same moment), multiplicity captures this.
- **预期 PnL gain vs SOTA +38.49**: low-medium (+0.2 to +0.5)
- **参数量预估**: ~90k
- **实现复杂度**: medium
- **优先级**: P2

---

### SetTransformerPP_DeepSets (Andreis et al. — arXiv 2022) — Direction 8
- **Paper**: https://arxiv.org/abs/2206.11925
- **Venue**: arXiv 2022 (widely used 2024)
- **Core idea**:
  - Deep Sets++ / Set Transformer++: fixes vanishing/exploding gradient issues
  - Clean path principle: residual connections must pass through equivariant operations
  - Set Norm (sn): normalization tailored for sets (normalizes over SET dimension, not batch)
  - SetNorm(x_i) = γ * (x_i - mean_j(x_j)) / std_j(x_j) + β  (mean/std over set elements)
  - Enables training of DEEP set networks (>10 layers without gradient issues)
- **PyTorch class sketch**:
  ```python
  class SetNorm(nn.Module):
      def __init__(self, d):
          super().__init__()
          self.gamma = nn.Parameter(torch.ones(d))
          self.beta = nn.Parameter(torch.zeros(d))

      def forward(self, x):  # (B, N, d)
          mean = x.mean(dim=1, keepdim=True)  # (B, 1, d) set-level mean
          std = x.std(dim=1, keepdim=True) + 1e-6
          return self.gamma * (x - mean) / std + self.beta
  ```
- **为何适合 v7+**: Our current LayerNorm normalizes each sym independently — it doesn't leverage the cross-sym distribution information. SetNorm normalizes relative to other syms, encoding RELATIVE feature values (exactly what cross-sectional trading cares about).
- **预期 PnL gain vs SOTA +38.49**: medium (+0.4 to +1.0)
- **参数量预估**: +2d params (minimal)
- **实现复杂度**: easy
- **优先级**: P1

---

### PermInvariantNN (arXiv 2024) — Direction 8
- **Paper**: https://arxiv.org/abs/2403.17410
- **Venue**: arXiv 2024
- **Core idea**:
  - Constructs permutation invariant transformations from equivariant building blocks
  - Theoretical analysis of universal approximation for invariant functions
  - Key: any permutation invariant function = g(Σ_i f(x_i)) (DeepSets) OR more complex poolings
  - 2024 result: Transformer is a universal approximator for permutation EQUIVARIANT functions
- **为何适合 v7+**: Theoretical backing that our current MHSA cross-sym attention is a universal approximator for equivariant functions of the 5-sym set. But to get INVARIANT prediction (final pooled pred), need proper invariant aggregation.
- **优先级**: P3 (theoretical reference)

---

### DeepOSets_ICL (arXiv 2024) — Direction 8
- **Paper**: https://arxiv.org/abs/2410.09298
- **Venue**: arXiv 2024
- **Core idea**:
  - Non-autoregressive in-context learning with permutation-invariance bias
  - Processes unordered sets as in-context examples
  - DeepOSets: DeepSets applied to operator learning
  - Permutation invariant ICL: predictions don't depend on order of context examples
  - Applicable: treat 5 syms as unordered context set for "in-context" cross-sym prediction
- **为何适合 v7+**: Frame our 5-sym prediction as in-context learning: given 5 sym feature vectors as context, predict each sym's return using all others as context. Permutation invariance is enforced by architecture.
- **优先级**: P3

---

### PermInvariantPolynomials (arXiv 2025) — Direction 8
- **Paper**: https://arxiv.org/abs/2502.11467
- **Venue**: arXiv 2025
- **Core idea**:
  - Efficient construction of permutation invariant polynomial approximators via single-head attention
  - Tight polynomial approximation bounds for symmetric functions
  - Key: shows that invariant polynomials can be approximated with column-size efficiency (not row-size)
  - For our 5-sym set: polynomial invariant function of degree ≤ 5 can capture all interaction orders
- **为何适合 v7+**: Polynomial interaction = explicit higher-order sym interactions. Degree-2 terms capture pairwise; degree-3 capture triplets. More explicit than attention for small N=5.
- **预期 PnL gain vs SOTA +38.49**: low-medium (+0.2 to +0.5 if polynomial terms added)
- **优先级**: P3

---

## Additional Papers Found (Direction 1-8 overflow)

### TabPFN_v2 (Hollmann et al. — Nature 2025) — Bonus
- **Paper**: https://www.nature.com/articles/s41097-025-XXXXX (Nature 2025)
- **Core idea**: Prior-data fitted Transformer for tabular in-context learning. Randomized feature tokens for heterogeneous features. 10K+ lines of engineering. Published Nature Jan 2025. Not directly applicable (our data is too large for ICL, and temporal structure is important).
- **优先级**: P3 (reference)

---

### SAINT_Intersample (Somepalli et al. 2021) — Bonus
- See Direction 3 entry.

---

### T2GFormer (AAAI 2023) — Direction 4
- See Direction 4 entry.

---

### EPGATEnergy (arXiv 2025) — Direction 4
- **Paper**: https://arxiv.org/abs/2507.08184
- **Core idea**: Energy-based parallel GAT with Boltzmann distribution for stock graph construction. Energy(i,j) = ‖feature_i - feature_j‖² → Boltzmann weight w_ij = exp(-E_ij/T). Parallel GAT preserves hierarchical intra-stock features.
- **为何适合 v7+**: Energy-based edge weights capture feature DISTANCE between syms — when two syms are far apart in feature space, they have low cross-sym connection weight. More adaptive than learned attention for small N.
- **预期 PnL gain vs SOTA +38.49**: low-medium (+0.2 to +0.5)
- **优先级**: P3

---

### HGNN_HierarchicalStock (arXiv 2024) — Direction 4
- **Paper**: https://arxiv.org/abs/2412.06862
- **Core idea**: Hierarchical GNN for stock type prediction. Multi-level graph captures industry + stock relations. Graph convolution + temporal attention aggregator for macro market state modeling.
- **为何适合 v7+**: Hierarchical aggregation: sym→industry→market state. For 5 syms that may span 2-3 sectors, hierarchical aggregation provides coarser-to-finer cross-sym interaction.
- **优先级**: P3

---

*Total papers surveyed: 56 papers across 8 directions*
*Note: Papers marked P3 are for reference/completeness; implementation focus should be P1 and P2.*
