# HFT LOB Prediction: 2024-2026 Research Scan
## Methods, Papers, and Implementation Roadmap

**Generated**: 2026-05-08  
**Context**: 5-stock HFT LOB competition, 100-tick sliding window, 3-class prediction (up/flat/down), fee=0.0001, ~12 PnL gap to leaders  
**Baseline**: LGB + MLP + conformal prediction ensemble

---

## 1. Executive Summary

Top findings from scanning 2024-2026 literature:

- **TLOB (Feb 2025)** is currently the strongest published single-model for LOB mid-price 3-class prediction. Its dual-attention (temporal + spatial) architecture with de-biased labeling beats all FI-2010 SOTA by avg +3.7 F1. Crucially it is stateless and sym-agnostic. Implementation is open-source. **High-priority drop-in candidate.**

- **MLPLOB (Feb 2025)** from the same TLOB paper shows that a pure MLP with bilinear normalization beats nearly all prior CNN/RNN/Transformer baselines on short horizons. This is a signal that normalization and label engineering matter more than model complexity — our current MLP may be miscalibrated rather than architecturally wrong.

- **LOB-aware feature engineering (OFI + Multi-level OBI)** consistently outperforms raw price/volume LOB inputs across multiple papers. Order Flow Imbalance derived features (deep OFI at 10 levels, multi-horizon) provide cheap alpha that our baseline likely misses.

- **BINCTABL (2024)** wins on the most rigorous robustness benchmark (15 models × FI-2010): an adaptive bilinear normalization layer added to CNN-LSTM-attention gives +9.2 F1 over next best and a 99.7 robustness score. The normalization trick alone is worth investigating.

- **Siamese bid/ask symmetric architecture (2025)** with OFI features demonstrated >75% win rate across 149 test sets. The structural symmetry prior (ask side mirrors bid side) is a strong inductive bias that should reduce overfitting on our 5-stock set.

- **T-KAN (Jan 2026)** replaces LSTM gates with learnable KAN spline activations, yielding +19.1% F1 on FI-2010 and +132% vs -83% trading returns. The spline gates create learnable "dead zones" near noise, functioning as adaptive denoising — directly relevant to our ternary classification problem.

- **LOBERT (Nov 2025)** — BERT-style foundation model for LOB messages, pretrained on large unlabeled data. Reduces context length ~20x vs prior methods. Demonstrates that pretraining on unlabeled LOB data transfers to mid-price prediction. Not immediately deployable (needs message-level data), but shows the direction.

- **Order Book Filtration (Jul 2025)**: removing fleeting/spoofing orders from OBI computation (lifetime < threshold, modification count > threshold) improves directional signal Pearson correlation and regime detection. This is a pure feature preprocessing step compatible with our architecture.

- **LOBFrame/Deep LOB microstructural guide (2024)**: large-tick vs small-tick stock categorization predicts model effectiveness. If our 5 stocks include small-tick stocks (spread >= 3× tick size), deep learning's edge shrinks; feature engineering and simpler models can catch up.

- **Better inputs > deeper models (Jun 2025)**: Kalman/Savitzky-Golay filtering on LOB snapshots before feature extraction lets logistic regression match DeepLOB. For our competition, clean price normalization may be the single highest-leverage intervention.

---

## 2. Top 10 Methods to Evaluate (Ranked by Impact × Feasibility)

| Rank | Method | Impact | Difficulty | Stateless | Sym-Agnostic | Key Gain |
|------|--------|--------|------------|-----------|--------------|---------|
| 1 | **TLOB Dual-Attention Transformer** | Large | Medium | Yes | Yes | +3.7 F1 on FI-2010 SOTA |
| 2 | **Multi-level OFI Features + Siamese bid/ask** | Large | Easy | Yes | Yes | >75% win rate over baselines |
| 3 | **Adaptive Bilinear Normalization (BINCTABL trick)** | Medium-Large | Easy | Yes | Yes | +9.2 F1, best robustness |
| 4 | **T-KAN: KAN spline gates in LSTM** | Medium-Large | Medium | Yes | Yes | +19.1% F1, +132% returns |
| 5 | **LOB Filtration (lifetime/count filtering of OBI)** | Medium | Easy | Yes | Yes | Cleaner OBI signal |
| 6 | **LiT patch-based Transformer + LSTM hybrid** | Medium | Medium | Yes | Yes | Beats all baselines on crypto LOB |
| 7 | **CVML: Convolutional Cross-Variate Mixing Layers** | Medium | Medium | Yes | Yes | +244.9% avg on futures MPRF |
| 8 | **Improved Labeling (TLOB horizon-bias removal)** | Medium | Easy | Yes | Yes | Removes systematic label leakage |
| 9 | **Volume-based event sampling** | Small-Medium | Medium | Yes | Yes | Better captures activity magnitude |
| 10 | **LOBERT-style pretraining on unlabeled LOB data** | Large (future) | Hard | Yes | Yes | Foundation model transfer |

---

## 3. Detailed Method Profiles

---

### Method 1: TLOB — Dual Attention Transformer for LOB

**Paper**: "TLOB: A Novel Transformer Model with Dual Attention for Price Trend Prediction with Limit Order Book Data"  
**Authors**: Leonardo Berti, Gjergji Kasneci  
**Year**: 2025 (Feb)  
**URL**: https://arxiv.org/abs/2502.15757  
**Code**: https://github.com/LeonardoBerti00/TLOB

**Key Insight**: Standard Transformers process tokens along a single dimension, but LOB data requires two distinct attention patterns — temporal (across the 100-tick window) and spatial (across the 40 LOB features per snapshot). TLOB uses two sequential self-attention heads per block: one along time axis, one along feature axis, separated so each learns its own inductive bias. The critical addition is Bilinear Normalization at input and an MLPLOB feed-forward block replacing the standard FFN.

**Architecture Details**:
- 4 TLOB blocks, each with: Temporal Self-Attention → Spatial Self-Attention → MLPLOB FFN
- Sequence length: 128 ticks (optimal from grid search; 100 should work)
- 1 attention head (finds sufficient for LOB's structured patterns)
- Sinusoidal positional embeddings
- Input: T × 40 tensor (T ticks × 10 levels × 4 features)
- Bilinear Normalization layer before transformer blocks

**FI-2010 F1 Scores**:
- h=10: TLOB=81.55, MLPLOB=81.64 (MLPLOB wins short-horizon)
- h=20: TLOB=82.68, MLPLOB=84.88
- h=50: TLOB=90.03, MLPLOB=91.39
- h=100: TLOB=92.81, MLPLOB=92.62 (TLOB wins long-horizon)

**Improved Labeling Formula**:
```
l(t,h,k) = [w+(t,h,k) - w-(t,h,k)] / w-(t,h,k)
w+(t,h,k) = (1/(k+1)) * sum_{i=0}^{k} p(t+h-i)   # forward smoothing window
w-(t,h,k) = (1/(k+1)) * sum_{i=0}^{k} p(t-i)     # backward smoothing window
```
The key innovation: k (smoothing window) and h (prediction horizon) are now independent parameters. Prior methods coupled them (k=h), creating "horizon bias" where longer horizon labels were artificially smoother. With decoupled k, labels are unbiased across horizons.

**Compatibility**:
- Stateless: Yes (window-based, no cross-call state)
- Sym-agnostic: Yes (no stock embeddings)
- Date-independent: Yes
- 100-tick window: Yes (128 is default, 100 is compatible)

**Estimated Impact**: Large — direct drop-in replacement for our MLP backbone  
**Implementation Difficulty**: Medium — PyTorch, open-source code available

**Key Warning**: Despite FI-2010 gains, TLOB authors note F1 drops significantly when threshold θ is set to average spread (transaction cost). High classification accuracy ≠ high PnL. Need to tune θ to match our fee=0.0001 regime.

---

### Method 2: Multi-Level OFI Features + Siamese Bid/Ask Architecture

**Paper**: "An Efficient Deep Learning Model to Predict Stock Price Movement Based on Limit Order Book"  
**Authors**: (arXiv 2505.22678)  
**Year**: 2025 (May)  
**URL**: https://arxiv.org/abs/2505.22678

**Supporting Paper**: "Deep Order Flow Imbalance: Extracting Alpha at Multiple Horizons from the LOB"  
**Authors**: Kolm, Turiel, Westray  
**Year**: 2023 (published in Mathematical Finance), highly cited and used in 2024-2025 work  
**URL**: https://onlinelibrary.wiley.com/doi/10.1111/mafi.12413

**Key Insight**: Two orthogonal improvements:

1. **OFI as feature** beats raw LOB for mid-price prediction across all horizons. OFI at level l is:
   ```
   OFI_l(t) = ΔV_bid_l(t) × I(ΔP_bid_l > 0) - ΔV_ask_l(t) × I(ΔP_ask_l < 0)
   ```
   Using all 10 levels gives a 10-dimensional OFI vector that captures directional pressure deep into the book.

2. **Siamese architecture** exploits the structural symmetry: bid side and ask side of LOB are inherently mirror images. Processing them through identical weight-sharing encoders then combining via feature subtraction (f_ask - f_bid) → 2-layer MLP decoder reduces parameters, reduces overfitting, and makes the model invariant to absolute price level.

**Performance**: >75% win rate vs non-Siamese baseline across 149 test sets. Best: LSTM-MHA Siamese (4 heads, D=64) with OFI features.

**OFI Feature Engineering (complete recipe)**:
- For each of 10 levels: compute bid/ask price changes and volume changes tick-by-tick
- If price increases: add current volume; if price decreases: add negative previous volume; else add volume diff
- Stack levels into vector: OFI_1 ... OFI_10 (10-dim per tick)
- Compute rolling sums at multiple horizons: h=1, 5, 10, 20, 50 ticks → 50 additional features
- Divide by average volume to normalize across stocks (sym-agnostic normalization)

**Compatibility**:
- Stateless: Yes
- Sym-agnostic: Yes (volume-normalized OFI)
- 100-tick window: Yes

**Estimated Impact**: Large — OFI features provide direct microstructure signal our LGB/MLP may be computing only implicitly  
**Implementation Difficulty**: Easy — pure NumPy vectorized computation

---

### Method 3: Adaptive Bilinear Normalization (BINCTABL)

**Paper**: "LOB-Based Deep Learning Models for Stock Price Trend Prediction: A Benchmark Study"  
**Authors**: (Artificial Intelligence Review, 2024)  
**Year**: 2024  
**URL**: https://link.springer.com/article/10.1007/s10462-024-10715-4  
**Framework**: https://github.com/FinancialComputingUCL/LOBFrame (LOBCAST framework)

**Key Insight**: The BINCTABL model — CNN-LSTM-Attention + **Bilinear Normalization** — won the most comprehensive robustness benchmark over 15 state-of-the-art models. The innovation is normalizing jointly along BOTH temporal and feature dimensions simultaneously (bilinear = across both axes), unlike standard batch norm (feature axis only) or layer norm (temporal axis only).

**Why it matters for non-stationary LOB**: Standard normalization assumes the data distribution is stable. LOB data violates this with regime changes, tick-size jumps, and volatility clustering. Bilinear normalization adapts to both within-window temporal drift AND cross-feature covariance shifts.

**Implementation (from TLOB paper supplementary)**:
```
BN(X) = (X - μ_row) / σ_row  [normalize each time step across features]
then:   (result - μ_col) / σ_col [normalize each feature across time steps]
learnable scale γ, shift β applied after
```

**Benchmark Results (FI-2010, avg over all horizons and seeds)**:
- BINCTABL: 82.6% ± 7.0 F1 (best)
- Second best (DLA): 73.4% F1
- Third (CTABL without bilinear): 69.6% F1
- BINCTABL robustness score: 99.7/100 (best)
- On unseen stocks: 73.5% retention (best generalization)

**Compatibility**:
- Stateless: Yes
- Sym-agnostic: Yes
- Drop-in: This normalization layer can be added to any architecture including our current MLP

**Estimated Impact**: Medium-Large — normalization trick that robustifies any model  
**Implementation Difficulty**: Easy — 10 lines of code, plug into existing MLP/LGB feature pipeline

---

### Method 4: T-KAN — Temporal KAN for LOB (LSTM with Spline Gates)

**Paper**: "Temporal Kolmogorov-Arnold Networks (T-KAN) for High-Frequency Limit Order Book Forecasting: Efficiency, Interpretability, and Alpha Decay"  
**Year**: 2026 (Jan)  
**URL**: https://arxiv.org/html/2601.02310

**Key Insight**: Replace the fixed linear weight matrices inside LSTM gates with learnable B-spline activation functions (KAN layers). Each gate (input, forget, output, candidate) now applies a univariate spline to transform inputs rather than a linear map followed by sigmoid/tanh. The splines learned on FI-2010 converge to sigmoidal S-curves with **"dead zones" near zero** — effectively learning to filter micro-structure noise below a learned threshold while amplifying larger signals.

**Architecture**:
- Dual-layer LSTM encoder (64 hidden units) where weights W, U are replaced by KAN spline functions
- KAN classification head (replaces standard linear + softmax)
- B-spline degree: 3 (cubic), grid points: 5
- Input: standard DeepLOB preprocessing (10-level LOB, 40 features)

**Performance on FI-2010 (horizon k=100)**:
- DeepLOB F1: 0.3354 → T-KAN F1: **0.3995** (+19.1%)
- DeepLOB precision: 0.4604 → T-KAN: 0.5343 (+16.0%)
- **Trading returns**: DeepLOB: -82.76% drawdown → T-KAN: **+132.48%** at 1.0 bps costs

The trading result is remarkable — T-KAN doesn't just improve classification, it dramatically improves actual profitability. This suggests the spline gates are filtering "false positives" that DeepLOB would trade on but T-KAN abstains from.

**Why relevant to our 3-class problem**: The "flat" class in ternary prediction is precisely the regime where a model should abstain. T-KAN's learned dead zones naturally create an abstention zone that maps to the "stable/flat" prediction, potentially improving our flat-class recall.

**Compatibility**:
- Stateless: Yes
- Sym-agnostic: Yes (no stock-specific parameters)
- 100-tick window: Yes

**Estimated Impact**: Medium-Large — especially for reducing false positives (our cost structure penalizes wrong directional bets)  
**Implementation Difficulty**: Medium — needs `efficient-kan` or `pykan` library, then swap LSTM weight matrices

---

### Method 5: LOB Order Flow Filtration (Signal Denoising via Order Filtering)

**Paper**: "Order Book Filtration and Directional Signal Extraction at High Frequency"  
**Authors**: Aditya Nittur Anantha, Shashi Jain, Prithwish Maiti  
**Year**: 2025 (Jul)  
**URL**: https://arxiv.org/abs/2507.22712

**Key Insight**: Before computing Order Book Imbalance (OBI), filter out three categories of noisy orders:

1. **Fleeting orders** (lifetime < threshold T̄): Cancel-and-replace in milliseconds, pure noise/spoofing
2. **Probing orders** (modification count > M̄): Repeatedly repriced to test liquidity — not genuine supply/demand
3. **Rapid-fire updates** (inter-update time < ℳ̄): Automated HFT ping-flooding, distorts true book depth

After filtering, filtered OBI shows higher Pearson correlation with contemporaneous returns and better regime-to-return R² than raw OBI.

**Key finding**: "OBI computed using trade events exhibits stronger causal alignment with future price movements than filtered quote OBI." This means trade-direction OBI (aggressor side of each trade) is the highest-quality signal — better than any filtering of the passive quote book.

**Feature Engineering Recipe (compatible with snapshot-only LOB data)**:
Even without message-level data, we can approximate:
- Compute bid/ask depth change at each level tick-by-tick
- Large one-tick changes (> X%) flag potential fleeting orders → down-weight or clip
- Multi-level OBI using only levels with persistent presence (> Y% of window ticks)
- Trade-flow-weighted OBI: weight each OBI observation by volume traded at that tick

**Compatibility**:
- Stateless: Yes
- Sym-agnostic: Yes (threshold can be relative to avg spread)
- LOB snapshot input: Partially (full filtration needs message-level data; we can approximate)

**Estimated Impact**: Medium — incremental improvement to OBI features  
**Implementation Difficulty**: Easy (approximation) to Hard (full message-level)

---

### Method 6: LiT — Limit Order Book Transformer with Structured Patching

**Paper**: "LiT: Limit Order Book Transformer"  
**Journal**: Frontiers in Artificial Intelligence, 2025  
**URL**: https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1616485/full

**Key Insight**: Treat LOB snapshots like vision patches. Instead of flattening the 100×40 input into a sequence, divide it into structured patches where:
- **Vertical patch height = number of price levels** (capture the full depth structure at a time point)
- **Horizontal patch width = narrow temporal window** (4 ticks per patch)

The crucial finding: "narrower temporal windows (W=4) with deeper spatial windows (H=40)" delivered optimal performance. This means the model learns full book-depth structure at each moment before integrating temporally, rather than learning shallow per-level temporal patterns. The patch embedding projects each patch to a token, then Transformer self-attention operates on the token sequence.

**Architecture**:
- Patch embedding: (100 ticks × 40 features) → patches of size (4 ticks × 40 features)
- Each patch → linear projection to d-dim token
- Transformer encoder (multi-head attention, no CNN)
- LSTM layer on top of transformer output (hybrid)
- Final classification head

**Performance (Binance cryptocurrency LOB)**:
- 300-500ms horizon: 58.99% F1 (best)
- 300-700ms: 63.65% F1
- 300-1000ms: 66.40% F1
- Zero-shot transfer to new asset: ~5% F1 drop; fine-tuning recovers and exceeds

**Key difference from TLOB**: LiT uses patches (ViT-style) while TLOB uses per-snapshot tokens. LiT also adds an LSTM after the transformer. For 100-tick windows, LiT would create 25 patches (100/4), making the sequence short enough for efficient attention.

**Transfer learning note**: Fine-tuned LiT transfers well across assets with distribution shift. Relevant if our 5 stocks have heterogeneous microstructure.

**Compatibility**:
- Stateless: Yes
- Sym-agnostic: Yes
- 100-tick window: Yes (exact fit with W=4, 25 patches)

**Estimated Impact**: Medium  
**Implementation Difficulty**: Medium

---

### Method 7: CVML — Convolutional Cross-Variate Mixing for LOB Time Series

**Paper**: "A Benchmark Study For Limit Order Book (LOB) Models and Time Series Forecasting Models on LOB Data"  
**Year**: 2024 (Oct), OpenReview  
**URL**: https://openreview.net/forum?id=MhD9rLeU31

**Key Insight**: Standard multivariate time series models (Transformer, PatchTST, TimesNet, etc.) process LOB features independently (channel-independent) or with full attention (O(d²) cross-feature cost). CVML introduces **convolutional cross-variate mixing** as an add-on layer: a 1D depthwise separable convolution mixes adjacent LOB features (e.g., bid level 1 price with bid level 1 volume, bid level 2, etc.) using the structural ordering of the LOB tensor.

This is the LOB-aware inductive bias missing from generic time series models: the 10-level × 2-side structure means adjacent features are physically adjacent in the order book, and a local convolution along the feature dimension captures this spatial structure.

**Performance**: +244.9% average improvement in mid-price return forecasting when CVML is added to various time series backbone models (PatchTST, TimesNet, etc.) on LOB data.

**Implementation**: Add as a plug-in layer before the backbone model. Essentially a 1D Conv with groups=40 (one per feature) that performs cross-feature local mixing using the LOB structural ordering.

**Compatibility**:
- Stateless: Yes
- Sym-agnostic: Yes
- Plug-in: Can add to our existing MLP/LGB feature extractor

**Estimated Impact**: Medium — mainly helps generic time series models; less benefit if model is already LOB-aware  
**Implementation Difficulty**: Medium (need to reshape input correctly)

---

### Method 8: Improved Labeling (Horizon-Bias Removal)

**Paper**: TLOB paper (same as Method 1), labeling section  
**Year**: 2025  
**URL**: https://arxiv.org/abs/2502.15757

**Key Insight**: Most LOB papers (including DeepLOB and predecessors) use:
```
l(t,h) = [mean(p(t+1)...p(t+h)) - mean(p(t-h+1)...p(t))] / mean(p(t-h+1)...p(t))
```
Here, the smoothing window equals the prediction horizon. For h=100 ticks, both the future window AND the past reference window are 100 ticks long. This creates "horizon bias": longer horizons use smoother references, making classes imbalanced and prediction artificially easier at long horizons.

**TLOB fix**: Decouple k (smoothing) from h (prediction):
```
l(t,h,k) = [w+(t,h,k) - w-(t,h,k)] / w-(t,h,k)
```
where k is a fixed small window (e.g., k=5 ticks) regardless of h.

**For our competition** (100-tick window, predict immediate next move): Our labels may currently use h=1 (1-tick-ahead), which has minimal bias. But if we experiment with longer horizons, this fix becomes critical.

**Additional insight**: Setting classification threshold θ to the average transaction cost (fee=0.0001) rather than a fixed α×std is more economically meaningful. Moves smaller than fee are predicted as "flat" (not worth trading).

**Estimated Impact**: Medium — improves label quality especially for multi-horizon experiments  
**Implementation Difficulty**: Easy — pure Python, replace label function

---

### Method 9: LOBERT — Foundation Model for LOB Messages (Pretraining Approach)

**Paper**: "LOBERT: Generative AI Foundation Model for Limit Order Book Messages"  
**Year**: 2025 (Nov, NeurIPS GenAI in Finance Workshop)  
**URL**: https://arxiv.org/abs/2511.12563

**Key Insight**: BERT-style masked autoencoding pretraining on large-scale LOB message data. Each "token" is a complete multi-dimensional LOB event (order type, price, volume, time), not individual features. This reduces context length ~20× vs feature-flattened approaches. Pretraining on unlabeled data from many stocks and time periods learns a universal representation of market microstructure dynamics. Fine-tuning on mid-price prediction then needs only small labeled datasets.

**Why this matters for a 5-stock competition**: If pretrained model exists publicly, fine-tuning on our 5-stock labeled data could give substantial gains without overfitting. The foundation model has "seen" thousands of market regimes and generalizes across stocks (sym-agnostic by design).

**Current limitation**: Requires message-level LOBSTER/NASDAQ ITCH data, not just snapshot data. Our competition likely provides snapshots. However:
1. Authors plan to release pretrained weights
2. The self-supervised pretraining principle can be adapted to snapshot data (mask-and-predict on LOB levels)

**Estimated Impact**: Large (future) — could be transformative once weights are public  
**Implementation Difficulty**: Hard — needs message data; snapshot-adapted version requires custom pretraining

---

### Method 10: Deep LOB Microstructural Guide (LOBFrame + Operational Framework)

**Paper**: "Deep Limit Order Book Forecasting: A Microstructural Guide"  
**Authors**: (published in Quantitative Finance 2025, submitted 2024)  
**URL**: https://arxiv.org/abs/2403.09267  
**Code**: https://github.com/FinancialComputingUCL/LOBFrame

**Key Insight**: Not a single model, but an operational framework for evaluating LOB models. Key actionable findings:

1. **Tick-size categorization predicts model choice**: Compute avg_spread / tick_size for each stock:
   - Large-tick (ratio ≤ 1.5): Deep learning wins; DeepLOB MCC ~0.29 at h=10
   - Small-tick (ratio ≥ 3): Deep learning advantage shrinks; simple models competitive
   - For each stock type, choose appropriate model

2. **5-day rolling z-score normalization** per feature is better than global normalization. The rolling window adapts to non-stationarity while remaining fully stateless (uses only the 100-tick window + a slightly larger lookback).

3. **LOBFrame pipeline** provides preprocessing, model training infrastructure, and a "probability of profitable execution" metric that better captures real-world trading performance than classification F1.

4. **Volume distributions** at best quotes cluster by tick-size category → can be used for unsupervised stock categorization without stock-specific models.

**For our 5 stocks**: Profile each stock's avg_spread/tick_size ratio, then select model architecture accordingly. Different stocks may warrant different model types in the ensemble.

**Estimated Impact**: Medium — operational insight that improves ensemble design  
**Implementation Difficulty**: Easy (analysis) to Medium (architecture-per-stock)

---

### Bonus: Attention-Based Multi-Asset Order Flow Networks

**Paper**: "Attention-Based Multi-Asset Order Flow Networks for Enhanced Mid-Price Prediction"  
**Year**: 2025 (ICAIF Nov 2025)  
**URL**: https://dl.acm.org/doi/10.1145/3768292.3770430

**Key Insight**: For multi-stock datasets (exactly our 5-stock setup), order flow signals from correlated stocks carry predictive information for each other. An attention network over cross-asset OFI features captures these correlations dynamically, unlike static correlation matrices.

**CRITICAL CONSTRAINT CHECK**: This method uses cross-asset information (OFI from other stocks in the window). It does NOT use stock-specific embeddings — the attention is over flow features, not stock IDs. Therefore it is sym-agnostic IF we treat stocks as interchangeable sources of flow signals. However, since test points have "sym 0-4 but may contain training-external stocks," if the model uses fixed per-slot positions (slot 0 = stock A always), this breaks. Solution: make cross-asset attention position-invariant (attention over a set of stock OFI vectors, order-agnostic → pooling or sorted input).

**Estimated Impact**: Small-Medium (our 5 stocks may not have strong cross-asset predictability)  
**Implementation Difficulty**: Medium-Hard (cross-asset feature alignment complexity)

---

## 4. Implementation Roadmap

### Phase 1 (Week 1): Low-Effort, High-Value Feature Engineering

**Priority 1A: Multi-Level OFI Features** (1-2 days)

Compute OFI for all 10 bid/ask levels and add to feature set:
```python
# For each tick t, level l:
# dPbid = P_bid_l(t) - P_bid_l(t-1)
# dPask = P_ask_l(t) - P_ask_l(t-1)
# OFI_bid_l(t) = V_bid_l(t) if dPbid > 0 else (-V_bid_l(t-1) if dPbid < 0 else V_bid_l(t) - V_bid_l(t-1))
# OFI_ask_l(t) = similar for ask side
# Net_OFI_l(t) = OFI_bid_l(t) - OFI_ask_l(t)
```
Rolling sums at t=1, 5, 10, 20, 50 give 50 new features. These go directly into LGB.

**Priority 1B: Improved Labeling** (0.5 days)

Implement TLOB's decoupled k,h labeling. Even if h=1 for immediate prediction, fix k=5 and verify class balance. Adjust threshold θ to transaction-cost-equivalent (0.0001).

**Priority 1C: Bilinear Normalization** (0.5 days)

Add bilinear normalization to existing MLP: normalize each 100-tick window both temporally (per-feature z-score across ticks) and spatially (per-tick z-score across features). Replace current batch norm.

**Expected gain from Phase 1**: +3-8 PnL points from feature quality alone.

---

### Phase 2 (Week 1-2): Architecture Experiments

**Priority 2A: TLOB Drop-In** (2-3 days)

Clone https://github.com/LeonardoBerti00/TLOB. Adapt input to our 100-tick × (10-level × 4-feature) format. Key changes:
- Set sequence_length=100
- Use our OFI-augmented features (can stack OFI on top of raw LOB)
- Train with decoupled labeling (k=5, h=1 for our competition)
- GPU training: `device='cuda'`; inference: CPU (no GPU-specific ops at inference)

**Priority 2B: Siamese OFI Architecture** (2 days)

Implement symmetry-based model:
1. Split 100-tick input into bid channels and ask channels (each 10 levels × price/volume)
2. Process through identical LSTM-MHA encoder (shared weights)
3. Combine: subtract ask embedding from bid embedding
4. Feed to 2-layer MLP classifier

This is simpler than TLOB and may outperform at our 100-tick horizon.

**Priority 2C: T-KAN Replacement** (3 days)

Install `pip install efficient-kan`. Replace LSTM in current MLP with KAN-gated LSTM:
- Keep LSTM architecture but replace weight matrices with KAN spline functions
- Start with cubic B-splines, 5 grid points
- Ablate: KAN head only vs KAN gates vs full T-KAN

---

### Phase 3 (Week 2): Ensemble and Calibration

**Priority 3A: Multi-Architecture Ensemble**

Stack predictions from:
- LGB (current baseline, now with OFI features)
- TLOB (dual attention)
- Siamese OFI LSTM-MHA
- Possibly T-KAN

Use conformal prediction (already in baseline) to calibrate abstention thresholds per model. Key: ensemble by averaging probability distributions, not class votes.

**Priority 3B: Microstructural Stock Profiling**

For each of 5 stocks, compute avg_spread/tick_size ratio. If any stock is small-tick (ratio ≥ 3), consider downweighting deep learning predictions and upweighting feature-engineered LGB for those stocks. This is sym-agnostic (the routing is by stock microstructure features, not stock ID).

**Priority 3C: OBI Filtration**

Even with snapshot data (no message-level data), approximate filtration:
- Flag ticks where bid/ask depth changes by >50% in one step (potential fleeting order artifact)
- Compute OBI with and without these ticks; use the "stable" OBI as the primary feature

---

## 5. Raw Search Notes and URLs Found

### Directly Relevant Papers (2024-2026)

| Paper | Year | URL | Relevance |
|-------|------|-----|-----------|
| TLOB: Dual Attention Transformer | 2025 | https://arxiv.org/abs/2502.15757 | High - SOTA, open source |
| LOBERT: Foundation Model for LOB | 2025 | https://arxiv.org/abs/2511.12563 | High - pretrain |
| LOBench: Representation Learning Benchmark | 2025 | https://arxiv.org/abs/2505.02139 | High - benchmark |
| Deep LOB Forecasting Microstructural Guide | 2024/2025 | https://arxiv.org/abs/2403.09267 | High - operational framework |
| T-KAN for LOB Forecasting | 2026 | https://arxiv.org/html/2601.02310 | High - +19% F1 |
| LiT: LOB Transformer | 2025 | https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1616485/full | Medium |
| Siamese OFI Deep Learning | 2025 | https://arxiv.org/abs/2505.22678 | High - OFI features |
| HLOB: Homological Conv LOB | 2024 | https://arxiv.org/abs/2405.18938 | Medium |
| LOB Filtration + Signal Extraction | 2025 | https://arxiv.org/abs/2507.22712 | Medium |
| Better Inputs > Deeper Models (Crypto) | 2025 | https://arxiv.org/abs/2506.05764 | High - practical |
| Attention-Based LOB Reading+Forecasting | 2024 | https://arxiv.org/abs/2409.02277 | Medium |
| Painting the Market: Diffusion LOB | 2025 | https://arxiv.org/abs/2509.05107 | Low (simulation) |
| LOB-Bench: Generative AI Benchmark | 2025 | https://arxiv.org/html/2502.09172v1 | Low (generative) |
| BINCTABL Benchmark Study | 2024 | https://link.springer.com/article/10.1007/s10462-024-10715-4 | High - best robust |
| CVML Benchmark Study | 2024 | https://openreview.net/forum?id=MhD9rLeU31 | Medium |
| Multi-Asset Order Flow Networks | 2025 | https://dl.acm.org/doi/10.1145/3768292.3770430 | Medium |
| CMDMamba Financial Time Series | 2025 | https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1599799/full | Low |
| ALPE Adaptive RL for Mid-Price | 2024 | https://arxiv.org/html/2412.19372 | Low |
| ContraSim Contrastive Learning | 2025 | https://arxiv.org/abs/2502.16023 | Low (news-based) |
| Deep Order Flow Imbalance (Kolm) | 2023 | https://onlinelibrary.wiley.com/doi/10.1111/mafi.12413 | High - OFI features |
| Microprice High Resolution TM | 2024 | https://arxiv.org/abs/2411.13594 | Low |

### Code Repositories

| Repository | URL | What's There |
|------------|-----|-------------|
| TLOB (official) | https://github.com/LeonardoBerti00/TLOB | Full training code, data loaders |
| LOBFrame | https://github.com/FinancialComputingUCL/LOBFrame | Preprocessing + 9 model benchmarks |
| DeepLOB original | https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books | Classic baseline |

### Key Competition Resources

| Resource | URL | Notes |
|----------|-----|-------|
| ICAIF 2025 Conference | https://dl.acm.org/action/showFmPdf?doi=10.1145/3768292 | Finance AI papers Nov 2025 |
| ICAIF 2024 FinRL Contest | https://open-finance-lab.github.io/finrl-contest-2024.github.io/ | RL for trading |
| Stevens HFTC 2025 | https://fsc.stevens.edu/2025-high-frequency-trading-competition-recap/ | Academic HFT competition |

### Negative Results Worth Noting

1. **Mamba SSM for LOB**: No strong LOB-specific Mamba paper found. CMDMamba works for lower-frequency index prediction but LOB-specific Mamba appears absent. Not recommended currently.

2. **Graph Neural Networks for LOB**: No strong GNN paper found for LOB mid-price prediction. HLOB uses information filtering networks but not message-passing GNNs.

3. **Contrastive Pretraining for LOB**: ContraSim is news-headline based (not applicable). No LOB-specific contrastive pretraining paper found beyond LOBERT's masked-autoencoding approach.

4. **Diffusion Models for LOB**: "Painting the Market" is for simulation/generation, not direct prediction. Too slow for real-time inference.

5. **Cross-asset attention** (Multi-Asset Order Flow Networks): Potentially powerful but introduces cross-call state risks if not carefully designed. Defer to Phase 3.

---

## 6. Key Technical Insights Summary

### What Actually Works for LOB Prediction in 2024-2026

1. **Feature engineering > model complexity**: Multiple papers confirm that Kalman/SG filtering, OFI features, and rolling z-score normalization often close the gap between simple and complex models. Start here.

2. **Bilinear normalization is a superpower**: The single factor that made BINCTABL the most robust across all 15 models tested was normalizing both time and feature axes jointly. Easy to add to any model.

3. **Dual attention (temporal + spatial) > single attention**: TLOB's empirical ablation shows both attention types contribute. But MLPLOB (no attention at all) is competitive at short horizons, suggesting the normalization matters most.

4. **Symmetry priors reduce overfitting**: Siamese bid/ask weight sharing is a strong inductive bias that consistently helps in small-data regimes (exactly our 5-stock situation).

5. **OFI beats raw LOB features**: This result is consistent across multiple 2023-2025 papers. Order Flow Imbalance at all levels is a cleaner signal than raw price/volume arrays.

6. **Horizon-biased labels are the silent killer**: Check your labeling formula. If smoothing window equals prediction horizon, your model has a systematic label quality issue at long horizons.

7. **Small-tick stocks resist deep learning**: If spread >> tick size, deep learning's advantage disappears. Use simpler LOB-aware features for these stocks.

8. **Historical predictability is declining**: TLOB shows -6.68 F1 decline over time in their NASDAQ dataset. This means more recent test data is harder. Models that are robust (BINCTABL) matter more than those that are accurate on easy periods.

---

## 7. Constraints Verification

All top-3 methods pass the 3 hard constraints:

| Method | date=0 OK? | Stateless? | Sym-agnostic? |
|--------|-----------|-----------|---------------|
| TLOB | Yes (no date feature) | Yes (window-based) | Yes (no stock embeddings) |
| OFI + Siamese | Yes | Yes | Yes (volume-normalized) |
| Bilinear Normalization | Yes | Yes | Yes |
| T-KAN | Yes | Yes | Yes |
| LOB Filtration | Yes | Yes | Yes (relative thresholds) |

---

*Report end. Next steps: implement Phase 1 feature engineering within 1-2 days, then run TLOB experiment in Week 1.*
