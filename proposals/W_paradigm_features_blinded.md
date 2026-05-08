# 颠覆性 Feature Paradigm（Blinded Brainstorm）

> **Worker**: opus-xhigh, blinded mode
> **Date**: 2026-05-08
> **任务**: 不只是"加更多 features"——从 first principles 想思路上完全不同的 feature 类型，破解 local↑/platform↓ 的 +12 gap。

---

## Part 1: 题目理解（First Principles）

### 1.1 题目核心信息源是什么？

把 5 sym × 120 day × 2 session × 2001 ticks × 154 col 拆成 3 个独立信息层级：

| 层级 | 内容 | 我们用了多少？ |
|---|---|---|
| **Level A**：当前快照 (1 tick × 154 col) | 量价 + 10 档 LOB + 6 类订单流 + Kercheval-Zhang 衍生 | 充分用 |
| **Level B**：100-tick 时序 (100 × 154) | 价格 path, 订单流强度时序, depth shape 演化 | **只用了简单 summary**（mean/std/lag returns）；时序的**结构信息**几乎没用 |
| **Level C**：跨 sym × train-set 全局先验 | 5 sym 共享市场结构（撤单率分布、价差分布的"市场常识"） | 几乎没用（约束 §3 限制 sym embedding，但**全局非 sym-specific 先验**没违反约束） |

### 1.2 我们 LightGBM regression-on-Δmid 的盲点

按"信息榨取效率"看，我们暴露了 4 个未被榨干的方向：

1. **Path-as-object**：100-tick window 不只是 100 个独立时间点；它是**一条路径**。路径有：
   - 算子结构（path signatures, iterated integrals）
   - 拓扑结构（持续同调 persistent homology）
   - 复杂度结构（permutation/sample entropy）
   - 谱结构（wavelet, FFT）
   我们目前用的是"取 mean / std / lag return"，等于**把路径压成 4 个数**。

2. **Order arrival 是 self-/cross-exciting，不是泊松**：
   - 现在我们用 `lb_intst, la_intst, ..., ca_intst` 作为 raw count
   - **真正预测力来自 Hawkes 分支比 (branching ratio)**：到达事件激发后续事件的概率。如果 mb_intst 暴涨且分支比 > 0.7，说明 momentum 自激；< 0.3 说明 mean-reverting。**这是结构参数，不是统计量。**

3. **Midprice 包含 microstructure noise**（Roll/TSRV 文献）：
   - 我们对 midprice 做 Δ 然后 regression
   - 但 midprice 一阶差分有 **noise/signal SNR ≈ 1:1** 的污染（bid-ask bounce, queue jitter）
   - Δmid_t = Δsignal_t + (ε_t − ε_{t-1})
   - **如果模型学的是 Δmid，它一半的 capacity 在拟合 noise**
   - 这可能正是"local↑ platform↓"：local 442k 中的 noise pattern 不复现到 platform

4. **Generative manifold of LOB states 没建模**：
   - 我们 supervised-only 看 LOB → label 映射
   - **但 LOB 状态本身在 100-tick window 上有 manifold 结构**（depth shape 的"正常形状"）
   - VAE/AE 学这个 manifold，**reconstruction error = anomaly score**
   - 异常状态（深度突然异常 thin / thick）= 大移动概率激增 = predictive feature
   - **完全 unsupervised 学到的特征不会过拟合 442k label 分布**

### 1.3 为什么 local↑ platform↓ 撞墙？关键诊断

> **我们在 fit feature → label 的 mapping，但 platform test 集来自不同的 path-time 分布**。
> 加更多 hand-crafted feature = 学更多 mapping 细节 = local 涨，但 mapping 在分布漂移下不稳。

**真正能 transmit 的 feature** 必须：
1. 是**不变量**（对路径变形/时间重参数化稳健，如 path signature 的低阶系数）
2. 是**结构参数**（如 Hawkes 分支比、Hurst 指数），不是 raw 统计量
3. 是**对 noise 鲁棒的 pre-processing**（如 TSRV 滤波后的 clean midprice），而不是在污染信号上加更多特征

这才是"颠覆性 paradigm" vs "incremental feature"的分界线。

---

## Part 2: 5-10 个颠覆性 Feature Paradigm

### Paradigm 1: Path Signature Features（Lyons 1998, Chevyrev-Kormilitzin 2016）

- **描述**：把 100-tick window 上的多变量轨迹（midprice, OFI_1, spread1, log_vol）编码为**截断的 iterated integrals 系数**（level-2 / level-3 truncation → ~10-50 个数）。
- **出处**：[Rough Path Theory & Signatures Applied To Quantitative Finance (QuantStart)](https://www.quantstart.com/articles/rough-path-theory-and-signatures-applied-to-quantitative-finance-part-1/), [A Primer on the Signature Method in Machine Learning (Chevyrev-Kormilitzin 2016, arXiv:1603.03788)](https://arxiv.org/pdf/1603.03788), [Lyons (Oxford rough paths)](https://www.ox.ac.uk/research/research-impact/rough-paths-gaining-insights-and-building-solutions)
- **为什么颠覆性**：signature 是**对路径取算子代数表示**，与"取统计量"完全不同的范式。它的核心性质：
  - **时间重参数化不变**（path 重采样不变）→ 100→90 tick 缺数据时仍 robust
  - **L^∞-dense in path-functionals**：理论上任意连续 path-functional 都可以用 signature 多项式逼近
  - **non-commutative**：编码 path 各分量的**先后顺序**（midprice 涨之前/之后 OFI 涨，意义不同）
  - **stateless-friendly**：每个 100×4 window 独立计算，O(n × d^k)
- **实施成本**：中等。`iisignature` 或 `signatory` PyPI 包，~30 行；level-2 over 4 channels = 10 features，level-3 = 30 features
- **预估 platform 改进**：+3 ~ +6（保守）。理由：(a) 与现有 lag-return 特征**几乎正交**；(b) 时间不变性让它在 platform shuffling 下稳定；(c) 文献证明它能压缩高阶 path 信息到少量数字
- **风险**：(1) signature 系数 scale 跨 level 差很多（需要 log-signature 或 normalization）；(2) 对 path "抖动"（bid-ask bounce）敏感→应先做 Roll filter

---

### Paradigm 2: Within-Window Hawkes Branching Ratio Estimation

- **描述**：对每个 100-tick window，**用 6 个订单流强度时序 (lb/la/mb/ma/cb/ca _intst) 拟合一个简化 multivariate Hawkes process**，输出：
  - 6 维 baseline μ̂
  - 6×6 self/cross-excitation matrix α̂_ij
  - 衰减率 β̂
  - 总分支比 n̂ = ‖α̂‖/β̂（< 1 稳定，→1 critical，> 1 explosive）
- **出处**：[Bacry-Muzy: Hawkes processes in finance (arXiv:1502.04592)](https://arxiv.org/abs/1502.04592), [Lu & Abergel: High-dimensional Hawkes processes for limit order books (J Banking Financial Tech 2024)](https://link.springer.com/article/10.1007/s42786-024-00049-8), [Hawkes Processes in High-Frequency Trading (arXiv:2503.14814)](https://arxiv.org/pdf/2503.14814)
- **为什么颠覆性**：
  - 我们目前用 `lb_intst` 等的 **mean / lag**——这是泊松假设下的点估计
  - **真实 order flow 是 self-exciting**（mb 触发后 5-10 ticks 内更可能再 mb）
  - **branching ratio 是结构参数**，捕捉"order flow 是 momentum 还是 mean-reverting"
  - 直接 ties to 评分公式：分支比高 = 短期 momentum 强 → label_5 / label_10 信号；分支比低 + 高 baseline → mean-reverting → 抑制信号
  - **跨 sym / 跨 day 量级一致**（无需 normalization），platform 鲁棒
- **实施成本**：中-高。简化方案：用 method-of-moments 闭式估计（不用 MLE），20-30 行 NumPy；完整 MLE 用 `tick.hawkes` 包，~50 行但慢
- **预估 platform 改进**：+4 ~ +8。理由：直接利用我们独特的 6-type order flow 数据，比赛对手大概率没用上结构参数
- **风险**：(1) 100 tick 太短，估计噪声大→可降到 6-d → 1 维总强度的简化；(2) 需要 fit 时长合理（method-of-moments 很快）

---

### Paradigm 3: Microstructure Noise Filtering (Roll + TSRV) → Clean Midprice

- **描述**：把 midprice 序列分解为 `midprice_t = signal_t + ε_t`，用 Roll (1984) 估计 ε_t variance（基于一阶自协方差），用 Two-Scale Realized Volatility (Zhang-Mykland-Aït-Sahalia 2005) 估计 σ²_signal。**用 clean signal 替代 raw midprice 计算所有下游特征**。
- **出处**：[Roll 1984 (foundational, see Verousis Ch. 20)](https://repository.essex.ac.uk/24183/1/Chapter%2020%20T.%20Verousis.pdf), [Zhang-Mykland-Aït-Sahalia: TSRV (Princeton)](http://www.princeton.edu/~yacine/depnoise.pdf), [Hansen-Lunde "Realized Variance and Market Microstructure Noise"](https://www.federalreserve.gov/pubs/ifdp/2007/905/ifdp905.htm)
- **为什么颠覆性**：
  - 我们一直在 raw midprice 上做 regression。**这是错的**——midprice 一阶差分含 bid-ask bounce noise（≈ ε_t − ε_{t-1}，AR(-1) 模式）
  - 模型 50% capacity 在拟合 noise → 在 local 442k 上 noise pattern 是固定的（同一些 442k 行）所以 fit 上去了，但 platform shuffle 后 noise pattern 变化 → 性能掉
  - **这是 root cause 修正，不是 feature add**——它影响所有依赖 Δmid 的下游（regression target, lag return features, momentum）
  - 同时，估计出的 `noise_variance` / `signal_variance` 比例本身就是个**预测特征**：noise 占比高的窗口 = 预测难度高 = EV-gate 应该收紧
- **实施成本**：低。Roll 估计 = 1 行（自协方差 -ρ₁σ²），TSRV = 5 行；clean signal 重算 ~10 行
- **预估 platform 改进**：+3 ~ +10。理由：(a) 解决"local-fit 但 noise 不 transmit"的根因；(b) 同时给所有下游 feature 加 free 增益；(c) σ_noise/σ_signal 本身是 high-info 特征（gate signal）
- **风险**：(1) Roll 假设市价单 driven 的 bid-ask bounce，可能与 LOB 实际 microstructure 不符→可改用 ZMA Pre-Averaging；(2) TSRV 在 100 tick 上 small-sample，估计方差大

---

### Paradigm 4: Permutation Entropy + Sample Entropy（窗口复杂度）

- **描述**：对 midprice、avg_imbalance、spread 三个序列，分别计算 100-tick 窗口的 **permutation entropy** (Bandt-Pompe 2002) 和 sample entropy。低值 = 模式重复（trending/predictable）；高值 = 随机/chaos。
- **出处**：[Bandt-Pompe 2002 PE](https://pmc.ncbi.nlm.nih.gov/articles/PMC7516788/), [Multiscale PE for stock volatility (Preprints 2025)](https://www.preprints.org/manuscript/202511.1980), [PE in HFT microstructure (Physica A 2014)](https://www.sciencedirect.com/science/article/abs/pii/S0378437114005020)
- **为什么颠覆性**：
  - 我们没有任何"窗口可预测性"的元特征
  - PE 是 **ordinal-based、对 marginal distribution 不变**（heavy-tail 友好）
  - **作为 EV-gate 增强**：PE 高的窗口 = 模型置信度该低 = 不出手
  - 多尺度 PE（scale 5/10/20/40/60 对应 5 个 horizon）→ 每 horizon 都有自己的 "predictability score"
- **实施成本**：极低。`antropy` PyPI 包，每特征 1 行
- **预估 platform 改进**：+1 ~ +3（中等）。预测力来自 gating 而非直接 prediction；但 stateless-clean，platform 稳健
- **风险**：低。已有大量金融文献验证

---

### Paradigm 5: Variational Autoencoder Latent + Reconstruction Anomaly

- **描述**：用所有训练 100-tick × 154-col window 训练一个 VAE（encoder: Conv1D + MLP → 16-d latent；decoder: 反向）。推理时输出：
  - **16 个 latent 维度**作为压缩 features
  - **reconstruction MSE** 作为 anomaly score（异常 = 模型没见过的 LOB shape = 大移动 risk）
- **出处**：[Representation Learning of LOB: Comprehensive Study & Benchmarking (LOBench, arXiv:2505.02139)](https://arxiv.org/html/2505.02139v1), [Detecting Multilevel Manipulation via Cascaded Contrastive Representation Learning (arXiv:2508.17086)](https://arxiv.org/html/2508.17086v2)
- **为什么颠覆性**：
  - **Unsupervised**——不会过拟合 442k 标签分布
  - learns the *manifold of LOB states*；任何已存在 generative model 都没在我们栈里
  - latent 空间是**学习的**，对手手工特征都达不到
  - reconstruction error 在 platform 上信号强：anomalous state（无论什么造成的）通常对应大移动
- **实施成本**：中-高。Conv1D-VAE + 2 hour 训练（GPU）；推理时只跑 encoder（CPU 友好，100×154 → 16 latent < 1ms）
- **预估 platform 改进**：+3 ~ +7。Anomaly score 可能直接是 EV-gate 因子；latent 16 维 给 LightGBM 当 features
- **风险**：(1) VAE training instability；(2) recon-error 的标定（reconstruction MSE 跨 sym 量级要 normalize 但不能 per-sym→用 train-set 全局 quantile）；(3) latent 16-d 可能有冗余但树模型能处理

---

### Paradigm 6: Topological Data Analysis（持续同调 of LOB shape）

- **描述**：把 100-tick × (midprice, totalbsize, totalasize, spread1, imbalance) 看作 5-d 点云；计算 **Vietoris-Rips persistent homology**；输出：
  - 0-th persistence diagram L²-norm（连通性变化）
  - 1-st persistence diagram L²-norm（环结构出现/消失）
  - persistence entropy
- **出处**：[Gidea-Katz: TDA of financial time series (Physica A 2018)](https://www.sciencedirect.com/science/article/abs/pii/S0378437117309202), [Enhancing financial forecasting through TDA (Neural Computing 2024)](https://link.springer.com/article/10.1007/s00521-024-10787-x), [Topological ML for Financial Crisis (MDPI 2025)](https://www.mdpi.com/2073-431X/14/10/408)
- **为什么颠覆性**：
  - TDA 抓的是**形状的拓扑不变量**——对噪声、scaling 鲁棒
  - 文献证明 TDA L^p norm 在金融 crash 前几天显著上升（regime change detector）
  - **现存所有团队几乎没人这么做**——远超 LOB 圈常规思路
- **实施成本**：中。`giotto-tda` 或 `ripser` PyPI；100-tick × 5-d 点云 → 持续图 5-10s/window；可能太慢需要 down-sample
- **预估 platform 改进**：+1 ~ +5。新颖度高，但 100 tick 点云对 1-st homology 信息可能不够（需要至少几百点）
- **风险**：(1) 计算成本可能不允许（推理时 < 1ms 限制 → 必须 lite 实现，如只取 0-th persistence）；(2) 100 tick 太少→可能 noise dominate

---

### Paradigm 7: Bayesian Online Change-Point Detection（regime feature）

- **描述**：对 100-tick midprice + spread + imbalance 序列，在线运行 BOCPD（Adams-MacKay 2007）。输出每个时刻的 **run-length posterior**：
  - `tick_since_last_changepoint`（distance to regime change）
  - `P(changepoint at last tick)`
- **出处**：[Adams-MacKay 2007 BOCPD (arXiv:0710.3742)](https://arxiv.org/abs/0710.3742), [Real-time financial surveillance via QCD (arXiv:1509.01570)](https://arxiv.org/pdf/1509.01570)
- **为什么颠覆性**：
  - **基于贝叶斯后验，不是统计量**——量化"regime 持续了多久"
  - 直觉：刚换 regime → 持续动量未稳定 → 大移动概率高；regime 持续久 → mean-revert
  - 100 tick 限制内可在线计算，stateless-friendly（每窗内独立）
- **实施成本**：中。NumPy 实现 ~50 行；conjugate Gaussian-Inverse-Gamma model 简单
- **预估 platform 改进**：+2 ~ +4
- **风险**：BOCPD 假设 piecewise stationary；微观结构是连续 evolution，模型 mis-spec

---

### Paradigm 8: Wavelet/Spectral Decomposition of midprice path

- **描述**：100-tick midprice 序列做 Daubechies-4 离散小波分解（4 levels）；每 level 的 detail coefficients 的 energy 比例 = features（4 维）。
- **出处**：[Murtagh "Wavelets in Forecasting"](https://arxiv.org/abs/1208.5662), 标准多分辨率分析
- **为什么颠覆性**：
  - 把"短期噪声"和"中期趋势"在频域**显式分离**
  - low-freq 占比高 = 趋势 dominate；high-freq 占比高 = 噪声 dominate（与 paradigm 3 互补）
  - 比 Fourier 更适合非平稳信号
- **实施成本**：极低。`pywt.wavedec`，1 行
- **预估 platform 改进**：+1 ~ +2（低，因为 lag-return 已捕捉部分多尺度信息）
- **风险**：可能与已有 lag-return 高相关；建议作为辅助 feature

---

### Paradigm 9: Generalized OFI (GOFI / MLOFI) at All 10 Levels

- **描述**：Cont-Stoikov-Kukanov 的 OFI 只在 best bid/ask 计算。**MLOFI**: 在 bid_k / ask_k 各 level 单独计算 OFI，再用 1/k 加权聚合。**GOFI**: 进一步用 LOB depth shape 做加权。
- **出处**：[Cont-Kukanov-Stoikov: The Price Impact of Order Book Events (SSRN 1712822)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1712822), [Cross-impact of OFI in equity markets (Tandfonline 2023)](https://www.tandfonline.com/doi/full/10.1080/14697688.2023.2236159), [The Price Impact of Generalized OFI (arXiv:2112.02947)](https://arxiv.org/pdf/2112.02947)
- **为什么颠覆性 / 不**：
  - **半颠覆性**——是 paradigm 升级（multi-level 而不是 best-only），但同 family
  - 文献证明 MLOFI 比 best-only OFI 在 5-30s 预测窗口下显著更强
  - 我们目前 imbalance 是 cumulative size，没做"价格变化驱动"的 OFI 算法
- **实施成本**：低。10 个 level × ΔSize - ΔPrice 公式，~20 行 NumPy
- **预估 platform 改进**：+2 ~ +4
- **风险**：低，理论 well-established

---

### Paradigm 10: VPIN / Toxicity（per-window）

- **描述**：把 100-tick 按 cumulative volume 分成 K=10 个等量 bucket；每 bucket 用 BV（Bulk Volume Classification）分类成 buy/sell-driven；VPIN = E[|V_buy − V_sell|/V_total]。
- **出处**：[Easley-de Prado-O'Hara: VPIN (Review of Financial Studies 2012)](https://www.jstor.org/stable/41306113), [BV-VPIN paper (SSRN 2791243)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2791243)
- **为什么颠覆性**：
  - VPIN 衡量"信息不对称下 informed trader 强度"——直接等价于 "next move 是哪一方"
  - 100 tick 短，但 PIN 风格的 toxicity 可在 short window 上估计（micro-VPIN）
- **实施成本**：低。20 行
- **预估 platform 改进**：+1 ~ +3
- **风险**：(1) BV 分类需要 trade direction，我们有 mb/ma 数据可以替代；(2) 100 tick 内 K=10 bucket 太粗→可降到 K=5

---

## Part 3: Top 3 推荐（按"未做过 × high impact × stateless-friendly × cost"）

排序原因：
- **Top 1 / 2** 是 paradigm-level 颠覆，不是常规 feature add，且与现有栈正交，platform-transmit 概率最高
- **Top 3** 是 root-cause 修复，预期收益高且风险极低

### 🥇 Top 1: Path Signatures (Truncated to Level 3) — Paradigm 1

**为什么 #1**：
- **正交性最强**：所有现存团队（包括我们）都在做 statistics-of-path（mean/std/lag），signature 是 algebra-of-path——正交
- **平台鲁棒性最高**：signature **time-warp invariant**——platform shuffle / 不同 sym tick 频率扰动下值不变
- **理论保证**：dense in path-functionals，可以表示任意我们手工想得到的 path feature 的极限
- **stateless 完美**：每窗口独立计算
- **cost 可控**：4-channel × level-3 = 30 features，`signatory` 包 GPU 加速
- **platform 改进估计**：+3 ~ +6
- **唯一风险**：scale 跨 level 差大 → 必须 log-signature 或 z-score normalize

**实施 quick-start**：
```python
import signatory  # pip install signatory
# Build 4-channel path: [midprice_t, OFI_t, spread1_t, log(totalbsize_t+totalasize_t)]
path = build_path(window_100tick)  # shape [batch, 100, 4]
sig = signatory.signature(path, depth=3)  # shape [batch, 4 + 16 + 64] = 84-dim
log_sig = signatory.logsignature(path, depth=3)  # 更紧凑
features = log_sig  # 30-d
```

### 🥈 Top 2: Within-Window Hawkes Branching Ratio — Paradigm 2

**为什么 #2**：
- **直接利用我们独特的 6-type order flow 数据**（lb/la/mb/ma/cb/ca）——其他比赛不一定有
- 输出"branching ratio + cross-excitation matrix"是**结构参数**，跨 sym/day platform 稳定
- 直接 ties to 评分：分支比 → 短期 momentum 强度 → label_5 / label_10
- **stateless 友好**：每窗内 fit
- **cost 中等**：method-of-moments 闭式 ~30 行
- **platform 改进估计**：+4 ~ +8（最大潜在收益）
- **风险**：100 tick 估 6×6 矩阵 noisy → 降维到 3 类（market: mb+ma, limit: lb+la, cancel: cb+ca）

**实施 quick-start**：
```python
# Method of moments for univariate Hawkes (per type)
# Var(N_T)/E(N_T) = 1/(1-n)^2  where n = branching ratio
def estimate_branching(intensity_series):  # 100 ticks
    counts = intensity_series  # 用 _intst 作为 proxy of arrivals
    over_dispersion = counts.var() / max(counts.mean(), 1e-9)
    n_hat = 1 - 1/sqrt(over_dispersion)  # ∈ [0,1)
    return clip(n_hat, 0, 0.99)
# 然后 cross-excitation 用 covariance matrix 估
```

### 🥉 Top 3: Microstructure Noise Filtering (Roll/TSRV) — Paradigm 3

**为什么 #3**：
- **Root-cause 修复**——直接攻击 local↑ platform↓ 问题（noise pattern 不复现）
- **不是 feature add，是 upstream cleaner**——影响所有下游 features 的质量
- **cost 极低**（~20 行）；几乎无风险
- σ_noise/σ_signal **本身就是 EV-gate 信号**（noise 占比高 → 该窗预测难 → 抑制 signal）
- **platform 改进估计**：+3 ~ +10（区间大；如果 root cause 诊断正确，上限高）

**实施 quick-start**：
```python
# Roll noise variance estimate (1984)
def roll_noise_var(midprice):
    dp = np.diff(midprice)  # ΔP
    cov1 = np.cov(dp[:-1], dp[1:])[0,1]  # lag-1 autocovariance
    if cov1 < 0:
        sigma_noise = np.sqrt(-cov1)
        return sigma_noise**2
    return 0.0  # no noise detected

# TSRV: clean signal volatility (Zhang-Mykland-Aït-Sahalia 2005)
def tsrv_clean_var(midprice, K=5):
    # Two-scale: K-skip vs all-tick
    rv_all = (np.diff(midprice)**2).sum()
    rv_K = (np.diff(midprice[::K])**2).sum() * K  # scale up
    return rv_K - (1/K) * rv_all  # bias-corrected signal variance

# Cleaned midprice (Pre-Averaging style)
def preavg_clean(midprice, win=3):
    return np.convolve(midprice, np.ones(win)/win, mode='same')

# 使用：所有依赖 midprice 的下游 feature 改用 preavg_clean(midprice)
# 同时把 noise_var/signal_var 作为单独的 EV-gate 因子加进模型
```

---

## Part 4: 现状 Cross-Check

> 已读 `PROGRESS.md`。Cross-check 范围：PROGRESS.md（项目基础说明）+ 对话 prompt 中给出的 SOTA 描述 + git status / commit log 中可见的实验 folder 名（**不读 folder 内容**，只看名字判断 paradigm 是否被尝试过）。

### 4.1 PROGRESS.md 现状要点（与我 Part 1 推断对比）

| 我的推断 | PROGRESS.md 验证 | 一致性 |
|---|---|---|
| precision 远比 recall 重要（PnL 评分） | ✅ "precision 远比 recall 重要——讲座原文：交易侧重 precision，F0.5" | 完全一致 |
| label_60 (α=0.1%) 比 label_5 (α=0.05%) 更有出手余地 | ✅ "α=0.05% 的任务扣完手续费余地极薄；α=0.1% 任务更有利可图。**初期建议优先打 label_60**" | 完全一致 |
| 类别不均衡 → 全预测 1 = acc 高 PnL=0 | ✅ "label=1 占 60-80%。SVM 不加 weight 全预测 1" | 一致 |
| 评分 = max-over-h，不是 avg | ✅ "5 个任务取最好的那个参与排名" | 一致 |
| 必须 sym-agnostic | ✅ §1 三红线 | 一致 |

PROGRESS.md 较 stale（定格在项目启动期，§7"下一步候选路径"还是 ABCD 4 选 1），后续迭代记录在 experiments/ 下（按 blind 规则不读）。

### 4.2 Paradigm × 现状 cross-check

仅基于 git status 中可见的 **folder 名称**（不读内容）判断是否被试过。

| # | Paradigm | git folder 命名匹配 | 判断 |
|---|---|---|---|
| 1 | Path Signatures | 未见 `*signature*` 或 `*signatory*` 类目录 | 🟢 **未做过** |
| 2 | Hawkes branching ratio | ⚠️ **`R_Hawkes_OFI/`** 存在 | 🟡 **可能做过/部分做过**——需在 cross-check 后查实做的是 OFI 加权还是 branching ratio 估计 |
| 3 | Microstructure noise filter (Roll/TSRV) | 未见 `*noise*` `*roll*` `*tsrv*` `*denoise*` | 🟢 **未做过** |
| 4 | Permutation entropy | 未见 `*entropy*` `*complexity*` | 🟢 **未做过** |
| 5 | VAE / Autoencoder | 未见 `*vae*` `*autoencoder*` `*ae_*` | 🟢 **未做过** |
| 6 | TDA / persistent homology | 未见 `*tda*` `*topology*` `*persist*` | 🟢 **未做过** |
| 7 | BOCPD / change-point | 未见 `*bocpd*` `*changepoint*` `*regime*` | 🟢 **未做过** |
| 8 | Wavelet / spectral | 未见 `*wavelet*` `*fft*` `*spectral*` | 🟢 **未做过** |
| 9 | GOFI / MLOFI | ⚠️ `R_Hawkes_OFI/` 含 OFI 字样；`R_Pairwise_ALL/` 可能涵盖 | 🟡 **可能涵盖** |
| 10 | VPIN / toxicity | 未见 `*vpin*` `*toxic*` `*pin*` | 🟢 **未做过** |

### 4.3 殊途同归（已做过但 motivation 重叠）

PROGRESS 边角和 git folder 命名透露这些方向**与我的 paradigm 部分撞 motivation**：

- **`T115_scale_invariance`**：与 Paradigm 1 (Path Signatures 的 time-warp invariance) 部分撞——应去看是否做的是 `(midprice − midprice.mean()) / midprice.std()` 这种 input scaling，还是 path-level invariance。**如果只是 input scaling，path signatures 仍是颠覆性升级**。
- **`R_multitick_window`**：与 Paradigm 8 (Wavelet 多尺度) 部分撞——应去看做的是不同 window 长度还是不同尺度的频域分解。
- **`T57_pnl_aware_loss` / `T87 SPO+ DFL NN`**：与 paradigm 无关，是 loss-side 改造（决策 focused learning）。但**这正好是 Paradigm 3 的反向证据**——SPO+ 帮 +8.93 platform，因为它对噪声鲁棒；而 raw Δmid regression 则不然。**Roll/TSRV 滤波 (Paradigm 3) 是从 input 端做同一件事**——降低对 noise 的拟合压力。
- **`T66_adversarial_val`**：local-platform shift 已被识别。这强化我的 Part 1.3 诊断："local↑ platform↓ 是 distribution shift"。**Adversarial val 是诊断；Paradigm 3 (noise filter) + Paradigm 1 (signature time-invariance) 是从 feature 端的解决方案**——互补不冲突。

### 4.4 黄金（未试过且高价值）

按 Top-3 顺序复审：

1. **🥇 Path Signatures**：folder 名无匹配，PROGRESS.md 无相关讨论 → **金矿**。Paradigm-orthogonal、time-warp invariant 的特性对 platform shuffling 完美。`T115_scale_invariance` 撞 motivation 但很可能是 input scaling 而非 path signature；可以并行不冲突。

2. **🥈 Hawkes branching ratio**：⚠️ `R_Hawkes_OFI/` **存在**，是潜在 collision。**但**："R_" 前缀通常表示"研究/探索"型——可能只是 OFI 风格特征工程（如把 OFI 在 hawkes weighting 下加权），而不是**fit Hawkes process estimating branching ratio n̂ 作为 structural parameter**。这两件事差别巨大：
   - 前者：`feature_i = decay_weighted_sum(OFI[t-k:t])` ← 还是 statistic
   - 后者：`feature = n̂` (structural) + `α̂_ij` matrix (cross-excitation) ← 这是 paradigm 颠覆
   - **如果 PM 确认 R_Hawkes_OFI 只做了前者，本 paradigm 仍黄金**。
   - 即使做过 branching ratio 估计，仍可加 method-of-moments 估计 + multivariate cross-excitation matrix 作为升级。

3. **🥉 Microstructure noise filter (Roll/TSRV)**：folder 名无匹配 → **金矿**。这是**最容易实现、最容易上线、风险最低**的一项。理由：
   - 实施 < 30 行
   - 不动模型架构
   - 不动 loss
   - 不动 LGB feature 集（只换 midprice 计算源）
   - 直接攻击 root cause
   - **建议作为 Day-1 优先做的**，即使 path signature / Hawkes 还没做完

### 4.5 推荐执行优先级（基于 cross-check 修正后）

```
Week 1 (low-risk wins):
  Day 1-2:  Paradigm 3 (Roll + TSRV noise filter) ← 最低 cost, 立即上 platform
            同时计算 noise_var/signal_var 作为 EV-gate 因子
  Day 3-5:  Paradigm 1 (Path Signatures level-3) ← 最大潜在 ceiling
            建议同时计算 log-signature on 4 channels:
              [clean_mid, OFI_best, spread1, log(sum_size)]

Week 2 (medium-risk):
  先与 PM 确认 R_Hawkes_OFI 做的是什么——
    若是 OFI 加权: 上 Paradigm 2 完整版（branching ratio + cross matrix）
    若是 structural: 跳到 Paradigm 4 (PE) + Paradigm 5 (VAE)
  Day 6-8:  根据上面分支决定 (Paradigm 2 或 PE+VAE)
  Day 9-10: 上 Paradigm 5 VAE latent + recon-error EV gate

Backlog (if time):
  Paradigm 6 TDA (高新颖度但 cost 高)
  Paradigm 7 BOCPD (中新颖度)
  Paradigm 8/9/10 都偏 incremental，最后做
```

### 4.6 关键告知 PM

- **R_Hawkes_OFI 的实施细节**是决定 Paradigm 2 价值的关键。请确认**是否估计了 branching ratio n̂ 作为 feature**——这一个数本身就值得作为独立 feature。
- **Paradigm 3 (noise filter) 不需要等任何人**——它是 input-side 改造，可以立即上，与现有所有 trick 兼容。

---

```
RESULT: task=paradigm_features_blinded top_3=[path_signatures, hawkes_branching_ratio, microstructure_noise_filter] notes=10个paradigm评估，3个金矿(signature/Roll-TSRV/Hawkes-structural)。R_Hawkes_OFI存在需PM确认是OFI加权还是branching-ratio估计。Roll/TSRV最低成本可立即上(攻击noise→local↑platform↓的root cause)。T115_scale_invariance / R_multitick_window 撞motivation但很可能是浅层版本，paradigm版本仍正交可上。
```

