# 答辩小抄：Feature 工程 Q&A 速查

> 用于答辩时被老师追问因子细节、跨股 / 跨日泛化、KS-drop-11 的现场速查。
> 来源：`factor_families_report_v4.pdf` §4 / `ANSWERS_P3_FEATURES.md` §6 / `onepage_v2/sweep_5seed_summary.json` / `ablation_runs/feat_family_progressive/build_partition.py` / `factor_per_feature_audit/MASTER_summary.md` / `h_horizon_study/REPORT.md`（5 horizon × 5 seed 实测）。

---

## §0 符号与缩写速查表（老师翻这里）

### §0.1 英文缩写

| 缩写 | 英文全称 | 中文 |
|---|---|---|
| LOB | Limit Order Book | 限价订单簿 |
| OFI | Order Flow Imbalance | 订单流不平衡 |
| MLOFI | Multi-Level Order Flow Imbalance | 多档订单流不平衡 |
| GOFI | Generalized Order Flow Imbalance | 广义订单流不平衡 |
| EWMA | Exponentially Weighted Moving Average | 指数加权移动平均 |
| WMP | Weighted Mid-Price (Stoikov 2014) | 加权微价格 |
| RV | Realized Volatility | 实现波动率 |
| BV | Bipower Variation | 二次幂变差 |
| Roll | Roll (1984) effective spread | Roll 有效价差 |
| Kyle λ | Kyle (1985) lambda | Kyle 价格冲击系数 |
| KyleInv | Kyle inverse impact | Kyle 反向冲击系数 |
| BNS | Barndorff-Nielsen-Shephard | BNS 跳跃份额估计 |
| KS | Kolmogorov-Smirnov test | K-S 分布检验 |
| PSI | Population Stability Index | 群体稳定性指数 |
| SPO+ | Smart Predict-then-Optimize Plus | 决策焦点损失（增强版） |
| DFL | Decision-Focused Learning | 决策焦点学习 |
| ETUD | Execution Threshold Up / Down | 非对称执行阈值 |
| EV | Expected Value | 期望收益 |
| NN | Neural Network | 神经网络（此处特指 MLP） |
| MLP | Multi-Layer Perceptron | 多层感知机 |
| LGB | LightGBM | 微软梯度提升决策树 |
| HFT | High-Frequency Trading | 高频交易 |
| OOD | Out-of-Distribution | 分布外 |
| LOSO | Leave-One-Sym-Out | 留一只股票出来（交叉验证） |
| IIR | Infinite Impulse Response | 无限脉冲响应（滤波器） |
| NNLS | Non-Negative Least Squares | 非负最小二乘 |
| HL | Half-Life | 半衰期 |
| IQR | Interquartile Range | 四分位距 |
| OHLC | Open-High-Low-Close | 开高低收四价 |
| DeepLOB | Deep LOB CNN | 主办方 baseline 模型 |
| CNN | Convolutional Neural Network | 卷积神经网络 |
| RNN | Recurrent Neural Network | 循环神经网络 |
| TCN | Temporal Convolutional Network | 时序卷积网络 |
| KAN | Kolmogorov-Arnold Network | KA 网络 |
| PnL | Profit and Loss | 盈亏（评分指标） |
| intst | intensity | 订单事件到达强度（换手率 % 单位） |
| ind | indicator | 订单事件 0/1 指示（过去 1 tick 均值 > 过去 60 tick 均值 → 1） |
| acc | acceleration | 订单事件强度的单 tick 变化率（"到达加速度"） |
| cumspread | Cumulative Spread | 累积价差（10 档价差求和） |
| cancel_imb | Cancel Imbalance | 撤单不平衡 |
| dualz | Dual-window z-score | 双窗 z-score（短窗 z − 长窗 z） |
| qrank | Quantile Rank | 分位排名 |
| adapt_mom | Adaptive Momentum | 自适应动量（方向 × 强度） |
| spread_reg | Spread Regime | 价差体制（鲁棒 z-score on spread） |
| trade_pers | Trade Persistence | 交易持久性（相邻方向自相关） |
| mid_ewma_resid | Midprice EWMA Residual | 中间价 EWMA 残差 |
| signed_rv | Signed Realized Volatility | 方向化实现波动率 |
| signed_bv | Signed Bipower Variation | 方向化二次幂变差 |
| rskew | Return Skewness | log-return 偏度 |
| vol_burst | Volume Burst | 成交量突发倍数 |
| amt_burst | Amount Burst | 成交金额突发倍数 |
| rv_ratio | RV Ratio | 短/长窗 RV 比（活跃度变化率） |
| jshare | Jump Share | 跳跃份额 |
| roll_eff_spr_ratio | Roll Effective Spread Ratio | Roll 有效价差比 |
| ofi_tox | OFI Toxicity | 有毒订单流比例 |
| ewma_ofi | EWMA-OFI | 指数加权 OFI |
| liq_asym | Liquidity Asymmetry | 流动性不对称 |
| wmp_lvl | WMP Level-k | 第 k 档加权微价格 |
| bid_diff / ask_diff | Bid/Ask Level Spread | 同侧相邻档位价差 |
| bid_rate / ask_rate | Bid/Ask Price Rate | 第 k 档买/卖价滑动平均变化率（主办方提供） |
| bsize_rate / asize_rate | Bid/Ask Size Rate | 第 k 档买/卖挂量滑动平均变化率（主办方提供） |
| gofi | Generalized OFI | 广义订单流不平衡（含同价挂量净变化） |
| mlofi | Multi-Level OFI | 多档订单流不平衡 |
| kyle_inv | KyleInv | Kyle 反向冲击系数 |
| kyle_lam | Kyle λ | Kyle 价格冲击系数（lambda） |

### §0.2 数学符号

| 符号 | 定义 |
|---|---|
| $t$ | tick 索引（每 3 秒一个 tick） |
| $k$ | LOB 档位索引，$k \in \{1, 2, \ldots, 10\}$ |
| $W$ | 滑动窗口长度（tick 数），常用 $W \in \{5, 20, 50, 100\}$ |
| $h$ | 预测视距（horizon），常用 $h \in \{10, 40, 60\}$ tick |
| $\alpha$ | EWMA 衰减系数，$\alpha \in \{0.05, 0.1, 0.3, 0.5\}$，对应 HL ≈ 13 / 7 / 2 / 1 ticks（HL ≈ $\ln 2 / (-\ln(1-\alpha))$） |
| $b_t^{(k)}$ | bid_k：第 $k$ 档买盘报价（在 tick $t$） |
| $a_t^{(k)}$ | ask_k：第 $k$ 档卖盘报价 |
| $bs_t^{(k)}$ | bsize_k：第 $k$ 档买盘挂单量 |
| $as_t^{(k)}$ | asize_k：第 $k$ 档卖盘挂单量 |
| $mp_t$ | midprice：$(b_t^{(1)} + a_t^{(1)}) / 2$ |
| $\Delta mp_t$ | 未来 $h$-tick 中间价变化：$mp_{t+h} - mp_t$（回归目标的基础） |
| $\Delta amt_t$ | amount_delta：单 tick 成交金额变化 |
| $r_t$ | log-return：$\log(mp_t / mp_{t-1})$ |
| $\mu_W(\cdot)$ | 长度 $W$ 的滑窗均值 |
| $\sigma_W(\cdot)$ | 长度 $W$ 的滑窗标准差 |
| $\mathbf{1}[\cdot]$ | indicator 函数（条件成立为 1，否则为 0） |
| $\varepsilon$ | 分母小数兜底（防除零，典型 $\varepsilon = 10^{-8}$） |
| $\mathrm{sgn}(\cdot)$ | 符号函数 |
| $\mathrm{cov}(\cdot,\cdot)$ | 协方差 |
| $\mathrm{var}(\cdot)$ | 方差 |
| signed_dvol | 方向化 dollar volume：$\mathrm{sgn}(\Delta mp) \cdot \sqrt{|\Delta amt| + \varepsilon}$ |
| intst / ind / acc | 6 类订单事件的「到达强度 / 0-1 指示 / 单 tick 变化率（一阶）」 |
| y_reg | 回归目标（$\Delta mp$ 归一化后的标签） |
| y_cls | 3 分类标签（涨 = 2 / 平 = 1 / 跌 = 0） |
| $\hat{a}$ | 模型动作，$\hat{a} \in \{0, 1, 2\}$，分别表示「卖 / 不动 / 买」 |
| $f$ | 单边手续费率（约 $0.02\%$） |

### §0.3 概念一句话解释

- **OFI（Order Flow Imbalance，订单流不平衡）**：衡量 LOB 上 bid 侧净增量减去 ask 侧净增量。正值 = 买压增强 = 价格大概率上涨。
- **WMP（Weighted Mid-Price，加权微价格）**：用对侧 size 作权重去加权同侧价格。**挂单少的那一侧权重大**，反映「哪侧将先被耗尽」。
- **dualz（Dual-window z-score，双窗 z-score）**：短窗 z-score 减长窗 z-score，**消除全局基差**只保留「短期相对长期的偏离」。
- **KS-drop（基于 KS 检验的特征剔除）**：用 Kolmogorov-Smirnov 检验筛掉「单 sym 分布显著偏离全 sym 分布」的特征，保证 sym-agnostic。

### §0.4 论文引用

| 引用简称 | 完整引用 | 出处 |
|---|---|---|
| **Kercheval & Zhang 2015** | Kercheval & Zhang, "Modelling high-frequency limit order book dynamics with support vector machines" | *Quantitative Finance* 15(8) — bid/ask rate、GOFI 等 LOB 衍生特征的来源 |
| **Cont, Kukanov & Stoikov 2014** | Cont, Kukanov & Stoikov, "The Price Impact of Order Book Events" | *Journal of Financial Econometrics* — MLOFI 理论基础 |
| **Stoikov 2014** | Stoikov, "The Micro-Price" | SSRN Working Paper — WMP 定义 |
| **Kyle 1985** | Kyle, "Continuous Auctions and Insider Trading" | *Econometrica* — Kyle λ（价格冲击系数）|
| **BNS / Barndorff-Nielsen & Shephard 2004** | Barndorff-Nielsen & Shephard, "Power and Bipower Variation with Stochastic Volatility and Jumps" | *Journal of Financial Econometrics* — BV / jshare 跳跃检验 |
| **Roll 1984** | Roll, "A Simple Implicit Measure of the Effective Bid-Ask Spread in an Efficient Market" | *Journal of Finance* — Roll 有效价差估计 |
| **Easley, López de Prado & O'Hara 2012** | Easley, López de Prado & O'Hara, "Flow Toxicity and Liquidity in a High-Frequency World" | *Review of Financial Studies* — VPIN / OFI Toxicity 框架 |

---

## Q1：6 大家族的因子定义与计算公式

> 维度账：154 raw + 216 derived = 370 维，**KS drop 11 → 359 喂模型**。`SchemeP` 的 F1–F6 是 370 维的完备分组（互斥、并集 = 100%）。

### F1 LOB（Limit Order Book，限价订单簿）派生：94 raw + 11 派生 = 105 维 · LGB gain 40.4%

- **OHLC + 10 档报价 / 挂量（44 raw）**：open / high / low / close、$b_t^{(k)}$、$a_t^{(k)}$、$bs_t^{(k)}$、$as_t^{(k)}$，$k = 1, \ldots, 10$。
- **聚合（8 raw）**：
  - **avgbid**（Volume-Weighted Average Bid，量权买价均值）$= \sum_k b_t^{(k)} \cdot bs_t^{(k)} / \sum_k bs_t^{(k)}$；**avgask** 同理（卖侧）。
  - **totalbsize**（Total Bid Depth，总买盘深度 / 市场深度）$= \sum_{k=1}^{10} bs_t^{(k)}$；**totalasize** 同理。总挂单量越大 → **市场深度（market depth）**越充裕 → 流动性越好。
  - 主办方另给 4 个均值字段：**bid_mean / ask_mean**（10 档买/卖价等权均值 $= \frac{1}{10}\sum_k b_t^{(k)}$）、**bsize_mean / asize_mean**（10 档挂量等权均值）。与 avgbid/avgask 的区别：avgbid 是**量权**，bid_mean 是**等权**；ask_mean 存在约 3.9% NaN（深档偶无报价）。
- **midprice_k（10 raw）**：

$$
mp_t^{(k)} = \frac{b_t^{(k)} + a_t^{(k)}}{2}
$$

- **spread_k（10 raw）**：$\mathrm{spread}_t^{(k)} = a_t^{(k)} - b_t^{(k)}$，第 $k$ 档买卖价差。衍生两个聚合指标：
  - **cumspread**（Cumulative Spread，累积价差）$= \sum_{k=1}^{10} \mathrm{spread}_t^{(k)} = 10(\mathrm{ask\_mean}_t - \mathrm{bid\_mean}_t)$。**10 档价差累加，刻画整个价格梯队总宽度**（spread 越宽 → 流动性越差）。
  - **imbalance**（Order-Book Imbalance，订单簿不平衡；源自 **Kercheval & Zhang 2015**）$= (\mathrm{totalbsize}_t - \mathrm{totalasize}_t) / (\mathrm{totalbsize}_t + \mathrm{totalasize}_t) \in (-1, +1)$。官方定义「十档累计买卖量差（归一化）」：正值 → 买压强（价格倾向上行），负值 → 卖压强。
- **bid_diff_k / ask_diff_k**（Bid/Ask Level Spread，同侧相邻档位价差）**（20 raw）**：$b_t^{(k)} - b_t^{(k+1)}$（买侧，$k = 1, \ldots, 9$）及 $a_t^{(k+1)} - a_t^{(k)}$（卖侧）。**为什么有用**：相邻档位差越小 → LOB 密度越高 → 大单冲击成本低；价格梯队稀疏 → 流动性脆弱。深档 ask_diff8-10 因涨跌停日 ask 侧无报价存在 NaN（最高 ≈ 16.86%）；bid 侧 NaN = 0%。
- **WMP**（Weighted Mid-Price，加权微价格；Stoikov 2014, "The Micro-Price"）**派生 11 维（wmp_lvl1–10 + wmp_balance_12）**：

$$
\mathrm{wmp\_lvl}_t^{(k)} = \frac{a_t^{(k)} \cdot bs_t^{(k)} + b_t^{(k)} \cdot as_t^{(k)}}{bs_t^{(k)} + as_t^{(k)} + \varepsilon}
$$

> **重点**：ask 价权 bs，bid 价权 as——挂单少的那一侧权重更大（短缺侧把 mid 拉向它），符合「哪边将耗尽就先走哪边」的微观结构直觉。
> 另加 1 个 **wmp_balance_12**（WMP Level Balance，WMP 层间漂移）$= \mathrm{wmp\_lvl}_t^{(1)} - \mathrm{wmp\_lvl}_t^{(2)}$。正值（L1 WMP > L2 WMP）→ 短期下行压力：L1 ask 侧挂单将先被耗尽，价格向 L2 下移。

> **F1 NaN 摘要**：94 raw 字段（bid/ask/bsize/asize、OHLC、avgbid/totalbsize/bid_mean 等）公式层无兜底，NN 路径走 `nanmean/nanstd → nan_to_num(0) → clip(±10)` 三层；LGB 完全不动（内置 missing routing）。**深档（ask8–10 / asize8–10 / spread8–10 / ask_diff8–10）在涨跌停日 ask 侧无报价**，实测 train NaN ≈ 12.4–16.86%（ask10/asize10 最高）；其余 raw ≈ 0%。11 个 WMP 派生已在公式层用 `np.where(isfinite, x, 0)` 兜底，实测 NaN = 0%。

### F2 多尺度 OFI（Order Flow Imbalance，订单流不平衡）：0 raw + 50 派生 · gain 5.5%

100% 派生、家族内 $|r| = 0.33$ 全场最高（强冗余 → importance 被稀释）。

> **「多尺度」澄清**：这里「多尺度」**不是**固定术语，而是指 OFI 系列指标在多个时间窗口 $W \in \{5, 20, 60\}$ 上同时计算，覆盖短中期不同时间尺度的订单流信息。

- **MLOFI**（Multi-Level Order Flow Imbalance，多档订单流不平衡；**Cont, Kukanov & Stoikov 2014, "The Price Impact of Order Book Events", Journal of Financial Econometrics**）**30 维**：$W \in \{5, 20, 60\}$ × $k \in \{1, \ldots, 10\}$。核心单 tick 增量 $e_t^{(k)}$：

$$
\begin{aligned}
e_t^{(k)} = &\ \mathbf{1}[b_t^{(k)} \geq b_{t-1}^{(k)}] \cdot bs_t^{(k)} \\
&- \mathbf{1}[b_t^{(k)} \leq b_{t-1}^{(k)}] \cdot bs_{t-1}^{(k)} \\
&- \mathbf{1}[a_t^{(k)} \leq a_{t-1}^{(k)}] \cdot as_t^{(k)} \\
&+ \mathbf{1}[a_t^{(k)} \geq a_{t-1}^{(k)}] \cdot as_{t-1}^{(k)}
\end{aligned}
$$

$$
\mathrm{MLOFI}_{t, W}^{(k)} = \sum_{s = t - W + 1}^{t} e_s^{(k)}
$$

直觉：bid 上行（或不变）贡献 $+bs_t$；bid 下行扣前一档 $bs_{t-1}$（即被吃单 / 撤单消失的挂量），左右对称构成净 bid pressure。

- **EWMA-OFI**（Exponentially Weighted Moving Average of OFI，指数加权 OFI）**12 维**：$\alpha \in \{0.05, 0.1, 0.3, 0.5\}$ × $k \in \{1, 5, 10\}$：

$$
\mathrm{EWMA\_OFI}_t^{(\alpha, k)} = \alpha \cdot e_t^{(k)} + (1-\alpha) \cdot \mathrm{EWMA\_OFI}_{t-1}^{(\alpha, k)}
$$

**为什么比固定累加窗口好**：(i) 平滑（减少单 tick 极端值影响）；(ii) 自动指数衰减历史，近期权重高、久远权重低，无需指定截断 $W$；(iii) 无边界效应（不存在窗口截断造成的跳变）。
- **KyleInv**（Kyle Inverse Impact，Kyle 反向冲击系数）**2 维**，$W \in \{50, 100\}$：

$$
\mathrm{KyleInv}_{t, W} = \frac{\Delta amt_t}{\sqrt[3]{|\Delta amt_t|} \cdot \sigma_W(r) + \varepsilon}
$$

**分子** $\Delta amt_t$：单 tick 成交金额变化（带方向，正 = 净买入）。**分母** $\sqrt[3]{|\Delta amt_t|} \cdot \sigma_W(r)$：「活跃度 × 波动」的量纲归一化因子。**物理含义**：高波动期（$\sigma_W(r)$ 大）下单位金额能解释的方向变化弱 → 信号强度自适应缩小，**避免高波动期被随机大额成交误导**；$\sqrt[3]{}$ 比 $\sqrt{}$ 对大额 $\Delta amt$ 压缩更激进，跨 sym 量纲更稳。

- **Kyle λ**（Kyle's Lambda，Kyle 价格冲击系数；**Kyle 1985, "Continuous Auctions and Insider Trading", Econometrica**）**2 维**，$W \in \{50, 100\}$：

$$
\lambda_{t, W} = \frac{\mathrm{cov}_W(\mathrm{signed\_dvol},\ \Delta mp)}{\mathrm{var}_W(\mathrm{signed\_dvol}) + \varepsilon}
$$

**分子** $\mathrm{cov}_W(\mathrm{signed\_dvol}, \Delta mp)$：方向化 dollar volume（$= \mathrm{sgn}(\Delta mp) \cdot \sqrt{|\Delta amt| + \varepsilon}$）与未来 mid 变化的协方差；分子大 → 资金流确实能推动价格。**分母**：signed_dvol 自身方差（标准化系数）。**整族 KS drop** 原因：不同股票的资金—价格弹性量纲和量级根本不同（ETF vs 主板各异），训练 sym 上估计的 λ 分布与 OOD sym 强烈偏移（见 Q3）。

- **OFI Toxicity**（有毒订单流比例；基于 **Easley, López de Prado & O'Hara 2012, "Flow Toxicity and Liquidity in a High-Frequency World", Review of Financial Studies** VPIN 框架）**4 维**，$k \in \{1, 5\}$ × $W \in \{20, 50\}$：

$$
\mathrm{ofi\_tox}_{t}^{(k, W)} = \frac{1}{W}\sum_{s=t-W+1}^{t} \mathbf{1}\!\left[\mathrm{sgn}(e_s^{(k)}) \neq \mathrm{sgn}(\Delta mp_s)\right]
$$

即窗口内「OFI 方向与价格变化方向不一致」的占比。**高 toxicity → 订单流被信息驱动（知情者 / HFT 套利）→ 单向价格冲击，普通挂单无法对冲**。

> **F2 NaN 摘要**：100% 派生，公式层全部含 `+ε` 防除零 + `np.where(isfinite, x, 0)` 兜底；NN 路径 `nanmean/nanstd → nan_to_num → clip(±10)`；LGB 不动（实际无 NaN）；实测**所有 F2 特征 NaN = 0%**。

### F3 订单流强度：18 raw + 27 派生 = 45 维 · gain 35.1% — **Top-4 全在此**

主办方直接给的 6 类订单事件 × 3 变体：limit / market / cancel × buy / sell × intst / ind / acc。

**6 类事件命名对照**：

| 缩写 | 英文全称 | 中文 |
|---|---|---|
| lb | limit-bid | 限价买 |
| la | limit-ask | 限价卖 |
| mb | market-bid | 市价买 |
| ma | market-ask | 市价卖 |
| cb | cancel-bid | 撤单买 |
| ca | cancel-ask | 撤单卖 |

- **intst**（intensity，订单事件到达强度；官方：「从上个 tick 到当前 tick 的平均到达强度，量已转化为换手率 %」）**6 raw**。**模型 Top-4 全在此**（主要靠 intst）：`ma_intst` 0.945（市价卖）、`mb_intst` 0.860（市价买）、`la_intst` 0.65（限价卖）、`lb_intst` 0.567（限价买）。
- **ind**（indicator，订单事件 0/1 指示；官方：「过去 1 tick 平均强度 > 过去 60 tick 平均强度 → 1，否则 0」）**6 raw**：鲁棒二值兜底，抗量纲差异，树模型直接可用。
- **acc**（acceleration，订单事件强度变化率；官方：「从上个 tick 到当前 tick 的到达强度变化率，可理解为到达的加速度」）**6 raw**：**注意**：acc 是「上个 tick 到当前 tick 的一阶变化率」而非二阶差分。6 维 gain 全部 < 0.05，是死特征但保留（drop 有风险，不如让树模型自行忽略）。
- **EWMA-intst（24 派生）**：6 类 × 4 个 $\alpha$（IIR，Infinite Impulse Response 无限脉冲响应 滤波器）：

$$
\tilde{x}_t^{(\alpha)} = \alpha \cdot x_t + (1 - \alpha) \cdot \tilde{x}_{t-1}^{(\alpha)}
$$

- **cancel_imb**（Cancel Imbalance，撤单不平衡）**3 派生**，$W \in \{20, 50, 100\}$：

$$
\mathrm{cancel\_imb}_{t, W} = \frac{\sum_{s=t-W+1}^{t} cb_s^{\text{intst}}}{\sum_{s=t-W+1}^{t} (lb_s^{\text{intst}} + mb_s^{\text{intst}} + cb_s^{\text{intst}}) + \varepsilon} - \frac{\sum_{s=t-W+1}^{t} ca_s^{\text{intst}}}{\sum_{s=t-W+1}^{t} (la_s^{\text{intst}} + ma_s^{\text{intst}} + ca_s^{\text{intst}}) + \varepsilon}
$$

公式中所有变量为 **intst（到达强度）** 序列的窗口 $W$ 内累加。正值 → 买侧撤单比例高于卖侧 → 买盘在收缩 → 潜在下行压力；分母含 clip(−1, 1) 兜底。

> **F3 NaN 摘要**：18 raw（intst/ind/acc × 6）公式层无兜底，NN 路径走 wz 三层或 nan_to_num(0)；LGB 不动。实测 NaN ≈ 0%（主办方 cache 保证 finite）。24 个 EWMA-intst 派生含 `np.where(isfinite, x, 0)` 兜底；3 个 cancel_imb 含 `denom + EPS` + `clip(-1, 1)` 兜底；实测 NaN = 0%。

### F4 微结构波动：2 raw + 27 派生 = 29 维 · gain 7.5%

- **volume_delta**（Volume Delta，成交量变化）/ **amount_delta**（Amount Delta，成交金额变化）**（2 raw）**：均是 $t$ 与 $t-1$ 之间的**单 tick 差**，不是窗口累计。注：`amount_delta` 是唯一未归一化字段（单位：元，量级 $10^3 \sim 10^6$），需 sign-log1p 处理。
- **rv_W（RV，Realized Volatility 实现波动率）4 维**：

$$
\mathrm{RV}_{t, W} = \sum_{s = t - W + 1}^{t} r_s^2, \quad W \in \{5, 10, 20, 50\}
$$

- **signed_rv**（Signed Realized Volatility，方向化实现波动率）**3 维**，$W \in \{20, 50, 100\}$，$\in [-1, 1]$：

$$
\mathrm{signed\_rv}_{t, W} = \frac{\mathrm{RV}^+_W - \mathrm{RV}^-_W}{\mathrm{RV}^+_W + \mathrm{RV}^-_W + \varepsilon}
$$

其中：

$$
\mathrm{RV}^+_W = \sum_{s=t-W+1}^{t} r_s^2 \cdot \mathbf{1}[r_s > 0], \quad \mathrm{RV}^-_W = \sum_{s=t-W+1}^{t} r_s^2 \cdot \mathbf{1}[r_s < 0]
$$

signed_rv = +1 → 波动全由上涨贡献（涨主导），−1 → 全由下跌贡献，0 → 涨跌对称。

- **rskew**（Return Skewness，log-return 偏度）**3 维**，$W \in \{20, 50, 100\}$：

$$
\mathrm{rskew}_W = \mathrm{clip}\!\left(\frac{m_3(r)}{\sigma_W(r)^3 + \varepsilon},\ -10,\ +10\right)
$$

$m_3(r) = \frac{1}{W}\sum_s (r_s - \bar{r})^3$ 为三阶中心矩。**正偏（右偏）→ 大涨概率大于大跌；负偏 → 尾部下行风险更重**。
- **signed_bv**（Signed Bipower Variation，方向化二次幂变差；**BNS 2004**）**3 维**，$W \in \{20, 50, 100\}$：

$$
\mathrm{signed\_bv}_{t, W} = \mathrm{clip}\!\left(\frac{\frac{\pi}{2}\sum_{s=t-W+2}^{t} \mathrm{sgn}(r_s) \cdot |r_s| \cdot |r_{s-1}|}{\mathrm{RV}_W + \varepsilon},\ -10,\ +10\right)
$$

与 signed_rv 互补：**含 $|r_{s-1}|$ 项，单 tick 极端值（跳跃）不会单独放大比率** → 比 signed_rv 更抗噪，信号鲁棒。
- **vol_burst**（Volume Burst，成交量突发倍数）/ **amt_burst**（Amount Burst，成交金额突发倍数）**（4 维，$W \in \{20, 50\}$）**：

$$
\mathrm{vol\_burst}_{t, W} = \mathrm{clip}\!\left(\frac{v_t}{\mu_W(v) + \varepsilon},\ 0,\ 100\right), \quad \mathrm{amt\_burst}_{t, W} = \mathrm{clip}\!\left(\frac{|\Delta amt_t|}{|\mu_W(\Delta amt)| + \varepsilon},\ 0,\ 100\right)
$$

**物理含义**：当前 tick 活跃度是过去 $W$ tick 平均的几倍；> 1 = 活跃突增，< 1 = 平静期。clip(0, 100) 防极端值爆炸。

- **rv_ratio**（RV Ratio，短/长窗实现波动率比）**3 维**：

$$
\mathrm{rv\_ratio}_{W_n, W_d} = \mathrm{clip}\!\left(\frac{\mathrm{RV}_{W_n}}{\mathrm{RV}_{W_d} + \varepsilon},\ 0,\ 100\right), \quad (W_n, W_d) \in \{(5, 50),\ (20, 100),\ (50, 100)\}
$$

**短窗 / 长窗 RV 之比 → 局部活跃度变化率**：比值 > 1 → 最近 $W_n$ tick 波动率高于长期 baseline → 可能是新信息到来 / 订单流冲击。
- **jshare**（Jump Share，跳跃份额；**Barndorff-Nielsen & Shephard 2004, "Power and Bipower Variation with Stochastic Volatility and Jumps", Journal of Financial Econometrics**）**4 维**，$W \in \{20, 30, 50, 100\}$：

$$
\mathrm{BV}_{t, W} = \frac{\pi}{2} \sum_{s = t - W + 2}^{t} |r_s| \cdot |r_{s-1}|
$$

$$
\mathrm{jshare}_{t, W} = \mathrm{clip}\!\left(\frac{\mathrm{RV}_{t, W} - \mathrm{BV}_{t, W}}{\mathrm{RV}_{t, W} + \varepsilon},\ 0,\ 1\right)
$$

**BNS 跳跃不变量**：BV 是相邻两 tick 绝对 return 的乘积之和，**单 tick 极大值（跳跃）不会同时影响相邻两项** → BV 是「连续扩散部分的二次变差」的稳健估计。因此 $\mathrm{RV} - \mathrm{BV} \approx$ 跳跃贡献，jshare = 跳跃占总波动的比例 $\in [0, 1]$。

- **roll_eff_spr_ratio**（Roll Effective Spread Ratio，Roll 有效价差比；**Roll 1984, "A Simple Implicit Measure of the Effective Bid-Ask Spread in an Efficient Market", Journal of Finance**）**3 维**，$W \in \{30, 50, 100\}$：

$$
\mathrm{roll\_eff\_spr\_ratio}_{t, W} = \mathrm{clip}\!\left(\frac{2\sqrt{\max(0,\ -\mathrm{cov}_W(\Delta P_t,\ \Delta P_{t-1}))}}{\mathrm{spread}_t^{(1)} + 1},\ 0,\ 10\right)
$$

**分子** = Roll (1984) 经典有效价差估计：**负自协方差** → 买卖方向交替（bid-ask bounce）→ 价格在 mid 附近反弹 → 分子即有效价差。**分母** $\mathrm{spread}^{(1)} + 1$（+1 为 magic constant，防 spread=0 时除零）。ratio < 1 → 有效价差小于报价价差 → 市场流动性良好；**$W = 100$ 被 KS drop**（跨 sym Roll 估计不稳，见 Q3）。

> **F4 NaN 摘要**：2 raw（volume_delta / amount_delta）公式层无兜底，NN 路径 wz 三层；LGB 不动；实测 NaN ≈ 0%。所有 27 个派生字段含 `+ε` 防除零 + `np.where(isfinite, x, 0)` + clip 三重兜底；实测 NaN = 0%。

### F5 窗口统计：0 raw + 67 派生 · gain 5.2% — **OOD 主力 family**

100% 派生、家族内 $|r| = 0.07$ 全场最低（近正交）、与其他 5 族 $|r| \leq 0.07$。5.2% gain 是「无冗余 5.2%」，是跨股泛化的几何基础。

- **dualz_c**（Dual-window z-score，双窗 z-score）**37 维 — 决胜 OOD 的 trick**：对 **37 个 raw 字段**（spread1–5、bid_diff1–5、ask_diff1–5、bsize1–5、asize1–5 等共 37 列）分别做

$$
\mathrm{dualz}(c)_t = \frac{c_t - \mu_{20}(c)}{\sigma_{20}(c) + \varepsilon} - \frac{c_t - \mu_{100}(c)}{\sigma_{100}(c) + \varepsilon}
$$

直觉：当前 tick 相对短期均值偏离 **多于** 长期均值偏离的程度；**对均值水平不敏感**（两 z 相减消除全局基差）、对方差鲁棒（除以 $\sigma$）。

- **qrank_W100_c**（Quantile Rank，分位排名）**20 维**：

$$
\mathrm{qrank}_{W=100}(c)_t = \frac{1}{100}\sum_{s=t-99}^{t} \mathbf{1}[c_s \leq c_t] \in \left[\tfrac{1}{100},\ 1\right]
$$

当前值在过去 100 tick 的经验分位数。**注**：tick-quantized 列（如 spread1/5/10 大多等于 1 tick，cumspread 也是）qrank 几乎恒 = 1 → 死特征 → 被 KS drop（drop 11 中有 4 个来自 qrank_spread）。
- **mid_ewma_resid_a0.05**（Midprice EWMA Residual，中间价 EWMA 残差）**1 维**：

$$
\mathrm{resid}_t = mp_t - \mathrm{EWMA}_{\alpha=0.05}(mp)_t
$$

$\alpha = 0.05$ 对应半衰期 $\mathrm{HL} \approx 13$ ticks $\approx 40$ 秒（HL $= \ln 2 / (-\ln(1-\alpha))$，见 §0.2）。mid 减去其低通 EWMA → **高频偏离量**：正值 = mid 暂时高于趋势，负值 = 暂时低于趋势。clip 到 $[-0.05, +0.05]$。
- **adapt_mom**（Adaptive Momentum，自适应动量）**3 维**，$W \in \{20, 50, 100\}$：

$$
\mathrm{adapt\_mom}_W = \mathrm{clip}\!\left(\mathrm{sgn}\!\left(\sum_{s=t-W+1}^{t} r_s\right) \cdot \sqrt{\mathrm{RV}_W},\ -10,\ +10\right)
$$

**方向 × 强度**：符号由窗口内累计 return 方向决定，幅度由 $\sqrt{\mathrm{RV}_W}$ 决定。波动大时信号幅度更强；$\sqrt{\cdot}$ 比直接用 RV 量纲压缩更均匀。
- **spread_reg**（Spread Regime，价差体制 = 鲁棒 z-score on spread）**3 维**，$W \in \{20, 50, 100\}$：

$$
\mathrm{spread\_reg}_W = \mathrm{clip}\!\left(\frac{\mathrm{spread}_t^{(1)} - \mathrm{median}_W(\mathrm{spread}^{(1)})}{\mathrm{IQR}_W(\mathrm{spread}^{(1)}) + \varepsilon},\ -10,\ +10\right)
$$

**鲁棒 z-score**（对常数 spread 不爆：IQR = 0 时 $\approx$ 0，而非 inf）。刻画当前 spread 相对近期中位数偏高 / 低 → 流动性 regime 变化信号。
- **trade_pers**（Trade Persistence，交易持久性）**3 维**，$W \in \{20, 50, 100\}$：

$$
\mathrm{trade\_pers}_W = \mathrm{clip}\!\left(\mathrm{corr}_W\!\left(\mathrm{sgn}(\Delta mp_s),\ \mathrm{sgn}(\Delta mp_{s-1})\right),\ -1,\ +1\right)
$$

**相邻方向自相关**：> 0 = 趋势（涨继续涨、跌继续跌），< 0 = 反转（价格在 mid 附近震荡），≈ 0 = 随机游走。

> **F5 NaN 摘要**：100% 派生。dualz（37 维）含 EPS + `np.where(isfinite, x, 0)`；qrank（20 维）比较运算恒 finite；mid_ewma_resid / adapt_mom / spread_reg / trade_pers 均含 EPS + isfinite 兜底 + clip。NN 路径 wz 三层或 nan_to_num(0)；LGB 不动。实测**所有 F5 特征 NaN = 0%**。

### F6 不对称性：40 raw + 34 派生 = 74 维 · gain 6.3%

> **F6 家族定位**：「买卖两侧不对称」家族——对买侧（bid）和卖侧（ask）**分开建特征，不假设价格方向对称**。买侧行为（bid_rate、bsize_rate）和卖侧行为（ask_rate、asize_rate）往往非对称，天然可作为方向预测信号。

- **bid_rate_k / ask_rate_k**（Bid/Ask Price Rate，第 $k$ 档买/卖价滑动平均变化率）**（20 raw，$k = 1, \ldots, 10$）**：主办方直接提供（来源：**Kercheval & Zhang 2015**，赛题 PDF 定义「十档价格平均变化率」）。官方未给出精确公式，按论文命名约定推断为：

$$
\mathrm{bid\_rate}_t^{(k)} = \frac{\bar{b}_{[t-W:t]}^{(k)} - \bar{b}_{[t-2W:t-W]}^{(k)}}{\left|\bar{b}_{[t-2W:t-W]}^{(k)}\right| + \varepsilon}
$$

即**前 $W$ tick 均值**与**再前 $W$ tick 均值**的相对变化（具体 $W$ 未披露，经验 $\approx 10$–20 tick）；ask_rate 同理（卖侧）。

- **bsize_rate_k / asize_rate_k**（Bid/Ask Size Rate，第 $k$ 档买/卖挂量滑动平均变化率）**（20 raw）**：与 bid/ask_rate 结构相同，对象换为 $bs_t^{(k)}$（买侧挂量）/ $as_t^{(k)}$（卖侧挂量）。
- **GOFI**（Generalized OFI，广义订单流不平衡；**Kercheval & Zhang 2015**）**30 派生**：$W \in \{5, 20, 60\}$ × $k \in \{1, \ldots, 10\}$，单 tick 增量：

$$
\begin{aligned}
e_t^{(k)} = &\ \mathbf{1}[b\!\uparrow] \cdot bs_t - \mathbf{1}[b\!\downarrow] \cdot bs_{t-1} + \mathbf{1}[b\!=] \cdot (bs_t - bs_{t-1}) \\
&+ \mathbf{1}[a\!\uparrow] \cdot as_{t-1} - \mathbf{1}[a\!\downarrow] \cdot as_t - \mathbf{1}[a\!=] \cdot (as_t - as_{t-1})
\end{aligned}
$$

（这里所有 $bs$、$as$ 都指第 $k$ 档；$b\!\uparrow$ 等表示 bid 价格相对 $t-1$ 上行 / 下行 / 不变。）

与 MLOFI 的区别：MLOFI 严格按「价格上下行」分类，**GOFI 把「同价不变但挂量净变化」也算入流**——同价（$b\!=\,$）情况下用 $bs_t - bs_{t-1}$ 这种「净 delta」（即挂单量的纯粹变化，无论是撤单还是新挂）。这是 F2 ↔ F6 corr 高达 0.176 的原因（MLOFI / GOFI 同源）。

- **liq_asym_top5**（Liquidity Asymmetry top-5，前 5 档流动性不对称）**（4 派生，$W \in \{5, 20, 50, 100\}$）**：

$$
\mathrm{liq\_asym}_{t, W} = \log\!\left(\frac{1 + \sum_{k = 1}^{5} \sum_{s = t - W + 1}^{t} bs_s^{(k)}}{1 + \sum_{k = 1}^{5} \sum_{s = t - W + 1}^{t} as_s^{(k)}}\right)
$$

**top-5 物理含义**：「前 5 档」是最优 5 档挂单量（距成交最近的流动性）。如果买侧 5 档总量 > 卖侧 5 档总量 → log 正 → 买侧流动性更厚 → 大概率上涨（买压较难穿透）；反之 log 负 → 卖侧更厚 → 偏向下行。**$W = 5$ 被 KS drop**（$W = 5$ 太短，本质单 tick 比值，噪音太大）。

> **F6 NaN 摘要**：40 raw（bid/ask_rate / bsize/asize_rate）公式层无兜底，NN 路径 wz 三层；LGB 不动。**ask_rate / asize_rate 深档（k = 7–10）因涨跌停日 ask 侧无报价存在 NaN**，实测 ask_rate10 / asize_rate10 train NaN ≈ 17.13%，ask_rate1 ≈ 3.95%，bid_rate 系列 NaN = 0%（bid 侧无涨停影响）。30 个 GOFI 派生含 `np.where(isfinite, x, 0)` 兜底，实测 NaN = 0%；4 个 liq_asym 含 clip + isfinite，NaN = 0%。

### 三类关键预处理（PPT p7）

1. **window-z score（输入归一化）—— 单贡献 +22.88 PnL**

$$
\mathrm{wz}(x)_t = \frac{x_t - \mu_{100}(x_t)}{\sigma_{100}(x_t) + \varepsilon}
$$

100-tick 滚动窗口内的 z-score（trailing window，**不含未来**）。所有统计量在 100-tick 切片内自适应归一化，**跨股分布差异被窗口吸收 → 训练外股票也能 zero-shot**（见 Q2 跨股泛化证据）。

2. **买卖镜像数据增强 —— +2.41 PnL**

   每条样本 $(X, y_{\mathrm{reg}}, y_{\mathrm{cls}})$ 复制一份镜像：

   | 原始 | 镜像 |
   |---|---|
   | $X$ 中所有 bid\_\* 列 | 换成对应 ask\_\* 列（互换） |
   | $X$ 中所有 bsize\_\* 列 | 换成对应 asize\_\* 列（互换） |
   | $y_{\mathrm{reg}}$ | $-y_{\mathrm{reg}}$ |
   | $y_{\mathrm{cls}}$ | $2 - y_{\mathrm{cls}}$（涨↔跌，平保持） |

   训练集 ×2，**强制方向对称**：买侧 = 卖侧的镜像 → 模型对价格上行 / 下行一视同仁。

3. **稳健归一化 sign-log1p —— LGB 受益，NN 中性**

$$
\mathrm{sign\_log1p}(x) = \mathrm{sgn}(x) \cdot \log(1 + |x|)
$$

仅对 `amount_delta` 这种 $10^3 \sim 10^6$ 重尾字段（唯一未归一化字段）。**保号对数压重尾**：保留方向符号，量纲从百万压到 ~14–18 范围，对数变换抑制极端值对 LGB 分裂阈值的干扰。

---

## Q2：跨日 / 跨股 泛化性证据

### 设计层：硬约束 + 3 个 eval setting

**评测端 3 条硬约束**（PPT p3 红字）：

1. `date` 被置 0；
2. 测试点顺序被打乱；
3. `sym` ∈ 0–4 可能含训练外股票 → 必须 **sym-agnostic（与 sym 无关）**。

所有 370 个因子都满足：(i) 不显式用 `date` / `sym`；(ii) 每条样本独立可算（不依赖跨样本状态）；(iii) 不嵌入 sym embedding / sym-conditional 归一化。

**3 个 eval setting**（PPT p5）：

| Setting | 划分 | OOD 维度 |
|---|---|---|
| **Train-Val-Test** | `date` 0–79 / 80–95 / 96–119 单一切分 | 日期 OOD；test zero-tune 最终评分 |
| **LOSO-date** | K-fold by date（K = 5） | 日期 OOD；stock 全 in-sample |
| **LOSO-stock** | 5-fold by stock（LOSO = Leave-One-Sym-Out） | 股票 OOD；验证 sym-agnostic 假设 |

### 跨股泛化关键证据

1. **window-z 单 trick +22.88 PnL**（NN 路径 154 raw + L2 + window-z vs 154 raw + L2：+0.52 → +23.40）。LGB 分裂阈值是绝对值的；若 train 一只 sym spread $\sim 10^{-3}$、test OOD sym spread $\sim 3 \times 10^{-3}$，训练阈值在测试上全错位。window-z 把绝对值映射到无量纲 z，「分裂可比性」得以恢复。
2. **T7**（实验 SchemeD1：154 raw + L2 + window-z）**加 window-z 后**，sym = 2（异常蓝筹 / ETF）cum_pnl 从 $-0.98$ 翻到 $+2.24$ — 全 sym 翻正，证明 window-z 解决了 sym 间分布差异。
3. **T10**（实验 SchemeF：全局 z-score 替换窗口 z-score）**反例**：把窗口 z-score 换成全局 z-score（基于整训练集统计量）→ LOSO（Leave-One-Sym-Out）$h_{10}$ +13.40，**反而退化**。原因：全局统计量被训练 sym 主导，OOD sym 时 $(x - \mu_{\mathrm{global}}) / \sigma_{\mathrm{global}}$ 不再以 0 为中心。**窗口 > 全局 > per-sym（per-sym 直接违约束）**。
4. **F5 与所有其他族 $|r| < 0.07$（近正交）** — dualz / qrank 把「绝对水平」扔掉、留「相对位置」，这是 F5 提供独立 OOD 信号的几何解释；5.2% gain 是无冗余 5.2%。
5. **F1 + F3 块间 corr = 0.035** — 两个最重要的家族几乎正交，LGB 同时拿到两者获得「几乎无冗余」的高信息组合。
6. **公榜结果**：baseline DeepLOB $-6.65$ → 加 SchemeC +4.07 → +window-z & 回归 +19.23 → +SPO+ DFL +28.16 → +LGB 全量重训 +34.44 → ensemble +35.64。公榜本身就是 OOD（日期 80–119 之后的 sym 0–4 + 可能新股票），window-z 是其中两个核心泛化 trick 之一。
7. **私榜 #1**（$h = 60$ 时 +41.61、$h = 40$ 时 +38.65）：私榜与公榜不同日期、不同样本，整体性能甚至更高 — 说明因子工程的泛化没有「踩公榜运气」。

### 跨日泛化关键证据

1. **L1 → L7 progressive 5-seed**（按 family 累加加入派生，时间切分 Train 0–79 / Val 80–95 / Test 96–119）：
   - L1 154 raw → 17.7 ± 1.33
   - +F1（wmp 11）→ 17.5（≈ 0，wmp 是 raw bid / ask 的加权和，信息冗余）
   - +F2（OFI 48）→ 22.6（+5.1）
   - +F3（intst 27）→ 25.7（+3.1）
   - +F4（vol 32）→ 24.9（−0.8）
   - +F5（win 63）→ 26.3（+1.4）
   - +F6（asym 35）→ 27.3（+1.0）

   每一步加入 family 都用 **训练区完全没看过的 `date` 96–119** 评分，证明累加因子都不是 in-sample 过拟合。
2. **M7 全量重训**（train + val 全量 `date` 0–119 重训，迭代数 × 1.1）单 trick 给 LGB +4.3、给最终公榜 +5.51。能成立的前提是因子在跨日上稳定（否则全量重训只会 overfit 验证集）。
3. **mid_ewma_resid_a0.05** 和 **dualz** 等长窗 EWMA / z 因子刻意选 $W = 100$（≈ 5 min），覆盖跨日内的「分钟级 regime shift」但不跨日。

### 跨股 / 跨日 PSI（Population Stability Index 群体稳定性指数）实测

> verdict 阈值（按 PSI）：stable < 0.10 / mild_drift 0.10–0.25 / strong_drift > 0.25。

| family | n | 跨股 stable | 跨股 mild | 跨股 strong | 跨日 stable | 跨日 mild | 跨日 strong |
|---|---:|---:|---:|---:|---:|---:|---:|
| F1 | 105 | 1 | 35 | 69 | 25 | 44 | 36 |
| F2 | 50 | 0 | 3 | 47 | 50 | 0 | 0 |
| F3 | 45 | 10 | 6 | 29 | 41 | 4 | 0 |
| F4 | 29 | 1 | 18 | 10 | 22 | 7 | 0 |
| F5 | 67 | 22 | 9 | 36 | 64 | 3 | 0 |
| F6 | 74 | 1 | 11 | 62 | 67 | 0 | 7 |

**关键观察**：F5 跨日 64 / 67 stable + 跨股 22 / 67 stable，是所有 family 里跨股 stable 占比最高、跨日几乎全 stable 的族 → 这就是「F5 是 OOD 主力 family」的硬数据证据。

---

## Q3：KS-drop 11 features 的过程与效果

### §Q3.0 KS-2sample 检验的数学定义

**双样本 Kolmogorov-Smirnov 检验**：给定两组独立样本 $X = \{x_1, \ldots, x_n\}$（sym $i$ 上该特征的取值）和 $Y = \{y_1, \ldots, y_m\}$（sym $j$ 上该特征的取值），定义各自的**经验累积分布函数（empirical CDF）**：

$$
F_n(x) = \frac{1}{n}\sum_{k=1}^{n} \mathbf{1}[x_k \leq x], \quad G_m(x) = \frac{1}{m}\sum_{k=1}^{m} \mathbf{1}[y_k \leq x]
$$

**KS 统计量** $D_{n,m}$ = 两个 CDF 之差的最大绝对值（sup-norm 距离）：

$$
D_{n,m} = \sup_x \left|F_n(x) - G_m(x)\right| \in [0, 1]
$$

物理含义：
- $D = 0$ → 两组完全同分布
- $D = 1$ → 完全分离（一组所有值大于另一组最大值）
- $D > 0.5$ → 在某一点两个 CDF 至少差 50 个百分点 → 严重分布漂移

**理论 p-value**（基于 Kolmogorov 分布）：

$$
p = 2\sum_{i=1}^{\infty} (-1)^{i-1} \exp(-2 i^2 \lambda^2), \quad \lambda = \sqrt{\tfrac{nm}{n+m}}\,D_{n,m}
$$

**为什么本项目不用 p-value 而用 D 阈值**：N ≈ 1.4M（每只 sym ≈ 295K），$\lambda$ 巨大，即使 $D=0.001$ 也得 $p \to 0$ → 所有 feature 都「统计显著」→ p-value 完全失去判别力。改用 **effect size D 阈值**直接量化分布差异强度，这是工程上更稳健的做法。代码注释明确说明（`validate_sym_invariance.py:4-5`）：

> "max KS statistic across pairs. With 1.4M samples KS p-values are always tiny"

### §Q3.1 失效标准

**实际执行流程**（来自 `T68_stage5_features/validate_sym_invariance.py:22-38`）：

```python
for each feature column ci in 370:
    # 1. 按 sym 切分（5 只股票 → 5 组样本）
    arr_per_sym = [X[sym == s, ci] for s in {0, 1, 2, 3, 4}]
    arr_per_sym = [a[np.isfinite(a)] for a in arr_per_sym]    # 丢 NaN/Inf

    # 2. Pairwise KS：C(5, 2) = 10 对全部两两比
    stats = []
    for (i, j) in combinations(range(5), 2):
        x = subsample(arr_per_sym[i], 50_000)                  # 各采 50K 行加速
        y = subsample(arr_per_sym[j], 50_000)
        D_ij, p_ij = scipy.stats.ks_2samp(x, y)
        stats.append((i, j, D_ij))

    # 3. 取 10 对里"最差"的（worst KS = 最大 D）
    worst_D = max(D for (_, _, D) in stats)

    # 4. 三档判定
    if worst_D > 0.5:    flag = "FAIL"   # → drop
    elif worst_D > 0.1:  flag = "WARN"   # → 保留但警告
    else:                flag = "PASS"
```

> ⚠️ **关键纠正**：之前 `ANSWERS_P3_FEATURES.md` 里写的「对每个 sym 单独看该特征在该 sym 上的分布 vs 全局训练分布做 KS test，$p < \alpha$」**两处不严格**：
> 1. 不是「sym vs 全局」，而是 **5 只 sym pairwise 10 对两两比**，取最差 $D$
> 2. 不是「$p < \alpha$」，而是「**worst pairwise $D > 0.5$**」（$N$ 太大时 p-value 永远显著）

### §Q3.2 11 个被 drop 的特征

| # | 特征名 | 族 | worst $D$ | stage | 失效原因（数据 + 理论） |
|---|---|---|---:|---|---|
| 1 | `dualz_ask_diff1` | F5 | 0.5011 | stage1 | ask 侧 1-level diff 跨 sym CDF 差约 50%（sym 间深档报价行为差异显著） |
| 2 | `dualz_bid_diff5` | F5 | 0.5029 | stage1 | 5-level bid 价差跨 sym 漂 |
| 3 | `dualz_ask_diff5` | F5 | 0.5019 | stage1 | 同上，ask 侧 5-level |
| 4 | `qrank_W100_spread1` | F5 | **0.8564** | stage2 | spread1 多为 1 tick → qrank 几乎恒 1 → 跨 sym CDF 极端分离 |
| 5 | `qrank_W100_spread5` | F5 | **0.8484** | stage2 | spread5 tick-quantized 同因 |
| 6 | `qrank_W100_spread10` | F5 | **0.8374** | stage2 | 同上 |
| 7 | `qrank_W100_cumspread` | F5 | **0.8549** | stage2 | 累积 spread 同因 |
| 8 | `kyle_lam_W50` | F2 | **0.9503** | stage2 | **接近完全分离！** Kyle $\lambda$ 定义为资金流与 $\Delta\mathrm{mid}$ 的协方差除以资金流方差，跨股资金—价格弹性根本不同 → 整族 KS fail |
| 9 | `kyle_lam_W100` | F2 | **0.9758** | stage2 | 同上，更长窗更严重 |
| 10 | `roll_eff_spr_ratio_W100` | F4 | 0.5348 | stage3 | $W=100$ 太长，跨 sym Roll 估计在不同活跃度股上偏差大 |
| 11 | `liq_asym_top5_W5` | F6 | 0.5161 | stage5 | $W=5$ 太短，单 tick top-5 比值噪声 + 跨 sym 流动性结构异 |

数据来源：`experiments/T59_fullsym_train/sym_invariance_report.json`（10 个）+ `experiments/T68_stage5_features/sym_invariance_report.json`（1 个 STAGE5）。

**额外洞察**：Kyle $\lambda$ 两档 worst $D$ 高达 0.95–0.98（CDF 几乎完全分离）→ 这是 Kyle $\lambda$ 整族被 drop 最硬的证据；qrank-on-spread 4 个 0.83–0.86 → 死特征 + tick 量化双重失败；3 个 `dualz_*_diff` 都贴 0.50 阈值 → 边界 fail。

### §Q3.3 Design Debt

- **阈值 0.5 是经验值，没做 sensitivity sweep**：试过 0.3 / 0.7 名单怎么变？没做。
- **subsample 固定 50K + seed=0**：换 seed 名单稳定性如何？没测。
- **没做 Bonferroni 多重检验修正**：370 列 × 10 对 = 3700 次检验。但**这其实不重要**，因为最终用的是 D 阈值（effect size）而非 p-value test → Bonferroni / FWER 控制的是 p-value 的 false discovery，对 effect size 不适用。
- **WARN 档（28 个，0.1 < D ≤ 0.5）选择保留而不 drop**：保留理由是 D 没到 0.5 → 跨 sym 漂移有限，模型 + window-z 可以兜底；drop 这些会减少模型能用的信号维度（28 个 feature 不少）。**没做 ablation 验证这个选择**——是否 drop WARN 也能保 std 不变？未知。
- **对称性 anomaly**：`dualz_bid_diff1` 保留（$D=0.456$ WARN）但 `dualz_ask_diff1` drop（$D=0.5011$ FAIL）——bid 侧勉强卡在 WARN 上限，ask 侧勉强越过 FAIL 下限。A 股流动性买卖侧本就不对称（涨跌停日 ask 侧深档无报价），ask 侧分布差异天然大。**这不是 design bug，是阈值 0.5 卡边界时的自然结果**。

### §Q3.4 效果（5-seed 直接对比，Train-Val-Test setting，test = `date` 96–119）

| Backbone | 370-d no drop | 359-d drop 11 | Δ mean | Δ std |
|---|---:|---:|---:|---:|
| LGB L2 | 27.28 ± 2.34 | 26.58 ± **0.96** | −0.70 | std **−59%** |
| NN L2 + window-z | 34.77 ± 1.39 | 34.34 ± **0.50** | −0.43 | std **−64%** |

**关键解读**：drop 11 牺牲约 0.5 个 mean PnL，但 **seed-to-seed std 砍掉 60%**。

- seed std 衡量「模型对训练随机性的敏感度」。
- OOD 上 std 高的模型在公榜 / 私榜会「运气依赖」，单次提交可能高也可能低。
- drop 11 后的 359-d 模型才是 50-seed ensemble + 公榜部署的 **稳定基底**。
- 一页纸消融表 row 9 → row 10 写的「drop 11 → −1.57」是消融表链式差值（NN 路径 +F6 累加后再 drop 11），与 5-seed 直比的 −0.43 方向一致、数值不同。

**Why drop 让 std 大降**：被 drop 的 11 个都是「在某些 sym 上几乎是常数 / 死特征 / 极噪声」的列。LGB 偶尔会用这种「死列」做分裂、形成 spurious 路径；不同 seed 下走的 spurious 路径不同 → seed 间分歧大。drop 后强制模型只能用「跨 sym 同分布」的列，决策面更稳。

---

## Q4：为什么 h=60 为主线？— horizon 选择的 5 层论证

> 评分规则 $\mathrm{Score} = \max_{h \in \{5, 10, 20, 40, 60\}} \sum_i \mathrm{pnl}^{(h)}_i$（5 个 horizon 取最大），看上去给了 5 个选择，实际上 **只有 h=60 在 OOD 上最终胜出**。下面 5 层论证为什么。

### Q4.1 评分公式 + 实测：短 h 在 OOD 上完全失效

回顾单笔 PnL 公式：

$$
\mathrm{pnl}_i^{(h)} = (\hat{a}_i - 1) \cdot \frac{\Delta mp^{(h)}}{mp_{t+1}} - f \cdot |\hat{a}_i - 1| \cdot \frac{mp_{t+h+1} + mp_{t+1}}{mp_{t+1}}
$$

其中 $f \approx 2 \times 10^{-4}$（双边手续费 0.02%），$\Delta mp^{(h)} = mp_{t+h} - mp_t$。

iter_002 SchemeC 阶段一次直接对比（同样的 LGB 3-class CE + 多 horizon 训练 + 高置信度阈值 gate）：

| horizon | LOSO 本地 | 公榜 OOD | gap (透传 $\Delta$) |
|---:|---:|---:|---:|
| $h = 10$ | **+21.86** | **−8.64** | **−30.50（灾难）** |
| $h = 60$ | +6.30 | +4.07 | −2.23（可接受） |

**短 h 的 LOSO 看上去好得多（+21.86 vs +6.30），但在 OOD 上立刻原形毕露（−8.64）**。从此后所有调参 / ensemble / 决策焦点全部以 h=60 为主线。

> ⚠️ 注：以上是 iter_002 早期 LOSO 方法学对比。更严谨的 5 horizon × 5 seed 实验（IC 分解、PnL 分解）详见 Q4.3b / Q4.2 / Q4.8，揭示了短 h 失效的**真实机制**（非 IC 低，而是 σ(y_h) 太小撑不过手续费）。

### Q4.2 信号 / 成本权衡的数学

- **成本项**：$f \cdot |\hat{a}_i - 1| \cdot (mp_{t+h+1} + mp_{t+1})/mp_{t+1} \approx 2f \approx 4 \,\mathrm{bp}$，**与 h 几乎无关**
- **信号项**：在中间价近似随机游走的假设下 $\mathrm{std}(\Delta mp^{(h)}) \propto \sigma \cdot \sqrt{h}$；含 alpha 时期望 $\mathbb{E}|\Delta mp^{(h)}| \propto \sigma \cdot \sqrt{h} \cdot \mathrm{IC}$
- **信号 / 成本比** $\propto \sqrt{h}$

$\sqrt{60/5} \approx 3.46$ — **h=60 的信噪比是 h=5 的 3.5 倍**。

**实测 IC × σ vs 成本分解**（5 horizon × 5 seed LGB，test 集，来源：`h_horizon_study/REPORT.md §D`）：

| h | IC_test | $\sigma(y_\mathrm{test})$ | $\mathrm{IC} \times \sigma$ | 单边成本 | net signal/笔 |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.372 | $5.33\times10^{-4}$ | **2.0 bp** | 2.0 bp | **≈ 0 bp** |
| 10 | 0.320 | $7.62\times10^{-4}$ | 2.4 bp | 2.0 bp | +0.4 bp |
| 20 | 0.250 | $1.10\times10^{-3}$ | 2.7 bp | 2.0 bp | +0.7 bp |
| 40 | 0.184 | $1.55\times10^{-3}$ | 2.9 bp | 2.0 bp | +0.9 bp |
| 60 | **0.144** | **$1.90\times10^{-3}$** | **2.7 bp** | 2.0 bp | **+0.7 bp** |

**这才是 h=60 主线的真正原因**：不是 IC 最高（实测反过来：IC 随 h 单调递减），而是 $\mathrm{IC} \times \sigma$ 在 h=60 撑得过 4 bp（round-trip）成本，h=5 时刚好持平。

### Q4.3 微结构噪声（bid-ask bounce）——修正版

高频中间价存在 **bid-ask bounce noise**：成交在 best bid（$b^{(1)}$）和 best ask（$a^{(1)}$）之间来回切换，单 tick Δmid 有 ±tick 的虚假波动，不反映真实方向。

**实测数字**（全局 1-lag 自相关，来源：`h_horizon_study/REPORT.md §B`）：

- 全局 $\rho = -0.150$（负值 = bounce 信号）
- sym3 $\rho = -0.23$（最严重），sym2 $\rho = -0.05$（最轻）

> ⚠️ **反直觉发现（旧说法已被实测推翻）**：原先答辩小抄 Q4.3 声称「短 h 模型 in-sample 拟合 bounce → OOD 失效 → IC 低」——**实测正好相反**：
>
> - **短 h IC 反而更高**：IC_test(h=5) = 0.372 >> IC_test(h=60) = 0.144
> - **短 h train-test gap 更小**：h=5 gap = 0.086（最稳），h=60 gap = 0.428（严重 overfit）
> - 短 h 模型预测力更强，bounce 不是「让短 h 不可学」，而是让 $\sigma(y_h)$ 比 $\sqrt{h}$ 基准略小（负自相关压缩了短 h 的方差），进一步收窄信号空间。

Bounce 的实际作用：负自相关（bounce share ≈ 22%）部分抵消 Δmid 方差，使 $\sigma(y_5) \approx 5.3\times10^{-4}$ 略低于纯 $\sqrt{h}$ 预测，让 h=5 信号更难越过手续费门槛——但模型对短 h 的预测力并未因 bounce 而失效。

### Q4.3b IC by Horizon 完整实测（5 horizon × 5 seed）

来源：`h_horizon_study/REPORT.md §C`，5 horizon × 5 seed LGB L2，359-d schemeP，train 0–79 / val 80–95 / test 96–119

| h | IC(train) | IC(test) | train-test gap |
|---:|---:|---:|---:|
| 5 | 0.458 ± 0.020 | **0.372 ± 0.003** | 0.086（最稳） |
| 10 | 0.447 ± 0.034 | 0.320 ± 0.003 | 0.126 |
| 20 | 0.460 ± 0.056 | 0.250 ± 0.003 | 0.210 |
| 40 | 0.516 ± 0.076 | 0.184 ± 0.004 | 0.332 |
| 60 | 0.572 ± 0.080 | **0.144 ± 0.006** | **0.428（严重 overfit）** |

关键观察：
1. IC(test) 随 h **单调递减**（短 h 模型逐 tick 预测力更强）
2. h=60 in-sample IC = 0.572 但 OOD 暴跌到 0.144，train-test gap 最大——h=60 自身也严重 overfit，仅靠 $\sigma(y_{60})$ 足够大才让净信号仍为正
3. **短 h 不是 IC 低的问题，是 $\sigma(y_h) \propto \sqrt{h}$ 决定的成本可行性问题**

### Q4.4 LOB 派生因子的时间尺度匹配

我们的 feature 是 **$W \in \{5, 20, 50, 100\}$ ticks 的窗口统计**（dualz / rv_W / cancel_imb_W 等），最长记忆 $W = 100$ ≈ 5 min。

- $h = 5$（15 秒）：远短于 $W = 100$ → 「用 5 分钟特征预测 15 秒事件」 → 特征-标的尺度严重失配
- $h = 60$（180 秒）：≈ $W = 100$ × 60% → **特征信息正好积累足够、又没过期** — 信号-标的尺度的 sweet spot
- $h = 120$（如果有）：远长于 $W = 100$ → 特征对未来 6 分钟的预测力衰减

### Q4.5 OOD 透传率：长 h 模型对分布漂移更鲁棒

| 节点 | LOSO 本地 h=60 | 公榜 h=60 | 透传率 |
|---|---:|---:|---:|
| iter_002 SchemeC | +6.30 | +4.07 | 65% |
| iter_013 回归突破 | +36.23 | +19.23 | 53% |
| iter_015 SPO+ DFL | +40.09 | +28.16 | 71% |
| iter_019 M7 LGB | +41.49 | +34.44 | 83% |

**h=60 的本地 → 公榜透传率随训练成熟稳定在 50-80%**，且随着模型 / 集成提升单调改善。而短 h 透传率为负或近零（iter_002 h=10 直接 −8.64）。

### Q4.6 一句话答辩答案

> **「h=60 不是因为 IC 最高（实测反过来：h=5 IC=0.37 > h=60 IC=0.14），是因为 $\sigma(y_h) \propto \sqrt{h}$，短 h 的目标变量幅度太小，$\mathrm{IC} \times \sigma \approx 2\,\mathrm{bp}$ 刚好等于单边手续费，net PnL ≈ 0；h=60 的 $\sigma$ 是 h=5 的 3.7×（实测 $1.9\times10^{-3}$ vs $5.3\times10^{-4}$），即使 IC 低，$\mathrm{IC} \times \sigma \approx 2.7\,\mathrm{bp}$ >> 成本，net PnL ≈ +0.7 bp。这不是预测能力问题，是信号尺度问题。」**

### Q4.7 h<60 实战：$\sqrt{h/60}$ 阈值缩放**实测失败**，h<60 必须单独训练

> ⚠️ **旧说法（已被实测推翻）**：原答辩小抄 Q4.7 称「按 $\sqrt{h/60}$ 缩放阈值 → 私榜 h=40 +38.65 #1，说明 h=60 模型有跨 horizon 迁移力」。实测发现：

| h | $\mathrm{scale}\,\sqrt{h/60}$ | h=60 模型缩放 PnL | 专训模型 PnL | $\Delta$ |
|---:|---:|---:|---:|---:|
| 5 | 0.289 | **−32.2（破产！）** | +18.6 | +50.8 |
| 10 | 0.408 | **−14.3** | +25.5 | +39.8 |
| 20 | 0.577 | +4.4 | +29.7 | +25.2 |
| 40 | 0.816 | +19.5 | +30.1 | +10.6 |
| 60 | 1.000 | +22.9 | +22.9 | 0 |

> 数据来源：`h_horizon_study/REPORT.md §F`

**真相**：用 h=60 模型 + $\sqrt{h/60}$ 阈值缩放推到 h < 20 完全破产（h=5 PnL = −32！），与专训该 h 差 +51。所以**实战中 h < 60 必须单独训练，不能简单 $\sqrt{h/60}$ 阈值缩放**。

私榜 h=40 +38.65（#1）实际是因为我们对 h=40 做过单独阈值调优（非简单 $\sqrt{}$ 缩放）。

### Q4.8 实测 PnL 对比：h=40 in-sample 最优，h=60 仍是 OOD 主线

DE-优化阈值后 test 集 PnL（5 seed 平均）：

| h | $\theta_\mathrm{up}$ | $\theta_\mathrm{dn}$ | n_trades | test PnL | per-trade |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.00033 | 0.00034 | 92,500 | +18.6 | $+2.01\times10^{-4}$ |
| 10 | 0.00038 | 0.00032 | 130,691 | +25.5 | $+1.95\times10^{-4}$ |
| 20 | 0.00036 | 0.00031 | 178,081 | +29.7 | $+1.67\times10^{-4}$ |
| **40** | **0.00046** | **0.00029** | **193,119** | **+30.1** | $+1.56\times10^{-4}$ |
| 60 | 0.00063 | 0.00030 | 177,480 | +22.9 | $+1.29\times10^{-4}$ |

> 数据来源：`h_horizon_study/REPORT.md §D`

**In-sample 最优是 h=40（+30.1），不是 h=60（+22.9）**——解释了为何私榜 h=40 +38.65（#1）与 h=60 +41.61（#1）几乎打平。但 h=60 OOD 透传更稳（私榜 #1），实战仍主押 h=60，h=40 兜底。

### Q4 答辩 talking point flow

> **答辩建议流程**（展示顺序 → 对应图/表）：
>
> 1. 「我们测试了 5 个时间尺度，发现一个反直觉结果」— 展示 `fig_C_IC_by_horizon.pdf`
> 2. 「短 h 模型 IC 反而更高（h=5 IC=0.37 vs h=60 IC=0.14），train-test gap 更小，不是 IC 低的问题」
> 3. 「真正原因」— 展示 Q4.2 新表（IC × σ vs cost）
> 4. 「$\sigma(y_h) \propto \sqrt{h}$（实测 $h^{0.5007}$），短 h $\sigma$ 太小，net signal ≈ 0」
> 5. 「h=40 in-sample 最优（+30.1），h=60 OOD 最稳（公榜 +35.64 / 私榜 +41.61 #1）」
> 6. 「$\sqrt{h/60}$ 阈值缩放对 h < 20 破产，h < 60 必须单独训练」

---

## 老师可能追问的 7 个刁钻问题（含答案）

**Q1：为什么 F1 importance 40% 但 progressive 增益 ≈ 0？**

> F1 里 94 / 105 维是 raw bid / ask / size / OHLC，已在 L1 baseline。F1 派生只剩 11 个 wmp_*，wmp 是 raw bid / ask 加权和、信息高度冗余。importance 反映的是「模型用了谁」，被 raw 价格 baseline 主导；progressive 反映的是「相对 baseline 新增了什么」，wmp 没新增信息。

**Q2：F2 importance 仅 5.5% 但 progressive 加入 +5.08 是最大跳变，为啥？**

> F2 内部 $|r| = 0.33$ 全场最高（50 个 MLOFI 因子高度相关）。LGB 的 gain 在高相关族内被分摊到很多 feature 上，单看 share 被稀释；但作为一个整体，OFI 提供了 baseline 完全没有的「订单流方向 × 时间尺度」维度，相对 baseline 跳变最大。

**Q3：为什么 ma_intst（市价卖单强度）是单因子 gain 第一？它和 OFI 不重复吗？**

> ma_intst = market-ask intst = 已脱敏的市价卖单总量 / 次数，**直接来自主办方**，不需要从价格反推 OFI 那种「逆向工程」。OFI 是从 $(b, a, bs, as)$ 的变化间接推订单流，会有「挂单跳价」的歧义；ma_intst 没歧义。LGB 优先用最低歧义的信号当根分裂，所以 gain 最高。

**Q4：为什么 5 个 OFI / Kyle 都用 $W \in \{5, 20, 60\}$ 或 $\{50, 100\}$，没做窗口 sweep？**

> 这是 R34 调研阶段的固定选择，没正式 sweep（design debt）。理由：$(20, 50, 100)$ ticks ≈ $(1, 2.5, 5)$ min 是 HFT（High-Frequency Trading，高频交易）文献的常用尺度；窗口 sweep 一次要重算 1.47M 行 feature cache，成本高，没做。Roll $W = 30$、jshare $W = 30$ 等「独有窗口」是各 stage 提议时的遗留，没归一化掉。

**Q5：Kyle λ 整族 KS drop 了，为什么不直接整族删除算了？**

> 我们就是这么做的（$W = 50$ 和 $W = 100$ 两个都 drop）。但 family report 里 F2 50 维仍包含 Kyle λ — 这是 dimension 账（370 维分组），不是「模型用了多少」。喂模型时是 359 维（drop 11）。Kyle λ 的位置由 KyleInv 顶上了（KyleInv 通过 $\sqrt[3]{\cdot}$ 重整 $\Delta amt$ → 跨 sym 量纲更稳）。

**Q6：window-z 用 100-tick（≈ 5 min）会不会泄露未来？**

> 不会。window-z 是 **trailing window**：对当前 tick $t$，用的统计量是 $[t-99, t]$ 全部历史 tick；不含 $t+1$ 及之后。两个关键代码位置：
>
> - `cfg["wz"]: mu = np.nanmean(tr["X"], axis=0)` 在 train 上算全局 mu / sd 套到 val / test，**保证三集同口径**，避免 train / val / test 分布偏移。
> - dualz 算法里 seg20 / seg100 都从 `X3d[:, -W:, :]` 取，永远是过去 $W$ tick。

**Q7：KS test drop 11 是不是会 overfit 训练集的 sym 0–4？OOD 新 sym 上还有效吗？**

> 风险确实存在（KS 用训练 sym 算分布）。我们的兜底是：(i) drop 的列要么 in-sample 就近乎死特征（qrank on quantized spread）、要么 $W$ 太短 / 长极噪声（liq_asym $W = 5$、roll $W = 100$），这些「病态特征」在任何 sym 上都不该贡献信号；(ii) 公榜 / 私榜两套不同 OOD test 上 359-d 模型都拿到 #1，证明 drop 列表对训练外 sym 也鲁棒。

---

## 一句话总结

> 6 大家族（370 维）+ 3 类 sym-agnostic 预处理（window-z / 镜像 / sign-log1p）+ KS drop 11（359 维稳定基底）+ 50×50 异质 ensemble，把跨股 + 跨日 OOD 风险层层吸收，最终公榜 +35.64（baseline $-6.65$）/ 私榜 #1（$+41.61$）。
