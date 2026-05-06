# Kaggle / 比赛社区的特征工程与建模 trick

> 整理自实际 1st place writeup 与社区博客；着重提炼**可迁移到我们良文杯**的做法。

## 1. Optiver Realized Volatility Prediction (2021)

- **任务**：用过去 10 分钟的 LOB+trade 数据预测下一个 10 分钟的 realized volatility（RMSPE）。
- **赛题特点**：完全匿名 stock_id 和 time_id，time_id 顺序是打乱的；3-second snapshot。
- **1st place: "Nearest Neighbors"** ([writeup](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/274970))
  - **核心 trick**：用 **price tick size 反推 stock 真实价格**，从而 reorder time_ids（赛方认为不可恢复的信息），让所有股票共享一个时间轴 → 可以做 lag features。
  - 用 **kNN over time_ids**（用 stocks 之间在某个 time_id 上的特征向量找最近邻）做 augmentation 与 ensembling。
  - 模型：LightGBM + MLP / Transformer ensemble。

### 关键特征公式（社区共识）

| 特征 | 公式 | 出处 |
|---|---|---|
| WAP | `(BidPrice₁·AskSize₁ + AskPrice₁·BidSize₁) / (BidSize₁ + AskSize₁)` | Optiver 官方 starter |
| Log return | `log(WAP_t / WAP_{t-1})` | 标准 |
| Realized volatility | `√Σ r²_t` over window | Wikipedia |
| WAP balance | `WAP1 - WAP2`（两档 WAP 的差） | EDA notebook |
| Bid-ask spread | `ask - bid` | 标准 |

### 衍生 / 强特征
- **realized_vol per minute**: 把 10 分钟切成 10 个 1-min bucket，每个算 RV → 10 维向量。然后 EWMA decay 加权聚合。
- **time_id-level cross-sectional**: pivot data by (time_id, stock_id)，再 KMeans cluster stocks，对每个 time_id 在每个 cluster 内做 mean aggregation。
- **5-min vs 10-min ratio**: `realized_vol_5 / realized_vol_10` 表征市场动量趋势变化。

### CV 策略
- **GroupKFold by time_id**：因为 time_id 是单位（每个 sample 是一个 time_id × stock_id），所以 CV split 必须按 time_id 分组防止泄漏。

---

## 2. Optiver Trading at the Close (2023)

- **任务**：Nasdaq 收盘前 10 分钟集合竞价，预测每只股票相对合成 index 的 60-second-ahead return。
- **1st place: HYD** ([writeup](https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/hyd-1st-place-solution))
- **核心结论**："Feature is all you need"——比 model 更重要。
- **模型**: LightGBM Robust Single Model + LightGBM × MLP Ensemble。
- **关键 trick**:
  1. **Triplet imbalance**：对任意 3 个价格 (a, b, c)，sort 出 max/mid/min，算 `(max - mid)/(mid - min)`。这是非线性的"三价相对位置"特征。
  2. **Pairwise price imbalance**：对所有价格对 `(p_a, p_b)` 算 `(p_a - p_b)/(p_a + p_b)`。
  3. **Stock-level global features**: median size, std price, price range（cached over last 21 days）→ 给每只股票一个"个性向量"。
  4. **Synthetic index**: `index_wap = Σ_i stock_weight_i · weighted_wap_i`，再做相对 deviation。
  5. **Time features**: day-of-week (`date_id % 5`), seconds (`seconds_in_bucket % 60`), minute (`seconds_in_bucket // 60`)。
  6. **Rolling features**: shift 和 pct_change over windows {1,2,3,5,10}。

### Trading at Close 的 V1/V2/V3 特征 layer（来自 fan2goa1 复盘）

完整公式 list 见 feature_ideas.md。要点：
- **Volume = ask_size + bid_size**
- **Liquidity Imbalance = (bid_size - ask_size) / (bid_size + ask_size)**
- **Matched Imbalance = (imbalance_size - matched_size) / (matched_size + imbalance_size)**
- **Micro Price = (bid_price · ask_size + ask_price · bid_size) / (bid_size + ask_size)**
- **Price Pressure = imbalance_size · (ask_price - bid_price)**
- **Market Urgency = price_spread · liquidity_imbalance**
- **Depth Pressure = (ask_size - bid_size) · (far_price - near_price)**
- **Spread-Depth Ratio = price_spread / (bid_size + ask_size)**
- **Relative Spread = price_spread / wap**

---

## 3. Jane Street Market Prediction (2020-2021)

- **任务**：500 个 anonymized features + 5 个 resp targets，每个 sample 决定是否交易（utility 函数评分，不是 acc）。
- **1st place (Yirun)**: ["Training Supervised Autoencoder with MLP"](https://www.kaggle.com/competitions/jane-street-market-prediction/writeups/cats-trading-yirun-s-solution-1st-place-training-s)

### 核心做法
1. **Supervised Autoencoder + MLP**：
   - Autoencoder（Encoder→bottleneck→Decoder）训练 reconstruction，**同时**带一个 MLP head 做监督，joint loss。
   - 锁住 encoder 后，把它当作"feature extractor"放在 MLP 前，concat encoder output + raw input 喂给 MLP。
   - **bottleneck features** 是非常 robust 的低维表示。
2. **Multi-target learning**：同时预测 resp, resp_1..4 五个 horizon。
3. **CV**: 3-fold time-grouped with **10-day embargo gap**（防止 label leakage）。
4. **Ensemble**: PT (skip connection) + S (categorical embedding for spike features) + AE + TF (residual MLP) — 多架构 blending，"middle 60% averaging" 防止极端预测。
5. **Inference 优化**: 把 frozen encoder 内嵌到 MLP 第一层，省 forward pass。
6. **Outlier 处理**: 排除 partial trading days, 移除高 vol outlier days。

### 对我们比赛迁移的启示
- **多 head 共训**：我们要预测 5/10/20/40/60 ticks 五个 horizon——天然 multi-task setup，可以借 AE+MLP 思路，bottleneck 共享，5 个 head 各自预测。
- **3-fold time-grouped CV with embargo**：必做。我们 120 天 × 5 sym，用 day 做 group key + 5-10 day embargo。
- **Multi-arch ensemble**：CNN(DeepLOB) + MLP(MLPLOB) + Transformer(TLOB) + GBDT(LightGBM on derived features) 4 路 → 中位数 60% 平均。

---

## 4. Jane Street Real-Time Market Data Forecasting (2024-2025)

- **任务**：实时市场数据预测，比 2020 版更接近真实生产。
- 比赛 2025 年才结束，1st place writeup 还在汇总；社区公认有效的方法仍是 transformer + AE + GBDT 的组合，加严格 CV。

---

## 5. G-Research Crypto Forecasting (2022)

- **任务**：14 种加密货币 1-minute 数据预测 15-min-ahead 回报。
- **关键 trick（社区共识）**:
  1. **Hull Moving Average (HMA)** 是单个最强特征。HMA 公式：
     ```
     HMA(n) = WMA(2 · WMA(n/2) - WMA(n)) of period sqrt(n)
     ```
  2. **Fibonacci-window lags**：lag = {55, 210, 340, 890, 3750} 而不是均匀间隔。
  3. **Market regime gating**：分别训练 up/down/stable 三种市场模型，用现成 indicator 切换。
  4. **Temporal aggregation**：past 15min 的 mean/max/min/range，作为时序上下文。

### 对我们的启示
- **Fibonacci windows** 在我们 100-tick window 内可以是 {5, 13, 34, 89}，多尺度滚动 stats。
- **Regime gating**：用 VPIN 或 spread 水平把 sample 分高低波段，分别训练。

---

## 6. Two Sigma Financial Modeling Challenge (2017) — 老但经典

- **任务**：anonymized features 预测短期 return。
- **共识做法**：feature engineering 重点在 **rolling cross-sectional rank**, **lagged returns**, **z-score across instruments at same time**。
- **rank features**：对每个 timestamp 把所有 instrument 按某 feature 排序得到 rank（cross-sectional rank），然后用这个 rank 作为新 feature。在我们 5 sym 的小 cross-section 上效果可能有限，但**短期 momentum vs reversal** 用 sym-pair 价比还是能拿信号。

---

## 跨比赛的通用 lesson（极重要）

| Lesson | Concrete action for 良文杯 |
|---|---|
| **Feature engineering > Model complexity** | 先把 OFI / WMP / triplet / rolling stats 全做，再卷模型 |
| **GBDT 是 tabular 的强 baseline** | LightGBM 在我们衍生 feature 上必跑，作为 DL 的对照 |
| **Multi-architecture ensemble** | 至少 CNN + Transformer + GBDT 三路 |
| **Time-grouped CV with embargo** | day-level group + 5-10 day embargo |
| **Class imbalance via balanced sampling**（见 microstructural guide） | 我们 label=1（平）多，需要按 class oversample 或 focal loss |
| **PnL-aware**：分类好不一定 PnL 好（见 TLOB 论文 + Optiver Close） | 直接训练 PnL-aware loss 或 threshold 后处理 |
| **Multi-target / multi-horizon 共训** | 5 个 head（5/10/20/40/60）共享 backbone |

---

## 来源

- Optiver Realized Volatility 1st: https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/274970
- Optiver Trading Close 1st (HYD): https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/hyd-1st-place-solution
- Trading Close feature blog (fan2goa1): https://fan2goa1.github.io/mkdocs-material/blog/2023/12/24/kaggle-optiver---trading-at-the-close/
- Jane Street 1st (Yirun): https://www.kaggle.com/competitions/jane-street-market-prediction/writeups/cats-trading-yirun-s-solution-1st-place-training-s
- Jane Street community SAE+MLP: https://www.kaggle.com/code/gogo827jz/jane-street-supervised-autoencoder-mlp
- G-Research Crypto: https://www.kaggle.com/competitions/g-research-crypto-forecasting
- Optiver 91st place blog (Chris Richard Miles): https://chrisrichardmiles.github.io/chrisrichardmiles/projects/optiver/index_optiver.html
