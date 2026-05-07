# R33: Kaggle / HFT 比赛冠军方案 Reverse Engineering

> 目标：搞清楚公榜较好 team 怎么做到 **5x recall（多交易）+ 1.4x per-trade alpha**。每个冠军方案 1-2 页详细 reverse engineering，含 model / 特征 / OOD / recall trick 各维度。
>
> 调研日期：2026-05-07
>
> ⚠️ 限制说明：
> - YouTube auto-caption 在本机走不通（cloud IP 被 YouTube 封；`/etc/network_turbo` 不存在）。HAO LI / Patrick Yam / Evgeniia Grigoreva 三个 walkthrough 视频未能直读字幕。所列内容均来自 GitHub solution.md / Kaggle discussion / 第三方 blog（含一份日文 docswell ppt）/ Medium / 公开 LinkedIn 摘要。
> - 良文杯（Liangwenbei）历史届方案不在公网索引，未找到。
> - Jane Street 2024 公开最完整的是 **Evgenia Volkova（8th, public LB ~0.0112）solution.md**，其他 top 方案信息分散在 Kaggle 讨论区 + LinkedIn 摘要 + 第三方推断，写出来的部分会标注置信度。

---

## 0. TL;DR — 与我们项目最相关的 6 条结论

| # | 结论 | 来源 | 对我们的启示 |
|---|------|------|----------|
| 1 | **Auxiliary targets（多任务）几乎所有 top 方案都用** | Volkova（resp_7/8/9/10）、Yirun 2020（5 个 action）、hyd 2023（多个 horizon） | iter_006 只用单一 close-to-close 目标 → 加 1m / 5m / 30m 多 horizon 辅助头，预计 +0.5–1.5 LOSO |
| 2 | **Cross-symbol attention/Transformer**：把同 timestep 全部 200 stocks 喂进一个 Transformer，预测 200 stocks | Optiver 2023 hyd 1st，Cross-stock 0.2 权重 | 我们的 Scheme C 特征之间没有 cross-sym 信号；可以做 **同 sym=4 内的 cross-bar attention**（不违反 sym-agnostic 约束） |
| 3 | **Online learning / 增量更新**给 +0.008 R²（Volkova），是单一最大涨分手段 | Volkova solution.md | 平台 Predictor 不维护 state，但 **训练时 online ablation** 仍能推断出 dx/dt 局部相关性，再编码进静态特征 |
| 4 | **Rolling stats over 1000 last time_ids per symbol**（Volkova 关键特征）| solution.md 节 2.3 | iter_006 已有 100/200/600 bar EMA，没到 1000；尝试 **更长 window（如 2000）** 可能有用 |
| 5 | **Triplet imbalance** `(max-mid)/(mid-min)` 在 3 个数值列上做出非线性比率特征 | Optiver 2023 多个 top 方案 | 我们的 226 d 里有大量 LOB ratio，但 **三元 triplet imbalance 还没用过**；低成本可以补 |
| 6 | **Tick-size 反推 time_id 顺序 + nearest-neighbor 特征** | Optiver Realized Vol 2021 nyanp 1st | 我们的 date 在评测时被置 0，但**测试点 order 被 shuffle 不代表 timestamp 信号被毁**；可以在训练时做 NN over train，把 test 点最近邻 train 点的 target/feature 拿来当 feature（**sym-agnostic safe**） |

---

## 1. Jane Street Real-Time Market Data Forecasting (2024–2025)
评估指标：weighted zero-mean R² on `responder_6`。
最相关：因为这次比赛**就有 39 sym**（实际 0–38），目标也是预测 6h 后 return，weighting 也类似 PnL；比赛持续到 2025-Q1，私榜直到 2025-Q3。

### 1.1 [8th place, public LB 0.0112] Evgenia Volkova（已读完整 solution.md）

**模型架构**
- **Time-series GRU**，sequence length = 1 day（968 time_ids per day after `date_id ≥ 700`）。
- 两个变体合奏：
  - GRU-1: 3-layer GRU
  - GRU-2: 1-layer GRU + 2× linear + ReLU + dropout
- **MLP / Transformer / cross-symbol attention / sym embedding 都没工作**（"didn't work for me"）。
- ⚠️ **她明确尝试了 sym embedding 但不工作**——和我们的 sym-agnostic 约束完全契合。

**特征（共约 90 维）**
1. 78 个 raw feature（去掉 categorical 09–11）
2. 选 16 个高相关 feature 上做：
   - **Market averages**：per `date_id` 平均 + per `time_id` 平均（**cross-stock pooling，sym-agnostic**！）
   - **Rolling stats per symbol**：last 1000 time_ids 的 mean / std
3. 加了 `time_id` 原始值作为 feature

**辅助 target（关键 +0.001 LB）**
```python
# responder_6 = 20-day rolling avg of underlying
# responder_7 = 120-day rolling avg
# responder_8 = 4-day rolling avg
responder_9  = responder_8 + responder_8.shift(-4)   # ≈ 8-day rolling avg
responder_10 = responder_6 + responder_6.shift(-20) + responder_6.shift(-40)  # ≈ 60-day rolling avg
```
- 4 个 aux target，每个一个独立的 GRU base model
- 4 个 base prediction → 一个 linear layer → final responder_6
- Loss = sum of weighted-zero-mean-R² across all responders

**训练**
- Time-series CV，2 fold，每 fold 200 dates val。
- Batch = 1 day，lr = 5e-4，约 6–10 epochs。

**Online learning（最大单点 +0.008）**
- 推理时每天结束，用 `responder_6` 单 target、lr=3e-4、做 **1 次 forward+backward** 更新。
- 不动 aux target，只用主 target 做 online update。
- 「performing one-day updates for almost a year is enough」——增量更新 9 个月就够，不需要全量重训。

**Ensemble**
- (GRU-1 + GRU-2) × 3 seeds = 6 个模型 simple average → LB 0.0112
- 单模最佳 LB 0.0105 → 合奏 +6.7%

**对我们 LOSO（h_60）的启示**
| 招数 | 适配性 | 预计 +LOSO | 实施成本 |
|---|---|---|---|
| Aux target = 长短 horizon 平均 | 高（直接平移） | +0.5–1.0 | 低（已有 horizon=60 框架） |
| Per-stock rolling stats over **1000-2000** bar | 高 | +0.3 | 极低（h_2000 EMA 没试过） |
| Market average per timestep（cross-sym pooling） | 中（sym 0–4 + unseen） | +0.2 | 低 |
| GRU sequence model | 中（NN 不算 LightGBM 替换，但可加） | +0.3 | 高（重新搭 NN pipe） |
| **Online learning at inference time** | **❌违反约束 2**（test 顺序乱） | 0 | — |

> 关键：Volkova solution **明确放弃了 sym embedding 和 sym-specific 模型**，而 8th place 已经站稳在 top 0.3%（3757 队），说明 **sym-agnostic 是一条可行路线**——但要用好 cross-sym pooling 来「补」掉 embedding 失去的信息。

---

### 1.2 [推断] HAO LI 1st place（来源：Kaggle Winners Walkthrough YouTube + Kaggle discussion 摘录，未拿到字幕）

来源不全；以下是 **可信度中等** 的推断（基于 Kaggle 公开讨论 + LinkedIn / 视频简介）：
- 主模型疑似 **Transformer + GRU 混合 + LightGBM 残差校正**（Patrick Yam LinkedIn 提到「information-theoretic features and ensembles」）
- 强调 online learning（与 Volkova 一致），可能用 **每日 1–2 次 fine-tune**
- LB Top1 final 约 **0.0157–0.0180**（vs 我们的 8th-equivalent 0.0112）

⚠️ 没拿到原始字幕，**不要照抄**；只作为「8th vs 1st 大概差 +50–60% R² 是怎么拿到的」的方向。

---

### 1.3 [推断] Patrick Yam 2nd place（LinkedIn 简介 + 视频标题）
- 关键词：**"information-theoretic features"**——很可能是 mutual-information–based feature importance / interaction screening / minimum redundancy maximum relevance (mRMR) 来挑 cross-feature
- Ensembles（多模型 stacking）
- 完整解法未公开

---

### 1.4 [13th, public 0.0090+] codefluence GitHub
- LightGBM + MLP ensemble
- 78 raw features only，没做高级 feature engineering
- 没做 online learning → 比 Volkova 差 ~0.002 R²，印证 online learning 是 +0.008 的关键

---

## 2. Optiver — Trading at the Close (Kaggle 2023, $100k, ~200 stocks NASDAQ)
评估指标：MAE on 60-second-ahead return.

### 2.1 [1st place] hyd（hydantess1993）

> 来源：日文 docswell ppt + Kaggle public writeup 摘要 + X 推文。**结构最完整、可借鉴度最高**。

**数据特征：300 features**

特征工程「Magic features」核心是 **三种 grouping 方式做比率/排名**：
1. **Within-stock / time-bucket ratio**：grouped by (stock × day × second-bucket)，feature 在 expanding mean / first-value 上的比率
2. **Cross-stock ratio**：grouped by (day × second)，每个 feature 跨 200 stocks 算 **mean ratio + percentile rank**
3. **Time-bucket cutoffs**：0–290s / 300–470s / 480–540s 三段（基于观察到的 size 行为分段）
- 应用对象：bid/ask price、bid/ask size

**模型 Ensemble (weights = 0.5 / 0.3 / 0.2)**
| 模型 | 输入结构 | 输出 | 学到什么 |
|---|---|---|---|
| **CatBoost** (0.5) | flat 300 features + categorical | 单点 target | tabular interactions |
| **GRU** (0.3) | per stock, **55 timesteps**, 输出最后 1 个 | 单点 target | **time-series intra-stock** |
| **Transformer** (0.2) | **per timestep, all 200 stocks**, 输出 200 stocks | 200 个 target 同时 | **cross-stock 同步** |

> ✨ **核心 idea：用两个不同维度的 sequence model 分别捕获 time-axis 和 stock-axis 信号**，再用 GBDT 兜底。

**Categorical features for CatBoost**
- stock_id, day_of_week, seconds, minute, imbalance_buy_sell_flag, imbalance_buy_sell_flag.shift(1/3/5/10)
- ⚠️ 这里用了 stock_id ——但 Optiver 数据里测试集就是这 200 只 stock，**没有 OOD stock**，所以可以；**对我们违反约束 3，不能照抄**。

**训练 + Online learning**
- **每 12 天做一次增量更新，5 个 cycle（约 60 天）的 fine-tune**
- 等于在 validation set 上做 5 次 incremental learning
- 这是 hyd 跟 2nd 拉开差距的关键

**Post-processing（关键涨分）**
- 论文 / writeup 没有公开具体公式，但提到「post-processing contributed significantly」
- 推断是 **per-stock bias correction** 或 **rank-residualization**
- 我们的 T35 ReVol+SG / T34 calibration v2 已经做了类似的事，可以复盘有无空间

**对我们的启示**
- **「GRU per-stock + Transformer cross-stock」双轴架构** → 在我们 5 sym 上等价于 **「sym-axis sliding ↔ time-axis sliding」**：
  - sym=4 个，太少做 attention，但可做 **mean / std / rank pooling across all sym** → 这就是 Volkova 的 market average
  - time-axis：60-bar 序列建 GRU 头
- **300 features 都来自 ratio + rank**，我们的 226 features 里 ratio 类只占 ~1/4，**新增 cross-stock rank percentile** 是低成本 +1% pnl 候选

### 2.2 [9th place] ChunhanLi（GitHub 仅 README + xgboost-training.ipynb）
- XGBoost 单模为主
- 详细 README 缺失，只能看 notebook（300+ feature engineering，purged time-series CV）

### 2.3 nimashahbazi LSTM + ConvNet
- 反向案例：**没做特征工程，全靠 raw feature + target lag** → 公榜 5.3508 / 5.3439
- 比 hyd 1st (5.3070) 差 8% MAE
- 教训：「raw features only 是不够的」

---

## 3. Optiver — Realized Volatility Prediction (Kaggle 2021, ~112 stocks)
评估指标：RMSPE。

### 3.1 [1st place] nyanp — "Nearest Neighbors"

**核心 trick（震撼但难复制）**：
- 训练 / 测试集的 `time_id` 被 Optiver **打乱了顺序**
- nyanp **用 price tick size 反推真实时间顺序**：观察每个 stock 在每个 time_id 的最小价差，因为 tick size 是固定的，相邻的 time_id 的 tick size 模式应该相似 → 用 KNN 重建图 → 用 shortest Hamiltonian path 算法找到 likely 顺序
- **重建顺序后**，可以做 **lag features**（即使原数据不让做 lag）

**Nearest-neighbor 特征**：
- 在每个 time_id 上，用 **跨 stock 的 feature vector** 找最相似的历史 time_id
- 把那个最相似 time_id 上的同 stock target 作为 NN target feature
- 这个 trick 等价于「过去市场状态 retrieval」

**模型**：LightGBM 主导 + 少量 NN 校正

**对我们的启示**
- 我们的约束：**「测试点 order 被 shuffle」**——和 Optiver 2021 的 time_id shuffle 几乎一样
- **可以用 nyanp 的同一招**：训练时记录每个 (sym, t_local) 的 226-d feature 和 target；推理时每个测试点查最近邻 train 点的 target 当 NN feature
- 完全 **sym-agnostic 安全**：feature 是 226 维，不依赖 sym index
- ⚠️ 风险：训练集 1.47M 行 × 226 d 全跑 KNN 推理慢；可以用 **FAISS** + **only nearest-1 / nearest-3**

### 3.2 [7th place] orvp（poluektov GitHub，公开度高）
- KNN 重建 time_id ordering（同 nyanp）
- LightGBM dart mode + 标准 mode 合奏
- Final RMSPE = 0.20013

### 3.3 普遍 top10 共识
- 大量用 **realized vol over different windows**（10-min realized vol → exponentially weighted）
- **WAP（weighted avg price）= (bid_price * ask_size + ask_price * bid_size) / (bid_size + ask_size)**
- 跨 stock affinity propagation clustering，**stock cluster ID** 作为 feature（这条对我们 OOD-sym 危险，慎用）

---

## 4. Jane Street Market Prediction (Kaggle 2020-2021, $100k)
评估指标：weighted utility on action ∈ {0, 1}（这次是 binary！）

### 4.1 [1st place] gogo827jz / Yirun Zhang — "Supervised Autoencoder + MLP"

> 这次跟我们最像：**binary trade decision + sym-agnostic + multi-target**

**Architecture**：3 个 head 共享 encoder
```
Input ─→ GaussianNoise(σ=0.035) ─→ Dense ─→ BatchNorm ─→ Swish
                                              │
                                              ├─→ Decoder ──→ MSE(reconstruct input)
                                              ├─→ Aux head  ──→ BCE(reconstruct target)
                                              └─→ concat with raw input ─→ MLP ─→ BCE(main target)
```
- **3 个 loss 同时反传**：reconstruction (MSE) + aux supervised (BCE) + main target (BCE)
- Encoder 出来的 latent **同时作为去噪后特征 + raw feature concat 喂给 MLP head**

**关键技巧**
- **Multi-target / multitask**：5 个 action target（不同 horizon 的 weighted return cutoff）作为辅助
- **Swap noise / Gaussian noise** 是 denoising autoencoder 的 input augmentation
- **Purged 5-fold CV**（避免 label leakage）

**对我们的启示**
- **Aux target = 多 horizon 多 cutoff 的 binary** 完全可以加（h_30/h_60/h_120 三个 binary classification head 共享 encoder）
- **Encoder + raw concat 一起进 head**：等价于「latent + raw 都给 LightGBM」——可以 **用一个小 NN 提取 32-d latent，concat 到 226-d 一起喂 LightGBM**
- **Gaussian noise on input** 这条我们 T39/T41 系列里实际没系统试过

---

## 5. JPX Tokyo Stock Exchange Prediction (Kaggle 2022, ~2000 stocks)
评估指标：Sharpe-ish ranking.

### 5.1 [1st place] Shoki Sakai
- 来源仅 Kaggle Winners Walkthrough YouTube（未拿到字幕）+ J-Quants GitHub（数据样例，无 solution.md）
- 公开度低
- 从 J-Quants repo + 第三方推断：
  - **Linear Tree（线性叶子的 GBDT）** + **Noise-reduced target**（rolling avg 当 target）+ **Ranking 而非 regression**
  - 来自 62nd place 的方法摘要（masahiro takeya），1st 思路相近
  - 强调 **噪声目标 → 平滑 target / multi-horizon target**

**对我们的启示**：和 Volkova 完全一致——**用更长 horizon 的 rolling avg 当 aux target** 是稳定 +的招。

---

## 6. G-Research Crypto Forecasting (Kaggle 2021–2022, 14 coins)
评估指标：weighted Pearson on log return.
- 公开 1st place writeup 极少（host 不希望暴露 alpha）
- 已知 top 方案：
  - **Hull moving average** 是最重要 feature，**Fibonacci 窗口** [55, 210, 340, 890, 3750]
  - **Spacetimeformer（Long-Range Transformers for Spatiotemporal）**
  - 强调 spatial（跨 coin）+ temporal 双轴

---

## 7. Jane Street Market Prediction 2020 ↔ 2024-2025 → Trading at the Close 2023 跨方案对比

| 维度 | JS 2020 (1st Yirun) | Optiver 2023 (1st hyd) | JS 2024 (8th Volkova) |
|---|---|---|---|
| 主模型 | MLP+SAE | CatBoost+GRU+Transformer | GRU only |
| 跨 stock pooling | 否（混在 NN） | Transformer over 200 stocks | per-time mean / std |
| Aux target | 5 actions multitask | 多 horizon | 4 responders |
| Online learning | 否 | 12-day × 5 cycles | per-day forward+backward |
| OOD stock 处理 | 不存在（测试 stock 一致） | 不存在 | sym embedding **试过不工作** |
| Augmentation | Gaussian noise σ=0.035 | — | — |
| Recall 提升核心 | multitask + denoising | cross-stock attention + post-proc | aux target + online learn |
| GBDT vs NN | NN | GBDT 主 + NN 辅 | NN |

> **共同模式**：
> 1. 多任务（multitask aux target） 几乎 100% 出现
> 2. 跨 stock 信息利用（cross-stock attention 或 cross-stock pooling） 几乎 100% 出现
> 3. 「ensemble GBDT + NN」 ≥ 50% 出现
> 4. Online learning / incremental fine-tune 在 sequence model 上是 +0.008 大招

---

## 8. 把上述 reverse engineering 投影到我们的硬约束

### 我们的硬约束
| # | 约束 | 阻断什么 | 不阻断什么 |
|---|---|---|---|
| C1 | `date` 评测时被置 0 | 任何 date / day-of-week 特征；rolling stats 不能跨 date 锚定 | 同 sym 内 t_local（intra-day 时间）特征 |
| C2 | 测试点顺序被打乱 | Predictor 不能维护跨调用 state；不能 incremental online learn | 训练时学到的「最近邻 train 点」可以编码进 inference 静态 logic |
| C3 | sym 0–4 训练，**测试可能含 unseen sym** | sym embedding；per-sym 模型；per-sym normalization | 跨 sym pooling（mean / std / rank）；training 内 sym-agnostic feature engineering |

### 投影结果
| 招数 | 阻断状态 | 投影后版本（合规版） |
|---|---|---|
| Volkova online learning（per-day fine-tune）| ❌C2 阻断 | 无 |
| Volkova rolling stats over 1000 t per sym | ✅合规 | 当前已用到 600，可拓到 1000–2000 |
| Volkova market avg per (date, time) | ⚠️half（date 部分死了） | **改成「同 sym 内 t_local 的 cross-sym 当时 mean / std」** |
| Volkova aux target = 多 horizon | ✅合规 | 加 h_30 / h_120 / h_240 三个头 |
| hyd cross-stock Transformer | ⚠️sym=4 太少做 attention | 改成 **sym=4 cross-pool（mean/std/rank percentile）** as feature |
| hyd online learning ×5 cycles | ❌C2 阻断 | 无 |
| hyd categorical = stock_id | ❌C3 阻断 | 砍掉 |
| Yirun Gaussian noise | ✅合规 | 训练时 input 加噪 σ=0.02–0.05 |
| Yirun multitask binary | ✅合规 | 加 binary classifier head 当 aux task |
| Yirun SAE concat latent + raw | ✅合规 | 跑一个轻 NN 提 32-d latent，concat 到 226-d 给 LightGBM |
| nyanp time_id reorder + NN feature | ✅合规且对症（C2） | **训练点存 226-d KDTree，inference 测试点查最近 K-NN train 点的 target，取 mean 当 1 个 feature** |
| Optiver Triplet imbalance | ✅合规 | 在 226 d 里取 30 组三列做 (max-mid)/(mid-min) |
| Hull MA + Fibonacci windows | ✅合规 | 加 5 个 Fibonacci window EMA on 16 个最强 feature |

---

## 9. 优先级排序（建议下一周实验队列）

| 优先级 | 实验 ID 建议 | 内容 | 预计 LOSO h_60 | 实施时长 |
|---|---|---|---|---|
| 🥇 P0 | T44 | Multi-horizon aux target（h_30 + h_60 + h_120 三头共享 226-d input + LightGBM 多 round 训练 / 各自模型 average）| **+0.5 ~ +1.5** | 1 day |
| 🥇 P0 | T45 | Cross-sym pooling features（每个 sample 加 4 个 cross-sym mean/std/rank/range over 16 高相关 feature 在同 t_local ± 1 bar 内）| +0.3 ~ +0.8 | 0.5 day |
| 🥈 P1 | T46 | KNN-target retrieval feature（FAISS over 226 d 训练特征，inference 时查最近 5 train 点的 target mean / std 当 2 个新 feature）| +0.3 ~ +1.0 | 1 day |
| 🥈 P1 | T47 | Triplet imbalance × 30 组 + Hull MA Fibonacci × 5 个 window | +0.2 ~ +0.5 | 0.5 day |
| 🥉 P2 | T48 | Lightweight SAE latent（4-layer NN 提 32d latent → concat 到 LightGBM）| +0.3 ~ +1.0 | 2 days |
| 🥉 P2 | T49 | Gaussian noise input augmentation σ=0.03，与 T44 联合 | +0.1 ~ +0.3 | 0.5 day |
| 🥉 P3 | T50 | NN（GRU 60-bar 序列模型，sym-agnostic）+ LightGBM ensemble | +0.5 ~ +2.0（高方差） | 3 days |

> **iter_007 候选拼装**：建议 T44 + T45 + T47 三个 features-based 实验先跑（共 ~2 days），再决定要不要拉 T46 / T48 / T50 重型工程。

---

## 10. 来源 & 链接
**Kaggle 比赛页 & writeup**
- [Jane Street Real-Time Market Data Forecasting](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting) (2024-2025)
- [Jane Street Market Prediction](https://www.kaggle.com/competitions/jane-street-market-prediction) (2020-2021)
- [Optiver - Trading at the Close](https://www.kaggle.com/competitions/optiver-trading-at-the-close) (2023)
- [Optiver Realized Volatility Prediction](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction) (2021)
- [JPX Tokyo Stock Exchange Prediction](https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction) (2022)
- [G-Research Crypto Forecasting](https://www.kaggle.com/c/g-research-crypto-forecasting) (2022)

**Solution writeups（部分能直读）**
- [Volkova 8th place GitHub solution.md](https://github.com/evgeniavolkova/kagglejanestreet/blob/master/solution.md) ✅完整
- [Volkova 8th Kaggle discussion 556542](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/discussion/556542)
- [hyd 1st Optiver Trading at Close writeup](https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/hyd-1st-place-solution) ⚠️只能拿到摘要
- [docswell hyd 1st 解读 (日文)](https://www.docswell.com/s/8980249862/K6YQ3E-2024-05-23-200638) ✅大部分关键点
- [Yirun 1st Jane Street 2020 SAE+MLP](https://www.kaggle.com/competitions/jane-street-market-prediction/writeups/cats-trading-yirun-s-solution-1st-place-training-s)
- [Numerai forum: SAE+multitask MLP analysis](https://forum.numer.ai/t/autoencoder-and-multitask-mlp-on-new-dataset-from-kaggle-jane-street/4338) ✅技术细节
- [nyanp 1st Optiver Realized Vol Nearest Neighbors](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/274970)
- [orvp 7th place GitHub](https://github.com/michaelpoluektov/orvp)
- [ChunhanLi 9th Optiver Trading at Close GitHub](https://github.com/ChunhanLi/9th-kaggle-optiver-trading-close)
- [nimashahbazi LSTM/ConvNet GitHub](https://github.com/nimashahbazi/optiver-trading-close)
- [fan2goa1 blog: Optiver Trading at the Close 解析](https://fan2goa1.github.io/mkdocs-material/blog/2023/12/24/kaggle-optiver---trading-at-the-close/)
- [joehbridges Medium: Gauging the Market](https://medium.com/@joehbridges/gauging-the-market-optivers-trading-at-the-close-kaggle-competition-27b73f7789c0)

**视频（未能直读字幕，仅引用 metadata）**
- [Kaggle Winners Walkthroughs HAO LI 1st JS2024](https://www.youtube.com/watch?v=gzgg6txCfd8)
- [Kaggle Winners Walkthroughs Patrick Yam 2nd JS2024](https://www.youtube.com/watch?v=lfzzPZZyzjE) ⚠️用户给的 26NozcM6X3k 已被 owner 设为 private
- [Kaggle Winners Walkthroughs Evgeniia Grigoreva 8th JS2024](https://www.youtube.com/watch?v=lXYC0c7wJFU)
- [Kaggle Winners Walkthroughs Shoki Sakai JPX 1st](https://www.youtube.com/watch?v=vHww9x7w0RY)
