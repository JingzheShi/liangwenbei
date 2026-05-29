# 答辩小抄：Feature 工程 Q&A 速查

> 用于答辩时被老师追问因子细节、跨股/跨日泛化、KS-drop-11 的现场速查。
> 来源：`factor_families_report_v4.pdf` §4 / `ANSWERS_P3_FEATURES.md` §6 / `onepage_v2/sweep_5seed_summary.json` / `ablation_runs/feat_family_progressive/build_partition.py`。

---

## Q1：6 大家族的因子定义与计算公式

> 维度账：154 raw + 216 derived = 370，**KS drop 11 → 359 喂模型**。`SchemeP` 的 F1–F6 是 370 维的完备分组（互斥、并 = 100%）。

### F1 LOB 派生（94 raw + 11 派生 = 105 维 · LGB gain 40.4%）

- **OHLC + 10 档报价/挂量 (44 raw)**：open/high/low/close、bid_k/ask_k、bsize_k/asize_k (k=1..10)
- **聚合 (4 raw)**：avgbid、avgask、totalbsize、totalasize（主办方给的"4 mean"是另一组：bid_mean / ask_mean / bsize_mean / asize_mean）
- **midprice_k (10 raw) = (bid_k + ask_k) / 2**
- **spread_k (10 raw) = ask_k − bid_k**；cumspread、imbalance = (bs1−as1)/(bs1+as1)
- **bid_diff_k / ask_diff_k (20 raw)**：同侧相邻档位价差 bid_k − bid_{k+1}
- **WMP 微价格（Stoikov 2014, 11 派生）**：
  $$wmp\_lvl_k = (a_k \cdot bs_k + b_k \cdot as_k) / (bs_k + as_k)$$
  **重点：用 ask 价权 bs，bid 价权 as——挂单少的那一侧权重更大**（短缺侧把 mid 拉向它），符合"哪边将耗尽就先走哪边"的微观结构直觉。
  + 1 个 wmp_balance_12 = wmp_lvl1 − wmp_lvl2（层间漂移）

### F2 多尺度 OFI（0 raw + 50 派生 · gain 5.5%）

100% 派生、家族内 |r|=0.33 全场最高（强冗余 → importance 被稀释）。
- **MLOFI (Cont 2014, 30 维)**：W ∈ {5, 20, 60} × 10 档。核心：
  $$e_t^{(k)} = \mathbb{1}_{b_t^{(k)}\geq b_{t-1}^{(k)}} bs_t^{(k)} - \mathbb{1}_{b_t^{(k)}\leq b_{t-1}^{(k)}} bs_{t-1}^{(k)} - \mathbb{1}_{a_t^{(k)}\leq a_{t-1}^{(k)}} as_t^{(k)} + \mathbb{1}_{a_t^{(k)}\geq a_{t-1}^{(k)}} as_{t-1}^{(k)}$$
  $$MLOFI_{t,W}^{(k)} = \sum_{s=t-W+1}^{t} e_s^{(k)}$$
  直觉：b 上行（或不变）贡献 +bs_t；b 下行扣前一档 bs_{t-1}（即被吃单/撤单消失的挂量），左右对称构成净 bid pressure。
- **EWMA-OFI (12 维)**：α ∈ {0.05, 0.1, 0.3, 0.5} × k ∈ {1, 5, 10}，OFI 的指数加权
- **Kyle 反冲击 KyleInv_W (1 维)**：`Δamt / (³√|Δamt|·σ(r_W)+ε)`，"活跃度下净流金额能解释多少方向"
- **Kyle λ (2 维)**：`cov(signed_dvol, Δmid) / var(signed_dvol)` for W ∈ {50, 100} — **整族 KS drop**
- **OFI Toxicity (4 维)**：W 内 OFI 与价格变化方向不一致的占比

### F3 订单流强度（18 raw + 27 派生 = 45 维 · gain 35.1%）— **Top-4 全在此**

主办方直接给的 6 类订单事件 × 3 变体：limit/market/cancel × buy/sell × intst/ind/acc。
- **intst (6 raw)**：单位时间该类事件的强度/频次/量
  - 全场重要性 Top-4：`ma_intst 0.945`（市价卖）、`mb_intst 0.860`（市价买）、`la_intst 0.65`（限价卖）、`lb_intst 0.567`（限价买）
- **ind (6 raw)**：0/1 该 tick 是否发生
- **acc (6 raw)**：二阶差分（加速度）— 6 维 gain 全 <0.05，是死特征但留着
- **EWMA-intst (24 派生)**：6 类 × 4 个 α（IIR 滤波器）：
  $$\tilde{x}_t^{(\alpha)} = \alpha x_t + (1-\alpha)\tilde{x}_{t-1}^{(\alpha)}$$
- **cancel_imb_W (3 派生)**：撤单不平衡，W ∈ {20, 50, 100}
  $$\frac{\sum cb}{\sum(lb+mb+cb)+\varepsilon} - \frac{\sum ca}{\sum(la+ma+ca)+\varepsilon}$$

### F4 微结构波动（2 raw + 27 派生 = 29 维 · gain 7.5%）

- **volume_delta, amount_delta (2 raw)**：单 tick 成交量/金额变化
- **rv_W (4 维) = Σr_t²**：实现波动率 RV, W ∈ {5,10,20,50}
- **signed_rv_W (3 维) = (RV+ − RV−)/(RV+ + RV−) ∈ [−1,1]**：方向化 RV
- **rskew_W (3 维)**：log-return 偏度（m3/sd³，clip ±10）
- **signed_bv_W (3 维)**：带符号 bipower 与 RV 的比
- **vol_burst / amt_burst (4 维)**：`v[-1] / mean(v[-W:])`，活跃度突发倍数
- **rv_ratio (3 维)**：短/长窗 RV 比
- **jshare_W (4 维) = max(0, (RV−BV)/RV)**：Barndorff-Nielsen-Shephard **跳跃份额**；BV = (π/2) Σ|r_t||r_{t-1}|（不含跳跃的二次变差稳健估计 → 跳跃 = RV − BV）
- **roll_eff_spr_ratio_W (3 维)**：Roll 1984 有效价差与 quoted spread 的比
  $$\frac{2\sqrt{\max(0, -\text{cov}(\Delta P_t, \Delta P_{t-1}))}}{\text{spread}_1+1}$$
  反映买卖压力切换时的负自协方差恢复；**W=100 被 KS drop**

### F5 窗口统计（0 raw + 67 派生 · gain 5.2%）— **OOD 主力 family**

100% 派生、家族内 |r|=0.07 全场最低（近正交）、与其他 5 族 |r| ≤ 0.07。它的 5.2% gain 是"无冗余 5.2%"，是跨股泛化的几何基础。
- **dualz_c (37 维) Dual-window z-score（决胜 OOD 的 trick）**：对 37 个 base 列做
  $$dualz(c)_t = \frac{c_t - \mu_{20}(c)}{\sigma_{20}(c)+\varepsilon} - \frac{c_t - \mu_{100}(c)}{\sigma_{100}(c)+\varepsilon}$$
  直觉：当前 tick 相对短期均值偏离 多于 长期均值偏离的程度；**对均值水平不敏感（两 z 相减消除全局基差）**、对方差鲁棒（除以 σ）。
- **qrank_W100_c (20 维)**：当前值在过去 100 tick 中的分位 rank ∈ [0, 1]
- **mid_ewma_resid_a0.05 (1 维)**：midprice 减 EWMA(α=0.05) 残差（长期偏离）
- **adapt_mom_W (3 维)**：方向化 RV × 窗口收益符号（Adaptive momentum）
- **spread_reg_W (3 维)**：spread 当前值的窗口内 z-score
- **trade_pers_W (3 维)**：连续同方向 trade 占比（trade persistence）

### F6 不对称性（40 raw + 34 派生 = 74 维 · gain 6.3%）

- **bid_rate_k / ask_rate_k (20 raw)**：主办方给的第 k 档买/卖价滑动平均变化率
- **bsize_rate_k / asize_rate_k (20 raw)**：第 k 档挂量滑动变化率
- **GOFI (30 派生, Generalized OFI)**：W ∈ {5,20,60} × 10 档：
  $$e_t^{(k)} = \big(\mathbb{1}_{b\uparrow}bs_t - \mathbb{1}_{b\downarrow}bs_{t-1} + \mathbb{1}_{b=}(bs_t - bs_{t-1})\big) + \big(\mathbb{1}_{a\uparrow}as_{t-1} - \mathbb{1}_{a\downarrow}as_t - \mathbb{1}_{a=}(as_t - as_{t-1})\big)$$
  与 MLOFI 区别：MLOFI 严格按"价格上下行"，**GOFI 把"同价不变但挂量净变化也算入流"**。这是 F2↔F6 corr 高达 0.176 的原因（MLOFI/GOFI 同源）。
- **liq_asym_top5_W (4 派生)**：top-5 流动性买/卖侧不对称的对数
  $$\log\left(\frac{1+\sum_{k=1}^{5}bs_k[-W:]}{1+\sum_{k=1}^{5}as_k[-W:]}\right)$$
  **W=5 被 KS drop**（W=5 太短，本质单 tick 比值噪音）

### 三类关键预处理（PPT p7）

1. **window-z score（输入归一化）—— 单贡献 +22.88 PnL**
   所有统计量在 100-tick 切片内自适应归一化，**跨股分布差异被窗口吸收 → 训练外股票也能 zero-shot**
2. **买卖镜像数据增强 —— +2.41 PnL**
   每条 (X, y_reg, y_cls) 复制一份镜像：bid/ask 类列互换，y_reg → −y_reg、y_cls → 2−y_cls。训练集 ×2，强制方向对称
3. **稳健归一化 sign-log1p —— LGB 受益，NN 中性**
   仅对 amount_delta 这种 10³~10⁶ 重尾字段：v → sign(v)·log(1+|v|)，保留方向、量纲压到 ~18

---

## Q2：跨日期 / 跨股票 泛化性证据

### 设计层：硬约束 + 3 个 eval setting

**评测端 3 条硬约束**（PPT p3 红字）：
1. date 被置 0；
2. 测试点顺序被打乱；
3. sym 0–4 可能含训练外股票 → 必须 **sym-agnostic**

所有 370 个因子都满足：(i) 不显式用 date / sym；(ii) 每条样本独立可算（不依赖跨样本状态）；(iii) 不嵌入 sym embedding / sym-conditional 归一化。

**3 个 eval setting**（PPT p5）：
| Setting | 划分 | OOD 维度 |
|---|---|---|
| **Train-Val-Test** | date 0–79 / 80–95 / 96–119 单一切分 | 日期 OOD；test zero-tune 最终评分 |
| **LOSO-date** | K-fold by date (K=5) | 日期 OOD；stock 全 in-sample |
| **LOSO-stock** | 5-fold by stock | 股票 OOD；验证 sym-agnostic 假设 |

### 跨股泛化关键证据

1. **window-z 单 trick +22.88 PnL**（NN 路径 154 raw + L2 + window-z vs 154 raw + L2：+0.52 → +23.40）。LGB 分裂阈值是绝对值的；若 train 一只 sym spread~10⁻³、test OOD sym spread~3×10⁻³，训练阈值在测试上全错位。window-z 把绝对值映射到无量纲 z，"分裂可比性"得以恢复。
2. **T7 加 window-z 后，sym=2（异常蓝筹/ETF）cum_pnl 从 −0.98 翻到 +2.24** —— 全 sym 翻正，证明 window-z 解决了 sym 间分布差异。
3. **T10 反例**：把窗口 z-score 换成全局 z-score（基于整训练集统计量）→ LOSO h_10 +13.40，**反而退化**。原因：全局统计量被训练 sym 主导，OOD sym 时 (x−global_μ)/global_σ 不再以 0 为中心。**窗口 > 全局 > per-sym（per-sym 直接违约束）**。
4. **F5 与所有其他族 |r| < 0.07（近正交）** —— Dual-z / qrank 把"绝对水平"扔掉、留"相对位置"，这是 F5 提供独立 OOD 信号的几何解释。它的 5.2% gain 是无冗余 5.2%。
5. **F1+F3 块间 corr = 0.035** —— 两个最重要的家族几乎正交，LGB 同时拿到两者获得"几乎无冗余"的高信息组合。
6. **公榜结果**：baseline DeepLOB −6.65 → 加 SchemeC +4.07 → +window-z & 回归 +19.23 → +SPO+ DFL +28.16 → +LGB 全量重训 +34.44 → ensemble +35.64。公榜本身就是 OOD（日期 80–119 之后的 sym 0–4 + 可能新股票），window-z 是其中两个核心泛化 trick 之一。
7. **私榜 #1**（h=60 +41.61、h=40 +38.65）：私榜与公榜不同日期、不同样本，整体性能甚至更高 —— 说明因子工程的泛化没有"踩公榜运气"。

### 跨日泛化关键证据

1. **L1 → L7 progressive 5-seed**（按 family 累加加入派生，时间切分 Train 0–79 / Val 80–95 / Test 96–119）：
   - L1 154 raw → 17.7 ± 1.33
   - +F1 (wmp 11) → 17.5（≈0，wmp 是 raw bid/ask 的加权和，信息冗余）
   - +F2 (OFI 48) → 22.6 (+5.1)
   - +F3 (INTST 27) → 25.7 (+3.1)
   - +F4 (VOL 32) → 24.9 (−0.8)
   - +F5 (WIN 63) → 26.3 (+1.4)
   - +F6 (ASYM 35) → 27.3 (+1.0)
   每一步加入 family 都用 **训练区完全没看过的 date 96–119** 评分，证明累加因子都不是 in-sample 过拟合。
2. **M7 全量重训（train + val 全量 0–119 重训，迭代数 × 1.1）** 单 trick 给 LGB +4.3、给最终公榜 +5.51。能成立的前提是因子在跨日上稳定（否则全量重训只会 overfit 验证集）。
3. **mid_ewma_resid_a0.05** 和 **dualz** 等长窗 EWMA/z 因子刻意选 W=100（≈5 min），覆盖跨日内的"分钟级 regime shift" 但不跨日。

---

## Q3：KS-drop 11 features 的过程与效果

### 11 个被 drop 的特征（按家族归类）

| # | 特征名 | 族 | 失效原因 |
|---|---|---|---|
| 1 | `dualz_ask_diff1` | F5 | ask 侧 1-level diff 跨 sym 分布差大 |
| 2 | `dualz_bid_diff5` | F5 | 5-level diff 噪音 |
| 3 | `dualz_ask_diff5` | F5 | 同上 |
| 4 | `qrank_W100_spread1` | F5 | spread1 大多 = 1 tick → qrank 几乎恒 1，死特征 |
| 5 | `qrank_W100_spread5` | F5 | spread5 tick-quantized，rank ≈ 1 |
| 6 | `qrank_W100_spread10` | F5 | 同上 |
| 7 | `qrank_W100_cumspread` | F5 | 累积 spread 同问题 |
| 8 | `kyle_lam_W50` | F2 | 跨 sym 协方差分布差异大，**整个 Kyle λ 族 KS fail** |
| 9 | `kyle_lam_W100` | F2 | 同上 |
| 10 | `roll_eff_spr_ratio_W100` | F4 | W=100 太长，跨 sym Roll 估计不稳 |
| 11 | `liq_asym_top5_W5` | F6 | W=5 太短，本质单 tick top-5 比值，噪音大 |

> **对称性 design debt**（如果被追问）：`dualz_bid_diff1` 保留但 `dualz_ask_diff1` drop，原因可能是 bid 侧在某些 sym 上 distribution shift 小于 ask 侧（A 股流动性结构本就买卖不对称），但**没有可复现的理论证据**，是经验观察。

### 失效标准

**Kolmogorov-Smirnov（KS）分布检验** —— 对每个 sym 单独看该特征在该 sym 上的分布 vs 全局训练分布做 KS test，p < α 即"sym-non-agnostic"→ drop。

具体执行：在一组 sym-agnostic 训练（T59 全 sym 训练）之后，逐 feature 跑 KS 检验，把不通过的列名固定为 `T59_FAIL_NAMES (10) + STAGE5_FAIL_NAMES (1) = 11`。
**具体 p 值阈值 / Bonferroni / val split 不在代码内（design debt 🔴），但 11 个名字已 freeze 进 SchemeP**。

### 效果（5-seed 直接对比，Train-Val-Test setting，test = date 96–119）

| Backbone | 370-d no drop | 359-d drop 11 | Δ mean | Δ std |
|---|---:|---:|---:|---:|
| LGB L2 | 27.28 ± 2.34 | 26.58 ± **0.96** | −0.70 | std **−59%** |
| NN L2 + window-z | 34.77 ± 1.39 | 34.34 ± **0.50** | −0.43 | std **−64%** |

**关键解读**：drop 11 牺牲约 0.5 个 mean PnL，但 **seed-to-seed std 砍掉 60%**。
- seed std 衡量"模型对训练随机性的敏感度"
- OOD 上 std 高的模型在公榜/私榜会"运气依赖"，单次提交可能高也可能低
- drop 11 后的 359-d 模型才是 50-seed ensemble + 公榜部署的 **稳定基底**
- 一页纸消融表 row 9→row 10 写的 "drop 11 → −1.57"，是消融表链式差值（NN 路径 +F6 累加后再 drop 11），与 5-seed 直比的 −0.43 数值不同但方向一致

**Why drop 让 std 大降**：被 drop 的 11 个都是"在某些 sym 上几乎是常数 / 死特征 / 极噪声"的列。LGB 偶尔会用这种"死列"做分裂、形成 spurious 路径；不同 seed 下走的 spurious 路径不同 → seed 间分歧大。drop 后强制模型只能用"跨 sym 同分布"的列，决策面更稳。

---

## 老师可能追问的 7 个刁钻问题（含答案）

**Q1: 为什么 F1 importance 40% 但 progressive 增益 ≈ 0？**
> F1 里 94/105 维是 raw bid/ask/size/OHLC，已在 L1 baseline。F1 派生只剩 11 个 wmp_*，wmp 是 raw bid/ask 加权和、信息高度冗余。importance 反映的是"模型用了谁"，被 raw 价格 baseline 主导；progressive 反映的是"相对 baseline 新增了什么"，wmp 没新增信息。

**Q2: F2 importance 仅 5.5% 但 progressive 加入 +5.08 是最大跳变，为啥？**
> F2 内部 |r|=0.33 全场最高（50 个 MLOFI 因子高度相关）。LGB 的 gain 在高相关族内被分摊到很多 feature 上，单看 share 被稀释；但作为一个整体，OFI 提供了 baseline 完全没有的"订单流方向 × 时间尺度"维度，相对 baseline 跳变最大。

**Q3: 为什么 ma_intst 是单因子 gain 第一？它和 OFI 不重复吗？**
> ma_intst（市价卖）是已脱敏的市价卖单总量/次数，**直接来自主办方**，不需要从价格反推 OFI 那种"逆向工程"。OFI 是从 (bid/ask, bsize/asize) 的变化间接推订单流，会有"挂单跳价"的歧义；ma_intst 没歧义。LGB 优先用最低歧义的信号当根分裂，所以 gain 最高。

**Q4: 为什么 5 个 OFI/Kyle 都用 W ∈ {5,20,60} 或 {50,100}，没做窗口 sweep？**
> 这是 R34 调研阶段的固定选择，没正式 sweep（design debt）。理由：(20,50,100) ticks ≈ (1, 2.5, 5) min 是 HFT 文献的常用尺度；窗口 sweep 一次要重算 1.47M 行 feature cache，成本高，没做。Roll W=30、jshare W=30 等"独有窗口"是各 stage 提议时的遗留，没归一化掉。

**Q5: Kyle λ 整族 KS drop 了，为什么不直接整族删除算了？**
> 我们就是这么做的（W50 W100 两个都 drop）。但 family report 里 F2 50 维仍包含 Kyle λ —— 这是 dimension 账（370 维分组），不是"模型用了多少"。喂模型时是 359 维（drop 11）。Kyle λ 的位置由 KyleInv 顶上了（KyleInv 通过 ³√ 重整 amt → 跨 sym 量纲更稳）。

**Q6: window-z 用 100-tick (~5min) 会不会泄露未来？**
> 不会。window-z 是 **trailing window**：对当前 tick t，用的统计量是 [t−99, t]，全部历史 tick；不含 t+1 及之后。两个关键代码位置：
> - `cfg["wz"]: mu = np.nanmean(tr["X"], axis=0)` 在 train 上算全局 mu/sd 套到 val/test，**保证三集同口径**，避免 train/val/test 分布偏移
> - dualz 算法里 seg20 / seg100 都从 X3d[:, -W:, :] 取，永远是过去 W tick

**Q7: KS test drop 11 是不是会 overfit 训练集的 sym 0-4？OOD 新 sym 上还有效吗？**
> 风险确实存在（KS 用训练 sym 算分布）。我们的兜底是 (i) drop 的列要么 in-sample 就近乎死特征（qrank on quantized spread）、要么 W 太短/长极噪声（liq_asym_W5、roll_W100），这些"病态特征"在任何 sym 上都不该贡献信号；(ii) 公榜/私榜两套不同 OOD test 上 359-d 模型都拿到 #1，证明 drop 列表对训练外 sym 也鲁棒。

---

## 一句话总结

> 6 大家族（370 维）+ 3 类 sym-agnostic 预处理（window-z / 镜像 / sign-log1p）+ KS drop 11（359 维稳定基底）+ 50×50 异质 ensemble，把跨股 + 跨日 OOD 风险层层吸收，最终公榜 +35.64（baseline −6.65）/ 私榜 #1（+41.61）。
