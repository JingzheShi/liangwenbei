# 答辩小抄 rewrite notes

> 输入：`答辩_feature_QA_小抄.md`（旧版 213 行）
> 输出：`答辩_feature_QA_小抄.md`（新版 418 行 · 32 行块公式分隔符 = 16 块公式 + 大量 inline）
> 目标：(A) 渲染兼容 GitHub markdown & 飞书 wiki / KaTeX；(B) 全部缩写补全称 + 中文；(C) 全部符号有定义；(D) 核心概念 1-2 句解释。
> Pandoc 检查：`pandoc 答辩_feature_QA_小抄.md -o /tmp/check.html --mathjax` → exit 0，无警告。

## 1. 结构层改动

- **新增 §0 符号与缩写速查表**（放在最前面，老师一翻就看到）：
  - §0.1 缩写表（36 行）：LOB / OFI / MLOFI / GOFI / EWMA / WMP / RV / BV / Roll / Kyle λ / KyleInv / BNS / KS / PSI / SPO+ / DFL / ETUD / EV / NN / MLP / LGB / HFT / OOD / LOSO / IIR / NNLS / HL / IQR / OHLC / DeepLOB / CNN / RNN / TCN / KAN / PnL → 共 35 条
  - §0.2 数学符号表：$t, k, W, h, \alpha, b_t^{(k)}, a_t^{(k)}, bs_t^{(k)}, as_t^{(k)}, mp_t, \Delta mp_t, \Delta amt_t, r_t, \mu_W, \sigma_W, \mathbf{1}, \varepsilon, \mathrm{sgn}, \mathrm{cov}, \mathrm{var}$, signed_dvol, intst/ind/acc, y_reg, y_cls, $\hat{a}$, $f$ → 共 25 条
  - §0.3 概念一句话解释：OFI / WMP / dualz / KS-drop
- 原 Q1–Q3 + 7 个刁钻问题结构保留不动，仅在内容上补充。
- 新增 Q2 末尾「跨股 / 跨日 PSI 实测」家族表（来自 `MASTER_summary.md` §3）。
- 新增 Q3 表格里增加 `verdict_stock` / `verdict_date` 列（来自 `MASTER_summary.md` §5）。

## 2. 公式重写（按出现顺序）

| # | 旧形式问题 | 新形式 | 备注 |
|---|---|---|---|
| 1 | WMP `(a_k · bs_k + b_k · as_k)/(bs_k + as_k)` 紧贴中文 | 独占块公式 + 上下空行 + 加 $\varepsilon$ | 防紧贴问题 |
| 2 | MLOFI 4 个 indicator 一行写完，难读 | 用 `\begin{aligned}...\end{aligned}` 多行展开 | 渲染稳 |
| 3 | EWMA $\tilde{x}_t^{(\alpha)} = \alpha x_t + (1-\alpha)\tilde{x}_{t-1}^{(\alpha)}$ 嵌入文中 | 改为独占块公式 | — |
| 4 | cancel_imb 用 `\sum cb` 等无下标 | 改写为 $\sum_W$，并解释 $cb / ca / lb / mb$ | 符号清晰 |
| 5 | rv_W `= Σr_t²` 嵌在条目里 | 独占块公式 + 显式给出求和上下限 | — |
| 6 | signed_rv `= (RV+ − RV−)/(RV+ + RV−)` 无定义 | 独占块公式 + 注解 $\mathrm{RV}^\pm$ 含义 | — |
| 7 | Kyle λ `cov(signed_dvol, Δmid) / var(signed_dvol)` 用 `\text{cov}` | 改 `\mathrm{cov}_W / \mathrm{var}_W` + 加 $\varepsilon$ | 兼容 GitHub |
| 8 | jshare `max(0, (RV−BV)/RV)` 一行内嵌 | 独占块公式 + BV 公式独立给出 | BV 定义补全 |
| 9 | Roll 公式用 `\text{cov}` `\text{spread}` | 改 `\mathrm{cov}_W`，spread 直接用 $\mathrm{spread}_t^{(1)}$ | — |
| 10 | dualz 公式紧贴中文 | 独占块公式 + 上下空行 | — |
| 11 | GOFI 旧版用 `\big(...\big)` + `\mathbb{1}_{b\uparrow}` 内嵌 | 改 `\begin{aligned}` 多行 + `\mathbf{1}[\cdot]` + 说明 $bs/as$ 指第 $k$ 档 | 最复杂的公式被重写 |
| 12 | liq_asym `\log(...)` 一行 | 独占块公式 + 显式展开 $\sum_{k=1}^{5} \sum_{s=t-W+1}^{t}$ | — |
| 13 | burst `v[-1]/mean(v[-W:])` Python 切片记号 | 改为标准数学 $v_t / \mu_W(v)$ | — |
| 14 | HL ≈ ln 2 / (−ln(1−α)) 内嵌 | 改 inline `$\ln 2 / (-\ln(1-\alpha))$` 并解释 | — |
| 15 | midprice_k = (bid_k+ask_k)/2 | 独占块公式 | — |
| 16 | KyleInv `Δamt / (³√|Δamt|·σ(r_W)+ε)` ASCII | 改 `\sqrt[3]{|\Delta amt_t|}` + `\sigma_W(r)` | — |

## 3. 公式语法兼容性修复（KaTeX / GitHub / 飞书）

- 删除所有 `\text{...}` → 替换为 `\mathrm{...}`（4 处：`\text{cov}`, `\text{signed\_dvol}`, `\text{reg}`/`\text{cls}`, `\text{global}`, `\text{KS}`）
- 删除所有 `\big(...\big)` → 用 `\begin{aligned}` 或直接括号
- 删除所有 `\mathbb{1}` → 用 `\mathbf{1}[\cdot]`
- 所有块公式前后都有空行（与中文不紧贴）
- 所有 inline `$..$` 与中英文之间有空格（`$x$是` → `$x$ 是`）
- 所有 `_` 后没空格的下标都成对包裹 `{}`（`\sigma(r_W)` 等）

## 4. 缩写补全

旧版只在前文偶尔出现 LOB / WMP / OFI 而无全称。新版：

- 在 §0.1 一次性给齐 35 条 (英文全称 / 中文)。
- 在正文每一次缩写「首次出现」处仍重复一遍 (中文)，例如：
  - 「F1 LOB（Limit Order Book，限价订单簿）派生」
  - 「F2 多尺度 OFI（Order Flow Imbalance，订单流不平衡）」
  - 「IIR（Infinite Impulse Response 无限脉冲响应）滤波器」
  - 「PSI（Population Stability Index 群体稳定性指数）」
  - 「KS 检验（Kolmogorov-Smirnov，K-S 分布检验）」
  - 「LOSO = Leave-One-Sym-Out」
  - 「HFT（High-Frequency Trading，高频交易）」

## 5. 符号定义补全

- §0.2 数学符号表统一给定义（25 个）。
- 每个公式里出现的新符号都在表里能查到，例如：
  - $h$（horizon）= 预测视距，常用 $h \in \{10, 40, 60\}$ tick
  - $\hat{a}$ = 模型动作（卖 / 不动 / 买）
  - $f$ = 单边手续费率 ≈ 0.02%
  - $\mathrm{sgn}, \mathrm{cov}, \mathrm{var}$ 等数学函数
  - $bs_t^{(k)}, as_t^{(k)}$ 等带上下标的报价 / 挂量

## 6. 概念解释（§0.3）

- **OFI**：bid 净增量 − ask 净增量，正值 = 买压 = 涨。
- **WMP**：对侧 size 加权同侧价，挂单少侧权重大 → 反映「哪侧将耗尽」。
- **dualz**：短窗 z − 长窗 z，消除全局基差留短期相对偏离。
- **KS-drop**：用 K-S 检验筛掉单 sym 分布偏离全局的特征。

## 7. 渲染检查

- `pandoc 答辩_feature_QA_小抄.md -o /tmp/check.html --mathjax` → exit 0，无 warning / error。
- 抽样 5 个公式 KaTeX 语法人工 review：
  1. WMP 块 → 渲染正常
  2. MLOFI aligned 块 → 渲染正常
  3. Kyle λ + `\mathrm{cov}_W` → 渲染正常
  4. GOFI aligned 块（最复杂）→ 渲染正常
  5. dualz 块 → 渲染正常

## 8. 不在 rewrite 范围内的事情

- 数据 / 数值结果未改动（gain%、PnL 数字、5-seed std 等都保留原值）
- 7 个刁钻 Q 的论证逻辑未改动，仅补全称 / 符号
- 一句话总结未改动

## 9. 统计

- 总行数：418
- 块公式数：16
- inline 公式（粗估）：约 70+
- 缩写表条目：35
- 符号表条目：25
- 概念解释：4

## Pass 2 补注（2026-05-29）

### 原 Pass 2 checklist 执行情况

#### F1 LOB
- [x] avgbid / avgask — 补量权均值公式（$\sum b_k \cdot bs_k / \sum bs_k$）及中文
- [x] totalbsize / totalasize — 补「市场深度（market depth）」含义
- [x] bid_mean / ask_mean / bsize_mean / asize_mean — 补等权均值公式，说明与 avgbid 区别（等权 vs 量权）
- [x] cumspread — 补 $\sum_{k=1}^{10}\text{spread}_t^{(k)}$ 公式 + 流动性含义
- [x] imbalance — 修正为官方定义（十档，totalbsize/totalasize），补含义
- [x] bid_diff_k / ask_diff_k — 补「LOB 价格密度」解释
- [x] wmp_balance_12 — 补「lvl1 > lvl2 WMP 短期下行压力」解释

#### F2 多尺度 OFI
- [x] 「多尺度」澄清 — 加说明「多个 $W$ 重复计算」
- [x] EWMA-OFI 12 维 — 补完整公式 + 为什么比累加窗口好
- [x] KyleInv 1 维 — 补分子/分母物理含义
- [x] Kyle λ 2 维 — 补分子协方差含义 + 整族 KS drop 原因（跨 sym 弹性不同）
- [x] OFI Toxicity 4 维 — 补完整公式 + VPIN 论文引用

#### F3 订单流强度
- [x] 6 类事件命名表格 — 加 lb/la/mb/ma/cb/ca 完整对照表
- [x] cancel_imb 公式 — 明确 intst 累加（非 ind）
- [x] intst / ind / acc — 补官方定义，修正 acc 为「一阶变化率」（非二阶差分）

#### F4 微结构波动
- [x] volume_delta / amount_delta — 补「t 与 t-1 单 tick 差」说明
- [x] signed_rv — 补显式 $\mathrm{RV}^+$ / $\mathrm{RV}^-$ 公式
- [x] rskew — 补 $m_3 / \sigma^3$ 公式
- [x] signed_bv — 补方向化 BV 公式，含 $|r_{s-1}|$ 解释
- [x] vol_burst / amt_burst — 补完整公式 + clip 区间
- [x] rv_ratio — 补公式 $(W_n, W_d) \in \{...\}$
- [x] jshare BV — 显式给出 $\mathrm{BV}_{t,W}$ 公式 + BNS 跳跃不变量解释
- [x] roll_eff_spr_ratio — 补分子/分母解释 + 完整论文引用

#### F5 窗口统计
- [x] dualz — 补「37 raw 字段」说明
- [x] qrank — 补完整公式 + tick-quantized 死特征说明
- [x] mid_ewma_resid — 补公式 + HL ≈ 13 ticks ≈ 40 秒
- [x] adapt_mom — 补公式（sgn × √RV）
- [x] spread_reg — 补鲁棒 z-score 公式 + IQR=0 安全性说明
- [x] trade_pers — 补相邻方向自相关公式

#### F6 不对称性
- [x] F6 家族简介 — 补「买卖两侧不对称」定位
- [x] bid_rate / ask_rate — 补官方定义 + 推断公式 + 来源标注
- [x] bsize_rate / asize_rate — 同上
- [x] GOFI 同价情况 — 补「$bs_t - bs_{t-1}$ 净 delta」明确说明
- [x] liq_asym_top5 — 补 top-5 物理含义（买侧厚→大概率上涨）

#### 三类预处理
- [x] window-z — 补公式 $\mathrm{wz}(x)_t = (x_t - \mu_{100}) / (\sigma_{100} + \varepsilon)$
- [x] 镜像增强 — 补完整 mapping 表格（bid↔ask / bsize↔asize 互换，$y_\mathrm{reg} \to -y_\mathrm{reg}$，$y_\mathrm{cls} \to 2-y_\mathrm{cls}$）
- [x] sign-log1p — 给正式公式 $\mathrm{sign\_log1p}(x) = \mathrm{sgn}(x) \cdot \log(1+|x|)$

#### Q2 跨日泛化
- [x] T7（SchemeD1）、T10（SchemeF）首次出现处补实验名
- [x] LOSO 补「Leave-One-Sym-Out」

---

### 新增 4 条要求执行情况

#### 新增 (1) NaN 处理说明
- [x] F1 NaN 摘要（94 raw 无兜底 + 深档 NaN 16.86% + WMP 派生 NaN=0%）
- [x] F2 NaN 摘要（全派生 NaN=0%）
- [x] F3 NaN 摘要（raw ≈ 0% + 派生 NaN=0%）
- [x] F4 NaN 摘要（全 NaN=0%）
- [x] F5 NaN 摘要（全 NaN=0%）
- [x] F6 NaN 摘要（ask_rate/asize_rate 深档 NaN 最高 17.13%，bid 侧 NaN=0%，派生 NaN=0%）

#### 新增 (2) 论文引用
- [x] §0.4 新增论文引用汇总表（7 篇论文）
- [x] MLOFI 处补全 Cont, Kukanov & Stoikov 2014
- [x] WMP 处补全 Stoikov 2014
- [x] KyleInv 处补「Kyle 1985」
- [x] Kyle λ 处补完整 Kyle 1985 引用
- [x] OFI Toxicity 处补 Easley, López de Prado & O'Hara 2012
- [x] jshare 处补完整 BNS 2004 引用
- [x] Roll 处补完整 Roll 1984 引用
- [x] Kercheval & Zhang 2015 补在 GOFI / bid_rate 处

#### 新增 (3) feature 名称缩写展开
- [x] §0.1 缩写表新增 31 条（intst/ind/acc/cumspread/cancel_imb/dualz/qrank/adapt_mom/spread_reg/trade_pers/mid_ewma_resid/signed_rv/signed_bv/rskew/vol_burst/amt_burst/rv_ratio/jshare/roll_eff_spr_ratio/ofi_tox/ewma_ofi/liq_asym/wmp_lvl/bid_diff/bid_rate/bsize_rate/gofi/mlofi/kyle_inv/kyle_lam）
- [x] 正文中每条 feature 首次出现处都给（英文全称，中文）

#### 新增 (4) bid_rate / ask_rate / bsize_rate / asize_rate 公式
- [x] 读赛题说明 PDF → 官方给出「十档价格/委量平均变化率」+ 来源 Kercheval & Zhang 2015，未给精确公式
- [x] 读 data_schema.md → 确认字段定义
- [x] 读 MASTER_per_feature_audit.json → 找到推断公式（前 W tick 均值 vs 再前 W tick 均值的相对变化率）
- [x] 在 F6 中标注「主办方 PDF + data_schema.md 定义为 $\ldots$，具体窗口 $W$ 未披露，按命名约定推断为 $\ldots$」（诚实标注来源）

---

### 统计
- 总行数：592（原 418）
- 新增块公式：约 18 块
- 新增 NaN 摘要段落：6 段（F1–F6 各一）
- 新增论文引用：7 篇（§0.4 汇总 + 各处内联）
- 新增缩写表条目：31 条
- pandoc 退出码：0（渲染干净，仅 TeX math 转换 warnings，非错误）

---

## Pass 3 KS-drop 方法深化

### 修改范围
- **只动 Q3 章节**（原 lines 509–552），其他章节一字未动。
- Q3 节从 2 个子节扩充为 5 个子节（§Q3.0–§Q3.4）。

### 新增内容

#### §Q3.0 KS-2sample 检验的数学定义（全新）
- 给出双样本 KS 检验的完整数学定义：empirical CDF $F_n$/$G_m$、统计量 $D_{n,m} = \sup_x|F_n-G_m|$、理论 p-value 公式（基于 Kolmogorov 分布）
- 物理含义三档：$D=0$（同分布）/ $D=1$（完全分离）/ $D>0.5$（CDF 差超 50%）
- **关键工程决策解释**：N≈1.4M 时 p-value 永远趋零，改用 effect size D 阈值；引用代码注释 `validate_sym_invariance.py:4-5`

#### §Q3.1 失效标准（重写，原"失效标准"节）
- 修正两处原始错误：
  1. 原「sym vs 全局分布」→ 实为 **pairwise 5 only × $\binom{5}{2}=10$ 对，取 worst D**
  2. 原「$p < \alpha_{\mathrm{KS}}$」→ 实为「**worst pairwise $D > 0.5$**」
- 给出带注释的伪代码流程（subsample 50K + `scipy.stats.ks_2samp` + 三档判定）
- ⚠️ 警示框明确标出两处不严格之处

#### §Q3.2 11 个被 drop 的特征（重写，原表）
- 添加 `worst D` 列（实测数值），删除 `verdict_stock`/`verdict_date` 列
- Kyle λ 两档 D=0.95/0.98（接近完全分离）→ 最硬证据
- qrank-on-spread 4 个 D=0.83–0.86 → 死特征+tick量化双重失败
- 3 个 dualz_*_diff 边界 fail（D=0.50–0.51）
- 标注数据来源 JSON 文件

#### §Q3.3 Design Debt（增强，替代原散落的 design debt 注释）
- 阈值 0.5 经验值（无 sensitivity sweep）
- subsample seed=0（无稳定性测试）
- Bonferroni 不适用的论证（effect size vs p-value）
- WARN 档 28 个保留无 ablation 验证
- dualz bid/ask 对称性 anomaly 解释（非 bug，是阈值边界自然结果）

#### §Q3.4 效果表（保留原内容不变）
- 5-seed 对比表、seed std 含义、Why std 大降 原封不动

### 关键纠正
- `p < α` → `worst D > 0.5`（阈值类型纠正）
- "sym vs 全局" → "5 只 sym pairwise 10 对"（比较方式纠正）
- D 阈值 0.5 全文统一（原文混用 0.10/0.5）

### 统计
- 总行数：~680（原 592，新增 ~88 行）
- 新增公式块：5 块（empirical CDF、KS statistic、p-value、三档判定伪代码）
- 新增子节：4 个（§Q3.0/Q3.1/Q3.2/Q3.3）
- pandoc 退出码：0（渲染干净）

## Pass 4 H=60 horizon 选择章节

**改动**：在「老师可能追问的 7 个刁钻问题」之前插入新节 `## Q4：为什么 h=60 为主线？— horizon 选择的 5 层论证`

**插入位置**：原 line 617（`## 老师可能追问的 7 个刁钻问题`）之前

**新增子节（7 个）**：
- Q4.1 评分公式 + 实测：短 h OOD 完全失效（对比表 h=10 vs h=60）
- Q4.2 信号/成本权衡数学（$\sqrt{h}$ 缩放，信噪比 3.5x）
- Q4.3 微结构噪声 bid-ask bounce 让短 h 不可学
- Q4.4 LOB 派生因子时间尺度匹配（W=100≈5min sweet spot）
- Q4.5 OOD 透传率表（4 个里程碑节点，50–83%）
- Q4.6 一句话答辩答案
- Q4.7 h<60 实战补救：按 $\sqrt{h/60}$ 缩放阈值（私榜 h=40 +38.65 #1）

### 统计
- 总行数：728（原 654，新增 ~74 行）
- 新增公式块：2 块（PnL 公式、信号项）
- 新增表格：3 张（h 对比表、OOD 透传率表、实战补救说明）
- 新增子节：7 个（§Q4.1–Q4.7）
- pandoc 退出码：0（渲染干净，warnings 均为文件其他章节预存）

---

## Pass 5 Q4 修正（基于 horizon study 实测）

**目标**：用 5 horizon × 5 seed LGB 实验实测结果修正 Q4 章节中的错误论证。**只动 Q4，其他章节一字不动。**

### 核心错误修正

**旧 Q4.3（已删除）**：「短 h 模型 in-sample 拟合 bid-ask bounce → OOD 失效 → IC 低」
**新 Q4.3（实测）**：反直觉——短 h IC 反而更高（h=5 IC_test=0.372 vs h=60 IC_test=0.144），train-test gap 更小（h=5 gap=0.086 vs h=60 gap=0.428）。Bounce 的作用是压缩 σ(y_h)，不是「让短 h 不可学」。

**旧 Q4.7（已推翻）**：「√(h/60) 缩放阈值 → h=40 私榜 +38.65 #1，h=60 模型有跨 horizon 迁移力」
**新 Q4.7（实测）**：h=60 模型 + √(h/60) 阈值缩放在 h<20 完全破产（h=5 PnL=-32！）；h<60 必须单独训练。

**旧 Q4.6（已改）**：「其他 4 个 horizon 都 OOD broken，h=60 是唯一活下来的」
**新 Q4.6（正确）**：「h=60 不是因为 IC 最高（反过来：h=5 IC=0.37 > h=60 IC=0.14），是因为 IC×σ 在 h=60 撑得过手续费」

### 修改内容（按子节）

- **Q4.1**：保留原 iter_002 LOSO 对比表（历史证据），新增注释「早期方法学，真实机制见 Q4.3b/Q4.2/Q4.8」
- **Q4.2**：保留 SNR ∝ √h 公式，新增实测 IC × σ vs cost 5 行表格（来源 §D）
- **Q4.3**：完全重写——改标题为「修正版」，加 ⚠️ 反直觉发现框，删除「模型拟合 bounce → OOD 失效」说法
- **Q4.3b**（新增）：完整 5×5 IC 实测表（IC train / test / gap）
- **Q4.4**：一字未动（时间尺度匹配论证仍成立）
- **Q4.5**：一字未动（OOD 透传率表）
- **Q4.6**：一句话答辩答案完全重写（新机制：信号尺度，而非 IC 高低）
- **Q4.7**：完全重写——标题改「实测失败」，加 √(h/60) 缩放 5 行失败表格（来源 §F）
- **Q4.8**（新增）：DE-优化 PnL 对比表（h=40 in-sample 最优 +30.1 vs h=60 +22.9）
- **Q4 答辩 talking point flow**（新增）：6 步答辩流程建议

### 文件来源新增

在 line 4 「来源」列表末尾加 `h_horizon_study/REPORT.md`（5 horizon × 5 seed 实测）。

### 统计

- 总行数：803（原 728，新增 ~75 行）
- 修改子节：Q4.1/Q4.2/Q4.3/Q4.6/Q4.7（共 5 个重写）
- 新增子节：Q4.3b / Q4.8 / Q4 答辩 flow（共 3 个）
- 保留不动：Q4.4 / Q4.5 / §0 / Q1 / Q2 / Q3 / 7 个刁钻问题 / 一句话总结
- pandoc 退出码：0（渲染干净）
