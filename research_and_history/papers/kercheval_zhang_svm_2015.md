# Modelling High-Frequency Limit Order Book Dynamics with Support Vector Machines

- **Authors**: Alec N. Kercheval, Yuan Zhang
- **Venue**: Quantitative Finance 15(8), 2015, pp. 1315–1329
- **Links**: https://www.tandfonline.com/doi/abs/10.1080/14697688.2015.1032546 ; FSU PDF: https://www.math.fsu.edu/~aluffi/archive/paper462.pdf

## 核心贡献

1. 第一篇系统性把 LOB 用机器学习预测的 paper，**为后续所有 LOB DL 工作奠定了 144 维特征 schema**。
2. 用 multi-class SVM 在两个任务上做：(a) 中价 mid-price 移动方向，(b) bid-ask spread 跨越 / 价差。
3. 提出 **三类手工特征**——这套特征后来被 FI-2010 benchmark 采纳，成为 LOB 数据集的标准 schema：
   - **Basic set** (40 维)：raw 10 档量价
   - **Time-insensitive set**：spread, mid-price, price differences, mean prices/volumes, accumulated differences
   - **Time-sensitive set**：price/volume derivatives over time, intensities, accelerations, relative intensity indicators

## 144 特征 schema（关键参考）

虽然原文表格在我们的 fetch 里读不到，但根据 FI-2010 论文（Ntakaris 2017）和后续整理，标准 144 维特征大致是：

### v1: Basic Set (40-d, raw)
```
{p_a^i, v_a^i, p_b^i, v_b^i}_{i=1..10}
```

### v2: Time-insensitive (~32-d)
- bid-ask spread per level: `p_a^i - p_b^i`
- mid-price per level: `(p_a^i + p_b^i) / 2`
- price differences:
  - `p_a^10 - p_a^1`, `p_b^1 - p_b^10`（10档 vs 1档价差）
  - `|p_a^{i+1} - p_a^i|`, `|p_b^{i+1} - p_b^i|` for i=1..9
- mean prices/volumes:
  - `mean(p_a^1..p_a^10)`, `mean(p_b^1..p_b^10)`, `mean(v_a^1..v_a^10)`, `mean(v_b^1..v_b^10)`
- accumulated differences:
  - `Σ p_a^i - Σ p_b^i`, `Σ v_a^i - Σ v_b^i`

### v3: Time-sensitive (~多维)
- price/volume derivatives:
  - `dp/dt`, `dv/dt` per level
- order arrival intensities:
  - `λ_lo^a`, `λ_lo^b`, `λ_mo^a`, `λ_mo^b`, `λ_co^a`, `λ_co^b`（limit, market, cancel × ask/bid）
- relative intensity indicators (1{λ_short > λ_long})
- intensity accelerations / spread crossings

## 对我们比赛的可借鉴点

**这套 schema 几乎完全对应我们的现有 154 维特征**！可以一一对应：

| Kercheval & Zhang 类别 | 我们 schema 中的对应 |
|---|---|
| Basic raw 量价 | bid1..bid10, ask1..ask10, bsize*, asize* |
| Per-level spread | spread1..spread10 |
| Per-level mid | midprice1..midprice10 |
| 价差 / 邻级差 | bid_diff*, ask_diff* |
| mean price/vol | bid_mean, ask_mean, bsize_mean, asize_mean |
| accumulated/cum spread | cumspread |
| derivative 成交价 | bid_rate*, ask_rate*, bsize_rate*, asize_rate* |
| arrival intensities | *_intst (lb, la, mb, ma, cb, ca) |
| intensity indicators | *_ind |
| intensity accelerations | *_acc |

**结论**：我们的 154 维 schema 已经是 Kercheval & Zhang 144 + 一些扩展（avgbid, avgask, totalbsize 等）。剩下的优化方向是：
1. 加 OFI / MLOFI（schema 里没显式有，需要从 bid_diff/ask_diff 拼）
2. 加 weighted mid-price / micro-price（schema 没有）
3. 加多窗 rolling stats / EWMA（schema 没有）
4. 加跨 sym 截面排名特征

## 我的评分

- 实现成本：low（schema 已经基本提供了）
- 优先级：参考价值 **high**（理解我们 schema 出处）；新增特征 priority 看 feature_ideas.md
