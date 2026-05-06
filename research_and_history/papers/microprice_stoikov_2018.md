# The Micro-Price: A High Frequency Estimator of Future Prices

- **Author**: Sasha Stoikov (Cornell)
- **Year**: 2017/2018
- **Links**: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2970694 ; slides: https://www.ma.imperial.ac.uk/~ajacquie/Gatheral60/Slides/Gatheral60%20-%20Stoikov.pdf

## 核心贡献

1. 定义了 LOB 中"未来价格的 martingale 估计"——micro-price（M_t）：当前所有 LOB 信息下，mid-price 的长期条件期望。
2. 关键变量：**imbalance** I_t = bsize_1 / (bsize_1 + asize_1)，**spread** S_t = ask - bid。
3. 给出 micro-price 的迭代公式：M_t = (mid_t) + g(I_t, S_t)，g 是从 mid-price 跳跃概率拟合的修正项。
4. 实证证明 micro-price 比 mid-price 和 weighted mid-price 都是更准的 short-term predictor，并且本身是 martingale。

## 关键公式

**Weighted mid-price (WMP)**（baseline）：
```
WMP = bid · asize / (bsize + asize) + ask · bsize / (bsize + asize)
```
注意是反过来加权的！bid 用 ask size 权重，ask 用 bid size 权重——这反映了 imbalance 推动方向。

**Micro-price**（迭代版本）：
```
M_t = mid_t + Σ_{k=0..∞} G_k(I_t, S_t)
G_k 是从经验数据估计的"k 步后的 mid-price 跳跃期望"
```
实际工程上，常见近似是：
```
M_t ≈ ask · bsize / (bsize+asize) + bid · asize / (bsize+asize)   (WMP)
```
或者 Stoikov 推荐的迭代估计 G。

## 对我们比赛的可借鉴点

- **WMP 必加**：上面的 weighted mid-price 是几乎所有 HFT 论文都用的基础特征。我们 schema 里没有，必须加。
- 多档 weighted mid-price：用 level k 的量价：
```
WMP_k = bid_k · asize_k / (bsize_k + asize_k) + ask_k · bsize_k / (bsize_k + asize_k)
```
- 把 micro-price 当目标的"无偏估计"——比 mid-price 噪声小，可以用作训练的目标平滑（但比赛 label 已固定，做监督需要小心）。
- **imbalance feature** I = bsize / (bsize + asize) 是 LOB 最 robust 的方向预测器。

## 我的评分

- 实现成本：low (WMP)；medium (full Stoikov micro-price 需要拟合 G)
- 优先级：**high（top 5 必选特征）**
