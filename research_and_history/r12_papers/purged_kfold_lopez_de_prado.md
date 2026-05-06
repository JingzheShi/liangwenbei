# Purged K-Fold + Embargo + Combinatorial CV (López de Prado)

**作者**：Marcos López de Prado
**书**：*Advances in Financial Machine Learning*, Wiley 2018, Ch. 7
**Wiki**：[Purged cross-validation](https://en.wikipedia.org/wiki/Purged_cross-validation)
**实现**：[skfolio.CombinatorialPurgedCV](https://skfolio.org/generated/skfolio.model_selection.CombinatorialPurgedCV.html)

---

## 解决的问题

金融 ML 标签**依赖未来事件**：例如 label_60 用 t+60 的 midprice 算 → t 时刻的 feature 与 t+60 的 label**在时间上重叠**。
随机 K-Fold 会把 t 放在 train、t+30（label 还在和 t 重叠）放在 test → **leakage**。

López de Prado 在 *Advances in FinML* Ch.7 给出系统的解决方案。

## 三个技术

### 1. Purging

如果一个 train 样本的 label 形成时间区间与 test 样本重叠 → 把它**从 train 集合中移除**。

对我们 label_60：
- test sample at time t 的 label 用了 [t, t+60] 的 midprice
- 任何 train sample at time t' 满足 t' ∈ [t-60, t+60] 都要 purge

### 2. Embargo

即使 train 和 test 的 label 时间不直接重叠，**市场 reaction lag** 仍可能让 t+60+ε 的 sample 携带 t 的信息。
解法：在 test fold 之后再禁掉一段时间（典型 1%~5% of total time）。

对我们：t+60 之后再 embargo 100~300 ticks。

### 3. Combinatorial Purged Cross-Validation (CPCV)

普通 K-Fold 5 次给 5 个 OOF；CPCV 用所有 C(N,k) 组合（如 C(6,2)=15 个），每次以两个 fold 当 test，其它当 train（purged + embargoed）→ 得到 **多条 OOF 路径** 而不是一条。

每条路径 = 一个完整的 train→test 时间序列。
统计意义：把 OOF 性能从"单一估计"升级为"分布"，可做 hypothesis test、confidence interval。

## 对我们 LOSO 的关系

我们当前用 **LOSO (Leave-One-Sym-Out)** —— 这其实是 *cross-section* purging：每 fold 把一个 sym 整体拉出来当 test，**不存在时间重叠**（每个 sym 完整时间序列都在 train 或 test，互斥）。
所以**我们已经隐式做了 purging**——这是为什么 LOSO 是平台行为最忠实代理（T4 验证：LOSO -22.10 vs 平台 -23.13）。

但 LOSO 还能加强：

### 改进 1：跨 sym + 跨时间双 purging

LOSO + 在 holdout sym 内部再做时间 split（前 80% 当 LOSO test，后 20% 不用，避免 future-leak through cross-sym signal propagation）。

### 改进 2：LOSO + bootstrap = "Combinatorial sym CV"

5 个 sym 取 2 个当 holdout、3 个当 train → C(5,2)=10 个 fold，每个 fold 都做 stacking OOF。
这能极大缓解 "5 LOSO 数据点太少导致 stacking weight 不稳" 的问题（方案 12 / Yao 2018 提到的痛点）。

### 改进 3：embargo 的"sym 等价"

如果某些 sym 之间存在 cross-asset 协动（比如同板块），LOSO holdout sym=2 时其它 sym 可能携带 sym=2 的信息——这时需要在 train 里再 purge "cross-correlated" 的 sample。但这要先做 EDA，识别哪些 sym 协动；目前我们没有这种证据，**LOSO 默认够用**。

## 为什么对 ensemble / stacking 至关重要

Stacking 最大的失败模式是 **meta-learner 在 leaked OOF 上学到 fake signal**。
正确流程：

1. 第一层：base 模型在 train fold 训练（已 purge + embargo）。
2. 在 holdout fold 生成 OOF prediction。
3. 把所有 fold 的 OOF concat → meta target。
4. Meta-learner 在这个**真正未泄露的** prediction 矩阵上训。
5. Test 时：base 用全数据重训（or 5-fold 平均），meta 套上去。

任何一步偷懒就漏 → meta 会"看穿"base 的 train memorization → test 时崩。

## 对方案 2 (stacking) 的硬性要求

- ✅ 每个 base 在每个 LOSO fold 上独立训练，**绝不能用见过该 sym 的 model 给该 sym 出 OOF**。
- ✅ Meta-learner 只在 OOF 上训，**不在 in-fold prediction 上训**。
- ✅ 推理时用全数据重训的 base + meta。否则 test set 上 base 见过的样本比 train 时多 → distribution shift。

## 关键 takeaway

我们已经在用 LOSO（自动 purging）；下一步上 stacking 时只需保证 meta 流程正确，无需新写 purged time-series CV。
但**要避免在 LOSO 5 fold 上拟合 K=10 模型的 stacking weight**——5 数据点学 10 维过拟合严重；改用 simplex constrained / bootstrapped weight / 限制 K ≤ 5 的 base 数量。
