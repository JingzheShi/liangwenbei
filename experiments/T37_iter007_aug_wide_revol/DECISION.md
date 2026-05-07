# T37 决策：不打包 iter_007

**时间**：2026-05-07 00:45

## 实验设计
| step | 配置 | LOSO h_60 5-seed avg + DE 4D | per_fold |
|---|---|---|---|
| stepA | aug [0.75, 1.25]，无 revol | **+12.94** | [1.67, 3.19, 1.74, 0.11, 6.23] 5/5 pos |
| stepB | stepA + `revol_wmp1_sigma_hat` | **+12.27** | [1.22, 2.66, 1.54, 0.10, 6.75] 5/5 pos |
| **iter_006**（现行最强）| aug [0.80, 1.20]，无 revol | **+13.61** | — |

## 决策
**不打包 iter_007。iter_006 (`submission_050606_iter006.zip`) 仍是最强候选。**

## 失败诊断
1. **T31 phase1 单 seed +0.81 优势在 5-seed ensemble + DE thresh 后反转**
   - 单 seed: 0.75-1.25 (+10.80) > 0.80-1.20 (+9.99)
   - 5-seed + DE: 0.75-1.25 (+12.94) < 0.80-1.20 (+13.61)
   - 解释：更宽 aug 提升单模型 robustness 但降低 ensemble 多样性，DE 阈值无法补偿
2. **revol_wmp1_sigma_hat 加进去反而跌 -0.67**
   - 单特征 LightGBM gain rank #1 不代表能加进生产特征集
   - 可能 revol 本来就被 226-d Scheme C 中的某些特征隐式表达

## 互补信号（可被 T38 利用）
stepA / stepB 的 per_fold profile 与 iter_006 显著不同：
- iter_006: 强在 sym=2（+6.96 我估的）
- stepA / stepB: 强在 sym=4（+6.2 / +6.8），sym=2 仅 +1.5-1.7

**3 路 ensemble** 可能各取所长。
