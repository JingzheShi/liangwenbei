# T38 决策：不打包 iter_007

**时间**：2026-05-07 ~01:30

## 实验
13 combo × DE 4D thresh，3 路 OOF 概率：
- M1 = iter_005b 5-seed avg (aug [0.80, 1.20])
- M2 = T37 stepA 5-seed avg (aug [0.75, 1.25])
- M3 = T37 stepB 5-seed avg (stepA + revol_wmp1_sigma_hat)

## 完整 ranking

| Rank | combo | sum | n_pos | std |
|---|---|---|---|---|
| 1 | **M12_w_6_4** | **+13.62** | 5/5 | 2.62 |
| 2 | M1_iter005b | +13.59 | 5/5 | 2.57 |
| 3 | M12_avg | +13.55 | 5/5 | 2.60 |
| 4 | M12_w_7_3 | +13.50 | 5/5 | 2.67 |
| 5 | M123_w_7_15_15 | +13.39 | 5/5 | 2.54 |
| 6 | M123_w_6_2_2 | +13.38 | 5/5 | 2.48 |
| 7 | M123_w_4_4_2 | +13.29 | 5/5 | 2.41 |
| 8 | M123_w_5_25_25 | +13.29 | 5/5 | 2.38 |
| 9 | M123_avg | +13.22 | 5/5 | 2.46 |
| 10 | M123_w_4_2_4 | +13.19 | 5/5 | 2.47 |
| 11 | M13_avg | +13.11 | 5/5 | 2.42 |
| 12 | M2_stepA | +12.92 | 5/5 | 2.07 |
| 13 | M23_avg | +12.82 | 5/5 | 2.26 |
| 14 | M3_stepB | +12.26 | 5/5 | 2.30 |

## 决策：不打包 iter_007

**理由**：
1. 最佳 M12_w_6_4 (+13.62) 比 M1 单独 (+13.59) 仅 +0.03
2. M1 重测 +13.59 vs iter_006 历史 +13.61 已差 0.02 → DE 噪声 ±0.05
3. +0.03 在噪声内 → **改进不可信**
4. 不浪费 12h cooldown 提 marginal 改进

## 关键洞察

1. **加 revol (M3) 全面有害**：M123 < M12 always，M3 < M2 alone。revol_wmp1_sigma_hat 在单模型 LightGBM gain rank #1 是欺骗信号
2. **stepA 比独立 useful**：M12 > M1（虽然 +0.03）说明 [0.75,1.25] aug 有少量正交信号
3. **diversity 不是瓶颈**：13 combo 里没有任何超 +0.5 的，说明 226-d Scheme C 的可提取 alpha 已被 iter_005b 5-seed 榨完

## 下一步建议（不立即 dispatch，等平台反馈）

1. 等 iter_006 提交后的平台反馈 → 校准 LOSO ↔ 平台 gap
2. 看历史冠军 / Kaggle 顶级方案找新特征思路（R31 提到的 video writeup 还没 mining）
3. CatBoost 5-seed × LightGBM 5-seed 跨模型 ensemble
4. 训练时合并 train+val 给 final model 使用
