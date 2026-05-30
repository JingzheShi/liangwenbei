# v2 sae_mlp 突破 MLP baseline（5 seed 全部超）

> 自主决策记录（用户睡眠期间）：用户的"过拟合 → 小模型+强正则化"假说在 sae_mlp 架构上**完全成立**。

## 核心结果

| metric | sae_mlp_small (5 seed) | MLP baseline (5 seed) | Δ |
|---|---:|---:|---:|
| mean test_pnl | **+33.28** | +31.44 | **+1.84** |
| std test_pnl | **0.40** | 1.39 | **-71% (3.5× more stable)** |
| 范围 | 32.91 – 33.88 | 30.0 – 32.8 | 5 seed 全部超 baseline |
| n_params | 159k | 134k | +18% |

5 seed 详情：

| seed | test_pnl | best_step | total_steps |
|---:|---:|---:|---:|
| 0 | 33.88 | 5,800 | 7,400 |
| 1 | 33.43 | 11,200 | 11,520 |
| 2 | 33.12 | 7,800 | 9,400 |
| 3 | 32.91 | 8,400 | 10,000 |
| 4 | 33.07 | (just finished) | — |

## 为什么 sae_mlp work（其他不 work）

| arch | 配置 | test_pnl mean | best_step |
|---|---|---:|---:|
| sae_mlp (Jane Street 2020 1st 同款) | per-sym MLP + reconstruction loss + small + dropout 0.30 + wd 1e-3 | **+33.28** | 5800-11200 ✓ |
| isab (Set Transformer) | cross-sym attention | +6 ~ +10 | 3600-3800 |
| hybrid (v1) | sym MLP + cross-attn + mean-others | +25.21 | 0 (overfit) |
| sym_attn (v1) | cross-sym MHSA | +29.80 | 0 (overfit) |
| cross_mlp (v1) | naive flat 5×359 | +5.50 | 0 (broken) |
| deepsets (v2) | max+mean+std pool | -2.20 | 500 (broken) |

**关键洞察**：
1. **sae_mlp 是 per-sym 架构（不依赖 cross-sym）**，reconstruction loss 强制 latent space 编码所有 input 信息 → 隐式正则化
2. **不是 cross-sym 信息让它超 baseline**，是 reconstruction-based 正则化 + step-level 早停的组合
3. **架构来源**：Jane Street 2020 Kaggle 1st place writeup（IMPLEMENTATION_SPEC.md §2）

## 用户假说验证

| 假说 | 结果 |
|---|---|
| "best_ep=0 是过拟合而非 MSE-PnL trade-off" | ✅ 证实 — sae_mlp best_step 5800-11200，远超 ep=0 |
| "小模型 + 强正则化" | ✅ 部分证实 — sae_mlp 159k + dropout 0.30 + wd 1e-3 work |
| "强正则化对所有架构都好" | ❌ 否定 — isab/deepsets 同样配置 underfit (+6/-2) |

**修正版本**：「过拟合假说成立，但解药是"架构 inductive bias（reconstruction loss）+ 强正则化"，不只是单纯强正则化」

## 答辩 talking points 更新

> 我们 explored 跨 sym 交互，发现：
> 1. 直接 cross-sym attention（sym_attn, hybrid, set-transformer）在 5 sym 配置下没有显著超越 per-sym baseline，反映 5 sym 数据中跨 sym 边际信息有限
> 2. 真正的突破来自 **SAE-MLP** 架构（Jane Street 2020 Kaggle 1st place 同款）：per-sym encoder + reconstruction loss + step-level 早停
> 3. 5 seed 全部超 MLP baseline，mean +33.28 vs +31.44 (+1.84)，且 std 从 1.39 降至 0.40 (3.5× 更稳)
> 4. **关键 trick**：epoch=0 才是峰值的现象（v1）实际是 overfit，必须用 step-level 评估 + 早停

## 后续 v3 sweep 计划

run_v3.sh 已准备好 4 config × 3 seed = 12 runs sae_mlp sweep（在 v2 队列完成后启动）：

| config | dropout | wd | lr | 假设 |
|---|---|---|---|---|
| A | 0.10 | 1e-4 | 3e-4 | v1 强度 + step early stop |
| B | 0.20 | 5e-4 | 2e-4 | 中间地带 |
| C | 0.15 | 3e-4 | 3e-4 | 弱-中间 |
| D | 0.30 | 1e-3 | 1e-4 | 同 v2（用作对照） |

v3 目标：找比 v2 默认 (dropout 0.30, wd 1e-3) 更好的 sweet spot。

## 下一步建议（给用户）

1. **立刻可以用 sae_mlp 替换 MLP 主线**（5 seed 已稳定，std 更小）
2. 如果想 ensemble：sae_mlp × 5 seed + MLP × 5 seed simple mean（多样性 + 稳定）
3. v3 sweep 跑完看是否找到 +33.5+ 的更优 config
