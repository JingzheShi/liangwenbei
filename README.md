# 第二届良文杯 — LOB 中间价方向预测

> **Liangwenbei 2026 — LOB Mid-Price Direction Prediction**
> **最终成绩**：+40.13 LOSO-equiv（平台提交 iter_016，50 NN + 50 LGB 集成）
> **比赛时间**：2026-05-01 ~ 2026-05-12（7 天比赛 + 答辩）

## 项目概述

本项目解决 A 股高频 LOB（Limit Order Book）中间价方向预测问题：给定当前时刻 10 档量价数据（154 维 raw feature），预测未来 h=5/10/20/40/60 ticks 的中间价变动方向（上/平/下），评估指标为按 |Δmid| 加权的 PnL。

### 核心贡献

| 贡献 | 量化增益 |
|------|---------|
| 多尺度 OFI / WMP / BNS 微观结构特征（359 维） | F2/F3/F5 特征体系 |
| Δmid 回归 + 非对称 EV gate（取代分类 CE） | +9.79 LOSO（iter_013） |
| SPO+ 决策焦点学习（T87 DFL，LGB → NN 微调） | +1.85 LOSO（iter_015） |
| 50 NN × 50 LGB 异质集成，简单加权均值 | +1.66 LOSO（iter_016） |

### 关键约束（违反 = 提交报错或 0 分）

1. `date` 字段在评测时被置 0 → **不能用 date 特征**
2. 测试点顺序被打乱 → **Predictor 必须 stateless**（无跨调用状态）
3. sym 可能含训练外新股 → **模型必须 sym-agnostic**（无 sym embedding）

---

## 仓库结构

```
liangwenbei_workdir/
├── final_submission_code/    # 复现最终提交的完整流程
│   ├── 01_build_features/    # 特征构建（359 维）
│   ├── 02_train_lgb/         # 50 LGB seeds
│   ├── 03_train_nn/          # 50 NN seeds（T87 SPO+）
│   └── 04_build_pkg/         # Predictor.py 打包
├── experiments/              # 200+ 实验记录（T* / R* 目录）
├── research_and_history/     # 研究笔记（132 个 r*.md + papers/）
├── proposals/                # W_* / T10x blinded 提案
├── slides/                   # 答辩幻灯片（slides.tex / slides.pdf）
├── onepage_v2/               # One-page summary
├── h_horizon_study/          # 5 horizon × 5 seed LGB 实测
├── REFERENCES.md             # 完整参考文献库（130+ 条）
├── REFERENCES.pdf            # 参考文献 PDF 版
├── CRITICAL_CONSTRAINTS.md   # 三大硬约束文档
├── 答辩_feature_QA_小抄.md   # 答辩 Q&A 小抄
└── key_designchoices_and_tricks.md
```

---

## 参考文献 / References

完整参考文献见 [REFERENCES.md](./REFERENCES.md)（130+ 条，§1–§12 分类）。

以下为各类别最核心的 10 条引用：

### 微观结构理论

- **[Cont, Kukanov & Stoikov 2014]** The Price Impact of Order Book Events. *Journal of Financial Econometrics*. [arXiv:1011.6402](https://arxiv.org/abs/1011.6402) — MLOFI 30 维特征的理论基础 ✅
- **[Stoikov 2018]** The Micro-Price. *Quantitative Finance*. [arXiv:1702.02867](https://arxiv.org/abs/1702.02867) — WMP 11 维特征 ✅
- **[Easley, López de Prado & O'Hara 2012]** Flow Toxicity and Liquidity. *Review of Financial Studies*. — VPIN / OFI Toxicity ✅
- **[Barndorff-Nielsen & Shephard 2004]** Bipower Variation. *Journal of Financial Econometrics*. — BV + jshare 跳跃检验 ✅

### ML for LOB

- **[Zhang, Zohren & Roberts 2018]** DeepLOB. *IEEE Trans Signal Processing*. [arXiv:1808.03668](https://arxiv.org/abs/1808.03668) — 比赛 baseline ⚠️
- **[Berti et al. 2025]** TLOB/MLPLOB. [arXiv:2502.15757](https://arxiv.org/abs/2502.15757) — 简单 MLP 达 SOTA 验证 ⚠️

### 决策焦点学习

- **[Elmachtoub & Grigas 2022]** Smart Predict, Then Optimize (SPO+). *Management Science*. [arXiv:1710.08005](https://arxiv.org/abs/1710.08005) — **T87 核心 loss，+1.85 LOSO** ✅

### OOD / 数据增强

- **[Zhang et al. 2018]** Mixup. *ICLR 2018*. [arXiv:1710.09412](https://arxiv.org/abs/1710.09412) — 跨 sym Mixup 数据增强 ✅
- **[Sagawa et al. 2020]** Group DRO. *ICLR 2020*. [arXiv:1911.08731](https://arxiv.org/abs/1911.08731) — sym-aware 训练权重 ⚠️

### 工具与比赛参考

- **[Yirun 2021]** Jane Street 1st Place. [Kaggle writeup](https://www.kaggle.com/competitions/jane-street-market-prediction/writeups/cats-trading-yirun-s-solution-1st-place-training-s) — Mixup 工业验证 ✅
- **[LightGBM: Ke et al. 2017]** *NeurIPS 2017*. [arXiv:1711.01830](https://arxiv.org/abs/1711.01830) — 主 GBDT 框架 ✅

> **注**：iTransformer / PatchTST / TimesNet 等 TS 预测模型均已调研但**明确拒绝**用于 LOB tick 级预测（详见 REFERENCES.md §3）。

---

## 快速复现最终提交

```bash
cd final_submission_code
bash run_pipeline.sh      # 端到端：特征 → 训练 → 打包
# 输出: 04_build_pkg/submission.zip
```

详细步骤见 [final_submission_code/README.md](./final_submission_code/README.md)。

---

## 答辩资料

- [slides/slides.pdf](./slides/slides.pdf) — 答辩幻灯片
- [onepage_v2/one_page_summary.pdf](./onepage_v2/one_page_summary.pdf) — One-page summary
- [答辩_feature_QA_小抄.md](./答辩_feature_QA_小抄.md) — Q&A 备忘
- [factor_families_report_v4.pdf](./factor_families_report_v4.pdf) — 因子工程详细报告

---

*项目由 Claude Code Agent System 辅助完成，200+ 实验，WandB 项目：`liangwenbei` (entity: cjxh21-Tsinghua University)*
