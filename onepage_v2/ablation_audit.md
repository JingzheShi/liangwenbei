# `one_page_summary.tex` Ablation 表 Audit 报告

**日期**：2026-05-26
**Auditor**：worker (亲自完成，未派 sub-worker)
**时间预算**：≤ 2 小时
**WandB project**：`liangwenbei-ablation-rerun` (entity `cjxh21-Tsinghua University`)

---

## Step 2 — 10 张 ablation 表 audit

| # | 表 (caption 简称) | 当前数据 (setting → metric) | Eval | Audit 结论 |
|---|---|---|---|---|
| T1 | **特征家族** | 154d → 226d → 359d, $-22.10$/$+21.86$/$+26.44$ | LOSO 5-fold + LOSO-eq **混用** | 跨 protocol，注释中已自承"$\sim+7$ 方法论位移" — **不重做**，时间不够 |
| T2 | **窗口内 z-score** | SchemeB no/with z-score, $+11.11 \to +13.91$ | LOSO 5-fold | 同 backbone 同 eval，**干净**；历史 alpha，不重做 |
| T3 | **方向对称增强 aug_a** | $+6.30 \to +9.67 \to +11.46$ (no-aug → +mirror → +5-seed bag) | LOSO 5-fold | 同 backbone；多步串联但每步含 caption 说明，可读，**不重做** |
| T4 | **重尾压制 sign-log1p** | 仅给出"$10^3$–$10^6 \to 5$–$15$"幅度，**无 PnL ablation** | "diagnostic" | **P1 重做**：补做 raw vs log1p amount_delta 的 LGB L2 PnL 对照 |
| T5 | **NN 架构** | MLP+LGB $+28.16$ vs +GRU $+25.08$ vs GroupTransformer holdout $+149.91$ | public + holdout 混 | caption 已注明 iter\_016 同改 GRU+OOF DE；GroupTransformer in-sample → **不重做**（NN 训练贵） |
| T6 | **树模型目标** | CE $+26.44 \to$ L2 $+36.23$ | LOSO-eq, "clean swap" | **P2 重做**：缩规模在 V4 同 backbone 复现 (CE vs L2 同 SchemeP 同 train/val split) |
| T7 | **SPO+ DFL** | L2 only $+19.23 \to$ +SPO+ $+28.16$ | public | NN 训练贵，**P5 跳过** |
| T8 | **M7 retrain** | train 0-79 $+28.93$ → M7 $+34.44$ | public | M7 训于 0-119 全数据，public 是 in-sample → **P4 重做**：本地 V4 train 0-79 vs M7 on holdout 96-119 OOD 对照 |
| T9 | **Ensemble 规模** | 5×5 $+34.64$ / 50×50 $+35.64$ / 150×150 $+34.59$ | public | 50×50 / 150×150 重训不可行（项目原成本 ~小时级） — **不重做** |
| T10 | **执行层 threshold** | symmetric $+33.72$ → asym $+36.23$ | LOSO-eq | **P3 重做**：用 P2 L2 预测的 val 阈值 sym vs asym 对照 |

---

## Step 3 — 补做实验进度

(待补写：实验完成后填入下方)

### P1 — `amount_delta` sign-log1p ablation
- Backbone: **154 raw-last** (无 216 extras)，LGB L2 regression on Δmid, train 0-79
- Eval: **V4 walk-forward holdout = dates 80-95**, h=60 cumulative PnL，asymmetric EV gate (val 2D grid search)
- 状态：训练中
- WandB run: `p1_amount_delta_raw` / `p1_amount_delta_log1p`

### P2 — L2 vs 3-class CE
- Backbone: **SchemeP 370-d** (含 216 extras)，train 0-79, eval val 80-95
- 同 LGB HP / num\_rounds，仅切换 objective
- 状态：370-d cache 构建中

### P3 — Symmetric vs Asymmetric threshold
- 用 P2 L2 预测在 val 上的 ŷ
- (a) symmetric: $k \cdot \overline{|\hat y|}$，k 网格 [0.2, 3.0]
- (b) asymmetric: 21×21 2D 网格搜
- 状态：依赖 P2

### P4 — M7 retrain vs train 0-79
- (a) train 0-79，eval test 96-119 h=60 PnL (OOD)
- (b) train 0-119 (M7, iters × 1.1)，eval test 96-119 h=60 PnL (in-sample, 加 caveat)
- 状态：依赖 P2 cache

---

(下面 Step 4 / Step 5 / Step 6 完成后补写)
