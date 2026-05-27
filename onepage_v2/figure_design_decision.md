# Feature_Img.pdf 设计决策

## 选择方案：候选 B（3-panel 横向并排）

### 三个 panel
1. **左 Panel：6 家族 dim 数 + gain%**
   - 双柱（横向）：每个家族两根 bar，#dim（浅色）和 gain%（深色 + lblue 系），共享 y 轴
   - 在 bar 上直接写数值，避免重复读图例
2. **中 Panel：Top-10 因子 横向 bar**
   - 每根 bar 按家族着色，与左侧一致，名称在 bar 内部
   - 直观显示：F3（4 个）+ F1（6 个）= Top10 全部，对应 75.5% gain share
3. **右 Panel：6×6 family corr heatmap**
   - 显示 family 间 block-mean |corr|；对角线为族内 |corr|
   - F5 行/列偏暗（≤0.07），视觉佐证 OOD 稳健性

### 为什么这样
- 一张图同时覆盖 PM 提出的 3 个目标：结构 / 主力 / 正交性
- 3 panel 横向并排适合单栏 \linewidth（≈ 9.8 cm × 4.8 cm），密度高但 panel 内不拥挤
- 颜色统一：F1-F6 同一 palette，跨 panel 视觉链路：左看占比、中看实例、右看相关
- 与 onepage lblue 主色协调（F1 LOB 派生作为头号家族用主蓝色）

### 字号
- 标题 7pt，标签 6.5pt，数值 6pt
- 最小 6pt，配合 \scriptsize tex 主体

### 尺寸
- 物理：宽 9.8cm × 高 4.5cm（fig: 3.86" × 1.77"）
- 输出 PDF（矢量），无栅格化
