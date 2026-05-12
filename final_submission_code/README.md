# T188v2 50+50 全 Horizon 提交流程

## 概述

本代码库完整复现 **第二届"良文杯"** SOTA 提交 `submission_050911_iter019_v2N_50plus50_optimized_fullhorizon.zip`，平台得分 **+35.64**（h=60 列，5 horizon 取最优）。

### 模型结构

- **50 NN**（T87 SPO+ DFL）：每个 seed 独立完成 T81-style L2 预训练（dates 0-79，val dates 80-95，早停）后，T170 协议 SPO+ 决策焦点微调（全数据 M7 dates 0-119，固定 11 epoch）。
- **50 LGB**（T75 回归）：5 个 HP 配置 × 10 seeds，M7 全数据重训，目标为 Δmid 回归，num_boost_round=330。
- **集成**：简单均值（w_NN=1.0, w_LGB=1.5），per-sym beta conformal abstain band。
- **全 horizon 发射**：h=5/10/20/40 共享 h=60 ensemble 的预测，阈值按 sqrt(H/60) 缩放；h=60 直接使用原始阈值。平台取 5 horizon 最优计分，添加短 horizon 只会保持或提升总分。

### 平台得分

| horizon | 策略 | 平台 PnL |
|---------|------|---------|
| h=60 | 主 ensemble（thr_up=0.0003, thr_dn=0.000216） | **+35.64** (SOTA) |
| h=5/10/20/40 | share_with=60，sqrt 缩放阈值 | 不影响 h=60 得分 |

---

## 环境要求

- **Python 3.10+**（推荐 3.11）
- **GPU（训练阶段）**：建议 RTX 3080 / 3090（10GB+ VRAM）
  - 5 台 GPU 并行训 50 NN ≈ 18 min；单卡 ≈ 90 min
  - LGB 也可用 GPU 加速（`--gpu` flag），50 seeds ≈ 12 min；CPU ≈ 30 min
- **CPU 推理（提交阶段）**：requirements.txt 使用 CPU-only torch，轻量快速
- **磁盘**：~50 GB（特征 cache ~20 GB + NN/LGB 模型 ~5 GB + 原始数据 ~0.5 GB）
- **RAM**：16 GB+（特征构建时需加载所有 parquet）

### 安装依赖

```bash
# 训练环境（需要 GPU torch）
pip install numpy pandas lightgbm scipy torch

# 提交包运行环境（CPU-only，由 requirements.txt 安装）
pip install -r 04_build_pkg/requirements.txt
```

---

## 数据准备

**本仓库不含训练数据**，用户需自行提供原始 parquet 文件，放置于 `./data/` 目录：

```
data/
├── snapshot_sym0_date0_am.parquet
├── snapshot_sym0_date0_pm.parquet
├── snapshot_sym0_date1_am.parquet
├── ...
└── snapshot_sym4_date119_pm.parquet
```

- **格式**：5 sym × 120 dates × 2 sessions (am/pm) × 2001 ticks，共 1200 个 parquet
- **列**：154 维原始 LOB 特征 + label_{5,10,20,40,60}（0/1/2 三分类）
- **无需 `date` 列**（平台测试时被置 0，特征中从不使用）

---

## 一键运行

```bash
# 将原始 parquet 放到 ./data/
./run_pipeline.sh ./data 0
# 第一个参数：data_dir；第二个参数：CUDA device ID（默认 0）
```

产物：`outputs/submission_050911_iter019_v2N_50plus50_optimized_fullhorizon.zip`（约 148 MB）

---

## 分步说明

### Step 1: 特征构建（`01_build_features/`）

```bash
python3 01_build_features/build_schemeP_cache.py \
    --data_dir ./data --out ./outputs/cache
```

**功能**：原始 parquet → schemeP 370-d 特征 cache（npz 格式）

**特征组成**：
- 154 维原始末 tick 特征（含 amount_delta 的 sign-preserving log1p）
- 196 维 extra 特征（T3 无时间 69 + Stage1 54 + Stage2 59 + Stage3 14）
- 20 维 Stage5 特征（自适应动量、OFI 毒性、多尺度 bipower RV、价差机制、交易方向持续性、流动性不对称）

**产物**：
- `outputs/cache/schemeP_train.npz`（dates 0-79，约 800 session）
- `outputs/cache/schemeP_val.npz`（dates 80-95，约 160 session）
- `outputs/cache/schemeP_test.npz`（dates 96-119，约 240 session）
- `outputs/cache/schemeP_feat_names.txt`（370 个特征名）

**耗时**：约 30 min（视磁盘 IO 速度）

---

### Step 2: 训练 50 LGB 模型（`02_train_lgb/`）

```bash
# 单个 seed
python3 02_train_lgb/train_T188v2_lgb_seed.py \
    --seed 1 --cache-dir ./outputs/cache --out-dir ./outputs/models

# 全部 50 seeds（串行）
bash 02_train_lgb/run_all_lgb_seeds.sh ./outputs/cache ./outputs/models

# 使用 GPU 加速（可选）
bash 02_train_lgb/run_all_lgb_seeds.sh ./outputs/cache ./outputs/models --gpu
```

**超参多样性**：5 HP 配置（来自 T75 系列）× seeds 1-50 cycling。  
`hp_group = (seed - 1) % 5`：

| HP 组 | feature_fraction | bagging_fraction | num_leaves | lambda_l2 |
|------|-----------------|-----------------|-----------|----------|
| 0 | 0.8 | 0.8 | 127 | 1.0 |
| 1 | 0.6 | 0.7 | 127 | 1.0 |
| 2 | 0.7 | 0.85 | 63 | 2.0 |
| 3 | 0.5 | 0.6 | 255 | 0.5 |
| 4 | 0.4 | 0.5 | 127 | 3.0 |

**训练协议**：M7 全数据重训（all dates 0-119），num_boost_round=330，aug_a 数据增强（随机缩放 0.8-1.2）。  
**目标**：Δmid 回归 `y = (mp_{t+H} - mp_t) / (mp_t + 1)`

**产物**：`outputs/models/model_h60_seed{1..50}.txt`（LightGBM booster）

**耗时**：CPU ≈ 30 min；GPU ≈ 12 min（需安装 LightGBM GPU 版本）

---

### Step 3: 训练 50 NN 模型（`03_train_nn/`）

```bash
# 单个 seed（使用 GPU 0）
python3 03_train_nn/train_T188v2_nn_seed.py \
    --seed 1 --cache-dir ./outputs/cache --out-dir ./outputs/models --cuda 0

# 全部 50 seeds（串行，GPU 0）
bash 03_train_nn/run_all_nn_seeds.sh ./outputs/cache ./outputs/models 0

# 并行加速（5 台 GPU，每台各跑 10 seeds）示例：
# GPU 0: seeds 1-10;  GPU 1: seeds 11-20; etc.
```

**两阶段训练**：

**Phase 1（L2 预训练）**：
- 架构：MLP [359→256→128→64→1]，LayerNorm + GELU + Dropout，Kaiming 初始化
- 训练集：dates 0-79（schemeP_train.npz）
- 验证集：dates 80-95（schemeP_val.npz）——15 天，足以给出稳定早停信号
- 优化：AdamW + CosineAnnealingLR，patience=10，max 50 epochs
- aug_a 数据增强（随机特征缩放，2× 数据量）
- 产物：`outputs/models/T81_pretrained/anchor_seed{S}.pt`

**Phase 2（SPO+ DFL 微调）**：
- 从 Phase 1 最优 checkpoint 热启动
- 全数据 M7（dates 0-119）+ aug_a
- 决策焦点损失：L2 + λ_spo × SPO+（λ_spo=30）
- lr=3e-5，固定 11 epochs
- 产物：`outputs/models/nn_h60_seed{S}.npz`（Numpy 格式权重，平台推理用）

**HP 多样性**（cycling by seed，3×4×3=36 combos）：
- lr_p1：[1e-4, 3e-4, 1e-3]，index = (seed-1) % 3
- dropout：[0.05, 0.10, 0.15, 0.20]，index = (seed-1) % 4
- batch_p1：[2048, 4096, 8192]，index = (seed-1) % 3

**耗时**：单 GPU 约 90 min（~50s Phase1 + ~50s Phase2 per seed）；5 GPU 并行 ≈ 18 min

---

### Step 4: 组装提交包（`04_build_pkg/`）

```bash
python3 04_build_pkg/build_pkg.py \
    --models ./outputs/models \
    --pkg ./outputs/pkg

cd ./outputs/pkg && zip -qr ../submission.zip .
```

**产物**：
- `outputs/pkg/` — 包含 Predictor.py + 100 个模型文件 + 配置
- `outputs/submission*.zip` — 提交给平台的压缩包（~148 MB）

---

## 提交包内容说明

| 文件 | 说明 |
|------|------|
| `Predictor.py` | 主推理类，批量化 NN ensemble + LGB 集成 + conformal abstain |
| `thresholds.json` | h=60 阈值（SOTA）+ h=5/10/20/40 share_with=60 |
| `config.json` | 154 维原始特征列表 |
| `fast_features.py` | 单窗口特征计算（Predictor 调用） |
| `fast_features_batch.py` | 批量特征计算（特征构建阶段使用） |
| `requirements.txt` | CPU-only torch 依赖（平台安装 ≈ 3-5 min） |
| `nn_h60_seed{1..50}.npz` | 50 NN 权重（numpy 格式，无 torch 依赖序列化） |
| `model_h60_seed{1..50}.txt` | 50 LightGBM booster（文本格式） |

---

## 推理逻辑（Predictor.predict）

1. 从输入 DataFrame 提取 100-tick 滑动窗口（WINDOW=100）
2. 调用 `fast_features.py` 计算 370-d schemeP 特征
3. 丢弃 11 个已知失效特征（DROP_NAMES），得到 359-d 输入
4. **LGB 推理**：50 个 booster 各预测 Δmid，取均值 `pred_lgb`
5. **NN 推理**：使用 `_BatchedMLPEnsemble` 将 50 个 MLP 合并为一次 torch.bmm 前向（CPU 或 CUDA 自动选择），取均值 `pred_nn`
6. **集成**：`pred = (1.0 × pred_nn + 1.5 × pred_lgb) / 2.5`
7. **全 horizon 发射**：对每个 horizon 查找 thresholds.json，如有 `share_with`，用 h=60 的预测值；阈值已按 sqrt(H/60) 缩放
8. **Conformal abstain**：`eff_thr_up = thr_up + beta_sym × sigma_sym`；若 |pred| 在带内则 action=1

**Constraint compliance**：
- `date` 字段从不使用
- 无跨调用 state（每次 predict 完全独立）
- sym 仅用于 conformal beta 查找（不输入模型，OOD sym 使用默认值）

---

## 关键设计决策

| 决策 | 原因 |
|------|------|
| M7 全数据重训 | 平台 +34.44 vs 早停 +28.16（+5.51），GBDT 全数据安全 |
| 50+50 mega ensemble | 相比 5+5 降低方差；平台 +35.64 vs +34.64（+1.00） |
| SPO+ DFL Phase 2 | 对齐 PnL 梯度；平台 +28.16 vs L2-only +19.23（+8.93） |
| h=5/10/20/40 share_with=60 | 不引入新模型，阈值缩放；max(5 horizon) 评分只升不降 |
| conformal per-sym abstain | sym=1/2/3 各有最优 beta；总体 +0.77 |
| aug_a 随机特征缩放 | 等效数据增广，提高泛化；0.8-1.2 均匀缩放 |

---

## 已知限制

- **无 GPU 平台**：requirements.txt 使用 CPU-only torch，推理时 NN ensemble 在 CPU 运行（约 500ms per batch）
- **训练权重不含**：本仓库不含 100 个已训练模型文件（太大，~5 GB）；用户需重新训练
- **硬约束**：平台评测时 date=0，sym 可能含训练外股票——这两点已在 Predictor.py 中处理

---

## 引用 / 致谢

本工作基于良文杯官方 mmpc_demo 基线，经过 192 个实验迭代优化。
核心方法参考：
- T75 LightGBM Δmid 回归（iter_013 突破）
- T81 MLP L2 预训练协议
- T87 SPO+（Smart Predict-then-Optimize）决策焦点学习
- T170/T188 M7 全数据重训 trick
