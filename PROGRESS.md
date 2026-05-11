# 第二届"良文杯"统计建模与 AI 预测挑战赛 — 项目进度

## ⚠️ 评测协议硬约束（最重要！）

完整说明见 [`CRITICAL_CONSTRAINTS.md`](CRITICAL_CONSTRAINTS.md)。**任何模型 / 特征 / Predictor 设计前必读**。
3 条核心红线：
1. **`date` 评测时被置 0**——不能当 feature
2. **测试点顺序被打乱**——Predictor 不能维护跨调用 state
3. **sym 0-4 但可能含训练外股票**——模型必须 sym-agnostic

---

## 1. 赛题速览

| 项 | 内容 |
|---|---|
| 主办 | 厦门大学经济学院 + 黄良文统计学科基金会 (协办: AITOPIA / 艾托邦) |
| 任务 | 用 LOB + 订单流数据预测股票**中间价 midprice** 未来移动方向（涨/平/跌 三分类） |
| 数据 | 5 只匿名 A 股 × 120 个交易日 × AM/PM × 2001 ticks (3s/tick) = 1200 个 parquet, 共 474MB |
| 输入 | 每次预测可用过去 ≤ 100 ticks（含当前），154 维特征 |
| 输出 | 5 个 head：label_5 / label_10 / label_20 / label_40 / label_60，分别对应 t+5/10/20/40/60 ticks 后的方向 |
| 评分 | **累计 PnL 收益率（扣除手续费 0.01% 双边）**——不是 accuracy；预测"涨/跌"才出手，"平"不出手 |
| 排名 | 5 个任务取最好的那个参与排名 |

### 关键时间节点（2026 年）
- 报名截止：3-29
- 初赛提交开放：4-初 起，**6 周窗口**
- 初赛提交截止：5-11
- 初赛成绩：5-18
- 决赛答辩：5-30

### 提交规则
- 每账号 **每 12 小时仅 1 次评测**——必须本地充分验证
- 单次评测时长 ≤ 3 小时；模型 ≤ 2GB；FP32
- 私榜：自动取公榜最后一次有效提交，公榜评测结束后 1-2 天跑

---

## 2. 标签定义（注意 α 阈值不同）

```
x = midprice_{t+n} − midprice_t
ϕ(x) = 0  if x < -α   (跌)
       1  if -α ≤ x ≤ α  (平)
       2  if α < x   (涨)
```
- label_5, label_10：α = **0.05%**
- label_20, label_40, label_60：α = **0.1%**

### 评分公式（关键）
```
pnl_single = [ (label−1)·(midprice_{t+n}−midprice_t)
               − 0.0001·|label−1|·((midprice_{t+n}+1)+(midprice_t+1)) ]
             / (midprice_t + 1)
```
- label=2(涨) → 买入再卖出；label=0(跌) → 卖出再买回（假设有底仓）
- label=1(平) → 不出手，pnl=0
- 累计收益率 = 全部 single 之和（**模型评分**）
- 单次收益率 = 累计 / (预测涨数 + 预测跌数)

**含义**：不是 acc 优化目标，而是 **PnL 优化目标**。多预测"平"不会扣分；预测错"涨/跌"才扣（手续费 + 反向移动）。**precision 远比 recall 重要**——讲座原文："交易侧重 precision，F0.5"。

---

## 3. 数据 schema（每 parquet 2001 行 × 163 列，无 NaN）

> **完整字段权威说明 → [`docs/data_schema.md`](docs/data_schema.md)**（官方版，任何代码与之不一致以官方为准）。
> 下面是速查版。

### 信息戳
- `date`(0–119), `sym`(0–4), `time`(实际时间戳, 3s/档)
- AM 09:40:00–11:20:00, PM 13:10:00–14:50:00（两端各剃掉 10 分钟）

### 量价（已无量纲化）
- `open/high/low/close`：相对昨收涨跌幅
- `volume_delta`：换手率%；`amount_delta`：单位元（**唯一未归一化字段，量级 e3–e6**）

### 10 档 LOB
- `bid1..bid10`, `ask1..ask10`：相对昨收涨跌幅
- `bsize1..bsize10`, `asize1..asize10`：换手率%
- `avgbid/avgask/totalbsize/totalasize`：千档汇总

### 订单流统计（六类订单：lb/la/mb/ma/cb/ca）
- `*_intst`：上 tick 到当 tick 平均到达强度
- `*_ind`：过去 1 tick 强度 > 过去 60 tick 强度? (0/1)
- `*_acc`：到达强度的平均变化率（加速度）
- 命名：l=limit, m=market, c=cancel；b=buy, a=ask(sell)

### 衍生特征（基础 LOB 的派生值，节约脑子用）
- `midprice1..10`, `spread1..10`, `bid_diff1..10`, `ask_diff1..10`
- `bid_mean/ask_mean/bsize_mean/asize_mean`
- `cumspread/imbalance`
- `bid_rate*/ask_rate*/bsize_rate*/asize_rate*`：滑动平均变化率

### 标签
- `midprice`：中间价 = (ask1+bid1)/2，跌停时=ask1，涨停时=bid1
- `label_5/10/20/40/60`：3 分类 0/1/2

---

## 4. 当前 workdir 结构

```
liangwenbei_workdir/
├── data/                              # 1200 parquet, 474M（不入 git）
├── docs/                              # 两份官方 PDF
│   ├── 赛题说明会.pdf
│   └── 平台提交说明及模型训练讲座.pdf
├── examples/
│   ├── example_official/              # 官方原版示例（DeepLOB 单 head, 3 分类）
│   │   ├── main.py                    # 滑窗预测调用方法
│   │   ├── mmpc/                      # 提交包结构示意
│   │   │   ├── Predictor.py           # 必填：__init__ + predict(List[DataFrame]) -> List[List[int]]
│   │   │   ├── model.py               # DeepLOB CNN
│   │   │   ├── best_val_model.pth     # 单任务权重
│   │   │   ├── config.json            # python_version/batch/feature/label
│   │   │   └── requirements.txt
│   │   └── snapshot_sym0_date0_am.parquet
│   └── mmpc_demo/                     # 升级版示例（DeepLOB 5 head 多任务）
│       ├── Predictor.py               # 多 head 版本
│       ├── model.py                   # DeepLOB / WideCNN / ResCNN / DropoutCNN 4 个变体
│       ├── best_model.pt
│       ├── config.json                # 全量 154 特征 + 5 个 label
│       └── requirements.txt
├── src/                               # 我们自己的代码（待写）
├── PROGRESS.md
└── .gitignore
```

---

## 5. Predictor 接口契约（提交必须遵循）

```python
class Predictor:
    def __init__(self):
        # 加载模型；用 os.path.join(os.path.dirname(__file__), 'xxx.pt') 取权重路径
        ...

    def predict(self, x: List[pd.DataFrame]) -> List[List[int]]:
        """
        x: 长度=batch（config 里指定）的 list，每个 DataFrame 100 行 × len(feature) 列
        返回: List[List[int]]，外层 batch，内层 len(label)，元素 ∈ {0,1,2}
        """
```

### 评测期注意（PDF 2.4）
- 测试点输入顺序**会被打乱**——不能跨 batch 假设时序
- `date` 字段被置 0；`sym` 0–4（**可能含训练外股票**！这条很关键）
- `time` 保留实际时间戳——可用于推断盘口阶段（开盘后/午前/午后等）

---

## 6. 已知潜在 trap & 建议方向

1. **类别极不均衡**：label=1（平）占 60–80%。讲座里官方 SVM baseline 不加 class_weight 时全预测 1（acc 64% 但 0 收益）。
2. **测试集可能有训练外股票**——不能 over-fit 单只 sym。
3. **PnL > F0.5 > acc**：宁可少出手（高 precision），不要为了 recall 乱预测涨/跌。
4. **手续费 0.01% 双边 = 0.02%/笔**：α=0.05% 的任务（label_5/10）扣完手续费余地极薄；α=0.1% 任务（label_20/40/60）更有利可图。**初期建议优先打 label_60。**
5. **特征已分三级**：原始量价(68) + 订单流统计(18) + 衍生(72) = 154。复杂模型可吃全量；简单模型可只用基础+订单流。
6. **mmpc_demo 已经是多 head 多任务模型**——直接 fork 它做基线最快。
7. **每 12h 一次提交**——本地 OOF/CV 必须能稳定预估线上分。

---

## 7. 下一步建议（待你确认方向）

候选路径（优先级递减）：
- **[A] 复现 mmpc_demo 多 head DeepLOB 基线**：本地训练→评测→生成第一份提交（占榜单坑位、跑通流程）
- **[B] 写一个本地 PnL 评测器**：严格按官方公式打分，未来所有迭代以此为准
- **[C] 探索性 EDA**：5 个 sym 的标签分布、midprice 跳变频率、各特征对 label 的 IC
- **[D] 调研更强 baseline**：DeepLOB 后续工作（TransLOB、TabLOB、N-BEATS 等高频文献）

**我的推荐**：先 B + A 并行——B 不依赖训练，可直接给出"如果什么都不预测=0 分"和"全预测涨=负分"的下限，并且让 A 的迭代有反馈。然后再看 C/D。

等你拍方向后我就去派 worker。

---

## 8. 实验历程摘要（最新）

| Iter | 实验 | 本地 holdout PnL | 平台 PnL | 备注 |
|---|---|---|---|---|
| v2 | T87 MLP + T75 LGB | — | +34.44 | **SOTA** |
| T163 | v2 + T97 MLP | — | −3.70 | MLP太相似，平台有毒 |
| T172 | v2 + GroupTransformer | **完成** | — (in-sample inflated) | 提交包已构建，需平台验证 |

## 10. T180 OOD密度抽象包装器（2026-05-10）

**目标**：用GMM密度估计在推断时检测OOD输入，对OOD行弃权（action=1）

**设计**：在日期0-95的370维特征上拟合PCA(20维) + GMM(k=5)；测试时log_p < threshold → abstain

**结果（本地holdout dates 96-119）**：

| 变体 | Holdout PnL | Delta vs T170 | 弃权比例 |
|---|---|---|---|
| T170 基线 | +149.95 | — | — |
| V1 (p10) | +117.63 | **-32.31** | 8.0% |
| V2 (p20) | +92.65 | **-57.29** | 15.0% |
| V3 (p30) | +71.50 | **-78.44** | 22.6% |

**关键发现**：测试数据(96-119)的mean log-p比训练高+4.12（更in-distribution，非OOD）；低密度点反而贡献高PnL。无zip构建。

---

## 9. T172b 任务完成（2026-05-09）

**目标**：GroupTransformer（特征分组+注意力）+ v2 LGB/MLP 集成，测试架构多样性

**架构**：GroupTransformer (d_model=64, nhead=4, 2 layers, n_tokens=32, CLS token)
- 输入：359维特征 → Linear(359, 32×64) → (B, 32, 64) super-tokens
- 2层 TransformerEncoder (pre-norm)
- CLS token → head FC1 → LN → FC2 → /target_scale

**In-sample 评估（全部 in-sample，transformer 训练于 0-119）**:
| 变体 | In-sample PnL (96-119) | 备注 |
|---|---|---|
| v2 baseline | +109.68 | w_nn=1.0, w_lgb=0.5 |
| Transformer only | +149.91 | 强in-sample拟合 |
| Best: 1.0:0.5:1.0 | +164.71 | w_nn=1, w_tr=0.5, w_lgb=1 |

**提交包**：`submission_050909_iter019_v2N_transformer.zip`
- md5: `1fd4b0815b4fe9950c12771a17c44a3c`
- Predictor: torch CPU transformer (13x faster than numpy: 372ms vs 4.8s for 5 seeds x 1024)
- 全部 in-sample，真实性能需平台评测
- ⚠️ transformer 加入会 inflate holdout，最终效果未知

## 11. T188v2 50+50 mega ensemble (2026-05-11) — 🟢 新 SOTA

**目标**：把 T170 (5 NN T87M7 + 5 LGB v2) 扩到 50 NN + 50 LGB ensemble。
- 50 个独立 T81-style L2 pretrain (HP 多样性) → 50 个独立 T87 SPO+ M7 fine-tune
- 50 个 LGB (5 HP × 10 seed) M7 full-retrain on date 0-119
- 同 v2 thresholds: thr_up=0.0003, thr_dn=0.000216, w_lgb=1.5, w_nn=1.0, conformal per-sym

**本地 holdout (date 96-119, in-sample)**：
| 模型 | PnL | n_act |
|---|---|---|
| T170 (5+5) | +149.96 | 192,821 |
| T188v2 (50+50) | +150.43 | 192,413 |
| delta | +0.48 (噪音级) | −408 |

**平台 (OOD)**：
| 模型 | 平台 PnL |
|---|---|
| T170 | +34.64 |
| **T188v2 (50+50)** | **+35.64 ← 新 SOTA** |
| **delta** | **+1.00** |

**关键洞察**：
- in-sample holdout 看不出 ensemble 红利（+0.3% 噪音），OOD 上拿到 +2.9%
- 50 模型的 variance reduction 在 distribution shift 上才生效
- NN 50 倍本地 holdout NN-only 反而 −1.4（fine-tune 收敛同一 minimum），但叠加 LGB ensemble + conformal 后整体 +1.0
- 推断时间：本地 ~7 min（GPU bmm）；平台预计 ~10-15 min（CPU/GPU 看环境）

**zip**：`submission_050911_iter019_v2N_50plus50_optimized.zip` (148MB, MD5 `95993e6250ea56361dfecb62c9123e63`)
- T188v3 batched bmm 优化版，actions 与原 T188v2 100% 一致 (max diff 6.98e-10)
- 备用 CPU-only 版（装环境快 ~10x）：`..._cputorch.zip` (148MB, MD5 `c67e4d25665a2a7a12f4f6839cc8d0fe`)
