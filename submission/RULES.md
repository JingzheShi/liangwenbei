# 良文杯提交规则（完整版）

> 整理自：官方 PDF（`docs/`）+ 用户提供的"平台提交说明（最新版）"。
> 本文件是 self-contained 的——读完就懂提交全流程。
> 凡与 PDF 不一致或 PDF 没有的"新信息"，会在 §0 单独标出。
>
> **数据字段权威说明 → [`../docs/data_schema.md`](../docs/data_schema.md)**（任何 EDA / 训练代码与之不一致以官方文档为准）

---

## §0 PDF 之外的新信息（重点！）

| 项 | 平台说明 |
| --- | --- |
| CPU 评测 | 16 CPU 核 |
| GPU 评测 | 1 块 NVIDIA RTX A6000，driver `580.126.09`，CUDA `13.0` |
| GPU 显存 | A6000 = 48GB（够用，主要瓶颈在 2GB 模型上限） |
| 提交频率 | **同账号 12 小时仅可 1 次提交**；任务异常可下载日志 + 删除任务后重提 |
| 单次评测 | ≤ 3 小时（超时直接失败） |
| 模型大小 | ≤ 2GB；**必须 FP32**（不接受 FP16/INT8 量化） |
| 私榜 | 自动取公榜最后一次有效提交（即提交策略要保证"最后一次跑得过") |
| zip 结构 | **所有文件直接平铺在 zip 顶层**——不能再有任何子文件夹 |

→ 详细 CUDA 13 / torch 兼容性 见 [`ENV_NOTES.md`](./ENV_NOTES.md)。

---

## §1 提交压缩包要求

### 1.1 必备文件（zip 顶层平铺，禁止子目录）

| 文件 | 作用 | 必填 |
| --- | --- | --- |
| `Predictor.py` | 含 `class Predictor`，平台调用入口 | ✅ |
| `config.json` | 平台读取的元数据（`python_version`/`batch`/`feature`/`label`） | ✅ |
| `requirements.txt` | 平台 `pip install -r` 用 | ✅ |
| `model.py`（或其他源码） | 被 `Predictor.py` import 的代码 | 视情况 |
| `model.pth` / `*.pt` / `*.bin` 等 | 模型权重 | 视情况 |

> ⚠️ **绝对禁止**：在 zip 里再套一层目录（比如 `submission/Predictor.py`）。
> 平台会"把所有文件直接平铺到一个目录"，子目录会变成路径错误而炸掉。
> 我们的 `src/submit/package.py` 默认就是顶层平铺，不要绕过它手动 zip。

### 1.2 `config.json` 字段（严格）

```json
{
  "python_version": "3.10",
  "batch": 1024,
  "feature": ["bid1", "bid2", "bid3", "ask1", "ask2", "ask3",
              "bsize1", "bsize2", "bsize3", "asize1", "asize2", "asize3"],
  "label": ["label_5", "label_10", "label_20", "label_40", "label_60"]
}
```

| 字段 | 含义 | 注意 |
| --- | --- | --- |
| `python_version` | 平台启动的 Python 版本（字符串） | 用 `"3.10"` 或 `"3.11"`，与本地训练环境一致 |
| `batch` | 平台一次调用 `predict()` 的 DataFrame 数量 | 必须是 **正整数**；不要硬编码到代码里——`predict` 应根据 `len(x)` 动态处理 |
| `feature` | 输入 DataFrame 的列名顺序（**严格匹配**） | 顺序敏感；本地训练时的列序就是这里的列序 |
| `label` | 输出每行的标签列名顺序 | 长度 = `predict` 返回的内层 list 长度 |

**`batch` 的取舍**（见 §0：单次评测 ≤ 3h）：
- 太小（<128）：CPU/GPU 利用率低，可能 3h 跑不完。
- 太大（>4096）：单批 OOM 风险（A6000 48G，但还要留权重的内存）。
- 经验值：CPU 模式选 `256–1024`；GPU 模式选 `512–4096`。

### 1.3 `requirements.txt` —— 易踩坑！

❌ **绝对不要**直接从训练环境 `pip freeze > requirements.txt`：
- 会混入 `jupyter`/`tensorboard`/`wandb`/`matplotlib` 等推理无关的库；
- 这些库的子依赖在评测环境里可能装不上 → 整个评测构建失败 → 直接失去这次提交机会（12h 才能再提）。

✅ **正确步骤**：

1. 新建独立文件夹（即本仓库的 `submission/iter_xxx_yyy/`），仅放推理相关文件：
   ```
   Predictor.py
   model.py
   model.pth
   config.json
   ```
2. 在该文件夹下用 `pipreqs` 生成纯净依赖：
   ```bash
   pip install pipreqs  # 一次性
   pipreqs ./ --encoding=utf8 --force
   ```
   `pipreqs` 只看 `import` 语句，不会污染。
3. **手动复查并写死版本号**——pipreqs 出的版本可能滞后，对照本地 `pip show <pkg>` 修：
   ```
   torch==2.10.0
   numpy==2.4.3
   pandas==3.0.1
   ```
4. 在空白环境验证（提交前做一次）：
   ```bash
   conda create -n test_submit python=3.10 -y
   conda activate test_submit
   pip install -r requirements.txt
   python -c "from Predictor import Predictor; Predictor()"
   ```

---

## §2 `Predictor.py` 接口

### 2.1 模板骨架

```python
import os
import torch
import pandas as pd
import numpy as np
from typing import List
from .model import YourModel  # 同包导入

class Predictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # ✅ 用 os.path.join + __file__，不要用相对路径！
        pth_path = os.path.join(os.path.dirname(__file__), 'model.pth')
        self.model = YourModel(...)
        self.model.load_state_dict(torch.load(pth_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, x: List[pd.DataFrame]) -> List[List[int]]:
        # x 长度 = config.batch（最后一批可能不足，要动态处理）
        # 每个 DataFrame：100 行 × len(config.feature) 列
        arr = np.stack([df.to_numpy(dtype=np.float32) for df in x], axis=0)  # (B, 100, D)
        t = torch.from_numpy(arr).to(self.device)
        with torch.no_grad():
            out = self.model(t)  # tuple of K tensors, each (B, num_classes)
        preds = np.stack([h.argmax(1).cpu().numpy() for h in out], axis=1)  # (B, K)
        return preds.astype(int).tolist()
```

### 2.2 硬性约定

- `__init__` 无参数
- `predict(x: List[pd.DataFrame]) -> List[List[int]]`
  - 外层 `len = len(x)`（动态读，**不要硬编码 `batch`**）
  - 内层 `len == len(config.label)`
  - 元素 `∈ {0, 1, 2}`（int，0=跌，1=平，2=涨）
- 模型权重路径用 `os.path.join(os.path.dirname(__file__), '...')`
- GPU：模型 `.to(device)`，输入 `.to(device)`，输出 `.cpu().numpy().tolist()`

---

## §3 测试时平台的行为（容易踩坑！）

### 3.1 评测点顺序被打乱

> 评测程序会**打乱**测试点顺序后送入 predict。

→ 后果：**不能跨 batch 假设连续时序**。
- ❌ 用 LSTM/Transformer 的 hidden state 跨 batch 累计
- ❌ 在 `__init__` 里建一个 sym → buffer 的 map 缓存历史
- ✅ 每次 predict 调用只用当前 100 行做完全独立的预测

### 3.2 sym 可能含训练外股票

> sym ∈ {0, 1, 2, 3, 4}，**可能含训练数据外的新股票**。

→ 后果：**不能 hard-code sym** 当 categorical embedding 用。
- ❌ `nn.Embedding(num_sym, dim)` 当 sym 是模型输入 → 训练外 sym 会 IndexError
- ✅ 完全不依赖 sym ID（输入特征只看 100×D 的 LOB 切片）
- ✅ 或者用 sym-agnostic 的 representation（embedding hash trick / 全部映射到同一个）

### 3.3 date 字段被置 0

> 评测时所有 date 都是 0（无意义）。

→ 后果：**不能用 date 做特征** 或 推理时间段。
- ❌ "周一/周五效应"、"季末效应" 等 date-derived 特征
- ✅ 只用 100 行窗口内部的特征

### 3.4 time 保留

> time 字段保留（评测时仍是真实的 09:30:00 等）。

→ 后果：**可以用 time 做特征**——这是平台特意保留的。
- ✅ 时段相关特征（开盘/收盘前后波动率不同）
- ⚠️ 但 time 不在 `config.feature` 列表里 → 平台不会把 time 列送进 DataFrame；
  如果想用 time，要么把 time 编码进特征（如 "minutes_since_open"）后加入 feature 列，
  要么放弃时段特征。**确认平台是否会把 time 列附加到 DataFrame**——如果配置里没显式列出 time，应该不会送，所以稳妥起见**不要依赖** time 在 DataFrame 里。

### 3.5 评测点的窗口

每个评测点平台会给：
- 一个 100 行的 DataFrame（列 = `config.feature`，顺序严格）
- 你需要返回该窗口对应的 5 个 horizon 的预测

不会给你 midprice、不会给你 t、不会给你 sym。

---

## §4 容易踩的坑（一节专门列）

| # | 坑 | 后果 | 防御 |
| --- | --- | --- | --- |
| 1 | zip 内套子目录 | 路径错误，导入失败 | 用 `package.py`，不手动 zip |
| 2 | 模型权重用相对路径 `'model.pth'` | 评测目录不一定是 cwd → 找不到文件 | 必须用 `os.path.join(os.path.dirname(__file__), ...)` |
| 3 | requirements.txt 有 jupyter/tensorboard 等 | 装包失败，整次评测炸 | 用 `pipreqs`，不要 pip freeze |
| 4 | requirements.txt 没写死版本号 | 评测装的版本与本地不同 → 推理结果异常 | 每个包写 `==X.Y.Z` |
| 5 | predict 硬编码 batch | 末批不足 batch 时报错 | `for df in x:` 或 `len(x)` |
| 6 | 模型用 FP16/quantize | 平台拒收 | 保持 FP32 |
| 7 | 用 sym 做 embedding | 训练外 sym 报 IndexError | 不依赖 sym |
| 8 | 用 date 做特征 | 评测时被置 0，预测崩塌 | 不依赖 date |
| 9 | 跨 batch 假设连续时序 | shuffle 后预测错乱 | 每个 100 行窗口独立预测 |
| 10 | 推理时调用网络（HF download 等） | 评测无网，hang 或报错 | 模型权重打进 zip |
| 11 | 一次提交后 12h 不能再提 | 浪费评测机会 | 提交前一定本地完整跑通 `validate_contract.py` |
| 12 | 评测 > 3h 超时 | 直接失败 | 测好单 batch 时长 × 总 batch 数 < 3h；GPU 模式 throughput 一般够 |
| 13 | 模型 > 2GB | 平台拒收 | `ls -lh model.pth` 提前看 |

---

## §5 提交前 checklist（每次都过一遍）

```
[ ] 1. zip 用 src/submit/package.py 生成（不要手动 zip）
[ ] 2. 解压一次，确认是顶层平铺，无子目录
[ ] 3. config.json 的 4 个字段都齐
[ ] 4. config.feature 顺序与训练时一致
[ ] 5. config.label = ["label_5", "label_10", "label_20", "label_40", "label_60"]
[ ] 6. requirements.txt 不含 jupyter / tensorboard / wandb / matplotlib / ipython
[ ] 7. requirements.txt 的版本号都写死（==X.Y.Z）
[ ] 8. 模型大小 ls -lh < 2GB
[ ] 9. 模型权重为 FP32（torch.float32），不是 half/quantized
[ ] 10. Predictor.__init__ 无参数
[ ] 11. Predictor.predict 用 len(x) 动态处理 batch
[ ] 12. 模型 weight 路径用 os.path.join(os.path.dirname(__file__), ...)
[ ] 13. 预测不依赖 sym ID（不会因为训练外 sym 崩溃）
[ ] 14. 预测不依赖 date（评测时置 0）
[ ] 15. 跑 src/submit/validate_contract.py 全过
[ ] 16. 跑 submission/scripts/sanity_check.py 全过
[ ] 17. 单批 predict 时长 × 总 batch < 3h（throughput 估算）
[ ] 18. SUBMISSION_LOG.md 已记录本次 iter
[ ] 19. 距离上次提交已超过 12 小时
[ ] 20. 公榜上传后**等评测完成再发起下一次准备**——避免被自动覆盖
```

---

## §6 提交策略建议

- **12h 限制 → 一天最多 2 次提交**。所以：
  - 周末抓住"早+晚 2 次"做激进改动 + 安全回退；
  - 工作日间隔规律提（早上 9 点 / 晚上 9 点）；
  - 比赛快收官时，留好"安全提交"——不要把最后一次留给"未充分验证"的版本（私榜会取最后一次有效，毁了就是毁了）。
- **每次提交至少有 2 个 horizon 的本地 PnL 是正的**——避免提了个负分上去。
- **本地 PnL 计算用 `src/eval/pnl.py`**（严格按官方公式）。
- **每次提交都跑一遍 `submission/scripts/sanity_check.py`**——防止 last-minute 改动引入 contract 违规。

---

## §7 评分公式（核心，公榜排名依据）

来源：
- 平台官网"评测指南 → 收益率计算"页（2026-05-06 截图归档于本仓库）
- `docs/第二届"良文杯"统计建模与AI预测挑战赛-平台提交说明及模型训练讲座.pdf` 第 1.5 节

注：网页版与 PDF 版仅在分子第二项的 `|·|` 写法上略有差异（网页含 abs，PDF 用 `[·]` 仅作括号）；
**数学上对合法输入完全等价**（midprice ∈ (-1, ∞) → midprice+1 > 0 → 求和恒正）。
本仓库 `src/eval/pnl.py` 实现以**网页版为准、保留显式 abs**，并已通过 6 项 sanity 断言。

### 7.1 单笔预测的 PnL（核心公式 — 网页原文）

对每个评测点 `t`，对每个 horizon `n ∈ {5, 10, 20, 40, 60}`：

```
                  (label − 1) · (midprice_{t+n} − midprice_t)
                  − 0.0001 · |label − 1| · | (midprice_{t+n} + 1) + (midprice_t + 1) |
pnl_single  =  ──────────────────────────────────────────────────────────────────────
                                     midprice_t + 1
```

平台原文（2026-05-06 截图）：

> 预测方向：
> - 预测上涨（label=2）视为：此时出手买入，到预测的 5/10/20/40/60 tick 后卖出，**买卖双边均收取手续费 0.01%**
> - 预测不变（label=1）视为：**不出手**；
> - 预测下跌（label=0）视为：此时卖出，到预测的 5/10/20/40/60 tick 后买入（**假设有底仓**），买卖双边均收取手续费 0.01%
>
> 注：midprice+1 将相对昨收的涨跌幅还原到相对昨收的比例。
>
> 累计收益率：所有预测上涨/下跌的收益率之和。
>
> 单次收益率 = 累计收益率 / (预测的上涨总数 + 预测的下跌总数)，**特别注意是预测为上涨或下跌的总数**。

其中：
- `fee = 0.0001`（万 1 = 0.01%；**双边都收**）
- `label_pred ∈ {0, 1, 2}`：模型对该 t 的预测
- `midprice_*` 是"相对昨收的涨跌幅"（无量纲）；分母 `midprice_t + 1` 把涨跌幅还原成"相对昨收的价格倍数"

含义解读：
- `label_pred = 2`（预测涨）：相当于 t 时刻买入，t+n 卖出。差价 = `midprice_{t+n} − midprice_t`。
- `label_pred = 0`（预测跌）：相当于 t 时刻卖空（假设有底仓），t+n 买回。
  公式里 `(label−1) = −1`，自动反号。
- `label_pred = 1`（预测平）：`(label−1) = 0`，**不出手 → 0 PnL，0 手续费**。
- 手续费项 `fee · |label−1| · (P_{t+n} + P_t)` ≈ `2 · fee · 平均价`（双边）。
  转化为相对收益率约为 `0.0002`（即每笔出手必扣的 0.02% 摩擦）。

### 7.2 累计 / 单次收益率

| 指标 | 定义 | 用途 |
| --- | --- | --- |
| **累计收益率（cum_pnl）** | `Σ pnl_single` 对所有评测点求和 | **= 模型评分 = 公榜排名依据** |
| **单次收益率（single_pnl）** | `cum_pnl / (n_pred_up + n_pred_down)` | 平均每笔"主动交易"的收益；分母**只数预测涨/跌**，不含预测平 |
| **准确率 / 召回 / F0.5** | 标准多分类指标 | 公榜上有列，**但不是排名依据**；F0.5 体现"交易侧重 precision" |

### 7.3 5 个 horizon 的处理

- **公榜上每个 horizon 都有一行成绩**（同一个模型一次提交，会自动跑 5 个 horizon）。
- **排名规则**：5 个 horizon 中**取最高 cum_pnl**作为该队伍的排名依据。
- 也就是说：**只要你某一个 horizon 强就行**——可以专攻某一个 horizon。
- 但平台并不知道你"专攻哪个"，它会公平地跑全部 5 个 → 你的 `config.json` 的 `label` 字段决定输出哪些 horizon。

### 7.4 critical 隐含规则

1. **不出手是绝对安全的零风险底盘**——cum_pnl 永远 ≥ 0 的策略是"全预测 1"。
2. **预测涨/跌的盈亏对称是手续费**——单笔需要 |Δmidprice| > 2·fee≈0.02% 才有正期望。
3. **完美预测 ≠ PnL 上限**——当真实 |Δmidprice| 略大于 2·fee 但被 α 阈值规整为 label=1 时，预测涨/跌反而比预测平赚。所以**阈值后处理**是个潜在 alpha。
4. **沉淀 single_pnl 高 + 累计 cum_pnl 高**才是"两条腿"——如果 single_pnl 高但出手次数少，cum_pnl 可能低；如果出手多但 single_pnl 接近 0 也没用。
5. **5 个 horizon 取最好** → 训练时不必强求 5 个都好，**重点优化 label_60**（PnL 上限最高，sanity check 测过）。
6. **平台还会显示"准确率/召回/F0.5"等参考列**——但模型评分只看 cum_pnl，**不要被准确率带偏**。

### 7.5 实现锚点

本仓库的 `src/eval/pnl.py` **严格按本公式实现且已通过 6 项断言**（包括 `cum_pnl(全预测 1) == 0`、`cum_pnl(完美预测) ≥ 任何其他策略`）。

每次提交前都用它在本地 test set 上算一遍 5 个 horizon 的 cum_pnl，记录到 `SUBMISSION_LOG.md`。

### 7.6 验证过的 baseline 数字（单 session = sym0 date0 am, 1842 评测点）

| 策略 | label_5 | label_10 | label_20 | label_40 | label_60 |
|---|---|---|---|---|---|
| 完美预测（PnL 上限近似） | 0.568 | 0.967 | 1.216 | 2.065 | **2.743** |
| 全预测平 | 0 | 0 | 0 | 0 | 0 |
| 全预测涨 | -0.328 | -0.296 | -0.264 | -0.170 | -0.062 |
| 全预测跌 | -0.409 | -0.441 | -0.473 | -0.567 | -0.675 |
| 随机三分类 | -0.272 | -0.261 | -0.250 | -0.267 | -0.368 |
| **mmpc_demo（官方多任务）** | **+0.0006** | **+0.0011** | **0** | **0** | **0** | (几乎全预测 1，模型塌缩)

**任何"接近 0"的 cum_pnl 都意味着模型在装睡**——必须迭代到 label_60 cum_pnl > 0.5 / session 才算入门。

---

## §8 与本仓库的关系

| 角色 | 路径 |
| --- | --- |
| 通用打包工具 | `src/submit/package.py` |
| 端到端契约校验 | `src/submit/validate_contract.py` |
| 本地伪评测器 | `src/submit/local_evaluator.py` |
| 严格 PnL 评测 | `src/eval/pnl.py` |
| 提交治理 | `submission/`（本目录） |
| 每次提交的产物 | `submission/iter_<NNN>_<name>/` |
| 历史 tracker | `submission/SUBMISSION_LOG.md` |

`submission/` 这层不再实现打包逻辑——它复用 `src/submit/` 的工具，只负责"治理 + 历史 + 实际产物"。
