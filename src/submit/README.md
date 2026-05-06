# 良文杯 — 提交流程工具集

把"训练完模型 → 打包 → 在评测协议下能跑通"这条链路工程化。任何最终
要提交的目录都能用这套工具在本地复现平台行为，确保提交前已通过契约校验。

## 目录

```
src/submit/
├── __init__.py
├── local_evaluator.py     # 模拟平台调用 Predictor.predict 的伪评测器
├── run_demo_baseline.py   # 用 examples/mmpc_demo 跑端到端基线
├── package.py             # 把源目录打成顶层平铺 zip
├── validate_contract.py   # 打包→解压到 tempdir→跑通，断言契约
└── README.md
```

## 提交包必须的文件清单

平台要求 zip **所有文件直接放在顶层**，不要再包子目录（PDF Step 0 红字）。
最少需要包含：

| 文件 | 作用 | 必填 |
| --- | --- | --- |
| `Predictor.py` | 含 `class Predictor`，`predict(x: List[pd.DataFrame]) -> List[List[int]]` | ✅ |
| `config.json` | 平台读取的元数据 | ✅ |
| `requirements.txt` | 平台 `pip install` 用 | ✅ |
| `model.py` 等源码 | 被 `Predictor.py` import 的代码 | 视情况 |
| `*.pt / *.pth / *.bin` | 模型权重 | 视情况 |

**总大小 < 2GB；FP32；单次评测 ≤ 3 小时**。

## `config.json` 字段

```json
{
  "python_version": "3.11",
  "batch": 1024,
  "feature": ["open", "high", ...],
  "label":   ["label_5", "label_10", "label_20", "label_40", "label_60"]
}
```

| 字段 | 含义 |
| --- | --- |
| `python_version` | 平台启动的 python 版本（如 `"3.9"`、`"3.11"`） |
| `batch` | 平台调用 `predict()` 时单批 DataFrame 数量 |
| `feature` | 输入 DataFrame 的列名顺序（严格匹配） |
| `label` | 输出每行的标签列名顺序，长度 = 内层 list 长度 |

## `Predictor` 接口约定

```python
class Predictor:
    def __init__(self): ...
    def predict(self, x: List[pd.DataFrame]) -> List[List[int]]:
        # x: 长度 = batch；每个 DataFrame：100 行 × len(feature) 列
        # 返回：长度 batch；每内层长度 = len(label)；元素 ∈ {0, 1, 2}
        ...
```

约束：

- **不要硬编码 batch_size**。最后一个批可能不足 `config["batch"]`（虽然平台
  通常是整除的，本地评测器为了不浪费样本会送末批，作为防御性测试）。
- 模型必须能在无网环境下加载（权重一并打进 zip）。
- 所有 import 用相对路径或同包导入；平台按 `from <pkg>.Predictor import Predictor`
  方式导入，所以可以也建议使用 `from .model import ...`。

## 评测协议要点

1. 平台读 `config.json` → `pip install -r requirements.txt` → `from <submit>.Predictor import Predictor`
2. 反复调用 `predict()`，直到所有评测点处理完
3. **评测点顺序被打乱**——不能依赖 batch 顺序累计 state（不能假设连续两次
   predict 是同一只股票的连续时间步）
4. **`date` 字段被置 0**；`sym` 仍是 0–4，但**可能含训练外股票**；`time` 保留
5. 每个评测窗口长度固定 = 100；单次评测 ≤ 3 小时；模型 ≤ 2GB；FP32

## 如何打包

### 一步式

```bash
python src/submit/package.py --src examples/mmpc_demo --output submission.zip
```

输出会包含：
- 校验报告（必填文件、config 字段、模型权重候选）
- 烟测结果（用随机 100×D 输入调一次 `predict()` 看返回 shape）
- 最终 zip 大小（必须 < 2GB）

### 完整契约校验（推荐提交前做）

```bash
python src/submit/validate_contract.py --src examples/mmpc_demo
```

它会：
1. 打包成临时 zip
2. 解压到全新 tempdir（包名默认 `mmpc`）
3. 用真实 `data/snapshot_sym0_date119_am.parquet` 调 `LocalEvaluator`
4. 逐 batch 断言：长度 == `batch`、内层长度 == `len(label)`、元素 ∈ {0,1,2}
5. 完整跑完一个 session，确认无报错

### 端到端基线指标

```bash
python src/submit/run_demo_baseline.py
```

跑 `examples/mmpc_demo` 在 `sym=0 date=119 am` 上的全部评测点，打印每个 horizon 的
accuracy / precision_macro / 预测分布，并把预测落盘到
`analysis/mmpc_demo_baseline_predictions.parquet`，列：

```
sym, date, session, t,
midprice_t,
true_label_5, ..., true_label_60,
pred_label_5, ..., pred_label_60,
midprice_t5, midprice_t10, midprice_t20, midprice_t40, midprice_t60
```

这个 parquet 是后续 PnL 评测器（另一个 worker）的输入。

## 常见坑

- **导入失败：`ImportError: attempted relative import with no known parent package`**
  原因：`Predictor.py` 用了 `from .model import ...`，但你直接 `python Predictor.py`
  跑了。LocalEvaluator/平台会按包导入（`from mmpc.Predictor import Predictor`），
  不会有这个问题。
- **形状错误：`predict 返回长度不匹配`**
  原因：`predict` 写死了 batch，或最后一个不足批没处理。改成动态读取
  `len(x)`。
- **塌缩到全预测 1（flat）**
  示例 `mmpc_demo` 就是这种状态：所有 5 个 head 在测试 session 上几乎全部
  输出 1。这是模型本身的问题（多任务训练严重欠拟合），与提交流程无关。
  Accuracy 数值看似不错（85–95%）只是因为真实分布也以 1 为主。

## 与 PnL 评测器的边界

本工具集 **只**负责：
- 模拟平台调用形态
- 生成预测 + ground-truth + midprice 的统一表

**不**负责：
- 计算 PnL / Sharpe / max drawdown 等收益指标
- 训练任何模型

PnL 评测器读取 `analysis/mmpc_demo_baseline_predictions.parquet`，自定义信号→仓位→收益逻辑。
