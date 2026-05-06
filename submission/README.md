# 良文杯提交工作区

> 本目录承担"治理 + 历史 + 实际产物"——每次准备提交的 zip 都生成在这里。
> 打包/契约校验/PnL 评测的逻辑在 `src/submit/` 和 `src/eval/`，本目录复用它们。

## 目录结构

```
submission/
├── README.md              # 本文件 — 怎么走流程
├── RULES.md               # 平台规则完整记录（中文）
├── SUBMISSION_LOG.md      # 提交迭代历史 tracker（一行一次）
├── ENV_NOTES.md           # 评测环境兼容性笔记（CUDA 13 / torch / pipreqs）
├── scripts/
│   ├── prepare.py             # 一键打包 + 契约校验 + 自动登记 LOG
│   ├── sanity_check.py        # 提交前飞行检查清单
│   └── compare_local_vs_eval.py  # LocalEvaluator vs 本地训练指标对比
├── template/              # 新 iter 的骨架（拷过去再填）
│   ├── Predictor.py
│   ├── model.py
│   ├── config.json
│   └── requirements.txt
└── iter_<NNN>_<name>/     # 每次提交的实际产物
    ├── Predictor.py
    ├── model.py
    ├── config.json
    ├── requirements.txt
    ├── model.pth          # 或 best_model.pt 等
    └── submission.zip     # prepare.py 生成
```

## 必读

- 提交规则：[`RULES.md`](./RULES.md)
- 环境兼容性：[`ENV_NOTES.md`](./ENV_NOTES.md)
- 历史记录：[`SUBMISSION_LOG.md`](./SUBMISSION_LOG.md)

## 创建一个新的 iter（标准步骤）

### Step 1 — 拷骨架

```bash
NNN=001
NAME=deeplob-mlofi
cp -r submission/template submission/iter_${NNN}_${NAME}
```

### Step 2 — 填代码 + 权重

把训练好的 `model.pth`（或 `best_model.pt`）放进 `submission/iter_${NNN}_${NAME}/`，
改 `Predictor.py`、`model.py`、`config.json` 适配你的模型。

### Step 3 — 生成纯净 requirements.txt

```bash
cd submission/iter_${NNN}_${NAME}
pip install pipreqs  # 一次性
pipreqs ./ --encoding=utf8 --force --mode no-pin
# 然后人工编辑加 ==X.Y.Z（参考 ../ENV_NOTES.md §2）
cd ../..
```

### Step 4 — 飞行检查

```bash
python submission/scripts/sanity_check.py --src submission/iter_${NNN}_${NAME}
```

任何 ✗ 都修掉，⚠ 自行判断。

### Step 5 — 打包 + 契约校验 + 登记

```bash
python submission/scripts/prepare.py \
    --iter $NNN \
    --name $NAME \
    --src submission/iter_${NNN}_${NAME} \
    --note "MLOFI multi-window features"
```

成功后：
- `submission/iter_${NNN}_${NAME}/submission.zip` 已生成
- `submission/SUBMISSION_LOG.md` 已自动追加一行

### Step 6 — 跑 LocalEvaluator 看本地 PnL

```bash
python submission/scripts/compare_local_vs_eval.py \
    --src submission/iter_${NNN}_${NAME}
```

会输出 5 个 horizon 的 cum_pnl + accuracy + 预测分布；并把
`compare_report.json` 落到该 iter 目录。

把 best cum_pnl 手动填回 `SUBMISSION_LOG.md` 的对应行。

### Step 7 — 上传到平台

把 `submission/iter_${NNN}_${NAME}/submission.zip` 上传到平台。
公榜结果出来后，回填 `SUBMISSION_LOG.md` 的"公榜 PnL / 单次收益"列。

## 已有 iter

- [`iter_000_mmpc_demo/`](./iter_000_mmpc_demo/) — 用现成 mmpc_demo 走通流程，
  作为后续 iter 的参考样例和 PnL baseline floor。
