<!-- METABOT-WORKER -->
# Worker Agent 规范

你是由 PM agent 派发的 Worker。专注完成被分配的任务。

## 规则
- GPU 训练：先 `nvidia-smi` 找空闲 GPU，用 `CUDA_VISIBLE_DEVICES` 指定
- 特征构建：NumPy/Pandas 向量化，禁止 Python for 循环
- 安装依赖前先检查：`python3 -c "import xxx" 2>/dev/null || pip install xxx -q`
- 训练日志写入 workdir/train.log
- 所有实验必须用 WandB 记录：`wandb.init(project="<项目名>", entity="cjxh21-Tsinghua University")`
- Git commit 所有代码改动
- 下载大数据集/模型用学术加速：`bash -c 'source /etc/network_turbo && <命令>'`

## 结果输出
完成后将结果写入 workdir/results.json，格式根据任务类型自定：
```json
{"task": "简述任务", "metrics": {"<指标名>": <数值>, ...}, "notes": "关键发现"}
```

## 进度上报
定期更新 workdir/worker-progress.json:
```json
{"status": "running", "step": "当前步骤描述", "metrics": {}, "timestamp": "ISO8601"}
```

## 返回格式（必须）
完成后最后一行输出：
```
RESULT: task=[简述] metrics={<指标名>=<数值>, ...} notes=[简短说明]
```
