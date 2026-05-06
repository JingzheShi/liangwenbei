# Snapshot Ensembles: Train 1, Get M for Free

**作者**：Gao Huang, Yixuan Li, Geoff Pleiss, Zhuang Liu, John Hopcroft, Kilian Q. Weinberger
**会议/年**：ICLR 2017
**arXiv**：[1704.00109](https://arxiv.org/abs/1704.00109)

---

## 核心 idea

训练**单个**神经网络，但 learning rate 用 **cyclic cosine annealing**：

$$\eta(t) = \frac{\eta_0}{2}\Big(\cos\Big(\frac{\pi \cdot \mathrm{mod}(t-1, T/M)}{T/M}\Big)+1\Big)$$

每个 cycle 末（lr 降到接近 0），网络收敛到一个 local minimum；保存 weights = 一份 snapshot。重复 M 次 → 得到 M 份 snapshot。

总训练 budget T 不变，于是"训一次得到 M 个模型"。

## 关键点

1. **lr 突然回升**到 η₀ 的作用：把网络从当前 minimum 踢出去，让它重新找一个不同的 basin。
2. **每个 snapshot 在 weight space 上不同**：作者用 t-SNE / parameter distance 验证 M 个 snapshot 的解相距足够远，prediction 上有 diversity。
3. **inference 时 ensemble**：M 个 snapshot 各自 forward，输出取平均（softmax 概率均值）。
4. **后期 snapshot 权重可加大**：作者建议给 cycle 数大的 snapshot 更高 weight（因为它们看了更多 data），但简单等权重也基本够用。

## 实验结果

| 数据集 / 模型 | Single error | Snapshot ensemble error |
|---|---|---|
| CIFAR-10 / DenseNet-100 | 4.10 | **3.44** |
| CIFAR-100 / DenseNet-100 | 20.20 | **17.41** |
| ImageNet / ResNet-50 | 23.62 | **23.04** |

→ **0.5 ~ 2.7% 绝对误差下降**，训练成本与单模型一致。

## 与 traditional ensemble 比较

- Traditional ensemble M 次独立训练：成本 M×，diversity 来自不同初始化。
- Snapshot ensemble：成本 1×，diversity 来自 cyclic LR 跳出不同 basin。
- 论文证明：CIFAR-100 上 Snapshot ensemble (M=5) 与 5× 独立训练 ensemble 几乎打平（误差差 < 0.5%）。

## 对我们的启示

1. **NN 路线的"免费多样性"**：如果我们要训 MLPLOB / DeepLOB / BiN-CTABL，加 cosine warm restart 调度几乎零成本就能得到 M=5 ensemble，**比 multi-seed 独立训练便宜 5×**。
2. **GBDT 路线不直接适用**：LightGBM 没有 weight space 的 cyclic 概念；但类似思想 = 训 N 个不同 `feature_fraction` / `bagging_fraction` 的 LightGBM 模型（本研究的方案 11 / 方案 3）。
3. **失败模式**：cycle 太短 → snapshots 没差异（≈ 多 seed simple mean，类似 T6b 失败）。建议 cycle 长度 ≥ 0.5 epoch 或者 ≥ 50 LightGBM rounds。
4. **inference 成本**：M× forward → 在 LOB tabular 上不是问题（每 sample 几 ms），可以放心用。

## 实现伪代码（PyTorch）

```python
optim = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optim, T_0=cycle_len_in_iters, T_mult=1
)
snapshots = []
for it, batch in enumerate(loader):
    train_step(model, batch)
    sched.step()
    if (it + 1) % cycle_len_in_iters == 0:
        snapshots.append(copy.deepcopy(model.state_dict()))

# inference
preds = [forward_with(s, x) for s in snapshots]
final = torch.stack(preds).mean(0)
```
