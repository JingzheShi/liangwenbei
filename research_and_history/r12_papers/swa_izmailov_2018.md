# Averaging Weights Leads to Wider Optima and Better Generalization (SWA)

**作者**：Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, Andrew Gordon Wilson
**会议/年**：UAI 2018
**arXiv**：[1803.05407](https://arxiv.org/abs/1803.05407)

---

## 核心 idea

**SWA (Stochastic Weight Averaging)** = 训练后期把 SGD 轨迹上的 N 个 weight 做 running mean。

不是把 N 个模型的 prediction 平均（那是 ensemble），而是**weight space 直接平均**得到一组新 weight。这组 SWA weight 落在一个 **flatter minimum**，generalization 比 SGD 终点更好。

## 算法

1. 用 SGD 训到接近收敛（pre-training phase, ~75% 总 budget）。
2. 切换到 **constant learning rate** （或 cyclical schedule），继续训 25%。
3. 每个 epoch 末把当前 weight 加到 running average：
   $$w_\mathrm{SWA}^{(n)} = \frac{n \cdot w_\mathrm{SWA}^{(n-1)} + w^{(n)}}{n+1}$$
4. 最后 BatchNorm 层用 SWA weight 做一遍 train data 的 forward，重算 BN running stats（**这步不能省**）。

## 为什么"flat minimum"更好

- SGD 在 loss landscape 上终点是一个**点估计**（可能落在 narrow basin）。
- SWA 把多个高 lr 阶段的解平均 → 平均向量在 weight space 是一个**flatter region 的中心**。
- Flatter minima 对 train→test distribution shift 更鲁棒（Hochreiter & Schmidhuber 1997 的 flat-minimum 假说）。

## 与 Snapshot ensemble / FGE 关系

- **Fast Geometric Ensembling (FGE, Garipov 2018)**：训 N 个 high-LR cycle 的 snapshot，**inference 时 ensemble**（保留 N 份 weight）。
- **SWA** 是 FGE 的"压缩版"：用 weight averaging 把 N 份压成 1 份。
- 实验上：SWA ≈ FGE 性能，但 inference 成本 = 1×（FGE = N×）。

## 实验结果（节选）

| 数据集 / 模型 | SGD | SWA | Δ |
|---|---|---|---|
| CIFAR-100 / PreResNet-164 | 22.20 | **20.83** | -1.37 |
| ImageNet / DenseNet-161 | 22.86 | **22.27** | -0.59 |
| CIFAR-10 / Shake-Shake | 2.86 | **2.68** | -0.18 |

## 对我们的启示

1. **NN 路线建议把 SWA 当默认开关**：PyTorch 已经原生支持（`torch.optim.swa_utils.AveragedModel`、`SWALR`、`update_bn`），代码几乎免费。
2. **配合 Snapshot Ensemble 可叠加**：先 SWA 得到一组 flat-minimum weight，再 Snapshot 出 M 份不同 cycle 的 SWA-weight → M 份 SWA model ensemble。
3. **GBDT 不适用**：LightGBM 没有 weight space 的"几何"概念；boosting iteration 之间 weight 可叠加但不正交。
4. **对 LOSO 任务重要**：SWA 找的 flat region 通常跨不同 sub-distribution（不同 sym 的 distribution 差异）更鲁棒——这正是 LOSO 关心的。
5. **失败模式**：pre-training 没收敛就开始 SWA → averaging junk weight；高 lr phase 太短 → average 退化为 single-checkpoint。

## 实现伪代码（PyTorch）

```python
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

model = MLPLOB(...).cuda()
optim = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
swa_model = AveragedModel(model)
swa_scheduler = SWALR(optim, swa_lr=0.01)

for epoch in range(100):
    train_one_epoch(model, optim, train_loader)
    if epoch >= 75:                # pretrain done
        swa_scheduler.step()
        swa_model.update_parameters(model)

update_bn(train_loader, swa_model)  # 必须！
torch.save(swa_model.state_dict(), 'swa.pt')
```
