# 评测环境兼容性笔记

调研日期：2026-05-06。资料来源：[PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/) +
[NVIDIA CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/) + 社区反馈。

## §1 平台规格（重述）

| 项 | 值 |
| --- | --- |
| GPU | 1 × NVIDIA RTX A6000 |
| GPU compute capability | **8.6**（Ampere） |
| GPU 显存 | 48GB |
| Driver | `580.126.09` (CUDA 13 LTS branch) |
| CUDA Toolkit (driver 报告) | `13.0` |
| CPU | 16 核 |
| OS（推断） | Linux x86_64 |

## §2 PyTorch ↔ CUDA 兼容性矩阵

> NVIDIA driver 580 是 **LTS 长期支持** 分支，向前兼容所有 ≤13.0 的 CUDA Toolkit。
> 即：driver 580 既能跑 cu130 wheel（原生），也能跑 cu128 / cu126 / cu118 wheel（minor version compatibility）。

| PyTorch 版本 | 官方 wheel CUDA 选项 | 推荐用于本平台 |
| --- | --- | --- |
| **2.10.0** | cu126, cu128, **cu130**, ROCm7.1 | ✅ **首选**（与 CUDA 13.0 原生匹配） |
| 2.9.1 | cu126, cu128, **cu130** | ✅ 备选 |
| 2.9.0 | cu126, cu128, **cu130** | ✅ 备选 |
| 2.8.0 | cu126, cu128, cu129 | ⚠️ cu129 也可跑（向前兼容），但非原生 |
| 2.7.x | cu118, cu126, cu128 | ⚠️ 走 cu128，向前兼容 |
| 2.6.0 | cu118, cu124, cu126 | ⚠️ 走 cu126，向前兼容；老 |
| ≤ 2.5 | 没有 cu126+ wheel | ❌ 不推荐（虽然能装但落后多个 release） |

### §2.1 推荐组合

```
torch==2.10.0+cu130
torchvision==0.25.0+cu130   # 如需视觉算子
torchaudio==2.10.0+cu130    # 如需音频算子
```

或保守一点（更稳定，社区版本数更多）：

```
torch==2.8.0+cu128
```

### §2.2 在 `requirements.txt` 中指定 CUDA wheel

```text
# requirements.txt
--index-url https://download.pytorch.org/whl/cu130
torch==2.10.0
numpy==2.4.3
pandas==3.0.1
```

或用 `--extra-index-url` 不破坏 PyPI 默认源：

```text
--extra-index-url https://download.pytorch.org/whl/cu130
torch==2.10.0
numpy==2.4.3
pandas==3.0.1
```

⚠️ **如果只写 `torch==2.10.0`**（无 index URL）：pip 会从 PyPI 拿默认 wheel，
默认 wheel 在不同 PyTorch 版本上对应的 CUDA 不同（最近通常是 cu126）。这在
driver 580 上仍能跑（向前兼容），但 throughput 略损（不走原生 cu130 算子优化）。

## §3 A6000 (compute capability 8.6) 兼容性

A6000 是 Ampere 架构，sm_86。所有现代 PyTorch wheel（≥ 2.0，cu118+）都内置 sm_80/86/89 PTX。
**没有兼容性陷阱**。

⚠️ 但是：cu128+ wheel 不再支持 compute capability < 7.5 的旧卡（K80、P40 这种），
对 A6000 (8.6) 完全没影响。

## §4 16 CPU 核 — 数值库线程数

平台跑 CPU 评测时给 16 核。numpy/pandas/torch CPU 算子默认会**抢光所有可见核**，
有时反而拖慢（线程切换开销）。建议在 `Predictor.__init__` 里**显式限制**：

```python
import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
import torch
torch.set_num_threads(8)
torch.set_num_interop_threads(2)
```

> **重要**：`os.environ` 必须在 `import torch / numpy` **之前**设置才生效。
> 在 `__init__` 里设其实已经太晚（`torch` 已被 `Predictor.py` 顶部 import 走）。
> 所以正确做法是把这几行放到 `Predictor.py` 文件**最顶部**，import 之前。

注意：**只在 CPU 评测模式下需要**这些设置。GPU 模式下 OMP/MKL 几乎无影响。

## §5 pipreqs 用法的常见坑

### §5.1 漏检 import

`pipreqs` 通过静态分析 import 语句生成 `requirements.txt`，**会漏的情况**：
- 动态 import（`importlib.import_module(name)`）
- 字符串构造的 import（`exec("import ...")`）
- 通过 entry_points 间接依赖的库

→ 防御：生成后**人工过一遍**自己的 import 列表。

### §5.2 包名不一致

`pip install` 名 ≠ `import` 名是常见情况。pipreqs 会写 `import` 名，可能装不上：

| `import` 名 | `pip install` 名 |
| --- | --- |
| `cv2` | `opencv-python` |
| `sklearn` | `scikit-learn` |
| `PIL` | `Pillow` |
| `yaml` | `PyYAML` |

→ 防御：检查 pipreqs 输出，把不对的改正确。

### §5.3 版本号过期

pipreqs 默认抓 PyPI 上的最新版，可能与本地实际装的版本不一致。

→ 防御：每行用 `pip show <pkg>` 复查并改成本地版本：
```bash
pip show torch | grep Version  # → Version: 2.10.0
```

### §5.4 未安装的包

pipreqs 不要求本地装着这个包，只看代码 import。意味着：
- 如果代码里有 `import foo` 但本地没装，pipreqs 仍会写到 requirements 里
- 平台装 `foo` 时可能版本不对、源没有，崩

→ 防御：本地建空环境装一遍 requirements，确认能装上、能 import。

### §5.5 推荐 pipreqs 命令

```bash
cd submission/iter_<NNN>_<name>/
pipreqs ./ --encoding=utf8 --force --mode no-pin   # no-pin 先无版本，再人工写
# 然后人工编辑加版本号
```

## §6 验证清单：提交前的环境隔离测试

```bash
# 在新空环境里测一次（最接近平台行为）
conda create -n test_submit python=3.10 -y
conda activate test_submit

cd submission/iter_<NNN>_<name>/
pip install -r requirements.txt --quiet

# 模拟平台调用
python -c "
from Predictor import Predictor
import pandas as pd, numpy as np
p = Predictor()
df = pd.DataFrame(np.random.randn(100, 12).astype(np.float32),
                  columns=['bid1','bid2','bid3','ask1','ask2','ask3',
                           'bsize1','bsize2','bsize3','asize1','asize2','asize3'])
out = p.predict([df] * 4)
assert len(out) == 4 and len(out[0]) == 5 and all(v in (0,1,2) for v in out[0])
print('OK', out)
"

conda deactivate
conda env remove -n test_submit -y
```

## §7 已知风险

1. **driver 580 的 minor version compatibility 边界**——
   理论上 cu118 wheel 也能在 driver 580 跑，但越早的 cu wheel 在 driver 580 上越没人测过。
   → 不要使用 cu11x；至少用 cu126+。
2. **Python 版本对齐**——
   `config.json.python_version` 是平台启动 Python 的依据。本地训练用什么版本，配置里就写什么。
   不要本地训练用 3.11、配置写 3.10，会出现"训练时 dataclass 行为不同"等隐性 bug。
3. **PyTorch nightly**——
   不要用 nightly。版本号（如 `2.11.0.dev2026...`）在 PyPI 找不到，平台装不上。

---

## 来源

- [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/)
- [NVIDIA CUDA Compatibility — Driver vs Toolkit](https://docs.nvidia.com/deploy/cuda-compatibility/)
- [What's New in CUDA Toolkit 13.0](https://developer.nvidia.com/blog/whats-new-and-important-in-cuda-toolkit-13-0/)
- 社区讨论：[PyTorch Forums - PyTorch GPU compatibility hell](https://discuss.pytorch.org/t/how-to-deal-with-pytorch-gpu-compatibility-hell/224182)
