"""sanity_check.py — 提交前的飞行检查清单。

把每条规则单独验一次，输出 ✓/✗ + 原因。任何 ✗ 都应在提交前修掉。

用法：
    python submission/scripts/sanity_check.py --src submission/iter_001_xxx
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
SUBMISSION_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(SUBMISSION_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

MAX_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB
REQUIRED_FILES = ("Predictor.py", "config.json", "requirements.txt")
DEV_LIBS_BLACKLIST = (
    "jupyter", "ipython", "ipykernel", "jupyterlab", "notebook",
    "tensorboard", "tensorboardx",
    "wandb", "mlflow", "neptune",
    "matplotlib", "seaborn", "plotly",
    "tqdm",  # tqdm 不算严格禁用，但通常推理不需要 → 警告
    "pytest", "black", "ruff", "mypy", "isort",
)


class Check:
    def __init__(self, name: str):
        self.name = name
        self.ok = True
        self.detail = ""

    def fail(self, why: str) -> "Check":
        self.ok = False
        self.detail = why
        return self

    def warn(self, why: str) -> "Check":
        self.ok = True  # 不算 fail
        self.detail = f"WARN: {why}"
        return self

    def passed(self, info: str = "") -> "Check":
        self.ok = True
        self.detail = info
        return self

    def __str__(self) -> str:
        mark = "✓" if self.ok and not self.detail.startswith("WARN") else (
            "⚠" if self.detail.startswith("WARN") else "✗"
        )
        return f"  {mark} [{self.name}] {self.detail}"


def check_required_files(src: str) -> list[Check]:
    out = []
    for fn in REQUIRED_FILES:
        c = Check(f"必备文件 {fn}")
        if os.path.isfile(os.path.join(src, fn)):
            out.append(c.passed(f"存在"))
        else:
            out.append(c.fail(f"缺失：{fn}"))
    return out


def check_no_subdirs(src: str) -> Check:
    c = Check("无业务子目录（平台禁止）")
    bad = [
        e for e in os.listdir(src)
        if os.path.isdir(os.path.join(src, e)) and not e.startswith(("_", "."))
    ]
    return c.passed("无") if not bad else c.fail(f"发现子目录: {bad}")


def check_config(src: str) -> list[Check]:
    out = []
    p = os.path.join(src, "config.json")
    c = Check("config.json 可解析 JSON")
    try:
        with open(p, encoding="utf-8") as fh:
            cfg = json.load(fh)
    except Exception as e:
        out.append(c.fail(f"json 解析失败: {e}"))
        return out
    out.append(c.passed())

    for key in ("python_version", "batch", "feature", "label"):
        ck = Check(f"config.{key} 存在")
        if key not in cfg:
            out.append(ck.fail(f"缺字段 {key}"))
        else:
            out.append(ck.passed(repr(cfg[key])[:60]))

    if "batch" in cfg:
        ck = Check("config.batch 是正整数")
        b = cfg["batch"]
        if isinstance(b, int) and b > 0:
            out.append(ck.passed(str(b)))
        else:
            out.append(ck.fail(f"非正整数: {b!r}"))
        if isinstance(b, int) and b > 4096:
            out.append(Check("config.batch 不过大").warn(
                f"batch={b} 偏大，注意 OOM；A6000 48GB 视模型而定"
            ))

    if isinstance(cfg.get("label"), list):
        ck = Check("config.label = 5 个 horizon")
        expected = ["label_5", "label_10", "label_20", "label_40", "label_60"]
        if cfg["label"] == expected:
            out.append(ck.passed())
        else:
            out.append(ck.warn(f"非标准 5 horizon：{cfg['label']}"))

    if isinstance(cfg.get("feature"), list):
        ck = Check("config.feature 非空")
        if cfg["feature"]:
            out.append(ck.passed(f"{len(cfg['feature'])} cols"))
        else:
            out.append(ck.fail("空 list"))

    return out


def check_requirements(src: str) -> list[Check]:
    out = []
    p = os.path.join(src, "requirements.txt")
    if not os.path.isfile(p):
        return out
    with open(p, encoding="utf-8") as fh:
        lines = [
            ln.strip().split("#", 1)[0].strip()
            for ln in fh.readlines()
        ]
    pkgs = []
    for ln in lines:
        if not ln or ln.startswith("-") or ln.startswith("--"):
            continue
        # 形如 "torch==2.10.0" 或 "numpy>=1.24" 或 "torch"
        m = re.match(r"^([A-Za-z0-9_.\-]+)", ln)
        if m:
            pkgs.append(m.group(1).lower().replace("_", "-"))

    ck = Check("requirements 非空")
    out.append(ck.passed(f"{len(pkgs)} 个包") if pkgs else ck.fail("空"))

    blacklist_hits = [p for p in pkgs if p in DEV_LIBS_BLACKLIST]
    ck = Check("requirements 不含开发库")
    if blacklist_hits:
        out.append(ck.fail(f"含开发库: {blacklist_hits}（清掉重打）"))
    else:
        out.append(ck.passed())

    # 检查版本固定情况
    pinned = sum(1 for ln in lines if "==" in ln)
    unpinned = sum(1 for ln in lines if ln and not ln.startswith(("-", "#")) and "==" not in ln)
    ck = Check("requirements 写死版本号")
    if unpinned == 0:
        out.append(ck.passed(f"{pinned} 个全 pin"))
    else:
        out.append(ck.warn(f"{unpinned} 个未 pin（建议全部 ==X.Y.Z）"))

    return out


def check_model_size(src: str) -> Check:
    c = Check("模型权重文件 < 2GB（含其他附件）")
    total = 0
    for fn in os.listdir(src):
        full = os.path.join(src, fn)
        if os.path.isfile(full):
            total += os.path.getsize(full)
    mb = total / 1024 / 1024
    if total > MAX_BYTES:
        return c.fail(f"目录总大小 {mb:.0f} MB > 2GB")
    return c.passed(f"目录总大小 {mb:.2f} MB")


def check_predictor_loadable(src: str) -> list[Check]:
    """临时把 src 当包导入，实例化 Predictor 并跑一次 predict。"""
    out = []
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from src.submit.local_evaluator import _import_predictor  # noqa: E402

    c = Check("Predictor 可 import + 实例化")
    try:
        Predictor, created_init = _import_predictor(src)
        predictor = Predictor()
    except Exception as e:
        out.append(c.fail(f"{type(e).__name__}: {e}"))
        return out
    out.append(c.passed())

    # 跑一次 random predict
    cfg = json.load(open(os.path.join(src, "config.json"), encoding="utf-8"))
    feats = cfg["feature"]
    labels = cfg["label"]
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        rng.standard_normal((100, len(feats))).astype(np.float32),
        columns=feats,
    )

    for batch_size in (1, 4):
        c = Check(f"predict batch={batch_size} 形状/取值合法")
        try:
            preds = predictor.predict([df] * batch_size)
        except Exception as e:
            out.append(c.fail(f"{type(e).__name__}: {e}"))
            continue
        if not isinstance(preds, list) or len(preds) != batch_size:
            out.append(c.fail(f"外层 len={len(preds) if hasattr(preds,'__len__') else type(preds)}"))
            continue
        if any(len(p) != len(labels) for p in preds):
            out.append(c.fail(f"内层 len 不全 = {len(labels)}"))
            continue
        flat = [int(v) for p in preds for v in p]
        if any(v not in (0, 1, 2) for v in flat):
            out.append(c.fail(f"含非法类标: {set(flat) - {0,1,2}}"))
            continue
        out.append(c.passed(f"out[0]={preds[0]}"))

    if created_init:
        init_py = os.path.join(src, "__init__.py")
        if os.path.isfile(init_py):
            try:
                os.remove(init_py)
            except OSError:
                pass

    return out


def check_zip_layout(src: str) -> list[Check]:
    """如果 submission.zip 已存在，验证它"""
    out = []
    z = os.path.join(src, "submission.zip")
    if not os.path.isfile(z):
        out.append(Check("submission.zip 已生成").warn("未生成（可后续 prepare.py 生成）"))
        return out

    out.append(Check("submission.zip 存在").passed(f"{os.path.getsize(z)/1024/1024:.2f} MB"))

    c = Check("zip 顶层无嵌套子目录")
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist()
    bad = [n for n in names if "/" in n or "\\" in n]
    if bad:
        out.append(c.fail(f"含路径分隔符的项: {bad}"))
    else:
        out.append(c.passed(f"{len(names)} 个顶层文件"))

    c = Check("zip 大小 < 2GB")
    sz = os.path.getsize(z)
    if sz > MAX_BYTES:
        out.append(c.fail(f"{sz/1024/1024/1024:.2f} GB > 2GB"))
    else:
        out.append(c.passed(f"{sz/1024/1024:.2f} MB"))

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="iter 目录路径")
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    if not os.path.isdir(src):
        sys.exit(f"[fail] src 不是目录: {src}")

    print(f"=== sanity check: {src} ===")
    all_checks: list[Check] = []
    all_checks.extend(check_required_files(src))
    all_checks.append(check_no_subdirs(src))
    all_checks.extend(check_config(src))
    all_checks.extend(check_requirements(src))
    all_checks.append(check_model_size(src))
    all_checks.extend(check_predictor_loadable(src))
    all_checks.extend(check_zip_layout(src))

    n_pass = sum(1 for c in all_checks if c.ok and not c.detail.startswith("WARN"))
    n_warn = sum(1 for c in all_checks if c.detail.startswith("WARN"))
    n_fail = sum(1 for c in all_checks if not c.ok)

    print()
    for c in all_checks:
        print(c)
    print()
    print(f"pass={n_pass}  warn={n_warn}  fail={n_fail}")
    if n_fail > 0:
        print("[FAIL] 修掉所有 ✗ 后再提交")
        sys.exit(1)
    if n_warn > 0:
        print("[OK with warnings] 注意 ⚠ 项")
    else:
        print("[OK] 所有检查通过")


if __name__ == "__main__":
    main()
