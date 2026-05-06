"""把 src_dir 打包成可提交的 zip。

平台要求（PDF Step 0 红字）：所有文件直接放在 zip 顶层，不要嵌套子目录。
也就是说提交者本地的目录结构 examples/mmpc_demo/Predictor.py 在 zip 里
就是顶层的 Predictor.py。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import zipfile
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

REQUIRED_FILES = ("Predictor.py", "config.json", "requirements.txt")
REQUIRED_CONFIG_KEYS = ("python_version", "batch", "feature", "label")
MAX_ZIP_BYTES = 2 * 1024 * 1024 * 1024  # 2GB


def _find_model_files(src_dir: str) -> List[str]:
    """启发式收集模型权重 / 必要源码（除了必填三件套）。"""
    out: List[str] = []
    for fn in sorted(os.listdir(src_dir)):
        full = os.path.join(src_dir, fn)
        if not os.path.isfile(full):
            continue
        if fn in REQUIRED_FILES:
            continue
        if fn.startswith("."):
            continue
        # 忽略隐藏 / pycache
        if fn.endswith(".pyc"):
            continue
        # 模型权重 / model.py / 其他源码都收
        out.append(fn)
    return out


def _validate_dir(src_dir: str) -> dict:
    info = {"src_dir": os.path.abspath(src_dir), "files": [], "missing": [],
            "config": None, "model_candidates": []}
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"src_dir 不存在: {src_dir}")
    for f in REQUIRED_FILES:
        if not os.path.isfile(os.path.join(src_dir, f)):
            info["missing"].append(f)
    if info["missing"]:
        raise FileNotFoundError(f"缺少必要文件: {info['missing']}")

    cfg_path = os.path.join(src_dir, "config.json")
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    missing_keys = [k for k in REQUIRED_CONFIG_KEYS if k not in cfg]
    if missing_keys:
        raise ValueError(f"config.json 缺字段: {missing_keys}")
    if not isinstance(cfg["batch"], int) or cfg["batch"] <= 0:
        raise ValueError(f"config.batch 必须是正整数: {cfg['batch']!r}")
    if not isinstance(cfg["feature"], list) or not cfg["feature"]:
        raise ValueError("config.feature 必须是非空 list")
    if not isinstance(cfg["label"], list) or not cfg["label"]:
        raise ValueError("config.label 必须是非空 list")
    info["config"] = cfg

    info["model_candidates"] = [
        f for f in _find_model_files(src_dir)
        if f.endswith((".pt", ".pth", ".pkl", ".bin", ".onnx", ".safetensors"))
    ]
    info["files"] = sorted(REQUIRED_FILES + tuple(_find_model_files(src_dir)))
    return info


def _smoke_test_predictor(src_dir: str, cfg: dict) -> dict:
    """临时把 src_dir 当包导入并跑一次 predict。"""
    # 用 local_evaluator 那套 import 逻辑，避免分裂。
    HERE = os.path.dirname(os.path.abspath(__file__))
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    from local_evaluator import _import_predictor  # type: ignore

    Predictor, created_init = _import_predictor(src_dir)
    try:
        predictor = Predictor()
        feature_cols = list(cfg["feature"])
        label_cols = list(cfg["label"])
        # 制造 sample DataFrame：100 行，列名严格 = feature
        rng = np.random.default_rng(0)
        sample = pd.DataFrame(
            rng.standard_normal((100, len(feature_cols))).astype(np.float32),
            columns=feature_cols,
        )
        # 用 1 个 batch 做最小调用（平台 batch 可能很大，但形状校验只要 1 即可）
        out = predictor.predict([sample])
        if not isinstance(out, list) or len(out) != 1:
            raise RuntimeError(f"predict 返回 list 长度应 = 1，实际 {out}")
        inner = out[0]
        if not isinstance(inner, (list, tuple)) or len(inner) != len(label_cols):
            raise RuntimeError(
                f"predict 返回内层长度应 = {len(label_cols)}，实际 {inner}"
            )
        for v in inner:
            iv = int(v)
            if iv not in (0, 1, 2):
                raise RuntimeError(f"predict 返回非法类标 {v}（必须 ∈ {{0,1,2}}）")
        return {"smoke_test": "ok", "sample_output": [int(v) for v in inner]}
    finally:
        if created_init:
            init_py = os.path.join(src_dir, "__init__.py")
            if os.path.isfile(init_py):
                try:
                    os.remove(init_py)
                except OSError:
                    pass


def _zip_contents_flat(src_dir: str, output_zip: str, extra_files: Optional[List[str]]) -> dict:
    """把 src_dir 下所有文件写入 zip，且都放在顶层（无嵌套目录）。"""
    src_dir = os.path.abspath(src_dir)
    if os.path.exists(output_zip):
        os.remove(output_zip)
    written: List[str] = []
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for fn in sorted(os.listdir(src_dir)):
            full = os.path.join(src_dir, fn)
            if not os.path.isfile(full):
                continue
            if fn.startswith(".") or fn.endswith(".pyc"):
                continue
            if fn == "__init__.py":
                # 这是 LocalEvaluator 临时塞的，不该进 zip
                continue
            zf.write(full, arcname=fn)
            written.append(fn)
        for ef in (extra_files or []):
            if not os.path.isfile(ef):
                raise FileNotFoundError(f"extra_files 不存在: {ef}")
            arc = os.path.basename(ef)
            zf.write(ef, arcname=arc)
            written.append(arc)
    size = os.path.getsize(output_zip)
    return {"output_zip": os.path.abspath(output_zip), "size_bytes": size,
            "size_mb": size / 1024 / 1024, "written": written}


def package_submission(
    src_dir: str,
    output_zip: str,
    extra_files: Optional[Iterable[str]] = None,
    validate_first: bool = True,
) -> dict:
    """打包提交 zip。

    步骤：
      1. 校验 src_dir 必备文件
      2. 校验 config.json 字段
      3. 实例化 Predictor 并用 sample 调用 predict() 校验返回 shape
      4. 打包到 zip（顶层无嵌套）
      5. 返回大小并断言 < 2GB
    """
    extra = list(extra_files) if extra_files else []
    report: dict = {"src_dir": os.path.abspath(src_dir),
                    "output_zip": os.path.abspath(output_zip),
                    "extra_files": extra}

    val = _validate_dir(src_dir)
    report["validate"] = {
        "missing": val["missing"],
        "config_keys": list(val["config"].keys()),
        "n_features": len(val["config"]["feature"]),
        "n_labels": len(val["config"]["label"]),
        "files": val["files"],
        "model_candidates": val["model_candidates"],
    }

    if validate_first:
        report["smoke"] = _smoke_test_predictor(src_dir, val["config"])

    pkg = _zip_contents_flat(src_dir, output_zip, extra)
    report["package"] = pkg
    if pkg["size_bytes"] > MAX_ZIP_BYTES:
        raise RuntimeError(
            f"提交 zip 超过 2GB 限制：{pkg['size_mb']:.1f} MB"
        )
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="源目录，含 Predictor.py 等")
    ap.add_argument("--output", required=True, help="输出 zip 路径")
    ap.add_argument("--extra", nargs="*", default=[], help="附加文件路径列表")
    ap.add_argument("--no-validate", action="store_true", help="跳过 import + 调 predict 的烟测")
    args = ap.parse_args()

    report = package_submission(
        src_dir=args.src,
        output_zip=args.output,
        extra_files=args.extra,
        validate_first=not args.no_validate,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n[OK] 已生成 {report['output_zip']} "
          f"({report['package']['size_mb']:.2f} MB, {len(report['package']['written'])} 个文件)")


if __name__ == "__main__":
    main()
