"""把一个 src 目录打包成 zip，再解压到一个全新临时目录、用 LocalEvaluator
跑通，断言形状/类型/取值符合契约。

设计意图：尽量真实地模拟"评测平台拿到 zip → 解压 → import → 调 predict"
那一刻的环境，避免开发者本地侥幸跑通但平台失败。

断言（PDF 协议要求）：
  - len(predictions) == batch_size （每个 batch）
  - len(predictions[0]) == len(label)
  - 每个元素 ∈ {0, 1, 2}
  - 整轮预测不报错
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.submit.local_evaluator import LocalEvaluator, load_test_sessions  # noqa: E402
from src.submit.package import package_submission  # noqa: E402


def _extract_to_tempdir(zip_path: str, pkg_name: str) -> str:
    """解压到 tempdir/<pkg_name>/，返回 tempdir 路径（其下含 pkg_name 子目录）。"""
    tmpdir = tempfile.mkdtemp(prefix="liangwenbei_validate_")
    submit_root = os.path.join(tmpdir, pkg_name)
    os.makedirs(submit_root, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        # 平台拿到的 zip 顶层无嵌套；解压到 submit_root 下。
        zf.extractall(submit_root)
        # 安全检查：zip 内不能有路径分隔符（除了文件名）
        for name in zf.namelist():
            if "/" in name or "\\" in name:
                raise AssertionError(
                    f"zip 内不允许嵌套子目录，违规项: {name!r}"
                )
    return tmpdir, submit_root


def validate_contract(
    src_dir: str,
    sym: int = 0,
    date: int = 119,
    session: str = "am",
    data_dir: str = "data",
    pkg_name: str = "mmpc",
    n_batches_to_run: int = 2,
    keep_tempdir: bool = False,
) -> dict:
    report: dict = {"src_dir": os.path.abspath(src_dir),
                    "pkg_name": pkg_name,
                    "asserts": []}

    # 1) 打包
    tmp_zip = tempfile.NamedTemporaryFile(suffix=".zip", delete=False).name
    try:
        pkg_report = package_submission(
            src_dir=src_dir,
            output_zip=tmp_zip,
            extra_files=None,
            validate_first=False,  # 我们下面会用真实数据再跑一次，这里先不重复
        )
        report["package"] = {"size_mb": pkg_report["package"]["size_mb"],
                             "files": pkg_report["package"]["written"]}
        report["asserts"].append(("zip 创建成功", True))

        # 2) 解压到 tempdir/<pkg_name>/
        tmpdir, submit_root = _extract_to_tempdir(tmp_zip, pkg_name)
        report["tempdir"] = tmpdir
        try:
            # 顶层文件 sanity
            files = sorted(os.listdir(submit_root))
            for must in ("Predictor.py", "config.json", "requirements.txt"):
                assert must in files, f"解压后缺 {must}"
                report["asserts"].append((f"解压后存在 {must}", True))

            # 3) 加载测试 session
            sessions = load_test_sessions([(sym, date, session)], data_dir=data_dir)

            # 4) 用 LocalEvaluator 跑（shuffle + anonymize 均开启，模拟平台）
            ev = LocalEvaluator(
                submit_dir=submit_root,
                test_data=sessions,
                shuffle=True,
                anonymize=True,
                progress=False,
            )

            # 5) 每个 batch 单独检查 shape——这是契约的硬性要求
            B = ev.batch_size
            K = len(ev.label_cols)

            n_check = min(n_batches_to_run, (len(ev.eval_points) + B - 1) // B)
            for bi in range(n_check):
                lo = bi * B
                hi = min(lo + B, len(ev.eval_points))
                batch_pts = ev.eval_points[lo:hi]
                batch_dfs = [ev._make_window_df(p) for p in batch_pts]

                # 形状预检
                for df in batch_dfs:
                    assert df.shape == (100, len(ev.feature_cols)), (
                        f"输入 DataFrame 形状错误：期望 (100, {len(ev.feature_cols)})，"
                        f"实际 {df.shape}"
                    )
                    assert list(df.columns) == ev.feature_cols, "输入列名不严格匹配 config.feature"

                preds = ev.predictor.predict(batch_dfs)
                expected_len = len(batch_pts)
                assert isinstance(preds, list), f"predict 返回必须是 list，实际 {type(preds)}"
                assert len(preds) == expected_len, (
                    f"batch#{bi}: len(predictions)={len(preds)} ≠ expected_batch_size={expected_len}"
                )
                report["asserts"].append((f"batch#{bi}: len==batch_size ({expected_len})", True))
                for j, inner in enumerate(preds):
                    assert isinstance(inner, (list, tuple)), (
                        f"batch#{bi}[{j}] 内层必须是 list/tuple，实际 {type(inner)}"
                    )
                    assert len(inner) == K, (
                        f"batch#{bi}[{j}]: len={len(inner)} ≠ len(label)={K}"
                    )
                    for v in inner:
                        iv = int(v)
                        assert iv in (0, 1, 2), (
                            f"batch#{bi}[{j}]: 非法类标 {v}"
                        )
                report["asserts"].append((f"batch#{bi}: 内层 len=={K}, 元素 ∈ {{0,1,2}}", True))

            # 6) 跑一次完整 run() 确保不报错
            res = ev.run()
            assert res.predictions.shape == (len(ev.eval_points), K), (
                f"run() 输出形状错误：{res.predictions.shape}"
            )
            assert ((res.predictions >= 0) & (res.predictions <= 2)).all(), (
                "run() 输出范围越界"
            )
            report["asserts"].append((f"run() 完整执行 → {res.predictions.shape}", True))
            report["timing"] = res.timing
            report["pred_dist_per_head"] = {
                lc: np.bincount(res.predictions[:, ki], minlength=3).tolist()
                for ki, lc in enumerate(ev.label_cols)
            }
            ev.cleanup()
        finally:
            if not keep_tempdir:
                shutil.rmtree(tmpdir, ignore_errors=True)
            # sys.path 中临时插的项不主动清理 —— 一次性脚本进程结束即可
    finally:
        if os.path.exists(tmp_zip):
            os.remove(tmp_zip)

    report["all_passed"] = all(ok for _, ok in report["asserts"])
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="examples/mmpc_demo")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--sym", type=int, default=0)
    ap.add_argument("--date", type=int, default=119)
    ap.add_argument("--session", default="am", choices=["am", "pm"])
    ap.add_argument("--pkg-name", default="mmpc",
                    help="解压后包名（决定 from <pkg>.Predictor import）")
    ap.add_argument("--n-batches", type=int, default=2)
    ap.add_argument("--keep-tempdir", action="store_true")
    args = ap.parse_args()

    rep = validate_contract(
        src_dir=args.src,
        sym=args.sym,
        date=args.date,
        session=args.session,
        data_dir=args.data_dir,
        pkg_name=args.pkg_name,
        n_batches_to_run=args.n_batches,
        keep_tempdir=args.keep_tempdir,
    )
    print(json.dumps(rep, indent=2, ensure_ascii=False, default=str))
    print("\n========== Assert checklist ==========")
    for name, ok in rep["asserts"]:
        mark = "✓" if ok else "✗"
        print(f"  {mark} {name}")
    if rep["all_passed"]:
        print("\n[PASS] 所有断言通过 ✓")
    else:
        print("\n[FAIL] 有断言失败 ✗")
        sys.exit(1)


if __name__ == "__main__":
    main()
