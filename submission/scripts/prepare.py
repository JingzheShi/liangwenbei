"""prepare.py — 把一个 iter 目录打包成提交 zip + 自动登记 SUBMISSION_LOG.md。

用法：
    python submission/scripts/prepare.py \\
        --iter 001 \\
        --name "deeplob-mlofi" \\
        --src submission/iter_001_deeplob-mlofi \\
        --note "MLOFI multi-window features"

可选：
    --skip-validate         不跑契约校验（不推荐；只在权重特别大、调试时用）
    --pkg-name <name>       LocalEvaluator 解压后的伪 pkg 名，默认 mmpc
    --no-log                不写 SUBMISSION_LOG.md（dry run）

行为：
1. 验证 src 目录结构合法（含必备文件、无子文件夹）
2. 调用 src/submit/package.py 打包到 <src>/submission.zip
3. 调用 src/submit/validate_contract.py 端到端校验（解压 + LocalEvaluator 跑通）
4. 自动追加一行到 submission/SUBMISSION_LOG.md
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SUBMISSION_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(SUBMISSION_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.submit.package import package_submission  # noqa: E402
from src.submit.validate_contract import validate_contract  # noqa: E402

LOG_PATH = os.path.join(SUBMISSION_DIR, "SUBMISSION_LOG.md")


def _ensure_iter_dirname(iter_id: str, name: str, src: str) -> str:
    """src 路径与 iter/name 一致性校验。建议命名：iter_<NNN>_<slug>。"""
    if not re.fullmatch(r"\d{3}", iter_id):
        raise ValueError(f"iter 必须是 3 位数字，例如 001。得到 {iter_id!r}")
    expected = f"iter_{iter_id}_{name}"
    base = os.path.basename(os.path.normpath(src))
    if base != expected:
        print(
            f"[warn] src 目录名 {base!r} 与 expected {expected!r} 不一致；"
            "继续，但建议规范化。",
            file=sys.stderr,
        )
    return expected


def _validate_no_subdirs(src: str) -> list[str]:
    """src 目录下不能有子文件夹（除了 __pycache__ 之类的隐藏/临时目录）。"""
    bad: list[str] = []
    for entry in sorted(os.listdir(src)):
        full = os.path.join(src, entry)
        if os.path.isdir(full) and not entry.startswith(("_", ".")):
            bad.append(entry)
    return bad


def _zip_listing(zip_path: str) -> list[str]:
    import zipfile
    with zipfile.ZipFile(zip_path) as zf:
        return sorted(zf.namelist())


def _append_log(
    iter_id: str,
    name: str,
    note: str,
    zip_path: str,
    contract_report: dict | None,
) -> None:
    if not os.path.isfile(LOG_PATH):
        raise FileNotFoundError(f"SUBMISSION_LOG.md 不存在: {LOG_PATH}")

    today = dt.date.today().isoformat()
    size_mb = os.path.getsize(zip_path) / 1024 / 1024
    n_files = len(_zip_listing(zip_path))

    pred_dist_str = "_(待填)_"
    if contract_report and "pred_dist_per_head" in contract_report:
        # 简短摘要：取 best 看起来分布最多样的 horizon
        pdh = contract_report["pred_dist_per_head"]
        pred_dist_str = ", ".join(f"{k}=[{a},{b},{c}]" for k, (a, b, c) in pdh.items())

    line = (
        f"| {iter_id}  | {today} | {name} | _(待填)_ | _(待跑 PnL)_ "
        f"| _(待填)_ | _未提交_ | _未提交_ | _(待填)_ "
        f"| zip={size_mb:.2f}MB, {n_files} files; pred_dist={pred_dist_str}; note={note} |\n"
    )
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line)
    print(f"[log] 追加一行到 {LOG_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", required=True, help="iter 编号，3 位数字（例如 001）")
    ap.add_argument("--name", required=True, help="iter 简名，slug-style（例如 deeplob-mlofi）")
    ap.add_argument("--src", required=True, help="该 iter 的源目录")
    ap.add_argument("--note", default="", help="一句话说明，会追加到 SUBMISSION_LOG.md")
    ap.add_argument("--pkg-name", default="mmpc", help="LocalEvaluator 解压后的伪 pkg 名")
    ap.add_argument("--skip-validate", action="store_true", help="跳过 validate_contract（不推荐）")
    ap.add_argument("--no-log", action="store_true", help="不写 SUBMISSION_LOG.md（dry run）")
    ap.add_argument(
        "--data-dir", default="data",
        help="LocalEvaluator 用的 parquet 数据根目录（相对 ROOT）",
    )
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    if not os.path.isdir(src):
        sys.exit(f"[fail] src 不是目录: {src}")

    _ensure_iter_dirname(args.iter, args.name, src)

    bad = _validate_no_subdirs(src)
    if bad:
        sys.exit(f"[fail] src 含子目录（平台禁止）: {bad}")

    zip_path = os.path.join(src, "submission.zip")
    # 重要：package.py 会遍历 src 目录下所有文件；如果 output_zip 也落在 src 里，
    # 它会把（部分写好的）zip 自身打进去 → 体积爆炸 / 损坏。
    # 解决：先写到临时位置，最后再 mv 进 src。
    if os.path.exists(zip_path):
        os.remove(zip_path)
    tmp_zip = tempfile.NamedTemporaryFile(
        suffix=".zip", prefix="prepare_", delete=False
    ).name
    print(f"[1/3] 打包 → {zip_path}（先写到 {tmp_zip} 再 mv）")
    try:
        pkg_report = package_submission(
            src_dir=src,
            output_zip=tmp_zip,
            validate_first=True,  # 用随机数据 smoke test
        )
        shutil.move(tmp_zip, zip_path)
        pkg_report["package"]["output_zip"] = zip_path
    finally:
        if os.path.exists(tmp_zip):
            os.remove(tmp_zip)
    print(json.dumps(pkg_report["package"], indent=2, ensure_ascii=False))

    contract_report = None
    if not args.skip_validate:
        print(f"[2/3] 契约端到端校验（解压 + LocalEvaluator 跑通）")
        data_dir_abs = os.path.join(ROOT, args.data_dir)
        contract_report = validate_contract(
            src_dir=src,
            data_dir=data_dir_abs,
            pkg_name=args.pkg_name,
            n_batches_to_run=2,
        )
        if not contract_report["all_passed"]:
            sys.exit("[fail] 契约校验未通过；请修复后重跑")
        print("[ok] 所有契约断言通过")

    if not args.no_log:
        print(f"[3/3] 登记到 SUBMISSION_LOG.md")
        _append_log(args.iter, args.name, args.note, zip_path, contract_report)

    print(f"\n[OK] iter_{args.iter}_{args.name} 准备完成")
    print(f"     zip: {zip_path} ({os.path.getsize(zip_path)/1024/1024:.2f} MB)")


if __name__ == "__main__":
    main()
