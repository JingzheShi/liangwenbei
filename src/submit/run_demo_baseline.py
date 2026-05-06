"""用 examples/mmpc_demo/ 跑一次端到端基线推理，验证整条链路。

输出：
  - 控制台打印：每个 horizon 的 accuracy / precision_macro / 预测分布
  - parquet：analysis/mmpc_demo_baseline_predictions.parquet
        列：sym, date, session, t,
             true_label_5, ..., true_label_60,
             pred_label_5, ..., pred_label_60,
             midprice_t, midprice_t5, midprice_t10, midprice_t20, midprice_t40, midprice_t60
  - JSON：results 字典通过 stdout 也写一份摘要
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

# 确保能 import 包内模块
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.submit.local_evaluator import (  # noqa: E402
    DEFAULT_HORIZONS,
    LocalEvaluator,
    TestSession,
    load_test_sessions,
)


def per_head_metrics(true: np.ndarray, pred: np.ndarray, num_classes: int = 3) -> dict:
    """按 head 计算 acc 和 macro-precision；忽略 true == -1 的样本。"""
    mask = true != -1
    if mask.sum() == 0:
        return {"acc": float("nan"), "precision_macro": float("nan"), "n_valid": 0}
    t = true[mask]
    p = pred[mask]
    acc = float((t == p).mean())
    precs = []
    for c in range(num_classes):
        denom = (p == c).sum()
        if denom == 0:
            continue
        precs.append(float(((p == c) & (t == c)).sum() / denom))
    precision_macro = float(np.mean(precs)) if precs else 0.0
    return {"acc": acc, "precision_macro": precision_macro, "n_valid": int(mask.sum())}


def class_distribution(arr: np.ndarray, num_classes: int = 3) -> dict:
    arr = arr[arr != -1]
    n = max(len(arr), 1)
    counts = Counter(arr.tolist())
    return {str(c): {"count": int(counts.get(c, 0)), "frac": float(counts.get(c, 0) / n)}
            for c in range(num_classes)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit-dir", default="examples/mmpc_demo")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--sym", type=int, default=0)
    ap.add_argument("--date", type=int, default=119)
    ap.add_argument("--session", default="am", choices=["am", "pm"])
    ap.add_argument("--out-parquet", default="analysis/mmpc_demo_baseline_predictions.parquet")
    ap.add_argument("--shuffle", action="store_true",
                    help="模拟评测平台打乱评测点顺序（默认关闭，方便对照）")
    ap.add_argument("--no-anonymize", action="store_true",
                    help="关闭 date=0 替换（默认开启）")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_parquet) or ".", exist_ok=True)

    print(f"[INFO] 加载测试 session: sym={args.sym} date={args.date} {args.session}")
    sessions = load_test_sessions([(args.sym, args.date, args.session)], data_dir=args.data_dir)
    print(f"[INFO] 加载完成：1 个 session, {len(sessions[0].full_df)} 行")

    print(f"[INFO] 实例化 LocalEvaluator (submit_dir={args.submit_dir})")
    ev = LocalEvaluator(
        submit_dir=args.submit_dir,
        test_data=sessions,
        shuffle=args.shuffle,
        anonymize=not args.no_anonymize,
        progress=True,
    )
    print(f"[INFO] config: batch={ev.batch_size}, "
          f"#feature={len(ev.feature_cols)}, label={ev.label_cols}")
    print(f"[INFO] 评测点总数: {len(ev.eval_points)}")

    res = ev.run()
    ev.cleanup()

    # 计算各 head 指标
    metrics_per_head: dict = {}
    for ki, lc in enumerate(res.label_cols):
        m = per_head_metrics(res.true_labels[:, ki], res.predictions[:, ki])
        dist_pred = class_distribution(res.predictions[:, ki])
        dist_true = class_distribution(res.true_labels[:, ki])
        m["pred_dist"] = dist_pred
        m["true_dist"] = dist_true
        m["collapsed_to_class"] = (
            int(np.bincount(res.predictions[:, ki][res.predictions[:, ki] != -1],
                            minlength=3).argmax())
        )
        m["collapsed_frac"] = max(d["frac"] for d in dist_pred.values())
        metrics_per_head[lc] = m

    print("\n========== Per-head metrics ==========")
    for lc, m in metrics_per_head.items():
        print(f"\n[{lc}] n_valid={m['n_valid']} | acc={m['acc']:.4f} "
              f"prec_macro={m['precision_macro']:.4f}")
        print(f"  pred dist: " + ", ".join(
            f"{c}={m['pred_dist'][c]['count']}({m['pred_dist'][c]['frac']*100:.1f}%)"
            for c in ("0", "1", "2")))
        print(f"  true dist: " + ", ".join(
            f"{c}={m['true_dist'][c]['count']}({m['true_dist'][c]['frac']*100:.1f}%)"
            for c in ("0", "1", "2")))
        if m["collapsed_frac"] > 0.95:
            print(f"  ⚠ 模型预测塌缩到类 {m['collapsed_to_class']}（占比 {m['collapsed_frac']*100:.1f}%）")

    print("\n========== Timing ==========")
    print(json.dumps(res.timing, indent=2))

    # 写 parquet
    df_out = pd.DataFrame({
        "sym": res.meta_sym,
        "date": res.meta_date,
        "session": res.meta_session,
        "t": res.meta_t,
        "midprice_t": res.midprice_t,
    })
    horizons = [int(lc.split("_")[-1]) for lc in res.label_cols]
    for ki, lc in enumerate(res.label_cols):
        df_out[f"true_{lc}"] = res.true_labels[:, ki]
        df_out[f"pred_{lc}"] = res.predictions[:, ki]
    for ki, h in enumerate(horizons):
        df_out[f"midprice_t{h}"] = res.midprice_tn[:, ki]
    # 排序回 (sym, date, session, t) 让下游使用更直观
    df_out.sort_values(["sym", "date", "session", "t"], inplace=True)
    df_out.reset_index(drop=True, inplace=True)
    df_out.to_parquet(args.out_parquet, index=False)
    print(f"\n[INFO] 写出 {args.out_parquet}: rows={len(df_out)}, cols={len(df_out.columns)}")

    # 把摘要也以 dict 形式回传
    summary = {
        "submit_dir": args.submit_dir,
        "session": {"sym": args.sym, "date": args.date, "session": args.session},
        "n_eval_points": int(res.num_points),
        "config": {
            "batch": ev.batch_size,
            "n_features": len(res.feature_cols),
            "label_cols": res.label_cols,
        },
        "per_head_metrics": metrics_per_head,
        "timing": res.timing,
        "out_parquet": args.out_parquet,
    }
    print("\n========== JSON Summary ==========")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
