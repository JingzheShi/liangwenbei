"""compare_local_vs_eval.py — 用 LocalEvaluator 跑提交包，跟本地训练时的指标做对照。

用途：模型本地训练算出来 cum_pnl=X，但 zip 打包 + LocalEvaluator 跑可能因为
shuffle / batch 边界 / dtype 转换等差异，得到不同的 PnL。这个脚本把两边一起跑出来对比，
**任何差异 > 阈值** 都应当排查。

用法：
    python submission/scripts/compare_local_vs_eval.py \\
        --src submission/iter_001_xxx \\
        --sym 0 --date 119 --session am

输出：
- 每个 horizon 的 cum_pnl / accuracy / pred_dist 两边比对
- 整体 pred 一致率（pred 一样的样本占比）
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUBMISSION_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(SUBMISSION_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from src.submit.local_evaluator import LocalEvaluator, load_test_sessions
from src.eval.pnl import compute_pnl, sanitize_for_json


def run_via_local_eval(
    src_dir: str,
    data_dir: str,
    sym: int, date: int, session: str,
    shuffle: bool = True,
) -> dict:
    sessions = load_test_sessions([(sym, date, session)], data_dir=data_dir)
    ev = LocalEvaluator(
        submit_dir=src_dir,
        test_data=sessions,
        shuffle=shuffle,
        anonymize=True,
        progress=False,
    )
    res = ev.run()
    ev.cleanup()

    # 真实 label 中的 -1（NaN）跳过
    valid = (res.true_labels != -1).all(axis=1)
    pnl = compute_pnl(
        res.predictions[valid],
        res.true_labels[valid],
        res.midprice_t[valid],
        res.midprice_tn[valid],
    )
    return {
        "predictions": res.predictions,
        "true_labels": res.true_labels,
        "midprice_t": res.midprice_t,
        "midprice_tn": res.midprice_tn,
        "meta_t": res.meta_t,
        "meta_sym": res.meta_sym,
        "valid_mask": valid,
        "pnl": pnl,
        "timing": res.timing,
        "label_cols": res.label_cols,
    }


def summarize(tag: str, out: dict) -> None:
    print(f"\n--- {tag} ---")
    print(f"  N points: {out['predictions'].shape[0]}")
    print(f"  N valid:  {int(out['valid_mask'].sum())}")
    print(f"  predict_total_s: {out['timing']['predict_total_s']:.2f}")
    print(f"  best_horizon: {out['pnl']['best_horizon']}  best_score: {out['pnl']['best_score']:.6g}")
    for lc in out["label_cols"]:
        m = out["pnl"][lc]
        pdist = m["pred_distribution"]
        print(
            f"  {lc}: cum_pnl={m['cum_pnl']:.6g}  acc={m['accuracy']:.4f}  "
            f"pred_dist=[{pdist[0]},{pdist[1]},{pdist[2]}]"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--sym", type=int, default=0)
    ap.add_argument("--date", type=int, default=119)
    ap.add_argument("--session", default="am", choices=["am", "pm"])
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    data_dir = os.path.abspath(os.path.join(ROOT, args.data_dir))

    print(f"=== compare local vs eval: {src} ===")
    print(f"data: sym={args.sym} date={args.date} session={args.session}")

    # 跑两次：一次 shuffle=True（模拟平台），一次 shuffle=False（与训练 t 顺序一致）
    out_sh = run_via_local_eval(src, data_dir, args.sym, args.date, args.session, shuffle=True)
    out_no = run_via_local_eval(src, data_dir, args.sym, args.date, args.session, shuffle=False)

    summarize("LocalEval (shuffle=True, 模拟平台)", out_sh)
    summarize("LocalEval (shuffle=False, 时序顺序)", out_no)

    # 一致率：用 t 索引对齐，比较 pred 是否相同
    t_sh = out_sh["meta_t"]
    t_no = out_no["meta_t"]
    pred_sh = out_sh["predictions"]
    pred_no = out_no["predictions"]
    order_sh = np.argsort(t_sh)
    order_no = np.argsort(t_no)
    pred_sh_sorted = pred_sh[order_sh]
    pred_no_sorted = pred_no[order_no]

    if pred_sh_sorted.shape == pred_no_sorted.shape:
        same = (pred_sh_sorted == pred_no_sorted).all(axis=1).mean()
        print(f"\n[一致率] shuffle vs no-shuffle, 全 head 一致的样本占比：{same*100:.2f}%")
        if same < 1.0:
            print(
                "[!] shuffle 后 pred 与时序顺序不一致 — 模型可能依赖了"
                " batch 内顺序（hidden state / running stats），需要排查"
            )

    # 把摘要落盘到 src/compare_report.json，方便回看
    report = {
        "src": src,
        "sym": args.sym, "date": args.date, "session": args.session,
        "shuffle_true": {
            "pnl": out_sh["pnl"],
            "timing": out_sh["timing"],
            "n_total": int(out_sh["predictions"].shape[0]),
            "n_valid": int(out_sh["valid_mask"].sum()),
        },
        "shuffle_false": {
            "pnl": out_no["pnl"],
            "timing": out_no["timing"],
            "n_total": int(out_no["predictions"].shape[0]),
            "n_valid": int(out_no["valid_mask"].sum()),
        },
    }
    out_path = os.path.join(src, "compare_report.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(sanitize_for_json(report), fh, indent=2, ensure_ascii=False)
    print(f"\n[ok] report → {out_path}")


if __name__ == "__main__":
    main()
