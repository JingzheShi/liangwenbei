"""
关键审计：在本地全 240-session test 集上跑 mmpc_demo，对比平台真实成绩。

平台 iter_000（任务 2340）：
  label_5  cum_pnl=-9.079  acc=0.306
  label_10 cum_pnl=-13.319 acc=0.336
  label_20 cum_pnl=-6.649  acc=0.314
  label_40 cum_pnl=-16.065 acc=0.339
  label_60 cum_pnl=-23.132 acc=0.333

如果本地全集结果与平台**量级一致 + 全负 + acc ≈ 0.3**，说明:
  ✓ LocalEvaluator + PnL evaluator + 数据 pipeline 正确
  ✓ mmpc_demo 本身是 zero-alpha 模型

如果本地全集结果**显著不同**（比如全 ~0 或正分），说明:
  ⚠️ pipeline 有 bug，T1/T2 的 +6.85/+9.36 都不可信
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "examples"))  # for mmpc_demo import

from src.data.split import get_split  # noqa: E402
from src.eval.dataset_eval import evaluate_on_dataset  # noqa: E402

# Import mmpc_demo Predictor as-is (no modification)
from mmpc_demo.Predictor import Predictor  # noqa: E402


def main():
    print("[audit] loading mmpc_demo Predictor (official, unchanged) ...")
    predictor = Predictor()

    # Use the SAME features as mmpc_demo's config.json
    cfg_path = os.path.join(ROOT, "examples/mmpc_demo/config.json")
    cfg = json.load(open(cfg_path))
    feature_cols = cfg["feature"]
    print(f"[audit] mmpc_demo config: {len(feature_cols)} features, batch={cfg['batch']}")

    splits = get_split(mode="local_debug")
    test_sessions = splits["test"]
    print(f"[audit] {len(test_sessions)} test sessions (date 96-119, 5 sym, am+pm)")

    t0 = time.time()
    result = evaluate_on_dataset(
        predict_fn=predictor.predict,
        dataset_dir=os.path.join(ROOT, "data"),
        sessions=test_sessions,
        feature_cols=feature_cols,
        window=100,
        batch_size=cfg.get("batch", 1024),
        verbose=False,
    )
    elapsed = time.time() - t0

    print(f"\n=== mmpc_demo on FULL local test set ({len(test_sessions)} sessions, {elapsed:.1f}s) ===\n")
    horizons = ["label_5", "label_10", "label_20", "label_40", "label_60"]
    print(f"{'horizon':<10s}  {'cum_pnl':>12s}  {'single_pnl':>12s}  {'n_active':>9s}  {'n_total':>9s}  {'acc':>6s}  {'pred_dist':<24s}")
    for h in horizons:
        m = result[h]
        d = m["pred_distribution"]
        print(
            f"{h:<10s}  {m['cum_pnl']:>+12.4f}  {m['single_pnl']:>+12.6f}  "
            f"{m['n_predictions_active']:>9d}  {m['n_total']:>9d}  {m['accuracy']:>6.4f}  "
            f"0:{d.get(0,0):>6d} 1:{d.get(1,0):>7d} 2:{d.get(2,0):>6d}"
        )
    print(f"\nbest_score: {result['best_score']:+.4f} @ {result['best_horizon']}")

    # Compare with platform
    platform = {
        "label_5": (-9.079, -0.000200, 0.306),
        "label_10": (-13.319, -0.000199, 0.336),
        "label_20": (-6.649, -0.000134, 0.314),
        "label_40": (-16.065, -0.000212, 0.339),
        "label_60": (-23.132, -0.000184, 0.333),
    }
    print("\n=== Compare to platform (task 2340) ===")
    print(f"{'horizon':<10s}  {'local_cum':>12s}  {'plat_cum':>12s}  {'ratio':>7s}  {'local_acc':>10s}  {'plat_acc':>9s}")
    for h in horizons:
        local = result[h]["cum_pnl"]
        local_acc = result[h]["accuracy"]
        p_cum, p_single, p_acc = platform[h]
        ratio = local / p_cum if p_cum != 0 else float("nan")
        print(f"{h:<10s}  {local:>+12.4f}  {p_cum:>+12.4f}  {ratio:>+7.3f}  {local_acc:>10.4f}  {p_acc:>9.4f}")

    # Save full result
    out_path = os.path.join(HERE, "audit_mmpc_demo_full_test_result.json")
    # numpy types -> python
    def cleanup(o):
        if isinstance(o, dict):
            return {k: cleanup(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [cleanup(v) for v in o]
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        return o
    with open(out_path, "w") as f:
        json.dump(cleanup(result), f, indent=2, ensure_ascii=False)
    print(f"\n[audit] saved full result to {out_path}")


if __name__ == "__main__":
    main()
