"""
彻底审计本地 pipeline，定位本地 +6 / 平台 -7 不一致的根源。

依次检查：
A. train-val-test split 定义是否正确
B. PnL 公式逐手计算 1 个样本
C. 评测器的 windowing 是否与官方 example_official/main.py 一致
D. mmpc_demo 在 1 个 session 上的预测分布 + 单样本 PnL
E. 全 240 session 的 per-session PnL 分布——是不是被少数 session 拖正？
F. 比较 raw 输入 vs 用 train 集 z-score normalize 输入对预测的影响
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "examples"))

from src.data.split import get_split, get_file_path  # noqa: E402
from src.eval.dataset_eval import evaluate_on_dataset, evaluate_on_session  # noqa: E402
from src.eval.pnl import compute_pnl  # noqa: E402
from mmpc_demo.Predictor import Predictor  # noqa: E402


def section(s):
    print(f"\n{'='*78}\n{s}\n{'='*78}")


# ============================================================
# A. Split 检查
# ============================================================
section("A. Split 定义审计")

splits = get_split(mode="local_debug")
for k in ["train", "val", "test"]:
    sess = splits[k]
    print(f"  {k}: {len(sess)} sessions")
    syms = sorted({s for s, d, ss in sess})
    dates = sorted({d for s, d, ss in sess})
    sessions = sorted({ss for s, d, ss in sess})
    print(f"    syms: {syms}")
    print(f"    dates: {dates[0]}..{dates[-1]} ({len(dates)} dates)")
    print(f"    am/pm: {sessions}")

# 重叠检查
train_set = set(splits["train"])
val_set = set(splits["val"])
test_set = set(splits["test"])
print(f"\n  train ∩ val: {len(train_set & val_set)} (must be 0)")
print(f"  train ∩ test: {len(train_set & test_set)} (must be 0)")
print(f"  val ∩ test: {len(val_set & test_set)} (must be 0)")


# ============================================================
# B. 单样本 PnL 手算 + 比对 compute_pnl
# ============================================================
section("B. PnL 公式逐手验证（单样本）")

sym, date, sess = splits["test"][0]  # sym=0 date=96 am or similar
session_path = get_file_path(sym, date, sess, data_dir=os.path.join(ROOT, "data"))
df = pd.read_parquet(session_path)

t = 100
mp_t = float(df["midprice"].iloc[t])
mp_tn_arr = np.array([float(df["midprice"].iloc[t + h]) for h in [5, 10, 20, 40, 60]])
labels = np.array([int(df[f"label_{h}"].iloc[t]) for h in [5, 10, 20, 40, 60]])
print(f"\n  test session: sym={sym} date={date} {sess}, t={t}")
print(f"  midprice_t = {mp_t:.6f}")
print(f"  midprice_tn (5/10/20/40/60) = {mp_tn_arr.tolist()}")
print(f"  labels = {labels.tolist()} (0=down, 1=flat, 2=up)")

fee = 0.0001

# 手算：假设 pred=labels（完美预测）
pred = labels.copy()
print(f"\n  Hand-computed pnl_single (pred = labels, perfect):")
for i, h in enumerate([5, 10, 20, 40, 60]):
    side = pred[i] - 1
    diff = mp_tn_arr[i] - mp_t
    fee_term = fee * abs(side) * abs((mp_tn_arr[i] + 1) + (mp_t + 1))
    pnl_hand = (side * diff - fee_term) / (mp_t + 1)
    print(f"    label_{h}: side={side} diff={diff:+.8f} fee_term={fee_term:.8f} pnl={pnl_hand:+.8f}")

# 用 compute_pnl 算同样的事
pred_5 = pred.reshape(1, 5)
label_5 = labels.reshape(1, 5)
mp_t_arr = np.array([mp_t])
mp_tn_2d = mp_tn_arr.reshape(1, 5)
result = compute_pnl(pred_5, label_5, mp_t_arr, mp_tn_2d)
print(f"\n  compute_pnl on same single point (pred=labels):")
for h in ["label_5", "label_10", "label_20", "label_40", "label_60"]:
    print(f"    {h}: cum_pnl={result[h]['cum_pnl']:+.8f}")

# 假设 pred=2 全做多
pred_long = np.array([2, 2, 2, 2, 2]).reshape(1, 5)
result_long = compute_pnl(pred_long, label_5, mp_t_arr, mp_tn_2d)
print(f"\n  compute_pnl (pred = all 2 = always long):")
for h in ["label_5", "label_10", "label_20", "label_40", "label_60"]:
    print(f"    {h}: cum_pnl={result_long[h]['cum_pnl']:+.8f}")


# ============================================================
# C. windowing 与官方 example_official/main.py 对照
# ============================================================
section("C. Windowing 与官方 example_official/main.py 对照")

# 官方代码：
#   for index in range(0, len(np_data) - slide_window):
#       data = np_data.iloc[index : index + slide_window,:]
#       r = predictor.predict([data])
# 这里 data 是 [index, index+100) = iloc[index:index+100] 共 100 行

# 我们的代码（dataset_eval.py）：
#   valid_t in [99, 1940]
#   window = feat_df.iloc[t-99 : t+1] 共 100 行
# 对应官方的 index = t - 99

# 即官方 index=0 ↔ 我们 t=99，二者切的是同一片 100 行
# 不同：官方主循环到 index = 1900（不到 1901），即 t = 1999
#       我们 t 到 1940（保留 60 行未来给 label_60）

# 对照：在共有的范围内（t in [99, 1940] 即 index in [0, 1841]），切片是否一致
n = 2001
window = 100
print(f"\n  session length n = {n}, window = 100, max horizon = 60")
print(f"  我们的 valid_t 范围: [99, 1940] = {1940-99+1} = 1842 个评测点")
print(f"  官方 main.py 的 index 范围: [0, {n-window-1}] = {n-window} = 1901 个，未限定 horizon\n")
print("  在官方的 index=0 ↔ 我们 t=99 上比对窗口：")

# 用官方的写法切：
np_data = df.iloc[:, [df.columns.get_loc(c) for c in ["bid1", "ask1", "midprice"]]]
official_w0 = df.iloc[0:100][["bid1", "ask1", "midprice"]]
ours_w0 = df.iloc[99-99 : 99+1][["bid1", "ask1", "midprice"]]
print(f"    official iloc[0:100]   first row: {official_w0.iloc[0].tolist()}")
print(f"    ours     iloc[0:100]   first row: {ours_w0.iloc[0].tolist()}")
print(f"    official iloc[0:100]   last  row: {official_w0.iloc[-1].tolist()}")
print(f"    ours     iloc[0:100]   last  row: {ours_w0.iloc[-1].tolist()}")
print(f"    diff: {np.abs(official_w0.values - ours_w0.values).max():.2e}")

# 关键问题: 在 t=99 的预测，期待的标签是哪一行？
# 我们当前: label = df['label_5'].iloc[t=99]
#   ↔ midprice[t=99+5=104] vs midprice[t=99]  即"未来 5 ticks 的方向"
# 官方 main.py: 没用标签（仅 demo），所以无法直接验证
print(f"\n  关键: 我们假设 label_5 at row 99 = direction(midprice[104] vs midprice[99])")
print(f"        即模型在 t=99 看 [0:100] 数据，预测 [99 → 104] 的方向。")
print(f"        如果平台实际是 [100 → 105]（off-by-one），则我们一直用错 label！")


# ============================================================
# D. 单 session 上 mmpc_demo 预测分布
# ============================================================
section("D. mmpc_demo 在 1 个 test session 上预测分布")

predictor = Predictor()
cfg = json.load(open(os.path.join(ROOT, "examples/mmpc_demo/config.json")))
feat_cols = cfg["feature"]

session_df = df.copy()
print(f"  session: sym={sym} date={date} {sess}, n={len(session_df)}, feat_cols={len(feat_cols)}")
result = evaluate_on_session(
    predict_fn=predictor.predict,
    session_df=session_df,
    feature_cols=feat_cols,
    window=100,
    batch_size=1024,
)
for h in ["label_5", "label_10", "label_20", "label_40", "label_60"]:
    m = result[h]
    d = m["pred_distribution"]
    n_total = m["n_total"]
    print(
        f"    {h}: cum={m['cum_pnl']:+8.4f} acc={m['accuracy']:.4f} "
        f"pred 0/1/2 = {d.get(0,0)}/{d.get(1,0)}/{d.get(2,0)} ({100*d.get(0,0)/n_total:.0f}%/{100*d.get(1,0)/n_total:.0f}%/{100*d.get(2,0)/n_total:.0f}%)"
    )


# ============================================================
# E. 全 240 sessions per-session PnL 分布
# ============================================================
section("E. 240 sessions per-session 分布——是不是少数 session 拖正？")

t0 = time.time()
per_session_pnl = {h: [] for h in ["label_5", "label_10", "label_20", "label_40", "label_60"]}
per_session_acc = {h: [] for h in ["label_5", "label_10", "label_20", "label_40", "label_60"]}
per_session_predflat = {h: [] for h in ["label_5", "label_10", "label_20", "label_40", "label_60"]}
session_keys = []
for i, (s, d, ss) in enumerate(splits["test"]):
    p = get_file_path(s, d, ss, data_dir=os.path.join(ROOT, "data"))
    session_df = pd.read_parquet(p)
    r = evaluate_on_session(
        predict_fn=predictor.predict,
        session_df=session_df,
        feature_cols=feat_cols,
        window=100,
        batch_size=1024,
    )
    if r is None:
        continue
    session_keys.append((s, d, ss))
    for h in ["label_5", "label_10", "label_20", "label_40", "label_60"]:
        m = r[h]
        per_session_pnl[h].append(m["cum_pnl"])
        per_session_acc[h].append(m["accuracy"])
        n_total = m["n_total"]
        per_session_predflat[h].append(m["pred_distribution"].get(1, 0) / n_total)
    if (i + 1) % 40 == 0:
        print(f"  [{i+1}/240] elapsed {time.time()-t0:.0f}s")

print(f"\n  240 sessions evaluated in {time.time()-t0:.0f}s\n")
print(f"{'horizon':<10s}  {'sum':>9s}  {'mean':>9s}  {'pos':>5s}  {'neg':>5s}  {'min':>9s}  {'max':>9s}  {'mean_acc':>8s}  {'mean_predflat':>12s}")
for h in ["label_5", "label_10", "label_20", "label_40", "label_60"]:
    arr = np.array(per_session_pnl[h])
    accs = np.array(per_session_acc[h])
    pf = np.array(per_session_predflat[h])
    print(
        f"{h:<10s}  {arr.sum():>+9.3f}  {arr.mean():>+9.4f}  "
        f"{(arr>0).sum():>5d}  {(arr<0).sum():>5d}  {arr.min():>+9.3f}  {arr.max():>+9.3f}  "
        f"{accs.mean():>8.4f}  {pf.mean():>12.4f}"
    )

# 极值 session
print("\n  最赚钱的 5 个 session（label_20）:")
arr20 = np.array(per_session_pnl["label_20"])
top5 = np.argsort(arr20)[-5:][::-1]
for i in top5:
    s, d, ss = session_keys[i]
    print(f"    sym={s} date={d} {ss}: cum_pnl={arr20[i]:+.4f} acc={per_session_acc['label_20'][i]:.4f} predflat={per_session_predflat['label_20'][i]:.4f}")
print("\n  最亏损的 5 个 session（label_20）:")
bot5 = np.argsort(arr20)[:5]
for i in bot5:
    s, d, ss = session_keys[i]
    print(f"    sym={s} date={d} {ss}: cum_pnl={arr20[i]:+.4f} acc={per_session_acc['label_20'][i]:.4f} predflat={per_session_predflat['label_20'][i]:.4f}")

# 按 sym 分组
print("\n  按 sym 分组（label_20）:")
import collections
by_sym = collections.defaultdict(list)
for i, (s, d, ss) in enumerate(session_keys):
    by_sym[s].append(per_session_pnl["label_20"][i])
for s in sorted(by_sym):
    arr = np.array(by_sym[s])
    print(f"    sym={s}: n={len(arr)}, sum={arr.sum():+.3f}, mean={arr.mean():+.4f}, pos/neg={(arr>0).sum()}/{(arr<0).sum()}")

# 保存数据
out = {
    "per_session_pnl": {h: per_session_pnl[h] for h in per_session_pnl},
    "per_session_acc": {h: per_session_acc[h] for h in per_session_acc},
    "per_session_predflat": {h: per_session_predflat[h] for h in per_session_predflat},
    "session_keys": [list(k) for k in session_keys],
}
with open(os.path.join(HERE, "audit_pipeline_per_session.json"), "w") as f:
    json.dump(out, f)
print(f"\n  saved per-session breakdown to {os.path.join(HERE, 'audit_pipeline_per_session.json')}")
