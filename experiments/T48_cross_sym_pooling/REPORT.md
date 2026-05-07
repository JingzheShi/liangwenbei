# T48 — Cross-Sym Pooling Features (R33 Trick #2) — NO-GO

**Status**: ABORT after Step 1 (inference path verification)
**Decision**: Cannot proceed — fundamental incompatibility with the platform's evaluation contract.
**Time spent**: ~30 min (Step 1 only; Steps 2–4 not executed).
**LOSO sum vs iter_006**: N/A (no training run).

---

## 1. Task summary

Trick 2 in `research_and_history/r33_recall_tricks.md` proposes:
> 每个样本在 `(t_local)` 这个时间锚点附近 ± 1 bar 内，对 sym ∈ {0,1,2,3,4} 中**所有可见 sym** 的同 feature 算 mean / std / rank percentile / range。新增约 16 高相关 feature × 4 stat = 64 d。

This requires, at inference time, the ability to compute per-`t_local` cross-sectional statistics over multiple syms.

The PM dispatch flagged the key risk up-front:
> **Inference path 是否可行**：如果 Predictor 不能访问同时 t 的所有 sym 数据，这个 trick 就不能用。**Step 1 必须先验证！如果不行，立即停止并报告**。

After verification, the answer is **NO** — the trick cannot be implemented in a way that survives the platform's `Predictor` contract. Below is the analysis.

---

## 2. What the platform actually delivers to `Predictor.predict`

Reviewed:
- `submission/iter_006_aug_a_5seed_h60_asymthresh/Predictor.py`
- `submission/iter_006_aug_a_5seed_h60_asymthresh/config.json`
- `examples/example_official/main.py` (official inference loop reference)
- `examples/example_official/Predictor.py`
- `examples/example_official/config.json`
- `CRITICAL_CONSTRAINTS.md`
- `docs/data_schema.md`

### 2.1 Signature and shape
```python
predict(self, batches: List[pd.DataFrame]) -> List[List[int]]
```
Each `batches[i]` is a 100-row × D-col sliding window for **one sample**.

### 2.2 Columns delivered to Predictor
The platform calls predict with `df[config["feature"]]` columns only — confirmed by the official example (`examples/example_official/main.py:31-35`):
```python
np_data = df[(df['sym']==sym) & (df['date']==date)]
np_data = np_data[columns['feature']].copy()
for index in range(0, len(np_data) - slide_window):
    data = np_data.iloc[index : index + slide_window, :].copy()
    r = predictor.predict([data])
```

Our iter_006 `config.json` `feature` list contains **none of**: `sym`, `date`, `time`. So at inference each window is a 100×154 (or 100×226 with derived) feature matrix with **no identity tag**.

### 2.3 Constraints from CRITICAL_CONSTRAINTS.md
- **C1**: `date` is set to 0 — useless as a pooling anchor.
- **C2**: "评测程序会打乱测试点输入顺序，不同次输入数据无先后关系" — windows are shuffled across the entire test set; the Predictor must not rely on order or maintain cross-call state.
- **C3**: `sym` cannot enter the model (and may include training-out-of-distribution sym IDs).

### 2.4 Batching reality
- `examples/example_official/config.json` uses `batch:1` — platform calls `predict([single_window])` per sample.
- `submission/iter_006_aug_a_5seed_h60_asymthresh/config.json` requests `batch:1024`. The platform may comply, but **C2 makes no guarantee about which windows end up in the same batch** — they are sampled from a globally-shuffled test set.

---

## 3. Why cross-sym pooling cannot be made compliant

To compute "mean/std/rank/range across the 5 syms at the same `t_local`" at inference time, the Predictor needs at least one of:

| Requirement | Available? | Why |
|---|---|---|
| Know each window's `sym` ID | ❌ No | Not in feature columns; forbidden by C3 |
| Know each window's `t_local` | ❌ No | `date` is zero (C1); `time` not in feature columns; even if added, `t_local` ≠ `time` (per-episode index, not global timestamp) |
| Group windows in batch by `(t_local)` | ❌ No reliable mechanism | Even with `time`, C2 shuffle gives no guarantee that windows from the same `(date, t)` end up in the same `predict` call |
| Maintain a cache of "latest cross-sym snapshot" | ❌ Forbidden by C2 | `Predictor.predict` must not rely on cross-call state |

The trick's core assumption — *"all 5 syms at the same anchor are simultaneously available at the moment a sample is featurized"* — only holds at training time. At inference, the platform delivers windows one-at-a-time (or in shuffled batches with no group guarantees), and the Predictor has no way to resolve the cross-sectional ensemble.

---

## 4. Workarounds considered and rejected

### 4.1 Within-batch pooling using `time` as group key
**Idea**: Add `time` to feature columns; in `predict`, group windows by their last-row `time`; compute pool stats within each group.

**Rejected because**:
1. **Group size is unknown and unstable.** With C2's full shuffle, a single batch may contain 0, 1, ..., 5 windows for any given `t_local`. The example config uses `batch:1`, in which case every group is size 1 → mean = identity, std/range = 0, rank = 0.5 → constant degenerate features.
2. **Severe train/inference distribution shift.** At training every `t_local` has 5 syms (rich pool stats); at inference the pool size varies and is often 1. The model would learn to use "real" pool signals it never sees at test time. This is exactly the failure mode T40 (probability calibration) and other recent ablations have flagged.
3. **Even within batch, `time` ≠ `t_local`.** `time` is a wall-clock timestamp; two episodes (different dates / different syms with different session boundaries) can share the same `time` value. Grouping by `time` would conflate cross-day windows.
4. **Identity ambiguity remains.** Even if we group "windows whose last-row time == X", we still don't know which syms they are — but the trick's `rank_percentile` requires ranking each window relative to the others in the group, which is well-defined; however the model would learn to depend on group sizes that don't appear at inference.

### 4.2 Static lookup: pre-compute training-time cross-sym stats per intraday `time`, embed as fixed features
**Idea**: At training, build a table `time → mean(feature)` averaged over (sym × date); at inference, look up the training-time average for the window's last `time`.

**Rejected because**:
- This is not "cross-sym pooling" — it's a **time-of-day feature**. It does not capture *current* market-wide state at the test time, which is the entire purpose of the trick (Volkova's "averages per date_id and time_id" specifically used the *current test time* averages).
- Equivalent to a fixed time-of-day prior; that signal is already implicitly captured by intraday features (`*_intst`, `*_acc`, etc.) and any explicit time-of-day feature we'd add (T22 already explored intraday features). Expected gain ≈ 0.
- Doesn't replicate the Volkova / hyd Optiver result, which depends on *cross-sectional* signal at the test moment.

### 4.3 Sym-pair correlations baked into the model (no inference-time pooling)
**Idea**: At training, instead of building cross-sym pool *features*, train the model on residualized targets where the "market component" is removed.

**Rejected because**:
- This is a **different trick** (target adjustment, not feature pooling) and is essentially the standard market-beta residualization. It's not what R33 #2 specifies, and it has its own pitfalls (residualization at inference requires knowing the market component, which we again can't compute).

---

## 5. Why the cited references don't transfer

- **Volkova 8th JS2024**: Jane Street Kaggle exposes `date_id` and `time_id` as canonical features in the API; multi-symbol rows for the same `(date_id, time_id)` are present in each timestep's API window by construction. Their "averages per date_id and time_id" is computable directly from the API row group.
- **hyd 1st Optiver 2023**: Optiver Trading at the Close also provides `stock_id`, `date_id`, `seconds_in_bucket` per row; cross-stock pooling at each `(date_id, seconds_in_bucket)` is again directly computable from each API frame.

In both Kaggle tasks the **API supplies the cross-sectional grouping for free**. Our platform does the opposite: it shuffles windows globally, strips identifying tags from the DataFrame, and forbids per-call state. The structural symmetry the references rely on does not exist here.

---

## 6. Outcome and recommendation

**Outcome**: Cross-sym pooling cannot be implemented in a way that simultaneously
- respects C1 (no `date`),
- respects C2 (no order/state assumptions, robust to shuffle and arbitrary batching),
- respects C3 (sym-agnostic),
- and provides at inference time the same statistics it provides at training time.

Any compliant implementation degrades to either a constant feature or a time-of-day prior, neither of which matches the trick's intended signal.

**Recommendation**:
- Drop R33 #2 from the candidate list (or down-rank it from "P0" to "infeasible under current platform contract").
- **R33 #3 (KNN-Target Retrieval)** remains a viable cross-sample-information trick — it does not require cross-sectional grouping at inference; each test window is independently looked up in a static FAISS index. Suggest re-prioritizing T48 budget toward T-? KNN if not already running.
- **R33 #5 (Triplet Imbalance)** and **R33 #6 (Hull MA)** are within-window features with no cross-sectional dependence — both fully compliant.
- For the "market state signal" gap that #2 was meant to fill, consider:
  - Per-window "market regime" proxies derivable from the LOB itself (book-imbalance Z-scores, recent volatility) — these are within-window and sym-agnostic.
  - A static intraday `time` feature (cheap, single feature) — won't replicate Volkova's gain but will recover the intraday seasonality fraction at zero risk.

---

## 7. Files produced

- `experiments/T48_cross_sym_pooling/REPORT.md` — this file
- `experiments/T48_cross_sym_pooling/results.json` — machine-readable summary

No OOF parquet, no model artifacts, no thresholds were produced (Step 2+ skipped).
