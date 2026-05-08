# R_cross_sym — Cross-Sym Feature Engineering — **EARLY ABORT (Infeasible)**

**Date**: 2026-05-08
**Worker**: opus xhigh (claude-opus-4-7)
**Status**: ❌ **ABORTED — fundamentally incompatible with platform evaluation protocol**
**GPU spent**: 0 (decision made from contract analysis, before any training run)

---

## TL;DR

The proposed cross-sym features (mean / std / rank / z-score across the *other* 4 syms at the same `(date, t)`) **cannot be computed at inference time** under the platform's evaluation contract. Both the naive design ("compute online from batch") and the PM-proposed fallback ("train-set per-sym constants") fail one of the three §1 hard constraints in `CRITICAL_CONSTRAINTS.md`. Training a model that **uses** these features but **doesn't have them at test time** is strictly worse than baseline (model over-relies on a then-missing signal). Best to early-abort and free the 90 min budget for a different trick.

I recommend instead exploring **train-time-only cross-sym features as a teacher signal for distillation** (sketched in §6) — this is the architecturally correct way to capture cross-sym information without violating the inference contract.

---

## 1. Goal of the Experiment (per dispatch)

PM asked: build ~24-30 cross-sym features (mean / std / rank / z-score across the other 4 syms at the same `(date, t)`) for 6 raw fields (`midprice1`, `spread1`, `imbalance`, `totalbsize`, `lb_intst`, `mb_intst`), train T75 LGB L2 baseline + variant on 3 seeds, compare LOSO. Top public Optiver team (HYD-1st) reported using cross-sym features → potentially +1~3 LOSO if applicable.

PM flagged a **stateless concern** in the dispatch and asked for it to be the gate.

---

## 2. The Stateless Gate — Verified

### 2.1 What the platform sends to `predict()`

From `examples/example_official/Predictor.py` and `experiments/T123_iter017_v1_pkg/{Predictor.py, config.json}`:

- `predict(self, x: List[pd.DataFrame]) → List[List[int]]` is the platform contract.
- Each `pd.DataFrame` is a **single-sym, 100-tick** historical LOB window for ONE `(sym, date, t)` test point.
- The DataFrame columns sent are exactly `config.json["feature"]` (154 raw cols). Verified:

  ```
  # feature cols sent to Predictor: 154
    - includes 'sym'?  False
    - includes 'date'? False
    - includes 'time'? False
  ```

- The 154 cols are: `open/high/low/close, volume_delta, amount_delta, bid1..bid10, bsize1..bsize10, ask1..ask10, asize1..asize10, avgbid, avgask, totalbsize, totalasize, {l,m,c}{b,a}_{intst,ind,acc}, midprice1..10, spread1..10, bid_diff1..10, ask_diff1..10, bid_mean, ask_mean, bsize_mean, asize_mean, cumspread, imbalance, bid_rate1..10, ask_rate1..10, bsize_rate1..10, asize_rate1..10`. **No `sym`, no `date`, no `time`**.

### 2.2 Three platform constraints relevant to cross-sym

From `CRITICAL_CONSTRAINTS.md` §1:

1. **`date` is zeroed** at evaluation — no way to identify which calendar day a window came from.
2. **Test point order is shuffled** — `predict(x)` calls are independent; no cross-call state allowed; same-`(date, t)` rows from different syms are not guaranteed to land in the same batch.
3. **`sym` may be unseen** (e.g., `sym=99` or sym=0..4 mapped to a stock not in the train set) — cannot use `sym` as an indexer.

### 2.3 Why cross-sym is infeasible — three independent failures

For a test row at `(sym=k, date=d, t=tt)`, the cross-sym features need values of (e.g.) `midprice1` at `(sym=k', date=d, t=tt)` for `k' ≠ k`. To compute these online inside `predict()`:

| Required information | Available at inference? |
|---|---|
| Identify the test row's `date d` | ❌ No — `date` is zeroed |
| Identify the test row's `time t` | ❌ No — `time` not in `config.feature` |
| Identify the test row's `sym k` | ❌ No — `sym` not in `config.feature`, and even if it were, mapping is untrusted |
| Receive other syms' rows for `(d, tt)` in the same batch | ❌ No guarantee — shuffled order; batch composition is platform-controlled |

**Each row independently kills the design.** Even if any single one of the four were resolvable, the others would still block it.

### 2.4 The PM-proposed fallback ("train-set globals, constants per sym") also fails

> "退化为 train-set globals (constants per sym)"

To use a per-sym lookup, we need to know which sym we're predicting on. We can't:

- `sym` column is not in `config.feature` (not sent to Predictor).
- Even if it were, constraint §1.3 says `sym` **may map to an unseen stock** — `sym=0` at eval is not guaranteed to be the same stock as `sym=0` in train, and `sym=99` might appear (older audit trail in `CRITICAL_CONSTRAINTS.md`).

So "per-sym constants" → either we use `sym` and risk `IndexError` on an unseen sym, or we use a **single global constant** which means every cross-sym feature column becomes a literal constant at inference (no information). A model trained with N "real" cross-sym features but seeing N constants at inference will produce systematically wrong predictions on the affected leaf splits.

### 2.5 The "model trained with feature, served with constant" anti-pattern

Even if we just trained the LGB model with cross-sym features as part of the 359+24 feature vector and at inference set those 24 dims to their training-set means:

- LGB splits would still partition feature space along these now-constant dims.
- The model would route all test rows down the "training-mean" branch, which corresponds to whatever subpopulation had near-mean cross-sym values in training — usually a biased subset.
- This systematically biases predictions, almost always **worse than not having the feature at all**. (This is well-known: training with a feature that's missing at inference is among the worst things you can do.)

So both naive design and fallback are dead.

---

## 3. Training-Time Cross-Sym Alignment — Verified Possible

For completeness, I verified the train-time prerequisite:

```
rows per sym: {0: 2001, 1: 2001, 2: 2001, 3: 2001, 4: 2001}  # one date, one session
unique (date, time) groups: 2001
groups with 5 syms: 2001                                      # 100% aligned
```

And from the schemeP train cache (`experiments/T68_stage5_features/cache/schemeP_train.npz`):

```
sym unique: [0 1 2 3 4]
date with 5 syms: 80 / 80                                      # all train dates have all 5 syms
```

So at train time, cross-sym alignment is trivial: `groupby(date, t)` gives 5-row groups for every `t`, every date. The PM's design works **at train time**.

The problem is purely the inference-time gap, which the platform's contract makes unbridgeable in a sym-agnostic, stateless, shuffled-batch setting.

---

## 4. Empirical Validation Skipped — Why

I considered running a 1-seed minimal experiment to empirically demonstrate the "constants-at-test" failure. I declined because:

1. The math is unambiguous: a model with cross-sym features but constants at inference is at best equivalent to a baseline that drops those features, and almost always strictly worse. No new information would come from running it.
2. The 90-min budget is better returned to the PM for a feasible trick.
3. Burning ~30 min of remote GPU on `r2` for a confirmed-infeasible design is wasteful when the contract analysis is conclusive.

`results.json` records `metrics.baseline = null, metrics.trick = null` to make the abort explicit.

---

## 5. Cross-Reference: Existing Audits

`audits/AUDIT-LEAKAGE/REPORT.md` line 79, 113, 144-146 explicitly establishes that the project's stance is **no per-sym normalization, no cross-sym leakage**, with the model being sym-agnostic. This was a deliberate compliance choice. The cross-sym design proposed in this dispatch is a step in the *opposite* direction and conflicts with that established posture.

`audits/PRACTICE-FUNDAMENTAL/REPORT.md` line 244 already noted the same cross-sym-generalization tension and recommended *better sym-agnostic features (e.g., features normalized relative to local market state)* — i.e., **within-window** cross-sectional features (across LOB levels), not **cross-sym** features. That's the right architectural direction.

---

## 6. Recommended Alternative: Cross-Sym Distillation (future work)

If the PM still wants to capture cross-sym signal, the architecturally clean way is **distillation**:

1. **Train a TEACHER** (`T_xs`) with the full 359 + 24 cross-sym features, on the train cache where alignment is trivial. This teacher cannot ship — it needs cross-sym at inference.
2. **Generate teacher predictions** `ŷ_teacher` for every train row.
3. **Train a STUDENT** (`S`) with **only** the 359 sym-agnostic features, using a soft target like `α · y_orig + (1-α) · ŷ_teacher`, or a multi-task setup.
4. **Deploy the STUDENT** — fully compliant with the inference contract, but its 359-d features now encode whatever sym-agnostic shadow of the cross-sym signal is recoverable.

Effort: ~2x baseline training time. Diminishing-returns risk: if the teacher's lift is not large to begin with (HYD's reported 1-3 LOSO is moderate), the student's residual lift may be `<0.5 LOSO`.

This is the correct *future* experiment. It is **not** what the dispatch asked for, so I'm not running it without explicit redirect.

---

## 7. Time Returned to PM

Dispatched budget: 90 min remote GPU + feature build.
Actual: ~25 min for contract analysis + this report. **Returned: ~65 min.**

PM can redirect to (suggestions, lowest-effort first):
- An **iter_017 ablation** that wasn't run (e.g., HYD ablation, monotone constraint sweep, longer-window roll features).
- Threshold revisit on iter_017 with current 4D DE on val (vs the unchanged iter_015 thresholds currently shipped).
- Distillation track described in §6 (higher effort, 2-3 hour budget).

---

## Appendix: Files in this experiment

- `REPORT.md` — this document
- `results.json` — abort record (machine-readable)
