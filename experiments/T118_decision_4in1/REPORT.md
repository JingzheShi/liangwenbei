# T118: Decision-Rule 4-in-1 Comparison vs DE Asym Threshold

**Goal.** T112 + T114 proposed four post-hoc decision rules to replace the
DE-tuned 4D asymmetric threshold (`thr_up=2.883e-4`, `thr_dn=2.186e-4`) used in
iter_016 v3. Each rule is evaluated on the 442k local test set with the existing
4-way 5-seed ensemble (T87 NN + 0.7×T99 CB Huber + 1.0×T99 LGB Huber + 1.0×T95 GRU).

**Headline.** **No rule clearly beats the DE asym baseline.** The promising-looking
BAT rule (+47.13 LOSO, +0.77 per_sym_min) turned out to be a **sym-sorted-data
artifact** — under production-realistic shuffled order it drops to **+43.4 LOSO
(−1.3 vs baseline)**, so it is **disqualified**. The other rules either match
baseline within noise or lose 0.3–1.5 LOSO. Recommendation: **stick with DE asym
thresholds for iter_017**.

## Setup

* Predictions: existing 5-seed parquets per model
  (`pred_T87_seed*_main.parquet`, `pred_T99_cb_huber_a0.001_seed*.parquet`,
  `pred_T99_huber_a0.001_seed*.parquet`, `pred_T95_gru_w100_C_seed*.parquet`)
  averaged within model, then combined with weights (1.0, 0.7, 1.0, 1.0)
  matching iter_016 v3.
* Spread feature for NSF: `ask1 − bid1` from
  `experiments/T68_stage5_features/cache/schemeP_test.npz` (mid-relative units,
  same scale as `pred_dmid_norm`). mean=9.7e-4, std=8.5e-4.
* PnL formula matches `T109/full_test_verify.py`. Verified: baseline reproduces
  `+44.7178` exactly.

## Sorted-data results (matches T68 cache row order)

| rule                             | LOSO       | per_sym_min | active_rate | per_sym                                          |
| -------------------------------- | ---------- | ----------- | ----------- | ------------------------------------------------ |
| baseline DE asym                 | **+44.72** | **+4.46**   | 0.445       | +4.69 +4.46 +4.54 +12.99 +18.04                  |
| NSF best (fe=2.50e-4, g=0)       | +44.39     | +4.05       | 0.438       | +4.05 +4.42 +4.39 +12.74 +18.79                  |
| NSF asym (DE base + 0.05·spread) | +42.93     | +4.64       | 0.371       | +4.65 +4.64 +4.69 +11.85 +17.08                  |
| TBT (top-K, K=44.5%·N)           | +44.32     | +3.99       | 0.445       | +5.14 +3.99 +4.31 +12.82 +18.06                  |
| STK PnL (cutoff=0.42, 10 bckt)   | +44.28     | +4.40       | 0.400       | +4.84 +4.40 +4.34 +12.46 +18.24                  |
| BAT (batch=1024)                 | +47.13 ⚠️   | +5.23 ⚠️     | 0.445       | +6.04 +6.56 +5.23 +12.45 +16.84                  |

⚠️ BAT result is misleading — see next section.

## BAT shuffle-invariance test (production scenario)

The local test cache is **sorted by sym** (only 4 transitions in 442 080 rows).
BAT batches of 1024 therefore overlap one symbol almost perfectly, so BAT acts
effectively as **per-sym thresholding** — which both (a) cheats the sym-sorting
artifact and (b) violates the sym-agnostic constraint in spirit. The platform's
test set is shuffled, so the relevant evaluation is under permuted rows:

| rule                                  | LOSO   | per_sym_min | active_rate |
| ------------------------------------- | ------ | ----------- | ----------- |
| baseline DE (shuffle)                 | +44.72 | +4.46       | 0.445       |
| NSF (fe=2.88e-4, g=0) shuffle         | +44.22 | +4.80       | 0.375       |
| TBT shuffle                           | +44.32 | +3.99       | 0.445       |
| **BAT b=1024 shuffle**                | **+43.44** | **+3.65** | 0.445   |
| BAT b=128 shuffle                     | +42.86 | +3.45       | 0.453       |
| BAT b=4096 shuffle                    | +43.50 | +3.64       | 0.445       |
| BAT b=100k shuffle                    | +43.64 | +3.68       | 0.445       |

BAT is consistent across permutation seeds (5-seed mean LOSO +43.34 ± 0.13),
confirming the +47.13 was not noise but **structural sym-sorted leakage**.
BAT is **disqualified** for production.

NSF / TBT / STK are order-invariant by construction (per-row rule, global top-K,
fold-OOF — verified numerically above).

## Per-rule analysis

### Rule 1 — NSF (Net Spread-aware EV Floor)
Symmetric form `|pred| > FEE_EFF + γ·spread_t` and asymmetric form
`pred > thr_up_base + γ·spread_t` / `pred < −(thr_dn_base + γ·spread_t)` were
both tested. Best symmetric (fe=2.5e-4, g=0) gives **−0.33 LOSO, −0.41 pmin**.
Best asymmetric (g=0.05 over DE base) gives **−1.79 LOSO, +0.18 pmin**. Adding
spread to the EV floor monotonically degrades LOSO — the model already learns
spread implicitly, so γ>0 over-filters profitable wide-spread trades. NSF γ=0
is just symmetric thresholding (no scale-invariance benefit).

### Rule 2 — TBT (Trade-Budget Top-K)
Global top-K by |pred| with K = baseline-active_rate·N. **−0.40 LOSO, −0.46
pmin**. Same active rate as DE but different selection: TBT can fire long when
DE would fire short and vice-versa near boundaries (since |pred| tied positive
side wins the tie). True scale-invariant rule, but ranking ≠ asym-thresholded
EV gate, so loses the asym structure DE encodes (LOSO of upside vs downside in
this dataset is not symmetric).

### Rule 3 — STK (5-fold OOF Win-Rate Bucketing)
Original sign-match definition gave only ~0.54 hit rate in the top bucket
(many tied mp_t == mp_th rows depress sign-correctness). Refined PnL-based
(win = pnl > 0) version: hit rates monotonically increase with |pred| bucket
(from 0.36 in lowest to 0.50+ in top), but cutoff=0.5 fires only top decile
(10% rate, +25.83 LOSO). Lowering cutoff to 0.42 matches baseline rate (40%):
**−0.43 LOSO, −0.06 pmin**. STK basically rediscovers a single-threshold rule
ranked by |pred| → equivalent to TBT-like behavior.

### Rule 4 — BAT (Batch-Adaptive Threshold)
**Disqualified** under shuffle test (see above). Apparent gain was per-sym
side-effect of sorted local cache.

## Stacking

NSF AND TBT (intersection of fire sets): same as NSF (−0.50 LOSO, +0.34 pmin)
because at fe=2.88e-4 NSF-fire ⊂ TBT-fire.
NSF OR TBT (union): same as TBT (−0.40 LOSO, −0.46 pmin). No stack-level gain.

## Recommendation

* **Do not promote any rule to iter_017.** DE asym thr is already the empirical
  optimum on this 4-way ensemble.
* BAT looked great but was a sym-sorted artifact — **important negative result**
  to record so that future audits don't re-propose batch-quantile rules without
  shuffle-testing.
* Best per_sym_min trade-off is NSF asym g=0.05 (+0.18 pmin for −1.79 LOSO),
  which is not worth it. Best LOSO match within rule family is NSF g=0 fe=2.5e-4
  (effectively a slightly tighter symmetric thresh).
* If transmission concern is paramount, a **scale-invariant fallback** like TBT
  (−0.40 LOSO) could be a backup if platform's data shifts heavily (since it
  doesn't depend on absolute pred-scale calibration). But baseline DE thr is
  ~1× FEE in scale and should travel well.

## Files

* `eval_4in1.py` — main eval (sorted order, all 4 rules)
* `eval_shuffle.py` — shuffle-invariance test (BAT validation)
* `eval_stk_stack.py` — STK PnL-based variant + stacking
* `eval_nsf_asym.py` — asymmetric NSF using DE base
* `results.json`, `results_shuffle.json`, `results_stk_stack.json`,
  `results_nsf_asym.json` — full numbers
* `eval*.log` — run logs

```
RESULT: task=decision_4in1 metrics={baseline_de_loso=44.72, nsf_loso=44.39, tbt_loso=44.32, stk_loso=44.28, bat_loso=43.44, best_rule=baseline_de, best_per_sym_min=4.46} notes=[BAT +47.13 was sym-sorted artifact (drops to +43.44 under shuffle); no rule beats DE asym thr; NSF g=0 ≈ symmetric thresh, TBT -0.40 LOSO, STK reduces to TBT-like behavior]
```
