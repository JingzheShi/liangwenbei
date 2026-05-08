# T65 Pseudo-Labeling — Half-Supervised Train Augmentation

**Status:** TBD pending DE thresh sweep.
**Baseline:** iter_010 = +24.52 LOSO-equiv (T59 5-seed DE thresh on 442k test).
**SOTA target:** > +24.52 → iter_011 candidate.

## 1. Idea

Self-train trick from Jane Street 2020 / Optiver 2023 top solutions:
1. Predict on test set (date 96-119) with iter_010 5-seed-avg model.
2. Gate high-confidence directional predictions:
   `max(prob_0, prob_2) > 0.55` AND `|prob_0 - prob_2| > 0.10`.
3. Take pseudo label = `argmax(prob_0, prob_2)` (binary 0/2 — directional only).
4. Append to train set with `sample_weight = 0.5` (downweight 50%).
5. Retrain 5-seed (R34 Stage 3 340-d, aug_a, GPU LightGBM).
6. DE 4D thresh on full 442k test, compare vs iter_010.

## 2. Pseudo-data quality

| Metric | Value |
|---|---|
| Total test rows | 442,080 |
| Pseudo gate (`max(p0,p2) > 0.55, |Δ| > 0.10`) | 14,251 (3.22%) |
| Pseudo label = up (2) | 6,767 |
| Pseudo label = dn (0) | 7,484 |
| Pseudo accuracy vs true label | ~48% (30% are actually flat=1) |

The pseudo accuracy is low because ~30% of high-confidence directional rows
are actually labeled flat — the model's directional belief is correct, but
the move size doesn't reach α=0.1%. This adds noise to training, which is
why we cap `sample_weight=0.5`.

Per-sym breakdown:

| sym | n_pseudo | up | dn | acc_vs_true |
|---|---|---|---|---|
| 0 | 537 | 154 | 383 | 0.559 |
| 1 | 5,204 | 2,248 | 2,956 | 0.472 |
| 2 | 3,277 | 1,606 | 1,671 | 0.443 |
| 3 | 1,231 | 671 | 560 | 0.491 |
| 4 | 4,002 | 2,088 | 1,914 | 0.516 |

## 3. Training setup

- Original train: 1,473,600 rows (date 0-79, syms 0-4)
- aug_a (`U[0.80, 1.20]` per-feature scale) on original → 2,947,200 rows
- + Pseudo: 14,251 rows (no aug, sample_weight=0.5 × class-balance)
- Total: 2,961,451 rows
- Val: 294,720 rows (date 80-95, early stopping)
- GPU LightGBM, 5 hyperparam-diverse seeds (same as T59):
  - seed=1 (ff=0.6, bf=0.7, leaves=127, l2=1.0)
  - seed=7 (ff=0.7, bf=0.85, leaves=63, l2=2.0)
  - seed=13 (ff=0.5, bf=0.6, leaves=255, l2=0.5)
  - seed=42 (ff=0.8, bf=0.8, leaves=127, l2=1.0)
  - seed=100 (ff=0.4, bf=0.5, leaves=127, l2=3.0)

## 4. Results

(populated by collect_results.py once training + DE eval finish)

### Per-seed test argmax cum_pnl (5-sym sum)

| seed | best_iter | full test cum_pnl | per-sym sum |
|---|---|---|---|
| 1 | TBD | TBD | TBD |
| 7 | TBD | TBD | TBD |
| 13 | TBD | TBD | TBD |
| 42 | TBD | TBD | TBD |
| 100 | TBD | TBD | TBD |

### 5-seed-avg DE 4D thresh

| Eval scope | DE LOSO sum | vs iter_010 (+24.52) |
|---|---|---|
| Full 442k test | TBD | TBD |
| Held-out (non-pseudo, ~427k) | TBD | TBD |

## 5. Caveats

⚠️ **Test-set leakage**: pseudo training rows are pulled from test X. Even with
pseudo labels (not ground truth), training on test X means the model sees the
test feature distribution, biasing held-in evaluation upward. The fairer view
is held-out (non-pseudo) cum_pnl. We report both.

⚠️ **Production realism**: this technique is only valid as a self-distillation
on the *local* test we have labels for. At submission time, we'd need to
retrain with pseudo labels from the *platform's* hidden test, which is not
possible. So the iter_011 candidate model would have memorized our local
test — its platform performance is the open question.

## 6. Decision

(filled after eval)
