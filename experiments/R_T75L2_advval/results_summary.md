# R_T75L2_advval — Adversarial-validation reweight on T75 LGB L2 (NEGATIVE)

## TL;DR
**Adversarial-val sample reweighting on T75 LGB L2 does not help, hurts at strong settings.**
Hypothesis falsified.

| Variant                 | LOSO-equiv | Δ vs baseline | min per-sym Δ |
|-------------------------|-----------:|--------------:|--------------:|
| Baseline T75 LGB L2     | +35.4487   | (anchor)      | (anchor)      |
| Trick aggressive [0.25,4] | +25.2739 | **−10.1748**  | −3.5525       |
| Trick mild [0.5,2.0]    | +35.1879   | **−0.2608**   | −0.5634       |

3-seed average over {42, 7, 13}; baseline reuses canonical T75 LGB L2 preds at
those seeds. DE asymmetric thresh search (5 DE seeds × 80 iter, sobol init).
Test = local 5-sym × 24-day (442k samples). Same V4 split (train 0-75, val 76-79).

## Adversarial classifier
- 5-fold date-block CV: each fold trains LGB binary on
  (past minus held dates) ∪ ALL future, evaluates on held past + future sample
- past = train cache (date 0-79, 1.47M samples, label=0)
- future = val + test caches (date 80-119, 736k samples, label=1)
- date / sym / time dropped → 359 schemeP features
- **Mean fold AUC = 0.9955** (folds: 0.996, 0.996, 0.996, 0.990, 0.999)
- Top discriminative features (by gain, all price-level):
  ask_mean, bid_mean, totalasize, open, totalbsize, ask_diff10, high, avgask, avgbid

→ Past/future are **trivially separable** because absolute price levels and
order-book sizes drift over 120 days. The classifier is so strong that the
OOF proba on past samples is nearly degenerate (p10=5.6e-5, p50=4.2e-3, p90=0.53).

## Sample-weight calibration
Computed sw_raw = clip(p, 0.01, 0.99) / (1 − clip(p, 0.01, 0.99)), then clip
to a configurable range, then mean-normalize to 1.

| Cap range  | mean | std  | p10  | p50  | p90  | max  |
|------------|-----:|-----:|-----:|-----:|-----:|-----:|
| [0.25, 4]  | 1.00 | 1.74 | 0.42 | 0.42 | 1.87 | 6.77 |
| [0.5, 2.0] | 1.00 | 0.67 | 0.77 | 0.77 | 1.71 | 3.09 |

The aggressive scheme leaves >50% of past samples deweighted to 0.42 (the cap)
while a small tail is upweighted up to 6.77×, concentrating training mass on a
non-representative subset. The mild scheme barely moves anything (most samples
within ±25% of nominal).

## Per-sym PnL (5 DE seeds → identical to 1e-5)

### Baseline
sym 0=+3.89, sym 1=+5.88, sym 2=+3.76, sym 3=+10.36, sym 4=+11.56

### Aggressive trick (sw cap [0.25, 4.0])
sym 0=+3.94, sym 1=+2.42, sym 2=+3.00, sym 3=+6.81, sym 4=+9.11
**Δ:** +0.05, **−3.47, −0.75, −3.55, −2.45** (only sym 0 unchanged; others bleed)

### Mild trick (sw cap [0.5, 2.0])
sym 0=+4.83, sym 1=+5.33, sym 2=+3.44, sym 3=+10.59, sym 4=+10.99
**Δ:** +0.94, −0.55, −0.32, +0.23, −0.56 (mixed; net Δ=−0.26 ≈ noise)

## Why adv val fails on this dataset
1. **AUC too high (0.9955).** Standard practice is to be wary above 0.7-0.8;
   above 0.95 the classifier is mostly identifying *price level drift* rather
   than meaningful regime change in label-relevant features. Reweighting on
   such a classifier moves you toward the long-tail outliers of the past
   distribution (high price levels, large book size), not toward "future-like
   trading regimes."
2. **Aggressive cap concentrates weight on outliers** that are
   non-representative for label dynamics — the resulting model trades
   off correlation (test_corr ↓ from ~0.16 baseline to 0.121-0.133 aggressive)
   for chasing the future feature distribution.
3. **Mild cap merely adds noise** — it doesn't sample-select aggressively
   enough to either help OR hurt.

Compare to **R1 date-decay** (linear w = 1 + 1.5·date/79 → +0.72 LOSO confirmed):
that's a smooth, principled prior weighting that doesn't concentrate on
outliers. Adv-val replaces that prior with a noisy data-driven signal that
turns out to be no better, and at strong settings actively counterproductive.

## Side-bonus stack with R1 date-decay
**NOT pursued** — adv-val itself doesn't help, so stacking with date-decay would
just be noise on top of R1's confirmed +0.72.

## Files
- `train_advval_classifier.py` — 5-fold date-block CV adversarial classifier
- `train_T75_l2_advval.py` — T75 LGB L2 with optional advval reweight
  (--no-advval reproduces baseline)
- `de_eval_advval.py` / `de_eval_mild.py` — DE thresh + LOSO-equiv eval
- `oof_proba_past.npy` — OOF proba for 1.47M past samples
- `sw_advval.npy` — aggressive [0.25, 4.0] weights
- `sw_advval_mild.npy` — mild [0.5, 2.0] weights
- `results.json` / `results_mild.json` — full DE results
- `advval_cls_summary.json` — classifier AUC + top features

## Decision
**NOT iter_017 candidate.** R1 date-decay (+0.72) remains the active L2
training-time gradient-shaping trick.

RESULT: task=t75l2_advval metrics={baseline=35.4487, trick_aggressive=25.2739, delta_aggressive=-10.1748, trick_mild=35.1879, delta_mild=-0.2608, adv_val_auc=0.9955, delta_per_sym_min_aggressive=-3.5525, delta_per_sym_min_mild=-0.5634} notes=[adv-val reweight HURTS T75 LGB L2; AUC=0.9955 too high → reweight concentrates on outliers; mild cap = noise (-0.26); R1 date-decay (+0.72) remains preferred. NOT iter_017 candidate. Side-bonus stack not pursued.]
