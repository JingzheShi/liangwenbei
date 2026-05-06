# Yirun's Solution (1st place) — Jane Street Market Prediction 2021
**Kaggle competition, $100k prize**
URL: https://www.kaggle.com/competitions/jane-street-market-prediction/writeups/cats-trading-yirun-s-solution-1st-place-training-s

(Direct page only returns title without auth — content reconstructed from public
Kaggle discussion threads + community notes that quote the writeup.)

## Architecture (verified from community summaries)

* **Supervised Autoencoder + MLP** (SAE+MLP)
* Bottleneck features shared across 5 `resp` heads (responder targets at
  multiple time horizons: resp, resp_1, …, resp_4)
* All heads trained jointly
* 3-fold time-grouped CV with **10-day embargo**
* Multi-architecture ensemble (Yirun + others) using **middle-60% averaging**

## Loss / training (verbatim from competition discussions)

> "One approach was to plug the utility score function directly as loss
> function (multiplied by -1 to maximize), which already weights the importance
> of trade opportunities based on response and Jane Street weight."

> "A fine-tuning regularizer was applied to maximize the utility function by
> choosing action as the sigmoid of the outputs."

So Yirun (or a top competitor) **directly used the leaderboard utility as
training loss** — not cross-entropy. This is the PnL-aware loss strategy in its
purest form, applied to a winning submission.

## Sample weighting

* Used `weight = ln(1 + w_competition)` rather than raw `w` (log-bounded
  variance).
* Some competitors zeroed out `weight==0` rows; Yirun used a small nonzero
  weight (1e-5) so those samples still contributed gradient.
* "Monster deals" with high weight were essential — the model was tuned
  specifically to get these right because they dominate cumulative score.

## What we'd borrow

1. **Direct utility-as-loss is feasible at production scale** — Yirun won
   $100k with it. This validates our S2/S4 approach.

2. **Sample weighting by importance** — for Jane Street, `weight` is given.
   For us, `|Δp_t|` is the natural analogue (S1).

3. **Auxiliary targets / multi-horizon shared bottleneck** — already the
   structure of our Scheme C (multi-horizon LightGBM with shared features).
   We could extend to "for each horizon, train both CE-warmed AND PnL-fine-tuned
   booster, ensemble". Cheap win.

4. **Embargo in CV** — we already do LOSO; adding a small temporal embargo
   between LOSO splits could matter once we have time-aware features.

## What we'd skip

* SAE bottleneck — Jane Street had ~130 anonymised features and benefited
  from learned compression. Our 154-d schema is interpretable; an SAE would
  hurt.
* `weight` log-transform — irrelevant; we don't have a `weight` column,
  we have `Δp` (already linear).

## Critical caveat

Yirun's "utility-as-loss" was applied via fine-tuning — i.e. **CE pretraining
first**, then utility loss as a refinement. Same warm-start structure we
recommend in `r10_pnl_loss.md` S2 (50 rounds CE → switch to PnL obj). Without
warm-start, the pure utility loss often collapses (model finds trivial
"never trade" solution).
