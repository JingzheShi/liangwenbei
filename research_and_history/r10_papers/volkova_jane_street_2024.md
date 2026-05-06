# Volkova's Solution (1st place) — Jane Street Real-Time Market Data Forecasting 2024
**Kaggle competition (Oct 2024 – early 2025)**
URLs:
* GitHub: https://github.com/evgeniavolkova/kagglejanestreet
* solution.md (full): https://raw.githubusercontent.com/evgeniavolkova/kagglejanestreet/master/solution.md

## Architecture

* **Time-series GRU** with sequence = one day per stock
* Two variants:
  * 3-layer GRU
  * 1-layer GRU + 2 linear layers + ReLU + dropout (slightly better, +0.001 CV)
* Both kept in ensemble (3 seeds each → 6 models, simple unweighted average).
  Ensemble LB 0.0112 vs single model best 0.0105.

## Loss function (verbatim from solution.md)

> "**The sum of losses (weighted zero-mean R²) for each responder was used to
> train the model**."

So **NOT a custom utility loss** — Volkova used **weighted MSE** (equivalent
to weighted zero-mean R² up to a constant) with `weight` from the competition
as sample weight. Auxiliary responders for multi-task learning.

This is **Scheme S1 (sample-weighted CE/MSE)** in pure regression form —
the simplest PnL-aware modification.

## Auxiliary targets (interesting trick)

```python
df = df.with_columns(
    (pl.col("responder_8")
     + pl.col("responder_8").shift(-4).over("symbol_id")
    ).fill_null(0.0).alias("responder_9"),
    (pl.col("responder_6")
     + pl.col("responder_6").shift(-20).over("symbol_id")
     + pl.col("responder_6").shift(-40).over("symbol_id")
    ).fill_null(0.0).alias("responder_10"),
)
```

Equivalent to multi-horizon target rolling-sum. **Same idea as our Scheme C
multi-horizon training.** Confirmed +0.001 CV gain.

## Online learning (key trick)

> "During inference, when new data with targets becomes available, I perform
> one forward pass to update the model weights with a learning rate of 0.0003.
> This approach significantly improved the model's performance on CV (+0.008)."

**Online updates dwarf the loss-function gain.** +0.001 from auxiliary,
+0.002 from features, **+0.008 from online**. Loss design is not the dominant
lever in this competition's setting.

## Performance progression

| Configuration | CV avg score |
|---|---|
| GRU 1, no aux, no online | 0.0112 |
| + auxiliary targets | 0.0190 |
| + online learning | 0.0201 |
| GRU 2 | 0.0214 |
| 3-seed ensemble of both | 0.0222 |

## What we'd borrow

1. **Sample-weighted regression loss** (= our S1) — confirmed winning approach
   in 2024. Despite all the recent "differentiable Sharpe" / "PnL loss"
   literature, the *actual 1st place* used weighted MSE.

2. **Auxiliary multi-horizon targets** — already in our Scheme C. ✓

3. **Online learning** — *not applicable* for us (the platform doesn't expose
   targets at inference; submission must be pure inference). Cannot adapt.

4. **3-seed ensembling** — already explored in our T6b experiment, showed
   negligible gain. Skip.

## What this tells us

The takeaway is sobering: a Kaggle 1st place in the most relevant competition
(Jane Street market data forecasting 2024) was won with **plain weighted MSE +
multi-task auxiliary targets + online learning**, NOT with a custom utility
loss. This argues for the simplest scheme (S1) as our default and the more
ambitious S2/S7 only as parallel experiments.

But: the leaderboard metric for Jane Street 2024 was a weighted-R²-based
"utility score" that **already includes** sample weighting in the metric. Our
metric is `Σ pnl_t` with no sample weight — flatter — so weighted CE/MSE is
LESS automatic for us. We may need to be more explicit about S2.
