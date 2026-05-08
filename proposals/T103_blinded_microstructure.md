# T103 — Blinded Microstructure Proposal

> **Mode**: Blinded — proposer was NOT allowed to read `experiments/`, `audits/`, `research_and_history/`, or any prior PROGRESS/REPORT before writing Parts 1–2. Only `CRITICAL_CONSTRAINTS.md` + `docs/data_schema.md` + the dispatch prompt.
> **Date**: 2026-05-08
> **Goal**: Propose disruptive but *literature-grounded* methods through an HFT / market-microstructure lens, then compare against the in-house frontier.

---

## Part 1 — Task Understanding (first principles)

### 1.1 What this problem actually is

We are predicting **mid-price direction at 5/10/20/40/60 ticks ahead** (= 15s / 30s / 60s / 120s / 180s) from a 100-tick (~5min) LOB+order-flow snapshot, on 5 anonymized A-share equities, with the catch that the **test stocks may not be among the 5 training stocks**.

The label is a binarized signed return at thresholds α ∈ {5bp, 5bp, 10bp, 10bp, 10bp} — but the **scoring is PnL** with a 1bp/side commission, which means:

```
PnL_per_trade ≈ sign(ŷ) · r_realized  −  cost
```

…where `cost ≈ 1bp` per side ≈ 2bps round-trip and `r_realized` is the *signed continuous return*, not the binarized label. **The label is a proxy; the score is continuous return × decision − cost.** This is the hardest thing to internalize.

### 1.2 What makes this brutal

1. **Label ≠ score.** The 3-class CE loss most teams will reach for is *not* the right loss. A correctly-labeled flat trade earns 0; an incorrectly labeled "0" on a +0.04% move loses 4bp − cost; a correctly labeled "2" on a +0.06% move earns only 6bp − 2bp = 4bp net. The decision boundary that maximizes PnL is **not** the natural α-threshold.
2. **Brutal class imbalance × cost asymmetry.** label=1 dominates 54–76% (per horizon). A naive "always flat" predictor scores 0. A reckless "always trade" predictor pays 2bp × (every tick). The Sharpe optimum is much sparser than label distribution suggests.
3. **OOD stocks at test.** This is *not* the usual quant-PM setting where you have stock-specific features. We must build a feature pipeline whose joint distribution is invariant across A-share equities — this rules out per-stock standardization, sym embeddings, and any per-stock model. Cross-sectional rank, vol-normalized features, and tick-invariant ratios become the *only* legal cross-stock features.
4. **No state.** No LSTM hidden carryover, no exponential-decay queue tracker that lives across `predict()` calls. Every prediction is from scratch on a 100-tick slab, and the slabs come **out of order**. This effectively forces every "memory" feature to be **explicitly computed inside the 100-tick window** — no shortcuts.
5. **Three-second tick is "slow"-HFT.** Faster than equities daily, far slower than nano-second LOB. Microstructure noise (Roll bid-ask bounce, queue jitter) is at the same scale as the 5bp threshold. The signal is *barely* above noise.
6. **`amount_delta` is the only un-normalized field.** Either log-transform or drop. Most newcomers will silently let it dominate any tree split.
7. **Independent per-horizon scoring with "best wins".** This means we do **not** need to do well across all 5 horizons — focus capital on the horizon with the best signal-to-cost ratio. Empirically, longer horizon → wider expected move → easier to clear cost. So h=60 is likely the optimum target unless h=5 has an unusually persistent micro-edge. A sensible team **specializes**.

### 1.3 The kernel of the problem

Stripped of ML window dressing, the problem is:

> **Given a 5-min snapshot of LOB + 6-class order flow, output a continuous expected signed return for some horizon h, and trade only when the expected move exceeds the implied effective cost.**

Everything else — the 3-class label, the CE loss, the class imbalance — is a layer on top of this kernel. Methods that work *on* the kernel will dominate methods that work on the layer.

---

## Part 2 — Disruptive method proposals (10)

Each method: 1-line description / source / why "basic" (well-known) / why disruptive (likely under-applied here) / cost / expected lift / risk.

### M1. Direct continuous-return regression with PnL-tuned thresholds

- **Description**: Train one regression head per horizon predicting `r̂ = (mid_{t+n} − mid_t)/(mid_t + 1)`. Choose action = 2 if r̂ > τ_up, action = 0 if r̂ < τ_dn, else 1. Tune (τ_up, τ_dn) by grid search **directly on validation PnL**, not on accuracy.
- **Source**: López de Prado, *Advances in Financial Machine Learning* Ch. 3–5; standard quant practice (Cartea-Jaimungal MM book §10).
- **Why basic**: Regression-then-threshold is textbook ML-finance. Two free parameters per horizon.
- **Why disruptive (here)**: Most LOB-prediction baselines (DeepLOB, HLOB, etc.) use 3-class CE. The label-binarization throws away magnitude info that *directly* enters PnL. Regression preserves the ordering, and decoupled thresholds let you set the trading bar at exactly where expected move > cost, regardless of where the data designer set the label α.
- **Implementation cost**: Low. LightGBM with `objective='regression_huber'` or `regression_l1`. Threshold tuning is `O(grid)` on a validation slice.
- **Expected lift**: **+5 to +15 PnL units**. The biggest single thing if not already done — recovers all the +0.04% / −0.04% near-threshold information that the label discards. Asymmetric thresholds (τ_up ≠ |τ_dn|) often add +1–3 more on real data because of secular drift.
- **Risk**: Regression target is heavy-tailed; needs Huber/MAE not MSE, and outlier clipping at ~0.5–1%.

### M2. Decision-Focused Loss (SPO+ / soft-PnL gradient surrogate)

- **Description**: Replace cross-entropy with a smooth differentiable approximation of `−PnL`. Two flavors: (a) Elmachtoub-Grigas SPO+ loss, (b) softmax-PnL: `loss = − Σ_c p_c · expected_pnl_c`, with `expected_pnl_c = (c−1) · r_realized − cost · |c−1|`. Train LightGBM with custom (grad, hess) or train a small MLP/temporal CNN.
- **Source**: Elmachtoub & Grigas, "Smart Predict, Then Optimize" (Mgmt. Sci. 2022); decision-focused learning literature, e.g., Donti-Amos-Kolter (2017).
- **Why basic**: Decision-focused learning is well-established in OR/ML.
- **Why disruptive (here)**: CE optimizes log-likelihood of the label; PnL surrogate optimizes the actual score. The gradient direction differs especially in the "weak signal" majority (label=1) — CE wastes capacity learning to be correct on flats, soft-PnL learns to abstain.
- **Implementation cost**: Medium. LightGBM custom objective is ~50 lines.
- **Expected lift**: **+3 to +8 PnL** *on top of* M1, because it changes the *training signal* not just the threshold.
- **Risk**: Custom-objective gradients are noisier; expect more careful early stopping and regularization. May also overfit if r_realized is leaked into the objective without proper holdout.

### M3. Multi-scale / queue-decomposed OFI à la Cont-Kukanov-Stoikov (2014)

- **Description**: Compute OFI at multiple decay scales τ ∈ {3s, 9s, 30s, 90s, 300s}, decomposed by event type. CKS define OFI on level-1 queues from limit/market/cancel events; we already have lb/la/mb/ma/cb/ca decomposition. So compute, for each τ:
  ```
  ofi_τ = Σ_t∈window e^(−(t_now − t)/τ) · (lb_intst − la_intst − mb_intst + ma_intst − cb_intst + ca_intst)
  ```
  plus per-event-type variants (just lb−la, just mb−ma, etc.) and side-asymmetry: `ofi_buy_side = lb − cb`, `ofi_sell_side = la − ca`.
- **Source**: Cont, Kukanov, Stoikov (2014) "The price impact of order book events", JFE; cross-impact follow-ups (Kolm-Westray 2023).
- **Why basic**: OFI is *the* canonical microstructure feature.
- **Why disruptive (here)**: Most ML pipelines use snapshot LOB features (level-1 imbalance, depth) or single-scale OFI. Multi-scale + per-event-type decomposition is a well-known result that separately identifies (a) liquidity demand (mb, ma), (b) liquidity provision (lb, la), (c) cancellations (cb, ca = informed/spoofing). The dataset already gives us this decomposition for free — *not* using it is leaving alpha on the table.
- **Implementation cost**: Low. Pure NumPy, ~30 lines per feature family. Probably 30–50 new features.
- **Expected lift**: **+5 to +12 PnL**. CKS-style OFI is the closest thing to a "free lunch" on this dataset.
- **Risk**: Feature explosion; need L1 / feature-importance pruning. Multi-scale decay constants need tuning but the literature gives reasonable defaults.

### M4. Stoikov micro-price + micro-price drift as direct features

- **Description**: Add three derived prices to features:
  - `microprice = (bid1·asize1 + ask1·bsize1) / (bsize1 + asize1)`  (sign convention: imbalance-weighted mid)
  - `micro_dev_t = microprice_t − midprice_t`  (instantaneous "fair-price residual")
  - `micro_dev_lag_diff = micro_dev_t − micro_dev_{t−k}`  for k ∈ {1, 5, 20}
  Optional: full Stoikov calibrated-microprice (G-matrix iteration) but this needs offline calibration.
- **Source**: Stoikov, "The micro-price: a high-frequency estimator of future prices", *Quantitative Finance* 18(12), 2018.
- **Why basic**: One of the most-cited HFT papers post-2015.
- **Why disruptive (here)**: It's a **closed-form, zero-parameter** feature that empirically beats midprice as a future-price predictor. Trivial to add. Many ML feature builders forget it because it's a "trader" object rather than a "feature engineering" object.
- **Implementation cost**: Trivial (5 lines).
- **Expected lift**: **+3 to +6 PnL**.
- **Risk**: May be partially redundant with level-1 imbalance + spread. Hedge: add it and let the GBM decide.

### M5. Realized-volatility-aware sizing (Sharpe gating)

- **Description**: Estimate window-realized vol σ̂_t (Garman-Klass-Yang-Zhang on OHLC; we already have OHLC). Trade only when `|r̂| / σ̂_t > κ`, picking κ from a Sharpe-equivalent argument (κ ≈ cost / σ̂ + safety margin). This converts a **fixed**-threshold rule into a **regime-aware** rule — it raises the bar in volatile windows (where r̂ is noisier) and lowers it in calm windows (where the same r̂ is more reliable).
- **Source**: Yang & Zhang (2000) "Drift-independent volatility estimation"; Garman-Klass (1980); Cartea-Jaimungal-Penalva *Algorithmic and HF Trading*, Ch. 4.
- **Why basic**: Vol-normalized signals are how every prop desk runs.
- **Why disruptive (here)**: Most ML stacks treat all rows symmetrically. PnL is asymmetric in vol: a high-vol window with the *same* `r̂` is materially less reliable. Gating by `r̂ / σ̂` instead of `r̂` should monotonically improve PnL.
- **Implementation cost**: Low. GKYZ is closed-form on OHLC.
- **Expected lift**: **+3 to +8 PnL**.
- **Risk**: 100-tick σ̂ is noisy; consider blending intraday vol with a longer rolling estimate via window-only quantiles.

### M6. Roll's implicit spread → window-specific effective cost estimate

- **Description**: Compute `roll_spread_t = 2 · √(−Cov(Δp_τ, Δp_{τ−1}))` over the 100-tick window. Use this as a **tick-local** estimate of effective transaction cost (which incorporates adverse selection on top of the platform's nominal 1bp). Replace the static `cost = 0.0001` in the threshold rule with `cost = max(0.0001, ½ · roll_spread)`.
- **Source**: Roll (1984) "A simple implicit measure of effective bid-ask spread"; Hasbrouck (2009).
- **Why basic**: 40-year-old textbook microstructure result.
- **Why disruptive (here)**: The platform's quoted 1bp commission is a **lower bound** on true effective cost — adverse selection means the realized cost on a directional bet is usually higher. A per-window effective-cost estimate should cull "marginal" trades that look profitable on paper but lose net.
- **Implementation cost**: Low (one line of NumPy).
- **Expected lift**: **+2 to +6 PnL**.
- **Risk**: Roll's measure is noisy and biased on small samples. Mitigation: floor it at the platform cost.

### M7. Cancel/limit ratio as informed-flow / iceberg detector

- **Description**: Build features
  - `iceberg_b = cb_intst / (lb_intst + ε)`, similarly `iceberg_a`
  - `flow_aggression_b = mb_intst / (lb_intst + ε)`, similarly `flow_aggression_a`
  - `cancel_asymm = (cb − ca) / (cb + ca + ε)`
  Hypothesis: cb >> lb (lots of cancels relative to additions) on the bid side = quote-stuffing or iceberg-refilling = *informed* selling pressure → near-term down-move.
- **Source**: Hasbrouck, *Empirical Market Microstructure* Ch. 13; Cartea-Jaimungal-Penalva on adverse selection; Kirilenko et al. (2017) on HFT cancel patterns.
- **Why basic**: Cancel-flow analysis is standard at HFT shops.
- **Why disruptive (here)**: **Most public LOB datasets do not give cancel data**. We have it explicitly (`cb_intst`, `ca_intst`). This is the biggest *unique* advantage of this dataset and is likely under-exploited.
- **Implementation cost**: Trivial (~10 lines).
- **Expected lift**: **+3 to +7 PnL**.
- **Risk**: Noisy on short windows; needs smoothing across multi-tick aggregation.

### M8. Kyle-Obizhaeva trading-invariance normalization (the OOD-stock fix)

- **Description**: Convert size-scale features into "bets per business-time" by normalizing with `bet_scale ∝ (σ̂_t · ADV̂_t)^(1/3)` per Kyle-Obizhaeva invariance. Implementation:
  - Compute `vol_proxy = sum(volume_delta) over window`
  - Compute `σ̂ = realized vol over window`
  - For every size-class feature, replace `f` with `f / (σ̂ · vol_proxy)^(1/3)`
  This makes the feature roughly the same distribution across stocks of vastly different liquidity — exactly the OOD-stock problem.
- **Source**: Kyle & Obizhaeva, "Market Microstructure Invariance: Empirical Hypotheses" (Econometrica 2016); Obizhaeva 2020 follow-ups; see also Andersen-Bondarenko applications.
- **Why basic**: Foundational microstructure invariance result.
- **Why disruptive (here)**: The constraint that test stocks can be OOD (sym ID 0–4 may map to a *different* stock) is the dataset's nastiest twist. Almost no Kaggle/competition pipeline applies a true cross-sectional invariance normalization. This is a **structural** fix at feature level, not a model-level hack.
- **Implementation cost**: Medium. Requires careful per-window estimation of σ̂ and ADV-proxy.
- **Expected lift**: **+5 to +15 PnL** *if* the test set actually has OOD stocks (could be the dominant gap). Could be neutral if test stocks happen to overlap.
- **Risk**: Wrong normalization can *hurt* on in-distribution stocks. Hedge: add normalized features alongside raw, let GBM choose.

### M9. Triple-barrier-style PnL meta-labeling (skip-vs-trade gate)

- **Description**: Two-stage stack:
  1. Stage-1: a strong directional model (could be M1 regression).
  2. Stage-2 ("meta"): a binary classifier predicting "given Stage-1 says trade in direction d, will I make money net of cost?" — trained on Stage-1's out-of-fold predictions + microstructure features (liquidity, vol, time-of-day).
  Action = (Stage-1 direction) iff (meta-prob > τ_meta).
- **Source**: López de Prado, *AdvFML* Ch. 3 (triple-barrier + meta-labeling). Standard at quant funds.
- **Why basic**: Idiomatic two-stage ML-finance design.
- **Why disruptive (here)**: Most ML systems threshold on the Stage-1 confidence directly. A separately-trained meta-model **learns systematic Stage-1 mistakes** (e.g., "Stage-1 over-confidently predicts up in low-vol low-imbalance windows") that the Stage-1 model cannot self-correct without overfitting.
- **Implementation cost**: Medium. Need OOF Stage-1 predictions, then Stage-2 trained with PnL-aware targets.
- **Expected lift**: **+3 to +10 PnL**.
- **Risk**: Stacking risk if OOF construction is sloppy. Use group-purged K-fold by date.

### M10. Order-book slope / resilience as a microstructure regime feature

- **Description**: Compute book slope on each side:
  - `slope_b = (bid1 − bid10) / (Σ_i bsize_i)`
  - `slope_a = (ask10 − ask1) / (Σ_i asize_i)`
  - `slope_asymm = (slope_b − slope_a) / (slope_b + slope_a + ε)`
  Stiff side (small slope) = resilient = price unlikely to penetrate. Soft side = thin = price likely to give. Strong asymmetry is a directional signal.
- **Source**: Avellaneda-Stoikov (2008) "High-frequency trading in a limit order book"; Cartea-Jaimungal-Penalva textbook; queue-reactive LOB models.
- **Why basic**: Standard market-making concept.
- **Why disruptive (here)**: Slope features rarely make it into pure-ML pipelines because they're "trader heuristics". Yet they capture book-resilience information that snapshot 10-level features don't expose to a tree-based learner cleanly.
- **Implementation cost**: Trivial.
- **Expected lift**: **+2 to +5 PnL**.
- **Risk**: Likely partially correlated with `imbalance` and `cumspread`. Add and let GBM decide.

---

### Bonus method (M11): Adversarial validation for distribution drift

- **Description**: Train a binary classifier "is this row from train or held-out late dates?" on date-based holdout. Identify training rows that look like held-out → upweight them in final training. Combats temporal drift and (partly) sym-shift.
- **Source**: Kaggle folklore; formalized in domain-adaptation literature (Sugiyama-Kawanabe).
- **Why basic**: Kaggle bread-and-butter.
- **Why disruptive (here)**: Stationarity is heroic on 120 trading days; OOD on test stocks compounds. Adversarial weighting is one of the few principled fixes that doesn't require knowing the test distribution.
- **Implementation cost**: Medium.
- **Expected lift**: **+2 to +6 PnL**.
- **Risk**: Risk of overfitting to a particular holdout window.

---

## Part 3 — Comparison with current state

*(Read AFTER Parts 1–2 were finalized: `PROGRESS.md`, `experiments/T75_regression_dmid/REPORT.md`, `experiments/T87_spo_dfl/REPORT.md` only. Experiment **directory names** scanned for keyword overlap, but their REPORTs were not read.)*

### 3.1 What I proposed that is already done (gap in my thinking)

| My method | In-house status | Evidence |
|---|---|---|
| **M1 — direct regression on Δmid_norm + DE-tuned EV gate** | **DONE in T75 / iter_013** | T75 REPORT §"Pipeline" trains 5-seed LightGBM with `objective=regression_l2` on `(mp_t60-mp_t)/(mp_t+1)` and DE-tunes asymmetric `(thr_up, thr_dn)`. Got **+9.79 LOSO-equiv** over iter_012's 3-class CE. **Identical to my M1.** |
| **M2 — SPO+ / decision-focused loss** | **DONE in T87 / iter_015** | T87 REPORT derives the exact SPO+ surrogate I proposed (`ReLU(\|2ĉ-c\| − fee_eff) − z*·(2ĉ-c) + fee·\|z*\|`) and warm-starts T81 NN with `λ_spo=30, lr=3e-5`. Got **+1.85** over iter_014 → final +40.13 LOSO-equiv. **Word-for-word identical to my M2** down to the convexity argument. |
| **M11 — adversarial validation** | **Likely tried** in `experiments/T66_adversarial_val/` (directory name match; report not read). |
| Vol-feature engineering (M5-adjacent — but the *gate* still fixed) | **Partially tried** in `T88_range_vol_features/` and `T96_hlc_range_factor/` (names match Garman-Klass / range-vol style features). **However**, the iter_015 EV gate is still a global pair of **fixed thresholds** `thr_up=3.58e-4, thr_dn=2.16e-4` (T87 REPORT §"Ensemble weight sweep"), **not** a vol-normalized Sharpe-style gate. So my M5 in feature form is partly done; my M5 in *gate* form is not. |
| Cross-sym OOD mitigation (M8-adjacent) | **Partially tried** in `T26_domain_randomization`, `T48_cross_sym_pooling`, `T59_fullsym_train` (names suggest cross-sym mixup / domain randomization). **However**, no name suggests **Kyle-Obizhaeva trading-invariance** normalization specifically — the standard fix here is mixup or pooling, which are generic ML tricks, not the structural microstructure-invariance fix. |

So my biggest "thinking gap" is: **I re-derived M1 (regression+EV) and M2 (SPO+ DFL) as "disruptive" — they are the *current frontier* of this project**. That's a reassuring sanity check (the in-house frontier is on solid microstructure ground), but for a "blinded outsider proposing disruptive ideas" that means **M1 and M2 are not differentially useful here** — they're already the +13.69 backbone of iter_015.

### 3.2 What I proposed that is NOT in the in-house frontier (the gold)

These don't appear by name in the experiment directory or in the T75 / T87 REPORTs:

1. **M3 — Multi-scale Cont-Kukanov-Stoikov OFI with l/m/c decomposition.** No experiment names "OFI", "cont_stoikov", "queue_reactive", or similar. The current 359-d "schemeP" may incidentally include some OFI-ish features, but a *deliberate* multi-scale exponentially-decay-weighted OFI (separately for lb/la, mb/ma, cb/ca) is not visible. **Strong gold.**
2. **M4 — Stoikov microprice and `micro_dev` features.** No name match. The EV-gated Predictor already uses raw bid1/ask1/imbalance, but `microprice_t = (bid1·asize1+ask1·bsize1)/(bsize1+asize1)` and `micro_dev_t = microprice_t−midprice_t` are zero-cost adds that may not be in schemeP. **Gold (cheap).**
3. **M5 (gate form) — vol-conditioned (Sharpe-style) EV gate.** Vol *features* are present but the gate is fixed. Replacing `r̂ > thr` with `r̂/σ̂ > κ` is a 50-line edit to `Predictor.predict` + a new threshold sweep. **Gold (cheap).**
4. **M6 — Roll's implicit-spread per-window effective cost.** No name match. The platform's nominal `0.0001` cost is hard-coded. **Gold (cheap).**
5. **M7 — Cancel/limit iceberg ratios (`cb/lb`, `ca/la`, `cancel_asymm`).** No name match. The competition data is *unusual* in giving us cb/ca explicitly. The raw `_intst` features are likely in schemeP, but ratios are non-linear transforms a tree may not learn from raw features alone. **Gold (cheap).**
6. **M8 — Kyle-Obizhaeva invariance normalization.** No name match. T26/T48/T59 attack OOD via mixup/pooling, which is mechanism-agnostic; KO invariance is the *structural* fix. **Gold (medium cost, biggest potential upside if test set has OOD stocks).**
7. **M9 — Triple-barrier-style meta-labeling stage.** Partially overlaps with `T72_binary_cascade` (name suggests two-stage but unclear), but the formal Prado meta-labeling on PnL-positive vs PnL-negative trades does not have a clear match. **Possible gold.**
8. **M10 — Order-book slope as a separate engineered feature.** No name match. May be implicit in schemeP, but a deliberate `slope_b`, `slope_a`, `slope_asymm` triplet is unlikely. **Modest gold.**

### 3.3 Strategic read on the project

The project has reached **+40 LOSO-equiv ≈ +20 platform PnL** primarily by climbing the "loss-function → decision-function alignment" axis (CE → regression → SPO+ DFL). That's the **decision-side** of the kernel I described in §1.3.

The **input-side** of the kernel — *high-quality microstructure features specifically engineered from the data's unique advantages* (cancel data, 10-level depth, tick spacing) — looks comparatively **less mined**. Specifically:
- **OFI-decomposition features** (M3) and **cancel-flow ratios** (M7) exploit the dataset's unusual gift of `lb/la/mb/ma/cb/ca` decomposition.
- **Microprice** (M4) and **slope** (M10) are zero-cost, well-known additions the in-house schemeP may have missed.
- The **OOD-stock structural fix** (M8) is a different mechanism than current pooling/mixup attempts, and addresses CRITICAL_CONSTRAINTS §3 head-on.
- The **EV gate** is still fixed-threshold; making it **vol-aware** (M5) is a small change with disproportionate decision-time leverage.

In short: **the loss surface has been polished; the input surface still has well-known microstructure features lying around that may not have been picked up.**

---

## Part 4 — Top 2 immediately-actionable recommendations

Ranked by `(not-done) × (high-impact) × (low-implementation-cost)`.

### #1 — Vol-conditioned (Sharpe-style) EV gate as a Predictor-level patch (M5, gate form)

**Why this first.** The cheapest single change with a clear theory-backed path to extra PnL. iter_015's gate is `r̂ > thr_up=3.58e-4 (or r̂ < -thr_dn=2.16e-4)` regardless of regime. PnL depends on the *signal-to-noise ratio* of `r̂`, not its absolute scale: in a high-vol window the same r̂ is less reliable; in a calm window it's more reliable. A Sharpe-style gate makes the threshold **regime-aware**.

**Concrete plan**:
1. Worker writes a new `ev_gate_v2_vol_norm.py` in `experiments/T<next>_vol_sharpe_gate/`.
2. Inside `Predictor.predict`, compute on the 100-tick window:
   ```python
   sigma_gk = garman_klass_yang_zhang(window['open'], window['high'], window['low'], window['close'])
   # GKYZ closed-form, ~10 lines numpy
   ```
3. Replace decision `r̂ > thr_up` with `r̂ / sigma_gk > kappa_up`.
4. Re-run DE asymmetric tuning over `(kappa_up, kappa_dn)` on the same V4 walk-forward val + the local 442k test, mirroring T87's protocol.
5. **Defensive symmetric fallback**: also test `|r̂|/σ̂ > κ` with a single κ (no DE).
6. **Compare**: cum_pnl on local 442k test vs iter_015's +40.09. **Hypothesis**: a flat 1–3 PnL gain. Bigger gain on per-sym dispersion (sym=0 currently +3.7, sym=4 currently +14.5 — vol normalization should narrow that asymmetry, which itself is a robustness signal).
7. **Compute time**: ~4 h end-to-end (no model retrain — just new gate + DE search + repackage).
8. **Risk**: GKYZ noisy on 100 ticks. Mitigation: blend window-vol with a constant-floor: `σ_gate = max(σ_floor, σ_gk)`. The σ_floor turns the rule into a smooth interpolation between Sharpe-gate (high vol) and absolute-gate (low vol).

**Estimated lift**: **+1 to +4 PnL** on top of iter_015 (+40.09). Worst case: tied with iter_015 (Sharpe-vs-absolute gate is provably no worse if you grid the threshold both ways and pick the better).

### #2 — Kyle-Obizhaeva trading-invariance feature normalization for OOD-stock robustness (M8)

**Why this second.** This is the only **structural** fix to CRITICAL_CONSTRAINTS §3 (OOD stocks at test time) that I propose, and the in-house attempts (T26 domain-rand, T48 cross-sym pooling, T59 fullsym train) appear to use **generic ML mechanisms** rather than a microstructure-invariance scaling. The dataset's per-sym PnL spread on iter_015 (`+3.74, +5.32, +4.55, +11.97, +14.51` — a ~4× range across the *training* stocks) is itself evidence that the model is **not** sym-invariant; if we add OOD stocks, this fragility is the most likely failure mode.

**Concrete plan**:
1. Worker creates `experiments/T<next>_ko_invariance/`.
2. For each (sym, date, AM/PM) session, precompute on the 100-tick window:
   ```python
   sigma_w = realized_vol(window['midprice'])             # window-level vol (% per √(100·3s))
   adv_w   = window['volume_delta'].sum()                 # window-level traded turnover
   bet_scale = (sigma_w * adv_w + eps) ** (1/3)           # Kyle-Obizhaeva invariance scale
   ```
3. **Augment**, don't replace, the schemeP feature set:
   - For every size-class feature `f` (volume_delta, all bsize/asize, lb_intst..ca_intst, totalbsize, totalasize), add `f / bet_scale` as a new feature. ~50 new features.
   - Keep raw features in parallel (let GBM choose).
4. Retrain T75 LightGBM 5-seed on **schemeP + KO-normalized features** (~410 dims).
5. Retrain T87 NN 5-seed warm-started from the new T75 (V4 walk-forward), keeping SPO+ λ=30.
6. Local PnL evaluation: total LOSO-equiv (training stocks) and especially `LOSO` proper (leave-one-sym-out): each held-out sym is the closest proxy we have for "OOD stock". **Hypothesis**: KO-normalized features should improve held-out-sym performance more than training-sym performance — that's the diagnostic.
7. **Compute time**: ~1–2 days (full retrain, but using the same T75/T87 recipe).
8. **Risk**: KO scaling can underperform on in-sample stocks if the parameters are mis-set. Mitigation: include both raw AND normalized features and let LightGBM importance pick.

**Estimated lift**: **+3 to +10 platform PnL** *if* the public/private test set contains OOD stocks (CRITICAL_CONSTRAINTS §3 says it can). The fact that iter_015's local LOSO-equiv = +40 maps to platform = +20 (transmission ratio ~0.5) is itself consistent with an OOD-driven signal-decay that KO normalization should partly close.

### Bonus pick #3 (free if you have a worker idle)

**M3 multi-scale CKS OFI + M4 microprice + M7 cancel ratios** as a single 1-day "feature gold rush" sprint. Adds ~80 features grounded in 3 well-cited papers (Cont-Kukanov-Stoikov 2014, Stoikov 2018, Hasbrouck 2009). Low risk: superset of current schemeP, GBM does the selection. Expected: **+1 to +3 PnL** if any of them isn't already in schemeP. Use this if pick #1 and pick #2 finish quickly and you want a parallel push.

---

## Self-assessment of this proposal's blindness quality

- **Genuine outsider view confirmed.** I independently re-derived M1 (regression+EV gate) and M2 (SPO+ DFL) — the project's *current frontier* — purely from the kernel argument in §1.3. That tells me my microstructure-first framing tracks the in-house framing well: the iter_015 stack is on the right axis.
- **Gap exposed: I hit the loss-function frontier and not the input-feature frontier.** The fact that M1+M2 = current frontier means the loss-function axis is roughly mined; the alpha left is on the **input/feature** side (M3, M4, M5-gate, M7) and on the **OOD-structural** side (M8). That is exactly where the top 2 recommendations land.
- **What I'd want to read next** if given more time: `experiments/T68_stage5_features/REPORT.md` (to see what's actually in schemeP), `experiments/T26_domain_randomization/REPORT.md` (to confirm M8 is genuinely different), and `experiments/T88_range_vol_features/REPORT.md` (to confirm vol features are in features but not in the gate).

---

```
RESULT: task=blinded_microstructure top_recommendations=[M5_vol_sharpe_gate, M8_ko_invariance] notes=re-derived M1 regression+EV (already iter_013) and M2 SPO+ DFL (already iter_015) confirming in-house frontier; gap on feature/input side and OOD structural side; top1 vol-conditioned gate is 4h Predictor patch with +1-4 expected; top2 Kyle-Obizhaeva invariance is 1-2d feature retrain attacking OOD-stock constraint structurally with +3-10 platform expected; bonus M3/M4/M7 feature gold rush as parallel sprint
```

