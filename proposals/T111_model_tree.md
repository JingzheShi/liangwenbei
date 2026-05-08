# T111 — Pure-Tree Model Tricks (No NN)

**Constraint axis:** model only (features fixed at T68 schemeP 359-d). No NN, no Transformer, no RNN.
**Reference targets:** iter_015 (+40.09 LOSO), iter_016 candidate (+44.72), T99 single LGB Huber +40.79, T99 single CB Huber +40.24.
**Leaderboard intel:** top is **CPU-based and runs in minutes** → pure tree single-model + clean training likely beats our 4-way ensemble.

---

## Part 1 — Diagnosis: where is the headroom in tree-only land?

### What worked in T99 (LGB regression evolution)

| variant | DE LOSO | Δ vs prior |
|---|---:|---:|
| LGB L2 5-seed (T75) | +36.23 | baseline |
| LGB MAE 1-seed | +34.05 (1-seed) | — |
| LGB Huber α=2e-3 1-seed | +38.18 | — |
| **LGB Huber α=1e-3 5-seed (T99)** | **+40.79** | **+4.56 over T75** |
| CB Huber α=1e-3 5-seed | +40.24 | parallel jump |

The single biggest model-axis lever was loss shape: L2 → Huber. Everything else (regularization, learning rate, num_leaves, augmentation) was kept identical.

### Where there is unused headroom (model-only)

**A. The Huber α optimum is probably below 1e-3.**
- Single-seed α=5e-4 gave **+34.65 raw PnL@k=1**, vs α=1e-3 **+34.30** (+0.35 raw, untested DE).
- α=1.5e-3 gave +34.15 raw.
- T99 stopped at 1e-3 because it was the first 5-seed run; the α scan was 1-seed only.
- Estimated 5-seed DE LOSO at α=5e-4: **+41 to +42** (extrapolating +0.35 raw → ~+1 DE).

**B. The 5 seeds are not really diverse — they are correlated >0.9 in T99.**
- Cross-corr matrix in T99 REPORT shows LGB_Huber/CB_Huber correlation 0.95.
- Within-LGB 5-seed corr almost certainly higher (same loss, same data, same feature set; only `feature_fraction`, `bagging_fraction`, `num_leaves` and `lambda_l2` differ).
- Real "ensemble effect" in 5-seed LGB Huber is probably ~+0.5 LOSO over single seed (single best seed = 13 at +31.30 raw, mean = +30.20, ensemble = +40.79 → much of the lift is just averaging out noise, not real diversity).
- Pseudo-ensemble — if we want real model diversity within tree family, we need **structurally different boosting** (DART, GOSS, ordered, linear-leaf, ExtraTrees) rather than 5 hyperparam tweaks.

**C. We early-stop on val 76-79 (4 dates, ~5% of train). best_iter is 70-124 → very short trees.**
- Means we are leaving training data (val) on the table at final eval.
- Standard Kaggle trick: refit on 0-95 (train+val) with fixed `best_iter × ratio_data_increase` and **no early-stopping**.
- 25% more train data, no validation contamination by selection bias on a 4-date val.

**D. Boosting algorithm space largely unexplored at the new (Huber, schemeP) operating point.**
- DART tested only at horizon=10 with old SchemeC features and L2 loss (T50 stub, never finished).
- GOSS never tried.
- CatBoost ordered tried only with multiclass classification (T49, blocked by GPU compatibility), never with Huber regression.
- XGBoost tried only at h=10 multiclass (T18), never with Huber/pseudohuber regression on h=60.
- ExtraTrees / linear-leaf / NGBoost / GBDT-PL never touched.

**E. Hyperparameter sweep is shallow.**
- T36 Optuna study existed but only ran trial 0 then stopped.
- min_data_in_leaf fixed at 100; never tried 1000/3000 (strong reg, longer trees).
- max_bin = default 255; never swept.
- bagging_freq = 5 fixed.
- Tree depth × num_leaves grid never properly mapped.

### Net diagnosis

The LGB-Huber-5-seed +40.79 is mostly a **single-seed +30.20 noise-averaged**, not a real ensemble. **Two orthogonal levers untouched:**
1. **Loss shape** — Huber α below 1e-3, or pseudohuber, or quantile-pair, or NGBoost's likelihood.
2. **Tree-structure diversity** — DART / GOSS / linear-leaf / ordered / ExtraTrees as **structurally different** ensemble members (not just hyperparam tweaks of GBDT).

If leaderboard top runs in minutes on CPU, they likely use **one well-trained tree model**, not a 5-seed ensemble. The "well-trained" axis is what we have under-explored.

---

## Part 2 — 10 tree-model tricks

For each: 1-line description / config / why basic / why disruptive / cost / estimated LOSO improvement / risk.

### Trick 1: Huber α 5-seed sweep (α ∈ {3e-4, 5e-4, 7e-4, 1e-3, 1.5e-3})

- **Description:** Re-do the T99 5-seed LGB+CB Huber training but at 4 α values; pick the one with best DE LOSO 5-seed average.
- **Config:** `objective='huber', alpha=α` for LGB; `loss_function='Huber:delta=α'` for CB.
- **Basic:** literally one parameter; no code change beyond CLI flag.
- **Disruptive:** T99 single-seed scan suggests α=5e-4 might be **above** α=1e-3 by ~0.35 raw → ~+1 LOSO on 5-seed; we may be sitting on a free +1 right now.
- **Cost:** 4 αs × 5 seeds × 2 (LGB+CB) = 40 trains × ~12s GPU each = ~10 min compute + DE eval per α (~30s × 4) = ~12 min total.
- **Estimated LOSO Δ:** **+0.5 to +1.5** over T99 +40.79 → ~+41-42 standalone, propagating ~+0.5 to iter_016 ensemble.
- **Risk:** very low. The α surface is smooth and the local optimum is well-bracketed by the existing scan (5e-4, 1e-3, 1.5e-3 all known).

### Trick 2: Final-train on full data 0-95 with fixed best_iter (no early-stop)

- **Description:** After Trick 1 picks best α, retrain final model on dates **0-95** (train + val merged) with `num_boost_round = best_iter × 1.06` and **no early stopping**.
- **Config:** drop `valid_sets`, drop `lgb.early_stopping`, set `num_boost_round=int(best_iter * 1.06)` per seed.
- **Basic:** standard Kaggle/competition closing trick. ~5 lines of code change.
- **Disruptive:** current training uses 0-75 only (best_iter ~100); refitting on 0-95 (full available pre-test data) gives **+25% more training data with no leakage** — the val 76-79 was only used for early-stopping anchor, not for any feature engineering or threshold tuning. We should be able to lift the standalone single-seed +30 raw → ~+31-32, and the 5-seed ensemble +40.79 → ~+41-42.
- **Cost:** 5-seed × 2 (LGB+CB) = 10 final trains, ~12s each = 2 min. Reuses all eval scripts.
- **Estimated LOSO Δ:** **+0.5 to +1.5** standalone; propagates to ~+0.5 in 4-way ensemble.
- **Risk:** medium. The 1.06 ratio is heuristic; real best_iter at +25% data could be +20-30% higher. Mitigation: do a quick 1-seed sanity check at best_iter × {1.0, 1.06, 1.15, 1.25} to find the sweet spot, then lock for all 5 seeds. Also: the 4 dates in val 76-79 are the **closest-in-time to test 96-119**, so by using them for training (rather than early-stop selection) we trade selection noise for distribution-relevance training signal — strong positive expected value.

### Trick 3: LightGBM linear-leaf (`linear_tree=True`)

- **Description:** LightGBM's `linear_tree` option fits a linear regression in each leaf (like model-tree / GBDT-PL) instead of a constant. Same boosting logic, just smooth leaves.
- **Config:** `linear_tree=True`, `linear_lambda=1.0` (regularizes the leaf linear model).
- **Basic:** one parameter flip.
- **Disruptive:** Constant leaves are a step-function approximator; linear leaves let each leaf express a continuous local trend. For a regression target whose mean is dominated by smooth microstructure (spread + imbalance + flow direction), a linear leaf model captures **gradient inside the leaf** — exactly the small-signal regime where Huber's gradient cap gave us +4.5. **First-time linear-leaf trial in this codebase.**
- **Cost:** same as T99 LGB Huber (~12s/seed × 5 = 1 min). Predictor.py needs zero change (LGB serializes linear-leaf models in the same `.txt` format).
- **Estimated LOSO Δ:** **+0.5 to +2.0** over T99 LGB Huber. Linear-leaf typically gives +0.5-2% on smooth regression tasks. With the 359-d schemeP feature set having many smooth (spread, mid-rate, imbalance) features, expected upper end.
- **Risk:** medium. Linear-leaf can overfit on small leaves; need `min_data_in_leaf >= 200` and `linear_lambda >= 1`. Falls back gracefully to constant if not enough data per leaf.

### Trick 4: DART boosting + Huber loss + schemeP

- **Description:** Replace `boosting_type='gbdt'` with `boosting_type='dart'` in LightGBM; keep Huber α=1e-3.
- **Config:** `boosting_type='dart', drop_rate=0.10, max_drop=50, skip_drop=0.5, uniform_drop=False, num_boost_round=1500` (DART has no early stopping).
- **Basic:** one parameter change. T50 attempted DART before but at h=10, old features, L2 loss — never tested at the (h60, schemeP, Huber) operating point.
- **Disruptive:** DART forces every new tree to compensate for a randomly dropped subset of existing trees → builds **redundant** representation, which is the opposite of GBDT's leverage-the-marginal-tree behavior. R32+R35 hypothesis: DART trees have lower variance under sym distribution shift (the LOSO setting). Combined with Huber's loss-shape robustness, DART+Huber should compose **multiplicatively** for OOD trade decisions.
- **Cost:** ~3 min/seed × 5 seeds = 15 min (DART is slower than GBDT because no early-stop). Single-precision GPU keeps memory low.
- **Estimated LOSO Δ:** **+0.5 to +2.0** vs T99. Particularly likely to help the per-sym worst case (sym 0/1/2) where current LGB_Huber is at +4.0-4.7. DART's redundancy property targets exactly the per-sym-divergence failure mode.
- **Risk:** medium. DART num_boost_round is fixed (no early-stop), so cost is calibrated by training the budget rather than the model — could undertrain or overtrain. Mitigation: 1-seed pilot at num_boost_round ∈ {800, 1500, 2500} before full sweep.

### Trick 5: GOSS (Gradient-based One-Side Sampling)

- **Description:** LightGBM `boosting_type='goss'` keeps top-`top_rate` % rows by gradient magnitude every tree, randomly samples `other_rate` from the rest. Trains on subset → faster + de-emphasizes already-fit examples.
- **Config:** `boosting_type='goss', top_rate=0.2, other_rate=0.1` (default).
- **Basic:** one parameter; LightGBM-native, zero install cost.
- **Disruptive:** GOSS focuses on **high-gradient samples** (the ones the current ensemble is most wrong about). Combined with Huber's gradient cap at α, GOSS will keep samples with |y| just below α (the boundary cases that matter for the EV gate) and downsample the deep-tail outliers (which Huber is already gradient-capped on). The pair is gradient-aligned with the EV decision surface in a way GBDT+L2 isn't.
- **Cost:** ~10s/seed × 5 = 1 min, even faster than GBDT due to subsampling.
- **Estimated LOSO Δ:** **+0.3 to +1.5** vs T99 LGB Huber. GOSS+Huber is theoretically well-aligned but not always empirically dominant. Worth a 5-seed run.
- **Risk:** low-medium. GOSS occasionally underperforms on low-noise data; if the worst-gradient samples are pure noise, GOSS wastes capacity on them. Falls back to GBDT-equivalent at top_rate=1.0.

### Trick 6: Aggressive hyperparam diversity 5-seed (real ensemble, not pseudo)

- **Description:** Replace the 5 seed configs in T99 (which differ only in `feature_fraction`, `bagging_fraction`, `num_leaves`, `lambda_l2`) with **5 structurally-different** configs:
  - seed 1: depth=4, num_leaves=15, min_data=1000, lambda_l2=20 — shallow regularized trees
  - seed 7: depth=10, num_leaves=255, min_data=50, lambda_l2=0.1 — deep tall trees
  - seed 13: extra_trees=True, feature_fraction_bynode=0.5 — randomized splits
  - seed 42: linear_tree=True, num_leaves=63, min_data=300 — linear leaves
  - seed 100: dart, num_boost_round=1500 — dart redundancy
- **Config:** see above; same Huber α=1e-3 across all.
- **Basic:** five separate trains with five different configs. Each config is a known-good single setup.
- **Disruptive:** current 5-seed cross-corr (per T99) is ~0.95 within LGB Huber. Real-diversity ensemble could drop intra-corr to ~0.7-0.8, **doubling the ensemble effect**. T99's ensemble lift over single seed was ~+10 PnL (raw). Real-diversity could double to ~+20.
- **Cost:** ~5 trains × varying time = ~10 min total. DE eval same as before.
- **Estimated LOSO Δ:** **+1.0 to +3.0** vs T99 5-seed Huber +40.79. Highest-EV trick on this list if it works.
- **Risk:** medium-high. Some configs (deep trees with min_data=50) may overfit and drag the ensemble down. Mitigation: evaluate each seed standalone first; drop any seed with single-seed DE LOSO < +30. Also: validate ensemble corr matrix → confirm spread of 0.6-0.85 (not all clumped at 0.9+).

### Trick 7: CatBoost ordered boosting + Huber regression (CPU)

- **Description:** Use CatBoost's ordered boosting mode with `loss_function='Huber:delta=0.001'`. Ordered boosting was T49's target but was blocked because GPU+ordered+multiclass is unsupported. **Regression doesn't have that constraint**, so ordered+Huber on CPU should work.
- **Config:** `boosting_type='Ordered', loss_function='Huber:delta=0.001', task_type='CPU', depth=6, l2_leaf_reg=3, iterations=2000, od_type='Iter'`.
- **Basic:** flag flip + CPU. Each train takes ~5-10 min on CPU but Predictor.py is CPU anyway.
- **Disruptive:** Ordered boosting eliminates the **target leakage** that plain GBDT introduces (each tree uses leaf statistics computed from samples it was trained on). On a regression target with std ≈ 2e-3 (noise-dominated), this leakage manifests as tiny systematic bias toward seen examples. Ordered explicitly addresses this. **Never tried in regression form in this codebase.**
- **Cost:** 5-seed × ~7 min CPU each = 35 min serial, or 5-7 min parallel on 5 cores. Well under 1 hour.
- **Estimated LOSO Δ:** **+0.5 to +2.0** standalone vs CB Huber Plain (T99 +40.24 → ~+41-42).
- **Risk:** low-medium. Ordered is mathematically more correct; the only risk is computational and the per-seed train time.

### Trick 8: XGBoost regression with Huber (`reg:pseudohubererror`) and per-feature monotone constraints

- **Description:** XGBoost has `reg:pseudohubererror` (parameterized by `huber_slope`). Train the same 5-seed regression on schemeP features. Optionally add monotone constraints on intuitive features (e.g., `imbalance` should monotonically influence pred — bid-imbalance → up-pred).
- **Config:** `objective='reg:pseudohubererror', huber_slope=0.001, tree_method='hist', device='cuda', max_depth=7`.
- **Basic:** XGBoost was tried only at h=10 multiclass (T18). Pseudohuber is a different loss formulation than LGB's Huber (smoother near zero); empirically often beats LGB Huber by 0.3-0.5 on regression tasks.
- **Disruptive:** XGBoost's tree algorithm is structurally different from LightGBM's (level-wise vs leaf-wise; histogram-based but different binning). At the **same loss family**, XGB and LGB typically have ~0.85 correlation — adding XGB Huber to the LGB+CB Huber pair gives a third tree-family member with **lower marginal correlation** than LGB↔CB (which are 0.95).
- **Cost:** ~15s/seed × 5 = 1-2 min on GPU.
- **Estimated LOSO Δ:** **+0.3 to +1.0** vs CB Huber as third stream. As ensemble add: ~+0.3 LOSO on iter_016 (4-way → 5-way).
- **Risk:** low. Worst case it's a redundant member and we drop it.

### Trick 9: NGBoost (predicts mean + variance)

- **Description:** NGBoost is a gradient-boosting variant that outputs a **distributional prediction** (e.g., Normal with both mean μ and scale σ predicted). Use μ as point estimate, but **σ as a confidence gate**: if σ is high, decline to trade even if |μ| > threshold.
- **Config:** `NGBRegressor(Dist=Normal, n_estimators=500, learning_rate=0.05, minibatch_frac=0.5)`. CPU-only (no GPU support).
- **Basic:** one library call. NGBoost is a single tree-based model with a different update rule.
- **Disruptive:** Our current decision rule `pred > thr_up → action=2` ignores prediction confidence entirely. A high-σ-high-μ prediction is **just as likely to be wrong as right**; a low-σ-low-μ prediction is **highly reliable to be small**. Adding σ-gating to the EV rule could reduce trades in the noise-dominated tail (where Huber's gradient cap can't reach). Decision rule: action=2 iff `μ - λ·σ > thr_up`, similar for short.
- **Cost:** CPU train ~30 min for 1.4M rows (NGBoost is slow). 1-seed pilot first.
- **Estimated LOSO Δ:** **+0.5 to +2.0** if the σ-gating is well-calibrated. **0 to +0.5** if σ is just monotone with |y| (then it's a redundant flag).
- **Risk:** medium-high. NGBoost trains on CPU, ~30 min/seed. If it doesn't beat LGB Huber standalone, ensemble lift is uncertain. Mitigation: pilot at 1 seed, decide on 5-seed.

### Trick 10: ExtraTrees auxiliary stream (sklearn) + LGB Huber stacking

- **Description:** Train a single sklearn `ExtraTreesRegressor` with Huber-equivalent loss (sklearn ET uses MSE, but we can apply the same class-balanced sample weighting). Use it as a parallel stream alongside LGB+CB Huber, weighted at 0.3-0.5 in the linear ensemble.
- **Config:** `ExtraTreesRegressor(n_estimators=500, max_depth=12, min_samples_leaf=200, max_features=0.5, n_jobs=-1)`. CPU-only.
- **Basic:** one sklearn line. ExtraTrees uses **completely random splits** (no information-gain optimization) — opposite of GBDT's greedy splits.
- **Disruptive:** ExtraTrees is **structurally orthogonal** to gradient boosting. Each tree is an unbiased estimator (no overfit-residuals dynamic), and the ensemble averages many such unbiased trees. ET typically has corr ~0.6-0.7 with GBDT — much lower than LGB↔CB (0.95). Adding ET at small weight (0.2-0.4) can give marginal lift on noisy regression tasks like ours. **Never tried in this codebase.**
- **Cost:** 1 train ~5-10 min CPU for 500 trees on 1.4M × 359. Single seed enough.
- **Estimated LOSO Δ:** **+0.2 to +1.0** as small-weight ensemble add. Lower-end if ET's noise is not orthogonal; higher-end if it captures coarse threshold behavior that GBDT smooths out.
- **Risk:** low. Worst case we set weight=0 in ensemble.

---

## Part 3 — Top 3 recommendations

Selecting on the criterion: **most basic to implement, most likely to give >+0.5 LOSO with low-to-medium risk**, and aligned with the leaderboard intel "single tree model + good training in minutes".

### #1: Huber α 5-seed sweep + final-train on full data (Tricks 1+2 combined)

**Why combine these two:** they are independent levers (Trick 1 changes loss surface, Trick 2 changes training data), and they are both essentially free (~15 min compute for the pair). The combined effect is roughly additive — if Trick 1 alone gives +1 and Trick 2 alone gives +1, together they give ~+1.5-2.

- **Standalone LOSO estimate:** +41 to +43 (vs T99 LGB Huber +40.79).
- **Iter_016 ensemble propagation:** +0.5 to +1.0 (the LGB and CB streams both improve, but ensemble has flat optimum).
- **Why first:** lowest implementation cost (CLI flag + ~5 lines of code), highest-confidence positive outcome, no risk of regression.
- **Implementation:** add `--alpha` and `--final-full-data` flags to `train_lgb.py` and `train_cb.py`; run a 1-seed α scan at α ∈ {3e-4, 5e-4, 7e-4, 1e-3, 1.5e-3} on V4 split first, then 5-seed at the winning α with refit on 0-95.

### #2: Real-diversity 5-seed ensemble (Trick 6) + DART boosting (Trick 4)

**Why combine:** Trick 6 inserts DART as one of its 5 members anyway, so they overlap. The pair is the **real ensemble** alternative to T99's pseudo-ensemble. This is the highest-EV single change if our diagnosis ("the 5-seed ensemble is mostly noise-averaging") is right.

- **Standalone LOSO estimate:** +42 to +44 (T99 LGB Huber 5-seed = +40.79; real-diverse 5-seed could capture an additional +1-3 from genuine ensemble lift).
- **Iter_016 ensemble propagation:** if real-diverse LGB Huber 5-seed = +43, weighting it at 1.0 in the 4-way (T87 + 0.7×CB_Huber + 1.0×LGB_Huber + 1.0×T95) lifts to ~+45.5-46.
- **Why second:** higher complexity + higher payoff than #1. The risk is tunable — we evaluate each seed standalone before aggregating, so any bad config gets dropped.
- **Implementation:** parameterize `train_lgb.py` to accept `--config-name` mapping to (depth, num_leaves, min_data, lambda_l2, boosting_type, linear_tree, extra_trees) tuple; run 5 configs in parallel; eval cross-corr matrix; aggregate the surviving configs.

### #3: LightGBM linear-leaf with Huber (Trick 3)

**Why third:** highest-novelty single trick. Linear leaves transform LightGBM from a step-function approximator into a piecewise-linear approximator — a categorical change in expressive power, not just a hyperparameter tweak. The schemeP 359-d feature set has many smooth features (spread/mid-rate/imbalance/flow-rates), exactly the regime where linear leaves help most.

- **Standalone LOSO estimate:** +41 to +43 vs T99 LGB Huber +40.79.
- **Iter_016 ensemble propagation:** could replace LGB Huber stream entirely (linear-leaf is a strict superset in expressive power). Expect ~+0.5-1.0 to ensemble.
- **Why third:** higher implementation risk than #1 (need to tune `linear_lambda` and `min_data_in_leaf`; small leaves can overfit). But the upside is a categorically different model.
- **Implementation:** add `--linear-tree` flag to `train_lgb.py`; pilot 1-seed at `linear_lambda ∈ {0.1, 1, 10}, min_data_in_leaf ∈ {100, 300}`; pick best for 5-seed run.

### Notes on what was deprioritized (and why)

- **CatBoost ordered (Trick 7):** strong theoretical motivation but slow CPU training (35 min/run); deprioritized in favor of cheaper wins.
- **XGBoost pseudohuber (Trick 8):** likely a redundant member with LGB Huber + CB Huber (corr probably ~0.85+). Worth including if Trick 6 gives less diversity than expected.
- **GOSS (Trick 5):** lower expected lift than DART (Trick 4) for similar cost; left as a fallback.
- **NGBoost (Trick 9):** slow CPU training and uncertain σ-calibration. Higher-risk; revisit if first three exhaust.
- **ExtraTrees (Trick 10):** small expected lift; worth bundling with #2 as one of the 5 diverse seeds rather than as a standalone.

---

## Risks and constraints

- **Compliance:** all 10 tricks are sym-agnostic and stateless (no per-call buffers, no per-sym branching). Only feature input is the schemeP cache; predictions feed the existing global EV-gate. ✅
- **Predictor.py impact:** Tricks 1, 2, 3, 4, 5, 6, 8 — `.txt`/`.cbm`/`.ubj` model files; existing `Predictor.py` loads them with no change. Tricks 7 (CatBoost ordered) — same `.cbm` format. Tricks 9 (NGBoost) and 10 (ExtraTrees) — pickle files, need a new loader branch.
- **Submission size:** all model files combined are <50 MB; well under 2 GB.
- **Inference time:** linear-leaf LGB and DART add ~5-10% to predict-batch time; well within 1024-batch < 3s budget.

---

```
RESULT: task=model_tree top_3=[huber_alpha_sweep+full_refit, real_diversity_5seed+dart, linear_leaf_lgb] notes=10 pure-tree tricks proposed. Top diagnosis: T99 5-seed Huber ensemble is pseudo-ensemble (intra-corr ~0.95), real headroom in (a) Huber α below 1e-3 (b) full-data refit 0-95 (c) structural ensemble diversity via DART/linear-leaf/extra-trees. Top 3 estimated +41-44 LOSO standalone; ~+1-2 to iter_016 ensemble. Aligned with "CPU-based top leaderboard runs in minutes" intel — single well-trained tree model > our 4-way ensemble.
```
