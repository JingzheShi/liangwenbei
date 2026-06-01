# QREVIEW: Inference / Predictor Submission Package

Scope: `final_submission_code/04_build_pkg/{Predictor.py, build_pkg.py, config.json, thresholds.json, requirements.txt}`. Whiteboard read. No README/PROGRESS consulted.

---

## A. Thresholds & Gating Logic

**Q1.** `thresholds.json:46` h=60 `thr_up = 0.0003`. Where does **3e-4** come from? Why not 2e-4 (which is the *fallback default* in `_gate_with_band`)? Was this tuned on validation? Held-out test? Which?

**Q2.** `thresholds.json:47` h=60 `thr_dn = 0.000216`. Why exactly **2.16e-4**? Ratio `0.0003/0.000216 = 1.3889`. Suspicious neat ratio. Was this a *fitted* number or `0.0003 / 1.3889 = 0.000216` derived from a chosen asymmetry?

**Q3.** Every horizon (h=5,10,20,40,60) has the *exact same* `thr_up/thr_dn = 1.3889`. Confirm: this asymmetry is the same across all horizons. Why is the asymmetry horizon-invariant — physical reason?

**Q4.** Asymmetry direction: `thr_dn < thr_up`. Smaller threshold for SHORT → model needs *less* negative prediction to short than positive prediction to long. Does this imply the model has a *positive bias*? If yes, why not de-bias the model and use symmetric thresholds?

**Q5.** Asymmetry could also be driven by market structure (short fills harder, etc.) — but for a *prediction* threshold (not execution cost), what justifies asymmetric gating?

**Q6.** `thresholds.json:7,16,25,34` short-horizon thresholds = `h=60 * sqrt(H/60)`. Where is this **sqrt(t) scaling** derived? It's Brownian-motion scaling for *price moves*, but you're using a 60-tick *predicted return* and gating it as if it were a 5/10/20/40-tick return. These are not the same random variable.

**Q7.** Specifically: at h=5, you compare a `pred_dmid_60` (predicted 60-tick return) to a threshold of `3e-4 * sqrt(5/60) = 8.66e-5`. Why is the threshold scaled DOWN for h=5 — wouldn't this trigger many MORE trades, when the predicted quantity is unchanged? What economic intuition supports this?

**Q8.** `_doc` says "Platform score = max(per-horizon PnL)" — is this verified against the platform spec? If the platform actually *sums* per-horizon PnL (or averages, or weights), adding noisier short-horizon actions could *reduce* total score, not preserve it.

**Q9.** `_doc` claims "adding shorter horizons can only stay same or improve". This assumes max. If max is wrong, the design is wrong. Where is `max` evidence?

**Q10.** Per-sym `beta`: `{"0":0.1, "1":0.4, "2":0.3, "3":0.0, "4":0.0}`. Why is sym 1 = 0.4 (highest)? Was this from a per-sym validation calibration? Show your work.

**Q11.** Sym 3 and 4 have `beta=0.0`. **No abstain widening** for these syms. Why? Are their model predictions known to be well-calibrated? Or did you not bother to tune them?

**Q12.** Per-sym `sigma`: `{"0":0.0002403901, ...}` — 10 significant digits. Strongly suggests these are *computed* from validation residuals. What was the data window? How many samples per sym? If small N, sigma is noisy and `beta*sigma` is noisy too.

**Q13.** Are these sigmas std of `pred_dmid` (signal magnitude), std of residual (`pred - true`), or something else? Pick one and explain.

**Q14.** `default_beta_for_ood = 0.16` — how chosen? Average of (0.1, 0.4, 0.3, 0.0, 0.0) = 0.16. So it's the *mean*? Why mean and not median (0.1) or max (0.4 — conservative)?

**Q15.** `default_sigma_for_ood = 0.0003998317` — what is this number? Mean of per-sym sigmas? Why so many digits?

**Q16.** OOD band = `0.16 * 3.998e-4 = 6.4e-5`. For sym=1, band = `0.4 * 4.71e-4 = 1.88e-4`. The OOD gets a **smaller** abstain band than a high-band known sym. This means OOD syms are traded MORE AGGRESSIVELY than uncertain known syms. Is this intentional? Default should arguably be MAX, not mean.

**Q17.** `_gate_with_band` uses `>` and `<` strict inequalities. `pred == thr` → flat. Why strict? Was training labels' boundary also strict?

**Q18.** Two gate functions: `_gate_with_band` (instance) and `_ev_gate_predict` (static). Why both? `_gate_with_band` with `band_per_row=0` reduces to `_ev_gate_predict`. The dual implementation invites drift between them.

**Q19.** `band_per_row` is computed even when `_cw_enabled=False`? Let me look — line 326: `band_per_row = np.zeros(B, dtype=np.float64)` only if disabled. So when disabled, we still allocate then ignore. Why not skip allocation when disabled?

**Q20.** `np.zeros(B, dtype=np.float64)` for an unused array — small cost but pointless. Why allocate?

**Q21.** Code path when `_cw_enabled=True`: ALL 5 horizons use the SAME band (looked up from last-tick sym). Conformal abstention is therefore horizon-independent. Was conformal calibrated per horizon? If yes, why discarded?

**Q22.** If you trained conformal on h=60 residuals, applying same `beta*sigma` to h=5 (which uses h=60 predictions but at scaled thresholds) is questionable. The band magnitude is fixed but thresholds shrink. So at h=5, the band may be LARGER than the threshold itself.

**Q23.** Concretely: at h=5, thr_up = 8.66e-5, sym=1 band = 1.88e-4. The band is **2.17× the threshold**. So `effective_thr_up = 8.66e-5 + 1.88e-4 = 2.75e-4`, basically the *unscaled* h=60 threshold. The sqrt scaling is effectively neutered for high-band syms at short horizons. Was this analyzed?

**Q24.** For sym 3,4 (band=0), the sqrt scaling is fully effective. So *different syms get different effective scalings* at short horizons. Is this intentional or accidental side effect?

---

## B. Ensemble Combination

**Q25.** `w_nn = 1.0, w_lgb = 1.5` (per-horizon, but they all share src=60). Why **1.5×** LGB weight? Where validated? On what split?

**Q26.** Why LGB > NN weight? If NN ensemble of 50 is "better" or "additive", you'd expect symmetric. Was 1.5 found by grid search? Tested values?

**Q27.** Effective weight share: `w_lgb / (w_lgb + w_nn) = 1.5/2.5 = 60% LGB, 40% NN`. Was this tested vs 50/50, vs 70/30?

**Q28.** Each horizon redundantly stores `w_nn=1.0, w_lgb=1.5` in config (even h=5/10/20/40 that share with 60). The weights from h=5/10/20/40 are **dead config** — the code in `predict()` looks up `self._weights[src_H]`. Why duplicate? Easy source of bugs if someone edits the "wrong" copy.

**Q29.** Ensemble math: `pred = (1.0*nn_mean + 1.5*lgb_mean) / 2.5`. Linear pool. Why not log-pool? Why not rank-averaging?

**Q30.** Why mean over 50 NNs and mean over 50 LGBs (i.e., compute each family's mean then weight), instead of treating all 100 models equally (with 100 weights)? Mathematically different when individual model variances differ across families.

**Q31.** Why 50 NNs and 50 LGBs (symmetric)? Was 100/0 tried? 0/100? 25/25? Why this exact count?

**Q32.** 50 seeds of LightGBM with identical hyperparameters likely have *very similar* predictions (boosting is highly deterministic given same data + similar seeds). What is the *empirical* correlation between the 50 LGBs? If ~0.99, the "ensemble" gives marginal benefit over 1 model.

**Q33.** Same Q for 50 NNs — what's avg pairwise corr between NN predictions? If high, ensembling is theatre.

**Q34.** `_ensemble_predict_lgb`: special-cases `len==1`, otherwise accumulates. Why not always `np.mean(np.stack([b.predict(X) for b in boosters]))`? Stack is more readable, no special case.

**Q35.** Per-booster `.predict()` returns float64 by default. Code casts to float32 then accumulates. Accumulating 50 float32 values loses precision. Why not accumulate in float64?

**Q36.** `predict()` line 348: `preds.append(self._ensemble_predict_lgb(lgbs, feats))`. 50× CPU LGB sequential. No threading/parallelism. LightGBM's own `predict` has internal parallelism via OMP but each call is one model. Could batch all 50 with `predict_proba` or stack? Why not?

**Q37.** `_BatchedMLPEnsemble`: parallelizes NN forward via `torch.bmm`. Good for NN. But LGB is left sequential. Why asymmetric optimization effort?

**Q38.** `_BatchedMLPEnsemble.__init__`: loads d0 once to read `in_dim, hidden, use_layernorm, keep_idx`. **Assumes all 50 NNs share these.** Where verified? What if seed-3's NPZ has different hidden dim? Code crashes mid-loop with cryptic shape error.

**Q39.** `self.clip = float(max(clips))` — uses **MAX clip across all 50 NNs**. But each NN was trained with its own clip. Applying max means some NNs see WIDER input range at inference than during training. Bias!

**Q40.** Why max and not min? Min would be most conservative (no NN sees data beyond what it saw in training). Max means all NNs MAY see data they never trained on. Was this consciously chosen?

**Q41.** Or — why aggregate at all? Use **per-NN clip**: `h[i] = clamp(h[i], -clips[i], clips[i])`. Trivial change, more faithful to training. Why simplified?

**Q42.** `target_scale` per-NN: stored, applied per-NN before mean. Different NNs trained with different target scales? Why? Was target scaled differently across seeds? Or all same and storage redundant?

**Q43.** `feat_mean / feat_std` per-NN (stored as `(N, in_dim)`). Different per NN means trained on different splits with different normalization stats. Or same? Stored redundantly?

**Q44.** Code line 158: `h = (h - feat_mean.unsqueeze(1)) / feat_std.unsqueeze(1)`. If `feat_std` has zeros (constant feature), NaN/Inf. Then line 159 replaces NaN with 0 but does NOT handle Inf. After clamp, becomes ±clip. Subtle bug?

**Q45.** Where is `feat_std` checked for zeros? If a feature was constant in some seed's training subset, `std=0`. Inference is silently broken for that NN.

**Q46.** Manual LayerNorm with `eps=1e-5`. Did training use exactly 1e-5? PyTorch's `F.layer_norm` default is 1e-5 too — but trained model checkpoints might have used a custom eps. Where verified?

**Q47.** `F.gelu(h, approximate="tanh")`. Why tanh approximation? Did training use `approximate="tanh"` or the exact GELU? Mismatch → predictions drift. Module docstring claims ~1e-6 typical, but where measured?

**Q48.** `h = h.unsqueeze(0).expand(self.N, -1, -1).contiguous()` — `expand` is view, `contiguous()` materializes. Memory cost: N×B×in_dim. For B=1024, N=50, in_dim=359: 73 MB allocation. Why materialize? `bmm` requires contiguous strides on broadcast axis.

**Q49.** `torch.where(torch.isnan(h), torch.zeros_like(h), h)` — only handles NaN, not Inf. Inf will survive normalization and propagate to GELU output. Why only NaN?

**Q50.** `out = out / self.target_scale` BEFORE `mean(0)`. So each NN's prediction is scaled back, then averaged. If all NNs use same scale, ordering doesn't matter. If different, ordering matters and current order is correct. Verify all NNs trained with the **same** target_scale.

**Q51.** `out.mean(0)` — equal-weight average over 50 NNs. No down-weighting of underperformers. Was per-NN validation done to find weights? If yes, where applied?

**Q52.** `predict_mean` uses `keep_idx` if `X_in.shape[1] != in_dim`. So either feats matches (no slice) or doesn't (slice). What if feats has wrong COLUMNS but right SHAPE? Silently uses garbage features. No column-name check.

**Q53.** `predict()` line 357: `pred_dmid = (stacked * ws_arr).sum(axis=0) / ws_arr.sum()`. ws_arr is float32, stacked is float32. Sum of float32 → potential precision loss with extreme values. Acceptable for this scale.

**Q54.** `pred_cache: Dict[int, np.ndarray] = {}` — sharing across horizons is correct (cache hit on src_H). But what if `share_with` cycles? E.g., h=5 shares with h=60, and h=60 shares with h=5? Code would compute h=60 first (active), cache it. Then h=5 sees src=60, hit. Fine. But what if h=60.share_with=70 (non-existent)? `self._lgb_lists.get(70)` returns None, `self._nn_batched.get(70)` returns None, `preds=[]`, `continue`. Silent failure → flat actions.

**Q55.** `self._weights.get(src_H, (1.0, 1.0))` — fallback to (1,1). So if `share_with` points to a horizon without weights configured, defaults to symmetric. Surprising silent behavior.

**Q56.** What if `w_lgb == 0` and only LGB available? `if lgbs and w_lgb > 0` skips LGB. preds list empty. continue. Horizon stays flat. Was this intentional?

**Q57.** What if `w_nn < 0`? Code says `if nn_ens is not None and w_nn > 0`, so negative weights skipped. But negative weights are not validated upstream. Silent.

---

## C. Feature Processing

**Q58.** `RAW_COLS_TRAIN_ORDER` (line 46-65) — hardcoded 154 column names. Why hardcoded? Why not derived from `config.json["feature"]` (minus sym)?

**Q59.** `assert len(RAW_COLS_TRAIN_ORDER) == 154` — assertion in module load. **Python `-O` flag disables asserts.** If platform runs with `-O`, this silently passes even if list is wrong length. Replace with explicit `if/raise`.

**Q60.** `config.json["feature"]` has 155 entries (154 + sym at end). `RAW_COLS_TRAIN_ORDER` has 154 (no sym). The relationship is implicit, undocumented. Why two sources of truth? What enforces they stay aligned?

**Q61.** Module-level `WINDOW = 100`. Hardcoded. What if platform sends window of 80? `X3d = np.empty((N, 100, K))`, then `df[cols].to_numpy()` returns (80, K), `X3d[n] = (80, K)` → shape mismatch crash. No graceful handling.

**Q62.** What if df has > 100 rows? Same crash. Why not slice `df.tail(100)` defensively?

**Q63.** What if df rows aren't time-ordered? `X3d[:, -1, :]` takes the LAST ROW assuming most-recent. But platform may not guarantee row ordering. Where verified?

**Q64.** `df[self._raw_feat_cols]` — if any column missing, raises KeyError. No try/except. So if platform omits a column, hard crash. Platform contract?

**Q65.** `to_numpy(dtype=np.float64, copy=False)` — `copy=False` is a *hint*, not a guarantee. Pandas may still copy if dtypes differ. Why force float64 at extraction? Memory waste.

**Q66.** `X3d` is float64 — memory hog. For B=1024, 100 ticks, 154 cols: 126 MB. Why float64 when downstream uses float32?

**Q67.** `for n, df in enumerate(batches): X3d[n] = df[self._raw_feat_cols].to_numpy(...)` — Python loop. Could use `pd.concat` + reshape for vectorization. Maybe negligible cost?

**Q68.** `raw_last = X3d[:, -1, :].astype(np.float32, copy=True)` — copy=True. Why copy when slicing creates a view that's then cast? Cast already creates new array.

**Q69.** **Why only `amount_delta` gets the `sign(v) * log1p(|v|)` transform?** Volume_delta is same family (delta of volume), can be very skewed. Why not it?

**Q70.** `np.sign(v) * np.log1p(np.abs(v))` — symmetric log. Where derived? Training used same transform? If training used `log(1+max(0,v))` for positive-only, sign-symmetric here is wrong.

**Q71.** `amount_delta` transform applied to `raw_last` AFTER float32 cast. So `np.log1p` operates on float32. Precision loss for large amounts?

**Q72.** Why transform AFTER concat with extras (well, before extras built into result)? Why not transform in the X3d (so extras see transformed amount)? Are extras computed on raw amount_delta?

**Q73.** `extras_full = self._ffb.compute_batch_features(X3d, self._col_idx)` — external function in fast_features_batch.py. Not reviewed here. What's its column count? Order? Is it stable across versions?

**Q74.** `np.where(np.isfinite(extras_full), extras_full, 0.0)` — replaces NaN, +Inf, -Inf with 0. Aggressive. A NaN often means "couldn't compute" — replacing with 0 makes it indistinguishable from "value is exactly 0". Different semantics!

**Q75.** Tree models handle NaN natively (treat as missing → take default direction). By zeroing NaN, you LOSE LightGBM's NaN handling. Was training data also zeroed? If not, train/test skew.

**Q76.** `extras_kept = extras_full[:, self._extra_keep_idx]` — fancy index, allocates new array. Cost OK.

**Q77.** Concatenation order: `np.concatenate([raw_last, extras_kept], axis=1)`. raw_last (154) first, extras_kept second. **Must match training order EXACTLY.** Where verified?

**Q78.** `FAIL_NAMES` — 11 specific feature names dropped: dualz_*, qrank_W100_spread*, qrank_W100_cumspread, kyle_lam_W{50,100}, roll_eff_spr_ratio_W100, liq_asym_top5_W5. What FAILED? Numerical stability? Sym-agnostic verification? Document the criterion.

**Q79.** `kyle_lam_W50` AND `kyle_lam_W100` both dropped — entire kyle_lam family failed? Or just W50 and W100?

**Q80.** `liq_asym_top5_W5` dropped (W5 window). Why this specific one and not other window sizes? Were W10/W20/W50/W100 versions tested?

**Q81.** Hardcoded `FAIL_NAMES` tuple in module — change requires code edit + redeploy. Why not config-driven?

**Q82.** What if FAIL_NAMES contains a name NOT in `self._extra_names`? Code: `[i for i, n in enumerate(self._extra_names) if n not in FAIL_NAMES]`. The non-match is silently ignored. So typo in FAIL_NAMES gives no error.

**Q83.** `self._extra_keep_idx` computed from FIRST call to `all_feature_names()`. If `all_feature_names()` ever returns variable-length list, downstream breaks.

**Q84.** `self._col_idx["midprice"] = self._raw_col_to_idx["midprice1"]` — aliases "midprice" → "midprice1". Why? Only used inside `compute_batch_features`. Implicit contract with ffb code.

**Q85.** Why alias `midprice` and not also `bid`, `ask`, `spread`, etc.? Only midprice gets special treatment. Inconsistent.

**Q86.** `self._amount_delta_idx = self._raw_feat_cols.index("amount_delta")`. The check `if self._amount_delta_idx >= 0` is always True (index always non-negative or ValueError raised). Dead defensive check.

**Q87.** Feature extraction is `O(B)` Python loop with pandas operations. For B=1024, dominated by ffb's batch_features. Is ffb truly batched (vectorized) or also looped?

**Q88.** Are there any features that depend on the last *N* ticks where N > 100? `kyle_lam_W100` is W=100, but is W relative to start of window? If W=100 in W=100 window, that's the whole window — only 1 sample. Statistic of 1 sample is degenerate (e.g., variance is 0).

**Q89.** What if a window has all-zero or constant features (e.g., during market open with no movement)? Stats degenerate. Some features would be NaN, then zeroed. Plausible silent corruption.

---

## D. Model File Format & Loading

**Q90.** NN weights stored as `.npz` (numpy archive), not `.pt` or `state_dict`. Why? Reason given is none. Trade-offs:
- `.npz` is dependency-free for inference (no torch needed to *load*).
- But you still need torch for `bmm`.
- `.npz` is verbose for many keys (one file per checkpoint, no nested structure).
- Why not `state_dict`?

**Q91.** Does the npz format separate `LN0_W, LN0_b` (LayerNorm params) per layer? Naming convention is `L{i}_W, L{i}_b, LN{i}_W, LN{i}_b, LF_W, LF_b`. Why suffix `F` for final, but `0,1,2..` for hidden? Why not `L{len(hidden)}_W` for final too?

**Q92.** Final layer (`LF_W, LF_b`) shape `(1, hidden_last)` and `(1,)`. So output dim is 1 (regression). Hardcoded everywhere. What if you want multi-task output?

**Q93.** `target_scale` stored as 1-element array `d["target_scale"][0]`. Why 1-element array? Why not scalar?

**Q94.** Similarly `clip[0]`, `use_layernorm[0]`, `in_dim[0]`. Why wrap scalars in arrays in npz? Numpy convention vs. clarity.

**Q95.** LGB models stored as `.txt` via `Booster.save_model()`. Why `.txt` and not `pickle`? `.txt` is forward-compatible across LightGBM versions, but `pickle` is fragile. **OK choice** — but what if a hyperparameter (e.g., `early_stopping`) is lost in `.txt` serialization?

**Q96.** Loading 50 LGB Boosters at `__init__`: `[lgb.Booster(model_file=p) for p in lgb_paths]`. Each `.txt` is parsed independently. RAM: 50 × (booster size). For 100-leaf, 500-iteration boosters: ~few MB each → ~100-300 MB total. Verify on platform.

**Q97.** Model loading is **eager** (in `__init__`). Total `__init__` time = sum of all loads. Could exceed platform init timeout. Lazy loading not used. Why not?

**Q98.** No retry/error handling on Booster load. If one `.txt` is corrupt, `__init__` crashes. No graceful skip.

**Q99.** Files loaded based on `seeds` list in config. Code: `if os.path.isfile(lp): lgb_paths.append(lp)` — silently skips missing files. So if 5 files missing, you get a 45-model ensemble silently. Was this intended? No min-count check.

**Q100.** Hardcoded filename pattern: `model_h{H}_seed{s}.txt`, `nn_h{H}_seed{s}.npz`. What if pattern changes? Code break.

**Q101.** No checksum/version field in npz. No way to detect "this npz was trained with old code".

**Q102.** No model schema version. If you change npz key names later, old/new mixing → crash.

**Q103.** Variable name `np_` (line 243): `np_ = os.path.join(here, f"nn_h{H}_seed{s}.npz")`. **`np_` shadows numpy alias if you weren't careful.** Trailing underscore distinguishes — but `npz_path` would be cleaner.

**Q104.** `np.load(npz_paths[0], allow_pickle=False)` — `allow_pickle=False` is good (safety). But same call later in loop. Consistent.

**Q105.** Each NN's npz contains feat_mean and feat_std of shape `(in_dim,)` — 50 copies of essentially the same data (if NNs share normalization). Storage cost: 50 × in_dim × 4 = ~70 KB. Negligible but redundant.

**Q106.** All `L{i}_W` for layer i are stacked to `(N, out, in)`. Memory cost per layer: N × out × in × 4 bytes. For layer 1 (256×359): 50 × 256 × 359 × 4 = 18 MB. Acceptable.

**Q107.** `to_t` helper uses `torch.from_numpy(np.stack(lst, 0)).to(device)` — np.stack copies, then transfer. Could `torch.tensor(np.stack(...))` be faster? Or use `torch.from_numpy(np.ascontiguousarray(np.stack(...)))`?

**Q108.** `_BatchedMLPEnsemble.__init__` doesn't verify that `d["L{i}_W"].shape == (hidden[i], hidden[i-1] or in_dim)` for each i. If a seed has wrong arch, mismatched shape causes torch.stack to fail with cryptic error.

**Q109.** What if `feat_mean` has different length across NNs? `np.stack` raises. Cryptic.

**Q110.** What if `keep_idx` differs across NNs? Code uses d0's keep_idx only. Other NNs' keep_idx silently ignored.

---

## E. config.json / build_pkg.py / requirements.txt

**Q111.** `config.json: python_version = "3.11"`. Why 3.11 specifically? Why not 3.12 (newer)? Platform pinned?

**Q112.** `config.json: batch = 1024`. What does this mean? Suggested batch size for platform? Maximum supported batch? Code doesn't enforce this anywhere. Where consumed?

**Q113.** `config.json: feature` ends with "sym" but `RAW_COLS_TRAIN_ORDER` doesn't include sym. Discrepancy is by design (sym is metadata) but undocumented in config.

**Q114.** `config.json: label = ["label_5", "label_10", "label_20", "label_40", "label_60"]`. 5 labels. Why these? Matches HORIZON_LIST. Why these specific horizons (powers + 60)?

**Q115.** No `version` field in config. Hard to evolve schema.

**Q116.** No `model_type` / `framework` field. Implicit.

**Q117.** `build_pkg.py`: `md5_file` function defined but **NEVER CALLED**. Dead code. Either should compute & store MD5s in manifest or remove.

**Q118.** `build_pkg.py: N_NN = 50, N_LGB = 50` hardcoded. If you want 100/100 next experiment, two places to edit (here + ensemble_seeds in JSON).

**Q119.** `check_models` prints missing but doesn't fail / exit. `build` proceeds even with 0 models. No `--strict` flag.

**Q120.** `shutil.rmtree(pkg_dir)` — **destructive!** If you accidentally pass `--pkg /` or `--pkg /home`, catastrophe. No safety check.

**Q121.** No backup before rmtree. Why no `pkg_dir.bak` for safety?

**Q122.** Static files list hardcoded: `["Predictor.py", "config.json", "fast_features.py", "fast_features_batch.py", "requirements.txt", "thresholds.json"]`. Where are these names validated against the actual filesystem layout?

**Q123.** Missing static file → prints ERROR but continues. Should `sys.exit(1)`. Otherwise package ships with missing files.

**Q124.** `shutil.copy2(src, dst)` preserves metadata (timestamps, perms). Useful, but if perms are wrong (000), copy succeeds and platform can't read. Should `chmod`.

**Q125.** No final `zip` step. User must run `cd pkg && zip -qr ../submission.zip .`. **Easy to forget**, and the zip would have wrong structure (with leading `./`?). Why not automate?

**Q126.** Manifest stores `nn_seeds, lgb_seeds, static_files` but NOT MD5s, version, build date, git SHA. Useless for traceability.

**Q127.** `argparse defaults: "./outputs/models"`, `"./outputs/pkg"` — relative paths. User must `cd` to right directory. Why not absolute defaults?

**Q128.** `os.makedirs(os.path.dirname(args.pkg) if os.path.dirname(args.pkg) else ".", exist_ok=True)` — handles bare filename edge case. But `args.pkg` is treated as a directory, and `dirname` gives parent. Why create parent? `rmtree`+`makedirs(pkg_dir)` handles it.

**Q129.** No sanity check: after build, attempt `Predictor()` on the pkg to verify it can construct. Trivial test, omitted.

**Q130.** No check that thresholds.json `share_with` references exist in active horizons.

**Q131.** No check that `ensemble_seeds` matches available `nn_*` and `model_*` files.

**Q132.** No check on file SIZE. A 0-byte npz file passes `isfile()` but fails on load.

**Q133.** `requirements.txt: --extra-index-url https://download.pytorch.org/whl/cpu`. This pulls torch from PyTorch's CPU wheel index. Platform pip must respect `--extra-index-url`. Some restricted environments strip this. Then platform installs default torch (possibly GPU), wasting download time.

**Q134.** `numpy==2.4.4` — numpy 2.x. Did anyone verify lightgbm 4.6.0 works with numpy 2.x? pandas 2.3.3 with numpy 2.x? scipy 1.17.1 with numpy 2.x?

**Q135.** `scipy==1.17.1` — let me check if this version even exists / makes sense. Predictor.py doesn't import scipy directly. Why required? Presumably `fast_features.py` / `fast_features_batch.py`.

**Q136.** Exact pinning (`==`) everywhere — brittle. If platform has slightly different version (`numpy==2.4.5`), install fails or downgrades, causing cascade.

**Q137.** No `pyarrow`. If platform passes Parquet-backed DataFrame, code may need pyarrow for fast access. Missing dep?

**Q138.** No `numexpr`, `bottleneck` — Pandas accelerators. Optional but often expected.

**Q139.** No `numba`. Does ffb use numba? If yes, missing dep.

**Q140.** `torch==2.5.1+cpu` — CPU-only build. But `Predictor.py` auto-selects CUDA if `torch.cuda.is_available()`. With `+cpu` build, `cuda.is_available()` returns False, so always CPU. Then why have CUDA branch at all? Confusing.

**Q141.** What if platform installs torch *without* the `+cpu` suffix (some pip configurations strip it)? Then CUDA may be detected on a GPU platform. Model loads to CUDA. Then platform expected CPU inference — surprise overhead.

**Q142.** No `lightgbm[gpu]` — CPU inference only. OK, but if platform had GPU and used GPU inference, would not work.

**Q143.** No pyversion constraint in requirements (e.g., `; python_version >= "3.11"`). Could install on wrong Python.

---

## F. Fault Tolerance / Edge Cases

**Q144.** `predict([])` returns `[]`. OK. But returns `[]` not `[[]]` — is this a valid platform output? Confirm contract.

**Q145.** What if a single df has wrong column dtype (e.g., `bid1` is string)? `.to_numpy(dtype=np.float64)` raises. No try/except.

**Q146.** What if df has NaN in raw columns? raw_last includes NaN. LGB handles NaN. NN: NaN → normalize → still NaN → zeroed. Different families see different treatment. Train/test consistency?

**Q147.** What if `sym` column is float (1.0)? `int(df["sym"].iloc[-1])` truncates. OK.

**Q148.** What if `sym` is string "0"? `int("0")` → 0. OK.

**Q149.** What if `sym` is None? `int(None)` raises TypeError. Caught. Default band used.

**Q150.** What if `sym` column has multiple values within window (e.g., during sym rollover)? Code uses `iloc[-1]`. Silent. Other rows ignored.

**Q151.** What if `batches` contains a non-DataFrame (e.g., dict, list)? `df[cols]` fails differently for dict vs. list. Not caught.

**Q152.** What if a `df` is empty (0 rows)? `df[cols].to_numpy()` → shape (0, K). `X3d[n] = ...` shape mismatch. Crash.

**Q153.** What if `df` index is not 0..99 but, e.g., a DatetimeIndex? `.to_numpy()` is index-agnostic. Should be fine.

**Q154.** What if input contains duplicate column names? `df[cols]` returns multiple matching columns. Shape mismatch downstream.

**Q155.** What if `predict()` is called with `len(batches) > 10000`? Memory: X3d float64 is 100 × 154 × 8 × 10000 = 12 GB. OOM.

**Q156.** Is there a memory guard / chunking? No. Platform should call with reasonable batch.

**Q157.** What if model file paths contain Unicode / spaces? `os.path.join(here, fname)` handles. `lgb.Booster(model_file=...)` may or may not. Untested.

**Q158.** What if `here = os.path.dirname(os.path.abspath(__file__))` returns wrong path because `Predictor.py` was imported via `imp.load_source` or similar? Then model files aren't found. Silent (0-model ensemble).

**Q159.** What if `thresholds.json` is corrupt (invalid JSON)? `json.load` raises in `__init__`. Hard crash, no fallback. Acceptable?

**Q160.** What if a horizon is missing required keys (no `h`, no `thr_up`)? `int(hcfg["h"])` raises KeyError. Crash. No graceful skip.

**Q161.** What if `ensemble_seeds` is missing in h=60 entry? `seeds = []`. No files loaded. Ensemble empty. predict returns all-flat. **Silent regression.**

**Q162.** What if `horizons` list is empty? `__init__` loads nothing. predict iterates over empty list → all output is `np.ones((B, 5))` (all flat). Silent.

**Q163.** What if `HORIZON_TO_IDX[H]` fails because hcfg has H=7 (not in (5,10,20,40,60))? KeyError. Crash mid-predict.

**Q164.** What if `_extract_band_per_row` is called with empty batches? `B=0`, `np.full(0, default)` → empty array. OK, but subsequent gate still tries to apply. `pred_dmid` would be empty too. Numpy handles empty arrays.

**Q165.** What if device is CUDA but model loaded to CPU? `to_t` puts tensors on device, so all on same device. OK.

**Q166.** What if device changes mid-execution (GPU disabled)? `_device` is captured in `__init__`. Subsequent calls use cached device. CUDA OOM possible if GPU memory tight.

**Q167.** No `torch.cuda.empty_cache()` after predict. GPU memory accumulates. For long-running predictor, OOM possible.

**Q168.** No `try/except` around `predict()` body. Any error → propagate. Platform expected to handle?

**Q169.** What if `lgb.Booster.predict()` returns shape mismatch (e.g., MultiOutput model)? Code assumes 1D output. Crash.

**Q170.** What if NN output `out.cpu().numpy()` returns NaN? Downstream gate: `pred > thr_up + band` is False for NaN, `pred < -(thr_dn + band)` is False for NaN. Stays at flat (1). **Silently flat for any NaN prediction.** Probably benign.

**Q171.** What if pred is +Inf? `Inf > thr_up + band` is True → action=2 (long). Could fire spurious longs from numerical issues.

---

## G. Sym-Agnostic Compliance & CRITICAL_CONSTRAINTS

**Q172.** CRITICAL_CONSTRAINTS.md: "模型必须 sym-agnostic". But `_extract_band_per_row` uses `sym` to pick a band → **model behavior depends on sym**. Is this within spec? The docstring claims sym is used only for "per-sym beta lookup" and "not fed to model.forward". Is gate widening still "sym-agnostic"?

**Q173.** A strict reading of sym-agnostic: same input → same output regardless of sym. With per-sym band, this is violated. The constraint may forbid this. Risk of submission rejection or 0 score.

**Q174.** CRITICAL_CONSTRAINTS: "可能含训练外股票" (may include out-of-training stocks). OOD fallback exists, but `int(s)` returns numeric sym IDs. If OOD sym is e.g. sym=5 (not in train), int() works, lookup misses, falls to default. OK.

**Q175.** But what if OOD sym is e.g. `100` (large int)? Same path: int(100)→100, not in `_cw_band`, fallback. OK.

**Q176.** CRITICAL_CONSTRAINTS: "测试点顺序被打乱". The Predictor processes each batch independently. ROW order within a batch matters (window-based features). Confirm platform shuffles BATCHES but not ROWS within a batch.

**Q177.** CRITICAL_CONSTRAINTS: "date 评测时被置 0". Predictor never uses date. Good. But verify no implicit dependency in `fast_features.py` / `fast_features_batch.py`.

**Q178.** Does `compute_batch_features` (ffb) ever look up a column named "date"? Not visible from this review. Worth checking ffb.

**Q179.** Sym-agnostic check: `feat_mean / feat_std` are GLOBAL (per-NN but not per-sym). Good, claimed in docstring.

**Q180.** Sym-agnostic check: `clip` is GLOBAL (max over NNs). Good.

**Q181.** LGB boosters trained on data spanning syms — but tree splits may happen to perfectly partition syms via correlated features. Is there an audit that LGB doesn't implicitly learn sym-specific behavior?

**Q182.** NN trained with sym-agnostic input but with sym-mixed batches. If batches accidentally over-represent one sym, NN may bias. Was train-time class balancing done?

---

## H. Hardcoded Magic Numbers & Oddities

**Q183.** Module-level `WINDOW = 100`. Why 100? Why not 50, 200?

**Q184.** `HORIZON_LIST = (5, 10, 20, 40, 60)`. Why these specific horizons? Why not 1, 5, 30? Why "60" max (5 min for tick data)?

**Q185.** `assert len(RAW_COLS_TRAIN_ORDER) == 154` — magic 154.

**Q186.** `_gate_with_band` default `thr_up=2.0e-4, thr_dn=2.0e-4`. Why 2e-4? Symmetric default? Where calibrated?

**Q187.** `_ev_gate_predict` same defaults. Magic 2e-4. Why?

**Q188.** `_cw_default_band = 0.16 * 4.0e-4` — magic 0.16 and 4.0e-4.

**Q189.** `1e-5` eps in LayerNorm — magic.

**Q190.** `torch.no_grad()` decorator on predict_mean — good practice, but why not `torch.inference_mode()` (slightly faster)?

**Q191.** `out.cpu().numpy().astype(np.float32)` — already float32, but explicit cast. Defensive but redundant.

**Q192.** `np.empty((N, WINDOW, K), dtype=np.float64)` then assigned in loop. `np.empty` is uninitialized — if loop fails partway, X3d has garbage. Use `np.zeros` for safety? Trivial cost.

**Q193.** `out = np.ones((B, 5), dtype=np.int64)` — initialize ALL to flat (1). Why action 1 = flat? Where defined? Confirm with platform.

**Q194.** Action enum: `0 = short, 1 = flat, 2 = long`. Confirm matches platform spec.

**Q195.** `for hcfg in self._horizons` iterates in order [h=5, h=10, h=20, h=40, h=60]. Since h=60 is LAST and others share with it, h=5 computes h=60 prediction first (cache miss), then h=10 hits cache. Wait — h=5 has `src_H = 60`, so cache miss on first iter computes 60. Then h=10,20,40 cache hit. Then h=60 (already cached). 

**Q196.** But the computation for h=5 (src=60) requires LGB models h=60 — these ARE loaded (since h=60 is also a horizon in config). OK.

**Q197.** If h=60 entry were missing or inactive, h=5's `src_H=60` lookup returns None for models. preds empty. Silent flat. Brittle.

**Q198.** `pred_cache` cleared each `predict()` call (local var). Good (no cross-call state).

**Q199.** Smoke test in `__main__`: creates df with `len(feats)` columns (155 incl. sym), then sets `df["sym"]=1`. Total 156? Or `len(feats)=155` and sym is set in-place. Re-check:
   - `feats = cfg["feature"]` is 155 items
   - `df = pd.DataFrame(rng..., columns=feats)` creates df with 155 cols (incl sym)
   - `df["sym"] = 1` overwrites existing sym col with 1
   - OK, 155 cols total.

**Q200.** Smoke test runs only h=60 models (since other horizons share). 100 rows passed, last row taken, etc. But it only tests with B=3. Doesn't stress with B=1024.

**Q201.** Smoke test doesn't validate output values (only prints). No regression check.

**Q202.** `print("Device:", p._device)` — prints to stdout. If smoke test runs in CI, this is fine. If unintentionally triggered in production import, noise.

**Q203.** `print(...)` in `__main__` block — fine (only on direct execution).

**Q204.** `if __name__ == "__main__":` block at bottom — but if Predictor.py is *imported* (which it always is in submission), this block is skipped. Good. But it could be removed in submission to reduce code.

**Q205.** `importlib.util.spec_from_file_location("iter_018_ffb", ...)` — module name "iter_018_ffb" leaks the experiment lineage (iter_018) into the runtime module name. If platform debugs / logs, this name shows up.

**Q206.** `mod_name = "iter_018_ffb"` magic name — could collide with other modules named similarly. Use UUID? Unique-enough?

**Q207.** `spec_from_file_location` doesn't add module to `sys.modules`. So if ffb internally imports something that does `import iter_018_ffb`, it would re-load — possibly with side effects.

**Q208.** Why use `importlib` instead of placing fast_features_batch in a package and `import .fast_features_batch`? Because submission is a flat directory (not a package)?

**Q209.** If `fast_features_batch.py` imports `fast_features.py` (probably), does that work via importlib? Python's normal import search runs. Should work because both are in same dir which is `sys.path[0]` or similar.

**Q210.** Actually — `here` is the script directory. But is `here` in `sys.path`? On platform import, `Predictor.py` is imported by platform, so `here` may not be in `sys.path`. Then ffb's `import fast_features` may fail.

**Q211.** Test this: does inference work on platform if ffb does internal imports? Probably yes since `importlib.util.spec_from_file_location` doesn't run path-based imports — but if ffb does, it depends on sys.path.

**Q212.** `_load_module` returns the loaded module — used as `self._ffb`. So `self._ffb.all_feature_names()` etc. work via attribute access on the module object.

**Q213.** No explicit `dispose` / cleanup. When Predictor is GC'd, models stay until process exits. Acceptable for one-shot eval.

**Q214.** `lgb.Booster` doesn't expose memory cleanup. Held until Python GC. Probably fine.

**Q215.** torch tensors held in `self.W`, `self.b`, etc. — GC'd with Predictor instance. Acceptable.

**Q216.** Why `torch.from_numpy(np.stack(lst, 0)).to(device)` — could be `torch.tensor(np.stack(...), device=device)`. Subtle: `from_numpy` shares memory with NumPy, `tensor` copies. `.to(device)` copies anyway. Either works.

**Q217.** `feats` is float32 (after concat of float32 raw_last and float32 extras_kept). Passed to LGB `.predict()` — LGB internally converts to its native dtype. OK.

**Q218.** Why LGB predict accepts float32 (numpy)? LGB happily takes float32 (auto-converts internally). Was training data float32 or float64? Mismatch → tiny prediction diff (boundary cases).

**Q219.** `_doc` in thresholds.json says "h=60 unchanged (platform SOTA +35.64)". What unit is 35.64? Score points? Percent? Where defined?

**Q220.** `iter_018_ffb` naming and `T188v3` package name — git lineage in code. Acceptable but not professional for a final submission.

**Q221.** No `LICENSE` file in package. Required by platform?

**Q222.** No `__init__.py` in package directory. So if platform tries `import package`, fails. But platform should `import Predictor` directly.

**Q223.** `Predictor.py:39 import torch` — torch is ALWAYS imported, even on platforms without GPU. CPU torch is heavy (300+ MB install). Big dependency.

**Q224.** `import torch.nn.functional as F` — only `F.gelu` used. Could `from torch.nn.functional import gelu` to avoid full F namespace import? Marginal cost.

**Q225.** Module-level constants `RAW_COLS_TRAIN_ORDER`, `FAIL_NAMES` etc. — repeated in package. If you change one, must update both training and inference. No shared source of truth.

**Q226.** Predictor docstring claims "Outputs match T188v2 within float32 precision (~1e-6 typical, 1e-4 max)" — claims tolerance vs. some baseline. Where measured? Test suite?

**Q227.** Tolerance "1e-4 max" — for what? Prediction values? Action outputs? If 1e-4 in pred and threshold is 8.66e-5, a 1e-4 swing can FLIP the action. So bit-equivalence between v2 and v3 NOT guaranteed for some inputs.

**Q228.** Action flips between v2 and v3 are possible at boundary. Was tested?

**Q229.** Why bother with v3 (batched NN) if action output may differ from v2? Latency win must justify. Was benchmarked?

**Q230.** No `setup.py` / `pyproject.toml`. Pip installs from requirements only. No package metadata.

**Q231.** `requirements.txt` doesn't have an explicit pip command — but it's an install file, not a script. OK.

**Q232.** No way to verify requirements are satisfied at runtime. If user runs without installing, `import torch` fails. Bare ImportError.

**Q233.** `build_pkg.py` accesses `HERE = os.path.dirname(os.path.abspath(__file__))` — assumes script is at `04_build_pkg/`. If moved, breaks. Not tested.

**Q234.** `build_pkg.py: if not os.path.dirname(args.pkg) else "."` — edge case for bare filename. Convoluted.

**Q235.** `build_pkg.py` doesn't `sys.exit(0)` on success — just prints. Caller may not know success vs warning.

---

## I. Cross-File Consistency / Misc

**Q236.** `config.json:feature` list has 155 entries (incl sym). `RAW_COLS_TRAIN_ORDER` in `Predictor.py` has 154 (no sym). Independent sources — must stay aligned manually.

**Q237.** If config.json's feature list grows (e.g., add bid11), the Predictor's `RAW_COLS_TRAIN_ORDER` doesn't auto-update. Train/inference skew.

**Q238.** `config.json:label` has 5 labels. `HORIZON_LIST` in Predictor has 5 horizons. If you add a label_120, must edit both. Tight coupling, loose tooling.

**Q239.** `thresholds.json:horizons` lists 5 horizons matching HORIZON_LIST. Trinity of label/horizon/threshold must be consistent. No checker.

**Q240.** `build_pkg.py` hardcodes `N_NN=50, N_LGB=50`. `thresholds.json` hardcodes `ensemble_seeds=[1..50]`. If you train 60 seeds, must edit both. Bug-prone.

**Q241.** `Predictor.py` loads files based on `ensemble_seeds` from JSON — if you remove seeds from JSON but leave files, those files are NOT loaded. So JSON is the authority. But `build_pkg.py` copies based on filesystem presence, not JSON list. Mismatch possible.

**Q242.** If a NN trained with seed 51 exists in filesystem but JSON has only [1..50], build_pkg COPIES it (filesystem-based copy), but Predictor IGNORES it. Wasted package size.

**Q243.** `requirements.txt` doesn't list `joblib` — does any code use joblib? Not in Predictor.py. Maybe ffb uses it.

**Q244.** No `protobuf`, `flatbuffers` — for any serialization? Probably fine.

**Q245.** `_doc` JSON keys: thresholds.json has `_doc` and `_note` keys. These are documentation comments. Code skips them gracefully (they're not "h" entries). But if you accidentally add `"_doc"` as a horizon entry's key, it would pass through.

**Q246.** Horizon entry parsing: `for hcfg in self._horizons` — assumes every dict in list is a horizon. No `if "h" in hcfg` check. If you add a `{"_note": "..."}` dict to the list, `int(hcfg["h"])` crashes.

**Q247.** What about `share_with` chain? h=5 → 60 → ? If 60 had `share_with=60`, infinite reference? Code: `src_H = int(hcfg.get("share_with", H))` — for h=60 with share_with=60, src_H=60. No recursion. Just a fixpoint. OK.

**Q248.** If h=60 had `share_with=5` and h=5 had `share_with=60`, what happens? Iter h=5 first: src=60. Cache miss, computes h=60 models. Cache hit pred_dmid. h=60 next: src=5. Cache miss for src=5 (not in cache yet since we cached as src=60). Computes h=5 models — but h=5 has no models loaded (only h=60 has ensemble_seeds). preds=[], continue. So h=60 stays flat. Silent broken behavior. No share_with cycle detection.

**Q249.** No logging during inference. If something goes wrong, no breadcrumbs.

**Q250.** No metrics emission (counts of long/flat/short, abstention rate). Useful for production monitoring.

**Q251.** No version stamp in stdout/log on Predictor instantiation. Hard to verify which build is running.

**Q252.** thresholds.json `enabled: true` — but config doesn't have a flag to globally disable conformal. Must edit JSON.

**Q253.** `per_sym_beta` keys are strings, `cw_band` keys are ints (after `int(k_str)`). Type conversion in `__init__`. OK.

**Q254.** Beta values are floats (0.1, 0.4 etc). What if JSON has integer (0)? `float(0)` = 0.0. OK.

**Q255.** Should beta be capped? `beta * sigma` with beta=10 (typo) would make band huge → never trades. No validation.

**Q256.** Where in code is `pred_dmid` clamped or sanity-checked? Nowhere. Trust upstream.

**Q257.** If NN's `feat_mean` and `feat_std` were learned from a window without certain syms (e.g., only sym 0 in train), normalizing OOD data with these stats yields out-of-distribution normalized values. Then clamp to ±clip. Predictions may still be poor.

**Q258.** `target_scale.unsqueeze(1)` — `target_scale` is shape `(N,)` not `(N,1)`. Then `.unsqueeze(1)` makes it `(N,1)`. Wait, line 131-133:
```
target_scale: torch.Tensor = torch.tensor(target_scales, dtype=torch.float32, device=device).unsqueeze(1)  # (N,1)
```
So already `(N,1)` after init. Then line 176 `out / self.target_scale` — `(N, B) / (N, 1)` broadcasts. OK.

**Q259.** Or is target_scale (N, 1, 1) needed? Let me re-trace. After line 175: `out + self.bF`. bF is (N, 1). Initial out is (N, B). bF broadcast (N, 1) → (N, B). OK. Then `out / target_scale` (N, B) / (N, 1) → (N, B). OK. Then `out.mean(0)` → (B,). OK.

**Q260.** `target_scale` semantics: model output is `pred_scaled`, true target was `target / scale`. So `pred_unscaled = pred_scaled * scale`. But code does `out / target_scale`. So it DIVIDES by scale. So target_scale here is `1/scale_factor` of training. Confusing naming.

**Q261.** Or: target_scale is the divisor applied to training target. So during training, model fit `target / scale`. At inference, output is `target / scale`-like, so multiply by `scale` to recover. But code DIVIDES. Inverse?

**Q262.** Need to check training code to understand. If target_scale=1.0 (likely), this doesn't matter. But if !=1, prediction may be off by `target_scale^2`.

**Q263.** Are all 50 NNs' target_scale = 1.0? If yes, division is no-op. Should verify.

**Q264.** All these are subtle. Without training-side check, can't verify.

**Q265.** No assertion that all NN's hidden dims match across the 50 files in `_BatchedMLPEnsemble.__init__`. d0's hidden is used; other npz's hidden ignored. If they differ, `torch.stack` later fails (shape mismatch on `L0_W`).

**Q266.** `use_layernorm` from d0 — assumes all same. If one NN trained without LN and one with, mismatch.

**Q267.** Line 119-121: `if self.use_layernorm: lLNW[i].append(d["LN{i}_W"])`. If `use_layernorm=False` from d0 but a specific npz has LN keys, they're ignored. Could mask training inconsistency.

**Q268.** What's the expected hidden architecture? Docstring says `[359 -> 256 -> 128 -> 64 -> 1]`. So hidden=[256, 128, 64], in_dim=359. Final layer LF: hidden_last=64 → 1.

**Q269.** 3 hidden layers, 359 in, 1 out. ~100K params per NN. 50 NNs = 5M params total. Tiny. Should infer in ms even sequential. Why bother batching?

**Q270.** Maybe batching is for CUDA throughput, not CPU. On CPU, sequential might be similar. Was benchmarked?

**Q271.** ZIP file size estimate: 50 LGB txt files (say 1-2 MB each compressed) + 50 NN npz (small, <100 KB each) + code ~50 KB. Total: 50-100 MB. Submission size limits?

**Q272.** Smoke test passes `df_no_sym` (sym column dropped). The code's band lookup uses `if "sym" not in df.columns: continue` — leaves default band. OK.

**Q273.** But `_compute_batch_features` uses `df[self._raw_feat_cols]` — doesn't access sym. So missing sym OK for feature computation.

**Q274.** Final question that's a meta-concern: this Predictor wires together v2 (50 LGB + 50 NN) + v3 batched NN + sqrt-scaled multi-horizon + per-sym conformal. Lots of moving parts, no integration test in repo. **Confidence in joint correctness is by inspection only.** Any single bug in any layer (scaling, gate, ensemble weight, band lookup) silently degrades performance.

**Q275.** What's the FAST-FEATURES-BATCH dependency contract? `compute_batch_features(X3d, col_idx)` — input shape (N, W, K), col_idx dict. Output shape assumed (N, M). Where documented? Where versioned?

**Q276.** `all_feature_names()` — returns list of M names. Order must match output column order. Where verified?

**Q277.** If ffb's feature list changes (e.g., new feature added), `self._extra_keep_idx` is computed at `__init__` using current ffb. But NN was trained with OLD ffb. Now mismatch. No versioning.

---

## J. Summary Concerns

**Q278.** The combination of (a) heavy hardcoded constants in Predictor.py, (b) duplication between config.json and module constants, (c) duplication in thresholds.json (w_nn/w_lgb in shared horizons), (d) lack of integration test, (e) silent fallback paths (missing models, OOD syms, NaN preds) — creates a system that's fragile to upstream changes and silent on degraded modes. What is the upstream change-management story?

**Q279.** No README / no comment block in code explaining the training pipeline that produced these files. If someone has to debug 6 months from now, lots of archaeology needed.

**Q280.** The `_doc` string in thresholds.json is the closest thing to documentation. Cute, but easily lost if someone regenerates config without preserving comments.

---

## K. USER-RAISED — sqrt(h/60) threshold scaling 的方向是错的

**Q281.** 【用户提出的精确推导】The `sqrt(H/60)` scaling for shared-horizon thresholds is **mathematically backwards** under a random-walk-with-drift model.

- 位置: `thresholds.json:7,16,25,34` + `Predictor.py` 决策门 `|pred| > thr_h`
- 模型实际回归目标 `y = (mp_{t+60} − mp_t) / (mp_t + 1)` ⇒ pred 是 **60-tick 归一化 Δmid 的估计**
- 短 horizon 直接 share `pred_h60`（**pred 不缩放**）
- thresholds: `thr_h = thr_60 × sqrt(h/60)`（短 h 阈值变小）

**推导**（drift-with-noise）：
- `E[Δmid_h] = μ·h`（drift 线性于 h）
- `std[Δmid_h] = σ·sqrt(h)`（noise sqrt(h)）

若想把 "h=60 预测" 真正用作 "h=h0 决策"，**pred 应缩 h/60（线性）**，而不是 thr 缩 sqrt(h/60)：
- 正确做法 A：`pred_h60 × (h/60) > thr_h_native`
- 等价做法 B：`pred_h60 > thr_h_native × (60/h)`（thr 反向缩 60/h，**变大**）
- 若 `thr_h_native = thr_60 × sqrt(h/60)`（按 noise std 缩）⇒ 等效 thr = `thr_60 × sqrt(60/h)`
- 代码实际：`thr_60 × sqrt(h/60)` ⇒ 与"理论正确"差 `60/h` 倍（h=5 时差 **12 倍**）

**问题**：
- (a) `_doc` 注释 "shorter horizons trade fewer/smaller signals" —— 实际效果完全相反（thr 变小 ⇒ trade 更多）
- (b) 忽略 `label_5/10` 的 α=0.05% vs `label_20/40/60` α=0.1% 这个 label 定义差异（短 h 的"涨/跌"门槛本来就更严）
- (c) 没有 ablation：sqrt(h/60) vs h/60 vs 1.0（不缩）vs sqrt(60/h)（理论正确方向）的平台对比
- (d) HISTORY_EVIDENCE_DB.md 主题 10 未找到该 ablation 记录；猜测作者把短 h 当 "score=max(per-horizon) 不会扣分"的安全占位，没认真推导

**可能的辩护（站不太住）**：
- 把 pred 当 horizon-agnostic "signal strength"（与 h 无关）⇒ thr 按 noise std 缩。但 pred 显式回归 60-tick Δmid，这个解读违反训练目标
- empirical：作者可能跑过几种 scaling，sqrt 在平台上表现最好。但 history 里看不到此 ablation

**结论**：sqrt(h/60) 在数学上没有清晰正当性，方向与 random-walk-with-drift 理论预测相反。现状能"work"是因为：
1. 平台 score = max(per-horizon) ⇒ 短 h 即使更差也不影响 +35.64 排名
2. 短 h 还有 conformal 弹性带兜底（per-sym β·σ 加在 thr 上，hi-band sym 在短 h 上几乎抵消了 sqrt 缩放，见 Q23）

**建议补做的实验**：在不动 h=60 的情况下，submit 4 个变体（thr 缩 h/60 / 1.0 / sqrt(60/h) / 关掉短 h 提交），对比短 h 平台分。如果其中任何一个让短 h 平台分超过 h=60 的 +35.64，就会改变 max-rank 结果。

---

RESULT: task=[predictor review] metrics={n_questions=281} notes=[Critical questions across thresholds (24 incl. sqrt scaling justification, OOD asymmetry, asymmetric thr_up/thr_dn ratio), ensemble combination (33 incl. w_lgb=1.5 rationale, max-clip choice, NN/LGB asymmetric optimization), feature processing (32 incl. log1p only on amount_delta, NaN-to-0 vs LGB native NaN, FAIL_NAMES criterion), model file format (21 incl. npz vs state_dict, lazy loading, schema versioning), config/build/requirements (33 incl. dead md5_file, no zip step, hardcoded N=50, numpy 2.x compat, torch CPU branch), fault tolerance (28 incl. empty/wrong-shape df, OOM, share_with cycles, NaN propagation), sym-agnostic compliance (11 incl. whether per-sym band violates spec, OOD treatment), magic numbers/oddities (50+ incl. WINDOW=100, eps=1e-5, default thr 2e-4), cross-file consistency (13 incl. config vs RAW_COLS, JSON-vs-filesystem authority). Many concerns about: (1) silent failure modes giving flat actions, (2) duplication across config sources, (3) sqrt(t) threshold scaling derivation, (4) OOD getting smaller band than uncertain known syms, (5) batched NN clip aggregation using max not per-NN.]
