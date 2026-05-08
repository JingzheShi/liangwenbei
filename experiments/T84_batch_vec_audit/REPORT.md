# T84 — Batch-vec inference cross-window leakage audit

**Target:** `submission/iter_013_time_features/fast_features_batch.py`
**Verdict:** **NO leakage detected.** Batch and single-row inference are byte-for-byte identical (max |Δ| = 0.0) across 8 independent invariance checks. No fix needed.

## Setup

- 24 real 100-tick windows pulled from 4 parquets in `data/` (sym0, multiple dates / sessions, end-indices 99 / 200 / 500 / 999 / 1500 / 1900)
- A larger heterogeneous batch of 36 windows spanning **sym 0 .. 4** (×3 dates × 2 sessions × 3 end-indices) for check 8
- Feature count F = 216 (T3-no-time 69 + Stage1 54 + Stage2 59 + Stage3 14 + Stage5 20)
- Repro: `python3 experiments/T84_batch_vec_audit/audit.py` (writes `audit_results.json`)

## Results — empirical invariance checks

| # | Check | Predicate (no-leak) | Observed max \|Δ\| | Verdict |
|---|---|---|---|---|
| 1 | **Single-row vs batched** — feed `X3d[i:i+1]` alone vs the full N-batch and compare row i for each of the 24 windows | rows must equal byte-for-byte | **0.000e+00** | OK |
| 2 | **Shuffle invariance** — random-permute window order, undo permutation | features unchanged | **0.000e+00** | OK |
| 3 | **Replication** — N=8 copies of the same window | all 8 output rows identical | **0.000e+00** | OK |
| 4 | **Reverse ordering** — windows fed back-to-front (future-then-past) | each row equals time-ordered counterpart | **0.000e+00** | OK |
| 5 | **Mixed-source interleave** — interleave even/odd indices | features unchanged | **0.000e+00** | OK |
| 6 | **Adversarial neighbours** — target window placed next to garbage rows scaled ×1e6 / ×1e-6 / -100 | target rows match target-alone | **0.000e+00** | OK |
| 8 | **Heterogeneous syms** — 36 windows from sym 0..4, batched vs each alone | byte-for-byte | **0.000e+00** | OK |
| 9 | **Cross-implementation** — `compute_batch_features(X3d)[:, :196]` vs `fast_features.py::compute_window_features` (single-window reference) | within 1e-3 documented tolerance | **5.329e-07** | OK (well within tol — float reduction-order, NOT leakage) |

The fact that *every batch-vs-single check is exactly 0.0* (not just within 1e-9) is strong evidence: the same floating-point operations execute on a (1, T) array as on the i-th slab of (N, T), so there is literally no opportunity for batch-dim influence.

## Static axis review (line-by-line)

I read every `axis=…` reduction in `fast_features_batch.py`. Summary:

| Operation kind | Lines | Axis used | Correct? |
|---|---|---|---|
| `lfilter(...)` (EWMA) | `_ewma_last_batch` line 98 | `axis=-1` | ✓ time |
| EWMA `zi` initial | line 97: `(1.0 - a) * x[:, :1]` | per-window slice | ✓ |
| `np.cumsum(x, axis=-1)` | line 117 | `axis=-1` | ✓ time |
| `seg.sum(axis=-1)` | many | `axis=-1` | ✓ time |
| `seg.mean(axis=-1)` | many | `axis=-1` | ✓ time |
| `seg.std(axis=-1, ddof=0)` | many | `axis=-1` | ✓ time |
| `seg20.mean(axis=1)` (Stage1 dual_z) | line 282 | shape `(N, 20, 37)`, axis=1 = time | ✓ |
| `np.median(seg, axis=-1)` (Stage5 spread) | line 691 | `axis=-1` | ✓ time |
| `np.quantile(seg, q, axis=-1)` | lines 692-3 | `axis=-1` | ✓ time |
| `(seg <= last_val).mean(axis=-1)` (qrank) | line 371 | `axis=-1` | ✓ time |
| Lag/diff (`b_prev[:, 1:] = b[:, :-1]`) | many | per-window time shift | ✓ |
| `np.argwhere`, `np.where` reductions over batch | none found | — | ✓ |

There are **no `axis=0` reductions, no global `mean/std/quantile`, no cross-row indexing, and no in-place writes that could leak**. All EWMA initial conditions are computed per-window from `x[:, :1]`. `lfilter`'s axis=-1 mode is documented to run an independent IIR filter for each row of the input.

## Per-feature leakage attribution

Skipped — no leakage was detected, so there is nothing to attribute. Check 1 produced an all-zeros diff matrix.

## Conclusion

`fast_features_batch.py` is **provably leakage-free** under the empirical invariance protocol:

1. Batch dimension is independent: every `(N, …)` reduction is along the time axis (`axis=-1` or, in the `(N, T, F)` Stage-1 stack, `axis=1`).
2. EWMA initial conditions are derived per-window.
3. No global statistics or cross-window broadcasting.
4. Output is byte-for-byte identical to running each window alone.

The 5.3e-7 residual against `fast_features.py` is purely floating-point reduction-order noise (consistent with the file's docstring "max diff < 1e-3"), and would still hold even with N=1 batches.

**Recommendation:** No fix needed. The iter_010+ batch-vec optimisation is safe to keep; the leaderboard gap reported in `iter_013` (LOSO-equiv +36.23 vs platform +19.23) is **not** caused by inference-time cross-window leakage.

## Artifacts

- `audit.py` — repro script
- `audit_results.json` — machine-readable per-check results
- `REPORT.md` — this file

## RESULT line

```
RESULT: task=t84_audit metrics={leakage_found=N, severity=0.000e+00, fix_needed=N}
```
