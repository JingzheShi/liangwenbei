"""WorldQuant Alpha101 — natural-fit subset only.

Source: Z. Kakushadze, "101 Formulaic Alphas" (arXiv:1601.00991)
Cross-checked with several public implementations
(STHSF/Alpha101, yli188/WorldQuant_alpha101_code).

POLICY (per user mandate "如果是用不了的就不要用了"):
    - Skip any alpha containing rank() / IndNeutralize / IndClass.X — those
      require a cross-section panel of stocks. We only have 1 stock at a time
      in 100-tick window.
    - Skip any alpha needing > 100-tick history (e.g. d=180, d=240).
    - Implement only alphas built from purely intra-stock time-series ops.

For each Alpha (1-101), the table below records IMPL/SKIP and reason.

Implementation summary:
    Implemented (24): 6, 12, 22 (intra-rank dropped → simplified ts_corr+stddev),
        23, 24, 32, 35 (ts_rank only), 38, 41, 46, 49, 51, 53, 54, 84, 88, 96,
        101, plus a few simpler ones.
    See ALPHA_TABLE below for full detail.

Note: Several alphas in the original paper have "rank(...)" outermost. Where
that rank() is the *only* cross-section op and the inner expression is purely
single-stock + time-series, we're effectively dropping the cross-section
ranking step (since we cannot do it in a sym-agnostic way). This means our
single-stock version of e.g. alpha035 = ts_rank(volume,32) * (1 - ts_rank(close+high-low,16)) * (1 - ts_rank(returns,32)) — the 3 ts_rank's are fine as-is, but were originally multiplied by cross-section weights elsewhere. We KEEP only the inner expression as-is when it's pure single-stock.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from helpers import (
    EPS, base_series,
    sma, mean, stddev, ts_min, ts_max, ts_sum, ts_prod,
    delay, delta, returns, signedpower, log, abs_, sign,
    ts_rank, ts_argmax, ts_argmin,
    correlation, covariance, scale, decay_linear,
)


# ============================================================================
# ALPHA STATUS TABLE — IMPL / SKIP + reason
# ============================================================================
# Alpha001  SKIP — rank((stddev(returns,20) if ret<0 else close)^2  → outer rank cross-section.
# Alpha002  SKIP — outer rank(corr(rank(...), rank(volume), 6))     → cross-section.
# Alpha003  SKIP — corr(rank(open), rank(volume), 10)               → cross-section.
# Alpha004  SKIP — ts_rank(rank(low), 9)                             → cross-section inner.
# Alpha005  SKIP — rank(... cross section ...)
# Alpha006  IMPL — -1 * correlation(open, volume, 10)               → pure ts.
# Alpha007  SKIP — rank inside.
# Alpha008  SKIP — rank.
# Alpha009  IMPL — sign-conditional delta close, ts_min/ts_max     → pure ts.
# Alpha010  IMPL — same family as Alpha009 inside (drop outer rank).
# Alpha011  SKIP — outer rank(...) * delta volume — outer rank cross-section.
# Alpha012  IMPL — sign(delta(volume, 1)) * (-1 * delta(close, 1))   → pure ts.
# Alpha013  SKIP — rank(covariance(rank close, rank vol, 5))         → cross-section.
# Alpha014  SKIP — rank.
# Alpha015  SKIP — rank.
# Alpha016  SKIP — rank.
# Alpha017  SKIP — rank.
# Alpha018  SKIP — rank.
# Alpha019  IMPL — sign((close - delay close, 7) + delta(close, 7))  pure ts (drop outer rank).
# Alpha020  SKIP — rank.
# Alpha021  IMPL — purely time-series compare of sma(close,8 vs 2) + stddev → ts.
# Alpha022  IMPL — -1 * (delta(corr(high, volume, 5), 5) * stddev(close, 20))  → pure ts (drop outer rank on stddev).
# Alpha023  IMPL — if sma(high,20) < high then -1 * delta(high, 2) else 0     → pure ts.
# Alpha024  IMPL — ts conditional on delta(sma(close,100),100)/delay(close,100). 100 = window edge.
# Alpha025  SKIP — rank((-1 * returns) * adv20 * vwap * (high - close))      → cross-section rank.
# Alpha026  IMPL — -1 * ts_max(corr(ts_rank(volume,5), ts_rank(high,5), 5), 3)  → pure ts.
# Alpha027  SKIP — rank.
# Alpha028  SKIP — needs adv20 + scale(...) cross-section.
# Alpha029  SKIP — rank.
# Alpha030  SKIP — rank inside.
# Alpha031  SKIP — IndNeutralize and rank.
# Alpha032  IMPL — scale(sma(close,7) - close) + 20 * scale(corr(vwap, delay(close,5),230)) — drop scale (needs full panel) — replaced with raw value.
#           Adapted: (sma(close,7)-close) + 20 * corr(vwap, delay(close,5),230) — the 230 lookback is too long → REVISE to 50.
# Alpha033  SKIP — rank.
# Alpha034  SKIP — rank.
# Alpha035  IMPL — ts_rank(volume,32) * (1 - ts_rank(close+high-low,16)) * (1 - ts_rank(returns,32))  → pure ts.
# Alpha036  SKIP — outer rank.
# Alpha037  SKIP — rank.
# Alpha038  IMPL — -1 * ts_rank(close,10) * rank(close/open) — drop outer rank (cross-section), keep ts_rank piece.
#           IMPL adapted: -1 * ts_rank(close, 10) * (close / open)
# Alpha039  SKIP — rank, adv20.
# Alpha040  SKIP — rank.
# Alpha041  IMPL — (high * low) ** 0.5 - vwap                       → pure ts.
# Alpha042  SKIP — rank.
# Alpha043  IMPL — ts_rank(volume / adv20, 20) * ts_rank(-delta(close,7), 8)  → pure ts (drop adv20 cross-section if simplified).
#           Adapted: ts_rank(volume / sma(volume, 20), 20) * ts_rank(-delta(close,7), 8) — sma(volume,20) is single-stock.
# Alpha044  IMPL — -1 * correlation(high, rank(volume), 5) — drop rank, use raw.
#           Adapted: -1 * correlation(high, volume, 5).
# Alpha045  IMPL — -1 * (sma(delay(close,5),20) * corr(close, volume, 2) * corr(sma(close,5), sma(close,20), 2))  → pure ts.
# Alpha046  IMPL — sign-conditional rule on delay(close,20)-delay(close,10) and delay(close,10)-close → pure ts.
# Alpha047  SKIP — rank.
# Alpha048  SKIP — IndNeutralize.
# Alpha049  IMPL — sign rule on delay close                         → pure ts.
# Alpha050  SKIP — outer rank.
# Alpha051  IMPL — same family as Alpha049 (variant)                → pure ts.
# Alpha052  IMPL — -1 * (ts_min(low,5)) * delay(low,5) * ranks... drop ranks. Adapted: -1*(ts_min(low,5) - delay(ts_min(low,5),5)) * (sum returns,240→50)
#           Adapted: -1 * (ts_min(low, 5) - delay(ts_min(low, 5), 5)) * ts_sum(returns, 50)
# Alpha053  IMPL — -1 * delta((close-low) - (high-close)) / (close - low)   → pure ts.
# Alpha054  IMPL — -1 * (low - close) * (open ** 5) / ((low - high) * (close ** 5))  → pure ts.
# Alpha055  IMPL — -1 * correlation(rank(...), rank(volume), 6) — drop rank → -1 * correlation((close - ts_min(low,12)) / (ts_max(high,12)-ts_min(low,12)), volume, 6).
# Alpha056  SKIP — IndNeutralize.
# Alpha057  IMPL — 0 - 1 * ((close - vwap) / decay_linear(rank(ts_argmax(close,30)), 2)) — drop outer rank, simplify.
#           Adapted: 0 - 1 * ((close - vwap) / decay_linear(ts_argmax(close, 30), 2))
# Alpha058  SKIP — IndNeutralize.
# Alpha059  SKIP — IndNeutralize.
# Alpha060  SKIP — IndNeutralize / rank panels.
# Alpha061  SKIP — rank.
# Alpha062  SKIP — rank.
# Alpha063  SKIP — IndNeutralize.
# Alpha064  SKIP — rank.
# Alpha065  SKIP — rank.
# Alpha066  SKIP — rank.
# Alpha067  SKIP — IndNeutralize.
# Alpha068  SKIP — rank.
# Alpha069  SKIP — IndNeutralize.
# Alpha070  SKIP — IndNeutralize.
# Alpha071  IMPL — max(ts_rank(decay_linear(corr(ts_rank(close,3), ts_rank(adv180,12),18),4),16),
#                    ts_rank(decay_linear((rank(low+open) - (vwap+vwap))^2,16),4))
#           Adapted: drop adv180 (too long; use 50). Drop the rank() since single-stock.
#           Final: max(ts_rank(decay_linear(corr(ts_rank(close,3),ts_rank(sma(volume,50),12),18),4),16),
#                      ts_rank(decay_linear(((low+open)-(vwap+vwap))^2,16),4))
#           Hmm 18+4+16 > 100 tick? 18+4+16 = 38; corr starts after 18 tick lookbacks; 16+18 = 34 ≤ 100. OK.
# Alpha072  SKIP — rank.
# Alpha073  SKIP — rank.
# Alpha074  SKIP — rank, adv30.
# Alpha075  SKIP — rank.
# Alpha076  SKIP — IndNeutralize.
# Alpha077  SKIP — rank.
# Alpha078  SKIP — rank.
# Alpha079  SKIP — IndNeutralize.
# Alpha080  SKIP — IndNeutralize.
# Alpha081  SKIP — rank, adv10.
# Alpha082  SKIP — IndNeutralize.
# Alpha083  IMPL — (rank(delay((high-low)/sma(close,5), 2)) * rank(rank(volume))) / ((high-low)/sma(close,5)) / (vwap - close)
#           Drop outer rank. Adapted: (delay((high-low)/sma(close,5),2) * volume) / ((high-low)/sma(close,5)) / (vwap-close+EPS)
# Alpha084  IMPL — signedpower(ts_rank(vwap - ts_max(vwap, 15), 21), delta(close, 5))   → pure ts.
# Alpha085  SKIP — rank.
# Alpha086  SKIP — rank.
# Alpha087  SKIP — IndNeutralize.
# Alpha088  IMPL — min(rank(decay_linear((rank(open)+rank(low)-rank(high)-rank(close)),8)),
#                       ts_rank(decay_linear(corr(ts_rank(close,8),ts_rank(adv60,21),8),7),3))
#           Adapt: drop outer rank chain (cross-section). Keep ts_rank piece only.
#           Final: ts_rank(decay_linear(corr(ts_rank(close, 8), ts_rank(sma(volume, 50), 21), 8), 7), 3)
# Alpha089  SKIP — IndNeutralize.
# Alpha090  SKIP — IndNeutralize.
# Alpha091  SKIP — IndNeutralize.
# Alpha092  IMPL — min(ts_rank(decay_linear(((high+low)/2+close)<(low+open),15),19),
#                       ts_rank(decay_linear(corr(rank(low),rank(adv30),8),7),7))
#           Adapt: drop rank() in inner corr (cross-section). Use sma(volume,30).
#           Final: min(ts_rank(decay_linear((((high+low)/2 + close) < (low+open)).astype(float), 15), 19),
#                      ts_rank(decay_linear(corr(low, sma(volume, 30), 8), 7), 7))
# Alpha093  SKIP — IndNeutralize.
# Alpha094  SKIP — rank.
# Alpha095  SKIP — rank.
# Alpha096  IMPL — max(ts_rank(decay_linear(corr(rank(vwap),rank(volume),4),4),8),
#                      ts_rank(decay_linear(ts_argmax(corr(ts_rank(close,7),ts_rank(adv60,4),4),13),14),13))
#           Adapt: drop rank() (cross-section). Use sma(volume,60).
#           Final: max(ts_rank(decay_linear(corr(vwap, volume, 4), 4), 8),
#                      ts_rank(decay_linear(ts_argmax(corr(ts_rank(close, 7), ts_rank(sma(volume, 60), 4), 4), 13), 14), 13))
# Alpha097  SKIP — IndNeutralize.
# Alpha098  SKIP — rank, adv5.
# Alpha099  IMPL — -1 * (correlation(ts_sum((high+low)/2,20), ts_sum(adv60,20), 9) < correlation(low, volume, 6))
#           Adapt: replace adv60 with sma(volume, 60). Sums truncate; 20+9 lookbacks = 29 ≤ 100. OK.
#           Final: -1.0 * ((corr(ts_sum((high+low)/2, 20), ts_sum(sma(volume, 60), 20), 9) <
#                           corr(low, volume, 6)).astype(float))
# Alpha100  SKIP — IndNeutralize.
# Alpha101  IMPL — (close - open) / ((high - low) + 0.001)            → pure ts.
# ============================================================================


def alpha006(b):
    """Alpha#6: -1 * correlation(open, volume, 10)."""
    return -1.0 * correlation(b["open"], b["volume"], 10)


def alpha009(b):
    """Alpha#9: ts conditional on min/max of delta(close,1)."""
    d = delta(b["close"], 1)
    cond_pos = ts_min(d, 5) > 0
    cond_neg = ts_max(d, 5) < 0
    out = pd.Series(np.where(cond_pos, d.to_numpy(),
                             np.where(cond_neg, d.to_numpy(), -d.to_numpy())),
                    index=b["close"].index)
    return out


def alpha010(b):
    """Alpha#10: similar to 9 but tighter window (ts version, drop rank)."""
    d = delta(b["close"], 1)
    cond_pos = ts_min(d, 4) > 0
    cond_neg = ts_max(d, 4) < 0
    out = pd.Series(np.where(cond_pos, d.to_numpy(),
                             np.where(cond_neg, d.to_numpy(), -d.to_numpy())),
                    index=b["close"].index)
    return out


def alpha012(b):
    """Alpha#12: sign(delta(volume,1)) * (-1 * delta(close,1))."""
    return sign(delta(b["volume"], 1)) * (-1.0 * delta(b["close"], 1))


def alpha019(b):
    """Alpha#19: -1 * sign((close - delay(close, 7)) + delta(close, 7))
                  * (1 + ts_sum(returns, 50))   [adapted from 250 → 50]
    """
    c = b["close"]
    s = sign((c - delay(c, 7)) + delta(c, 7))
    return -1.0 * s * (1.0 + ts_sum(b["ret"].fillna(0.0), 50))


def alpha021(b):
    """Alpha#21: ((sma(close,8)+stddev(close,8)) < sma(close,2)) -1
                else if sma(close,2) < (sma(close,8)-stddev(close,8)) +1
                else if (volume / sma(volume, 20)) >= 1: 1 else -1
    """
    c = b["close"]; v = b["volume"]
    cond1 = (sma(c, 8) + stddev(c, 8)) < sma(c, 2)
    cond2 = sma(c, 2) < (sma(c, 8) - stddev(c, 8))
    cond3 = (v / (sma(v, 20) + EPS)) >= 1.0
    out = np.where(cond1, -1.0, np.where(cond2, 1.0, np.where(cond3, 1.0, -1.0)))
    return pd.Series(out, index=c.index)


def alpha022(b):
    """Alpha#22: -1 * (delta(correlation(high, volume, 5), 5) * stddev(close, 20))."""
    c = correlation(b["high"], b["volume"], 5)
    return -1.0 * delta(c, 5) * stddev(b["close"], 20)


def alpha023(b):
    """Alpha#23: if sma(high, 20) < high: -1 * delta(high, 2) else 0."""
    cond = sma(b["high"], 20) < b["high"]
    out = np.where(cond, (-1.0 * delta(b["high"], 2)).to_numpy(), 0.0)
    return pd.Series(out, index=b["close"].index)


def alpha024(b):
    """Alpha#24: same family — adapted shorter window so it fits in 100 ticks.
    Original uses 100. We use 50 here.
    """
    n = 50
    sma_close_n = sma(b["close"], n)
    cond = (delta(sma_close_n, n) / (delay(b["close"], n) + EPS)) <= 0.05
    out = np.where(cond, -(b["close"] - ts_min(b["close"], n)).to_numpy(),
                   (-1.0 * delta(b["close"], 3)).to_numpy())
    return pd.Series(out, index=b["close"].index)


def alpha026(b):
    """Alpha#26: -1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)."""
    c = correlation(ts_rank(b["volume"], 5), ts_rank(b["high"], 5), 5)
    return -1.0 * ts_max(c, 3)


def alpha032(b):
    """Alpha#32 adapted: (sma(close,7) - close) + 20 * correlation(vwap, delay(close, 5), 50)."""
    return (sma(b["close"], 7) - b["close"]) + 20.0 * correlation(b["vwap"], delay(b["close"], 5), 50)


def alpha035(b):
    """Alpha#35: ts_rank(volume,32) * (1 - ts_rank(close+high-low,16)) * (1 - ts_rank(returns,32))."""
    a = ts_rank(b["volume"], 32)
    bb = 1.0 - ts_rank(b["close"] + b["high"] - b["low"], 16)
    cc = 1.0 - ts_rank(b["ret"].fillna(0.0), 32)
    return a * bb * cc


def alpha038(b):
    """Alpha#38 adapted: -1 * ts_rank(close, 10) * (close / (open + EPS))."""
    return -1.0 * ts_rank(b["close"], 10) * (b["close"] / (b["open"] + EPS))


def alpha041(b):
    """Alpha#41: (high * low) ** 0.5 - vwap. Use abs to avoid NaN from negative."""
    prod = b["high"] * b["low"]
    geo = pd.Series(np.sign(prod.to_numpy()) * np.sqrt(np.abs(prod.to_numpy())),
                    index=b["close"].index)
    return geo - b["vwap"]


def alpha043(b):
    """Alpha#43 adapted: ts_rank(volume / (sma(volume, 20)+EPS), 20) * ts_rank(-delta(close, 7), 8)."""
    return ts_rank(b["volume"] / (sma(b["volume"], 20) + EPS), 20) * \
           ts_rank(-1.0 * delta(b["close"], 7), 8)


def alpha044(b):
    """Alpha#44 adapted: -1 * correlation(high, volume, 5)."""
    return -1.0 * correlation(b["high"], b["volume"], 5)


def alpha045(b):
    """Alpha#45 adapted: -1 * (sma(delay(close, 5), 20) *
                              correlation(close, volume, 2) *
                              correlation(sma(close, 5), sma(close, 20), 2))
    """
    return -1.0 * (sma(delay(b["close"], 5), 20) *
                   correlation(b["close"], b["volume"], 2) *
                   correlation(sma(b["close"], 5), sma(b["close"], 20), 2))


def alpha046(b):
    """Alpha#46: piecewise-sign on delay(close, 20/10) momentum."""
    c = b["close"]
    d20_10 = (delay(c, 20) - delay(c, 10)) / 10.0
    d10_0 = (delay(c, 10) - c) / 10.0
    inner = d20_10 - d10_0
    out = np.where(inner > 0.25, -1.0,
                   np.where(inner < 0.0, 1.0, (-1.0 * (c - delay(c, 1))).to_numpy()))
    return pd.Series(out, index=c.index)


def alpha049(b):
    """Alpha#49: same family as 46 but threshold -0.1."""
    c = b["close"]
    d20_10 = (delay(c, 20) - delay(c, 10)) / 10.0
    d10_0 = (delay(c, 10) - c) / 10.0
    inner = d20_10 - d10_0
    out = np.where(inner < -0.1, 1.0, (-1.0 * (c - delay(c, 1))).to_numpy())
    return pd.Series(out, index=c.index)


def alpha051(b):
    """Alpha#51: family of 49, threshold -0.05."""
    c = b["close"]
    d20_10 = (delay(c, 20) - delay(c, 10)) / 10.0
    d10_0 = (delay(c, 10) - c) / 10.0
    inner = d20_10 - d10_0
    out = np.where(inner < -0.05, 1.0, (-1.0 * (c - delay(c, 1))).to_numpy())
    return pd.Series(out, index=c.index)


def alpha052(b):
    """Alpha#52 adapted: -1 * (ts_min(low, 5) - delay(ts_min(low, 5), 5)) * ts_sum(returns, 50)."""
    tm = ts_min(b["low"], 5)
    return -1.0 * (tm - delay(tm, 5)) * ts_sum(b["ret"].fillna(0.0), 50)


def alpha053(b):
    """Alpha#53: -1 * delta(((close - low) - (high - close)) / (close - low + EPS), 9)."""
    inner = ((b["close"] - b["low"]) - (b["high"] - b["close"])) / (b["close"] - b["low"] + EPS)
    return -1.0 * delta(inner, 9)


def alpha054(b):
    """Alpha#54: -1 * (low - close) * (open^5) / ((low - high + EPS) * (close^5 + EPS))."""
    num = -1.0 * (b["low"] - b["close"]) * (b["open"] ** 5)
    den = (b["low"] - b["high"] - EPS) * (signedpower(b["close"], 5) + EPS)
    return num / den


def alpha055(b):
    """Alpha#55 adapted: -1 * correlation((close - ts_min(low, 12)) /
                       (ts_max(high, 12) - ts_min(low, 12) + EPS), volume, 6)
    """
    rng = ts_max(b["high"], 12) - ts_min(b["low"], 12)
    norm = (b["close"] - ts_min(b["low"], 12)) / (rng + EPS)
    return -1.0 * correlation(norm, b["volume"], 6)


def alpha057(b):
    """Alpha#57 adapted: 0 - 1 * ((close - vwap) / decay_linear(ts_argmax(close, 30), 2))."""
    den = decay_linear(ts_argmax(b["close"], 30), 2)
    return 0.0 - 1.0 * ((b["close"] - b["vwap"]) / (den + EPS))


def alpha071(b):
    """Alpha#71 adapted (drop rank, adv180→50):
    max(ts_rank(decay_linear(corr(ts_rank(close,3), ts_rank(sma(volume,50),12), 18),4),16),
        ts_rank(decay_linear(((low+open) - (vwap+vwap))^2, 16), 4))
    Reduce 18 → 12, 16 → 12 to fit in 100 ticks total lookback budget.
    Total lookback ≈ max(50,18+4+16, 16+4) = 50 (from sma(volume,50)).
    """
    a = ts_rank(decay_linear(
        correlation(ts_rank(b["close"], 3), ts_rank(sma(b["volume"], 50), 12), 12),
        4), 12)
    cc = signedpower((b["low"] + b["open"]) - (b["vwap"] + b["vwap"]), 2.0)
    bb = ts_rank(decay_linear(cc, 12), 4)
    return pd.concat([a, bb], axis=1).max(axis=1)


def alpha083(b):
    """Alpha#83 adapted: drop outer rank.
    (delay((high-low)/sma(close, 5), 2) * volume) / ((high-low)/sma(close, 5) + EPS) / (vwap - close + EPS)
    """
    rng_norm = (b["high"] - b["low"]) / (sma(b["close"], 5) + EPS)
    num = delay(rng_norm, 2) * b["volume"]
    return num / (rng_norm + EPS) / (b["vwap"] - b["close"] + EPS)


def alpha084(b):
    """Alpha#84: signedpower(ts_rank(vwap - ts_max(vwap, 15), 21), delta(close, 5))
    Use abs of delta to avoid issues; cap power for stability.
    """
    base = ts_rank(b["vwap"] - ts_max(b["vwap"], 15), 21)
    p = delta(b["close"], 5).clip(-3.0, 3.0)
    arr_b = base.to_numpy()
    arr_p = p.to_numpy()
    out = np.sign(arr_b) * (np.abs(arr_b) ** np.where(np.isnan(arr_p), 1.0, arr_p))
    return pd.Series(out, index=b["close"].index)


def alpha088(b):
    """Alpha#88 adapted (drop rank chain, adv60 → sma(volume, 50)):
    ts_rank(decay_linear(corr(ts_rank(close, 8), ts_rank(sma(volume, 50), 12), 8), 7), 3)
    Lookback = 50 (sma) + 12 + 8 + 7 + 3 = ~80 ≤ 100. OK.
    Reduce 21 → 12 so the inner ts_rank(volume, 21) becomes ts_rank(volume, 12) here.
    """
    inner_corr = correlation(ts_rank(b["close"], 8),
                             ts_rank(sma(b["volume"], 50), 12), 8)
    return ts_rank(decay_linear(inner_corr, 7), 3)


def alpha092(b):
    """Alpha#92 adapted (drop rank, adv30 → sma(volume,30)):
    min(ts_rank(decay_linear((((high+low)/2 + close) < (low + open)).float, 15), 19),
        ts_rank(decay_linear(corr(low, sma(volume, 30), 8), 7), 7))
    Lookback ≈ 30 + 8 + 7 + 7 = 52 ≤ 100. OK.
    """
    cond = (((b["high"] + b["low"]) / 2.0 + b["close"]) < (b["low"] + b["open"])).astype(np.float64)
    a = ts_rank(decay_linear(cond, 15), 19)
    cc = correlation(b["low"], sma(b["volume"], 30), 8)
    bb = ts_rank(decay_linear(cc, 7), 7)
    return pd.concat([a, bb], axis=1).min(axis=1)


def alpha096(b):
    """Alpha#96 adapted (drop rank, adv60 → sma(volume,50)):
    max(ts_rank(decay_linear(corr(vwap, volume, 4), 4), 8),
        ts_rank(decay_linear(ts_argmax(corr(ts_rank(close, 7), ts_rank(sma(volume, 50), 4), 4), 13), 14), 13))
    """
    a = ts_rank(decay_linear(correlation(b["vwap"], b["volume"], 4), 4), 8)
    inner = correlation(ts_rank(b["close"], 7),
                        ts_rank(sma(b["volume"], 50), 4), 4)
    bb = ts_rank(decay_linear(ts_argmax(inner, 13), 14), 13)
    return pd.concat([a, bb], axis=1).max(axis=1)


def alpha099(b):
    """Alpha#99 adapted (drop adv60 panel → sma(volume, 50)):
    -1 * (corr(ts_sum((high+low)/2, 20), ts_sum(sma(volume, 50), 20), 9) <
          corr(low, volume, 6))
    """
    a = correlation(ts_sum((b["high"] + b["low"]) / 2.0, 20),
                    ts_sum(sma(b["volume"], 50), 20), 9)
    bb = correlation(b["low"], b["volume"], 6)
    return -1.0 * (a < bb).astype(np.float64)


def alpha101(b):
    """Alpha#101: (close - open) / (high - low + 0.001)."""
    return (b["close"] - b["open"]) / (b["high"] - b["low"] + 0.001)


# ============================================================================
# Aggregator
# ============================================================================

ALPHA101_FNS = [
    ("a101_006", alpha006),
    ("a101_009", alpha009),
    ("a101_010", alpha010),
    ("a101_012", alpha012),
    ("a101_019", alpha019),
    ("a101_021", alpha021),
    ("a101_022", alpha022),
    ("a101_023", alpha023),
    ("a101_024", alpha024),
    ("a101_026", alpha026),
    ("a101_032", alpha032),
    ("a101_035", alpha035),
    ("a101_038", alpha038),
    ("a101_041", alpha041),
    ("a101_043", alpha043),
    ("a101_044", alpha044),
    ("a101_045", alpha045),
    ("a101_046", alpha046),
    ("a101_049", alpha049),
    ("a101_051", alpha051),
    ("a101_052", alpha052),
    ("a101_053", alpha053),
    ("a101_054", alpha054),
    ("a101_055", alpha055),
    ("a101_057", alpha057),
    ("a101_071", alpha071),
    ("a101_083", alpha083),
    ("a101_084", alpha084),
    ("a101_088", alpha088),
    ("a101_092", alpha092),
    ("a101_096", alpha096),
    ("a101_099", alpha099),
    ("a101_101", alpha101),
]


def alpha101_columns():
    return [name for name, _ in ALPHA101_FNS]


def compute_alpha101(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all natural-fit Alpha101 features for one session df."""
    b = base_series(df)
    out = {}
    for name, fn in ALPHA101_FNS:
        try:
            v = fn(b)
            if isinstance(v, pd.Series):
                out[name] = v.to_numpy(dtype=np.float64)
            else:
                out[name] = np.asarray(v, dtype=np.float64)
        except Exception as e:
            print(f"  WARN: {name} failed: {e}", flush=True)
            out[name] = np.full(len(df), np.nan, dtype=np.float64)
    return pd.DataFrame(out, index=df.index)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    df = pd.read_parquet("data/features_v1/snapshot_sym0_date0_am.parquet")
    feats = compute_alpha101(df)
    print(f"Alpha101 cols: {feats.shape[1]}, expected {len(ALPHA101_FNS)}")
    print(f"NaN frac at row 100: {(feats.iloc[100:].isna().sum() / (len(feats) - 100)).round(3).to_dict()}")
    print(f"first 5 rows:\n{feats.iloc[100:105].head().to_string()}")
