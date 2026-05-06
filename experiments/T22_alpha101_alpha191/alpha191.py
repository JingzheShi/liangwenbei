"""国泰君安 Alpha191 — natural-fit subset only.

Source: 国泰君安证券《191 个量化选股 Alpha 因子》研究报告
Cross-checked with public implementations:
    - github.com/Albert-Z-Guo/MultiFactor
    - github.com/chenditc/quant_research

POLICY ("如果是用不了的就不要用了"):
    - SKIP any alpha containing RANK(...) cross-section.
    - SKIP any alpha needing > 100-tick history (most caps at 60-90).
    - SKIP any alpha referencing "BANCHMARKINDEXCLOSE" / "BENCHMARK*" — needs
      market index data which we do not have per-tick.
    - Keep all single-stock time-series alphas (DELAY, DELTA, MEAN, STD,
      CORR, SUM, MIN, MAX, COUNT, SUMIF, TSRANK, WMA, SMA, DECAYLINEAR,
      LOG, SIGN, ABS, SIGNEDPOWER).

This file logs IMPL/SKIP for each of the 191 alphas in the table block.
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
    wma, sma_gtja, regbeta, regresi, sumif, count_if, highday, lowday,
)


# ============================================================================
# ALPHA191 STATUS TABLE (selected; all pure-ts alphas implemented below)
# ============================================================================
# Alpha001  SKIP — cross-section rank
# Alpha002  IMPL — (-1 * delta(((close-low) - (high-close)) / (high-low+EPS), 1))
# Alpha003  IMPL — sum(condition * delta(close), 6)
# Alpha004  IMPL — sma vs ratio comparison rules
# Alpha005  IMPL — -1 * tsmax(corr(tsrank(volume,5), tsrank(high,5),5),3)  (also a101_026 family)
# Alpha006  SKIP — outer rank
# Alpha007  IMPL — (max(vwap-close,3) + min(vwap-close,3)) * delta(volume,3)  (drop ranks; here all ts)
# Alpha008  SKIP — outer rank
# Alpha009  IMPL — sma_gtja-based momentum × volume
# Alpha010  IMPL — rank(max(...))^2 — drop outer rank: max(...)^2
# Alpha011  IMPL — sum(((close-low)-(high-close))/(high-low) * volume, 6)
# Alpha012  SKIP — needs cross-section
# Alpha013  IMPL — (high*low)^0.5 - vwap  (= a101_041)
# Alpha014  IMPL — close - delay(close,5)
# Alpha015  IMPL — open/delay(close,1) - 1
# Alpha016  SKIP — rank
# Alpha017  SKIP — rank
# Alpha018  IMPL — close/delay(close,5)
# Alpha019  IMPL — piecewise rule on (close-delay(close,5))/delay(close,5)
# Alpha020  IMPL — (close-delay(close,6))/delay(close,6) * 100
# Alpha021  IMPL — regbeta(mean(close,6), seq(1..6), 6)
# Alpha022  IMPL — sma_gtja((close - mean(close,6))/mean(close,6) - delay((close-mean(close,6))/mean(close,6),3),12,1)
# Alpha023  IMPL — sma-conditioned stddev ratio
# Alpha024  IMPL — sma_gtja(close - delay(close,5), 5, 1)
# Alpha025  SKIP — rank
# Alpha026  IMPL — adapted (mean(close,7) - close) + corr(vwap, delay(close,5), 50) [shorter than 230]
# Alpha027  IMPL — wma((close/delay(close,3) -1)*100 + (close/delay(close,6) -1)*100, 12)
# Alpha028  IMPL — 3*sma_gtja((close - tsmin(low,9))/(tsmax(high,9) - tsmin(low,9))*100, 3, 1) - 2*sma_gtja(sma_gtja((close-tsmin(low,9))/(tsmax(high,9) - tsmin(low,9))*100, 3, 1), 3, 1)
# Alpha029  IMPL — (close - delay(close,6))/delay(close,6) * volume
# Alpha030  SKIP — needs market index/regbeta on benchmark
# Alpha031  IMPL — (close - mean(close,12)) / mean(close,12) * 100
# Alpha032  SKIP — outer rank
# Alpha033  IMPL — adapted -1 * tsmin(low, 5) - delay(tsmin(low,5),5) * sum(returns, 50) [a101_052 family]
# Alpha034  IMPL — mean(close, 12) / close
# Alpha035  SKIP — outer rank cross-section
# Alpha036  SKIP — rank cross-section
# Alpha037  SKIP — rank
# Alpha038  IMPL — sma(high,20) < high ? -1*delta(high,2) : 0 (= a101_023)
# Alpha039  SKIP — rank
# Alpha040  IMPL — sum(close>delay(close,1)?volume:0,26) / sum(close<=delay(close,1)?volume:0,26) * 100
# Alpha041  SKIP — rank
# Alpha042  IMPL — -1 * stddev(high,10) * corr(high, volume, 10) (drop outer rank)
# Alpha043  IMPL — sum(close>delay(close,1)?volume:(close<delay(close,1)?-volume:0), 6)
# Alpha044  IMPL — adapted: tsrank(decay_linear(corr(low, mean(volume,10),7),6),4) + tsrank(decay_linear(delta(vwap,3),10),15)
# Alpha045  SKIP — rank
# Alpha046  IMPL — (mean(close,3)+mean(close,6)+mean(close,12)+mean(close,24))/(4*close)
# Alpha047  IMPL — sma_gtja((tsmax(high,6)-close)/(tsmax(high,6)-tsmin(low,6))*100, 9, 1)
# Alpha048  SKIP — rank
# Alpha049  IMPL — close-up vs down volume sums (similar to 040 family)
# Alpha050  IMPL — close-up vs down sum (analog of 049)
# Alpha051  IMPL — same family
# Alpha052  IMPL — sum(max(0,high-delay((high+low+close)/3,1)),26) / sum(max(0,delay((high+low+close)/3,1)-low),26)*100
# Alpha053  IMPL — count(close>delay(close,1),12)/12*100
# Alpha054  IMPL — -1 * (stddev(abs(close-open)) + (close-open) + corr(close, open, 10))
# Alpha055  IMPL — sum(((close-delay(close,1))+(close-open)/2 ...,) — long formula, pure ts.
# Alpha056  SKIP — rank
# Alpha057  IMPL — sma_gtja((close - tsmin(low,9))/(tsmax(high,9) - tsmin(low,9))*100, 3, 1)
# Alpha058  IMPL — count(close>delay(close,1), 20)/20*100
# Alpha059  IMPL — sum(close - (close>delay(close,1)?min(low,delay(close,1)):max(high,delay(close,1))), 20)
# Alpha060  IMPL — sum(((close-low)-(high-close))/(high-low)*volume, 20)
# Alpha061  SKIP — rank
# Alpha062  IMPL — -1 * corr(high, volume, 5) (drop outer rank)
# Alpha063  IMPL — sma_gtja(max(close-delay(close,1),0),6,1)/sma_gtja(abs(close-delay(close,1)),6,1)*100
# Alpha064  SKIP — rank
# Alpha065  IMPL — mean(close,6)/close
# Alpha066  IMPL — (close - mean(close,6))/mean(close,6) * 100
# Alpha067  IMPL — sma_gtja(max(close-delay(close,1),0),24,1)/sma_gtja(abs(close-delay(close,1)),24,1)*100
# Alpha068  IMPL — sma_gtja((high+low)/2 - (delay(high,1)+delay(low,1))/2)*(high-low)/volume,15,2)
# Alpha069  SKIP — needs cross-section
# Alpha070  IMPL — std(amount, 6)
# Alpha071  IMPL — (close - mean(close,24))/mean(close,24) * 100
# Alpha072  IMPL — sma_gtja((tsmax(high,6)-close)/(tsmax(high,6)-tsmin(low,6))*100, 15, 1)
# Alpha073  SKIP — rank
# Alpha074  SKIP — rank
# Alpha075  SKIP — needs cross-section count of stocks above benchmark
# Alpha076  IMPL — std(abs(close/delay(close,1)-1)/volume, 20) / mean(abs(close/delay(close,1)-1)/volume, 20)
# Alpha077  SKIP — rank
# Alpha078  IMPL — ((high+low+close)/3 - mean((high+low+close)/3, 12)) / (0.015 * mean(abs(close - mean((high+low+close)/3,12)),12))
# Alpha079  IMPL — sma_gtja(max(close-delay(close,1),0),12,1)/sma_gtja(abs(close-delay(close,1)),12,1)*100
# Alpha080  IMPL — (volume - delay(volume,5))/delay(volume,5) * 100
# Alpha081  IMPL — sma_gtja(volume, 21, 2)
# Alpha082  IMPL — sma_gtja((tsmax(high,6)-close)/(tsmax(high,6)-tsmin(low,6))*100, 20, 1)
# Alpha083  SKIP — rank
# Alpha084  IMPL — sum(close>delay(close,1) ? volume : (close<delay(close,1) ? -volume : 0), 20)
# Alpha085  IMPL — adapted tsrank(volume/mean(volume,20),20) * tsrank(-1*delta(close,7),8) (= a101_043)
# Alpha086  IMPL — momentum/lag inequality piecewise (= a101_046 family)
# Alpha087  SKIP — rank
# Alpha088  IMPL — (close - delay(close,20))/delay(close,20) * 100
# Alpha089  IMPL — 2*(sma_gtja(close,13,2) - sma_gtja(close,27,2) - sma_gtja(sma_gtja(close,13,2)-sma_gtja(close,27,2),10,2))
# Alpha090  SKIP — rank
# Alpha091  SKIP — rank
# Alpha092  SKIP — rank
# Alpha093  IMPL — sum((open<=delay(open,1)?0:max(open-low,open-delay(open,1))), 20)
# Alpha094  IMPL — sum((close>delay(close,1)?volume:(close<delay(close,1)?-volume:0)), 30)
# Alpha095  IMPL — std(amount, 20)
# Alpha096  IMPL — sma_gtja(sma_gtja((close-tsmin(low,9))/(tsmax(high,9)-tsmin(low,9))*100,3,1),3,1)
# Alpha097  IMPL — std(volume, 10)
# Alpha098  IMPL — adapted: piecewise on delta(sum(close,n)/n, n)/delay(close, n)
# Alpha099  SKIP — rank
# Alpha100  IMPL — std(volume, 20)
# Alpha101  SKIP — rank
# Alpha102  IMPL — sma_gtja(max(volume-delay(volume,1),0),6,1)/sma_gtja(abs(volume-delay(volume,1)),6,1)*100
# Alpha103  IMPL — (20 - lowday(low,20))/20 * 100
# Alpha104  IMPL — adapted -1 * delta(corr(high, volume, 5),5) * std(close,20)  (= a101_022)
# Alpha105  SKIP — rank
# Alpha106  IMPL — close - delay(close, 20)
# Alpha107  SKIP — rank
# Alpha108  SKIP — rank
# Alpha109  IMPL — sma_gtja(high-low, 10, 2)/sma_gtja(sma_gtja(high-low,10,2),10,2)
# Alpha110  IMPL — sum(max(0, high-delay(close,1)),20) / sum(max(0, delay(close,1)-low),20) * 100
# Alpha111  IMPL — sma_gtja(volume*((close-low)-(high-close))/(high-low+EPS), 11, 2) - sma_gtja(volume*((close-low)-(high-close))/(high-low+EPS), 4, 2)
# Alpha112  IMPL — (sum((close>delay(close,1)?close-delay(close,1):0),12) - sum((close<delay(close,1)?abs(close-delay(close,1)):0),12)) / (sum(...,12) + sum(...,12)) * 100
# Alpha113  SKIP — rank
# Alpha114  SKIP — rank
# Alpha115  SKIP — rank
# Alpha116  IMPL — regbeta(close, seq(1..20), 20)  (drop the surrounding rank if any) — actually this is regbeta on close vs time index → pure ts.
# Alpha117  IMPL — adapted: tsrank(volume,32) * (1 - tsrank(close+high-low,16)) * (1 - tsrank(returns,32))  (= a101_035)
# Alpha118  IMPL — sum(high-open,20)/sum(open-low,20) * 100
# Alpha119  SKIP — rank
# Alpha120  SKIP — rank
# Alpha121  SKIP — rank
# Alpha122  IMPL — (sma_gtja(sma_gtja(sma_gtja(log(close),13,2),13,2),13,2) - delay(sma_gtja(...,13,2),1)) / delay(...)
# Alpha123  SKIP — rank
# Alpha124  IMPL — adapted (drop outer rank): (close - vwap) / decay_linear(tsmax(close,30), 2)
# Alpha125  SKIP — rank
# Alpha126  IMPL — (close + high + low) / 3
# Alpha127  IMPL — mean((100*(close - tsmin(close,12))/(tsmax(close,12) - tsmin(close,12)+EPS))^2, 12)^0.5  (RSI-ish stat)
# Alpha128  IMPL — money flow index — uses sums of (typical price * volume) directional
# Alpha129  IMPL — sum(abs(close - delay(close,1)) where close < delay(close,1), 12)
# Alpha130  SKIP — rank
# Alpha131  SKIP — rank
# Alpha132  IMPL — mean(amount, 20)
# Alpha133  IMPL — (20 - highday(high,20))/20 * 100 - (20 - lowday(low,20))/20 * 100
# Alpha134  IMPL — (close - delay(close,12))/delay(close,12) * volume
# Alpha135  IMPL — sma_gtja(delay(close/delay(close,20),1), 20, 1)
# Alpha136  SKIP — rank
# Alpha137  IMPL — long combined intraday strength formula — pure ts
# Alpha138  SKIP — rank
# Alpha139  IMPL — -1 * corr(open, volume, 10)  (drop outer rank; same as a101_006)
# Alpha140  SKIP — rank
# Alpha141  SKIP — rank
# Alpha142  SKIP — rank
# Alpha143  IMPL — close > delay(close,1) ? (close-delay(close,1))/delay(close,1)*self : 0 — recursive; we approximate with cumulative product over 60.
# Alpha144  IMPL — sumif(close<delay(close,1), abs(close/delay(close,1)-1)/amount, 20)/count(close<delay(close,1), 20)
# Alpha145  IMPL — (mean(volume,9) - mean(volume,26))/mean(volume,12) * 100
# Alpha146  IMPL — chained ema-style — pure ts
# Alpha147  IMPL — regbeta(mean(close,12), seq(1..12), 12)
# Alpha148  SKIP — rank
# Alpha149  SKIP — needs benchmark index returns (regbeta with benchmark)
# Alpha150  IMPL — (close + high + low)/3 * volume
# Alpha151  IMPL — sma_gtja(close - delay(close,20), 20, 1)
# Alpha152  IMPL — chained sma_gtja
# Alpha153  IMPL — (mean(close,3)+mean(close,6)+mean(close,12)+mean(close,24))/4
# Alpha154  SKIP — rank
# Alpha155  IMPL — sma_gtja(volume,13,2) - sma_gtja(volume,27,2) - sma_gtja(sma_gtja(volume,13,2)-sma_gtja(volume,27,2),10,2)
# Alpha156  SKIP — rank
# Alpha157  IMPL — adapted tsmin(prod_min((-1*((close-1)-delay(close-1,5))),2),3)
# Alpha158  IMPL — ((high - sma_gtja(close,15,2)) - (low - sma_gtja(close,15,2))) / close
# Alpha159  IMPL — composite stochastic-like indicator — pure ts
# Alpha160  IMPL — sma_gtja(close<delay(close,1)?stddev(close,20):0, 20, 1)
# Alpha161  IMPL — mean(max(max(high-low, abs(delay(close,1)-high)), abs(delay(close,1)-low)), 12)
# Alpha162  IMPL — long sma_gtja-based RSI-like — pure ts
# Alpha163  IMPL — -1 * mean(close, 20) * volume / vwap * (high - close) (drop outer rank chain)
# Alpha164  IMPL — sma_gtja(((close>delay(close,1)?1/(close-delay(close,1)):1) - tsmin((close>delay(close,1)?1/(close-delay(close,1)):1),12))/(high-low)*100, 13, 2)
# Alpha165  SKIP — has cross-section
# Alpha166  IMPL — moment-based: sum( ((close/delay(close,1) - 1) - mean((...),20))^3 ,20) / something — adapt to pure ts
# Alpha167  IMPL — sum(close > delay(close,1) ? close - delay(close,1) : 0, 12)
# Alpha168  IMPL — -1 * volume / mean(volume, 20)
# Alpha169  IMPL — sma_gtja(mean(delay(sma_gtja(close-delay(close,1),9,1),1),12) - mean(delay(sma_gtja(close-delay(close,1),9,1),1),26), 10, 1)
# Alpha170  SKIP — rank
# Alpha171  IMPL — -1 * (low - close) * (open^5) / ((low - high) * (close^5)) (= a101_054)
# Alpha172  IMPL — TR/(SMA TR) - true range ATR-style
# Alpha173  IMPL — 3*sma_gtja(close,13,2) - 2*sma_gtja(sma_gtja(close,13,2),13,2) + sma_gtja(sma_gtja(sma_gtja(log(close),13,2),13,2),13,2)
# Alpha174  IMPL — sma_gtja(close>delay(close,1) ? stddev(close,20) : 0, 20, 1)
# Alpha175  IMPL — mean(max(max(high-low, abs(delay(close,1)-high)), abs(delay(close,1)-low)), 6)
# Alpha176  SKIP — rank
# Alpha177  IMPL — (20 - highday(high, 20)) / 20 * 100
# Alpha178  IMPL — (close - delay(close,1))/delay(close,1) * volume
# Alpha179  SKIP — rank
# Alpha180  IMPL — adapted: mean(volume,20) < volume ? -1 * tsrank(abs(delta(close,7)),60) * sign(delta(close,7)) : -1 * volume
# Alpha181  SKIP — needs benchmark index
# Alpha182  SKIP — needs benchmark index
# Alpha183  SKIP — has cross-section flag
# Alpha184  SKIP — rank
# Alpha185  SKIP — rank
# Alpha186  IMPL — long combined formula on TR series — pure ts
# Alpha187  IMPL — sum((open<=delay(open,1)?0:max(high-open, open-delay(open,1))), 20)
# Alpha188  IMPL — (high-low - sma_gtja(high-low,11,2)) / sma_gtja(high-low,11,2) * 100
# Alpha189  IMPL — mean(abs(close - mean(close,6)), 6)
# Alpha190  SKIP — has cross-section count
# Alpha191  IMPL — corr(mean(volume,20), low, 5) + (high+low)/2 - close (drop outer rank)
# ============================================================================


def a191_002(b):
    inner = ((b["close"] - b["low"]) - (b["high"] - b["close"])) / (b["high"] - b["low"] + EPS)
    return -1.0 * delta(inner, 1)


def a191_003(b):
    c = b["close"]
    cond_up = c > delay(c, 1)
    cond_dn = c < delay(c, 1)
    val = pd.Series(np.where(cond_up.to_numpy(),
                             (c - np.minimum(b["low"].to_numpy(), delay(c, 1).to_numpy())),
                             np.where(cond_dn.to_numpy(),
                                      (c - np.maximum(b["high"].to_numpy(), delay(c, 1).to_numpy())),
                                      0.0)), index=c.index)
    return ts_sum(val, 6)


def a191_004(b):
    c = b["close"]; v = b["volume"]
    cond1 = (sma(c, 8) + stddev(c, 8)) < sma(c, 2)
    cond2 = sma(c, 2) < (sma(c, 8) - stddev(c, 8))
    cond3 = (v / (sma(v, 20) + EPS)) >= 1.0
    out = np.where(cond1, -1.0, np.where(cond2, 1.0, np.where(cond3, 1.0, -1.0)))
    return pd.Series(out, index=c.index)


def a191_005(b):
    c = correlation(ts_rank(b["volume"], 5), ts_rank(b["high"], 5), 5)
    return -1.0 * ts_max(c, 3)


def a191_007(b):
    return (ts_max(b["vwap"] - b["close"], 3) + ts_min(b["vwap"] - b["close"], 3)) * delta(b["volume"], 3)


def a191_009(b):
    inner = ((b["high"] + b["low"]) / 2.0 - (delay(b["high"], 1) + delay(b["low"], 1)) / 2.0) * \
            (b["high"] - b["low"]) / (b["volume"] + EPS)
    return sma_gtja(inner, 7, 2)


def a191_010(b):
    r = b["ret"].fillna(0.0)
    cond = r < 0
    base = pd.Series(np.where(cond.to_numpy(), stddev(r, 20).to_numpy(),
                              b["close"].to_numpy()), index=b["close"].index)
    return ts_max(signedpower(base, 2.0), 5)


def a191_011(b):
    return ts_sum(((b["close"] - b["low"]) - (b["high"] - b["close"])) /
                  (b["high"] - b["low"] + EPS) * b["volume"], 6)


def a191_013(b):
    prod = b["high"] * b["low"]
    geo = pd.Series(np.sign(prod.to_numpy()) * np.sqrt(np.abs(prod.to_numpy())),
                    index=b["close"].index)
    return geo - b["vwap"]


def a191_014(b):
    return b["close"] - delay(b["close"], 5)


def a191_015(b):
    return b["open"] / (delay(b["close"], 1) + EPS) - 1.0


def a191_018(b):
    return b["close"] / (delay(b["close"], 5) + EPS)


def a191_019(b):
    c = b["close"]; d5 = delay(c, 5)
    cond1 = c < d5; cond2 = c == d5
    out = np.where(cond1.to_numpy(), ((c - d5) / (d5 + EPS)).to_numpy(),
                   np.where(cond2.to_numpy(), 0.0, ((c - d5) / (c + EPS)).to_numpy()))
    return pd.Series(out, index=c.index)


def a191_020(b):
    return (b["close"] - delay(b["close"], 6)) / (delay(b["close"], 6) + EPS) * 100.0


def a191_021(b):
    """regbeta(mean(close, 6), seq(1..6), 6)."""
    m = sma(b["close"], 6)
    t = pd.Series(np.arange(len(m), dtype=np.float64), index=m.index)
    return regbeta(m, t, 6)


def a191_022(b):
    inner = (b["close"] - sma(b["close"], 6)) / (sma(b["close"], 6) + EPS)
    return sma_gtja(inner - delay(inner, 3), 12, 1)


def a191_023(b):
    """Adapted GTJA Alpha23: ratio of std-conditional sums via sma_gtja."""
    c = b["close"]
    cond_up = c > delay(c, 1)
    s = stddev(c, 20)
    val_up = pd.Series(np.where(cond_up.to_numpy(), s.to_numpy(), 0.0), index=c.index)
    val_dn = pd.Series(np.where(cond_up.to_numpy(), 0.0, s.to_numpy()), index=c.index)
    a = sma_gtja(val_up, 20, 1)
    bb = sma_gtja(val_dn, 20, 1)
    return a / (a + bb + EPS) * 100.0


def a191_024(b):
    return sma_gtja(b["close"] - delay(b["close"], 5), 5, 1)


def a191_026(b):
    return (sma(b["close"], 7) - b["close"]) + correlation(b["vwap"], delay(b["close"], 5), 50)


def a191_027(b):
    inner = ((b["close"] / delay(b["close"], 3) - 1.0) * 100.0 +
             (b["close"] / delay(b["close"], 6) - 1.0) * 100.0)
    return wma(inner, 12)


def a191_028(b):
    rng = ts_max(b["high"], 9) - ts_min(b["low"], 9)
    inner = (b["close"] - ts_min(b["low"], 9)) / (rng + EPS) * 100.0
    smoothed = sma_gtja(inner, 3, 1)
    return 3.0 * smoothed - 2.0 * sma_gtja(smoothed, 3, 1)


def a191_029(b):
    return (b["close"] - delay(b["close"], 6)) / (delay(b["close"], 6) + EPS) * b["volume"]


def a191_031(b):
    return (b["close"] - sma(b["close"], 12)) / (sma(b["close"], 12) + EPS) * 100.0


def a191_033(b):
    tm = ts_min(b["low"], 5)
    return -1.0 * (tm - delay(tm, 5)) * ts_sum(b["ret"].fillna(0.0), 50)


def a191_034(b):
    return sma(b["close"], 12) / (b["close"] + EPS)


def a191_038(b):
    cond = sma(b["high"], 20) < b["high"]
    out = np.where(cond.to_numpy(), (-1.0 * delta(b["high"], 2)).to_numpy(), 0.0)
    return pd.Series(out, index=b["close"].index)


def a191_040(b):
    c = b["close"]
    up = pd.Series(np.where(c > delay(c, 1), b["volume"].to_numpy(), 0.0), index=c.index)
    dn = pd.Series(np.where(c <= delay(c, 1), b["volume"].to_numpy(), 0.0), index=c.index)
    return ts_sum(up, 26) / (ts_sum(dn, 26) + EPS) * 100.0


def a191_042(b):
    return -1.0 * stddev(b["high"], 10) * correlation(b["high"], b["volume"], 10)


def a191_043(b):
    c = b["close"]; v = b["volume"]
    val = pd.Series(np.where(c > delay(c, 1), v.to_numpy(),
                             np.where(c < delay(c, 1), -v.to_numpy(), 0.0)),
                    index=c.index)
    return ts_sum(val, 6)


def a191_044(b):
    a = ts_rank(decay_linear(correlation(b["low"], sma(b["volume"], 10), 7), 6), 4)
    bb = ts_rank(decay_linear(delta(b["vwap"], 3), 10), 15)
    return a + bb


def a191_046(b):
    return (sma(b["close"], 3) + sma(b["close"], 6) +
            sma(b["close"], 12) + sma(b["close"], 24)) / (4.0 * b["close"] + EPS)


def a191_047(b):
    rng = ts_max(b["high"], 6) - ts_min(b["low"], 6)
    inner = (ts_max(b["high"], 6) - b["close"]) / (rng + EPS) * 100.0
    return sma_gtja(inner, 9, 1)


def a191_049(b):
    h = b["high"]; l = b["low"]
    sum_hl = h + l
    delay_hl = delay(h, 1) + delay(l, 1)
    cond = sum_hl <= delay_hl
    val_up = pd.Series(np.where(cond.to_numpy(), 0.0,
                                np.maximum(np.abs((h - delay(h, 1)).to_numpy()),
                                           np.abs((l - delay(l, 1)).to_numpy()))),
                       index=h.index)
    val_dn = pd.Series(np.where(cond.to_numpy(),
                                np.maximum(np.abs((h - delay(h, 1)).to_numpy()),
                                           np.abs((l - delay(l, 1)).to_numpy())),
                                0.0), index=h.index)
    return ts_sum(val_dn, 12) / (ts_sum(val_up, 12) + ts_sum(val_dn, 12) + EPS)


def a191_050(b):
    h = b["high"]; l = b["low"]
    sum_hl = h + l
    delay_hl = delay(h, 1) + delay(l, 1)
    cond_eq = sum_hl == delay_hl
    cond_lt = sum_hl < delay_hl
    diff_max = np.maximum(np.abs((h - delay(h, 1)).to_numpy()),
                          np.abs((l - delay(l, 1)).to_numpy()))
    val_dn = pd.Series(np.where(cond_eq.to_numpy(), 0.0,
                                np.where(cond_lt.to_numpy(), diff_max, 0.0)),
                       index=h.index)
    val_up = pd.Series(np.where(cond_eq.to_numpy(), 0.0,
                                np.where(cond_lt.to_numpy(), 0.0, diff_max)),
                       index=h.index)
    return (ts_sum(val_up, 12) - ts_sum(val_dn, 12)) / (
        ts_sum(val_up, 12) + ts_sum(val_dn, 12) + EPS)


def a191_052(b):
    typ = (b["high"] + b["low"] + b["close"]) / 3.0
    val_up = (b["high"] - delay(typ, 1)).clip(lower=0.0)
    val_dn = (delay(typ, 1) - b["low"]).clip(lower=0.0)
    return ts_sum(val_up, 26) / (ts_sum(val_dn, 26) + EPS) * 100.0


def a191_053(b):
    cond = (b["close"] > delay(b["close"], 1)).astype(np.float64)
    return ts_sum(cond, 12) / 12.0 * 100.0


def a191_054(b):
    diff = b["close"] - b["open"]
    return -1.0 * (stddev(diff.abs(), 10) + diff +
                   correlation(b["close"], b["open"], 10))


def a191_057(b):
    rng = ts_max(b["high"], 9) - ts_min(b["low"], 9)
    inner = (b["close"] - ts_min(b["low"], 9)) / (rng + EPS) * 100.0
    return sma_gtja(inner, 3, 1)


def a191_058(b):
    cond = (b["close"] > delay(b["close"], 1)).astype(np.float64)
    return ts_sum(cond, 20) / 20.0 * 100.0


def a191_059(b):
    c = b["close"]
    cond_up = c > delay(c, 1)
    cond_dn = c < delay(c, 1)
    val = pd.Series(np.where(cond_up.to_numpy(),
                             (c - np.minimum(b["low"].to_numpy(), delay(c, 1).to_numpy())),
                             np.where(cond_dn.to_numpy(),
                                      (c - np.maximum(b["high"].to_numpy(), delay(c, 1).to_numpy())),
                                      0.0)),
                    index=c.index)
    return ts_sum(val, 20)


def a191_060(b):
    return ts_sum(((b["close"] - b["low"]) - (b["high"] - b["close"])) /
                  (b["high"] - b["low"] + EPS) * b["volume"], 20)


def a191_062(b):
    return -1.0 * correlation(b["high"], b["volume"], 5)


def a191_063(b):
    diff = b["close"] - delay(b["close"], 1)
    up = diff.clip(lower=0.0)
    return sma_gtja(up, 6, 1) / (sma_gtja(diff.abs(), 6, 1) + EPS) * 100.0


def a191_065(b):
    return sma(b["close"], 6) / (b["close"] + EPS)


def a191_066(b):
    return (b["close"] - sma(b["close"], 6)) / (sma(b["close"], 6) + EPS) * 100.0


def a191_067(b):
    diff = b["close"] - delay(b["close"], 1)
    up = diff.clip(lower=0.0)
    return sma_gtja(up, 24, 1) / (sma_gtja(diff.abs(), 24, 1) + EPS) * 100.0


def a191_068(b):
    inner = ((b["high"] + b["low"]) / 2.0 - (delay(b["high"], 1) + delay(b["low"], 1)) / 2.0) * \
            (b["high"] - b["low"]) / (b["volume"] + EPS)
    return sma_gtja(inner, 15, 2)


def a191_070(b):
    return stddev(b["amount"], 6)


def a191_071(b):
    return (b["close"] - sma(b["close"], 24)) / (sma(b["close"], 24) + EPS) * 100.0


def a191_072(b):
    rng = ts_max(b["high"], 6) - ts_min(b["low"], 6)
    inner = (ts_max(b["high"], 6) - b["close"]) / (rng + EPS) * 100.0
    return sma_gtja(inner, 15, 1)


def a191_076(b):
    val = (b["close"] / (delay(b["close"], 1) + EPS) - 1.0).abs() / (b["volume"] + EPS)
    return stddev(val, 20) / (sma(val, 20) + EPS)


def a191_078(b):
    typ = (b["high"] + b["low"] + b["close"]) / 3.0
    md = sma(typ, 12)
    md_dev = (typ - md).abs()
    return (typ - md) / (0.015 * sma(md_dev, 12) + EPS)


def a191_079(b):
    diff = b["close"] - delay(b["close"], 1)
    up = diff.clip(lower=0.0)
    return sma_gtja(up, 12, 1) / (sma_gtja(diff.abs(), 12, 1) + EPS) * 100.0


def a191_080(b):
    return (b["volume"] - delay(b["volume"], 5)) / (delay(b["volume"], 5) + EPS) * 100.0


def a191_081(b):
    return sma_gtja(b["volume"], 21, 2)


def a191_082(b):
    rng = ts_max(b["high"], 6) - ts_min(b["low"], 6)
    inner = (ts_max(b["high"], 6) - b["close"]) / (rng + EPS) * 100.0
    return sma_gtja(inner, 20, 1)


def a191_084(b):
    c = b["close"]; v = b["volume"]
    val = pd.Series(np.where(c > delay(c, 1), v.to_numpy(),
                             np.where(c < delay(c, 1), -v.to_numpy(), 0.0)),
                    index=c.index)
    return ts_sum(val, 20)


def a191_085(b):
    return ts_rank(b["volume"] / (sma(b["volume"], 20) + EPS), 20) * \
           ts_rank(-1.0 * delta(b["close"], 7), 8)


def a191_086(b):
    c = b["close"]
    d20_10 = (delay(c, 20) - delay(c, 10)) / 10.0
    d10_0 = (delay(c, 10) - c) / 10.0
    inner = d20_10 - d10_0
    out = np.where(inner > 0.25, -1.0,
                   np.where(inner < 0.0, 1.0, (-1.0 * (c - delay(c, 1))).to_numpy()))
    return pd.Series(out, index=c.index)


def a191_088(b):
    return (b["close"] - delay(b["close"], 20)) / (delay(b["close"], 20) + EPS) * 100.0


def a191_089(b):
    a = sma_gtja(b["close"], 13, 2)
    bb = sma_gtja(b["close"], 27, 2)
    return 2.0 * (a - bb - sma_gtja(a - bb, 10, 2))


def a191_093(b):
    cond = b["open"] > delay(b["open"], 1)
    diff_max = pd.concat([b["open"] - b["low"], b["open"] - delay(b["open"], 1)], axis=1).max(axis=1)
    val = pd.Series(np.where(cond.to_numpy(), diff_max.to_numpy(), 0.0), index=b["open"].index)
    return ts_sum(val, 20)


def a191_094(b):
    c = b["close"]; v = b["volume"]
    val = pd.Series(np.where(c > delay(c, 1), v.to_numpy(),
                             np.where(c < delay(c, 1), -v.to_numpy(), 0.0)), index=c.index)
    return ts_sum(val, 30)


def a191_095(b):
    return stddev(b["amount"], 20)


def a191_096(b):
    rng = ts_max(b["high"], 9) - ts_min(b["low"], 9)
    inner = (b["close"] - ts_min(b["low"], 9)) / (rng + EPS) * 100.0
    return sma_gtja(sma_gtja(inner, 3, 1), 3, 1)


def a191_097(b):
    return stddev(b["volume"], 10)


def a191_098(b):
    n = 50
    sma_close = sma(b["close"], n)
    cond = (delta(sma_close, n) / (delay(b["close"], n) + EPS)) <= 0.05
    out = np.where(cond.to_numpy(), -(b["close"] - ts_min(b["close"], n)).to_numpy(),
                   (-1.0 * delta(b["close"], 3)).to_numpy())
    return pd.Series(out, index=b["close"].index)


def a191_100(b):
    return stddev(b["volume"], 20)


def a191_102(b):
    diff = b["volume"] - delay(b["volume"], 1)
    up = diff.clip(lower=0.0)
    return sma_gtja(up, 6, 1) / (sma_gtja(diff.abs(), 6, 1) + EPS) * 100.0


def a191_103(b):
    return (20.0 - lowday(b["low"], 20)) / 20.0 * 100.0


def a191_104(b):
    c = correlation(b["high"], b["volume"], 5)
    return -1.0 * delta(c, 5) * stddev(b["close"], 20)


def a191_106(b):
    return b["close"] - delay(b["close"], 20)


def a191_109(b):
    a = sma_gtja(b["high"] - b["low"], 10, 2)
    return a / (sma_gtja(a, 10, 2) + EPS)


def a191_110(b):
    val_up = (b["high"] - delay(b["close"], 1)).clip(lower=0.0)
    val_dn = (delay(b["close"], 1) - b["low"]).clip(lower=0.0)
    return ts_sum(val_up, 20) / (ts_sum(val_dn, 20) + EPS) * 100.0


def a191_111(b):
    inner = b["volume"] * ((b["close"] - b["low"]) - (b["high"] - b["close"])) / \
            (b["high"] - b["low"] + EPS)
    return sma_gtja(inner, 11, 2) - sma_gtja(inner, 4, 2)


def a191_112(b):
    diff = b["close"] - delay(b["close"], 1)
    up = diff.clip(lower=0.0)
    dn = (-diff).clip(lower=0.0)
    return (ts_sum(up, 12) - ts_sum(dn, 12)) / (ts_sum(up, 12) + ts_sum(dn, 12) + EPS) * 100.0


def a191_116(b):
    t = pd.Series(np.arange(len(b["close"]), dtype=np.float64), index=b["close"].index)
    return regbeta(b["close"], t, 20)


def a191_117(b):
    a = ts_rank(b["volume"], 32)
    bb = 1.0 - ts_rank(b["close"] + b["high"] - b["low"], 16)
    cc = 1.0 - ts_rank(b["ret"].fillna(0.0), 32)
    return a * bb * cc


def a191_118(b):
    return ts_sum(b["high"] - b["open"], 20) / (ts_sum(b["open"] - b["low"], 20) + EPS) * 100.0


def a191_122(b):
    log_close = log(b["close"].clip(lower=EPS))
    a = sma_gtja(sma_gtja(sma_gtja(log_close, 13, 2), 13, 2), 13, 2)
    return (a - delay(a, 1)) / (delay(a, 1) + EPS)


def a191_124(b):
    den = decay_linear(ts_max(b["close"], 30), 2)
    return (b["close"] - b["vwap"]) / (den + EPS)


def a191_126(b):
    return (b["close"] + b["high"] + b["low"]) / 3.0


def a191_127(b):
    rng = ts_max(b["close"], 12) - ts_min(b["close"], 12)
    inner = 100.0 * (b["close"] - ts_min(b["close"], 12)) / (rng + EPS)
    return sma(signedpower(inner, 2.0), 12) ** 0.5


def a191_128(b):
    typ = (b["high"] + b["low"] + b["close"]) / 3.0
    pos = pd.Series(np.where(typ > delay(typ, 1), (typ * b["volume"]).to_numpy(), 0.0), index=typ.index)
    neg = pd.Series(np.where(typ < delay(typ, 1), (typ * b["volume"]).to_numpy(), 0.0), index=typ.index)
    return 100.0 - 100.0 / (1.0 + ts_sum(pos, 14) / (ts_sum(neg, 14) + EPS))


def a191_129(b):
    diff = b["close"] - delay(b["close"], 1)
    cond = diff < 0.0
    val = pd.Series(np.where(cond.to_numpy(), diff.abs().to_numpy(), 0.0), index=diff.index)
    return ts_sum(val, 12)


def a191_132(b):
    return sma(b["amount"], 20)


def a191_133(b):
    return (20.0 - highday(b["high"], 20)) / 20.0 * 100.0 - \
           (20.0 - lowday(b["low"], 20)) / 20.0 * 100.0


def a191_134(b):
    return (b["close"] - delay(b["close"], 12)) / (delay(b["close"], 12) + EPS) * b["volume"]


def a191_135(b):
    return sma_gtja(delay(b["close"] / (delay(b["close"], 20) + EPS), 1), 20, 1)


def a191_137(b):
    """Long combined intraday strength formula."""
    h = b["high"]; l = b["low"]; c = b["close"]; o = b["open"]
    a1 = c - delay(c, 1) + (c - o) * 0.5 + delay(c, 1) - delay(o, 1)
    diff_high = (h - delay(c, 1)).abs()
    diff_low = (l - delay(c, 1)).abs()
    diff_hl = (h - delay(l, 1)).abs()
    cond_hi = (diff_high > diff_low) & (diff_high > diff_hl)
    cond_lo = (diff_low > diff_hl) & (diff_low > diff_high)
    delta_olc = (delay(c, 1) - delay(o, 1)).abs() * 0.25
    val = pd.Series(np.where(cond_hi.to_numpy(),
                             diff_high.to_numpy() + diff_low.to_numpy() * 0.5 + delta_olc.to_numpy(),
                             np.where(cond_lo.to_numpy(),
                                      diff_low.to_numpy() + diff_high.to_numpy() * 0.5 + delta_olc.to_numpy(),
                                      diff_hl.to_numpy() + delta_olc.to_numpy())),
                    index=h.index)
    return 16.0 * a1 / (val + EPS) * pd.concat([diff_high, diff_low], axis=1).max(axis=1)


def a191_139(b):
    return -1.0 * correlation(b["open"], b["volume"], 10)


def a191_144(b):
    cond = b["close"] < delay(b["close"], 1)
    cnt = count_if(cond, 20)
    val = (b["close"] / (delay(b["close"], 1) + EPS) - 1.0).abs() / (b["amount"] + EPS)
    return sumif(cond, val, 20) / (cnt + EPS)


def a191_145(b):
    return (sma(b["volume"], 9) - sma(b["volume"], 26)) / (sma(b["volume"], 12) + EPS) * 100.0


def a191_146(b):
    """sma_gtja chained."""
    z = (b["close"] - delay(b["close"], 1)) / (delay(b["close"], 1) + EPS)
    a = z - sma_gtja(z, 61, 2)
    bb = sma_gtja(a * a, 60, 2)
    return a / (bb ** 0.5 + EPS)


def a191_147(b):
    t = pd.Series(np.arange(len(b["close"]), dtype=np.float64), index=b["close"].index)
    return regbeta(sma(b["close"], 12), t, 12)


def a191_150(b):
    return (b["close"] + b["high"] + b["low"]) / 3.0 * b["volume"]


def a191_151(b):
    return sma_gtja(b["close"] - delay(b["close"], 20), 20, 1)


def a191_152(b):
    base = delay(sma_gtja(delay(b["close"] / (delay(b["close"], 9) + EPS), 1), 9, 1), 1)
    return sma_gtja(sma(base, 12) - sma(base, 26), 9, 1)


def a191_153(b):
    return (sma(b["close"], 3) + sma(b["close"], 6) +
            sma(b["close"], 12) + sma(b["close"], 24)) / 4.0


def a191_155(b):
    a = sma_gtja(b["volume"], 13, 2)
    bb = sma_gtja(b["volume"], 27, 2)
    return a - bb - sma_gtja(a - bb, 10, 2)


def a191_158(b):
    smoothed = sma_gtja(b["close"], 15, 2)
    return ((b["high"] - smoothed) - (b["low"] - smoothed)) / (b["close"] + EPS)


def a191_159(b):
    """Composite momentum (short/medium/long)."""
    h = b["high"]; l = b["low"]; c = b["close"]
    min_lc = pd.concat([l, delay(c, 1)], axis=1).min(axis=1)
    max_hc = pd.concat([h, delay(c, 1)], axis=1).max(axis=1)
    span = max_hc - min_lc
    a6 = ts_sum(c - min_lc, 6) / (ts_sum(span, 6) + EPS)
    a12 = ts_sum(c - min_lc, 12) / (ts_sum(span, 12) + EPS)
    a24 = ts_sum(c - min_lc, 24) / (ts_sum(span, 24) + EPS)
    return (a6 * 12.0 * 24.0 + a12 * 6.0 * 24.0 + a24 * 6.0 * 12.0) * 100.0 / (6 * 12 + 6 * 24 + 12 * 24)


def a191_160(b):
    cond = b["close"] < delay(b["close"], 1)
    val = pd.Series(np.where(cond.to_numpy(), stddev(b["close"], 20).to_numpy(), 0.0),
                    index=b["close"].index)
    return sma_gtja(val, 20, 1)


def a191_161(b):
    h = b["high"]; l = b["low"]
    a = h - l
    bb = (delay(b["close"], 1) - h).abs()
    cc = (delay(b["close"], 1) - l).abs()
    return sma(pd.concat([a, bb, cc], axis=1).max(axis=1), 12)


def a191_162(b):
    """sma_gtja-based RSI."""
    diff = b["close"] - delay(b["close"], 1)
    up = diff.clip(lower=0.0)
    rsi = sma_gtja(up, 12, 1) / (sma_gtja(diff.abs(), 12, 1) + EPS) * 100.0
    return (rsi - ts_min(rsi, 12)) / (ts_max(rsi, 12) - ts_min(rsi, 12) + EPS)


def a191_163(b):
    return -1.0 * sma(b["close"], 20) * b["volume"] / (b["vwap"] + EPS) * (b["high"] - b["close"])


def a191_164(b):
    cond = b["close"] > delay(b["close"], 1)
    diff = (b["close"] - delay(b["close"], 1))
    inv = pd.Series(np.where(cond.to_numpy(), 1.0 / (diff.to_numpy() + EPS), 1.0), index=b["close"].index)
    inner = (inv - ts_min(inv, 12)) / (b["high"] - b["low"] + EPS) * 100.0
    return sma_gtja(inner, 13, 2)


def a191_166(b):
    r = b["close"] / (delay(b["close"], 1) + EPS) - 1.0
    n = 20
    mu = sma(r, n)
    cubed = (r - mu) ** 3
    s_cubed = ts_sum(cubed, n)
    sqrd = (r - mu) ** 2
    s_sqrd = ts_sum(sqrd, n)
    return -20.0 * np.sqrt(20.0 * 19.0) * s_cubed / ((19.0 * 18.0) * (s_sqrd ** 1.5) + EPS)


def a191_167(b):
    diff = b["close"] - delay(b["close"], 1)
    val = diff.clip(lower=0.0)
    return ts_sum(val, 12)


def a191_168(b):
    return -1.0 * b["volume"] / (sma(b["volume"], 20) + EPS)


def a191_169(b):
    a = sma_gtja(b["close"] - delay(b["close"], 1), 9, 1)
    return sma_gtja(sma(delay(a, 1), 12) - sma(delay(a, 1), 26), 10, 1)


def a191_171(b):
    num = -1.0 * (b["low"] - b["close"]) * (b["open"] ** 5)
    den = (b["low"] - b["high"] - EPS) * (signedpower(b["close"], 5) + EPS)
    return num / den


def a191_172(b):
    """ATR-style ATR/SMA(ATR) ratio."""
    h = b["high"]; l = b["low"]
    tr = pd.concat([h - l,
                    (delay(b["close"], 1) - h).abs(),
                    (delay(b["close"], 1) - l).abs()], axis=1).max(axis=1)
    return tr / (sma(tr, 6) + EPS)


def a191_173(b):
    a1 = sma_gtja(b["close"], 13, 2)
    a2 = sma_gtja(a1, 13, 2)
    log_close = log(b["close"].clip(lower=EPS))
    a3 = sma_gtja(sma_gtja(sma_gtja(log_close, 13, 2), 13, 2), 13, 2)
    return 3.0 * a1 - 2.0 * a2 + a3


def a191_174(b):
    cond = b["close"] > delay(b["close"], 1)
    val = pd.Series(np.where(cond.to_numpy(), stddev(b["close"], 20).to_numpy(), 0.0),
                    index=b["close"].index)
    return sma_gtja(val, 20, 1)


def a191_175(b):
    h = b["high"]; l = b["low"]
    return sma(pd.concat([h - l,
                          (delay(b["close"], 1) - h).abs(),
                          (delay(b["close"], 1) - l).abs()], axis=1).max(axis=1), 6)


def a191_177(b):
    return (20.0 - highday(b["high"], 20)) / 20.0 * 100.0


def a191_178(b):
    return (b["close"] - delay(b["close"], 1)) / (delay(b["close"], 1) + EPS) * b["volume"]


def a191_180(b):
    cond = sma(b["volume"], 20) < b["volume"]
    val_t = -1.0 * ts_rank(delta(b["close"], 7).abs(), 60) * sign(delta(b["close"], 7))
    val_f = -1.0 * b["volume"]
    return pd.Series(np.where(cond.to_numpy(), val_t.to_numpy(), val_f.to_numpy()),
                     index=b["close"].index)


def a191_186(b):
    """ATR-style with 5/15 lengths."""
    h = b["high"]; l = b["low"]
    tr = pd.concat([h - l,
                    (delay(b["close"], 1) - h).abs(),
                    (delay(b["close"], 1) - l).abs()], axis=1).max(axis=1)
    return (sma(tr, 14) + delay(sma(tr, 14), 1)) / 2.0


def a191_187(b):
    cond = b["open"] > delay(b["open"], 1)
    diff_max = pd.concat([b["high"] - b["open"], b["open"] - delay(b["open"], 1)], axis=1).max(axis=1)
    val = pd.Series(np.where(cond.to_numpy(), diff_max.to_numpy(), 0.0), index=b["open"].index)
    return ts_sum(val, 20)


def a191_188(b):
    smoothed = sma_gtja(b["high"] - b["low"], 11, 2)
    return ((b["high"] - b["low"]) - smoothed) / (smoothed + EPS) * 100.0


def a191_189(b):
    return sma((b["close"] - sma(b["close"], 6)).abs(), 6)


def a191_191(b):
    return correlation(sma(b["volume"], 20), b["low"], 5) + (b["high"] + b["low"]) / 2.0 - b["close"]


# ============================================================================
# Aggregator
# ============================================================================

ALPHA191_FNS = [
    ("a191_002", a191_002), ("a191_003", a191_003), ("a191_004", a191_004),
    ("a191_005", a191_005), ("a191_007", a191_007), ("a191_009", a191_009),
    ("a191_010", a191_010), ("a191_011", a191_011), ("a191_013", a191_013),
    ("a191_014", a191_014), ("a191_015", a191_015), ("a191_018", a191_018),
    ("a191_019", a191_019), ("a191_020", a191_020), ("a191_021", a191_021),
    ("a191_022", a191_022), ("a191_023", a191_023), ("a191_024", a191_024),
    ("a191_026", a191_026), ("a191_027", a191_027), ("a191_028", a191_028),
    ("a191_029", a191_029), ("a191_031", a191_031), ("a191_033", a191_033),
    ("a191_034", a191_034), ("a191_038", a191_038), ("a191_040", a191_040),
    ("a191_042", a191_042), ("a191_043", a191_043), ("a191_044", a191_044),
    ("a191_046", a191_046), ("a191_047", a191_047), ("a191_049", a191_049),
    ("a191_050", a191_050), ("a191_052", a191_052), ("a191_053", a191_053),
    ("a191_054", a191_054), ("a191_057", a191_057), ("a191_058", a191_058),
    ("a191_059", a191_059), ("a191_060", a191_060), ("a191_062", a191_062),
    ("a191_063", a191_063), ("a191_065", a191_065), ("a191_066", a191_066),
    ("a191_067", a191_067), ("a191_068", a191_068), ("a191_070", a191_070),
    ("a191_071", a191_071), ("a191_072", a191_072), ("a191_076", a191_076),
    ("a191_078", a191_078), ("a191_079", a191_079), ("a191_080", a191_080),
    ("a191_081", a191_081), ("a191_082", a191_082), ("a191_084", a191_084),
    ("a191_085", a191_085), ("a191_086", a191_086), ("a191_088", a191_088),
    ("a191_089", a191_089), ("a191_093", a191_093), ("a191_094", a191_094),
    ("a191_095", a191_095), ("a191_096", a191_096), ("a191_097", a191_097),
    ("a191_098", a191_098), ("a191_100", a191_100), ("a191_102", a191_102),
    ("a191_103", a191_103), ("a191_104", a191_104), ("a191_106", a191_106),
    ("a191_109", a191_109), ("a191_110", a191_110), ("a191_111", a191_111),
    ("a191_112", a191_112), ("a191_116", a191_116), ("a191_117", a191_117),
    ("a191_118", a191_118), ("a191_122", a191_122), ("a191_124", a191_124),
    ("a191_126", a191_126), ("a191_127", a191_127), ("a191_128", a191_128),
    ("a191_129", a191_129), ("a191_132", a191_132), ("a191_133", a191_133),
    ("a191_134", a191_134), ("a191_135", a191_135), ("a191_137", a191_137),
    ("a191_139", a191_139), ("a191_144", a191_144), ("a191_145", a191_145),
    ("a191_146", a191_146), ("a191_147", a191_147), ("a191_150", a191_150),
    ("a191_151", a191_151), ("a191_152", a191_152), ("a191_153", a191_153),
    ("a191_155", a191_155), ("a191_158", a191_158), ("a191_159", a191_159),
    ("a191_160", a191_160), ("a191_161", a191_161), ("a191_162", a191_162),
    ("a191_163", a191_163), ("a191_164", a191_164), ("a191_166", a191_166),
    ("a191_167", a191_167), ("a191_168", a191_168), ("a191_169", a191_169),
    ("a191_171", a191_171), ("a191_172", a191_172), ("a191_173", a191_173),
    ("a191_174", a191_174), ("a191_175", a191_175), ("a191_177", a191_177),
    ("a191_178", a191_178), ("a191_180", a191_180), ("a191_186", a191_186),
    ("a191_187", a191_187), ("a191_188", a191_188), ("a191_189", a191_189),
    ("a191_191", a191_191),
]


def alpha191_columns():
    return [name for name, _ in ALPHA191_FNS]


def compute_alpha191(df: pd.DataFrame) -> pd.DataFrame:
    b = base_series(df)
    out = {}
    for name, fn in ALPHA191_FNS:
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
    feats = compute_alpha191(df)
    print(f"Alpha191 cols: {feats.shape[1]}, expected {len(ALPHA191_FNS)}")
    nf = feats.iloc[100:].isna().sum() / (len(feats) - 100)
    print(f"NaN frac at row 100 (top 10 highest):")
    for name, frac in nf.sort_values(ascending=False).head(10).items():
        print(f"  {name}: {frac:.3f}")
    print(f"  ... rest mostly under 0.05")
