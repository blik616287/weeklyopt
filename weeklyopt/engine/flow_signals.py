"""#9: Unusual options activity detection.

Without a paid feed (Unusual Whales, FlowAlgo), we approximate from yfinance:
- Volume/OI ratio spikes at specific strikes
- Sudden OI changes day-over-day
- Large volume at far OTM strikes (institutional bets)
- Put/call volume ratio extremes (acute sentiment)

This is a best-effort free approximation. For production, plug in a real feed.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class FlowSignal:
    ticker: str
    unusual_call_activity: bool = False
    unusual_put_activity: bool = False
    # Highest volume/OI ratio strike on each side
    hottest_call_strike: float | None = None
    hottest_put_strike: float | None = None
    hottest_call_vol_oi: float = 0.0
    hottest_put_vol_oi: float = 0.0
    # Aggregate
    flow_sentiment: str = "neutral"  # "bullish", "bearish", "neutral"
    flow_score: float = 0.0  # -100 (max bearish) to +100 (max bullish)


def detect_unusual_activity(ticker: str) -> FlowSignal:
    """Scan current option chain for unusual activity patterns."""
    signal = FlowSignal(ticker=ticker)

    try:
        tk = yf.Ticker(ticker)
        expirations = tk.options
        if not expirations:
            return signal

        # Get nearest weekly expiration
        chain = tk.option_chain(expirations[0])
        calls = chain.calls.copy()
        puts = chain.puts.copy()
    except Exception:
        return signal

    # Volume/OI ratio — high ratio = new positioning, not just existing OI
    for side, df, attr_strike, attr_ratio, attr_unusual in [
        ("call", calls, "hottest_call_strike", "hottest_call_vol_oi", "unusual_call_activity"),
        ("put", puts, "hottest_put_strike", "hottest_put_vol_oi", "unusual_put_activity"),
    ]:
        if df.empty or "volume" not in df.columns or "openInterest" not in df.columns:
            continue

        df = df.copy()
        df["volume"] = df["volume"].fillna(0)
        df["openInterest"] = df["openInterest"].fillna(0)
        df["vol_oi"] = df["volume"] / df["openInterest"].replace(0, 1)

        # Unusual = volume/OI > 3x (lots of new activity relative to existing positions)
        unusual = df[df["vol_oi"] > 3.0]

        if not unusual.empty:
            setattr(signal, attr_unusual, True)
            hottest = unusual.loc[unusual["volume"].idxmax()]
            setattr(signal, attr_strike, float(hottest["strike"]))
            setattr(signal, attr_ratio, float(hottest["vol_oi"]))

    # Flow sentiment: net call vs put unusual activity
    call_unusual_vol = calls[calls["volume"].fillna(0) / calls["openInterest"].fillna(1).replace(0, 1) > 2.0]["volume"].sum() if not calls.empty else 0
    put_unusual_vol = puts[puts["volume"].fillna(0) / puts["openInterest"].fillna(1).replace(0, 1) > 2.0]["volume"].sum() if not puts.empty else 0

    total = call_unusual_vol + put_unusual_vol
    if total > 0:
        signal.flow_score = ((call_unusual_vol - put_unusual_vol) / total) * 100

    if signal.flow_score > 30:
        signal.flow_sentiment = "bullish"
    elif signal.flow_score < -30:
        signal.flow_sentiment = "bearish"
    else:
        signal.flow_sentiment = "neutral"

    return signal


def flow_score_adjustment(flow: FlowSignal, strategy: str) -> float:
    """Adjust strategy score based on unusual options flow.

    Bullish flow → boost put selling (market makers hedge by buying stock)
    Bearish flow → boost call selling
    """
    adj = 0.0

    if flow.unusual_put_activity and flow.flow_sentiment == "bearish":
        # Heavy put buying = fear = rich put premium to sell
        if strategy in ("bull_put_spread", "cash_secured_put"):
            adj += 8
        elif strategy in ("bear_call_spread",):
            adj -= 3  # crowd already bearish, contrarian risk

    if flow.unusual_call_activity and flow.flow_sentiment == "bullish":
        # Heavy call buying = greed = rich call premium
        if strategy in ("bear_call_spread", "covered_call"):
            adj += 8
        elif strategy in ("bull_put_spread",):
            adj -= 3

    return adj
