"""Strike-level open interest analysis for credit spreads.

Heavy OI at a strike acts as:
1. Support/resistance — market makers hedge at these levels, creating a "wall"
2. Max pain gravity — stocks tend to pin near the strike with highest total OI value
3. Short strike validation — if your short strike has heavy OI on the same side,
   you're in crowded company (market makers will defend it)

This module analyzes the live option chain to:
- Score how well a proposed short strike is protected by OI structure
- Identify OI walls (support below, resistance above)
- Detect unusual OI buildup at specific strikes
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class OIAnalysis:
    ticker: str
    spot_price: float = 0.0
    expiration: str = ""

    # OI structure
    nearest_put_wall: float | None = None      # nearest strike below spot with heavy put OI
    nearest_call_wall: float | None = None     # nearest strike above spot with heavy call OI
    put_wall_oi: int = 0
    call_wall_oi: int = 0
    put_wall_distance_pct: float = 0.0         # how far below spot (%)
    call_wall_distance_pct: float = 0.0        # how far above spot (%)

    # Max pain
    max_pain_strike: float | None = None
    max_pain_distance_pct: float = 0.0

    # Short strike scoring
    short_strike_oi_score: float = 0.0         # 0-100: how much OI supports your short strike

    # Unusual OI
    unusual_oi_strikes: list = None            # strikes with OI > 3x median

    def __post_init__(self):
        if self.unusual_oi_strikes is None:
            self.unusual_oi_strikes = []


def analyze_oi_structure(ticker: str) -> OIAnalysis:
    """Analyze the full OI structure of the nearest expiration chain."""
    analysis = OIAnalysis(ticker=ticker)

    try:
        tk = yf.Ticker(ticker)
        expirations = tk.options
        if not expirations:
            return analysis

        chain = tk.option_chain(expirations[0])
        calls = chain.calls.copy()
        puts = chain.puts.copy()
        analysis.expiration = expirations[0]

        hist = tk.history(period="5d")
        if hist.empty:
            return analysis
        analysis.spot_price = float(hist["Close"].iloc[-1])
    except Exception:
        return analysis

    spot = analysis.spot_price
    if spot <= 0:
        return analysis

    # Clean OI data
    for df in [calls, puts]:
        if "openInterest" in df.columns:
            df["openInterest"] = df["openInterest"].fillna(0).astype(int)
        else:
            return analysis

    # ── Find OI walls ──
    # Put wall: highest put OI strike below spot (acts as support)
    puts_below = puts[puts["strike"] < spot].copy()
    if not puts_below.empty:
        top_put = puts_below.loc[puts_below["openInterest"].idxmax()]
        analysis.nearest_put_wall = float(top_put["strike"])
        analysis.put_wall_oi = int(top_put["openInterest"])
        analysis.put_wall_distance_pct = (spot - analysis.nearest_put_wall) / spot

    # Call wall: highest call OI strike above spot (acts as resistance)
    calls_above = calls[calls["strike"] > spot].copy()
    if not calls_above.empty:
        top_call = calls_above.loc[calls_above["openInterest"].idxmax()]
        analysis.nearest_call_wall = float(top_call["strike"])
        analysis.call_wall_oi = int(top_call["openInterest"])
        analysis.call_wall_distance_pct = (analysis.nearest_call_wall - spot) / spot

    # ── Max pain ──
    all_strikes = sorted(set(calls["strike"].tolist() + puts["strike"].tolist()))
    if all_strikes:
        min_pain = float("inf")
        mp_strike = all_strikes[len(all_strikes) // 2]

        for strike in all_strikes:
            pain = 0
            for _, row in calls.iterrows():
                oi = int(row["openInterest"])
                if strike > row["strike"]:
                    pain += (strike - row["strike"]) * oi
            for _, row in puts.iterrows():
                oi = int(row["openInterest"])
                if strike < row["strike"]:
                    pain += (row["strike"] - strike) * oi

            if pain < min_pain:
                min_pain = pain
                mp_strike = strike

        analysis.max_pain_strike = mp_strike
        analysis.max_pain_distance_pct = abs(spot - mp_strike) / spot

    # ── Unusual OI detection ──
    all_oi = pd.concat([
        puts[["strike", "openInterest"]].assign(side="put"),
        calls[["strike", "openInterest"]].assign(side="call"),
    ])
    median_oi = all_oi["openInterest"].median()
    if median_oi > 0:
        unusual = all_oi[all_oi["openInterest"] > median_oi * 3]
        analysis.unusual_oi_strikes = [
            {"strike": float(r["strike"]), "side": r["side"], "oi": int(r["openInterest"])}
            for _, r in unusual.iterrows()
        ]

    return analysis


def score_short_strike(
    analysis: OIAnalysis,
    short_strike: float,
    option_type: str,  # "put" or "call"
) -> float:
    """Score how well the OI structure supports a specific short strike.

    Returns 0-100. Higher = more OI protection for your short strike.

    For a short PUT at strike K:
    - Heavy put OI at or below K = good (support, market makers defend)
    - K is between spot and put wall = ideal (cushion above your strike)
    - K is below the put wall = even better (behind the wall)

    For a short CALL at strike K:
    - Heavy call OI at or above K = good (resistance, caps upside)
    - K is between spot and call wall = ideal
    """
    spot = analysis.spot_price
    if spot <= 0:
        return 50.0  # neutral if no data

    score = 50.0

    if option_type == "put":
        # Put wall protection
        if analysis.nearest_put_wall is not None:
            if short_strike <= analysis.nearest_put_wall:
                # Short strike is at or below the OI wall — excellent protection
                score += 25
            elif short_strike < spot and analysis.nearest_put_wall < short_strike:
                # Short strike between spot and wall — wall provides some support
                # Score by how close the wall is to the short strike
                gap = (short_strike - analysis.nearest_put_wall) / spot
                if gap < 0.02:
                    score += 15  # wall is very close below
                elif gap < 0.05:
                    score += 8

            # Scale by how large the wall is
            if analysis.put_wall_oi > 10_000:
                score += 10
            elif analysis.put_wall_oi > 5_000:
                score += 5

        # Max pain proximity — if short strike is near max pain, pinning helps
        if analysis.max_pain_strike is not None:
            mp_dist = abs(short_strike - analysis.max_pain_strike) / spot
            if mp_dist < 0.01:
                score += 10
            elif mp_dist < 0.02:
                score += 5

    elif option_type == "call":
        if analysis.nearest_call_wall is not None:
            if short_strike >= analysis.nearest_call_wall:
                score += 25
            elif short_strike > spot and analysis.nearest_call_wall > short_strike:
                gap = (analysis.nearest_call_wall - short_strike) / spot
                if gap < 0.02:
                    score += 15
                elif gap < 0.05:
                    score += 8

            if analysis.call_wall_oi > 10_000:
                score += 10
            elif analysis.call_wall_oi > 5_000:
                score += 5

        if analysis.max_pain_strike is not None:
            mp_dist = abs(short_strike - analysis.max_pain_strike) / spot
            if mp_dist < 0.01:
                score += 10
            elif mp_dist < 0.02:
                score += 5

    # Check for unusual OI at the short strike itself (crowded = defended)
    for u in analysis.unusual_oi_strikes:
        if abs(u["strike"] - short_strike) < 0.5:
            if u["side"] == option_type:
                score += 10  # heavy OI at your exact strike = market makers will defend

    return max(0, min(100, score))


def print_oi_analysis(analysis: OIAnalysis) -> None:
    """Print OI structure analysis."""
    print(f"\n  OI Structure: {analysis.ticker} @ ${analysis.spot_price:.2f}  (exp: {analysis.expiration})")
    print(f"  {'-'*55}")

    if analysis.nearest_call_wall:
        print(f"  Call wall:   ${analysis.nearest_call_wall:.0f}  "
              f"({analysis.call_wall_distance_pct:+.1%} from spot)  "
              f"OI: {analysis.call_wall_oi:,}")

    print(f"  Spot:        ${analysis.spot_price:.2f}")

    if analysis.nearest_put_wall:
        print(f"  Put wall:    ${analysis.nearest_put_wall:.0f}  "
              f"({-analysis.put_wall_distance_pct:.1%} from spot)  "
              f"OI: {analysis.put_wall_oi:,}")

    if analysis.max_pain_strike:
        print(f"  Max pain:    ${analysis.max_pain_strike:.0f}  "
              f"({analysis.max_pain_distance_pct:.1%} away)")

    if analysis.unusual_oi_strikes:
        print(f"  Unusual OI:  {len(analysis.unusual_oi_strikes)} strikes")
        for u in sorted(analysis.unusual_oi_strikes, key=lambda x: -x["oi"])[:5]:
            print(f"    ${u['strike']:.0f} {u['side']}: {u['oi']:,}")
