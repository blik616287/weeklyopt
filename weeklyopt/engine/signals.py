"""Signal generation: determine which strategy to run on which ticker each week.

Analyzes current market conditions per ticker and scores strategy fit:
- Directional regime (trend vs mean-reversion)
- IV rank (current IV vs historical range)
- Skew richness (put premium vs call premium)
- Momentum strength and direction
- Support/resistance proximity

Returns a ranked recommendation: which strategy + which tickers to trade this week.
"""

from dataclasses import dataclass, field
from enum import Enum
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ..pricing.volatility import historical_vol
from ..pricing.calibration import TickerCalibration, IVCalibrator
from .fundamentals import FundamentalData


class Regime(Enum):
    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    SIDEWAYS = "sideways"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"


class VolRegime(Enum):
    LOW = "low"         # IV rank < 25
    NORMAL = "normal"   # IV rank 25-50
    ELEVATED = "elevated"  # IV rank 50-75
    HIGH = "high"       # IV rank > 75


@dataclass
class TickerSignal:
    """Market signal analysis for a single ticker on a specific date."""
    ticker: str
    date: date
    price: float

    # Directional
    regime: Regime = Regime.SIDEWAYS
    momentum_5d: float = 0.0    # 5-day return
    momentum_20d: float = 0.0   # 20-day return
    rsi_14: float = 50.0
    distance_from_20sma: float = 0.0  # % above/below 20-day SMA

    # Volatility
    vol_regime: VolRegime = VolRegime.NORMAL
    current_hv: float = 0.0
    iv_rank: float = 50.0       # percentile rank of current HV vs 1yr range (0-100)
    iv_percentile: float = 50.0

    # Skew (from calibration)
    put_iv_ratio: float = 1.0   # calibrated put IV / HV
    call_iv_ratio: float = 1.0  # calibrated call IV / HV
    skew: float = 1.0           # put/call skew

    # Fundamentals & flow
    earnings_this_week: bool = False
    days_to_earnings: int | None = None
    put_call_oi_ratio: float = 1.0    # >1 = bearish positioning
    put_call_vol_ratio: float = 1.0
    max_pain_strike: float | None = None
    short_pct_float: float | None = None
    eps_surprise_pct: float | None = None

    # Strategy recommendations (score 0-100, higher = better fit)
    scores: dict = field(default_factory=dict)
    recommended_strategy: str = ""
    recommendation_reason: str = ""


def compute_rsi(prices: pd.Series, period: int = 14) -> float:
    """Compute RSI for the most recent value."""
    delta = prices.diff()
    gains = delta.clip(lower=0)
    losses = (-delta.clip(upper=0))

    avg_gain = gains.rolling(window=period, min_periods=period).mean()
    avg_loss = losses.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    last = rsi.dropna()
    return float(last.iloc[-1]) if not last.empty else 50.0


def compute_iv_rank(hv_series: pd.Series, lookback: int = 252) -> tuple[float, float]:
    """Compute IV rank and percentile.

    IV rank = (current - 1yr low) / (1yr high - 1yr low) * 100
    IV percentile = % of days in last year that were below current
    """
    recent = hv_series.dropna().tail(lookback)
    if len(recent) < 20:
        return 50.0, 50.0

    current = float(recent.iloc[-1])
    high = float(recent.max())
    low = float(recent.min())

    if high == low:
        rank = 50.0
    else:
        rank = (current - low) / (high - low) * 100

    percentile = (recent < current).sum() / len(recent) * 100

    return rank, percentile


def classify_regime(prices: pd.Series) -> Regime:
    """Classify directional regime from price action."""
    if len(prices) < 20:
        return Regime.SIDEWAYS

    ret_5 = (prices.iloc[-1] / prices.iloc[-6] - 1) if len(prices) >= 6 else 0
    ret_20 = (prices.iloc[-1] / prices.iloc[-21] - 1) if len(prices) >= 21 else 0

    sma_20 = prices.rolling(20).mean()
    sma_50 = prices.rolling(50).mean()

    above_20 = prices.iloc[-1] > sma_20.iloc[-1] if not sma_20.empty else True
    above_50 = prices.iloc[-1] > sma_50.iloc[-1] if len(sma_50.dropna()) > 0 else True

    if ret_20 > 0.05 and above_20 and above_50:
        return Regime.STRONG_UPTREND
    elif ret_20 > 0.02 and above_20:
        return Regime.UPTREND
    elif ret_20 < -0.05 and not above_20 and not above_50:
        return Regime.STRONG_DOWNTREND
    elif ret_20 < -0.02 and not above_20:
        return Regime.DOWNTREND
    else:
        return Regime.SIDEWAYS


def classify_vol_regime(iv_rank: float) -> VolRegime:
    if iv_rank < 25:
        return VolRegime.LOW
    elif iv_rank < 50:
        return VolRegime.NORMAL
    elif iv_rank < 75:
        return VolRegime.ELEVATED
    else:
        return VolRegime.HIGH


def score_strategies(signal: TickerSignal) -> dict[str, float]:
    """Score each strategy 0-100 based on current conditions.

    Higher score = better fit for current conditions.
    """
    scores = {}

    regime = signal.regime
    vol = signal.vol_regime
    iv_rank = signal.iv_rank
    skew = signal.skew
    rsi = signal.rsi_14
    put_ratio = signal.put_iv_ratio
    call_ratio = signal.call_iv_ratio

    # ── Cash-Secured Put ──
    # Best: uptrend + high IV + rich put skew + oversold dips
    score = 50.0
    if regime in (Regime.STRONG_UPTREND, Regime.UPTREND):
        score += 15
    elif regime == Regime.SIDEWAYS:
        score += 5
    elif regime == Regime.DOWNTREND:
        score -= 10
    elif regime == Regime.STRONG_DOWNTREND:
        score -= 25

    if vol in (VolRegime.ELEVATED, VolRegime.HIGH):
        score += 15  # high IV = fat premium
    elif vol == VolRegime.LOW:
        score -= 15  # thin premium

    if put_ratio > 1.4:
        score += 10  # puts are richly priced
    elif put_ratio < 1.1:
        score -= 10

    if rsi < 35:
        score += 10  # oversold bounce candidate = good for put selling
    elif rsi > 75:
        score -= 5   # extended, risky to sell puts

    scores["cash_secured_put"] = max(0, min(100, score))

    # ── Covered Call ──
    # Best: sideways/slight uptrend + high IV + rich call skew
    score = 40.0
    if regime == Regime.SIDEWAYS:
        score += 15
    elif regime == Regime.UPTREND:
        score += 5
    elif regime in (Regime.STRONG_UPTREND,):
        score -= 10  # will cap your gains
    elif regime in (Regime.DOWNTREND, Regime.STRONG_DOWNTREND):
        score -= 15  # stock falls, call premium doesn't save you

    if vol in (VolRegime.ELEVATED, VolRegime.HIGH):
        score += 10
    elif vol == VolRegime.LOW:
        score -= 10

    if call_ratio > 1.2:
        score += 10
    elif call_ratio < 1.0:
        score -= 15  # calls are cheap, nothing to sell

    scores["covered_call"] = max(0, min(100, score))

    # ── Bull Put Spread ──
    # Similar to CSP but with defined risk — better in uncertain uptrends
    score = scores["cash_secured_put"] - 5  # slightly less edge but defined risk
    if vol == VolRegime.HIGH:
        score += 5  # defined risk is better in high vol
    scores["bull_put_spread"] = max(0, min(100, score))

    # ── Bear Call Spread ──
    # Best: downtrend + high IV + calls still have some premium
    score = 35.0
    if regime in (Regime.DOWNTREND, Regime.STRONG_DOWNTREND):
        score += 20
    elif regime == Regime.SIDEWAYS:
        score += 5
    elif regime in (Regime.UPTREND, Regime.STRONG_UPTREND):
        score -= 20

    if vol in (VolRegime.ELEVATED, VolRegime.HIGH):
        score += 10
    if call_ratio > 1.2:
        score += 10
    elif call_ratio < 1.0:
        score -= 15

    if rsi > 70:
        score += 10  # overbought = good for selling calls

    scores["bear_call_spread"] = max(0, min(100, score))

    # ── Iron Condor ──
    # Best: sideways + high IV + balanced skew
    score = 40.0
    if regime == Regime.SIDEWAYS:
        score += 20
    elif regime in (Regime.UPTREND, Regime.DOWNTREND):
        score -= 5
    elif regime in (Regime.STRONG_UPTREND, Regime.STRONG_DOWNTREND):
        score -= 20  # trending kills condors

    if vol in (VolRegime.ELEVATED, VolRegime.HIGH):
        score += 15
    elif vol == VolRegime.LOW:
        score -= 20  # no premium to collect

    # Balanced skew helps condors
    if 0.95 < skew < 1.15:
        score += 5
    else:
        score -= 5

    scores["iron_condor"] = max(0, min(100, score))

    # ── Short Straddle ──
    # Best: sideways + very high IV + expect vol crush
    score = 35.0
    if regime == Regime.SIDEWAYS:
        score += 20
    elif regime in (Regime.STRONG_UPTREND, Regime.STRONG_DOWNTREND):
        score -= 25

    if vol == VolRegime.HIGH:
        score += 20
    elif vol == VolRegime.ELEVATED:
        score += 10
    elif vol == VolRegime.LOW:
        score -= 20

    if iv_rank > 75:
        score += 10  # mean-reversion of vol likely

    scores["short_straddle"] = max(0, min(100, score))

    # ── Short Strangle ──
    # Like straddle but wider — more forgiving of direction
    score = scores["short_straddle"] + 5  # slightly more room
    if regime in (Regime.UPTREND, Regime.DOWNTREND):
        score += 5  # can handle mild trends
    scores["short_strangle"] = max(0, min(100, score))

    # ── DEBIT STRATEGIES (defined risk, max loss = premium paid) ──

    # ── Bull Call Spread (debit) ──
    # Best: strong uptrend + low IV (cheap options) + momentum
    score = 40.0
    if regime == Regime.STRONG_UPTREND:
        score += 20
    elif regime == Regime.UPTREND:
        score += 15
    elif regime == Regime.SIDEWAYS:
        score -= 10  # theta eats you alive
    elif regime in (Regime.DOWNTREND, Regime.STRONG_DOWNTREND):
        score -= 25

    if vol == VolRegime.LOW:
        score += 10  # cheap entry, vol expansion helps
    elif vol == VolRegime.HIGH:
        score -= 10  # expensive entry, vol crush hurts

    if rsi > 50 and rsi < 70:
        score += 5  # momentum but not overbought
    elif rsi > 75:
        score -= 10  # too extended

    if signal.momentum_5d > 0.02:
        score += 5

    scores["bull_call_spread"] = max(0, min(100, score))

    # ── Bear Put Spread (debit) ──
    # Best: strong downtrend + low IV + bearish momentum
    score = 40.0
    if regime == Regime.STRONG_DOWNTREND:
        score += 20
    elif regime == Regime.DOWNTREND:
        score += 15
    elif regime == Regime.SIDEWAYS:
        score -= 10
    elif regime in (Regime.UPTREND, Regime.STRONG_UPTREND):
        score -= 25

    if vol == VolRegime.LOW:
        score += 10
    elif vol == VolRegime.HIGH:
        score -= 10

    if rsi < 50 and rsi > 30:
        score += 5
    elif rsi < 25:
        score -= 10  # too oversold, bounce likely

    if signal.momentum_5d < -0.02:
        score += 5

    scores["bear_put_spread"] = max(0, min(100, score))

    # ── Long Call ──
    # Best: breakout/strong momentum + low IV + conviction
    score = 35.0
    if regime == Regime.STRONG_UPTREND:
        score += 20
    elif regime == Regime.UPTREND:
        score += 10
    elif regime == Regime.SIDEWAYS:
        score -= 15
    elif regime in (Regime.DOWNTREND, Regime.STRONG_DOWNTREND):
        score -= 25

    if vol == VolRegime.LOW:
        score += 15  # cheap premium, vol expansion = bonus
    elif vol == VolRegime.NORMAL:
        score += 5
    elif vol == VolRegime.HIGH:
        score -= 15  # overpaying

    if signal.momentum_5d > 0.03:
        score += 10  # strong recent momentum

    scores["long_call"] = max(0, min(100, score))

    # ── Long Put ──
    # Best: breakdown/strong downward momentum + low IV
    score = 35.0
    if regime == Regime.STRONG_DOWNTREND:
        score += 20
    elif regime == Regime.DOWNTREND:
        score += 10
    elif regime == Regime.SIDEWAYS:
        score -= 15
    elif regime in (Regime.UPTREND, Regime.STRONG_UPTREND):
        score -= 25

    if vol == VolRegime.LOW:
        score += 15
    elif vol == VolRegime.NORMAL:
        score += 5
    elif vol == VolRegime.HIGH:
        score -= 15

    if signal.momentum_5d < -0.03:
        score += 10

    scores["long_put"] = max(0, min(100, score))

    # ── Managed Long Straddle ──
    # Best when: expecting big move but unsure of direction
    # High IV rank (big move expected), strong recent momentum (picking a direction),
    # NOT sideways (need movement)
    score = 40.0

    # Regime: need movement, any direction works
    if regime in (Regime.STRONG_UPTREND, Regime.STRONG_DOWNTREND):
        score += 15  # strong trend = one leg wins big
    elif regime in (Regime.UPTREND, Regime.DOWNTREND):
        score += 10
    elif regime == Regime.SIDEWAYS:
        score -= 20  # death for long straddles

    # IV: counterintuitive — we want LOW IV (cheap entry) but EXPECT vol expansion
    # High IV rank means vol is already priced in (expensive)
    if vol == VolRegime.LOW:
        score += 15  # cheap premium, potential vol expansion
    elif vol == VolRegime.NORMAL:
        score += 5
    elif vol == VolRegime.HIGH:
        score -= 15  # overpaying for both legs

    # Absolute momentum magnitude matters more than direction
    abs_mom_5d = abs(signal.momentum_5d)
    if abs_mom_5d > 0.04:
        score += 10  # stock is already moving
    elif abs_mom_5d > 0.02:
        score += 5
    elif abs_mom_5d < 0.005:
        score -= 10  # dead stock

    # Extreme RSI = potential for continued move (momentum) or snap-back (also a move)
    if rsi > 75 or rsi < 25:
        score += 5  # extreme = something is happening

    scores["managed_straddle"] = max(0, min(100, score))

    # ── Managed Long Strangle ──
    # Same logic but cheaper (OTM on both sides), needs bigger move
    score = scores["managed_straddle"] - 5  # slightly less edge (wider breakevens)
    if abs_mom_5d > 0.05:
        score += 5  # big movers justify the wider strikes
    scores["managed_strangle"] = max(0, min(100, score))

    # ── Fundamental adjustments (applied to ALL strategies) ──

    # Earnings this week: KILL credit strategies, but debit plays can work
    if signal.earnings_this_week:
        credit_strats = ["cash_secured_put", "covered_call", "bull_put_spread",
                         "bear_call_spread", "iron_condor", "short_straddle", "short_strangle"]
        for strat in credit_strats:
            if strat in scores:
                scores[strat] = max(0, scores[strat] - 40)
        # Debit strategies get a small boost (momentum plays into earnings)
        for strat in ["bull_call_spread", "bear_put_spread", "long_call", "long_put"]:
            if strat in scores:
                scores[strat] += 5
        # Managed straddle LOVES earnings — big move guaranteed, direction unknown
        for strat in ["managed_straddle", "managed_strangle"]:
            if strat in scores:
                scores[strat] += 15

    # Put/call OI ratio
    if signal.put_call_oi_ratio > 1.5:
        scores["cash_secured_put"] += 8
        scores["bull_put_spread"] += 8
        scores["bear_call_spread"] -= 5
        scores.setdefault("bear_put_spread", 0)
        scores["bear_put_spread"] = scores.get("bear_put_spread", 0) + 5  # crowd bearish, join them
        scores["long_put"] = scores.get("long_put", 0) + 5
    elif signal.put_call_oi_ratio < 0.7:
        scores["cash_secured_put"] -= 5
        scores["covered_call"] += 5
        scores["bear_call_spread"] += 5
        scores["bull_call_spread"] = scores.get("bull_call_spread", 0) + 5  # crowd bullish, join
        scores["long_call"] = scores.get("long_call", 0) + 5

    # Put/call volume ratio
    if signal.put_call_vol_ratio > 2.0:
        scores["cash_secured_put"] += 5   # panic buying puts = sell into it
        scores["bull_put_spread"] += 5
    elif signal.put_call_vol_ratio < 0.5:
        scores["covered_call"] += 5
        scores["bear_call_spread"] += 5

    # High short interest: squeeze risk = don't sell calls, do sell puts
    if signal.short_pct_float and signal.short_pct_float > 0.15:
        scores["covered_call"] -= 10      # squeeze blows through call strikes
        scores["bear_call_spread"] -= 10
        scores["cash_secured_put"] += 5   # squeeze is bullish for put sellers

    # EPS surprise: momentum continuation after beats, reversal risk after misses
    if signal.eps_surprise_pct is not None:
        if signal.eps_surprise_pct > 5:
            scores["cash_secured_put"] += 5  # post-beat momentum = safe to sell puts
        elif signal.eps_surprise_pct < -5:
            scores["cash_secured_put"] -= 5  # post-miss = continued selling risk
            scores["bear_call_spread"] += 5

    # Clamp all scores
    for strat in scores:
        scores[strat] = max(0, min(100, scores[strat]))

    return scores


def analyze_ticker(
    ticker: str,
    prices: pd.Series,
    calibration: TickerCalibration | None = None,
    fundamentals: FundamentalData | None = None,
    hv_window: int = 20,
) -> TickerSignal:
    """Generate a complete signal analysis for one ticker."""
    if len(prices) < 50:
        return TickerSignal(ticker=ticker, date=date.today(), price=float(prices.iloc[-1]))

    current_price = float(prices.iloc[-1])
    current_date = prices.index[-1].date() if hasattr(prices.index[-1], 'date') else date.today()

    # Directional
    regime = classify_regime(prices)
    mom_5 = float(prices.iloc[-1] / prices.iloc[-6] - 1) if len(prices) >= 6 else 0
    mom_20 = float(prices.iloc[-1] / prices.iloc[-21] - 1) if len(prices) >= 21 else 0
    rsi = compute_rsi(prices)
    sma_20 = float(prices.rolling(20).mean().iloc[-1])
    dist_sma = (current_price - sma_20) / sma_20

    # Volatility
    hv_series = historical_vol(prices, hv_window)
    current_hv = float(hv_series.dropna().iloc[-1]) if not hv_series.dropna().empty else 0.20
    iv_rank, iv_pct = compute_iv_rank(hv_series)
    vol_regime = classify_vol_regime(iv_rank)

    # Skew from calibration
    put_ratio = calibration.otm_put_iv_ratio if calibration and calibration.sample_dates > 0 else 1.15
    call_ratio = calibration.otm_call_iv_ratio if calibration and calibration.sample_dates > 0 else 1.02
    skew = calibration.put_call_skew if calibration and calibration.sample_dates > 0 else 1.15

    signal = TickerSignal(
        ticker=ticker,
        date=current_date,
        price=current_price,
        regime=regime,
        momentum_5d=mom_5,
        momentum_20d=mom_20,
        rsi_14=rsi,
        distance_from_20sma=dist_sma,
        vol_regime=vol_regime,
        current_hv=current_hv,
        iv_rank=iv_rank,
        iv_percentile=iv_pct,
        put_iv_ratio=put_ratio,
        call_iv_ratio=call_ratio,
        skew=skew,
        # Fundamentals
        earnings_this_week=fundamentals.earnings_this_week if fundamentals else False,
        days_to_earnings=fundamentals.days_to_earnings if fundamentals else None,
        put_call_oi_ratio=fundamentals.put_call_oi_ratio if fundamentals else 1.0,
        put_call_vol_ratio=fundamentals.put_call_volume_ratio if fundamentals else 1.0,
        max_pain_strike=fundamentals.max_pain_strike if fundamentals else None,
        short_pct_float=fundamentals.short_pct_float if fundamentals else None,
        eps_surprise_pct=fundamentals.eps_surprise_pct if fundamentals else None,
    )

    # Score all strategies
    signal.scores = score_strategies(signal)

    # Pick the best
    best = max(signal.scores, key=signal.scores.get)
    signal.recommended_strategy = best
    signal.recommendation_reason = _explain_recommendation(signal, best)

    return signal


def _explain_recommendation(signal: TickerSignal, strategy: str) -> str:
    """Generate a human-readable explanation for the recommendation."""
    parts = []

    r = signal.regime.value.replace("_", " ")
    parts.append(f"{r}")

    if signal.vol_regime in (VolRegime.ELEVATED, VolRegime.HIGH):
        parts.append(f"IV rank {signal.iv_rank:.0f} (rich premium)")
    elif signal.vol_regime == VolRegime.LOW:
        parts.append(f"IV rank {signal.iv_rank:.0f} (thin premium)")

    if signal.put_iv_ratio > 1.3:
        parts.append(f"puts rich ({signal.put_iv_ratio:.2f}x HV)")
    if signal.call_iv_ratio > 1.2:
        parts.append(f"calls rich ({signal.call_iv_ratio:.2f}x HV)")

    if signal.rsi_14 < 30:
        parts.append("oversold")
    elif signal.rsi_14 > 70:
        parts.append("overbought")

    return " | ".join(parts)


def scan_all_tickers(
    equity_data: dict[str, pd.DataFrame],
    calibrations: dict[str, TickerCalibration] | None = None,
    fundamentals: dict[str, FundamentalData] | None = None,
    min_score: float = 50.0,
) -> list[TickerSignal]:
    """Scan all tickers and return signals sorted by best opportunity."""
    calibrations = calibrations or {}
    fundamentals = fundamentals or {}
    signals = []

    for ticker, df in equity_data.items():
        cal = calibrations.get(ticker)
        fund = fundamentals.get(ticker)
        signal = analyze_ticker(ticker, df["Close"], cal, fund)
        signals.append(signal)

    # Sort by best score descending
    signals.sort(key=lambda s: max(s.scores.values()) if s.scores else 0, reverse=True)

    return signals


def print_scan_results(signals: list[TickerSignal], top_n: int = 15) -> None:
    """Print a formatted scan dashboard."""
    print(f"\n{'='*105}")
    print(f"  Weekly Options Signal Scanner")
    print(f"{'='*105}")
    print(
        f"  {'Ticker':>6}  {'Price':>8}  {'Regime':>18}  {'IV Rank':>7}  {'RSI':>5}  "
        f"{'Best Strategy':>18}  {'Score':>5}  {'Reason'}"
    )
    print(f"  {'-'*100}")

    for s in signals[:top_n]:
        best_strat = s.recommended_strategy
        best_score = s.scores.get(best_strat, 0)

        # Color-code score
        if best_score >= 70:
            indicator = "+++"
        elif best_score >= 55:
            indicator = "++"
        elif best_score >= 40:
            indicator = "+"
        else:
            indicator = "-"

        print(
            f"  {s.ticker:>6}  ${s.price:>7.2f}  {s.regime.value:>18}  "
            f"{s.iv_rank:>5.0f}%  {s.rsi_14:>5.1f}  "
            f"{best_strat:>18}  {best_score:>4.0f}{indicator}  "
            f"{s.recommendation_reason}"
        )

    # Summary by strategy
    print(f"\n  Strategy Distribution:")
    strat_counts = {}
    for s in signals[:top_n]:
        strat = s.recommended_strategy
        strat_counts[strat] = strat_counts.get(strat, 0) + 1

    for strat, count in sorted(strat_counts.items(), key=lambda x: -x[1]):
        tickers = [s.ticker for s in signals[:top_n] if s.recommended_strategy == strat]
        print(f"    {strat:>20s}: {count} tickers — {', '.join(tickers)}")

    print(f"{'='*105}")


def print_detailed_signal(signal: TickerSignal) -> None:
    """Print detailed analysis for a single ticker."""
    print(f"\n{'='*60}")
    print(f"  {signal.ticker} — ${signal.price:.2f} — {signal.date}")
    print(f"{'='*60}")
    print(f"  Regime:          {signal.regime.value}")
    print(f"  Momentum 5d:     {signal.momentum_5d:+.1%}")
    print(f"  Momentum 20d:    {signal.momentum_20d:+.1%}")
    print(f"  RSI(14):         {signal.rsi_14:.1f}")
    print(f"  Dist from 20SMA: {signal.distance_from_20sma:+.1%}")
    print(f"  HV(20):          {signal.current_hv:.1%}")
    print(f"  IV Rank:         {signal.iv_rank:.0f}%  ({signal.vol_regime.value})")
    print(f"  Put IV/HV:       {signal.put_iv_ratio:.2f}x")
    print(f"  Call IV/HV:      {signal.call_iv_ratio:.2f}x")
    print(f"  Put/Call Skew:   {signal.skew:.2f}")

    print(f"\n  Strategy Scores:")
    for strat, score in sorted(signal.scores.items(), key=lambda x: -x[1]):
        bar = "#" * int(score / 2)
        marker = " <-- BEST" if strat == signal.recommended_strategy else ""
        print(f"    {strat:>20s}: {score:>3.0f}  {bar}{marker}")

    print(f"\n  Recommendation: {signal.recommended_strategy}")
    print(f"  Reason: {signal.recommendation_reason}")
    print(f"{'='*60}")
