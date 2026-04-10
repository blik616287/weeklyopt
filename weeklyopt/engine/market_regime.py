"""Market-wide regime signals: VIX term structure, breadth, sentiment.

These are macro signals that apply to ALL positions, not per-ticker.
They answer: "should I be selling premium at all this week?"

Signals:
- VIX level and percentile rank
- VIX term structure (contango vs backwardation)
- Market breadth (% of tickers above key SMAs)
- Aggregate put/call ratio across universe
- Correlation regime (are stocks moving together?)
"""

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class MarketRegime:
    """Market-wide conditions for portfolio-level decisions."""
    date: date = date.today()

    # VIX
    vix: float = 0.0
    vix_20d_avg: float = 0.0
    vix_rank: float = 50.0       # percentile vs last year (0-100)
    vix_percentile: float = 50.0

    # VIX term structure
    vix_spot: float = 0.0        # front month VIX
    vix3m: float = 0.0           # 3-month VIX (VIX3M)
    term_structure_ratio: float = 1.0  # VIX / VIX3M: <1 = contango, >1 = backwardation
    in_contango: bool = True     # normal/calm market
    in_backwardation: bool = False  # fear/panic

    # Market breadth
    pct_above_20sma: float = 0.5  # % of universe above 20-day SMA
    pct_above_50sma: float = 0.5
    advance_decline: float = 0.0  # net advancers - decliners this week

    # Cross-correlation
    avg_correlation: float = 0.3   # avg pairwise correlation of universe
    correlation_regime: str = "normal"  # "low", "normal", "high"

    # Composite
    fear_gauge: float = 50.0  # 0 = extreme greed, 100 = extreme fear
    regime_label: str = "normal"  # "risk_on", "normal", "cautious", "risk_off"

    # Recommendation
    premium_selling_score: float = 50.0  # 0-100: how good is the environment for selling premium?
    position_size_modifier: float = 1.0  # scale positions up/down based on regime


def fetch_vix_data(lookback_days: int = 365) -> tuple[pd.Series, pd.Series | None]:
    """Fetch VIX and VIX3M historical data."""
    end = date.today()
    start = end - timedelta(days=lookback_days + 30)

    vix = yf.download("^VIX", start=str(start), end=str(end), progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    vix3m = None
    try:
        vix3m_df = yf.download("^VIX3M", start=str(start), end=str(end), progress=False)
        if isinstance(vix3m_df.columns, pd.MultiIndex):
            vix3m_df.columns = vix3m_df.columns.get_level_values(0)
        if not vix3m_df.empty:
            vix3m = vix3m_df["Close"]
    except Exception:
        pass

    return vix["Close"] if not vix.empty else pd.Series(dtype=float), vix3m


def compute_breadth(equity_data: dict[str, pd.DataFrame]) -> tuple[float, float, float]:
    """Compute market breadth from our ticker universe.

    Returns: (pct_above_20sma, pct_above_50sma, advance_decline_ratio)
    """
    above_20 = 0
    above_50 = 0
    advancers = 0
    total = 0

    for ticker, df in equity_data.items():
        if len(df) < 50:
            continue
        total += 1
        price = df["Close"].iloc[-1]
        sma20 = df["Close"].rolling(20).mean().iloc[-1]
        sma50 = df["Close"].rolling(50).mean().iloc[-1]

        if price > sma20:
            above_20 += 1
        if price > sma50:
            above_50 += 1

        # Weekly return
        if len(df) >= 5:
            week_ret = df["Close"].iloc[-1] / df["Close"].iloc[-6] - 1
            if week_ret > 0:
                advancers += 1

    if total == 0:
        return 0.5, 0.5, 0.0

    return above_20 / total, above_50 / total, advancers / total


def compute_correlation(equity_data: dict[str, pd.DataFrame], window: int = 20) -> float:
    """Compute average pairwise correlation of the universe."""
    returns = {}
    for ticker, df in equity_data.items():
        if len(df) >= window + 1:
            ret = df["Close"].pct_change().dropna().tail(window)
            if len(ret) == window:
                returns[ticker] = ret.values

    if len(returns) < 3:
        return 0.3

    tickers = list(returns.keys())
    corrs = []
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            c = np.corrcoef(returns[tickers[i]], returns[tickers[j]])[0, 1]
            if not np.isnan(c):
                corrs.append(c)

    return float(np.mean(corrs)) if corrs else 0.3


def analyze_market_regime(
    equity_data: dict[str, pd.DataFrame] | None = None,
) -> MarketRegime:
    """Full market regime analysis."""
    regime = MarketRegime(date=date.today())

    # ── VIX ──
    vix_series, vix3m_series = fetch_vix_data()

    if not vix_series.empty:
        regime.vix = float(vix_series.iloc[-1])
        regime.vix_spot = regime.vix
        regime.vix_20d_avg = float(vix_series.tail(20).mean())

        # Rank vs last year
        last_year = vix_series.tail(252)
        if len(last_year) > 20:
            regime.vix_rank = float(
                (regime.vix - last_year.min()) / (last_year.max() - last_year.min()) * 100
            ) if last_year.max() != last_year.min() else 50
            regime.vix_percentile = float((last_year < regime.vix).sum() / len(last_year) * 100)

    if vix3m_series is not None and not vix3m_series.empty:
        regime.vix3m = float(vix3m_series.iloc[-1])
        if regime.vix3m > 0:
            regime.term_structure_ratio = regime.vix_spot / regime.vix3m
            regime.in_contango = regime.term_structure_ratio < 1.0
            regime.in_backwardation = regime.term_structure_ratio > 1.0

    # ── Breadth ──
    if equity_data:
        regime.pct_above_20sma, regime.pct_above_50sma, regime.advance_decline = compute_breadth(equity_data)

    # ── Correlation ──
    if equity_data:
        regime.avg_correlation = compute_correlation(equity_data)
        if regime.avg_correlation > 0.6:
            regime.correlation_regime = "high"  # stocks moving together = systematic risk
        elif regime.avg_correlation < 0.2:
            regime.correlation_regime = "low"   # dispersed = stock-picking matters more
        else:
            regime.correlation_regime = "normal"

    # ── Fear Gauge (composite 0-100) ──
    fear = 50.0

    # VIX contribution (0-40 points)
    if regime.vix > 30:
        fear += 20
    elif regime.vix > 25:
        fear += 15
    elif regime.vix > 20:
        fear += 5
    elif regime.vix < 13:
        fear -= 15
    elif regime.vix < 16:
        fear -= 10

    # Term structure (0-20 points)
    if regime.in_backwardation:
        fear += 15  # fear is acute
    elif regime.in_contango and regime.term_structure_ratio < 0.85:
        fear -= 10  # deep contango = complacency

    # Breadth (0-20 points)
    if regime.pct_above_20sma < 0.30:
        fear += 10
    elif regime.pct_above_20sma > 0.80:
        fear -= 10

    # Correlation (0-10 points)
    if regime.avg_correlation > 0.6:
        fear += 10  # everything selling together = panic

    regime.fear_gauge = max(0, min(100, fear))

    # ── Regime Label ──
    if regime.fear_gauge >= 70:
        regime.regime_label = "risk_off"
    elif regime.fear_gauge >= 55:
        regime.regime_label = "cautious"
    elif regime.fear_gauge <= 30:
        regime.regime_label = "risk_on"
    else:
        regime.regime_label = "normal"

    # ── Premium selling score ──
    # High VIX + contango = best environment for selling premium
    ps = 50.0

    # VIX level: elevated = rich premium
    if regime.vix_rank > 70:
        ps += 20
    elif regime.vix_rank > 50:
        ps += 10
    elif regime.vix_rank < 25:
        ps -= 15

    # Contango: normal market structure, vol sellers win
    if regime.in_contango:
        ps += 10
    elif regime.in_backwardation:
        ps -= 15  # vol is spiking, dangerous to sell

    # Breadth: broad participation = stable market
    if regime.pct_above_50sma > 0.60:
        ps += 10
    elif regime.pct_above_50sma < 0.30:
        ps -= 10

    # Low correlation = better for diversified premium selling
    if regime.correlation_regime == "low":
        ps += 5
    elif regime.correlation_regime == "high":
        ps -= 10  # systematic risk, all positions lose together

    regime.premium_selling_score = max(0, min(100, ps))

    # Position size modifier
    if regime.premium_selling_score >= 70:
        regime.position_size_modifier = 1.25  # size up in ideal conditions
    elif regime.premium_selling_score >= 50:
        regime.position_size_modifier = 1.0
    elif regime.premium_selling_score >= 35:
        regime.position_size_modifier = 0.75  # size down
    else:
        regime.position_size_modifier = 0.50  # half size or sit out

    return regime


def print_market_regime(regime: MarketRegime) -> None:
    """Print market regime dashboard."""
    print(f"\n{'='*70}")
    print(f"  Market Regime Dashboard — {regime.date}")
    print(f"{'='*70}")

    print(f"\n  VIX")
    print(f"    Current:         {regime.vix:.1f}")
    print(f"    20d Average:     {regime.vix_20d_avg:.1f}")
    print(f"    1yr Rank:        {regime.vix_rank:.0f}%")
    print(f"    1yr Percentile:  {regime.vix_percentile:.0f}%")

    print(f"\n  Term Structure")
    print(f"    VIX Spot:        {regime.vix_spot:.1f}")
    print(f"    VIX3M:           {regime.vix3m:.1f}")
    print(f"    Ratio:           {regime.term_structure_ratio:.3f}")
    ts_status = "CONTANGO (normal)" if regime.in_contango else "BACKWARDATION (fear)"
    print(f"    Status:          {ts_status}")

    print(f"\n  Market Breadth")
    print(f"    % Above 20 SMA: {regime.pct_above_20sma:.0%}")
    print(f"    % Above 50 SMA: {regime.pct_above_50sma:.0%}")
    print(f"    Advance/Decline: {regime.advance_decline:.0%}")

    print(f"\n  Correlation")
    print(f"    Avg Pairwise:    {regime.avg_correlation:.2f}")
    print(f"    Regime:          {regime.correlation_regime}")

    print(f"\n  Composite")
    fear_bar = "#" * int(regime.fear_gauge / 2)
    greed_bar = " " * (50 - int(regime.fear_gauge / 2))
    print(f"    Fear Gauge:      [{greed_bar}{fear_bar}] {regime.fear_gauge:.0f}/100")
    print(f"    Regime:          {regime.regime_label.upper()}")

    print(f"\n  Premium Selling Environment")
    ps_bar = "#" * int(regime.premium_selling_score / 2)
    print(f"    Score:           [{ps_bar}] {regime.premium_selling_score:.0f}/100")
    print(f"    Position Size:   {regime.position_size_modifier:.0%} of normal")

    if regime.premium_selling_score >= 65:
        print(f"    Signal:          FAVORABLE — sell premium, normal size")
    elif regime.premium_selling_score >= 45:
        print(f"    Signal:          NEUTRAL — selective premium selling")
    elif regime.premium_selling_score >= 30:
        print(f"    Signal:          CAUTION — reduce size, tighter stops")
    else:
        print(f"    Signal:          UNFAVORABLE — sit out or buy protection")

    print(f"{'='*70}")
