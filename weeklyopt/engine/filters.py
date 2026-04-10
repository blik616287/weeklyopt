"""Portfolio-level filters and enhancements.

#2  IV rank hard gate (>50 only)
#3  Earnings calendar: skip pre-earnings, boost post-earnings
#4  VIX backwardation filter
#5  Entry timing premium (open vs close)
#6  Dynamic delta based on IV rank
#7  Correlated pair avoidance
#8  Max pain gravitational scoring
"""

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ..config import CORRELATION_CLUSTERS


@dataclass
class FilterConfig:
    """Configuration for all portfolio filters."""
    # #2: IV rank gate
    min_iv_rank: float = 50.0  # don't trade if IV rank below this
    iv_rank_enabled: bool = True

    # #3: Earnings filter
    skip_pre_earnings_days: int = 5  # skip N days before earnings
    boost_post_earnings: bool = True  # size up week after earnings
    post_earnings_boost: float = 1.3  # 30% more capital post-earnings
    earnings_enabled: bool = True

    # #4: VIX backwardation filter
    skip_backwardation: bool = True  # don't trade if VIX in backwardation
    backwardation_exception: list = None  # allow these strategies even in backwardation

    # #5: Entry timing
    open_premium_markup: float = 1.05  # Monday open premium ~5% richer than close
    entry_timing_enabled: bool = True

    # #6: Dynamic delta
    dynamic_delta_enabled: bool = True
    # IV rank -> delta mapping
    # High IV: sell further OTM (more room)
    # Low IV: sell closer to ATM (need premium)

    # #7: Correlated pair avoidance
    max_per_cluster: int = 1  # max positions per correlation cluster
    correlation_filter_enabled: bool = True

    # #8: Max pain scoring
    max_pain_boost: float = 10.0  # score bonus when short strike near max pain
    max_pain_enabled: bool = True

    def __post_init__(self):
        if self.backwardation_exception is None:
            self.backwardation_exception = ["bull_put_spread"]  # put premium richest in fear


def dynamic_delta_for_iv_rank(iv_rank: float) -> float:
    """#6: Adjust delta based on IV environment.

    High IV → sell further OTM (0.15 delta) — more room, still good premium
    Normal IV → standard (0.25 delta)
    Low IV → sell closer to ATM (0.35 delta) — need the premium
    """
    if iv_rank >= 75:
        return 0.15  # far OTM — vol is high, plenty of premium out there
    elif iv_rank >= 50:
        return 0.20  # standard OTM
    elif iv_rank >= 30:
        return 0.25  # slightly closer
    else:
        return 0.30  # near ATM — scraping for premium


def check_iv_rank_gate(iv_rank: float, config: FilterConfig) -> bool:
    """#2: Should we trade this ticker based on IV rank?"""
    if not config.iv_rank_enabled:
        return True
    return iv_rank >= config.min_iv_rank


def check_earnings_filter(
    days_to_earnings: int | None,
    config: FilterConfig,
) -> tuple[bool, float]:
    """#3: Earnings proximity check.

    Returns (should_trade, size_multiplier).
    """
    if not config.earnings_enabled or days_to_earnings is None:
        return True, 1.0

    # Skip pre-earnings
    if 0 < days_to_earnings <= config.skip_pre_earnings_days:
        return False, 0.0

    # Boost post-earnings (negative days_to_earnings or within 7 days after)
    if days_to_earnings < 0 and days_to_earnings >= -7:
        return True, config.post_earnings_boost

    return True, 1.0


def check_backwardation_filter(
    vix_data: pd.DataFrame | None,
    as_of: pd.Timestamp,
    strategy_name: str,
    config: FilterConfig,
) -> bool:
    """#4: Skip if VIX term structure is in backwardation (fear).

    Exception: bull_put_spread can trade in backwardation (puts are richest then).
    """
    if not config.skip_backwardation or vix_data is None:
        return True

    # We'd need VIX3M data too — approximate from VIX 20d avg
    vix_hist = vix_data.loc[:as_of, "Close"] if "Close" in vix_data.columns else pd.Series(dtype=float)
    if len(vix_hist) < 20:
        return True

    vix_now = float(vix_hist.iloc[-1])
    vix_20d = float(vix_hist.tail(20).mean())

    # Proxy: VIX significantly above its 20d average = acute fear (backwardation-like)
    in_backwardation = vix_now > vix_20d * 1.15

    if in_backwardation:
        if strategy_name in (config.backwardation_exception or []):
            return True  # exception strategies can still trade
        return False

    return True


def apply_entry_timing(credit: float, config: FilterConfig) -> float:
    """#5: Adjust credit for entry timing premium.

    Monday open typically has ~5% more premium than close due to weekend decay.
    This models entering at the open for better fills.
    """
    if not config.entry_timing_enabled:
        return credit
    return credit * config.open_premium_markup


def check_correlation_cluster(
    ticker: str,
    already_selected: list[str],
    config: FilterConfig,
) -> bool:
    """#7: Don't hold too many positions in the same correlation cluster.

    E.g., don't hold bull put spreads on both NVDA and AMD simultaneously.
    """
    if not config.correlation_filter_enabled:
        return True

    # Find which cluster this ticker belongs to
    ticker_cluster = None
    for cluster_name, members in CORRELATION_CLUSTERS.items():
        if ticker in members:
            ticker_cluster = cluster_name
            break

    if ticker_cluster is None:
        return True  # not in any cluster, always allowed

    # Count how many from this cluster are already selected
    count = 0
    for selected in already_selected:
        for cluster_name, members in CORRELATION_CLUSTERS.items():
            if cluster_name == ticker_cluster and selected in members:
                count += 1
                break

    return count < config.max_per_cluster


def max_pain_score_adjustment(
    short_strike: float,
    max_pain_strike: float | None,
    underlying_price: float,
    config: FilterConfig,
) -> float:
    """#8: Score bonus when short strike is near max pain.

    Stocks tend to pin near max pain at Friday expiry.
    If our short strike is near max pain, higher probability of profit.
    """
    if not config.max_pain_enabled or max_pain_strike is None:
        return 0.0

    # How close is our short strike to max pain (as % of underlying)?
    distance_pct = abs(short_strike - max_pain_strike) / underlying_price

    if distance_pct < 0.01:  # within 1%
        return config.max_pain_boost
    elif distance_pct < 0.02:  # within 2%
        return config.max_pain_boost * 0.5
    elif distance_pct < 0.03:
        return config.max_pain_boost * 0.25

    return 0.0
