"""Covered call: long stock + short OTM call."""

from dataclasses import dataclass

from .base import Strategy, OptionLeg, StrategyDirection
from ..pricing.black_scholes import OptionType, bs_price, strike_from_delta


@dataclass
class CoveredCall(Strategy):
    """Sell weekly OTM calls against underlying shares.

    Parameters:
        call_delta: Delta of the short call (e.g., 0.30 for ~30 delta).
    """
    name: str = "covered_call"
    direction: StrategyDirection = StrategyDirection.BULLISH
    call_delta: float = 0.30

    def construct(self, underlying_price, vol, dte_years, risk_free_rate):
        K = strike_from_delta(underlying_price, dte_years, risk_free_rate, vol, self.call_delta, OptionType.CALL)
        premium = bs_price(underlying_price, K, dte_years, risk_free_rate, vol, OptionType.CALL)

        return [
            OptionLeg(
                option_type=OptionType.CALL,
                strike=K,
                is_long=False,
                entry_price=premium,
            )
        ]
        # Note: the long stock leg is handled implicitly by the engine
        # (P&L includes stock movement + premium collected)

    def max_risk(self, legs, underlying_price):
        # Risk is owning the stock minus premium collected
        premium = sum(leg.entry_price for leg in legs if not leg.is_long)
        return (underlying_price - premium) * 100
