"""Vertical spreads: bull put spread and bear call spread."""

from dataclasses import dataclass

from .base import Strategy, OptionLeg, StrategyDirection
from ..pricing.black_scholes import OptionType, bs_price, strike_from_delta


@dataclass
class BullPutSpread(Strategy):
    """Sell put spread (credit spread, bullish).

    Parameters:
        short_delta: Delta of the short put.
        spread_width: Distance between strikes in dollars.
    """
    name: str = "bull_put_spread"
    direction: StrategyDirection = StrategyDirection.BULLISH
    short_delta: float = 0.30
    spread_width: float = 5.0

    def construct(self, underlying_price, vol, dte_years, risk_free_rate):
        short_K = strike_from_delta(
            underlying_price, dte_years, risk_free_rate, vol, self.short_delta, OptionType.PUT
        )
        long_K = short_K - self.spread_width

        short_prem = bs_price(underlying_price, short_K, dte_years, risk_free_rate, vol, OptionType.PUT)
        long_prem = bs_price(underlying_price, long_K, dte_years, risk_free_rate, vol, OptionType.PUT)

        return [
            OptionLeg(OptionType.PUT, short_K, is_long=False, entry_price=short_prem),
            OptionLeg(OptionType.PUT, long_K, is_long=True, entry_price=long_prem),
        ]

    def max_risk(self, legs, underlying_price):
        net_credit = sum(leg.entry_price * (-leg.sign) for leg in legs)
        return (self.spread_width - net_credit) * 100


@dataclass
class BearCallSpread(Strategy):
    """Sell call spread (credit spread, bearish).

    Parameters:
        short_delta: Delta of the short call.
        spread_width: Distance between strikes in dollars.
    """
    name: str = "bear_call_spread"
    direction: StrategyDirection = StrategyDirection.BEARISH
    short_delta: float = 0.30
    spread_width: float = 5.0

    def construct(self, underlying_price, vol, dte_years, risk_free_rate):
        short_K = strike_from_delta(
            underlying_price, dte_years, risk_free_rate, vol, self.short_delta, OptionType.CALL
        )
        long_K = short_K + self.spread_width

        short_prem = bs_price(underlying_price, short_K, dte_years, risk_free_rate, vol, OptionType.CALL)
        long_prem = bs_price(underlying_price, long_K, dte_years, risk_free_rate, vol, OptionType.CALL)

        return [
            OptionLeg(OptionType.CALL, short_K, is_long=False, entry_price=short_prem),
            OptionLeg(OptionType.CALL, long_K, is_long=True, entry_price=long_prem),
        ]

    def max_risk(self, legs, underlying_price):
        net_credit = sum(leg.entry_price * (-leg.sign) for leg in legs)
        return (self.spread_width - net_credit) * 100
