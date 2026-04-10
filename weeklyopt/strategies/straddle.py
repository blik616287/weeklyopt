"""Straddle and strangle strategies."""

from dataclasses import dataclass

from .base import Strategy, OptionLeg, StrategyDirection
from ..pricing.black_scholes import OptionType, bs_price, strike_from_delta


@dataclass
class Straddle(Strategy):
    """Short straddle: sell ATM call + ATM put.

    High premium collection, unlimited risk. Use with position sizing.
    """
    name: str = "short_straddle"
    direction: StrategyDirection = StrategyDirection.NEUTRAL

    def construct(self, underlying_price, vol, dte_years, risk_free_rate):
        # ATM strike: round to nearest 0.50
        K = round(underlying_price)

        call_prem = bs_price(underlying_price, K, dte_years, risk_free_rate, vol, OptionType.CALL)
        put_prem = bs_price(underlying_price, K, dte_years, risk_free_rate, vol, OptionType.PUT)

        return [
            OptionLeg(OptionType.CALL, K, is_long=False, entry_price=call_prem),
            OptionLeg(OptionType.PUT, K, is_long=False, entry_price=put_prem),
        ]

    def max_risk(self, legs, underlying_price):
        # Undefined risk — use notional as proxy for position sizing
        total_premium = sum(leg.entry_price for leg in legs)
        return underlying_price * 100  # notional, capped by position sizing


@dataclass
class Strangle(Strategy):
    """Short strangle: sell OTM call + OTM put.

    Parameters:
        call_delta: Delta for short call (~0.20).
        put_delta: Absolute delta for short put (~0.20).
    """
    name: str = "short_strangle"
    direction: StrategyDirection = StrategyDirection.NEUTRAL
    call_delta: float = 0.20
    put_delta: float = 0.20

    def construct(self, underlying_price, vol, dte_years, risk_free_rate):
        call_K = strike_from_delta(
            underlying_price, dte_years, risk_free_rate, vol, self.call_delta, OptionType.CALL
        )
        put_K = strike_from_delta(
            underlying_price, dte_years, risk_free_rate, vol, self.put_delta, OptionType.PUT
        )

        call_prem = bs_price(underlying_price, call_K, dte_years, risk_free_rate, vol, OptionType.CALL)
        put_prem = bs_price(underlying_price, put_K, dte_years, risk_free_rate, vol, OptionType.PUT)

        return [
            OptionLeg(OptionType.CALL, call_K, is_long=False, entry_price=call_prem),
            OptionLeg(OptionType.PUT, put_K, is_long=False, entry_price=put_prem),
        ]

    def max_risk(self, legs, underlying_price):
        return underlying_price * 100  # undefined risk, use notional
