"""Managed long straddle: buy both sides, cut the loser, ride the winner.

Entry: Monday — buy ATM call + ATM put
Management:
  - Monitor daily which leg is winning
  - Close the losing leg when it drops below a threshold (recover time value)
  - Let the winning leg run with a trailing stop
  - If neither leg is winning by mid-week, close both (theta is eating you)

Max loss = total premium paid (defined risk).
"""

from dataclasses import dataclass

from .base import Strategy, OptionLeg, StrategyDirection
from ..pricing.black_scholes import OptionType, bs_price


@dataclass
class ManagedLongStraddle(Strategy):
    """Buy ATM straddle with active leg management.

    This is a LONG straddle — you pay premium, max loss = what you paid.
    The management happens in the engine via custom exit logic.

    Parameters:
        strike_offset: 0 = ATM, positive = slightly OTM both sides
    """
    name: str = "managed_straddle"
    direction: StrategyDirection = StrategyDirection.NEUTRAL
    strike_offset: float = 0.0  # ATM

    def construct(self, underlying_price, vol, dte_years, risk_free_rate):
        K = round(underlying_price + self.strike_offset)

        call_prem = bs_price(underlying_price, K, dte_years, risk_free_rate, vol, OptionType.CALL)
        put_prem = bs_price(underlying_price, K, dte_years, risk_free_rate, vol, OptionType.PUT)

        return [
            OptionLeg(OptionType.CALL, K, is_long=True, entry_price=call_prem),
            OptionLeg(OptionType.PUT, K, is_long=True, entry_price=put_prem),
        ]

    def max_risk(self, legs, underlying_price):
        # Max loss = total premium paid for both legs
        return sum(leg.entry_price for leg in legs) * 100


@dataclass
class ManagedLongStrangle(Strategy):
    """Buy OTM strangle with active leg management.

    Cheaper than straddle (both legs OTM), needs bigger move,
    but less premium at risk.

    Parameters:
        call_delta: Delta for the OTM call (~0.35).
        put_delta: Absolute delta for the OTM put (~0.35).
    """
    name: str = "managed_strangle"
    direction: StrategyDirection = StrategyDirection.NEUTRAL
    call_delta: float = 0.35
    put_delta: float = 0.35

    def construct(self, underlying_price, vol, dte_years, risk_free_rate):
        from ..pricing.black_scholes import strike_from_delta

        call_K = strike_from_delta(
            underlying_price, dte_years, risk_free_rate, vol, self.call_delta, OptionType.CALL
        )
        put_K = strike_from_delta(
            underlying_price, dte_years, risk_free_rate, vol, self.put_delta, OptionType.PUT
        )

        call_prem = bs_price(underlying_price, call_K, dte_years, risk_free_rate, vol, OptionType.CALL)
        put_prem = bs_price(underlying_price, put_K, dte_years, risk_free_rate, vol, OptionType.PUT)

        return [
            OptionLeg(OptionType.CALL, call_K, is_long=True, entry_price=call_prem),
            OptionLeg(OptionType.PUT, put_K, is_long=True, entry_price=put_prem),
        ]

    def max_risk(self, legs, underlying_price):
        return sum(leg.entry_price for leg in legs) * 100
