"""Debit spreads: defined-risk strategies where max loss = premium paid.

Bull Call Spread: buy lower strike call, sell higher strike call (bullish, debit)
Bear Put Spread: buy higher strike put, sell lower strike put (bearish, debit)

These are the inverse of our credit spreads. You PAY to enter, and your max loss
is exactly what you paid. No margin, no assignment surprise, no blowup risk.
"""

from dataclasses import dataclass

from .base import Strategy, OptionLeg, StrategyDirection
from ..pricing.black_scholes import OptionType, bs_price, strike_from_delta


@dataclass
class BullCallSpread(Strategy):
    """Buy call spread (debit spread, bullish).

    Buy a near-ATM call, sell a further OTM call.
    Max loss = net debit paid. Max gain = spread width - debit.

    Parameters:
        long_delta: Delta of the long call (closer to ATM, e.g. 0.45).
        spread_width: Distance between strikes in dollars.
    """
    name: str = "bull_call_spread"
    direction: StrategyDirection = StrategyDirection.BULLISH
    long_delta: float = 0.45
    spread_width: float = 5.0

    def construct(self, underlying_price, vol, dte_years, risk_free_rate):
        long_K = strike_from_delta(
            underlying_price, dte_years, risk_free_rate, vol, self.long_delta, OptionType.CALL
        )
        short_K = long_K + self.spread_width

        long_prem = bs_price(underlying_price, long_K, dte_years, risk_free_rate, vol, OptionType.CALL)
        short_prem = bs_price(underlying_price, short_K, dte_years, risk_free_rate, vol, OptionType.CALL)

        return [
            OptionLeg(OptionType.CALL, long_K, is_long=True, entry_price=long_prem),
            OptionLeg(OptionType.CALL, short_K, is_long=False, entry_price=short_prem),
        ]

    def max_risk(self, legs, underlying_price):
        # Max loss = net debit paid
        net_debit = sum(leg.entry_price * leg.sign for leg in legs)  # positive = cost
        return max(net_debit, 0.01) * 100


@dataclass
class BearPutSpread(Strategy):
    """Buy put spread (debit spread, bearish).

    Buy a near-ATM put, sell a further OTM put.
    Max loss = net debit paid. Max gain = spread width - debit.

    Parameters:
        long_delta: Absolute delta of the long put (closer to ATM, e.g. 0.45).
        spread_width: Distance between strikes in dollars.
    """
    name: str = "bear_put_spread"
    direction: StrategyDirection = StrategyDirection.BEARISH
    long_delta: float = 0.45
    spread_width: float = 5.0

    def construct(self, underlying_price, vol, dte_years, risk_free_rate):
        long_K = strike_from_delta(
            underlying_price, dte_years, risk_free_rate, vol, self.long_delta, OptionType.PUT
        )
        short_K = long_K - self.spread_width

        long_prem = bs_price(underlying_price, long_K, dte_years, risk_free_rate, vol, OptionType.PUT)
        short_prem = bs_price(underlying_price, short_K, dte_years, risk_free_rate, vol, OptionType.PUT)

        return [
            OptionLeg(OptionType.PUT, long_K, is_long=True, entry_price=long_prem),
            OptionLeg(OptionType.PUT, short_K, is_long=False, entry_price=short_prem),
        ]

    def max_risk(self, legs, underlying_price):
        net_debit = sum(leg.entry_price * leg.sign for leg in legs)
        return max(net_debit, 0.01) * 100


@dataclass
class LongCall(Strategy):
    """Buy a single call. Max loss = premium paid.

    Parameters:
        delta: Delta of the call to buy (e.g. 0.40 for slight OTM).
    """
    name: str = "long_call"
    direction: StrategyDirection = StrategyDirection.BULLISH
    delta: float = 0.40

    def construct(self, underlying_price, vol, dte_years, risk_free_rate):
        K = strike_from_delta(
            underlying_price, dte_years, risk_free_rate, vol, self.delta, OptionType.CALL
        )
        prem = bs_price(underlying_price, K, dte_years, risk_free_rate, vol, OptionType.CALL)

        return [
            OptionLeg(OptionType.CALL, K, is_long=True, entry_price=prem),
        ]

    def max_risk(self, legs, underlying_price):
        return legs[0].entry_price * 100


@dataclass
class LongPut(Strategy):
    """Buy a single put. Max loss = premium paid.

    Parameters:
        delta: Absolute delta of the put to buy (e.g. 0.40).
    """
    name: str = "long_put"
    direction: StrategyDirection = StrategyDirection.BEARISH
    delta: float = 0.40

    def construct(self, underlying_price, vol, dte_years, risk_free_rate):
        K = strike_from_delta(
            underlying_price, dte_years, risk_free_rate, vol, self.delta, OptionType.PUT
        )
        prem = bs_price(underlying_price, K, dte_years, risk_free_rate, vol, OptionType.PUT)

        return [
            OptionLeg(OptionType.PUT, K, is_long=True, entry_price=prem),
        ]

    def max_risk(self, legs, underlying_price):
        return legs[0].entry_price * 100
