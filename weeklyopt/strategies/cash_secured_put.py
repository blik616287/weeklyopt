"""Cash-secured put: short OTM put backed by cash."""

from dataclasses import dataclass

from .base import Strategy, OptionLeg, StrategyDirection
from ..pricing.black_scholes import OptionType, bs_price, strike_from_delta


@dataclass
class CashSecuredPut(Strategy):
    """Sell weekly OTM puts, secured by cash.

    Parameters:
        put_delta: Absolute delta of the short put (e.g., 0.30).
    """
    name: str = "cash_secured_put"
    direction: StrategyDirection = StrategyDirection.BULLISH
    put_delta: float = 0.30

    def construct(self, underlying_price, vol, dte_years, risk_free_rate):
        K = strike_from_delta(underlying_price, dte_years, risk_free_rate, vol, self.put_delta, OptionType.PUT)
        premium = bs_price(underlying_price, K, dte_years, risk_free_rate, vol, OptionType.PUT)

        return [
            OptionLeg(
                option_type=OptionType.PUT,
                strike=K,
                is_long=False,
                entry_price=premium,
            )
        ]

    def max_risk(self, legs, underlying_price):
        # Max loss = strike - premium (stock goes to zero)
        leg = legs[0]
        return (leg.strike - leg.entry_price) * 100
