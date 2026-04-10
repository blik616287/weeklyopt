"""Iron condor: sell OTM put spread + sell OTM call spread."""

from dataclasses import dataclass

from .base import Strategy, OptionLeg, StrategyDirection
from ..pricing.black_scholes import OptionType, bs_price, strike_from_delta


@dataclass
class IronCondor(Strategy):
    """Sell weekly iron condor.

    Parameters:
        short_put_delta: Delta of the short put leg (~0.20).
        short_call_delta: Delta of the short call leg (~0.20).
        wing_width: Distance in dollars between short and long strikes.
    """
    name: str = "iron_condor"
    direction: StrategyDirection = StrategyDirection.NEUTRAL
    short_put_delta: float = 0.20
    short_call_delta: float = 0.20
    wing_width: float = 5.0

    def construct(self, underlying_price, vol, dte_years, risk_free_rate):
        # Short put (higher strike, closer to ATM)
        sp_strike = strike_from_delta(
            underlying_price, dte_years, risk_free_rate, vol, self.short_put_delta, OptionType.PUT
        )
        # Long put (lower strike, further OTM)
        lp_strike = sp_strike - self.wing_width

        # Short call (lower strike, closer to ATM)
        sc_strike = strike_from_delta(
            underlying_price, dte_years, risk_free_rate, vol, self.short_call_delta, OptionType.CALL
        )
        # Long call (higher strike, further OTM)
        lc_strike = sc_strike + self.wing_width

        sp_prem = bs_price(underlying_price, sp_strike, dte_years, risk_free_rate, vol, OptionType.PUT)
        lp_prem = bs_price(underlying_price, lp_strike, dte_years, risk_free_rate, vol, OptionType.PUT)
        sc_prem = bs_price(underlying_price, sc_strike, dte_years, risk_free_rate, vol, OptionType.CALL)
        lc_prem = bs_price(underlying_price, lc_strike, dte_years, risk_free_rate, vol, OptionType.CALL)

        return [
            OptionLeg(OptionType.PUT, sp_strike, is_long=False, entry_price=sp_prem),
            OptionLeg(OptionType.PUT, lp_strike, is_long=True, entry_price=lp_prem),
            OptionLeg(OptionType.CALL, sc_strike, is_long=False, entry_price=sc_prem),
            OptionLeg(OptionType.CALL, lc_strike, is_long=True, entry_price=lc_prem),
        ]

    def max_risk(self, legs, underlying_price):
        # Max loss = wing width - net credit
        net_credit = sum(leg.entry_price * (-leg.sign) for leg in legs)
        return (self.wing_width - net_credit) * 100
