"""Exit rules for managing open positions.

Defines when to close a trade early based on P&L targets, stops,
Greeks thresholds, and time-based rules. These dramatically improve
real-world performance vs holding to expiry.
"""

from dataclasses import dataclass, field
from enum import Enum

from ..pricing.black_scholes import bs_price, bs_greeks, OptionType


class ExitReason(Enum):
    EXPIRY = "expiry"
    PROFIT_TARGET = "profit_target"
    STOP_LOSS = "stop_loss"
    PROFIT_FLOOR = "profit_floor"
    TIME_STOP = "time_stop"
    GAMMA_RISK = "gamma_risk"
    DELTA_BREACH = "delta_breach"


@dataclass
class ExitRules:
    """Configurable exit rules for option positions.

    All thresholds are expressed as fractions of the initial credit/debit.

    Example for a short iron condor collecting $2.00 credit:
      profit_target_pct=0.50 → close when you've captured $1.00 (50% of max)
      stop_loss_pct=2.0 → close when loss hits $4.00 (2x credit received)
      profit_floor_pct=0.30 → once profit hits 30%, set trailing stop
      trailing_stop_pct=0.50 → trailing stop at 50% of peak profit
    """
    # Profit taking: close when unrealized profit reaches this % of max profit
    profit_target_pct: float = 0.50  # close at 50% of max profit

    # Stop loss: close when loss reaches this multiple of credit received
    stop_loss_pct: float = 2.0  # close at 2x credit received

    # Profit floor + trailing stop
    profit_floor_pct: float = 0.30    # activate trailing stop once profit > 30% of max
    trailing_stop_pct: float = 0.50   # trail at 50% of peak unrealized profit

    # Time-based: close N days before expiry to avoid gamma risk
    time_stop_dte: int = 1  # close with 1 day left (Thursday for weekly)

    # Greeks-based thresholds
    max_position_delta: float = 0.50   # close if net |delta| exceeds this
    max_gamma_risk: float = 0.10       # close if any leg gamma exceeds this

    # Which rules are active
    use_profit_target: bool = True
    use_stop_loss: bool = True
    use_profit_floor: bool = True
    use_time_stop: bool = True
    use_greeks_stops: bool = True


@dataclass
class PositionState:
    """Track the evolving state of an open position."""
    entry_credit: float = 0.0     # net credit received (positive for credit spreads)
    max_profit: float = 0.0       # maximum possible profit
    peak_unrealized: float = 0.0  # highest unrealized profit seen so far
    trailing_active: bool = False  # whether trailing stop is armed
    current_pnl: float = 0.0
    days_held: int = 0


def evaluate_position_at_time(
    legs: list,
    underlying_price: float,
    vol: float,
    dte_years: float,
    risk_free_rate: float,
) -> tuple[float, float, list]:
    """Price all legs at current market conditions.

    Returns: (total_position_value, net_pnl_per_contract, greeks_per_leg)
    """
    total_value = 0.0
    total_entry = 0.0
    leg_greeks = []

    for leg in legs:
        current_price = bs_price(
            underlying_price, leg.strike, dte_years,
            risk_free_rate, vol, leg.option_type,
        )
        greeks = bs_greeks(
            underlying_price, leg.strike, dte_years,
            risk_free_rate, vol, leg.option_type,
        )

        # Position value: long legs are assets, short legs are liabilities
        total_value += leg.sign * current_price
        total_entry += leg.sign * leg.entry_price
        leg_greeks.append(greeks)

    # For credit spreads: entry credit is negative total_entry (we received money)
    # PnL = entry_credit - current_cost = -total_entry - total_value
    # Simplify: pnl = -(total_value - total_entry) for the position
    # Actually: pnl per share = (what we collected - what it costs to close)
    # For short positions: collected = -total_entry, cost_to_close = -total_value
    # pnl = -total_entry - (-total_value) = total_value - total_entry
    # Wait, let's be precise:
    # total_entry = sum(sign * entry_price): negative for net credit
    # total_value = sum(sign * current_price): negative when position profitable (short options worth less)
    # pnl = total_value - total_entry: when both are negative, pnl is positive if value < entry (good for sellers)

    pnl = -(total_value - total_entry)  # positive = profit for the position

    return total_value, pnl, leg_greeks


def check_exit(
    rules: ExitRules,
    state: PositionState,
    legs: list,
    underlying_price: float,
    vol: float,
    dte_years: float,
    risk_free_rate: float,
    trading_days_left: int,
) -> tuple[bool, ExitReason | None]:
    """Check if any exit rule triggers.

    Returns (should_exit, reason).
    """
    _, pnl, leg_greeks = evaluate_position_at_time(
        legs, underlying_price, vol, dte_years, risk_free_rate,
    )
    state.current_pnl = pnl

    # Track peak unrealized profit
    if pnl > state.peak_unrealized:
        state.peak_unrealized = pnl

    # 1. Profit target
    if rules.use_profit_target and state.max_profit > 0:
        if pnl >= state.max_profit * rules.profit_target_pct:
            return True, ExitReason.PROFIT_TARGET

    # 2. Stop loss
    if rules.use_stop_loss and state.entry_credit > 0:
        max_loss = state.entry_credit * rules.stop_loss_pct
        if pnl <= -max_loss:
            return True, ExitReason.STOP_LOSS

    # 3. Profit floor / trailing stop
    if rules.use_profit_floor and state.max_profit > 0:
        floor_threshold = state.max_profit * rules.profit_floor_pct
        if pnl >= floor_threshold:
            state.trailing_active = True

        if state.trailing_active and state.peak_unrealized > 0:
            trail_level = state.peak_unrealized * (1 - rules.trailing_stop_pct)
            if pnl <= trail_level:
                return True, ExitReason.PROFIT_FLOOR

    # 4. Time stop (close before expiry to avoid gamma)
    if rules.use_time_stop:
        if trading_days_left <= rules.time_stop_dte:
            return True, ExitReason.TIME_STOP

    # 5. Greeks-based stops
    if rules.use_greeks_stops:
        # Net delta check
        net_delta = sum(
            g.delta * leg.sign * leg.multiplier
            for g, leg in zip(leg_greeks, legs)
        )
        if abs(net_delta) > rules.max_position_delta * 100:  # scale to contract level
            return True, ExitReason.DELTA_BREACH

        # Gamma risk check
        for g in leg_greeks:
            if abs(g.gamma) > rules.max_gamma_risk:
                return True, ExitReason.GAMMA_RISK

    return False, None


# ─── Preset configurations ──────────────────────────────────────────────

AGGRESSIVE = ExitRules(
    profit_target_pct=0.75,
    stop_loss_pct=3.0,
    profit_floor_pct=0.50,
    trailing_stop_pct=0.40,
    time_stop_dte=0,  # hold to expiry
    use_greeks_stops=False,
)

CONSERVATIVE = ExitRules(
    profit_target_pct=0.50,
    stop_loss_pct=1.5,
    profit_floor_pct=0.25,
    trailing_stop_pct=0.50,
    time_stop_dte=1,
    use_greeks_stops=True,
    max_position_delta=0.40,
    max_gamma_risk=0.08,
)

THETA_HARVEST = ExitRules(
    profit_target_pct=0.50,
    stop_loss_pct=2.0,
    profit_floor_pct=0.30,
    trailing_stop_pct=0.50,
    time_stop_dte=1,  # exit Thursday, capture theta but dodge gamma
    use_greeks_stops=True,
    max_position_delta=0.50,
    max_gamma_risk=0.10,
)

HOLD_TO_EXPIRY = ExitRules(
    use_profit_target=False,
    use_stop_loss=False,
    use_profit_floor=False,
    use_time_stop=False,
    use_greeks_stops=False,
)
