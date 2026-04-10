"""#11: Multi-leg rolling — roll winning positions into next week.

Instead of:
  Week 1: Open spread → Close spread (pay bid/ask on all legs)
  Week 2: Open new spread → Close spread

Do:
  Week 1: Open spread → At expiry, if profitable, roll the untested leg
  Week 2: Keep the profitable short leg, just move the protective long leg

This saves one bid/ask spread crossing per roll and compounds theta collection.

In the backtest, we model this as a reduced transaction cost when rolling.
"""

from dataclasses import dataclass

from ..strategies.base import OptionLeg, TradeResult
from ..pricing.black_scholes import OptionType


@dataclass
class RollState:
    """Track positions that can be rolled into next week."""
    ticker: str
    strategy_name: str
    short_leg_strike: float
    short_leg_type: OptionType
    remaining_credit: float  # credit still held from the short leg
    weeks_held: int = 1
    cumulative_pnl: float = 0.0


def should_roll(
    trade: TradeResult,
    max_roll_weeks: int = 3,
    current_roll: RollState | None = None,
) -> bool:
    """Decide whether to roll a winning position into next week.

    Roll if:
    - Trade was profitable (expiry or early exit with profit)
    - The short leg expired worthless or near-worthless
    - Haven't been rolling too long (avoid concentration risk)
    """
    if trade.total_pnl <= 0:
        return False  # don't roll losers

    if current_roll and current_roll.weeks_held >= max_roll_weeks:
        return False  # max rolling duration reached

    # Check if short legs expired near worthless (< 10% of entry)
    for leg in trade.legs:
        if not leg.is_long:  # short leg
            if leg.exit_price > leg.entry_price * 0.10:
                return False  # short leg still has value, don't roll

    return True


def compute_roll_savings(
    num_legs: int,
    slippage_per_contract: float = 0.05,
    commission_per_contract: float = 0.65,
    contracts: int = 1,
) -> float:
    """Calculate cost savings from rolling vs closing and re-opening.

    Rolling saves: opening cost of the short leg (already in place).
    You still pay to move the long leg (protective wing).
    """
    # Normal: close all legs + open all legs = 2 x (legs x cost)
    normal_cost = 2 * num_legs * (slippage_per_contract * 100 + commission_per_contract) * contracts

    # Rolling: close long leg + open new long leg = 2 x (1 leg x cost)
    # Short leg stays in place — no cost
    roll_cost = 2 * 1 * (slippage_per_contract * 100 + commission_per_contract) * contracts

    return normal_cost - roll_cost


def apply_roll_to_portfolio(
    trades: list[TradeResult],
    slippage: float = 0.05,
    commission: float = 0.65,
) -> tuple[list[TradeResult], float]:
    """Post-process trades to apply rolling logic.

    Identifies consecutive winning trades on the same ticker/strategy
    and reduces their transaction costs to model rolling.

    Returns (modified_trades, total_savings).
    """
    if not trades:
        return trades, 0.0

    total_savings = 0.0
    roll_states: dict[str, RollState] = {}  # key = ticker

    for trade in trades:
        key = trade.ticker
        current_roll = roll_states.get(key)

        if current_roll and current_roll.strategy_name == trade.strategy_name:
            # Same ticker, same strategy, consecutive week — can we roll?
            if should_roll(trade, current_roll=current_roll):
                # Apply roll savings
                savings = compute_roll_savings(
                    len(trade.legs), slippage, commission, trade.contracts,
                )
                trade.total_pnl += savings
                total_savings += savings

                current_roll.weeks_held += 1
                current_roll.cumulative_pnl += trade.total_pnl
            else:
                # Can't roll — reset
                roll_states.pop(key, None)
        else:
            roll_states.pop(key, None)

        # Start new roll tracking if this trade was profitable
        if trade.total_pnl > 0:
            roll_states[key] = RollState(
                ticker=trade.ticker,
                strategy_name=trade.strategy_name,
                short_leg_strike=trade.legs[0].strike if trade.legs else 0,
                short_leg_type=trade.legs[0].option_type if trade.legs else OptionType.PUT,
                remaining_credit=trade.total_premium_collected / (trade.contracts * 100) if trade.contracts > 0 else 0,
                weeks_held=1,
                cumulative_pnl=trade.total_pnl,
            )

    return trades, total_savings
