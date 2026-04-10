"""Managed straddle execution: daily leg-by-leg management.

The logic:
  Day 1 (Mon): Open both legs
  Day 2 (Tue): If one leg is up 30%+, close the loser (recover remaining time value)
  Day 3 (Wed): If no clear winner yet, close both (theta is accelerating)
  Day 3-4: Ride the winner with a trailing stop at 50% of peak unrealized
  Day 4 (Thu): Close everything (don't hold into Friday gamma)

This turns a straddle from a binary bet into a managed directional play.
"""

from dataclasses import dataclass

import numpy as np

from ..pricing.black_scholes import bs_price, OptionType


@dataclass
class StraddleManagerConfig:
    # When the winning leg is up this % of its entry, close the losing leg
    winner_threshold_pct: float = 0.30  # 30% gain on winning leg triggers cut

    # If no winner by this day (0=Mon, 1=Tue, 2=Wed), close everything
    max_wait_days: int = 2  # close if no direction by Wednesday

    # Trailing stop on the winning leg (% of peak unrealized profit)
    trailing_stop_pct: float = 0.50

    # Close everything by this day (0=Mon, 4=Fri)
    final_exit_day: int = 3  # Thursday

    # If loser recovers to within this % of entry, don't cut it yet
    loser_recovery_zone: float = 0.15  # within 15% of entry = still alive


def simulate_managed_straddle(
    legs: list,
    daily_prices: list[float],  # underlying close each day [Mon, Tue, Wed, Thu, Fri]
    vol: float,
    risk_free_rate: float,
    config: StraddleManagerConfig | None = None,
) -> tuple[float, str, int]:
    """Simulate managed straddle over a week.

    Returns: (pnl_per_share, exit_reason, days_held)
    """
    if config is None:
        config = StraddleManagerConfig()

    if len(legs) != 2 or len(daily_prices) < 2:
        # Fallback: just evaluate at last price
        total_entry = sum(leg.entry_price for leg in legs)
        total_exit = 0
        last_price = daily_prices[-1]
        for leg in legs:
            if leg.option_type == OptionType.CALL:
                total_exit += max(last_price - leg.strike, 0)
            else:
                total_exit += max(leg.strike - last_price, 0)
        return total_exit - total_entry, "expiry", len(daily_prices)

    call_leg = next(l for l in legs if l.option_type == OptionType.CALL)
    put_leg = next(l for l in legs if l.option_type == OptionType.PUT)

    total_entry = call_leg.entry_price + put_leg.entry_price
    call_entry = call_leg.entry_price
    put_entry = put_leg.entry_price

    # Track state
    call_alive = True
    put_alive = True
    call_closed_pnl = 0.0
    put_closed_pnl = 0.0
    winner_peak = 0.0
    exit_reason = "expiry"

    total_days = len(daily_prices)

    for day_idx in range(1, total_days):  # skip entry day
        price = daily_prices[day_idx]
        days_remaining = total_days - day_idx - 1
        dte_years = max(days_remaining / 252, 0.25 / 252)

        # Price live legs
        call_now = bs_price(price, call_leg.strike, dte_years, risk_free_rate, vol, OptionType.CALL) if call_alive else 0
        put_now = bs_price(price, put_leg.strike, dte_years, risk_free_rate, vol, OptionType.PUT) if put_alive else 0

        call_pnl = call_now - call_entry if call_alive else call_closed_pnl
        put_pnl = put_now - put_entry if put_alive else put_closed_pnl

        total_pnl = call_pnl + put_pnl

        # ── Final exit day: close everything ──
        if day_idx >= config.final_exit_day:
            if call_alive:
                call_closed_pnl = call_pnl
                call_alive = False
            if put_alive:
                put_closed_pnl = put_pnl
                put_alive = False
            exit_reason = "time_exit"
            # Set exit prices on legs
            call_leg.exit_price = call_now + call_entry  # restore to price level
            put_leg.exit_price = put_now + put_entry
            return call_closed_pnl + put_closed_pnl, exit_reason, day_idx + 1

        # ── Both legs alive: check for winner/loser ──
        if call_alive and put_alive:
            call_return = call_pnl / call_entry if call_entry > 0 else 0
            put_return = put_pnl / put_entry if put_entry > 0 else 0

            # Check if one leg is a clear winner
            if call_return >= config.winner_threshold_pct and put_return < -config.loser_recovery_zone:
                # Call winning, cut the put
                put_closed_pnl = put_pnl
                put_alive = False
                winner_peak = call_pnl
                exit_reason = "cut_loser_put"

            elif put_return >= config.winner_threshold_pct and call_return < -config.loser_recovery_zone:
                # Put winning, cut the call
                call_closed_pnl = call_pnl
                call_alive = False
                winner_peak = put_pnl
                exit_reason = "cut_loser_call"

            elif day_idx >= config.max_wait_days:
                # No clear winner by deadline — close both
                call_closed_pnl = call_pnl
                put_closed_pnl = put_pnl
                call_alive = False
                put_alive = False
                exit_reason = "no_direction"
                return call_closed_pnl + put_closed_pnl, exit_reason, day_idx + 1

        # ── One leg alive: trailing stop on winner ──
        elif call_alive and not put_alive:
            if call_pnl > winner_peak:
                winner_peak = call_pnl
            trail_level = winner_peak * (1 - config.trailing_stop_pct)
            if winner_peak > 0 and call_pnl <= trail_level:
                call_closed_pnl = call_pnl
                call_alive = False
                exit_reason = "trailing_stop_call"
                return call_closed_pnl + put_closed_pnl, exit_reason, day_idx + 1

        elif put_alive and not call_alive:
            if put_pnl > winner_peak:
                winner_peak = put_pnl
            trail_level = winner_peak * (1 - config.trailing_stop_pct)
            if winner_peak > 0 and put_pnl <= trail_level:
                put_closed_pnl = put_pnl
                put_alive = False
                exit_reason = "trailing_stop_put"
                return call_closed_pnl + put_closed_pnl, exit_reason, day_idx + 1

        # Both closed somehow
        if not call_alive and not put_alive:
            return call_closed_pnl + put_closed_pnl, exit_reason, day_idx + 1

    # Held to expiry (shouldn't happen with final_exit_day)
    if call_alive:
        call_closed_pnl = max(daily_prices[-1] - call_leg.strike, 0) - call_entry
    if put_alive:
        put_closed_pnl = max(put_leg.strike - daily_prices[-1], 0) - put_entry

    return call_closed_pnl + put_closed_pnl, exit_reason, total_days
