"""Core backtesting engine — runs strategies over weekly cycles."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import BacktestConfig
from ..data.fetcher import fetch_all_equities
from ..pricing.black_scholes import bs_price
from ..pricing.volatility import implied_vol_estimate, calibrated_vol_estimate, vol_on_date
from ..pricing.calibration import TickerCalibration, IVCalibrator
from ..strategies.base import Strategy, TradeResult
from .exit_rules import ExitRules, HOLD_TO_EXPIRY


def _find_weekly_entries(dates: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Find (entry, expiry) pairs for weekly option cycles.

    Entry: Monday (or first trading day of the week).
    Expiry: Friday (or last trading day of the week).
    Only returns complete weeks.
    """
    df = pd.DataFrame({"date": dates})
    df["weekday"] = df["date"].dt.weekday  # 0=Mon, 4=Fri
    df["week"] = df["date"].dt.isocalendar().week.values
    df["year"] = df["date"].dt.year

    pairs = []
    for (year, week), group in df.groupby(["year", "week"]):
        days = group.sort_values("date")
        if len(days) < 3:  # skip short weeks (holidays)
            continue
        entry = days.iloc[0]["date"]   # first trading day of week
        expiry = days.iloc[-1]["date"]  # last trading day of week
        pairs.append((entry, expiry))

    return pairs


@dataclass
class BacktestEngine:
    config: BacktestConfig = field(default_factory=BacktestConfig)
    trades: list[TradeResult] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    def run(
        self,
        strategy: Strategy,
        tickers: list[str] | None = None,
        calibrations: dict[str, TickerCalibration] | None = None,
        exit_rules: ExitRules | None = None,
        verbose: bool = True,
    ) -> list[TradeResult]:
        """Run a strategy across all tickers over the backtest period.

        Args:
            calibrations: Per-ticker IV calibration. If None, tries to load
                          saved calibration, then falls back to flat IV markup.
        Returns list of TradeResult for every trade taken.
        """
        tickers = tickers or self.config.tickers
        cfg = self.config

        # Load calibration if not provided
        if calibrations is None:
            calibrations = IVCalibrator.load_calibration() or {}
            if calibrations and verbose:
                print(f"Using saved IV calibration for {len(calibrations)} tickers")

        if verbose:
            print(f"Fetching data for {len(tickers)} tickers...")
        all_data = fetch_all_equities(tickers, cfg.start_date, cfg.end_date)

        self.trades = []
        capital = cfg.initial_capital
        capital_ts = {}

        for ticker, df in all_data.items():
            cal = calibrations.get(ticker)
            mode = "calibrated" if cal and cal.sample_dates > 0 else "flat markup"
            if verbose:
                print(f"  Running {strategy.name} on {ticker} (IV: {mode})...")

            # Raw HV series (no markup) for regime scaling
            raw_hv_series = implied_vol_estimate(df["Close"], cfg.hv_window, 1.0)

            if cal and cal.sample_dates > 0:
                vol_series = calibrated_vol_estimate(
                    df["Close"], cal, moneyness=1.0, window=cfg.hv_window,
                )
            else:
                vol_series = implied_vol_estimate(df["Close"], cfg.hv_window, cfg.iv_markup)

            weekly_pairs = _find_weekly_entries(df.index)

            for entry_dt, expiry_dt in weekly_pairs:
                if entry_dt not in df.index or expiry_dt not in df.index:
                    continue

                entry_price = float(df.loc[entry_dt, "Close"])
                expiry_price = float(df.loc[expiry_dt, "Close"])
                hv_now = vol_on_date(raw_hv_series, entry_dt)

                # Use calibrated vol with moneyness awareness
                if cal and cal.sample_dates > 0:
                    vol = cal.iv_for_moneyness(1.0, None)  # ATM baseline
                    # Scale by current HV regime vs calibration snapshot
                    if cal.historical_vol > 0 and hv_now > 0:
                        vol = vol * (hv_now / cal.historical_vol)
                else:
                    vol = vol_on_date(vol_series, entry_dt)

                if vol <= 0.01:
                    continue

                # DTE in years (trading days between entry and expiry / 252)
                mask = (df.index >= entry_dt) & (df.index <= expiry_dt)
                trading_days = mask.sum()
                dte_years = max(trading_days / 252, 1 / 252)

                # Construct strategy legs
                try:
                    legs = strategy.construct(entry_price, vol, dte_years, cfg.risk_free_rate)
                except Exception:
                    continue

                # Re-price each leg with calibrated moneyness-specific IV
                if cal and cal.sample_dates > 0 and hv_now > 0:
                    from ..pricing.black_scholes import bs_price
                    for leg in legs:
                        moneyness = leg.strike / entry_price
                        leg_iv = cal.iv_for_moneyness(moneyness, leg.option_type)
                        # Scale by current HV regime
                        leg_iv = leg_iv * (hv_now / cal.historical_vol)
                        leg.entry_price = bs_price(
                            entry_price, leg.strike, dte_years,
                            cfg.risk_free_rate, leg_iv, leg.option_type,
                        )

                # Position sizing: how many contracts can we afford?
                max_risk_per = strategy.max_risk(legs, entry_price)
                if max_risk_per <= 0:
                    continue

                alloc = min(capital, cfg.initial_capital * 3) * cfg.max_ticker_allocation
                contracts = max(1, min(int(alloc / max_risk_per), 100))

                # Compute entry credit/debit for exit rules
                entry_credit = sum(
                    leg.entry_price * (-leg.sign) for leg in legs
                )  # positive for net credit positions

                # ── Daily simulation with exit rules ──
                week_dates = df.index[(df.index >= entry_dt) & (df.index <= expiry_dt)]
                exit_dt = expiry_dt
                exit_underlying = expiry_price
                exit_reason_str = "expiry"

                if exit_rules is not None and len(week_dates) > 1:
                    from .exit_rules import (
                        PositionState, check_exit, evaluate_position_at_time,
                    )

                    state = PositionState(
                        entry_credit=entry_credit,
                        max_profit=entry_credit * 100,  # per contract
                    )

                    exited_early = False
                    for day_idx, day_dt in enumerate(week_dates[1:], 1):  # skip entry day
                        day_price = float(df.loc[day_dt, "Close"])
                        days_left = len(week_dates) - day_idx - 1
                        remaining_dte = max(days_left / 252, 0.5 / 252)

                        should_exit, reason = check_exit(
                            exit_rules, state, legs,
                            day_price, vol, remaining_dte, cfg.risk_free_rate,
                            days_left,
                        )

                        state.days_held = day_idx

                        if should_exit and reason is not None:
                            exit_dt = day_dt
                            exit_underlying = day_price
                            exit_reason_str = reason.value

                            # Price legs at exit time (not intrinsic — still have time value)
                            for leg in legs:
                                leg.exit_price = bs_price(
                                    day_price, leg.strike, remaining_dte,
                                    cfg.risk_free_rate, vol, leg.option_type,
                                )
                            exited_early = True
                            break

                    if not exited_early:
                        # Held to expiry — use intrinsic value
                        legs = strategy.evaluate_at_expiry(legs, expiry_price)
                else:
                    # No exit rules — hold to expiry
                    legs = strategy.evaluate_at_expiry(legs, expiry_price)

                # Calculate P&L
                pnl = 0.0
                premium_collected = 0.0
                premium_paid = 0.0
                for leg in legs:
                    leg_pnl = leg.pnl_per_contract() * contracts
                    pnl += leg_pnl
                    if not leg.is_long:
                        premium_collected += leg.entry_price * leg.multiplier * contracts
                    else:
                        premium_paid += leg.entry_price * leg.multiplier * contracts

                # Apply costs (double for early exit — pay to open AND close)
                num_legs = len(legs)
                exit_cost_multiplier = 2 if exit_reason_str != "expiry" else 1
                total_slippage = cfg.slippage_per_contract * contracts * num_legs * 100 * exit_cost_multiplier
                total_commission = cfg.commission_per_contract * contracts * num_legs * exit_cost_multiplier
                pnl -= (total_slippage + total_commission)

                days_held = len(df.index[(df.index >= entry_dt) & (df.index <= exit_dt)])

                trade = TradeResult(
                    ticker=ticker,
                    strategy_name=strategy.name,
                    entry_date=entry_dt,
                    exit_date=exit_dt,
                    entry_underlying=entry_price,
                    exit_underlying=exit_underlying,
                    legs=legs,
                    contracts=contracts,
                    total_pnl=pnl,
                    total_premium_collected=premium_collected,
                    total_premium_paid=premium_paid,
                    max_risk=max_risk_per * contracts,
                    return_on_risk=pnl / (max_risk_per * contracts) if max_risk_per > 0 else 0,
                    exit_reason=exit_reason_str,
                    days_held=days_held,
                )
                self.trades.append(trade)
                capital += pnl
                capital_ts[exit_dt] = capital

        # Build equity curve
        if capital_ts:
            self.equity_curve = pd.Series(capital_ts).sort_index()
            self.equity_curve.iloc[0] = cfg.initial_capital if self.equity_curve.iloc[0] == cfg.initial_capital else self.equity_curve.iloc[0]

        if verbose:
            print(f"Completed: {len(self.trades)} trades across {len(all_data)} tickers")

        return self.trades

    def trades_df(self) -> pd.DataFrame:
        """Convert trades to a DataFrame for analysis."""
        if not self.trades:
            return pd.DataFrame()

        records = []
        for t in self.trades:
            records.append({
                "ticker": t.ticker,
                "strategy": t.strategy_name,
                "entry_date": t.entry_date,
                "exit_date": t.exit_date,
                "entry_price": t.entry_underlying,
                "exit_price": t.exit_underlying,
                "underlying_return": (t.exit_underlying - t.entry_underlying) / t.entry_underlying,
                "contracts": t.contracts,
                "premium_collected": t.total_premium_collected,
                "premium_paid": t.total_premium_paid,
                "pnl": t.total_pnl,
                "max_risk": t.max_risk,
                "return_on_risk": t.return_on_risk,
                "num_legs": len(t.legs),
                "strikes": [leg.strike for leg in t.legs],
                "exit_reason": t.exit_reason,
                "days_held": t.days_held,
            })

        return pd.DataFrame(records)
