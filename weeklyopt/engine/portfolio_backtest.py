"""Portfolio-level backtest driven by the signal scanner.

Each week:
1. Run the scanner on all tickers
2. Rank opportunities by score
3. Allocate capital across top picks (respecting budget + contract limits)
4. Execute, manage with exit rules, record results

This simulates what you'd actually do: let the data tell you what to trade each week.
"""

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from ..config import BacktestConfig, TICKERS
from ..data.fetcher import fetch_all_equities
from ..pricing.black_scholes import bs_price, OptionType
from ..pricing.volatility import historical_vol, implied_vol_estimate, vol_on_date
from ..data.fetcher import fetch_equity_data
from ..pricing.calibration import TickerCalibration, IVCalibrator
from ..strategies import (
    CoveredCall, CashSecuredPut, IronCondor,
    BullPutSpread, BearCallSpread, Straddle, Strangle,
    BullCallSpread, BearPutSpread, LongCall, LongPut,
    ManagedLongStraddle, ManagedLongStrangle,
)
from ..strategies.base import Strategy, TradeResult
from .signals import analyze_ticker, TickerSignal
from .exit_rules import ExitRules, check_exit, PositionState, THETA_HARVEST
from .fundamentals import FundamentalData
from .filters import (
    FilterConfig, check_iv_rank_gate, check_earnings_filter,
    check_backwardation_filter, check_correlation_cluster,
    dynamic_delta_for_iv_rank, apply_entry_timing, max_pain_score_adjustment,
)
from .rolling import apply_roll_to_portfolio
from .ml_exit import MLExitModel, ExitCheckpoint


STRATEGY_MAP = {
    "cash_secured_put": CashSecuredPut,
    "covered_call": CoveredCall,
    "iron_condor": IronCondor,
    "bull_put_spread": BullPutSpread,
    "bear_call_spread": BearCallSpread,
    "short_straddle": Straddle,
    "short_strangle": Strangle,
    "bull_call_spread": BullCallSpread,
    "bear_put_spread": BearPutSpread,
    "long_call": LongCall,
    "long_put": LongPut,
    "managed_straddle": ManagedLongStraddle,
    "managed_strangle": ManagedLongStrangle,
}

# Strategies where max loss = initial outlay (no blowup risk)
DEFINED_RISK_ONLY = {
    "bull_call_spread", "bear_put_spread", "long_call", "long_put",
    "managed_straddle", "managed_strangle",
}

# Credit spreads: sell theta + hard capped max loss = spread width - credit
CREDIT_SPREADS_ONLY = {
    "bull_put_spread", "bear_call_spread", "iron_condor",
}


@dataclass
class WeekAllocation:
    """What the scanner decided for a specific week."""
    week_start: pd.Timestamp
    week_end: pd.Timestamp
    picks: list[dict] = field(default_factory=list)
    # Each pick: {ticker, strategy, score, reason, contracts, capital_used}
    total_capital_used: float = 0.0
    total_contracts: int = 0
    regime_modifier: float = 1.0


@dataclass
class PortfolioConfig:
    """Portfolio-level constraints."""
    weekly_budget: float | None = None  # fixed budget (overrides dynamic)
    max_contracts_per_week: int = 10
    max_contracts_per_ticker: int = 3
    min_score: float = 50.0  # only trade signals scoring above this
    max_positions_per_week: int = 5  # don't spread too thin
    risk_limit: float = 0.05  # max % of current capital to risk per week
    defined_risk_only: bool = False  # only use strategies where max loss = outlay
    credit_spreads_only: bool = False  # only credit spreads (sell theta + capped risk)
    filters: FilterConfig = field(default_factory=FilterConfig)  # all 11 enhancements
    enable_rolling: bool = True  # #11: multi-leg rolling


@dataclass
class PortfolioBacktest:
    """Run a signal-driven portfolio backtest."""

    config: BacktestConfig = field(default_factory=BacktestConfig)
    portfolio_config: PortfolioConfig = field(default_factory=PortfolioConfig)
    exit_rules: ExitRules | None = None

    # Results
    trades: list[TradeResult] = field(default_factory=list)
    weekly_allocations: list[WeekAllocation] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    @staticmethod
    def _compute_regime_modifier(
        vix_data: pd.DataFrame | None,
        equity_data: dict[str, pd.DataFrame],
        as_of: pd.Timestamp,
    ) -> float:
        """Compute position size modifier from market regime at a point in time.

        Returns 0.5-1.25 multiplier on position size.
        """
        modifier = 1.0

        # VIX component
        if vix_data is not None and not vix_data.empty:
            vix_hist = vix_data.loc[:as_of, "Close"]
            if len(vix_hist) >= 20:
                vix_now = float(vix_hist.iloc[-1])
                vix_1yr = vix_hist.tail(252)
                vix_rank = (
                    (vix_now - vix_1yr.min()) / (vix_1yr.max() - vix_1yr.min()) * 100
                    if vix_1yr.max() != vix_1yr.min() else 50
                )

                # Elevated VIX = rich premium but riskier
                if vix_rank > 75:
                    modifier *= 0.75  # size down in extreme vol
                elif vix_rank > 50:
                    modifier *= 1.1   # slightly elevated = good for premium selling
                elif vix_rank < 20:
                    modifier *= 0.9   # thin premium, reduce

        # Correlation component
        returns = {}
        for ticker, df in equity_data.items():
            hist = df.loc[:as_of, "Close"]
            if len(hist) >= 21:
                ret = hist.pct_change().dropna().tail(20)
                if len(ret) == 20:
                    returns[ticker] = ret.values

        if len(returns) >= 5:
            tickers = list(returns.keys())
            corrs = []
            for i in range(len(tickers)):
                for j in range(i+1, len(tickers)):
                    c = np.corrcoef(returns[tickers[i]], returns[tickers[j]])[0, 1]
                    if not np.isnan(c):
                        corrs.append(c)
            avg_corr = np.mean(corrs) if corrs else 0.3

            if avg_corr > 0.6:
                modifier *= 0.7   # high correlation = systematic risk
            elif avg_corr < 0.2:
                modifier *= 1.1   # low correlation = diversification works

        # Breadth component
        above_20 = 0
        total = 0
        for ticker, df in equity_data.items():
            hist = df.loc[:as_of, "Close"]
            if len(hist) >= 20:
                total += 1
                if float(hist.iloc[-1]) > float(hist.rolling(20).mean().iloc[-1]):
                    above_20 += 1
        if total > 0:
            breadth = above_20 / total
            if breadth < 0.30:
                modifier *= 0.8
            elif breadth > 0.70:
                modifier *= 1.1

        return max(0.5, min(1.25, modifier))

    def run(self, verbose: bool = True) -> list[TradeResult]:
        cfg = self.config
        pcfg = self.portfolio_config

        # Load calibration
        calibrations = IVCalibrator.load_calibration() or {}
        if calibrations and verbose:
            print(f"Loaded IV calibration for {len(calibrations)} tickers")

        # Fetch all data
        if verbose:
            print(f"Fetching data for {len(cfg.tickers)} tickers...")
        all_data = fetch_all_equities(cfg.tickers, cfg.start_date, cfg.end_date)

        # Fetch VIX for regime analysis
        if verbose:
            print("Fetching VIX data for regime analysis...")
        try:
            import yfinance as yf
            vix_data = yf.download("^VIX", start=str(cfg.start_date - __import__('datetime').timedelta(days=365)),
                                    end=str(cfg.end_date), progress=False)
            if isinstance(vix_data.columns, pd.MultiIndex):
                vix_data.columns = vix_data.columns.get_level_values(0)
        except Exception:
            vix_data = None

        # Pre-compute HV and vol series
        hv_cache = {}
        vol_cache = {}
        for ticker, df in all_data.items():
            hv_cache[ticker] = historical_vol(df["Close"], cfg.hv_window)
            cal = calibrations.get(ticker)
            if cal and cal.sample_dates > 0:
                from ..pricing.volatility import calibrated_vol_estimate
                vol_cache[ticker] = calibrated_vol_estimate(
                    df["Close"], cal, moneyness=1.0, window=cfg.hv_window,
                )
            else:
                vol_cache[ticker] = implied_vol_estimate(df["Close"], cfg.hv_window, cfg.iv_markup)

        # Find weekly periods from the first ticker with data
        first_df = next(iter(all_data.values()))
        weekly_pairs = self._find_weekly_entries(first_df.index)

        self.trades = []
        self.weekly_allocations = []
        capital = cfg.initial_capital
        capital_ts = {}

        if verbose:
            if pcfg.weekly_budget is not None:
                budget_str = f"${pcfg.weekly_budget:.0f}/week fixed"
            else:
                budget_str = f"{pcfg.risk_limit:.0%} of capital/week (${cfg.initial_capital * pcfg.risk_limit:,.0f} starting)"
            print(f"Running portfolio backtest: {len(weekly_pairs)} weeks, "
                  f"{budget_str}, {pcfg.max_contracts_per_week} max contracts\n")

        for week_idx, (entry_dt, expiry_dt) in enumerate(weekly_pairs):
            # ── 1. Scan all tickers for this week ──
            signals = []
            for ticker, df in all_data.items():
                if entry_dt not in df.index:
                    continue
                # Use price history up to entry date for signal
                hist = df.loc[:entry_dt, "Close"]
                if len(hist) < 50:
                    continue

                cal = calibrations.get(ticker)
                signal = analyze_ticker(ticker, hist, cal)
                signals.append(signal)

            # ── 2. Rank and filter ──
            allowed_set = None
            if pcfg.credit_spreads_only:
                allowed_set = CREDIT_SPREADS_ONLY
            elif pcfg.defined_risk_only:
                allowed_set = DEFINED_RISK_ONLY

            if allowed_set is not None:
                for s in signals:
                    filtered = {k: v for k, v in s.scores.items() if k in allowed_set}
                    if filtered:
                        best = max(filtered, key=filtered.get)
                        s.recommended_strategy = best
                        s.scores = filtered
                    else:
                        s.scores = {}

            signals = [s for s in signals if s.scores and max(s.scores.values(), default=0) >= pcfg.min_score]
            signals.sort(key=lambda s: max(s.scores.values()), reverse=True)

            # ── 3. Allocate capital across top picks ──
            # Dynamic budget: risk_limit % of current capital, or fixed budget
            if pcfg.weekly_budget is not None:
                base_budget = pcfg.weekly_budget
            else:
                base_budget = capital * pcfg.risk_limit

            # Apply market regime modifier
            regime_mod = self._compute_regime_modifier(vix_data, all_data, entry_dt)
            adjusted_budget = base_budget * regime_mod
            adjusted_contracts = max(1, int(pcfg.max_contracts_per_week * regime_mod))

            week_alloc = WeekAllocation(week_start=entry_dt, week_end=expiry_dt)
            budget_remaining = adjusted_budget
            contracts_remaining = adjusted_contracts

            selected_tickers_this_week = []  # for correlation filter

            for signal in signals[:pcfg.max_positions_per_week * 2]:  # scan more, filter more
                if budget_remaining <= 0 or contracts_remaining <= 0:
                    break
                if len(selected_tickers_this_week) >= pcfg.max_positions_per_week:
                    break

                ticker = signal.ticker
                best_strat_name = signal.recommended_strategy
                best_score = signal.scores.get(best_strat_name, 0)
                filt = pcfg.filters

                if ticker not in all_data or entry_dt not in all_data[ticker].index:
                    continue
                if expiry_dt not in all_data[ticker].index:
                    continue

                # #2: IV rank hard gate
                if not check_iv_rank_gate(signal.iv_rank, filt):
                    continue

                # #3: Earnings filter
                can_trade, earnings_mult = check_earnings_filter(signal.days_to_earnings, filt)
                if not can_trade:
                    continue

                # #4: VIX backwardation filter
                if not check_backwardation_filter(vix_data, entry_dt, best_strat_name, filt):
                    continue

                # #7: Correlation cluster filter
                if not check_correlation_cluster(ticker, selected_tickers_this_week, filt):
                    continue

                df = all_data[ticker]
                entry_price = float(df.loc[entry_dt, "Close"])
                expiry_price = float(df.loc[expiry_dt, "Close"])

                # Get vol for this ticker/date
                vol = vol_on_date(vol_cache[ticker], entry_dt)
                if vol <= 0.01:
                    continue

                # Compute DTE
                mask = (df.index >= entry_dt) & (df.index <= expiry_dt)
                trading_days = mask.sum()
                dte_years = max(trading_days / 252, 1 / 252)

                # Build strategy with dynamic delta (#6)
                strat_cls = STRATEGY_MAP.get(best_strat_name)
                if strat_cls is None:
                    continue

                strat_kwargs = {}
                if filt.dynamic_delta_enabled:
                    dyn_delta = dynamic_delta_for_iv_rank(signal.iv_rank)
                    # Apply dynamic delta to whichever field the strategy uses
                    for field_name in ["short_delta", "call_delta", "put_delta",
                                       "short_put_delta", "short_call_delta", "delta",
                                       "long_delta"]:
                        if hasattr(strat_cls, field_name):
                            strat_kwargs[field_name] = dyn_delta

                strategy = strat_cls(**strat_kwargs)

                try:
                    legs = strategy.construct(entry_price, vol, dte_years, cfg.risk_free_rate)
                except Exception:
                    continue

                # Re-price with calibration
                cal = calibrations.get(ticker)
                raw_hv = vol_on_date(
                    implied_vol_estimate(df["Close"], cfg.hv_window, 1.0), entry_dt
                )
                if cal and cal.sample_dates > 0 and raw_hv > 0:
                    for leg in legs:
                        moneyness = leg.strike / entry_price
                        leg_iv = cal.iv_for_moneyness(moneyness, leg.option_type)
                        leg_iv = leg_iv * (raw_hv / cal.historical_vol)
                        leg.entry_price = bs_price(
                            entry_price, leg.strike, dte_years,
                            cfg.risk_free_rate, leg_iv, leg.option_type,
                        )

                # #5: Apply entry timing premium to credit legs
                if filt.entry_timing_enabled:
                    for leg in legs:
                        if not leg.is_long:
                            leg.entry_price = apply_entry_timing(leg.entry_price, filt)

                # Position sizing within budget
                max_risk_per = strategy.max_risk(legs, entry_price)
                if max_risk_per <= 0:
                    continue

                # #3: Earnings multiplier (size up post-earnings)
                effective_budget = budget_remaining * earnings_mult

                # How many contracts can we afford?
                cost_per_contract = max_risk_per
                affordable = max(1, int(effective_budget / (cost_per_contract / 100)))
                contracts = min(
                    affordable,
                    contracts_remaining,
                    pcfg.max_contracts_per_ticker,
                )

                if contracts <= 0:
                    continue

                capital_used = (cost_per_contract / 100) * contracts

                # ── 4. Execute trade with exit rules ──
                week_dates = df.index[(df.index >= entry_dt) & (df.index <= expiry_dt)]
                exit_dt = expiry_dt
                exit_underlying = expiry_price
                exit_reason_str = "expiry"

                # Managed straddle/strangle: use custom leg management
                if best_strat_name in ("managed_straddle", "managed_strangle") and len(week_dates) > 1:
                    from .straddle_manager import simulate_managed_straddle, StraddleManagerConfig

                    daily_prices = [float(df.loc[d, "Close"]) for d in week_dates]
                    pnl_per_share, mgmt_reason, days = simulate_managed_straddle(
                        legs, daily_prices, vol, cfg.risk_free_rate,
                    )

                    # Set exit prices so P&L calc works
                    total_entry = sum(leg.entry_price for leg in legs)
                    # Distribute the pnl back into legs proportionally
                    for leg in legs:
                        leg.exit_price = leg.entry_price + (pnl_per_share * leg.entry_price / total_entry) if total_entry > 0 else 0

                    exit_dt = week_dates[min(days, len(week_dates) - 1)]
                    exit_underlying = daily_prices[min(days, len(daily_prices) - 1)]
                    exit_reason_str = mgmt_reason

                elif self.exit_rules is not None and len(week_dates) > 1:
                    # Try ML exit model first, fall back to rule-based
                    ml_model = MLExitModel.load() if not hasattr(self, '_ml_model') else self._ml_model
                    self._ml_model = ml_model

                    entry_credit = sum(leg.entry_price * (-leg.sign) for leg in legs)
                    state = PositionState(
                        entry_credit=entry_credit,
                        max_profit=entry_credit * 100,
                    )

                    exited_early = False
                    for day_idx, day_dt in enumerate(week_dates[1:], 1):
                        day_price = float(df.loc[day_dt, "Close"])
                        days_left = len(week_dates) - day_idx - 1
                        remaining_dte = max(days_left / 252, 0.5 / 252)

                        should_exit, reason = check_exit(
                            self.exit_rules, state, legs,
                            day_price, vol, remaining_dte, cfg.risk_free_rate,
                            days_left,
                        )
                        state.days_held = day_idx

                        # Also consult ML model if available
                        if not should_exit and ml_model.is_trained:
                            ml_cp = ExitCheckpoint(
                                day_of_week=day_dt.weekday(),
                                days_remaining=days_left,
                                unrealized_pnl_pct=state.current_pnl / state.max_profit if state.max_profit > 0 else 0,
                                unrealized_loss_pct=max(0, -state.current_pnl / state.max_profit) if state.max_profit > 0 else 0,
                                position_delta=(day_price - entry_price) / entry_price * 0.3,
                                position_gamma=0.05 / max(days_left, 0.5),
                                underlying_1d_return=0,
                                underlying_3d_return=0,
                                vix_level=20,
                                vix_1d_change=0,
                                iv_rank=signal.iv_rank,
                            )
                            ml_exit, ml_conf = ml_model.predict_exit(ml_cp)
                            if ml_exit and ml_conf > 0.7:
                                should_exit = True
                                reason = type('R', (), {'value': 'ml_exit'})()

                        if should_exit and reason is not None:
                            exit_dt = day_dt
                            exit_underlying = day_price
                            exit_reason_str = reason.value
                            for leg in legs:
                                leg.exit_price = bs_price(
                                    day_price, leg.strike, remaining_dte,
                                    cfg.risk_free_rate, vol, leg.option_type,
                                )
                            exited_early = True
                            break

                    if not exited_early:
                        legs = strategy.evaluate_at_expiry(legs, expiry_price)
                else:
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

                # Costs
                num_legs = len(legs)
                exit_mult = 2 if exit_reason_str != "expiry" else 1
                pnl -= cfg.slippage_per_contract * contracts * num_legs * 100 * exit_mult
                pnl -= cfg.commission_per_contract * contracts * num_legs * exit_mult

                days_held = len(df.index[(df.index >= entry_dt) & (df.index <= exit_dt)])

                trade = TradeResult(
                    ticker=ticker,
                    strategy_name=best_strat_name,
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

                # Track allocation
                week_alloc.picks.append({
                    "ticker": ticker,
                    "strategy": best_strat_name,
                    "score": best_score,
                    "reason": signal.recommendation_reason,
                    "contracts": contracts,
                    "capital_used": capital_used,
                    "pnl": pnl,
                    "exit_reason": exit_reason_str,
                })
                budget_remaining -= capital_used
                selected_tickers_this_week.append(ticker)  # #7: track for correlation
                contracts_remaining -= contracts

            week_alloc.total_capital_used = adjusted_budget - budget_remaining
            week_alloc.regime_modifier = regime_mod
            week_alloc.total_contracts = pcfg.max_contracts_per_week - contracts_remaining
            self.weekly_allocations.append(week_alloc)
            capital_ts[expiry_dt] = capital

        # #11: Apply multi-leg rolling savings
        roll_savings = 0.0
        if pcfg.enable_rolling:
            self.trades, roll_savings = apply_roll_to_portfolio(
                self.trades, cfg.slippage_per_contract, cfg.commission_per_contract,
            )

        # Rebuild equity curve with rolling adjustments
        capital = cfg.initial_capital
        capital_ts = {}
        for trade in self.trades:
            capital += trade.total_pnl
            capital_ts[trade.exit_date] = capital

        if capital_ts:
            self.equity_curve = pd.Series(capital_ts).sort_index()

        if verbose:
            print(f"\nCompleted: {len(self.trades)} trades over {len(self.weekly_allocations)} weeks")
            if roll_savings > 0:
                print(f"Rolling savings: ${roll_savings:,.0f}")

        return self.trades

    def trades_df(self) -> pd.DataFrame:
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
                "exit_reason": t.exit_reason,
                "days_held": t.days_held,
            })
        return pd.DataFrame(records)

    def print_weekly_log(self, last_n: int | None = None):
        """Print what the scanner picked each week."""
        allocs = self.weekly_allocations
        if last_n:
            allocs = allocs[-last_n:]

        print(f"\n{'='*95}")
        print(f"  Weekly Portfolio Allocation Log")
        print(f"{'='*95}")

        for wa in allocs:
            week_pnl = sum(p["pnl"] for p in wa.picks)
            print(f"\n  Week: {wa.week_start.date()} → {wa.week_end.date()}  "
                  f"|  Positions: {len(wa.picks)}  |  Contracts: {wa.total_contracts}  "
                  f"|  Capital: ${wa.total_capital_used:.0f}  |  P&L: ${week_pnl:+,.0f}"
                  f"  |  Regime: {wa.regime_modifier:.0%}")

            if wa.picks:
                for p in wa.picks:
                    pnl_str = f"${p['pnl']:+,.0f}"
                    print(f"    {p['ticker']:>6} {p['strategy']:>20s}  "
                          f"score={p['score']:.0f}  x{p['contracts']}  "
                          f"{pnl_str:>8}  [{p['exit_reason']}]")
            else:
                print(f"    (no trades — no signals above threshold)")

        print(f"\n{'='*95}")

    def print_strategy_breakdown(self):
        """Show how each strategy performed when the scanner selected it."""
        df = self.trades_df()
        if df.empty:
            return

        print(f"\n{'='*85}")
        print(f"  Scanner Strategy Selection Performance")
        print(f"{'='*85}")
        print(f"  {'Strategy':>20s}  {'Trades':>6}  {'Win%':>5}  {'Total P&L':>10}  "
              f"{'Avg P&L':>8}  {'PF':>5}  {'Avg Days':>8}")
        print(f"  {'-'*80}")

        for strat in sorted(df["strategy"].unique()):
            subset = df[df["strategy"] == strat]
            wins = (subset["pnl"] > 0).sum()
            total = len(subset)
            total_pnl = subset["pnl"].sum()
            avg_pnl = subset["pnl"].mean()
            gross_win = subset[subset["pnl"] > 0]["pnl"].sum()
            gross_loss = abs(subset[subset["pnl"] <= 0]["pnl"].sum()) or 1e-9
            pf = gross_win / gross_loss
            avg_days = subset["days_held"].mean()

            print(f"  {strat:>20s}  {total:>6}  {wins/total:>4.0%}  "
                  f"${total_pnl:>9,.0f}  ${avg_pnl:>7,.0f}  {pf:>5.2f}  {avg_days:>7.1f}")

        print(f"\n  Total: {len(df)} trades, ${df['pnl'].sum():+,.0f} P&L")
        print(f"{'='*85}")

    @staticmethod
    def _find_weekly_entries(dates: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        df = pd.DataFrame({"date": dates})
        df["weekday"] = df["date"].dt.weekday
        df["week"] = df["date"].dt.isocalendar().week.values
        df["year"] = df["date"].dt.year

        pairs = []
        for (year, week), group in df.groupby(["year", "week"]):
            days = group.sort_values("date")
            if len(days) < 3:
                continue
            entry = days.iloc[0]["date"]
            expiry = days.iloc[-1]["date"]
            pairs.append((entry, expiry))

        return pairs
