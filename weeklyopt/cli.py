"""CLI entry point for weeklyopt."""

import argparse
import sys
from datetime import date

from .config import BacktestConfig, TICKERS
from .engine.backtest import BacktestEngine
from .strategies import (
    CoveredCall, CashSecuredPut, IronCondor,
    BullPutSpread, BearCallSpread, Straddle, Strangle,
)
from .analysis.metrics import compute_metrics, print_report
from .analysis.plots import plot_dashboard
from .validation.live_check import LiveValidator
from .pricing.black_scholes import OptionType


STRATEGIES = {
    "covered_call": CoveredCall,
    "cash_secured_put": CashSecuredPut,
    "iron_condor": IronCondor,
    "bull_put_spread": BullPutSpread,
    "bear_call_spread": BearCallSpread,
    "short_straddle": Straddle,
    "short_strangle": Strangle,
}


def main():
    parser = argparse.ArgumentParser(
        description="weeklyopt - Weekly options backtesting framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mode 3: Synthetic BS backtest (primary, free)
  weeklyopt backtest --strategy iron_condor
  weeklyopt backtest --strategy covered_call --tickers SPY QQQ AAPL --start 2022-01-01

  # Mode 1a: Spot-check BS vs live yfinance chains
  weeklyopt validate --tickers SPY AAPL --delta 0.30 --type call

  # Mode 1b: Historical validation against ThetaData (requires Theta Terminal)
  weeklyopt theta-validate --strategy iron_condor --tickers SPY --start 2024-06-01 --end 2024-12-31
  weeklyopt theta-validate --strategy iron_condor --tickers SPY --max-trades 20

  # Mode 1c: Full Lumibot + ThetaData production backtest
  weeklyopt lumibot --strategy iron_condor --symbol SPY --start 2024-01-01 --end 2025-01-01

  # Calibrate IV model from real ThetaData prices (run this first!)
  weeklyopt calibrate --tickers SPY QQQ AAPL TSLA --weeks 12

  # Compare: backtest then auto-validate winners
  weeklyopt compare --strategy iron_condor --tickers SPY

  # Scan: analyze all tickers and recommend strategies for this week
  weeklyopt scan
  weeklyopt scan --tickers SPY NVDA AAPL --detail
        """,
    )
    subparsers = parser.add_subparsers(dest="command")

    # Screen command (broad universe discovery)
    scr = subparsers.add_parser("screen", help="Screen broad universe for credit spread candidates")
    scr.add_argument("--top", type=int, default=30, help="Return top N tickers")
    scr.add_argument("--min-iv-rank", type=float, default=30, help="Minimum IV rank")
    scr.add_argument("--min-volume", type=int, default=500_000, help="Minimum avg daily volume")
    scr.add_argument("--min-oi", type=int, default=1_000, help="Minimum total option OI")

    # Scan command (weekly signal scanner)
    sc = subparsers.add_parser("scan", help="Scan tickers and recommend strategies for this week")
    sc.add_argument("--tickers", nargs="+", default=None, help="Tickers to scan (default: all 15)")
    sc.add_argument("--expanded", action="store_true", help="Use expanded 23-ticker universe")
    sc.add_argument("--dynamic", action="store_true", help="Use dynamic screener to find best tickers")
    sc.add_argument("--detail", action="store_true", help="Show detailed analysis per ticker")
    sc.add_argument("--trade-plan", action="store_true", help="Generate concrete trade plan for next week")
    sc.add_argument("--capital", type=float, default=20_000, help="Current account capital")
    sc.add_argument("--risk-limit", type=float, default=0.10, help="Max %% of capital to risk")
    sc.add_argument("--max-contracts", type=int, default=10, help="Max contracts per week")
    sc.add_argument("--credit-spreads", action="store_true", help="Only recommend credit spreads")

    # Portfolio backtest (scanner-driven)
    pb = subparsers.add_parser("portfolio", help="Backtest scanner-driven portfolio with weekly budget")
    pb.add_argument("--tickers", nargs="+", default=None)
    pb.add_argument("--expanded", action="store_true", help="Use expanded 23-ticker universe with sector ETFs")
    pb.add_argument("--start", type=str, default="2023-01-01")
    pb.add_argument("--end", type=str, default="2025-12-31")
    pb.add_argument("--budget", type=float, default=None, help="Fixed weekly budget (overrides dynamic sizing)")
    pb.add_argument("--max-contracts", type=int, default=10, help="Max contracts per week")
    pb.add_argument("--capital", type=float, default=20_000, help="Starting account capital ($)")
    pb.add_argument("--risk-limit", type=float, default=0.05, help="Max %% of capital to risk per week (0.05 = 5%%)")
    pb.add_argument("--exit-rules", choices=["hold", "greeks", "conservative", "theta", "aggressive"],
                    default="theta", help="Exit rule preset (default: theta)")
    pb.add_argument("--min-score", type=float, default=50, help="Min signal score to trade")
    pb.add_argument("--defined-risk", action="store_true",
                    help="Only debit strategies (max loss = premium paid)")
    pb.add_argument("--credit-spreads", action="store_true",
                    help="Only credit spreads (sell theta + capped max loss)")
    pb.add_argument("--weekly-log", type=int, default=None, help="Show last N weeks allocation log")
    pb.add_argument("--no-plot", action="store_true")

    # Calibrate command (run before backtest for accurate IV)
    cal = subparsers.add_parser("calibrate", help="Calibrate IV model from real ThetaData option prices")
    cal.add_argument("--tickers", nargs="+", default=None, help="Tickers to calibrate (default: all 15)")
    cal.add_argument("--weeks", type=int, default=12, help="Lookback period in weeks")

    # Backtest command (mode 3)
    bt = subparsers.add_parser("backtest", help="Run a backtest (mode 3: synthetic BS pricing)")
    bt.add_argument("--strategy", required=True, choices=list(STRATEGIES.keys()))
    bt.add_argument("--tickers", nargs="+", default=None, help="Tickers to test (default: all 15)")
    bt.add_argument("--start", type=str, default="2020-01-01")
    bt.add_argument("--end", type=str, default="2025-12-31")
    bt.add_argument("--capital", type=float, default=100_000)
    bt.add_argument("--delta", type=float, default=None, help="Override default delta for strategy")
    bt.add_argument("--spread-width", type=float, default=None, help="Override spread width (dollars)")
    bt.add_argument("--save-plot", type=str, default=None, help="Save dashboard plot to file")
    bt.add_argument("--no-plot", action="store_true", help="Skip plotting")
    bt.add_argument("--exit-rules", choices=["hold", "greeks", "conservative", "theta", "aggressive"],
                    default="hold", help="Exit rule preset (default: hold to expiry)")
    bt.add_argument("--profit-target", type=float, default=None, help="Override profit target %% (e.g., 0.50)")
    bt.add_argument("--stop-loss", type=float, default=None, help="Override stop loss multiple (e.g., 2.0)")
    bt.add_argument("--time-stop", type=int, default=None, help="Close N days before expiry")

    # Validate command (mode 1a: live snapshot)
    val = subparsers.add_parser("validate", help="Spot-check BS pricing against live yfinance chains")
    val.add_argument("--tickers", nargs="+", default=["SPY", "AAPL", "QQQ"])
    val.add_argument("--delta", type=float, default=0.30)
    val.add_argument("--type", choices=["call", "put"], default="call")

    # ThetaData historical validation (mode 1b)
    tv = subparsers.add_parser("theta-validate", help="Validate backtest trades against ThetaData historical prices")
    tv.add_argument("--strategy", required=True, choices=list(STRATEGIES.keys()))
    tv.add_argument("--tickers", nargs="+", default=None)
    tv.add_argument("--start", type=str, default="2024-06-01")
    tv.add_argument("--end", type=str, default="2025-03-31")
    tv.add_argument("--capital", type=float, default=100_000)
    tv.add_argument("--delta", type=float, default=None)
    tv.add_argument("--spread-width", type=float, default=None)
    tv.add_argument("--max-trades", type=int, default=50, help="Max trades to validate (rate limit)")
    tv.add_argument("--no-rate-limit", action="store_true", help="Disable rate limiting (paid tier)")

    # Lumibot full backtest (mode 1c)
    lb = subparsers.add_parser("lumibot", help="Full Lumibot + ThetaData production backtest")
    lb.add_argument("--strategy", required=True, choices=["iron_condor", "covered_call", "cash_secured_put"])
    lb.add_argument("--symbol", type=str, default="SPY")
    lb.add_argument("--start", type=str, default="2024-01-01")
    lb.add_argument("--end", type=str, default="2025-01-01")
    lb.add_argument("--capital", type=float, default=100_000)

    # Compare command
    cmp = subparsers.add_parser("compare", help="Backtest then validate top picks")
    cmp.add_argument("--strategy", required=True, choices=list(STRATEGIES.keys()))
    cmp.add_argument("--tickers", nargs="+", default=None)
    cmp.add_argument("--start", type=str, default="2022-01-01")
    cmp.add_argument("--end", type=str, default="2025-12-31")
    cmp.add_argument("--capital", type=float, default=100_000)
    cmp.add_argument("--top-n", type=int, default=3, help="Validate top N tickers by P&L")

    args = parser.parse_args()

    if args.command == "screen":
        _run_screen(args)
    elif args.command == "scan":
        _run_scan(args)
    elif args.command == "portfolio":
        _run_portfolio(args)
    elif args.command == "calibrate":
        _run_calibrate(args)
    elif args.command == "backtest":
        _run_backtest(args)
    elif args.command == "validate":
        _run_validate(args)
    elif args.command == "theta-validate":
        _run_theta_validate(args)
    elif args.command == "lumibot":
        _run_lumibot(args)
    elif args.command == "compare":
        _run_compare(args)
    else:
        parser.print_help()


def _run_calibrate(args):
    """Calibrate IV model using real ThetaData option prices."""
    from .pricing.calibration import IVCalibrator

    tickers = args.tickers or TICKERS
    calibrator = IVCalibrator()

    print(f"Calibrating IV model for {len(tickers)} tickers using last {args.weeks} weeks...")
    print(f"(Requires Theta Terminal running on port 25503)\n")

    calibrations = calibrator.calibrate_all(tickers, lookback_weeks=args.weeks, verbose=True)

    path = calibrator.save_calibration(calibrations)
    print(f"\nCalibration saved to: {path}")
    print("Future backtests will auto-load this calibration.")


def _run_portfolio(args):
    """Run scanner-driven portfolio backtest."""
    from .engine.portfolio_backtest import PortfolioBacktest, PortfolioConfig
    from .analysis.metrics import compute_metrics, print_report
    from .config import TICKERS_EXPANDED

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if args.tickers:
        tickers = args.tickers
    elif getattr(args, "expanded", False):
        tickers = TICKERS_EXPANDED
    else:
        tickers = TICKERS

    starting_capital = getattr(args, "capital", 20_000)

    bt_config = BacktestConfig(
        tickers=tickers,
        start_date=start,
        end_date=end,
        initial_capital=starting_capital,
    )

    port_config = PortfolioConfig(
        weekly_budget=args.budget,  # None = dynamic from risk_limit
        max_contracts_per_week=args.max_contracts,
        min_score=args.min_score,
        defined_risk_only=args.defined_risk,
        credit_spreads_only=getattr(args, "credit_spreads", False),
        risk_limit=getattr(args, "risk_limit", 0.05),
    )

    exit_rules = _build_exit_rules(args)

    engine = PortfolioBacktest(
        config=bt_config,
        portfolio_config=port_config,
        exit_rules=exit_rules,
    )

    if args.budget:
        budget_str = f"${args.budget:.0f}/week fixed"
    else:
        budget_str = f"{args.risk_limit:.0%} of capital/week"
    print(f"Portfolio Backtest: ${starting_capital:,.0f} capital, {budget_str}, {args.max_contracts} max contracts")
    print(f"Period: {start} → {end}, {len(tickers)} tickers, exit rules: {args.exit_rules}\n")

    engine.run(verbose=True)

    trades_df = engine.trades_df()
    if trades_df.empty:
        print("No trades generated.")
        return

    metrics = compute_metrics(trades_df, engine.equity_curve, starting_capital)
    print_report(metrics, "Scanner Portfolio", initial_capital=starting_capital)

    # Strategy breakdown
    engine.print_strategy_breakdown()

    # Exit reason breakdown
    if "exit_reason" in trades_df.columns:
        reason_counts = trades_df["exit_reason"].value_counts()
        if len(reason_counts) > 1 or reason_counts.index[0] != "expiry":
            print(f"\n  Exit Reasons:")
            for reason, count in reason_counts.items():
                pct = count / len(trades_df) * 100
                avg_pnl = trades_df[trades_df["exit_reason"] == reason]["pnl"].mean()
                print(f"    {reason:>20s}: {count:>4d} ({pct:>5.1f}%)  avg P&L: ${avg_pnl:>8,.0f}")

    # Weekly log
    if args.weekly_log:
        engine.print_weekly_log(args.weekly_log)

    if not args.no_plot:
        from .analysis.plots import plot_dashboard
        import matplotlib.pyplot as plt
        plot_dashboard(trades_df, engine.equity_curve, bt_config.initial_capital,
                       "Scanner Portfolio", None)
        plt.show()


def _run_screen(args):
    """Screen broad universe for credit spread candidates."""
    from .engine.screener import screen_universe, print_screen_results, ScreenerConfig

    config = ScreenerConfig(
        max_tickers=args.top,
        min_iv_rank=args.min_iv_rank,
        min_avg_volume=args.min_volume,
        min_option_oi=args.min_oi,
    )

    results = screen_universe(config=config, verbose=True)
    print_screen_results(results)

    # Show the tickers as a pasteable list
    tickers = [r.ticker for r in results]
    print(f"\n  Use these in scan:")
    print(f"  weeklyopt scan --tickers {' '.join(tickers)} --credit-spreads --trade-plan")


def _run_scan(args):
    """Scan all tickers and recommend strategies for this week."""
    from .data.fetcher import fetch_all_equities
    from .pricing.calibration import IVCalibrator
    from .engine.signals import scan_all_tickers, print_scan_results, print_detailed_signal
    from .engine.fundamentals import fetch_all_fundamentals, print_fundamentals_summary
    from datetime import timedelta

    from .config import TICKERS_EXPANDED

    if getattr(args, "dynamic", False):
        # Dynamic: screen broad universe first, then scan the survivors
        from .engine.screener import screen_universe, print_screen_results
        print("0. Running dynamic screener on broad universe...")
        screen_results = screen_universe(verbose=True)
        print_screen_results(screen_results)
        tickers = [r.ticker for r in screen_results]
        print(f"\n   Advancing {len(tickers)} tickers to signal analysis...\n")
    elif args.tickers:
        tickers = args.tickers
    elif getattr(args, "expanded", False):
        tickers = TICKERS_EXPANDED
    else:
        tickers = TICKERS

    # Fetch price data (last 6 months for signals)
    start = date.today() - timedelta(days=180)
    end = date.today()

    print(f"Scanning {len(tickers)} tickers for weekly options opportunities...\n")

    print("1. Fetching price data...")
    equity_data = fetch_all_equities(tickers, start, end)

    print("2. Fetching fundamentals & options flow...")
    fundamentals = fetch_all_fundamentals(tickers, verbose=True)
    print_fundamentals_summary(fundamentals)

    print("\n3. Loading IV calibration...")
    calibrations = IVCalibrator.load_calibration() or {}
    if calibrations:
        print(f"   Loaded calibration for {len(calibrations)} tickers")
    else:
        print("   No calibration found — run `weeklyopt calibrate` first for better results")

    print("\n4. Analyzing market regime...")
    from .engine.market_regime import analyze_market_regime, print_market_regime
    market = analyze_market_regime(equity_data)
    print_market_regime(market)

    print("\n5. Analyzing ticker signals...")
    signals = scan_all_tickers(equity_data, calibrations, fundamentals)

    print_scan_results(signals)

    if args.detail:
        for signal in signals:
            print_detailed_signal(signal)

    if getattr(args, "trade_plan", False):
        _print_trade_plan(args, signals, market, calibrations, fundamentals)


def _print_trade_plan(args, signals, market, calibrations, fundamentals):
    """Generate a concrete trade plan for next week."""
    from .engine.filters import (
        FilterConfig, check_iv_rank_gate, check_earnings_filter,
        check_correlation_cluster, dynamic_delta_for_iv_rank,
    )
    from .engine.oi_analysis import analyze_oi_structure, score_short_strike, print_oi_analysis
    from .pricing.black_scholes import bs_price, strike_from_delta, OptionType
    from .config import CORRELATION_CLUSTERS
    from datetime import timedelta

    filt = FilterConfig()
    capital = args.capital
    risk_limit = args.risk_limit
    max_contracts = args.max_contracts
    credit_spreads_only = getattr(args, "credit_spreads", False)

    # Apply regime modifier
    regime_mod = market.position_size_modifier
    budget = capital * risk_limit * regime_mod
    adj_contracts = max(1, int(max_contracts * regime_mod))

    # Next Monday
    today = __import__("datetime").date.today()
    days_to_monday = (7 - today.weekday()) % 7
    if days_to_monday == 0:
        days_to_monday = 7
    next_monday = today + timedelta(days=days_to_monday)
    next_friday = next_monday + timedelta(days=4)

    print(f"\n{'='*80}")
    print(f"  TRADE PLAN: Week of {next_monday}")
    print(f"{'='*80}")
    print(f"  Account:        ${capital:,.0f}")
    print(f"  Risk budget:    ${budget:,.0f}  ({risk_limit:.0%} x ${capital:,.0f} x {regime_mod:.0%} regime)")
    print(f"  Max contracts:  {adj_contracts}")
    print(f"  Expiry:         {next_friday}")
    print(f"  Regime:         {market.regime_label.upper()} (modifier: {regime_mod:.0%})")

    # Filter allowed strategies
    credit_set = {"bull_put_spread", "bear_call_spread", "iron_condor"}

    # Build picks
    picks = []
    selected = []
    budget_left = budget
    contracts_left = adj_contracts

    for signal in signals:
        if budget_left <= 0 or contracts_left <= 0 or len(picks) >= 5:
            break

        best = signal.recommended_strategy
        score = signal.scores.get(best, 0)

        if credit_spreads_only and best not in credit_set:
            # Find best credit spread for this ticker
            credit_scores = {k: v for k, v in signal.scores.items() if k in credit_set}
            if credit_scores:
                best = max(credit_scores, key=credit_scores.get)
                score = credit_scores[best]
            else:
                continue

        if score < 50:
            continue
        if not check_iv_rank_gate(signal.iv_rank, filt):
            continue
        can_trade, earn_mult = check_earnings_filter(signal.days_to_earnings, filt)
        if not can_trade:
            continue
        if not check_correlation_cluster(signal.ticker, selected, filt):
            continue

        # Compute strikes
        delta = dynamic_delta_for_iv_rank(signal.iv_rank)
        price = signal.price
        vol = signal.current_hv * 1.2  # rough IV estimate
        dte_years = 5 / 252  # ~1 week

        if best == "bull_put_spread":
            short_K = strike_from_delta(price, dte_years, 0.045, vol, delta, OptionType.PUT)
            long_K = short_K - 5
            short_prem = bs_price(price, short_K, dte_years, 0.045, vol, OptionType.PUT)
            long_prem = bs_price(price, long_K, dte_years, 0.045, vol, OptionType.PUT)
            credit = short_prem - long_prem
            max_loss = (5 - credit) * 100
            desc = f"Sell {short_K:.0f}P / Buy {long_K:.0f}P"
        elif best == "bear_call_spread":
            short_K = strike_from_delta(price, dte_years, 0.045, vol, delta, OptionType.CALL)
            long_K = short_K + 5
            short_prem = bs_price(price, short_K, dte_years, 0.045, vol, OptionType.CALL)
            long_prem = bs_price(price, long_K, dte_years, 0.045, vol, OptionType.CALL)
            credit = short_prem - long_prem
            max_loss = (5 - credit) * 100
            desc = f"Sell {short_K:.0f}C / Buy {long_K:.0f}C"
        elif best == "iron_condor":
            sp_K = strike_from_delta(price, dte_years, 0.045, vol, delta, OptionType.PUT)
            lp_K = sp_K - 5
            sc_K = strike_from_delta(price, dte_years, 0.045, vol, delta, OptionType.CALL)
            lc_K = sc_K + 5
            credit = (
                bs_price(price, sp_K, dte_years, 0.045, vol, OptionType.PUT)
                - bs_price(price, lp_K, dte_years, 0.045, vol, OptionType.PUT)
                + bs_price(price, sc_K, dte_years, 0.045, vol, OptionType.CALL)
                - bs_price(price, lc_K, dte_years, 0.045, vol, OptionType.CALL)
            )
            max_loss = (5 - credit) * 100
            desc = f"Sell {sp_K:.0f}P/{sc_K:.0f}C, Buy {lp_K:.0f}P/{lc_K:.0f}C"
        else:
            continue

        if max_loss <= 0:
            continue

        # OI analysis: score how well the chain structure protects this trade
        oi = analyze_oi_structure(signal.ticker)
        if best == "bull_put_spread":
            oi_score = score_short_strike(oi, short_K, "put")
        elif best == "bear_call_spread":
            oi_score = score_short_strike(oi, short_K, "call")
        elif best == "iron_condor":
            put_oi = score_short_strike(oi, sp_K, "put")
            call_oi = score_short_strike(oi, sc_K, "call")
            oi_score = (put_oi + call_oi) / 2
        else:
            oi_score = 50.0

        contracts = min(
            max(1, int(budget_left / max_loss)),
            contracts_left,
            3,
        )
        total_risk = max_loss * contracts
        total_credit = credit * 100 * contracts

        picks.append({
            "ticker": signal.ticker,
            "strategy": best,
            "score": score,
            "desc": desc,
            "delta": delta,
            "credit": credit,
            "max_loss": max_loss,
            "contracts": contracts,
            "total_risk": total_risk,
            "total_credit": total_credit,
            "iv_rank": signal.iv_rank,
            "regime": signal.regime.value,
            "reason": signal.recommendation_reason,
            "oi_score": oi_score,
            "oi_analysis": oi,
        })

        selected.append(signal.ticker)
        budget_left -= total_risk
        contracts_left -= contracts

    if not picks:
        print(f"\n  No trades meet criteria this week.")
        if market.premium_selling_score < 35:
            print(f"  Reason: Market regime unfavorable (premium score: {market.premium_selling_score:.0f}/100)")
        print(f"{'='*80}")
        return

    print(f"\n  {'#':>3}  {'Ticker':>6}  {'Strategy':>18}  {'Strikes':>30}  "
          f"{'Scr':>3}  {'OI':>3}  {'x':>2}  {'Credit':>8}  {'MaxLoss':>8}  {'Risk':>8}")
    print(f"  {'-'*112}")

    total_risk = 0
    total_credit = 0
    for i, p in enumerate(picks):
        oi_s = p.get("oi_score", 50)
        oi_flag = "+" if oi_s >= 65 else " " if oi_s >= 45 else "-"
        print(f"  {i+1:>3}  {p['ticker']:>6}  {p['strategy']:>18}  {p['desc']:>30}  "
              f"{p['score']:>3.0f}  {oi_s:>2.0f}{oi_flag} x{p['contracts']}  "
              f"${p['total_credit']:>6,.0f}  ${p['max_loss']:>6,.0f}  ${p['total_risk']:>6,.0f}")
        total_risk += p["total_risk"]
        total_credit += p["total_credit"]

    print(f"  {'-'*105}")
    print(f"  {'':>3}  {'':>6}  {'':>18}  {'TOTAL':>30}  "
          f"{'':>3}  {'':>2}  ${total_credit:>6,.0f}  {'':>8}  ${total_risk:>6,.0f}")

    print(f"\n  Capital at risk:  ${total_risk:,.0f} / ${budget:,.0f} budget "
          f"({total_risk/capital:.1%} of account)")
    print(f"  Max credit if all expire worthless: ${total_credit:,.0f}")

    # OI structure per pick
    print(f"\n  Open Interest Structure:")
    for p in picks:
        oi = p.get("oi_analysis")
        if oi and oi.spot_price > 0:
            parts = [f"  {p['ticker']:>6}:"]
            if oi.nearest_put_wall:
                parts.append(f"put wall ${oi.nearest_put_wall:.0f} ({oi.put_wall_oi:,} OI)")
            if oi.nearest_call_wall:
                parts.append(f"call wall ${oi.nearest_call_wall:.0f} ({oi.call_wall_oi:,} OI)")
            if oi.max_pain_strike:
                parts.append(f"max pain ${oi.max_pain_strike:.0f}")
            print("    " + "  |  ".join(parts))

    print(f"\n  Exit plan (greeks-only recommended):")
    print(f"    Gamma breach (>0.10): close immediately")
    print(f"    Delta breach (>0.50): close immediately")
    print(f"    Otherwise: hold to expiry, let theta work")

    # Warnings
    warnings = []
    if market.premium_selling_score < 45:
        warnings.append(f"Market regime cautious (score {market.premium_selling_score:.0f}) — consider half size")
    if market.avg_correlation > 0.6:
        warnings.append(f"High cross-correlation ({market.avg_correlation:.2f}) — all positions may move together")
    for p in picks:
        fund = fundamentals.get(p["ticker"])
        if fund and fund.earnings_this_week:
            warnings.append(f"{p['ticker']} has earnings this week — elevated risk")

    if warnings:
        print(f"\n  Warnings:")
        for w in warnings:
            print(f"    - {w}")

    print(f"{'='*80}")


def _build_strategy(name: str, delta: float | None = None, spread_width: float | None = None):
    cls = STRATEGIES[name]
    kwargs = {}
    if delta is not None:
        if hasattr(cls, "call_delta") and "call" in name:
            kwargs["call_delta"] = delta
        elif hasattr(cls, "put_delta") and "put" in name:
            kwargs["put_delta"] = delta
        elif hasattr(cls, "short_delta"):
            kwargs["short_delta"] = delta
        if name == "iron_condor":
            kwargs["short_put_delta"] = delta
            kwargs["short_call_delta"] = delta
        elif name == "short_strangle":
            kwargs["call_delta"] = delta
            kwargs["put_delta"] = delta
    if spread_width is not None and hasattr(cls, "spread_width"):
        kwargs["spread_width"] = spread_width
    if spread_width is not None and hasattr(cls, "wing_width"):
        kwargs["wing_width"] = spread_width
    return cls(**kwargs)


def _build_exit_rules(args):
    from .engine.exit_rules import (
        ExitRules, HOLD_TO_EXPIRY, CONSERVATIVE, THETA_HARVEST, AGGRESSIVE,
    )
    GREEKS_ONLY = ExitRules(
        use_profit_target=False,
        use_stop_loss=False,
        use_profit_floor=False,
        use_time_stop=False,
        use_greeks_stops=True,
        max_position_delta=0.50,
        max_gamma_risk=0.10,
    )
    preset_map = {
        "hold": None,  # None = hold to expiry (legacy behavior)
        "greeks": GREEKS_ONLY,
        "conservative": CONSERVATIVE,
        "theta": THETA_HARVEST,
        "aggressive": AGGRESSIVE,
    }
    rules = preset_map.get(getattr(args, "exit_rules", "hold"))

    # Apply overrides
    if any(getattr(args, f, None) is not None for f in ["profit_target", "stop_loss", "time_stop"]):
        if rules is None:
            rules = ExitRules()  # start from defaults
        if getattr(args, "profit_target", None) is not None:
            rules.profit_target_pct = args.profit_target
            rules.use_profit_target = True
        if getattr(args, "stop_loss", None) is not None:
            rules.stop_loss_pct = args.stop_loss
            rules.use_stop_loss = True
        if getattr(args, "time_stop", None) is not None:
            rules.time_stop_dte = args.time_stop
            rules.use_time_stop = True

    return rules


def _run_backtest(args):
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    tickers = args.tickers or TICKERS

    config = BacktestConfig(
        tickers=tickers,
        start_date=start,
        end_date=end,
        initial_capital=args.capital,
    )

    strategy = _build_strategy(args.strategy, args.delta, args.spread_width)
    exit_rules = _build_exit_rules(args)
    engine = BacktestEngine(config=config)
    engine.run(strategy, exit_rules=exit_rules)

    trades_df = engine.trades_df()
    if trades_df.empty:
        print("No trades generated.")
        return

    metrics = compute_metrics(trades_df, engine.equity_curve, args.capital)
    print_report(metrics, strategy.name)

    # Print exit reason breakdown
    if "exit_reason" in trades_df.columns:
        reason_counts = trades_df["exit_reason"].value_counts()
        if len(reason_counts) > 1 or reason_counts.index[0] != "expiry":
            print(f"\n  Exit Reasons:")
            for reason, count in reason_counts.items():
                pct = count / len(trades_df) * 100
                avg_pnl = trades_df[trades_df["exit_reason"] == reason]["pnl"].mean()
                avg_days = trades_df[trades_df["exit_reason"] == reason]["days_held"].mean()
                print(f"    {reason:>20s}: {count:>4d} ({pct:>5.1f}%)  avg P&L: ${avg_pnl:>8,.0f}  avg days: {avg_days:.1f}")

    if not args.no_plot:
        import matplotlib.pyplot as plt
        plot_dashboard(trades_df, engine.equity_curve, args.capital, strategy.name, args.save_plot)
        if not args.save_plot:
            plt.show()


def _run_validate(args):
    option_type = OptionType.CALL if args.type == "call" else OptionType.PUT
    validator = LiveValidator()

    for ticker in args.tickers:
        try:
            results = validator.validate_strategy_pricing(ticker, args.delta, option_type)
            validator.print_validation(results)
        except Exception as e:
            print(f"Validation failed for {ticker}: {e}")


def _run_theta_validate(args):
    """Run BS backtest, then replay trades against ThetaData historical prices."""
    from .validation.thetadata_client import ThetaDataClient
    from .validation.historical_validate import HistoricalValidator

    # Step 1: Run the BS backtest
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    tickers = args.tickers or TICKERS

    config = BacktestConfig(
        tickers=tickers,
        start_date=start,
        end_date=end,
        initial_capital=args.capital,
    )

    strategy = _build_strategy(args.strategy, args.delta, args.spread_width)
    engine = BacktestEngine(config=config)
    engine.run(strategy)

    trades_df = engine.trades_df()
    if trades_df.empty:
        print("No trades generated.")
        return

    metrics = compute_metrics(trades_df, engine.equity_curve, args.capital)
    print_report(metrics, strategy.name)

    # Step 2: Validate against ThetaData
    print("\n" + "=" * 60)
    print("  Now validating against ThetaData historical prices...")
    print("=" * 60)

    client = ThetaDataClient(rate_limit=not args.no_rate_limit)
    validator = HistoricalValidator(client=client)

    comparisons = validator.validate_trades(
        engine.trades,
        max_trades=args.max_trades,
        verbose=True,
    )

    if comparisons:
        validator.print_summary(comparisons)
    else:
        print("No trades could be validated. Check that Theta Terminal is running.")


def _run_lumibot(args):
    """Run full production backtest via Lumibot + ThetaData."""
    from .validation.lumibot_runner import LumibotRunner

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    runner = LumibotRunner()
    result = runner.run(
        strategy_name=args.strategy,
        symbol=args.symbol,
        start=start,
        end=end,
        initial_capital=args.capital,
    )

    if result:
        print("\nLumibot backtest complete. Results saved by Lumibot.")


def _run_compare(args):
    """Backtest, then validate top performers against live data."""
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    tickers = args.tickers or TICKERS

    config = BacktestConfig(
        tickers=tickers,
        start_date=start,
        end_date=end,
        initial_capital=args.capital,
    )

    strategy = _build_strategy(args.strategy)
    engine = BacktestEngine(config=config)
    engine.run(strategy)

    trades_df = engine.trades_df()
    if trades_df.empty:
        print("No trades generated.")
        return

    metrics = compute_metrics(trades_df, engine.equity_curve, args.capital)
    print_report(metrics, strategy.name)

    # Find top N tickers
    ticker_pnl = trades_df.groupby("ticker")["pnl"].sum().sort_values(ascending=False)
    top_tickers = ticker_pnl.head(args.top_n).index.tolist()

    print(f"\nValidating top {args.top_n} tickers against live chains: {top_tickers}")
    print("(Comparing synthetic BS prices vs real market bid/ask)\n")

    validator = LiveValidator()
    for ticker in top_tickers:
        try:
            results = validator.validate_strategy_pricing(ticker, 0.30, OptionType.CALL)
            validator.print_validation(results)
        except Exception as e:
            print(f"  Skipping {ticker}: {e}")


if __name__ == "__main__":
    main()
