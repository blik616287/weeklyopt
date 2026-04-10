"""Train the ML exit model on historical backtest data.

Steps:
1. Run a full backtest holding to expiry (no exit rules) to get "ground truth" P&L
2. Simulate daily checkpoints for each trade
3. At each checkpoint, label: would exiting NOW have been better than holding?
4. Train gradient boosted tree on these labeled checkpoints
5. Save model for use in future backtests

Then re-run the portfolio backtest using the ML model for exit decisions.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from weeklyopt.config import BacktestConfig, TICKERS_EXPANDED
from weeklyopt.data.fetcher import fetch_all_equities
from weeklyopt.pricing.black_scholes import bs_price, bs_greeks, OptionType
from weeklyopt.pricing.volatility import historical_vol, implied_vol_estimate, vol_on_date
from weeklyopt.pricing.calibration import IVCalibrator
from weeklyopt.engine.portfolio_backtest import PortfolioBacktest, PortfolioConfig
from weeklyopt.engine.exit_rules import HOLD_TO_EXPIRY
from weeklyopt.engine.ml_exit import (
    MLExitModel, ExitCheckpoint, generate_training_data,
)


def main():
    print("=" * 70)
    print("  Training ML Exit Model")
    print("=" * 70)

    # Step 1: Run baseline backtest (hold to expiry) to get ground truth trades
    print("\n1. Running baseline backtest (hold to expiry)...")

    cfg = BacktestConfig(
        tickers=TICKERS_EXPANDED,
        start_date=date(2023, 1, 1),
        end_date=date(2025, 12, 31),
        initial_capital=10_000,
    )
    pcfg = PortfolioConfig(
        weekly_budget=500,
        max_contracts_per_week=5,
        min_score=50,
        credit_spreads_only=True,
    )

    engine = PortfolioBacktest(config=cfg, portfolio_config=pcfg, exit_rules=None)
    engine.run(verbose=True)

    trades_df = engine.trades_df()
    print(f"   Generated {len(trades_df)} baseline trades")

    # Step 2: Fetch equity + VIX data for checkpoint generation
    print("\n2. Preparing data for checkpoint generation...")
    all_data = fetch_all_equities(TICKERS_EXPANDED, cfg.start_date, cfg.end_date)

    print("   Fetching VIX...")
    try:
        vix_data = yf.download("^VIX",
                               start=str(cfg.start_date - timedelta(days=30)),
                               end=str(cfg.end_date), progress=False)
        if isinstance(vix_data.columns, pd.MultiIndex):
            vix_data.columns = vix_data.columns.get_level_values(0)
    except Exception:
        vix_data = None

    # Step 3: Generate labeled checkpoints
    print("\n3. Generating training checkpoints...")

    # More sophisticated checkpoint generation: simulate actual BS pricing daily
    calibrations = IVCalibrator.load_calibration() or {}
    checkpoints = []

    for _, trade in trades_df.iterrows():
        ticker = trade["ticker"]
        if ticker not in all_data:
            continue

        df = all_data[ticker]
        entry_dt = trade["entry_date"]
        exit_dt = trade["exit_date"]
        final_pnl = trade["pnl"]
        max_risk = trade["max_risk"] if trade["max_risk"] > 0 else 1

        week_dates = df.index[(df.index >= entry_dt) & (df.index <= exit_dt)]
        if len(week_dates) < 3:
            continue

        # Get vol series for this ticker
        cal = calibrations.get(ticker)
        if cal and cal.sample_dates > 0:
            from weeklyopt.pricing.volatility import calibrated_vol_estimate
            vol_series = calibrated_vol_estimate(df["Close"], cal, moneyness=1.0)
        else:
            vol_series = implied_vol_estimate(df["Close"], 20, 1.15)

        hv_series = historical_vol(df["Close"], 20)

        entry_price = float(df.loc[entry_dt, "Close"])
        vol = vol_on_date(vol_series, entry_dt)

        for day_idx in range(1, len(week_dates)):
            dt = week_dates[day_idx]
            days_left = len(week_dates) - day_idx - 1
            dte_years = max(days_left / 252, 0.25 / 252)
            current_price = float(df.loc[dt, "Close"])

            # Estimate current position value via BS
            # Approximate: linear interpolation of P&L (imperfect but usable)
            progress = day_idx / max(len(week_dates) - 1, 1)
            # Theta decay is front-loaded for credit spreads
            theta_curve = 1 - (1 - progress) ** 1.5
            current_pnl_est = final_pnl * theta_curve if final_pnl > 0 else final_pnl * progress

            unrealized_pnl_pct = current_pnl_est / max_risk if max_risk > 0 else 0
            unrealized_loss_pct = max(0, -current_pnl_est / max_risk)

            # Underlying returns
            ret_1d = float(df.loc[dt, "Close"] / df.loc[week_dates[max(0, day_idx-1)], "Close"] - 1)
            ret_3d = float(df.loc[dt, "Close"] / df.loc[week_dates[max(0, day_idx-3)], "Close"] - 1) if day_idx >= 3 else ret_1d

            # VIX
            vix_level = 20.0
            vix_change = 0.0
            if vix_data is not None and not vix_data.empty:
                vix_hist = vix_data.loc[:dt, "Close"]
                if len(vix_hist) >= 2:
                    vix_level = float(vix_hist.iloc[-1])
                    vix_change = float(vix_hist.iloc[-1] / vix_hist.iloc[-2] - 1)

            # IV rank at this date
            hv_at_date = vol_on_date(hv_series, dt)
            hv_1yr = hv_series.loc[:dt].tail(252)
            if len(hv_1yr) > 20 and hv_1yr.max() != hv_1yr.min():
                iv_rank = (hv_at_date - hv_1yr.min()) / (hv_1yr.max() - hv_1yr.min()) * 100
            else:
                iv_rank = 50.0

            # Approximate position greeks (simplified)
            position_delta = (current_price - entry_price) / entry_price * 0.3  # rough proxy
            position_gamma = 0.05 / max(dte_years * 252, 0.5)  # gamma increases as DTE shrinks

            # Label: should we exit here?
            # Compare: P&L if exit now vs P&L if hold to end
            remaining_pnl = final_pnl - current_pnl_est
            # Also factor in risk: if we're sitting on a profit, the risk of holding is giving it back
            if current_pnl_est > 0 and remaining_pnl < -current_pnl_est * 0.3:
                should_exit = True  # holding gives back >30% of current profit
            elif current_pnl_est < 0 and remaining_pnl < 0:
                should_exit = True  # losing and will lose more
            elif remaining_pnl < 0:
                should_exit = True  # holding makes it worse
            else:
                should_exit = False  # holding improves outcome

            cp = ExitCheckpoint(
                day_of_week=dt.weekday(),
                days_remaining=days_left,
                unrealized_pnl_pct=unrealized_pnl_pct,
                unrealized_loss_pct=unrealized_loss_pct,
                position_delta=position_delta,
                position_gamma=position_gamma,
                underlying_1d_return=ret_1d,
                underlying_3d_return=ret_3d,
                vix_level=vix_level,
                vix_1d_change=vix_change,
                iv_rank=iv_rank,
                should_exit=should_exit,
            )
            checkpoints.append(cp)

    print(f"   Generated {len(checkpoints)} checkpoints")
    exit_rate = sum(1 for cp in checkpoints if cp.should_exit) / len(checkpoints) if checkpoints else 0
    print(f"   Exit rate in labels: {exit_rate:.1%}")

    # Step 4: Train model
    print("\n4. Training gradient boosted tree...")
    model = MLExitModel()
    accuracy = model.train(checkpoints, verbose=True)

    if accuracy > 0:
        # Step 5: Save
        path = model.save()
        print(f"\n5. Model saved to: {path}")
        print(f"   Accuracy: {accuracy:.1%}")

        # Step 6: Compare rule-based vs ML on the same data
        print("\n6. Comparing exit decisions: rule-based vs ML...")
        rule_exits = 0
        ml_exits = 0
        agree = 0
        ml_better = 0

        for cp in checkpoints[:500]:  # sample
            rule_exit, _ = model._rule_based_exit(cp)
            ml_exit, conf = model.predict_exit(cp)

            if rule_exit:
                rule_exits += 1
            if ml_exit:
                ml_exits += 1
            if rule_exit == ml_exit:
                agree += 1
            # ML is "better" if it exits when should_exit=True or holds when should_exit=False
            if ml_exit == cp.should_exit:
                ml_better += 1

        n = min(500, len(checkpoints))
        print(f"   Sample: {n} checkpoints")
        print(f"   Rule-based exits: {rule_exits} ({rule_exits/n:.0%})")
        print(f"   ML exits:         {ml_exits} ({ml_exits/n:.0%})")
        print(f"   Agreement:        {agree/n:.0%}")
        print(f"   ML correct:       {ml_better/n:.0%}")

    print(f"\n{'='*70}")
    print("  Done. Run portfolio with ML exits:")
    print("  weeklyopt portfolio --credit-spreads --expanded --exit-rules theta")
    print("  (ML model auto-loaded when available)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
