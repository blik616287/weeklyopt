"""Train ML exit model v2: proper train/test split with out-of-sample evaluation.

Train on 2023-2024, test on 2025. Compare actual P&L outcomes.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

from weeklyopt.config import BacktestConfig, TICKERS_EXPANDED
from weeklyopt.data.fetcher import fetch_all_equities
from weeklyopt.pricing.volatility import historical_vol, implied_vol_estimate, vol_on_date, calibrated_vol_estimate
from weeklyopt.pricing.calibration import IVCalibrator
from weeklyopt.engine.portfolio_backtest import PortfolioBacktest, PortfolioConfig
from weeklyopt.engine.ml_exit import MLExitModel, ExitCheckpoint


def generate_checkpoints(trades_df, all_data, vix_data, calibrations):
    """Generate training checkpoints from trades."""
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

        hv_series = historical_vol(df["Close"], 20)
        entry_price = float(df.loc[entry_dt, "Close"])

        for day_idx in range(1, len(week_dates)):
            dt = week_dates[day_idx]
            days_left = len(week_dates) - day_idx - 1
            current_price = float(df.loc[dt, "Close"])

            progress = day_idx / max(len(week_dates) - 1, 1)
            theta_curve = 1 - (1 - progress) ** 1.5
            current_pnl_est = final_pnl * theta_curve if final_pnl > 0 else final_pnl * progress

            unrealized_pnl_pct = current_pnl_est / max_risk if max_risk > 0 else 0
            unrealized_loss_pct = max(0, -current_pnl_est / max_risk)

            ret_1d = float(df.loc[dt, "Close"] / df.loc[week_dates[max(0, day_idx-1)], "Close"] - 1)
            ret_3d = float(df.loc[dt, "Close"] / df.loc[week_dates[max(0, day_idx-3)], "Close"] - 1) if day_idx >= 3 else ret_1d

            vix_level, vix_change = 20.0, 0.0
            if vix_data is not None and not vix_data.empty:
                vh = vix_data.loc[:dt, "Close"]
                if len(vh) >= 2:
                    vix_level = float(vh.iloc[-1])
                    vix_change = float(vh.iloc[-1] / vh.iloc[-2] - 1)

            hv = vol_on_date(hv_series, dt)
            hv_1yr = hv_series.loc[:dt].tail(252)
            iv_rank = ((hv - hv_1yr.min()) / (hv_1yr.max() - hv_1yr.min()) * 100
                      if len(hv_1yr) > 20 and hv_1yr.max() != hv_1yr.min() else 50.0)

            position_delta = (current_price - entry_price) / entry_price * 0.3
            position_gamma = 0.05 / max(days_left, 0.5)

            remaining_pnl = final_pnl - current_pnl_est
            if current_pnl_est > 0 and remaining_pnl < -current_pnl_est * 0.3:
                should_exit = True
            elif current_pnl_est < 0 and remaining_pnl < 0:
                should_exit = True
            elif remaining_pnl < 0:
                should_exit = True
            else:
                should_exit = False

            checkpoints.append(ExitCheckpoint(
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
            ))

    return checkpoints


def checkpoints_to_arrays(cps):
    X = np.array([[
        cp.day_of_week, cp.days_remaining, cp.unrealized_pnl_pct,
        cp.unrealized_loss_pct, cp.position_delta, cp.position_gamma,
        cp.underlying_1d_return, cp.underlying_3d_return,
        cp.vix_level, cp.vix_1d_change, cp.iv_rank,
    ] for cp in cps])
    y = np.array([cp.should_exit for cp in cps]).astype(int)
    return X, y


def main():
    print("=" * 70)
    print("  ML Exit Model v2: Train/Test Split")
    print("=" * 70)

    calibrations = IVCalibrator.load_calibration() or {}

    # Fetch all data
    print("\nFetching data...")
    all_data = fetch_all_equities(TICKERS_EXPANDED, date(2023, 1, 1), date(2025, 12, 31))

    vix_data = yf.download("^VIX", start="2022-06-01", end="2025-12-31", progress=False)
    if isinstance(vix_data.columns, pd.MultiIndex):
        vix_data.columns = vix_data.columns.get_level_values(0)

    # ── Train set: 2023-2024 ──
    print("\n--- TRAINING SET: 2023-2024 ---")
    train_cfg = BacktestConfig(tickers=TICKERS_EXPANDED, start_date=date(2023, 1, 1),
                                end_date=date(2024, 12, 31), initial_capital=10_000)
    train_pcfg = PortfolioConfig(weekly_budget=500, max_contracts_per_week=5,
                                  min_score=50, credit_spreads_only=True)
    train_engine = PortfolioBacktest(config=train_cfg, portfolio_config=train_pcfg, exit_rules=None)
    train_engine.run(verbose=False)
    train_trades = train_engine.trades_df()
    print(f"  {len(train_trades)} training trades")

    train_cps = generate_checkpoints(train_trades, all_data, vix_data, calibrations)
    print(f"  {len(train_cps)} training checkpoints, exit rate: {sum(c.should_exit for c in train_cps)/len(train_cps):.1%}")

    # ── Test set: 2025 ──
    print("\n--- TEST SET: 2025 ---")
    test_cfg = BacktestConfig(tickers=TICKERS_EXPANDED, start_date=date(2025, 1, 1),
                               end_date=date(2025, 12, 31), initial_capital=10_000)
    test_pcfg = PortfolioConfig(weekly_budget=500, max_contracts_per_week=5,
                                 min_score=50, credit_spreads_only=True)
    test_engine = PortfolioBacktest(config=test_cfg, portfolio_config=test_pcfg, exit_rules=None)
    test_engine.run(verbose=False)
    test_trades = test_engine.trades_df()
    print(f"  {len(test_trades)} test trades")

    test_cps = generate_checkpoints(test_trades, all_data, vix_data, calibrations)
    print(f"  {len(test_cps)} test checkpoints, exit rate: {sum(c.should_exit for c in test_cps)/len(test_cps):.1%}")

    # ── Train model on 2023-2024 only ──
    print("\n--- TRAINING ---")
    X_train, y_train = checkpoints_to_arrays(train_cps)
    X_test, y_test = checkpoints_to_arrays(test_cps)

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        min_samples_leaf=20,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # ── Evaluate on 2025 ──
    print("\n--- OUT-OF-SAMPLE RESULTS (2025) ---")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    feature_names = [
        "day_of_week", "days_remaining", "unrealized_pnl_pct",
        "unrealized_loss_pct", "position_delta", "position_gamma",
        "underlying_1d_return", "underlying_3d_return",
        "vix_level", "vix_1d_change", "iv_rank",
    ]

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Hold", "Exit"]))

    importances = list(zip(feature_names, model.feature_importances_))
    importances.sort(key=lambda x: -x[1])
    print("  Feature Importances:")
    for name, imp in importances:
        bar = "#" * int(imp * 50)
        print(f"    {name:>25s}: {imp:.3f}  {bar}")

    # Compare rule-based vs ML
    print("\n  --- Decision Comparison (2025) ---")
    ml_model = MLExitModel()
    rule_correct = sum(1 for cp in test_cps if ml_model._rule_based_exit(cp)[0] == cp.should_exit)
    ml_correct = sum(1 for cp, pred in zip(test_cps, y_pred) if bool(pred) == cp.should_exit)

    print(f"  Rule-based accuracy: {rule_correct/len(test_cps):.1%}")
    print(f"  ML accuracy:         {ml_correct/len(test_cps):.1%}")
    print(f"  Improvement:         {(ml_correct - rule_correct)/len(test_cps):+.1%}")

    # Save the model trained on full data
    print("\n--- RETRAINING ON FULL DATA (2023-2025) ---")
    all_cps = train_cps + test_cps
    X_all, y_all = checkpoints_to_arrays(all_cps)
    model.fit(X_all, y_all)

    ml_model.model = model
    ml_model.is_trained = True
    path = ml_model.save()
    print(f"  Model saved to: {path}")
    print(f"  Total training samples: {len(all_cps)}")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
