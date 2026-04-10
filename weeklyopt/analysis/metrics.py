"""Performance metrics computation."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_pnl_per_trade: float
    median_pnl_per_trade: float
    max_win: float
    max_loss: float
    avg_winner: float
    avg_loser: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    avg_return_on_risk: float
    total_premium_collected: float
    total_premium_paid: float
    best_ticker: str
    worst_ticker: str


def compute_metrics(
    trades_df: pd.DataFrame,
    equity_curve: pd.Series | None = None,
    initial_capital: float = 100_000.0,
) -> PerformanceMetrics:
    """Compute comprehensive performance metrics from trade results."""
    if trades_df.empty:
        raise ValueError("No trades to analyze")

    pnls = trades_df["pnl"]
    winners = pnls[pnls > 0]
    losers = pnls[pnls <= 0]

    total_pnl = pnls.sum()
    gross_profit = winners.sum() if len(winners) > 0 else 0
    gross_loss = abs(losers.sum()) if len(losers) > 0 else 1e-9

    # Sharpe / Sortino from weekly returns
    weekly_returns = pnls / trades_df["max_risk"].replace(0, np.nan)
    weekly_returns = weekly_returns.dropna()

    sharpe = 0.0
    sortino = 0.0
    if len(weekly_returns) > 1:
        ann_factor = np.sqrt(52)  # weekly -> annualized
        sharpe = (weekly_returns.mean() / weekly_returns.std()) * ann_factor if weekly_returns.std() > 0 else 0
        downside = weekly_returns[weekly_returns < 0].std()
        sortino = (weekly_returns.mean() / downside) * ann_factor if downside > 0 else 0

    # Max drawdown from equity curve
    max_dd = 0.0
    max_dd_pct = 0.0
    if equity_curve is not None and len(equity_curve) > 1:
        peak = equity_curve.expanding().max()
        drawdown = equity_curve - peak
        max_dd = drawdown.min()
        dd_pct = drawdown / peak
        max_dd_pct = dd_pct.min()

    # By ticker
    ticker_pnl = trades_df.groupby("ticker")["pnl"].sum()
    best_ticker = ticker_pnl.idxmax() if not ticker_pnl.empty else "N/A"
    worst_ticker = ticker_pnl.idxmin() if not ticker_pnl.empty else "N/A"

    return PerformanceMetrics(
        total_trades=len(pnls),
        winning_trades=len(winners),
        losing_trades=len(losers),
        win_rate=len(winners) / len(pnls) if len(pnls) > 0 else 0,
        total_pnl=total_pnl,
        avg_pnl_per_trade=pnls.mean(),
        median_pnl_per_trade=pnls.median(),
        max_win=winners.max() if len(winners) > 0 else 0,
        max_loss=losers.min() if len(losers) > 0 else 0,
        avg_winner=winners.mean() if len(winners) > 0 else 0,
        avg_loser=losers.mean() if len(losers) > 0 else 0,
        profit_factor=gross_profit / gross_loss,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        avg_return_on_risk=trades_df["return_on_risk"].mean(),
        total_premium_collected=trades_df["premium_collected"].sum(),
        total_premium_paid=trades_df["premium_paid"].sum(),
        best_ticker=best_ticker,
        worst_ticker=worst_ticker,
    )


def print_report(metrics: PerformanceMetrics, strategy_name: str = "", initial_capital: float = 0) -> None:
    """Print a formatted performance report."""
    header = f"  Performance Report: {strategy_name}  " if strategy_name else "  Performance Report  "
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    if initial_capital > 0:
        ending = initial_capital + metrics.total_pnl
        total_return = metrics.total_pnl / initial_capital
        # Approximate years from trade dates
        years = max(metrics.total_trades / 52, 1)  # rough: ~1 trade/week
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        weekly_avg = metrics.total_pnl / max(metrics.total_trades / 2, 1)  # ~2 trades/week avg

        print(f"  Starting Capital:    ${initial_capital:,.0f}")
        print(f"  Ending Capital:      ${ending:,.0f}")
        print(f"  Total Return:        {total_return:+.1%}")
        print(f"  Annualized Return:   {annual_return:+.1%}")
        print(f"  Avg Weekly P&L:      ${weekly_avg:,.0f}")
        print()

    print(f"  Total Trades:        {metrics.total_trades}")
    print(f"  Win Rate:            {metrics.win_rate:.1%}")
    print(f"  Winners / Losers:    {metrics.winning_trades} / {metrics.losing_trades}")
    print()
    print(f"  Total P&L:           ${metrics.total_pnl:,.2f}")
    print(f"  Avg P&L/Trade:       ${metrics.avg_pnl_per_trade:,.2f}")
    print(f"  Median P&L/Trade:    ${metrics.median_pnl_per_trade:,.2f}")
    print(f"  Max Win:             ${metrics.max_win:,.2f}")
    print(f"  Max Loss:            ${metrics.max_loss:,.2f}")
    print(f"  Avg Winner:          ${metrics.avg_winner:,.2f}")
    print(f"  Avg Loser:           ${metrics.avg_loser:,.2f}")
    print(f"  Profit Factor:       {metrics.profit_factor:.2f}")
    print()
    print(f"  Sharpe Ratio:        {metrics.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio:       {metrics.sortino_ratio:.2f}")
    print(f"  Max Drawdown:        ${metrics.max_drawdown:,.2f} ({metrics.max_drawdown_pct:.1%})")
    print(f"  Avg Return on Risk:  {metrics.avg_return_on_risk:.2%}")
    print()
    net_premium = metrics.total_premium_collected - metrics.total_premium_paid
    print(f"  Net Premium:         ${net_premium:,.2f}")
    if initial_capital > 0:
        print(f"  Net Premium/Capital: {net_premium / initial_capital:.1%}")
    print(f"  Best Ticker:         {metrics.best_ticker}")
    print(f"  Worst Ticker:        {metrics.worst_ticker}")
    print("=" * len(header))
