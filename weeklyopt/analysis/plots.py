"""Visualization for backtest results."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


def plot_equity_curve(
    equity_curve: pd.Series,
    initial_capital: float = 100_000.0,
    title: str = "Equity Curve",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot portfolio equity over time."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(equity_curve.index, equity_curve.values, linewidth=1.5, color="#2196F3")
    ax.axhline(initial_capital, color="gray", linestyle="--", alpha=0.5, label="Initial Capital")

    # Drawdown shading
    peak = equity_curve.expanding().max()
    ax.fill_between(equity_curve.index, equity_curve, peak, alpha=0.15, color="red", label="Drawdown")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Portfolio Value ($)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_pnl_distribution(
    trades_df: pd.DataFrame,
    title: str = "P&L Distribution",
    save_path: str | None = None,
) -> plt.Figure:
    """Histogram of trade P&L."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # P&L histogram
    ax = axes[0]
    pnls = trades_df["pnl"]
    colors = ["#4CAF50" if x > 0 else "#F44336" for x in pnls]
    ax.hist(pnls, bins=50, color="#2196F3", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="black", linewidth=1)
    ax.axvline(pnls.mean(), color="orange", linestyle="--", label=f"Mean: ${pnls.mean():.0f}")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("P&L ($)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Return on risk histogram
    ax = axes[1]
    ror = trades_df["return_on_risk"]
    ax.hist(ror, bins=50, color="#9C27B0", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="black", linewidth=1)
    ax.axvline(ror.mean(), color="orange", linestyle="--", label=f"Mean: {ror.mean():.1%}")
    ax.set_title("Return on Risk Distribution", fontweight="bold")
    ax.set_xlabel("Return on Risk")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_by_ticker(
    trades_df: pd.DataFrame,
    title: str = "P&L by Ticker",
    save_path: str | None = None,
) -> plt.Figure:
    """Bar chart of cumulative P&L per ticker."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ticker_pnl = trades_df.groupby("ticker")["pnl"].sum().sort_values(ascending=True)
    colors = ["#4CAF50" if x > 0 else "#F44336" for x in ticker_pnl.values]

    ax = axes[0]
    ax.barh(ticker_pnl.index, ticker_pnl.values, color=colors)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Cumulative P&L ($)")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="x")

    # Win rate by ticker
    ax = axes[1]
    ticker_wr = trades_df.groupby("ticker")["pnl"].apply(lambda x: (x > 0).mean()).sort_values(ascending=True)
    colors = ["#4CAF50" if x > 0.5 else "#FF9800" if x > 0.4 else "#F44336" for x in ticker_wr.values]
    ax.barh(ticker_wr.index, ticker_wr.values, color=colors)
    ax.set_title("Win Rate by Ticker", fontweight="bold")
    ax.set_xlabel("Win Rate")
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_dashboard(
    trades_df: pd.DataFrame,
    equity_curve: pd.Series,
    initial_capital: float = 100_000.0,
    strategy_name: str = "",
    save_path: str | None = None,
) -> plt.Figure:
    """Full dashboard: equity curve, P&L dist, by-ticker, cumulative by month."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f"Backtest Dashboard: {strategy_name}", fontsize=16, fontweight="bold")

    # Equity curve
    ax = axes[0, 0]
    ax.plot(equity_curve.index, equity_curve.values, linewidth=1.5, color="#2196F3")
    ax.axhline(initial_capital, color="gray", linestyle="--", alpha=0.5)
    peak = equity_curve.expanding().max()
    ax.fill_between(equity_curve.index, equity_curve, peak, alpha=0.15, color="red")
    ax.set_title("Equity Curve", fontweight="bold")
    ax.set_ylabel("Portfolio Value ($)")
    ax.grid(True, alpha=0.3)

    # P&L distribution
    ax = axes[0, 1]
    pnls = trades_df["pnl"]
    ax.hist(pnls, bins=50, color="#2196F3", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="black", linewidth=1)
    ax.axvline(pnls.mean(), color="orange", linestyle="--", label=f"Mean: ${pnls.mean():.0f}")
    ax.set_title("P&L Distribution", fontweight="bold")
    ax.set_xlabel("P&L ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # By ticker
    ax = axes[1, 0]
    ticker_pnl = trades_df.groupby("ticker")["pnl"].sum().sort_values(ascending=True)
    colors = ["#4CAF50" if x > 0 else "#F44336" for x in ticker_pnl.values]
    ax.barh(ticker_pnl.index, ticker_pnl.values, color=colors)
    ax.set_title("Cumulative P&L by Ticker", fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="x")

    # Monthly P&L
    ax = axes[1, 1]
    trades_df = trades_df.copy()
    trades_df["month"] = trades_df["exit_date"].dt.to_period("M")
    monthly = trades_df.groupby("month")["pnl"].sum()
    colors = ["#4CAF50" if x > 0 else "#F44336" for x in monthly.values]
    ax.bar(range(len(monthly)), monthly.values, color=colors, alpha=0.8)
    # Show every 6th month label
    tick_positions = range(0, len(monthly), 6)
    ax.set_xticks(list(tick_positions))
    ax.set_xticklabels([str(monthly.index[i]) for i in tick_positions], rotation=45)
    ax.set_title("Monthly P&L", fontweight="bold")
    ax.set_ylabel("P&L ($)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
