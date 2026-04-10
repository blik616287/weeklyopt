"""Parameter optimization for the credit-spread portfolio.

Sweeps across spread width, delta, min score, max contracts,
and exit rule presets to find the best configuration.
"""

import itertools
from datetime import date
from dataclasses import dataclass

import pandas as pd
import numpy as np

from weeklyopt.config import BacktestConfig, TICKERS
from weeklyopt.engine.portfolio_backtest import PortfolioBacktest, PortfolioConfig
from weeklyopt.engine.exit_rules import ExitRules, THETA_HARVEST, CONSERVATIVE, AGGRESSIVE
from weeklyopt.strategies.vertical_spread import BullPutSpread, BearCallSpread
from weeklyopt.strategies.iron_condor import IronCondor


@dataclass
class RunResult:
    params: dict
    total_pnl: float
    win_rate: float
    profit_factor: float
    sharpe: float
    max_drawdown_pct: float
    avg_pnl: float
    total_trades: int
    max_loss: float


def run_one(
    spread_width: float,
    delta: float,
    min_score: float,
    max_contracts: int,
    exit_preset: str,
    budget: float = 500.0,
) -> RunResult:
    """Run a single parameter combo and return metrics."""

    # Patch the default strategy params by modifying class defaults
    BullPutSpread.spread_width = spread_width
    BullPutSpread.short_delta = delta
    BearCallSpread.spread_width = spread_width
    BearCallSpread.short_delta = delta
    IronCondor.wing_width = spread_width
    IronCondor.short_put_delta = delta
    IronCondor.short_call_delta = delta

    exit_map = {
        "theta": THETA_HARVEST,
        "conservative": CONSERVATIVE,
        "aggressive": AGGRESSIVE,
    }

    bt_config = BacktestConfig(
        tickers=TICKERS,
        start_date=date(2023, 1, 1),
        end_date=date(2025, 12, 31),
        initial_capital=10_000,
    )

    port_config = PortfolioConfig(
        weekly_budget=budget,
        max_contracts_per_week=max_contracts,
        min_score=min_score,
        credit_spreads_only=True,
    )

    engine = PortfolioBacktest(
        config=bt_config,
        portfolio_config=port_config,
        exit_rules=exit_map.get(exit_preset),
    )

    engine.run(verbose=False)
    df = engine.trades_df()

    if df.empty:
        return RunResult(
            params={}, total_pnl=0, win_rate=0, profit_factor=0,
            sharpe=0, max_drawdown_pct=0, avg_pnl=0, total_trades=0, max_loss=0,
        )

    pnls = df["pnl"]
    winners = pnls[pnls > 0]
    losers = pnls[pnls <= 0]
    gross_win = winners.sum() if len(winners) > 0 else 0
    gross_loss = abs(losers.sum()) if len(losers) > 0 else 1e-9

    # Sharpe from trade returns
    ror = df["pnl"] / df["max_risk"].replace(0, np.nan)
    ror = ror.dropna()
    sharpe = (ror.mean() / ror.std()) * np.sqrt(52) if len(ror) > 1 and ror.std() > 0 else 0

    # Max drawdown
    eq = engine.equity_curve
    max_dd_pct = 0.0
    if len(eq) > 1:
        peak = eq.expanding().max()
        dd_pct = (eq - peak) / peak
        max_dd_pct = float(dd_pct.min())

    params = {
        "width": spread_width,
        "delta": delta,
        "min_score": min_score,
        "max_contracts": max_contracts,
        "exit": exit_preset,
    }

    return RunResult(
        params=params,
        total_pnl=float(pnls.sum()),
        win_rate=len(winners) / len(pnls),
        profit_factor=gross_win / gross_loss,
        sharpe=sharpe,
        max_drawdown_pct=max_dd_pct,
        avg_pnl=float(pnls.mean()),
        total_trades=len(pnls),
        max_loss=float(pnls.min()),
    )


def sweep():
    """Run parameter sweep and print results."""

    # Parameter grid
    spread_widths = [3.0, 5.0, 7.0, 10.0]
    deltas = [0.20, 0.25, 0.30, 0.35]
    min_scores = [40, 50, 60]
    max_contracts_list = [5, 10]
    exit_presets = ["theta", "conservative", "aggressive"]

    combos = list(itertools.product(
        spread_widths, deltas, min_scores, max_contracts_list, exit_presets,
    ))

    print(f"Running {len(combos)} parameter combinations...\n")

    results = []
    for i, (width, delta, min_score, max_contracts, exit_preset) in enumerate(combos):
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(combos)}]...")

        r = run_one(width, delta, min_score, max_contracts, exit_preset)
        results.append(r)

    # Sort by composite score: Sharpe * PF (balanced risk-adjusted + consistency)
    results.sort(key=lambda r: r.sharpe * r.profit_factor, reverse=True)

    # Print top 20
    print(f"\n{'='*115}")
    print(f"  Top 20 Parameter Combinations (sorted by Sharpe x PF)")
    print(f"{'='*115}")
    print(
        f"  {'#':>3}  {'Width':>5}  {'Delta':>5}  {'MinScr':>6}  {'MaxK':>4}  {'Exit':>12}  "
        f"{'Trades':>6}  {'Win%':>5}  {'P&L':>10}  {'Avg':>7}  {'PF':>5}  "
        f"{'Sharpe':>6}  {'MaxDD':>7}  {'MaxLoss':>8}  {'Score':>6}"
    )
    print(f"  {'-'*110}")

    for i, r in enumerate(results[:20]):
        p = r.params
        composite = r.sharpe * r.profit_factor
        print(
            f"  {i+1:>3}  {p['width']:>5.0f}  {p['delta']:>5.2f}  {p['min_score']:>6.0f}  "
            f"{p['max_contracts']:>4}  {p['exit']:>12}  "
            f"{r.total_trades:>6}  {r.win_rate:>4.0%}  ${r.total_pnl:>9,.0f}  "
            f"${r.avg_pnl:>6,.0f}  {r.profit_factor:>5.2f}  "
            f"{r.sharpe:>6.2f}  {r.max_drawdown_pct:>6.1%}  ${r.max_loss:>7,.0f}  "
            f"{composite:>6.2f}"
        )

    # Also show worst 5
    print(f"\n  Bottom 5:")
    print(f"  {'-'*110}")
    for r in results[-5:]:
        p = r.params
        composite = r.sharpe * r.profit_factor
        print(
            f"       {p['width']:>5.0f}  {p['delta']:>5.2f}  {p['min_score']:>6.0f}  "
            f"{p['max_contracts']:>4}  {p['exit']:>12}  "
            f"{r.total_trades:>6}  {r.win_rate:>4.0%}  ${r.total_pnl:>9,.0f}  "
            f"${r.avg_pnl:>6,.0f}  {r.profit_factor:>5.2f}  "
            f"{r.sharpe:>6.2f}  {r.max_drawdown_pct:>6.1%}  ${r.max_loss:>7,.0f}  "
            f"{composite:>6.2f}"
        )

    print(f"\n{'='*115}")

    # Print the winner
    best = results[0]
    p = best.params
    print(f"\n  OPTIMAL CONFIGURATION:")
    print(f"    Spread width:    ${p['width']:.0f}")
    print(f"    Delta:           {p['delta']:.2f}")
    print(f"    Min score:       {p['min_score']:.0f}")
    print(f"    Max contracts:   {p['max_contracts']}")
    print(f"    Exit rules:      {p['exit']}")
    print(f"")
    print(f"    Expected: {best.win_rate:.0%} win rate, ${best.avg_pnl:.0f}/trade, "
          f"PF {best.profit_factor:.2f}, Sharpe {best.sharpe:.2f}")
    print(f"    3yr P&L: ${best.total_pnl:+,.0f} on {best.total_trades} trades")
    print(f"")
    print(f"  Run with:")
    print(f"    weeklyopt portfolio --budget 500 --max-contracts {p['max_contracts']} "
          f"--exit-rules {p['exit']} --credit-spreads "
          f"--min-score {p['min_score']:.0f}")


if __name__ == "__main__":
    sweep()
