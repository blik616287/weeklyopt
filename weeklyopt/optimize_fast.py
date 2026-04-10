"""Fast parameter optimization — reduced grid, credit spreads only."""

import itertools
from datetime import date
from dataclasses import dataclass

import numpy as np

from weeklyopt.config import BacktestConfig, TICKERS
from weeklyopt.engine.portfolio_backtest import PortfolioBacktest, PortfolioConfig
from weeklyopt.engine.exit_rules import THETA_HARVEST, CONSERVATIVE, AGGRESSIVE
from weeklyopt.strategies.vertical_spread import BullPutSpread, BearCallSpread
from weeklyopt.strategies.iron_condor import IronCondor


def run_one(width, delta, min_score, max_contracts, exit_preset, budget=500.0):
    BullPutSpread.spread_width = width
    BullPutSpread.short_delta = delta
    BearCallSpread.spread_width = width
    BearCallSpread.short_delta = delta
    IronCondor.wing_width = width
    IronCondor.short_put_delta = delta
    IronCondor.short_call_delta = delta

    exit_map = {"theta": THETA_HARVEST, "conservative": CONSERVATIVE, "aggressive": AGGRESSIVE}

    engine = PortfolioBacktest(
        config=BacktestConfig(tickers=TICKERS, start_date=date(2023, 1, 1),
                              end_date=date(2025, 12, 31), initial_capital=10_000),
        portfolio_config=PortfolioConfig(weekly_budget=budget, max_contracts_per_week=max_contracts,
                                         min_score=min_score, credit_spreads_only=True),
        exit_rules=exit_map.get(exit_preset),
    )
    engine.run(verbose=False)
    df = engine.trades_df()

    if df.empty:
        return None

    pnls = df["pnl"]
    w = pnls[pnls > 0]
    l = pnls[pnls <= 0]
    gw = w.sum() if len(w) else 0
    gl = abs(l.sum()) if len(l) else 1e-9
    ror = pnls / df["max_risk"].replace(0, np.nan)
    ror = ror.dropna()
    sharpe = (ror.mean() / ror.std()) * np.sqrt(52) if len(ror) > 1 and ror.std() > 0 else 0

    eq = engine.equity_curve
    max_dd = 0.0
    if len(eq) > 1:
        peak = eq.expanding().max()
        max_dd = float(((eq - peak) / peak).min())

    return {
        "width": width, "delta": delta, "min_score": min_score,
        "max_k": max_contracts, "exit": exit_preset,
        "trades": len(pnls), "win%": len(w)/len(pnls),
        "pnl": float(pnls.sum()), "avg": float(pnls.mean()),
        "pf": gw/gl, "sharpe": sharpe, "max_dd": max_dd,
        "max_loss": float(pnls.min()),
        "score": sharpe * (gw/gl),  # composite
    }


if __name__ == "__main__":
    # Focused grid: skip obvious losers
    grid = list(itertools.product(
        [3, 5, 7, 10],           # spread width
        [0.20, 0.25, 0.30],     # delta
        [40, 50, 60],           # min score
        [5, 10],                # max contracts
        ["theta", "conservative"],  # exit (skip aggressive — already tested)
    ))

    print(f"Running {len(grid)} combinations...\n")
    results = []
    for i, (w, d, ms, mk, ex) in enumerate(grid):
        if (i+1) % 12 == 0:
            print(f"  [{i+1}/{len(grid)}]...")
        r = run_one(w, d, ms, mk, ex)
        if r:
            results.append(r)

    results.sort(key=lambda r: r["score"], reverse=True)

    print(f"\n{'='*120}")
    print(f"  Top 15 Configurations (Sharpe x PF)")
    print(f"{'='*120}")
    print(f"  {'#':>3}  {'W':>3}  {'Delta':>5}  {'MinS':>4}  {'MaxK':>4}  {'Exit':>12}  "
          f"{'Trades':>6}  {'Win%':>5}  {'P&L':>10}  {'Avg':>7}  {'PF':>5}  "
          f"{'Sharpe':>6}  {'MaxDD':>7}  {'MaxLoss':>8}  {'Score':>6}")
    print(f"  {'-'*115}")

    for i, r in enumerate(results[:15]):
        print(f"  {i+1:>3}  {r['width']:>3.0f}  {r['delta']:>5.2f}  {r['min_score']:>4.0f}  "
              f"{r['max_k']:>4}  {r['exit']:>12}  "
              f"{r['trades']:>6}  {r['win%']:>4.0%}  ${r['pnl']:>9,.0f}  "
              f"${r['avg']:>6,.0f}  {r['pf']:>5.2f}  "
              f"{r['sharpe']:>6.2f}  {r['max_dd']:>6.1%}  ${r['max_loss']:>7,.0f}  "
              f"{r['score']:>6.2f}")

    print(f"\n{'='*120}")
    best = results[0]
    print(f"\n  OPTIMAL: ${'width'} ${best['width']:.0f} wide, delta {best['delta']:.2f}, "
          f"min score {best['min_score']:.0f}, max {best['max_k']} contracts, {best['exit']} exit")
    print(f"  → {best['win%']:.0%} win rate, PF {best['pf']:.2f}, Sharpe {best['sharpe']:.2f}, "
          f"P&L ${best['pnl']:+,.0f}")
    print(f"\n  weeklyopt portfolio --budget 500 --max-contracts {best['max_k']} "
          f"--exit-rules {best['exit']} --credit-spreads --min-score {best['min_score']:.0f}")
