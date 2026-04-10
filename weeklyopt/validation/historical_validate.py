"""Historical validation: replay backtest trades against real ThetaData option prices.

This is the real mode 1 — compare what our BS engine predicted vs what the market
actually priced for the same strikes/expirations historically.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ..strategies.base import TradeResult
from ..pricing.black_scholes import OptionType
from .thetadata_client import ThetaDataClient


@dataclass
class HistoricalComparison:
    ticker: str
    entry_date: date
    expiry_date: date
    strike: float
    option_type: str
    bs_entry_price: float
    real_entry_bid: float
    real_entry_ask: float
    real_entry_mid: float
    bs_exit_price: float
    real_exit_bid: float
    real_exit_ask: float
    real_exit_mid: float
    bs_pnl: float
    real_pnl: float  # using mid prices
    pnl_difference: float
    entry_price_diff_pct: float


@dataclass
class HistoricalValidator:
    """Validate backtest trades against real historical option prices from ThetaData."""

    client: ThetaDataClient = field(default_factory=ThetaDataClient)

    def validate_trades(
        self,
        trades: list[TradeResult],
        max_trades: int | None = None,
        verbose: bool = True,
    ) -> list[HistoricalComparison]:
        """Look up real historical prices for backtest trades.

        Args:
            trades: List of TradeResult from the backtest engine.
            max_trades: Limit number of trades to validate (rate limit friendly).
            verbose: Print progress.
        """
        if not self.client.check_connection():
            raise ConnectionError(
                "Cannot connect to Theta Terminal at localhost:25510.\n"
                "Make sure the Theta Terminal is running.\n"
                "Download from: https://www.thetadata.net/terminal"
            )

        comparisons = []
        trades_to_check = trades[:max_trades] if max_trades else trades
        total = len(trades_to_check)

        if verbose:
            print(f"Validating {total} trades against ThetaData historical prices...")
            print(f"(Free tier: ~20 req/min, this may take a while)\n")

        for i, trade in enumerate(trades_to_check):
            if verbose and (i + 1) % 5 == 0:
                print(f"  [{i+1}/{total}] Processing {trade.ticker} {trade.entry_date.date()}...")

            for leg in trade.legs:
                try:
                    comp = self._validate_leg(trade, leg)
                    if comp is not None:
                        comparisons.append(comp)
                except Exception as e:
                    if verbose:
                        print(f"  Skipping {trade.ticker} {leg.strike} {leg.option_type.value}: {e}")
                    continue

        if verbose:
            print(f"\nValidated {len(comparisons)} option legs.")

        return comparisons

    def _validate_leg(
        self,
        trade: TradeResult,
        leg,
    ) -> HistoricalComparison | None:
        """Validate a single option leg against ThetaData v3 EOD data."""
        right = "call" if leg.option_type == OptionType.CALL else "put"
        entry_dt = trade.entry_date.date() if hasattr(trade.entry_date, 'date') else trade.entry_date
        expiry_dt = trade.exit_date.date() if hasattr(trade.exit_date, 'date') else trade.exit_date

        # Fetch EOD data for the contract covering entry and expiry
        eod_df = self.client.get_option_eod(
            symbol=trade.ticker,
            expiration=expiry_dt,
            strike=leg.strike,
            right=right,
            start_date=entry_dt,
            end_date=expiry_dt,
        )

        if eod_df.empty or "mid" not in eod_df.columns:
            return None

        # Find entry-day row (closest to entry date)
        if "created" in eod_df.columns:
            eod_df["_date"] = eod_df["created"].dt.date
        elif "last_trade" in eod_df.columns:
            eod_df["_date"] = eod_df["last_trade"].dt.date
        else:
            return None

        entry_rows = eod_df[eod_df["_date"] == entry_dt]
        if entry_rows.empty:
            # Try nearest date after entry
            entry_rows = eod_df[eod_df["_date"] >= entry_dt].head(1)
        if entry_rows.empty:
            return None

        entry_row = entry_rows.iloc[-1]
        real_entry_mid = float(entry_row["mid"])
        real_entry_bid = float(entry_row.get("bid", real_entry_mid))
        real_entry_ask = float(entry_row.get("ask", real_entry_mid))

        # Find expiry-day row
        exit_rows = eod_df[eod_df["_date"] == expiry_dt]
        if exit_rows.empty:
            exit_rows = eod_df[eod_df["_date"] <= expiry_dt].tail(1)

        if exit_rows.empty:
            # At expiry, if no data, assume intrinsic value
            if leg.option_type == OptionType.CALL:
                real_exit_mid = max(trade.exit_underlying - leg.strike, 0)
            else:
                real_exit_mid = max(leg.strike - trade.exit_underlying, 0)
            real_exit_bid = real_exit_mid
            real_exit_ask = real_exit_mid
        else:
            exit_row = exit_rows.iloc[-1]
            real_exit_mid = float(exit_row["mid"])
            real_exit_bid = float(exit_row.get("bid", real_exit_mid))
            real_exit_ask = float(exit_row.get("ask", real_exit_mid))

        # Compute P&L comparison (per contract, 100 multiplier)
        sign = 1 if leg.is_long else -1
        bs_pnl = sign * (leg.exit_price - leg.entry_price) * 100
        real_pnl = sign * (real_exit_mid - real_entry_mid) * 100

        entry_diff_pct = (
            (leg.entry_price - real_entry_mid) / real_entry_mid
            if real_entry_mid > 0.01 else 0
        )

        return HistoricalComparison(
            ticker=trade.ticker,
            entry_date=entry_dt,
            expiry_date=expiry_dt,
            strike=leg.strike,
            option_type=leg.option_type.value,
            bs_entry_price=leg.entry_price,
            real_entry_bid=real_entry_bid,
            real_entry_ask=real_entry_ask,
            real_entry_mid=real_entry_mid,
            bs_exit_price=leg.exit_price,
            real_exit_bid=real_exit_bid,
            real_exit_ask=real_exit_ask,
            real_exit_mid=real_exit_mid,
            bs_pnl=bs_pnl,
            real_pnl=real_pnl,
            pnl_difference=bs_pnl - real_pnl,
            entry_price_diff_pct=entry_diff_pct,
        )

    def comparison_df(self, comparisons: list[HistoricalComparison]) -> pd.DataFrame:
        """Convert comparisons to DataFrame."""
        if not comparisons:
            return pd.DataFrame()

        records = []
        for c in comparisons:
            records.append({
                "ticker": c.ticker,
                "entry_date": c.entry_date,
                "expiry_date": c.expiry_date,
                "strike": c.strike,
                "type": c.option_type,
                "bs_entry": c.bs_entry_price,
                "real_entry_mid": c.real_entry_mid,
                "real_entry_bid": c.real_entry_bid,
                "real_entry_ask": c.real_entry_ask,
                "entry_diff%": c.entry_price_diff_pct,
                "bs_exit": c.bs_exit_price,
                "real_exit_mid": c.real_exit_mid,
                "bs_pnl": c.bs_pnl,
                "real_pnl": c.real_pnl,
                "pnl_diff": c.pnl_difference,
            })

        return pd.DataFrame(records)

    def print_summary(self, comparisons: list[HistoricalComparison]) -> None:
        """Print validation summary report."""
        if not comparisons:
            print("No comparisons to report.")
            return

        df = self.comparison_df(comparisons)

        print(f"\n{'='*78}")
        print(f"  Historical Validation Report (ThetaData)")
        print(f"{'='*78}")
        print(f"  Legs validated:     {len(df)}")
        print(f"  Tickers:            {', '.join(df['ticker'].unique())}")
        print(f"  Date range:         {df['entry_date'].min()} to {df['expiry_date'].max()}")

        # Pricing accuracy
        abs_diff = df["entry_diff%"].abs()
        print(f"\n  --- Pricing Accuracy (entry) ---")
        print(f"  Mean |BS - Real| / Real:    {abs_diff.mean():.1%}")
        print(f"  Median |BS - Real| / Real:  {abs_diff.median():.1%}")
        print(f"  90th percentile:            {abs_diff.quantile(0.9):.1%}")
        print(f"  BS overprices entry:        {(df['entry_diff%'] > 0).sum()} / {len(df)}")
        print(f"  BS underprices entry:       {(df['entry_diff%'] < 0).sum()} / {len(df)}")

        # P&L accuracy
        print(f"\n  --- P&L Comparison (per leg, per contract) ---")
        print(f"  Total BS P&L:       ${df['bs_pnl'].sum():,.2f}")
        print(f"  Total Real P&L:     ${df['real_pnl'].sum():,.2f}")
        print(f"  P&L Difference:     ${df['pnl_diff'].sum():,.2f}")

        # Correlation
        if len(df) > 2:
            corr = df["bs_pnl"].corr(df["real_pnl"])
            print(f"  BS vs Real P&L corr: {corr:.3f}")

        # By ticker
        print(f"\n  --- By Ticker ---")
        ticker_summary = df.groupby("ticker").agg(
            legs=("bs_pnl", "count"),
            bs_total=("bs_pnl", "sum"),
            real_total=("real_pnl", "sum"),
            avg_entry_diff=("entry_diff%", lambda x: x.abs().mean()),
        ).sort_values("avg_entry_diff")

        print(f"  {'Ticker':>8}  {'Legs':>5}  {'BS P&L':>10}  {'Real P&L':>10}  {'Avg |Diff|':>10}")
        print(f"  {'-'*50}")
        for ticker, row in ticker_summary.iterrows():
            print(
                f"  {ticker:>8}  {int(row['legs']):>5}  "
                f"${row['bs_total']:>9,.0f}  ${row['real_total']:>9,.0f}  "
                f"{row['avg_entry_diff']:>9.1%}"
            )

        print(f"\n{'='*78}")

        # Verdict
        overall_diff = abs_diff.mean()
        if overall_diff < 0.10:
            print("  VERDICT: BS model is well-calibrated. Backtest results are reliable.")
        elif overall_diff < 0.25:
            print("  VERDICT: BS model is directionally correct. Use backtest for strategy")
            print("           selection but expect real P&L to differ by ~10-25%.")
        else:
            print("  VERDICT: Significant pricing gap. Consider tuning IV markup per ticker")
            print("           or using ThetaData directly for production backtesting.")
        print(f"{'='*78}")
