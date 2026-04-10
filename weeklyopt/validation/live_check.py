"""Validate backtest results against live option chain data.

Uses yfinance for current snapshots and optionally ThetaData for richer data.
This is the 'mode 1' validator — compare synthetic BS prices against real bid/asks.
"""

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
import numpy as np
import yfinance as yf

from ..pricing.black_scholes import bs_price, bs_greeks, OptionType, strike_from_delta
from ..pricing.volatility import historical_vol


@dataclass
class ValidationResult:
    ticker: str
    expiration: str
    strike: float
    option_type: str
    bs_price: float
    market_bid: float
    market_ask: float
    market_mid: float
    price_diff: float
    price_diff_pct: float
    bs_delta: float
    market_iv: float | None


@dataclass
class LiveValidator:
    """Compare Black-Scholes synthetic prices against live option chains.

    This bridges mode 3 (synthetic backtest) and mode 1 (real data validation).
    Run this on strategies that looked good in backtest to see how BS prices
    compare to actual market pricing.
    """
    risk_free_rate: float = 0.045
    hv_window: int = 20

    def fetch_live_chain(self, ticker: str) -> tuple[pd.DataFrame, list[str]]:
        """Fetch current option chain from yfinance.

        Returns (chain_df, expirations_list).
        """
        tk = yf.Ticker(ticker)
        expirations = tk.options  # list of expiration date strings

        if not expirations:
            raise ValueError(f"No options available for {ticker}")

        # Get the nearest weekly expiration
        nearest_exp = expirations[0]

        chain = tk.option_chain(nearest_exp)
        calls = chain.calls.copy()
        calls["option_type"] = "call"
        puts = chain.puts.copy()
        puts["option_type"] = "put"

        df = pd.concat([calls, puts], ignore_index=True)
        df["expiration"] = nearest_exp

        return df, expirations

    def validate_strategy_pricing(
        self,
        ticker: str,
        target_delta: float = 0.30,
        option_type: OptionType = OptionType.CALL,
    ) -> list[ValidationResult]:
        """Compare BS prices to live market for strikes near a target delta.

        Returns ValidationResult for each relevant strike.
        """
        chain_df, expirations = self.fetch_live_chain(ticker)

        # Get current price and historical vol
        tk = yf.Ticker(ticker)
        hist = tk.history(period="3mo")
        if hist.empty:
            raise ValueError(f"No price history for {ticker}")

        current_price = float(hist["Close"].iloc[-1])
        hv = historical_vol(hist["Close"], self.hv_window)
        current_vol = float(hv.iloc[-1]) if not hv.empty else 0.20

        results = []
        nearest_exp = expirations[0]
        exp_date = pd.to_datetime(nearest_exp)
        dte_days = max((exp_date - pd.Timestamp.now()).days, 1)
        dte_years = dte_days / 365

        # Filter to the option type we care about
        type_str = "call" if option_type == OptionType.CALL else "put"
        subset = chain_df[chain_df["option_type"] == type_str].copy()

        if subset.empty:
            return results

        # Find the BS-implied strike for target delta
        bs_strike = strike_from_delta(
            current_price, dte_years, self.risk_free_rate, current_vol, target_delta, option_type
        )

        # Compare around that strike (+/- a few strikes)
        subset["dist"] = abs(subset["strike"] - bs_strike)
        nearby = subset.nsmallest(5, "dist")

        for _, row in nearby.iterrows():
            K = float(row["strike"])
            bs_px = bs_price(current_price, K, dte_years, self.risk_free_rate, current_vol, option_type)
            greeks = bs_greeks(current_price, K, dte_years, self.risk_free_rate, current_vol, option_type)

            bid = float(row.get("bid", 0))
            ask = float(row.get("ask", 0))
            mid = (bid + ask) / 2 if (bid + ask) > 0 else 0
            market_iv = float(row["impliedVolatility"]) if "impliedVolatility" in row and pd.notna(row["impliedVolatility"]) else None

            diff = bs_px - mid
            diff_pct = diff / mid if mid > 0 else 0

            results.append(ValidationResult(
                ticker=ticker,
                expiration=nearest_exp,
                strike=K,
                option_type=type_str,
                bs_price=bs_px,
                market_bid=bid,
                market_ask=ask,
                market_mid=mid,
                price_diff=diff,
                price_diff_pct=diff_pct,
                bs_delta=greeks.delta,
                market_iv=market_iv,
            ))

        return results

    def print_validation(self, results: list[ValidationResult]) -> None:
        """Print a formatted validation comparison."""
        if not results:
            print("No validation results.")
            return

        ticker = results[0].ticker
        exp = results[0].expiration
        print(f"\n{'='*70}")
        print(f"  Validation: {ticker}  |  Expiry: {exp}")
        print(f"{'='*70}")
        print(f"  {'Strike':>8}  {'Type':>5}  {'BS Price':>9}  {'Mkt Mid':>8}  {'Bid':>6}  {'Ask':>6}  {'Diff%':>7}  {'Delta':>6}")
        print(f"  {'-'*62}")

        for r in results:
            print(
                f"  {r.strike:>8.1f}  {r.option_type:>5}  "
                f"${r.bs_price:>7.2f}  ${r.market_mid:>6.2f}  "
                f"${r.market_bid:>4.2f}  ${r.market_ask:>4.2f}  "
                f"{r.price_diff_pct:>6.1%}  {r.bs_delta:>6.3f}"
            )

        # Summary
        avg_diff = np.mean([abs(r.price_diff_pct) for r in results])
        print(f"\n  Avg absolute price diff: {avg_diff:.1%}")
        if avg_diff < 0.10:
            print("  BS pricing is reasonably close to market.")
        elif avg_diff < 0.25:
            print("  BS pricing shows moderate deviation — results are directionally useful.")
        else:
            print("  BS pricing diverges significantly — treat backtest results with caution.")
        print(f"{'='*70}")
