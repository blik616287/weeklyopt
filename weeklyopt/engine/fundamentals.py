"""Fundamental and options flow data for signal enrichment.

Pulls from yfinance:
- Earnings dates (avoid selling premium into earnings week)
- Open interest and volume (liquidity, crowded trades)
- Put/call ratio (sentiment — high = bearish fear = rich put premium)
- Market cap, EPS, P/E (size/quality filters)
- Short interest (squeeze risk)
"""

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class FundamentalData:
    ticker: str
    # Earnings
    next_earnings_date: date | None = None
    days_to_earnings: int | None = None
    earnings_this_week: bool = False
    eps_surprise_pct: float | None = None  # last quarter beat/miss %

    # Options flow
    total_call_oi: int = 0
    total_put_oi: int = 0
    put_call_oi_ratio: float = 1.0  # >1 = more puts = bearish sentiment
    total_call_volume: int = 0
    total_put_volume: int = 0
    put_call_volume_ratio: float = 1.0
    max_pain_strike: float | None = None

    # Fundamentals
    market_cap: float = 0
    trailing_pe: float | None = None
    forward_pe: float | None = None
    eps_trailing: float | None = None
    short_pct_float: float | None = None  # % of float sold short

    # Liquidity
    avg_volume_30d: int = 0
    options_volume_avg: int = 0  # rough proxy


def fetch_fundamentals(ticker: str) -> FundamentalData:
    """Fetch fundamental and options flow data from yfinance."""
    tk = yf.Ticker(ticker)
    info = tk.info or {}
    fd = FundamentalData(ticker=ticker)

    # ── Earnings ──
    try:
        cal = tk.calendar
        if cal is not None:
            if isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
                earnings_dates = cal.loc["Earnings Date"]
                if len(earnings_dates) > 0:
                    next_earn = pd.to_datetime(earnings_dates.iloc[0]).date()
                    fd.next_earnings_date = next_earn
                    fd.days_to_earnings = (next_earn - date.today()).days
                    fd.earnings_this_week = 0 <= fd.days_to_earnings <= 5
            elif isinstance(cal, dict):
                earn_dates = cal.get("Earnings Date", [])
                if earn_dates:
                    next_earn = pd.to_datetime(earn_dates[0]).date()
                    fd.next_earnings_date = next_earn
                    fd.days_to_earnings = (next_earn - date.today()).days
                    fd.earnings_this_week = 0 <= fd.days_to_earnings <= 5
    except Exception:
        pass

    # EPS surprise from last quarter
    try:
        earnings_hist = tk.earnings_dates
        if earnings_hist is not None and not earnings_hist.empty:
            # Find most recent past earnings
            past = earnings_hist[earnings_hist.index <= pd.Timestamp.now()]
            if not past.empty and "Surprise(%)" in past.columns:
                fd.eps_surprise_pct = float(past["Surprise(%)"].iloc[0])
    except Exception:
        pass

    # ── Options flow (current chain) ──
    try:
        expirations = tk.options
        if expirations:
            # Get nearest expiration
            chain = tk.option_chain(expirations[0])

            calls = chain.calls
            puts = chain.puts

            fd.total_call_oi = int(calls["openInterest"].sum()) if "openInterest" in calls.columns else 0
            fd.total_put_oi = int(puts["openInterest"].sum()) if "openInterest" in puts.columns else 0
            fd.total_call_volume = int(calls["volume"].fillna(0).sum()) if "volume" in calls.columns else 0
            fd.total_put_volume = int(puts["volume"].fillna(0).sum()) if "volume" in puts.columns else 0

            if fd.total_call_oi > 0:
                fd.put_call_oi_ratio = fd.total_put_oi / fd.total_call_oi
            if fd.total_call_volume > 0:
                fd.put_call_volume_ratio = fd.total_put_volume / fd.total_call_volume

            # Max pain: strike where total OI (calls + puts) value is minimized
            fd.max_pain_strike = _compute_max_pain(calls, puts)

    except Exception:
        pass

    # ── Fundamentals ──
    fd.market_cap = info.get("marketCap", 0) or 0
    fd.trailing_pe = info.get("trailingPE")
    fd.forward_pe = info.get("forwardPE")
    fd.eps_trailing = info.get("trailingEps")
    fd.short_pct_float = info.get("shortPercentOfFloat")
    fd.avg_volume_30d = info.get("averageVolume", 0) or 0

    return fd


def _compute_max_pain(calls: pd.DataFrame, puts: pd.DataFrame) -> float | None:
    """Compute max pain strike — where option writers have minimum payout.

    Price tends to gravitate toward max pain at expiry (pinning effect).
    """
    try:
        all_strikes = sorted(set(calls["strike"].tolist() + puts["strike"].tolist()))
        if not all_strikes:
            return None

        min_pain = float("inf")
        max_pain_strike = all_strikes[len(all_strikes) // 2]

        for strike in all_strikes:
            # Call pain: for each call, if stock at `strike`, what do call holders lose?
            call_pain = 0
            for _, row in calls.iterrows():
                oi = row.get("openInterest", 0) or 0
                call_strike = row["strike"]
                if strike > call_strike:
                    call_pain += (strike - call_strike) * oi * 100

            # Put pain
            put_pain = 0
            for _, row in puts.iterrows():
                oi = row.get("openInterest", 0) or 0
                put_strike = row["strike"]
                if strike < put_strike:
                    put_pain += (put_strike - strike) * oi * 100

            total = call_pain + put_pain
            if total < min_pain:
                min_pain = total
                max_pain_strike = strike

        return max_pain_strike
    except Exception:
        return None


def fetch_all_fundamentals(tickers: list[str], verbose: bool = True) -> dict[str, FundamentalData]:
    """Fetch fundamentals for all tickers."""
    results = {}
    for i, ticker in enumerate(tickers):
        if verbose:
            print(f"  [{i+1}/{len(tickers)}] Fetching {ticker} fundamentals...")
        try:
            results[ticker] = fetch_fundamentals(ticker)
        except Exception as e:
            if verbose:
                print(f"    Failed: {e}")
            results[ticker] = FundamentalData(ticker=ticker)
    return results


def print_fundamentals_summary(fundamentals: dict[str, FundamentalData]) -> None:
    """Print a compact fundamentals overview."""
    print(f"\n{'='*100}")
    print(f"  Fundamentals & Options Flow")
    print(f"{'='*100}")
    print(
        f"  {'Ticker':>6}  {'MktCap':>10}  {'P/E':>6}  {'EPS':>6}  "
        f"{'P/C OI':>6}  {'P/C Vol':>7}  {'Max Pain':>9}  "
        f"{'Short%':>6}  {'Earnings':>12}  {'Flag'}"
    )
    print(f"  {'-'*95}")

    for ticker in sorted(fundamentals.keys()):
        f = fundamentals[ticker]
        mc = f"${f.market_cap/1e9:.0f}B" if f.market_cap > 1e9 else f"${f.market_cap/1e6:.0f}M"
        pe = f"{f.trailing_pe:.1f}" if f.trailing_pe else "—"
        eps = f"{f.eps_trailing:.2f}" if f.eps_trailing else "—"
        pcoi = f"{f.put_call_oi_ratio:.2f}" if f.total_call_oi > 0 else "—"
        pcvol = f"{f.put_call_volume_ratio:.2f}" if f.total_call_volume > 0 else "—"
        mp = f"${f.max_pain_strike:.0f}" if f.max_pain_strike else "—"
        short = f"{f.short_pct_float:.1%}" if f.short_pct_float else "—"
        earn = f"{f.days_to_earnings}d" if f.days_to_earnings is not None else "—"

        flags = []
        if f.earnings_this_week:
            flags.append("EARNINGS!")
        if f.put_call_oi_ratio > 1.5:
            flags.append("bearish flow")
        elif f.put_call_oi_ratio < 0.5:
            flags.append("bullish flow")
        if f.short_pct_float and f.short_pct_float > 0.10:
            flags.append("high short")
        if f.eps_surprise_pct and f.eps_surprise_pct > 10:
            flags.append(f"beat {f.eps_surprise_pct:.0f}%")
        elif f.eps_surprise_pct and f.eps_surprise_pct < -10:
            flags.append(f"miss {f.eps_surprise_pct:.0f}%")

        flag_str = " | ".join(flags) if flags else ""

        print(
            f"  {ticker:>6}  {mc:>10}  {pe:>6}  {eps:>6}  "
            f"{pcoi:>6}  {pcvol:>7}  {mp:>9}  "
            f"{short:>6}  {earn:>12}  {flag_str}"
        )

    print(f"{'='*100}")
