"""Dynamic ticker screener: find the best credit spread candidates from a broad universe.

Instead of a curated 23-ticker list, scan a wider pool of weekly-options-eligible
stocks and ETFs, filter for tradeable characteristics, and feed survivors to the
signal scanner.

Pipeline:
1. Start from a broad universe (~100-150 liquid tickers with weekly options)
2. Filter for minimum liquidity (volume, OI, bid/ask width)
3. Compute IV rank and put IV/HV ratio from price history
4. Rank by credit spread suitability
5. Return top N candidates for the signal scanner

Data source: yfinance (free, rate-limited). Caches aggressively.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import yfinance as yf

from ..pricing.volatility import historical_vol
from ..pricing.calibration import IVCalibrator


# Broad universe: stocks and ETFs known to have weekly options and decent volume.
# This is the starting pool — the screener filters it down each week.
WEEKLY_OPTIONS_UNIVERSE = [
    # Mega-cap tech
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "AVGO", "ORCL", "CRM",
    "ADBE", "NFLX", "AMD", "INTC", "QCOM", "MU", "AMAT", "LRCX", "KLAC", "MRVL",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "SCHW", "BLK", "AXP", "COF",
    # Healthcare
    "UNH", "JNJ", "PFE", "ABBV", "MRK", "LLY", "BMY", "AMGN", "GILD", "MRNA",
    # Consumer
    "WMT", "COST", "HD", "LOW", "TGT", "SBUX", "MCD", "NKE", "DIS", "ABNB",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "DVN", "MPC", "VLO", "PSX",
    # Industrials
    "CAT", "DE", "BA", "RTX", "LMT", "GE", "HON", "UPS", "FDX", "UNP",
    # Other
    "COIN", "SQ", "PYPL", "SHOP", "SNAP", "UBER", "LYFT", "RBLX", "PLTR", "SOFI",
    # ETFs — sectors and thematic
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XBI", "XLK", "XLV", "XLP",
    "XLI", "XLU", "XLRE", "GLD", "SLV", "TLT", "HYG", "EEM", "FXI", "KWEB",
    "SMH", "KRE", "ARKK", "ARKG", "BITO", "USO", "GDX", "VXX",
]

SCREEN_CACHE_DIR = Path.home() / ".weeklyopt_cache" / "screener"


@dataclass
class ScreenResult:
    """Screening result for a single ticker."""
    ticker: str
    price: float = 0.0
    avg_volume: int = 0
    # Options metrics
    has_weekly_options: bool = False
    nearest_expiry_days: int = 0
    total_put_oi: int = 0
    total_call_oi: int = 0
    put_bid_ask_avg: float = 0.0  # avg bid/ask spread on puts near ATM
    call_bid_ask_avg: float = 0.0
    put_volume_daily: int = 0
    call_volume_daily: int = 0
    # IV metrics
    hv_20d: float = 0.0
    iv_rank: float = 0.0
    # Suitability score (0-100)
    liquidity_score: float = 0.0
    iv_score: float = 0.0
    overall_score: float = 0.0
    rejection_reason: str = ""


@dataclass
class ScreenerConfig:
    """Screening criteria."""
    min_price: float = 10.0
    max_price: float = 5000.0
    min_avg_volume: int = 500_000  # shares/day
    min_option_oi: int = 1_000  # total OI across chain
    max_bid_ask_spread: float = 0.20  # dollars, on near-ATM options
    min_iv_rank: float = 30.0
    min_option_volume: int = 100  # daily option contracts
    max_tickers: int = 30  # return top N


def screen_universe(
    tickers: list[str] | None = None,
    config: ScreenerConfig | None = None,
    verbose: bool = True,
) -> list[ScreenResult]:
    """Screen the broad universe and return ranked candidates.

    This is rate-limit aware — caches results for 4 hours.
    """
    tickers = tickers or WEEKLY_OPTIONS_UNIVERSE
    config = config or ScreenerConfig()

    if verbose:
        print(f"Screening {len(tickers)} tickers for credit spread candidates...\n")

    # Check cache
    cache = _load_cache()
    if cache and verbose:
        print(f"  Using cached data for {len(cache)} tickers (< 4hrs old)")

    results = []
    scanned = 0
    skipped_cache = 0

    for i, ticker in enumerate(tickers):
        if verbose and (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(tickers)}] scanned={scanned} cached={skipped_cache}...")

        # Use cache if fresh
        if ticker in cache:
            results.append(cache[ticker])
            skipped_cache += 1
            continue

        try:
            result = _screen_ticker(ticker, config)
            results.append(result)
            scanned += 1
            # Rate limit: yfinance gets angry above ~2 req/sec
            if scanned % 5 == 0:
                time.sleep(1)
        except Exception as e:
            results.append(ScreenResult(ticker=ticker, rejection_reason=f"error: {e}"))

    # Save cache
    _save_cache({r.ticker: r for r in results})

    # Filter and rank
    passed = [r for r in results if not r.rejection_reason and r.overall_score > 0]
    passed.sort(key=lambda r: r.overall_score, reverse=True)

    if verbose:
        rejected = [r for r in results if r.rejection_reason]
        print(f"\n  Results: {len(passed)} passed / {len(rejected)} rejected / {len(results)} total")

    return passed[:config.max_tickers]


def _screen_ticker(ticker: str, config: ScreenerConfig) -> ScreenResult:
    """Screen a single ticker."""
    result = ScreenResult(ticker=ticker)

    tk = yf.Ticker(ticker)

    # Price and volume from recent history
    hist = tk.history(period="3mo")
    if hist.empty or len(hist) < 20:
        result.rejection_reason = "insufficient price history"
        return result

    result.price = float(hist["Close"].iloc[-1])
    result.avg_volume = int(hist["Volume"].tail(20).mean())

    if result.price < config.min_price or result.price > config.max_price:
        result.rejection_reason = f"price ${result.price:.0f} out of range"
        return result

    if result.avg_volume < config.min_avg_volume:
        result.rejection_reason = f"avg volume {result.avg_volume:,} below {config.min_avg_volume:,}"
        return result

    # Check for weekly options
    try:
        expirations = tk.options
    except Exception:
        result.rejection_reason = "no options chain"
        return result

    if not expirations:
        result.rejection_reason = "no options available"
        return result

    # Check nearest expiry is within 7 days (weekly)
    nearest = pd.to_datetime(expirations[0]).date()
    days_to_exp = (nearest - date.today()).days
    result.nearest_expiry_days = days_to_exp
    result.has_weekly_options = days_to_exp <= 8

    if not result.has_weekly_options:
        result.rejection_reason = f"no weekly options (nearest: {days_to_exp}d)"
        return result

    # Options chain analysis
    try:
        chain = tk.option_chain(expirations[0])
        calls = chain.calls
        puts = chain.puts
    except Exception:
        result.rejection_reason = "failed to fetch chain"
        return result

    # OI and volume
    result.total_put_oi = int(puts["openInterest"].fillna(0).sum()) if "openInterest" in puts.columns else 0
    result.total_call_oi = int(calls["openInterest"].fillna(0).sum()) if "openInterest" in calls.columns else 0
    result.put_volume_daily = int(puts["volume"].fillna(0).sum()) if "volume" in puts.columns else 0
    result.call_volume_daily = int(calls["volume"].fillna(0).sum()) if "volume" in calls.columns else 0

    total_oi = result.total_put_oi + result.total_call_oi
    if total_oi < config.min_option_oi:
        result.rejection_reason = f"option OI {total_oi:,} below {config.min_option_oi:,}"
        return result

    total_opt_vol = result.put_volume_daily + result.call_volume_daily
    if total_opt_vol < config.min_option_volume:
        result.rejection_reason = f"option volume {total_opt_vol} below {config.min_option_volume}"
        return result

    # Bid/ask spread on near-ATM options
    atm_range = result.price * 0.03  # within 3% of spot
    near_atm_puts = puts[abs(puts["strike"] - result.price) < atm_range]
    near_atm_calls = calls[abs(calls["strike"] - result.price) < atm_range]

    if not near_atm_puts.empty and "bid" in near_atm_puts.columns and "ask" in near_atm_puts.columns:
        spreads = near_atm_puts["ask"].fillna(0) - near_atm_puts["bid"].fillna(0)
        result.put_bid_ask_avg = float(spreads.mean())
    if not near_atm_calls.empty and "bid" in near_atm_calls.columns and "ask" in near_atm_calls.columns:
        spreads = near_atm_calls["ask"].fillna(0) - near_atm_calls["bid"].fillna(0)
        result.call_bid_ask_avg = float(spreads.mean())

    if result.put_bid_ask_avg > config.max_bid_ask_spread and result.call_bid_ask_avg > config.max_bid_ask_spread:
        result.rejection_reason = f"wide spreads (put: ${result.put_bid_ask_avg:.2f}, call: ${result.call_bid_ask_avg:.2f})"
        return result

    # HV and IV rank
    hv = historical_vol(hist["Close"], 20)
    hv_clean = hv.dropna()
    if len(hv_clean) > 0:
        result.hv_20d = float(hv_clean.iloc[-1])
        hv_1yr = hv_clean.tail(252)
        if len(hv_1yr) > 20 and hv_1yr.max() != hv_1yr.min():
            result.iv_rank = float(
                (result.hv_20d - hv_1yr.min()) / (hv_1yr.max() - hv_1yr.min()) * 100
            )

    # Scoring
    # Liquidity: volume, OI, tight spreads
    vol_score = min(100, result.avg_volume / 5_000_000 * 100)
    oi_score = min(100, total_oi / 50_000 * 100)
    spread_score = max(0, 100 - result.put_bid_ask_avg / 0.20 * 100)
    opt_vol_score = min(100, total_opt_vol / 10_000 * 100)
    result.liquidity_score = (vol_score * 0.2 + oi_score * 0.3 + spread_score * 0.3 + opt_vol_score * 0.2)

    # IV: high IV rank = rich premium
    result.iv_score = min(100, result.iv_rank * 1.2)

    # Overall: weighted combination
    result.overall_score = result.liquidity_score * 0.4 + result.iv_score * 0.6

    return result


def _cache_path() -> Path:
    SCREEN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return SCREEN_CACHE_DIR / "screen_results.json"


def _load_cache() -> dict[str, ScreenResult]:
    """Load cached results if less than 4 hours old."""
    path = _cache_path()
    if not path.exists():
        return {}

    # Check age
    import os
    age_hours = (time.time() - os.path.getmtime(path)) / 3600
    if age_hours > 4:
        return {}

    try:
        data = json.loads(path.read_text())
        results = {}
        for ticker, vals in data.items():
            r = ScreenResult(ticker=ticker)
            for k, v in vals.items():
                if hasattr(r, k):
                    setattr(r, k, v)
            results[ticker] = r
        return results
    except Exception:
        return {}


def _save_cache(results: dict[str, ScreenResult]):
    """Save screening results to cache."""
    path = _cache_path()
    data = {}
    for ticker, r in results.items():
        data[ticker] = {
            "ticker": r.ticker, "price": r.price, "avg_volume": r.avg_volume,
            "has_weekly_options": r.has_weekly_options, "nearest_expiry_days": r.nearest_expiry_days,
            "total_put_oi": r.total_put_oi, "total_call_oi": r.total_call_oi,
            "put_bid_ask_avg": r.put_bid_ask_avg, "call_bid_ask_avg": r.call_bid_ask_avg,
            "put_volume_daily": r.put_volume_daily, "call_volume_daily": r.call_volume_daily,
            "hv_20d": r.hv_20d, "iv_rank": r.iv_rank,
            "liquidity_score": r.liquidity_score, "iv_score": r.iv_score,
            "overall_score": r.overall_score, "rejection_reason": r.rejection_reason,
        }
    path.write_text(json.dumps(data, indent=2))


def print_screen_results(results: list[ScreenResult], show_all: bool = False) -> None:
    """Print screening results."""
    print(f"\n{'='*110}")
    print(f"  Dynamic Screener: Top {len(results)} Credit Spread Candidates")
    print(f"{'='*110}")
    print(f"  {'#':>3}  {'Ticker':>6}  {'Price':>8}  {'AvgVol':>8}  {'OptOI':>8}  "
          f"{'PutSprd':>7}  {'HV':>6}  {'IVRank':>6}  "
          f"{'Liq':>4}  {'IV':>4}  {'Score':>5}")
    print(f"  {'-'*105}")

    for i, r in enumerate(results):
        vol_str = f"{r.avg_volume/1e6:.1f}M" if r.avg_volume >= 1e6 else f"{r.avg_volume/1e3:.0f}K"
        oi_str = f"{(r.total_put_oi + r.total_call_oi)/1e3:.0f}K"
        print(
            f"  {i+1:>3}  {r.ticker:>6}  ${r.price:>7.2f}  {vol_str:>8}  {oi_str:>8}  "
            f"${r.put_bid_ask_avg:>5.2f}  {r.hv_20d:>5.1%}  {r.iv_rank:>5.0f}%  "
            f"{r.liquidity_score:>4.0f}  {r.iv_score:>4.0f}  {r.overall_score:>5.0f}"
        )

    print(f"{'='*110}")
