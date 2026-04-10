"""Default configuration for the backtesting framework."""

from dataclasses import dataclass, field
from datetime import date

TICKERS = [
    "SPY", "QQQ", "AAPL", "MSFT", "AMZN",
    "TSLA", "NVDA", "META", "GOOGL", "AMD",
    "JPM", "BAC", "XLF", "IWM", "GLD",
]

# Expanded universe: sector ETFs for decorrelation (#1)
TICKERS_EXPANDED = TICKERS + [
    "TLT",   # Long-term bonds — inverse equity correlation
    "XLE",   # Energy — oil cycle, decorrelated from tech
    "XBI",   # Biotech — fat tails, rich put premium
    "KRE",   # Regional banks — event-driven vol
    "COIN",  # Crypto proxy — massive IV, rich skew
    "SLV",   # Silver — commodity vol
    "EEM",   # Emerging markets — different macro cycle
    "SMH",   # Semiconductors ETF — tighter spreads than single names
]

# Correlation clusters for pair avoidance (#7)
CORRELATION_CLUSTERS = {
    "mega_tech": {"AAPL", "MSFT", "GOOGL", "META"},
    "semiconductors": {"NVDA", "AMD", "SMH"},
    "broad_market": {"SPY", "QQQ", "IWM"},
    "financials": {"JPM", "BAC", "KRE", "XLF"},
    "commodities": {"GLD", "SLV", "XLE"},
    "high_beta": {"TSLA", "COIN", "ARKK"},
    "bonds": {"TLT"},
    "biotech": {"XBI"},
    "emerging": {"EEM"},
}

# Risk-free rate proxy (10Y Treasury ~4.5% as of 2024-2025)
RISK_FREE_RATE = 0.045

# Historical volatility lookback (trading days)
HV_WINDOW = 20

# Default IV markup over historical vol (options tend to price higher)
IV_MARKUP = 1.15


@dataclass
class BacktestConfig:
    tickers: list[str] = field(default_factory=lambda: TICKERS.copy())
    start_date: date = date(2020, 1, 1)
    end_date: date = date(2025, 12, 31)
    initial_capital: float = 100_000.0
    risk_free_rate: float = RISK_FREE_RATE
    hv_window: int = HV_WINDOW
    iv_markup: float = IV_MARKUP
    # max allocation per ticker as fraction of portfolio
    max_ticker_allocation: float = 0.15
    # slippage per contract leg (dollars)
    slippage_per_contract: float = 0.05
    # commission per contract
    commission_per_contract: float = 0.65
