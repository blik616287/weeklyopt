# weeklyopt

Weekly options backtesting framework with signal-driven portfolio management, calibrated IV pricing, and ML-based exit timing. Built for credit spread strategies on high-liquidity equities and sector ETFs.

## Quick Start

```bash
# Install everything (Python deps + Java 21 + Theta Terminal jar)
make install-all

# Run smoke tests
make test

# Scan this week's opportunities and generate a trade plan
make plan

# Run the optimized portfolio backtest
make portfolio
```

### First Time Setup

```bash
git clone <repo> && cd weekly

# 1. Install all dependencies
make install-all

# 2. Set up ThetaData credentials (free account at thetadata.net)
make setup-creds

# 3. Start Theta Terminal and calibrate IV model
make theta-start
make calibrate-expanded

# 4. Train the ML exit model
make train-ml

# 5. Run your first backtest
make portfolio

# 6. Generate next week's trade plan
make plan
```

### Weekly Workflow

```bash
# Every Sunday/Monday:
make scan                # Market regime + signals + recommendations
make plan                # Concrete trade plan with strikes, contracts, risk

# After placing trades (Thursday):
# Positions auto-exit on gamma/delta breach, otherwise hold to expiry

# Monthly:
make theta-start         # Start terminal if not running
make calibrate-expanded  # Recalibrate IV ratios
make train-ml            # Retrain ML model on new data
make theta-stop          # Stop terminal when done
```

## Exit Strategy Comparison

The framework supports 5 exit presets. Backtested on $20k, 10% weekly risk, credit spreads:

| Exit Rules | Ending Capital | Annual Return | Max Drawdown | Sharpe |
|:-----------|--:|--:|--:|--:|
| `hold` (expiry) | $77,783 | +21.5% | **-44.7%** | 3.54 |
| **`greeks`** | **$32,433** | **+7.2%** | **-3.5%** | **2.82** |
| `theta` (ML) | $30,897 | +6.4% | -4.5% | 2.32 |
| `conservative` | ~$28,000 | ~+5% | ~-4% | ~2.5 |
| `aggressive` | ~$35,000 | ~+8% | ~-10% | ~2.0 |

**`greeks` is recommended** -- holds positions and lets theta work, only exits when gamma or delta breach thresholds. Best risk-adjusted returns with the smallest max drawdown (-3.5%).

`hold` makes the most money but has a -44.7% drawdown -- that's $9k gone on a $20k account at the worst point.

## Commands

### Scanner and Trade Plan
```bash
weeklyopt scan --expanded --credit-spreads              # Signal dashboard
weeklyopt scan --expanded --credit-spreads --detail     # Per-ticker breakdown
weeklyopt scan --expanded --credit-spreads --trade-plan --capital 20000 --risk-limit 0.10
```

The `--trade-plan` flag generates a concrete plan for next week:
- Exact strikes and spread widths
- Number of contracts per position (sized to your account)
- Total capital at risk vs budget
- Exit rules and warnings

### Portfolio Backtest
```bash
# Recommended config
weeklyopt portfolio \
  --capital 20000 \
  --risk-limit 0.10 \
  --max-contracts 10 \
  --credit-spreads \
  --expanded \
  --exit-rules greeks

# Conservative
weeklyopt portfolio --capital 20000 --risk-limit 0.05 --max-contracts 5 \
  --credit-spreads --expanded --exit-rules conservative

# Fixed weekly budget instead of dynamic
weeklyopt portfolio --budget 500 --max-contracts 5 --credit-spreads --expanded

# Show weekly allocation log
weeklyopt portfolio --capital 20000 --risk-limit 0.10 --max-contracts 10 \
  --credit-spreads --expanded --exit-rules greeks --weekly-log 20
```

### Single Strategy Backtest
```bash
weeklyopt backtest --strategy bull_put_spread --exit-rules greeks
weeklyopt backtest --strategy iron_condor --tickers SPY QQQ --start 2023-01-01
weeklyopt backtest --strategy cash_secured_put --delta 0.25 --exit-rules conservative --no-plot
```

### ThetaData Management
```bash
make theta-start         # Start terminal (auto-downloads jar if missing)
make theta-stop          # Stop terminal
make theta-status        # Check if running and API connected
make calibrate-expanded  # Calibrate all 23 tickers
```

### ML Exit Model
```bash
make train-ml            # Train on 2023-2024, test on 2025, save model
```

## Architecture

```
weeklyopt/
├── config.py                   # 23 tickers, correlation clusters, defaults
├── cli.py                      # CLI entry point
├── data/fetcher.py             # yfinance download + parquet caching
├── pricing/
│   ├── black_scholes.py        # BS pricing, Greeks, strike-from-delta
│   ├── volatility.py           # HV, calibrated IV estimation
│   └── calibration.py          # Per-ticker IV/HV ratios from ThetaData
├── strategies/
│   ├── base.py                 # Strategy ABC, OptionLeg, TradeResult
│   ├── covered_call.py         # Sell OTM call against shares
│   ├── cash_secured_put.py     # Sell OTM put backed by cash
│   ├── iron_condor.py          # Sell OTM put spread + call spread
│   ├── vertical_spread.py      # BullPutSpread, BearCallSpread (credit)
│   ├── debit_spreads.py        # BullCallSpread, BearPutSpread, LongCall, LongPut
│   ├── straddle.py             # Short straddle/strangle
│   └── managed_straddle.py     # Long straddle with leg management
├── engine/
│   ├── backtest.py             # Single-strategy backtest engine
│   ├── portfolio_backtest.py   # Scanner-driven portfolio (all enhancements)
│   ├── signals.py              # Per-ticker scoring (regime, IV, momentum, skew)
│   ├── fundamentals.py         # Earnings, OI, P/C ratio, max pain, short interest
│   ├── market_regime.py        # VIX, term structure, breadth, correlation
│   ├── exit_rules.py           # Profit targets, stop losses, Greeks stops
│   ├── filters.py              # IV rank gate, earnings, backwardation, clusters
│   ├── flow_signals.py         # Unusual options activity detection
│   ├── ml_exit.py              # Gradient boosted tree for exit timing
│   ├── rolling.py              # Multi-leg rolling for cost savings
│   └── straddle_manager.py     # Daily leg management for long straddles
├── analysis/
│   ├── metrics.py              # Sharpe, Sortino, PF, drawdown, win rate
│   └── plots.py                # Equity curve, P&L distribution, dashboard
├── validation/
│   ├── live_check.py           # BS vs live yfinance chain comparison
│   ├── thetadata_client.py     # ThetaData v3 REST API client
│   ├── historical_validate.py  # Replay trades against real historical prices
│   └── lumibot_runner.py       # Lumibot + ThetaData production backtest
├── optimize.py                 # Parameter grid search
└── train_ml_exit_v2.py         # ML exit model training
```

## How It Works

### Signal Pipeline (each week)

1. **Market Regime** -- VIX level/rank, term structure (contango vs backwardation), cross-correlation, breadth. Produces a position size modifier (50%-125%).

2. **Per-Ticker Signals** -- Directional regime, IV rank, RSI, momentum, calibrated put/call skew. Scores 11 strategies 0-100 per ticker.

3. **Fundamentals** -- Earnings proximity (skip pre-earnings, boost post-earnings), put/call OI ratio, unusual flow, short interest, max pain.

4. **Filters** -- IV rank >50 gate, VIX backwardation filter, correlation cluster avoidance (max 1 per cluster), dynamic delta by IV rank.

5. **Execution** -- Credit spread at dynamically selected delta, sized to risk budget.

6. **Exit Management** -- Greeks-only: hold until gamma >0.10 or delta >0.50 breach. ML model consulted as secondary signal.

7. **Rolling** -- Consecutive winners on same ticker get reduced transaction costs.

### Exit Rule Presets

| Preset | Profit Target | Stop Loss | Time Stop | Greeks | Best For |
|--------|:---:|:---:|:---:|:---:|---------|
| `hold` | -- | -- | -- | -- | Max return, can stomach -45% DD |
| `greeks` | -- | -- | -- | gamma/delta | **Recommended.** Best risk-adjusted |
| `theta` | 50% | 2x credit | Thursday | gamma/delta | ML-driven, very protective |
| `conservative` | 50% | 1.5x | Thursday | tighter | Minimal drawdown |
| `aggressive` | 75% | 3x | -- | -- | More room to run |

### Portfolio Flags

| Flag | Effect |
|------|--------|
| `--credit-spreads` | Bull put, bear call, iron condor only (capped risk) |
| `--defined-risk` | Debit strategies only (max loss = premium paid) |
| `--expanded` | 23-ticker universe including sector ETFs |
| `--risk-limit 0.10` | Dynamic budget = 10% of current capital per week |
| `--capital 20000` | Starting account size |
| `--exit-rules greeks` | Only exit on gamma/delta breach |
| `--trade-plan` | Generate concrete trade plan with strikes |

## Ticker Universe

**Equities (12):** SPY, QQQ, AAPL, MSFT, AMZN, TSLA, NVDA, META, GOOGL, AMD, JPM, BAC

**Sector ETFs (11):** XLF, IWM, GLD, TLT, XLE, XBI, KRE, COIN, SLV, EEM, SMH

Tickers are grouped into correlation clusters (mega_tech, semiconductors, financials, commodities, etc.) -- the portfolio holds max 1 position per cluster to prevent correlated losses.

## IV Calibration

Per-ticker IV/HV ratios from ThetaData. Richest put premium (best for selling):

| Ticker | Put IV/HV | Notes |
|--------|:---------:|-------|
| KRE | 2.48x | Regional bank fear premium |
| XLF | 1.79x | Financial sector skew |
| BAC | 1.75x | Bank puts very rich |
| SLV | 1.58x | Commodity vol |
| MSFT | 1.58x | Mega-cap put demand |
| AAPL | 1.54x | |
| NVDA | 1.47x | Semiconductor vol |

Avoid selling premium on: META (0.77x), EEM (1.00x).

## ThetaData Setup

Required for IV calibration and historical validation. Not needed for backtesting.

```bash
# Full automated setup
make install-all      # Installs Java 21 + downloads jar + Python deps
make setup-creds      # Interactive prompt for ThetaData email/password
make theta-start      # Starts terminal, writes PID file, verifies connection
make theta-status     # Check if running
make theta-stop       # Stop terminal, clean PID
```

Or manually:
```bash
# 1. Get a free account at thetadata.net
# 2. Create creds file
echo "your@email.com" > creds.txt
echo "your_password" >> creds.txt
chmod 600 creds.txt

# 3. Start
make theta-start
```

Free tier: 1 year EOD options data, 20 requests/min.

## Dependencies

**Required:** `yfinance numpy pandas scipy matplotlib seaborn httpx`

**Optional:** `scikit-learn` (ML exit model), `lumibot thetadata` (production validation)

**System:** Java 21+ (for Theta Terminal only)

```bash
make install-all    # installs everything
```
