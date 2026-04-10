"""Run strategies through Lumibot with ThetaData for full production-grade backtesting.

This is the gold-standard validation: real historical bid/ask data,
proper fill simulation, and Lumibot's broker emulation.
"""

from datetime import datetime, date
from dataclasses import dataclass

try:
    from lumibot.backtesting import ThetaDataBacktesting
    from lumibot.strategies import Strategy as LumiStrategy
    HAS_LUMIBOT = True
except ImportError:
    HAS_LUMIBOT = False


def _check_lumibot():
    if not HAS_LUMIBOT:
        raise ImportError(
            "Lumibot is not installed. Install with:\n"
            "  pip install 'weeklyopt[validate]'\n"
            "Or: pip install lumibot thetadata"
        )


# ─── Iron Condor Strategy for Lumibot ──────────────────────────────────────

if HAS_LUMIBOT:

    class LumiIronCondor(LumiStrategy):
        """Weekly iron condor executed through Lumibot with real ThetaData pricing."""

        parameters = {
            "symbol": "SPY",
            "short_delta_target": 0.20,
            "wing_width": 5,
            "allocation_pct": 0.10,
        }

        def initialize(self):
            self.sleeptime = "1D"
            self._traded_this_week = False
            self._last_trade_week = -1

        def on_trading_iteration(self):
            dt = self.get_datetime()
            week_num = dt.isocalendar()[1]

            # Trade on Monday (or first day of week)
            if dt.weekday() == 0 and week_num != self._last_trade_week:
                self._open_iron_condor(dt)
                self._last_trade_week = week_num

        def _open_iron_condor(self, dt):
            symbol = self.parameters["symbol"]
            price = self.get_last_price(symbol)
            if price is None:
                return

            wing = self.parameters["wing_width"]

            # Find nearest Friday expiration
            days_to_friday = (4 - dt.weekday()) % 7
            if days_to_friday == 0:
                days_to_friday = 7
            expiry = dt + __import__("datetime").timedelta(days=days_to_friday)
            expiry_str = expiry.strftime("%Y-%m-%d")

            # Approximate strikes from delta targets
            # ~0.20 delta ≈ 1 standard deviation move for the week
            # Rough: short strikes ~2-3% OTM for weeklies
            otm_pct = 0.025  # ~2.5% OTM
            short_put_strike = round((price * (1 - otm_pct)) * 2) / 2
            long_put_strike = short_put_strike - wing
            short_call_strike = round((price * (1 + otm_pct)) * 2) / 2
            long_call_strike = short_call_strike + wing

            alloc = self.portfolio_value * self.parameters["allocation_pct"]
            max_loss_per = wing * 100  # rough max loss per contract
            contracts = max(1, int(alloc / max_loss_per))

            try:
                # Sell put spread
                sell_put = self.create_order(
                    self.create_asset(symbol, asset_type="option",
                                      expiration=expiry, strike=short_put_strike, right="put"),
                    quantity=contracts,
                    side="sell",
                )
                buy_put = self.create_order(
                    self.create_asset(symbol, asset_type="option",
                                      expiration=expiry, strike=long_put_strike, right="put"),
                    quantity=contracts,
                    side="buy",
                )
                # Sell call spread
                sell_call = self.create_order(
                    self.create_asset(symbol, asset_type="option",
                                      expiration=expiry, strike=short_call_strike, right="call"),
                    quantity=contracts,
                    side="sell",
                )
                buy_call = self.create_order(
                    self.create_asset(symbol, asset_type="option",
                                      expiration=expiry, strike=long_call_strike, right="call"),
                    quantity=contracts,
                    side="buy",
                )

                self.submit_order(sell_put)
                self.submit_order(buy_put)
                self.submit_order(sell_call)
                self.submit_order(buy_call)

            except Exception as e:
                self.log_message(f"Failed to open iron condor: {e}")

    class LumiCoveredCall(LumiStrategy):
        """Weekly covered call through Lumibot."""

        parameters = {
            "symbol": "SPY",
            "otm_pct": 0.02,
            "allocation_pct": 0.15,
        }

        def initialize(self):
            self.sleeptime = "1D"
            self._last_trade_week = -1

        def on_trading_iteration(self):
            dt = self.get_datetime()
            week_num = dt.isocalendar()[1]

            if dt.weekday() == 0 and week_num != self._last_trade_week:
                self._open_covered_call(dt)
                self._last_trade_week = week_num

        def _open_covered_call(self, dt):
            symbol = self.parameters["symbol"]
            price = self.get_last_price(symbol)
            if price is None:
                return

            days_to_friday = (4 - dt.weekday()) % 7
            if days_to_friday == 0:
                days_to_friday = 7
            expiry = dt + __import__("datetime").timedelta(days=days_to_friday)

            call_strike = round((price * (1 + self.parameters["otm_pct"])) * 2) / 2

            alloc = self.portfolio_value * self.parameters["allocation_pct"]
            shares = int(alloc / price / 100) * 100  # round to 100s
            contracts = shares // 100

            if contracts < 1:
                return

            try:
                # Buy underlying
                stock_order = self.create_order(symbol, quantity=shares, side="buy")
                self.submit_order(stock_order)

                # Sell call
                call_order = self.create_order(
                    self.create_asset(symbol, asset_type="option",
                                      expiration=expiry, strike=call_strike, right="call"),
                    quantity=contracts,
                    side="sell",
                )
                self.submit_order(call_order)
            except Exception as e:
                self.log_message(f"Failed to open covered call: {e}")

    class LumiCashSecuredPut(LumiStrategy):
        """Weekly cash-secured put through Lumibot."""

        parameters = {
            "symbol": "SPY",
            "otm_pct": 0.03,
            "allocation_pct": 0.15,
        }

        def initialize(self):
            self.sleeptime = "1D"
            self._last_trade_week = -1

        def on_trading_iteration(self):
            dt = self.get_datetime()
            week_num = dt.isocalendar()[1]

            if dt.weekday() == 0 and week_num != self._last_trade_week:
                self._open_csp(dt)
                self._last_trade_week = week_num

        def _open_csp(self, dt):
            symbol = self.parameters["symbol"]
            price = self.get_last_price(symbol)
            if price is None:
                return

            days_to_friday = (4 - dt.weekday()) % 7
            if days_to_friday == 0:
                days_to_friday = 7
            expiry = dt + __import__("datetime").timedelta(days=days_to_friday)

            put_strike = round((price * (1 - self.parameters["otm_pct"])) * 2) / 2

            alloc = self.portfolio_value * self.parameters["allocation_pct"]
            contracts = max(1, int(alloc / (put_strike * 100)))

            try:
                put_order = self.create_order(
                    self.create_asset(symbol, asset_type="option",
                                      expiration=expiry, strike=put_strike, right="put"),
                    quantity=contracts,
                    side="sell",
                )
                self.submit_order(put_order)
            except Exception as e:
                self.log_message(f"Failed to open CSP: {e}")


@dataclass
class LumibotRunner:
    """Run strategies through Lumibot + ThetaData for production backtesting."""

    STRATEGIES = {}

    def __init__(self):
        _check_lumibot()
        self.STRATEGIES = {
            "iron_condor": LumiIronCondor,
            "covered_call": LumiCoveredCall,
            "cash_secured_put": LumiCashSecuredPut,
        }

    def run(
        self,
        strategy_name: str,
        symbol: str = "SPY",
        start: date = date(2024, 1, 1),
        end: date = date(2025, 1, 1),
        initial_capital: float = 100_000,
        **strategy_params,
    ):
        """Run a strategy through Lumibot with ThetaData.

        Returns Lumibot's result object with full stats.
        """
        if strategy_name not in self.STRATEGIES:
            available = ", ".join(self.STRATEGIES.keys())
            raise ValueError(f"Unknown strategy: {strategy_name}. Available: {available}")

        strategy_cls = self.STRATEGIES[strategy_name]
        params = {"symbol": symbol}
        params.update(strategy_params)

        start_dt = datetime(start.year, start.month, start.day)
        end_dt = datetime(end.year, end.month, end.day)

        print(f"Running {strategy_name} on {symbol} via Lumibot + ThetaData")
        print(f"Period: {start} to {end}")
        print(f"Capital: ${initial_capital:,.0f}")
        print("(This uses real historical option bid/ask data)\n")

        result = strategy_cls.run_backtest(
            ThetaDataBacktesting,
            start_dt,
            end_dt,
            benchmark_asset=symbol,
            budget=initial_capital,
            parameters=params,
        )

        return result
