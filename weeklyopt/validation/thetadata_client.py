"""ThetaData REST API client for historical options data.

Supports v3 (port 25503, default) and v2 (port 25510) terminals.
Free tier: 1 year EOD data, 20 req/min rate limit.

Requires Theta Terminal running locally:
  v3: https://download-unstable.thetadata.us/ThetaTerminalv3.jar
"""

import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import httpx
import numpy as np
import pandas as pd


V3_BASE = "http://127.0.0.1:25503/v3"
V2_BASE = "http://127.0.0.1:25510/v2"

# Rate limit: 20 requests/min on free tier
RATE_LIMIT_DELAY = 3.1  # seconds between requests


def _date_fmt(d: date) -> str:
    """Format date as YYYY-MM-DD for v3 API."""
    return d.strftime("%Y-%m-%d")


@dataclass
class ThetaDataClient:
    """Client for ThetaData REST API (v3 by default).

    The Theta Terminal must be running locally.
    v3 terminal: java -jar ThetaTerminalv3.jar
    """
    base_url: str = V3_BASE
    rate_limit: bool = True  # set False if on paid tier
    timeout: int = 30
    _last_request: float = 0.0

    def _throttle(self):
        if self.rate_limit:
            elapsed = time.time() - self._last_request
            if elapsed < RATE_LIMIT_DELAY:
                time.sleep(RATE_LIMIT_DELAY - elapsed)
        self._last_request = time.time()

    def _get_json(self, endpoint: str, params: dict) -> list[dict]:
        """Make a GET request, return parsed JSON response."""
        self._throttle()
        url = f"{self.base_url}{endpoint}"
        params["format"] = "json"

        resp = httpx.get(url, params=params, timeout=self.timeout)

        # 472 = ThetaData "no data" (contract doesn't exist at that strike/exp)
        if resp.status_code == 472:
            return []
        resp.raise_for_status()

        data = resp.json()

        # v3 returns list of objects directly, or {"response": [...]} in v2
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "response" in data:
            return data["response"]
        return []

    def get_option_eod(
        self,
        symbol: str,
        expiration: date,
        strike: float,
        right: str = "both",  # "call", "put", or "both"
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """Fetch historical EOD option data (OHLC + bid/ask at close).

        Returns DataFrame with columns: date, expiration, strike, right,
        open, high, low, close, volume, bid, ask, mid
        """
        params = {
            "symbol": symbol,
            "expiration": _date_fmt(expiration),
            "strike": f"{strike:.2f}",
            "right": right,
            "start_date": _date_fmt(start_date or expiration - timedelta(days=30)),
            "end_date": _date_fmt(end_date or expiration),
        }

        rows = self._get_json("/option/history/eod", params)

        if not rows:
            return pd.DataFrame()

        # v3 returns nested: [{contract: {...}, data: [{day1}, {day2}]}, ...]
        # Flatten into one row per day with contract info attached
        flat_rows = []
        for item in rows:
            if isinstance(item, dict) and "data" in item and "contract" in item:
                contract = item["contract"]
                for day in item["data"]:
                    row = {**day}
                    row["symbol"] = contract.get("symbol", symbol)
                    row["strike"] = contract.get("strike", strike)
                    row["right"] = contract.get("right", right)
                    row["expiration"] = contract.get("expiration", "")
                    flat_rows.append(row)
            elif isinstance(item, dict):
                # Flat format (v2 or single-level v3)
                flat_rows.append(item)

        if not flat_rows:
            return pd.DataFrame()

        df = pd.DataFrame(flat_rows)

        # Normalize column names
        col_map = {col: col.lower().replace(" ", "_") for col in df.columns}
        df = df.rename(columns=col_map)

        # Compute mid price
        if "bid" in df.columns and "ask" in df.columns:
            df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
            df["ask"] = pd.to_numeric(df["ask"], errors="coerce")
            df["mid"] = (df["bid"] + df["ask"]) / 2

        # Parse dates
        for date_col in ["created", "last_trade"]:
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        if "expiration" in df.columns:
            df["expiration"] = pd.to_datetime(df["expiration"], errors="coerce")

        return df

    def get_option_eod_chain(
        self,
        symbol: str,
        expiration: date,
        start_date: date | None = None,
        end_date: date | None = None,
        right: str = "both",
        strike_range: int | None = 10,
    ) -> pd.DataFrame:
        """Fetch full EOD chain for an expiration (all strikes).

        strike_range: number of strikes above/below spot to include.
        """
        params = {
            "symbol": symbol,
            "expiration": _date_fmt(expiration),
            "strike": "*",  # all strikes
            "right": right,
            "start_date": _date_fmt(start_date or expiration - timedelta(days=7)),
            "end_date": _date_fmt(end_date or expiration),
        }
        if strike_range is not None:
            params["strike_range"] = str(strike_range)

        rows = self._get_json("/option/history/eod", params)

        if not rows:
            return pd.DataFrame()

        # Flatten nested v3 format
        flat_rows = []
        for item in rows:
            if isinstance(item, dict) and "data" in item and "contract" in item:
                contract = item["contract"]
                for day in item["data"]:
                    row = {**day}
                    row["symbol"] = contract.get("symbol", symbol)
                    row["strike"] = contract.get("strike")
                    row["right"] = contract.get("right")
                    row["expiration"] = contract.get("expiration", "")
                    flat_rows.append(row)
            elif isinstance(item, dict):
                flat_rows.append(item)

        if not flat_rows:
            return pd.DataFrame()

        df = pd.DataFrame(flat_rows)
        col_map = {col: col.lower().replace(" ", "_") for col in df.columns}
        df = df.rename(columns=col_map)

        if "bid" in df.columns and "ask" in df.columns:
            df["mid"] = (df["bid"].astype(float) + df["ask"].astype(float)) / 2

        for c in ["strike", "open", "high", "low", "close", "bid", "ask", "mid"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)

        return df

    def get_option_quote_history(
        self,
        symbol: str,
        expiration: date,
        strike: float,
        right: str = "call",
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """Fetch historical intraday quotes for a specific contract."""
        params = {
            "symbol": symbol,
            "expiration": _date_fmt(expiration),
            "strike": f"{strike:.2f}",
            "right": right,
            "start_date": _date_fmt(start_date or expiration - timedelta(days=7)),
            "end_date": _date_fmt(end_date or expiration),
        }

        rows = self._get_json("/option/history/quote", params)
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        col_map = {col: col.lower().replace(" ", "_") for col in df.columns}
        df = df.rename(columns=col_map)
        return df

    def get_option_greeks_eod(
        self,
        symbol: str,
        expiration: date,
        strike: float | str = "*",
        right: str = "both",
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """Fetch historical EOD Greeks (delta, gamma, theta, vega, IV)."""
        strike_str = f"{strike:.2f}" if isinstance(strike, (int, float)) else strike
        params = {
            "symbol": symbol,
            "expiration": _date_fmt(expiration),
            "strike": strike_str,
            "right": right,
            "start_date": _date_fmt(start_date or expiration - timedelta(days=30)),
            "end_date": _date_fmt(end_date or expiration),
        }

        rows = self._get_json("/option/history/greeks/eod", params)
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        col_map = {col: col.lower().replace(" ", "_") for col in df.columns}
        df = df.rename(columns=col_map)
        return df

    def list_expirations(self, symbol: str) -> list[date]:
        """Get available option expirations for a symbol."""
        rows = self._get_json("/option/list/expirations", {"symbol": symbol})

        expirations = []
        for item in rows:
            if isinstance(item, dict) and "expiration" in item:
                expirations.append(pd.to_datetime(item["expiration"]).date())
            elif isinstance(item, str):
                expirations.append(pd.to_datetime(item).date())
            elif isinstance(item, (int, float)):
                expirations.append(datetime.strptime(str(int(item)), "%Y%m%d").date())
        return sorted(expirations)

    def list_strikes(self, symbol: str, expiration: date) -> list[float]:
        """Get available strikes for a given expiration."""
        rows = self._get_json(
            "/option/list/strikes",
            {"symbol": symbol, "expiration": _date_fmt(expiration)},
        )

        strikes = []
        for item in rows:
            if isinstance(item, dict) and "strike" in item:
                strikes.append(float(item["strike"]))
            elif isinstance(item, (int, float)):
                strikes.append(float(item))
        return sorted(strikes)

    def list_symbols(self) -> list[str]:
        """List all available stock symbols."""
        rows = self._get_json("/stock/list/symbols", {})
        symbols = []
        for item in rows:
            if isinstance(item, dict) and "symbol" in item:
                symbols.append(item["symbol"])
            elif isinstance(item, str):
                symbols.append(item)
        return sorted(symbols)

    def check_connection(self) -> bool:
        """Check if Theta Terminal is running."""
        try:
            resp = httpx.get(
                f"{self.base_url}/stock/list/symbols",
                params={"format": "json"},
                timeout=5,
            )
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False
