"""IV calibration using real market data from ThetaData.

Instead of a flat IV markup, we derive per-ticker calibration that captures:
1. IV/HV ratio — how much the market marks up over realized vol (varies by ticker)
2. Skew — OTM puts trade richer than OTM calls (varies by ticker)
3. Term structure — not modeled here (weekly-only so DTE is ~constant)

The calibration fetches recent ATM and OTM option prices from ThetaData,
backs out implied vol from market mid prices, and compares to our HV estimate.
This gives us a realistic per-ticker, per-moneyness IV surface.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .black_scholes import bs_price, OptionType
from .volatility import historical_vol


CALIBRATION_DIR = Path.home() / ".weeklyopt_cache" / "calibration"


@dataclass
class TickerCalibration:
    """Calibrated IV parameters for a single ticker."""
    ticker: str
    # IV/HV ratio at different moneyness levels
    atm_iv_ratio: float = 1.0       # ATM: IV / HV
    otm_put_iv_ratio: float = 1.0   # ~20-30 delta put: IV / HV
    otm_call_iv_ratio: float = 1.0  # ~20-30 delta call: IV / HV
    # Raw IV levels observed
    atm_iv: float = 0.0
    otm_put_iv: float = 0.0
    otm_call_iv: float = 0.0
    historical_vol: float = 0.0
    # Skew: how much richer puts trade vs calls (ratio)
    put_call_skew: float = 1.0
    # Data quality
    sample_dates: int = 0
    calibration_date: str = ""

    def iv_for_moneyness(self, moneyness: float, option_type: OptionType) -> float:
        """Get calibrated IV for a given moneyness level.

        moneyness: strike/spot (1.0 = ATM, 0.97 = 3% OTM put, 1.03 = 3% OTM call)
        Returns: annualized IV estimate.
        """
        hv = self.historical_vol
        if hv <= 0:
            hv = 0.20  # fallback

        if abs(moneyness - 1.0) < 0.01:
            # ATM
            return hv * self.atm_iv_ratio
        elif moneyness < 1.0:
            # OTM put side — interpolate between ATM and OTM put
            otm_degree = min((1.0 - moneyness) / 0.05, 1.0)  # normalize to ~5% OTM
            ratio = self.atm_iv_ratio + otm_degree * (self.otm_put_iv_ratio - self.atm_iv_ratio)
            return hv * ratio
        else:
            # OTM call side
            otm_degree = min((moneyness - 1.0) / 0.05, 1.0)
            ratio = self.atm_iv_ratio + otm_degree * (self.otm_call_iv_ratio - self.atm_iv_ratio)
            return hv * ratio


@dataclass
class IVCalibrator:
    """Calibrate IV model against real ThetaData option prices."""

    # Injected at runtime to avoid circular import
    _client: object = None

    def _get_client(self):
        if self._client is None:
            from ..validation.thetadata_client import ThetaDataClient
            self._client = ThetaDataClient()
        return self._client

    def implied_vol_from_price(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: OptionType,
    ) -> float | None:
        """Back out implied volatility from a market price using Brent's method."""
        if market_price <= 0 or T <= 0:
            return None

        # Intrinsic value check
        if option_type == OptionType.CALL:
            intrinsic = max(S - K, 0)
        else:
            intrinsic = max(K - S, 0)

        if market_price < intrinsic:
            return None

        def objective(sigma):
            return bs_price(S, K, T, r, sigma, option_type) - market_price

        try:
            iv = brentq(objective, 0.01, 5.0, xtol=1e-6, maxiter=100)
            return iv
        except (ValueError, RuntimeError):
            return None

    def calibrate_ticker(
        self,
        ticker: str,
        lookback_weeks: int = 12,
        risk_free_rate: float = 0.045,
        hv_window: int = 20,
    ) -> TickerCalibration:
        """Calibrate IV parameters for a ticker using recent weekly option chains.

        Fetches the last N weeks of Friday expirations, gets EOD chain data,
        backs out IV at various strikes, and compares to historical vol.
        """
        client = self._get_client()

        # Get recent expirations
        expirations = client.list_expirations(ticker)
        today = date.today()
        cutoff = today - timedelta(weeks=lookback_weeks)

        # Filter to recent past Friday expirations
        recent_exps = [
            e for e in expirations
            if cutoff <= e <= today and e.weekday() == 4  # Fridays
        ]

        if not recent_exps:
            # Fallback: any recent expirations
            recent_exps = [e for e in expirations if cutoff <= e <= today][:lookback_weeks]

        if not recent_exps:
            print(f"  No recent expirations found for {ticker}")
            return TickerCalibration(ticker=ticker, calibration_date=str(today))

        # Fetch underlying price history for HV calculation
        import yfinance as yf
        hist = yf.download(ticker, start=str(cutoff - timedelta(days=60)), end=str(today), progress=False)
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)

        if hist.empty:
            print(f"  No price history for {ticker}")
            return TickerCalibration(ticker=ticker, calibration_date=str(today))

        hv_series = historical_vol(hist["Close"], hv_window)
        current_hv = float(hv_series.dropna().iloc[-1]) if not hv_series.dropna().empty else 0.20

        # Collect IV observations across expirations
        atm_ivs = []
        otm_put_ivs = []
        otm_call_ivs = []

        for exp in recent_exps[-8:]:  # last 8 weeks max (rate limit friendly)
            # Entry date = Monday of that week
            entry_date = exp - timedelta(days=4)
            if entry_date.weekday() != 0:
                entry_date = exp - timedelta(days=exp.weekday())

            # Get spot price on entry date
            spot_candidates = hist.index[hist.index <= str(entry_date)]
            if spot_candidates.empty:
                continue
            spot = float(hist.loc[spot_candidates[-1], "Close"])

            # DTE in years
            dte_days = (exp - entry_date).days
            if dte_days <= 0:
                continue
            T = dte_days / 365

            # Fetch chain for entry date
            try:
                chain = client.get_option_eod_chain(
                    ticker, exp,
                    start_date=entry_date,
                    end_date=entry_date,
                    right="both",
                    strike_range=10,
                )
            except Exception:
                continue

            if chain.empty or "mid" not in chain.columns or "strike" not in chain.columns:
                continue

            # Filter to entry date only
            if "created" in chain.columns:
                chain["created"] = pd.to_datetime(chain["created"], errors="coerce")
                chain["_date"] = chain["created"].dt.date
                chain = chain[chain["_date"] == entry_date]

            if chain.empty:
                continue

            chain["strike"] = pd.to_numeric(chain["strike"], errors="coerce")
            chain["mid"] = pd.to_numeric(chain["mid"], errors="coerce")
            chain = chain.dropna(subset=["strike", "mid"])
            chain = chain[chain["mid"] > 0.05]  # filter dust

            if "right" not in chain.columns:
                continue

            calls = chain[chain["right"].str.upper() == "CALL"]
            puts = chain[chain["right"].str.upper() == "PUT"]

            # ATM: nearest strike to spot
            for df_slice, opt_type in [(calls, OptionType.CALL), (puts, OptionType.PUT)]:
                if df_slice.empty:
                    continue
                df_slice = df_slice.copy()
                df_slice["dist_atm"] = abs(df_slice["strike"] - spot)
                atm_row = df_slice.loc[df_slice["dist_atm"].idxmin()]

                iv = self.implied_vol_from_price(
                    float(atm_row["mid"]), spot, float(atm_row["strike"]),
                    T, risk_free_rate, opt_type,
                )
                if iv and 0.05 < iv < 3.0:
                    atm_ivs.append(iv)

            # OTM puts: ~3-5% below spot
            otm_put_target = spot * 0.97
            if not puts.empty:
                puts_c = puts.copy()
                puts_c["dist"] = abs(puts_c["strike"] - otm_put_target)
                otm_put_row = puts_c.loc[puts_c["dist"].idxmin()]
                iv = self.implied_vol_from_price(
                    float(otm_put_row["mid"]), spot, float(otm_put_row["strike"]),
                    T, risk_free_rate, OptionType.PUT,
                )
                if iv and 0.05 < iv < 3.0:
                    otm_put_ivs.append(iv)

            # OTM calls: ~3-5% above spot
            otm_call_target = spot * 1.03
            if not calls.empty:
                calls_c = calls.copy()
                calls_c["dist"] = abs(calls_c["strike"] - otm_call_target)
                otm_call_row = calls_c.loc[calls_c["dist"].idxmin()]
                iv = self.implied_vol_from_price(
                    float(otm_call_row["mid"]), spot, float(otm_call_row["strike"]),
                    T, risk_free_rate, OptionType.CALL,
                )
                if iv and 0.05 < iv < 3.0:
                    otm_call_ivs.append(iv)

        # Compute ratios
        atm_iv = float(np.median(atm_ivs)) if atm_ivs else current_hv
        otm_put_iv = float(np.median(otm_put_ivs)) if otm_put_ivs else atm_iv
        otm_call_iv = float(np.median(otm_call_ivs)) if otm_call_ivs else atm_iv

        atm_ratio = atm_iv / current_hv if current_hv > 0 else 1.0
        put_ratio = otm_put_iv / current_hv if current_hv > 0 else 1.0
        call_ratio = otm_call_iv / current_hv if current_hv > 0 else 1.0
        skew = otm_put_iv / otm_call_iv if otm_call_iv > 0 else 1.0

        cal = TickerCalibration(
            ticker=ticker,
            atm_iv_ratio=round(atm_ratio, 4),
            otm_put_iv_ratio=round(put_ratio, 4),
            otm_call_iv_ratio=round(call_ratio, 4),
            atm_iv=round(atm_iv, 4),
            otm_put_iv=round(otm_put_iv, 4),
            otm_call_iv=round(otm_call_iv, 4),
            historical_vol=round(current_hv, 4),
            put_call_skew=round(skew, 4),
            sample_dates=len(atm_ivs),
            calibration_date=str(today),
        )

        return cal

    def calibrate_all(
        self,
        tickers: list[str],
        lookback_weeks: int = 12,
        verbose: bool = True,
    ) -> dict[str, TickerCalibration]:
        """Calibrate all tickers and save results."""
        results = {}

        for i, ticker in enumerate(tickers):
            if verbose:
                print(f"  [{i+1}/{len(tickers)}] Calibrating {ticker}...")

            cal = self.calibrate_ticker(ticker, lookback_weeks)
            results[ticker] = cal

            if verbose and cal.sample_dates > 0:
                print(
                    f"    HV={cal.historical_vol:.1%}  ATM IV={cal.atm_iv:.1%} "
                    f"(ratio={cal.atm_iv_ratio:.2f})  "
                    f"OTM Put IV={cal.otm_put_iv:.1%} ({cal.otm_put_iv_ratio:.2f})  "
                    f"OTM Call IV={cal.otm_call_iv:.1%} ({cal.otm_call_iv_ratio:.2f})  "
                    f"Skew={cal.put_call_skew:.2f}  "
                    f"Samples={cal.sample_dates}"
                )

        # Save to disk
        self.save_calibration(results)

        if verbose:
            self._print_summary(results)

        return results

    def save_calibration(self, calibrations: dict[str, TickerCalibration]) -> Path:
        """Save calibration to JSON file."""
        CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
        path = CALIBRATION_DIR / "iv_calibration.json"

        data = {}
        for ticker, cal in calibrations.items():
            data[ticker] = {
                "atm_iv_ratio": cal.atm_iv_ratio,
                "otm_put_iv_ratio": cal.otm_put_iv_ratio,
                "otm_call_iv_ratio": cal.otm_call_iv_ratio,
                "atm_iv": cal.atm_iv,
                "otm_put_iv": cal.otm_put_iv,
                "otm_call_iv": cal.otm_call_iv,
                "historical_vol": cal.historical_vol,
                "put_call_skew": cal.put_call_skew,
                "sample_dates": cal.sample_dates,
                "calibration_date": cal.calibration_date,
            }

        path.write_text(json.dumps(data, indent=2))
        return path

    @staticmethod
    def load_calibration() -> dict[str, TickerCalibration] | None:
        """Load saved calibration from disk."""
        path = CALIBRATION_DIR / "iv_calibration.json"
        if not path.exists():
            return None

        data = json.loads(path.read_text())
        result = {}
        for ticker, vals in data.items():
            result[ticker] = TickerCalibration(ticker=ticker, **vals)
        return result

    def _print_summary(self, calibrations: dict[str, TickerCalibration]) -> None:
        """Print calibration summary table."""
        print(f"\n{'='*90}")
        print(f"  IV Calibration Summary")
        print(f"{'='*90}")
        print(f"  {'Ticker':>6}  {'HV':>6}  {'ATM IV':>7}  {'ATM Ratio':>9}  "
              f"{'Put IV':>7}  {'Put Ratio':>9}  {'Call IV':>8}  {'Call Ratio':>10}  {'Skew':>5}")
        print(f"  {'-'*82}")

        for ticker in sorted(calibrations.keys()):
            c = calibrations[ticker]
            if c.sample_dates == 0:
                print(f"  {ticker:>6}  {'---':>6}  {'no data':>7}")
                continue
            print(
                f"  {ticker:>6}  {c.historical_vol:>5.1%}  {c.atm_iv:>6.1%}  "
                f"{c.atm_iv_ratio:>9.2f}  {c.otm_put_iv:>6.1%}  "
                f"{c.otm_put_iv_ratio:>9.2f}  {c.otm_call_iv:>7.1%}  "
                f"{c.otm_call_iv_ratio:>10.2f}  {c.put_call_skew:>5.2f}"
            )

        print(f"\n  Key insight:")
        ratios = [c.atm_iv_ratio for c in calibrations.values() if c.sample_dates > 0]
        if ratios:
            print(f"  - ATM IV/HV ratio range: {min(ratios):.2f} to {max(ratios):.2f}")
            print(f"  - Your old flat markup was 1.15x — actual market varies per ticker")
        skews = [c.put_call_skew for c in calibrations.values() if c.sample_dates > 0]
        if skews:
            print(f"  - Put/Call skew range: {min(skews):.2f} to {max(skews):.2f}")
            print(f"    (>1.0 = puts trade richer than calls)")
        print(f"{'='*90}")
