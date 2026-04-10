"""Volatility estimation from historical price data."""

import numpy as np
import pandas as pd

from ..config import HV_WINDOW, IV_MARKUP


def historical_vol(prices: pd.Series, window: int = HV_WINDOW) -> pd.Series:
    """Compute annualized historical volatility from close prices.

    Uses log returns with a rolling window.
    """
    log_returns = np.log(prices / prices.shift(1))
    hv = log_returns.rolling(window=window).std() * np.sqrt(252)
    return hv


def implied_vol_estimate(
    prices: pd.Series,
    window: int = HV_WINDOW,
    markup: float = IV_MARKUP,
) -> pd.Series:
    """Estimate implied volatility as marked-up historical vol.

    Options typically trade at a premium to realized vol.
    The markup factor approximates this spread.
    """
    hv = historical_vol(prices, window)
    return hv * markup


def calibrated_vol_estimate(
    prices: pd.Series,
    calibration,  # TickerCalibration
    moneyness: float = 1.0,
    option_type=None,
    window: int = HV_WINDOW,
) -> pd.Series:
    """Estimate IV using per-ticker calibration.

    Uses the calibrated IV/HV ratio at the given moneyness level
    instead of a flat markup.
    """
    from .black_scholes import OptionType as OT

    hv = historical_vol(prices, window)

    if calibration is None or calibration.sample_dates == 0:
        # Fallback to flat markup
        return hv * IV_MARKUP

    # Get the appropriate ratio from calibration
    ratio = calibration.iv_for_moneyness(moneyness, option_type or OT.CALL) / calibration.historical_vol
    if ratio <= 0 or np.isnan(ratio):
        ratio = IV_MARKUP

    return hv * ratio


def vol_on_date(vol_series: pd.Series, dt: pd.Timestamp) -> float:
    """Get volatility for a specific date, falling back to nearest prior."""
    if dt in vol_series.index:
        v = vol_series.loc[dt]
    else:
        mask = vol_series.index <= dt
        if mask.any():
            v = vol_series.loc[mask].iloc[-1]
        else:
            v = vol_series.dropna().iloc[0] if not vol_series.dropna().empty else 0.20
    return float(v) if not np.isnan(v) else 0.20
