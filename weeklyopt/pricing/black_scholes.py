"""Black-Scholes option pricing and Greeks."""

from enum import Enum
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


class OptionType(Enum):
    CALL = "call"
    PUT = "put"


@dataclass
class Greeks:
    delta: float
    gamma: float
    theta: float  # per day
    vega: float
    rho: float


def _d1d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    """Compute d1 and d2 for Black-Scholes."""
    if T <= 0 or sigma <= 0:
        return 0.0, 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = OptionType.CALL,
) -> float:
    """Black-Scholes option price.

    Args:
        S: Underlying price
        K: Strike price
        T: Time to expiry in years (e.g., 5/252 for 5 trading days)
        r: Risk-free rate (annualized)
        sigma: Implied volatility (annualized)
        option_type: CALL or PUT
    """
    if T <= 0:
        # At expiry: intrinsic value
        if option_type == OptionType.CALL:
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    d1, d2 = _d1d2(S, K, T, r, sigma)

    if option_type == OptionType.CALL:
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = OptionType.CALL,
) -> Greeks:
    """Compute all major Greeks."""
    if T <= 0 or sigma <= 0:
        intrinsic_delta = 1.0 if (S > K and option_type == OptionType.CALL) else (
            -1.0 if (S < K and option_type == OptionType.PUT) else 0.0
        )
        return Greeks(delta=intrinsic_delta, gamma=0, theta=0, vega=0, rho=0)

    d1, d2 = _d1d2(S, K, T, r, sigma)
    sqrt_T = np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    discount = np.exp(-r * T)

    if option_type == OptionType.CALL:
        delta = norm.cdf(d1)
        rho = K * T * discount * norm.cdf(d2) / 100
    else:
        delta = norm.cdf(d1) - 1.0
        rho = -K * T * discount * norm.cdf(-d2) / 100

    gamma = pdf_d1 / (S * sigma * sqrt_T)
    vega = S * pdf_d1 * sqrt_T / 100  # per 1% vol change
    theta = (
        -(S * pdf_d1 * sigma) / (2 * sqrt_T)
        - r * K * discount * (norm.cdf(d2) if option_type == OptionType.CALL else norm.cdf(-d2))
    ) / 252  # per trading day

    return Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)


def strike_from_delta(
    S: float,
    T: float,
    r: float,
    sigma: float,
    target_delta: float,
    option_type: OptionType = OptionType.CALL,
) -> float:
    """Find the strike price that corresponds to a target delta.

    Uses Newton's method to invert the BS delta formula.
    """
    if option_type == OptionType.PUT:
        target_delta = abs(target_delta)
        # Put delta = Call delta - 1, so call_delta = 1 - |put_delta|
        target_call_delta = 1.0 - target_delta
    else:
        target_call_delta = target_delta

    # Initial guess: ATM
    K = S
    for _ in range(50):
        d1, _ = _d1d2(S, K, T, r, sigma)
        current_delta = norm.cdf(d1)
        # derivative of delta w.r.t. K
        d_delta_dK = -norm.pdf(d1) / (K * sigma * np.sqrt(T))
        if abs(d_delta_dK) < 1e-12:
            break
        K = K - (current_delta - target_call_delta) / d_delta_dK
        K = max(K, S * 0.5)
        K = min(K, S * 1.5)
        if abs(current_delta - target_call_delta) < 1e-6:
            break

    # Round to nearest whole dollar (standard for high-liquidity like SPY/QQQ)
    # Most weekly options on large-cap equities use $1 increments
    return round(K)
