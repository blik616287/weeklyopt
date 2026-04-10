"""Base classes for option strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from ..pricing.black_scholes import OptionType


class StrategyDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class OptionLeg:
    option_type: OptionType
    strike: float
    is_long: bool  # True = buy, False = sell/write
    contracts: int = 1
    entry_price: float = 0.0  # premium per share at entry
    exit_price: float = 0.0   # premium per share at exit/expiry

    @property
    def multiplier(self) -> int:
        return 100

    @property
    def sign(self) -> int:
        return 1 if self.is_long else -1

    def pnl_per_contract(self) -> float:
        """P&L per contract (100 shares)."""
        return self.sign * (self.exit_price - self.entry_price) * self.multiplier


@dataclass
class TradeResult:
    ticker: str
    strategy_name: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_underlying: float
    exit_underlying: float
    legs: list[OptionLeg]
    contracts: int = 1
    # Filled in by engine
    total_pnl: float = 0.0
    total_premium_collected: float = 0.0
    total_premium_paid: float = 0.0
    max_risk: float = 0.0
    return_on_risk: float = 0.0
    exit_reason: str = "expiry"
    days_held: int = 0


class Strategy(ABC):
    """Base class for weekly option strategies."""

    name: str = "base"
    direction: StrategyDirection = StrategyDirection.NEUTRAL

    @abstractmethod
    def construct(
        self,
        underlying_price: float,
        vol: float,
        dte_years: float,
        risk_free_rate: float,
    ) -> list[OptionLeg]:
        """Build the option legs for this strategy given market conditions.

        Returns list of OptionLeg with entry_price filled in.
        """
        ...

    @abstractmethod
    def max_risk(self, legs: list[OptionLeg], underlying_price: float) -> float:
        """Maximum possible loss for the position (positive number)."""
        ...

    def evaluate_at_expiry(
        self,
        legs: list[OptionLeg],
        expiry_price: float,
    ) -> list[OptionLeg]:
        """Set exit_price on each leg based on intrinsic value at expiry."""
        for leg in legs:
            if leg.option_type == OptionType.CALL:
                leg.exit_price = max(expiry_price - leg.strike, 0.0)
            else:
                leg.exit_price = max(leg.strike - expiry_price, 0.0)
        return legs
