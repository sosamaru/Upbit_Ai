"""Replaceable market-data and broker connectors.

The AI core never depends directly on a specific broker. Implement the protocols below
when adding Korea Investment, Alpaca, Interactive Brokers, or another provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from stock_ai import validate_ohlcv


class MarketDataProvider(Protocol):
    def fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame: ...


class Broker(Protocol):
    def get_cash(self) -> float: ...

    def get_position(self, symbol: str) -> float: ...

    def buy(self, symbol: str, quantity: float) -> str: ...

    def sell(self, symbol: str, quantity: float) -> str: ...


class CSVMarketData:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def fetch(self, symbol: str = "", start: str = "", end: str = "") -> pd.DataFrame:
        del symbol, start, end
        frame = pd.read_csv(self.path, index_col=0, parse_dates=True)
        return validate_ohlcv(frame)


class YFinanceMarketData:
    """Public research connector. Do not treat it as execution-grade market data."""

    def fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        import yfinance as yf

        frame = yf.download(
            symbol,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            multi_level_index=False,
        )
        if frame.empty:
            raise ValueError(f"no market data returned for {symbol}")
        return validate_ohlcv(frame)


@dataclass
class PaperBroker:
    """Minimal in-memory broker used before any real-money integration."""

    cash: float

    def __post_init__(self) -> None:
        if self.cash <= 0:
            raise ValueError("paper broker cash must be positive")
        self.positions: dict[str, float] = {}
        self.last_prices: dict[str, float] = {}
        self.order_sequence = 0

    def update_price(self, symbol: str, price: float) -> None:
        if price <= 0:
            raise ValueError("price must be positive")
        self.last_prices[symbol] = price

    def get_cash(self) -> float:
        return self.cash

    def get_position(self, symbol: str) -> float:
        return self.positions.get(symbol, 0.0)

    def buy(self, symbol: str, quantity: float) -> str:
        price = self._price(symbol)
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        cost = price * quantity
        if cost > self.cash:
            raise ValueError("insufficient paper cash")
        self.cash -= cost
        self.positions[symbol] = self.get_position(symbol) + quantity
        return self._order_id("BUY")

    def sell(self, symbol: str, quantity: float) -> str:
        price = self._price(symbol)
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if quantity > self.get_position(symbol):
            raise ValueError("insufficient paper position")
        self.positions[symbol] = self.get_position(symbol) - quantity
        self.cash += price * quantity
        return self._order_id("SELL")

    def _price(self, symbol: str) -> float:
        if symbol not in self.last_prices:
            raise ValueError(f"missing paper price for {symbol}")
        return self.last_prices[symbol]

    def _order_id(self, side: str) -> str:
        self.order_sequence += 1
        return f"PAPER-{side}-{self.order_sequence:06d}"
