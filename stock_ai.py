"""Core AI, policy configuration, feature engineering, and backtesting.

The project stays compact: external market/broker connections live in ``connectors.py``
while this module owns the validated research policy and AI logic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")
FEATURE_COLUMNS = (
    "return_1",
    "return_5",
    "return_20",
    "volatility_10",
    "volatility_20",
    "volume_ratio_20",
    "ma_gap_5_20",
    "ma_gap_20_60",
    "rsi_14",
    "atr_pct_14",
)


@dataclass(frozen=True)
class AIConfig:
    """Single source of truth for the stage-2 trading objective and risk limits."""

    # Research scope. Live trading is intentionally unavailable before stage 20.
    trade_mode: str = "paper"
    market: str = "US_EQUITY"
    bar_interval: str = "1d"
    strategy_style: str = "swing"
    long_only: bool = True
    allow_leverage: bool = False
    allow_short: bool = False

    # Prediction objective: seek at least 1% gross return over five daily bars.
    prediction_horizon: int = 5
    target_return: float = 0.01
    buy_threshold: float = 0.60
    sell_threshold: float = 0.45
    max_holding_bars: int = 10

    # Portfolio and loss limits. These are policy limits; stage 17 builds the full
    # account-level enforcement engine for daily/weekly losses and multi-asset risk.
    max_positions: int = 3
    max_position_fraction: float = 0.20
    minimum_cash_fraction: float = 0.30
    stop_loss: float = 0.03
    take_profit: float = 0.06
    max_daily_loss: float = 0.02
    max_weekly_loss: float = 0.05
    max_drawdown_limit: float = 0.10
    max_consecutive_losses: int = 3

    # Backtest assumptions.
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0005
    initial_cash: float = 10_000_000.0
    random_state: int = 42

    def validate(self) -> None:
        if self.trade_mode not in {"research", "paper"}:
            raise ValueError("trade_mode must be research or paper before stage 20")
        if self.market not in {"US_EQUITY", "KR_EQUITY"}:
            raise ValueError("market must be US_EQUITY or KR_EQUITY")
        if self.bar_interval not in {"1d", "1h", "30m", "15m"}:
            raise ValueError("unsupported bar_interval")
        if self.strategy_style not in {"swing", "position", "intraday"}:
            raise ValueError("unsupported strategy_style")
        if not self.long_only or self.allow_leverage or self.allow_short:
            raise ValueError("stage-2 policy is long-only without leverage or shorting")
        if self.prediction_horizon < 1 or self.max_holding_bars < 1:
            raise ValueError("prediction and holding horizons must be at least 1")
        if self.target_return <= 0:
            raise ValueError("target_return must be positive")
        if not 0 < self.buy_threshold <= 1:
            raise ValueError("buy_threshold must be in (0, 1]")
        if not 0 <= self.sell_threshold < self.buy_threshold:
            raise ValueError("sell_threshold must be below buy_threshold")
        if self.max_positions < 1:
            raise ValueError("max_positions must be at least 1")
        if self.max_consecutive_losses < 1:
            raise ValueError("max_consecutive_losses must be at least 1")

        fraction_fields = (
            "max_position_fraction",
            "minimum_cash_fraction",
            "stop_loss",
            "take_profit",
            "max_daily_loss",
            "max_weekly_loss",
            "max_drawdown_limit",
        )
        for name in fraction_fields:
            value = float(getattr(self, name))
            if not 0 < value < 1:
                raise ValueError(f"{name} must be in (0, 1)")

        if self.stop_loss >= self.take_profit:
            raise ValueError("stop_loss must be below take_profit")
        if not self.max_daily_loss <= self.max_weekly_loss <= self.max_drawdown_limit:
            raise ValueError("loss limits must satisfy daily <= weekly <= drawdown")
        deployable_fraction = 1 - self.minimum_cash_fraction
        if self.max_positions * self.max_position_fraction > deployable_fraction + 1e-12:
            raise ValueError("position limits exceed capital available after cash reserve")
        if self.fee_rate < 0 or self.slippage_rate < 0:
            raise ValueError("trading costs cannot be negative")
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")

    def policy_summary(self) -> dict[str, Any]:
        """Return the fixed stage-2 policy in a serializable form."""

        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class Prediction:
    timestamp: str
    close: float
    probability_up: float
    action: str


@dataclass(frozen=True)
class BacktestResult:
    initial_cash: float
    final_equity: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    trade_count: int
    win_rate: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def validate_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a clean, sorted OHLCV frame or raise a precise error."""

    if frame.empty:
        raise ValueError("OHLCV data is empty")

    normalized = frame.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]
    missing = [column for column in REQUIRED_COLUMNS if column not in normalized.columns]
    if missing:
        raise ValueError(f"missing OHLCV columns: {', '.join(missing)}")

    normalized = normalized.loc[:, REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    normalized = normalized.replace([np.inf, -np.inf], np.nan).dropna()
    normalized = normalized[~normalized.index.duplicated(keep="last")].sort_index()

    if len(normalized) < 80:
        raise ValueError("at least 80 rows of OHLCV data are required")
    if (normalized[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("price columns must be positive")
    if (normalized["volume"] < 0).any():
        raise ValueError("volume cannot be negative")
    if (normalized["high"] < normalized[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("high price is inconsistent")
    if (normalized["low"] > normalized[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("low price is inconsistent")

    return cast(pd.DataFrame, normalized.astype(float))


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    relative_strength = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + relative_strength))


def _atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period).mean()


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build causal features using only current and past bars."""

    data = validate_ohlcv(frame)
    close = data["close"]
    returns = close.pct_change()

    features = pd.DataFrame(index=data.index)
    features["return_1"] = returns
    features["return_5"] = close.pct_change(5)
    features["return_20"] = close.pct_change(20)
    features["volatility_10"] = returns.rolling(10).std()
    features["volatility_20"] = returns.rolling(20).std()
    features["volume_ratio_20"] = data["volume"] / data["volume"].rolling(20).mean()
    features["ma_gap_5_20"] = close.rolling(5).mean() / close.rolling(20).mean() - 1
    features["ma_gap_20_60"] = close.rolling(20).mean() / close.rolling(60).mean() - 1
    features["rsi_14"] = _rsi(close) / 100
    features["atr_pct_14"] = _atr(data) / close

    return features.replace([np.inf, -np.inf], np.nan)


def build_training_set(
    frame: pd.DataFrame, config: AIConfig
) -> tuple[pd.DataFrame, pd.Series]:
    """Create features and a forward-return classification target."""

    config.validate()
    data = validate_ohlcv(frame)
    features = build_features(data)
    forward_return = data["close"].shift(-config.prediction_horizon) / data["close"] - 1
    target = (forward_return >= config.target_return).astype(float)
    target[forward_return.isna()] = np.nan

    joined = features.join(target.rename("target")).dropna()
    if len(joined) < 40:
        raise ValueError("not enough usable rows after feature generation")
    if joined["target"].nunique() < 2:
        raise ValueError("training target contains only one class")

    return joined.loc[:, FEATURE_COLUMNS], joined["target"].astype(int)


class StockProfitAI:
    """Compact stock direction model with persistence and walk-forward-safe output."""

    def __init__(self, config: AIConfig | None = None) -> None:
        self.config = config or AIConfig()
        self.config.validate()
        self.model = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=0.5,
                        class_weight="balanced",
                        max_iter=1_000,
                        random_state=self.config.random_state,
                    ),
                ),
            ]
        )
        self.is_fitted = False

    def train(self, frame: pd.DataFrame, test_fraction: float = 0.2) -> dict[str, float]:
        features, target = build_training_set(frame, self.config)
        if not 0.1 <= test_fraction <= 0.4:
            raise ValueError("test_fraction must be between 0.1 and 0.4")

        split_index = int(len(features) * (1 - test_fraction))
        x_train, x_test = features.iloc[:split_index], features.iloc[split_index:]
        y_train, y_test = target.iloc[:split_index], target.iloc[split_index:]
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            raise ValueError("time split must contain both target classes")

        self.model.fit(x_train, y_train)
        self.is_fitted = True
        probability = self.model.predict_proba(x_test)[:, 1]
        predicted = (probability >= self.config.buy_threshold).astype(int)

        return {
            "train_rows": float(len(x_train)),
            "test_rows": float(len(x_test)),
            "accuracy": float(accuracy_score(y_test, predicted)),
            "precision": float(precision_score(y_test, predicted, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probability)),
        }

    def predict_probabilities(self, frame: pd.DataFrame) -> pd.Series:
        self._require_fitted()
        features = build_features(frame).dropna()
        if features.empty:
            raise ValueError("no usable feature rows for prediction")
        values = self.model.predict_proba(features.loc[:, FEATURE_COLUMNS])[:, 1]
        return pd.Series(values, index=features.index, name="probability_up")

    def latest_prediction(self, frame: pd.DataFrame) -> Prediction:
        data = validate_ohlcv(frame)
        probabilities = self.predict_probabilities(data)
        timestamp = probabilities.index[-1]
        probability = float(probabilities.iloc[-1])
        action = self._action(probability)
        return Prediction(
            timestamp=str(timestamp),
            close=float(data.loc[timestamp, "close"]),
            probability_up=probability,
            action=action,
        )

    def save(self, path: str | Path) -> None:
        self._require_fitted()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"config": asdict(self.config), "model": self.model}, destination)

    @classmethod
    def load(cls, path: str | Path) -> "StockProfitAI":
        payload: dict[str, Any] = joblib.load(Path(path))
        instance = cls(AIConfig(**payload["config"]))
        instance.model = payload["model"]
        instance.is_fitted = True
        return instance

    def _action(self, probability: float) -> str:
        if probability >= self.config.buy_threshold:
            return "BUY"
        if probability <= self.config.sell_threshold:
            return "SELL"
        return "HOLD"

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("model is not trained or loaded")


def backtest(
    frame: pd.DataFrame,
    probabilities: pd.Series,
    config: AIConfig,
) -> BacktestResult:
    """Run a long-only simulation with next-bar execution and stage-2 limits."""

    config.validate()
    data = validate_ohlcv(frame)
    aligned_probability = probabilities.reindex(data.index).shift(1)
    cash = config.initial_cash
    quantity = 0.0
    entry_price = 0.0
    holding_bars = 0
    trades: list[float] = []
    equity_curve: list[float] = []
    trading_cost = config.fee_rate + config.slippage_rate

    for timestamp, row in data.iterrows():
        open_price = float(row["open"])
        close_price = float(row["close"])
        probability_raw = aligned_probability.reindex([timestamp]).iloc[0]
        probability = float(probability_raw) if pd.notna(probability_raw) else np.nan

        if quantity > 0:
            holding_bars += 1
            return_from_entry = close_price / entry_price - 1
            should_exit = (
                return_from_entry <= -config.stop_loss
                or return_from_entry >= config.take_profit
                or holding_bars >= config.max_holding_bars
                or (pd.notna(probability) and probability <= config.sell_threshold)
            )
            if should_exit:
                exit_price = open_price * (1 - config.slippage_rate)
                proceeds = quantity * exit_price * (1 - config.fee_rate)
                cost_basis = quantity * entry_price * (1 + config.fee_rate)
                trades.append(proceeds - cost_basis)
                cash += proceeds
                quantity = 0.0
                entry_price = 0.0
                holding_bars = 0

        if (
            quantity == 0
            and pd.notna(probability)
            and probability >= config.buy_threshold
        ):
            budget = cash * config.max_position_fraction
            buy_price = open_price * (1 + config.slippage_rate)
            quantity = budget / (buy_price * (1 + config.fee_rate))
            spent = quantity * buy_price * (1 + config.fee_rate)
            cash -= spent
            entry_price = buy_price
            holding_bars = 0

        equity_curve.append(cash + quantity * close_price * (1 - trading_cost))

    if quantity > 0:
        final_price = float(data["close"].iloc[-1]) * (1 - config.slippage_rate)
        proceeds = quantity * final_price * (1 - config.fee_rate)
        cost_basis = quantity * entry_price * (1 + config.fee_rate)
        trades.append(proceeds - cost_basis)
        cash += proceeds
        equity_curve[-1] = cash

    equity = pd.Series(equity_curve, index=data.index, dtype=float)
    returns = equity.pct_change().dropna()
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1
    sharpe = 0.0
    if not returns.empty and returns.std(ddof=0) > 0:
        sharpe = float(np.sqrt(252) * returns.mean() / returns.std(ddof=0))

    wins = sum(profit > 0 for profit in trades)
    return BacktestResult(
        initial_cash=config.initial_cash,
        final_equity=float(equity.iloc[-1]),
        total_return=float(equity.iloc[-1] / config.initial_cash - 1),
        max_drawdown=float(drawdown.min()),
        sharpe_ratio=sharpe,
        trade_count=len(trades),
        win_rate=float(wins / len(trades)) if trades else 0.0,
    )
