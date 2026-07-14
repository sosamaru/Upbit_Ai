from __future__ import annotations

import numpy as np
import pandas as pd

from connectors import PaperBroker
from stock_ai import FEATURE_COLUMNS, AIConfig, StockProfitAI, backtest, build_features


def sample_frame(rows: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2022-01-01", periods=rows, freq="D")
    regime = np.sin(np.arange(rows) / 20) * 0.004
    returns = regime + rng.normal(0.0005, 0.012, rows)
    close = 100 * np.cumprod(1 + returns)
    open_price = close * (1 + rng.normal(0, 0.002, rows))
    spread = np.abs(rng.normal(0.006, 0.002, rows))
    high = np.maximum(open_price, close) * (1 + spread)
    low = np.minimum(open_price, close) * (1 - spread)
    volume = rng.integers(100_000, 500_000, rows)
    return pd.DataFrame(
        {"open": open_price, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def test_features_are_complete_after_warmup() -> None:
    features = build_features(sample_frame())
    assert tuple(features.columns) == FEATURE_COLUMNS
    assert len(features.dropna()) > 350


def test_model_training_prediction_and_backtest(tmp_path) -> None:
    frame = sample_frame()
    config = AIConfig(target_return=0.0, buy_threshold=0.55, sell_threshold=0.45)
    ai = StockProfitAI(config)
    metrics = ai.train(frame)
    assert 0 <= metrics["roc_auc"] <= 1

    model_path = tmp_path / "model.joblib"
    ai.save(model_path)
    loaded = StockProfitAI.load(model_path)
    prediction = loaded.latest_prediction(frame)
    assert prediction.action in {"BUY", "SELL", "HOLD"}

    probabilities = loaded.predict_probabilities(frame)
    result = backtest(frame, probabilities, loaded.config)
    assert result.final_equity > 0
    assert result.max_drawdown <= 0


def test_paper_broker_prevents_overspending() -> None:
    broker = PaperBroker(1_000)
    broker.update_price("TEST", 100)
    broker.buy("TEST", 5)
    assert broker.get_cash() == 500
    assert broker.get_position("TEST") == 5
