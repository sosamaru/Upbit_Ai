from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from market_data import MarketDataCollector, verify_dataset


def sample_frame(rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    close = 100 * np.cumprod(1 + rng.normal(0.0004, 0.01, rows))
    open_price = close * (1 + rng.normal(0, 0.002, rows))
    spread = np.abs(rng.normal(0.005, 0.001, rows))
    high = np.maximum(open_price, close) * (1 + spread)
    low = np.minimum(open_price, close) * (1 - spread)
    volume = rng.integers(100_000, 500_000, rows)
    return pd.DataFrame(
        {"open": open_price, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


class FakeProvider:
    def fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        del symbol, start, end
        return sample_frame()


class FailingProvider:
    def fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        del start, end
        if symbol == "BAD":
            raise RuntimeError("provider failure")
        return sample_frame()


def test_collect_writes_versioned_files_and_manifest(tmp_path) -> None:
    collector = MarketDataCollector(FakeProvider(), "fake", tmp_path)
    result = collector.collect(["aapl", "MSFT", "AAPL"], "2024-01-01", "2025-01-01")

    assert [record.symbol for record in result.records] == ["AAPL", "MSFT"]
    assert len(verify_dataset(result.manifest_file)) == 2

    manifest = json.loads((tmp_path / result.dataset_dir.split("/")[-1] / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert {record["symbol"] for record in manifest["records"]} == {"AAPL", "MSFT"}


def test_verify_detects_tampered_dataset(tmp_path) -> None:
    collector = MarketDataCollector(FakeProvider(), "fake", tmp_path)
    result = collector.collect(["AAPL"], "2024-01-01", "2025-01-01")
    data_file = next((tmp_path / result.dataset_dir.split("/")[-1]).glob("AAPL.csv"))
    data_file.write_text(data_file.read_text() + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_dataset(result.manifest_file)


def test_failed_collection_removes_partial_dataset(tmp_path) -> None:
    collector = MarketDataCollector(FailingProvider(), "fake", tmp_path)
    with pytest.raises(RuntimeError, match="provider failure"):
        collector.collect(["AAPL", "BAD"], "2024-01-01", "2025-01-01")
    assert list(tmp_path.iterdir()) == []


def test_collection_rejects_invalid_range(tmp_path) -> None:
    collector = MarketDataCollector(FakeProvider(), "fake", tmp_path)
    with pytest.raises(ValueError, match="start must be earlier"):
        collector.collect(["AAPL"], "2025-01-01", "2024-01-01")
