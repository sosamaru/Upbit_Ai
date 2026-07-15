"""Reproducible market-data collection and dataset manifest management."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import pandas as pd

from stock_ai import validate_ohlcv


class HistoricalDataProvider(Protocol):
    """Minimal interface required by the stage-3 collector."""

    def fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame: ...


@dataclass(frozen=True)
class DatasetRecord:
    symbol: str
    source: str
    start: str
    end: str
    interval: str
    collected_at_utc: str
    first_timestamp: str
    last_timestamp: str
    rows: int
    adjusted: bool
    timezone: str
    file: str
    sha256: str


@dataclass(frozen=True)
class CollectionResult:
    dataset_dir: str
    manifest_file: str
    records: tuple[DatasetRecord, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_dir": self.dataset_dir,
            "manifest_file": self.manifest_file,
            "records": [asdict(record) for record in self.records],
        }


def _safe_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper()
    if not normalized:
        raise ValueError("symbol cannot be empty")
    safe = re.sub(r"[^A-Z0-9._-]+", "_", normalized)
    if safe in {".", ".."}:
        raise ValueError("invalid symbol")
    return safe


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp_text(value: object) -> str:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


class MarketDataCollector:
    """Collect validated OHLCV files and an auditable JSON manifest."""

    def __init__(
        self,
        provider: HistoricalDataProvider,
        source_name: str,
        root_dir: str | Path = "data/raw",
        interval: str = "1d",
        adjusted: bool = False,
        timezone: str = "America/New_York",
    ) -> None:
        if not source_name.strip():
            raise ValueError("source_name cannot be empty")
        if interval not in {"1d", "1h", "30m", "15m"}:
            raise ValueError("unsupported interval")
        self.provider = provider
        self.source_name = source_name.strip()
        self.root_dir = Path(root_dir)
        self.interval = interval
        self.adjusted = adjusted
        self.timezone = timezone

    def collect(self, symbols: list[str], start: str, end: str) -> CollectionResult:
        unique_symbols = tuple(dict.fromkeys(_safe_symbol(symbol) for symbol in symbols))
        if not unique_symbols:
            raise ValueError("at least one symbol is required")
        if pd.Timestamp(start) >= pd.Timestamp(end):
            raise ValueError("start must be earlier than end")

        dataset_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        dataset_dir = self.root_dir / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=False)

        records: list[DatasetRecord] = []
        try:
            for symbol in unique_symbols:
                frame = validate_ohlcv(self.provider.fetch(symbol, start, end))
                file_path = dataset_dir / f"{symbol}.csv"
                frame.to_csv(file_path, index=True, index_label="timestamp")
                records.append(
                    DatasetRecord(
                        symbol=symbol,
                        source=self.source_name,
                        start=start,
                        end=end,
                        interval=self.interval,
                        collected_at_utc=datetime.now(UTC).isoformat(),
                        first_timestamp=_timestamp_text(frame.index[0]),
                        last_timestamp=_timestamp_text(frame.index[-1]),
                        rows=len(frame),
                        adjusted=self.adjusted,
                        timezone=self.timezone,
                        file=file_path.name,
                        sha256=_sha256(file_path),
                    )
                )

            manifest_path = dataset_dir / "manifest.json"
            payload = {
                "schema_version": 1,
                "dataset_id": dataset_id,
                "created_at_utc": datetime.now(UTC).isoformat(),
                "records": [asdict(record) for record in records],
            }
            temporary_path = manifest_path.with_suffix(".json.tmp")
            temporary_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temporary_path.replace(manifest_path)
            return CollectionResult(
                dataset_dir=str(dataset_dir),
                manifest_file=str(manifest_path),
                records=tuple(records),
            )
        except Exception:
            for child in dataset_dir.glob("*"):
                child.unlink(missing_ok=True)
            dataset_dir.rmdir()
            raise


def verify_dataset(manifest_file: str | Path) -> tuple[DatasetRecord, ...]:
    """Verify every file recorded in a stage-3 dataset manifest."""

    manifest_path = Path(manifest_file)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported manifest schema")

    records = tuple(DatasetRecord(**item) for item in payload.get("records", []))
    if not records:
        raise ValueError("manifest contains no dataset records")

    for record in records:
        file_path = manifest_path.parent / record.file
        if not file_path.is_file():
            raise ValueError(f"missing dataset file: {record.file}")
        if _sha256(file_path) != record.sha256:
            raise ValueError(f"checksum mismatch: {record.file}")
        frame = validate_ohlcv(pd.read_csv(file_path, index_col="timestamp", parse_dates=True))
        if len(frame) != record.rows:
            raise ValueError(f"row count mismatch: {record.file}")

    return records
