"""Command-line entry point for configuration, data, training, and backtesting."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from connectors import CSVMarketData, YFinanceMarketData
from market_data import MarketDataCollector, verify_dataset
from stock_ai import AIConfig, StockProfitAI, backtest


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "true" if default else "false").strip().lower()
    if raw not in {"true", "false", "1", "0", "yes", "no"}:
        raise ValueError(f"{name} must be a boolean")
    return raw in {"true", "1", "yes"}


def load_config() -> AIConfig:
    load_dotenv()
    config = AIConfig(
        trade_mode=os.getenv("AI_TRADE_MODE", "paper"),
        market=os.getenv("AI_MARKET", "US_EQUITY"),
        bar_interval=os.getenv("AI_BAR_INTERVAL", "1d"),
        strategy_style=os.getenv("AI_STRATEGY_STYLE", "swing"),
        long_only=_env_bool("AI_LONG_ONLY", True),
        allow_leverage=_env_bool("AI_ALLOW_LEVERAGE", False),
        allow_short=_env_bool("AI_ALLOW_SHORT", False),
        prediction_horizon=int(os.getenv("AI_PREDICTION_HORIZON", "5")),
        target_return=float(os.getenv("AI_TARGET_RETURN", "0.01")),
        buy_threshold=float(os.getenv("AI_BUY_THRESHOLD", "0.60")),
        sell_threshold=float(os.getenv("AI_SELL_THRESHOLD", "0.45")),
        max_holding_bars=int(os.getenv("AI_MAX_HOLDING_BARS", "10")),
        max_positions=int(os.getenv("AI_MAX_POSITIONS", "3")),
        max_position_fraction=float(os.getenv("AI_MAX_POSITION_FRACTION", "0.20")),
        minimum_cash_fraction=float(os.getenv("AI_MINIMUM_CASH_FRACTION", "0.30")),
        stop_loss=float(os.getenv("AI_STOP_LOSS", "0.03")),
        take_profit=float(os.getenv("AI_TAKE_PROFIT", "0.06")),
        max_daily_loss=float(os.getenv("AI_MAX_DAILY_LOSS", "0.02")),
        max_weekly_loss=float(os.getenv("AI_MAX_WEEKLY_LOSS", "0.05")),
        max_drawdown_limit=float(os.getenv("AI_MAX_DRAWDOWN_LIMIT", "0.10")),
        max_consecutive_losses=int(os.getenv("AI_MAX_CONSECUTIVE_LOSSES", "3")),
        fee_rate=float(os.getenv("AI_FEE_RATE", "0.0005")),
        slippage_rate=float(os.getenv("AI_SLIPPAGE_RATE", "0.0005")),
        initial_cash=float(os.getenv("AI_INITIAL_CASH", "10000000")),
        random_state=int(os.getenv("AI_RANDOM_STATE", "42")),
    )
    config.validate()
    return config


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Stock Profit Maximizer AI")
    subcommands = command.add_subparsers(dest="command", required=True)

    subcommands.add_parser("config", help="validate and print the fixed trading policy")

    download = subcommands.add_parser("download", help="download one public research CSV")
    download.add_argument("--symbol", required=True)
    download.add_argument("--start", required=True)
    download.add_argument("--end", required=True)
    download.add_argument("--output", default="data.csv")

    collect = subcommands.add_parser("collect", help="collect a versioned multi-symbol dataset")
    collect.add_argument("--symbols", nargs="+", required=True)
    collect.add_argument("--start", required=True)
    collect.add_argument("--end", required=True)
    collect.add_argument("--output-dir", default="data/raw")

    verify = subcommands.add_parser("verify-data", help="verify a dataset manifest and checksums")
    verify.add_argument("--manifest", required=True)

    train = subcommands.add_parser("train", help="train and save a model")
    train.add_argument("--data", required=True)
    train.add_argument("--model", default="model.joblib")

    predict = subcommands.add_parser("predict", help="show the latest prediction")
    predict.add_argument("--data", required=True)
    predict.add_argument("--model", default="model.joblib")

    simulation = subcommands.add_parser("backtest", help="run a risk-aware backtest")
    simulation.add_argument("--data", required=True)
    simulation.add_argument("--model", default="model.joblib")

    return command


def main() -> None:
    args = parser().parse_args()
    config = load_config()

    if args.command == "config":
        print(json.dumps(config.policy_summary(), ensure_ascii=False, indent=2))
        return
    if args.command == "download":
        frame = YFinanceMarketData().fetch(args.symbol, args.start, args.end)
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination)
        print(json.dumps({"rows": len(frame), "output": str(destination)}, ensure_ascii=False))
        return
    if args.command == "collect":
        collector = MarketDataCollector(
            provider=YFinanceMarketData(),
            source_name="yfinance",
            root_dir=args.output_dir,
            interval=config.bar_interval,
            adjusted=False,
        )
        result = collector.collect(args.symbols, args.start, args.end)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return
    if args.command == "verify-data":
        records = verify_dataset(args.manifest)
        print(
            json.dumps(
                {"verified": len(records), "symbols": [record.symbol for record in records]},
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    frame = CSVMarketData(args.data).fetch()
    if args.command == "train":
        ai = StockProfitAI(config)
        metrics = ai.train(frame)
        ai.save(args.model)
        print(json.dumps({"model": args.model, "metrics": metrics}, ensure_ascii=False, indent=2))
        return

    ai = StockProfitAI.load(args.model)
    if args.command == "predict":
        print(json.dumps(asdict(ai.latest_prediction(frame)), ensure_ascii=False, indent=2))
        return

    probabilities = ai.predict_probabilities(frame)
    result = backtest(frame, probabilities, ai.config)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
