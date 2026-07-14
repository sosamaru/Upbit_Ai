"""Command-line entry point for data download, training, prediction, and backtesting."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from connectors import CSVMarketData, YFinanceMarketData
from stock_ai import AIConfig, StockProfitAI, backtest


def load_config() -> AIConfig:
    load_dotenv()
    return AIConfig(
        prediction_horizon=int(os.getenv("AI_PREDICTION_HORIZON", "5")),
        target_return=float(os.getenv("AI_TARGET_RETURN", "0.01")),
        buy_threshold=float(os.getenv("AI_BUY_THRESHOLD", "0.60")),
        sell_threshold=float(os.getenv("AI_SELL_THRESHOLD", "0.45")),
        stop_loss=float(os.getenv("AI_STOP_LOSS", "0.03")),
        take_profit=float(os.getenv("AI_TAKE_PROFIT", "0.06")),
        max_position_fraction=float(os.getenv("AI_MAX_POSITION_FRACTION", "0.25")),
        fee_rate=float(os.getenv("AI_FEE_RATE", "0.0005")),
        slippage_rate=float(os.getenv("AI_SLIPPAGE_RATE", "0.0005")),
        initial_cash=float(os.getenv("AI_INITIAL_CASH", "10000000")),
        random_state=int(os.getenv("AI_RANDOM_STATE", "42")),
    )


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Stock Profit Maximizer AI")
    subcommands = command.add_subparsers(dest="command", required=True)

    download = subcommands.add_parser("download", help="download public research data")
    download.add_argument("--symbol", required=True)
    download.add_argument("--start", required=True)
    download.add_argument("--end", required=True)
    download.add_argument("--output", default="data.csv")

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

    if args.command == "download":
        frame = YFinanceMarketData().fetch(args.symbol, args.start, args.end)
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination)
        print(json.dumps({"rows": len(frame), "output": str(destination)}, ensure_ascii=False))
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
