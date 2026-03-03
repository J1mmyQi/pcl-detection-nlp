"""CLI entrypoint for data, training, and export workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data_pipeline import run_stats
from .training_pipeline import export_split_predictions, train_from_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PCL detection project scaffold CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    stats_parser = subparsers.add_parser("stats", help="Generate a basic EDA report")
    stats_parser.add_argument("--config", default="configs/baseline.json")

    train_parser = subparsers.add_parser("train", help="Train a configured baseline")
    train_parser.add_argument("--config", default="configs/baseline.json")

    export_parser = subparsers.add_parser("export", help="Export dev/test predictions")
    export_parser.add_argument("--config", default="configs/baseline.json")
    export_parser.add_argument("--model-dir", required=True)
    export_parser.add_argument("--split", choices=("dev", "test"), required=True)
    export_parser.add_argument("--output-file", required=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "stats":
        report = run_stats(args.config)
        print(json.dumps(report, indent=2))
        return

    if args.command == "train":
        result = train_from_config(args.config)
        print(json.dumps(result, indent=2))
        return

    if args.command == "export":
        output_path = export_split_predictions(
            config_path=args.config,
            model_dir=Path(args.model_dir),
            split=args.split,
            output_file=Path(args.output_file),
        )
        print(json.dumps({"output_file": str(output_path)}, indent=2))
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
