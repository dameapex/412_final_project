from __future__ import annotations

import argparse
from pathlib import Path

from src.train_baseline_lstm import run_lstm_baseline_training
from src.train import run_demo_training, run_training_with_split_metrics


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="CS412 final project scaffold")
	subparsers = parser.add_subparsers(dest="command")
	parser.set_defaults(command="run-all")

	run_all_parser = subparsers.add_parser("run-all", help="Train and report train/validation/test metrics")
	run_all_parser.set_defaults(command="run-all")

	train_parser = subparsers.add_parser("demo-train", help="Run a tiny synthetic training loop")
	train_parser.add_argument("--config", default="configs/common.yaml")
	train_parser.set_defaults(command="demo-train")

	plot_parser = subparsers.add_parser("plot-preprocess", help="Generate simple raw/processed preprocessing plots")
	plot_parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
	plot_parser.add_argument("--start-date", default=None)
	plot_parser.add_argument("--start-row-index", type=int, default=0)
	plot_parser.add_argument("--num-days", type=int, default=20)
	plot_parser.add_argument("--output-prefix", default=None)
	plot_parser.set_defaults(command="plot-preprocess")

	lstm_baseline_parser = subparsers.add_parser("lstm-baseline", help="Train and evaluate LSTM baseline")
	lstm_baseline_parser.add_argument("--config", default="configs/common.yaml")
	lstm_baseline_parser.set_defaults(command="lstm-baseline")
	return parser


def main() -> None:
	args = build_parser().parse_args()
	if args.command == "run-all":
		split_metrics = run_training_with_split_metrics()
		print(split_metrics)
		return

	if args.command == "plot-preprocess":
		from src.data.visualize import plot_window_curves

		raw_output_path, processed_output_path = plot_window_curves(
			base_dir=Path(__file__).resolve().parent,
			split=args.split,
			start_date=args.start_date,
			start_row_index=args.start_row_index,
			num_days=args.num_days,
			output_prefix=args.output_prefix,
		)
		print(raw_output_path)
		print(processed_output_path)
		return

	if args.command == "lstm-baseline":
		split_metrics = run_lstm_baseline_training(config_path=args.config)
		print(split_metrics)
		return

	metrics = run_demo_training(config_path=args.config)
	print(metrics)


if __name__ == "__main__":
	main()
