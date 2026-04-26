from __future__ import annotations

import argparse

from src.train import run_demo_training
from tests.smoke_test import run_smoke_test


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="CS412 final project scaffold")
	subparsers = parser.add_subparsers(dest="command", required=True)

	smoke_parser = subparsers.add_parser("smoke-test", help="Check the local PyTorch environment")
	smoke_parser.set_defaults(command="smoke-test")

	train_parser = subparsers.add_parser("demo-train", help="Run a tiny synthetic training loop")
	train_parser.set_defaults(command="demo-train")

	plot_parser = subparsers.add_parser("plot-preprocess", help="Generate simple raw/processed preprocessing plots")
	plot_parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
	plot_parser.add_argument("--start-date", default=None)
	plot_parser.add_argument("--start-row-index", type=int, default=0)
	plot_parser.add_argument("--num-days", type=int, default=20)
	plot_parser.add_argument("--output-prefix", default=None)
	plot_parser.set_defaults(command="plot-preprocess")
	return parser


def main() -> None:
	args = build_parser().parse_args()
	if args.command == "smoke-test":
		run_smoke_test()
		return

	if args.command == "plot-preprocess":
		from src.data.visualize import plot_window_curves

		raw_output_path, processed_output_path = plot_window_curves(
			base_dir="d:/sp26/412/final_project",
			split=args.split,
			start_date=args.start_date,
			start_row_index=args.start_row_index,
			num_days=args.num_days,
			output_prefix=args.output_prefix,
		)
		print(raw_output_path)
		print(processed_output_path)
		return

	metrics = run_demo_training()
	print(metrics)


if __name__ == "__main__":
	main()
