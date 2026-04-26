from __future__ import annotations

import argparse

from src.train import run_demo_training
from tests.smoke_test import run_smoke_test


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="CS412 final project scaffold")
	parser.add_argument(
		"command",
		choices=["smoke-test", "demo-train"],
		help="smoke-test checks the local PyTorch environment; demo-train runs a tiny synthetic training loop.",
	)
	return parser


def main() -> None:
	args = build_parser().parse_args()
	if args.command == "smoke-test":
		run_smoke_test()
		return

	metrics = run_demo_training()
	print(metrics)


if __name__ == "__main__":
	main()
