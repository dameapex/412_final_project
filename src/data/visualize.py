from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data.preprocess import (
    DATE_COLUMN,
    LOAD_COLUMNS,
    build_all_split_stages,
)


def plot_window_curves(
    base_dir: str | Path,
    split: str = "train",
    start_date: str | None = None,
    start_row_index: int = 0,
    num_days: int = 20,
    output_prefix: str | Path | None = None,
) -> Path:
    """Generate two simple report-ready figures for a concatenated multi-day window."""

    base_dir = Path(base_dir)
    all_stages = build_all_split_stages(base_dir)
    if split not in all_stages:
        raise ValueError(f"Unsupported split: {split}")

    raw_frame = all_stages[split]["raw"].copy()
    cleaned_frame = all_stages[split]["outlier_repaired"].copy()
    raw_frame[DATE_COLUMN] = pd.to_datetime(raw_frame[DATE_COLUMN])
    cleaned_frame[DATE_COLUMN] = pd.to_datetime(cleaned_frame[DATE_COLUMN])

    if start_date:
        anchor_date = pd.to_datetime(start_date).normalize()
        raw_window = raw_frame.loc[raw_frame[DATE_COLUMN] >= anchor_date].head(num_days).copy()
    else:
        raw_window = raw_frame.iloc[start_row_index : start_row_index + num_days].copy()

    if len(raw_window) == 0:
        raise ValueError("No rows available for the requested window")

    start_label = raw_window.iloc[0][DATE_COLUMN].strftime("%Y-%m-%d")
    end_label = raw_window.iloc[-1][DATE_COLUMN].strftime("%Y-%m-%d")
    cleaned_window = cleaned_frame.loc[
        (cleaned_frame[DATE_COLUMN] >= raw_window.iloc[0][DATE_COLUMN])
        & (cleaned_frame[DATE_COLUMN] <= raw_window.iloc[-1][DATE_COLUMN])
    ].copy()

    if cleaned_window.empty:
        raise ValueError("Selected window is not available after cleaning; try a different date range")

    if output_prefix is None:
        output_prefix = base_dir / "outputs" / "figures" / f"window_{split}_{start_label}_to_{end_label}"
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    raw_output_path = output_prefix.with_name(f"{output_prefix.name}_raw.png")
    processed_output_path = output_prefix.with_name(f"{output_prefix.name}_processed.png")

    raw_window = raw_window.sort_values(DATE_COLUMN).reset_index(drop=True)
    cleaned_window = cleaned_window.sort_values(DATE_COLUMN).reset_index(drop=True)

    raw_sequence = raw_window[LOAD_COLUMNS].to_numpy(dtype=float).reshape(-1)
    cleaned_sequence = cleaned_window[LOAD_COLUMNS].to_numpy(dtype=float).reshape(-1)
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_concatenated_sequence(
        sequence=raw_sequence,
        dates=raw_window[DATE_COLUMN].tolist(),
        title=f"Raw values ({start_label} to {end_label})",
        y_label="Raw load",
        color="#4c78a8",
        output_path=raw_output_path,
    )
    _plot_concatenated_sequence(
        sequence=cleaned_sequence,
        dates=cleaned_window[DATE_COLUMN].tolist(),
        title=f"Processed values ({start_label} to {end_label})",
        y_label="Processed load",
        color="#f58518",
        output_path=processed_output_path,
    )
    return raw_output_path, processed_output_path


def _plot_concatenated_sequence(
    sequence: pd.Series | list[float] | pd.Index | object,
    dates: list[pd.Timestamp],
    title: str,
    y_label: str,
    color: str,
    output_path: Path,
) -> None:
    x_axis = list(range(len(sequence)))
    figure, axis = plt.subplots(figsize=(16, 5.5))
    axis.plot(x_axis, sequence, linewidth=1.5, color=color)
    axis.set_title(title)
    axis.set_xlabel("Concatenated sample index across the selected time window")
    axis.set_ylabel(y_label)
    axis.grid(True, alpha=0.25)

    for boundary in range(96, len(x_axis), 96):
        axis.axvline(boundary, linestyle="--", linewidth=0.8, color="gray", alpha=0.5)

    if dates:
        step = max(1, len(dates) // 10)
        tick_positions = [day_index * 96 + 48 for day_index in range(0, len(dates), step)]
        tick_labels = [pd.to_datetime(dates[day_index]).strftime("%m-%d") for day_index in range(0, len(dates), step)]
        axis.set_xticks(tick_positions)
        axis.set_xticklabels(tick_labels)

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate simple before/after preprocessing plots")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--start-date", default=None, help="Window start date in YYYY-MM-DD format")
    parser.add_argument("--start-row-index", type=int, default=0, help="Fallback row index if date is not provided")
    parser.add_argument("--num-days", type=int, default=20, help="Number of days to concatenate in the plot")
    parser.add_argument("--base-dir", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--output-prefix", default=None, help="Optional output path prefix without suffix")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raw_output_path, processed_output_path = plot_window_curves(
        base_dir=args.base_dir,
        split=args.split,
        start_date=args.start_date,
        start_row_index=args.start_row_index,
        num_days=args.num_days,
        output_prefix=args.output_prefix,
    )
    print(raw_output_path)
    print(processed_output_path)


if __name__ == "__main__":
    main()