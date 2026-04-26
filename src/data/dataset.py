from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.preprocess import DATE_COLUMN, LOAD_COLUMNS


@dataclass
class SampleBatch:
    """Container used by the training loop for readability."""

    # One multi-channel input tensor per sample: (features, time).
    inputs: torch.Tensor
    target: torch.Tensor


class ProcessedLoadDataset(Dataset):
    """Dataset backed by the exported processed daily csv files.

    Input channels follow the current model assumption:
    - channel 0: previous day's load profile
    - channels 1-4: target day's calendar features broadcast across the day
    - channels 5-7: previous day's summary statistics broadcast across the day
    """

    target_calendar_columns = ["day_of_week", "is_weekend", "month", "day_of_year"]
    previous_summary_columns = ["daily_mean", "daily_std", "daily_range"]

    def __init__(
        self,
        split: str,
        processed_dir: str | Path = "data/processed",
        use_standardized: bool = True,
    ) -> None:
        super().__init__()
        processed_dir = Path(processed_dir)
        suffix = "standardized" if use_standardized else "cleaned"
        file_path = processed_dir / f"{split}_daily_{suffix}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Processed dataset file not found: {file_path}")

        frame = pd.read_csv(file_path)
        frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN])
        self.frame = frame.sort_values(DATE_COLUMN).reset_index(drop=True)
        self.sample_indices = list(range(len(self.frame) - 1))
        if not self.sample_indices:
            raise ValueError(f"Split {split} does not contain enough rows to build next-day samples")

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, index: int) -> SampleBatch:
        current_index = self.sample_indices[index]
        current_row = self.frame.iloc[current_index]
        target_row = self.frame.iloc[current_index + 1]

        previous_day_load = torch.tensor(current_row[LOAD_COLUMNS].to_numpy(dtype=float), dtype=torch.float32)
        repeated_target_calendar = [
            torch.full_like(previous_day_load, float(target_row[column]))
            for column in self.target_calendar_columns
        ]
        repeated_previous_summary = [
            torch.full_like(previous_day_load, float(current_row[column]))
            for column in self.previous_summary_columns
        ]
        inputs = torch.stack(
            [previous_day_load, *repeated_target_calendar, *repeated_previous_summary],
            dim=0,
        )
        target = torch.tensor(target_row[LOAD_COLUMNS].to_numpy(dtype=float), dtype=torch.float32)
        return SampleBatch(inputs=inputs, target=target)


class SyntheticLoadDataset(Dataset):
    """Small synthetic dataset for smoke tests and pipeline validation.

    Each sample uses one day's multi-feature 96-point sequence to predict the
    next day's 96-point target load curve.
    """

    def __init__(
        self,
        num_samples: int = 256,
        input_steps: int = 96,
        output_steps: int = 96,
        num_features: int = 8,
        seed: int = 42,
    ) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)

        time_axis = torch.linspace(0, 2 * torch.pi, input_steps)
        base_curve = 0.6 * torch.sin(time_axis) + 0.3 * torch.cos(2 * time_axis)

        inputs = []
        target = []
        for index in range(num_samples):
            phase = index / 14.0
            daily_shift = torch.rand(1, generator=generator).item() * 0.2
            noise = 0.05 * torch.randn(input_steps, generator=generator)
            curve = base_curve + daily_shift + 0.2 * torch.sin(time_axis + phase) + noise
            next_curve = curve.roll(-4) + 0.03 * torch.randn(output_steps, generator=generator)

            # Channel 0 is the target load history baseline used by the model residual output.
            feature_channels = [curve]
            # Additional channels emulate exogenous features (weather/calendar-like signals).
            for feature_index in range(max(0, num_features - 1)):
                freq = 1.0 + (feature_index % 3)
                phase_shift = 0.15 * feature_index + phase
                seasonal = torch.sin(freq * time_axis + phase_shift)
                trend = (feature_index + 1) * 0.02 * torch.linspace(0, 1, input_steps)
                feature_noise = 0.02 * torch.randn(input_steps, generator=generator)
                feature_channels.append(seasonal + trend + feature_noise)

            # Final input shape per sample: (num_features, input_steps).
            inputs.append(torch.stack(feature_channels[:num_features], dim=0))
            target.append(next_curve)

        self.inputs = torch.stack(inputs).float()
        self.target = torch.stack(target).float()

    def __len__(self) -> int:
        return self.inputs.size(0)

    def __getitem__(self, index: int) -> SampleBatch:
        return SampleBatch(
            inputs=self.inputs[index],
            target=self.target[index],
        )
