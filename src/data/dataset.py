from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

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


class _SampleSpec(NamedTuple):
    sample_index: int
    slice_offset: int


class ProcessedLoadDataset(Dataset):
    """Dataset backed by the exported processed daily csv files.

    Input channels follow the current model assumption:
    - channel 0: previous day's load profile
    - channels 1-4: target day's calendar features broadcast across the day
    - channels 5-7: previous day's summary statistics broadcast across the day
    """

    target_calendar_columns = ["day_of_week", "is_weekend", "month", "day_of_year"]
    previous_summary_columns = ["daily_mean", "daily_std", "daily_range"]
    summary_channel_start = 1 + len(target_calendar_columns)

    def __init__(
        self,
        split: str,
        processed_dir: str | Path = "data/processed",
        use_standardized: bool = True,
        time_slice_stride: int = 96,
        enable_augmentation: bool = False,
        amplitude_jitter_prob: float = 0.0,
        amplitude_jitter_min: float = 1.0,
        amplitude_jitter_max: float = 1.0,
        input_noise_std: float = 0.0,
        time_mask_prob: float = 0.0,
        time_mask_min_width: int = 0,
        time_mask_max_width: int = 0,
    ) -> None:
        super().__init__()
        if time_slice_stride < 1:
            raise ValueError("time_slice_stride must be >= 1")

        processed_dir = Path(processed_dir)
        suffix = "standardized" if use_standardized else "cleaned"
        file_path = processed_dir / f"{split}_daily_{suffix}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Processed dataset file not found: {file_path}")

        frame = pd.read_csv(file_path)
        frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN])
        self.frame = frame.sort_values(DATE_COLUMN).reset_index(drop=True)
        base_sample_indices = list(range(len(self.frame) - 1))
        if not base_sample_indices:
            raise ValueError(f"Split {split} does not contain enough rows to build next-day samples")
        self.sample_specs = [
            _SampleSpec(sample_index=sample_index, slice_offset=offset)
            for sample_index in base_sample_indices
            for offset in range(0, len(LOAD_COLUMNS), time_slice_stride)
        ]
        self.summary_channel_start = 1 + len(self.target_calendar_columns)
        self.enable_augmentation = enable_augmentation
        self.amplitude_jitter_prob = amplitude_jitter_prob
        self.amplitude_jitter_min = amplitude_jitter_min
        self.amplitude_jitter_max = amplitude_jitter_max
        self.input_noise_std = input_noise_std
        self.time_mask_prob = time_mask_prob
        self.time_mask_min_width = time_mask_min_width
        self.time_mask_max_width = time_mask_max_width

    def _apply_time_mask(self, sequence: torch.Tensor) -> torch.Tensor:
        if self.time_mask_prob <= 0 or self.time_mask_max_width <= 0:
            return sequence
        if torch.rand(1).item() >= self.time_mask_prob:
            return sequence

        length = sequence.numel()
        min_width = max(1, min(self.time_mask_min_width, length))
        max_width = max(min_width, min(self.time_mask_max_width, length))
        if max_width <= 0:
            return sequence

        width = int(torch.randint(min_width, max_width + 1, (1,)).item())
        start = int(torch.randint(0, max(1, length - width + 1), (1,)).item())
        end = start + width

        masked = sequence.clone()
        left_value = masked[start - 1] if start > 0 else masked[end] if end < length else masked.mean()
        right_value = masked[end] if end < length else left_value
        masked[start:end] = torch.linspace(left_value, right_value, width, device=sequence.device, dtype=sequence.dtype)
        return masked

    def _augment_sample(self, inputs: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        augmented_inputs = inputs.clone()
        augmented_target = target.clone()
        augmented_load = augmented_inputs[0]

        if torch.rand(1).item() < self.amplitude_jitter_prob:
            scale = torch.empty(1).uniform_(self.amplitude_jitter_min, self.amplitude_jitter_max).item()
            augmented_load = augmented_load * scale
            augmented_target = augmented_target * scale

        if self.input_noise_std > 0:
            augmented_load = augmented_load + torch.randn_like(augmented_load) * self.input_noise_std

        augmented_load = self._apply_time_mask(augmented_load)

        augmented_inputs[0] = augmented_load

        # Keep summary channels consistent with the possibly augmented load channel.
        summary_values = [
            float(augmented_load.mean()),
            float(augmented_load.std(unbiased=False)),
            float(augmented_load.max() - augmented_load.min()),
        ]
        for offset, value in enumerate(summary_values):
            augmented_inputs[self.summary_channel_start + offset] = torch.full_like(augmented_load, value)

        return augmented_inputs, augmented_target

    def __len__(self) -> int:
        return len(self.sample_specs)

    def _slice_sequence(self, sequence: torch.Tensor, offset: int) -> torch.Tensor:
        if offset == 0:
            return sequence
        tiled = torch.cat([sequence, sequence], dim=-1)
        return tiled[..., offset : offset + len(LOAD_COLUMNS)]

    def __getitem__(self, index: int) -> SampleBatch:
        spec = self.sample_specs[index]
        current_index = spec.sample_index
        slice_offset = spec.slice_offset
        current_row = self.frame.iloc[current_index]
        target_row = self.frame.iloc[current_index + 1]

        previous_day_load = self._slice_sequence(
            torch.tensor(current_row[LOAD_COLUMNS].to_numpy(dtype=float), dtype=torch.float32),
            slice_offset,
        )
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
        target = self._slice_sequence(
            torch.tensor(target_row[LOAD_COLUMNS].to_numpy(dtype=float), dtype=torch.float32),
            slice_offset,
        )
        if self.enable_augmentation:
            inputs, target = self._augment_sample(inputs, target)
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
        time_slice_stride: int = 96,
        enable_augmentation: bool = False,
        amplitude_jitter_prob: float = 0.0,
        amplitude_jitter_min: float = 1.0,
        amplitude_jitter_max: float = 1.0,
        input_noise_std: float = 0.0,
        time_mask_prob: float = 0.0,
        time_mask_min_width: int = 0,
        time_mask_max_width: int = 0,
    ) -> None:
        super().__init__()
        if time_slice_stride < 1:
            raise ValueError("time_slice_stride must be >= 1")

        expected_features = 1 + len(ProcessedLoadDataset.target_calendar_columns) + len(ProcessedLoadDataset.previous_summary_columns)
        if num_features != expected_features:
            raise ValueError(
                f"SyntheticLoadDataset expects num_features={expected_features} "
                "(load + calendar features + summary features)"
            )

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
            target_curve = curve.roll(-4) + 0.03 * torch.randn(output_steps, generator=generator)

            # Channel 0 is the target load history baseline used by the model residual output.
            feature_channels = [curve]
            latest_curve = curve

            # Channels 1-4 are repeated calendar features for the target day.
            day_of_week = (index % 7) / 6.0
            is_weekend = 1.0 if (index % 7) >= 5 else 0.0
            month = ((index % 12) + 1) / 12.0
            day_of_year = ((index % 365) + 1) / 366.0
            calendar_features = [day_of_week, is_weekend, month, day_of_year]
            for value in calendar_features:
                feature_channels.append(torch.full_like(latest_curve, value))

            # Channels 5-7 are repeated summary statistics for the previous day.
            daily_mean = float(latest_curve.mean())
            daily_std = float(latest_curve.std(unbiased=False))
            daily_range = float(latest_curve.max() - latest_curve.min())
            summary_features = [daily_mean, daily_std, daily_range]
            for value in summary_features:
                feature_channels.append(torch.full_like(latest_curve, value))

            # Final input shape per sample: (num_features, input_steps).
            inputs.append(torch.stack(feature_channels[:num_features], dim=0))
            target.append(target_curve)

        self.inputs = torch.stack(inputs).float()
        self.target = torch.stack(target).float()
        self.enable_augmentation = enable_augmentation
        self.time_slice_stride = time_slice_stride
        self.amplitude_jitter_prob = amplitude_jitter_prob
        self.amplitude_jitter_min = amplitude_jitter_min
        self.amplitude_jitter_max = amplitude_jitter_max
        self.input_noise_std = input_noise_std
        self.time_mask_prob = time_mask_prob
        self.time_mask_min_width = time_mask_min_width
        self.time_mask_max_width = time_mask_max_width
        self.summary_channel_start = 1 + len(ProcessedLoadDataset.target_calendar_columns)
        self.sample_specs = [
            _SampleSpec(sample_index=sample_index, slice_offset=offset)
            for sample_index in range(self.inputs.size(0))
            for offset in range(0, self.inputs.size(-1), self.time_slice_stride)
        ]

    def _augment_sample(self, inputs: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        augmented_inputs = inputs.clone()
        augmented_target = target.clone()
        augmented_load = augmented_inputs[0]

        if torch.rand(1).item() < self.amplitude_jitter_prob:
            scale = torch.empty(1).uniform_(self.amplitude_jitter_min, self.amplitude_jitter_max).item()
            augmented_load = augmented_load * scale
            augmented_target = augmented_target * scale

        if self.input_noise_std > 0:
            augmented_load = augmented_load + torch.randn_like(augmented_load) * self.input_noise_std
        if self.time_mask_prob > 0 and self.time_mask_max_width > 0 and torch.rand(1).item() < self.time_mask_prob:
            length = augmented_load.numel()
            min_width = max(1, min(self.time_mask_min_width, length))
            max_width = max(min_width, min(self.time_mask_max_width, length))
            width = int(torch.randint(min_width, max_width + 1, (1,)).item())
            start = int(torch.randint(0, max(1, length - width + 1), (1,)).item())
            end = start + width
            left_value = augmented_load[start - 1] if start > 0 else augmented_load[end] if end < length else augmented_load.mean()
            right_value = augmented_load[end] if end < length else left_value
            augmented_load[start:end] = torch.linspace(
                left_value,
                right_value,
                width,
                device=augmented_load.device,
                dtype=augmented_load.dtype,
            )
        augmented_inputs[0] = augmented_load
        summary_values = [
            float(augmented_load.mean()),
            float(augmented_load.std(unbiased=False)),
            float(augmented_load.max() - augmented_load.min()),
        ]
        for offset, value in enumerate(summary_values):
            augmented_inputs[self.summary_channel_start + offset] = torch.full_like(augmented_load, value)

        return augmented_inputs, augmented_target

    def __len__(self) -> int:
        return len(self.sample_specs)

    def _slice_sequence(self, sequence: torch.Tensor, offset: int) -> torch.Tensor:
        if offset == 0:
            return sequence
        tiled = torch.cat([sequence, sequence], dim=-1)
        return tiled[..., offset : offset + self.inputs.size(-1)]

    def __getitem__(self, index: int) -> SampleBatch:
        spec = self.sample_specs[index]
        inputs = self._slice_sequence(self.inputs[spec.sample_index], spec.slice_offset)
        target = self._slice_sequence(self.target[spec.sample_index], spec.slice_offset)
        if self.enable_augmentation:
            inputs, target = self._augment_sample(inputs, target)
        return SampleBatch(
            inputs=inputs,
            target=target,
        )
