from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass
class SampleBatch:
    """Container used by the training loop for readability."""

    history: torch.Tensor
    context: torch.Tensor
    target: torch.Tensor


class SyntheticLoadDataset(Dataset):
    """Small synthetic dataset for smoke tests and pipeline validation.

    Each sample uses one day's 96-point load curve plus a small context vector
    to predict the next day's 96-point curve.
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

        history = []
        context = []
        target = []
        for index in range(num_samples):
            phase = index / 14.0
            daily_shift = torch.rand(1, generator=generator).item() * 0.2
            noise = 0.05 * torch.randn(input_steps, generator=generator)
            curve = base_curve + daily_shift + 0.2 * torch.sin(time_axis + phase) + noise
            next_curve = curve.roll(-4) + 0.03 * torch.randn(output_steps, generator=generator)

            history.append(curve.unsqueeze(0))
            context.append(torch.rand(num_features, generator=generator))
            target.append(next_curve)

        self.history = torch.stack(history).float()
        self.context = torch.stack(context).float()
        self.target = torch.stack(target).float()

    def __len__(self) -> int:
        return self.history.size(0)

    def __getitem__(self, index: int) -> SampleBatch:
        return SampleBatch(
            history=self.history[index],
            context=self.context[index],
            target=self.target[index],
        )
