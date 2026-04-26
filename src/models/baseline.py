from __future__ import annotations

import torch


def persistence_baseline(history: torch.Tensor) -> torch.Tensor:
    """Use the previous day's curve as the next-day prediction baseline."""

    if history.dim() != 3:
        raise ValueError("history must have shape [batch, channels, steps]")
    return history[:, 0, :]
