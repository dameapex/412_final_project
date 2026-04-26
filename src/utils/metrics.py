from __future__ import annotations

import torch


def mae(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(predictions - targets))


def rmse(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((predictions - targets) ** 2))


def mape(predictions: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    safe_targets = torch.clamp(torch.abs(targets), min=epsilon)
    return torch.mean(torch.abs((predictions - targets) / safe_targets)) * 100.0
