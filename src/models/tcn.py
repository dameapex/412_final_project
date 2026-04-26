from __future__ import annotations

import torch
from torch import nn


class TemporalBlock(nn.Module):
    """Basic residual TCN block used for 1D time-series encoding."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (3 - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(inputs)
        outputs = self.net(inputs)
        outputs = outputs[..., : inputs.size(-1)]
        return outputs + residual


class TCNForecaster(nn.Module):
    """Forecast next-day 96-point load using history and context features."""

    def __init__(
        self,
        input_channels: int = 1,
        context_dim: int = 8,
        output_steps: int = 96,
        channels: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        channels = channels or [32, 64, 64]

        blocks = []
        current_channels = input_channels
        for level, next_channels in enumerate(channels):
            blocks.append(
                TemporalBlock(
                    in_channels=current_channels,
                    out_channels=next_channels,
                    dilation=2**level,
                    dropout=dropout,
                )
            )
            current_channels = next_channels

        self.encoder = nn.Sequential(*blocks)
        self.context_mlp = nn.Sequential(
            nn.Linear(context_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(current_channels + 32, 128),
            nn.ReLU(),
            nn.Linear(128, output_steps),
        )

    def forward(self, history: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(history)
        pooled = encoded.mean(dim=-1)
        context_embed = self.context_mlp(context)
        fused = torch.cat([pooled, context_embed], dim=1)
        return self.head(fused)
