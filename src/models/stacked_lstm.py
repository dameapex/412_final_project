from __future__ import annotations

import torch
from torch import nn


class StackedLSTMBlock(nn.Module):
    """A single sequential BiLSTM block used in the pure LSTM ablation model."""

    def __init__(self, input_size: int, hidden_size: int, lstm_layers: int, dropout: float) -> None:
        super().__init__()
        self.output_size = hidden_size * 2
        self.recurrent = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(self.output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.recurrent(sequence)
        outputs = self.norm(outputs)
        return self.dropout(outputs)


class StackedLSTMForecaster(nn.Module):
    """Pure BiLSTM forecaster with channel-matched stacked LSTM blocks.

    The architecture ablates the multi-scale U-shaped path by directly stacking
    LSTM blocks whose hidden sizes follow the provided channel list.
    """

    def __init__(
        self,
        input_channels: int,
        output_steps: int = 96,
        channels: list[int] | None = None,
        lstm_layers: int = 1,
        dropout: float = 0.1,
        use_residual_output: bool = True,
    ) -> None:
        super().__init__()
        channels = channels or [16, 32, 64]
        if len(channels) < 1:
            raise ValueError("StackedLSTMForecaster requires at least one stage in channels")

        blocks = []
        current_size = input_channels
        for hidden_size in channels:
            block = StackedLSTMBlock(
                input_size=current_size,
                hidden_size=hidden_size,
                lstm_layers=lstm_layers,
                dropout=dropout,
            )
            blocks.append(block)
            current_size = block.output_size
        self.blocks = nn.ModuleList(blocks)

        self.output_projection = nn.Linear(current_size, 1)
        self.residual_refine = nn.Sequential(
            nn.Linear(output_steps, 128),
            nn.ReLU(),
            nn.Linear(128, output_steps),
        )
        self.output_steps = output_steps
        self.use_residual_output = use_residual_output

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        squeeze_batch = False
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(0)
            squeeze_batch = True
        elif inputs.dim() != 3:
            raise ValueError("inputs must have shape (features, time) or (batch, features, time)")

        # LSTM expects [B, T, C].
        sequence = inputs.transpose(1, 2)
        for block in self.blocks:
            sequence = block(sequence)

        forecast = self.output_projection(sequence).squeeze(-1)
        if forecast.size(-1) != self.output_steps:
            forecast = nn.functional.interpolate(
                forecast.unsqueeze(1),
                size=self.output_steps,
                mode="linear",
                align_corners=False,
            ).squeeze(1)

        if self.use_residual_output:
            baseline = inputs[:, 0, :]
            if baseline.size(-1) != self.output_steps:
                baseline = nn.functional.interpolate(
                    baseline.unsqueeze(1),
                    size=self.output_steps,
                    mode="linear",
                    align_corners=False,
                ).squeeze(1)
            outputs = baseline + self.residual_refine(forecast)
        else:
            outputs = forecast

        if squeeze_batch:
            return outputs.squeeze(0)
        return outputs
