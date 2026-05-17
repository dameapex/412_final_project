from __future__ import annotations

import torch
from torch import nn


def persistence_baseline(history: torch.Tensor) -> torch.Tensor:
    """Use the previous day's curve as the next-day prediction baseline."""

    if history.dim() != 3:
        raise ValueError("history must have shape [batch, channels, steps]")
    return history[:, 0, :]


class LSTMBaselineForecaster(nn.Module):
    """Simple LSTM baseline with tensor IO shape [B, C, T] -> [B, T_out]."""

    def __init__(
        self,
        input_channels: int,
        input_steps: int = 96,
        output_steps: int = 96,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.recurrent = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.head = nn.Linear(hidden_size, output_steps)
        self.input_steps = input_steps
        self.output_steps = output_steps

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.dim() != 3:
            raise ValueError("history must have shape [batch, channels, steps]")
        if history.size(-1) != self.input_steps:
            raise ValueError(
                f"history steps must equal input_steps={self.input_steps}, "
                f"got {history.size(-1)}"
            )

        sequence_first = history.transpose(1, 2)
        _, (hidden_state, _) = self.recurrent(sequence_first)
        final_hidden = hidden_state[-1]
        return self.head(final_hidden)
