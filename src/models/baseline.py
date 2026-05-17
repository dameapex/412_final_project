from __future__ import annotations

import torch
from torch import nn


def persistence_baseline(history: torch.Tensor) -> torch.Tensor:
    """Use the previous day's curve as the next-day prediction baseline."""

    if history.dim() != 3:
        raise ValueError("history must have shape [batch, channels, steps]")
    return history[:, 0, :]


def _validate_history(history: torch.Tensor, input_steps: int) -> None:
    if history.dim() != 3:
        raise ValueError("history must have shape [batch, channels, steps]")
    if history.size(-1) != input_steps:
        raise ValueError(
            f"history steps must equal input_steps={input_steps}, "
            f"got {history.size(-1)}"
        )
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
        _validate_history(history, self.input_steps)

        sequence_first = history.transpose(1, 2)
        _, (hidden_state, _) = self.recurrent(sequence_first)
        final_hidden = hidden_state[-1]
        return self.head(final_hidden)


class _CausalChomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return inputs
        return inputs[..., :-self.chomp_size]


class _TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            _CausalChomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            _CausalChomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.residual = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.net(inputs) + self.residual(inputs))


class TCNBaselineForecaster(nn.Module):
    """Simple TCN baseline with tensor IO shape [B, C, T] -> [B, T]."""

    def __init__(
        self,
        input_channels: int,
        input_steps: int = 96,
        output_steps: int = 96,
        channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        channels = channels or [32, 32, 32]
        blocks = []
        current_channels = input_channels
        for index, next_channels in enumerate(channels):
            blocks.append(
                _TemporalBlock(
                    in_channels=current_channels,
                    out_channels=next_channels,
                    kernel_size=kernel_size,
                    dilation=2 ** index,
                    dropout=dropout,
                )
            )
            current_channels = next_channels

        self.network = nn.Sequential(*blocks)
        self.head = nn.Conv1d(current_channels, 1, kernel_size=1)
        self.input_steps = input_steps
        self.output_steps = output_steps

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        _validate_history(history, self.input_steps)
        encoded = self.network(history)
        prediction = self.head(encoded).squeeze(1)
        if prediction.size(-1) != self.output_steps:
            raise ValueError(
                f"prediction steps must equal output_steps={self.output_steps}, got {prediction.size(-1)}"
            )
        return prediction
