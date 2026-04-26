from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Two-layer 1D convolution block used on both encoder and decoder paths."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class EncoderStage(nn.Module):
    """Extract local temporal features and then downsample the sequence length."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.features = ConvBlock(in_channels, out_channels, dropout)
        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.features(inputs)
        downsampled = self.pool(skip)
        return skip, downsampled


class DecoderStage(nn.Module):
    """Upsample and fuse bottleneck features with matching skip features."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        self.fuse = ConvBlock(out_channels + skip_channels, out_channels, dropout)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        upsampled = self.upsample(inputs)
        if upsampled.size(-1) != skip.size(-1):
            upsampled = upsampled[..., : skip.size(-1)]
        merged = torch.cat([upsampled, skip], dim=1)
        return self.fuse(merged)


class SkipTemporalBlock(nn.Module):
    """Enhance skip features with an LSTM before decoder fusion."""

    def __init__(self, channels: int, hidden_size: int, dropout: float, lstm_layers: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=channels,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.projection = nn.Linear(hidden_size, channels)

    def forward(self, skip: torch.Tensor) -> torch.Tensor:
        sequence_first = skip.transpose(1, 2)
        recurrent_outputs, _ = self.lstm(sequence_first)
        projected = self.projection(recurrent_outputs).transpose(1, 2)
        return projected + skip


class UNetLSTMForecaster(nn.Module):
    """U-shaped 1D forecaster with LSTM-enhanced skip connections.

    The model consumes one multi-channel input tensor (for example: previous
    day's load, weather, and calendar features) and predicts a single target
    load sequence.

    Encoder skip features are first passed through per-scale LSTM blocks and the
    temporally enhanced skips are then fused by matching decoder stages. The
    final output is produced as a residual correction on top of the previous
    day's load curve (channel 0 of the input).
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_steps: int = 96,
        channels: list[int] | None = None,
        lstm_hidden_size: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        channels = channels or [16, 32, 64]
        if len(channels) < 2:
            raise ValueError("UNetLSTMForecaster requires at least two channel stages")

        encoder_stages = []
        current_channels = input_channels
        for next_channels in channels:
            encoder_stages.append(EncoderStage(current_channels, next_channels, dropout))
            current_channels = next_channels
        self.encoder_stages = nn.ModuleList(encoder_stages)

        self.bottleneck_conv = ConvBlock(channels[-1], channels[-1], dropout)

        self.skip_temporal_blocks = nn.ModuleList(
            [
                SkipTemporalBlock(
                    channels=stage_channels,
                    hidden_size=lstm_hidden_size,
                    dropout=dropout,
                    lstm_layers=lstm_layers,
                )
                for stage_channels in channels
            ]
        )

        decoder_stages = []
        decoder_input_channels = channels[-1]
        for skip_channels in reversed(channels):
            decoder_stages.append(DecoderStage(decoder_input_channels, skip_channels, skip_channels, dropout))
            decoder_input_channels = skip_channels
        self.decoder_stages = nn.ModuleList(decoder_stages)
        self.output_head = nn.Conv1d(channels[0], 1, kernel_size=1)
        self.residual_refine = nn.Sequential(
            nn.Linear(output_steps, 128),
            nn.ReLU(),
            nn.Linear(128, output_steps),
        )
        self.output_steps = output_steps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        squeeze_batch = False
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(0)
            squeeze_batch = True
        elif inputs.dim() != 3:
            raise ValueError("inputs must have shape (features, time) or (batch, features, time)")

        skips: list[torch.Tensor] = []
        encoded = inputs
        for stage in self.encoder_stages:
            skip, encoded = stage(encoded)
            skips.append(skip)

        bottleneck = self.bottleneck_conv(encoded)

        temporal_skips = [
            block(skip)
            for block, skip in zip(self.skip_temporal_blocks, skips)
        ]

        decoded = bottleneck
        for stage, skip in zip(self.decoder_stages, reversed(temporal_skips)):
            decoded = stage(decoded, skip)

        decoder_forecast = self.output_head(decoded).squeeze(1)
        if decoder_forecast.size(-1) != self.output_steps:
            decoder_forecast = nn.functional.interpolate(
                decoder_forecast.unsqueeze(1),
                size=self.output_steps,
                mode="linear",
                align_corners=False,
            ).squeeze(1)

        baseline = inputs[:, 0, :]
        if baseline.size(-1) != self.output_steps:
            baseline = nn.functional.interpolate(
                baseline.unsqueeze(1),
                size=self.output_steps,
                mode="linear",
                align_corners=False,
            ).squeeze(1)

        residual_input = decoder_forecast
        residual_delta = self.residual_refine(residual_input)
        outputs = baseline + residual_delta
        if squeeze_batch:
            return outputs.squeeze(0)
        return outputs
