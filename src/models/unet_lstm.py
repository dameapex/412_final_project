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


class ScaleRecurrentHead(nn.Module):
    """Run recurrent modeling on one temporal scale and emit a full-resolution forecast."""

    def __init__(self, channels: int, hidden_size: int, output_steps: int, dropout: float, lstm_layers: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=channels,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.projection = nn.Linear(hidden_size, 1)
        self.output_steps = output_steps

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_first = features.transpose(1, 2)
        recurrent_outputs, _ = self.lstm(sequence_first)
        scale_signal = self.projection(recurrent_outputs).transpose(1, 2)
        pooled_descriptor = recurrent_outputs.mean(dim=1)
        upsampled_signal = nn.functional.interpolate(
            scale_signal,
            size=self.output_steps,
            mode="linear",
            align_corners=False,
        )
        return upsampled_signal.squeeze(1), pooled_descriptor


class UNetLSTMForecaster(nn.Module):
    """U-shaped 1D forecaster with multi-scale recurrent prediction heads.

    Each downsampled temporal scale is sent through its own LSTM head so the
    model can make scale-specific forecasts instead of relying only on the
    coarsest bottleneck representation. A learned fusion module then combines the
    scale forecasts with the decoder forecast. The final output is produced as a
    residual correction on top of the previous day's load curve.
    """

    def __init__(
        self,
        input_channels: int = 1,
        context_dim: int = 8,
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

        self.context_projection = nn.Sequential(
            nn.Linear(context_dim, channels[-1]),
            nn.ReLU(),
        )
        self.bottleneck_conv = ConvBlock(channels[-1], channels[-1], dropout)

        self.scale_heads = nn.ModuleList(
            [
                ScaleRecurrentHead(
                    channels=stage_channels,
                    hidden_size=lstm_hidden_size,
                    output_steps=output_steps,
                    dropout=dropout,
                    lstm_layers=lstm_layers,
                )
                for stage_channels in channels
            ]
        )

        decoder_stages = []
        decoder_specs = [
            (channels[-1], channels[-1], channels[-1]),
            (channels[-1], channels[-2], channels[-2]),
            (channels[-2], channels[-3], channels[-3]),
        ]
        for in_channels, skip_channels, out_channels in decoder_specs:
            decoder_stages.append(DecoderStage(in_channels, skip_channels, out_channels, dropout))
        self.decoder_stages = nn.ModuleList(decoder_stages)
        self.output_head = nn.Conv1d(channels[0], 1, kernel_size=1)
        fusion_input_dim = (len(channels) * lstm_hidden_size) + context_dim
        self.scale_weight_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, len(channels) + 1),
        )
        self.residual_refine = nn.Sequential(
            nn.Linear(output_steps + context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_steps),
        )
        self.output_steps = output_steps

    def forward(self, history: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        scale_features: list[torch.Tensor] = []
        encoded = history
        for stage in self.encoder_stages:
            skip, encoded = stage(encoded)
            skips.append(skip)
            scale_features.append(encoded)

        bottleneck = self.bottleneck_conv(encoded)
        context_bias = self.context_projection(context).unsqueeze(-1)
        bottleneck = bottleneck + context_bias

        decoded = bottleneck
        for stage, skip in zip(self.decoder_stages, reversed(skips)):
            decoded = stage(decoded, skip)

        decoder_forecast = self.output_head(decoded).squeeze(1)
        if decoder_forecast.size(-1) != self.output_steps:
            decoder_forecast = nn.functional.interpolate(
                decoder_forecast.unsqueeze(1),
                size=self.output_steps,
                mode="linear",
                align_corners=False,
            ).squeeze(1)

        scale_forecasts = []
        scale_descriptors = []
        for head, features in zip(self.scale_heads, scale_features):
            forecast, descriptor = head(features)
            scale_forecasts.append(forecast)
            scale_descriptors.append(descriptor)

        fusion_features = torch.cat(scale_descriptors + [context], dim=1)
        fusion_weights = torch.softmax(self.scale_weight_mlp(fusion_features), dim=1)

        combined_forecasts = scale_forecasts + [decoder_forecast]
        fused = sum(
            fusion_weights[:, index].unsqueeze(1) * forecast
            for index, forecast in enumerate(combined_forecasts)
        )
        baseline = history[:, 0, : self.output_steps]
        residual_input = torch.cat([fused, context], dim=1)
        residual_delta = self.residual_refine(residual_input)
        return baseline + residual_delta
