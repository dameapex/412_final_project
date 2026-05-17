from __future__ import annotations

import torch
from torch import nn


def _attention_weighted_concat(tensors: list[torch.Tensor], dim: int = 1) -> torch.Tensor:
    """Concatenate tensors after per-source attention weighting.

    Attention scores are computed per sample using global average magnitude of
    each source tensor, then normalized with softmax across sources.
    """

    if len(tensors) == 1:
        return tensors[0]

    source_scores = torch.stack([tensor.abs().mean(dim=(1, 2)) for tensor in tensors], dim=1)
    source_weights = torch.softmax(source_scores, dim=1)
    weighted_tensors = [
        tensor * source_weights[:, source_index].view(-1, 1, 1)
        for source_index, tensor in enumerate(tensors)
    ]
    return torch.cat(weighted_tensors, dim=dim)


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
        merged = _attention_weighted_concat([upsampled, skip], dim=1)
        return self.fuse(merged)


class _CausalChomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return inputs
        return inputs[..., :-self.chomp_size]


class _SkipTemporalTCNBlock(nn.Module):
    """Residual temporal conv block used when skip enhancement mode is TCN."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            _CausalChomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            _CausalChomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.net(inputs) + inputs)


class SkipTemporalBlock(nn.Module):
    """Enhance skip features with LSTM, TCN, or sequential TCN->LSTM."""

    def __init__(
        self,
        channels: int,
        hidden_size: int,
        dropout: float,
        lstm_layers: int,
        bidirectional: bool,
        mode: str = "lstm",
        tcn_layers: int = 2,
        tcn_kernel_size: int = 3,
        tcn_dropout: float | None = None,
    ) -> None:
        super().__init__()
        self.mode = mode.lower()
        if self.mode == "lstm":
            self.lstm = nn.LSTM(
                input_size=channels,
                hidden_size=hidden_size,
                num_layers=lstm_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if lstm_layers > 1 else 0.0,
            )
            projection_in_features = hidden_size * (2 if bidirectional else 1)
            self.projection = nn.Linear(projection_in_features, channels)
            self.tcn = None
        elif self.mode == "tcn":
            if tcn_layers < 1:
                raise ValueError("tcn_layers must be >= 1 when skip mode is tcn")
            tcn_effective_dropout = dropout if tcn_dropout is None else tcn_dropout
            self.tcn = nn.Sequential(
                *[
                    _SkipTemporalTCNBlock(
                        channels=channels,
                        kernel_size=tcn_kernel_size,
                        dilation=2**layer_index,
                        dropout=tcn_effective_dropout,
                    )
                    for layer_index in range(tcn_layers)
                ]
            )
            self.lstm = None
            self.projection = None
        elif self.mode == "tcn_lstm":
            if tcn_layers < 1:
                raise ValueError("tcn_layers must be >= 1 when skip mode is tcn_lstm")
            tcn_effective_dropout = dropout if tcn_dropout is None else tcn_dropout
            self.tcn = nn.Sequential(
                *[
                    _SkipTemporalTCNBlock(
                        channels=channels,
                        kernel_size=tcn_kernel_size,
                        dilation=2**layer_index,
                        dropout=tcn_effective_dropout,
                    )
                    for layer_index in range(tcn_layers)
                ]
            )
            self.lstm = nn.LSTM(
                input_size=channels,
                hidden_size=hidden_size,
                num_layers=lstm_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if lstm_layers > 1 else 0.0,
            )
            projection_in_features = hidden_size * (2 if bidirectional else 1)
            self.projection = nn.Linear(projection_in_features, channels)
        else:
            raise ValueError(f"Unsupported skip temporal mode: {mode}")

    def forward(self, skip: torch.Tensor) -> torch.Tensor:
        if self.mode == "lstm":
            sequence_first = skip.transpose(1, 2)
            recurrent_outputs, _ = self.lstm(sequence_first)
            projected = self.projection(recurrent_outputs).transpose(1, 2)
            return projected + skip
        if self.mode == "tcn_lstm":
            tcn_skip = self.tcn(skip)
            sequence_first = tcn_skip.transpose(1, 2)
            recurrent_outputs, _ = self.lstm(sequence_first)
            projected = self.projection(recurrent_outputs).transpose(1, 2)
            return projected + tcn_skip
        return self.tcn(skip)


class UNetLSTMForecaster(nn.Module):
    """U-shaped 1D forecaster with LSTM-enhanced skip connections.

    The model consumes one multi-channel input tensor (for example: previous
    day's load, weather, and calendar features) and predicts a single target
    load sequence.

    Encoder skip features are first passed through per-scale LSTM blocks and the
    temporally enhanced skips are then fused by matching decoder stages. The
    final output is produced as a residual correction on top of the previous
    day's load curve (channel 0 of the input).

    Decoder stages use dense internal skip connections: each stage receives all
    preceding decoder outputs (aligned on the temporal axis), not only the
    output of the immediately previous stage. Concatenation in decoder fusion
    uses source-attention weighting for all contributing inputs.

    Residual formulation can be toggled for ablation:
    - use_residual_output=True: output = baseline + residual_delta
    - use_residual_output=False: output = decoder forecast directly
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_steps: int = 96,
        channels: list[int] | None = None,
        lstm_hidden_size: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.1,
        use_residual_output: bool = True,
        use_bidirectional_skip_lstm: bool = False,
        skip_temporal_mode: str = "lstm",
        skip_tcn_layers: int = 2,
        skip_tcn_kernel_size: int = 3,
        skip_tcn_dropout: float | None = None,
    ) -> None:
        super().__init__()
        channels = channels or [16, 32, 64]
        if len(channels) < 1:
            raise ValueError("UNetLSTMForecaster requires at least one channel stage")

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
                    bidirectional=use_bidirectional_skip_lstm,
                    mode=skip_temporal_mode,
                    tcn_layers=skip_tcn_layers,
                    tcn_kernel_size=skip_tcn_kernel_size,
                    tcn_dropout=skip_tcn_dropout,
                )
                for stage_channels in channels
            ]
        )

        decoder_stages = []
        reversed_channels = list(reversed(channels))
        for stage_index, skip_channels in enumerate(reversed_channels):
            if stage_index == 0:
                stage_input_channels = channels[-1]
            else:
                stage_input_channels = sum(reversed_channels[:stage_index])
            decoder_stages.append(DecoderStage(stage_input_channels, skip_channels, skip_channels, dropout))
        self.decoder_stages = nn.ModuleList(decoder_stages)
        self.output_head = nn.Conv1d(channels[0], 1, kernel_size=1)
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

        decoder_outputs: list[torch.Tensor] = []
        reversed_temporal_skips = list(reversed(temporal_skips))
        for stage_index, (stage, skip) in enumerate(zip(self.decoder_stages, reversed_temporal_skips)):
            if stage_index == 0:
                stage_inputs = bottleneck
            else:
                target_length = decoder_outputs[-1].size(-1)
                aligned_previous_outputs = []
                for previous_output in decoder_outputs:
                    if previous_output.size(-1) != target_length:
                        previous_output = nn.functional.interpolate(
                            previous_output,
                            size=target_length,
                            mode="linear",
                            align_corners=False,
                        )
                    aligned_previous_outputs.append(previous_output)
                stage_inputs = _attention_weighted_concat(aligned_previous_outputs, dim=1)
            decoded = stage(stage_inputs, skip)
            decoder_outputs.append(decoded)

        decoder_forecast = self.output_head(decoded).squeeze(1)
        if decoder_forecast.size(-1) != self.output_steps:
            decoder_forecast = nn.functional.interpolate(
                decoder_forecast.unsqueeze(1),
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

            residual_delta = self.residual_refine(decoder_forecast)
            outputs = baseline + residual_delta
        else:
            outputs = decoder_forecast
        if squeeze_batch:
            return outputs.squeeze(0)
        return outputs
