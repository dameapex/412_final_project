from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import yaml

from src.models.unet_lstm import UNetLSTMForecaster


def _load_config(config_path: str | Path) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _build_demo_features(num_features: int, input_steps: int) -> torch.Tensor:
    """Create one synthetic multi-channel day sample with shape (1, F, T)."""

    time_axis = torch.linspace(0, 2 * torch.pi, input_steps)
    load_curve = 0.9 + 0.25 * torch.sin(time_axis) + 0.1 * torch.cos(2 * time_axis)
    channels = [load_curve]
    for feature_index in range(max(0, num_features - 1)):
        freq = 1.0 + (feature_index % 3)
        phase = 0.2 * feature_index
        seasonal = torch.sin(freq * time_axis + phase)
        trend = (feature_index + 1) * 0.03 * torch.linspace(0, 1, input_steps)
        channels.append(seasonal + trend)
    return torch.stack(channels[:num_features], dim=0).unsqueeze(0)


def export_demo_submission(
    output_path: str | Path = "outputs/submissions/demo_submission.xlsx",
    config_path: str | Path = "configs/common.yaml",
) -> Path:
    """Create a placeholder submission file using the UNet-LSTM forecaster.

    Replace this synthetic example with the real test-data inference pipeline later.
    """

    config = _load_config(config_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = UNetLSTMForecaster(
        input_channels=config["num_features"],
        output_steps=config["output_steps"],
        channels=config["unet_channels"],
        lstm_hidden_size=config["lstm_hidden_size"],
        lstm_layers=config["lstm_layers"],
        dropout=config["dropout"],
    )
    model.eval()

    demo_inputs = _build_demo_features(config["num_features"], config["input_steps"])
    with torch.no_grad():
        prediction = model(demo_inputs).squeeze(0).cpu().numpy()

    frame = pd.DataFrame(
        {
            "date": ["2024-06-02"],
            **{f"slot_{index + 1:02d}": [float(prediction[index])] for index in range(config["output_steps"])},
        }
    )
    frame.to_excel(output_path, index=False)
    return output_path


if __name__ == "__main__":
    path = export_demo_submission()
    print(path)
