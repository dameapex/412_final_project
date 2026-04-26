from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import SampleBatch, SyntheticLoadDataset
from src.models.unet_lstm import UNetLSTMForecaster
from src.utils.metrics import mae, mape, rmse
from src.utils.seed import set_seed


def _collate_fn(batch: list[SampleBatch]) -> SampleBatch:
    """Keep the training loop explicit instead of unpacking raw tuples everywhere."""

    history = torch.stack([item.history for item in batch])
    context = torch.stack([item.context for item in batch])
    target = torch.stack([item.target for item in batch])
    return SampleBatch(history=history, context=context, target=target)


def load_config(config_path: str | Path) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_model(config: dict) -> nn.Module:
    model_name = config.get("model_name", "unet_lstm")
    if model_name == "unet_lstm":
        return UNetLSTMForecaster(
            input_channels=1,
            context_dim=config["num_features"],
            output_steps=config["output_steps"],
            channels=config["unet_channels"],
            lstm_hidden_size=config["lstm_hidden_size"],
            lstm_layers=config["lstm_layers"],
            dropout=config["dropout"],
        )

    raise ValueError(f"Unsupported model_name: {model_name}")


def run_demo_training(config_path: str | Path = "configs/common.yaml") -> dict[str, float]:
    """Run a tiny synthetic training job to verify the full pipeline."""

    config = load_config(config_path)
    set_seed(config["seed"])

    dataset = SyntheticLoadDataset(
        num_samples=256,
        input_steps=config["input_steps"],
        output_steps=config["output_steps"],
        num_features=config["num_features"],
        seed=config["seed"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=_collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_fn = nn.MSELoss()
    model.train()

    last_metrics: dict[str, float] = {}
    for epoch in range(config["epochs"]):
        epoch_loss = 0.0
        for batch in dataloader:
            history = batch.history.to(device)
            context = batch.context.to(device)
            target = batch.target.to(device)

            optimizer.zero_grad()
            predictions = model(history, context)
            loss = loss_fn(predictions, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        with torch.no_grad():
            batch_mae = mae(predictions, target).item()
            batch_rmse = rmse(predictions, target).item()
            batch_mape = mape(predictions, target).item()

        last_metrics = {
            "epoch": float(epoch + 1),
            "loss": epoch_loss / len(dataloader),
            "mae": batch_mae,
            "rmse": batch_rmse,
            "mape": batch_mape,
        }
        print(f"epoch={epoch + 1} loss={last_metrics['loss']:.4f} mae={batch_mae:.4f} rmse={batch_rmse:.4f} mape={batch_mape:.2f}")

    return last_metrics


if __name__ == "__main__":
    results = run_demo_training()
    print(asdict if False else results)
