from __future__ import annotations

import json
from math import sqrt
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import ProcessedLoadDataset, SampleBatch, SyntheticLoadDataset
from src.data.preprocess import LOAD_COLUMNS, save_processed_outputs
from src.models.baseline import LSTMBaselineForecaster
from src.utils.seed import set_seed


def _collate_fn(batch: list[SampleBatch]) -> SampleBatch:
    inputs = torch.stack([item.inputs for item in batch])
    target = torch.stack([item.target for item in batch])
    return SampleBatch(inputs=inputs, target=target)


def load_config(config_path: str | Path) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _ensure_processed_data() -> None:
    processed_train_path = Path("data/processed/train_daily_standardized.csv")
    if processed_train_path.exists():
        return

    required_raw_files = [
        Path("train_data.xlsx"),
        Path("validation_data.xlsx"),
        Path("test_data_to_students.xlsx"),
    ]
    if not all(path.exists() for path in required_raw_files):
        return

    save_processed_outputs(base_dir=Path.cwd())


def _load_denormalization_std(processed_dir: Path = Path("data/processed")) -> torch.Tensor | None:
    stats_path = processed_dir / "standardization_stats.json"
    if not stats_path.exists():
        return None

    with stats_path.open("r", encoding="utf-8") as handle:
        stats = json.load(handle)

    slot_std = [float(stats[column]["std"]) for column in LOAD_COLUMNS if column in stats]
    if len(slot_std) != len(LOAD_COLUMNS):
        return None
    return torch.tensor(slot_std, dtype=torch.float32)


def _build_split_datasets(config: dict) -> dict[str, Dataset]:
    processed_dir = Path("data/processed")
    expected_num_features = 1 + len(ProcessedLoadDataset.target_calendar_columns) + len(ProcessedLoadDataset.previous_summary_columns)
    config["num_features"] = expected_num_features
    use_standardized_data = config.get("use_standardized_data", True)
    time_slice_stride = config.get("train_time_slice_stride", 96)
    augmentation_enabled = config.get("train_augmentation", False)
    amplitude_jitter_prob = config.get("amplitude_jitter_prob", 0.0)
    amplitude_jitter_min = config.get("amplitude_jitter_min", 1.0)
    amplitude_jitter_max = config.get("amplitude_jitter_max", 1.0)
    input_noise_std = config.get("input_noise_std", 0.0)
    suffix = "standardized" if use_standardized_data else "cleaned"
    split_datasets: dict[str, Dataset] = {}
    for split_name in ["train", "validation", "test"]:
        split_path = processed_dir / f"{split_name}_daily_{suffix}.csv"
        if split_path.exists():
            is_train = split_name == "train"
            split_datasets[split_name] = ProcessedLoadDataset(
                split=split_name,
                processed_dir=processed_dir,
                use_standardized=use_standardized_data,
                time_slice_stride=time_slice_stride if is_train else 96,
                enable_augmentation=is_train and augmentation_enabled,
                amplitude_jitter_prob=amplitude_jitter_prob,
                amplitude_jitter_min=amplitude_jitter_min,
                amplitude_jitter_max=amplitude_jitter_max,
                input_noise_std=input_noise_std,
            )

    if split_datasets:
        return split_datasets

    split_datasets["train"] = SyntheticLoadDataset(
        num_samples=256,
        input_steps=config["input_steps"],
        output_steps=config["output_steps"],
        num_features=expected_num_features,
        seed=config["seed"],
        time_slice_stride=time_slice_stride,
        enable_augmentation=augmentation_enabled,
        amplitude_jitter_prob=amplitude_jitter_prob,
        amplitude_jitter_min=amplitude_jitter_min,
        amplitude_jitter_max=amplitude_jitter_max,
        input_noise_std=input_noise_std,
    )
    return split_datasets


def _evaluate_dataset(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
    denorm_std: torch.Tensor | None = None,
) -> dict[str, float]:
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_fn,
    )

    model.eval()
    total_elements = 0
    sum_abs_error = 0.0
    sum_squared_error = 0.0
    sum_squared_error_real = 0.0
    sum_absolute_percentage_error = 0.0
    epsilon = 1e-6
    denorm_std = denorm_std.to(device) if denorm_std is not None else None

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch.inputs.to(device)
            target = batch.target.to(device)
            predictions = model(inputs)
            error = predictions - target

            total_elements += error.numel()
            sum_abs_error += error.abs().sum().item()
            sum_squared_error += (error ** 2).sum().item()
            if denorm_std is not None:
                error_real = error * denorm_std.view(1, -1)
                sum_squared_error_real += (error_real ** 2).sum().item()
            sum_absolute_percentage_error += (error.abs() / target.abs().clamp(min=epsilon)).sum().item()

    mse_value = sum_squared_error / total_elements
    rmse_real = sqrt(sum_squared_error_real / total_elements) if denorm_std is not None else sqrt(mse_value)
    return {
        "loss": mse_value,
        "mae": sum_abs_error / total_elements,
        "rmse": sqrt(mse_value),
        "rmse_real": rmse_real,
        "mape": (sum_absolute_percentage_error / total_elements) * 100.0,
    }


def run_lstm_baseline_training(config_path: str | Path = "configs/common.yaml") -> dict[str, float]:
    """Train and evaluate LSTM baseline, reporting final RMSE by split."""

    _ensure_processed_data()
    config = load_config(config_path)
    set_seed(config["seed"])
    use_standardized_data = config.get("use_standardized_data", True)

    split_datasets = _build_split_datasets(config)
    train_dataset = split_datasets["train"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=_collate_fn,
    )

    input_channels = train_dataset[0].inputs.shape[0]
    baseline_hidden_size = config.get("baseline_lstm_hidden_size", config["lstm_hidden_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMBaselineForecaster(
        input_channels=input_channels,
        input_steps=config["input_steps"],
        output_steps=config["output_steps"],
        hidden_size=baseline_hidden_size,
        num_layers=1,
        dropout=config["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_fn = nn.MSELoss()
    denorm_std = _load_denormalization_std(Path("data/processed")) if use_standardized_data else None
    denorm_std_on_device = denorm_std.to(device) if denorm_std is not None else None
    validation_dataset = split_datasets.get("validation")

    for epoch in range(config["epochs"]):
        model.train()
        sum_squared_error = 0.0
        sum_squared_error_real = 0.0
        total_elements = 0
        for batch in train_loader:
            inputs = batch.inputs.to(device)
            target = batch.target.to(device)

            optimizer.zero_grad()
            predictions = model(inputs)
            loss = loss_fn(predictions, target)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                error = predictions.detach() - target
                sum_squared_error += (error ** 2).sum().item()
                if denorm_std_on_device is not None:
                    error_real = error * denorm_std_on_device.view(1, -1)
                    sum_squared_error_real += (error_real ** 2).sum().item()
                total_elements += error.numel()

        epoch_rmse = sqrt(sum_squared_error / total_elements)
        epoch_rmse_real = (
            sqrt(sum_squared_error_real / total_elements)
            if denorm_std_on_device is not None
            else epoch_rmse
        )
        if validation_dataset is not None:
            validation_metrics = _evaluate_dataset(
                model=model,
                dataset=validation_dataset,
                batch_size=config["batch_size"],
                device=device,
                denorm_std=denorm_std,
            )
            print(
                f"epoch={epoch + 1} "
                f"train_rmse_z={epoch_rmse:.4f} "
                f"train_rmse_real={epoch_rmse_real:.4f} "
                f"validation_rmse_z={validation_metrics['rmse']:.4f} "
                f"validation_rmse_real={validation_metrics['rmse_real']:.4f}"
            )
        else:
            print(
                f"epoch={epoch + 1} "
                f"train_rmse_z={epoch_rmse:.4f} "
                f"train_rmse_real={epoch_rmse_real:.4f}"
            )

    split_rmse: dict[str, float] = {}
    print("Final RMSE by split:")
    for split_name in ["train", "validation", "test"]:
        dataset = split_datasets.get(split_name)
        if dataset is None:
            continue
        metrics = _evaluate_dataset(
            model=model,
            dataset=dataset,
            batch_size=config["batch_size"],
            device=device,
            denorm_std=denorm_std,
        )
        split_rmse[split_name] = metrics["rmse"]
        print(f"{split_name}_rmse_z={metrics['rmse']:.4f} {split_name}_rmse_real={metrics['rmse_real']:.4f}")

    return split_rmse


if __name__ == "__main__":
    result = run_lstm_baseline_training()
    print(result)