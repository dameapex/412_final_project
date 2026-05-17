from __future__ import annotations

import json
from dataclasses import asdict
from math import sqrt
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import ProcessedLoadDataset, SampleBatch, SyntheticLoadDataset
from src.data.preprocess import LOAD_COLUMNS, save_processed_outputs
from src.models.stacked_lstm import StackedLSTMForecaster
from src.models.unet_lstm import UNetLSTMForecaster
from src.utils.metrics import mae, mape, rmse
from src.utils.seed import set_seed


def _collate_fn(batch: list[SampleBatch]) -> SampleBatch:
    """Keep the training loop explicit instead of unpacking raw tuples everywhere."""

    inputs = torch.stack([item.inputs for item in batch])
    target = torch.stack([item.target for item in batch])
    return SampleBatch(inputs=inputs, target=target)


def _apply_mixup(inputs: torch.Tensor, target: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    if inputs.size(0) < 2 or alpha <= 0:
        return inputs, target
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    permutation = torch.randperm(inputs.size(0), device=inputs.device)
    mixed_inputs = lam * inputs + (1.0 - lam) * inputs[permutation]
    mixed_target = lam * target + (1.0 - lam) * target[permutation]
    return mixed_inputs, mixed_target


def load_config(config_path: str | Path) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_model(config: dict) -> nn.Module:
    input_channels = config.get("input_channels", config["num_features"])
    use_multiscale_u = config.get("use_multiscale_u", True)
    use_residual_output = config.get("use_residual_output", True)
    if use_multiscale_u:
        return UNetLSTMForecaster(
            input_channels=input_channels,
            output_steps=config["output_steps"],
            channels=config["unet_channels"],
            lstm_hidden_size=config["lstm_hidden_size"],
            lstm_layers=config["lstm_layers"],
            dropout=config["dropout"],
            use_residual_output=use_residual_output,
            use_bidirectional_skip_lstm=config.get("use_bidirectional_skip_lstm", False),
            skip_temporal_mode=str(config.get("skip_temporal_mode", "lstm")),
            skip_tcn_layers=int(config.get("skip_tcn_layers", 2)),
            skip_tcn_kernel_size=int(config.get("skip_tcn_kernel_size", 3)),
            skip_tcn_dropout=config.get("skip_tcn_dropout", None),
        )
    return StackedLSTMForecaster(
        input_channels=input_channels,
        output_steps=config["output_steps"],
        channels=config["unet_channels"],
        lstm_layers=config["lstm_layers"],
        dropout=config["dropout"],
        use_residual_output=use_residual_output,
    )


def build_training_dataset(config: dict) -> Dataset:
    expected_num_features = 1 + len(ProcessedLoadDataset.target_calendar_columns) + len(ProcessedLoadDataset.previous_summary_columns)
    config["num_features"] = expected_num_features
    time_slice_stride = config.get("train_time_slice_stride", 96)
    use_standardized_data = config.get("use_standardized_data", True)
    augmentation_enabled = config.get("train_augmentation", False)
    amplitude_jitter_prob = config.get("amplitude_jitter_prob", 0.0)
    amplitude_jitter_min = config.get("amplitude_jitter_min", 1.0)
    amplitude_jitter_max = config.get("amplitude_jitter_max", 1.0)
    input_noise_std = config.get("input_noise_std", 0.0)
    time_mask_prob = config.get("time_mask_prob", 0.0)
    time_mask_min_width = int(config.get("time_mask_min_width", 0))
    time_mask_max_width = int(config.get("time_mask_max_width", 0))

    suffix = "standardized" if use_standardized_data else "cleaned"
    processed_path = Path(f"data/processed/train_daily_{suffix}.csv")
    if processed_path.exists():
        return ProcessedLoadDataset(
            split="train",
            processed_dir=processed_path.parent,
            use_standardized=use_standardized_data,
            time_slice_stride=time_slice_stride,
            enable_augmentation=augmentation_enabled,
            amplitude_jitter_prob=amplitude_jitter_prob,
            amplitude_jitter_min=amplitude_jitter_min,
            amplitude_jitter_max=amplitude_jitter_max,
            input_noise_std=input_noise_std,
            time_mask_prob=time_mask_prob,
            time_mask_min_width=time_mask_min_width,
            time_mask_max_width=time_mask_max_width,
        )

    return SyntheticLoadDataset(
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
        time_mask_prob=time_mask_prob,
        time_mask_min_width=time_mask_min_width,
        time_mask_max_width=time_mask_max_width,
    )


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


def _build_processed_split_datasets(processed_dir: Path, use_standardized_data: bool) -> dict[str, Dataset]:
    split_datasets: dict[str, Dataset] = {}
    suffix = "standardized" if use_standardized_data else "cleaned"
    for split_name in ["train", "validation", "test"]:
        split_path = processed_dir / f"{split_name}_daily_{suffix}.csv"
        if not split_path.exists():
            continue
        split_datasets[split_name] = ProcessedLoadDataset(
            split=split_name,
            processed_dir=processed_dir,
            use_standardized=use_standardized_data,
            time_slice_stride=96,
        )
    return split_datasets


def run_training_with_split_metrics(config_path: str | Path = "configs/common.yaml") -> dict[str, float]:
    """Train once and report final RMSE on train/validation/test splits."""

    _ensure_processed_data()
    config = load_config(config_path)
    set_seed(config["seed"])
    use_standardized_data = config.get("use_standardized_data", True)
    mixup_prob = float(config.get("mixup_prob", 0.0))
    mixup_alpha = float(config.get("mixup_alpha", 0.2))

    train_dataset = build_training_dataset(config)
    config["input_channels"] = train_dataset[0].inputs.shape[0]
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=_collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    use_cosine_annealing = config.get("use_cosine_annealing", True)
    cosine_t_max = max(1, int(config.get("cosine_t_max", 10)))
    cosine_eta_min = float(config.get("cosine_eta_min", 1e-6))
    scheduler = None
    if use_cosine_annealing:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_t_max,
            eta_min=cosine_eta_min,
        )
    loss_fn = nn.MSELoss()
    denorm_std = _load_denormalization_std(Path("data/processed")) if use_standardized_data else None
    denorm_std_on_device = denorm_std.to(device) if denorm_std is not None else None
    split_datasets = _build_processed_split_datasets(
        Path("data/processed"),
        use_standardized_data=use_standardized_data,
    )
    validation_dataset = split_datasets.get("validation")
    test_dataset = split_datasets.get("test")
    for epoch in range(config["epochs"]):
        model.train()
        sum_squared_error = 0.0
        sum_squared_error_real = 0.0
        sum_absolute_percentage_error = 0.0
        total_elements = 0
        epsilon = 1e-6
        for batch in train_loader:
            inputs = batch.inputs.to(device)
            target = batch.target.to(device)

            if mixup_prob > 0 and torch.rand(1).item() < mixup_prob:
                inputs, target = _apply_mixup(inputs, target, mixup_alpha)

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
                sum_absolute_percentage_error += (error.abs() / target.abs().clamp(min=epsilon)).sum().item()
                total_elements += error.numel()

        epoch_rmse = sqrt(sum_squared_error / total_elements)
        epoch_rmse_real = (
            sqrt(sum_squared_error_real / total_elements)
            if denorm_std_on_device is not None
            else epoch_rmse
        )
        epoch_mape = (sum_absolute_percentage_error / total_elements) * 100.0
        epoch_report = [
            f"epoch={epoch + 1}",
            f"train_rmse={epoch_rmse_real:.4f}",
            f"train_mape={epoch_mape:.2f}",
        ]
        if validation_dataset is not None:
            validation_metrics = _evaluate_dataset(
                model=model,
                dataset=validation_dataset,
                batch_size=config["batch_size"],
                device=device,
                denorm_std=denorm_std,
            )
            epoch_report.append(f"validation_rmse={validation_metrics['rmse_real']:.4f}")
            epoch_report.append(f"validation_mape={validation_metrics['mape']:.2f}")
        if test_dataset is not None:
            test_metrics = _evaluate_dataset(
                model=model,
                dataset=test_dataset,
                batch_size=config["batch_size"],
                device=device,
                denorm_std=denorm_std,
            )
            epoch_report.append(f"test_rmse={test_metrics['rmse_real']:.4f}")
            epoch_report.append(f"test_mape={test_metrics['mape']:.2f}")
        epoch_report.append(f"lr={optimizer.param_groups[0]['lr']:.8f}")
        print(" ".join(epoch_report))

        # Converges around epoch 10 in current setting; decay to eta_min then keep it stable.
        if scheduler is not None and epoch < cosine_t_max:
            scheduler.step()

    if not split_datasets:
        split_datasets = {"train": train_dataset}

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
        split_rmse[split_name] = metrics["rmse_real"]
        print(f"{split_name}_rmse={metrics['rmse_real']:.4f} {split_name}_mape={metrics['mape']:.2f}")

    return split_rmse


def run_demo_training(config_path: str | Path = "configs/common.yaml") -> dict[str, float]:
    """Run a tiny synthetic training job to verify the full pipeline."""

    config = load_config(config_path)
    set_seed(config["seed"])

    dataset = build_training_dataset(config)
    config["input_channels"] = dataset[0].inputs.shape[0]
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
            inputs = batch.inputs.to(device)
            target = batch.target.to(device)

            optimizer.zero_grad()
            predictions = model(inputs)
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
