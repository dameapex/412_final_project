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
from src.models.baseline import LSTMBaselineForecaster, TCNBaselineForecaster
from src.utils.seed import set_seed


def _collate_fn(batch: list[SampleBatch]) -> SampleBatch:
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


def _build_baseline_model(config: dict, input_channels: int) -> tuple[str, nn.Module]:
    baseline_model_type = str(config.get("baseline_model_type", "lstm")).strip().lower()
    if baseline_model_type == "lstm":
        baseline_hidden_size = int(config.get("baseline_lstm_hidden_size", config["lstm_hidden_size"]))
        model = LSTMBaselineForecaster(
            input_channels=input_channels,
            input_steps=config["input_steps"],
            output_steps=config["output_steps"],
            hidden_size=baseline_hidden_size,
            num_layers=1,
            dropout=config["dropout"],
        )
        return baseline_model_type, model

    if baseline_model_type == "tcn":
        tcn_channels = config.get("baseline_tcn_channels", [32, 32, 32])
        model = TCNBaselineForecaster(
            input_channels=input_channels,
            input_steps=config["input_steps"],
            output_steps=config["output_steps"],
            channels=[int(channel) for channel in tcn_channels],
            kernel_size=int(config.get("baseline_tcn_kernel_size", 3)),
            dropout=float(config.get("baseline_tcn_dropout", config["dropout"])),
        )
        return baseline_model_type, model

    raise ValueError(
        "Unsupported baseline_model_type. Use 'lstm' or 'tcn'. "
        f"Got: {baseline_model_type}"
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
    time_mask_prob = config.get("time_mask_prob", 0.0)
    time_mask_min_width = int(config.get("time_mask_min_width", 0))
    time_mask_max_width = int(config.get("time_mask_max_width", 0))
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
                time_mask_prob=time_mask_prob,
                time_mask_min_width=time_mask_min_width,
                time_mask_max_width=time_mask_max_width,
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
        time_mask_prob=time_mask_prob,
        time_mask_min_width=time_mask_min_width,
        time_mask_max_width=time_mask_max_width,
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


def run_lstm_baseline_training(
    config_path: str | Path = "configs/common.yaml",
    baseline_model_type: str | None = None,
) -> dict[str, float]:
    """Train and evaluate baseline model (LSTM/TCN), reporting final RMSE by split."""

    _ensure_processed_data()
    config = load_config(config_path)
    if baseline_model_type is not None:
        config["baseline_model_type"] = baseline_model_type
    set_seed(config["seed"])
    use_standardized_data = config.get("use_standardized_data", True)
    mixup_prob = float(config.get("mixup_prob", 0.0))
    mixup_alpha = float(config.get("mixup_alpha", 0.2))

    split_datasets = _build_split_datasets(config)
    train_dataset = split_datasets["train"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=_collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_channels = train_dataset[0].inputs.shape[0]
    selected_model_type, model = _build_baseline_model(config, input_channels)
    model = model.to(device)

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
    validation_dataset = split_datasets.get("validation")
    test_dataset = split_datasets.get("test")
    print(f"baseline_model_type={selected_model_type}")

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


def run_baseline_training(
    config_path: str | Path = "configs/common.yaml",
    baseline_model_type: str | None = None,
) -> dict[str, float]:
    """Public unified baseline training entrypoint."""

    return run_lstm_baseline_training(
        config_path=config_path,
        baseline_model_type=baseline_model_type,
    )


if __name__ == "__main__":
    result = run_lstm_baseline_training()
    print(result)