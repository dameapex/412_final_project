from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd


def load_raw_table(file_path: str | Path) -> pd.DataFrame:
    # Load a raw electricity table from csv or xlsx.
    # This function is intentionally lightweight so the midterm report can show a
    # clear data-ingestion entry point before the final dataset schema is fixed.

    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    raise ValueError(f"Unsupported file format: {suffix}")


DATE_COLUMN = "date"
TYPE_COLUMN = "data_type"
TIME_INDEX_COLUMN = "sample_index"
TIME_LABEL_COLUMN = "time_label"
DAYPART_COLUMN = "day_part"
TRAIN_FILE = "train_data.xlsx"
VALIDATION_FILE = "validation_data.xlsx"
TEST_FILE = "test_data_to_students.xlsx"
WEATHER_DAILY_FILE = "data/weather/haining_weather_2022_2024_daily.csv"
DEFAULT_TIME_INTERVAL_MINUTES = 15
MAX_NEGATIVE_POINTS_PER_DAY = 8         # Allow some negative points to be repaired, but if a day is mostly negative then it's probably too noisy to learn from.
MIN_STANDARDIZER_STD = 0.05


def _build_load_columns() -> list[str]:
    # Generate load_01, load_02, ..., load_96 for the 96 time slots in each day.
    return [f"load_{index:02d}" for index in range(1, 97)]

LOAD_COLUMNS = _build_load_columns()
WEATHER_COLUMNS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "precipitation_hours",
    "relative_humidity_2m_mean",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "shortwave_radiation_sum",
]

def _convert_excel_date(value: object) -> pd.Timestamp:
    # Training/validation dates are Excel serial numbers; test dates are already real datetimes.
    if pd.isna(value):
        return pd.NaT
    if isinstance(value, (int, float, np.integer, np.floating)):
        return pd.to_datetime(float(value), unit="D", origin="1899-12-30").normalize()
    return pd.to_datetime(value).normalize()


def normalize_raw_columns(frame: pd.DataFrame) -> pd.DataFrame:
    # Map the provided Chinese column names into a stable English schema.

    renamed = frame.copy()
    rename_map = {"数据日期": DATE_COLUMN, "数据类型": TYPE_COLUMN}
    rename_map.update({f"功率{index}": LOAD_COLUMNS[index - 1] for index in range(1, 97)})
    renamed = renamed.rename(columns=rename_map)
    renamed.columns = [str(column).strip() for column in renamed.columns]
    missing = [column for column in [DATE_COLUMN, TYPE_COLUMN, *LOAD_COLUMNS] if column not in renamed.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return renamed[[DATE_COLUMN, TYPE_COLUMN, *LOAD_COLUMNS]].copy()


def standardize_daily_frame(frame: pd.DataFrame) -> pd.DataFrame:
    # Normalize date/type columns and coerce the 96 load readings to numeric.

    standardized = normalize_raw_columns(frame)
    standardized[DATE_COLUMN] = standardized[DATE_COLUMN].apply(_convert_excel_date)
    standardized[TYPE_COLUMN] = pd.to_numeric(standardized[TYPE_COLUMN], errors="coerce").fillna(0).astype(int)
    standardized[LOAD_COLUMNS] = standardized[LOAD_COLUMNS].apply(pd.to_numeric, errors="coerce")
    standardized = standardized.sort_values(DATE_COLUMN).reset_index(drop=True)
    return standardized


def repair_missing_values(frame: pd.DataFrame) -> pd.DataFrame:
    # Repair missing values and across days without breaking row count.
    # Using simple interpolation and forward/backward fill here since the missingness is sparse and doesn't seem to have a strong temporal pattern.

    repaired = frame.copy()
    repaired[LOAD_COLUMNS] = repaired[LOAD_COLUMNS].interpolate(axis=1, limit_direction="both")
    repaired[LOAD_COLUMNS] = repaired[LOAD_COLUMNS].ffill().bfill()
    return repaired


def repair_negative_values(
    frame: pd.DataFrame,
    split_name: str,
    max_negative_points_per_day: int = MAX_NEGATIVE_POINTS_PER_DAY,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Treat negative load values as noise.

    # Strategy:
    # - If one day contains only a small number of negative points, replace each
    #   negative value with the previous day's value at the same slot when
    #   available, otherwise fall back to the non-negative median of that slot.
    # - If a day contains too many negative points, drop it for train/validation.
    # - For test, never drop rows because every day still needs a prediction; all
    #   negative values are repaired in place.

    repaired = frame.copy().sort_values(DATE_COLUMN).reset_index(drop=True)
    negative_mask = repaired[LOAD_COLUMNS] < 0
    negative_count = negative_mask.sum(axis=1)
    report = pd.DataFrame(
        {
            DATE_COLUMN: repaired[DATE_COLUMN],
            "negative_points": negative_count,
            "dropped_for_noise": False,
        }
    )

    keep_mask = negative_count <= max_negative_points_per_day
    if split_name != "test":
        report.loc[~keep_mask, "dropped_for_noise"] = True
        repaired = repaired.loc[keep_mask].reset_index(drop=True)
        negative_mask = repaired[LOAD_COLUMNS] < 0

    slot_medians = repaired[LOAD_COLUMNS].mask(repaired[LOAD_COLUMNS] < 0).median(axis=0).fillna(0.0)

    for row_index in range(len(repaired)):
        for column in LOAD_COLUMNS:
            if repaired.at[row_index, column] < 0:
                replacement = np.nan
                if row_index > 0:
                    previous_value = repaired.at[row_index - 1, column]
                    if pd.notna(previous_value) and previous_value >= 0:
                        replacement = previous_value
                if pd.isna(replacement):
                    replacement = slot_medians[column]
                repaired.at[row_index, column] = replacement

    repaired[LOAD_COLUMNS] = repaired[LOAD_COLUMNS].interpolate(axis=1, limit_direction="both")
    repaired[LOAD_COLUMNS] = repaired[LOAD_COLUMNS].ffill().bfill()
    return repaired, report


def repair_outliers(frame: pd.DataFrame, z_threshold: float = 4.0) -> pd.DataFrame:
    # Detect extreme point-wise anomalies using robust z-scores and smooth them.

    # The replacement strategy uses the median profile at the same sample index
    # computed on the current split, then reapplies local interpolation.

    repaired = frame.copy()
    values = repaired[LOAD_COLUMNS]
    medians = values.median(axis=0)
    mad = (values - medians).abs().median(axis=0).replace(0, np.nan)
    robust_scale = 1.4826 * mad
    robust_z = (values - medians).abs().divide(robust_scale)
    outlier_mask = robust_z > z_threshold
    repaired[LOAD_COLUMNS] = values.mask(outlier_mask, other=pd.DataFrame([medians] * len(values), columns=LOAD_COLUMNS))
    repaired[LOAD_COLUMNS] = repaired[LOAD_COLUMNS].interpolate(axis=1, limit_direction="both")
    repaired[LOAD_COLUMNS] = repaired[LOAD_COLUMNS].ffill().bfill()
    return repaired


def add_daily_features(frame: pd.DataFrame) -> pd.DataFrame:
    # Attach daily calendar and summary features to the wide daily table.

    enriched = frame.copy()
    date_series = pd.to_datetime(enriched[DATE_COLUMN])
    enriched["day_of_week"] = date_series.dt.dayofweek
    enriched["is_weekend"] = (enriched["day_of_week"] >= 5).astype(int)
    enriched["month"] = date_series.dt.month
    enriched["day_of_year"] = date_series.dt.dayofyear
    enriched["week_of_year"] = date_series.dt.isocalendar().week.astype(int)
    enriched["daily_mean"] = enriched[LOAD_COLUMNS].mean(axis=1)
    enriched["daily_std"] = enriched[LOAD_COLUMNS].std(axis=1)
    enriched["daily_max"] = enriched[LOAD_COLUMNS].max(axis=1)
    enriched["daily_min"] = enriched[LOAD_COLUMNS].min(axis=1)
    enriched["daily_range"] = enriched["daily_max"] - enriched["daily_min"]
    return enriched


def load_weather_daily(base_dir: str | Path) -> pd.DataFrame:
    weather_path = Path(base_dir) / WEATHER_DAILY_FILE
    if not weather_path.exists():
        raise FileNotFoundError(
            f"Weather table not found: {weather_path}. "
            "Please fetch weather data first."
        )

    weather = pd.read_csv(weather_path)
    required_columns = [DATE_COLUMN, *WEATHER_COLUMNS]
    missing = [column for column in required_columns if column not in weather.columns]
    if missing:
        raise ValueError(f"Weather table missing columns: {missing}")

    weather = weather[required_columns].copy()
    weather[DATE_COLUMN] = pd.to_datetime(weather[DATE_COLUMN]).dt.normalize()
    weather = weather.sort_values(DATE_COLUMN).drop_duplicates(subset=[DATE_COLUMN], keep="last").reset_index(drop=True)
    weather[WEATHER_COLUMNS] = weather[WEATHER_COLUMNS].apply(pd.to_numeric, errors="coerce")
    weather[WEATHER_COLUMNS] = weather[WEATHER_COLUMNS].ffill().bfill()
    return weather


def add_weather_features(frame: pd.DataFrame, weather_daily: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched[DATE_COLUMN] = pd.to_datetime(enriched[DATE_COLUMN]).dt.normalize()
    merged = enriched.merge(weather_daily, on=DATE_COLUMN, how="left")
    merged[WEATHER_COLUMNS] = merged[WEATHER_COLUMNS].ffill().bfill()
    merged[WEATHER_COLUMNS] = merged[WEATHER_COLUMNS].fillna(0.0)
    return merged


def daily_to_long(frame: pd.DataFrame) -> pd.DataFrame:
    # Explode the daily 96-point table into a long table with explicit time tags.

    long_frame = frame.melt(
        id_vars=[column for column in frame.columns if column not in LOAD_COLUMNS],
        value_vars=LOAD_COLUMNS,
        var_name="load_slot",
        value_name="load_value",
    )
    long_frame[TIME_INDEX_COLUMN] = long_frame["load_slot"].str.extract(r"(\d+)").astype(int) - 1
    minutes = long_frame[TIME_INDEX_COLUMN] * DEFAULT_TIME_INTERVAL_MINUTES
    long_frame[TIME_LABEL_COLUMN] = pd.to_datetime(minutes, unit="m").dt.strftime("%H:%M")
    long_frame["hour"] = (minutes // 60).astype(int)
    long_frame["minute"] = (minutes % 60).astype(int)
    long_frame[DAYPART_COLUMN] = long_frame["hour"].map(_hour_to_day_part)
    long_frame["is_morning_peak"] = long_frame["hour"].between(7, 10).astype(int)
    long_frame["is_evening_peak"] = long_frame["hour"].between(18, 21).astype(int)
    return long_frame.sort_values([DATE_COLUMN, TIME_INDEX_COLUMN]).reset_index(drop=True)


def _hour_to_day_part(hour: int) -> str:
    if 5 <= hour < 11:
        return "morning"
    if 11 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 22:
        return "evening"
    return "night"


def fit_standardizer(
    frame: pd.DataFrame,
    columns: list[str],
    min_std: float = MIN_STANDARDIZER_STD,
) -> dict[str, dict[str, float]]:
    # Fit z-score parameters on the training split only.

    stats: dict[str, dict[str, float]] = {}
    for column in columns:
        mean = float(frame[column].mean())
        std = float(frame[column].std())
        if std == 0 or np.isnan(std):
            std = min_std
        std = max(std, min_std)
        stats[column] = {"mean": mean, "std": std}
    return stats


def apply_standardizer(frame: pd.DataFrame, stats: dict[str, dict[str, float]]) -> pd.DataFrame:
    # Apply z-score standardization using the provided statistics.
    
    standardized = frame.copy()
    for column, params in stats.items():
        standardized[column] = (standardized[column] - params["mean"]) / params["std"]
    return standardized


def build_split_stages(
    file_path: str | Path,
    split_name: str,
    weather_daily: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    # Return intermediate preprocessing stages for one split.

    # This is useful for diagnostics and plotting so we can compare the raw daily
    # curve, the repaired curve, and the standardized curve on the same sample.

    raw_frame = load_raw_table(file_path)
    standardized_schema = standardize_daily_frame(raw_frame)
    missing_repaired = repair_missing_values(standardized_schema)
    negative_repaired, noise_report = repair_negative_values(missing_repaired, split_name=split_name)
    outlier_repaired = repair_outliers(negative_repaired)
    featured = add_daily_features(outlier_repaired)
    featured = add_weather_features(featured, weather_daily)
    return {
        "raw": standardized_schema,
        "missing_repaired": missing_repaired,
        "negative_repaired": negative_repaired,
        "outlier_repaired": outlier_repaired,
        "featured": featured,
        "noise_report": noise_report,
    }


def build_all_split_stages(base_dir: str | Path, weather_daily: pd.DataFrame) -> dict[str, dict[str, pd.DataFrame]]:
    # Build intermediate preprocessing stages for train/validation/test splits.

    base_dir = Path(base_dir)
    file_map = {
        "train": base_dir / TRAIN_FILE,
        "validation": base_dir / VALIDATION_FILE,
        "test": base_dir / TEST_FILE,
    }
    return {
        split_name: build_split_stages(file_path, split_name=split_name, weather_daily=weather_daily)
        for split_name, file_path in file_map.items()
    }


def save_processed_outputs(base_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, Path]:
    # Run the full preprocessing pipeline and persist wide/long outputs plus stats.

    base_dir = Path(base_dir)
    output_dir = Path(output_dir) if output_dir else base_dir / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    weather_daily = load_weather_daily(base_dir)
    all_stages = build_all_split_stages(base_dir, weather_daily=weather_daily)
    processed = {split_name: stages["featured"] for split_name, stages in all_stages.items()}
    feature_columns = [
        *LOAD_COLUMNS,
        "day_of_week",
        "month",
        "day_of_year",
        "week_of_year",
        *WEATHER_COLUMNS,
        "daily_mean",
        "daily_std",
        "daily_max",
        "daily_min",
        "daily_range",
    ]
    stats = fit_standardizer(processed["train"], feature_columns)

    saved_paths: dict[str, Path] = {}
    noise_reports: dict[str, list[dict[str, object]]] = {}
    for split_name, wide_frame in processed.items():
        stages = all_stages[split_name]
        cleaned_wide = wide_frame.copy()
        standardized_wide = apply_standardizer(wide_frame, stats)
        cleaned_long = daily_to_long(cleaned_wide)
        standardized_long = daily_to_long(standardized_wide)
        noise_reports[split_name] = stages["noise_report"].assign(date=lambda frame: frame[DATE_COLUMN].astype(str)).to_dict(orient="records")

        cleaned_daily_path = output_dir / f"{split_name}_daily_cleaned.csv"
        standardized_daily_path = output_dir / f"{split_name}_daily_standardized.csv"
        cleaned_long_path = output_dir / f"{split_name}_long_cleaned.csv"
        standardized_long_path = output_dir / f"{split_name}_long_standardized.csv"

        cleaned_wide.to_csv(cleaned_daily_path, index=False, encoding="utf-8-sig")
        standardized_wide.to_csv(standardized_daily_path, index=False, encoding="utf-8-sig")
        cleaned_long.to_csv(cleaned_long_path, index=False, encoding="utf-8-sig")
        standardized_long.to_csv(standardized_long_path, index=False, encoding="utf-8-sig")

        saved_paths[f"{split_name}_daily_cleaned"] = cleaned_daily_path
        saved_paths[f"{split_name}_daily_standardized"] = standardized_daily_path
        saved_paths[f"{split_name}_long_cleaned"] = cleaned_long_path
        saved_paths[f"{split_name}_long_standardized"] = standardized_long_path

    stats_path = output_dir / "standardization_stats.json"
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
    saved_paths["stats"] = stats_path

    noise_report_path = output_dir / "noise_report.json"
    with noise_report_path.open("w", encoding="utf-8") as handle:
        json.dump(noise_reports, handle, ensure_ascii=False, indent=2)
    saved_paths["noise_report"] = noise_report_path
    return saved_paths


if __name__ == "__main__":
    outputs = save_processed_outputs(Path(__file__).resolve().parents[2])
    for name, path in outputs.items():
        print(f"{name}: {path}")
