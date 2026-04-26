from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_raw_table(file_path: str | Path) -> pd.DataFrame:
    """Load a raw electricity table from csv or xlsx.

    This function is intentionally lightweight so the midterm report can show a
    clear data-ingestion entry point before the final dataset schema is fixed.
    """

    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    raise ValueError(f"Unsupported file format: {suffix}")


def standardize_load_frame(frame: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """Normalize a raw daily load table into a report-friendly schema.

    Expected result: one row per day, one date column, and 96 load columns.
    """

    normalized = frame.copy()
    normalized.columns = [str(column).strip() for column in normalized.columns]
    if date_column not in normalized.columns:
        raise ValueError(f"Missing required date column: {date_column}")

    normalized[date_column] = pd.to_datetime(normalized[date_column])
    load_columns = [column for column in normalized.columns if column != date_column]
    if len(load_columns) != 96:
        raise ValueError(f"Expected 96 load columns, found {len(load_columns)}")

    normalized = normalized.sort_values(date_column).reset_index(drop=True)
    normalized[load_columns] = normalized[load_columns].interpolate(axis=1).ffill().bfill()
    return normalized


def add_calendar_features(frame: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """Attach simple calendar covariates that are always known at inference time."""

    enriched = frame.copy()
    date_series = pd.to_datetime(enriched[date_column])
    enriched["day_of_week"] = date_series.dt.dayofweek
    enriched["is_weekend"] = (date_series.dt.dayofweek >= 5).astype(int)
    enriched["month"] = date_series.dt.month
    enriched["day_of_year"] = date_series.dt.dayofyear
    return enriched


def split_by_project_schedule(frame: pd.DataFrame, date_column: str = "date") -> dict[str, pd.DataFrame]:
    """Split the dataset according to the course project specification."""

    dated = frame.copy()
    dated[date_column] = pd.to_datetime(dated[date_column])
    train_mask = (dated[date_column] >= "2022-01-01") & (dated[date_column] <= "2023-09-30")
    val_mask = (dated[date_column] >= "2023-10-01") & (dated[date_column] <= "2024-05-31")
    test_mask = (dated[date_column] >= "2024-06-01") & (dated[date_column] <= "2024-12-01")
    return {
        "train": dated.loc[train_mask].reset_index(drop=True),
        "validation": dated.loc[val_mask].reset_index(drop=True),
        "test": dated.loc[test_mask].reset_index(drop=True),
    }
