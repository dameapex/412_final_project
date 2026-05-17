
import pandas as pd
import numpy as np
import json

files = {
    "train": "data/processed/train_daily_standardized.csv",
    "validation": "data/processed/validation_daily_standardized.csv"
}

load_cols = [f"load_{i:02d}" for i in range(1, 97)]

for name, path in files.items():
    df = pd.read_csv(path)
    loads = df[load_cols].values
    print(f"Dataset: {name}")
    print(f"  Mean: {np.mean(loads):.4f}")
    print(f"  Std: {np.std(loads):.4f}")
    print(f"  Abs 95th Percentile: {np.percentile(np.abs(loads), 95):.4f}")
    print(f"  Day Range: {df.day_of_year.min()} to {df.day_of_year.max()}")
    if name == "validation":
        extreme_ratio = np.mean(np.abs(loads) > 10)
        print(f"  Extreme Value Ratio (|z|>10): {extreme_ratio:.6f}")

