from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from src.models.baseline import persistence_baseline


def export_demo_submission(output_path: str | Path = "outputs/submissions/demo_submission.xlsx") -> Path:
    """Create a placeholder submission file using the persistence baseline.

    Replace this synthetic example with the real test-data inference pipeline later.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    history = torch.linspace(0.8, 1.2, 96).view(1, 1, 96)
    prediction = persistence_baseline(history).squeeze(0).numpy()

    frame = pd.DataFrame(
        {
            "date": ["2024-06-02"],
            **{f"slot_{index + 1:02d}": [float(prediction[index])] for index in range(96)},
        }
    )
    frame.to_excel(output_path, index=False)
    return output_path


if __name__ == "__main__":
    path = export_demo_submission()
    print(path)
