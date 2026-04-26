# CS412 Final Project Skeleton

This repository provides a minimal PyTorch scaffold for short-term load forecasting.

## Project layout

- `configs/`: model and training configuration.
- `src/data/`: dataset and feature preparation helpers.
- `src/models/`: PyTorch models.
- `src/utils/`: metrics, random seed, and shared utilities.
- `tests/`: environment and smoke tests.
- `project.py`: simple entry point for smoke test or demo training.

## Quick start

```powershell
.venv/Scripts/python.exe project.py smoke-test
.venv/Scripts/python.exe project.py demo-train
.venv/Scripts/python.exe -m src.infer
```

The demo train command uses synthetic data so you can verify the training loop before wiring in the real dataset.
