# CS412 Final Project Skeleton

This repository provides a minimal PyTorch scaffold for short-term load forecasting.

## Current midterm direction

The current primary model is a 1D U-Net with an LSTM bottleneck.

- Encoder: repeated convolution blocks + max-pooling for multi-scale temporal feature extraction.
- Bottleneck: LSTM over the compressed sequence to model longer-range dependency.
- Decoder: transposed convolution upsampling with skip connections from the encoder.
- Context fusion: calendar or weather features can be projected into the bottleneck.

This keeps the architecture aligned with the project goal: learn short-term local load patterns with convolutions, then refine global temporal dependency with recurrent modeling.

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
.venv/Scripts/python.exe project.py lstm-baseline
.venv/Scripts/python.exe project.py tcn-baseline
.venv/Scripts/python.exe project.py baseline --baseline-type tcn
.venv/Scripts/python.exe -m src.infer
```

The demo train command uses synthetic data so you can verify the training loop before wiring in the real dataset.

For raw-data integration, start from `src/data/preprocess.py`. It provides a first-pass pipeline for loading csv/xlsx files, checking the 96-point daily layout, adding simple calendar features, and splitting the dataset according to the course schedule.
