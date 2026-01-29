# DualTimesField Interpolation Experiments

## Overview

This directory contains code for evaluating DualTimesField on irregular time series interpolation tasks, following the experimental setup from iHyperTime (Zhou et al., 2023).

## Datasets

- **PhysioNet-2012**: ICU patient health monitoring data (41 signals, 8,000 patients)
- **USHCN**: US Historical Climatology Network (1,218 weather stations)

Both datasets are automatically downloaded when running experiments.

## Experiments

### Running All Experiments

```bash
bash imputation/run_all_experiments.sh
```

### Running Individual Experiments

```bash
# PhysioNet
python imputation/run_experiments.py --dataset physionet

# USHCN
python imputation/run_experiments.py --dataset ushcn
```

### Arguments

- `--dataset`: Dataset to use (`physionet` or `ushcn`)
- `--sample-rate`: Fraction of observed points to use for training (default: 0.5)
- `--num-epochs`: Number of training epochs per sample (default: 1000)
- `--lr`: Learning rate (default: 1e-3)
- `--device`: Device to use (`cuda` or `cpu`)

## Results Comparison

After running experiments, generate comparison tables with baseline methods:

```bash
python imputation/baseline_results.py
```

This will create comparison tables with the following baseline methods:
- RNN
- RNN-VAE
- ODE-RNN
- GRU-D
- Latent ODE
- LS4
- iHT (iHyperTime)

## File Structure

```
imputation/
├── __init__.py              # Package initialization
├── datasets.py              # PhysioNet and USHCN dataset loaders
├── models.py                # DualTimesField interpolation models
├── train.py                 # Training and evaluation functions
├── utils.py                 # Utility functions and baseline results
├── run_experiments.py       # Main experiment script
├── run_all_experiments.sh   # Batch experiment runner
├── baseline_results.py      # Results comparison and visualization
└── README.md               # This file
```

## Expected Results

Based on iHyperTime paper (Table 14):

| Method | PhysioNet MSE (×10^-3) | USHCN MSE (×10^-3) |
|--------|------------------------|---------------------|
| LS4    | 0.63                   | 0.06                |
| iHT    | 3.35                   | 1.46                |
| ODE-RNN| 2.23                   | 2.47                |
| RNN    | 2.92                   | 4.32                |

DualTimesField aims to outperform iHT by leveraging the dual-field architecture (CTF + DGF).

## Notes

- Each sample is trained independently (per-sample optimization)
- Training time depends on dataset size and number of epochs
- GPU is recommended for faster training
- Results are saved in `./outputs/interpolation/`
