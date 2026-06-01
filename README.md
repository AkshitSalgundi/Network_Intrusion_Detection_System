# Network Intrusion Detection System (NIDS)

This repository contains a machine learning project for network intrusion detection using the UNSW-NB15 dataset.

## Contents

- `app.py` - A web app interface for the trained intrusion detection model.
- `train_model.py` - Training script for building the model from dataset files.
- `NIDS.ipynb` - Exploratory notebook and analysis.
- `.gitignore` - Excludes large dataset files and trained model artifacts.

## Dataset and Large Files

The original dataset CSV files and model artifacts are not included in this repository because they are large:

- `archive/UNSW-NB15_*.csv`
- `model/*.pkl`

To reproduce the project locally:

1. Download the UNSW-NB15 dataset from a trusted source.
2. Place the dataset files into the `archive/` directory.
3. Run `train_model.py` to train models and generate the `model/` artifacts.

## Notes

- `archive/` and `model/` are intentionally ignored by git to keep this repository lightweight.
- The current repository includes only code, notebook, and configuration files.
