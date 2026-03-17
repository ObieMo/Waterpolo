# Waterpolo Action Spotting with CALF

This repository contains a reduced, project-focused version of the CALF action spotting pipeline adapted for waterpolo.

It covers two main steps:

1. Feature extraction from video with a ResNet + PCA pipeline
2. Action spotting with a CALF-based temporal model

The spotting model is based on the CALF architecture (`ContextAwareModel`) from the SoccerNet action spotting benchmark, adapted here for waterpolo event spotting.

## What The Model Does

The goal is to detect action timestamps in untrimmed waterpolo videos.

The current waterpolo setup uses 2 classes:

- `GOAL`
- `MissedShot`

These classes are defined in:

- `Task1-ActionSpotting/CALF/src/config/classes_waterpolo.py`

The model predicts action locations over time from precomputed video features. The training and testing entrypoint is:

- `Task1-ActionSpotting/CALF/src/main_waterpolo.py`

## Repository Structure

```text
Waterpolo/
|-- Features/
|   |-- VideoFeatureExtractor.py
|   |-- ExtractResNET_TF2.py
|   |-- ReduceFeaturesPCA.py
|   |-- pca_512_TF2.pkl
|   `-- average_512_TF2.pkl
|-- Task1-ActionSpotting/
|   `-- CALF/
|       |-- src/
|       |   |-- main_waterpolo.py
|       |   |-- dataset_waterpolo.py
|       |   |-- model.py
|       |   `-- config/classes_waterpolo.py
|       `-- models/
|-- requirements.txt
`-- README.md
```

## Environment Setup

This project is intended to run with the same stack as the previously working `calf-py38` environment.

Create the environment with:

```powershell
conda create -n waterpolo python=3.8 -y
conda activate waterpolo
conda install pytorch=1.6 torchvision=0.7 cudatoolkit=10.1 -c pytorch
pip install -r requirements.txt
```

## Feature Extraction

Use `Features/VideoFeatureExtractor.py` to extract ResNet features and optionally apply the provided PCA reduction.

Example:

```powershell
python Features/VideoFeatureExtractor.py --path_video Videos/4.mp4 --path_features features/match2_ResNET_TF2_PCA512.npy --overwrite --PCA Features/pca_512_TF2.pkl --PCA_scaler Features/average_512_TF2.pkl
```

What this does:

- reads an input video
- extracts frame-level ResNet features
- applies PCA reduction to 512 dimensions
- saves the resulting `.npy` feature file

Important files for feature extraction:

- `Features/VideoFeatureExtractor.py`
- `Features/pca_512_TF2.pkl`
- `Features/average_512_TF2.pkl`

## Dataset Format For Training

The waterpolo training code expects a dataset root with `train`, `valid`, and `test` folders, each containing match folders.

Expected structure:

```text
dataset_root/
|-- train/
|   |-- match_001/
|   |   |-- features.npy
|   |   `-- Labels.json
|   `-- match_002/
|-- valid/
|   `-- match_003/
`-- test/
    `-- match_004/
```

Each match folder should contain:

- `features.npy`: extracted video features
- `Labels.json`: annotations with event timestamps

## Training

Train or resume the waterpolo spotting model with:

```powershell
python src\main_waterpolo.py `
  --dataset_path "C:\Users\Obie\Desktop\testwater" `
  --model_name CALF_benchmark_waterpolo_init `
  --load_weights model `
  --max_epochs 1001 `
  --max_num_worker 0
```

Run this command from:

```text
Task1-ActionSpotting/CALF
```

Notes:

- `--dataset_path` points to the dataset root containing `train`, `valid`, and `test`
- `--model_name` selects the model folder inside `Task1-ActionSpotting/CALF/models`
- `--load_weights model` resolves to `models/<model_name>/checkpoints/model.pth.tar`
- outputs and checkpoints are written under `Task1-ActionSpotting/CALF/models` and `Task1-ActionSpotting/CALF/outputs`

## Main Files

Core files for action spotting:

- `Task1-ActionSpotting/CALF/src/main_waterpolo.py`
- `Task1-ActionSpotting/CALF/src/train_waterpolo.py`
- `Task1-ActionSpotting/CALF/src/dataset_waterpolo.py`
- `Task1-ActionSpotting/CALF/src/model.py`
- `Task1-ActionSpotting/CALF/src/loss.py`
- `Task1-ActionSpotting/CALF/src/config/classes_waterpolo.py`

## Quick Checks

After setting up the environment, you can run:

```powershell
python Features/VideoFeatureExtractor.py --help
python Task1-ActionSpotting/CALF/src/main_waterpolo.py --help
```

If both commands run successfully, the environment is usually in a good state.
