# FGI Deepfake Detection (PyTorch)

## Overview

This project focuses on **audio-visual deepfake detection** using Fine-Grained Inconsistencies (FGI).  
The goal is to reimplement and analyze a recent research method that detects subtle inconsistencies between audio and visual signals in deepfake videos.

## Objectives

- Reproduce the methodology from the FGI paper
- Understand multimodal deepfake detection (audio + video)
- Build a clean and modular PyTorch pipeline
- Compare baseline and advanced models
- Analyze results and limitations

## Reference Paper

> **Detecting Audio-Visual Deepfakes with Fine-Grained Inconsistencies**  
> BMVC 2024

Main idea:
- Detect subtle inconsistencies between audio and visual modalities
- Focus on fine-grained spatial and temporal mismatches
- Improve robustness against realistic deepfakes

## Original Repository

This project is inspired by the following repository:

`https://github.com/aseuteurideu/FGI`

This repository does **not directly reuse the original code**, but instead aims to reimplement the main ideas from scratch for better understanding, modularity, and experimentation.

## Project Approach

1. Build a simple baseline (audio-only or video-only)
2. Implement a multimodal fusion pipeline
3. Reproduce an FGI-inspired method
4. Run experiments and compare results

## Project Structure

```text
.
├── data/              # datasets (not tracked)
├── configs/           # experiment configurations
├── experiments/       # logs, results, checkpoints
├── jobs/              # jobs for cluster runs
├── notebooks/         # exploration
├── src/
│   ├── data/          # preprocessing and PyTorch datasets
│   ├── models/        # architectures
│   ├── training/      # training logic
│   ├── evaluation/    # metrics
│   └── utils/
├── tests/             # tests
├── main.py
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/TON-USERNAME/fgi-deepfake-detection-pytorch.git
cd fgi-deepfake-detection-pytorch
```

### 2. Create a virtual environment (recommended)

#### On Windows (PowerShell)

```powershell
python -m venv venv
venv\Scripts\activate
```

#### On Linux / Mac

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

Train an audio or video baseline:

```bash
python3 main.py train --config configs/baseline_audio.yaml
python3 main.py train --config configs/baseline_video.yaml
```

Evaluate a checkpoint:

```bash
python3 main.py eval \
  --config configs/baseline_audio.yaml \
  --checkpoint runs/baseline-audio/<run-id>/checkpoints/best.pt \
  --split test
```

Resume an interrupted training run. `--epochs` is the final target epoch:

```bash
python3 main.py train \
  --config configs/baseline_video.yaml \
  --resume runs/baseline-video/<run-id>/checkpoints/last.pt \
  --epochs 20
```

Compare audio and video checkpoints and evaluate their probability ensemble:

```bash
python3 main.py ensemble-eval \
  --config configs/baseline_ensemble.yaml \
  --audio-checkpoint runs/baseline-audio/<run-id>/checkpoints/best.pt \
  --video-checkpoint runs/baseline-video/<run-id>/checkpoints/best.pt
```

Slurm entry points are available in `jobs/`:

```bash
sbatch jobs/train_audio_baseline.sbatch
sbatch jobs/train_video_baseline.sbatch
sbatch jobs/eval_baseline.sbatch
sbatch jobs/eval_ensemble.sbatch
```

## Experiments

Experiment outputs are stored in `runs/<experiment>/<run-id>/`:

- copied configuration and Git metadata
- checkpoints (`best.pt`, `last.pt`)
- training and evaluation metrics
- per-clip predictions
- SVG plots and confusion matrices

Each run is organized in a separate folder for reproducibility.

## Dataset analysis

The static dataset audit is available in
`notebooks/dataset_static_analysis.ipynb`. It reports class and split balance,
video-level distributions, clips per video, leakage and duplicate checks,
training class weights, and an optional clip file inventory.
