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
├── src/
│   ├── data/          # preprocessing
│   ├── models/        # architectures
│   ├── training/      # training logic
│   ├── evaluation/    # metrics
│   └── utils/
├── configs/           # experiment configurations
├── experiments/       # logs, results, checkpoints
├── notebooks/         # exploration
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

Run an experiment using a configuration file:

```bash
python main.py --config configs/baseline.yaml
```

## Experiments

All experiment outputs are stored in the `experiments/` directory:

- training logs
- saved models (.pth)
- evaluation metrics

Each run is organized in a separate folder for reproducibility.