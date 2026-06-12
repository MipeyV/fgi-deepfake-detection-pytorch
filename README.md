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

Train the R3D-18 video model initialized from Kinetics-400 weights:

```bash
python3 main.py train --config configs/r3d18_video.yaml
```

Submit the same experiment to Slurm:

```bash
sbatch --export=ALL,CONFIG=configs/r3d18_video.yaml \
  jobs/train_video_baseline.sbatch
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

## Extensible video pipeline

Video experiments separate input processing from model architecture:

- `src/data/video/` builds the configured preprocessing and dataloaders.
- `src/models/video/` builds the configured classifier.
- training and evaluation consume the shared batch contract without knowing
  which implementation produced it.

Current video input pipelines return `frames` with shape
`[batch, frames, channels, height, width]`. Video classifiers return logits
with shape `[batch, classes]`.

The two components are selected independently in YAML:

```yaml
video:
  preprocessing:
    name: resize_center_crop
    resize_size: [128, 171]
    crop_size: 112

model:
  name: r3d18
```

Supported visual preprocessing strategies are:

- `resize_square`: direct square resize used by the original video baseline.
- `resize_center_crop`: resize followed by a centered crop for R3D-18.
- `resize_normalize`: square resize followed by configurable RGB
  normalization. With `mean` and `std` set to `0.5`, frames are mapped to
  `[-1, 1]` as in the official FGI visual pipeline.

The preparatory FGI configuration demonstrates that preprocessing and model
selection remain independent:

```yaml
video:
  preprocessing:
    name: resize_normalize
    frame_size: 224
    mean: [0.5, 0.5, 0.5]
    std: [0.5, 0.5, 0.5]

model:
  name: video_cnn_baseline
```

See `configs/fgi_preprocessing.yaml`. It is an executable input-pipeline
experiment, not yet the multimodal FGI model. The next FGI stages are separate
audio and visual encoders, local audio-visual distances, spatial attention, and
temporally local pseudo-fake augmentation.

## FGI face crops

FGI experiments use a separate offline cache of stable face crops. The source
clips and audio remain unchanged under `data/processed/`; cropped frames and
copied synchronized audio are written under `data/processed_fgi/`.

Download the official YuNet model from
[`opencv/opencv_zoo`](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet):

```bash
mkdir -p models
curl -L \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx \
  -o models/face_detection_yunet_2023mar.onnx
```

Process each split independently so its existing split metadata is preserved:

```bash
python3 main.py fgi-face-crops \
  --manifest data/manifests/train_manifest.csv \
  --output-dir data/processed_fgi \
  --output-manifest data/manifests_fgi/train_manifest.csv \
  --detector-model models/face_detection_yunet_2023mar.onnx \
  --missing-face-policy skip \
  --contact-sheet runs/fgi-face-crops/train_contact_sheet.png
```

Repeat with `val_manifest.csv` and `test_manifest.csv`. The command associates
face detections over time using IoU, selects the longest consistent track,
aggregates it into one stable square crop for the whole clip, adds a
configurable margin, resizes to `256x256`, copies `audio.wav`, and rewrites
`clip_path` in the new manifest. Each contact-sheet cell shows the beginning,
middle, and end of a clip.

Always inspect the contact sheet and skipped-clip count before training. The
optional `--detector haar` backend requires no model file, but it is less
reliable and should only be used for explicit development checks.

## FGI multimodal dataset

`configs/fgi_inspired.yaml` defines the strict synchronized input contract for
the FGI-inspired model:

```text
frames: [batch, 30, 3, 224, 224]
audio:  [batch, 48000]
label:  [batch]
```

The dataset validates every clip before returning it:

- exactly 30 face frames;
- mono 16-bit PCM audio;
- exactly 48,000 samples at 48 kHz;
- face pixels normalized to `[-1, 1]`;
- raw audio normalized per clip to `[-1, 1]`;
- labels encoded as `real=0`, `fake=1`.

After generating `data/manifests_fgi/`, test a split without constructing a
model:

```bash
python3 main.py fgi-data-smoke \
  --config configs/fgi_inspired.yaml \
  --split train \
  --batch-size 1
```

The command prints batch shapes and numeric ranges. The config deliberately
tracks component readiness through `model.implementation_status`.

## FGI encoders

The first model components now transform synchronized inputs into aligned local
features:

```text
faces [B, 30, 3, 224, 224] -> video [B, 128, 15, 28, 28]
audio [B, 48000]            -> audio [B, 128, 15]
```

The video encoder is a compact residual 3D CNN. The raw-audio encoder follows
the convolutional structure used by FGI and adaptively projects its temporal
axis to the same 15 positions as video. Both embedding dimensions and output
sizes are configurable under `model.encoders`.

Run the complete data-to-encoder smoke test with:

```bash
python3 main.py fgi-encoder-smoke \
  --config configs/fgi_inspired.yaml \
  --split train \
  --batch-size 1 \
  --device cpu
```

## FGI inconsistency classifier

The complete forward pass compares every spatial video location with the
synchronized audio representation:

```text
video [B, 128, 15, 28, 28] + audio [B, 128, 15]
    -> local inconsistency map [B, 28, 28]
    -> spatial attention map [B, 28, 28]
    -> classification logits [B, 2]
```

The distance map, attention map, and encoder features remain available in the
model output for inspection. Dropout, attention dimensions, attention mode,
and encoder dimensions are configurable under `model`.

Run the complete data-to-logits smoke test with:

```bash
python3 main.py fgi-model-smoke \
  --config configs/fgi_inspired.yaml \
  --split train \
  --batch-size 1 \
  --device cpu
```

The configuration is now marked `model_ready`. This means the forward model is
implemented; integration with the generic `train` and `eval` commands is the
next step.

## Dataset analysis

The static dataset audit is available in
`notebooks/dataset_static_analysis.ipynb`. It reports class and split balance,
video-level distributions, clips per video, leakage and duplicate checks,
training class weights, and an optional clip file inventory.
