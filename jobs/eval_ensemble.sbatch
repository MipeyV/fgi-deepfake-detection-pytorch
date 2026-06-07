#!/usr/bin/env bash
#SBATCH --job-name=fgi-ensemble
#SBATCH --partition=3090
#SBATCH --output=jobs/logs/%x-%j.out
#SBATCH --error=jobs/logs/%x-%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
CONFIG="${CONFIG:-configs/baseline_ensemble.yaml}"
AUDIO_CHECKPOINT="${AUDIO_CHECKPOINT:?Set AUDIO_CHECKPOINT}"
VIDEO_CHECKPOINT="${VIDEO_CHECKPOINT:?Set VIDEO_CHECKPOINT}"
SPLIT="${SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DEVICE="${DEVICE:-cuda}"
RUN_ID="${RUN_ID:-}"

cd "$PROJECT_DIR"
mkdir -p jobs/logs

if [[ -n "${VENV_PATH:-}" ]]; then
  source "$VENV_PATH/bin/activate"
fi

COMMAND=(
  python3 -u main.py ensemble-eval
  --config "$CONFIG"
  --audio-checkpoint "$AUDIO_CHECKPOINT"
  --video-checkpoint "$VIDEO_CHECKPOINT"
  --split "$SPLIT"
  --batch-size "$BATCH_SIZE"
  --device "$DEVICE"
)

if [[ -n "$RUN_ID" ]]; then
  COMMAND+=(--run-id "$RUN_ID")
fi

if [[ -n "${MAX_BATCHES:-}" ]]; then
  COMMAND+=(--max-batches "$MAX_BATCHES")
fi

printf 'Running command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'

"${COMMAND[@]}"
