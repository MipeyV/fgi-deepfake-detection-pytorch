#!/usr/bin/env bash
#SBATCH --job-name=fgi-video
#SBATCH --output=jobs/logs/%x-%j.out
#SBATCH --error=jobs/logs/%x-%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
CONFIG="${CONFIG:-configs/baseline_video.yaml}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DEVICE="${DEVICE:-cuda}"

cd "$PROJECT_DIR"
mkdir -p jobs/logs

if [[ -n "${VENV_PATH:-}" ]]; then
  source "$VENV_PATH/bin/activate"
fi

COMMAND=(
  python3 main.py train
  --config "$CONFIG"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --device "$DEVICE"
)

if [[ -n "${MAX_BATCHES:-}" ]]; then
  COMMAND+=(--max-batches "$MAX_BATCHES")
fi

printf 'Running command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'

"${COMMAND[@]}"
