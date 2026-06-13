#!/usr/bin/env bash
#SBATCH --job-name=fgi-train
#SBATCH --partition=3090
#SBATCH --output=jobs/logs/%x-%j.out
#SBATCH --error=jobs/logs/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
CONFIG="${CONFIG:-configs/fgi_inspired.yaml}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DEVICE="${DEVICE:-cuda}"
RUN_ID="${RUN_ID:-}"
RESUME="${RESUME:-}"

cd "$PROJECT_DIR"
mkdir -p jobs/logs

if [[ -n "${VENV_PATH:-}" ]]; then
  source "$VENV_PATH/bin/activate"
fi

COMMAND=(
  python3 -u main.py train
  --config "$CONFIG"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --device "$DEVICE"
)

if [[ -n "${MAX_BATCHES:-}" ]]; then
  COMMAND+=(--max-batches "$MAX_BATCHES")
fi

if [[ -n "$RUN_ID" ]]; then
  COMMAND+=(--run-id "$RUN_ID")
fi

if [[ -n "$RESUME" ]]; then
  COMMAND+=(--resume "$RESUME")
fi

printf 'Running command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'

"${COMMAND[@]}"
