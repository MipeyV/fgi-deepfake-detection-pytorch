#!/usr/bin/env bash
#SBATCH --job-name=fgi-eval
#SBATCH --partition=3090
#SBATCH --output=jobs/logs/%x-%j.out
#SBATCH --error=jobs/logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
CONFIG="${CONFIG:?Set CONFIG to the experiment config path}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the checkpoint path}"
SPLIT="${SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-}"
DEVICE="${DEVICE:-cuda}"
RUN_ID="${RUN_ID:-}"

cd "$PROJECT_DIR"
mkdir -p jobs/logs

if [[ -n "${VENV_PATH:-}" ]]; then
  source "$VENV_PATH/bin/activate"
fi

COMMAND=(
  python3 -u main.py eval
  --config "$CONFIG"
  --checkpoint "$CHECKPOINT"
  --split "$SPLIT"
  --device "$DEVICE"
)

if [[ -n "$BATCH_SIZE" ]]; then
  COMMAND+=(--batch-size "$BATCH_SIZE")
fi

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
