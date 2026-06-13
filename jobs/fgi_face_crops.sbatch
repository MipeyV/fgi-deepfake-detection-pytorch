#!/usr/bin/env bash
#SBATCH --job-name=fgi-crops
#SBATCH --partition=3090
#SBATCH --output=jobs/logs/%x-%j.out
#SBATCH --error=jobs/logs/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
SPLIT="${SPLIT:?Set SPLIT to train, val, or test}"
DETECTOR_MODEL="${DETECTOR_MODEL:-models/face_detection_yunet_2023mar.onnx}"

case "$SPLIT" in
  train|val|test) ;;
  *)
    echo "SPLIT must be train, val, or test" >&2
    exit 2
    ;;
esac

cd "$PROJECT_DIR"
mkdir -p jobs/logs runs/fgi-face-crops

if [[ -n "${VENV_PATH:-}" ]]; then
  source "$VENV_PATH/bin/activate"
fi

COMMAND=(
  python3 -u main.py fgi-face-crops
  --manifest "data/manifests/${SPLIT}_manifest.csv"
  --output-dir data/processed_fgi
  --output-manifest "data/manifests_fgi/${SPLIT}_manifest.csv"
  --detector-model "$DETECTOR_MODEL"
  --missing-face-policy skip
  --contact-sheet "runs/fgi-face-crops/${SPLIT}_contact_sheet.png"
)

printf 'Running command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'

"${COMMAND[@]}"
