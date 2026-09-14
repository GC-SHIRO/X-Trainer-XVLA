#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
TRAIN_CONFIG="${XTRAINER_LORA_TRAIN_CONFIG:-${REPO_ROOT}/configs/xtrainer/train_xvla_lora.yaml}"
VALIDATOR="${SCRIPT_DIR}/validate_dataset_v21.py"
DATASET_ROOT=""
BASE_MODEL=""
OUTPUT_DIR=""
DEVICE=""
BATCH_SIZE=""
STEPS=""
RESUME_CHECKPOINT=""
SKIP_VALIDATION=false

usage() {
    cat <<EOF
Usage: scripts/xtrainer/train_xvla_lora.sh --dataset-root PATH --base-model PATH [options]
  --dataset-root PATH        Local LeRobot Dataset v2.1 directory.
  --base-model PATH          Local original XVLA checkpoint directory.
  --config PATH              LoRA YAML override.
  --output-dir PATH          Training output override.
  --device DEVICE            Device override.
  --batch-size N             Batch size override.
  --steps N                  Training steps override.
  --resume-checkpoint PATH   Resume from a LoRA checkpoint.
  --skip-validation          Skip dataset validation.
  -h, --help                 Show this help.
EOF
}

require_value() {
    if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
        echo "error: $1 requires a value" >&2
        usage >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset-root|--base-model|--config|--output-dir|--device|--batch-size|--steps|--resume-checkpoint)
            require_value "$1" "${2:-}"
            case "$1" in
                --dataset-root) DATASET_ROOT="$2" ;;
                --base-model) BASE_MODEL="$2" ;;
                --config) TRAIN_CONFIG="$2" ;;
                --output-dir) OUTPUT_DIR="$2" ;;
                --device) DEVICE="$2" ;;
                --batch-size) BATCH_SIZE="$2" ;;
                --steps) STEPS="$2" ;;
                --resume-checkpoint) RESUME_CHECKPOINT="$2" ;;
            esac
            shift 2 ;;
        --skip-validation) SKIP_VALIDATION=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "error: unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "$DATASET_ROOT" ]] || { echo "error: --dataset-root is required" >&2; exit 2; }
[[ -n "$BASE_MODEL" ]] || { echo "error: --base-model is required" >&2; exit 2; }
[[ -d "$DATASET_ROOT" ]] || { echo "error: dataset root does not exist: $DATASET_ROOT" >&2; exit 2; }
[[ -d "$BASE_MODEL" ]] || { echo "error: base model does not exist: $BASE_MODEL" >&2; exit 2; }
[[ -f "$TRAIN_CONFIG" ]] || { echo "error: training config does not exist: $TRAIN_CONFIG" >&2; exit 2; }
command -v python >/dev/null 2>&1 || { echo "error: python was not found" >&2; exit 127; }
command -v lerobot-train >/dev/null 2>&1 || { echo "error: lerobot-train was not found" >&2; exit 127; }

if [[ "$SKIP_VALIDATION" == false ]]; then
    python "$VALIDATOR" --root "$DATASET_ROOT"
fi

train_args=()
if [[ -n "$RESUME_CHECKPOINT" ]]; then
    train_args+=("--resume=true" "--config_path=$RESUME_CHECKPOINT")
else
    train_args+=("--config_path=$TRAIN_CONFIG" "--policy.path=$BASE_MODEL")
fi
train_args+=("--dataset.root=$DATASET_ROOT")
[[ -n "$OUTPUT_DIR" ]] && train_args+=("--output_dir=$OUTPUT_DIR")
[[ -n "$DEVICE" ]] && train_args+=("--policy.device=$DEVICE")
[[ -n "$BATCH_SIZE" ]] && train_args+=("--batch_size=$BATCH_SIZE")
[[ -n "$STEPS" ]] && train_args+=("--steps=$STEPS")

echo "starting XVLA LoRA training"
echo "base model: $BASE_MODEL"
echo "config: $TRAIN_CONFIG"
exec lerobot-train "${train_args[@]}"
