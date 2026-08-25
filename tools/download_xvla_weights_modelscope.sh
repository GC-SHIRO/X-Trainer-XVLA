#!/usr/bin/env bash
set -Eeuo pipefail

ENV_NAME="xtrainer-xvla"
MODEL_ID="lerobot/xvla-base"
TOKENIZER_MODEL_ID="AI-ModelScope/bart-large"
REVISION=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/models/xvla-base"
TOKENIZER_DIR="${OUTPUT_DIR}/tokenizer"

usage() {
  cat <<'USAGE'
Usage: bash tools/download_xvla_weights_modelscope.sh [OPTIONS]

Download the complete official XVLA base-model repository and the BART tokenizer
required by its processor from ModelScope. The default model and destination
match tools/download_xvla_weights_hf.sh exactly.

Options:
  --model-id ID         ModelScope model ID (default: lerobot/xvla-base)
  --tokenizer-model-id ID
                        ModelScope tokenizer model ID (default: AI-ModelScope/bart-large)
  --output-dir PATH     Local model directory (default: models/xvla-base)
  --revision REVISION   Optional branch, tag, or commit ID
  --env-name NAME       Conda environment name (default: xtrainer-xvla)
  -h, --help            Show this help

For private repositories, run `modelscope login` or provide
MODELSCOPE_API_TOKEN. Existing downloaded files are reused.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-id) MODEL_ID="${2:?--model-id requires a value}"; shift 2 ;;
    --tokenizer-model-id) TOKENIZER_MODEL_ID="${2:?--tokenizer-model-id requires a value}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:?--output-dir requires a value}"; shift 2 ;;
    --revision) REVISION="${2:?--revision requires a value}"; shift 2 ;;
    --env-name) ENV_NAME="${2:?--env-name requires a value}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'ERROR: unknown argument: %s\n' "$1" >&2; exit 1 ;;
  esac
done

TOKENIZER_DIR="${OUTPUT_DIR}/tokenizer"

[[ "${ENV_NAME}" =~ ^[A-Za-z0-9._-]+$ ]] || {
  printf 'ERROR: invalid Conda environment name: %s\n' "${ENV_NAME}" >&2
  exit 1
}
command -v conda >/dev/null 2>&1 || { printf 'ERROR: conda was not found\n' >&2; exit 1; }

mkdir -p "${OUTPUT_DIR}"
printf '[xtrainer-download-modelscope] XVLA checkpoint: %s\n' "${MODEL_ID}"
printf '[xtrainer-download-modelscope] BART tokenizer: %s\n' "${TOKENIZER_MODEL_ID}"
printf '[xtrainer-download-modelscope] destination: %s\n' "${OUTPUT_DIR}"

conda run --no-capture-output -n "${ENV_NAME}" \
  python - "${MODEL_ID}" "${OUTPUT_DIR}" "${TOKENIZER_MODEL_ID}" "${TOKENIZER_DIR}" "${REVISION}" <<'PY'
import json
import os
import sys
from pathlib import Path

from modelscope import snapshot_download
from transformers import AutoTokenizer

model_id, output_dir_value, tokenizer_model_id, tokenizer_dir_value, revision = sys.argv[1:]
output_dir = Path(output_dir_value).expanduser().resolve()
tokenizer_dir = Path(tokenizer_dir_value).expanduser().resolve()
token = os.environ.get("MODELSCOPE_API_TOKEN")
if token:
    from modelscope.hub.api import HubApi

    HubApi().login(token)

# Download the whole remote repository. Do not filter files here: the local
# ModelScope snapshot must retain every checkpoint, configuration, processor,
# and repository metadata artifact published by the upstream model.
downloaded_path = snapshot_download(
    model_id=model_id,
    revision=revision or None,
    local_dir=str(output_dir),
)

# XVLA uses BART only for text tokenization. Download exactly the tokenizer
# artifacts, not unused BART parameter checkpoints, into the shared local path.
tokenizer_files = (
    "config.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
downloaded_tokenizer_path = snapshot_download(
    model_id=tokenizer_model_id,
    # The XVLA revision does not exist in the independent BART repository.
    revision=None,
    local_dir=str(tokenizer_dir),
    allow_patterns=tokenizer_files,
)

required_files = (
    "config.json",
    "model.safetensors",
    "policy_postprocessor.json",
    "policy_preprocessor.json",
)
missing = [name for name in required_files if not (output_dir / name).is_file()]
empty = [name for name in required_files if (output_dir / name).is_file() and (output_dir / name).stat().st_size == 0]
if missing or empty:
    raise RuntimeError(f"incomplete XVLA checkpoint: missing={missing}, empty={empty}")

with (output_dir / "config.json").open(encoding="utf-8") as file:
    config = json.load(file)
if config.get("type") != "xvla":
    raise RuntimeError(f"unexpected policy type in config.json: {config.get('type')!r}")
if config.get("output_features", {}).get("action", {}).get("shape") != [20]:
    raise RuntimeError("XVLA base checkpoint must expose a 20-dimensional action head")
if config.get("florence_config", {}).get("model_type") != "florence2":
    raise RuntimeError("XVLA checkpoint does not contain the expected Florence-2 configuration")
for filename in ("policy_preprocessor.json", "policy_postprocessor.json"):
    with (output_dir / filename).open(encoding="utf-8") as file:
        json.load(file)

missing_tokenizer = [name for name in tokenizer_files if not (tokenizer_dir / name).is_file()]
if missing_tokenizer:
    raise RuntimeError(f"incomplete BART tokenizer: missing={missing_tokenizer}")
AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)

print(f"model download complete: {downloaded_path}")
print(f"tokenizer download complete: {downloaded_tokenizer_path}")
print(f"validated XVLA checkpoint: {output_dir}")
print(f"validated offline BART tokenizer: {tokenizer_dir}")
PY

printf '[xtrainer-download-modelscope] deployment checkpoint: %s\n' "${OUTPUT_DIR}"
