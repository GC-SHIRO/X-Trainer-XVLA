#!/usr/bin/env bash
set -Eeuo pipefail

ENV_NAME="xtrainer-xvla"
REPO_ID="lerobot/xvla-base"
TOKENIZER_REPO_ID="facebook/bart-large"
REVISION=""
ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/models/xvla-base"
TOKENIZER_DIR="${OUTPUT_DIR}/tokenizer"

usage() {
  cat <<'USAGE'
Usage: bash tools/download_xvla_weights_hf.sh [OPTIONS]

Download the complete official XVLA base-model repository and the BART tokenizer
required by its processor from Hugging Face.

Options:
  --repo-id ID          Model ID (default: lerobot/xvla-base)
  --tokenizer-repo-id ID
                        Tokenizer model ID (default: facebook/bart-large)
  --output-dir PATH     Local model directory (default: models/xvla-base)
  --revision REVISION   Optional branch, tag, or commit SHA
  --endpoint URL        Hugging Face endpoint (default: https://huggingface.co)
  --env-name NAME       Conda environment name (default: xtrainer-xvla)
  -h, --help            Show this help

For private or gated repositories, authenticate with `hf auth login` or provide
HF_TOKEN. Existing downloaded files are reused.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-id) REPO_ID="${2:?--repo-id requires a value}"; shift 2 ;;
    --tokenizer-repo-id) TOKENIZER_REPO_ID="${2:?--tokenizer-repo-id requires a value}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:?--output-dir requires a value}"; shift 2 ;;
    --revision) REVISION="${2:?--revision requires a value}"; shift 2 ;;
    --endpoint) ENDPOINT="${2:?--endpoint requires a value}"; shift 2 ;;
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
printf '[xtrainer-download-hf] XVLA checkpoint: %s\n' "${REPO_ID}"
printf '[xtrainer-download-hf] BART tokenizer: %s\n' "${TOKENIZER_REPO_ID}"
printf '[xtrainer-download-hf] endpoint: %s\n' "${ENDPOINT}"
printf '[xtrainer-download-hf] destination: %s\n' "${OUTPUT_DIR}"

conda run --no-capture-output -n "${ENV_NAME}" \
  python - "${REPO_ID}" "${OUTPUT_DIR}" "${TOKENIZER_REPO_ID}" "${TOKENIZER_DIR}" "${REVISION}" "${ENDPOINT}" <<'PY'
import json
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

repo_id, output_dir_value, tokenizer_repo_id, tokenizer_dir_value, revision, endpoint = sys.argv[1:]
output_dir = Path(output_dir_value).expanduser().resolve()
tokenizer_dir = Path(tokenizer_dir_value).expanduser().resolve()
downloaded_path = snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    revision=revision or None,
    local_dir=str(output_dir),
    endpoint=endpoint,
    token=os.environ.get("HF_TOKEN"),
)
tokenizer_files = (
    "config.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
downloaded_tokenizer_path = snapshot_download(
    repo_id=tokenizer_repo_id,
    repo_type="model",
    # The XVLA revision does not exist in the independent BART repository.
    revision=None,
    local_dir=str(tokenizer_dir),
    endpoint=endpoint,
    token=os.environ.get("HF_TOKEN"),
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

missing_tokenizer = [name for name in tokenizer_files if not (tokenizer_dir / name).is_file()]
if missing_tokenizer:
    raise RuntimeError(f"incomplete BART tokenizer: missing={missing_tokenizer}")
AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)

print(f"model download complete: {downloaded_path}")
print(f"tokenizer download complete: {downloaded_tokenizer_path}")
print(f"validated offline BART tokenizer: {tokenizer_dir}")
PY

printf '[xtrainer-download-hf] deployment checkpoint: %s\n' "${OUTPUT_DIR}"
