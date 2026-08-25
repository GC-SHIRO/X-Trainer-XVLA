#!/usr/bin/env bash
set -Eeuo pipefail

ENV_NAME="xtrainer-xvla"
MODEL_ID="lerobot/xvla-base"
REVISION=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/models/xvla-base"

usage() {
  cat <<'USAGE'
Usage: bash tools/download_xvla_weights_modelscope.sh [OPTIONS]

Download the official XVLA base checkpoint from ModelScope. The default model
and destination match tools/download_xvla_weights_hf.sh exactly.

Options:
  --model-id ID         ModelScope model ID (default: lerobot/xvla-base)
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
    --output-dir) OUTPUT_DIR="${2:?--output-dir requires a value}"; shift 2 ;;
    --revision) REVISION="${2:?--revision requires a value}"; shift 2 ;;
    --env-name) ENV_NAME="${2:?--env-name requires a value}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'ERROR: unknown argument: %s\n' "$1" >&2; exit 1 ;;
  esac
done

[[ "${ENV_NAME}" =~ ^[A-Za-z0-9._-]+$ ]] || {
  printf 'ERROR: invalid Conda environment name: %s\n' "${ENV_NAME}" >&2
  exit 1
}
command -v conda >/dev/null 2>&1 || { printf 'ERROR: conda was not found\n' >&2; exit 1; }

mkdir -p "${OUTPUT_DIR}"
printf '[xtrainer-download-modelscope] XVLA checkpoint: %s\n' "${MODEL_ID}"
printf '[xtrainer-download-modelscope] destination: %s\n' "${OUTPUT_DIR}"

conda run --no-capture-output -n "${ENV_NAME}" \
  python - "${MODEL_ID}" "${OUTPUT_DIR}" "${REVISION}" <<'PY'
import json
import os
import sys
from pathlib import Path

from modelscope import snapshot_download

model_id, output_dir_value, revision = sys.argv[1:]
output_dir = Path(output_dir_value).expanduser().resolve()
token = os.environ.get("MODELSCOPE_API_TOKEN")
if token:
    from modelscope.hub.api import HubApi

    HubApi().login(token)

# Keep the local checkpoint payload aligned with the Hugging Face repository.
# ModelScope adds a service-specific configuration.json that LeRobot does not
# consume, so only the shared XVLA payload is downloaded.
shared_files = [
    "README.md",
    "config.json",
    "model.safetensors",
    "policy_postprocessor.json",
    "policy_preprocessor.json",
]
downloaded_path = snapshot_download(
    model_id=model_id,
    revision=revision or None,
    local_dir=str(output_dir),
    allow_patterns=shared_files,
)

required_files = shared_files[1:]
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

print(f"model download complete: {downloaded_path}")
print(f"validated XVLA checkpoint: {output_dir}")
PY

printf '[xtrainer-download-modelscope] deployment checkpoint: %s\n' "${OUTPUT_DIR}"
