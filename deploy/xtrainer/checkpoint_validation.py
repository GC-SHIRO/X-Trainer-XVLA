"""Lightweight local deployment checks, without importing the model runtime."""

import json
import math
from pathlib import Path


def validate_deployment_checkpoint(checkpoint: str) -> bool:
    """Return whether this is a validated merge export; leave legacy models unchanged."""
    root = Path(checkpoint)
    if not root.is_dir():
        return False
    if (root / "EXPORT_FAILED.txt").exists():
        raise ValueError("This checkpoint failed export validation and must not be deployed.")
    if (root / "adapter_config.json").exists():
        raise ValueError("Deploy a merged full checkpoint, not a standalone PEFT adapter.")
    report_path = root / "merge_report.json"
    if not report_path.exists():
        if (root / "xvla_base_manifest.json").exists():
            raise ValueError("LoRA export has no successful merge_report.json.")
        return False
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("validated") is not True:
        raise ValueError("LoRA merge report does not confirm successful validation.")
    for key in ("merge_max_abs_error", "reload_max_abs_error"):
        value = report.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid merge report metric: {key}")
    for name in ("config.json", "model.safetensors", "policy_preprocessor.json",
                 "policy_postprocessor.json"):
        if not (root / name).is_file():
            raise ValueError(f"Merged deployment checkpoint is missing {name}")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if config.get("use_peft", False):
        raise ValueError("Merged deployment config must have use_peft=false.")
    if not (root / "tokenizer").is_dir():
        raise ValueError("Merged deployment checkpoint is missing tokenizer resources.")
    return True
