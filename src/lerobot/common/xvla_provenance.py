"""Local XVLA adapter provenance, independent of torch and PEFT imports."""

import hashlib
import json
import logging
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

MANIFEST = "xvla_base_manifest.json"


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_base(directory: str | Path) -> dict:
    root = Path(directory).resolve(strict=True)
    if not (root / "config.json").is_file():
        raise ValueError(f"XVLA base is missing config.json: {root}")
    weights = sorted(
        set(root.glob("model*.safetensors")) | set(root.glob("pytorch_model*.bin"))
    )
    if not weights:
        raise ValueError(f"XVLA base has no supported weight files: {root}")
    files = sorted(set(weights) | {root / "config.json"} | set(root.glob("*.index.json")))
    return {
        "source": str(root),
        "files": {
            path.name: {"size": path.stat().st_size, "sha256": _digest(path)} for path in files
        },
    }


def create_manifest(base: str | Path, tokenizer: str) -> dict:
    dependencies = {}
    for name in ("lerobot", "torch", "transformers", "peft", "accelerate"):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3],
            capture_output=True, text=True, check=True, timeout=10,
        )
        commit = result.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=Path(__file__).resolve().parents[3],
            capture_output=True, text=True, check=True, timeout=10,
        )
        dirty = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit = None
        dirty = None
    tokenizer_path = Path(tokenizer)
    tokenizer_info = {"source": tokenizer, "files": {}}
    if tokenizer_path.is_dir():
        tokenizer_info["source"] = str(tokenizer_path.resolve())
        tokenizer_info["files"] = {
            path.relative_to(tokenizer_path).as_posix(): _digest(path)
            for path in sorted(tokenizer_path.rglob("*")) if path.is_file()
        }
    return {
        "schema_version": 1,
        "base": fingerprint_base(base),
        "tokenizer": tokenizer_info,
        "code_commit": commit,
        "code_dirty": dirty,
        "python": platform.python_version(),
        "dependencies": dependencies,
    }


def verify_manifest(adapter: str | Path, base: str | Path) -> dict | None:
    path = Path(adapter) / MANIFEST
    if not path.is_file():
        logging.warning("XVLA adapter has no base manifest; base identity cannot be verified: %s", adapter)
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported XVLA base manifest version")
    actual = fingerprint_base(base)
    if actual["files"] != manifest["base"]["files"]:
        raise ValueError("XVLA base checkpoint identity mismatch; refusing to load adapter")
    return manifest


def save_manifest(policy, directory: str | Path) -> None:
    manifest = getattr(policy, "_xvla_base_manifest", None)
    if manifest is None:
        return
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    saved = dict(manifest)
    saved["artifacts"] = {
        path.relative_to(target).as_posix(): {"size": path.stat().st_size, "sha256": _digest(path)}
        for path in sorted(target.rglob("*"))
        if path.is_file() and path.name != MANIFEST
    }
    (target / MANIFEST).write_text(json.dumps(saved, indent=2) + "\n", encoding="utf-8")
