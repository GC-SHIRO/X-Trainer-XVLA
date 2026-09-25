import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "scripts/xtrainer/train_xvla_lora.sh"


def _fake_env(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    captured = tmp_path / "args.txt"
    fake = bin_dir / "lerobot-train"
    fake.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURED"\n', encoding="utf-8")
    fake.chmod(0o755)
    return {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "CAPTURED": str(captured)}, captured


def test_lora_launcher_requires_base_model(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    result = subprocess.run(
        ["bash", str(LAUNCHER), "--dataset-root", str(dataset)],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "--base-model is required" in result.stderr


def test_lora_launcher_forwards_explicit_base_model(tmp_path):
    dataset, base = tmp_path / "dataset", tmp_path / "base"
    dataset.mkdir()
    base.mkdir()
    env, captured = _fake_env(tmp_path)
    result = subprocess.run(
        ["bash", str(LAUNCHER), "--dataset-root", str(dataset), "--base-model", str(base), "--skip-validation"],
        capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0, result.stderr
    args = captured.read_text(encoding="utf-8").splitlines()
    assert f"--policy.path={base}" in args
    assert any(arg.endswith("configs/xtrainer/train_xvla_lora.yaml") for arg in args)


def test_lora_launcher_resume_uses_checkpoint_config(tmp_path):
    dataset, base = tmp_path / "dataset", tmp_path / "base"
    dataset.mkdir()
    base.mkdir()
    env, captured = _fake_env(tmp_path)
    resume = "outputs/lora/checkpoints/last/pretrained_model"
    result = subprocess.run(
        ["bash", str(LAUNCHER), "--dataset-root", str(dataset), "--base-model", str(base),
         "--resume-checkpoint", resume, "--skip-validation"],
        capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0, result.stderr
    assert captured.read_text(encoding="utf-8").splitlines() == [
        "--resume=true", f"--config_path={resume}", f"--dataset.root={dataset}"
    ]
