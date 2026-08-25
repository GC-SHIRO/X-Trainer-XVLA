"""Contracts for the XVLA-only environment, model download, and deployment files."""

import subprocess
from pathlib import Path

import pytest
import yaml

from lerobot.policies.xvla.configuration_xvla import XVLAConfig


REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALLER = REPO_ROOT / "tools" / "install_xtrainer_env.sh"
HF_DOWNLOADER = REPO_ROOT / "tools" / "download_xvla_weights_hf.sh"
MODELSCOPE_DOWNLOADER = REPO_ROOT / "tools" / "download_xvla_weights_modelscope.sh"
DEPLOY_CONFIG = REPO_ROOT / "configs" / "xtrainer" / "deploy.yaml"


def test_environment_installer_help_is_xvla_only():
    result = subprocess.run(["bash", str(INSTALLER), "--help"], capture_output=True, text=True)

    assert result.returncode == 0
    assert "X-trainer XVLA" in result.stdout
    assert "xtrainer-xvla" in result.stdout
    assert "Smol" not in result.stdout


def test_xvla_downloader_help_uses_official_checkpoint():
    result = subprocess.run(["bash", str(HF_DOWNLOADER), "--help"], capture_output=True, text=True)

    assert result.returncode == 0
    assert "lerobot/xvla-base" in result.stdout
    assert "models/xvla-base" in result.stdout
    assert "VLM" not in result.stdout


def test_modelscope_downloader_matches_huggingface_layout():
    result = subprocess.run(["bash", str(MODELSCOPE_DOWNLOADER), "--help"], capture_output=True, text=True)

    assert result.returncode == 0
    assert "lerobot/xvla-base" in result.stdout
    assert "models/xvla-base" in result.stdout
    assert "xtrainer-xvla" in result.stdout
    assert "VLM" not in result.stdout

    script = MODELSCOPE_DOWNLOADER.read_text(encoding="utf-8")
    assert '"model.safetensors"' in script
    assert '"policy_preprocessor.json"' in script
    assert '"policy_postprocessor.json"' in script
    assert 'config.get("type") != "xvla"' in script


def test_deploy_config_matches_xvla_training_contract():
    config = yaml.safe_load(DEPLOY_CONFIG.read_text(encoding="utf-8"))

    assert config["policy"]["type"] == "xvla"
    assert config["policy"]["domain_id"] == 19
    assert config["policy"]["actions_per_chunk"] == 32
    assert config["xtrainer"]["action_dim"] == 14
    assert config["xtrainer"]["state_dim"] == 14
    assert config["xtrainer"]["chunk_size"] == 32


def test_xvla_domain_id_is_validated_against_domain_table():
    assert XVLAConfig(domain_id=19).domain_id == 19

    with pytest.raises(ValueError, match="domain_id"):
        XVLAConfig(domain_id=30, num_domains=30)
