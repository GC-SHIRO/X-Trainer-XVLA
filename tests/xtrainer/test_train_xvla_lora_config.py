"""Configuration contracts only; these tests do not load checkpoint weights."""

from pathlib import Path

import draccus
import yaml

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "configs/xtrainer/train_xvla_lora.yaml"
FULL_CONFIG = REPO_ROOT / "configs/xtrainer/train_xvla.yaml"


def test_lora_config_preserves_robot_contract():
    lora = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    full = yaml.safe_load(FULL_CONFIG.read_text(encoding="utf-8"))
    assert lora["rename_map"] == full["rename_map"]
    assert lora["xtrainer"] == full["xtrainer"]
    for key, value in full["policy"].items():
        if key not in {"path", "tokenizer_name"}:
            assert lora["policy"][key] == value
    assert lora["output_dir"] != full["output_dir"]
    assert lora["job_name"] != full["job_name"]
    assert "peft" not in full


def test_lora_config_parses_without_loading_base():
    parser._config_path_args.clear()
    parser._config_yaml_overrides.clear()
    cleaned = CONFIG
    try:
        cleaned = Path(
            parser.extract_path_fields_from_config(
                str(CONFIG), TrainPipelineConfig.__get_path_fields__()
            )
        )
        assert parser._config_path_args["policy"] == "/path/to/original-xvla-checkpoint"
        cfg = draccus.parse(config_class=TrainPipelineConfig, config_path=str(cleaned), args=[])
        assert cfg.peft.method_type == "LORA"
        assert cfg.peft.r == 8
        assert cfg.peft.lora_alpha == 16
        assert cfg.peft.target_modules == "all-linear"
        assert cfg.peft.full_training_modules == [
            "model.transformer.soft_prompt_hub",
            "model.transformer.action_encoder",
            "model.transformer.action_decoder",
        ]
        assert cfg.dataset.format_version == "v2.1"
        assert cfg.use_policy_training_preset is True
    finally:
        parser._config_path_args.clear()
        parser._config_yaml_overrides.clear()
        if cleaned != CONFIG:
            cleaned.unlink(missing_ok=True)
