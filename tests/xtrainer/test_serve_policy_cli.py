import os
import subprocess
import sys
from pathlib import Path

from scripts.xtrainer.serve_policy import parse_args


def test_policy_server_accepts_reference_cli_aliases():
    args = parse_args(["--model-path", "checkpoint", "--use-length", "25"])

    assert args.checkpoint == "checkpoint"
    assert args.actions_per_chunk == 25


def test_policy_server_action_logging_is_disabled_by_default():
    args = parse_args([])

    assert args.log_actions is False
    assert args.action_log_path is None


def test_policy_server_accepts_action_log_options():
    args = parse_args(["--log-actions", "--action-log-path", "actions.jsonl"])

    assert args.log_actions is True
    assert args.action_log_path == "actions.jsonl"


def test_policy_server_accepts_domain_override():
    args = parse_args(["--domain-id", "19"])

    assert args.domain_id == 19


def test_launcher_prefers_checkout_over_another_lerobot_installation(tmp_path):
    old_package = tmp_path / "lerobot"
    old_package.mkdir()
    (old_package / "__init__.py").write_text('raise RuntimeError("wrong LeRobot installation")\n')
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts/xtrainer/serve_policy.py"), "--help"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "--actions-per-chunk" in result.stdout
