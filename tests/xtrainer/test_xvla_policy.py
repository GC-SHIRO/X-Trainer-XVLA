"""CPU-only tests for the XVLA X-trainer policy adapter.

These tests avoid loading a real XVLA checkpoint: XVLAPolicy.from_pretrained
and the processor factory are monkeypatched with lightweight fakes so the tests
run on CPU without network access or GPU.
"""

from __future__ import annotations

import json
import numpy as np
import pytest
import torch

from deploy.xtrainer import xvla_policy as xvla_policy_module
from deploy.xtrainer.xvla_policy import ACTION_DIM, STATE_DIM, XVLAXTrainerPolicy

CAMERA_KEYS = {
    "top": "observation.images.top",
    "left_wrist": "observation.images.left_wrist",
    "right_wrist": "observation.images.right_wrist",
}


class FakeConfig:
    domain_id = 19
    action_mode = "auto"
    chunk_size = 32


class FakeActionSpace:
    real_dim = ACTION_DIM


class FakeModel:
    def __init__(self):
        self.action_space = FakeActionSpace()


class FakePolicy:
    """Stand in for XVLAPolicy: record calls and return a deterministic chunk."""

    def __init__(self):
        self.config = FakeConfig()
        self.model = FakeModel()
        self.reset_calls = 0
        self.chunk_len = 5
        self.last_batch = None

    def to(self, _device):
        return self

    def eval(self):
        return self

    def reset(self):
        self.reset_calls += 1

    def predict_action_chunk(self, batch):
        self.last_batch = batch
        return torch.zeros(1, self.chunk_len, ACTION_DIM)


class IdentityPipeline:
    """Stands in for a PolicyProcessorPipeline: passes data through unchanged."""

    def __call__(self, data):
        return data


def _make_policy(monkeypatch, *, fake_policy=None, load_calls=None, action_log_path=None):
    fake_policy = fake_policy or FakePolicy()
    load_calls = load_calls if load_calls is not None else []

    def fake_from_pretrained(path):
        load_calls.append(path)
        return fake_policy

    monkeypatch.setattr(xvla_policy_module.XVLAPolicy, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(
        xvla_policy_module,
        "make_xvla_pre_post_processors",
        lambda config: (IdentityPipeline(), IdentityPipeline()),
    )

    def fail_make_pre_post_processors(**_kwargs):
        raise FileNotFoundError("no saved processors for this fake checkpoint")

    monkeypatch.setattr(xvla_policy_module, "make_pre_post_processors", fail_make_pre_post_processors)

    policy = XVLAXTrainerPolicy(
        checkpoint="fake/checkpoint",
        device="cpu",
        camera_keys=CAMERA_KEYS,
        warmup=False,
        action_log_path=action_log_path,
    )
    return policy, fake_policy


def _valid_payload():
    return {
        "state": np.zeros(STATE_DIM, dtype=np.float32),
        "images": {name: np.zeros((480, 640, 3), dtype=np.uint8) for name in CAMERA_KEYS},
        "task": "pick up the object",
    }


def test_infer_returns_expected_action_shape(monkeypatch):
    policy, fake_policy = _make_policy(monkeypatch)

    result = policy.infer(_valid_payload())

    assert set(result.keys()) == {"action"}
    assert result["action"].shape == (fake_policy.chunk_len, ACTION_DIM)
    assert result["action"].dtype == np.float32
    assert np.isfinite(result["action"]).all()


def test_infer_truncates_to_actions_per_chunk(monkeypatch):
    policy, _ = _make_policy(monkeypatch)
    policy.actions_per_chunk = 3

    result = policy.infer(_valid_payload())

    assert result["action"].shape == (3, ACTION_DIM)


def test_infer_writes_action_log_when_enabled(monkeypatch, tmp_path):
    log_path = tmp_path / "actions.jsonl"
    policy, _ = _make_policy(monkeypatch, action_log_path=log_path)
    payload = _valid_payload()
    payload["images"] = {
        name: np.full((2, 3, 3), index, dtype=np.uint8)
        for index, name in enumerate(CAMERA_KEYS)
    }

    policy.infer(payload)
    policy.close()

    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert np.asarray(records[0]["action"]).shape == (5, ACTION_DIM)
    assert set(records[0]["images"]) == set(CAMERA_KEYS)
    for name, image in payload["images"].items():
        np.testing.assert_array_equal(records[0]["images"][name], image)


def test_infer_rejects_wrong_state_shape_before_calling_model(monkeypatch):
    policy, fake_policy = _make_policy(monkeypatch)
    payload = _valid_payload()
    payload["state"] = np.zeros(10, dtype=np.float32)

    with pytest.raises(ValueError, match="state"):
        policy.infer(payload)
    assert fake_policy.last_batch is None


def test_infer_rejects_non_finite_state_before_calling_model(monkeypatch):
    policy, fake_policy = _make_policy(monkeypatch)
    payload = _valid_payload()
    payload["state"][0] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        policy.infer(payload)
    assert fake_policy.last_batch is None


def test_infer_rejects_missing_camera_before_calling_model(monkeypatch):
    policy, fake_policy = _make_policy(monkeypatch)
    payload = _valid_payload()
    del payload["images"]["left_wrist"]

    with pytest.raises(ValueError, match="left_wrist"):
        policy.infer(payload)
    assert fake_policy.last_batch is None


def test_infer_rejects_wrong_image_dtype_before_calling_model(monkeypatch):
    policy, fake_policy = _make_policy(monkeypatch)
    payload = _valid_payload()
    payload["images"]["top"] = np.zeros((480, 640, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="uint8"):
        policy.infer(payload)
    assert fake_policy.last_batch is None


def test_infer_rejects_wrong_image_shape_before_calling_model(monkeypatch):
    policy, fake_policy = _make_policy(monkeypatch)
    payload = _valid_payload()
    payload["images"]["top"] = np.zeros((480, 640, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match="shape"):
        policy.infer(payload)
    assert fake_policy.last_batch is None


def test_infer_rejects_missing_task_before_calling_model(monkeypatch):
    policy, fake_policy = _make_policy(monkeypatch)
    payload = _valid_payload()
    payload["task"] = ""

    with pytest.raises(ValueError, match="task"):
        policy.infer(payload)
    assert fake_policy.last_batch is None


def test_infer_rejects_non_finite_model_output(monkeypatch):
    fake_policy = FakePolicy()
    fake_policy.predict_action_chunk = lambda batch: torch.full((1, 5, ACTION_DIM), float("nan"))
    policy, _ = _make_policy(monkeypatch, fake_policy=fake_policy)

    with pytest.raises(ValueError, match="non-finite"):
        policy.infer(_valid_payload())


def test_reset_delegates_to_underlying_policy(monkeypatch):
    policy, fake_policy = _make_policy(monkeypatch)

    policy.reset()

    assert fake_policy.reset_calls == 1


def test_metadata_reports_schema_and_dims(monkeypatch):
    policy, _ = _make_policy(monkeypatch)

    metadata = policy.metadata()

    assert metadata["model_type"] == "xvla"
    assert metadata["action_dim"] == ACTION_DIM
    assert metadata["state_dim"] == STATE_DIM
    assert metadata["domain_id"] == 19


def test_checkpoint_must_be_adapted_to_14d_auto_action_space(monkeypatch):
    fake_policy = FakePolicy()
    fake_policy.model.action_space.real_dim = 20

    with pytest.raises(ValueError, match="real action dim 14"):
        _make_policy(monkeypatch, fake_policy=fake_policy)


def test_checkpoint_domain_must_match_deployment_domain(monkeypatch):
    fake_policy = FakePolicy()
    fake_policy.config.domain_id = 18

    with pytest.raises(ValueError, match="does not match"):
        _make_policy(monkeypatch, fake_policy=fake_policy)


def test_saved_processor_receives_device_and_domain_overrides(monkeypatch):
    fake_policy = FakePolicy()
    captured = {}

    def fake_from_pretrained(path):
        return fake_policy

    def fake_processors(**kwargs):
        captured.update(kwargs)
        return IdentityPipeline(), IdentityPipeline()

    monkeypatch.setattr(xvla_policy_module.XVLAPolicy, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(xvla_policy_module, "make_pre_post_processors", fake_processors)

    XVLAXTrainerPolicy(
        checkpoint="fake/checkpoint",
        device="cpu",
        domain_id=19,
        camera_keys=CAMERA_KEYS,
        warmup=False,
    )

    assert fake_policy.config.domain_id == 19
    assert captured["preprocessor_overrides"]["device_processor"] == {"device": "cpu"}
    assert captured["preprocessor_overrides"]["xvla_add_domain_id"] == {"domain_id": 19}
