import asyncio
import json
import math

import cv2
import numpy as np
import pytest

from deploy.xtrainer.msgpack_numpy import protocol_metadata
from deploy.xtrainer.real.environment import (
    LEFT_WRIST_IMAGE_KEY,
    RIGHT_WRIST_IMAGE_KEY,
    STATE_KEY,
    TASK_KEY,
    TOP_IMAGE_KEY,
)
from scripts.xtrainer.run_real import (
    ControlActionLog,
    _apply_binary_gripper_close,
    _extract_action_chunk,
    _policy_payload,
    _rate_limit_action,
    parse_args,
    run,
    run_control_loop,
)


def _observation():
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    return {
        STATE_KEY: np.zeros(14, dtype=np.float32),
        TOP_IMAGE_KEY: image.copy(),
        LEFT_WRIST_IMAGE_KEY: image.copy(),
        RIGHT_WRIST_IMAGE_KEY: image.copy(),
        TASK_KEY: "pick",
    }


class MockEnvironment:
    def __init__(self):
        self.actions = []
        self.reset_called = False
        self.enabled = False
        self.closed = False
        self.reset_pose = None

    def reset(self):
        self.reset_called = True
        return _observation()

    def enable_arms(self):
        self.enabled = True

    def smooth_reset(self, reset_pose):
        self.reset_pose = np.asarray(reset_pose, dtype=np.float64).copy()

    def get_observation(self):
        return _observation()

    def apply_action(self, action, *, pace=True):
        assert pace is False
        applied = np.asarray(action, dtype=np.float64).copy()
        self.actions.append(applied)
        return applied

    def close(self):
        self.closed = True


class MockPolicy:
    def __init__(self, chunks, *, metadata=None, hang_after=None):
        self.chunks = [np.asarray(chunk, dtype=np.float64) for chunk in chunks]
        self.server_metadata = metadata or protocol_metadata(
            {
                "policy": {
                    "model_type": "xvla",
                    "schema_version": 1,
                    "action_dim": 14,
                    "state_dim": 14,
                    "domain_id": 19,
                }
            }
        )
        self.hang_after = hang_after
        self.infer_calls = 0
        self.active_infers = 0
        self.max_active_infers = 0
        self.payloads = []
        self.connected = False
        self.reset_called = False
        self.closed = False

    async def connect(self):
        self.connected = True
        return self.server_metadata

    async def reset(self):
        self.reset_called = True
        return {"ok": True}

    async def infer(self, payload):
        call_index = self.infer_calls
        self.infer_calls += 1
        self.payloads.append(payload)
        self.active_infers += 1
        self.max_active_infers = max(self.max_active_infers, self.active_infers)
        try:
            if self.hang_after is not None and call_index >= self.hang_after:
                await asyncio.Event().wait()
            await asyncio.sleep(0)
            return {"action": self.chunks[min(call_index, len(self.chunks) - 1)].copy()}
        finally:
            self.active_infers -= 1

    async def close(self):
        self.closed = True


def test_extract_action_chunk_validates_shape_and_finite_values():
    chunk = np.zeros((5, 14), dtype=np.float32)

    assert _extract_action_chunk({"action": chunk}, 3).shape == (3, 14)
    with pytest.raises(ValueError, match="shape"):
        _extract_action_chunk({"action": np.zeros((5, 13))}, 5)
    chunk[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        _extract_action_chunk({"action": chunk}, 5)


def test_policy_payload_applies_top_camera_crop_and_flips_only():
    observation = _observation()
    top_image = np.arange(10 * 20 * 3, dtype=np.uint8).reshape(10, 20, 3)
    left_wrist_image = np.arange(10 * 20 * 3, dtype=np.uint8).reshape(10, 20, 3)
    right_wrist_image = np.arange(10 * 20 * 3, dtype=np.uint8).reshape(10, 20, 3)
    observation[TOP_IMAGE_KEY] = top_image
    observation[LEFT_WRIST_IMAGE_KEY] = left_wrist_image
    observation[RIGHT_WRIST_IMAGE_KEY] = right_wrist_image

    payload = _policy_payload(observation)

    expected_top = cv2.resize(top_image[2:8, 4:16], (20, 10))[::-1, ::-1]
    np.testing.assert_array_equal(payload["images"]["top"], expected_top)
    assert payload["images"]["top"].shape == top_image.shape
    np.testing.assert_array_equal(payload["images"]["left_wrist"], left_wrist_image)
    np.testing.assert_array_equal(payload["images"]["right_wrist"], right_wrist_image)


def test_final_rate_limit_caps_action_change():
    target = np.full(14, 7.0)

    limited = _rate_limit_action(target, np.zeros(14), max_delta_per_step=0.2)

    np.testing.assert_allclose(limited, 0.2)


def test_bin_gripper_smoothly_closes_values_below_threshold():
    action = np.full(14, 0.8)
    action[6] = 0.49
    action[13] = 0.2
    previous = np.full(14, 0.8)

    transformed = _apply_binary_gripper_close(action, previous)

    assert transformed[6] == pytest.approx(0.75)
    assert transformed[13] == pytest.approx(0.75)
    assert transformed[0] == pytest.approx(0.8)


def test_cli_uses_planned_camera_defaults_and_reserved_switch():
    args = parse_args(["--host", "127.0.0.1"])

    assert args.camera_top_serial == "409122273405"
    assert args.camera_left_wrist_serial == "412622272997"
    assert args.camera_right_wrist_serial == "412622271417"
    assert args.action_horizon == 32
    assert args.domain_id == 19
    assert args.observation_similarity_epsilon is None
    assert args.execute is False
    assert args.log_control is False
    assert args.bin_gripper is False
    assert args.control_log_path is None
    assert math.isinf(args.max_joint_delta)
    assert math.isinf(args.max_gripper_delta)
    assert args.ramp_step == pytest.approx(0.01)
    assert args.ramp_max_steps == 100
    assert args.gripper_update_threshold == 0.0


def test_cli_accepts_optional_client_control_log_path(tmp_path):
    log_path = tmp_path / "client.jsonl"

    args = parse_args(["--host", "127.0.0.1", "--log-control", "--control-log-path", str(log_path)])

    assert args.log_control is True
    assert args.control_log_path == log_path


def test_cli_accepts_reference_hardware_option_names():
    args = parse_args(
        [
            "--host",
            "127.0.0.1",
            "--left-arm-ip",
            "192.168.5.11",
            "--right-arm-ip",
            "192.168.5.12",
            "--top-camera-serial",
            "top",
            "--left-wrist-camera-serial",
            "left",
            "--right-wrist-camera-serial",
            "right",
        ]
    )

    assert args.left_robot_ip == "192.168.5.11"
    assert args.right_robot_ip == "192.168.5.12"
    assert (args.camera_top_serial, args.camera_left_wrist_serial, args.camera_right_wrist_serial) == (
        "top",
        "left",
        "right",
    )
def test_control_loop_executes_complete_chunks_then_holds_measured_pose():
    async def exercise():
        policy = MockPolicy([np.zeros((4, 14)), np.full((4, 14), 10.0)])
        environment = MockEnvironment()

        async def yield_control(_seconds):
            for _ in range(3):
                await asyncio.sleep(0)

        await run_control_loop(
            policy,
            environment,
            action_horizon=4,
            control_hz=20.0,
            max_steps=5,
            request_timeout_s=1.0,
            max_delta_per_step=0.0,
            monotonic_fn=lambda: 0.0,
            sleep_fn=yield_control,
        )
        return policy, environment

    policy, environment = asyncio.run(exercise())

    assert len(environment.actions) == 6
    np.testing.assert_allclose(environment.actions[:5], 0.0)
    np.testing.assert_allclose(environment.actions[5], 10.0)
    assert policy.infer_calls == 2
    assert policy.max_active_infers == 1
    assert set(policy.payloads[0]) == {"state", "images", "task"}
    assert set(policy.payloads[0]["images"]) == {"top", "left_wrist", "right_wrist"}


def test_control_loop_records_hold_without_fallback(tmp_path):
    async def exercise(log_path):
        policy = MockPolicy([np.zeros((2, 14)), np.ones((2, 14))])
        environment = MockEnvironment()
        control_log = ControlActionLog(log_path)
        try:
            await run_control_loop(
                policy,
                environment,
                action_horizon=2,
                control_hz=1000.0,
                max_steps=3,
                request_timeout_s=1.0,
                max_delta_per_step=0.0,
                control_log=control_log,
            )
        finally:
            control_log.close()

    log_path = tmp_path / "control.jsonl"
    asyncio.run(exercise(log_path))
    records = [json.loads(line) for line in log_path.read_text().splitlines()]

    assert [record["event"] for record in records] == [
        "inference_request",
        "inference_response",
        "control_step",
        "control_step",
        "control_hold",
        "inference_request",
        "inference_response",
        "control_step",
    ]
    assert records[4]["control_timestep"] == 2
    assert all(not record.get("used_fallback", False) for record in records)


def test_client_control_log_records_state_response_and_applied_action(tmp_path):
    async def exercise(log_path):
        policy = MockPolicy([np.full((2, 14), 0.25)])
        environment = MockEnvironment()
        control_log = ControlActionLog(log_path)
        try:
            await run_control_loop(
                policy,
                environment,
                action_horizon=2,
                control_hz=100.0,
                max_steps=1,
                request_timeout_s=1.0,
                max_delta_per_step=0.0,
                control_log=control_log,
                monotonic_fn=lambda: 0.0,
                sleep_fn=asyncio.sleep,
            )
        finally:
            control_log.close()

    log_path = tmp_path / "control.jsonl"
    asyncio.run(exercise(log_path))
    records = [json.loads(line) for line in log_path.read_text().splitlines()]

    assert [record["event"] for record in records] == [
        "inference_request",
        "inference_response",
        "control_step",
    ]
    assert records[0]["state"] == [0.0] * 14
    assert records[0]["task"] == "pick"
    assert records[1]["retained_actions"] == [[0.25] * 14, [0.25] * 14]
    assert records[2]["source_observation_timestep"] == 0
    assert records[2]["used_fallback"] is False
    assert records[2]["queued_action"] == [0.25] * 14
    assert records[2]["rate_limited_action"] == [0.25] * 14
    assert records[2]["applied_action"] == [0.25] * 14


def test_chunk_request_timeout_closes_policy_and_hardware():
    args = parse_args(
        [
            "--host",
            "127.0.0.1",
            "--execute",
            "--action-horizon",
            "2",
            "--max-steps",
            "5",
            "--control-hz",
            "100",
            "--request-timeout",
            "0.001",
        ]
    )
    policy = MockPolicy([np.zeros((2, 14))], hang_after=1)
    environment = MockEnvironment()

    with pytest.raises(TimeoutError):
        asyncio.run(run(args, policy=policy, environment=environment))

    assert environment.reset_called
    assert environment.enabled
    assert environment.closed
    assert policy.connected
    assert policy.reset_called
    assert policy.closed
    assert policy.max_active_infers == 1


def test_run_applies_reset_pose_from_policy_metadata_and_cleans_up():
    reset_pose = np.arange(14, dtype=np.float64) / 10
    metadata = protocol_metadata(
        {
            "policy": {
                "model_type": "xvla",
                "schema_version": 1,
                "action_dim": 14,
                "state_dim": 14,
                "domain_id": 19,
                "reset_pose": reset_pose.tolist(),
            }
        }
    )
    args = parse_args(
        [
            "--host",
            "127.0.0.1",
            "--execute",
            "--action-horizon",
            "1",
            "--max-steps",
            "1",
            "--control-hz",
            "1000",
        ]
    )
    policy = MockPolicy([np.zeros((1, 14))], metadata=metadata)
    environment = MockEnvironment()

    asyncio.run(run(args, policy=policy, environment=environment))

    np.testing.assert_allclose(environment.reset_pose, reset_pose)
    assert environment.closed
    assert policy.closed


def test_run_requires_explicit_execute_before_connecting_resources():
    args = parse_args(["--host", "127.0.0.1"])
    policy = MockPolicy([np.zeros((1, 14))])
    environment = MockEnvironment()

    with pytest.raises(RuntimeError, match="--execute"):
        asyncio.run(run(args, policy=policy, environment=environment))

    assert not policy.connected
    assert not environment.reset_called
