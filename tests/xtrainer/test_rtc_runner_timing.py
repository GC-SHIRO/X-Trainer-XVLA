"""RTC prefix alignment under slow observation acquisition."""

import asyncio

# Load the hardware-free runner without importing optional camera dependencies.
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "rtc_runner_under_test", Path(__file__).resolve().parents[2] / "deploy/xtrainer/rtc_runner.py"
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
execute_chunks = _module.execute_chunks


@pytest.mark.parametrize("observation_delay", [0.0, 0.12])
@pytest.mark.parametrize("ema_enabled", [False, True])
def test_rtc_switch_preserves_action_sequence_when_control_ticks_overrun(observation_delay, ema_enabled):
    async def scenario():
        now = [0.0]
        sent = []
        events = []
        switches = []

        class Environment:
            def get_observation(self):
                now[0] += observation_delay
                return {"state": np.zeros(14)}

            def apply_action(self, action, *, pace):
                sent.append(np.asarray(action).copy())
                now[0] += 0.005
                return action

        class Policy:
            metadata = {"policy": {"rtc_enabled": True, "chunk_size": 50}}

            async def infer(self, payload):
                # Allow execution to proceed while this prediction is pending.
                for _ in range(3):
                    await asyncio.sleep(0)
                prefix = payload.get("prev_chunk_left_over")
                start = 0 if prefix is None else int(prefix[0, 0])
                actions = np.repeat(np.arange(start, start + 50)[:, None], 14, axis=1)
                return {"action": actions, "model_action": actions}

        class Log:
            def write(self, event, **fields):
                if event == "control_step":
                    events.append(fields)
                elif event == "rtc_switch":
                    switches.append(fields)

        async def sleep(seconds):
            now[0] += seconds
            await asyncio.sleep(0)

        await execute_chunks(
            Policy(),
            Environment(),
            payload_fn=lambda obs: dict(obs),
            extract_fn=lambda result, horizon: result["action"][:horizon],
            rate_limit_fn=lambda action, previous, limit: action,
            gripper_fn=None,
            state_key="state",
            action_horizon=50,
            control_hz=30,
            max_steps=100,
            timeout=5,
            max_delta=0,
            ema_enabled=ema_enabled,
            log=Log(),
            clock=lambda: now[0],
            sleep=sleep,
        )
        # Exclude the final cleanup hold. Predictions reproduce the prefix
        # exactly, so any missing value is a client-side alignment error.
        expected = np.arange(100, dtype=float)
        if ema_enabled:
            for index in range(1, len(expected)):
                expected[index] = expected[index - 1] + 0.25 * (index - expected[index - 1])
        np.testing.assert_allclose(np.asarray(sent[:100])[:, 0], expected)
        np.testing.assert_array_equal(np.asarray([e["queued_action"] for e in events])[:, 0], np.arange(100))
        assert len(events) == 100
        assert switches
        assert all(event["revised_next_target_delta_rad"] == 0 for event in switches)
        assert events[0]["send_interval_ms"] is None
        assert all(event["apply_action_ms"] >= 0 for event in events)
        if observation_delay:
            assert max(event["send_interval_ms"] or 0 for event in events) > 100

    asyncio.run(scenario())


@pytest.mark.parametrize("ema_enabled", [False, True])
@pytest.mark.parametrize("binary_gripper", [False, True])
def test_ema_uses_applied_target_and_preserves_binary_gripper(ema_enabled, binary_gripper):
    async def scenario():
        initial = np.full(14, 0.2)
        raw = np.full((3, 14), 1.0)
        raw[1, :6] = -1.0
        raw[1, 7:13] = -1.0
        raw[:, 6] = [0.4, 0.9, 0.8]
        raw[:, 13] = [0.9, 0.8, 0.7]
        sent, events = [], []

        class Environment:
            def get_observation(self):
                return {"state": initial.copy()}

            def apply_action(self, action, *, pace):
                applied = np.clip(action, -0.3, 0.3)
                sent.append(applied.copy())
                return applied

        class Policy:
            async def infer(self, payload):
                return {"action": raw}

        class Log:
            def write(self, event, **fields):
                if event == "control_step":
                    events.append(fields)

        def binary(action, previous):
            result = action.copy()
            result[[6, 13]] = 0.1
            return result

        await execute_chunks(
            Policy(),
            Environment(),
            payload_fn=dict,
            extract_fn=lambda result, horizon: result["action"],
            rate_limit_fn=lambda action, previous, limit: action,
            gripper_fn=binary if binary_gripper else None,
            state_key="state",
            action_horizon=3,
            control_hz=33,
            max_steps=3,
            timeout=5,
            max_delta=0,
            log=Log(),
            clock=lambda: 0,
            sleep=asyncio.sleep,
            ema_enabled=ema_enabled,
        )
        filtered = np.array([event["filtered_action"] for event in events])
        joints = np.r_[0:6, 7:13]
        if ema_enabled:
            # Step 1 starts from hardware-clamped 0.3, not the previous EMA target 0.4.
            np.testing.assert_allclose(
                filtered[:, joints], np.repeat([[0.4], [-0.025], [0.23125]], 12, axis=1)
            )
        else:
            np.testing.assert_array_equal(filtered[:, joints], raw[:, joints])
        if binary_gripper:
            np.testing.assert_allclose(filtered[:, [6, 13]], 0.1)
        elif ema_enabled:
            np.testing.assert_allclose(filtered[:, [6, 13]], [[0.3, 0.9], [0.9, 0.8], [0.8, 0.5]])
            assert events[0]["gripper_ema_jump"] == [False, True]
        else:
            np.testing.assert_array_equal(filtered[:, [6, 13]], raw[:, [6, 13]])
        np.testing.assert_array_equal(raw[:, 0], [1, -1, 1])

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "values",
    [(0, 0.5, 0.4), (1.1, 0.5, 0.4), (0.25, float("nan"), 0.4), (0.25, 0.5, -1), (0.25, 0.5, float("inf"))],
)
def test_invalid_ema_parameters(values):
    with pytest.raises(ValueError):
        _module.validate_ema_config(*values)


@pytest.mark.parametrize("failure", ["exhausted", "expired", "invalid", "server", "timeout", "initial"])
def test_rtc_failures_hold_measured_pose_and_cancel_pending_request(failure):
    async def scenario():
        now = [0.0]
        sent = []
        calls = [0]
        active = [0]

        class Environment:
            def get_observation(self):
                return {"state": np.full(14, 0.2)}

            def apply_action(self, action, *, pace):
                sent.append(np.asarray(action).copy())
                return action

        class Policy:
            metadata = {"policy": {"rtc_enabled": True, "chunk_size": 32}}

            async def infer(self, payload):
                calls[0] += 1
                active[0] += 1
                try:
                    if failure == "initial":
                        raise RuntimeError("initial inference failed")
                    if calls[0] > 1:
                        if failure == "server":
                            raise RuntimeError("server inference failed")
                        if failure in ("exhausted", "timeout"):
                            await asyncio.Event().wait()
                        if failure == "expired":
                            now[0] += 2.0
                        if failure == "invalid":
                            return {"action": np.zeros((32, 14)), "model_action": np.zeros((31, 14))}
                    return {"action": np.ones((32, 14)), "model_action": np.full((32, 14), 0.3)}
                finally:
                    active[0] -= 1

        async def sleep(seconds):
            now[0] += seconds
            await asyncio.sleep(0.01 if failure == "timeout" else 0)

        error = {
            "exhausted": "queue exhausted",
            "expired": "expired",
            "invalid": "model-space",
            "server": "server inference",
            "initial": "initial inference",
        }.get(failure)
        with pytest.raises((RuntimeError, ValueError, TimeoutError), match=error):
            await execute_chunks(
                Policy(),
                Environment(),
                payload_fn=dict,
                extract_fn=lambda result, horizon: result["action"],
                rate_limit_fn=lambda action, previous, limit: action,
                gripper_fn=None,
                state_key="state",
                action_horizon=32,
                control_hz=30,
                max_steps=60,
                timeout=0.001 if failure == "timeout" else 5,
                max_delta=0,
                log=None,
                clock=lambda: now[0],
                sleep=sleep,
            )
        assert active[0] == 0
        np.testing.assert_allclose(sent[-1], 0.2)
        if failure == "initial":
            assert len(sent) == 1

    asyncio.run(scenario())
