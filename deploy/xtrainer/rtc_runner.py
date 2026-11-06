"""Sequential or guided RTC execution, with one outstanding prediction."""

import asyncio
import logging

import numpy as np


def validate_ema_config(joint_alpha, gripper_alpha, gripper_jump_threshold):
    """Validate the per-step joint and gripper filtering parameters."""
    for name, value in (("joint_ema_alpha", joint_alpha), ("gripper_ema_alpha", gripper_alpha)):
        if not np.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{name} must be finite and within (0, 1]")
    if not np.isfinite(gripper_jump_threshold) or gripper_jump_threshold < 0:
        raise ValueError("gripper_ema_jump_threshold must be finite and nonnegative")


async def execute_chunks(
    policy,
    environment,
    *,
    payload_fn,
    extract_fn,
    rate_limit_fn,
    gripper_fn,
    state_key,
    action_horizon,
    control_hz,
    max_steps,
    timeout,
    max_delta,
    log,
    clock,
    sleep,
    ema_enabled=True,
    joint_ema_alpha=0.25,
    gripper_ema_alpha=0.5,
    gripper_ema_jump_threshold=0.4,
):
    """Overlap one prediction with playback and filter only actions being sent."""
    validate_ema_config(joint_ema_alpha, gripper_ema_alpha, gripper_ema_jump_threshold)
    metadata = (getattr(policy, "metadata", None) or {}).get("policy", {})
    rtc = bool(metadata.get("rtc_enabled"))
    horizon = int(metadata["chunk_size"]) if rtc else action_horizon
    logger = logging.getLogger(__name__)
    logger.info(
        "Execution mode=%s prediction_steps=%d control_hz=%.2f",
        "guided RTC" if rtc else "sequential",
        horizon,
        control_hz,
    )
    logger.info(
        "EMA enabled=%s joint_alpha=%.3f gripper_alpha=%.3f gripper_jump_threshold=%.3f binary_gripper=%s",
        ema_enabled,
        joint_ema_alpha,
        gripper_ema_alpha,
        gripper_ema_jump_threshold,
        gripper_fn is not None,
    )
    joint_indices = np.r_[0:6, 7:13]
    gripper_indices = np.array([6, 13])
    period = 1 / control_hz
    pending = None
    last_sent = None
    delay = 1
    step = 0
    previous_send_started = None
    initial_state = None
    source_step = 0
    request_step = 0
    request_started = 0.0

    async def request(observation, prefix=None, observation_step=0):
        payload = payload_fn(observation)
        if rtc:
            payload.update(prev_chunk_left_over=prefix, inference_delay=delay if prefix is not None else 0)
        if log is not None:
            log.write(
                "inference_request",
                observation_timestep=observation_step,
                state=np.asarray(payload["state"]).tolist(),
                task=payload.get("task"),
                inference_delay=payload.get("inference_delay", 0),
            )
        response = await asyncio.wait_for(policy.infer(payload), timeout)
        if log is not None:
            log.write(
                "inference_response",
                observation_timestep=observation_step,
                retained_actions=extract_fn(response, horizon).tolist(),
            )
        return response

    def unpack(response):
        actions = extract_fn(response, horizon)
        model = None
        if rtc:
            model = np.asarray(response.get("model_action"), dtype=np.float32)
            if actions.shape != (horizon, 14) or model.shape != actions.shape or not np.isfinite(model).all():
                raise ValueError("Invalid RTC model-space actions")
        return actions, model

    try:
        started = clock()
        observation = environment.get_observation()
        initial_state = np.asarray(observation[state_key], dtype=np.float64).copy()
        actions, model = unpack(await request(observation))
        delay = max(1, int(np.ceil((clock() - started) / period)) + 2)
        index = 0
        deadline = clock()
        while step < max_steps:
            if pending is not None and pending.done():
                response = pending.result()
                pending = None
                incoming, incoming_model = unpack(response)
                consumed = step - request_step
                elapsed = max(consumed, int(np.ceil((clock() - request_started) / period)))
                delay = max(1, elapsed + 2)
                logger.info(
                    "RTC result: elapsed_steps=%d executed_steps=%d alignment_steps=%d next_delay=%d",
                    elapsed,
                    consumed,
                    consumed,
                    delay,
                )
                if elapsed >= len(incoming):
                    raise RuntimeError("RTC prediction expired; stopping")
                joints = np.r_[0:6, 7:13]
                seam_delta = (
                    None
                    if last_sent is None
                    else float(np.max(np.abs(incoming[consumed, joints] - last_sent[joints])))
                )
                revision_delta = (
                    float(np.max(np.abs(incoming[consumed, joints] - actions[index, joints])))
                    if index < len(actions)
                    else None
                )
                logger.info(
                    "RTC seam: target_step_delta_rad=%s revised_next_target_delta_rad=%s",
                    seam_delta,
                    revision_delta,
                )
                if log is not None:
                    log.write(
                        "rtc_switch",
                        executed_steps=consumed,
                        elapsed_steps=elapsed,
                        target_step_delta_rad=seam_delta,
                        revised_next_target_delta_rad=revision_delta,
                    )
                # This executor advances one action per successful apply_action,
                # including when a tick overruns. The prefix is indexed by those
                # actions, not wall-clock slots. Time-based slicing skips targets.
                actions, model, index = incoming[consumed:], incoming_model[consumed:], 0
                source_step = request_step
            remaining = len(actions) - index
            if remaining == 0:
                if rtc:
                    raise RuntimeError("RTC queue exhausted; stopping")
                observation = environment.get_observation()
                last_sent = np.asarray(environment.apply_action(observation[state_key], pace=False))
                if log is not None:
                    log.write("control_hold", control_timestep=step, applied_action=last_sent.tolist())
                actions, model = unpack(await request(observation, observation_step=step))
                source_step = step
                index = 0
                deadline = clock()
                remaining = len(actions)
            if rtc and pending is None and step + remaining < max_steps and remaining <= max(15, delay + 5):
                if remaining <= delay:
                    raise RuntimeError("RTC prefix cannot cover observed latency")
                request_started = clock()
                observation = environment.get_observation()
                observation_ms = (clock() - request_started) * 1000
                logger.info("RTC observation acquisition %.1f ms", observation_ms)
                request_step = step
                pending = asyncio.create_task(request(observation, model[index:].copy(), request_step))
            action = np.array(actions[index], dtype=np.float64, copy=True)
            previous = last_sent if last_sent is not None else initial_state
            gripper_jump = np.zeros(2, dtype=bool)
            if ema_enabled:
                # Only executed targets advance the filter. The model-space RTC
                # prefix stays raw, and hardware-limited output anchors the next step.
                action[joint_indices] = previous[joint_indices] + joint_ema_alpha * (
                    action[joint_indices] - previous[joint_indices]
                )
                if gripper_fn is None:
                    delta = action[gripper_indices] - previous[gripper_indices]
                    gripper_jump = np.abs(delta) > gripper_ema_jump_threshold
                    action[gripper_indices] = np.where(
                        gripper_jump,
                        action[gripper_indices],
                        previous[gripper_indices] + gripper_ema_alpha * delta,
                    )
            if gripper_fn is not None:
                action = gripper_fn(action, previous)
            filtered_action = action.copy()
            action = rate_limit_fn(action, last_sent, max_delta)
            send_started = clock()
            interval_ms = (
                None if previous_send_started is None else (send_started - previous_send_started) * 1000
            )
            previous_send_started = send_started
            joint_delta = (
                None
                if last_sent is None
                else float(np.max(np.abs(action[joint_indices] - last_sent[joint_indices])))
            )
            last_sent = np.asarray(environment.apply_action(action, pace=False), dtype=np.float64)
            apply_ms = (clock() - send_started) * 1000
            if interval_ms is not None and interval_ms > period * 1500:
                logger.warning(
                    "Control gap: interval=%.1f ms target=%.1f ms apply=%.1f ms joint_target_delta=%.5f rad",
                    interval_ms,
                    period * 1000,
                    apply_ms,
                    joint_delta,
                )
            if log is not None:
                log.write(
                    "control_step",
                    control_timestep=step,
                    rtc=rtc,
                    source_observation_timestep=source_step,
                    used_fallback=False,
                    queued_action=actions[index].tolist(),
                    applied_action=last_sent.tolist(),
                    filtered_action=filtered_action.tolist(),
                    ema_enabled=ema_enabled,
                    rate_limited_action=action.tolist(),
                    gripper_ema_jump=gripper_jump.tolist(),
                    send_started_monotonic=send_started,
                    send_interval_ms=interval_ms,
                    apply_action_ms=apply_ms,
                    joint_target_delta_rad=joint_delta,
                    action_stages_ms=dict(getattr(environment, "last_action_timing_ms", {})),
                )
            index += 1
            step += 1
            deadline = max(deadline + period, clock())
            await sleep(max(0, deadline - clock()))
    finally:
        if pending is not None:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        if initial_state is not None:
            observation = environment.get_observation()
            applied = environment.apply_action(observation[state_key], pace=False)
            if log is not None:
                log.write(
                    "control_stop_hold", control_timestep=step, applied_action=np.asarray(applied).tolist()
                )
