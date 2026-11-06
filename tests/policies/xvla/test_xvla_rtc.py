"""Exercise the real XVLA sampling loop without downloading a checkpoint."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.policies.xvla.action_hub import build_action_space
from lerobot.policies.xvla.modeling_xvla import XVLAModel


class TinyActionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.calls = 0

    def forward(self, action_with_noise, **kwargs):
        self.calls += 1
        return action_with_noise * self.scale + 0.2


def make_model(dtype="float32"):
    model = XVLAModel.__new__(XVLAModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(dtype=dtype, action_mode="auto")
    model.chunk_size = 8
    model.dim_action = 20
    model.action_space = build_action_space("auto", real_dim=14, max_dim=20)
    model.transformer = TinyActionHead()
    model.forward_vlm = lambda *args: {}
    return model


def inputs():
    return {
        "input_ids": torch.zeros(1, 2, dtype=torch.long),
        "image_input": torch.zeros(1, 3, 4, 4),
        "image_mask": torch.ones(1, 1),
        "domain_id": torch.tensor([19]),
        "proprio": torch.zeros(1, 14),
        "steps": 4,
    }


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_no_prefix_is_exactly_the_original_sampling_loop(dtype):
    model = make_model(dtype)
    args = inputs()
    # Independent copy of the original recurrence, including dummy channels.
    target_dtype = model._get_target_dtype()
    torch.manual_seed(123)
    noise = torch.randn(1, 8, 20, dtype=target_dtype)
    action = torch.zeros_like(noise)
    with torch.no_grad():
        for i in range(4, 0, -1):
            t = torch.full((1,), i / 4, dtype=target_dtype)
            x_t = noise * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
            action = model.transformer(action_with_noise=x_t)
    expected = action[..., :14]
    for processor in (None, RTCProcessor(RTCConfig())):
        torch.manual_seed(123)
        actual = model.generate_actions(**args, rtc_processor=processor)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_guidance_moves_prefix_toward_old_chunk_without_guiding_padding(dtype):
    model = make_model(dtype)
    args = inputs() | {"steps": 1}
    torch.manual_seed(123)
    original = model.generate_actions(**args)
    prefix = torch.full((1, 4, 14), 0.8)
    processor = RTCProcessor(RTCConfig(max_guidance_weight=0.5, execution_horizon=4, debug=True))
    torch.manual_seed(123)
    guided = model.generate_actions(
        **args, rtc_processor=processor, prev_chunk_left_over=prefix, inference_delay=2
    )
    assert model.transformer.calls == 2  # One transformer call per sampling step.
    assert torch.isfinite(guided).all()
    assert not guided.requires_grad
    assert torch.mean(abs(guided[:, :2].float() - 0.8)) < torch.mean(abs(original[:, :2].float() - 0.8))
    torch.testing.assert_close(guided[:, 4:], original[:, 4:], atol=0.004, rtol=0)
    torch.testing.assert_close(prefix, torch.full_like(prefix, 0.8))
    assert model.transformer.scale.grad is None
    assert processor.get_all_debug_steps()[0].correction.shape[-1] == 14


def test_guidance_covers_delay_larger_than_configured_horizon():
    model = make_model()
    processor = RTCProcessor(RTCConfig(execution_horizon=2, max_guidance_weight=0.5, debug=True))
    result = model.generate_actions(
        **inputs(), rtc_processor=processor, prev_chunk_left_over=torch.ones(1, 7, 14), inference_delay=5
    )
    assert torch.isfinite(result).all()
    assert model.transformer.calls == 4
    assert len(processor.get_all_debug_steps()) == 4
    for step in processor.get_all_debug_steps():
        torch.testing.assert_close(step.weights[:, :5], torch.ones(1, 5, 1))
