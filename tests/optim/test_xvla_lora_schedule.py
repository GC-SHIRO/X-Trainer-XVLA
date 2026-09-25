import torch
import copy
from types import SimpleNamespace

import pytest

from lerobot.optim.optimizers import XVLALoRAAdamWConfig
from lerobot.optim.schedulers import XVLALoRAStagedSchedulerConfig


def test_xvla_lora_optimizer_builds_four_named_groups():
    model = torch.nn.Module()
    model.vlm_lora_A = torch.nn.Parameter(torch.ones(2))
    model.action_lora_A = torch.nn.Parameter(torch.ones(2))
    model.soft_prompt_hub = torch.nn.Parameter(torch.ones(2))
    model.action_encoder = torch.nn.Parameter(torch.ones(2))
    optimizer = XVLALoRAAdamWConfig(lr=1.0).build(dict(model.named_parameters()))
    assert [group["name"] for group in optimizer.param_groups] == [
        "vlm_lora", "action_lora", "soft_prompts", "action_modules"
    ]


def test_xvla_lora_staged_scheduler_holds_only_lora_groups():
    params = [torch.nn.Parameter(torch.ones(1)) for _ in range(4)]
    optimizer = torch.optim.AdamW(
        [
            {"params": [params[0]], "lr": 1.0, "name": "vlm_lora"},
            {"params": [params[1]], "lr": 1.0, "name": "action_lora"},
            {"params": [params[2]], "lr": 1.0, "name": "soft_prompts"},
            {"params": [params[3]], "lr": 1.0, "name": "action_modules"},
        ]
    )
    scheduler = XVLALoRAStagedSchedulerConfig(lora_start_step=2).build(optimizer, 10)
    assert scheduler.get_last_lr() == [0.0, 0.0, 1.0, 1.0]
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 0.0
    assert optimizer.param_groups[1]["lr"] == 0.0
    assert optimizer.param_groups[2]["lr"] == 1.0
    assert optimizer.param_groups[3]["lr"] == 1.0
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 1.0


def build_schedule(start=2, cosine=False):
    optimizer = XVLALoRAAdamWConfig(lr=1.0).build({
        "model.vlm.lora_A.weight": torch.nn.Parameter(torch.ones(1)),
        "model.transformer.lora_A.weight": torch.nn.Parameter(torch.ones(1)),
        "model.transformer.soft_prompt_hub.weight": torch.nn.Parameter(torch.ones(1)),
        "model.transformer.action_decoder.weight": torch.nn.Parameter(torch.ones(1)),
    })
    scheduler = XVLALoRAStagedSchedulerConfig(
        lora_start_step=start, use_cosine_decay=cosine
    ).build(optimizer, 10)
    return optimizer, scheduler


def test_zero_adaptation_steps():
    _, scheduler = build_schedule(start=0)
    assert scheduler.get_last_lr() == [1.0] * 4


@pytest.mark.parametrize("resume_step", [1, 2, 3])
def test_resume_matches_uninterrupted_schedule(resume_step):
    optimizer, scheduler = build_schedule(cosine=True)
    for _ in range(resume_step):
        optimizer.step()
        scheduler.step()
    opt_state = copy.deepcopy(optimizer.state_dict())
    sched_state = copy.deepcopy(scheduler.state_dict())
    resumed_opt, resumed_sched = build_schedule(cosine=True)
    resumed_opt.load_state_dict(opt_state)
    resumed_sched.load_state_dict(sched_state)
    for _ in range(12):
        assert resumed_sched.get_last_lr() == pytest.approx(scheduler.get_last_lr())
        assert [g["lr"] for g in resumed_opt.param_groups] == pytest.approx(scheduler.get_last_lr())
        optimizer.step()
        scheduler.step()
        resumed_opt.step()
        resumed_sched.step()


def test_factory_passes_named_parameters_without_preset():
    from lerobot.optim.factory import make_optimizer_and_scheduler

    model = torch.nn.Module()
    model.vlm_lora_A = torch.nn.Parameter(torch.ones(1))
    cfg = SimpleNamespace(
        optimizer=XVLALoRAAdamWConfig(), scheduler=XVLALoRAStagedSchedulerConfig(),
        use_policy_training_preset=False, steps=10,
    )
    optimizer, scheduler = make_optimizer_and_scheduler(cfg, model)
    assert optimizer.param_groups[0]["name"] == "vlm_lora"
    assert scheduler.get_last_lr() == [0.0]


def test_factory_keeps_standard_optimizer_parameter_iterable():
    from lerobot.optim.factory import make_optimizer_and_scheduler
    from lerobot.optim.optimizers import AdamWConfig

    model = torch.nn.Linear(2, 1)
    cfg = SimpleNamespace(
        optimizer=AdamWConfig(), scheduler=None, use_policy_training_preset=False, steps=10,
    )
    optimizer, scheduler = make_optimizer_and_scheduler(cfg, model)
    assert scheduler is None
    assert optimizer.param_groups[0]["params"] == list(model.parameters())
