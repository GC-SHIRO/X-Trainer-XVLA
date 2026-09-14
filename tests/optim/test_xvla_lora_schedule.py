import torch

from lerobot.optim.optimizers import XVLALoRAAdamWConfig
from lerobot.optim.schedulers import XVLALoRAStagedSchedulerConfig


def test_xvla_lora_optimizer_builds_four_named_groups():
    model = torch.nn.Module()
    model.vlm_lora = torch.nn.Parameter(torch.ones(2))
    model.action_lora = torch.nn.Parameter(torch.ones(2))
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
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 0.0
    assert optimizer.param_groups[1]["lr"] == 0.0
    assert optimizer.param_groups[2]["lr"] == 1.0
    assert optimizer.param_groups[3]["lr"] == 1.0
    scheduler.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 1.0
