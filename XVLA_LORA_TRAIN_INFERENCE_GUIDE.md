# X-VLA LoRA 训练与推理操作指南

更新日期：2026-09-14

## 1. 当前可用性与前置条件

本文命令按当前仓库的 CLI 参数编写，适用于已安装本工程和依赖的 Linux / Bash 训练环境。从仓库根目录执行，不要直接在 PowerShell 中使用 Bash 的换行符和变量语法。

**此前发现的两个构建阻塞已修复，但尚未完成实际训练验收：**

1. 工厂仅对 `XVLALoRAAdamWConfig` 传入命名参数字典，其他 optimizer 保持原路径。
2. 分阶段 `LambdaLR` 现按组接收返回标量的独立回调，不再将列表用作学习率倍率。
3. 配置测试已同步 `use_policy_training_preset=False`，并新增阶段边界、零适配步数及恢复轨迹测试。当前本机 Python 缺少 pytest，测试尚未执行通过。

不要通过简单打开 `use_policy_training_preset` 绕过问题，这可能让训练配置重新使用原全量 optimizer/scheduler。

真实 PEFT 训练、恢复、合并输出以及真机部署尚未完成端到端验证。此前通过的标准库测试仅验证文件路径和基座清单等辅助逻辑。修复上述问题后，仍需检查 warmup 时间基准、分布式步数和恢复学习率轨迹。

本指南只使用以下流程：

```text
本地官方原始基座 + 机器人数据
          -> LoRA 训练
          -> 原始基座 + adapter 合并
          -> 独立完整 checkpoint
          -> 现有推理服务
```

已有全量微调模型不是输入依赖，不需要在 LoRA 配置中填写其路径。

## 2. 准备路径和配置

以下路径全部是示例，替换为训练机器上的真实位置：

```bash
BASE=/data/models/original-xvla
DATASET=/data/datasets/xtrainer-v21
RUN=/data/outputs/xvla-lora-run01
ADAPTER="$RUN/checkpoints/last/pretrained_model"
MERGED=/data/exports/xvla-lora-merged-run01
VALIDATION=/data/validation/observation.safetensors
```

要求：

- `BASE` 是官方原始基座，当前加载器需要 `config.json` 和单个 `model.safetensors`。
- `DATASET` 是通过现有校验器检查的 LeRobot v2.1 数据集。
- `RUN` 使用新的训练目录，不覆盖已有实验。
- `MERGED` 必须不存在，且不在基座或 adapter 目录内部。
- 相对路径按进程工作目录解析，推荐绝对路径。

编辑 `configs/xtrainer/train_xvla_lora.yaml` 中的占位值：

```yaml
dataset:
  root: /data/datasets/xtrainer-v21

policy:
  path: /data/models/original-xvla
  tokenizer_name: /data/models/xvla-tokenizer

output_dir: /data/outputs/xvla-lora-run01
```

这是要替换的字段片段，不是完整 YAML；保留原文件其余内容。Tokenizer 路径必须实际可用，不假定基座一定有 `tokenizer/` 子目录。

确认现有数据契约适用于此次数据：

- 三相机名称、顺序和图像变换。
- 14 维状态/动作、角度单位、夹爪方向和范围。
- `action_mode=auto`、`domain_id=19`。
- `chunk_size=32` 和执行动作长度。

检查环境是否能找到训练入口和依赖：

```bash
command -v python
command -v lerobot-train
python -c "import torch, peft, transformers, accelerate, lerobot; print('imports OK')"
```

这些命令不证明 GPU、模型格式或训练链路兼容。

## 3. 数据校验

```bash
python scripts/xtrainer/validate_dataset_v21.py --root "$DATASET"
```

训练脚本默认也会执行校验。不建议首次训练使用 `--skip-validation`。

## 4. 首次 LoRA 训练

**先在训练环境运行针对性测试并进行短训练验收。** 使用独立 LoRA 启动脚本，不使用原 `train_xvla.sh`，后者可能自动覆盖模型路径。

### 4.1 短训练

```bash
bash scripts/xtrainer/train_xvla_lora.sh \
  --config configs/xtrainer/train_xvla_lora.yaml \
  --base-model "$BASE" \
  --dataset-root "$DATASET" \
  --output-dir /data/outputs/xvla-lora-smoke01 \
  --device cuda \
  --batch-size 1 \
  --steps 20
```

20 步只能检查初始化及短训练，不会跨过默认 1000 步适配阶段。验证阶段切换时，使用独立测试配置缩短 `scheduler.lora_start_step`，并设置适当保存频率。

### 4.2 正式训练

```bash
bash scripts/xtrainer/train_xvla_lora.sh \
  --config configs/xtrainer/train_xvla_lora.yaml \
  --base-model "$BASE" \
  --dataset-root "$DATASET" \
  --output-dir "$RUN" \
  --device cuda \
  --batch-size 8 \
  --steps 30000
```

Batch size 和步数只是当前配置示例，需要按显存、数据量及评估结果调整。脚本传入的基座、数据和输出路径覆盖 YAML 对应值；tokenizer 仍来自 YAML。

首次训练应检查实际可训练参数及四组学习率，不以 loss 能打印为成功标准。

## 5. 断点续训

保留完整 checkpoint，包括相邻的 `training_state`，不能只复制 adapter 权重。

```text
checkpoints/<step>/
  pretrained_model/
    adapter_config.json
    adapter_model.safetensors
    config.json
    train_config.json
    xvla_base_manifest.json
    ...处理器与 tokenizer 资源
  training_state/
    ...优化器、调度器、随机状态和步数
```

```bash
bash scripts/xtrainer/train_xvla_lora.sh \
  --base-model "$BASE" \
  --dataset-root "$DATASET" \
  --resume-checkpoint "$ADAPTER"
```

当前脚本的实际行为：

- 恢复时使用 checkpoint 保存的配置。
- `--base-model` 仍被脚本要求存在，但不会作为 `policy.path` 转发。
- 实际恢复基座路径来自 adapter 配置，不是该命令行参数。
- 基座搬迁后，仅改变 `--base-model` 不会修复 adapter 的旧来源路径。
- 不要随意修改基座、batch size、总步数或调度配置；恢复一致性还需要实测。

遇到基座身份不匹配时，应找回正确基座，不要删除清单绕过检查。

## 6. 准备合并验证观测

合并工具要求 `--validation-batch`：一份**已经经过匹配处理器处理的观测 Tensor 字典**，保存为 safetensors。

它应包含：

- 当前模型使用的图像键及预处理后图像张量。
- 状态张量。
- 语言 token。
- 正确的 `domain_id`，或模型配置指定的域字段。

不能直接提供数据集路径、原始图片或任意 pickle 文件。建议使用留出的真实观测。

在已有数据处理代码中获得 `batch` 后，可用以下代码保存：

```python
from safetensors.torch import save_file

save_file(
    {key: value.detach().cpu().contiguous().clone() for key, value in batch.items()},
    "/data/validation/observation.safetensors",
)
```

此片段要求 `batch` 已经是符合模型输入契约的 Tensor 字典；不负责读取数据集或运行处理器。当前仓库尚未提供专门的一键验证观测导出 CLI，不要把该片段当作独立可执行程序。

## 7. 合并导出

```bash
python scripts/xtrainer/merge_xvla_lora.py \
  --base-model "$BASE" \
  --adapter "$ADAPTER" \
  --output-dir "$MERGED" \
  --validation-batch "$VALIDATION" \
  --device cuda
```

默认保留模型配置精度。可显式添加 `--dtype float32` 或 `--dtype bfloat16`；精度变化后需要重新评估。

工具会：

1. 校验基座身份和清单内 adapter 文件哈希。
2. 加载 adapter，包括额外完整训练模块。
3. 固定随机种子，记录动作输出。
4. 合并并移除 PEFT 包装，核对额外模块权重未丢失。
5. 对比合并前后输出。
6. 导出权重、配置、处理器和 tokenizer。
7. 重新加载完整模型，再次比较输出并写入 `merge_report.json`。

输出比较默认 `--atol 0.001 --rtol 0.01`。这是数值比较容限，不是机器人动作安全阈值；不应只为使测试通过而放宽。

注意：

- 验证可能同时占用合并模型和重载模型的内存/显存。
- 出现 `EXPORT_FAILED.txt` 的目录禁止部署。
- 不覆盖失败目录重试，使用另一个新目录。
- 保留原始基座和 adapter，合并目录不是 LoRA 断点续训产物。

## 8. 启动推理服务

这一命令只启动模型服务，不启动真机客户端：

```bash
python scripts/xtrainer/serve_policy.py \
  --config configs/xtrainer/deploy.yaml \
  --checkpoint "$MERGED" \
  --device cuda \
  --host 127.0.0.1 \
  --port 8000
```

若机器人客户端在另一台机器上，改为受信任局域网可访问的服务器地址；需要监听所有网卡时显式指定 `--host 0.0.0.0`，不要将服务直接暴露到公网。

也可以修改 `deploy.yaml` 的 `policy.checkpoint`，省略 `--checkpoint`。CLI 优先于 YAML。不需要填写基座或 adapter 路径，也不增加 PEFT 开关。

默认执行启动 warmup。可开启动作日志：

```bash
python scripts/xtrainer/serve_policy.py \
  --config configs/xtrainer/deploy.yaml \
  --checkpoint "$MERGED" \
  --device cuda \
  --host 127.0.0.1 \
  --port 8000 \
  --log-actions \
  --action-log-path /data/logs/xvla-lora-actions.jsonl
```

在另一个终端检查服务：

```bash
curl --fail http://127.0.0.1:8000/healthz
```

健康检查不是任务成功率或机器人安全验证。`/ws` 使用现有 WebSocket/msgpack 协议，不是 JSON HTTP 推理接口。

## 9. 真机客户端

**下面命令会连接硬件并可能触发机器人复位和运动。必须先检查工作空间、急停、初始姿态、相机及动作单位，再由现场操作人员执行。**

以下为参数模板，不是已经确认适合当前机器的限幅数值：

```bash
SERVER_IP=192.168.1.100
# 必须按现场验证结果填写有限的正数，不能直接沿用无限限制。
MAX_JOINT_DELTA=现场确认的关节增量上限
MAX_GRIPPER_DELTA=现场确认的夹爪增量上限

python scripts/xtrainer/run_real.py \
  --host "$SERVER_IP" \
  --port 8000 \
  --task "pick up the object" \
  --domain-id 19 \
  --action-horizon 32 \
  --control-hz 30 \
  --max-steps 100 \
  --max-joint-delta "$MAX_JOINT_DELTA" \
  --max-gripper-delta "$MAX_GRIPPER_DELTA"
```

沿用已经验证过的硬件参数和现场操作流程。需要其他相机、机械臂或夹爪参数时查看：

```bash
python scripts/xtrainer/run_real.py --help
```

`run_real.py` 不是读取 `deploy.yaml` 的同一套 CLI；不要假定服务端 YAML 中的 safety 配置会自动应用到真机客户端。示例中的频率和动作长度也不是“低速安全”保证。

切换模型时先停止机器人执行，再停止服务、修改完整 checkpoint 路径并重新启动。不进行运行中热切换。

## 10. 常见问题

| 问题 | 处理 |
| --- | --- |
| LoRA optimizer 要求 named parameters | 确认正在使用已修复的工厂代码，而不是旧安装 |
| Scheduler 倍率类型错误 | 确认使用按组返回标量回调的修复版本，再测试恢复 |
| 路径仍为 `/path/to/...` | 填写 YAML 中实际路径，尤其 tokenizer |
| 恢复时仍读取旧基座目录 | `--base-model` 不覆盖 resume 来源；检查 adapter 配置 |
| 基座身份不匹配 | 使用训练时相同内容的官方基座 |
| 合并缺少验证 batch | 先通过匹配处理器导出真实观测 Tensor 字典 |
| 部署拒绝 adapter | 将 checkpoint 指向合并后的完整模型 |
| 合并目录含失败标记 | 排查验证错误，重新导出到新目录 |
| 域编号不一致 | 对齐训练配置、模型、服务及客户端，不强行绕过 |
| 合并模型缺少处理器 | 修复导出，不使用其他模型的预处理文件替代 |

配套设计文档：`XVLA_LORA_IMPLEMENTATION_PLAN.md`。
