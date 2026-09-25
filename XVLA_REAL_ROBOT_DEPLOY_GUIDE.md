# X-VLA 训练后真机部署指南

更新日期：2026-09-17

适用：本仓库的 X-Trainer 双臂部署。以下命令在已配置本工程依赖的 Linux/Bash 环境中，从仓库根目录运行。路径、IP、相机序列号和串口需要按实际设备填写。

本文只提供操作说明，不表示你的 checkpoint 已经完成真机验收，也不保证任务一定成功。

## 1. 训练命令没有 language，是不是没有语言输入？

**不是。当前训练启动命令没有单独的 `--language` 参数，语言指令来自数据集的 `task`。**

本仓库 v2.1 数据集的流程是：

```text
meta/tasks.jsonl 中的任务文本
          ↑
每帧 task_index 对应任务
          ↓
数据加载器返回 sample["task"]
          ↓
X-VLA 处理器进行 tokenizer 编码
          ↓
模型读取语言 token、图像和状态
```

相关实现：

- `src/lerobot/datasets/v21/dataset.py`
- `src/lerobot/policies/xvla/processor_xvla.py`
- `src/lerobot/processor/tokenizer_processor.py`

因此，配置中的 `tokenizer_name` 是 tokenizer 来源，不是任务指令。`observation_keys.task: task` 是字段名，也不是要执行的句子。

在训练机器查看实际使用的数据集：

```bash
DATASET=/data/datasets/你的训练数据
head -n 20 "$DATASET/meta/tasks.jsonl"
python scripts/xtrainer/validate_dataset_v21.py --root "$DATASET"
```

示例任务记录：

```json
{"task_index": 0, "task": "pick up the red block and put it in the tray"}
```

还应确认每条轨迹的 `task_index` 对应正确文本。校验器可检查引用关系和非空文本，但不能判断文字是否真实描述了示范动作。

**推理时通过 `--task` 提供任务文本。** 首轮验收建议使用训练数据中的原始指令，不要随意更换语言、物体名称或任务内容。数据中如果只有笼统文本或错误标注，仅在部署时换成更详细的指令，不能证明模型已经学会对应任务。

我没有读取你的实际训练数据和训练日志，因此这里说明的是代码中的语言输入链路，不是确认本次训练使用了哪一句文本。

## 2. 先确认训练产物类型

### 2.1 全量微调模型

如果使用 `configs/xtrainer/train_xvla.yaml` 训练，通常直接选择：

```bash
MODEL=/data/outputs/你的全量实验/checkpoints/last/pretrained_model
```

也可以选具体步数的 checkpoint；`last` 不一定是任务效果最好的模型。

部署应使用完整的 `pretrained_model` 目录，不是单独的权重文件：

```text
pretrained_model/
  config.json
  model.safetensors
  policy_preprocessor.json
  policy_postprocessor.json
  ...处理器状态与 tokenizer 文件
```

复制整个目录及其引用的本地资源。不要用其他实验的处理器或统计量替换。

### 2.2 LoRA 模型

如果目录主要包含 `adapter_model.safetensors` 和 `adapter_config.json`，这是 adapter，不能直接交给当前部署入口。

必须先合并：

```bash
python scripts/xtrainer/merge_xvla_lora.py \
  --base-model /data/models/训练时使用的官方原始基座 \
  --adapter /data/outputs/你的LoRA实验/checkpoints/last/pretrained_model \
  --output-dir /data/exports/你的LoRA完整模型 \
  --validation-batch /data/validation/observation.safetensors \
  --device cuda
```

`validation-batch` 必须是经过匹配处理器处理的真实观测 Tensor 字典，不是原始图片。准备方法见 `XVLA_LORA_TRAIN_INFERENCE_GUIDE.md`。

成功后：

```bash
MODEL=/data/exports/你的LoRA完整模型
```

确认 `merge_report.json` 验证成功且没有 `EXPORT_FAILED.txt`。输出目录必须新建，基座必须与训练时相同。全量模型不需要执行这个合并步骤。

## 3. 部署结构

```text
推理服务器：GPU + 完整模型
    serve_policy.py
           ↑ 状态、三路图像、task
           ↓ 动作 chunk
机器人控制机：
    run_real.py
           ↓
    双臂、夹爪、三路相机
```

两端可以在同一台机器上。同机使用 `127.0.0.1`；跨机使用服务器的局域网 IP。

现有接口是 WebSocket/msgpack，不是向 HTTP endpoint POST JSON 的推理服务。

## 4. 启动前核对

- 模型 `action_mode` 必须与现有 X-Trainer 入口匹配，当前要求 `auto`。
- 实际动作维度为 14：左臂 6 关节、左夹爪、右臂 6 关节、右夹爪。
- 状态和动作的关节顺序、单位、绝对值/增量语义必须与训练一致。
- 夹爪开合方向及范围必须一致。
- 三路相机视角、名称、图像变换及尺寸与训练对应。
- `domain_id` 与 checkpoint 一致，当前配置示例为 19，不是所有模型都固定为 19。
- 执行动作长度不超过模型 chunk 长度。
- 使用匹配 checkpoint 的 tokenizer、预处理和后处理。
- 核对机械臂 IP、夹爪串口/ID、相机序列号和实际接线。

**特别注意复位动作：** `run_real.py` 启动后会启用机械臂，并可能执行服务端提供的复位姿态。服务端默认姿态在 `deploy/xtrainer/real/constants.py` 的 `XTRAINER_RESET_POSE`。必须先审查该姿态及到达路径对当前工具、障碍物和工作空间是否适用。

配置中的无限关节范围/增量不构成安全保护。现场应明确限幅、关节边界、碰撞风险及硬件急停，不要仅依赖软件异常处理。

## 5. 启动推理服务器，不启动机器人

在模型服务器运行：

```bash
python scripts/xtrainer/serve_policy.py \
  --config configs/xtrainer/deploy.yaml \
  --checkpoint "$MODEL" \
  --device cuda \
  --host 127.0.0.1 \
  --port 8000 \
  --domain-id 19 \
  --actions-per-chunk 32
```

跨机连接时，将 `--host` 改为服务器局域网地址，或在受信任网络内使用 `0.0.0.0`。不要把无认证的控制服务直接暴露到公网。

也可以修改 `configs/xtrainer/deploy.yaml` 的 `policy.checkpoint`，省略 `--checkpoint`；命令行覆盖 YAML。原配置通常也包含 `domain_id: 19`，必须核对而不是盲目沿用。

默认会进行 warmup 推理，不会连接机械臂。第一次部署不建议添加 `--no-warmup` 跳过模型输入检查。

在另一终端：

```bash
curl --fail http://127.0.0.1:8000/healthz
```

跨机把地址换成服务器实际 IP。健康检查通过只说明服务响应，不意味着实际场景动作正确。

可选服务端动作日志：

```bash
python scripts/xtrainer/serve_policy.py \
  --config configs/xtrainer/deploy.yaml \
  --checkpoint "$MODEL" \
  --device cuda \
  --host 127.0.0.1 \
  --port 8000 \
  --log-actions \
  --action-log-path /data/logs/xvla-actions.jsonl
```

当前服务端日志还记录输入图像数组，体积可能很大；建议仅短时诊断并预留磁盘空间。

## 6. 真机运行前检查参数

在机器人控制机：

```bash
python scripts/xtrainer/run_real.py --help
```

当前默认硬件值仅供对照，不能视为现场实际值：

| 项目 | 当前默认 |
| --- | --- |
| 左臂 IP | `192.168.5.1` |
| 右臂 IP | `192.168.5.2` |
| 左夹爪 | `/dev/ttyUSB1`，ID 21 |
| 右夹爪 | `/dev/ttyUSB0`，ID 22 |
| 图像尺寸 | 640 × 480 |
| 相机帧率 | 30 |

三路相机通过 `--camera-top-serial`、`--camera-left-wrist-serial`、`--camera-right-wrist-serial` 指定。

`run_real.py` 不会自动读取服务端 `deploy.yaml` 的所有硬件和 safety 参数。需要在机器人客户端显式设置相应选项。

没有 `--execute` 时，当前客户端会直接拒绝运行，**不是无动作采集或 dry-run 模式**。

## 7. 显式授权真机运动

以下是现场填写的模板；故意不提供未经验证的统一安全限幅值。

先准备变量，任务文本替换为训练数据中的实际指令：

```bash
SERVER_IP=192.168.1.100
TASK='pick up the red block and put it in the tray'

LEFT_ARM_IP=192.168.5.1
RIGHT_ARM_IP=192.168.5.2
TOP_SERIAL=填写顶置相机序列号
LEFT_SERIAL=填写左腕相机序列号
RIGHT_SERIAL=填写右腕相机序列号

MAX_JOINT_DELTA=填写现场验证过的有限正数
MAX_GRIPPER_DELTA=填写现场验证过的有限正数
MAX_CLIENT_DELTA=填写现场验证过的有限正数
```

确认人员远离运动空间、复位姿态安全、急停可用，并由现场操作人员执行：

```bash
python scripts/xtrainer/run_real.py \
  --host "$SERVER_IP" \
  --port 8000 \
  --task "$TASK" \
  --domain-id 19 \
  --action-horizon 32 \
  --control-hz 30 \
  --max-steps 100 \
  --left-robot-ip "$LEFT_ARM_IP" \
  --right-robot-ip "$RIGHT_ARM_IP" \
  --left-gripper-port /dev/ttyUSB1 \
  --right-gripper-port /dev/ttyUSB0 \
  --left-gripper-id 21 \
  --right-gripper-id 22 \
  --camera-top-serial "$TOP_SERIAL" \
  --camera-left-wrist-serial "$LEFT_SERIAL" \
  --camera-right-wrist-serial "$RIGHT_SERIAL" \
  --max-joint-delta "$MAX_JOINT_DELTA" \
  --max-gripper-delta "$MAX_GRIPPER_DELTA" \
  --max-delta-per-step "$MAX_CLIENT_DELTA" \
  --log-control \
  --control-log-path /data/logs/xvla-control.jsonl \
  --execute
```

警告：

- `--execute` 允许启用和移动机械臂，包括任务执行前的复位。
- 30 Hz、32 动作长度和 100 步只是接口示例，不保证低速或安全。按已经验证的控制约定配置，不能简单把降频当成安全限速。
- 小步数不排除启动时的复位运动。
- 本客户端是动作执行循环，没有自动任务成功判定器。模型输出动作不等于任务已经成功，需要现场按标准判断。
- 不要为了绕过启动错误关闭有限值检查或改掉域/动作契约。

## 8. 停止与切换模型

出现异常动作、相机错位、夹爪方向错误或通信异常时，优先按现场急停流程停止硬件。不要把终端 Ctrl+C 或网络断开视为硬件急停的等价替代。

正常切换模型：

1. 停止客户端并确认机械臂处于稳定状态。
2. 停止推理服务器。
3. 修改完整 checkpoint 路径并重新启动服务器。
4. 重新进行模型检查和现场确认，再授权客户端执行。

不使用运行中热切换，也不覆盖当前部署目录中的权重。

## 9. 常见问题

| 现象 | 优先检查 |
| --- | --- |
| 不知道推理 task 填什么 | 查看训练数据 `meta/tasks.jsonl` |
| 训练命令没有 language | 正常，语言来自数据样本的 task，不是启动参数 |
| 模型没有按新指令做事 | 检查训练是否包含相应任务和语言标注；不能仅靠换文本获得新技能 |
| `use_peft` 或 adapter 被拒绝 | LoRA 先合并，部署指向完整模型 |
| `domain_id` 不匹配 | 对齐模型、服务和客户端，不能绕过校验 |
| 动作维度/模式错误 | 核对 checkpoint 是否是适配 X-Trainer 的模型 |
| tokenizer/processor 找不到 | 复制完整 checkpoint 和相关资源，检查迁移后的路径 |
| healthz 正常但客户端连不上 | 检查监听地址、防火墙、IP 和端口 |
| 没加 execute 就退出 | 当前设计为禁止运动，不是 dry-run |
| 启动后先复位 | 现有启动行为，先审查服务端 reset pose |

建议先从训练分布内的单个任务、熟悉的物体位置和较短运行开始，记录成功率及异常类型，再逐步扩大测试范围。
