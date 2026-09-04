# X-Trainer XVLA：数据、训练与真机部署

版本：V1.0  
日期：2026-09-04  
适用代码：`GC-SHIRO/X-Trainer-XVLA` `main`

> 本 README 按 X-Trainer Pi0.5 手册的端到端标准整理。所有路径、设备序列号和任务文本应按实际环境替换；未在本机实测的结果不应视为验收结论。

本文描述 Dobot X-trainer 双臂平台的完整 XVLA 工作流。训练数据保持 LeRobot Dataset v2.1，模型使用
`lerobot/xvla-base`，策略服务和机器人控制程序通过 WebSocket + MessagePack 通信。

## 1. 系统结构

```text
collect_data
    │
    ▼
LeRobot Dataset v2.1 ──► lerobot-train ──► XVLA checkpoint
                                               │
                                               ▼
RealSense + Dobot + Feetech ◄── WebSocket ── XVLA policy server
```

训练机需要 NVIDIA GPU；机器人控制机负责三路 RealSense、两台 Dobot 和两个 Feetech 夹爪。两端可以是同一台
机器，也可以部署在可信局域网中的两台机器。

## 2. XVLA 与 X-trainer 契约

X-trainer 的状态与动作均为 14 维：

| 索引 | 内容 |
| --- | --- |
| 0–5 | 左臂关节 1–6，弧度 |
| 6 | 左夹爪，归一化到 `[0,1]` |
| 7–12 | 右臂关节 1–6，弧度 |
| 13 | 右夹爪，归一化到 `[0,1]` |

XVLA 基础模型使用 20 维动作头。配置采用 `action_mode: auto`：训练时 14 维标签在尾部补零到 20 维，只在真实
14 维上计算损失；推理后自动裁回 14 维。因此不要把配置改为 14 维动作头，否则无法严格加载基础 checkpoint。

本项目为 X-trainer 使用 `domain_id: 19`。该 ID 会选择独立的 domain-aware projection 和 soft prompt。训练、断点
续训和部署必须使用同一 ID；修改 ID 等同于切换机器人域。

三路视觉字段为：

- `observation.images.top`
- `observation.images.left_wrist`
- `observation.images.right_wrist`

XVLA 可以直接使用这些字段，不需要重命名为 `camera1/2/3`。processor 会完成 uint8/float 转换、ImageNet
归一化、tokenization、device 移动和 domain ID 注入。

## 3. 环境安装

支持 Ubuntu 24.04 x86_64、Python 3.12、PyTorch 2.8.0。默认 CUDA wheel 为 cu128，并要求驱动版本不低于
570.26：

```bash
bash tools/install_xtrainer_env.sh
conda activate xtrainer-xvla
```

使用官方软件源：

```bash
bash tools/install_xtrainer_env.sh --source official
```

只进行数据转换或 Mock 测试：

```bash
bash tools/install_xtrainer_env.sh --cpu-only
```

环境脚本安装 XVLA、训练、数据、WebSocket、Feetech 和 RealSense 依赖，但不会下载模型、连接硬件或修改串口权限。

## 4. 下载基础模型

```bash
bash tools/download_xvla_weights_hf.sh
```

或者使用 ModelScope：

```bash
bash tools/download_xvla_weights_modelscope.sh
```

两个脚本均将 `lerobot/xvla-base` 的完整远端仓库快照下载到 `models/xvla-base`，不筛选模型、配置、processor 或
仓库元数据文件；另外会把 processor 依赖的 BART tokenizer 下载到 `models/xvla-base/tokenizer`。训练配置会显式
使用这个本地 tokenizer，因此离线或禁止 Hub 出网时也无需再下载 `facebook/bart-large`。国内网络也可以为
Hugging Face 指定镜像：

```bash
bash tools/download_xvla_weights_hf.sh --endpoint https://hf-mirror.com
```

XVLA checkpoint 已包含 Florence-2 骨干，不需要额外下载 VLM。训练启动脚本检测到
`models/xvla-base/config.json` 后会优先使用本地权重，否则使用 YAML 中的 Hub ID。

基础模型尚未适配 X-trainer 的 14 维 action feature，不能直接用于真机；需要先完成至少一次 X-trainer 微调并部署
生成的 checkpoint。

## 5. 原始数据格式

原始数据按 episode 保存：

```text
collect_data/
└── <episode_id>/
    ├── topImg/<frame_id>.jpg
    ├── leftImg/<frame_id>.jpg
    ├── rightImg/<frame_id>.jpg
    └── observation/<frame_id>.pkl
```

每个 pkl 必须提供 14 维 `joint_positions` 和 14 维 `control`。转换命令：

```bash
python scripts/xtrainer/convert_raw_to_lerobot_2_1.py \
  --raw-root /data/xtrainer/collect_data \
  --output-root /data/xtrainer/dataset_v21 \
  --task "将试管放入试管架" \
  --fps 30 \
  --use-videos
```

转换器只保留状态、动作和三路图像均有效的帧。使用 `--fail-on-bad-frames` 可在首个坏帧处停止；使用
`--overwrite-output` 会替换目标目录，目标绝不能指向原始采集目录。

输出结构：

```text
dataset_v21/
├── meta/
│   ├── info.json
│   ├── stats.json
│   ├── tasks.jsonl
│   └── episodes.jsonl
├── data/chunk-000/episode_000000.parquet
└── videos/chunk-000/
    ├── observation.images.top/episode_000000.mp4
    ├── observation.images.left_wrist/episode_000000.mp4
    └── observation.images.right_wrist/episode_000000.mp4
```

## 6. 数据校验

训练前完整校验：

```bash
python scripts/xtrainer/validate_dataset_v21.py \
  --root /data/xtrainer/dataset_v21 \
  --all-episodes
```

校验内容包括格式版本、字段、14 维 shape、统计量、episode 长度、时间戳、任务索引、夹爪范围、视频存在性和抽样
解码。训练 launcher 默认再次执行快速校验；只有已经独立完成完整校验时才使用 `--skip-validation`。

如果旧数据的相机方向与当前安装不一致，使用 `tools/transform_xtrainer_dataset_images.py` 创建转换副本，不要直接
覆盖唯一的数据源。

## 7. 训练配置

主配置为 `configs/xtrainer/train_xvla.yaml`，关键参数：

```yaml
policy:
  path: lerobot/xvla-base
  type: xvla
  dtype: bfloat16
  action_mode: auto
  domain_id: 19
  max_action_dim: 20
  max_state_dim: 32
  num_image_views: 3
  chunk_size: 32
  n_action_steps: 32
  freeze_vision_encoder: false
  freeze_language_encoder: false
  train_policy_transformer: true
  train_soft_prompts: true
```

默认使用 XVLA optimizer preset：VLM 学习率是主学习率的 1/10，其余 transformer/action head 使用完整学习率。

启动训练：

```bash
bash scripts/xtrainer/train_xvla.sh \
  --dataset-root /data/xtrainer/dataset_v21 \
  --device cuda \
  --batch-size 4 \
  --steps 30000 \
  --output-dir outputs/train/xtrainer_xvla
```

显存不足时依次尝试：减小 batch size、开启分片训练、冻结视觉/语言 encoder。不要修改 `max_action_dim: 20`。

最小 smoke run：

```bash
bash scripts/xtrainer/train_xvla.sh \
  --dataset-root /data/xtrainer/smoke_v21 \
  --device cuda \
  --batch-size 1 \
  --steps 1 \
  --output-dir outputs/train/xtrainer_xvla_smoke
```

## 8. 断点续训

```bash
bash scripts/xtrainer/train_xvla.sh \
  --dataset-root /data/xtrainer/dataset_v21 \
  --resume-checkpoint outputs/train/xtrainer_xvla/checkpoints/last/pretrained_model \
  --device cuda
```

断点续训时 checkpoint 中保存的模型、processor、domain ID 和优化器配置为权威值；命令行仍可覆盖数据根目录、输出
目录、device、batch size 和总 steps。

## 9. Mock 联调

Mock 服务不会加载模型，也不会生成运动轨迹，只会把当前 14 维状态重复成 32 步动作块：

```bash
python scripts/xtrainer/serve_mock_policy.py --host 0.0.0.0 --port 8000 --chunk-size 32
```

自动化的协议、消息序列化和控制循环测试不会连接真实硬件：

```bash
pytest -q tests/xtrainer/test_websocket_transport.py tests/xtrainer/test_deploy_e2e.py
```

## 10. 真机硬件检查

先确认设备号、串口、相机序列号和 Dobot IP。硬件检查会依次读取三路相机、移动每个关节 ±5° 并开合夹爪，必须在
清空工作区且急停可触达后执行：

```bash
python scripts/xtrainer/check_real_hardware.py --execute
```

可先运行 `--help` 检查所有硬件覆盖参数。自动测试不会调用这个入口。

## 11. XVLA 策略服务

在 GPU 策略机启动：

```bash
python scripts/xtrainer/serve_policy.py \
  --config configs/xtrainer/deploy.yaml \
  --checkpoint outputs/train/xtrainer_xvla/checkpoints/last/pretrained_model \
  --device cuda \
  --domain-id 19 \
  --actions-per-chunk 32
```

服务启动时会完成一次 warmup。调试加载问题时可使用 `--no-warmup`。通过 `--log-actions` 可以记录返回动作与输入
图像，但图像以数组形式写入 JSONL，文件增长很快，只建议短时使用。

策略服务 metadata 固定声明：

```text
model_type=xvla
action_dim=14
state_dim=14
chunk_size=32
domain_id=19
```

## 12. 真机运行

在机器人控制机执行：

```bash
python scripts/xtrainer/run_real.py \
  --host <策略机IP> \
  --port 8000 \
  --task "将试管放入试管架" \
  --domain-id 19 \
  --action-horizon 32 \
  --control-hz 20 \
  --max-joint-delta 0.05 \
  --max-gripper-delta 0.03 \
  --max-steps 1000 \
  --execute
```

客户端完整执行每个策略返回的动作 chunk，不在 chunk 中途预取或混合新旧动作。chunk 结束后，客户端读取
当前真实状态并将其作为 hold 目标下发；在等待下一次推理结果期间不执行任何新的策略动作。最终动作仍经过
可选速率限制与环境安全限幅。

第一次上机建议：

1. 使用很小的 `--max-steps`。
2. 将关节与夹爪 delta 设为保守值。
3. 降低 `--control-hz`。
4. 开启 `--log-control` 保存请求、返回和实际下发动作。
5. 全程保持急停可触达。

## 13. 配置同步要求

以下值必须跨训练、服务和客户端保持一致：

| 配置 | 训练 | 服务 | 客户端 |
| --- | --- | --- | --- |
| action dim | 数据集 14D | 输出校验 14D | 输入校验 14D |
| model action dim | `max_action_dim=20` | checkpoint 内保存 | 无需感知 |
| domain | `domain_id=19` | `domain_id=19` | metadata 校验 |
| chunk | 32 | 32 | `action_horizon=32` |
| camera keys | top/left/right | top/left/right | top/left/right |

## 14. 常见问题

`Action dimension mismatch`：确认训练配置使用 `action_mode=auto` 和 `max_action_dim=20`，并部署微调后的 checkpoint，
不是未经适配的基础模型。

`Domain ID` 不一致：确认 YAML、服务参数和 checkpoint processor 都是 19。重新训练或切换 domain 后必须重新保存
processor。

图像范围错误：策略 wrapper 接收 HWC uint8，相机输入不要提前做 ImageNet normalization；XVLA processor 会统一处理。

CUDA OOM：先减小 batch size；正式训练推荐 bfloat16。CPU 只适合配置、转换和极小 smoke，不适合正常训练或实时推理。

视频无法解码：确认 TorchCodec、PyAV 和系统 FFmpeg 均可用，并用 `validate_dataset_v21.py --all-episodes` 定位损坏
episode。

策略服务无法连接：确认 8000 端口、防火墙和服务监听地址。该协议用于可信局域网，不应直接暴露公网。

## 15. 验收清单

- 数据集完整校验通过。
- XVLA 配置解析通过，domain=19、action mode=auto、max action dim=20。
- 1-step 训练能前向、反向并保存 checkpoint。
- checkpoint 目录包含模型 config、`model.safetensors` 与 pre/post processor 配置。
- XVLA wrapper 对合成三相机输入返回 `[N,14]` 有限动作。
- Mock WebSocket 端到端测试通过。
- 真机 metadata、reset pose 和动作安全限幅通过。
- 上机前完成单关节、夹爪和三相机检查。
