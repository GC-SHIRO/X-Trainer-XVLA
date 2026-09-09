# X-Trainer 部署 XVLA 手册

版本：V1.0  
日期：2026-09-06
适用代码：`GC-SHIRO/X-Trainer-XVLA` `main`

---

## 1. 文档目标

本文用于在 Dobot X-Trainer 双臂平台上完成 XVLA 的数据采集、数据转换、全量微调和真机部署。

流程：硬件检查 -> 遥操作采集 -> LeRobot Dataset v2.1 -> 数据校验 -> XVLA 全量微调 -> 策略服务 -> 真机执行。

## 2. 总体架构

### 2.1 端到端流程

```text
X-Trainer 硬件
    -> follower / leader / gripper / RealSense 检查
    -> raw episode 采集
    -> LeRobot Dataset v2.1
    -> XVLA 全量微调
    -> checkpoint
    -> policy server
    -> X-Trainer real client
```

### 2.2 项目分工

| 模块 | 位置 | 作用 |
|---|---|---|
| 采集和机器人控制 | `dobot_xtrainer` | 连接 Dobot、leader、夹爪和相机，保存 raw episode。 |
| 数据转换 | `scripts/xtrainer/convert_raw_to_lerobot_2_1.py` | 写入 LeRobot Dataset v2.1。 |
| 数据校验 | `scripts/xtrainer/validate_dataset_v21.py` | 检查字段、维度、统计量、视频和 episode。 |
| 训练 | `scripts/xtrainer/train_xvla.sh` | 使用 `configs/xtrainer/train_xvla.yaml` 训练。 |
| 策略服务 | `scripts/xtrainer/serve_policy.py` | 通过 WebSocket + MessagePack 提供动作。 |
| 真机客户端 | `scripts/xtrainer/run_real.py` | 读取状态和图像并下发动作。 |

### 2.3 关键数据契约

X-Trainer 的状态和真实动作均为 14 维：

| 索引 | 内容 |
|---|---|
| `0-5` | 左臂关节 1-6，弧度 |
| `6` | 左夹爪，归一化到 `[0,1]` |
| `7-12` | 右臂关节 1-6，弧度 |
| `13` | 右夹爪，归一化到 `[0,1]` |

视觉字段固定为 `observation.images.top`、`observation.images.left_wrist` 和 `observation.images.right_wrist`。

XVLA 基础模型的动作头为 20 维。`action_mode: auto` 会把 14 维标签补零到 20 维，只对真实 14 维计算损失，推理结果再裁回 14 维；`max_action_dim` 必须保持为 `20`。

本项目使用 `domain_id: 19`。训练配置、checkpoint、策略服务和真机客户端必须使用同一个 domain ID。

## 3. 硬件和软件前置条件

### 3.1 硬件组成

| 硬件 | 数量 | 用途 |
|---|---:|---|
| Dobot follower 机械臂 | 2 | 执行动作。 |
| leader 主手 | 2 | 输入遥操作动作。 |
| Feetech / X-Trainer 夹爪 | 2 | 控制左右夹爪。 |
| Intel RealSense | 3 | 顶部、左腕、右腕图像。 |
| GPU 训练机 | 1 | 训练和策略服务。 |
| 机器人控制机 | 1 | 连接硬件、采集和执行。 |

### 3.2 网络和设备约定

```text
左臂 follower: 192.168.5.1
右臂 follower: 192.168.5.2
```

```bash
ping 192.168.5.1
ping 192.168.5.2
ls -l /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || true
```

实际串口和相机序列号以现场配置为准，不能只按 USB 枚举顺序判断左右设备。

### 3.3 系统建议

安装脚本按 Ubuntu 24.04 x86_64、Python 3.12、PyTorch 2.8.0 CUDA 12.8 准备。GPU 模式要求 NVIDIA 驱动不低于 `570.26`。

## 4. 环境部署
本仓库提供全自动环境部署脚本，在对应的环境下可以直接运行脚本进行一键部署

在仓库根目录执行：

```bash
bash tools/install_xtrainer_env.sh
conda activate xtrainer-xvla
```

使用官方软件源：

```bash
bash tools/install_xtrainer_env.sh --source official
```

只做数据转换时可使用 CPU 环境：

```bash
bash tools/install_xtrainer_env.sh --cpu-only
conda activate xtrainer-xvla
```

脚本不安装显卡驱动、不下载模型，也不修改串口权限。

## 5. X-Trainer 硬件配置

### 5.1 配置内容

配置左右 Dobot IP、leader 串口、夹爪串口、夹爪 ID 和三台 RealSense 序列号。不要把密码、私有串口和设备序列号提交到公共仓库。

动作排列必须保持：

```text
left_j1 ... left_j6, left_gripper,
right_j1 ... right_j6, right_gripper
```

### 5.2 配置文件位置

采集侧沿用 `dobot_xtrainer/scripts/dobot_config/dobot_settings.ini`。其中包含相机序列号、leader 串口、夹爪串口、关节 ID、offset、方向和初始姿态。密码和现场设备信息不要提交到公共仓库。

### 5.3 自动扫描串口

```bash
cd /path/to/workspace/dobot_xtrainer
python scripts/1_find_port.py
```

脚本扫描 `/dev/ttyACM*` 和 `/dev/ttyUSB*`，识别左右 leader 与夹爪并写回配置。设备重插后应重新扫描。

### 5.4 标定 leader offset

```bash
python scripts/2_get_offset.py
```

标定前将 leader 放到约定的初始姿态，确认左右串口、`joint_ids`、`append_id` 和波特率正确。

### 5.5 检查相机和硬件

```bash
ping 192.168.5.1
ping 192.168.5.2
ls -l /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || true
python scripts/5_camera_read.py
python scripts/xtrainer/check_real_hardware.py --help
python scripts/xtrainer/check_real_hardware.py --execute
```

执行检查前清空工作区并确认急停可触达。确认相机序列号、左右映射和关节方向后再采集。

## 6. 遥操作与数据采集

### 6.1 启动 follower server

在终端 1 执行：

```bash
cd /path/to/workspace/dobot_xtrainer
conda activate xtrainer
python experiments/launch_nodes.py --hostname 127.0.0.1 --robot-port 6001
```

启动前确认左右 Dobot IP 可达、控制盒处于 TCP/IP 控制状态，且没有安全保护未复位。

### 6.2 启动遥操作和采集程序

在终端 2 执行：

```bash
cd /path/to/workspace/dobot_xtrainer
conda activate xtrainer
python experiments/run_control.py \
  --hostname 127.0.0.1 \
  --robot-port 6001 \
  --show-img True
```

### 6.3 按钮语义

| 操作 | 作用 |
|---|---|
| Button A 短按 | leader lock / unlock。 |
| Button A 长按超过 1 秒 | 对应侧 follower servo start / stop。 |
| Button B 按下 | 开始 / 停止 recording。 |

推荐顺序：启动 follower server，启动 `run_control.py`，短按 A 解锁 leader，长按 A 启动 servo，确认跟随方向后按 B 录制；停止录制后再停 servo，并锁定 leader。

### 6.4 采集前准备

确认 follower 上电、Dobot 网络可达、leader 和夹爪串口可用、三路相机画面正常。先用低速动作确认左右臂和夹爪方向。

### 6.5 采集输出结构

```text
collect_data/
└── <episode_id>/
    ├── topImg/<frame_id>.jpg
    ├── leftImg/<frame_id>.jpg
    ├── rightImg/<frame_id>.jpg
    └── observation/<frame_id>.pkl
```

每个 pkl 至少包含 `joint_positions` 和 `control`，两者均为 14 维。三路图像和 pkl 必须按同一帧号对应。

### 6.6 采集要求

1. 每个任务先采集少量短 episode，确认转换和训练链路可用。
2. 每条 episode 从稳定初始场景开始，到任务完成后结束。
3. 相机覆盖目标物体、末端执行器和操作区域。
4. leader 动作保持平滑，避免快速大幅移动。
5. 训练和部署使用固定任务描述，例如 `将桌面上的方块放入收纳盒`。

### 6.7 采集后检查

```bash
find /data/xtrainer/collect_data -maxdepth 1 -mindepth 1 -type d | wc -l
find /data/xtrainer/collect_data/<episode_id>/observation -name "*.pkl" | wc -l
find /data/xtrainer/collect_data/<episode_id>/topImg -name "*.jpg" | wc -l
```

帧数不一致时，先处理 raw 数据，再转换。

## 7. Raw 数据转换为 LeRobot 格式

本仓库默认使用 `lerobot V2.1` 数据集格式，在代码层已对数据集做只读兼容，并提供数据集转换脚本

### 7.1 转换命令

```bash
python scripts/xtrainer/convert_raw_to_lerobot_2_1.py \
  --raw-root /data/xtrainer/collect_data \
  --output-root /data/xtrainer/dataset_v21 \
  --task "将桌面上的方块放入收纳盒" \
  --fps 30 \
  --use-videos \
  --overwrite-output
```

`--output-root` 不能指向 raw 目录；需要在坏帧处停止时增加 `--fail-on-bad-frames`。

### 7.2 字段映射

| raw 字段 | LeRobot 字段 |
|---|---|
| `joint_positions` | `observation.state` |
| `control` | `action` |
| `topImg` | `observation.images.top` |
| `leftImg` | `observation.images.left_wrist` |
| `rightImg` | `observation.images.right_wrist` |
| `--task` | episode task |

输出至少包含 `meta/info.json`、`meta/stats.json`、`meta/tasks.jsonl`、`meta/episodes.jsonl`、parquet 数据和三路视频。

### 7.3 相机转换

由于原相机存在左右手反转不一致问题，因此本仓库增加了转换数据集的脚本，需要额外进行一次相机反转角度转换，使用 `tools/transform_xtrainer_dataset_images.py` 生成副本，不覆盖唯一数据源。

使用方式：

```bash
python tools/transform_xtrainer_dataset_images.py \
  --input-root 当前 collect_items 目录 \
  --output-root 目标目录 \
  --overwrite-output
```

### 7.4 如何理解转换

采集程序生成的是便于机器人实时写入的 raw 数据，而训练程序需要的是 LeRobot Dataset v2.1。转换脚本的作用可以理解为“整理和打包”，不会改变任务本身：


因此，转换前后的 episode 数量通常应该一致。转换成功并不代表数据一定适合训练，还需要执行第 9 节的数据校验，并抽查图片和动作是否对应。

### 7.5 参数怎么选

- `--raw-root` 是采集结果所在目录，下面应直接包含多个 episode 子目录；不要填写某一个 episode 的路径。
- `--output-root` 是新数据集目录。建议使用一个不存在的目录，或确认 `--overwrite-output` 不会覆盖仍需保留的数据。
- `--task` 是这批 episode 的任务描述。训练和真机执行时应使用相同或含义一致的文字，避免只改写同一任务的说法。
- `--fps` 应与采集时的实际帧率接近。帧率过高会让视频播放过快，过低会让动作和图像的时间关系变差。
- `--use-videos` 会生成视频并减少大量小图片文件，通常适合正式训练；调试单个样本时也可以先不使用它以便直接查看图片。
- `--fail-on-bad-frames` 会在遇到坏帧时立即停止，适合数据质量要求较高的正式转换；不加此参数时应在转换日志中确认是否跳过了坏帧。




## 8. 模型数据配置

XVLA 读取以下字段，不要改成其他相机名称：

```text
observation.state
observation.images.top
observation.images.left_wrist
observation.images.right_wrist
action
task
```

训练配置为 `configs/xtrainer/train_xvla.yaml`。关键值必须保持：

```yaml
policy:
  action_mode: auto
  domain_id: 19
  max_action_dim: 20
  max_state_dim: 20
  num_image_views: 3
```

数据集标签仍是 14 维，XVLA 内部负责补零、计算损失和裁剪输出。



## 9. 基础模型与数据校验

使用 Hugging Face 下载基础模型：

```bash
bash tools/download_xvla_weights_hf.sh
```

或使用 ModelScope：

```bash
bash tools/download_xvla_weights_modelscope.sh
```

默认目录为 `models/xvla-base`。需要使用 Hugging Face 镜像时：

```bash
bash tools/download_xvla_weights_hf.sh --endpoint https://hf-mirror.com
```

训练前执行完整数据校验：

```bash
python scripts/xtrainer/validate_dataset_v21.py \
  --root /data/xtrainer/dataset_v21 \
  --all-episodes
```



## 10. XVLA 全量微调

### 10.1 Smoke training

为了保证运行过程能够正常进行，需要先进行 Smoke run 确保正常运行。

```bash
bash scripts/xtrainer/train_xvla.sh \
  --dataset-root /data/xtrainer/smoke_v21 \
  --device cuda \
  --batch-size 1 \
  --steps 1 \
  --output-dir outputs/train/xtrainer_xvla_smoke
```

### 10.2 正式训练

```bash
bash scripts/xtrainer/train_xvla.sh \
  --dataset-root /data/xtrainer/dataset_v21 \
  --device cuda \
  --batch-size 8 \
  --steps 30000 \
  --output-dir outputs/train/xtrainer_xvla
```

显存不足时先减小 batch size，再根据训练配置调整冻结项。保持 `action_mode: auto`、`max_action_dim: 20` 和 `domain_id: 19` 不变。训练输出的 `pretrained_model` 目录用于策略服务。

先完成一次 Smoke training，只用来验证环境、模型和数据管线可以运行，并不代表模型已经学会任务。确认成功后再增加训练步数，并观察显存占用和 loss 是否正常。

## 11. 断点续训

```bash
bash scripts/xtrainer/train_xvla.sh \
  --dataset-root /data/xtrainer/dataset_v21 \
  --resume-checkpoint outputs/train/xtrainer_xvla/checkpoints/last/pretrained_model \
  --device cuda \
  --batch-size 8 \
  --steps 30000 \
  --output-dir outputs/train/xtrainer_xvla
```

断点续训时，checkpoint 中保存的模型、processor 和 domain ID 为准。数据根目录、输出目录、设备、batch size 和总步数可按任务调整。


续训时选择上一次训练生成的 `last/pretrained_model`。如果更换了数据集或任务，应重新确认训练结果是否仍然具有可比性。

## 12. 启动策略服务

```bash
python scripts/xtrainer/serve_policy.py \
  --checkpoint 对应的 pretrained_model 目录 \
  --device cuda \
  --actions-per-chunk 32 
```

`serve_policy.py` 的命令行参数均为可选覆盖项；未指定时从 `configs/xtrainer/deploy.yaml` 读取。checkpoint 和 domain 虽然已有配置默认值，但启动前必须确认。

| 类型 | 参数 | 默认值或来源 | 作用和注意事项 |
|---|---|---|---|
| 配置必查 | `--checkpoint` / `--model-path` | `policy.checkpoint` | 要部署的 `pretrained_model` 目录；两个参数名等价。必须使用完成 X-Trainer 微调后的 checkpoint。 |
| 配置必查 | `--domain-id` | `policy.domain_id` | 覆盖 XVLA domain，必须与 checkpoint 训练时的 domain 一致。命令行和 YAML 都不指定时自动读取 checkpoint；显式冲突时直接报错。 |
| 可选 | `--config` | `configs/xtrainer/deploy.yaml` | 部署配置文件路径。 |
| 可选 | `--device` | `policy.device`，当前为 `cuda` | 模型加载和推理设备。 |
| 可选 | `--host` | `network.host`，当前为 `0.0.0.0` | 服务监听地址。`0.0.0.0` 表示监听本机所有网络接口。 |
| 可选 | `--port` | `network.port`，当前为 `8000` | 服务监听端口，必须与 `run_real.py --port` 一致。 |
| 可选 | `--actions-per-chunk` / `--use-length` | `policy.actions_per_chunk`，当前为 `32` | 每次推理返回的动作步数；两个参数名等价，且不能超过 checkpoint 的 `chunk_size`。 |
| 可选 | `--log-actions` | 关闭 | 将每次返回的动作块和输入图像写入 JSONL。图像数据会使文件快速增大并影响推理速度，仅建议短时调试。 |
| 可选 | `--action-log-path` | `outputs/xtrainer/action_logs/actions_<UTC>.jsonl` | 自定义动作日志路径；仅在启用 `--log-actions` 时生效。 |
| 可选 | `--no-warmup` | 关闭 | 跳过服务启动时的首次 warmup 推理，主要用于排查模型加载问题；正常部署建议保留 warmup。 |

服务端和机器人控制机应位于可信局域网，监听地址和客户端目标 IP 必须对应。


## 13. 启动真机任务

首次运行使用较小步数和动作增量：

```bash
python scripts/xtrainer/run_real.py \
  --host 策略机的地址 \
  --task "左手拿盒子，右手夹取，并把桌上所有的杂物放进盒子中" \
  --control-hz 30 \
  --max-steps 10000 \
  --bin-gripper \
  --action-horizon 32 \
  --execute
```

`run_real.py` 的参数如下。`必须传入` 表示命令行缺少该参数就无法运行；`真机必查` 表示参数已有默认值，但启动前必须确认它与 checkpoint、任务或现场硬件一致；其余参数均为按需覆盖的可选项。

| 类型 | 参数 | 默认值 | 作用和注意事项 |
|---|---|---|---|
| 必须传入 | `--host` | 无 | XVLA 策略服务的 IP 地址或主机名。 |
| 必须传入 | `--execute` | 关闭 | 显式允许连接、使能并移动真机；不传时程序会拒绝执行。仅在完成硬件检查、清空工作区并确保急停可触达后启用。 |
| 真机必查 | `--task` | `pick up the object` | 发送给 XVLA 的任务文本，应与训练数据中的任务描述一致。中文任务可以直接传入。 |
| 真机必查 | `--domain-id` | `19` | 客户端期望的 XVLA domain，必须与策略服务 metadata 和训练 checkpoint 一致，范围为 `[0, 30)`。 |
| 可选 | `--port` | `8000` | 策略服务端口，必须与 `serve_policy.py` 的监听端口一致。 |
| 可选 | `--action-horizon` | `32` | 每次策略响应最多执行的动作步数。缩短动作块时，服务端的 `--actions-per-chunk` 也应同步调整。 |
| 可选 | `--control-hz` | `30` | 真机动作下发频率。32 步动作块在 30 Hz 下约执行 `1.07` 秒；实际频率还会受通信和硬件耗时影响。 |
| 可选 | `--max-steps` | `1000` | 整次任务最多执行的动作步总数，不是动作块数量。首次上机建议设为较小值。 |
| 可选 | `--camera-warmup-frames` | `10` | 每台相机启动后丢弃的预热帧数；设为 `0` 可跳过预热。 |
| 可选 | `--image-jpeg-quality` | `85` | 三路 RGB 图像的传输 JPEG 质量，范围为 `0–100`；`0` 关闭压缩并发送原始数组。不支持 JPEG 的服务端会自动回退到原始数组。 |
| 可选 | `--max-joint-delta` | `inf`（关闭） | 环境层单步关节变化上限，单位为弧度；默认不限制。首次上机可显式设置保守值。 |
| 可选 | `--max-gripper-delta` | `inf`（关闭） | 环境层单步夹爪变化上限；默认不限制。首次上机可显式设置保守值。 |
| 可选 | `--ramp-step` | `0.01` | 机器人移动到 reset pose 时使用的插值步长，不影响正常策略动作。 |
| 可选 | `--ramp-max-steps` | `100` | reset pose 插值的最大步数。 |
| 可选 | `--gripper-update-threshold` | `0.0` | 夹爪目标相对上次指令的最小变化量；`0` 表示发送所有变化。 |
| 可选 | `--max-delta-per-step` | `0.0`（关闭） | 客户端最终下发前对全部 14 维动作施加的单步变化上限；小于或等于 `0` 时关闭。 |
| 可选 | `--chunk-blend-steps` | `6` | 从第二个动作块开始，用前 N 步将双臂关节从 hold 位置平滑过渡到模型目标；不处理夹爪，也不增加动作步数。30 Hz、6 步时约过渡 `200 ms`；`0` 关闭。 |
| 可选 | `--bin-gripper` | 关闭 | 左右夹爪独立判断：模型值低于 `0.5` 时，从上一次实际位置开始，每周期向闭合方向 `0` 递减 `0.05`；不低于 `0.5` 时保留模型原值。30 Hz 下从 `1` 到 `0` 约需 `0.67` 秒。 |
| 可选 | `--log-control` | 关闭 | 将状态、模型动作和最终下发动作写入客户端 JSONL 日志。日志不保存图像，通常不会显著影响控制速度。 |
| 可选 | `--control-log-path` | `outputs/xtrainer/control_logs/control_<UTC>.jsonl` | 自定义 `--log-control` 的日志路径；仅在启用 `--log-control` 时生效。 |



## 14. 关键文件索引

```text
tools/install_xtrainer_env.sh
tools/download_xvla_weights_hf.sh
scripts/xtrainer/convert_raw_to_lerobot_2_1.py
scripts/xtrainer/validate_dataset_v21.py
scripts/xtrainer/train_xvla.sh
scripts/xtrainer/serve_policy.py
scripts/xtrainer/run_real.py
configs/xtrainer/train_xvla.yaml
configs/xtrainer/deploy.yaml
```


