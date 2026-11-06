# X-Trainer 部署 XVLA 手册

版本：V1.0  
更新日期：2026-09-25
适用代码：`GC-SHIRO/X-Trainer-XVLA` `main`

---

## 1. 文档目标

本文用于在 Dobot X-Trainer 双臂平台上完成 XVLA 的数据采集、数据转换、全量微调或 LoRA 微调和真机部署。

流程：硬件检查 -> 遥操作采集 -> LeRobot Dataset v2.1 -> 数据校验 -> XVLA 全量微调 -> 策略服务 -> 真机执行。

LoRA 路线见第 15 节：官方原始基座 -> LoRA 微调 -> 合并导出完整模型 -> 复用第 12、13 节的服务和真机客户端。

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
| LoRA 训练 | `scripts/xtrainer/train_xvla_lora.sh` | 独立 LoRA 配置、梯度累积和续训入口。 |
| LoRA 导出 | `scripts/xtrainer/merge_xvla_lora.py` | 校验基座、合并 adapter 并验证完整模型输出。 |
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

## 12. 启动策略服务（serve_policy.py）

在运行模型的机器上启动服务，再到连接机械臂和相机的控制机上运行第 13 节的客户端。以下命令使用 Linux/Bash，在仓库根目录执行，并使用已安装项目依赖的 Python 环境。

### 12.1 默认启动：guided RTC

```bash
python scripts/xtrainer/serve_policy.py \
  --config configs/xtrainer/deploy.yaml \
  --checkpoint outputs/train/xtrainer_xvla/checkpoints/last/pretrained_model \
  --device cuda \
  --host 0.0.0.0 \
  --port 8000 \
  --actions-per-chunk 32
```

将 `--checkpoint` 换成实际模型目录。checkpoint 是训练后保存的模型和配置；如果使用 LoRA 微调，需要先合并导出完整模型，见第 15 节。`32` 必须与该 checkpoint 的 `config.json` 中 `chunk_size` 一致，不能为了加快响应任意缩短。

默认的 `configs/xtrainer/deploy.yaml` 已启用以下设置。此处只是文件中的相关片段，修改现有字段即可，不要用它覆盖整个部署文件：

```yaml
policy:
  actions_per_chunk: 32
  rtc:
    enabled: true
    mode: guided
    execution_horizon: 10
    max_guidance_weight: 10.0
```

RTC（实时动作段衔接）让机器人执行当前动作段的同时，提前生成下一段。`guided` 表示生成时参考旧段中尚未执行的动作，减少切换时的突变，无需重新训练模型。

| YAML 字段 | 当前值 | 含义与约束 |
|---|---|---|
| `policy.rtc.enabled` | `true` | 开启 RTC；设为 `false` 后改为执行完一段、保持当前位置并等待下一次推理。省略整个 `rtc` 配置时也关闭。 |
| `policy.rtc.mode` | `guided` | 当前只支持此模式；填写 `trained` 或其他值会报错。 |
| `policy.rtc.execution_horizon` | `10` | 参与衔接引导的前缀范围，必须是 `1` 到 `chunk_size` 之间的整数。前缀就是旧段尚未执行的部分。预计延迟较大时，实际引导范围会扩大以覆盖延迟，但不超过已有前缀。 |
| `policy.rtc.max_guidance_weight` | `10.0` | 引导强度的上限，必须是有限正数；初始沿用 Pi05 的设置，效果需要实测。 |

RTC 通过 YAML 配置，**没有 `--rtc` 或 `--mode` 命令行参数**。修改后需要重启策略服务。客户端连接时读取服务端信息，自动选择 RTC 或同步分块模式。

### 12.2 服务端参数

模型、domain、设备、监听地址、端口和动作段长度的命令行选项优先于 YAML。domain 是训练时使用的任务域编号，用于选择模型内对应的提示参数；必须与 checkpoint 一致。

| 参数 | 默认值或来源 | 作用和注意事项 |
|---|---|---|
| `--config` | `configs/xtrainer/deploy.yaml` | 部署配置文件路径。 |
| `--checkpoint` / `--model-path` | `policy.checkpoint` | 完整 `pretrained_model` 目录；两个参数名等价。 |
| `--domain-id` | `policy.domain_id`，当前为 `19` | 覆盖任务域编号。命令行和 YAML 都不指定时，自动读取 checkpoint；显式指定但与 checkpoint 冲突时会报错。 |
| `--device` | `policy.device`，当前为 `cuda` | 模型推理设备。 |
| `--host` | `network.host`，当前为 `0.0.0.0` | 服务监听地址。`0.0.0.0` 表示监听本机所有网络接口；客户端应填写这台机器可访问的实际 IP。 |
| `--port` | `network.port`，当前为 `8000` | 服务端口，必须与客户端 `--port` 一致。 |
| `--actions-per-chunk` / `--use-length` | `policy.actions_per_chunk`，当前为 `32` | 每次返回的动作步数。RTC 开启时必须等于 checkpoint 的完整 `chunk_size`；关闭时可取 `1` 到 `chunk_size`。 |
| `--no-warmup` | 不启用，默认进行预热 | 跳过启动预热。RTC 开启时，默认会预热首次推理和带旧动作前缀的推理；预热完成后才启动网络监听。 |
| `--log-actions` | 关闭 | 将返回的机器人动作和输入图像写入 JSONL（一行一条记录）日志。图像会使文件快速增大并增加推理耗时，仅建议短时调试。 |
| `--action-log-path` | `outputs/xtrainer/action_logs/actions_<UTC>.jsonl` | 自定义服务端日志路径；需同时传入 `--log-actions`。文件名时间为 UTC。 |

服务端与控制机应位于可信局域网。服务端只负责模型推理；运行此命令不会连接或使能机械臂。

## 13. 启动真机任务（run_real.py）

### 13.1 默认启动：RTC + EMA 滤波

策略服务启动完成后，在连接机械臂和相机的控制机上执行。将示例 IP 换成策略机地址，任务文本换成与训练数据一致的描述；同机部署可使用 `127.0.0.1`。

```bash
python scripts/xtrainer/run_real.py \
  --host 192.168.1.100 \
  --port 8000 \
  --domain-id 19 \
  --task "左手拿盒子，右手夹取，并把桌上所有的杂物放进盒子中" \
  --control-hz 30 \
  --max-steps 300 \
  --joint-ema-alpha 0.25 \
  --gripper-ema-alpha 0.5 \
  --gripper-ema-jump-threshold 0.4 \
  --log-control \
  --execute
```

示例执行 300 个策略动作，按 30 Hz 计算约为 10 秒，不包含初始化、复位和首次推理时间，实际耗时还受通信与硬件影响。三个滤波参数已是默认值，可省略。`--execute` 显式允许真机运动；连接后客户端会校验模型信息、使能机械臂、按服务端提供的复位姿态移动，再开始执行策略。

**`run_real.py` 不读取 `deploy.yaml`，也没有 `--config` 参数。** 控制频率、相机、机械臂、滤波、超时及限幅设置均由客户端参数决定；只修改 YAML 中的 `xtrainer.control_hz`、`cameras`、`hardware` 或 `safety` 不会改变客户端的实际设置。

### 13.2 RTC 与滤波如何配合

RTC 负责新旧动作段的衔接；EMA（指数移动平均）负责每一步的平滑。关节默认每步从上一次实际下发目标向新目标靠近 25%；夹爪小幅变化时靠近 50%，变化超过 `0.4` 时直接通过滤波。比例越小，动作越平滑，但响应也越慢；设为 `1` 则该部分不做平滑。

滤波只处理即将执行的一步，跨段连续，并以上一次经过硬件限幅后的下发目标作为参考。模型使用的旧动作前缀保持未经滤波的原值。实际处理顺序为：模型动作 → EMA / 可选夹爪模式 → 客户端限幅 → 环境层限幅 → 硬件下发。

| 使用方式 | 服务端设置 | 客户端设置 |
|---|---|---|
| RTC + EMA（默认） | `policy.rtc.enabled: true` | 不传 `--no-ema` |
| 仅 RTC，关闭 EMA | `policy.rtc.enabled: true` | 添加 `--no-ema` |
| 同步分块 + EMA | `policy.rtc.enabled: false`，重启服务 | 不传 `--no-ema` |
| 同步分块，关闭 EMA | `policy.rtc.enabled: false`，重启服务 | 添加 `--no-ema` |

如需原有的逐步闭合夹爪行为，在启动命令中追加 `--bin-gripper`。此时关节仍做 EMA，夹爪不做 EMA：模型夹爪值低于 `0.5` 时，每周期从上次下发目标向闭合方向递减 `0.05`；不低于 `0.5` 时保留模型目标。后续限幅仍会生效。上面的默认示例使用连续夹爪滤波，因此没有附加此选项。

**旧的 `--chunk-blend-steps`、`--chunk-smoothing-strength`、`--chunk-stop-steps` 已删除。** 启动脚本中也需要移除这些参数，不能通过传 `0` 继续保留。

### 13.3 控制、滤波与日志参数

| 参数 | 默认值 | 作用和注意事项 |
|---|---|---|
| `--host` | 必填 | 策略机的 IP 或主机名，不带 `http://`。 |
| `--port` | `8000` | 与策略服务端口一致。 |
| `--execute` | 关闭 | 显式允许连接、使能和移动真机；不传时程序拒绝执行。 |
| `--task` | `pick up the object` | 提供给模型的任务文本，应与训练任务一致，支持中文。 |
| `--domain-id` | `19` | 客户端期望的任务域编号，必须与服务端和 checkpoint 一致。 |
| `--control-hz` | `30` | 动作下发频率，单位 Hz（每秒次数），必须大于 0。 |
| `--max-steps` | `1000` | 最多执行的策略动作步总数，必须为正数；不含复位和退出时的保持指令。 |
| `--action-horizon` | `32` | 仅同步分块模式生效：每次最多保留这么多步。RTC 模式忽略此值并使用服务端完整段长，但参数仍须为正数。 |
| `--request-timeout` | `10.0` 秒 | 单次推理请求超时；RTC 动作队列可能在此之前耗尽并触发退出。 |
| `--no-ema` | 不启用，EMA 默认开启 | 关闭关节和连续夹爪的 EMA；不关闭 RTC，也不关闭 `--bin-gripper` 或限幅。 |
| `--joint-ema-alpha` | `0.25` | 关节滤波比例，必须是 `(0,1]` 内的有限值。 |
| `--gripper-ema-alpha` | `0.5` | 连续夹爪滤波比例，必须是 `(0,1]` 内的有限值；启用 `--bin-gripper` 时不用于夹爪。 |
| `--gripper-ema-jump-threshold` | `0.4` | 夹爪变化超过此阈值时直接通过 EMA；必须是有限非负数。夹爪取值范围为 `[0,1]`。 |
| `--bin-gripper` | 关闭 | 将低于 `0.5` 的夹爪目标转为每步递减 `0.05` 的闭合指令；从 `1` 到 `0` 在 30 Hz 下约需 `0.67` 秒。 |
| `--max-delta-per-step` | `0.0`（关闭） | 最终下发前，对全部 14 维动作施加单步变化上限；必须非负，`0` 关闭。关节为弧度、夹爪为 `[0,1]` 数值。 |
| `--max-joint-delta` | `inf`（关闭） | 环境层单步关节变化上限，单位为弧度。 |
| `--max-gripper-delta` | `inf`（关闭） | 环境层单步夹爪变化上限。 |
| `--ramp-step` | `0.01` | 移动到复位姿态时的关节插值步长，单位为弧度。 |
| `--ramp-max-steps` | `100` | 复位插值最多执行的步数。 |
| `--gripper-update-threshold` | `0.0` | 夹爪目标相对上次指令的最小变化量，`0` 表示发送所有变化。 |
| `--log-control` | 关闭 | 写入客户端 JSONL 日志，包括推理请求、RTC 切换、原始动作、滤波动作、实际下发动作及控制间隔；不保存图像。 |
| `--control-log-path` | `outputs/xtrainer/control_logs/control_<UTC>.jsonl` | 自定义客户端日志路径；需同时传入 `--log-control`。文件名时间为 UTC。 |

### 13.4 硬件与图像参数

以下默认值对应当前 X-Trainer 配置，换控制机或硬件后应在客户端命令中覆盖。

| 参数 | 默认值 | 作用 |
|---|---|---|
| `--left-robot-ip` / `--left-arm-ip` | `192.168.5.1` | 左臂 IP，两种写法等价。 |
| `--right-robot-ip` / `--right-arm-ip` | `192.168.5.2` | 右臂 IP，两种写法等价。 |
| `--left-gripper-port` / `--right-gripper-port` | `/dev/ttyUSB1` / `/dev/ttyUSB0` | 左右夹爪串口。 |
| `--left-gripper-id` / `--right-gripper-id` | `21` / `22` | 左右夹爪电机 ID。 |
| `--camera-top-serial` / `--top-camera-serial` | `409122273405` | 顶部相机序列号，两种写法等价。 |
| `--camera-left-wrist-serial` / `--left-wrist-camera-serial` | `412622272997` | 左腕相机序列号，两种写法等价。 |
| `--camera-right-wrist-serial` / `--right-wrist-camera-serial` | `412622271417` | 右腕相机序列号，两种写法等价。 |
| `--camera-fps` | `30` | 相机采集帧率。 |
| `--camera-width` / `--camera-height` | `640` / `480` | 相机图像宽度、高度。 |
| `--camera-warmup-frames` | `10` | 每台相机启动后丢弃的预热帧数，`0` 可跳过。 |
| `--image-jpeg-quality` | `85` | 图像传输 JPEG 质量，范围 `0–100`；`0` 发送原始 RGB 数组。服务端不支持 JPEG 时自动回退。 |

### 13.5 段长、延迟与排查

一次模型预测生成完整动作段，客户端同时只保留一个待完成的推理请求。当剩余动作不超过 `max(15, delay + 5)` 步时提前请求下一段，其中 `delay` 是根据已观测耗时估计的延迟步数。新段返回后，按这段时间内实际执行的动作数对齐；实际经过的时间用于延迟估计和过期判断，不直接用于跳过动作。

32 步、30 Hz 对应约 1.07 秒的整段时间，但这并不意味着下一次推理总有 1.07 秒可用；例如剩 15 步时才触发请求，可用执行时间约为 0.5 秒。采集、传输、模型推理和控制耗时都需要实测。客户端日志中的 `send_interval_ms` 是下发间隔，30 Hz 下目标约为 33.3 ms；`rtc_switch` 记录切换时目标差值。

| 现象或报错 | 处理方法 |
|---|---|
| `RTC requires the full checkpoint chunk_size` | 将服务端 `--actions-per-chunk` 或 `policy.actions_per_chunk` 设为 checkpoint 的完整段长；调整客户端 `--action-horizon` 不能解决。 |
| `supports rtc.mode='guided' only` | 将 YAML 中的 `policy.rtc.mode` 改为 `guided`，重启服务。 |
| `RTC prefix cannot cover observed latency`、`RTC prediction expired` 或 `RTC queue exhausted` | 查看服务端推理耗时、客户端下发间隔和采集耗时；确认启动预热已完成，检查网络及模型性能。单纯增大 `--request-timeout` 不会增加队列里的动作数。 |
| domain 不匹配 | 核对 checkpoint 中的 `domain_id`、服务端配置和客户端 `--domain-id`。 |
| `unexpected keyword argument 'rtc_processor'` 或 `does not support guided RTC` | 部署代码与模型代码版本不一致。确认 `src/lerobot/policies/xvla/modeling_xvla.py` 和部署文件都已更新；在当前 Conda 环境、仓库根目录执行 `python -m pip install --no-deps -e .` 后重启服务。新版启动脚本优先使用当前仓库的 `src`，并打印 `XVLA model implementation` 路径供核对。 |
| `Connect call failed ('127.0.0.1', 8000)` | 确认策略服务没有在加载或预热阶段报错，并已开始监听；跨机部署时客户端 `--host` 应填写策略机实际 IP，`127.0.0.1` 只指向客户端本机。 |
| `unrecognized arguments: --chunk-...` | 从旧启动脚本中删除已移除的三个消抖参数，改用本节的 EMA 参数。 |

正常 RTC 段间不再停住等待；队列耗尽、预测过期或推理失败时会退出。结束时执行器尝试发送当前测得姿态作为保持目标，随后客户端清理流程撤销机械臂使能，并关闭硬件资源和网络连接。

## 14. 关键文件索引

```text
tools/install_xtrainer_env.sh
tools/download_xvla_weights_hf.sh
scripts/xtrainer/convert_raw_to_lerobot_2_1.py
scripts/xtrainer/validate_dataset_v21.py
scripts/xtrainer/train_xvla.sh
scripts/xtrainer/serve_policy.py
scripts/xtrainer/run_real.py
deploy/xtrainer/rtc_runner.py
configs/xtrainer/train_xvla.yaml
configs/xtrainer/deploy.yaml
```

## 15. XVLA LoRA 微调与部署

本节按当前仓库实现编写。LoRA 不依赖已有全量微调模型；使用本地官方原始 checkpoint 作为基座，训练后与同一份基座合并。

```text
本地官方原始基座 + LeRobot v2.1 数据
    -> LoRA adapter + 额外训练模块
    -> 与原始基座合并
    -> 完整 checkpoint
    -> serve_policy.py
    -> run_real.py
```

独立 adapter 不能直接传给当前真机部署入口。全量模型无需执行 LoRA 合并步骤。所有示例在已配置环境的 Linux/Bash 中，从仓库根目录执行；路径替换为实际值。

### 15.1 环境与路径

LoRA 需要 PEFT。安装脚本当前安装的 extras 没有显式包含 `peft`，请在同一个训练环境中检查：

```bash
conda activate xtrainer-xvla
python -c "import torch, peft, transformers, accelerate; print('LoRA imports OK')"
```

若缺少 PEFT，可在该环境补装项目对应 extra：

```bash
python -m pip install -e ".[peft]"
```

这会解析并可能调整相关依赖；已有可用环境应先记录版本。导入成功不代表真实训练或模型兼容性已经验证。

编辑 `configs/xtrainer/train_xvla_lora.yaml`：

| 字段 | 应填写的内容 |
|---|---|
| `dataset.root` | 本地 LeRobot v2.1 数据集目录 |
| `policy.path` | 本地官方原始模型目录，当前加载器使用 `config.json` 和 `model.safetensors` |
| `policy.tokenizer_name` | 实际可加载的 tokenizer 目录，不假定一定在基座的 `tokenizer/` 下 |
| `output_dir` | 新的 LoRA 实验目录，避免覆盖已有模型 |

启动脚本的 `--base-model`、`--dataset-root`、`--output-dir` 在首次训练时覆盖相应 YAML 值。Tokenizer 没有独立脚本参数，需在 YAML 中填写。相对路径以进程工作目录为基准，建议使用绝对路径。

### 15.2 当前训练配方

| 配置 | 当前值 / 含义 |
|---|---|
| `peft.r` / `lora_alpha` | `8` / `16` |
| `target_modules` | `all-linear`，不只适配动作 Transformer |
| `full_training_modules` | `model.transformer.soft_prompt_hub`、`action_encoder`、`action_decoder` 三个完整路径所指模块 |
| `optimizer.type` | `xvla-lora-adamw`，四组参数独立倍率 |
| `scheduler.type` | `xvla-lora-staged` |
| `lora_start_step` | `1000`，在此之前两类 LoRA 参数组学习率为 0 |
| Warmup / 余弦 | 当前关闭：`num_warmup_steps: null`、`use_cosine_decay: false` |
| `use_policy_training_preset` | `false`，保留显式 LoRA optimizer/scheduler |
| Batch / 总循环步数 | `8` / `30000` |
| 数据加载 workers | `4` |
| 保存频率 | `20000` |

适配阶段仍计算梯度，学习率为 0 不等于跳过反向传播。LoRA 也不自动减少输入图像、模型前向计算或总训练步数，因此不能只按可训练参数比例估计速度。

当前可选 warmup 按 scheduler 的全局计数计算，不是从 `lora_start_step` 后重新计数。启用 warmup 或余弦时应先验证实际各组学习率曲线。

YAML 文件头部“调度器尚未实现”“只能直接使用 lerobot-train”的注释已过时：代码已有独立调度器和下述 LoRA 启动入口。不要改用全量启动脚本，它有自动本地基座覆盖逻辑。

### 15.3 图像与语言输入

LoRA 与全量训练共用 X-VLA 图像处理器：相机键映射、float 范围转换、ImageNet 归一化。三路相机、裁剪/旋转、分辨率和数据增强设置应与实验约定一致。

两份 YAML 未显式指定 `resize_imgs_with_padding`，其实际值还可能继承基座配置。使用离线变换过的数据集时，部署也必须匹配，不能认为 LoRA 会自动替你缩图。

训练命令没有单独的 `--language`：语言来自数据集的 `task_index` 对应的 `meta/tasks.jsonl` 文本，经 tokenizer 编码输入模型。查看实际指令：

```bash
head -n 20 /data/xtrainer/dataset_v21/meta/tasks.jsonl
```

推理时通过 `run_real.py --task "训练数据中的任务文本"` 提供语言输入。`tokenizer_name` 是分词器路径，不是任务文本。

### 15.4 短训练与正式训练

先填写 YAML 中的 tokenizer 等路径，再运行短训练：

```bash
bash scripts/xtrainer/train_xvla_lora.sh \
  --base-model /data/models/xvla-original \
  --dataset-root /data/xtrainer/dataset_v21 \
  --config configs/xtrainer/train_xvla_lora.yaml \
  --output-dir outputs/train/xtrainer_xvla_lora_smoke \
  --device cuda \
  --batch-size 1 \
  --steps 20 \
  --grad-accum 1 \
  --no-wandb
```

20 步只检查初始化和数据/训练通路，不跨过默认适配阶段，也不证明模型学会任务。需要阶段切换测试时，使用独立测试配置缩短 `lora_start_step` 并调整保存频率。

正式训练示例：

```bash
bash scripts/xtrainer/train_xvla_lora.sh \
  --base-model /data/models/xvla-original \
  --dataset-root /data/xtrainer/dataset_v21 \
  --config configs/xtrainer/train_xvla_lora.yaml \
  --output-dir outputs/train/xtrainer_xvla_lora \
  --device cuda \
  --batch-size 8 \
  --steps 30000 \
  --grad-accum 1
```

默认执行数据校验。`--skip-validation` 仅跳过该步骤；首次运行不建议跳过。`--no-wandb` 覆盖 `wandb.enable=false`，适用于不使用 W&B 的运行。

脚本还会尝试发现当前 Python 中 `nvidia.cudnn` 的 `lib` 目录，并加入当前进程的 `LD_LIBRARY_PATH`。这不是 CUDA/cuDNN 安装命令，也不保证所有动态库问题都已解决。

### 15.5 梯度累积与耗时

显存不足时可以尝试减小单次 batch，并显式使用：

```text
--batch-size 2 --grad-accum 4
```

单卡上每次完整累积的有效 batch 为 `2 × 4 = 8`。梯度累积不是自动加速：它用更多 micro-batch 换取较小的单次 batch 内存需求。

**当前训练外层 `step` 每个 micro-batch 增加一次。** 所以 `--steps 30000 --grad-accum 4` 不能直接理解成 30000 次实际优化器更新，通常约为 7500 次，边界还受数据加载和 Accelerator 行为影响。日志、保存频率和 scheduler 计数不要混为同一含义；多卡及累积场景需核对真实学习率轨迹。

排查慢训练时记录 GPU 型号/利用率、图像实际尺寸、日志 `data_s` 和 `updt_s`，先区分数据读取瓶颈、模型计算和总循环次数。

### 15.6 LoRA 断点续训与基座身份

```bash
bash scripts/xtrainer/train_xvla_lora.sh \
  --base-model /data/models/xvla-original \
  --dataset-root /data/xtrainer/dataset_v21 \
  --resume-checkpoint outputs/train/xtrainer_xvla_lora/checkpoints/last/pretrained_model \
  --no-wandb
```

保留完整 checkpoint，包括 `pretrained_model` 旁边的 `training_state`，后者包含续训所需的优化器、调度器、随机状态和训练步数。

当前脚本恢复时仍要求 `--base-model` 指向存在的目录，但不会把它转发为 `policy.path`；实际基座来自 adapter 配置。**改变该参数并不能重定位 resume 所用的基座。** 基座迁移后要正确处理 adapter 来源路径，并保持内容一致。

本地 X-VLA adapter 会记录 `xvla_base_manifest.json`，保存基座配置/权重哈希及来源等信息。恢复时基座内容不匹配会被拒绝；不要删除清单绕过验证。首次训练会计算基座文件哈希，大模型初始化可能因此增加耗时。

为保证恢复一致性，不建议在续训时随意改变数据、梯度累积、总步数和调度参数。旧 adapter 无清单时加载会警告，但下述合并工具要求清单存在。

### 15.7 合并导出完整模型

准备一份与训练处理器一致的观测 Tensor 字典，保存为 safetensors，用于比较合并前后的动作输出。文件必须包含模型要求的图像、状态、语言 token 和正确的域字段；不是原始图片或数据集目录。

当前没有一键生成验证观测的专用 CLI。如何从已预处理的 batch 保存文件，见 [LoRA 训练与推理指南](XVLA_LORA_TRAIN_INFERENCE_GUIDE.md)第 6 节。

```bash
python scripts/xtrainer/merge_xvla_lora.py \
  --base-model /data/models/xvla-original \
  --adapter outputs/train/xtrainer_xvla_lora/checkpoints/last/pretrained_model \
  --output-dir outputs/exports/xtrainer_xvla_lora_merged \
  --validation-batch /data/validation/observation.safetensors \
  --device cuda
```

要求及产物：

- 基座与训练时一致，adapter 清单、处理器和权重齐全。
- 输出目录必须不存在，且不能与源目录重叠。
- 工具合并 LoRA，保留额外完整训练模块；校验这些模块权重及固定种子下的动作输出。
- 保存完整模型和 tokenizer/处理器，再用 `XVLAPolicy.from_pretrained` 严格重载验证。
- 成功写入 `merge_report.json`；出现 `EXPORT_FAILED.txt` 的目录不能部署。
- 默认保留精度，可选 `--dtype float32` / `bfloat16`；输出比较默认 `--atol 0.001 --rtol 0.01`，不是机器人动作安全阈值。
- 重载检查需要额外内存/显存；合并目录不包含用于 LoRA 续训的优化器状态，原 adapter 和基座应另外保留。

### 15.8 复用现有策略服务

```bash
python scripts/xtrainer/serve_policy.py \
  --config configs/xtrainer/deploy.yaml \
  --checkpoint outputs/exports/xtrainer_xvla_lora_merged \
  --device cuda \
  --host 127.0.0.1 \
  --port 8000 \
  --actions-per-chunk 32
```

同机可使用上述监听地址；跨机使用可信局域网地址或按现场网络配置监听。机器人客户端沿用第 13 节，不增加基座或 adapter 参数。切换完整模型时先停止机器人执行，再修改 checkpoint 路径并重启服务，不支持运行中热切换。

该命令同样默认读取 `configs/xtrainer/deploy.yaml` 并启用 guided RTC；`--actions-per-chunk` 必须与合并模型的完整 `chunk_size` 一致，RTC 和 EMA 的设置见第 12、13 节。

部署会拒绝独立 adapter、失败导出和不完整的合并产物；合并模型处理器加载失败不会回退默认预处理。

第 13 节的硬件和运动授权检查仍适用：`--execute` 允许启用及移动机械臂，可能包含任务前复位；小步数或低频率不保证安全，必须按现场条件检查限幅、初始姿态和硬件急停。

### 15.9 代码与验证范围

| 文件 | 作用 |
|---|---|
| `configs/xtrainer/train_xvla_lora.yaml` | LoRA 训练参数 |
| `scripts/xtrainer/train_xvla_lora.sh` | 本地训练、梯度累积及恢复启动 |
| `src/lerobot/optim/optimizers.py`、`schedulers.py` | LoRA 四组参数及独立调度 |
| `src/lerobot/common/xvla_provenance.py` | 基座清单与身份校验 |
| `scripts/xtrainer/merge_xvla_lora.py` | 合并、导出与数值验证 |
| `deploy/xtrainer/checkpoint_validation.py` | 部署产物启动检查 |

本文更新基于代码核对，没有重新运行真实训练、合并或真机任务。短训练、梯度累积的学习率轨迹、断点恢复以及合并输出仍需在实际训练环境中验收。

进一步说明：

- [LoRA 实施方案](XVLA_LORA_IMPLEMENTATION_PLAN.md)
- [LoRA 训练与推理操作指南](XVLA_LORA_TRAIN_INFERENCE_GUIDE.md)
- [训练后真机部署指南](XVLA_REAL_ROBOT_DEPLOY_GUIDE.md)


