# X-Trainer-XVLA

面向 Dobot X-trainer 双臂平台的 XVLA 训练与真机部署工程。项目基于 LeRobot，完整覆盖原始采集数据转换、
LeRobot Dataset v2.1 读取、XVLA Phase-II 微调、WebSocket 策略服务、异步动作队列和真实硬件控制。

## 核心约定

- 基础模型：`lerobot/xvla-base`。
- 数据格式：本地只读 LeRobot Dataset v2.1，不会在训练时修改源数据。
- 状态/动作：14 维，顺序为左臂 6 关节、左夹爪、右臂 6 关节、右夹爪。
- 相机：顶视、左腕、右腕三路 RGB。
- XVLA 动作模式：`auto`，训练时把 14 维动作补齐为模型的 20 维，推理后裁回 14 维。
- XVLA domain：`19`，训练和部署必须保持一致。
- 动作块长度：32。

## 快速开始

环境安装：

```bash
bash tools/install_xtrainer_env.sh
conda activate xtrainer-xvla
```

下载官方 XVLA 权重：

```bash
bash tools/download_xvla_weights_hf.sh
```

也可以从 ModelScope 下载完整仓库快照到完全相同的本地路径：

```bash
bash tools/download_xvla_weights_modelscope.sh
```

两个脚本也会下载 XVLA processor 所需的 BART tokenizer 到 `models/xvla-base/tokenizer`，可在离线训练时直接使用。

国内网络可指定 Hugging Face 镜像：

```bash
bash tools/download_xvla_weights_hf.sh --endpoint https://hf-mirror.com
```

转换并校验数据：

```bash
python scripts/xtrainer/convert_raw_to_lerobot_2_1.py \
  --raw-root /data/xtrainer/collect_data \
  --output-root /data/xtrainer/dataset_v21 \
  --task "将试管放入试管架" \
  --fps 30 \
  --use-videos

python scripts/xtrainer/validate_dataset_v21.py \
  --root /data/xtrainer/dataset_v21 \
  --all-episodes
```

启动训练：

```bash
bash scripts/xtrainer/train_xvla.sh \
  --dataset-root /data/xtrainer/dataset_v21 \
  --device cuda \
  --batch-size 4 \
  --steps 30000 \
  --output-dir outputs/train/xtrainer_xvla
```

启动策略服务：

```bash
python scripts/xtrainer/serve_policy.py \
  --checkpoint outputs/train/xtrainer_xvla/checkpoints/last/pretrained_model \
  --device cuda
```

确认安全条件后，在机器人控制机运行：

```bash
python scripts/xtrainer/run_real.py \
  --host <策略机IP> \
  --task "将试管放入试管架" \
  --domain-id 19 \
  --action-horizon 32 \
  --max-joint-delta 0.05 \
  --max-gripper-delta 0.03 \
  --execute
```

## 目录

```text
configs/xtrainer/          XVLA 训练和部署配置
deploy/xtrainer/           XVLA wrapper、WebSocket 协议与真机硬件适配
scripts/xtrainer/          数据转换、训练、服务、Mock 和真机入口
tools/                     环境安装、模型下载与图像转换工具
tests/xtrainer/            X-trainer 专用单元及端到端测试
src/lerobot/               LeRobot 数据、训练和 XVLA 实现
```

详细的数据契约、训练参数、Mock 联调、真机安全步骤和故障处理见
[X-trainer XVLA 完整指南](docs/XTRAINER_XVLA.md)。环境与模型下载参数见
[工具说明](tools/README.md)。XVLA 模型结构可参考 [LeRobot XVLA 文档](docs/source/xvla.mdx)。

## 安全提示

`run_real.py` 和 `check_real_hardware.py` 必须显式传入 `--execute` 才会连接并控制机器人。第一次上机应降低关节
增量、降低控制速度、保持急停可触达，并先运行 Mock WebSocket 测试和单关节硬件检查。策略服务只建议部署在可信
局域网中，默认协议没有面向公网的认证能力。

## License

继承上游 LeRobot 的 Apache-2.0 许可，详见 [LICENSE](LICENSE)。
