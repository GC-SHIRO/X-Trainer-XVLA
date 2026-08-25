# X-trainer XVLA 环境与模型工具

## 环境基线

| 项目 | 默认值 |
| --- | --- |
| 系统 | Ubuntu 24.04 x86_64 |
| Python | 3.12 |
| PyTorch | 2.8.0 |
| CUDA wheel | cu128 |
| TorchCodec | 0.6.0 |
| Conda 环境 | `xtrainer-xvla` |

默认 CUDA 安装要求 NVIDIA 驱动不低于 `570.26`。脚本不安装显卡驱动，不要求预装完整 CUDA Toolkit。

## 安装

```bash
bash tools/install_xtrainer_env.sh
conda activate xtrainer-xvla
```

可用选项：

```text
--recreate              删除并重建指定 Conda 环境
--cpu-only              安装 CPU PyTorch，用于转换和 Mock 检查
--source official       使用官方 Ubuntu/PyPI/PyTorch 源
--skip-system-packages  跳过 apt-get
--env-name NAME         自定义 Conda 环境名
```

默认使用国内镜像；`--source official` 只影响本次执行，不会永久修改系统软件源。安装内容包括训练、XVLA、
Datasets、PyArrow、PyAV、OpenCV、WebSocket、Feetech 和 RealSense 依赖。环境脚本不会下载模型或数据，
也不会连接机器人。

## 下载 XVLA 权重

官方 checkpoint 为 `lerobot/xvla-base`，其中已包含 Florence-2 视觉语言骨干，不需要单独下载 VLM：

```bash
bash tools/download_xvla_weights_hf.sh
```

默认保存到 `models/xvla-base`。下载中断后可直接重跑，已有文件会复用。

国内镜像：

```bash
bash tools/download_xvla_weights_hf.sh --endpoint https://hf-mirror.com
```

固定 revision 或自定义目录：

```bash
bash tools/download_xvla_weights_hf.sh \
  --repo-id lerobot/xvla-base \
  --output-dir /data/models/xvla-base \
  --revision main
```

私有模型使用 `hf auth login` 或设置 `HF_TOKEN`。如果环境名不是默认值，追加 `--env-name NAME`。

## 相机方向转换

已有 v2.1 视频数据需要按 X-trainer 相机安装方向校正时，创建独立副本：

```bash
python tools/transform_xtrainer_dataset_images.py \
  --input-root /data/xtrainer_dataset_original \
  --output-root /data/xtrainer_dataset_camera_aligned
```

预演而不写文件：

```bash
python tools/transform_xtrainer_dataset_images.py \
  --input-root /data/xtrainer_dataset_original \
  --output-root /data/xtrainer_dataset_camera_aligned \
  --dry-run
```

完整训练与部署流程见 [X-trainer XVLA 指南](../docs/XTRAINER_XVLA.md)。
