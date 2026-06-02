# GR00T N1.7 微调文档

## 环境准备

```bash
cd /home/vega/Competition/Isaac-GR00T
uv sync --all-extras
```

## 模型下载

```bash
# 禁用 brotli 避免解压错误
pip uninstall brotlicffi brotli -y
hf download nvidia/GR00T-N1.7-3B
```

模型路径：
```
/home/vega/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots/2fc962b973bccdd5d8ce4f67cc63b264d6886495
```

---

## 数据集

位于 `/home/vega/Competition/GlobalHumanoidRobotChallenge_2026_Baseline/datasets/`

| 任务 | 路径 | episodes |
|------|------|----------|
| Foam_Inlaying | `datasets/Foam_Inlaying` | 539 |
| Packing_Box | `datasets/Packing_Box` | — |
| Part_Sorting (long) | `datasets/Part_Sorting/part_sorting_long_756_episode` | 756 |
| Part_Sorting (short) | `datasets/Part_Sorting/part_sorting_short_1000_episode` | 1000 |

所有数据集 modality 兼容性：
- **video**: 4路摄像头（head_left / head_right / wrist_left / wrist_right）✓ 一致
- **action**: 20维关节位置（joint_action）✓ 一致
- **state**: `joint_state` 0-20 ✓ 一致；`object_state` 维度各异（混合训练时不使用）

---

## 单任务微调

使用官方 `finetune.sh`，以 Foam_Inlaying 为例：

```bash
cd /home/vega/Competition/Isaac-GR00T

CUDA_VISIBLE_DEVICES=0 \
USE_WANDB=0 \
bash examples/finetune.sh \
  --base-model-path /home/vega/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots/2fc962b973bccdd5d8ce4f67cc63b264d6886495 \
  --dataset-path /home/vega/Competition/GlobalHumanoidRobotChallenge_2026_Baseline/datasets/Foam_Inlaying \
  --modality-config-path examples/WalkerS2/foam_inlaying_config.py \
  --embodiment-tag new_embodiment \
  --output-dir /home/vega/Competition/outputs/foam_inlaying_ft
```

---

## 三任务联合训练（推荐）

使用 `launch_mixed_finetune.py`（已硬编码三个数据集路径，使用 `mixed_walker_s2_config.py` 只取 `joint_state`，兼容所有任务）：

```bash
cd /home/vega/Competition/Isaac-GR00T

CUDA_VISIBLE_DEVICES=0 \
python launch_mixed_finetune.py \
  --base_model_path /home/vega/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots/2fc962b973bccdd5d8ce4f67cc63b264d6886495 \
  --dataset_path /home/vega/Competition/GlobalHumanoidRobotChallenge_2026_Baseline/datasets/Foam_Inlaying \
  --modality_config_path examples/WalkerS2/mixed_walker_s2_config.py \
  --embodiment_tag new_embodiment \
  --output_dir /home/vega/Competition/outputs/mixed_walker_s2 \
  --max_steps 20000 \
  --save_steps 1000 \
  --global_batch_size 32 \
  --learning_rate 1e-4 \
  --no-use_wandb
```

> `--dataset_path` 为必填占位符，实际训练数据由脚本内 `DATASETS` 列表决定。

### 修改训练数据集

编辑 `launch_mixed_finetune.py` 中的 `DATASETS` 列表：

```python
DATASETS = [
    f"{DATASET_BASE}/Foam_Inlaying",
    f"{DATASET_BASE}/Packing_Box",
    f"{DATASET_BASE}/Part_Sorting/part_sorting_long_756_episode",
    # 可追加 Part_Sorting short 版本：
    # f"{DATASET_BASE}/Part_Sorting/part_sorting_short_1000_episode",
]
```

---

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max_steps` | 10000 | 总训练步数，联合训练建议 20000 |
| `--global_batch_size` | 32 | 显存不足改为 16 |
| `--save_steps` | 1000 | checkpoint 保存间隔 |
| `--learning_rate` | 1e-4 | 学习率 |
| `--no-use_wandb` | — | 禁用 wandb 日志 |

---

## Modality Config 说明

| Config 文件 | 适用场景 | state 字段 |
|-------------|----------|------------|
| `examples/WalkerS2/foam_inlaying_config.py` | 单任务 Foam_Inlaying | joint_state + object_state |
| `examples/WalkerS2/mixed_walker_s2_config.py` | 多任务联合训练 | joint_state only |

---

## 输出结构

```
/home/vega/Competition/outputs/mixed_walker_s2/
  checkpoint-1000/
  checkpoint-2000/
  ...
  checkpoint-final/
```

用最终 checkpoint 路径替换推理时的 `--model-path`。

---

## 零样本推理（基线模型）

使用基线模型对 WalkerS2 数据集进行推理（无需微调）：

```bash
source /home/vega/Competition/Isaac-GR00T/scripts/activate_spark.sh

PYTHONPATH=/home/vega/Competition/Isaac-GR00T \
/home/vega/Competition/Isaac-GR00T/.venv/bin/python scripts/deployment/standalone_inference_script.py \
  --model-path /home/vega/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots/2fc962b973bccdd5d8ce4f67cc63b264d6886495 \
  --dataset-path /home/vega/Competition/GlobalHumanoidRobotChallenge_2026_Baseline/datasets/Foam_Inlaying \
  --modality-config-path examples/WalkerS2/mixed_walker_s2_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --traj-ids 0 1 2 \
  --inference-mode pytorch \
  --action-horizon 16
```

**参数说明：**
- `--modality-config-path`：自定义 modality 配置（必须先注册 `new_embodiment` tag）
- `--traj-ids`：指定推理的轨迹 ID（空格分隔）
- `--inference-mode`：`pytorch`（默认）或 `tensorrt`（需先导出 TRT engine）

**环境说明（DGX Spark 平台）：**
- 必须先执行 `source scripts/activate_spark.sh` 设置 CUDA 13.0 环境变量
- 用 `.venv/bin/python` 绝对路径避免 conda 环境干扰
- 设置 `PYTHONPATH` 确保 `gr00t` 模块可导入

---

## 依赖问题排查

### 1. torchcodec 缺失

训练/推理时报错 `ImportError: Video backend 'torchcodec' is not available`：

```bash
# 安装系统依赖（需 sudo）
sudo apt-get install -y ffmpeg libavdevice-dev libavfilter-dev \
  libavformat-dev libavcodec-dev libavutil-dev libswresample-dev \
  libswscale-dev pkg-config pybind11-dev python3-dev

# 从源码编译（Spark 平台）
git clone --depth 1 --branch v0.10.0 https://github.com/pytorch/torchcodec.git /tmp/torchcodec
cd /tmp/torchcodec
I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1 uv pip install . --no-build-isolation
cd ~ && rm -rf /tmp/torchcodec
```

### 2. Cosmos-Reason2-2B gated repo

训练时报错 `GatedRepoError: Cannot access gated repo for url nvidia/Cosmos-Reason2-2B`：

```bash
# 1. 去 https://huggingface.co/nvidia/Cosmos-Reason2-2B 申请访问权限
# 2. 登录并下载
huggingface-cli login  # 输入 HF token
hf download nvidia/Cosmos-Reason2-2B

# 3. 获取本地路径并更新训练脚本
COSMOS_PATH=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('nvidia/Cosmos-Reason2-2B', local_files_only=True))")
sed -i "s|config.model.model_name = \"nvidia/Cosmos-Reason2-2B\"|config.model.model_name = \"$COSMOS_PATH\"|" launch_mixed_finetune.py
```

### 3. flash-attn 架构不匹配警告

训练时出现 `FATAL: kernel 'fmha_cutlassF_bf16_aligned_64x128_rf_sm80' is for sm80-sm100, but was built for sm121`：

**无需处理**，代码已自动回退到 PyTorch SDPA 注意力机制，训练正常进行。
