# 图像承载长上下文压缩实验

本仓库实现了一个完整的"图像承载长上下文压缩"论文复现实验框架，对比 Text-Context、Render-Context、Codebook-Context、Codebook-External 四种方案在 JSON restore 和 Needle-in-haystack 任务上的表现。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 完整工作流程

```bash
# 步骤 1: 生成训练数据 (CPU 可完成，约 5-10 分钟)
python scripts/generate_train_data.py --config configs/train.yaml --num_samples 500

# 步骤 2: 训练 LoRA (需要 GPU，约 2-3 小时)
python scripts/train_lora.py --config configs/train.yaml

# 步骤 3: 运行评估实验 (需要 GPU)
python scripts/run_exp.py --config configs/exp_sweep.yaml --lora_path outputs/checkpoints/final

# 步骤 4: 生成图表
python scripts/plot_results.py --input outputs/results/results.jsonl --output outputs/plots/
```

### 3. 快速测试 (Smoke Test)

```bash
# 不使用 LoRA (base model)
python scripts/run_exp.py --config configs/smoke_test.yaml

# 使用训练好的 LoRA
python scripts/run_exp.py --config configs/smoke_test.yaml --lora_path outputs/checkpoints/final
```

## 项目结构

```
image_context_compression/
├── README.md                    # 本文件
├── requirements.txt             # 依赖版本锁定
├── configs/                     # 配置文件
│   ├── default.yaml             # 默认配置
│   ├── exp_sweep.yaml           # 完整实验配置
│   ├── smoke_test.yaml          # 小规模测试配置
│   └── train.yaml               # LoRA 训练配置
├── src/                         # 源代码
│   ├── model/                   # 模型加载与工具
│   │   ├── loader.py            # 模型加载 (支持 LoRA)
│   │   └── tokenizer_utils.py   # Token 计数工具
│   ├── data/                    # 数据生成器
│   │   ├── json_task.py         # JSON 还原任务
│   │   ├── needle_task.py       # Needle 任务
│   │   ├── train_data_generator.py  # 训练数据生成
│   │   └── dataset.py           # PyTorch Dataset
│   ├── codec/                   # 编码器实现
│   │   ├── text_context.py      # 文本编码
│   │   ├── render_context.py    # 渲染图像编码
│   │   ├── codebook_context.py  # 自定义 2D 编码
│   │   └── codebook_external.py # QR Code 编码
│   ├── augment/                 # 图像增强
│   ├── eval/                    # 评测指标
│   └── utils/                   # 工具函数
├── assets/                      # 资源文件
│   └── fonts/                   # 内置字体
├── scripts/                     # 执行脚本
│   ├── generate_train_data.py   # 生成训练数据
│   ├── train_lora.py            # LoRA 训练
│   ├── run_exp.py               # 运行实验
│   └── plot_results.py          # 生成图表
├── outputs/                     # 实验输出
│   ├── train_data/              # 训练数据
│   ├── checkpoints/             # 模型 checkpoint
│   ├── results/                 # 实验结果
│   └── plots/                   # 可视化图表
└── tests/                       # 测试代码
```

## LoRA 训练

### 为什么需要训练？

原始的 Qwen2-VL 模型**无法理解** codebook/QR code 等自定义图像编码，因为它从未见过这类数据。需要通过 LoRA 微调让模型学会从这些图像中还原原始文本。

### 训练流程

1. **生成训练数据**：为每种 codec 生成 (图像, 原始文本) 配对
2. **LoRA 微调**：使用 QLoRA (4-bit 量化) 训练，只更新少量参数
3. **评估**：使用训练好的模型运行实验

### 训练配置

编辑 `configs/train.yaml` 调整训练参数：

```yaml
# LoRA 配置
lora:
  r: 16                    # LoRA rank
  alpha: 32                # LoRA alpha
  dropout: 0.05

# 训练配置
training:
  num_epochs: 3
  batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-4

# 数据配置
data:
  generation:
    num_samples_per_codec: 500   # 每种 codec 的样本数
    codecs:
      - "render"
      - "codebook"
      - "codebook_external"
```

### 硬件要求

| 配置 | 显存需求 | 训练时间 (1500 样本) |
|------|---------|-------------------|
| Qwen2-VL-2B + QLoRA (4-bit) | ~8GB | ~2-3 小时 |
| Qwen2-VL-2B + LoRA (fp16) | ~12GB | ~2-3 小时 |

**推荐**: Google Colab T4 (免费) 或 A10G (付费)

### 在 Colab 上训练

```python
# 1. 克隆仓库
!git clone https://github.com/ZechenLiu001/QR-code-compression-llm.git
%cd QR-code-compression-llm

# 2. 安装依赖
!pip install -r requirements.txt

# 3. 生成训练数据
!python scripts/generate_train_data.py --config configs/train.yaml --num_samples 500

# 4. 训练 LoRA
!python scripts/train_lora.py --config configs/train.yaml

# 5. 运行评估
!python scripts/run_exp.py --config configs/smoke_test.yaml --lora_path outputs/checkpoints/final

# 6. 生成图表
!python scripts/plot_results.py --input outputs/results/smoke_results.jsonl
```

## 实验配置

所有实验参数通过 YAML 配置文件管理，确保可复现性。主要配置项包括：

- **模型配置**: Qwen2-VL-2B-Instruct 加载参数，支持 LoRA
- **推理参数**: temperature=0.0（greedy decoding），max_new_tokens 按任务设置
- **数据配置**: 长度分桶（512/1k/2k/4k tokens），每个 bucket 样本数
- **Codec 配置**: 四种编码器的参数
- **增强配置**: 图像增强强度（clean/light/heavy）

## 实验结果

实验结果保存在 `outputs/results/results.jsonl`，包含：
- 预测结果与 ground truth
- 评测指标（JSON exact match, structural F1, needle hit@1）
- Token 成本（text/visual/total）
- 压缩比（orig_text_tokens / total_tokens）

使用 `scripts/plot_results.py` 生成可视化图表：
- **cost-accuracy**: Token 成本 vs 准确率
- **ratio-accuracy**: 压缩比 vs 准确率
- **robustness**: 不同增强级别下的性能

## 可复现性

- 固定随机种子（config 中指定）
- 所有配置保存为 config_hash
- 中间产物（图片、prompt）完整保存
- 推理参数锁定（temperature=0.0 等）

## 常见问题

### Q: 为什么 codebook/QR 的指标全是 0？
A: 因为没有训练 LoRA。原始模型无法理解这些编码，需要先运行训练流程。

### Q: 训练数据在哪里？
A: 运行 `python scripts/generate_train_data.py` 后，数据保存在 `outputs/train_data/`。

### Q: 如何恢复中断的训练？
A: 使用 `--resume` 参数：
```bash
python scripts/train_lora.py --config configs/train.yaml --resume outputs/checkpoints/checkpoint-100
```

### Q: 如何在推理时使用训练好的模型？
A: 使用 `--lora_path` 参数：
```bash
python scripts/run_exp.py --config configs/exp_sweep.yaml --lora_path outputs/checkpoints/final
```

## License

MIT
