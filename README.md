# 图像承载长上下文压缩实验

本仓库实现了一个完整的"图像承载长上下文压缩"论文复现实验框架，对比 Text-Context、Render-Context、Codebook-Context、Codebook-External 四种方案在 JSON restore 和 Needle-in-haystack 任务上的表现。

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行 Smoke Test

```bash
python scripts/run_exp.py --config configs/smoke_test.yaml
```

### 生成结果图表

```bash
python scripts/plot_results.py --input outputs/results/smoke_results.jsonl --output outputs/plots/
```

## 项目结构

```
image_context_compression/
├── README.md                    # 本文件
├── requirements.txt             # 依赖版本锁定
├── configs/                     # 配置文件
│   ├── default.yaml             # 默认配置
│   ├── exp_sweep.yaml           # 完整实验配置
│   └── smoke_test.yaml          # 小规模测试配置
├── src/                         # 源代码
│   ├── model/                   # 模型加载与工具
│   ├── data/                    # 数据生成器
│   ├── codec/                   # 编码器实现
│   ├── augment/                 # 图像增强
│   ├── eval/                    # 评测指标
│   └── utils/                   # 工具函数
├── assets/                      # 资源文件
│   └── fonts/                   # 内置字体
├── scripts/                      # 执行脚本
├── outputs/                     # 实验输出
└── tests/                       # 测试代码
```

## 实验配置

所有实验参数通过 YAML 配置文件管理，确保可复现性。主要配置项包括：

- **模型配置**: Qwen2-VL-2B-Instruct 加载参数
- **推理参数**: temperature=0.0（greedy decoding），max_new_tokens 按任务设置
- **数据配置**: 长度分桶（1k/2k/4k/8k/16k tokens），每个 bucket 样本数
- **Codec 配置**: 四种编码器的参数
- **增强配置**: 图像增强强度（clean/light/heavy）

## 实验结果

实验结果保存在 `outputs/results/results.jsonl`，包含：
- 预测结果与 ground truth
- 评测指标（JSON exact match, structural F1, needle hit@1）
- Token 成本（text/visual/total）
- 压缩比（orig_text_tokens / total_tokens）

使用 `scripts/plot_results.py` 生成可视化图表。

## 可复现性

- 固定随机种子（config 中指定）
- 所有配置保存为 config_hash
- 中间产物（图片、prompt）完整保存
- 推理参数锁定（temperature=0.0 等）

## 后续扩展

- LoRA 训练脚本（`scripts/train_lora.py`）支持从 codebook 图片学习文本还原
- 支持更多 codec 变体
- 支持更多评测任务
