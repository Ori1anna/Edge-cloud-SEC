# 🚀 Edge-Cloud Speculative Decoding 实验运行指南

本文档提供了运行Edge baseline、Cloud baseline和Speculative Decoding实验的完整命令。

## 📋 目录
- [基础配置](#基础配置)
- [Edge Baseline 运行命令](#edge-baseline-运行命令)
- [Cloud Baseline 运行命令](#cloud-baseline-运行命令)
- [Speculative Decoding 运行命令](#speculative-decoding-运行命令)
- [参数说明](#参数说明)
- [输出文件说明](#输出文件说明)
- [性能对比示例](#性能对比示例)

## 🔧 基础配置

### 环境要求
```bash
# 确保在项目根目录
cd /data/gpfs/projects/punim2341/jiajunlu/edgecloud-sec

# 检查配置文件
ls configs/default.yaml
```

### 数据集路径
```bash
# 主要数据集路径
DATA_PATH="data/processed/secap/manifest.json"
```

## 🖥️ Edge Baseline 运行命令

### 基础运行
```bash
python experiments/runs/run_edge_baseline.py \
    --dataset_path data/processed/secap/manifest.json \
    --output_name edge_secap_chinese \
    --caption_type audio_only \
    --language chinese \
    --prompt_type default
```

### 完整参数运行
```bash
python experiments/runs/run_edge_baseline.py \
    --config configs/default.yaml \
    --dataset_type unified \
    --dataset_path data/processed/secap/manifest.json \
    --output_name edge_secap_chinese_full \
    --max_samples 100 \
    --verbose \
    --caption_type audio_only \
    --language chinese \
    --prompt_type default
```

### 小规模测试
```bash
python experiments/runs/run_edge_baseline.py \
    --dataset_path data/processed/secap/manifest.json \
    --output_name edge_test \
    --max_samples 10 \
    --verbose \
    --caption_type audio_only \
    --language chinese \
    --prompt_type default
```

## ☁️ Cloud Baseline 运行命令

### 基础运行
```bash
python experiments/runs/run_cloud_baseline.py \
    --dataset_path data/processed/secap/manifest.json \
    --output_name cloud_secap_chinese \
    --caption_type audio_only \
    --language chinese \
    --prompt_type default
```

### 完整参数运行
```bash
python experiments/runs/run_cloud_baseline.py \
    --config configs/default.yaml \
    --dataset_type unified \
    --dataset_path data/processed/secap/manifest.json \
    --output_name cloud_secap_chinese_full \
    --max_samples 100 \
    --verbose \
    --caption_type audio_only \
    --language chinese \
    --prompt_type default
```

### 小规模测试
```bash
python experiments/runs/run_cloud_baseline.py \
    --dataset_path data/processed/secap/manifest.json \
    --output_name cloud_test \
    --max_samples 10 \
    --verbose \
    --caption_type audio_only \
    --language chinese \
    --prompt_type default
```

## ⚡ Speculative Decoding 运行命令

### 基础运行（推荐参数）
```bash
python experiments/runs/run_speculative_decoding.py \
    --dataset_path data/processed/secap/manifest.json \
    --output_name spec_secap_chinese \
    --caption_type audio_only \
    --language chinese \
    --prompt_type default \
    --entropy_threshold 1.5 \
    --k 3
```

### 完整参数运行
```bash
python experiments/runs/run_speculative_decoding.py \
    --config configs/default.yaml \
    --dataset_type unified \
    --dataset_path data/processed/secap/manifest.json \
    --output_name spec_secap_chinese_full \
    --max_samples 100 \
    --verbose \
    --caption_type audio_only \
    --language chinese \
    --prompt_type default \
    --entropy_threshold 1.5 \
    --k 3
```

### 小规模测试
```bash
python experiments/runs/run_speculative_decoding.py \
    --dataset_path data/processed/secap/manifest.json \
    --output_name spec_test \
    --max_samples 5 \
    --verbose \
    --caption_type audio_only \
    --language chinese \
    --prompt_type default \
    --entropy_threshold 1.45 \
    --k 5
```

### 参数调优实验
```bash
# 高接受率配置（减少Cloud调用）
python experiments/runs/run_speculative_decoding.py \
    --dataset_path data/processed/secap/manifest.json \
    --output_name spec_high_acceptance \
    --max_samples 20 \
    --verbose \
    --caption_type audio_only \
    --language chinese \
    --prompt_type default \
    --entropy_threshold 2.0 \
    --k 5

# 高精度配置（增加Cloud调用）
python experiments/runs/run_speculative_decoding.py \
    --dataset_path data/processed/secap/manifest.json \
    --output_name spec_high_precision \
    --max_samples 20 \
    --verbose \
    --caption_type audio_only \
    --language chinese \
    --prompt_type default \
    --entropy_threshold 1.0 \
    --k 3
```

## 📊 参数说明

### 通用参数
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--config` | str | `configs/default.yaml` | 配置文件路径 |
| `--dataset_type` | str | `unified` | 数据集类型 |
| `--dataset_path` | str | **必需** | 数据集manifest文件路径 |
| `--output_name` | str | `*_results` | 输出文件名前缀 |
| `--max_samples` | int | `None` | 最大处理样本数 |
| `--verbose` | flag | `False` | 详细输出模式 |

### 内容参数
| 参数 | 选择 | 默认值 | 说明 |
|------|------|--------|------|
| `--caption_type` | `original`, `audio_only` | `original` | 标注类型 |
| `--language` | `chinese`, `english` | `chinese` | 生成语言 |
| `--prompt_type` | `default`, `detailed`, `concise` | `default` | 提示类型 |

### Speculative Decoding 专用参数
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--entropy_threshold` | float | `1.5` | 熵不确定性阈值 |
| `--k` | int | `3` | 每次生成的draft token数量 |

### 参数调优建议
- **entropy_threshold**: 1.0-2.0，值越高Cloud调用越少
- **k**: 3-5，值越大每次draft越多但计算开销越大
- **注意**: 现在使用内部排名阈值策略，不再需要手动设置prob_threshold

## 📁 输出文件说明

### 文件位置
```
experiments/results/
├── edge_secap_chinese.json
├── cloud_secap_chinese.json
└── spec_secap_chinese.json
```

### 输出结构
```json
{
  "experiment_config": { ... },
  "metrics": {
    "avg_bleu": 0.0169,
    "avg_cider": 0.2387,
    "latency_metrics": {
      "ttft_mean": 0.4585,
      "otps_mean": 12.75,
      "cpu_percent_mean": 0.47,
      "gpu_util_mean": 46.33
    },
    "speculative_decoding_metrics": { ... }
  },
  "detailed_results": [ ... ]
}
```

## 🔬 性能对比示例

### 完整对比实验
```bash
# 1. 运行Edge Baseline
python experiments/runs/run_edge_baseline.py \
    --dataset_path data/processed/secap/manifest.json \
    --output_name edge_comparison \
    --max_samples 50 \
    --caption_type audio_only \
    --language chinese \
    --prompt_type default

# 2. 运行Cloud Baseline
python experiments/runs/run_cloud_baseline.py \
    --dataset_path data/processed/secap/manifest.json \
    --output_name cloud_comparison \
    --max_samples 50 \
    --caption_type audio_only \
    --language chinese \
    --prompt_type default

# 3. 运行Speculative Decoding
python experiments/runs/run_speculative_decoding.py \
    --dataset_path data/processed/secap/manifest.json \
    --output_name spec_comparison \
    --max_samples 50 \
    --caption_type audio_only \
    --language chinese \
    --prompt_type default \
    --entropy_threshold 1.5 \
    --k 3
```

### 快速测试（3个样本）
```bash
# Edge Baseline
python experiments/runs/run_edge_baseline.py \
    --dataset_path data/processed/secap/manifest.json \
    --output_name edge_quick_test \
    --max_samples 3 \
    --verbose \
    --caption_type audio_only \
    --language chinese \
    --prompt_type default

# Cloud Baseline  
python experiments/runs/run_cloud_baseline.py \
    --dataset_path data/processed/secap/manifest.json \
    --output_name cloud_quick_test \
    --max_samples 3 \
    --verbose \
    --caption_type audio_only \
    --language chinese \
    --prompt_type default

# Speculative Decoding
python experiments/runs/run_speculative_decoding.py \
    --dataset_path data/processed/secap/manifest.json \
    --output_name spec_quick_test \
    --max_samples 3 \
    --verbose \
    --caption_type audio_only \
    --language chinese \
    --prompt_type default \
    --entropy_threshold 1.45 \
    --k 5
```

## 🚀 CPU-Limited Edge + GPU Cloud 测试 (iPhone 15 Plus 模拟)

### 硬件限制的Edge-Only基线测试
```bash
# 模拟iPhone 15 Plus硬件约束的Edge模型测试
python experiments/runs/run_edge_baseline_cpu_limited.py \
    --dataset_path data/processed/secap/manifest.json \
    --output_name edge_cpu_limited_iphone15 \
    --max_samples 20 \
    --verbose \
    --caption_type audio_only \
    --language chinese \
    --prompt_type default \
    --max_cpu_cores 2 \
    --max_memory_gb 16.0
```

### 硬件限制的Speculative Decoding测试
```bash
# CPU-limited Edge + GPU Cloud 混合模式
python experiments/runs/run_speculative_decoding_cpu_limited.py \
    --dataset_path data/processed/secap/manifest.json \
    --output_name speculative_decoding_cpu_limited \
    --max_samples 20 \
    --verbose \
    --caption_type audio_only \
    --language chinese \
    --prompt_type default \
    --entropy_threshold 4.0 \
    --k 5 \
    --max_cpu_cores 2 \
    --max_memory_gb 16.0
```

## 🧪 三个关键测试对比

### 测试1: Cloud-Only Baseline (GPU)
```bash
# 纯GPU Cloud模型测试 - 作为性能上限参考
python experiments/runs/run_cloud_baseline.py \
    --dataset_path data/processed/secap/manifest.json \
    --output_name cloud_secap_chinese_test \
    --max_samples 20 \
    --verbose \
    --caption_type audio_only \
    --language chinese \
    --prompt_type default
```

### 测试2: Edge-Only Baseline (CPU Limited)
```bash
# CPU限制的Edge模型测试 - 模拟真实设备性能
python experiments/runs/run_edge_baseline_cpu_limited.py \
    --dataset_path data/processed/secap/manifest.json \
    --output_name edge_cpu_limited_test \
    --max_samples 20 \
    --verbose \
    --caption_type audio_only \
    --language chinese \
    --prompt_type default \
    --max_cpu_cores 2 \
    --max_memory_gb 16.0
```

### 测试3: Speculative Decoding (CPU Edge + GPU Cloud)
```bash
# CPU Edge + GPU Cloud 混合推理测试
python experiments/runs/run_speculative_decoding_cpu_limited.py \
    --dataset_path data/processed/secap/manifest.json \
    --output_name speculative_decoding_cpu_limited_test \
    --max_samples 20 \
    --verbose \
    --caption_type audio_only \
    --language chinese \
    --prompt_type default \
    --entropy_threshold 4.0 \
    --k 5 \
    --max_cpu_cores 2 \
    --max_memory_gb 16.0
```

### 🔄 完整对比测试流程
```bash
# 步骤1: 运行Cloud Baseline (性能上限)
echo "=== 运行Cloud Baseline ==="
python experiments/runs/run_cloud_baseline.py \
    --dataset_path data/processed/secap/manifest.json \
    --output_name cloud_baseline_comparison \
    --max_samples 20 \
    --verbose \
    --caption_type audio_only \
    --language chinese \
    --prompt_type default

# 步骤2: 运行Edge CPU Limited (真实设备性能)
echo "=== 运行Edge CPU Limited ==="
python experiments/runs/run_edge_baseline_cpu_limited.py \
    --dataset_path data/processed/secap/manifest.json \
    --output_name edge_cpu_limited_comparison \
    --max_samples 20 \
    --verbose \
    --caption_type audio_only \
    --language chinese \
    --prompt_type default \
    --max_cpu_cores 2 \
    --max_memory_gb 16.0

# 步骤3: 运行Speculative Decoding (混合推理)
echo "=== 运行Speculative Decoding ==="
python experiments/runs/run_speculative_decoding_cpu_limited.py \
    --dataset_path data/processed/secap/manifest.json \
    --output_name speculative_decoding_comparison \
    --max_samples 20 \
    --verbose \
    --caption_type audio_only \
    --language chinese \
    --prompt_type default \
    --entropy_threshold 4.0 \
    --k 5 \
    --max_cpu_cores 2 \
    --max_memory_gb 16.0

echo "=== 所有测试完成 ==="
echo "结果文件保存在 experiments/results/ 目录下"
```

### 硬件配置说明
- **Edge端 (iPhone 15 Plus模拟)**:
  - CPU: 2个性能核心 (A17 Pro芯片)
  - 内存: 16GB (Qwen2.5-Omni-3B模型需要更多内存)
  - 设备: CPU + float32精度
  - 存储: NVMe SSD
  - **注意**: 内存限制主要用于监控，不会在模型加载后严格限制

- **Cloud端 (G100 GPU)**:
  - GPU: G100 64GB显存
  - 设备: CUDA + float16精度
  - 高性能计算资源

## 📊 测试结果分析

### 关键指标对比
运行三个测试后，您可以对比以下指标：

#### 性能指标 (Latency Metrics)
- **TTFT (Time To First Token)**: 首token生成时间
- **OTPS (Output Tokens Per Second)**: 输出token速度
- **Total Time**: 总生成时间
- **CPU Usage**: CPU使用率
- **GPU Usage**: GPU使用率 (仅Cloud和Speculative Decoding)

#### 质量指标 (Quality Metrics)
- **BLEU Score**: 词面重叠度
- **CIDEr Score**: 语义相似度
- **BERTScore**: 语义相似度 (Precision/Recall/F1)

#### Speculative Decoding特有指标
- **Cloud Call Rate**: Cloud模型调用频率
- **Acceptance Rate**: Edge token接受率
- **Correction Rate**: Cloud纠正率

### 预期结果分析
1. **Cloud Baseline**: 最高质量，最快速度，但需要GPU资源
2. **Edge CPU Limited**: 较低质量，较慢速度，但节省资源
3. **Speculative Decoding**: 质量接近Cloud，速度接近Edge，资源使用平衡

## 🎯 推荐运行顺序

1. **快速验证**：先运行3个样本的快速测试，确保代码正常
2. **三个对比测试**：运行上述三个关键测试进行对比
3. **参数调优**：使用20-50个样本测试不同参数组合
4. **完整评估**：使用100+样本进行完整性能评估
5. **结果分析**：对比三个方法的latency metrics和quality metrics

## ⚠️ 注意事项

1. **GPU内存**：确保有足够的GPU内存运行Cloud模型
2. **网络连接**：首次运行需要下载模型，确保网络稳定
3. **存储空间**：结果文件可能较大，确保有足够存储空间
4. **运行时间**：Cloud模型和Speculative Decoding运行时间较长

## 📞 故障排除

### 常见错误
- **模型加载失败**：检查网络连接和Hugging Face认证
- **GPU内存不足**：减少batch size或使用CPU
- **数据集路径错误**：确认manifest.json文件存在
- **配置文件缺失**：检查configs/default.yaml是否存在

### 调试模式
```bash
# 使用verbose模式查看详细日志
--verbose

# 检查单个样本
--max_samples 1
```

---

**创建时间**: 2024年12月
**版本**: 1.0
**维护者**: Edge-Cloud Speculative Decoding Team

