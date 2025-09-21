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
    --prob_threshold 0.15 \
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
    --prob_threshold 0.15 \
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
    --prob_threshold 0.24 \
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
    --prob_threshold 0.1 \
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
    --prob_threshold 0.3 \
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
| `--prob_threshold` | float | `0.15` | Token接受概率阈值 |
| `--k` | int | `3` | 每次生成的draft token数量 |

### 参数调优建议
- **entropy_threshold**: 1.0-2.0，值越高Cloud调用越少
- **prob_threshold**: 0.1-0.3，值越低接受率越高
- **k**: 3-5，值越大每次draft越多但计算开销越大

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
    --prob_threshold 0.15 \
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
    --prob_threshold 0.24 \
    --k 5
```

## 🎯 推荐运行顺序

1. **快速验证**：先运行3个样本的快速测试，确保代码正常
2. **参数调优**：使用20-50个样本测试不同参数组合
3. **完整评估**：使用100+样本进行完整性能评估
4. **结果分析**：对比三个方法的latency metrics和quality metrics

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

