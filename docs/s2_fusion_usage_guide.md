# S2策略融合使用指南

## 概述

S2策略是一种两步走的方法，用于生成高质量的多模态情感描述。本指南将帮助您使用Qwen-3-7B模型将基于音频的情感描述与原始文本转录进行融合。

## 文件结构

```
experiments/runs/run_s2_fusion.py    # 主要的S2融合脚本
run_s2_fusion_demo.py               # 演示脚本
docs/s2_fusion_usage_guide.md       # 本使用指南
```

## 使用方法

### 方法1：使用演示脚本（推荐）

```bash
# 运行演示脚本（会处理5个样本进行测试）
python run_s2_fusion_demo.py
```

### 方法2：直接使用主脚本

```bash
python experiments/runs/run_s2_fusion.py \
    --audio_description_file experiments/results/cloud_mer_en_test9.json \
    --transcription_file data/processed/mer2024/manifest_audio_text_augmented_v5.json \
    --output_file experiments/results/s2_fusion_results.json \
    --language english \
    --max_samples 10 \
    --verbose
```

## 参数说明

- `--audio_description_file`: 音频情感描述结果文件（如cloud_baseline、edge_baseline或speculative_decoding的输出文件）
- `--transcription_file`: 包含转录文本的manifest文件
- `--output_file`: 输出文件路径
- `--language`: 转录语言（english或chinese）
- `--max_samples`: 最大处理样本数（用于测试）
- `--verbose`: 显示详细信息

## 输入文件格式

### 音频描述文件格式
```json
{
  "detailed_results": [
    {
      "file_id": "sample_00000000",
      "generated_text": "The voice is steady but carries a tone of uncertainty..."
    }
  ]
}
```

### 转录文件格式
```json
[
  {
    "file_id": "sample_00000000",
    "english_transcription": "I don't know! I don't have experience in this area.",
    "chinese_transcription": "我哪知道啊！我也没有这方面的经验。"
  }
]
```

## 输出文件格式

```json
{
  "experiment_config": {
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "strategy": "S2_Fusion",
    "language": "english",
    "total_samples": 5,
    "timestamp": "2024-01-01 12:00:00"
  },
  "fusion_results": [
    {
      "file_id": "sample_00000000",
      "audio_description": "The voice is steady but carries a tone of uncertainty...",
      "original_transcription": "I don't know! I don't have experience in this area.",
      "fusion_description": "Based on the acoustic analysis and textual content...",
      "timestamp": "2024-01-01 12:00:00"
    }
  ]
}
```

## 示例命令

### 使用Cloud Baseline结果进行融合
```bash
python experiments/runs/run_s2_fusion.py \
    --audio_description_file experiments/results/cloud_mer_en_test9.json \
    --transcription_file data/processed/mer2024/manifest_audio_text_augmented_v5.json \
    --output_file experiments/results/s2_fusion_cloud.json \
    --language english \
    --max_samples 20
```

### 使用Edge Baseline结果进行融合
```bash
python experiments/runs/run_s2_fusion.py \
    --audio_description_file experiments/results/edge_mer_en_test7.json \
    --transcription_file data/processed/mer2024/manifest_audio_text_augmented_v5.json \
    --output_file experiments/results/s2_fusion_edge.json \
    --language english \
    --max_samples 20
```

### 使用Speculative Decoding结果进行融合
```bash
python experiments/runs/run_s2_fusion.py \
    --audio_description_file experiments/results/speculative_decoding_mer_en_test5.json \
    --transcription_file data/processed/mer2024/manifest_audio_text_augmented_v5.json \
    --output_file experiments/results/s2_fusion_speculative.json \
    --language english \
    --max_samples 20
```

## 注意事项

1. **模型要求**: 确保您有足够的GPU内存来运行Qwen-3-7B模型
2. **文件路径**: 确保输入文件的路径正确
3. **内存使用**: 处理大量样本时注意内存使用情况
4. **测试建议**: 建议先用少量样本（如5-10个）进行测试

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减少`max_samples`参数
   - 使用CPU模式（会自动检测）

2. **文件未找到**
   - 检查文件路径是否正确
   - 确保文件存在且有读取权限

3. **模型加载失败**
   - 检查网络连接（需要下载模型）
   - 确保有足够的磁盘空间

### 日志信息

脚本会输出详细的日志信息，包括：
- 数据加载状态
- 模型加载进度
- 处理进度
- 错误信息

## 下一步

完成S2融合后，您可以：
1. 分析融合结果的质量
2. 与其他策略进行比较
3. 进一步优化prompt模板
4. 集成到您的评估流程中
