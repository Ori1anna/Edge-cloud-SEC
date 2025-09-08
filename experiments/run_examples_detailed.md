# Edge/Cloud Baseline 实验运行指南

本文档提供了使用 `run_edge_baseline.py` 和 `run_cloud_baseline.py` 脚本运行各种实验配置的详细说明。

## 脚本功能

### Edge Baseline (`run_edge_baseline.py`)
- **模型**: Qwen2.5-Omni-3B (边缘设备)
- **用途**: 边缘设备上的快速推理
- **特点**: 轻量级，适合实时应用

### Cloud Baseline (`run_cloud_baseline.py`)
- **模型**: Qwen2.5-Omni-7B (云端设备)
- **用途**: 云端高质量推理
- **特点**: 更高质量，适合离线分析

两个脚本都支持：
- **多数据集**：SECap、MER2024等
- **多语言**：中文、英文
- **多Prompt类型**：default、detailed、concise
- **详细Latency指标**：TTFT、ITPS、OTPS、OET、Total Time

## 基本语法

### Edge Baseline
```bash
python experiments/runs/run_edge_baseline.py \
    --dataset_path <数据集路径> \
    --output_name <输出文件名> \
    --caption_type <caption类型> \
    --language <语言> \
    --prompt_type <prompt类型> \
    [--max_samples <最大样本数>] \
    [--verbose]
```

### Cloud Baseline
```bash
python experiments/runs/run_cloud_baseline.py \
    --dataset_path <数据集路径> \
    --output_name <输出文件名> \
    --caption_type <caption类型> \
    --language <语言> \
    --prompt_type <prompt类型> \
    [--max_samples <最大样本数>] \
    [--verbose]
```

## 参数说明

| 参数 | 说明 | 选项 | 默认值 |
|------|------|------|--------|
| `--dataset_path` | 数据集manifest文件路径 | 必需 | - |
| `--output_name` | 输出结果文件名 | 可选 | edge_only_results |
| `--caption_type` | caption类型 | original, audio_only | original |
| `--language` | 生成语言 | chinese, english | chinese |
| `--prompt_type` | prompt类型 | default, detailed, concise | default |
| `--max_samples` | 最大处理样本数 | 整数 | 全部样本 |
| `--verbose` | 详细输出 | 标志 | False |

## 实验配置示例

### 1. SECap数据集实验

#### Edge Baseline - 中文 + Detailed Prompt
```bash
python experiments/runs/run_edge_baseline.py \
    --dataset_path data/processed/secap/manifest.json \
    --output_name edge_secap_chinese_detailed \
    --caption_type audio_only \
    --language chinese \
    --prompt_type detailed \
    --max_samples 100 \
    --verbose
```

#### Cloud Baseline - 中文 + Detailed Prompt
```bash
python experiments/runs/run_cloud_baseline.py \
    --dataset_path data/processed/secap/manifest.json \
    --output_name cloud_secap_chinese_detailed \
    --caption_type audio_only \
    --language chinese \
    --prompt_type detailed \
    --max_samples 100 \
    --verbose
```

#### Edge Baseline - 英文 + Default Prompt
```bash
python experiments/runs/run_edge_baseline.py \
    --dataset_path data/processed/secap/manifest.json \
    --output_name edge_secap_english_default \
    --caption_type audio_only \
    --language english \
    --prompt_type default \
    --max_samples 100 \
    --verbose
```

#### Cloud Baseline - 英文 + Default Prompt
```bash
python experiments/runs/run_cloud_baseline.py \
    --dataset_path data/processed/secap/manifest.json \
    --output_name cloud_secap_english_default \
    --caption_type audio_only \
    --language english \
    --prompt_type default \
    --max_samples 100 \
    --verbose
```

#### 中文 + Concise Prompt
```bash
python experiments/runs/run_edge_baseline.py \
    --dataset_path data/processed/secap/manifest.json \
    --output_name edge_secap_chinese_concise \
    --caption_type audio_only \
    --language chinese \
    --prompt_type concise \
    --max_samples 100 \
    --verbose
```

### 2. MER2024数据集实验

#### 中文 + Detailed Prompt
```bash
python experiments/runs/run_edge_baseline.py \
    --dataset_path data/processed/mer2024/manifest_audio_only_final.json \
    --output_name edge_mer2024_chinese_detailed \
    --caption_type audio_only \
    --language chinese \
    --prompt_type detailed \
    --max_samples 100 \
    --verbose
```

#### 英文 + Detailed Prompt
```bash
python experiments/runs/run_edge_baseline.py \
    --dataset_path data/processed/mer2024/manifest_audio_only_final.json \
    --output_name edge_mer2024_english_detailed \
    --caption_type audio_only \
    --language english \
    --prompt_type detailed \
    --max_samples 100 \
    --verbose
```

#### 中文 + Concise Prompt
```bash
python experiments/runs/run_edge_baseline.py \
    --dataset_path data/processed/mer2024/manifest_audio_only_final.json \
    --output_name edge_mer2024_chinese_concise \
    --caption_type audio_only \
    --language chinese \
    --prompt_type concise \
    --max_samples 100 \
    --verbose
```

### 3. 快速测试配置

#### 小样本测试（5个样本）
```bash
python experiments/runs/run_edge_baseline.py \
    --dataset_path data/processed/secap/manifest.json \
    --output_name edge_secap_test \
    --caption_type audio_only \
    --language chinese \
    --prompt_type default \
    --max_samples 5 \
    --verbose
```

#### 中等样本测试（50个样本）
```bash
python experiments/runs/run_edge_baseline.py \
    --dataset_path data/processed/mer2024/manifest_audio_only_final.json \
    --output_name edge_mer2024_test \
    --caption_type audio_only \
    --language english \
    --prompt_type detailed \
    --max_samples 50 \
    --verbose
```

## Prompt类型说明

### Default Prompt
- **中文**: "请根据音频内容生成简洁的中文情感描述。示例：'说话者声音颤抖，表达出悲伤和失望的情绪。'"
- **英文**: "Please generate a concise English emotion description based on the audio content. Example: 'The speaker's voice trembles, expressing sadness and disappointment.'"

### Detailed Prompt
- **中文**: "请详细分析音频中的情感特征，包括语调、语速、音量等，并生成详细的中文情感描述。"
- **英文**: "Please provide a detailed analysis of emotional features in the audio, including tone, speed, volume, etc., and generate a detailed English emotion description."

### Concise Prompt
- **中文**: "请用最简洁的中文描述音频中的情感状态。"
- **英文**: "Please describe the emotional state in the audio using the most concise English."

## 输出结果

### 结果文件位置
```
experiments/results/{output_name}.json
```

### 结果内容
- **实验配置**: 数据集、语言、prompt类型等
- **评估指标**: BLEU、CIDEr分数
- **详细Latency指标**:
  - TTFT (Time-to-First-Token)
  - ITPS (Input Token Per Second)
  - OTPS (Output Token Per Second)
  - OET (Output Evaluation Time)
  - Total Time
- **详细结果**: 每个样本的生成文本和指标

### 控制台输出示例
```
INFO:__main__:Results saved to: experiments/results/edge_secap_chinese_detailed.json
INFO:__main__:Average BLEU: 0.0087
INFO:__main__:Average CIDEr: 0.3032
INFO:__main__:Detailed Latency Metrics:
INFO:__main__:  TTFT (Time-to-First-Token): 2.2252s
INFO:__main__:  ITPS (Input Token Per Second): 45.67 tokens/sec
INFO:__main__:  OTPS (Output Token Per Second): 12.34 tokens/sec
INFO:__main__:  OET (Output Evaluation Time): 2.2252s
INFO:__main__:  Total Time: 2.2252s
```

## 常用实验组合

### Edge Baseline 实验组合（复制粘贴即可使用）

#### 组合1: SECap中文详细分析
```bash
python experiments/runs/run_edge_baseline.py --dataset_path data/processed/secap/manifest.json --output_name edge_secap_chinese_detailed --caption_type audio_only --language chinese --prompt_type detailed --max_samples 100 --verbose
```

#### 组合2: MER2024英文详细分析
```bash
python experiments/runs/run_edge_baseline.py --dataset_path data/processed/mer2024/manifest_audio_only_final.json --output_name edge_mer2024_english_detailed --caption_type audio_only --language english --prompt_type detailed --max_samples 100 --verbose
```

### Cloud Baseline 实验组合（复制粘贴即可使用）

#### 组合1: SECap中文详细分析
```bash
python experiments/runs/run_cloud_baseline.py --dataset_path data/processed/secap/manifest.json --output_name cloud_secap_chinese_detailed --caption_type audio_only --language chinese --prompt_type detailed --max_samples 100 --verbose
```

#### 组合2: MER2024英文详细分析
```bash
python experiments/runs/run_cloud_baseline.py --dataset_path data/processed/mer2024/manifest_audio_only_final.json --output_name cloud_mer2024_english_detailed --caption_type audio_only --language english --prompt_type detailed --max_samples 100 --verbose
```

#### 组合2: MER2024英文详细分析
```bash
python experiments/runs/run_edge_baseline.py --dataset_path data/processed/mer2024/manifest_audio_only_final.json --output_name edge_mer2024_english_detailed --caption_type audio_only --language english --prompt_type detailed --max_samples 100 --verbose
```

#### 组合3: SECap中文简洁描述
```bash
python experiments/runs/run_edge_baseline.py --dataset_path data/processed/secap/manifest.json --output_name edge_secap_chinese_concise --caption_type audio_only --language chinese --prompt_type concise --max_samples 100 --verbose
```

#### 组合4: MER2024中文默认prompt
```bash
python experiments/runs/run_edge_baseline.py --dataset_path data/processed/mer2024/manifest_audio_only_final.json --output_name edge_mer2024_chinese_default --caption_type audio_only --language chinese --prompt_type default --max_samples 100 --verbose
```

#### 组合5: 快速测试（5个样本）
```bash
python experiments/runs/run_edge_baseline.py --dataset_path data/processed/secap/manifest.json --output_name edge_secap_test --caption_type audio_only --language chinese --prompt_type default --max_samples 5 --verbose
```

## 注意事项

1. **数据集路径**: 确保manifest文件存在且格式正确
2. **输出文件名**: 建议使用有意义的名称，便于区分不同实验
3. **样本数量**: 建议先用小样本测试，确认无误后再运行完整实验
4. **资源监控**: 运行过程中注意GPU内存和计算资源使用情况
5. **结果备份**: 重要实验结果建议备份保存

## 故障排除

### 常见问题
1. **ModuleNotFoundError**: 确保在项目根目录运行
2. **CUDA内存不足**: 减少`max_samples`或使用更小的模型
3. **文件权限错误**: 确保有写入`experiments/results/`目录的权限

### 调试建议
- 使用`--verbose`参数查看详细输出
- 先用`--max_samples 5`进行小规模测试
- 检查日志输出中的错误信息
