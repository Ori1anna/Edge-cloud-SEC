# Speculative Decoding 输出质量修复

## 问题诊断

基于 `speculative_decoding_mer_chinese_newprompt2.json` 的分析，发现以下严重问题：

### 1. 重复问题
- **严重重复**：`"犹豫犹豫，犹豫犹豫，犹豫犹豫..."` 和 `"说话人用略显紧张的音调"` 重复
- **根本原因**：重复检测太晚（25个token后），且检测条件不够严格

### 2. 模板泄漏
- **问题**：出现 `"Human"` 等对话模板内容
- **根本原因**：没有检测模板泄漏

### 3. 配置未生效
- **问题**：`entropy_threshold: 4.0` 而不是我们设置的 1.5
- **根本原因**：脚本默认值没有更新

### 4. 截断和语无伦次
- **截断**：`"说话"` 等不完整句子
- **语无伦次**：`"哎，曹小强，很烦恼的事情"` 等无关内容

## 修复方案

### 1. 早期重复检测
```python
# 在15个token后就开始检测重复
if len(generated_tokens) > 15:
    recent_tokens = generated_tokens[-10:]
    if len(set(recent_tokens)) < 3:  # 少于3个不同token就停止
        break
```

### 2. 模板泄漏检测
```python
# 检测对话模板内容
if any(template in current_text for template in ['Human', 'User', 'Assistant', 'System']):
    break
```

### 3. 更新默认配置
```python
# 更新脚本默认值
entropy_threshold: float = 1.5,  # 从4.0改为1.5
```

### 4. N句停止策略
```python
# 在speculative decoding中实现N句停止
if len(generated_tokens) > 32:  # 最小token保护
    sentence_count = 0
    for token in generated_tokens:
        if self._is_sentence_end_token(token):
            sentence_count += 1
            if sentence_count >= 2:  # 2句后停止
                break
```

### 5. 更严格的质量控制
```python
# 多层次质量检测
- 早期重复检测（15 tokens）
- 严重重复检测（25 tokens）
- 模板泄漏检测（20 tokens）
- 长度限制（120 tokens）
```

## 预期效果

### 修复前 vs 修复后

| 问题 | 修复前 | 修复后 |
|------|--------|--------|
| 重复 | `"犹豫犹豫，犹豫犹豫..."` | 早期检测，及时停止 |
| 模板泄漏 | `"...Human"` | 检测并阻止模板内容 |
| 截断 | `"说话"` | 完整句子，自然停止 |
| 语无伦次 | `"哎，曹小强..."` | 逻辑连贯的描述 |
| 配置 | `entropy_threshold=4.0` | `entropy_threshold=1.5` |

### 质量指标预期改善
- **重复率**：从严重重复 → 基本无重复
- **完整性**：从不完整句子 → 完整1-2句描述
- **相关性**：从无关内容 → 专注情感和声学特征
- **连贯性**：从语无伦次 → 逻辑清晰的描述

## 测试建议

### 运行命令
```bash
python experiments/runs/run_speculative_decoding_cpu_limited.py \
  --dataset_path data/processed/mer2024/manifest_audio_only_final.json \
  --output_name speculative_decoding_mer_chinese_fixed \
  --max_samples 100 \
  --verbose \
  --caption_type audio_only \
  --language chinese \
  --prompt_type detailed \
  --entropy_threshold 1.5 \
  --k 5 \
  --max_memory_gb 16.0
```

### 预期输出特征
- **完整句子**：以`。`结尾的完整中文句子
- **无重复**：没有重复的词汇或短语
- **无模板**：不出现Human、User等对话模板
- **逻辑连贯**：专注情感和声学线索的描述
- **适当长度**：1-2句，30-70个汉字

这些修复应该能显著改善speculative decoding的输出质量，解决重复、截断和语无伦次的问题。
